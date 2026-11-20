"""Receiver-side observations for the VTemp4 continuous lookahead policy."""
from types import SimpleNamespace

import numpy as np

from simulator.ground_evidence import observed_fused_graph
from ground.ground_policy import CANDIDATE_LIMIT, ground_observation, ground_sensor_tile
from world_model.navigation_evidence import EvidenceGrid, corridor_fractions
from ground.survey_fleet_clearance import native_segment_clear
from simulator.survey_fleet_observation import _clearance


# Ring order is relative to the held guidance, positive counter-clockwise. The
# ring describes the evidence; it does not quantise the action of the actor.
DIRECTION_COUNT = 24
RELATIVE_ANGLES = np.arange(DIRECTION_COUNT) * (2 * np.pi / DIRECTION_COUNT)
FEATURE_NAMES = ('relative_cos', 'relative_sin', 'range_fraction', 'connected_road_fraction') + tuple(
    f'section_{section}_{state}' for section in range(8) for state in ('free', 'blocked', 'unknown'))
FEATURE_COUNT = len(FEATURE_NAMES)
CONTEXT_NAMES = ('heading_valid', 'heading_cos', 'heading_sin', 'log_held_seconds',
                 'last_dx_200m', 'last_dy_200m', 'last_goal_progress_200m', 'remaining_budget',
                 'terrain_receipt_valid', 'log_terrain_receipt_age_seconds')
CONTEXT_SIZE = len(CONTEXT_NAMES)


def ring_points(position, yaw, relative_angles, radius):
    """Convert issue-time vehicle-relative angles to world XY on a fixed-radius ring."""
    origin = np.asarray(position, dtype=float)
    angles = np.asarray(relative_angles, dtype=float) + yaw
    if (origin.shape != (2,) or not np.isfinite(origin).all() or angles.ndim != 1
            or not np.isfinite(angles).all() or not np.isfinite(radius) or radius <= 0):
        raise ValueError('Expected a measured XY pose, finite ring angles, and positive radius')
    return origin + radius * np.column_stack((np.cos(angles), np.sin(angles)))


def valid_ring_points(adapter, points):
    """Filter world bounds and known occupied endpoints. Unknown terrain may be surveyed."""
    points = np.asarray(points, dtype=float)
    world = float(adapter.ep.case['world_size'])
    valid = np.isfinite(points).all(axis=1) & ((points >= 0.) & (points < world)).all(axis=1)
    # Query small clearance neighbourhoods around each legal endpoint using
    # only the receiver's map. The native ground planner certifies motion to it.
    for index in np.flatnonzero(valid):
        xy = np.floor(points[index] / adapter.resolution).astype(int)
        lo = np.maximum(0, xy - 3)
        hi = np.minimum(adapter.cells.shape[::-1], xy + 4)
        patch = _clearance(adapter.cells[lo[1]:hi[1], lo[0]:hi[0]])
        valid[index] = patch[xy[1]-lo[1], xy[0]-lo[0]] != 1
    return valid


def _connected_roads(graph, car, rays, ranges, local_tile, resolution):
    """Furthest contiguous observed road in each 30 m half-width corridor.

    Start only where the car can connect using its own sensor. An edge is
    credited up to its first exit; a disconnected road beyond a gap adds zero.
    """
    result = np.zeros(len(rays))
    xy, edges = graph['graph_positions'], graph['graph_edge_index']
    if local_tile is None or not edges.size:
        return result
    delta = xy - car
    near = np.flatnonzero(np.linalg.norm(delta, axis=1) <= 15.)
    entries = [node for node in near
               if native_segment_clear(local_tile, car, resolution, car, xy[node])]
    if not entries:
        return result
    outgoing = {}
    for source, dest in edges.T:
        if source != dest:
            outgoing.setdefault(source, []).append(dest)
    width = 30.
    for i, (ray, extent) in enumerate(zip(rays, ranges)):
        along = delta @ ray
        side = delta[:, 1] * ray[0] - delta[:, 0] * ray[1]
        inside = (along >= -width) & (along <= extent) & (np.abs(side) <= width)
        queue = [node for node in entries if inside[node] and node in outgoing]
        seen = set(queue)
        while queue:
            source = queue.pop()
            for dest in outgoing.get(source, ()):
                if inside[dest]:
                    result[i] = max(result[i], along[source], along[dest])
                    if dest not in seen:
                        seen.add(dest)
                        queue.append(dest)
                elif along[dest] > along[source]:
                    fraction = 1.
                    if abs(side[dest]) > width:
                        boundary = np.copysign(width, side[dest])
                        fraction = (boundary - side[source]) / (side[dest] - side[source])
                    reached = along[source] + fraction * (along[dest] - along[source])
                    result[i] = max(result[i], min(extent, reached))
    return result


def build_observation(adapter, heading_rad, held_seconds, previous_position, *, lookahead_m=200., limit_by_goal=True):
    """Build one review from onboard measurements and packets already delivered.

    On the first review, a missing heading uses the goal bearing as reference.
    A missing previous position denotes no completed decision interval.
    """
    if (not np.isfinite(lookahead_m) or lookahead_m <= 0
            or not np.isfinite(held_seconds) or held_seconds < 0):
        raise ValueError('Expected a positive lookahead and nonnegative heading age')
    adapter.observe_receipts()
    episode, exp = adapter.ep, adapter.exp
    car = np.asarray(adapter.own['position'], dtype=float)[:2]
    goal = np.asarray(episode.goal, dtype=float)[:2]
    goal_delta = goal - car
    remaining = np.linalg.norm(goal_delta)
    heading = np.arctan2(goal_delta[1], goal_delta[0]) if heading_rad is None else heading_rad
    if not np.isfinite(heading):
        raise ValueError('Expected a finite guidance angle')
    angles = heading + RELATIVE_ANGLES
    rays = np.column_stack((np.cos(angles), np.sin(angles)))
    # The last receiver grid cell can extend beyond a fractional mission extent.
    world_size = float(episode.case['world_size'])
    boundary = np.where(rays >= 0, world_size - car, -car)
    axis_limits = np.divide(boundary, rays, out=np.full_like(rays, np.inf), where=np.abs(rays) > 1.e-12)
    horizon = min(lookahead_m, remaining) if limit_by_goal else lookahead_m
    ranges = np.clip(axis_limits.min(axis=1), 0., horizon)
    points = car + ranges[:, None] * rays

    # Grid work per decision is bounded by the sensing horizon on kilometre maps.
    # Include the full half-width of the VTemp2 terrain corridor along every ray.
    reach = ranges.max() + 13.
    low = np.maximum(0, np.floor((car - reach) / adapter.resolution).astype(int))
    high = np.minimum(adapter.cells.shape[::-1], np.ceil((car + reach) / adapter.resolution).astype(int) + 1)
    cells = adapter.cells[low[1]:high[1], low[0]:high[0]]
    grid = EvidenceGrid(cells, adapter.resolution, low * adapter.resolution, clearance=0.)
    fractions = corridor_fractions(grid, car, points,
                                   sample_bounds=((0., 0.), (world_size, world_size)))
    graph = observed_fused_graph(exp.ground_controller.unified_graph, adapter.now)
    adapter.last_graph_edges = graph['graph_edges']
    local = exp.obs[exp.ground]
    support = _connected_roads(graph, car, rays, ranges, ground_sensor_tile(local), local.get('map_resolution'))

    rows = np.zeros((CANDIDATE_LIMIT, FEATURE_COUNT), np.float32)
    rows[:DIRECTION_COUNT] = np.column_stack((np.cos(RELATIVE_ANGLES), np.sin(RELATIVE_ANGLES),
                                              ranges / lookahead_m, support / lookahead_m, fractions))
    points_padded = np.zeros((CANDIDATE_LIMIT, 2))
    points_padded[:DIRECTION_COUNT] = points
    present = np.arange(CANDIDATE_LIMIT) < DIRECTION_COUNT
    queries = dict(features=rows, targets=points_padded, present=present, mask=present.copy())
    comm = episode._relay_observation.observe()
    facade = SimpleNamespace(env=exp, goal=goal, belief=lambda: graph)
    # The local drive waypoint can be present before a shared guidance is issued.
    # VTemp4 has no independent survey request in the VTemp2 task block.
    task = dict(ground_target=adapter.ground_target, survey_target=None,
                survey_issued_at=None, survey_evidence_received=False)
    obs = ground_observation(facade, queries, comm['one_hop_neighbors'], comm['relay_tokens'], task_context=task)
    previous = car if previous_position is None else np.asarray(previous_position, dtype=float)[:2]
    if previous.shape != (2,) or not np.isfinite(previous).all():
        raise ValueError('Expected a finite XY position at the previous review')
    progress = np.linalg.norm(goal - previous) - remaining
    receipt = adapter.last_received
    context = np.array([heading_rad is not None, np.cos(heading), np.sin(heading), np.log1p(held_seconds),
                        *((car - previous) / 200.), progress / 200.,
                        np.clip(1. - adapter.now / episode.rollout_limit, 0., 1.),
                        receipt is not None, 0. if receipt is None else np.log1p(adapter.now - receipt)],
                       dtype=np.float32)
    if not np.isfinite(context).all():
        raise ValueError('Invalid guidance context')
    obs['context'] = context
    return obs
