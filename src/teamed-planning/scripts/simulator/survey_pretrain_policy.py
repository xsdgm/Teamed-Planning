"""Encode lightweight survey decisions for the full VTemp2 ground/survey actor.

The adapter takes only the receiver's public observation. It converts confirmed
free terrain into a local metric graph and reuses the deployment feature and GAT
packing code. Candidate indices are the probe actions offered by the planner.
"""
from types import SimpleNamespace

import numpy as np

from ground.ground_policy import CANDIDATE_LIMIT, SENSOR_SIDE, ground_observation
from world_model.hierarchical_runtime import SLOW_FEATURES
from world_model.navigation_evidence import EvidenceGrid, corridor_fractions, features


def _received_roads(grid, source_times):
    """Sample a regular graph with straight edges wholly inside measured free space."""
    stride = max(1, round(8. / grid.resolution))
    free = grid.cells == -1
    sampled = free[::stride, ::stride]
    ids = np.full(sampled.shape, -1, dtype=np.int32)
    row, col = np.nonzero(sampled)
    ids[row, col] = np.arange(len(row))
    y, x = row * stride, col * stride
    edges = []
    for dy, dx in ((0, 1), (1, 0)):
        inside = (row + dy < ids.shape[0]) & (col + dx < ids.shape[1])
        start = ids[row[inside], col[inside]]
        end = ids[row[inside] + dy, col[inside] + dx]
        visible = end >= 0
        for offset in range(1, stride):
            visible &= free[y[inside] + dy * offset, x[inside] + dx * offset]
        edges.append(np.stack((start[visible], end[visible])))
    unique = np.concatenate(edges, axis=1)
    xy = np.column_stack((x, y)) + .5
    return dict(graph_positions=xy * grid.resolution + np.asarray(grid.origin),
                graph_source_stamps=source_times[y, x],
                graph_confidence=np.ones(len(x), dtype=np.float32),
                graph_observation_counts=np.ones(len(x), dtype=np.int32),
                graph_edge_index=np.concatenate((unique, unique[::-1]), axis=1),
                graph_edges=unique.shape[1])


def _vehicle_tile(grid, position):
    """Sample the proxy's six-metre local sensor onto the VTemp2 ego raster."""
    resolution = 12. / SENSOR_SIDE
    offsets = (np.arange(SENSOR_SIDE) + .5 - SENSOR_SIDE / 2) * resolution
    dx, dy = np.meshgrid(offsets, offsets)
    points = np.column_stack((dx.ravel(), dy.ravel())) + position
    cells, inside = grid.sample(points)
    visible = inside & (np.square(dx.ravel()) + np.square(dy.ravel()) <= 6.**2)
    occupancy = np.full(cells.shape, -1., dtype=np.float32)
    occupancy[visible & (cells != 0)] = cells[visible & (cells != 0)] == 1
    return occupancy.reshape(SENSOR_SIDE, SENSOR_SIDE), resolution


def policy_observation(state, *, local_window=False):
    """Pack a `SurveyEnv.observe()` result, preserving the planner's action order."""
    position = np.asarray(state['position'], dtype=float)
    goal = np.asarray(state['goal'], dtype=float)
    now = float(state['time'])
    points = np.asarray(state['targets'], dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not 0 < len(points) <= CANDIDATE_LIMIT:
        raise ValueError('Expected an unpadded set of XY survey targets')
    cells = np.asarray(state['cells'])
    source_times = np.asarray(state['source_times'])
    if source_times.shape != cells.shape:
        raise ValueError('Received terrain and source time grids disagree')
    origin = np.asarray(state['origin'], dtype=float)
    resolution = float(state['resolution'])
    if local_window and not state.get('local_window', False):
        from simulator.survey_pretrain_env import survey_policy_window
        ys, xs = survey_policy_window(position, points, state['ground_target'], origin, resolution, cells.shape)
        origin = origin + np.array([xs.start, ys.start]) * resolution
        cells = cells[ys, xs]
        source_times = source_times[ys, xs]
    # The proxy measures vehicle-clearance-adjusted traversability directly.
    grid = EvidenceGrid(cells, resolution, origin, clearance=0.)
    velocity = np.asarray(state['velocity'], dtype=float)
    roads = _received_roads(grid, source_times)
    belief = dict(position=position, goal=goal, grid=grid, speed=float(np.linalg.norm(velocity)), **roads)
    receipt = state.get('last_survey_receipt_at')
    if receipt is not None:
        if not 0. <= receipt <= now:
            raise ValueError('Survey receipt is outside the receiver timeline')
        belief.update(receipt_age=now - receipt, recent_receipts=float(now - receipt <= 10.))
    legal = np.asarray(state.get('mask', np.ones(len(points), dtype=bool)), dtype=bool)
    if legal.shape != (len(points),) or not legal.any():
        raise ValueError('Expected at least one available survey target')
    rows = np.zeros((CANDIDATE_LIMIT, len(SLOW_FEATURES)), dtype=np.float32)
    targets = np.zeros((CANDIDATE_LIMIT, 2))
    present = np.arange(CANDIDATE_LIMIT) < len(points)
    mask = np.zeros(CANDIDATE_LIMIT, dtype=bool)
    targets[present], mask[present] = points, legal
    corridors = corridor_fractions(grid, position, points)
    for i, point in enumerate(points):
        base, _ = features(belief, point, corridor=corridors[i])
        rows[i, :len(base)] = base
    rows[present, SLOW_FEATURES.index('remaining_budget')] = state['remaining_budget']
    # Car control history stays empty. Prior probe results have a separate task block.
    candidates = dict(features=rows, targets=targets, mask=mask, present=present)
    sensor, resolution = _vehicle_tile(grid, position)
    own = dict(position=position, velocity=velocity, orientation=state['orientation'],
               angular_velocity=0.)
    vehicle = dict(self=own, occupancy_map=sensor, map_resolution=resolution)
    episode = SimpleNamespace(goal=goal, belief=lambda: belief,
                              env=SimpleNamespace(ground='ground', obs={'ground': vehicle},
                                                  env=SimpleNamespace(time=now)))
    task_context = {key: state[key] for key in
                    ('ground_target', 'survey_target', 'survey_issued_at', 'survey_evidence_received')}
    # The receiver in this proxy has no one-hop UAV sensor or V9 telemetry relay.
    return ground_observation(episode, candidates, [], [], task_context=task_context)
