"""Navigation-graph task metrics from delivered information, without map truth."""
from collections.abc import Mapping

import heapq
import math
import networkx as nx
import numpy as np


def route_quality(length, reference_length):
    """Bounded efficiency against a feasible approximate reference, not an oracle."""
    return min(1., reference_length/length) if np.isfinite(length) and length>0 else 0.


class RouteQualityTracker:
    """Credit improvements to the best delivered fixed-endpoint physical route.

    A disconnected map has no measurable route length. Its first connection
    gets quality credit, but never an invented infinity-to-finite meter saving.
    Keeping the best quality prevents repeated detour/reconnection rewards.
    """
    def __init__(self, reference_length):
        if not np.isfinite(reference_length) or reference_length<=0:
            raise ValueError('Reference route length must be finite and positive')
        self.reference_length=float(reference_length)
        self.previous_length=None
        self.best_length=None
        self.first_length=None

    def update(self, metrics):
        length=float(metrics['graph_task_path_length'])
        connected=bool(metrics['graph_task_connected']) and np.isfinite(length) and length>0
        before=self.previous_length
        comparable=connected and before is not None
        old_quality=route_quality(self.best_length or 0.,self.reference_length)
        first=connected and self.first_length is None
        saving=0.
        if connected:
            if first:self.first_length=length
            if self.best_length is not None:saving=max(self.best_length-length,0.)
            self.best_length=length if self.best_length is None else min(self.best_length,length)
        quality=route_quality(length,self.reference_length) if connected else 0.
        best_quality=route_quality(self.best_length or 0.,self.reference_length)
        gain=max(best_quality-old_quality,0.)
        self.previous_length=length if connected else None
        return dict(route_length_before_m=before or 0.,route_length_after_m=length if connected else 0.,
            route_lengths_comparable=float(comparable),
            route_length_delta_m=float(before-length) if comparable else 0.,
            route_new_best_saving_m=saving,route_best_length_m=self.best_length or 0.,
            route_first_length_m=self.first_length or 0.,route_quality=quality,
            route_best_quality=best_quality,route_quality_gain=gain,
            route_first_quality_gain=gain if first else 0.,
            route_shortening_quality_gain=0. if first else gain,
            route_excess_m=max(length-self.reference_length,0.) if connected else 0.)


def vehicle_route_update(before_metrics, after_metrics, start, goal):
    """Compare delivered maps at the same vehicle position and active target.

    Callers must measure both maps at these endpoints, including the endpoint
    evidence available before/after delivery. No previous-frame length or best
    route is retained: movement and target changes alone cannot earn credit.
    The straight-line distance is a lower bound, not a feasible-route oracle;
    its ratio to route length measures remaining-route efficiency, not proof
    of optimality. Signed changes charge map losses as well as paying gains.
    """
    reference = float(np.linalg.norm(np.asarray(goal, float)[:2]-np.asarray(start, float)[:2]))
    if not np.isfinite(reference):
        raise ValueError('Vehicle route endpoints must be finite')

    def measure(metrics):
        length = float(metrics['graph_task_path_length'])
        connected = (bool(metrics['graph_task_connected']) and np.isfinite(length)
                     and (length > 0. or (reference == 0. and length == 0.)))
        quality = (min(1., reference/length) if length > 0. else 1.) if connected else 0.
        remaining = float(metrics.get('graph_frontier_goal_distance_m', reference))
        progress = (1. if connected else
                    float(np.clip(1.-remaining/reference, 0., 1.)) if reference > 0. else 0.)
        return connected, length if connected else 0., quality, progress

    old_connected, old_length, old_quality, old_progress = measure(before_metrics)
    connected, length, quality, progress = measure(after_metrics)
    comparable = old_connected and connected
    gain = quality-old_quality
    first_gain = gain if connected and not old_connected else 0.
    return dict(route_reference_length_m=reference, route_reference_is_lower_bound=1.,
        route_length_before_m=old_length, route_length_after_m=length,
        route_lengths_comparable=float(comparable),
        route_length_delta_m=old_length-length if comparable else 0.,
        route_quality_before=old_quality, route_quality=quality, route_quality_gain=gain,
        route_first_quality_gain=first_gain, route_shortening_quality_gain=gain-first_gain,
        route_excess_m=max(length-reference, 0.) if connected else 0.,
        frontier_progress_before=old_progress, frontier_progress=progress,
        frontier_progress_gain=progress-old_progress)


def task_graph_metrics(graph, start, goal, connector_clear, endpoint_radius=10.):
    """graph_frontier_goal_distance_m: least straight-line goal distance of a node
    reachable from the start (start-goal distance before any start node, 0 once
    connected). Only physical reachability counts, not edge confidence."""
    result = dict(graph_task_connected=0., graph_task_path_length=0.,
        graph_task_path_ratio=0., graph_nodes=0., graph_edges=0., graph_components=0.,
        graph_frontier_goal_distance_m=float(np.linalg.norm(np.asarray(goal)-start)))
    if graph is None or not len(graph):
        return result
    g = graph.graph
    result.update(graph_nodes=float(len(g)), graph_edges=float(g.number_of_edges()),
        graph_components=float(nx.number_connected_components(g)))
    starts, goals = {}, {}
    for n, data in g.nodes(data=True):
        position = np.asarray(data['position'])[:2]
        for endpoint, candidates in ((start, starts), (goal, goals)):
            distance = float(np.linalg.norm(position-endpoint))
            if distance <= endpoint_radius and connector_clear(endpoint, position):
                candidates[n] = distance
    if not starts:
        return result
    distances = dict(starts)
    counter = 0
    queue = []
    for node, cost in starts.items():
        heapq.heappush(queue, (cost, counter, node)); counter += 1
    best = math.inf
    while queue:
        cost, _, node = heapq.heappop(queue)
        if cost != distances[node] or cost >= best:
            continue
        if node in goals:
            best = min(best, cost+goals[node])
        for other in g[node]:
            # Physical route length, independent of confidence weighting.
            length = float(np.linalg.norm(np.asarray(g.nodes[node]['position'])-
                                          g.nodes[other]['position']))
            next_cost = cost+length
            if next_cost < distances.get(other, math.inf):
                distances[other] = next_cost
                heapq.heappush(queue, (next_cost, counter, other)); counter += 1
    if math.isfinite(best):
        result.update(graph_task_connected=1., graph_task_path_length=best,
            graph_task_path_ratio=best/max(float(np.linalg.norm(np.asarray(goal)-start)), 1e-8),
            graph_frontier_goal_distance_m=0.)
    else:
        # Without a route the search above has visited the whole start component.
        result['graph_frontier_goal_distance_m'] = min(float(np.linalg.norm(
            np.asarray(g.nodes[n]['position'])[:2]-goal)) for n in distances)
    return result


def multi_target_graph_metrics(graph, start, goals, connector_clear, endpoint_radius=10.):
    """task_graph_metrics for several goals from one start, with one search.

    Returns per-goal arrays: connected, physical route length (inf without a
    route) and frontier distance (least straight-line goal distance of a node
    reachable from the start; 0 once connected)."""
    start = np.asarray(start, float)[:2]
    goals = np.asarray(goals, float).reshape(-1, 2)
    result = dict(connected=np.zeros(len(goals), bool), length=np.full(len(goals), np.inf),
                  frontier=np.linalg.norm(goals-start, axis=1), nodes=0, edges=0)
    if graph is None or not len(graph):
        return result
    g = graph.graph
    nodes = list(g.nodes)
    points = np.asarray([g.nodes[n]['position'][:2] for n in nodes], float)
    result.update(nodes=len(nodes), edges=g.number_of_edges())
    def endpoint(point):
        near = np.flatnonzero(np.linalg.norm(points-point, axis=1) <= endpoint_radius)
        return {i: float(np.linalg.norm(points[i]-point)) for i in near if connector_clear(point, points[i])}
    starts = endpoint(start)
    if not starts:
        return result
    index = {n: i for i, n in enumerate(nodes)}
    distances = dict(starts)
    queue = [(cost, i) for i, cost in starts.items()]
    heapq.heapify(queue)
    while queue:
        cost, i = heapq.heappop(queue)
        if cost != distances[i]:
            continue
        for other in g[nodes[i]]:
            j = index[other]
            next_cost = cost+float(np.linalg.norm(points[i]-points[j]))
            if next_cost < distances.get(j, math.inf):
                distances[j] = next_cost
                heapq.heappush(queue, (next_cost, j))
    reached = np.fromiter(distances, int)
    for m, goal in enumerate(goals):
        ends = endpoint(goal)
        lengths = [distances[i]+d for i, d in ends.items() if i in distances]
        if lengths:
            result['connected'][m], result['length'][m], result['frontier'][m] = True, min(lengths), 0.
        else:
            result['frontier'][m] = float(np.linalg.norm(points[reached]-goal, axis=1).min())
    return result


# Map-update credit for CoGraph's current, multi-vehicle option space.
#
# Both measurements must use the same post-motion vehicle positions and targets.
# For each target the shortest delivered route is normalized by the minimum
# straight-line distance from any vehicle.  That shared distance is a lower bound
# for the team's shortest route, not an assertion that an unobstructed route exists.

def _geometry(positions, goals):
    positions = np.asarray(positions, dtype=float)
    goals = np.asarray(goals, dtype=float)
    for name, values in (("positions", positions), ("goals", goals)):
        if values.ndim != 2 or values.shape[1] not in (2, 3) or not len(values):
            raise ValueError(f"{name} must be a nonempty array with shape (N, 2) or (N, 3)")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite coordinates")
    positions, goals = positions[:, :2], goals[:, :2]
    reference = np.linalg.norm(positions[:, None, :] - goals[None, :, :], axis=2).min(axis=0)
    return positions, goals, reference


def _route_arrays(metrics, count):
    connected = np.asarray(metrics['connected'], dtype=bool)
    length = np.asarray(metrics['length'], dtype=float)
    frontier = np.asarray(metrics['frontier'], dtype=float)
    if any(values.shape != (count,) for values in (connected, length, frontier)):
        raise ValueError('Route connected, length and frontier arrays must match the target count')
    if np.isnan(length).any() or (length < 0.).any():
        raise ValueError('Route lengths must be nonnegative or positive infinity')
    if not np.isfinite(frontier).all() or (frontier < 0.).any():
        raise ValueError('Route frontier distances must be finite and nonnegative')
    connected = connected & np.isfinite(length)
    return connected, np.where(connected, length, np.inf), np.where(connected, 0., frontier)


def _summary(connected, length, frontier, reference):
    if np.any(connected & (length == 0.) & (reference > 0.)):
        raise ValueError('A zero-length connected route requires a vehicle at the target')
    quality = np.zeros_like(reference)
    np.divide(reference, length, out=quality, where=connected & (length > 0.))
    quality[connected & (length == 0.)] = 1.
    quality = np.clip(quality, 0., 1.)
    progress = np.zeros_like(reference)
    np.divide(frontier, reference, out=progress, where=reference > 0.)
    progress = np.where(reference > 0., np.clip(1. - progress, 0., 1.), 0.)
    progress[connected] = 1.
    return dict(connected=connected.copy(), length=length.copy(), frontier=frontier.copy(),
                reference_length=reference.copy(), quality=quality, progress=progress)


def current_team_routes(per_vehicle, positions, goals):
    """Copy the current per-target team metrics before a map can mutate in place.

    ``per_vehicle`` contains the dictionaries returned by
    ``multi_target_graph_metrics`` for each vehicle's own delivered graph.
    Connectivity requires at least one vehicle; lengths and frontier distances
    are the minima over the current vehicles.  The quality denominator and its
    lower bound therefore refer to the same multi-source route problem.
    """
    positions, goals, reference = _geometry(positions, goals)
    per_vehicle = list(per_vehicle)
    if len(per_vehicle) != len(positions):
        raise ValueError('There must be one route-metrics dictionary per vehicle position')
    rows = [_route_arrays(metrics, len(goals)) for metrics in per_vehicle]
    connected = np.any([row[0] for row in rows], axis=0)
    length = np.min([row[1] for row in rows], axis=0)
    frontier = np.min([row[2] for row in rows], axis=0)
    return _summary(connected, length, frontier, reference)


def paired_team_route_update(before_metrics, after_metrics, positions, goals):
    """Compare maps at frozen current endpoints, with signed update credit.

    Inputs may be per-vehicle metric lists or copied ``current_team_routes``
    summaries.  The caller measures the old maps and evidence after physics but
    before fusion/delivery, then measures the new maps at the same positions and
    goals.  No previous-frame best value or initial task distance is consulted.
    """
    positions, goals, reference = _geometry(positions, goals)

    def aggregate(metrics):
        if isinstance(metrics, Mapping):
            connected, length, frontier = _route_arrays(metrics, len(goals))
            return _summary(connected, length, frontier, reference)
        return current_team_routes(metrics, positions, goals)

    before = aggregate(before_metrics)
    after = aggregate(after_metrics)
    comparable = before['connected'] & after['connected']
    delta = np.zeros(len(goals), dtype=float)
    np.subtract(before['length'], after['length'], out=delta, where=comparable)
    return dict(before=before, after=after,
                quality_gain=after['quality'] - before['quality'],
                frontier_gain=after['progress'] - before['progress'],
                length_delta_m=delta, lengths_comparable=comparable)
