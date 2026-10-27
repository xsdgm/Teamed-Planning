"""Navigation-graph task metrics from delivered information, without map truth."""
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
