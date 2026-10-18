"""Navigation-graph task metrics from delivered information, without map truth."""
import heapq
import math
import networkx as nx
import numpy as np


def task_graph_metrics(graph, start, goal, connector_clear, endpoint_radius=10.):
    result = dict(graph_task_connected=0., graph_task_path_length=0.,
        graph_task_path_ratio=0., graph_nodes=0., graph_edges=0., graph_components=0.)
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
    if not starts or not goals:
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
            graph_task_path_ratio=best/max(float(np.linalg.norm(np.asarray(goal)-start)), 1e-8))
    return result
