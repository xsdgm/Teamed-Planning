"""CoGraph v2 task geometry: which targets the drone network covers at once.

A *task* is a set of targets that the drone network can cover at the same time
from one vehicle position. The drones form one relative-neighbourhood tree
rooted at the vehicles, with 95 m links. CoGraph v2 trains on single-task scenes
only (every target under the network from the start); the scene generator uses
drones_to_cover for that.

Coverage estimate: a tree over {anchor, targets} with the fewest drones, one
drone at every target and a relay every link length. This is conservative:
Steiner points, sensing range and shared corridors can only lower the true count.

TaskDecomposer splits targets that cannot be covered together into ordered
tasks. It is a pure function of what one vehicle knows (its position, the
targets, its own fused graph), so any vehicle can run it for itself; it is not
used in single-task training. GraphTravel is the vehicles' travel estimate on
their own navigation graph. No obstacle truth and no reference routes.
"""
import math
from dataclasses import dataclass

import numpy as np

COMM_RANGE_M = 95.
CBF_LINK_MARGIN_M = .25
FLIGHT_HEIGHT_M = 15.
LINK_M = COMM_RANGE_M-CBF_LINK_MARGIN_M
# The first hop also climbs from the vehicle to the flight height.
GROUND_LINK_M = math.sqrt(LINK_M**2-FLIGHT_HEIGHT_M**2)


def _edge_drones(length, from_anchor, link=LINK_M, ground_link=GROUND_LINK_M):
    """Drones on one tree edge, counting the drone at its far end."""
    if from_anchor:
        return 1+max(0, math.ceil((length-ground_link)/link-1e-9))
    return max(1, math.ceil(length/link-1e-9))


def coverage_tree(anchor, targets, link=LINK_M, ground_link=GROUND_LINK_M, reach=0.):
    """Prim tree over {anchor, targets} minimising the drone count.

    reach: a drone within this distance of a target is taken to cover it (0 is
    conservative). Returns (drones, tree length in m, parent per target with -1
    for the anchor).
    """
    anchor = np.asarray(anchor, float)[:2]
    points = np.asarray(targets, float).reshape(-1, 2)
    count = len(points)
    if not count:
        return 0, 0., np.zeros(0, int)
    def drones(length, from_anchor):
        return _edge_drones(max(0., length-reach), from_anchor, link, ground_link)
    pair = np.linalg.norm(points[:, None]-points[None], axis=2)
    root = np.linalg.norm(points-anchor, axis=1)
    cost = np.array([drones(d, True) for d in root], float)
    length = root.copy()
    parent = np.full(count, -1)
    done = np.zeros(count, bool)
    total_drones, total_length = 0, 0.
    for _ in range(count):
        # Fewest drones first; equal counts take the shorter edge.
        j = int(np.lexsort((np.where(done, np.inf, length), np.where(done, np.inf, cost)))[0])
        done[j] = True
        total_drones += int(cost[j])
        total_length += float(length[j])
        for k in np.flatnonzero(~done):
            c = drones(pair[j, k], False)
            if c < cost[k] or (c == cost[k] and pair[j, k] < length[k]):
                cost[k], length[k], parent[k] = c, pair[j, k], j
    return total_drones, total_length, parent


def drones_to_cover(anchor, targets, **kwargs):
    return coverage_tree(anchor, targets, **kwargs)[0]


def tree_depth(anchor, targets, parent):
    """Longest anchor-to-target path along the coverage tree (m)."""
    anchor = np.asarray(anchor, float)[:2]
    points = np.asarray(targets, float).reshape(-1, 2)
    depth = np.full(len(points), -1.)
    def resolve(j):
        if depth[j] < 0:
            p = parent[j]
            depth[j] = (np.linalg.norm(points[j]-anchor) if p < 0
                        else resolve(p)+np.linalg.norm(points[j]-points[p]))
        return depth[j]
    return max((resolve(j) for j in range(len(points))), default=0.)


class GraphTravel:
    """A vehicle's travel estimates on its own fused navigation graph.

    Graph distance from the start to the best reachable node, plus a detour
    factor times the straight remainder. Without graph access at the start,
    the whole leg is straight times the detour.
    """
    def __init__(self, graph, start, detour=1.4, entry_radius=15.):
        self.start = np.asarray(start, float)[:2]
        self.detour = float(detour)
        self.points, self.distance = np.zeros((0, 2)), np.zeros(0)
        if graph is None or not len(graph):
            return
        import networkx as nx
        g = graph.graph
        nodes = list(g.nodes)
        points = np.asarray([g.nodes[n]['position'][:2] for n in nodes], float)
        entry = np.linalg.norm(points-self.start, axis=1)
        sources = np.flatnonzero(entry <= entry_radius)
        if not len(sources):
            return
        index = {n: i for i, n in enumerate(nodes)}
        h = nx.Graph()
        h.add_nodes_from(range(len(nodes)))
        h.add_weighted_edges_from((index[u], index[v], float(np.linalg.norm(points[index[u]]-points[index[v]])))
                                  for u, v in g.edges())
        source = len(nodes)
        h.add_weighted_edges_from((source, int(i), float(entry[i])) for i in sources)
        reached = nx.single_source_dijkstra_path_length(h, source)
        keep = np.asarray([i for i in reached if i != source], int)
        self.points = points[keep]
        self.distance = np.asarray([reached[i] for i in keep], float)

    def to(self, destination):
        q = np.asarray(destination, float)[:2]
        straight = self.detour*float(np.linalg.norm(q-self.start))
        if not len(self.points):
            return straight
        return float(min(straight, (self.distance+self.detour*np.linalg.norm(self.points-q, axis=1)).min()))


def snap_to_graph(graph, point, radius=25.):
    """The nearest delivered graph node within `radius` of `point`, else the point."""
    point = np.asarray(point, float)[:2]
    if graph is None or not len(graph):
        return point
    g = graph.graph
    positions = np.asarray([g.nodes[n]['position'][:2] for n in g.nodes], float)
    distance = np.linalg.norm(positions-point, axis=1)
    i = int(np.argmin(distance))
    return positions[i].copy() if distance[i] <= radius else point


@dataclass
class Task:
    targets: tuple          # indices into the full target list
    anchor: np.ndarray      # vehicle position from which the drones cover them
    drones: int             # estimated drones needed at the anchor
    travel_s: float         # drive to the anchor
    build_s: float          # drones' graph building at the anchor
    score: float            # (travel + build) per target


class TaskDecomposer:
    """Ordered tasks for targets the network cannot cover together.

    Pure: the result depends only on the arguments, so each vehicle can run it
    on its own information and broadcast what it decided.
    """
    def __init__(self, num_drones, vehicle_speed=1., build_speed=.5, detour=1.4, reach=0., anchor_step=10.):
        if num_drones < 1 or vehicle_speed <= 0 or build_speed <= 0 or detour < 1 or anchor_step <= 0:
            raise ValueError('Invalid task decomposer parameters')
        self.num_drones = int(num_drones)
        self.vehicle_speed, self.build_speed = float(vehicle_speed), float(build_speed)
        self.detour, self.reach, self.anchor_step = float(detour), float(reach), float(anchor_step)

    def covers(self, anchor, targets):
        return drones_to_cover(anchor, targets, reach=self.reach) <= self.num_drones

    def anchor_for(self, position, targets):
        """First point from `position` toward the targets' centroid that covers
        every target; None if even the centroid does not."""
        position = np.asarray(position, float)[:2]
        targets = np.asarray(targets, float).reshape(-1, 2)
        if self.covers(position, targets):
            return position.copy()
        centroid = targets.mean(0)
        steps = max(1, math.ceil(float(np.linalg.norm(centroid-position))/self.anchor_step))
        for t in np.linspace(0., 1., steps+1)[1:]:
            q = position+t*(centroid-position)
            if self.covers(q, targets):
                return q
        return None

    def evaluate(self, travel, position, targets, indices):
        chosen = targets[list(indices)]
        anchor = self.anchor_for(position, chosen)
        if anchor is None:
            return None
        drones, _, parent = coverage_tree(anchor, chosen, reach=self.reach)
        build = self.detour*tree_depth(anchor, chosen, parent)/self.build_speed
        drive = travel.to(anchor)/self.vehicle_speed
        return Task(tuple(sorted(int(i) for i in indices)), anchor, drones, drive, build,
                    (drive+build)/len(indices))

    def grow(self, travel, position, targets, remaining, seed):
        """Greedy maximal cluster: add the nearest target that keeps one anchor."""
        cluster, task = [seed], self.evaluate(travel, position, targets, [seed])
        while True:
            others = sorted((i for i in remaining if i not in cluster),
                            key=lambda i: min(np.linalg.norm(targets[i]-targets[j]) for j in cluster))
            for i in others:
                grown = self.evaluate(travel, position, targets, cluster+[i])
                if grown is not None:
                    cluster, task = cluster+[i], grown
                    break
            else:
                return task

    def decompose(self, position, targets, pending, graph):
        """Greedy task sequence.

        Each step grows a maximal cluster from every remaining seed (targets the
        network covers together are built together, never one after another),
        takes the cheapest per target, moves to its anchor and plans the rest
        from there.
        """
        targets = np.asarray(targets, float).reshape(-1, 2)
        remaining = [int(i) for i in pending]
        p = np.asarray(position, float)[:2]
        plan = []
        while remaining:
            travel = GraphTravel(graph, p, self.detour)
            candidates = {}
            for seed in remaining:
                task = self.grow(travel, p, targets, remaining, seed)
                if task is not None:
                    candidates.setdefault(task.targets, task)
            best = min(candidates.values(), key=lambda t: (t.score, t.targets))
            plan.append(best)
            remaining = [i for i in remaining if i not in best.targets]
            p = best.anchor
        return plan
