"""CoGraph v2 task geometry and decentralized vehicle target choice.

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
their own navigation graph. VehicleClaimAgent uses those estimates and
received claims to choose and transfer targets. No obstacle truth and no
reference routes.
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


class VehicleClaimAgent:
    """CoGraph v2 decentralized vehicle target choice: no leader, no central planner.

    Each vehicle runs its own VehicleClaimAgent on its own information: its
    position, its own fused navigation graph (its route status and frontier per
    target), and the claims other vehicles broadcast, received directly within
    radio range or forwarded hop by hop by the drones. A vehicle that has decided on
    a target broadcasts the claim; nobody assigns targets to anybody.

    Claim = (target, cost, since, rank). cost is the vehicle's own travel estimate
    to the target on its own graph (graph distance plus a detour factor times the
    straight remainder), refreshed every cycle; since is when it took the target.

    Ownership rule, applied by every vehicle to the same broadcast numbers (so a
    hand-over needs no extra handshake): among the valid claims on a target, the
    incumbent is the earliest (ties: lower cost, then lower rank); a challenger
    cheaper than the incumbent by transfer_margin takes the target over (the
    cheapest such challenger, then the lower rank). A vehicle that sees it no longer
    owns its target releases it and chooses again.

    Mid-course transfer: as the graph grows the costs change. A vehicle claims a
    target held by another one when it is cheaper by the margin; if it already holds
    a target, the new one must also be cheaper than its own by the margin. The
    displaced vehicle then chooses again, possibly the target just released. This
    also applies to a vehicle already driving.

    More targets than vehicles: each vehicle holds one target at a time; on arrival
    (within arrival_radius) the target is served, broadcast to all, and the vehicle
    claims the next one.

    Departure: a vehicle drives to its target once its own graph has built part of
    the way there, progress = 1 - frontier distance / straight distance >=
    depart_progress (or the target is connected): not before any graph exists, and
    without waiting for the whole graph. It drives along the delivered graph and
    waits at the graph frontier while the drones extend it.
    """

    def __init__(self, robot_id, rank, transfer_margin=.15, depart_progress=.5, claim_timeout=10.,
                 arrival_radius=8., detour=1.4, conflict_window=2.):
        if not (0. <= transfer_margin < 1. and 0. < depart_progress <= 1. and claim_timeout > 0
                and arrival_radius > 0 and detour >= 1.):
            raise ValueError('Invalid vehicle claim parameters')
        self.robot_id, self.rank = str(robot_id), int(rank)
        self.transfer_margin, self.depart_progress = float(transfer_margin), float(depart_progress)
        self.claim_timeout, self.arrival_radius, self.detour = float(claim_timeout), float(arrival_radius), float(detour)
        # A claim lost within conflict_window of taking it was a simultaneous claim
        # (a conflict), not a mid-course hand-over.
        self.conflict_window = float(conflict_window)
        self.reset()

    def reset(self):
        self.claim, self.since, self.cost, self.departed = None, None, math.inf, False
        self.served = set()
        self.heard = {}   # robot_id -> (timestamp, claim or None)
        self.claims_made = self.transfers_in = self.transfers_out = self.conflicts = 0
        self.switches = self.arrivals = self.departures = 0

    # Communication -------------------------------------------------------------
    def receive(self, messages):
        """Vehicle broadcasts (direct or drone-forwarded); the freshest per vehicle is kept."""
        for message in messages:
            if not isinstance(message, dict):
                continue
            rid, stamp = message.get('robot_id'), message.get('timestamp')
            if rid is None or rid == self.robot_id or stamp is None:
                continue
            self.served.update(int(j) for j in message.get('served', ()))
            old = self.heard.get(rid)
            if old is None or float(stamp) >= old[0]:
                self.heard[rid] = (float(stamp), message.get('claim'))

    def message(self):
        claim = None if self.claim is None else dict(target=int(self.claim), cost=float(self.cost),
                                                     since=float(self.since), rank=self.rank)
        return dict(claim=claim, served=sorted(int(j) for j in self.served))

    # Ownership -----------------------------------------------------------------
    def claims_on(self, target, now):
        """Valid claims of OTHER vehicles on `target`: (since, cost, rank, robot_id)."""
        rows = []
        for rid, (stamp, claim) in self.heard.items():
            if claim is not None and int(claim['target']) == target and now-stamp <= self.claim_timeout:
                rows.append((float(claim['since']), float(claim['cost']), int(claim['rank']), rid))
        return rows

    def owner(self, rows):
        """The rule every vehicle applies: earliest claim, unless a challenger is
        cheaper by transfer_margin (then the cheapest challenger)."""
        if not rows:
            return None
        incumbent = min(rows, key=lambda r: (r[0], r[1], r[2]))
        better = [r for r in rows if r is not incumbent and r[1] < incumbent[1]*(1.-self.transfer_margin)]
        return min(better, key=lambda r: (r[1], r[2])) if better else incumbent

    def _release(self):
        self.claim, self.since, self.cost, self.departed = None, None, math.inf, False

    # Decision ------------------------------------------------------------------
    def decide(self, now, position, targets, graph, progress, connected):
        """This vehicle's own decision for the cycle.

        progress/connected: per target, on this vehicle's own graph from its
        current position. Returns claim, cost, departed and the drive goal
        (the target once departed, else None).
        """
        targets = np.asarray(targets, float).reshape(-1, 2)
        position = np.asarray(position, float)[:2]
        if self.claim is not None and np.linalg.norm(targets[self.claim]-position) <= self.arrival_radius:
            self.served.add(self.claim)
            self.arrivals += 1
            self._release()
        travel = GraphTravel(graph, position, self.detour)
        costs = np.array([travel.to(t) for t in targets])
        if self.claim is not None:
            if self.claim in self.served:
                self._release()
            else:
                rows = self.claims_on(self.claim, now)+[(self.since, float(costs[self.claim]), self.rank, self.robot_id)]
                if self.owner(rows)[3] != self.robot_id:
                    # Another vehicle claimed it at the same time, or is now clearly
                    # better placed (mid-course hand-over): release it.
                    if now-self.since <= self.conflict_window:
                        self.conflicts += 1
                    else:
                        self.transfers_out += 1
                    self._release()
        best = None
        for j in range(len(targets)):
            if j in self.served or j == self.claim:
                continue
            holder = self.owner(self.claims_on(j, now))
            if holder is not None and not costs[j] < holder[1]*(1.-self.transfer_margin):
                continue
            if self.claim is not None and not costs[j] < costs[self.claim]*(1.-self.transfer_margin):
                continue
            if best is None or (costs[j], j) < (costs[best], best):
                best = j
        if best is not None:
            took = bool(self.claims_on(best, now))
            self.switches += self.claim is not None
            self._release()
            self.claim, self.since = best, float(now)
            self.claims_made += 1
            self.transfers_in += took
        if self.claim is not None:
            self.cost = float(costs[self.claim])
            if not self.departed and (bool(connected[self.claim]) or float(progress[self.claim]) >= self.depart_progress):
                self.departed = True
                self.departures += 1
        goal = targets[self.claim].copy() if self.claim is not None and self.departed else None
        return dict(claim=self.claim, cost=None if self.claim is None else self.cost,
                    departed=self.departed, goal=goal)
