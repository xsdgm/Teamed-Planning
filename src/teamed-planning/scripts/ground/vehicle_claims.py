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
import math

import numpy as np

from ground.task_planner import GraphTravel


class VehicleClaimAgent:
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
