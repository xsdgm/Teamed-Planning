"""A 30-second, single-vehicle local detour surrogate for vtemp5.

The hidden scene is a river crossing with two possible passages. A directional
survey sweep and delayed packets stand in for an aerial survey. The route prior
is geometric; this environment trains the lookahead policy, not a terrain net.
World x/y are metres. Positive goal-relative action angles point left.
"""
from __future__ import annotations

import math
import numpy as np


SIDES = (1, -1)
LOOKAHEAD_M = 200.
REVIEW_S = 30.
TAU = 2. * math.pi


def _wrap(angle):
    return (angle + math.pi) % TAU - math.pi


def _bearing(p, q):
    return math.atan2(q[1] - p[1], q[0] - p[0])


def _target(position, route):
    """Aim at the near gate until aligned, then continue over the crossing."""
    approach, bank_exit, reconnect = route
    if position[0] < approach[0] + 1. and math.dist(position, approach) > 4.:
        return approach
    if position[0] < bank_exit[0] - 1.:
        # A vehicle can be abreast of an approach on the wrong side of the bank.
        if abs(position[1] - approach[1]) > 4. and position[0] < approach[0] + 7.:
            return approach
        return bank_exit
    return reconnect


def rule_offset(observation):
    """Public-evidence reference controller; chooses left on an initial tie."""
    if not observation['detected']:
        return 0.
    status = observation['side_status']
    side = observation['locked_side'] or 1
    if status[side] == 'blocked':
        side = -side
    car = observation['position']
    target = _target(car, observation['routes'][side])
    return _wrap(_bearing(car, target) - _bearing(car, observation['goal']))


def _remaining(position, polyline):
    """Remaining arclength from nearest projection plus distance to the route."""
    if len(polyline) == 1:
        return math.dist(position, polyline[0])
    best_gap = math.inf
    along = total = 0.
    for a, b in zip(polyline, polyline[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length == 0.:
            continue
        projected = ((position[0]-a[0])*dx + (position[1]-a[1])*dy) / length
        projected = min(length, max(0., projected))
        gap = math.hypot(position[0]-a[0]-projected*dx/length,
                         position[1]-a[1]-projected*dy/length)
        if gap < best_gap:
            best_gap, along = gap, total + projected
        total += length
    return total - along + best_gap


class VTemp5Env:
    """One obstacle and a bounded local mission per episode.

    blocked_side is for paired diagnostics: +1 blocks left, -1 blocks right,
    zero opens both; None samples all three. It is never in observe().
    """

    def __init__(self, seed, *, step_m=6.209, horizon=40, wait_probability=0.02889,
                 blocked_side=None):
        if not math.isfinite(step_m) or step_m < 0. or horizon < 1 or not 0. <= wait_probability <= 1.:
            raise ValueError("invalid motion or time parameters")
        self.rng = np.random.default_rng(seed)
        self.step_m, self.horizon = float(step_m), int(horizon)
        self.wait_probability = float(wait_probability)
        self._front = float(self.rng.uniform(22., 30.))
        self._far = self._front + float(self.rng.uniform(15., 21.))
        self._gate_y = {side: side * float(self.rng.uniform(17., 27.)) for side in SIDES}
        scene_side = int(self.rng.choice([0, 1, -1]))
        self._blocked_side = scene_side if blocked_side is None else blocked_side
        if self._blocked_side not in (0, 1, -1):
            raise ValueError("blocked_side must be -1, 0, 1 or None")
        self._delay = int(self.rng.integers(1, 3))
        self.goal = (float(self.rng.uniform(340., 420.)), 0.)
        self.position = (0., 0.)
        self.heading_rad = 0.
        self.elapsed_steps = 0
        self.history = []
        self._packets = []
        self._sent = set()
        # Only delivered packets may populate these fields.
        self._observed_front = None
        self._observed_gates = {}
        self._observed_far = {}
        self.side_status = {side: 'unknown' for side in SIDES}
        self.locked_side = 0
        self.blocked_evidence_step = None
        self.done = False

    @property
    def detected(self):
        return self._observed_front is not None

    def _routes(self):
        if not self.detected:
            return {}
        front = self._observed_front
        routes = {}
        for side in SIDES:
            y = self._observed_gates.get(side, side * 22.)
            far = self._observed_far.get(side, front + 18.)
            routes[side] = [(front - 5., y), (far + 3., y), (far + 15., 0.)]
        return routes

    def _side_axis(self):
        """Bisector of the two route bearings, in the current goal frame."""
        headings = []
        for route in self._routes().values():
            target = _target(self.position, route)
            headings.append(_bearing(self.position, target))
        left, right = headings
        dx, dy = math.cos(left) - math.cos(right), math.sin(left) - math.sin(right)
        if math.hypot(dx, dy) < 1e-8:
            return None  # The two branches have rejoined the same exit.
        return _wrap(math.atan2(dy, dx) - _bearing(self.position, self.goal))

    def requested_side(self, offset_rad):
        if not self.detected:
            return 0
        axis = self._side_axis()
        if axis is None:
            return self.locked_side or 1
        margin = math.cos(offset_rad - axis)
        if abs(margin) < 1e-10:
            return self.locked_side or 1
        return 1 if margin > 0. else -1

    def legal_goal_arcs(self):
        if not self.locked_side or self.side_status[self.locked_side] == 'blocked':
            return [(-math.pi, math.pi)]
        axis = self._side_axis()
        if axis is None:
            return [(-math.pi, math.pi)]
        if self.locked_side == -1:
            axis = _wrap(axis + math.pi)
        lo, hi = _wrap(axis - math.pi/2), _wrap(axis + math.pi/2)
        if lo < hi:
            return [(lo, hi)]
        if hi == -math.pi:
            return [(lo, math.pi)]
        return [(-math.pi, hi), (lo, math.pi)]

    def observe(self):
        routes = self._routes()
        return dict(position=self.position, goal=self.goal, heading_rad=self.heading_rad,
                    detected=self.detected, locked_side=self.locked_side,
                    obstacle_id=1 if self.detected else None,
                    side_status=self.side_status.copy(), elapsed_steps=self.elapsed_steps,
                    routes=routes, route_observed={s: [s in self._observed_gates,
                    self.side_status[s] == 'open', False] for s in routes},
                    goal_arcs=self.legal_goal_arcs(), history=list(self.history))

    def _in_view(self, location, world_heading, reach, aperture):
        delta = abs(_wrap(_bearing(self.position, location) - world_heading))
        distance = math.dist(self.position, location)
        return distance <= reach and (distance < 3. or delta <= aperture)

    def _survey(self, world_heading, side):
        # One fresh strip per review. Duplicate visits cannot manufacture evidence.
        if not self.detected:
            location = (self._front, 0.)
            if not self._in_view(location, world_heading, 65., math.radians(80)):
                return
            key, data = ('front', 0), self._front
        elif side not in self._observed_gates:
            location = (self._front - 5., self._gate_y[side])
            if not self._in_view(location, world_heading, 48., math.radians(72)):
                return
            key, data = ('mouth', side), self._gate_y[side]
        else:
            # Cover the whole remaining crossing on this side. A first near-bank
            # observation alone cannot refute a branch, even at a blocked mouth.
            location = (self._front + 7., self._gate_y[side])
            if not self._in_view(location, world_heading, 32., math.radians(76)):
                return
            key = ('crossing', side)
            data = ('blocked' if self._blocked_side == side else 'open', self._far)
        if key not in self._sent:
            self._sent.add(key)
            self._packets.append((self.elapsed_steps + self._delay, key, data))

    def _deliver(self):
        delivered = []
        while self._packets and self._packets[0][0] <= self.elapsed_steps:
            _, (kind, side), data = self._packets.pop(0)
            delivered.append((kind, side))
            if kind == 'front':
                self._observed_front = data
            elif kind == 'mouth':
                self._observed_gates[side] = data
            else:
                self.side_status[side], self._observed_far[side] = data
                if self.side_status[side] == 'blocked':
                    self.blocked_evidence_step = self.elapsed_steps
        return delivered

    def _driveable(self, x, y):
        if x < -25. or abs(y) > 50.:
            return False
        if not self._front <= x <= self._far:
            return True
        # The tracker also enforces the committed passage in physical space.
        # A drivable bridge on the other side is unavailable to this command.
        candidates = (self.locked_side,) if self.locked_side else SIDES
        return any(side != self._blocked_side and abs(y-self._gate_y[side]) <= 5.5
                   for side in candidates)

    def _move(self, heading, travel_m=None):
        if self.rng.random() < self.wait_probability:
            return 0.
        # Ground motion slides along an encountered bank if commanded tangentially;
        # each substep stays short compared with the minimum simulated passage.
        travel_m = self.step_m if travel_m is None else travel_m
        pieces = max(1, math.ceil(travel_m / 0.8))
        dx = travel_m / pieces * math.cos(heading)
        dy = travel_m / pieces * math.sin(heading)
        x, y = self.position
        path_m = 0.
        for _ in range(pieces):
            if self._driveable(x + dx, y + dy):
                x, y = x + dx, y + dy
                path_m += travel_m / pieces
            elif self._driveable(x, y + dy):
                y += dy
                path_m += abs(dy)
            elif self._driveable(x + dx, y):
                x += dx
                path_m += abs(dx)
            else:
                break
        self.position = (x, y)
        return path_m

    def _reference_path(self, route):
        return [(0., 0.), *route]

    def _arrived(self):
        return self.position[0] >= self._far + 7.

    def step(self, offset_rad):
        if self.done:
            raise RuntimeError("episode has already ended")
        if not math.isfinite(offset_rad):
            raise ValueError("lookahead angle must be finite")
        raw = _wrap(float(offset_rad))
        requested = self.requested_side(raw)
        previous = self.locked_side
        refuted = bool(previous and self.side_status[previous] == 'blocked')
        denied = bool(previous and previous != requested and not refuted)
        angle = raw
        if denied:
            bounds = [a for interval in self.legal_goal_arcs() for a in interval]
            angle = min(bounds, key=lambda a: abs(_wrap(a - raw)))
        if requested and not denied and self.side_status[requested] != 'blocked':
            self.locked_side = requested
        switched = bool(previous and self.locked_side != previous)
        before = self.position
        world_heading = _bearing(before, self.goal) + angle
        waypoint = (before[0] + LOOKAHEAD_M * math.cos(world_heading),
                    before[1] + LOOKAHEAD_M * math.sin(world_heading))
        side = self.locked_side
        route = self._routes().get(side, [self.goal])
        frozen = self._reference_path(route)
        self._survey(world_heading, side)
        # Once a branch is disproven, another command to that branch can only hold.
        if not side or self.side_status[side] != 'blocked':
            self._move(world_heading)
        distance = math.dist(before, self.position)
        if distance > 1e-8:
            self.heading_rad = _bearing(before, self.position)
        progress = _remaining(before, frozen) - _remaining(self.position, frozen)
        # A nearest-segment change cannot count more metres than physically travelled.
        progress = min(distance, max(-distance, progress))
        self.elapsed_steps += 1
        delivered = self._deliver()
        self.history.append(dict(position_before=before, position_after=self.position,
                                 waypoint=waypoint, local_target=_target(before, route) if side else self.goal,
                                 distance_m=distance, elapsed_steps=self.elapsed_steps,
                                 heading_rad=self.heading_rad))
        success = self._arrived()
        self.done = success or self.elapsed_steps >= self.horizon
        # Local goal band is on the far bank; reaching it requires a real crossing.
        reward = 10. * progress / 100. - 1. / self.horizon + 10. * success
        return self.observe(), reward, self.done, dict(
            success=success, side_switch=switched, unauthorized_switch=bool(switched and not refuted),
            unauthorized_request=denied, requested_side=requested, locked_side=side,
            blocked_evidence_step=self.blocked_evidence_step, progress_m=progress,
            displacement_m=distance, delivered=delivered, executed_offset_rad=angle)
