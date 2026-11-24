"""vtemp1 directional lookahead for ground travel and aerial mapping."""
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from numbers import Real

import numpy as np


DEFAULT_REVIEW_INTERVAL_S = 15.0
ROAD_SUPPORT_WEIGHT = 1.5
SUPPORT_GAIN_M = 10.0


@dataclass(frozen=True)
class FailedApproach:
    position: np.ndarray  # Vehicle position when progress failed.
    direction: np.ndarray
    time: float
    support: float  # Connected road support for this approach at failure.


class LookaheadGuidance:
    """Represent a travel direction by a rolling point inside the survey radius."""

    def __init__(self, goal, radius=200., arrival=25., snap=20., snap_distance=None,
                 stall=180., max_wait=600., retreat=40., corridor=30., penalty_radius=5.,
                 penalty_decay=600., directions=24, world_bounds=None, margin=10.,
                 review_interval=DEFAULT_REVIEW_INTERVAL_S):
        if not 0 < arrival < radius or snap < 0 or not 0 < stall <= max_wait or retreat <= 0:
            raise ValueError('Require 0 < arrival < radius, snap >= 0, 0 < stall <= max_wait and retreat > 0')
        if type(review_interval) not in (float, int) or not np.isfinite(review_interval) or review_interval <= 0:
            raise ValueError('review_interval must be a positive finite duration in seconds')
        if directions < 3 or corridor <= 0 or penalty_radius < 0 or penalty_decay <= 0:
            raise ValueError('Require a positive corridor and decay, a nonnegative penalty radius, and >= 3 directions')
        self.goal = np.asarray(goal, float)[:2]
        self.radius, self.arrival = float(radius), float(arrival)
        self.stall, self.max_wait, self.retreat = float(stall), float(max_wait), float(retreat)
        self.corridor = float(corridor)
        self.penalty_radius, self.penalty_decay = float(penalty_radius), float(penalty_decay)
        self.directions = int(directions)
        self.world_bounds, self.margin = world_bounds, float(margin)
        self.review_interval = float(review_interval)
        self.target = self.origin = self._direction = self._route_direction = None
        self.final = self.finishing = self.direction_changed = False
        # Legacy configurations and recorders carry snap options/status. A
        # direction marker can pass an obstacle without snapping onto a node.
        self.snapped = False
        self.failures = []
        self.selected = self.arrivals = self.failed = self.reviews = 0
        self._last_review = self._last_update = None
        self._last_position = None
        self._progress = 0.
        self._best = self._support = self._since = self._approached = 0.

    def update(self, position, graph, now, entry_allowed=None):
        """Return (shared target, needs update) using this frame's filtered road graph.

        entry_allowed(xy) checks a short car-to-road connection using current
        local observations. Standalone callers without one can enter at the car.
        """
        p = np.asarray(position, float)[:2]
        self.direction_changed = False
        if self._last_update is not None and now < self._last_update:
            self.target = None
            self._direction = None
            self.finishing = False
            self.failures = [failure for failure in self.failures if failure.time <= now]
        self._last_update = now

        goal_distance = np.linalg.norm(self.goal - p)
        if self.finishing and goal_distance > self.radius:
            self.finishing = False
            self.target = self._direction = None
        if self.finishing or goal_distance <= min(25., self.arrival):
            self.finishing = True
            changed = self.target is None or not self.final
            if changed:
                self.target, self.origin, self.final = self.goal.copy(), p.copy(), True
                self.selected += 1
                self._since = now
            if now - self._since >= self.stall:
                self._since = now
                changed = True
            self._last_review = now
            self.direction_changed = changed
            return self.target, changed

        initial = self.target is None
        early = failed = False
        if not initial:
            # Accumulate actual motion along the bearing active during each
            # step, so a gradual bend does not look like a retreat.
            self._progress += float((p - self._last_position) @ self._direction)
            self._last_position = p.copy()
            if self._progress > self._best + 2.:
                self._best, self._since, self._approached = self._progress, now, now
            retreated = self._progress < self._best - self.retreat
            timed_out = now - self._approached >= self.max_wait
            failed = retreated or timed_out
            # A fast vehicle can pass the old marker between two scheduled
            # reviews. Refresh before the controller would steer back toward it.
            length = np.linalg.norm(self.target - self.origin)
            if not self.final and length > 1e-8:
                early = (np.linalg.norm(self.target - p) <= min(self.arrival, length / 2)
                         or (p - self.origin) @ self._direction >= length)
            if (not early and not failed and now - self._since < self.stall
                    and now - self._last_review < self.review_interval):
                return self.target, False

        points = self._candidates(p, goal_distance)
        offsets = points - p
        lengths = np.linalg.norm(offsets, axis=1)
        rays = offsets / np.maximum(lengths[:, None], 1e-9)
        remaining = np.linalg.norm(points - self.goal, axis=1)
        reference = self._direction if not initial else None
        if self.final and not initial and np.any(remaining < 1e-8):
            reference = (self.goal - p) / goal_distance
        support, outside_support = self._road_support(p, graph, rays, lengths, entry_allowed, reference)
        turn_threshold = np.cos(np.pi / self.directions)
        same_heading = (np.zeros(len(points), dtype=bool) if initial else
                        rays @ self._direction >= turn_threshold)

        if not initial:
            prior_support = float(np.max(support[same_heading], initial=0.))
            if prior_support > self._support + SUPPORT_GAIN_M:
                self._since, self._support = now, prior_support
            failed = failed or now - self._since >= self.stall
            if failed:
                self.failures.append(FailedApproach(p.copy(), self._direction.copy(), now, prior_support))
                self.failed += 1
            if early:
                self.arrivals += 1

        # Let a surveyed, continuous detour outweigh a modest increase in
        # remaining straight-line distance. Unknown directions can still win.
        penalties = self._approach_penalties(p, rays, support, turn_threshold, now)
        penalties[remaining < 1e-8] = 0.  # The mission destination is immutable.
        scoring_support = support
        holding = np.zeros(len(points), dtype=bool)
        if not initial and not failed:
            # Compare the actual bearings. Rolling the marker forward along
            # the existing bearing, or tracking the same fixed destination,
            # should not pay a turn cost.
            cosine = np.clip(rays @ reference, -1., 1.)
            holding = cosine >= 1. - 1e-12
            if holding.any():
                # A local bend that still fits the current corridor is already
                # available to the vehicle planner. Aligning a new bearing
                # with that same bend should not earn an extra mapping credit.
                held_support = np.max(support[holding])
                scoring_support = np.minimum(support, np.maximum(held_support, outside_support))
                angles = np.degrees(np.arccos(cosine))
                turn_cost = (10. + .2 * angles) * (self.radius / 200.)
                penalties += np.where(holding, 0., turn_cost)
        scores = remaining - ROAD_SUPPORT_WEIGHT * scoring_support + penalties
        index = int(np.lexsort((remaining, ~holding, np.round(scores, 6)))[0])
        new_target = points[index]
        final = remaining[index] < 1e-8
        moved = initial or np.linalg.norm(new_target - self.target) > 1e-6
        new_direction = initial or failed or final != self.final or (moved and not same_heading[index])
        # The vehicle may reuse its route across several small successive
        # rolls. Account for their cumulative turn relative to that route.
        self.direction_changed = new_direction or (moved and rays[index] @ self._route_direction < turn_threshold)
        if moved:
            self.selected += 1
        if self.direction_changed:
            self._route_direction = rays[index].copy()
        if new_direction:
            self._progress = self._best = 0.
            self._support = float(support[index])
            self._since = self._approached = now
        self._last_position = p.copy()
        self.target, self.origin, self.final = new_target.copy(), p.copy(), bool(final)
        self._direction = rays[index].copy()
        self._last_review = now
        self.reviews += 1
        return self.target, moved or self.direction_changed

    def _candidates(self, p, goal_distance):
        goal_offset = self.goal - p
        heading = np.arctan2(goal_offset[1], goal_offset[0])
        angles = heading + np.arange(self.directions) * (2 * np.pi / self.directions)
        rays = np.column_stack((np.cos(angles), np.sin(angles)))
        if self.target is not None:
            rays = np.vstack([rays, self._direction])
        horizon = min(goal_distance, self.radius)
        lengths = np.full(len(rays), horizon)
        if self.world_bounds is not None:
            bounds = np.asarray(self.world_bounds, float).reshape(2, 2)
            margin = np.minimum(self.margin, (bounds[1] - bounds[0]) / 2)
            low, high = bounds[0] + margin, bounds[1] - margin
            for axis in (0, 1):
                direction = rays[:, axis]
                moving = np.abs(direction) > 1e-9
                boundary = np.where(direction > 0, high[axis], low[axis])
                lengths[moving] = np.minimum(lengths[moving],
                    (boundary[moving] - p[axis]) / direction[moving])
            points = p + np.maximum(lengths, 0.)[:, None] * rays
            inside = np.all((points >= low - 1e-8) & (points <= high + 1e-8), axis=1)
            valid = inside & (lengths > 1e-6)
            # Prefer a marker far enough to convey a heading. In a tiny
            # survey box even the longest admissible ray may fall short.
            useful = valid & (lengths >= min(self.arrival, horizon / 2))
            points = points[useful if np.any(useful) else valid]
        else:
            points = p + lengths[:, None] * rays
        if goal_distance <= self.radius:
            points = np.vstack([self.goal, points])
        if not len(points):
            # Outside the surveyed box: approach its interior within the
            # radius until an in-bounds direction becomes available.
            toward = np.clip(self.goal, low, high) - p
            points = p + toward[None, :] * min(1., horizon / max(np.linalg.norm(toward), 1e-9))
        return points

    def _road_support(self, p, graph, rays, lengths, entry_allowed, reference=None):
        """Measure continuous road support, including reach outside the held corridor."""
        support = np.zeros(len(rays))
        outside = np.zeros(len(rays))
        if graph is None or not len(graph):
            return support, outside
        network = graph.graph
        roads = {}

        def neighbors(node):
            if node not in roads:
                roads[node] = []
                for other, edge in network[node].items():
                    weight = edge.get('weight')
                    if not edge.get('bridged') and isinstance(weight, Real) and 0 <= weight < np.inf:
                        roads[node].append((other, weight))
            return roads[node]

        # Seed all locally accessible road components with the actual entry
        # distance. With no vehicle observation only a coincident entry is known.
        entries = []
        for node, distance in graph.find_nearby_nodes(tuple(p), min(15., max(lengths))):
            if not neighbors(node):
                continue
            if distance > 1e-6 and (entry_allowed is None or not entry_allowed(graph.get_node_position(node))):
                continue
            entries.append((node, distance))
        if not entries:
            return support, outside

        # Every traversed edge stays within this heading's corridor. A branch
        # that leaves and later re-enters cannot supply evidence past the gap.
        # A little room behind the car also permits a local turn around a tree.
        for index, (ray, length) in enumerate(zip(rays, lengths)):
            coordinates = {}
            if reference is not None:
                along = reference @ ray
                across = reference[0] * ray[1] - reference[1] * ray[0]

            def credit(ahead, side, cost):
                value = ahead - abs(side) - max(0., cost - ahead)
                support[index] = max(support[index], value)
                if reference is not None:
                    forward = ahead * along - side * across
                    lateral = ahead * across + side * along
                    if forward < -self.corridor or abs(lateral) > self.corridor:
                        outside[index] = max(outside[index], value)

            def project(node):
                if node not in coordinates:
                    x, y = graph.get_node_position(node)[:2]
                    dx, dy = x - p[0], y - p[1]
                    coordinates[node] = (dx * ray[0] + dy * ray[1], dy * ray[0] - dx * ray[1])
                return coordinates[node]

            costs, queue = {}, []
            sequence = count()
            for node, distance in entries:
                ahead, side = project(node)
                if distance <= length and -self.corridor <= ahead <= length and abs(side) <= self.corridor:
                    costs[node] = distance
                    heappush(queue, (distance, next(sequence), node))

            while queue:
                cost, _, node = heappop(queue)
                if cost != costs[node]:
                    continue
                ahead, side = project(node)
                credit(ahead, side, cost)
                for other, weight in neighbors(node):
                    new_cost = cost + weight
                    if new_cost >= costs.get(other, np.inf):
                        continue
                    end, lateral = project(other)
                    if new_cost <= length and -self.corridor <= end <= length and abs(lateral) <= self.corridor:
                        costs[other] = new_cost
                        heappush(queue, (new_cost, next(sequence), other))
                        continue
                    # Count the continuous prefix of a long observed edge up
                    # to its first exit or the travel budget. Its far endpoint
                    # is available to the search only if the entire edge fits.
                    fraction = 1.
                    if new_cost > length:
                        fraction = (length - cost) / weight
                    if end > length:
                        fraction = min(fraction, (length - ahead) / (end - ahead))
                    elif end < -self.corridor:
                        fraction = min(fraction, (-self.corridor - ahead) / (end - ahead))
                    if abs(lateral) > self.corridor:
                        limit = np.copysign(self.corridor, lateral)
                        fraction = min(fraction, (limit - side) / (lateral - side))
                    end = ahead + fraction * (end - ahead)
                    lateral = side + fraction * (lateral - side)
                    partial_cost = cost + fraction * weight
                    credit(end, lateral, partial_cost)
        return support, outside

    def _approach_penalties(self, position, rays, support, turn_threshold, now):
        """Discount a failed departure only at its local entry and bearing."""
        penalties = np.zeros(len(rays))
        retained = []
        for failure in self.failures:
            distance = np.linalg.norm(position - failure.position)
            if distance <= self.penalty_radius:
                aligned = rays @ failure.direction >= turn_threshold
                # Require useful new reach beyond any gain that can be
                # explained by the car's movement within this entry area.
                if np.any(support[aligned] > failure.support + SUPPORT_GAIN_M + distance):
                    continue
                penalties[aligned] += self.radius * np.exp(-(now - failure.time) / self.penalty_decay)
            retained.append(failure)
        self.failures = retained
        return penalties

    def stats(self):
        return dict(subgoals_selected=float(self.selected), subgoal_arrivals=float(self.arrivals),
                    subgoal_failures=float(self.failed), subgoal_reviews=float(self.reviews))


# Existing callers use this name for the active vehicle guidance.
SubgoalManager = LookaheadGuidance
