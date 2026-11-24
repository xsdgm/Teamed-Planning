"""PPO selection of persistent world-coordinate guidance from vehicle-side evidence."""
from collections import deque
import math

import numpy as np
from scipy.ndimage import label

from survey.survey_fleet_observation import START_CONNECTOR, _clearance
from survey.survey_fleet_runtime import GOAL_TOLERANCE_M
from vtemp4.vtemp4_observation import build_observation, DIRECTION_COUNT, ring_points, valid_ring_points
from vtemp4.vtemp4_ring_contract import RING_RADIUS_M
from vtemp4.vtemp4_runtime import END_REWARD, FINAL_APPROACH_M, HeadingFleetEpisode, interval_reward


HISTORY_AGES_S = (60, 45, 30, 15)
REACHED_M = 4.
PROGRESS_M = 2.
STALL_S = 60.
RING_ANGLES = np.arange(2 * DIRECTION_COUNT) * (2 * math.pi / (2 * DIRECTION_COUNT))


def motion_history(samples, now, position, goal, active, lookahead, *, issued_at=0):
    """Car-relative past poses, signed closing distances and accumulated path length."""
    by_time = {stamp: (pose, travel) for stamp, pose, travel in samples}
    rows = np.zeros((len(HISTORY_AGES_S), 5), np.float32)
    mask = np.zeros(len(rows), bool)
    if now not in by_time:
        return rows, mask
    current_travel = by_time[now][1]
    for i, age in enumerate(HISTORY_AGES_S):
        if now - age in by_time:
            previous, travel = by_time[now - age]
            rows[i] = [*(previous - position),
                       np.linalg.norm(goal - previous) - np.linalg.norm(goal - position),
                       0. if active is None or now-age < issued_at else
                       np.linalg.norm(active - previous) - np.linalg.norm(active - position),
                       current_travel - travel]
            mask[i] = True
    return rows / lookahead, mask


def _last_approach(samples, target, issued_at, last_progress):
    """Time-stamp fresh closing movement, even before reclaiming an earlier best."""
    distances = {t: np.linalg.norm(target - xy) for t, xy, _ in samples if t >= issued_at}
    for t in reversed(distances):
        if t <= last_progress:
            break
        # A previous decrease may still be visible after the vehicle stops.
        # Attribute the credit to its motion tick, never to a later review.
        if t-1 not in distances or distances[t-1] - distances[t] < .01:
            continue
        for age in (15, 30, 60):
            if t-2 not in distances or not all(t-age+i in distances for i in range(3)):
                continue
            early = sum(distances[t-age+i] for i in range(3)) / 3.
            late = sum(distances[t-i] for i in range(3)) / 3.
            if early - late >= max(.5, age / 60.):
                return t
    return last_progress


def _occupied(adapter, point):
    """An endpoint is invalid only if onboard/received evidence rules out clearance."""
    xy = np.floor(np.asarray(point) / adapter.resolution).astype(int)
    if np.any(xy < 0) or np.any(xy >= adapter.cells.shape[::-1]):
        return True
    lo = np.maximum(0, xy - 3)
    hi = np.minimum(adapter.cells.shape[::-1], xy + 4)
    patch = _clearance(adapter.cells[lo[1]:hi[1], lo[0]:hi[0]])
    return bool(patch[xy[1]-lo[1], xy[0]-lo[0]] == 1)


def waypoint_options(adapter, position, endpoints, lookahead):
    """For each goal-relative ray, propose a near known point and a farther point/frontier.

    A proposal may request further observation of a frontier. It never authorizes
    ground motion: the native ground planner still validates each driven segment.
    """
    res = adapter.resolution
    size = adapter.cells.shape[::-1]
    lo = np.maximum(0, np.floor((position - lookahead) / res).astype(int) - 4)
    hi = np.minimum(size, np.ceil((position + lookahead) / res).astype(int) + 5)
    cells = _clearance(adapter.cells[lo[1]:hi[1], lo[0]:hi[0]])
    possible = label(cells != 1)[0]
    known = label(cells == -1)[0]
    source = np.floor(position/res).astype(int) - lo
    component = possible[source[1], source[0]]
    confirmed = known[source[1], source[0]]
    points = np.repeat(position[None, :], 2 * DIRECTION_COUNT, axis=0)
    mask = np.zeros(len(points), bool)
    for i, endpoint in enumerate(endpoints):
        delta = endpoint - position
        extent = np.linalg.norm(delta)
        if extent <= REACHED_M + res:
            continue
        lengths = np.linspace(0., extent, math.ceil(extent / res) + 1)
        ray = position + lengths[:, None] * (delta / extent)
        indices = np.floor(ray / res).astype(int) - lo
        inside = np.all((indices >= 0) & (indices < cells.shape[::-1]), axis=1)
        indices = np.clip(indices, 0, np.asarray(cells.shape[::-1]) - 1)
        xy = (indices[:, 1], indices[:, 0])
        allowed = inside & (lengths > REACHED_M + res) & (cells[xy] != 1)
        if component:
            allowed &= possible[xy] == component
        accessible = allowed & (known[xy] == confirmed) & (confirmed != 0)
        # If the car is on the edge of confirmed clearance, its own native
        # start connector may reattach it. An unknown point stays a survey request.
        near = np.flatnonzero(accessible & (lengths <= extent*.5))
        far = np.flatnonzero(accessible)
        reach = lengths[far[-1]] if len(far) else 0.
        frontier = np.flatnonzero(allowed & (cells[xy] == 0) & (lengths > reach)
                                  & (lengths <= reach + min(25., extent * .5)))
        if not len(near):
            near = far[:1] if len(far) else frontier[:1]
        if len(frontier):
            far = frontier
        for j, choices in enumerate((near, far)):
            if len(choices):
                q = ray[choices[-1]]
                if j == 0 or not mask[2*i] or np.linalg.norm(q - points[2*i]) > res:
                    points[2*i+j] = q
                    mask[2*i+j] = True
    return points, mask


class WaypointProgress:
    """Track the vehicle's measured movement toward its currently designated map point."""

    def __init__(self, *args, **kwargs):
        self.samples = deque(maxlen=max(HISTORY_AGES_S) + 1)
        self.last_progress_s = 0
        self.best_distance = None
        self.best_route = None
        self.route_history = deque()
        self.waiting_for_evidence = False
        super().__init__(*args, **kwargs)

    def _record_pose(self, now, position):
        if self.samples and self.samples[-1][0] == now:
            return
        travel = 0. if not self.samples else self.samples[-1][2] + np.linalg.norm(position - self.samples[-1][1])
        self.samples.append((now, position.copy(), travel))

    def _target(self):
        return None if self.current_request is None else np.asarray(self.current_request['target'], dtype=float)

    def _review_reason(self, now, position, active):
        if active is None or self.waiting_for_evidence:
            return 'initial' if active is None else 'awaiting_evidence'
        distance = float(np.linalg.norm(position - active))
        if distance <= REACHED_M:
            return 'reached'
        if _occupied(self.adapter, active):
            return 'blocked'
        if self.best_distance is None or distance <= self.best_distance - PROGRESS_M:
            self.best_distance = distance
            self.last_progress_s = now
        # Follow progress around a measured bend even while Euclidean distances
        # temporarily increase. Only a complete measured route is comparable.
        lo, _, path, _, _, reason, _ = self.adapter._plan(active)
        while self.route_history and now - self.route_history[0][0] > STALL_S:
            self.route_history.popleft()
        if reason == 'known_goal_route':
            endpoints = (np.asarray(path)[[0, -1]] + lo)[:, ::-1] * self.adapter.resolution + .5*self.adapter.resolution
            length = (len(path) - 1) * self.adapter.resolution + np.linalg.norm(position - endpoints[0]) + np.linalg.norm(active - endpoints[1])
            if not self.route_history or self.route_history[-1][0] != now:
                fresh_decrease = self.route_history and self.route_history[-1][1] - length >= .25
                if fresh_decrease and any(now-t >= self.review_interval and old-length >= PROGRESS_M
                                          for t, old in self.route_history):
                    self.last_progress_s = now
                self.route_history.append((now, length))
            if self.best_route is None or length <= self.best_route - PROGRESS_M:
                self.best_route = length
                self.last_progress_s = now
        self.last_progress_s = _last_approach(self.samples, active, self.current_request['issued_at_s'],
                                              self.last_progress_s)
        return 'stalled' if now - self.last_progress_s >= STALL_S else 'tracking'

    def _reset_progress(self, now, distance):
        self.best_distance = distance
        self.best_route = None
        self.route_history.clear()
        self.last_progress_s = now


class WaypointFleetEpisode(WaypointProgress, HeadingFleetEpisode):
    """A held point has one broadcast ID and one immutable world-frame location."""

    body_ring = False

    def __init__(self, *args, **kwargs):
        self.waypoint_reviews = []
        super().__init__(*args, **kwargs)

    def _options(self, position):
        # None anchors all 24 rays at the current final-goal bearing. The
        # heading-policy displacement slots are zero in both waypoint arms.
        obs = build_observation(self.adapter, None, 0., None, lookahead_m=self.lookahead_m)
        world, valid = waypoint_options(self.adapter, position, obs['targets'][:DIRECTION_COUNT], self.lookahead_m)
        return obs, world, valid

    def _waypoint_observation(self, reason):
        position = self.position()
        active = self._target()
        obs, world, valid = self._options(position)
        if active is not None:
            valid &= np.linalg.norm(world - active, axis=1) > self.adapter.resolution
        if reason == 'tracking':
            valid[:] = False
        can_hold = reason in ('tracking', 'stalled') or not valid.any()
        history, mask = motion_history(self.samples, self.clock_s, position, self.goal, active, self.lookahead_m,
                                      issued_at=0 if self.current_request is None else self.current_request['issued_at_s'])
        obs.update(waypoint_targets=world, waypoint_xy=((world-position)/self.lookahead_m).astype(np.float32),
                   waypoint_mask=valid, can_hold=bool(can_hold), active_waypoint_delta=(
                       np.zeros(2, np.float32) if active is None else ((active-position)/self.lookahead_m).astype(np.float32)),
                   history=history, history_mask=mask)
        return obs

    def _heading_observation(self):
        # Bootstrap a truncated rollout using the same masks as an ordinary review.
        now, position = self.clock_s, self.position()
        self._record_pose(now, position)
        return self._waypoint_observation(self._review_reason(now, position, self._target()))

    def _review(self, now, position, distance):
        old = self._target()
        reason = self._review_reason(now, position, old)
        obs = self._waypoint_observation(reason)
        with self.torch.inference_mode():
            replace, index, logp, value = self.slow.act(obs, deterministic=self.rollout is None,
                                                        generator=self.generator)
        index = int(index)
        if replace:
            if not (0 <= index < len(obs['waypoint_mask']) and obs['waypoint_mask'][index]):
                raise ValueError('Selected an unavailable map waypoint')
            target = obs['waypoint_targets'][index].copy()
        else:
            if not obs['can_hold'] or index != -1:
                raise ValueError('Invalid hold/index for a map waypoint')
            # Legacy waypoint v1 used the current pose as an empty-options fallback.
            # The ring contract permits a new broadcast only on explicit selection.
            target = old if self.body_ring or (old is not None and reason != 'blocked') else position.copy()
        self._settle(now, distance, value)
        fallback = not replace and reason in ('initial', 'blocked', 'reached', 'awaiting_evidence')
        changed = target is not None and (old is None or not np.allclose(target, old, atol=1.e-7, rtol=0.))
        angle = 0.
        if replace and old is not None:
            before, after = old-position, target-position
            angle = math.remainder(math.atan2(after[1], after[0]) - math.atan2(before[1], before[0]), 2*math.pi)
        if changed:
            self._set_marker(now, target)
            if self.body_ring:
                self.current_request.update(origin_xy=position.tolist(), vehicle_yaw_rad=obs['vehicle_yaw_rad'],
                                            relative_angle_rad=float(RING_ANGLES[index]), radius_m=self.lookahead_m)
            self._reset_progress(now, float(np.linalg.norm(target-position)))
        self.waiting_for_evidence = fallback
        event = dict(time_s=now, position=position.tolist(), remaining_m=distance, reason=reason,
                     replace=bool(replace), changed=changed, point_index=index,
                     waypoint_id=None if self.current_request is None else self.current_request['id'],
                     target=None if target is None else target.tolist(),
                     distance_to_waypoint_m=None if target is None else float(np.linalg.norm(target-position)),
                     switch_angle_rad=angle)
        self.waypoint_reviews.append(event)
        self.pending = dict(start=now, distance=distance, observation=obs, replace=bool(replace), point_index=index,
                            switch=bool(replace and changed and old is not None), angle=angle,
                            logp=float(logp), value=float(value), event=event)
        self.last_review = now

    def _after_fusion(self, graph, semantic_map):
        now = self.clock_s
        if self._last_control_time != now:
            self._last_control_time = now
            self._belief_cache = None
            self.adapter.observe_receipts()
            position = self.position()
            self._record_pose(now, position)
            distance = float(np.linalg.norm(position-self.goal))
            if now < self.limit and distance > GOAL_TOLERANCE_M:
                finish = distance <= min(FINAL_APPROACH_M, self.lookahead_m) or (
                    self.final_approach and distance <= self.lookahead_m)
                phase_change = finish != self.final_approach
                self.final_approach = finish
                due = not finish and (self.last_review is None or now-self.last_review >= self.review_interval)
                if due:
                    self._review(now, position, distance)
                if finish and not self.body_ring:
                    self._set_marker(now, self.goal)
                if due or phase_change or now % self.ground_interval == 0:
                    marker = self._target()
                    wait = self.body_ring and not finish and (marker is None or self.waiting_for_evidence)
                    if wait:
                        self._ground_decision(now, wait_for_guidance=True)
                    else:
                        self._ground_decision(now, guidance=None if finish else marker)
                    self.ground_events[-1].update(shared_marker=None if marker is None else marker.tolist(),
                                                  final_approach=finish, waypoint_id=None if self.current_request is None
                                                  else self.current_request['id'])
        filtered = self._native_filter(graph, semantic_map)
        return None if self.adapter.last_ground_reason == START_CONNECTOR else filtered

    def _settle(self, now, distance, next_value, reason=None):
        p = self.pending
        if p is None:
            return
        duration = now - p['start']
        reward = interval_reward(p['distance'], distance, self.case['direct_distance_m'], p['switch'], p['angle'],
                                 duration, self.budget, reason)
        self.heading_return += reward
        p['event'].update(duration_s=duration, reward=reward, end_reason=reason)
        if self.rollout is not None:
            self.rollout.append(dict(observation=p['observation'], replace=p['replace'], point_index=p['point_index'],
                                     reward=reward, logp=p['logp'], value=p['value'], next_value=float(next_value),
                                     terminated=reason in END_REWARD, truncated=reason == 'collector_limit',
                                     duration=duration))
        self.pending = None

    def fleet_summary(self):
        summary = super().fleet_summary()
        summary.pop('heading_reviews')
        summary.pop('turn_count')
        return dict(summary, waypoint_reviews=self.waypoint_reviews, waypoint_count=len(self.requests),
                    waypoint_replacements=sum(r['changed'] and r['replace'] and r['waypoint_id'] not in (None, 0)
                                              for r in self.waypoint_reviews),
                    link_violation_steps=self.link_violations)


class RingWaypointFleetEpisode(WaypointFleetEpisode):
    """Select a world point on the vehicle's 200 m ring and hold it until reissued."""

    body_ring = True

    def __init__(self, *args, lookahead_m=RING_RADIUS_M, **kwargs):
        if lookahead_m != RING_RADIUS_M:
            raise ValueError('Vehicle-relative waypoint selection requires a 200 m ring')
        super().__init__(*args, lookahead_m=lookahead_m, **kwargs)

    def _options(self, position):
        yaw = float(self.adapter.own['orientation'])
        obs = build_observation(self.adapter, yaw, 0., None, lookahead_m=self.lookahead_m, limit_by_goal=False)
        obs['vehicle_yaw_rad'] = yaw
        world = ring_points(position, yaw, RING_ANGLES, self.lookahead_m)
        return obs, world, valid_ring_points(self.adapter, world)
