"""Continuous, vehicle-relative angle actions that designate fixed map points."""
import math

import numpy as np

from simulator.survey_fleet_observation import START_CONNECTOR
from simulator.survey_fleet_runtime import GOAL_TOLERANCE_M
from ground.vtemp4_observation import build_observation, ring_points, valid_ring_points
from simulator.vtemp4_ring_contract import RING_RADIUS_M
from simulator.vtemp4_runtime import FINAL_APPROACH_M, HeadingFleetEpisode


class RingHeadingFleetEpisode(HeadingFleetEpisode):
    """A new angle is applied once to the measured vehicle pose at issue time."""

    def __init__(self, *args, lookahead_m=RING_RADIUS_M, **kwargs):
        if lookahead_m != RING_RADIUS_M:
            raise ValueError(f'Fixed ring navigation requires a {RING_RADIUS_M:g} m radius')
        super().__init__(*args, lookahead_m=lookahead_m, **kwargs)

    def _heading_observation(self):
        yaw = float(self.adapter.own['orientation'])
        obs = build_observation(self.adapter, yaw, self.clock_s - self.held_since,
                                self.previous_review_position, lookahead_m=self.lookahead_m, limit_by_goal=False)
        # The native ego block supplies the current yaw. The first context fields
        # describe whether there is a held point, and its current map XY offset.
        obs['context'][:3] = 0.
        if self.current_request is not None:
            delta = np.asarray(self.current_request['target']) - self.position()
            obs['context'][:3] = [1., *(delta / self.lookahead_m)]
        return obs

    def _choose_heading(self, now, position, distance):
        obs = self._heading_observation()
        with self.torch.inference_mode():
            turn, angle, logp, value = self.slow.act(
                obs, deterministic=self.rollout is None, generator=self.generator)
        angle = float(angle)
        if not math.isfinite(angle) or abs(angle) > float(np.float32(math.pi)) or (not turn and angle != 0.):
            raise ValueError('Expected a signed vehicle-relative turn or a zero-angle hold')
        angle = max(-math.pi, min(math.pi, angle))
        self._settle(now, distance, value)
        yaw = float(self.adapter.own['orientation'])
        old = self.current_request
        accepted = False
        reason = 'hold'
        if old is None or turn:
            # A first hold requests straight ahead. An invalid designation waits
            # for a new action, preserving the sampled action and its PPO logp.
            point = ring_points(position, yaw, [angle], self.lookahead_m)
            accepted = bool(valid_ring_points(self.adapter, point)[0])
            reason = 'invalid_proposal'
            if accepted:
                self._set_marker(now, point[0])
                if self.current_request is not old:
                    self.current_request.update(origin_xy=position.tolist(), vehicle_yaw_rad=yaw,
                                                relative_angle_rad=angle, radius_m=self.lookahead_m)
                    self.held_since = now
                reason = 'initial' if old is None else 'new_designation'
        request = self.current_request
        changed = request is not old
        event = dict(time_s=now, position=position.tolist(), remaining_m=distance, turn=bool(turn),
                     angle_rad=angle, vehicle_yaw_rad=yaw, changed=changed, accepted=accepted, reason=reason,
                     waypoint_id=None if request is None else request['id'],
                     target=None if request is None else list(request['target']))
        self.heading_reviews.append(event)
        self.pending = dict(start=now, distance=distance, observation=obs, turn=bool(turn), angle_rad=angle,
                            logp=float(logp), value=float(value), event=event)
        self.last_review = now
        self.previous_review_position = position.copy()

    def _after_fusion(self, graph, semantic_map):
        now = self.clock_s
        if self._last_control_time != now:
            self._last_control_time = now
            self._belief_cache = None
            self.adapter.observe_receipts()
            position = self.position()
            distance = float(np.linalg.norm(position - self.goal))
            if now < self.limit and distance > GOAL_TOLERANCE_M:
                finish = distance <= FINAL_APPROACH_M or (self.final_approach and distance <= self.lookahead_m)
                phase_changed = finish != self.final_approach
                self.final_approach = finish
                due = self.last_review is None or (not finish and now - self.last_review >= self.review_interval)
                if due:
                    self._choose_heading(now, position, distance)
                if due or phase_changed or now % self.ground_interval == 0:
                    request = self.current_request
                    marker = None if request is None else request['target']
                    # Local final approach can engage between decisions. The fleet
                    # broadcast still comes from the last accepted designation.
                    guidance = None if finish or marker is None else np.asarray(marker)
                    self._ground_decision(now, guidance=guidance, wait_for_guidance=marker is None and not finish)
                    self.ground_events[-1].update(shared_marker=None if marker is None else list(marker),
                                                  waypoint_id=None if request is None else request['id'],
                                                  final_approach=finish)
        filtered = self._native_filter(graph, semantic_map)
        return None if self.adapter.last_ground_reason == START_CONNECTOR else filtered
