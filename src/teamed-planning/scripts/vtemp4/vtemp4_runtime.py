"""Execute learned lookahead bearings in the frozen three-aircraft V9 simulator."""
from __future__ import annotations

import math

import numpy as np

from survey.survey_fleet_observation import START_CONNECTOR
from survey.survey_fleet_runtime import GOAL_TOLERANCE_M, SurveyFleetEpisode
from vtemp4.vtemp4_observation import build_observation


FINAL_APPROACH_M = 25.
DEFAULT_PROGRESS_REWARD_WEIGHT = 2.
END_REWARD = {'arrival': 10., 'safety_failure': -5., 'time_budget': 0.}


def lookahead_marker(position, heading, goal, radius, world_size):
    """Roll a point along the held world bearing, shortening its ray at the map edge."""
    direction = np.array([math.cos(heading), math.sin(heading)])
    distance = min(radius, float(np.linalg.norm(goal - position)))
    for coordinate, component in zip(position, direction):
        if abs(component) > 1.e-12:
            boundary = world_size if component > 0 else 0.
            distance = min(distance, max(0., (boundary - coordinate) / component))
    return position + distance * direction


def interval_reward(d_before, d_after, direct_distance, turn, angle_rad, duration, budget, end_reason=None,
                    *, progress_reward_weight=DEFAULT_PROGRESS_REWARD_WEIGHT):
    turn_metres = 0.5 + abs(angle_rad) / math.pi if turn else 0.
    distance_reward = (progress_reward_weight * (d_before - d_after) - 2. * turn_metres) / direct_distance
    return distance_reward - duration / budget + END_REWARD.get(end_reason, 0.)


class HeadingFleetEpisode(SurveyFleetEpisode):
    """A slow action controls the shared bearing. The local UGV planner executes it."""

    progress_reward_weight = DEFAULT_PROGRESS_REWARD_WEIGHT
    _route_field = None

    def __init__(self, repo, case, model, *, fast_seed, budget_s, review_interval_s=15,
                 lookahead_m=200., rollout=None, collector_limit_s=None,
                 progress_reward_weight=DEFAULT_PROGRESS_REWARD_WEIGHT, progress_distance='euclidean'):
        if not isinstance(review_interval_s, int) or review_interval_s < 1:
            raise ValueError('Review interval must be a positive number of whole simulation seconds')
        if not math.isfinite(lookahead_m) or lookahead_m <= 0 or case['direct_distance_m'] <= 0:
            raise ValueError('Lookahead and mission distance must be positive')
        if not math.isfinite(progress_reward_weight) or progress_reward_weight < 0:
            raise ValueError('Progress reward weight must be finite and nonnegative')
        if progress_distance == 'traversable':
            from vtemp4.vtemp4_route_distance import RouteDistanceField
            self._route_field = RouteDistanceField(case)
        elif progress_distance != 'euclidean':
            raise ValueError(f'Unknown progress distance {progress_distance!r}')
        self.progress_reward_weight = progress_reward_weight
        self.review_interval = review_interval_s
        self.lookahead_m = lookahead_m
        self.heading = None
        self.held_since = 0
        self.previous_review_position = None
        self.final_approach = False
        self.heading_reviews = []
        self.heading_return = 0.
        # The parent prepares the first native action; _after_fusion is already
        # active on that first call so time zero uses the same data flow as later reviews.
        super().__init__(repo, case, model, fast_seed=fast_seed, budget_s=budget_s,
                         rollout=rollout, collector_limit_s=collector_limit_s)
        if self._route_field is not None:
            self._route_initial_m = self._route_field.distance_m(self.position())
            if self._route_initial_m is None:
                self.close()
                raise ValueError(f"Vehicle start cannot reach the route field in {case['case_id']}")
            self._route_last_raw_m = self._route_last_scored_m = self._route_initial_m
            self._route_unavailable_intervals = 0

    def _heading_observation(self):
        return build_observation(self.adapter, self.heading, self.clock_s - self.held_since,
                                 self.previous_review_position, lookahead_m=self.lookahead_m)

    def _after_fusion(self, graph, semantic_map):
        now = self.clock_s
        if self._last_control_time != now:
            self._last_control_time = now
            self._belief_cache = None
            self.adapter.observe_receipts()
            position = self.position()
            distance = float(np.linalg.norm(position - self.goal))
            if now < self.limit and distance > GOAL_TOLERANCE_M:
                finishing = distance <= min(FINAL_APPROACH_M, self.lookahead_m) or (
                    self.final_approach and distance <= self.lookahead_m)
                changed_phase = finishing != self.final_approach
                self.final_approach = finishing
                due = not finishing and (self.last_review is None or now - self.last_review >= self.review_interval)
                if due:
                    self._choose_heading(now, position, distance)
                if due or changed_phase or now % self.ground_interval == 0:
                    marker = self.goal.copy() if finishing else lookahead_marker(
                        position, self.heading, self.goal, self.lookahead_m, self.case['world_size'])
                    self._set_marker(now, marker)
                    self._ground_decision(now, guidance=None if finishing else marker)
                    self.ground_events[-1]['shared_marker'] = marker.tolist()
                    self.ground_events[-1]['final_approach'] = finishing
        filtered = self._native_filter(graph, semantic_map)
        return None if self.adapter.last_ground_reason == START_CONNECTOR else filtered

    def _choose_heading(self, now, position, distance):
        observation = self._heading_observation()
        with self.torch.inference_mode():
            turn, angle, logp, value = self.slow.act(
                observation, deterministic=self.rollout is None, generator=self.generator)
        turn, angle = bool(turn), float(angle)
        if not math.isfinite(angle) or abs(angle) > float(np.float32(math.pi)) or (not turn and angle != 0.):
            raise ValueError('Expected hold with zero angle or a signed turn within a half circle')
        angle = max(-math.pi, min(math.pi, angle))
        self._settle(now, distance, value)
        reference = self.heading
        if reference is None:
            offset = self.goal - position
            reference = math.atan2(offset[1], offset[0])
        if self.heading is None or turn:
            self.held_since = now
        self.heading = math.atan2(math.sin(reference + angle), math.cos(reference + angle))
        event = dict(time_s=now, position=position.tolist(), remaining_m=distance,
                     previous_heading_rad=reference, turn=turn, angle_rad=angle, heading_rad=self.heading)
        self.heading_reviews.append(event)
        self.pending = dict(start=now, distance=distance, observation=observation, turn=turn, angle_rad=angle,
                            logp=float(logp), value=float(value), event=event)
        self.last_review = now
        self.previous_review_position = position.copy()

    def _set_marker(self, now, marker):
        # SurveyFleetEpisode.step verifies the actual physics broadcast and
        # records who heard it. A forward roll does not start a new slow action.
        if self.current_request is None or not np.allclose(self.current_request['target'], marker, rtol=0, atol=1.e-7):
            self.current_request = dict(id=len(self.requests), issued_at_s=now, target=marker.tolist(),
                                        physics_broadcasts=0, first_physics_broadcast_s=None, heard_by={})
            self.requests.append(self.current_request)

    def _reward_distances(self, pending, distance):
        return pending['distance'], distance

    def _settle(self, now, distance, next_value, reason=None):
        p = self.pending
        if p is None:
            return
        duration = now - p['start']
        turn, turn_angle = p.get('turn_for_reward', (p['turn'], p['angle_rad']))
        charge_turn = turn and p.get('policy_controlled', True)
        before, after = self._reward_distances(p, distance)
        if self._route_field is not None:
            raw_after = self._route_field.distance_m(self.position())
            before = self._route_last_scored_m
            # Hold the last connected potential through a gap. Re-entering the
            # connected graph then credits only the net change between known poses.
            after = before if raw_after is None else raw_after
            p['event'].update(route_remaining_start_m=self._route_last_raw_m,
                              route_remaining_end_m=raw_after, route_progress_m=before-after)
            self._route_unavailable_intervals += int(self._route_last_raw_m is None or raw_after is None)
            self._route_last_raw_m, self._route_last_scored_m = raw_after, after
        reward = interval_reward(before, after, self.case['direct_distance_m'], charge_turn,
                                 turn_angle, duration, self.budget, reason,
                                 progress_reward_weight=self.progress_reward_weight)
        if 'waypoint_progress_score' in p['event']:
            vehicle_gain = max(0., p['distance'] - distance)
            progress_reward = self.progress_reward_weight * vehicle_gain / self.case['direct_distance_m']
            bonus = self.waypoint_progress_bonus_ratio * p['event']['waypoint_progress_score'] * progress_reward
            p['event']['waypoint_progress_bonus'] = bonus
            reward += bonus
        self.heading_return += reward
        p['event'].update(duration_s=duration, reward=reward, end_reason=reason)
        if self.rollout is not None:
            row = dict(observation=p['observation'], turn=p['turn'], angle_rad=p['angle_rad'],
                       logp=p['logp'], value=p['value'], reward=reward, next_value=float(next_value),
                       terminated=reason in END_REWARD, truncated=reason == 'collector_limit', duration=duration)
            if 'policy_controlled' in p:
                row['policy_controlled'] = p['policy_controlled']
            self.rollout.append(row)
        self.pending = None

    def _end(self, reason):
        self.end_reason = reason
        self.finished = True
        self.terminated = reason in END_REWARD
        self.truncated = reason == 'collector_limit'
        if not (self.terminated or self.truncated):
            raise ValueError(f'Unexpected heading episode outcome {reason}')
        next_value = 0.
        if self.truncated and self.rollout is not None and self.pending is not None:
            with self.torch.inference_mode():
                next_value = self.slow.value(self._heading_observation())
        self._settle(self.clock_s, float(np.linalg.norm(self.position() - self.goal)), next_value, reason)

    def fleet_summary(self):
        route_metrics = {}
        if self._route_field is not None:
            route_metrics = dict(route_initial_m=self._route_initial_m, route_remaining_m=self._route_last_raw_m,
                                 route_progress_m=self._route_initial_m-self._route_last_scored_m,
                                 route_unavailable_intervals=self._route_unavailable_intervals)
        return dict(case_id=self.case['case_id'], seed=self.case['seed'], end_reason=self.end_reason,
                    goal_reached=self.end_reason == 'arrival', sim_time_s=self.clock_s, budget_s=self.budget,
                    direct_distance_m=self.case['direct_distance_m'], travelled_m=self.travelled_m,
                    remaining_m=float(np.linalg.norm(self.position() - self.goal)), reward=self.heading_return,
                    heading_reviews=self.heading_reviews, turn_count=sum(row['turn'] for row in self.heading_reviews),
                    ground_replan_events=self.ground_events, shared_markers=self.requests,
                    ground_received_packets=self.ground_packets, fusion_count=self.env.ground_controller.graphs_received,
                    adapter=self.adapter.stats(), ground_motion_guard=self.motion_guard.stats(),
                    collisions=self.collisions, safety_failure_details=self.safety_failure_details, **route_metrics)


def run_heading_case(repo, case, model, *, fast_seed, budget_s, review_interval_s=15, lookahead_m=200.,
                     rollout=None, collector_limit_s=None):
    episode = HeadingFleetEpisode(repo, case, model, fast_seed=fast_seed, budget_s=budget_s,
                                  review_interval_s=review_interval_s, lookahead_m=lookahead_m,
                                  rollout=rollout, collector_limit_s=collector_limit_s)
    try:
        while not episode.finished:
            episode.step()
        return episode.fleet_summary()
    finally:
        episode.close()
