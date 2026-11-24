"""Run independent ground and survey targets through the frozen V9 fleet simulator."""
from __future__ import annotations

from copy import copy, deepcopy

import numpy as np

from vtemp2_3.ground_policy import ground_sensor_tile
from survey.survey_fleet_clearance import native_segment_clear
from shared.simulator.ground_evidence import GroundPolicyEpisode
from vtemp2_3.hierarchical_safety import failure_diagnostics
from survey.survey_fleet_drive import GroundMotionGuard
from survey.survey_fleet_metrics import use_bounded_task_mask
from survey.survey_fleet_observation import FleetSurveyAdapter, START_CONNECTOR


GAMMA = .9995
GOAL_TOLERANCE_M = 2.


def _point(value):
    return np.asarray(value, dtype=float)[:2]


def _same(a, b):
    return a is not None and b is not None and np.allclose(a, b, rtol=0., atol=1.e-7)


def _survey_reason(now, current, last_review, answered, minimum, timeout):
    if current is None:
        return 'idle'
    age = now - last_review
    if age >= timeout:
        return 'timeout'
    if age >= minimum and answered:
        return 'evidence'
    return None


class _DriveRoadGraph:
    """Keep native graph filtering and copies local to this vehicle's planning call."""

    def __init__(self, graph, episode):
        object.__setattr__(self, '_graph', graph)
        object.__setattr__(self, '_episode', episode)

    def __getattr__(self, name):
        return getattr(self._graph, name)

    def __setattr__(self, name, value):
        setattr(self._graph, name, value)

    def __len__(self):
        return len(self._graph)

    def __copy__(self):
        # The native driver prunes observed obstacles from a shallow graph copy.
        # Its planning and confidence checks must both see that pruned copy.
        return type(self)(copy(self._graph), self._episode)

    def edge_confidence_from_positions(self, start, target):
        original = self._graph.edge_confidence_from_positions(start, target)
        return self._episode._road_entry_confidence(self._graph, start, target, original)


class SurveyFleetEpisode(GroundPolicyEpisode):
    ground_control = 'vtemp4'

    def __init__(self, repo, case, model, *, fast_seed, budget_s, ground_interval_s=5,
                 survey_check_s=5, survey_min_s=20, survey_timeout_s=60, rollout=None,
                 collector_limit_s=None):
        if (min(ground_interval_s, survey_check_s, survey_min_s, budget_s) < 1
                or survey_timeout_s < survey_min_s
                or collector_limit_s is not None and collector_limit_s < 1):
            raise ValueError('Invalid drive, survey or episode clocks')
        super().__init__(repo, case, fast_seed=fast_seed, rollout_limit=budget_s, task_context=False)
        self.budget = budget_s
        self.limit = min(collector_limit_s, budget_s) if collector_limit_s is not None else budget_s
        self.ground_interval = ground_interval_s
        self.survey_check = survey_check_s
        self.survey_min = survey_min_s
        self.survey_timeout = survey_timeout_s
        self.rollout = rollout
        self.slow = model
        self.generator = self.torch.Generator().manual_seed(int(fast_seed) + 1_000_003)
        self.current_request = None
        self.last_review = None
        self.pending = None
        self.adapter = None
        self.end_reason = 'running'
        self.received_neighbors = []
        self.ground_events = []
        self.survey_reviews = []
        self.requests = []
        self.travelled_m = 0.
        self.survey_return = 0.
        self.undiscounted_evidence = 0.
        self.air_distances = np.zeros(len(self.env.ids))
        self.air_samples = []
        self.minimum_separation = None
        self.link_violations = 0
        self.ground_packets = 0
        controller = self.env.ground_controller
        self._original_filter = controller._filter_observed_obstacles
        self._last_control_time = None
        try:
            if self.env.env.config.dt != 1.:
                raise ValueError('Survey clocks require one-second frozen V9 physics')
            use_bounded_task_mask(self.env)
            self.adapter = FleetSurveyAdapter(self, model)
            self.motion_guard = GroundMotionGuard(self.adapter)
            # The archived get_action calls this boundary just after real graph
            # fusion and local sensing, before it computes the next vehicle motion.
            # Keep its single invocation and its original perception/planning.
            controller._filter_observed_obstacles = self._after_fusion
            self.env.ground_action = self.env._ground_action([])
        except BaseException:
            self.close()
            raise

    @property
    def clock_s(self):
        return int(self.env.env.time)

    def position(self):
        return _point(self.env.obs[self.env.ground]['self']['position'])

    def _native_filter(self, graph, semantic_map):
        filtered = self._original_filter(graph, semantic_map)
        return _DriveRoadGraph(filtered, self) if filtered is not None else None

    def _road_entry_confidence(self, graph, start, target, confidence):
        driver = self.env.ground_controller
        planner = driver.global_planner
        if (confidence > driver.edge_confidence_threshold or not planner.using_graph
                or planner.current_waypoint_idx != 1 or planner.start_connection_idx is not None):
            return confidence
        path = planner.current_path
        if (not path or len(path) < 2 or not np.array_equal(start, path[0])
                or not np.array_equal(target, path[1])):
            return confidence
        destination = graph.find_nearest_node(target)
        if destination is None or destination[1] > 1.e-6:
            return confidence
        # A new route (including a shorter candidate adopted at map fusion)
        # can omit a road entry already inside the native arrival radius.
        car = np.asarray(driver.position)
        radius = driver.waypoint_threshold
        if np.linalg.norm(car-start) > radius:
            return confidence
        obs = self.env.obs[self.env.ground]
        for node, distance in sorted(graph.find_nearby_nodes(car, radius), key=lambda item: item[1]):
            entry = graph.get_node_position(node)
            if (distance > radius or np.array_equal(entry, start)
                    or not graph.graph.has_edge(node, destination[0])):
                continue
            actual = graph.edge_confidence_from_positions(entry, target)
            if (actual > driver.edge_confidence_threshold
                    and native_segment_clear(ground_sensor_tile(obs), car, obs['map_resolution'], car, entry)):
                # This entry was reached already: keep the same next waypoint.
                path[0] = entry
                return actual
        return confidence

    def _ground_action(self, neighbors=None):
        # These are the ground receiver's native drop-filtered packets for this step.
        self.received_neighbors = [] if neighbors is None else neighbors
        action = super()._ground_action(neighbors)
        if self.ground_control == 'vtemp4':
            action['motion'] = self.motion_guard.motion(action['motion'])
        action['broadcast']['target'] = self.current_request['target'] if self.current_request is not None else None
        return action

    def _after_fusion(self, graph, semantic_map):
        now = self.clock_s
        if self._last_control_time != now:
            self._last_control_time = now
            self._belief_cache = None
            self.adapter.observe_receipts()
            self._accrue(now)
            if now < self.limit and np.linalg.norm(self.position() - self.goal) > GOAL_TOLERANCE_M:
                if now % self.ground_interval == 0:
                    self._ground_decision(now)
                if now % self.survey_check == 0:
                    reason = _survey_reason(now, self.current_request, self.last_review,
                                            self.adapter.survey_answered, self.survey_min, self.survey_timeout)
                    if reason:
                        self._survey_decision(now, reason)
        filtered = self._native_filter(graph, semantic_map)
        # A certified local start segment already specifies the direct waypoint.
        # Continue fusing native graphs, but do not route this segment via them.
        return None if self.adapter.last_ground_reason == START_CONNECTOR else filtered

    def _ground_decision(self, now, guidance=None, *, wait_for_guidance=False):
        position = self.position()
        old = self.manager.target.copy() if self.manager.selected else None
        previous_reason = self.adapter.last_ground_reason
        if wait_for_guidance:
            target, reason = position.copy(), 'waiting_for_guidance'
            self.adapter.ground_guidance = None
            self.adapter.ground_target = target.copy()
            self.adapter.ground_path_xy = target[None, :].copy()
            self.adapter.last_ground_reason = reason
        else:
            target, reason = self.adapter.plan_ground(guidance)
        q = _point(target)
        if not np.isfinite(q).all():
            raise ValueError('Nonfinite planned drive target')
        self.manager.target = q.copy()
        self.manager.origin = position.copy()
        self.manager.final = _same(q, self.goal)
        self.manager.selected += 1
        self.holding = bool(np.linalg.norm(position - q) < 1.e-6 or reason == 'hold')
        controller = self.env.ground_controller
        mode_changed = (reason == START_CONNECTOR) != (previous_reason == START_CONNECTOR)
        if mode_changed or not _same(q, controller.goal_position):
            controller._drive_to(tuple(q))
        self.ground_events.append(dict(time_s=now, position=position.tolist(), target=q.tolist(),
                                       previous_target=None if old is None else old.tolist(), reason=reason,
                                       previous_target_reached=None if old is None else bool(
                                           np.linalg.norm(position-old) <= controller.waypoint_threshold)))

    def _accrue(self, now):
        if self.pending is None or self.pending['last_s'] >= now:
            return
        pending = self.pending
        credit = self.adapter.survey_credit()
        gain = float(credit['reward_evidence']) - pending['credit']
        duration = now - pending['last_s']
        if duration != 1:
            raise RuntimeError('Fleet evidence accounting missed a physics second')
        value = gain - .001 * duration
        pending['reward'] += GAMMA ** (pending['last_s'] - pending['start']) * value
        pending['credit'] += gain
        pending['last_s'] = now
        pending['event'].update(reward_evidence_total=pending['credit'],
                                useful_new_cells=int(credit['useful_new_cells']),
                                resolved_question=bool(credit['resolved_question']),
                                answer=credit['answer'])
        evidence = self.adapter.last_survey_evidence
        if evidence is not None and self.current_request is not None:
            if self.current_request['first_evidence'] is None:
                self.current_request['first_evidence'] = deepcopy(evidence)
            if credit['resolved_question'] and self.current_request['first_answer'] is None:
                self.current_request['first_answer'] = deepcopy(evidence)
        self.undiscounted_evidence += gain
        self.survey_return += value

    def _finish_choice(self, now, next_value, terminated, truncated):
        pending = self.pending
        if pending is None:
            return
        duration = now - pending['start']
        pending['event'].update(duration_s=duration, discounted_reward=pending['reward'])
        if self.rollout is not None and duration > 0 and pending['eligible']:
            row = pending['choice']
            observation = row['observation']
            self.rollout.append(dict(
                **{k: np.asarray(observation[k]).copy() for k in ('features', 'ground_inputs', 'mask', 'present')},
                action=int(row['action']), logp=float(row['logp']), value=float(row['value']),
                next_value=float(next_value), reward=float(pending['reward']), duration=duration,
                terminated=bool(terminated), truncated=bool(truncated),
            ))
        self.pending = None

    def _survey_decision(self, now, reason):
        prior = self.current_request
        target = self.adapter.select_survey(deterministic=self.rollout is None, generator=self.generator)
        choice = self.adapter.last_choice
        eligible = target is not None and bool(np.count_nonzero(choice['observation']['mask']) > 1)
        self._finish_choice(now, choice['value'], False, not eligible)
        self.last_review = now
        q = None if target is None else _point(target)
        if q is not None and not np.isfinite(q).all():
            raise ValueError('Nonfinite selected survey target')
        if q is None:
            self.current_request = None
        elif prior is None or not _same(q, prior['target']):
            self.record_survey_dispatch(q, now)
            self.current_request = dict(id=len(self.requests), issued_at_s=now, target=q.tolist(),
                                        physics_broadcasts=0, first_physics_broadcast_s=None,
                                        heard_by={}, first_evidence=None, first_answer=None)
            self.requests.append(self.current_request)
        event = dict(time_s=now, reason=reason, ground_target=self.manager.target.tolist(),
                     survey_target=None if q is None else q.tolist(),
                     request_id=None if self.current_request is None else self.current_request['id'],
                     same_request=prior is not None and self.current_request is prior,
                     gap_count=int(choice['gap_count']), route_relevance=float(choice['route_relevance']),
                     feedback_prior=prior is not None and prior['first_evidence'] is not None)
        self.survey_reviews.append(event)
        self.pending = dict(choice=choice, event=event, eligible=eligible, start=now,
                            last_s=now, credit=0., reward=0.)

    def _observe_heard(self, action_time):
        for drone in self.env.controllers:
            chain = drone.navigator.chain
            task = chain.task()
            if task is None:
                continue
            stamp = chain.mission_time
            for request in reversed(self.requests):
                if request['issued_at_s'] <= stamp and _same(request['target'], task['target']):
                    request['heard_by'].setdefault(drone.drone_id, dict(actor_time_s=action_time,
                                                                       source_time_s=float(stamp)))
                    break

    def _end(self, reason):
        self.end_reason = reason
        self.finished = True
        terminal = reason in ('arrival', 'time_budget')
        self.terminated, self.truncated = terminal, not terminal
        next_value = 0.
        if self.rollout is not None and self.pending is not None and not terminal:
            observation, _, _ = self.adapter.survey_observation()
            next_value = self.slow.value(observation)
        self._finish_choice(self.clock_s, next_value, terminal, not terminal)

    def step(self):
        if self.finished:
            raise RuntimeError('Fleet episode already ended')
        before = self.position().copy()
        positions = np.stack([_point(self.env.obs[r]['self']['position']) for r in self.env.ids])
        now = self.clock_s
        request = self.current_request
        prepared = self.env.ground_action['broadcast']['target']
        if (request is None and prepared is not None) or (request is not None and not _same(request['target'], prepared)):
            raise AssertionError('Outgoing UAV goal differs from selected survey target')
        self.attempts += 1
        try:
            with self.torch.inference_mode():
                self.env.step(self.frame)
        except self.me.SafetyTermination:
            self._observe_heard(now)
            self.safety_failure_details = failure_diagnostics(self.env)
            self._end('safety_failure')
            return True
        self._observe_heard(now)
        self.frame += 1
        if request is not None:
            sent = self.env.obs[self.env.ground]['self']['broadcast']
            if not _same(sent.get('target'), request['target']):
                raise AssertionError('Physics did not broadcast the selected survey target')
            if request['first_physics_broadcast_s'] is None:
                request['first_physics_broadcast_s'] = now
            request['physics_broadcasts'] += 1
        after = self.position()
        self.travelled_m += float(np.linalg.norm(after-before))
        self.trajectory.append(after.tolist())
        current = np.stack([_point(self.env.obs[r]['self']['position']) for r in self.env.ids])
        self.air_distances += np.linalg.norm(current - positions, axis=1)
        if self.frame == 1 or self.clock_s % 10 == 0:
            self.air_samples.append(dict(time_s=self.clock_s, positions=current.tolist()))
        metrics = self.env.trace[-1]
        self.collisions += int(metrics['ground_collisions'])
        self.ground_packets += int(metrics['received_ground_packets'])
        self.link_violations += int(metrics['chain_link_violation'])
        separation = float(metrics['min_separation'])
        self.minimum_separation = separation if self.minimum_separation is None else min(self.minimum_separation, separation)
        if self.env.goal_reached:
            if np.linalg.norm(after-self.goal) > GOAL_TOLERANCE_M:
                raise AssertionError('Intermediate target incorrectly terminated the V9 mission')
            self._end('arrival')
        elif self.clock_s >= self.limit:
            self._end('time_budget' if self.limit == self.budget else 'collector_limit')
        return self.finished

    def fleet_summary(self):
        return dict(case_id=self.case['case_id'], seed=self.case['seed'], end_reason=self.end_reason,
                    goal_reached=self.end_reason == 'arrival', sim_time_s=self.clock_s, budget_s=self.budget,
                    collector_limit_s=self.limit, direct_distance_m=self.case['direct_distance_m'],
                    remaining_m=float(np.linalg.norm(self.position()-self.goal)), travelled_m=self.travelled_m,
                    attempts=self.attempts, completed_steps=self.frame, collisions=self.collisions,
                    ground_replan_events=self.ground_events, survey_evaluations=self.survey_reviews,
                    survey_requests=len(self.requests), request_log=self.requests, survey_reward=self.survey_return,
                    evidence_reward=self.undiscounted_evidence,
                    heard_requests=sum(bool(x['heard_by']) for x in self.requests),
                    evidenced_requests=sum(x['first_evidence'] is not None for x in self.requests),
                    ground_received_packets=self.ground_packets,
                    fusion_count=self.env.ground_controller.graphs_received, adapter=self.adapter.stats(),
                    ground_motion_guard=self.motion_guard.stats(),
                    uav_travelled_m=dict(zip([c.drone_id for c in self.env.controllers], self.air_distances.tolist())),
                    uav_trajectory=self.air_samples, ground_trajectory=self.trajectory,
                    min_uav_separation_m=self.minimum_separation, link_violation_steps=self.link_violations,
                    safety_failure_details=self.safety_failure_details)

    def close(self):
        if getattr(self, 'adapter', None) is not None:
            self.adapter.close()
        if hasattr(self, '_original_filter'):
            self.env.ground_controller._filter_observed_obstacles = self._original_filter
        super().close()


def run_fleet_case(repo, case, model, *, fast_seed, budget_s, ground_interval_s=5, survey_check_s=5,
                   survey_min_s=20, survey_timeout_s=60, progress=None, rollout=None, collector_limit_s=None):
    episode = SurveyFleetEpisode(repo, case, model, fast_seed=fast_seed, budget_s=budget_s,
                                ground_interval_s=ground_interval_s, survey_check_s=survey_check_s,
                                survey_min_s=survey_min_s, survey_timeout_s=survey_timeout_s,
                                rollout=rollout, collector_limit_s=collector_limit_s)
    try:
        while not episode.finished:
            episode.step()
            if progress is not None and (episode.finished or episode.clock_s % 60 == 0):
                progress(dict(case_id=case['case_id'], sim_time_s=episode.clock_s, end_reason=episode.end_reason,
                              remaining_m=float(np.linalg.norm(episode.position()-episode.goal)),
                              survey_requests=len(episode.requests), transitions=0 if rollout is None else len(rollout)))
        return episode.fleet_summary()
    finally:
        episode.close()
