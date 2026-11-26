"""Execute same-tick frozen vehicle decisions in the kilometre V9.1 fleet."""
from contextlib import redirect_stdout
from copy import copy
import math
import os
import sys
from unittest.mock import patch

import numpy as np
import torch

from run_v91_extent_pilot import encode
from v91_extent_collect import _world_cells
from v91_km_wild import wild_runtime
from v91_lake_drive import _experiment
from v91_lake_metrics import model_fields


def _as_point(value, size):
    p = np.asarray(value, dtype=float)
    if p.shape != (2,) or not np.isfinite(p).all() or (p < 0).any() or (p >= size).any():
        raise ValueError('Expected a finite in-bounds physical target')
    return p


def _publish_vehicle_target(controller, accepted, same_intent, proposed, car, now, force):
    """Keep the actually issued point under jitter and roll forward along a healthy prefix."""
    distance = float(np.linalg.norm(proposed - car))
    tolerance = max(.25, min(2., .015 * distance))
    if accepted is not None and same_intent and not force:
        if np.linalg.norm(proposed - accepted) <= tolerance:
            return accepted, 'held'
        old = accepted - car
        previous_distance = np.linalg.norm(old)
        if (previous_distance > 25. and distance > 25. and controller.goal_position is not None
                and old @ (proposed-car) > math.cos(math.radians(3.)) * distance * previous_distance):
            # Match the native same-direction rolling-guidance update. The delta
            # rebases distance-to-target while retaining physical progress history.
            controller._local_route_best += distance - previous_distance
            controller._local_route_goal = controller.goal_position = tuple(proposed)
            if controller._route_map is not None:
                controller._route_map.goal = proposed.copy()
            return proposed, 'rolled'
    controller._drive_to(tuple(proposed))
    controller._start_route_progress(tuple(proposed), distance, now)
    return proposed, 'replanned'


def _current_air_evidence(env, neighbors):
    """Include this tick's actual receipts before the native reward ledger commits them."""
    ledger = copy(env.ledger)
    ledger.received = ledger.received.copy()
    ledger.seen, ledger.log_odds = ledger.seen.copy(), ledger.log_odds.copy()
    for neighbor in neighbors:
        for packet in (neighbor.get('broadcast') or {}).get('evidence_audits', ()):
            ledger.receive(packet['source'], packet['frame'], packet['position'], packet['occupancy'],
                           packet['resolution'], valid=packet['valid'])
    return ledger


class GuidedDrive:
    def __init__(self, env, predictor, actor, arm, steps, interface, on_progress, geometry_seed, fleet_seed):
        from shared import ground
        from shared.ground import global_planner
        from shared.simulator.wildscenes import wildscenes_map
        from survey.survey_fleet_metrics import use_bounded_task_mask
        from vtemp2_3.wildscenes_runtime import _ground_local_geometry, filter_ground_motion
        if env.obs[env.ground].get('ground_safety_map') is None:
            # This sensor is vehicle-centred. Its occupied cell boundaries move
            # as the vehicle translates. Source-aligned tiles retain the original
            # continuous geometry and last-line motion filter.
            from v91_rolling_clearance import guard_motion as filter_ground_motion
            from v91_rolling_clearance import local_geometry as _ground_local_geometry

        for name, module in [('ground',ground), ('ground.global_planner',global_planner)]:
            if sys.modules.setdefault(name,module) is not module:
                raise RuntimeError('Incompatible archived planner modules')
        from vtemp4.vtemp4_anchored_ground import install_anchored_ground

        self.env, self.actor, self.predictor, self.arm = env, actor, predictor, arm
        self.geometry_seed, self.fleet_seed = geometry_seed, fleet_seed
        self.interface, self.steps, self.progress = interface, steps, on_progress
        self.path, self.trace, self.reviews = [], [], []
        self.distance = 0.
        self.accepted = self.survey = self.intent = self.pending = self.local = None
        self.anchor = env.start.copy()
        self.last_progress = 0
        self.last_veto = None
        self.interval_travel = 0.
        self.interval_vetoes = 0
        self.interval_interventions = 0
        self.radio = next(robot for robot in env.env.ground_robots if robot.robot_id == env.ground)
        self.current_neighbors = []
        self.at_review_tick = False
        self.at_refresh_tick = False
        self.at_bootstrap_tick = False
        self.holding = False
        self.policy_tick = None
        self.ground = type(env.ledger)(world=env.world_size, resolution=env.ledger.resolution)
        self.guard = lambda action, obs: filter_ground_motion(action, obs, wildscenes_map,
                                    clearance=env.ground_safety_distance, max_speed=env.ground_controller.max_linear_vel)
        self.local_geometry = lambda obs: _ground_local_geometry(obs, wildscenes_map)
        c = env.ground_controller
        install_anchored_ground(c)
        c.local_planner.escape_distance = c.local_planner.safety_distance = env.ground_safety_distance
        use_bounded_task_mask(env)
        self.native_ground = env._ground_action
        self.native_filter = c._filter_observed_obstacles
        self.native_velocity = c.local_planner.compute_velocity_command
        self.native_local = c._local_waypoint
        c._filter_observed_obstacles = self.after_fusion
        c.local_planner.compute_velocity_command = self.capture_local
        c._local_waypoint = self.checked_local
        env._ground_action = self.prepare

    def capture_local(self, *args, **kwargs):
        pt = kwargs['target_waypoint'] if 'target_waypoint' in kwargs else args[2]
        self.local = np.asarray(pt, dtype=float).copy()
        return self.native_velocity(*args, **kwargs)

    def checked_local(self, current_pos, target, semantic_map, resolution):
        point = self.native_local(current_pos, target, semantic_map, resolution)
        veto = self.last_veto
        if point is None or veto is None or veto['local_xy'] is None:
            return point
        # Reject a previously vetoed local direction only while it still fails
        # the live onboard clearance test. An aligned side step remains eligible.
        if (np.linalg.norm(np.asarray(current_pos) - veto['car_xy']) > resolution
                or np.linalg.norm(np.asarray(point) - veto['local_xy']) > resolution):
            return point
        delta = np.asarray(point) - current_pos
        obs = self.env.obs[self.env.ground]
        aligned = dict(obs, self=dict(obs['self'], orientation=math.atan2(delta[1], delta[0])))
        action, _ = self.guard(dict(motion=(min(.5, np.linalg.norm(delta)), 0.)), aligned)
        return point if action['motion'][0] > 1e-5 else None

    def after_fusion(self, graph, semantic_map):
        if self.at_review_tick or self.at_bootstrap_tick:
            self.at_review_tick = False
            now = len(self.path) - 1
            car = self.path[-1]
            air = _current_air_evidence(self.env, self.current_neighbors)
            previous = self.reviews[-1] if self.reviews else None
            feedback = dict(interval_steps=now-previous['step'] if previous else 0,
                interval_displacement_m=float(np.linalg.norm(car-previous['car_xy'])) if previous else 0.,
                interval_travelled_m=self.interval_travel, veto_frames=self.interval_vetoes,
                guard_interventions=self.interval_interventions,
                stalled_seconds=now-self.last_progress, last_veto=self.last_veto,
                accepted_target_xy=None if self.accepted is None else self.accepted.tolist())
            snapshot = dict(frame=now, car_xy=car, goal_xy=self.env.goal, car_path=np.stack(self.path),
                resolution=air.resolution, air_cells=_world_cells(air), ground_cells=_world_cells(self.ground),
                received_packets=len(air.received), mask_threshold=self.predictor.mask_threshold,
                presence_threshold=self.predictor.presence_threshold, geometry_seed=self.geometry_seed,
                fleet_seed=self.fleet_seed, execution_feedback=feedback)
            if self.arm == 'predicted':
                fields = model_fields(self.predictor, snapshot)
            else:
                x, xy = encode(snapshot)
                fields = (x, xy, None, None)
            if self.at_bootstrap_tick:
                self.at_bootstrap_tick = False
                self.interface.bootstrap(self.env, snapshot, fields, self.actor, self.arm)
                return self.native_filter(graph, semantic_map)
            decision = self.interface.review(self.env, snapshot, fields, self.actor, self.arm)
            proposed = _as_point(decision['target_xy'], self.env.world_size)
            intent = decision.get('intent_key', decision.get('mode', decision.get('source')))
            accepted, kind = _publish_vehicle_target(self.env.ground_controller, self.accepted, intent == self.intent,
                                          proposed, car, now, bool(decision.get('force_replan', False)))
            self.intent, self.accepted = intent, accepted
            self.holding = decision.get('mode') == 'wait_for_survey'
            if decision.get('survey_target_xy') is None:
                self.survey = accepted.copy()
            else:
                self.survey = _as_point(decision['survey_target_xy'], self.env.world_size)
            self.interface.after_issue(decision, accepted, kind != 'held')
            self.reviews.append(dict(decision, step=now, car_xy=car.tolist(), proposed_target_xy=proposed.tolist(),
                target_xy=accepted.tolist(), broadcast_target=self.survey.tolist(), issue_kind=kind,
                issue_changed=kind != 'held', received_packets=len(air.received), execution_feedback=feedback,
                applied_target_frames=0, moving_target_frames=0, veto_frames=0, local_refreshes=[]))
            self.interval_vetoes = 0
            self.interval_interventions = 0
            self.interval_travel = 0.
            self.policy_tick = now
        elif self.at_refresh_tick:
            self.at_refresh_tick = False
            review = self.reviews[-1]
            car = self.path[-1]
            if np.linalg.norm(car - self.accepted) < 1.2:
                next_point = self.interface.refresh_target(self.env, review, car, self.accepted)
                if next_point is not None:
                    point = _as_point(next_point, self.env.world_size)
                    geometry = self.local_geometry(self.env.obs[self.env.ground])
                    if (np.linalg.norm(point - car) > 2.5 + 1e-6
                            or not geometry.segment_clear(car, point, self.env.ground_safety_distance)):
                        raise ValueError('A local target refresh exceeded the on-board certified segment')
                    self.accepted, update = _publish_vehicle_target(self.env.ground_controller, self.accepted,
                        True, point, car, len(self.path)-1, False)
                    if update != 'held':
                        review['local_refreshes'].append(dict(step=len(self.path)-1,
                                                              target_xy=self.accepted.tolist()))
        return self.native_filter(graph, semantic_map)

    def prepare(self, neighbors=None):
        env = self.env
        now = int(env.env.time)
        if now != len(self.path):
            raise RuntimeError('Ground action computed more than once for a physical tick')
        obs = env.obs[env.ground]
        car = np.asarray(obs['self']['position'][:2], dtype=float).copy()
        if self.pending is not None:
            moved = float(np.linalg.norm(car-self.path[-1]))
            previous = self.pending
            self.distance += moved
            self.interval_travel += moved
            self.interval_vetoes += int(previous['veto'])
            self.interval_interventions += int(previous['guard_intervened'])
            transmitted = self.radio.get_broadcast()['target']
            if transmitted != previous['broadcast_target']:
                raise RuntimeError('The actual ground radio did not transmit the issued survey target')
            heard = sum(any(n['id'] == env.ground and (n.get('broadcast') or {}).get('target') == transmitted
                           for n in env.obs[r]['neighbors']) for r in env.ids)
            self.trace.append(dict(previous, distance_m=moved, direct_uav_receivers=heard))
            if self.interface is not None:
                self.interface.record_local(now-1, previous['local_xy'])
                row = self.reviews[-1]
                if previous['review_step'] != row['step']:
                    raise RuntimeError('Executed command lost its issuing policy review')
                row['applied_target_frames'] += 1
                row['moving_target_frames'] += int(moved > 1e-6)
                row['veto_frames'] += int(previous['veto'])
            if np.linalg.norm(car-self.anchor) >= .5:
                self.anchor, self.last_progress = car.copy(), now
        self.path.append(car)
        terminal = (now == self.steps or np.linalg.norm(car-env.goal) <= 2.
                    or getattr(env.env, 'ground_safety_violations', 0) > 0)
        if self.progress is not None and (now % 60 == 0 or terminal):
            self.progress(dict(step=now, max_steps=self.steps, travelled_m=self.distance,
                goal_remaining_m=float(np.linalg.norm(car-env.goal)),
                actor_calls=sum(r['actor_called'] for r in self.reviews), uav_packets_received=len(env.ledger.received)))
        self.at_review_tick = self.interface is not None and not terminal and now % 30 == 0
        self.at_bootstrap_tick = (now == self.steps and hasattr(self.interface, 'bootstrap')
                                  and np.linalg.norm(car - env.goal) > 2.
                                  and not getattr(env.env, 'ground_safety_violations', 0))
        self.at_refresh_tick = (self.interface is not None and not terminal and not self.holding and
                                now % 30 != 0 and self.reviews[-1]['mode'] in
                                ('local_bank_recovery', 'follow_observed_branch'))
        self.current_neighbors = [] if neighbors is None else neighbors
        self.ground.receive(env.ground, now, car, obs['occupancy_map'], obs['map_resolution'],
                            valid=obs.get('sensor_valid',True))
        self.local = None
        action = self.native_ground(neighbors)
        if self.at_review_tick or self.at_bootstrap_tick:
            raise RuntimeError('Native ground controller skipped the fusion decision hook')
        if terminal:
            self.pending = None
            return action
        if self.survey is not None:
            action = dict(action, broadcast=dict(action['broadcast'], target=self.survey.tolist()))
        if self.holding:
            # A survey hold authorizes aircraft scouting and a stationary UGV.
            # Preserve physical state from this tick for the next stall decision.
            action = dict(action, motion=(0., 0.))
        guarded, event = self.guard(action, obs)
        velocity = tuple(map(float, guarded['motion']))
        local = None if self.local is None else self.local.tolist()
        veto = event['intervened'] and abs(velocity[0]) < 1e-5
        if veto:
            self.last_veto = dict(step=now, car_xy=car.tolist(), local_xy=local,
                                  requested_motion=event['requested_motion'], motion=list(velocity))
        intended = env.goal if self.accepted is None else self.accepted
        self.pending = dict(step=now, review_step=self.policy_tick, motion=velocity,
            requested_motion=event['requested_motion'], guard_intervened=bool(event['intervened']), veto=bool(veto),
            vehicle_target=intended.tolist(), broadcast_target=guarded['broadcast']['target'], local_xy=local)
        return guarded


def run_km_actor_arm(scene, uav, config, predictor, actor, arm, *, max_steps=None, on_progress=None,
                     guidance=None):
    """Run a frozen corrected-guidance arm with one real ground decision/command per second."""
    from mappo import mappo_experiment as me
    from v91_rejoin_guidance import RejoinGuidance

    size, goal = float(scene['world_size']), np.asarray(scene['goal'], dtype=float)
    direct = float(np.linalg.norm(_as_point(goal, size) - _as_point(scene['start'], size)))
    budget = int(scene['max_steps'])
    steps = budget if max_steps is None else max_steps
    mouth = guidance is not None and scene.get('curriculum_stage') == 'mouth'
    valid_course = (size == 1440. and 0. < direct <= scene['reference_distance_m'] + 1e-6
                    and budget == math.ceil(6 * scene['reference_distance_m'] + 120)
                    if mouth else direct >= 1000. and budget == math.ceil(6*direct + 120))
    if (arm not in ('native','observed','predicted') or (guidance is not None and arm == 'native') or not valid_course
            or type(steps) is not int or not 1 <= steps <= budget or config['ground_safety_distance'] != 1.5):
        raise ValueError('Invalid kilometre mission, physical clearance or simulation budget')
    reason = 'frame_limit'
    with wild_runtime(scene), torch.inference_mode(), open(os.devnull, 'w') as sink, redirect_stdout(sink):
        torch.manual_seed(int(scene['fleet_seed']))
        # The frozen constructor requests its first command before creating the
        # delivered ledger. Defer that single request until all sensors/ledgers
        # exist; no dummy command is sampled, sent, or executed.
        with patch.object(me.Experiment, '_ground_action', return_value=None):
            env = _experiment(scene, scene['seed'], uav, dict(config,world_size=size))
        try:
            expected_bounds = (0., 0., size, size)
            if (env.env.config.dt != 1. or env.subgoals is not None or env.ledger.resolution != 2.
                    or env.ledger.seen.shape != (int(size/2),) * 2 or tuple(env.env.config.world_size) != (size,size)
                    or tuple(env.ground_controller.world_bounds) != expected_bounds
                    or any(tuple(c.navigator.world_bounds) != expected_bounds for c in env.controllers)
                    or any(env.obs[rid]['map_axes'] != 'yx' or env.obs[rid]['map_resolution'] != res
                           for rid, res in [(env.ground,.5), *((r,2.) for r in env.ids)])
                    or len(env.ids) != 3 or env.communication_mode != 'backbone' or env.drop != 0.):
                raise ValueError('Expected the V9.1 physical, evidence and radio contracts')
            interface = guidance if guidance is not None else (None if arm == 'native' else RejoinGuidance())
            drive = GuidedDrive(env, predictor, actor, arm, steps, interface, on_progress,
                                int(scene['seed']), int(scene['fleet_seed']))
            env.ground_action = drive.prepare()
            for step in range(steps):
                try:
                    env.step(step)
                except me.SafetyTermination:
                    reason = 'safety_termination'
                    break
                if getattr(env.env, 'ground_safety_violations', 0):
                    reason = 'ground_safety_violation'
                    break
                if env.goal_reached:
                    reason = 'vehicle_arrival'
                    break
            if len(drive.trace) + 1 != len(drive.path) or len(drive.trace) != len(env.trace):
                raise RuntimeError('Physical command/pose clock mismatch')
            if interface is not None and sum(r['applied_target_frames'] for r in drive.reviews) != len(drive.trace):
                raise RuntimeError('Incomplete policy attribution')
            return dict(seed=int(scene['seed']), fleet_seed=int(scene['fleet_seed']), arm=arm, termination=reason,
                last_step=len(drive.trace), goal=goal.tolist(), goal_remaining_m=float(np.linalg.norm(drive.path[-1]-goal)),
                travelled_m=drive.distance, ground_collisions=int(sum(m['ground_collisions'] for m in env.trace)),
                ground_safety_violations=int(getattr(env.env,'ground_safety_violations',0)),
                ground_physical_collision_attempts=getattr(env.env,'raster_collision_attempts',None),
                boundary_violation_frames=sum(bool(m['out_of_bounds']) for m in env.trace),
                ground_motion_filter_interventions=sum(r['guard_intervened'] for r in drive.trace),
                ground_motion_full_vetoes=sum(r['veto'] for r in drive.trace),
                car_path=np.stack(drive.path).tolist(), trace_motion=drive.trace, reviews=drive.reviews,
                review_interval_s=30, actor_calls=sum(bool(r['actor_called']) for r in drive.reviews),
                waypoint_changes=sum(r['issue_changed'] for r in drive.reviews),
                moving_actor_target_frames=sum(r['moving_target_frames'] for r in drive.reviews if r['actor_called']),
                uav_packets_received=len(env.ledger.received), mission_distance_m=direct, world_size_m=size,
                mission_budget_s=budget, rollout_limit_s=steps, is_full_budget=steps == budget,
                number_of_drones=3, receiver_resolution_m=2., ground_control='vtemp1_connected_anchored')
        finally:
            env.env.close()
