"""Paired closed-loop test of a local goalward return gate on fixed UAV rollouts.

The physics, mission, native graph planner and aircraft policy are identical in
all arms. A review can request direct local tracking after a vehicle has seen
an occupied goalward corridor and deviated at least six metres laterally.
Unknown far cells default to blocked in the observed arm; the predicted arm
fills ONLY unknown cells from the frozen network. The native arm never overrides
a command. Every proposed step still passes the native onboard local planner.
Labels, raw simulator geometry and oracle safety scores do not enter the gate.
"""
from contextlib import redirect_stdout
import os

import numpy as np
import torch

from v91_extent_collect import _snapshot
from v91_lake_metrics import load_model as load_predictor, model_fields


RETURN_LENGTH_M = 24.
LATERAL_TRIGGER_M = 6.
DIRECT_ANGLE_RAD = np.deg2rad(25.)
CELL_M = 2.


def _corridor(xy, car, direction, length, half_width):
    delta = xy - car
    longitudinal = np.clip(delta @ direction, 0., length)
    residual = delta - longitudinal[..., None] * direction
    return np.linalg.norm(residual, axis=-1) <= half_width + CELL_M * np.sqrt(.5)


def _review(snapshot, predictor, encounter_seen, start, clearance, arm):
    car, goal = snapshot['car_xy'], snapshot['goal_xy']
    delta = goal - car
    distance = float(np.linalg.norm(delta))
    forward = delta / max(distance, 1e-9)
    x, xy, presence, probability = model_fields(predictor, snapshot)
    occupied, free = x[1].astype(bool), x[0].astype(bool)
    length = min(distance, RETURN_LENGTH_M)
    corridor = _corridor(xy, car, forward, length, clearance)
    near = _corridor(xy, car, forward, min(length, 2.), clearance)
    # The shared encounter detector uses only received occupancy, including small
    # blockers; it never queries a scenario ID, shoreline label or bridge position.
    encounter_seen |= bool(occupied[_corridor(xy, car, forward, min(64., distance), clearance)].any())
    mission = goal - start
    mission /= max(np.linalg.norm(mission), 1e-9)
    cross = abs(float((car - start) @ np.array([-mission[1], mission[0]])))
    eligible = bool(encounter_seen and cross >= LATERAL_TRIGGER_M and distance > 4.)
    known_blocked = bool(occupied[corridor].any())
    near_clear = bool(free[near].all())
    permitted = eligible and near_clear and not known_blocked
    observed = bool(permitted and free[corridor].all())
    unknown = corridor & ~free & ~occupied
    predicted = bool(permitted and not (probability[unknown] >= predictor.mask_threshold).any())
    apply = (arm == 'observed' and observed) or (arm == 'predicted' and predicted)
    return encounter_seen, dict(step=int(snapshot['frame']), car_xy=car.tolist(),
        corridor_length_m=length, eligible=eligible, cross_track_m=cross,
        encounter_reported=bool(encounter_seen), observed_blocked=known_blocked, near_clear=near_clear,
        unknown_cells=int(unknown.sum()), corridor_cells=int(corridor.sum()),
        presence_probability=float(presence), observed_approve=observed, predicted_approve=predicted,
        applied=bool(apply), local_tracker_steps=0, changed_command_steps=0)


def _direct_motion(env, car, goal):
    """Get a native locally certified waypoint and its executable 1 s command."""
    if env.goal_reached or env.obs[env.ground].get('motion_held', False):
        return None
    controller = env.ground_controller
    local = controller.local_planner
    delta = goal - car
    remaining = np.linalg.norm(delta)
    if remaining < 2.:
        return None
    waypoint = car + delta / remaining * min(2., remaining)
    semantic = controller._last_semantic_map
    target = local.observed_connection_waypoint(car, waypoint, semantic, controller.resolution,
                                               allow_escape=False)
    if target is None:
        return None
    step = np.asarray(target) - car
    step_length = np.linalg.norm(step)
    if step_length < .1 or float(step @ delta) < np.cos(DIRECT_ANGLE_RAD) * step_length * remaining:
        return None
    heading = float(controller.heading)
    error = local._normalize_angle(float(np.arctan2(step[1], step[0])) - heading)
    linear, angular = local.compute_velocity_command(car, heading, target,
        local_semantic_map=semantic, resolution=controller.resolution)
    # Same turn-before-translation convention as grounded observed-waypoint tracking.
    dt = env.env.config.dt
    angular = float(np.clip(angular, -abs(error)/dt, abs(error)/dt))
    return (0. if abs(error) > .25 else float(linear), angular)


def _experiment(scene, seed, model, config):
    from mappo.mappo_experiment import Experiment

    keys = ('drop exploration_weight exploration_candidates communication_mode safety_filter '
            'spacing_penalty_weight intervention_weight moving_ground reward_mode ground_speed '
            'path_reward_scale ground_safety_distance subgoal_radius subgoal_arrival subgoal_review_interval '
            'route_reward_weight time_penalty ground_stationary chain_soft_ratio chain_risk_weight '
            'chain_outward_weight chain_missing_weight path_quality_weight mapping_route_origin '
            'cbf_margin_weight cbf_margin_scale frontier_progress_weight side_exploration_weight '
            'chain_ground_range chain_link_range relay_loss relay_delay_jitter relay_period relay_capacity '
            'fusion_bridge_distance world_size').split()
    opts = {key: config[key] for key in keys}
    sizes = config['fleet_sizes']
    return Experiment(seed, model, stochastic=True, num_drones=sizes[seed % len(sizes)],
        scene=(scene['goal'], scene['circles'], scene.get('attempts', 1)), start=scene['start'],
        stop_on_ground_arrival=False, mapping_until_connected=config['mapping_reward_until_connected'], **opts)


def _execution_metrics(reviews, path, goal, interval):
    for row in reviews:
        begin = row['step']
        end = min(len(path) - 1, begin + interval)
        change = path[end] - path[begin]
        displacement = float(np.linalg.norm(change))
        direction = goal - path[begin]
        direction /= max(np.linalg.norm(direction), 1e-9)
        progress = float(change @ direction)
        row.update(executed_seconds=end - begin, displacement_m=displacement, goalward_m=progress,
                   actual_goal_aligned=bool(displacement >= 1. and progress >= displacement*np.cos(DIRECT_ANGLE_RAD)))


def run_arm(scene, seed, uav_model, config, predictor, arm, *, max_steps=480, interval=5, snapshot_steps=()):
    """Return (JSON-safe summary, requested raw observation snapshots). Times are seconds."""
    from mappo.mappo_experiment import DeliveredEvidence, SafetyTermination
    from v91_lake_cases import install_terrain

    if arm not in ('native', 'observed', 'predicted') or min(max_steps, interval) < 1:
        raise ValueError('Expected a known control arm and positive step and review limits')
    wanted = set(snapshot_steps)
    sample_seed = int(np.random.SeedSequence([config['seed'], seed, 0]).generate_state(1)[0])
    start = np.array(scene['start'], float)
    goal = np.array(scene['goal'], float)
    snapshots, reviews, car_path, drone_paths = [], [], [], []
    reason = 'frame_limit'
    with open(os.devnull, 'w') as sink, redirect_stdout(sink), torch.inference_mode():
        torch.manual_seed(sample_seed)
        env = _experiment(scene, seed, uav_model, config)
        try:
            install_terrain(env, scene)
            dt = env.env.config.dt
            if dt != 1.:
                raise ValueError('This evaluation requires one-second simulation steps')
            ground = DeliveredEvidence(world=env.world_size, resolution=env.ledger.resolution)
            encounter_seen = False
            review = None
            for step in range(max_steps + 1):
                obs = env.obs[env.ground]
                car = np.array(obs['self']['position'][:2], float)
                car_path.append(car)
                drone_paths.append(np.array([env.obs[i]['self']['position'][:2] for i in env.ids]))
                ground.receive(env.ground, step, car, obs['occupancy_map'], obs['map_resolution'],
                               valid=obs.get('sensor_valid', True))
                if step in wanted or (step % interval == 0 and step < max_steps and not env.goal_reached):
                    snapshot = _snapshot(env, ground, car_path, drone_paths, step)
                    if step in wanted:
                        snapshots.append(snapshot)
                    if step % interval == 0 and step < max_steps and not env.goal_reached:
                        encounter_seen, review = _review(snapshot, predictor, encounter_seen, start,
                                                         config['ground_safety_distance'], arm)
                        reviews.append(review)
                if env.goal_reached:
                    reason = 'vehicle_arrival'
                    break
                if step == max_steps:
                    break
                original_motion = env.ground_action['motion']
                motion = _direct_motion(env, car, goal) if review and review['applied'] else None
                if motion is not None:
                    env.ground_action = dict(env.ground_action, motion=motion)
                try:
                    env.step(step)
                except SafetyTermination:
                    reason = 'safety_termination'
                    break
                if motion is not None:
                    review['local_tracker_steps'] += 1
                    review['changed_command_steps'] += int(not np.allclose(motion, original_motion))
            path = np.stack(car_path)
            _execution_metrics(reviews, path, goal, interval)
            summary = dict(seed=int(seed), arm=arm, termination=reason, goal=goal.tolist(),
                last_step=len(path)-1, goal_remaining_m=float(np.linalg.norm(goal-path[-1])),
                travelled_m=float(np.linalg.norm(np.diff(path, axis=0),axis=1).sum()),
                ground_collisions=int(sum(m['ground_collisions'] for m in env.trace)),
                car_path=path.tolist(), reviews=reviews, review_interval_s=interval)
            return summary, snapshots
        finally:
            env.env.close()
