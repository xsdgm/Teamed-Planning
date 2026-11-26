"""Run frozen VTemp5 waypoint guidance in the V9.1 native fleet simulator."""
from contextlib import redirect_stdout
import os

import numpy as np
import torch

from v91_extent_collect import _world_cells
from v91_lake_cases import install_terrain
from v91_lake_drive import _experiment
from v91_lake_metrics import model_fields


def _same_point(a, b):
    return a is not None and b is not None and bool(np.allclose(a, b, rtol=0., atol=1e-6))


def run_actor_arm(scene, seed, uav_model, config, predictor, actor, arm, *, max_steps=480, interval=30,
                  prepare=None, action_seed=None, on_progress=None):
    """Run one closed loop, including the actual one-second vehicle commands.

    The learned arms issue native global-planner waypoints. Both execution and
    radio broadcast use the real V9.1 ground controller. A newly issued point
    first affects the next planner tick, after the already prepared command.
    """
    from mappo.mappo_experiment import DeliveredEvidence, SafetyTermination

    if arm not in ('native', 'observed', 'predicted') or interval != 30 or max_steps < 1:
        raise ValueError('Choose a supported arm, 30 s policy interval and a positive time limit')
    sample_seed = (int(np.random.SeedSequence([config['seed'], seed, 0]).generate_state(1)[0])
                   if action_seed is None else int(action_seed))
    goal = np.asarray(scene['goal'], dtype=float)
    car_path, trace, reviews = [], [], []
    reason = 'frame_limit'
    with open(os.devnull, 'w') as sink, redirect_stdout(sink), torch.inference_mode():
        torch.manual_seed(sample_seed)
        env = _experiment(scene, seed, uav_model, config)
        try:
            if scene.get('rects'):
                install_terrain(env, scene)
            if prepare is not None:
                prepare(env, scene)
            if env.env.config.dt != 1. or env.subgoals is not None:
                raise ValueError('The frozen waypoint driver requires 1 s steps and external subgoals')
            controller = env.ground_controller
            ground = DeliveredEvidence(world=env.world_size, resolution=env.ledger.resolution)
            interface = None
            if arm != 'native':
                from v91_frozen_actor_inputs import EpisodeActorInputs
                interface = EpisodeActorInputs()
            last_issue = goal.copy()
            actual_local_target = None
            original_velocity = controller.local_planner.compute_velocity_command

            def capture_velocity(*args, **kwargs):
                nonlocal actual_local_target
                target = kwargs['target_waypoint'] if 'target_waypoint' in kwargs else args[2]
                actual_local_target = np.asarray(target, dtype=float).copy()
                return original_velocity(*args, **kwargs)

            if interface is not None:
                controller.local_planner.compute_velocity_command = capture_velocity
            travelled = 0.
            for step in range(max_steps + 1):
                obs = env.obs[env.ground]
                car = np.asarray(obs['self']['position'][:2], dtype=float).copy()
                car_path.append(car)
                if on_progress is not None and (step % 60 == 0 or step == max_steps or env.goal_reached):
                    on_progress(dict(step=step, max_steps=max_steps, goal_remaining_m=float(np.linalg.norm(car-goal)),
                                     travelled_m=travelled, actor_calls=sum(r['actor_called'] for r in reviews),
                                     uav_packets_received=len(env.ledger.received)))
                if getattr(env.env, 'ground_safety_violations', 0) and prepare is not None:
                    reason = 'ground_safety_violation'
                    break
                if env.goal_reached:
                    reason = 'vehicle_arrival'
                    break
                if step == max_steps:
                    break
                if interface is not None:
                    # This is the local waypoint for the command already queued.
                    interface.record_local(step, actual_local_target)
                    ground.receive(env.ground, step, car, obs['occupancy_map'], obs['map_resolution'],
                                   valid=obs.get('sensor_valid', True))
                    if step % interval == 0:
                        snapshot = dict(car_xy=car, goal_xy=goal, frame=step, car_path=np.stack(car_path),
                            resolution=env.ledger.resolution, air_cells=_world_cells(env.ledger),
                            ground_cells=_world_cells(ground), received_packets=len(env.ledger.received),
                            mask_threshold=predictor.mask_threshold)
                        fields = model_fields(predictor, snapshot)
                        decision = interface.review(env, snapshot, fields, actor, arm)
                        target = np.asarray(decision['target_xy'], dtype=float)
                        if target.shape != (2,) or not np.isfinite(target).all():
                            raise ValueError('Frozen policy produced an invalid navigation target')
                        changed = not _same_point(last_issue, target)
                        if changed:
                            # Same waypoint update semantic as the formal fleet
                            # actor. Do not reinterpret a predicted cell as free
                            # in the ground planner or issue a velocity directly.
                            controller._drive_to(tuple(target))
                        last_issue = target.copy()
                        reviews.append(dict(decision, step=step, car_xy=car.tolist(),
                                            target_xy=target.tolist(), issue_changed=changed,
                                            received_packets=snapshot['received_packets'],
                                            applied_target_frames=0, moving_target_frames=0))

                motion = tuple(map(float, env.ground_action['motion']))
                broadcast = env.ground_action.get('broadcast') or {}
                broadcast_target = broadcast.get('target')
                local_xy = None if actual_local_target is None else actual_local_target.tolist()
                actual_local_target = None
                try:
                    env.step(step)
                except SafetyTermination:
                    reason = 'safety_termination'
                    break
                following_car = np.asarray(env.obs[env.ground]['self']['position'][:2])
                distance = float(np.linalg.norm(following_car - car))
                travelled += distance
                if reviews and _same_point(broadcast_target, last_issue):
                    reviews[-1]['applied_target_frames'] += 1
                    reviews[-1]['moving_target_frames'] += int(distance > 1e-6)
                trace.append(dict(step=step, motion=motion, distance_m=distance, local_xy=local_xy,
                    broadcast_target=None if broadcast_target is None else list(map(float, broadcast_target))))
            path = np.stack(car_path)
            if len(trace) != len(path) - 1:
                raise RuntimeError('The recorded command stream and physical vehicle trajectory differ in length')
            return dict(seed=int(seed), arm=arm, termination=reason, last_step=len(trace),
                        goal=goal.tolist(), goal_remaining_m=float(np.linalg.norm(path[-1] - goal)),
                        travelled_m=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                        ground_collisions=int(sum(m['ground_collisions'] for m in env.trace)),
                        ground_safety_violations=int(getattr(env.env, 'ground_safety_violations', 0)),
                        ground_physical_collision_attempts=getattr(env.env, 'raster_collision_attempts', None),
                        boundary_violation_frames=sum(bool(m['out_of_bounds']) for m in env.trace),
                        car_path=path.tolist(), trace_motion=trace, reviews=reviews, review_interval_s=interval,
                        actor_calls=sum(bool(r['actor_called']) for r in reviews),
                        waypoint_changes=sum(r['issue_changed'] for r in reviews),
                        moving_actor_target_frames=sum(r['moving_target_frames'] for r in reviews
                                                       if r['actor_called']),
                        uav_packets_received=len(env.ledger.received))
        finally:
            env.env.close()
