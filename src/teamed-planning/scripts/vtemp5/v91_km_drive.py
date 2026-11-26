"""Frozen checkpoint evaluation on the earlier kilometre-scale fleet missions."""
import math
import sys

import numpy as np

from v91_frozen_actor_drive import run_actor_arm
from v91_km_wild import install_wild_ground_filter, wild_runtime


def _prepare(env, scene):
    from shared import ground
    from shared.ground import global_planner
    from survey.survey_fleet_metrics import use_bounded_task_mask

    # The archived vehicle extension imports the same planner under the former
    # package name. Its live class and its per-controller factory must be identical.
    for name, module in [('ground', ground), ('ground.global_planner', global_planner)]:
        if sys.modules.setdefault(name, module) is not module:
            raise RuntimeError('Another vehicle planner is bound to the archived module name')
    from vtemp4.vtemp4_anchored_ground import install_anchored_ground

    size = scene['world_size']
    bounds = (0., 0., size, size)
    if (tuple(env.env.config.world_size) != (size, size)
            or tuple(env.ground_controller.world_bounds) != bounds
            or env.ledger.seen.shape != (int(size / 2),) * 2 or env.ledger.resolution != 2.
            or len(env.ids) != 3 or any(tuple(c.navigator.world_bounds) != bounds for c in env.controllers)
            or env.env.config.dt != 1. or env.ground_safety_distance != 1.5):
        raise ValueError('Kilometre fleet geometry, ground clearance or receiver resolution mismatch')
    if any(env.obs[rid]['map_axes'] != 'yx' or env.obs[rid]['map_resolution'] != res
           for rid, res in [(env.ground, .5), *((rid, 2.) for rid in env.ids)]):
        raise ValueError('The kilometre evaluation requires native vehicle and aircraft observations')
    install_anchored_ground(env.ground_controller)
    use_bounded_task_mask(env)
    install_wild_ground_filter(env, scene)
    env.ground_action = env._ground_action([])


def run_km_actor_arm(scene, uav, config, predictor, actor, arm, *, max_steps=None, on_progress=None):
    """Run one paired arm in physical metres, with native 1 s actions and 30 s reviews.

    The geometry seed and the archived fleet sampling seed serve distinct roles.
    A shorter max_steps is available for smoke tests; results record this explicitly.
    on_progress receives one dict after each 60 simulated seconds, including time zero.
    """
    size = float(scene['world_size'])
    start_goal = np.array([scene['start'], scene['goal']], dtype=float)
    distance = float(np.linalg.norm(start_goal[1] - start_goal[0]))
    budget = int(scene['max_steps'])
    if (start_goal.shape != (2, 2) or not np.isfinite(start_goal).all() or (start_goal < 0).any()
            or (start_goal >= size).any() or distance < 1000. or size % 2 != 0.
            or not np.isclose(distance, scene['direct_distance_m'], atol=1e-6, rtol=0.)
            or budget != math.ceil(6 * distance + 120)):
        raise ValueError('Expected an in-bounds kilometre mission with the archived physical time budget')
    steps = budget if max_steps is None else max_steps
    if type(steps) is not int or not 1 <= steps <= budget:
        raise ValueError('The evaluation time limit must be a positive integer within the mission budget')
    # The native V9.1 training normalizations remain in the weights and runtime.
    # Only the actual map bounds change from the UAV training arena.
    with wild_runtime(scene):
        result = run_actor_arm(scene, scene['seed'], uav, dict(config, world_size=size),
                               predictor, actor, arm, max_steps=steps, interval=30, prepare=_prepare,
                               action_seed=scene['fleet_seed'], on_progress=on_progress)
    path = np.asarray(result['car_path'])
    if (not np.isfinite(path).all() or np.any(path < 0.) or np.any(path > size)
            or not np.array_equal(path[0], start_goal[0]) or len(path) != result['last_step'] + 1):
        raise RuntimeError('Executed trajectory is inconsistent with its physical task')
    for row in result['reviews']:
        target = np.asarray(row['target_xy'])
        if np.any(target < 0.) or np.any(target >= size):
            raise RuntimeError('Vehicle policy issued an out-of-bounds waypoint')
        if row['actor_called'] and not np.isclose(np.linalg.norm(target - row['car_xy']), 200.,
                                                  rtol=0., atol=1e-6):
            raise RuntimeError('A learned designation is off the 200 m ring')
    result.update(fleet_seed=int(scene['fleet_seed']), mission_distance_m=distance,
                  world_size_m=size, mission_budget_s=budget, rollout_limit_s=steps,
                  is_full_budget=steps == budget, number_of_drones=3, receiver_resolution_m=2.,
                  ground_control='vtemp1_connected_anchored')
    return result
