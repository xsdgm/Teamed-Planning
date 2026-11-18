"""vtmp2 observations and interruptible candidate-conditioned collection.

Legacy collection retains the archived runtime. Independent missions combine
the current vtmp vehicle stack with frozen v9 aircraft weights/runtime/CBF;
archived sources are imported without modification.
"""
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .hierarchical_sources import DEFAULT_REPO as REPO, atomic_json, file_sha256 as sha256
from .runtime import ReconnaissanceEnvelope, make_vtmp2_controller
from .vtmp2 import Vtmp2SubgoalManager

ALGORITHM = 'vtmp2'
SCHEMA_VERSION = 2
MISSION_SCHEMA_VERSION = 3
MISSION_ACTION_CONTRACT = 'drive_survey_v1'
SIDE = 32
CHANNELS = 4
FEATURES = 16
HORIZON = 10
GRAPH_HALF_EXTENT = 200.
CHECKPOINT_SHA = '7ac3aedcc3876902773b2241f60e839eb3e3b5630ddeb3c8be660421f2c24198'
FROZEN = REPO / 'artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146'
# Position residuals are measured in 10 metres, graph-count residuals in 0.1
# log-normalized units. Final goal coordinates (6:8) are immutable.
DELTA_SCALE = np.array([.01, .01, 1, 1, 1, 1, 0, 0, .1, .1, 1, 1, 1, 1, 1, 1], np.float32)


def provenance():
    checkpoint = FROZEN / 'best.pt'
    actual = sha256(checkpoint)
    if actual != CHECKPOINT_SHA:
        raise ValueError(f'Frozen MAPPO checkpoint changed: {actual}')
    sources = {str(p.relative_to(FROZEN)): sha256(p)
               for p in sorted((FROZEN / 'runtime_sources').rglob('*.py'))}
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=actual,
                runtime_sources=sources, config_sha256=sha256(FROZEN / 'config.json'),
                policy_mode='deterministic_mean', high_level_algorithm=ALGORITHM,
                selector_mode='geometric_reconnaissance', schema_version=SCHEMA_VERSION)


# Kept as an import alias for earlier collection callers. Selection is now
# evidence-aware vtmp2; it is not a trained world-model policy.
CandidateSampler = Vtmp2SubgoalManager


def encode(controller, final_goal, now, velocity):
    """Whitelist only own state, local sensing, fused graph and receipt times.

    Graph channels cover 400 x 400 m around the vehicle, world aligned.
    Local occupancy uses its native 8 x 8 m extent, resized to 32 x 32.
    Empty graph cells mean no graph evidence, NOT known free terrain.
    """
    p = np.asarray(controller.position, float)[:2]
    raster = np.zeros((CHANNELS, SIDE, SIDE), np.float32)
    graph = controller.unified_graph
    nodes = edges = 0

    def pixels(points):
        xy = np.floor((np.asarray(points) - p + GRAPH_HALF_EXTENT) / (2 * GRAPH_HALF_EXTENT) * SIDE).astype(int)
        return xy[np.all((xy >= 0) & (xy < SIDE), axis=1)]

    if graph is not None:
        g = graph.graph
        nodes, edges = g.number_of_nodes(), g.number_of_edges()
        for _, d in g.nodes(data=True):
            xy = pixels([d['position'][:2]])
            if len(xy):
                x, y = xy[0]
                raster[0, y, x] += .2
                raster[1, y, x] = max(raster[1, y, x], float(d.get('confidence', 0)))
        for a, b in g.edges():
            a, b = np.asarray(g.nodes[a]['position'][:2]), np.asarray(g.nodes[b]['position'][:2])
            if np.any(np.maximum(a, b) < p - GRAPH_HALF_EXTENT) or np.any(np.minimum(a, b) > p + GRAPH_HALF_EXTENT):
                continue
            xy = pixels(np.linspace(a, b, max(2, int(np.linalg.norm(a-b) / 6) + 1)))
            if len(xy):
                raster[2, xy[:, 1], xy[:, 0]] = 1.
    semantic = controller._last_semantic_map
    if semantic is not None:
        yy = np.minimum(np.arange(SIDE) * semantic.shape[0] // SIDE, semantic.shape[0] - 1)
        xx = np.minimum(np.arange(SIDE) * semantic.shape[1] // SIDE, semantic.shape[1] - 1)
        raster[3] = np.isin(semantic[np.ix_(yy, xx)], [2, 4])
    ages, receipts = [], []
    for i in range(3):
        key = f'drone_{i}'
        stamp = controller.last_fusion_per_drone.get(key, -np.inf)
        ages.append(float(np.clip((now - stamp) / 60., 0, 1)))
        receipts.append(float(controller.current_drone_communications.get(key, {}).get('received', False)))
    vector = np.asarray([*(p / 1000.), np.sin(controller.heading), np.cos(controller.heading),
                         *np.asarray(velocity)[:2], *(np.asarray(final_goal)[:2] / 1000.),
                         np.log1p(nodes) / np.log(10001), np.log1p(edges) / np.log(20001),
                         *ages, *receipts], np.float32)
    if vector.shape != (FEATURES,) or not np.isfinite(vector).all():
        raise ValueError('Invalid vehicle observation')
    return np.clip(raster, 0, 1), vector


def build_environment(seed, *, max_target_distance=200., manager_options=None,
                      scene_path=None, scene_sha256=None, mission_planner=None,
                      decoupled_planner=None, terrain_backend='circles'):
    if mission_planner is not None and decoupled_planner is not None:
        raise ValueError('Select one action contract per environment')
    if terrain_backend not in ('circles', 'wildscenes_simplified'):
        raise ValueError('Unknown terrain backend')
    if terrain_backend != 'circles' and (decoupled_planner is None or scene_path is None):
        raise ValueError('WildScenes requires an independent planner and certified scene')
    if not np.isfinite(max_target_distance) or not 0 < max_target_distance <= GRAPH_HALF_EXTENT:
        raise ValueError('Target distance must fit the current 200m graph-input half extent')
    import torch
    sys.dont_write_bytecode = True
    runtime = FROZEN / 'runtime_sources'
    sys.path.insert(0, str(runtime))
    from drone.mappo import ActorCritic
    from simulator.mappo_experiment import Experiment, SafetyTermination
    import simulator.mappo_experiment as experiment_module
    from simulator.large_world_scenarios import generate_training_scene
    if not Path(inspect.getfile(Experiment)).resolve().is_relative_to(runtime):
        raise RuntimeError('Mutable simulator imported instead of frozen runtime')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    checkpoint = torch.load(FROZEN / 'best.pt', map_location='cpu', weights_only=False)
    model = ActorCritic.from_state_dict(checkpoint['model']).eval()
    model.requires_grad_(False)
    cfg = json.loads((FROZEN / 'config.json').read_text())
    certificate = None
    if scene_path is None:
        scene = generate_training_scene(seed, cfg['task_distances'],
                                        distance_range=cfg['task_distance_range'],
                                        distance_sampling=cfg['distance_sampling'])
    elif terrain_backend == 'wildscenes_simplified':
        from world_model.decoupled_scenes import load_scene
        scene, certificate = load_scene(scene_path, scene_sha256)
        cfg['world_size'] = scene['world']
        if int(scene['seed']) != int(seed):
            raise ValueError('Certified WildScenes seed differs from the requested seed')
    else:
        from .feasibility import load_certified_scene
        scene, certificate = load_certified_scene(scene_path, expected_sha256=scene_sha256,
                                                  clearance=1.5)
        if int(scene['seed']) != int(seed):
            raise ValueError('Certified scene does not match the requested episode seed')
    if not np.isclose(scene['world'], cfg['world_size']):
        raise ValueError('Certified scene bounds differ from the simulation configuration')
    keys = set(inspect.signature(Experiment).parameters) & set(cfg)
    options = {key: cfg[key] for key in keys if key not in ('seed', 'model')}
    options.update(mapping_until_connected=cfg['mapping_reward_until_connected'],
                   subgoal_radius=max_target_distance, subgoal_arrival=1.,
                   num_drones=3, stochastic=False,
                   scene=(scene['goal'], scene['circles'], scene['attempts']),
                   start=scene['start'], stop_on_ground_arrival=False)
    # These are the frozen experiment's configured fleet/sensor constants, not
    # simulator fleet positions or hidden obstacle truth.
    envelope = ReconnaissanceEnvelope(drone_count=options['num_drones'],
                                      communication_range=experiment_module.COMMUNICATION_RANGE_M,
                                      altitude=experiment_module.FLIGHT_HEIGHT_M,
                                      footprint_width=24.)
    original = experiment_module.GroundRobotController
    mission_options = {}
    if mission_planner is not None:
        from .mission import WorldModelMissionManager
        mission_options = dict(manager_factory=WorldModelMissionManager,
                               planner_factory=lambda: mission_planner)
    original_drone = experiment_module.DroneController
    original_environment = experiment_module.MultiRobotEnv
    if terrain_backend == 'wildscenes_simplified':
        from world_model.decoupled_scenes import geometry_adapter, geometry_for
        geometry = geometry_for(scene)
    if decoupled_planner is not None:
        from world_model.decoupled_runtime import (ground_navigation_controller,
            make_decoupled_controller, make_information_drone_controller)
        experiment_module.GroundRobotController = make_decoupled_controller(
            ground_navigation_controller(), reconnaissance=envelope, max_target_distance=max_target_distance,
            manager_options=manager_options, planner_factory=lambda: decoupled_planner)
        experiment_module.DroneController = make_information_drone_controller(original_drone)
        if terrain_backend == 'wildscenes_simplified':
            def native_environment(config):
                instance = original_environment(config)
                return geometry_adapter().install_legacy_environment(instance, geometry)

            experiment_module.MultiRobotEnv = native_environment
    else:
        experiment_module.GroundRobotController = make_vtmp2_controller(
            original, reconnaissance=envelope, max_target_distance=max_target_distance,
            manager_options=manager_options, **mission_options)
    try:
        env = Experiment(seed, model, **options)
    finally:
        experiment_module.GroundRobotController = original
        experiment_module.DroneController = original_drone
        experiment_module.MultiRobotEnv = original_environment
    sensor = env.env.config
    if (not np.isclose(sensor.occupancy_map_size * sensor.occupancy_map_resolution, envelope.footprint_width)
            or not np.isclose(sensor.communication_radius, envelope.communication_range)):
        raise ValueError('Reconnaissance envelope must match the configured radio and sensor footprint')
    env.high_level_algorithm = ALGORITHM
    # Audit metadata belongs to the experiment, never the controller/encoder.
    env.physical_feasibility = certificate
    if terrain_backend == 'wildscenes_simplified':
        from world_model.decoupled_scenes import geometry_hash
        env.scene_geometry_sha256 = geometry_hash(scene)
    else:
        geometry = {key: scene[key] for key in ('start', 'goal', 'world', 'circles')}
        env.scene_geometry_sha256 = hashlib.sha256(
            json.dumps(geometry, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if certificate is not None and not np.isclose(env.ground_controller.local_planner.safety_distance, 1.5):
        raise ValueError('Certified clearance and controller safety distance differ')
    return env, SafetyTermination


def collect_episode(spec):
    """Each file is one uninterrupted scene, with N+1 observations / N actions."""
    if isinstance(spec, dict):
        seed, split, steps, output = (spec[key] for key in ('seed', 'split', 'steps', 'output'))
        scene_path, scene_digest = spec['scene_path'], spec['scene_sha256']
        if json.loads(Path(scene_path).read_text())['split'] != split:
            raise ValueError('Certified scene split differs from the collection split')
    else:
        seed, split, steps, output = spec
        scene_path = scene_digest = None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = f'{split}_{seed}'
    started = time.time()
    provenance()  # Fail closed if the frozen controller changed.
    if steps < HORIZON or steps % HORIZON:
        raise ValueError('Episode budget must be a positive multiple of the horizon')
    if (output / (name + '.npz')).exists():
        raise FileExistsError(name)
    # Existing runtime logs are extremely verbose; keep exceptions, not stdout.
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
        environment_options = (dict(scene_path=scene_path, scene_sha256=scene_digest)
                               if scene_path is not None else {})
        env, safety_exception = build_environment(seed, **environment_options)
        maps, states, actions, labels, elapsed, terminal = [], [], [], [], [], []
        raw_targets, times, switches, switch_reasons, target_statuses = [], [float(env.env.time)], [], [], []
        m, s = encode(env.ground_controller, env.goal, env.env.time, (0, 0))
        maps.append(m); states.append(s)
        reason = 'budget'
        frame = 0
        velocity = np.zeros(2)
        while frame < steps:
            target = env.subgoals.target.copy()
            is_final = np.allclose(target, env.goal, atol=1e-6, rtol=0.)
            target_status = env.subgoals.target_status
            begin = float(env.env.time)
            p0 = np.asarray(env.ground_controller.position).copy()
            rejection = False
            reached = False
            switched = False
            for _ in range(min(HORIZON, steps-frame)):
                # env.step executes the cached old command; selection happens
                # once at its end. Never mix two targets in one transition.
                if not np.allclose(env.subgoals.target, target, atol=1e-6, rtol=0.):
                    raise AssertionError('Target changed inside one macro action')
                before = np.asarray(env.ground_controller.position).copy()
                try:
                    env.step(frame)
                except safety_exception:
                    rejection = True
                    reason = 'cbf_infeasible'
                    break
                frame += 1
                velocity = np.asarray(env.ground_controller.position) - before
                reached |= np.linalg.norm(np.asarray(env.ground_controller.position) - target) <= (2. if is_final else env.subgoals.arrival)
                switched = not np.allclose(env.subgoals.target, target, atol=1e-6, rtol=0.)
                if env.goal_reached:
                    reason = 'arrival'
                    break
                if switched:
                    break
            dt = float(env.env.time) - begin
            m, s = encode(env.ground_controller, env.goal, env.env.time, velocity)
            maps.append(m); states.append(s)
            actions.append([*(target / 1000.), 1.])  # Requested, NOT realized duration.
            raw_targets.append(target)
            switches.append(switched)
            switch_reasons.append(env.subgoals.last_reason if switched else '')
            target_statuses.append(target_status)
            displacement = np.linalg.norm(np.asarray(env.ground_controller.position) - p0)
            labels.append([float(reached), float(displacement < .5 and dt >= HORIZON), float(rejection)])
            elapsed.append(dt / HORIZON)
            terminal.append(bool(rejection or env.goal_reached))
            times.append(float(env.env.time))
            atomic_json(output / (name + '.progress.json'), dict(seed=seed, split=split, step=frame,
                        budget=steps, wall_seconds=time.time()-started, stage='collecting',
                        algorithm=ALGORITHM, target_status=env.subgoals.target_status))
            if terminal[-1]:
                break
    arrays = dict(maps=np.asarray(maps, np.float32), states=np.asarray(states, np.float32),
                  actions=np.asarray(actions, np.float32), labels=np.asarray(labels, np.float32),
                  elapsed=np.asarray(elapsed, np.float32), terminated=np.asarray(terminal, bool),
                  truncated=np.asarray([False]*(len(actions)-1)+[not terminal[-1]], bool),
                  targets=np.asarray(raw_targets, np.float32), times=np.asarray(times, np.float32),
                  target_switched=np.asarray(switches, bool), switch_reason=np.asarray(switch_reasons),
                  target_status=np.asarray(target_statuses), algorithm=np.asarray(ALGORITHM),
                  schema_version=np.asarray(SCHEMA_VERSION),
                  scene_sha256=np.asarray(scene_digest or ''),
                  scene_geometry_sha256=np.asarray(getattr(env, 'scene_geometry_sha256', '')),
                  feasibility_validated=np.asarray(scene_path is not None),
                  physical_clearance_m=np.asarray(1.5 if scene_path is not None else 0.),
                  seed=np.asarray(seed), split=np.asarray(split), checkpoint_sha256=np.asarray(CHECKPOINT_SHA))
    path = output / (name + '.npz')
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix('.tmp').replace(path)
    summary = dict(algorithm=ALGORITHM, schema_version=SCHEMA_VERSION,
                   scene_path=str(scene_path) if scene_path is not None else None,
                   scene_sha256=scene_digest,
                   scene_geometry_sha256=getattr(env, 'scene_geometry_sha256', None),
                   physical_feasibility=getattr(env, 'physical_feasibility', None),
                   selector_mode='geometric_reconnaissance',
                   reconnaissance=env.ground_controller.reconnaissance.describe(),
                   seed=seed, split=split, steps=frame, transitions=len(actions), reason=reason,
                   target_changes=env.subgoals.selected, wall_seconds=time.time()-started,
                   file=str(path), sha256=sha256(path), initial_distance_m=env.initial_goal_distance,
                   final_distance_m=float(np.linalg.norm(np.asarray(env.ground_controller.position)-env.goal)))
    atomic_json(output / (name + '.json'), summary)
    return summary
