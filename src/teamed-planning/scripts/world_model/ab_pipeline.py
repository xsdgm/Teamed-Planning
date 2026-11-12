"""Freeze source, collect proxy episodes, train, then test learned A/B decisions.

Run with python -I -B path/to/world_model/ab_pipeline.py --output <new directory>.
No v11-2 weights are loaded, modified or required by this first stage.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from simulator.large_world_scenarios import generate_training_scene
from world_model.ab_environment import (
    ACTION_CONTRACT, ALGORITHM, SurveyNavigationEnv, MappingConfig, ProxyMapper,
)
from world_model.ab_planner import ModelPlanner, bootstrap_select
from world_model.ab_training import atomic_json, load_model, train


def scene_for(seed, config):
    return generate_training_scene(seed, world=config.world, distance_range=(100., 400.),
                                   distance_sampling='annulus_area', ground_safety_distance=config.clearance)


def geometry_hash(scene):
    keys = ('world', 'start', 'goal', 'terrain_region_id') if 'terrain_region_id' in scene else ('world', 'start', 'goal', 'circles')
    geometry = {key: scene[key] for key in keys}
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def collect_episode(scene, config, max_steps, max_seconds, *, model=None, device='cpu', exploration=0.):
    env = SurveyNavigationEnv(scene, config)
    rng = np.random.default_rng(int(scene['seed']) ^ 0xABC3)
    planner = ModelPlanner(model, config, device) if model is not None else None
    observation = env.observe()
    maps, states = [observation[0]], [observation[1]]
    actions, outcomes, decisions = [], [], []
    for _ in range(max_steps):
        if planner is None:
            action, diagnostic = bootstrap_select(env, rng)
        else:
            action, diagnostic = planner.choose(observation, exploration=exploration, rng=rng)
        observation, outcome, info = env.step(action)
        maps.append(observation[0]); states.append(observation[1])
        actions.append(np.asarray(action).reshape(4)/1000.); outcomes.append(outcome)
        decisions.append(dict(**info, decision=diagnostic))
        if env.done or env.elapsed >= max_seconds:
            break
    n = len(actions)
    arrays = dict(maps=np.asarray(maps, np.float32), states=np.asarray(states, np.float32),
        actions=np.asarray(actions, np.float32), outcomes=np.asarray(outcomes, np.float32),
        terminated=np.array([False]*(n-1)+[env.done]), truncated=np.array([False]*(n-1)+[not env.done]),
        algorithm=np.asarray(ALGORITHM), action_contract=np.asarray(ACTION_CONTRACT),
        seed=np.asarray(scene['seed']), geometry_sha256=np.asarray(geometry_hash(scene)),
        backend=np.asarray(env.mapper.name))
    if 'terrain_region_id' in scene:
        arrays.update(terrain_region_id=np.asarray(scene['terrain_region_id']),
            terrain_source_sha256=np.asarray(scene['terrain_source_sha256']),
            terrain_origin_xy=np.asarray(scene['terrain_origin_xy']),
            terrain_world_m=np.asarray(scene['world']))
    summary = dict(seed=scene['seed'], reached=env.done, transitions=n,
        elapsed_seconds=env.elapsed, goal_distance_m=float(np.linalg.norm(env.goal-env.position)),
        failure_tasks=sum(row['status'] != 'complete' for row in decisions),
        received_cells=int(np.count_nonzero(env.received.cells >= 0)),
        survey_seconds=sum(row['survey_seconds'] for row in decisions),
        vehicle_distance_m=sum(row['vehicle_distance_m'] for row in decisions),
        max_survey_ground_displacement_m=max(row['survey_ground_displacement_m'] for row in decisions),
        selector='world_model_mpc' if planner else 'bootstrap_observed_map',
        termination='goal' if env.done else 'time_budget' if env.elapsed >= max_seconds else 'step_budget')
    return arrays, decisions, summary


def save_episode(output, split, scene, config, args, *, model=None, exploration=0.):
    seed = scene['seed']
    arrays, decisions, summary = collect_episode(scene, config, args.max_steps, args.max_seconds,
                                                model=model, device=args.device, exploration=exploration)
    arrays['split'] = np.asarray(split)
    path = output/'data'/f'{split}_{seed}.npz'
    if path.exists():
        raise ValueError(f'Would overwrite episode {path}')
    np.savez_compressed(path, **arrays)
    atomic_json(output/'data'/f'{split}_{seed}.decisions.json', decisions)
    print(json.dumps(dict(stage='collected', split=split, **summary)), flush=True)
    return summary


def freeze_sources(output, args):
    scripts = Path(__file__).resolve().parents[1]
    sources = output/'runtime_sources'
    manifest = {}
    for package in ('world_model', 'common', 'simulator', 'drone', 'ground', 'tests'):
        for source in sorted((scripts/package).rglob('*.py')):
            relative = source.relative_to(scripts)
            contents = source.read_bytes()
            destination = sources/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
            digest = hashlib.sha256(contents).hexdigest()
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Source changed while freezing: {relative}')
            manifest[str(relative)] = digest
    source_config = Path(args.v11_config).resolve()
    shutil.copyfile(source_config, output/'v11_2_config.json')
    manifest['v11_2_config.json'] = hashlib.sha256((output/'v11_2_config.json').read_bytes()).hexdigest()
    if args.wildscenes_manifest:
        from world_model.ab_wildscenes import freeze_manifest
        frozen = freeze_manifest(args.wildscenes_manifest, output/'wildscenes_inputs')
        for item in frozen.parent.glob('*.json'):
            manifest[str(item.relative_to(output))] = hashlib.sha256(item.read_bytes()).hexdigest()
    atomic_json(output/'source_manifest.json', manifest)
    return sources/'world_model/ab_pipeline.py'


def mapping_config(args):
    source = json.loads(Path(args.v11_config).read_text())
    world = source['world_size']
    if args.wildscenes_manifest:
        from world_model.ab_wildscenes import load_manifest
        world = load_manifest(args.wildscenes_manifest)['world_size_m']
    return MappingConfig(world=world, resolution=source['survey_resolution'],
        half_width=source['survey_half_width'], clearance=source['ground_safety_distance'],
        ground_speed=source['ground_speed'], fleet_size=source['fleet_sizes'][0],
        failure_probability=args.failure_probability,
        distance_failure_probability=args.distance_failure_probability,
        communication_failure_probability=args.communication_failure_probability,
        setup_seconds=args.setup_seconds, survey_speed=args.survey_speed,
        delay_jitter=args.delay_jitter, timeout_seconds=args.timeout_seconds)


def run(args):
    output = Path(args.output).resolve()
    cfg = mapping_config(args)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    provider = None
    backend_name = ProxyMapper.name
    if args.wildscenes_manifest:
        from world_model.ab_wildscenes import WildScenesSource, WildScenesMapper
        provider = WildScenesSource(args.wildscenes_manifest, cfg)
        backend_name = WildScenesMapper.name
        atomic_json(output/'terrain_manifest.json', provider.spec)
    def sample_scene(seed, split):
        return provider.scene(seed, split) if provider else scene_for(seed, cfg)
    atomic_json(output/'config.json', dict(algorithm=ALGORITHM, action_contract=ACTION_CONTRACT,
        args=vars(args), mapping=asdict(cfg), backend=backend_name,
        terrain='wildscenes_simplified' if provider else 'random_circles',
        proxy_calibrated_to_v11_2=False, real_uav_policy_used=False,
        navigation='Dijkstra on conservative received-cell navigation graph; bounded kinematic path follower',
        model_selection='one_macro_step_model_predictive_control',
        action='[ax, ay, bx, by] / 1000; width fixed to v11-2 config',
        scene_truth_in_policy=False))
    (output/'data').mkdir(); (output/'scenes').mkdir()
    manifest_hash = hashlib.sha256((output/'source_manifest.json').read_bytes()).hexdigest()
    summaries, scenes = [], {}
    seeds = {'train': list(range(args.train_seed, args.train_seed+args.train_scenes)),
             'val': list(range(args.val_seed, args.val_seed+args.val_scenes)),
             'test': list(range(args.test_seed, args.test_seed+args.test_scenes))}
    if len(set(sum(seeds.values(), []))) != sum(map(len, seeds.values())):
        raise ValueError('Scene splits must not overlap')
    for split in ('train', 'val'):
        for seed in seeds[split]:
            atomic_json(output/'status.json', dict(stage='collecting', split=split, seed=seed, pid=os.getpid()))
            scene = sample_scene(seed, split)
            scenes[seed] = scene
            atomic_json(output/'scenes'/f'{split}_{seed}.json', scene)
            summaries.append(save_episode(output, split, scene, cfg, args))
    atomic_json(output/'collection_manifest.json', summaries)
    model, results = None, []
    for round_index in range(args.rounds):
        if round_index:
            for index in range(args.train_scenes):
                seed = args.train_seed+round_index*10000+index
                if seed in set(sum(seeds.values(), [])) or seed in scenes:
                    raise ValueError('Aggregated training seed collides with an existing scene')
                atomic_json(output/'status.json', dict(stage='collecting', split='train', round=round_index+1,
                                                      seed=seed, pid=os.getpid()))
                scene = sample_scene(seed, 'train'); scenes[seed] = scene
                atomic_json(output/'scenes'/f'train_{seed}.json', scene)
                summaries.append(save_episode(output, 'train', scene, cfg, args, model=model, exploration=.2))
            atomic_json(output/'collection_manifest.json', summaries)
        atomic_json(output/'status.json', dict(stage='training', round=round_index+1, pid=os.getpid()))
        model, result = train(output/'data', output/f'training_{round_index+1}', epochs=args.epochs,
            device=args.device, model=model, seed=args.seed+round_index,
            mapping_config=asdict(cfg), source_manifest_sha256=manifest_hash)
        results.append(result)
    test_results = []
    for seed in seeds['test']:
        atomic_json(output/'status.json', dict(stage='closed_loop_test', seed=seed, pid=os.getpid()))
        scene = sample_scene(seed, 'test')
        if geometry_hash(scene) in {geometry_hash(s) for s in scenes.values()}:
            raise ValueError('Test scene geometry overlaps training or validation')
        atomic_json(output/'scenes'/f'test_{seed}.json', scene)
        for mode, selected_model in (('model', model), ('bootstrap', None)):
            arrays, decisions, summary = collect_episode(scene, cfg, args.max_steps, args.max_seconds,
                                                        model=selected_model, device=args.device)
            atomic_json(output/f'test_{seed}_{mode}.json', dict(summary=summary, decisions=decisions))
            test_results.append(summary)
            print(json.dumps(dict(stage='test', **summary)), flush=True)
    atomic_json(output/'closed_loop_test.json', dict(results=test_results,
        test_used_for_model_selection=False, real_v11_2_evaluated=False,
        comparison='same scenes and proxy RNG seed; task-dependent outcomes may diverge',
        limitations=['uncalibrated stochastic mapping backend', 'ideal occupancy inside delivered swaths',
                     'kinematic ground follower', 'one-step MPC', 'no UAV dynamics or CBF execution']))
    complete = dict(stage='complete', pid=os.getpid(), rounds=args.rounds, training=results,
        test_scenes=args.test_scenes, real_v11_2_used=False, output=str(output))
    atomic_json(output/'status.json', complete)
    print(json.dumps(dict(stage='complete', output=str(output))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    default_config = Path(__file__).resolve().parents[2]/'config/algorithm/mappo_training_v11_2.json'
    parser.add_argument('--v11-config', default=str(default_config))
    parser.add_argument('--wildscenes-manifest', help='Spatially separated simplified WildScenes regions')
    parser.add_argument('--frozen', action='store_true', help=argparse.SUPPRESS)
    for name, default in (('train-scenes', 32), ('val-scenes', 8), ('test-scenes', 4),
                          ('train-seed', 210000), ('val-seed', 310000), ('test-seed', 410000),
                          ('max-steps', 32), ('epochs', 100), ('rounds', 1), ('threads', 2), ('seed', 20261110)):
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--device', default='cpu')
    for name, default in (('max-seconds', 2400.), ('failure-probability', .15),
                          ('distance-failure-probability', .15), ('communication-failure-probability', .05),
                          ('setup-seconds', 8.), ('survey-speed', 2.), ('delay-jitter', .5), ('timeout-seconds', 180.)):
        parser.add_argument('--'+name, type=float, default=default)
    args = parser.parse_args()
    if any(getattr(args, name) < 1 for name in ('train_scenes', 'val_scenes', 'test_scenes', 'max_steps', 'epochs', 'rounds', 'threads')):
        parser.error('Scene counts, budgets, rounds and threads must be positive')
    if not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error('max-seconds must be finite and positive')
    mapping_config(args)  # Validate before creating a run directory.
    output = Path(args.output).resolve()
    if not args.frozen:
        output.mkdir(parents=True, exist_ok=False)
        frozen_entry = freeze_sources(output, args)
        forwarded = [value for value in sys.argv[1:]]
        command = [sys.executable, '-I', '-B', str(frozen_entry), *forwarded,
                   '--v11-config', str(output/'v11_2_config.json'), '--output', str(output), '--frozen']
        if args.wildscenes_manifest:
            command.extend(['--wildscenes-manifest', str(output/'wildscenes_inputs/manifest.json')])
        atomic_json(output/'launch.json', dict(command=command, started_at=time.time()))
        return subprocess.call(command)
    try:
        run(args)
    except BaseException as exc:
        atomic_json(output/'status.json', dict(stage='failed', error=repr(exc), pid=os.getpid()))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
