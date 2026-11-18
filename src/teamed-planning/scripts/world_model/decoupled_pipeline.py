"""Collect, train and evaluate vtmp2 with independent vehicle/aerial decisions.

The v9 aircraft checkpoint is always frozen. Full terrain is a training label
only; all online choices use received sensors/telemetry. Evaluation requires a
new schema-4 world-model checkpoint and separate physical scenes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

SOURCE = Path(__file__).resolve().parent
if __package__ in (None, ''):
    sys.path.insert(0, str(SOURCE.parent))
    __package__ = 'world_model'

from .hierarchical_sources import package_python_files

import numpy as np

from .data import CHECKPOINT_SHA, HORIZON, atomic_json, build_environment, provenance, sha256
from .decoupled_observation import (ACTION_CONTRACT, ARCHITECTURE, DRONE_FEATURES,
    GROUND_FEATURES, SCHEMA_VERSION, VEHICLE_RUNTIME, action_vector, cell_centers, motion_state)
from .decoupled_planner import TerrainPlanner


def terrain_labels(scene, origin):
    """Offline occupancy supervision; this function is never called by a policy."""
    xy = cell_centers(origin)
    if scene.get('terrain') == 'wildscenes_simplified':
        from .decoupled_scenes import geometry_for
        return geometry_for(scene).truth(xy).astype(np.float32)
    occupied = np.zeros(xy.shape[:2], bool)
    for center, radius in scene['circles']:
        occupied |= np.linalg.norm(xy-np.asarray(center), axis=-1) <= radius
    occupied |= np.any((xy < 0) | (xy >= scene['world']), axis=-1)
    return occupied.astype(np.float32)


def current_action(controller):
    return action_vector(controller.drive_target, controller.survey.target)


def task_step_budget(reference_length, maximum, steps_per_metre=4.):
    """Offline reference sets time allowance only; it never guides navigation."""
    if (not np.isfinite([reference_length, steps_per_metre]).all()
            or reference_length <= 0 or steps_per_metre <= 0
            or type(maximum) is not int or maximum < 1):
        raise ValueError('Positive reference length, budget and steps/metre required')
    return min(maximum, max(600, math.ceil(reference_length*steps_per_metre+240)))


def navigation_summary(results):
    kilometer = [row for row in results if row['initial_distance_m'] >= 1000.]
    successes = [row for row in kilometer if row['kilometer_success']]
    return dict(tasks=len(results), kilometer_tasks=len(kilometer),
        goal_arrivals=sum(row['goal_reached'] for row in results),
        kilometer_successes=len(successes),
        kilometer_success_rate=len(successes)/len(kilometer) if kilometer else None,
        definition='initial Euclidean distance >= 1000m, final arrival, no audited collisions or 1.5m clearance violations',
        selector_modes=sorted({row['selector_mode'] for row in results}),
        mean_success_elapsed_seconds=float(np.mean([row['elapsed_seconds'] for row in successes])) if successes else None,
        mean_success_path_length_m=float(np.mean([row['ground_path_length_m'] for row in successes])) if successes else None)


def collect_episode(job):
    """Split a transition whenever EITHER independently executing target changes."""
    seed, split = int(job['seed']), job['split']
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=True)
    path = output/f'{split}_{seed}.npz'
    if path.exists():
        raise FileExistsError(path)
    provenance()  # Verify exactly the requested frozen v9 weights.
    planner = (TerrainPlanner.from_checkpoint(job['checkpoint'], device='cpu')
               if job.get('checkpoint') else TerrainPlanner(seed=seed))
    document = json.loads(Path(job['scene_path']).read_text())
    if document['split'] != split:
        raise ValueError('Certified scene and episode splits differ')
    scene = document['scene']
    started = time.time()
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
        env, safety_exception = build_environment(seed, scene_path=job['scene_path'],
            scene_sha256=job['scene_sha256'], decoupled_planner=planner,
            terrain_backend=job.get('terrain', 'circles'),
            manager_options=dict(review_interval=5., min_hold=0., switch_improvement=2.))
        controller = env.ground_controller
        initial_position = np.asarray(controller.position).copy()
        initial_distance = float(np.linalg.norm(initial_position-env.goal))
        audit_geometry = None
        if scene.get('terrain') == 'wildscenes_simplified':
            from .decoupled_scenes import geometry_for
            audit_geometry = geometry_for(scene)
        maps, states, origins = [[np.asarray(value).copy()] for value in controller.world_observation]
        actions, durations, truths, information, motion, motion_masks = [], [], [], [], [], []
        terminal, times, decisions = [], [float(env.env.time)], []
        frame, reason, simultaneous_steps, simultaneous_distance = 0, 'budget', 0, 0.
        trajectory, trajectory_times = [initial_position], [float(env.env.time)]
        travelled = stationary = max_stationary = 0.
        clearance_violations = 0 if audit_geometry is not None else None
        while frame < job['steps']:
            action = current_action(controller)
            initial_maps, initial_states, origin = maps[-1], states[-1], origins[-1]
            begin = float(env.env.time)
            rejected = False
            for _ in range(min(HORIZON, job['steps']-frame)):
                if not np.allclose(current_action(controller), action, atol=1e-7, rtol=0):
                    raise AssertionError('Independent target changed inside a labelled action')
                before = np.asarray(controller.position).copy()
                step_time = float(env.env.time)
                try:
                    env.step(frame)
                except safety_exception:
                    rejected, reason = True, 'cbf_infeasible'
                    break
                frame += 1
                after = np.asarray(controller.position).copy()
                distance = float(np.linalg.norm(after-before))
                travelled += distance
                stationary = stationary+float(env.env.time)-step_time if distance < .01 else 0.
                max_stationary = max(max_stationary, stationary)
                trajectory.append(after); trajectory_times.append(float(env.env.time))
                if audit_geometry is not None:
                    clearance_violations += int(not audit_geometry.segment_clear(before, after, 1.5))
                if action[2] and action[5]:
                    simultaneous_steps += 1
                    simultaneous_distance += distance
                if env.goal_reached:
                    reason = 'arrival'
                    break
                if not np.allclose(current_action(controller), action, atol=1e-7, rtol=0):
                    break
            next_maps, next_states, next_origin = controller.world_observation
            # Newly received information is measured in the SAME world frame;
            # recentering the vehicle crop cannot create fake exploration gain.
            received_after = controller.observation_memory.raster(origin)[1]
            information.append(np.maximum(received_after-initial_maps[1], 0)*(1-initial_maps[2]))
            truths.append(terrain_labels(scene, origin))
            motion.append((motion_state(next_states, planner.drone_count)-
                           motion_state(initial_states, planner.drone_count))/.01)
            mask = np.ones(2+3*planner.drone_count, np.float32)
            for i in range(planner.drone_count):
                slot = GROUND_FEATURES+i*DRONE_FEATURES
                valid = (initial_states[slot+7] and next_states[slot+7]
                         and initial_states[slot+11] and next_states[slot+11])
                mask[2+3*i:5+3*i] = float(bool(valid))
            motion_masks.append(mask)
            maps.append(next_maps.copy()); states.append(next_states.copy()); origins.append(next_origin.copy())
            actions.append(action)
            times.append(float(env.env.time))
            durations.append(times[-1]-begin)
            terminal.append(bool(rejected or env.goal_reached))
            decisions.append(dict(time=times[-1], executed_action=action.tolist(),
                next_action=current_action(controller).tolist(),
                ground_position=np.asarray(controller.position).tolist(),
                ground_status=controller.subgoals.target_status,
                ground_reason=controller.subgoals.last_reason,
                ground_loop_recovery=bool(getattr(controller.subgoals, '_loop_recovery', False)),
                survey_completed=controller.survey.completed,
                predictions=dict(planner.last_diagnostics)))
            atomic_json(path.with_suffix('.progress.json'), dict(seed=seed, split=split,
                stage='collecting', step=frame, budget=job['steps'],
                elapsed_seconds=times[-1]-times[0], initial_distance_m=initial_distance,
                ground_position=np.asarray(controller.position).tolist(),
                remaining_distance_m=float(np.linalg.norm(np.asarray(controller.position)-env.goal)),
                ground_path_length_m=travelled, max_stationary_seconds=max_stationary,
                ground_loop_recovery=bool(getattr(controller.subgoals, '_loop_recovery', False)),
                drive_target=None if controller.drive_target is None else controller.drive_target.tolist(),
                execution_target=(None if getattr(controller, 'execution_target', None) is None
                                  else controller.execution_target.tolist()),
                survey_target=None if controller.survey.target is None else controller.survey.target.tolist()))
            if terminal[-1]:
                break
    arrays = dict(maps=np.asarray(maps, np.uint8), states=np.asarray(states, np.float32),
        origins=np.asarray(origins, np.int64), actions=np.asarray(actions, np.float32),
        durations=np.asarray(durations, np.float32), terrain=np.asarray(truths, np.uint8),
        newly_observed=np.asarray(information, np.uint8), motion=np.asarray(motion, np.float32),
        motion_mask=np.asarray(motion_masks, np.float32), times=np.asarray(times),
        terminated=np.asarray(terminal, bool),
        truncated=np.asarray([False]*(len(actions)-1)+[not terminal[-1]], bool),
        action_contract=np.asarray(ACTION_CONTRACT), algorithm=np.asarray('vtmp2'),
        schema_version=np.asarray(SCHEMA_VERSION), drone_count=np.asarray(planner.drone_count),
        seed=np.asarray(seed), split=np.asarray(split), checkpoint_sha256=np.asarray(CHECKPOINT_SHA),
        scene_sha256=np.asarray(job['scene_sha256']),
        scene_geometry_sha256=np.asarray(env.scene_geometry_sha256),
        feasibility_validated=np.asarray(True))
    arrays['vehicle_runtime'] = np.asarray(VEHICLE_RUNTIME)
    arrays.update(trajectory=np.asarray(trajectory, np.float32), trajectory_times=np.asarray(trajectory_times))
    arrays['terrain_backend'] = np.asarray(job.get('terrain', 'circles'))
    if scene.get('terrain') == 'wildscenes_simplified':
        for key in ('terrain_source_sha256', 'terrain_origin_xy', 'terrain_map_id', 'terrain_region_id', 'terrain_split'):
            arrays[key] = np.asarray(scene[key])
        arrays['terrain_world_m'] = np.asarray(scene['world'])
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix('.tmp').replace(path)
    collision_attempts = getattr(env.env, 'legacy_collision_attempts', None)
    final_distance = float(np.linalg.norm(np.asarray(controller.position)-env.goal))
    goal_reached = bool(env.goal_reached and final_distance <= 2.)
    summary = dict(algorithm='vtmp2', schema_version=SCHEMA_VERSION, action_contract=ACTION_CONTRACT,
        seed=seed, split=split, steps=frame, transitions=len(actions), reason=reason,
        selector_mode=planner.selector_mode, simultaneous_steps=simultaneous_steps,
        ground_distance_during_survey_m=simultaneous_distance,
        initial_distance_m=initial_distance, reference_length_m=scene.get('reference_length'),
        goal_reached=goal_reached, elapsed_seconds=times[-1]-times[0],
        budget_steps=job['steps'], ground_path_length_m=travelled,
        max_stationary_seconds=max_stationary, ground_target_changes=getattr(controller.subgoals, 'selected', 0),
        ground_target_failures=getattr(controller.subgoals, 'failed', 0),
        collision_attempts=collision_attempts, clearance_violation_steps=clearance_violations,
        kilometer_task=initial_distance >= 1000.,
        kilometer_success=bool(initial_distance >= 1000. and goal_reached
                               and collision_attempts == 0 and clearance_violations == 0),
        final_distance_m=final_distance,
        survey_selected=controller.survey.selected, survey_completed=controller.survey.completed,
        v9_checkpoint_sha256=CHECKPOINT_SHA, v9_weights_trained=False,
        vehicle_runtime=VEHICLE_RUNTIME,
        world_model_vehicle_runtime=planner.training_vehicle_runtime,
        model_runtime_transfer=bool(job.get('checkpoint') and
                                   planner.training_vehicle_runtime != VEHICLE_RUNTIME),
        terrain=job.get('terrain', 'circles'),
        source_unknown_space=scene.get('source_unknown_space'),
        scene_geometry_sha256=env.scene_geometry_sha256, file=str(path), sha256=sha256(path),
        wall_seconds=time.time()-started, world_model_checkpoint=job.get('checkpoint'))
    atomic_json(path.with_suffix('.json'), summary)
    atomic_json(path.with_suffix('.decisions.json'), decisions)
    atomic_json(path.with_suffix('.progress.json'), dict(stage='complete', seed=seed, split=split,
        step=frame, budget=job['steps'], reason=reason, goal_reached=goal_reached,
        remaining_distance_m=final_distance, ground_path_length_m=travelled,
        elapsed_seconds=times[-1]-times[0]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-config', type=Path)
    parser.add_argument('--train-scenes', type=int, default=32)
    parser.add_argument('--val-scenes', type=int, default=8)
    parser.add_argument('--train-seed', type=int)
    parser.add_argument('--val-seed', type=int)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=4000)
    parser.add_argument('--train-distance-min', type=float, default=100.)
    parser.add_argument('--train-distance-max', type=float, default=400.)
    parser.add_argument('--val-distance-min', type=float, default=100.)
    parser.add_argument('--val-distance-max', type=float, default=400.)
    parser.add_argument('--eval-distance-min', type=float, default=1000.)
    parser.add_argument('--eval-distance-max', type=float, default=1150.)
    parser.add_argument('--long-train-fraction', type=float, default=0., help='Fraction of training tasks with at least 1km endpoint distance')
    parser.add_argument('--steps-per-reference-metre', type=float, default=4.)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=24)
    parser.add_argument('--members', type=int, default=3)
    parser.add_argument('--terrain', choices=('circles', 'wildscenes_simplified'), default='circles')
    parser.add_argument('--wildscenes-manifest', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='Evaluate schema-4 world model; requires --epochs 0')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    frozen = args.run_config is not None
    if frozen:
        args = argparse.Namespace(**json.loads(args.run_config.read_text()))
        args.output = Path(args.output)
        args.checkpoint = Path(args.checkpoint) if args.checkpoint else None
    invalid_scenes = (min(args.train_scenes, args.val_scenes) < 0
        or args.train_scenes+args.val_scenes < 1
        or not args.checkpoint and min(args.train_scenes, args.val_scenes) < 1)
    if (invalid_scenes or min(args.workers, args.max_steps, args.batch_size) < 1
            or args.epochs < 0 or args.checkpoint and args.epochs):
        parser.error('Positive collection sizes required; checkpoint evaluation needs --epochs 0')
    if args.train_seed is None:
        args.train_seed = 94000 if args.checkpoint else 90000
    if args.val_seed is None:
        args.val_seed = 95000 if args.checkpoint else 92000
    if min(args.train_seed, args.val_seed) < 0:
        parser.error('Seeds must be nonnegative')
    if args.terrain == 'wildscenes_simplified' and not args.wildscenes_manifest:
        parser.error('--wildscenes-manifest is required for simplified WildScenes')
    for prefix in ('train', 'val', 'eval'):
        low, high = getattr(args, prefix+'_distance_min'), getattr(args, prefix+'_distance_max')
        if not np.isfinite([low, high]).all() or not 0 < low <= high:
            parser.error('Task distance ranges must be finite, positive and ordered')
    if (not np.isfinite([args.long_train_fraction, args.steps_per_reference_metre]).all()
            or not 0 <= args.long_train_fraction <= 1 or args.steps_per_reference_metre <= 0
            or args.long_train_fraction > 0 and args.terrain == 'wildscenes_simplified' and args.train_distance_max < 1000):
        parser.error('Invalid long-task fraction, training distance maximum or steps/metre')
    output = args.output.resolve()
    if output.parent != SOURCE.parents[3]/'artifacts':
        parser.error('Use a fresh directory directly under Teamed-Planning/artifacts')
    snapshot = output/'runtime_sources/world_model'
    if not frozen:
        if args.checkpoint:
            # Validate weights before allocating an output run or collecting.
            TerrainPlanner.from_checkpoint(args.checkpoint)
        output.mkdir(exist_ok=False)
        shutil.copytree(SOURCE.parent, output/'runtime_sources',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        if args.terrain == 'wildscenes_simplified':
            from .ab_wildscenes import freeze_manifest
            args.wildscenes_manifest = freeze_manifest(args.wildscenes_manifest, output/'terrain')
        args.output = output
        args.checkpoint = args.checkpoint.resolve() if args.checkpoint else None
        config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()
                  if name != 'run_config'}
        atomic_json(output/'config.json', config)
        os.execv(sys.executable, [sys.executable, '-I', '-B', str(snapshot/'decoupled_pipeline.py'),
            '--output', str(output), '--run-config', str(output/'config.json')])
    if SOURCE != snapshot:
        raise ValueError('Execute the frozen source belonging to this run')
    from .feasibility import generate_certified_scene
    provenance_record = provenance()
    provenance_record.update(model_architecture=ARCHITECTURE, schema_version=SCHEMA_VERSION,
        action_contract=ACTION_CONTRACT, selector_mode='world_model_independent' if args.checkpoint else 'bootstrap_independent',
        v9_weights_trained=False, vehicle_runtime=VEHICLE_RUNTIME,
        world_model_sources={str(rel): sha256(SOURCE / rel) for rel in package_python_files(SOURCE)},
        vehicle_sources={p.name: sha256(p) for p in (SOURCE.parent/'ground').glob('*.py')})
    atomic_json(output/'provenance.json', provenance_record)
    try:
        atomic_json(output/'status.json', dict(stage='certifying', pid=os.getpid()))
        specifications = [(split, seed) for split, start, count in
            (('train', args.train_seed, args.train_scenes), ('val', args.val_seed, args.val_scenes))
            for seed in range(start, start+count)]
        if len({seed for _, seed in specifications}) != len(specifications):
            raise ValueError('Train and validation seeds overlap')
        forbidden_seeds, forbidden_geometry, development = set(), set(), []
        if args.checkpoint:
            import torch
            checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
            for row in checkpoint.get('dataset', []):
                forbidden_seeds.add(row['seed'])
                forbidden_geometry.add(row['geometry_sha256'])
                development.append(row)
        provider = None
        if args.terrain == 'wildscenes_simplified':
            from .ab_environment import MappingConfig
            from .ab_wildscenes import WildScenesSource, load_manifest
            from .decoupled_scenes import generate_scene, geometry_hash, regions_overlap
            manifest = load_manifest(args.wildscenes_manifest)
            provider = WildScenesSource(args.wildscenes_manifest,
                MappingConfig(world=manifest['world_size_m'], resolution=2., clearance=1.5))
        jobs = []
        for split, seed in specifications:
            if seed in forbidden_seeds:
                raise ValueError('Evaluation seed overlaps the world-model development scenes')
            prefix = 'eval' if args.checkpoint else split
            low, high = getattr(args, prefix+'_distance_min'), getattr(args, prefix+'_distance_max')
            if (not args.checkpoint and split == 'train'
                    and seed-args.train_seed < math.ceil(args.train_scenes*args.long_train_fraction)):
                low = max(low, 1000.)
            certificate = (generate_scene(provider, seed, split, output, region_split='test' if args.checkpoint else split,
                                           distance_range=(low, high))
                if provider is not None else generate_certified_scene(seed=seed, split=split, output_dir=output, clearance=1.5))
            doc = json.loads(Path(certificate['scene_path']).read_text())
            if provider is not None:
                geometry_sha = geometry_hash(doc['scene'])
                region = dict(doc['scene'], terrain_world_m=doc['scene']['world'])
                if any(regions_overlap(region, item) for item in development):
                    raise ValueError('Evaluation terrain overlaps model development regions')
            else:
                geometry = {key: doc['scene'][key] for key in ('start', 'goal', 'world', 'circles')}
                geometry_sha = hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if geometry_sha in forbidden_geometry:
                raise ValueError('Evaluation geometry overlaps the world-model development scenes')
            budget = task_step_budget(certificate['certificate']['path_length_m'], args.max_steps,
                                      args.steps_per_reference_metre)
            jobs.append(dict(seed=seed, split=split, steps=budget, output=str(output/'data'),
                scene_path=certificate['scene_path'], scene_sha256=certificate['scene_sha256'],
                terrain=args.terrain,
                checkpoint=str(args.checkpoint) if args.checkpoint else None))
        atomic_json(output/'jobs.json', jobs)
        results = []
        atomic_json(output/'status.json', dict(stage='collecting', completed=0, total=len(jobs), pid=os.getpid()))
        with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for result in pool.map(collect_episode, jobs):
                results.append(result)
                atomic_json(output/'collection_manifest.json', results)
                atomic_json(output/'navigation_summary.json', navigation_summary(results))
                atomic_json(output/'status.json', dict(stage='collecting', completed=len(results), total=len(jobs)))
                print(json.dumps(result), flush=True)
        training = None
        if args.epochs:
            from .decoupled_model import train_model
            atomic_json(output/'status.json', dict(stage='training', pid=os.getpid()))
            training = train_model(output/'data', output/'training', epochs=args.epochs,
                batch_size=args.batch_size, hidden=args.hidden, members=args.members, device=args.device)
        atomic_json(output/'status.json', dict(stage='complete', training=training,
            closed_loop_evaluation=bool(args.checkpoint), v9_weights_trained=False))
    except Exception as exc:
        atomic_json(output/'status.json', dict(stage='failed', error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()
