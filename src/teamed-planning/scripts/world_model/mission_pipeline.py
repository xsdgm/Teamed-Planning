"""Collect/train/evaluate distinct ground-drive and aerial-survey decisions.

Schema 3: action=(target_x/1000, target_y/1000, purpose_sign), where +1
requests ten drive steps and -1 requests ten survey steps with ground held.
The bootstrap policy gathers data only; it is never called a learned policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import contextlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

SOURCE = Path(__file__).resolve().parent
if __package__ in (None, ''):
    sys.path.insert(0, str(SOURCE.parent))
    __package__ = 'world_model'

from .hierarchical_sources import copy_source_files, package_python_files
from .data import (ALGORITHM, CHECKPOINT_SHA, HORIZON, MISSION_ACTION_CONTRACT,
                  MISSION_SCHEMA_VERSION, atomic_json, build_environment, encode,
                  provenance, sha256)


class BootstrapMissionPolicy:
    """Explicit exploration-data policy; alternates both action semantics."""
    selector_mode = 'bootstrap_drive_survey'
    action_contract = MISSION_ACTION_CONTRACT

    def __init__(self, seed=0, period=40.):
        self.seed, self.period = int(seed), float(period)
        self.reset()

    def reset(self):
        self.now = 0.
        self.state = None
        self.last_diagnostics = {}

    def observe(self, maps, states, now, executed_target=None, executed_purpose='drive'):
        self.state = np.asarray(states).copy()
        self.now = float(now)

    def rank(self, candidates, goal):
        phase = int(self.now // self.period)
        preferred = 'drive' if phase % 2 == 0 else 'survey'
        position = self.state[:2]*1000.
        heading = math.atan2(*(np.asarray(goal)-position)[::-1])
        # Include lateral surveys in the data; these are not labels claiming
        # that the sampled point is optimal or that its terrain is free.
        heading += ((self.seed+phase//2) % 5-2)*math.pi/6
        scores = []
        for candidate in candidates:
            distance = float(np.linalg.norm(candidate.point-position))
            delta = candidate.point-position
            angle = math.atan2(delta[1], delta[0])
            angle_cost = 30.*(1.-math.cos(angle-heading))
            cost = (0. if candidate.purpose == preferred else 1000.)
            if candidate.purpose == 'survey':
                cost += angle_cost+abs(distance-100.)*.2
            else:
                cost += float(np.linalg.norm(candidate.point-goal))+.15*candidate.cost
            scores.append(float(cost))
        self.last_diagnostics = dict(selector_mode=self.selector_mode,
            preferred_purpose=preferred, cost=scores, trained=False)
        return scores


def _graph_size(controller):
    graph = getattr(controller, 'unified_graph', None)
    graph = getattr(graph, 'graph', None)
    return 0 if graph is None else len(graph)+graph.number_of_edges()


def collect_mission_episode(job):
    """One certified scene, using only legal observations for all decisions."""
    seed, split = int(job['seed']), job['split']
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=True)
    path = output/f'{split}_{seed}.npz'
    if path.exists():
        raise FileExistsError(path)
    provenance()
    if job.get('checkpoint'):
        from .planner import WorldModelPlanner
        planner = WorldModelPlanner.from_checkpoint(job['checkpoint'], device='cpu',
            horizon_steps=job.get('horizon_steps', 8), particles=job.get('particles', 4))
        if planner.action_contract != MISSION_ACTION_CONTRACT:
            raise ValueError('Joint mission evaluation requires a drive/survey checkpoint')
    else:
        planner = BootstrapMissionPolicy(seed)
    started = time.time()
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
        env, safety_exception = build_environment(seed, scene_path=job['scene_path'],
            scene_sha256=job['scene_sha256'], mission_planner=planner,
            manager_options=dict(review_interval=10., min_hold=10., candidate_limit=24))
        maps, states, actions, labels, elapsed, terminal = [], [], [], [], [], []
        targets, purposes, switches, reasons, statuses = [], [], [], [], []
        times = [float(env.env.time)]
        m, s = encode(env.ground_controller, env.goal, env.env.time, (0., 0.))
        maps.append(m); states.append(s)
        frame, reason, velocity = 0, 'budget', np.zeros(2)
        survey_steps = drive_steps = 0
        max_survey_displacement = 0.
        decisions = []
        while frame < job['steps']:
            target = env.subgoals.target.copy()
            purpose = env.subgoals.purpose
            status = env.subgoals.target_status
            begin = float(env.env.time)
            p0 = np.asarray(env.ground_controller.position).copy()
            graph_before = _graph_size(env.ground_controller)
            reached = rejected = switched = False
            for _ in range(min(HORIZON, job['steps']-frame)):
                if (env.subgoals.purpose != purpose or
                        not np.allclose(env.subgoals.target, target, rtol=0, atol=1e-6)):
                    raise AssertionError('Mission changed inside one labelled action')
                before = np.asarray(env.ground_controller.position).copy()
                try:
                    env.step(frame)
                except safety_exception:
                    rejected, reason = True, 'cbf_infeasible'
                    break
                frame += 1
                velocity = np.asarray(env.ground_controller.position)-before
                if purpose == 'survey':
                    survey_steps += 1
                    max_survey_displacement = max(max_survey_displacement, float(np.linalg.norm(velocity)))
                    reached |= env.subgoals.survey_observed(target, env.ground_controller.unified_graph)
                else:
                    drive_steps += 1
                    threshold = 2. if np.allclose(target, env.goal, rtol=0, atol=1e-6) else 1.
                    reached |= np.linalg.norm(np.asarray(env.ground_controller.position)-target) <= threshold
                switched = (purpose != env.subgoals.purpose or
                            not np.allclose(target, env.subgoals.target, rtol=0, atol=1e-6))
                if env.goal_reached:
                    reason = 'arrival'
                    break
                if switched:
                    break
            dt = float(env.env.time)-begin
            m, s = encode(env.ground_controller, env.goal, env.env.time, velocity)
            maps.append(m); states.append(s)
            actions.append([*(target/1000.), 1. if purpose == 'drive' else -1.])
            targets.append(target); purposes.append(purpose); statuses.append(status)
            switches.append(switched)
            reasons.append(env.subgoals.last_reason if switched else '')
            displacement = float(np.linalg.norm(np.asarray(env.ground_controller.position)-p0))
            stalled = ((_graph_size(env.ground_controller) <= graph_before and not reached)
                       if purpose == 'survey' else displacement < .5)
            labels.append([float(reached), float(stalled and dt >= HORIZON), float(rejected)])
            elapsed.append(dt/HORIZON)
            terminal.append(bool(rejected or env.goal_reached))
            times.append(float(env.env.time))
            decisions.append(dict(time=times[-1], executed_purpose=purpose,
                executed_target=target.tolist(), ground_position=np.asarray(env.ground_controller.position).tolist(),
                next_purpose=env.subgoals.purpose, next_target=env.subgoals.target.tolist(),
                decision=getattr(env.subgoals, 'last_decision', {}),
                predictions=planner.last_diagnostics))
            atomic_json(path.with_suffix('.progress.json'), dict(seed=seed, step=frame,
                budget=job['steps'], purpose=env.subgoals.purpose, stage='collecting'))
            if terminal[-1]:
                break
    arrays = dict(maps=np.asarray(maps, np.float32), states=np.asarray(states, np.float32),
        actions=np.asarray(actions, np.float32), labels=np.asarray(labels, np.float32),
        elapsed=np.asarray(elapsed, np.float32), terminated=np.asarray(terminal, bool),
        truncated=np.asarray([False]*(len(actions)-1)+[not terminal[-1]], bool),
        targets=np.asarray(targets, np.float32), times=np.asarray(times),
        target_switched=np.asarray(switches, bool), switch_reason=np.asarray(reasons),
        target_status=np.asarray(statuses), mission_purpose=np.asarray(purposes),
        action_contract=np.asarray(MISSION_ACTION_CONTRACT), algorithm=np.asarray(ALGORITHM),
        schema_version=np.asarray(MISSION_SCHEMA_VERSION), seed=np.asarray(seed), split=np.asarray(split),
        checkpoint_sha256=np.asarray(CHECKPOINT_SHA), scene_sha256=np.asarray(job['scene_sha256']),
        scene_geometry_sha256=np.asarray(env.scene_geometry_sha256),
        feasibility_validated=np.asarray(True), physical_clearance_m=np.asarray(1.5))
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix('.tmp').replace(path)
    summary = dict(algorithm=ALGORITHM, schema_version=MISSION_SCHEMA_VERSION,
        action_contract=MISSION_ACTION_CONTRACT, seed=seed, split=split,
        selector_mode=env.subgoals.selector_mode, steps=frame, budget=job['steps'],
        transitions=len(actions), purpose_transitions=dict(Counter(purposes)),
        drive_steps=drive_steps, survey_steps=survey_steps,
        max_survey_ground_displacement_m=max_survey_displacement,
        reason=reason, initial_distance_m=env.initial_goal_distance,
        final_distance_m=float(np.linalg.norm(np.asarray(env.ground_controller.position)-env.goal)),
        file=str(path), sha256=sha256(path), wall_seconds=time.time()-started,
        checkpoint=job.get('checkpoint'), checkpoint_sha256=(sha256(job['checkpoint']) if job.get('checkpoint') else None))
    atomic_json(path.with_suffix('.json'), summary)
    atomic_json(path.with_suffix('.decisions.json'), decisions)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--train-scenes', type=int, default=4)
    parser.add_argument('--val-scenes', type=int, default=2)
    parser.add_argument('--train-seed', type=int)
    parser.add_argument('--val-seed', type=int)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=4000)
    parser.add_argument('--checkpoint', type=Path, help='Evaluate learned missions; omit only for bootstrap data')
    parser.add_argument('--epochs', type=int, default=0, help='Train a fresh joint predictor after collection')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--horizon-steps', type=int, default=8)
    parser.add_argument('--particles', type=int, default=4)
    args = parser.parse_args()
    frozen = args.run_config is not None
    if frozen:
        args = argparse.Namespace(**json.loads(args.run_config.read_text()))
        args.output = Path(args.output)
        args.checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if args.output is None:
        parser.error('--output is required')
    if args.checkpoint and args.epochs:
        parser.error('Keep learned-policy evaluation separate from model training')
    if args.train_seed is None:
        args.train_seed = 88000 if args.checkpoint else 86000
    if args.val_seed is None:
        args.val_seed = 89000 if args.checkpoint else 87000
    if min(args.train_scenes, args.val_scenes, args.workers, args.max_steps) < 1 or args.epochs < 0:
        parser.error('Positive scene/worker/budget counts and nonnegative epochs required')
    output = args.output.resolve()
    if output.parent != SOURCE.parents[3]/'artifacts':
        parser.error('Output must be a fresh directory directly under Teamed-Planning/artifacts')
    snapshot = output/'runtime_sources/world_model'
    if not frozen:
        output.mkdir(exist_ok=False)
        snapshot.mkdir(parents=True)
        copy_source_files(SOURCE, snapshot, package_python_files(SOURCE))
        args.output = output
        args.checkpoint = args.checkpoint.resolve() if args.checkpoint else None
        config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items() if key != 'run_config'}
        atomic_json(output/'config.json', config)
        # All workers and training use this immutable per-run source copy.
        os.execv(sys.executable, [sys.executable, '-I', '-B', str(snapshot/'mission_pipeline.py'),
                                 '--run-config', str(output/'config.json')])
    if SOURCE != snapshot:
        raise ValueError('Run config must execute from its own source snapshot')
    evidence = provenance()
    evidence['world_model_sources'] = {str(rel): sha256(SOURCE / rel) for rel in package_python_files(SOURCE)}
    atomic_json(output/'provenance.json', evidence)
    atomic_json(output/'status.json', dict(stage='certifying', pid=os.getpid()))
    from .feasibility import generate_certified_scene
    jobs = []
    try:
        for split, start, count in [('train', args.train_seed, args.train_scenes), ('val', args.val_seed, args.val_scenes)]:
            for seed in range(start, start+count):
                record = generate_certified_scene(seed=seed, split=split, output_dir=output, clearance=1.5)
                scene = json.loads(Path(record['scene_path']).read_text())['scene']
                budget = min(args.max_steps, max(600, math.ceil(3*scene['reference_length']+240)))
                jobs.append(dict(seed=seed, split=split, steps=budget, output=str(output/'data'),
                    scene_path=record['scene_path'], scene_sha256=record['scene_sha256'],
                    checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
                    horizon_steps=args.horizon_steps, particles=args.particles))
        if len({job['seed'] for job in jobs}) != len(jobs):
            raise ValueError('Train and validation scene seeds overlap')
        atomic_json(output/'jobs.json', jobs)
        results = []
        with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for result in pool.map(collect_mission_episode, jobs):
                results.append(result)
                atomic_json(output/'collection_manifest.json', results)
                atomic_json(output/'status.json', dict(stage='collecting', completed=len(results), total=len(jobs), pid=os.getpid()))
                print(json.dumps(result), flush=True)
        training = None
        if args.epochs:
            if args.checkpoint:
                raise ValueError('Keep evaluation data separate from training data')
            atomic_json(output/'status.json', dict(stage='training', pid=os.getpid()))
            from .train import run_training
            training = run_training(output/'data', output/'training', device=args.device,
                epochs=args.epochs, sequence_length=16, rollout_steps=4, rollout_weight=.25)
        atomic_json(output/'status.json', dict(stage='complete', pid=os.getpid(),
            selector_mode='world_model_mission' if args.checkpoint else 'bootstrap_drive_survey',
            training=training, closed_loop_evaluation=bool(args.checkpoint)))
    except Exception as exc:
        atomic_json(output/'status.json', dict(stage='failed', pid=os.getpid(), error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()
