"""Evaluate a fixed human UGV target with frozen graph world/UAV fast control.

No slow policy or optimizer is loaded. The UGV keeps its original rule planner
and all original physical/CBF checks. The target is held for the full original
mission budget; success retains the simulator's strict two-metre tolerance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
from dataclasses import replace
import importlib
import importlib.util
import inspect
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

_helper_path = Path(__file__).with_name('hierarchical_sources.py')
_helper_spec = importlib.util.spec_from_file_location('_hierarchical_sources_compat', _helper_path)
_sources = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_sources)
digest = _sources.file_sha256
write_json = _sources.atomic_json


def frozen_modules(run):
    return _sources.load_runtime_modules(_sources.runtime_source(run),
        ('hierarchy', 'graph_data', 'graph_policy', 'graph_world', 'fast', 'validate', 'scenes'))


def evaluate_case(job):
    repo, run, checkpoint, output, case, deterministic = job
    import numpy as np
    import torch
    torch.set_num_threads(1)
    modules = frozen_modules(run)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = saved['config']
    graph_world = modules['graph_world']
    road_arguments = {'road_topology': True} if config.get('road_topology') else {}
    schema = getattr(graph_world, 'ROAD_MODEL_SCHEMA', None) if road_arguments else graph_world.MODEL_SCHEMA
    if schema is None or saved['world_schema'] != schema:
        raise ValueError('Checkpoint world schema differs from selected runtime')
    protocol = config.get('training_observation_protocol')
    observation_arguments = {'training_observation_protocol': protocol} if protocol else {}
    if config.get('fixed_mission_target') and 'v9_fast_reward' in inspect.signature(
            modules['hierarchy'].HierarchicalEpisode).parameters:
        observation_arguments['v9_fast_reward'] = True
    fresh_arguments = {'random_init': True} if (config.get('fresh_all', False)
                                               or config.get('fresh_fast_world', False)) else {}
    label = f"{case['kind']}-{case['direct_distance_m']:g}m-seed{case['seed']}"
    output = Path(output)
    started = time.monotonic()
    with (output / 'logs' / (label + '.log')).open('w') as log, contextlib.redirect_stdout(log):
        hierarchy_config = modules['hierarchy'].HierarchyConfig()
        # Leave room for the real mission-budget terminal, never a 257-step
        # collection cut. The same budget formula is used by the episode.
        budget = max(hierarchy_config.min_mission_budget,
                     math.ceil(hierarchy_config.budget_multiplier * case.get('reference_distance_m', case['direct_distance_m'])
                               + hierarchy_config.budget_extra))
        # One human instruction, held until arrival/failure/deadline. The UGV
        # local planner still checks obstacles on every physical control step.
        hierarchy_options = dict(option_steps=budget, evidence_interval=budget + 1)
        if hasattr(hierarchy_config, 'max_option_steps'):
            hierarchy_options.update(max_option_steps=max(budget, hierarchy_config.max_option_steps), adaptive_options=False)
        if config.get('ground_start_hold_seconds'):
            hierarchy_options['ground_start_hold_seconds'] = config['ground_start_hold_seconds']
        hierarchy_config = replace(hierarchy_config, **hierarchy_options)
        episode = modules['hierarchy'].HierarchicalEpisode(repo, case, hierarchy_config,
            fast_seed=case['seed'] + 43, rollout_limit=budget + 1, **fresh_arguments, **observation_arguments)
        try:
            fast = modules['graph_policy'].make_graph_fast(repo, **fresh_arguments)
            fast.load_state_dict(saved['fast']); fast.eval().requires_grad_(False)
            world = graph_world.GraphWorldModel(config['hidden'], config['horizon'], **road_arguments)
            world.load_state_dict(saved['world']); world.eval().requires_grad_(False)
            initial_hashes = {name: modules['fast'].state_hash(modules['fast'].numpy_state(model))
                              for name, model in (('fast', fast), ('world', world))}
            episode.fast_model = fast
            for controller in episode.env.controllers:
                controller.navigator.model = fast
                controller.navigator.deterministic = deterministic
            teacher = None if fresh_arguments else modules['validate'].runtime(repo)[2]
            status_path = output / 'progress' / (label + '.json')

            class EvaluationRecorder(modules['graph_data'].GraphRecorder):
                """Same inference hook, without training labels or critic calls."""
                def __init__(self, *args, **kwargs):
                    self.safety_rows = []
                    super().__init__(*args, **kwargs)

                def record_truth(self, frame):
                    pass

                def save_input(self, graph):
                    pass

                def step(self, frame):
                    self.prepare(frame)
                    if not np.array_equal(episode.manager.target, episode.goal):
                        raise AssertionError('The human target changed during execution')
                    try:
                        result = self.base_step(frame)
                    finally:
                        for controller in self.env.controllers:
                            stats = controller.safety_stats
                            margin = stats.get('feasibility_margin')
                            self.safety_rows.append(dict(frame=int(frame), owner=controller.drone_id,
                                intervention_m_s=float(stats.get('intervention', 0.)),
                                infeasible=bool(stats.get('infeasible', False)),
                                margin_m_s=None if margin is None or not math.isfinite(margin) else float(margin),
                                margin_kind='unconstrained' if margin is None else
                                            'finite' if math.isfinite(margin) else 'negative_infinity',
                                missing_links=list(stats.get('missing_links', ()))))
                    if not np.array_equal(episode.manager.target, episode.goal):
                        raise AssertionError('The human target changed after execution')
                    if frame % 20 == 0 or self.env.goal_reached:
                        remaining = float(np.linalg.norm(np.asarray(self.env.obs[self.env.ground]['self']['position'][:2])
                                                         - episode.goal))
                        write_json(status_path, dict(state='running', label=label, steps=frame + 1,
                            mission_budget=budget, remaining_m=remaining,
                            elapsed_seconds=round(time.monotonic() - started, 2)))
                    return result

            recorder = EvaluationRecorder(episode, world, fast, teacher, config)
            # A human can specify the goal without waiting for a candidate
            # generator. This does not bypass the UGV planner or physical CBF.
            observation = dict(targets=np.asarray([episode.goal.copy()]), mask=np.ones(1, bool),
                               kinds=['final'], statuses=['human_fixed_target'])
            while not episode.finished:
                episode.execute(observation, 0)
            summary = episode.summary()
            if any(not np.array_equal(np.asarray(row['target']), episode.goal) for row in summary['decisions']):
                raise AssertionError('An option selected a different target')
            final_hashes = {name: modules['fast'].state_hash(modules['fast'].numpy_state(model))
                            for name, model in (('fast', fast), ('world', world))}
            if final_hashes != initial_hashes:
                raise AssertionError('Frozen evaluation changed model parameters')
            if summary['goal_reached'] and summary['remaining_m'] > 2. + 1e-6:
                raise AssertionError('Arrival did not satisfy the strict two-metre goal')
            distances = [float(np.linalg.norm(np.asarray(point) - episode.goal)) for point in summary['trajectory']]
            intervention = np.asarray([row['intervention_m_s'] for row in recorder.safety_rows])
            finite_margins = [row['margin_m_s'] for row in recorder.safety_rows if row['margin_m_s'] is not None]
            result = dict(label=label, seed=case['seed'], kind=case['kind'],
                observation_protocol=getattr(episode, 'observation_protocol', 'native'),
                distance_m=case['direct_distance_m'], goal_reached=summary['goal_reached'],
                termination=summary['termination'], steps=summary['steps'], control_attempts=summary['control_attempts'],
                mission_budget=episode.mission_budget, remaining_m=summary['remaining_m'],
                min_remaining_m=min(distances), ground_path_length_m=float(np.linalg.norm(np.diff(summary['trajectory'], axis=0), axis=1).sum()),
                collisions=summary['collisions'], strict_arrival_tolerance_m=2., target_unchanged=True,
                deterministic=deterministic, world_calls=recorder.world_calls,
                cbf_decisions=len(recorder.safety_rows),
                cbf_infeasible_decisions=sum(row['infeasible'] for row in recorder.safety_rows),
                cbf_missing_link_decisions=sum(bool(row['missing_links']) for row in recorder.safety_rows),
                cbf_mean_intervention_m_s=float(intervention.mean()) if len(intervention) else None,
                cbf_intervention_fraction=float((intervention > 1e-6).mean()) if len(intervention) else None,
                cbf_min_finite_margin_m_s=min(finite_margins) if finite_margins else None,
                cbf_negative_infinite_margins=sum(row['margin_kind'] == 'negative_infinity' for row in recorder.safety_rows),
                frozen_hashes=final_hashes, elapsed_seconds=round(time.monotonic() - started, 2))
            write_json(output / 'episodes' / (label + '.json'),
                       dict(case=case, result=result, trajectory=summary['trajectory'],
                            options=summary['decisions'], cbf=recorder.safety_rows, transport=recorder.network.stats))
            write_json(status_path, dict(state='complete', **result))
            return result
        finally:
            episode.close()


def summarize(results):
    def group(rows):
        return dict(episodes=len(rows), successes=sum(row['goal_reached'] for row in rows),
                    success_rate=sum(row['goal_reached'] for row in rows) / len(rows),
                    mean_remaining_m=sum(row['remaining_m'] for row in rows) / len(rows),
                    safety_terminations=sum(row['termination'] == 'safety_failure' for row in rows),
                    mission_timeouts=sum(row['termination'] == 'mission_budget' for row in rows),
                    collisions=sum(row['collisions'] for row in rows))
    return dict(all=group(results),
        by_distance={str(distance): group([row for row in results if row['distance_m'] == distance])
                     for distance in sorted({row['distance_m'] for row in results})},
        by_kind={kind: group([row for row in results if row['kind'] == kind])
                 for kind in sorted({row['kind'] for row in results})})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', default='final.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, help='JSON list of independent evaluation cases, including V9 circle or WildScenes raster scenes')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--distances', nargs='+', type=float, default=[100., 150., 200.])
    parser.add_argument('--kinds', nargs='+', choices=['open', 'trees', 'wall_gap'], default=['open', 'trees', 'wall_gap'])
    parser.add_argument('--seed', type=int, default=26121600)
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--frozen', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1 or any(not math.isfinite(value) or not 0 < value <= 200. for value in args.distances):
        parser.error('Positive workers and target distances in (0, 200] are required')
    if len(set(args.distances)) != len(args.distances) or len(set(args.kinds)) != len(args.kinds):
        parser.error('Duplicate distances/kinds are not separate trials')
    repo, run, output = args.repo.resolve(), args.run.resolve(), args.output.resolve()
    checkpoint = Path(args.checkpoint)
    checkpoint = checkpoint.resolve() if checkpoint.is_absolute() else run / 'checkpoints' / checkpoint
    if not args.frozen:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Evaluation output is not empty')
        output.mkdir(parents=True, exist_ok=True)
        target = output / 'evaluate_fixed_target.py'
        shutil.copy2(__file__, target)
        shutil.copy2(_helper_path, output / 'hierarchical_sources.py')
        os.execv(sys.executable, [sys.executable, '-I', '-B', '-u', str(target), *sys.argv[1:],
            '--repo', str(repo), '--run', str(run), '--output', str(output),
            '--checkpoint', str(checkpoint), '--frozen'])
    for name in ('episodes', 'logs', 'progress'):
        (output / name).mkdir(exist_ok=True)
    modules = frozen_modules(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    source = _sources.runtime_source(run)
    for name, expected in manifest['source_sha256'].items():
        if digest(source / name) != expected:
            raise ValueError('Archived training source changed: ' + name)
    if args.cases is None:
        cases = [modules['scenes'].make_scene(args.seed + i, (distance, distance), 2000., kind)
                 for i, (distance, kind) in enumerate((distance, kind) for distance in args.distances for kind in args.kinds)]
    else:
        cases = json.loads(args.cases.read_text(encoding='utf-8'))
        if not isinstance(cases, list) or not cases or len({case['seed'] for case in cases}) != len(cases):
            raise ValueError('Evaluation cases must form a nonempty list with unique seeds')
    old_scenes = json.loads((run / 'scenes.json').read_text())
    used = {case['seed'] for group in old_scenes.values() for case in group}
    if used & {case['seed'] for case in cases}:
        raise ValueError('Evaluation seeds overlap training/validation scenes')
    write_json(output / 'manifest.json', dict(protocol='vtemp2_fixed_human_target_evaluation_v1',
        run=str(run), checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        evaluator_sha256=digest(__file__), source_sha256=manifest['source_sha256'],
        source_package=_sources.source_package(source), source_helper_sha256=digest(_helper_path),
        deterministic=not args.stochastic, workers=args.workers,
        case_source=str(args.cases.resolve()) if args.cases else 'generated_synthetic',
        mission_budget='Archived episode budget using the reference distance when supplied, otherwise the direct distance.',
        success='Original simulator UGV distance <= 2m, before full mission deadline.',
        control='Human final target held constant; graph world + UAV fast actor; UGV rule planner; no slow model and no learning.',
        evidence='Only local and delivered observations enter policies; offline truth verifies scene feasibility only.',
        limitations='Small synthetic diagnostic, one seed per distance/kind; neither a universal guarantee nor a safety certification.'))
    write_json(output / 'cases.json', cases)
    started = time.monotonic(); results = []
    def status(state, **extra):
        value = dict(state=state, pid=os.getpid(), completed=len(results), total=len(cases),
                     elapsed_seconds=round(time.monotonic() - started, 2), **extra)
        write_json(output / 'status.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
    status('running')
    try:
        jobs = [(str(repo), str(run), str(checkpoint), str(output), case, not args.stochastic) for case in cases]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(evaluate_case, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result(); results.append(result)
                write_json(output / 'results.json', sorted(results, key=lambda row: row['label']))
                status('running', latest=result)
        summary = summarize(results)
        write_json(output / 'summary.json', summary)
        status('complete', summary=summary)
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
