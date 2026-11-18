"""Evaluate current control repairs on checkpoint validation tasks without learning.

The evaluator freezes its own source and an immutable copy of the supplied
checkpoint. --max-steps is a diagnostic cutoff, never a full-mission result.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
from dataclasses import replace
import importlib
import importlib.util
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

SOURCE = Path(__file__).resolve().parent
PROTOCOL = 'vtemp2_repair_validation_frozen_weights_v1'

# Load without importing a mutable world_model package before selecting the
# frozen runtime. This same helper travels with standalone/legacy evaluators.
_helper_path = SOURCE / 'hierarchical_sources.py'
_helper_spec = importlib.util.spec_from_file_location('_hierarchical_sources_compat', _helper_path)
_sources = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_sources)
digest = _sources.file_sha256


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if hasattr(value, 'tolist'):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def freeze_inputs(output, checkpoint, source=SOURCE, evaluator=None):
    """Pin weights before spawning; a live latest.pt must not vary by worker."""
    output, checkpoint, source = map(lambda p: Path(p).resolve(), (output, checkpoint, source))
    if output.exists() and any(output.iterdir()):
        raise ValueError('Evaluation output is not empty; preserve earlier evidence')
    original_digest = digest(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    package = _sources.source_package(source)
    target = output / 'runtime_sources' / package
    _sources.snapshot_sources(source, target, include_backend=True)
    if evaluator is not None:
        shutil.copy2(evaluator, target / 'evaluate_safety_repairs.py')
    # Old snapshots never contained this compatibility helper. Copy it into
    # the new evaluation output, leaving the historical source untouched.
    if not (target / 'hierarchical_sources.py').is_file():
        shutil.copy2(_helper_path, target / 'hierarchical_sources.py')
    copied = output / 'input.pt'
    shutil.copy2(checkpoint, copied)
    if digest(copied) != original_digest or digest(checkpoint) != original_digest:
        raise ValueError('Parent checkpoint changed while copying; use a stable checkpoint and new output')
    snapshot = dict(parent_checkpoint=str(checkpoint), checkpoint_sha256=original_digest, source_origin=str(source),
        checkpoint=str(copied), source_package=package, source_sha256=_sources.source_hashes(target))
    write_json(output / 'input-manifest.json', snapshot)
    return target / 'evaluate_safety_repairs.py'


def load_modules(source=SOURCE):
    return _sources.load_runtime_modules(source)


def runtime_contract(modules):
    if 'control_safety' in modules:
        return modules['control_safety'].control_contract()
    return dict(version='archived_runtime_without_control_safety_adapter', communication_reserve_m=0.)


def select_cases(config, indices=None):
    cases = config.get('scene_manifest', {}).get('validation')
    if not isinstance(cases, list) or not cases:
        raise ValueError('Checkpoint must embed its fixed validation tasks')
    selected = list(range(len(cases))) if indices is None else list(indices)
    if (not selected or len(set(selected)) != len(selected)
            or any(type(i) is not int or not 0 <= i < len(cases) for i in selected)):
        raise ValueError('Case indices must be unique valid zero-based validation indices')
    return [(index, cases[index]) for index in selected]


def checkpoint_hashes(saved, state_hash):
    return {name: state_hash({key: tensor.detach().cpu().numpy()
                             for key, tensor in saved[name].items()})
            for name in ('fast', 'world', 'slow')}


def validate_checkpoint(saved, modules):
    if (saved.get('world_schema') != modules['graph_world'].MODEL_SCHEMA
            or saved.get('manifest', {}).get('fast_schema') != modules['graph_policy'].FAST_SCHEMA
            or saved.get('manifest', {}).get('slow_schema') != modules['graph_policy'].SLOW_SCHEMA):
        raise ValueError('Checkpoint graph model schemas differ from evaluation source')


class DiscardRecords:
    """The inherited actor hook appends CBF context; evaluation need not retain it."""
    def append(self, value):
        pass

    def __len__(self):
        return 0


def lean_recorder_type(base):
    class LeanRecorder(base):
        def __init__(self, *args, on_step=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.cbf_records = DiscardRecords()
            self.on_step = on_step

        def record_truth(self, frame):
            pass

        def save_input(self, graph):
            pass

        def append_transition(self, *args, **kwargs):
            raise AssertionError('Evaluation must never collect training transitions')

        def step(self, frame):
            self.prepare(frame)
            # Keep the same causal ground history refresh as training.
            self.predict(self.graph('ground_0', self.ground_observation()), 'ugv')
            try:
                return self.base_step(frame)
            finally:
                if self.on_step is not None:
                    self.on_step(frame)
    return LeanRecorder


def safety_row(frame, controller):
    stats = controller.safety_stats
    margin = stats.get('feasibility_margin')
    return dict(frame=int(frame), owner=controller.drone_id,
        intervention_m_s=float(stats.get('intervention', 0.)),
        infeasible=bool(stats.get('infeasible', False)),
        missing_links=list(stats.get('missing_links', ())),
        margin_m_s=float(margin) if margin is not None and math.isfinite(margin) else None,
        margin_kind='unconstrained' if margin is None else 'finite' if math.isfinite(margin) else str(margin),
        solver_status=stats.get('solver_status'), hard_solver_status=stats.get('hard_solver_status'),
        communication_guard_enabled=bool(stats.get('communication_guard_enabled', False)),
        communication_guard_feasible=stats.get('communication_guard_feasible'),
        communication_guard_applied=bool(stats.get('communication_guard_applied', False)),
        max_constraint_residual=stats.get('max_constraint_residual'))


def evaluate_case(job):
    repo, output, index, case, max_steps, expected_hashes, expected_digest = job
    import numpy as np
    import torch
    torch.set_num_threads(1)
    output = Path(output)
    modules = load_modules(_sources.runtime_source(output))
    checkpoint = output / 'input.pt'
    if digest(checkpoint) != expected_digest:
        raise ValueError('Pinned evaluation checkpoint changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    validate_checkpoint(saved, modules)
    config = saved['config']
    fresh_arguments = {'random_init': True} if (config.get('fresh_all', False)
                                               or config.get('fresh_fast_world', False)) else {}
    label = f'validation-{index:04d}-seed{case["seed"]}'
    progress_path = output / 'progress' / (label + '.json')
    started = time.monotonic()
    episode = None
    decisions, safety = [], []
    with (output / 'logs' / (label + '.log')).open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            write_json(progress_path, dict(state='initializing', label=label, pid=os.getpid()))
            hierarchy = replace(modules['hierarchy'].HierarchyConfig(), option_steps=config['option_steps'])
            episode = modules['hierarchy'].HierarchicalEpisode(repo, case, hierarchy,
                fast_seed=case['seed'] + 43, rollout_limit=max_steps if max_steps is not None else 1,
                **fresh_arguments)
            if max_steps is None:
                episode.rollout_limit = episode.mission_budget + 1
            models = dict(fast=modules['graph_policy'].make_graph_fast(repo, **fresh_arguments),
                world=modules['graph_world'].GraphWorldModel(config['hidden'], config['horizon']),
                slow=modules['graph_policy'].GraphSlowActorCritic(hidden=config['hidden']))
            for name, model in models.items():
                model.load_state_dict(saved[name], strict=True)
                model.eval().requires_grad_(False)
            hashes = lambda: {name: modules['fast'].state_hash(modules['fast'].numpy_state(model))
                              for name, model in models.items()}
            if hashes() != expected_hashes:
                raise AssertionError('Worker did not load the pinned model weights')
            del saved
            episode.fast_model = models['fast']
            for controller in episode.env.controllers:
                controller.navigator.model = models['fast']
                controller.navigator.deterministic = True

            def progress(frame):
                safety.extend(safety_row(frame, c) for c in episode.env.controllers)
                if frame % 20 == 0 or episode.env.goal_reached:
                    p = np.asarray(episode.env.obs[episode.env.ground]['self']['position'][:2])
                    write_json(progress_path, dict(state='running', label=label, pid=os.getpid(),
                        steps=episode.frame, control_frame=frame, mission_budget=episode.mission_budget,
                        max_steps=max_steps, remaining_m=float(np.linalg.norm(p - episode.goal)),
                        slow_decisions=len(decisions), elapsed_seconds=round(time.monotonic() - started, 2)))

            teacher = None if fresh_arguments else modules['validate'].runtime(repo)[2]
            recorder = lean_recorder_type(modules['graph_data'].GraphRecorder)(episode, models['world'],
                models['fast'], teacher, config, on_step=progress)
            with torch.inference_mode():
                while not episode.finished:
                    observation = recorder.slow_observation()
                    action, logp, value = models['slow'].act(observation, deterministic=True)
                    origin = np.asarray(episode.env.obs[episode.env.ground]['self']['position'][:2]).copy()
                    _, _, terminal, truncated, record = episode.execute(observation, action)
                    decisions.append(dict(record, origin=origin.tolist(),
                        valid_candidates=int(observation['mask'].sum()), selected_probability=math.exp(logp),
                        value=float(value), terminated=terminal, truncated=truncated))
            summary = episode.summary()
            failure_details = summary.get('safety_failure_details', [])
            if summary['termination'] == 'safety_failure' and not failure_details:
                failure_details = [dict(owner=c.drone_id, stats=json_value(c.safety_stats),
                    received_context=json_value(recorder.cbf.get(c.drone_id))) for c in episode.env.controllers]
            if hashes() != expected_hashes:
                raise AssertionError('Evaluation changed frozen model weights')
            if any((recorder.inputs, recorder.transitions, recorder.truth_states, recorder.truth_actions)):
                raise AssertionError('Lean evaluation accumulated training data')
            if summary['goal_reached'] and summary['remaining_m'] > 2. + 1e-6:
                raise AssertionError('Arrival exceeded the original two-metre tolerance')
            result = dict(case_index=index, label=label, seed=case['seed'], kind=case['kind'],
                map_group=case.get('map_group'), distance_m=case['direct_distance_m'],
                reference_distance_m=case.get('reference_distance_m', case['direct_distance_m']),
                goal_reached=summary['goal_reached'], termination=summary['termination'],
                steps=summary['steps'], control_attempts=summary['control_attempts'],
                remaining_m=summary['remaining_m'], mission_budget=episode.mission_budget,
                terminated=summary['terminated'], truncated=summary['truncated'], diagnostic=max_steps is not None,
                max_steps=max_steps, collisions=summary['collisions'],
                ground_safety_violations=summary.get('ground_safety_violations', 0),
                raster_collision_attempts=summary.get('raster_collision_attempts', 0),
                ground_motion_filter_interventions=summary.get('ground_motion_filter_interventions', 0),
                slow_decisions=len(decisions), cbf_infeasible_decisions=sum(r['infeasible'] for r in safety),
                communication_guard_applied_decisions=sum(r['communication_guard_applied'] for r in safety),
                frozen_hashes=expected_hashes, world_calls=recorder.world_calls,
                control_safety=summary.get('control_safety', runtime_contract(modules)),
                safety_failure_details=failure_details,
                elapsed_seconds=round(time.monotonic() - started, 2))
            write_json(output / 'episodes' / (label + '.json'), dict(case=case, result=result,
                trajectory=summary['trajectory'], decisions=decisions, cbf=safety,
                safety_failure_details=result['safety_failure_details'],
                ground_motion_filter_events=summary.get('ground_motion_filter_events', []),
                transport=recorder.network.stats))
            write_json(progress_path, dict(state='complete', **result))
            return result
        except BaseException as error:
            write_json(progress_path, dict(state='failed', label=label, error=repr(error),
                steps=getattr(episode, 'frame', 0), elapsed_seconds=round(time.monotonic() - started, 2)))
            if episode is not None:
                write_json(output / 'episodes' / (label + '-partial.json'), dict(case=case,
                    error=repr(error), trajectory=episode.trajectory, decisions=decisions, cbf=safety,
                    safety_failure_details=getattr(episode, 'safety_failure_details', [])))
            raise
        finally:
            if episode is not None:
                episode.close()


def summarize(results, diagnostic):
    return dict(episodes=len(results), successes=sum(row['goal_reached'] for row in results),
        success_rate=None if diagnostic or not results else sum(row['goal_reached'] for row in results) / len(results),
        diagnostic=diagnostic, truncated_episodes=sum(row['truncated'] for row in results),
        terminations={name: sum(row['termination'] == name for row in results)
                      for name in sorted({row['termination'] for row in results})},
        collisions=sum(row['collisions'] for row in results),
        ground_safety_violations=sum(row['ground_safety_violations'] for row in results))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=SOURCE.parents[3])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--case-indices', nargs='+', type=int, help='Zero-based indices in checkpoint validation cases')
    parser.add_argument('--max-steps', type=int, help='Bounded diagnostic only; omit for full mission budgets')
    parser.add_argument('--baseline-run', type=Path,
                        help='Use this run\'s archived source for a same-weight baseline, instead of current repairs')
    parser.add_argument('--frozen', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error('workers and max-steps must be positive')
    return args


def main():
    args = parse_args()
    repo, checkpoint, output = args.repo.resolve(), args.checkpoint.resolve(), args.output.resolve()
    if not args.frozen:
        source = SOURCE
        if args.baseline_run is not None:
            baseline = args.baseline_run.resolve()
            source = _sources.runtime_source(baseline)
            baseline_manifest = json.loads((baseline / 'manifest.json').read_text())
            actual = _sources.source_hashes(source)
            if actual != baseline_manifest['source_sha256']:
                raise ValueError('Baseline archived source differs from its manifest')
        entry = freeze_inputs(output, checkpoint, source, evaluator=SOURCE / 'evaluate_safety_repairs.py')
        os.execv(sys.executable, [sys.executable, '-I', '-B', '-u', str(entry), *sys.argv[1:],
            '--repo', str(repo), '--checkpoint', str(output / 'input.pt'), '--output', str(output), '--frozen'])
    source = _sources.runtime_source(output)
    pinned = json.loads((output / 'input-manifest.json').read_text())
    if checkpoint != output / 'input.pt' or digest(checkpoint) != pinned['checkpoint_sha256']:
        raise ValueError('Frozen evaluation must use its unchanged pinned checkpoint')
    if _sources.source_hashes(source) != pinned['source_sha256']:
        raise ValueError('Evaluation source snapshot changed')
    for name in ('logs', 'episodes', 'progress'):
        (output / name).mkdir(exist_ok=True)
    import torch
    torch.set_num_threads(1)
    modules = load_modules(source)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    validate_checkpoint(saved, modules)
    cases = select_cases(saved['config'], args.case_indices)
    expected_hashes = checkpoint_hashes(saved, modules['fast'].state_hash)
    write_json(output / 'cases.json', [dict(case_index=index, case=case) for index, case in cases])
    write_json(output / 'manifest.json', dict(protocol=PROTOCOL, **pinned,
        parent_policy_version=saved.get('policy_version'), parent_source_sha256=saved['manifest'].get('source_sha256'),
        model_hashes=expected_hashes, workers=args.workers, max_steps=args.max_steps,
        selected_case_indices=[index for index, _ in cases], total_validation_cases=len(saved['config']['scene_manifest']['validation']),
        deterministic_uav=True, deterministic_slow=True, model_config=saved['config'],
        control_safety=runtime_contract(modules), baseline_run=str(args.baseline_run.resolve()) if args.baseline_run else None,
        comparison='Same-weight comparisons require matching all three model hashes, task values and seeds. Source repairs intentionally differ.',
        diagnostic=args.max_steps is not None, learning=False, training_data_collected=False))
    del saved
    results = []
    started = time.monotonic()

    def status(state, **extra):
        value = dict(state=state, pid=os.getpid(), completed=len(results), total=len(cases),
                     elapsed_seconds=round(time.monotonic() - started, 2), **extra)
        write_json(output / 'status.json', value)
        print(json.dumps(json_value(value), ensure_ascii=False), flush=True)

    status('running')
    try:
        jobs = [(str(repo), str(output), index, case, args.max_steps, expected_hashes,
                 pinned['checkpoint_sha256']) for index, case in cases]
        # Release simulator maps and graph caches between missions as well as
        # disabling the much larger training tensor/replay collection.
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 max_tasks_per_child=1) as pool:
            for future in as_completed([pool.submit(evaluate_case, job) for job in jobs]):
                result = future.result()
                results.append(result)
                write_json(output / 'results.json', sorted(results, key=lambda row: row['case_index']))
                status('running', latest_case=result['case_index'], termination=result['termination'])
        summary = summarize(results, args.max_steps is not None)
        write_json(output / 'summary.json', summary)
        status('complete', summary=summary)
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
