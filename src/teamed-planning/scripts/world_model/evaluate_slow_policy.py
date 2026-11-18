"""Matched, frozen learned-slow evaluation on an existing fixed-target test set."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
from dataclasses import replace
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

_helper_path = Path(__file__).with_name('hierarchical_sources.py')
_helper_spec = importlib.util.spec_from_file_location('_hierarchical_sources_compat', _helper_path)
_sources = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_sources)


def helper(reference):
    path = Path(reference) / 'evaluate_fixed_target.py'
    spec = importlib.util.spec_from_file_location('_fixed_evaluation_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Preserve the reference evaluator's utilities while supporting a newly
    # organized selected runtime. Do not edit the reference artifact itself.
    module.frozen_modules = lambda run: _sources.load_runtime_modules(_sources.runtime_source(run))
    return module


def evaluate_case(job):
    repo, run, reference, checkpoint, output, case, option_steps, expected_hashes = job
    import numpy as np
    import torch
    torch.set_num_threads(1)
    util = helper(reference)
    modules = util.frozen_modules(run)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = saved['config']
    fresh_arguments = {'random_init': True} if (config.get('fresh_all', False)
                                               or config.get('fresh_fast_world', False)) else {}
    output = Path(output)
    label = f"{case['kind']}-{case['direct_distance_m']:g}m-seed{case['seed']}"
    started = time.monotonic()
    with (output / 'logs' / (label + '.log')).open('w') as log, contextlib.redirect_stdout(log):
        hierarchy = replace(modules['hierarchy'].HierarchyConfig(), option_steps=option_steps)
        budget = max(hierarchy.min_mission_budget,
                     math.ceil(hierarchy.budget_multiplier * case['direct_distance_m'] + hierarchy.budget_extra))
        episode = modules['hierarchy'].HierarchicalEpisode(repo, case, hierarchy,
            fast_seed=case['seed'] + 43, rollout_limit=budget + 1, **fresh_arguments)
        try:
            fast = modules['graph_policy'].make_graph_fast(repo, **fresh_arguments)
            world = modules['graph_world'].GraphWorldModel(config['hidden'], config['horizon'])
            slow = modules['graph_policy'].GraphSlowActorCritic(hidden=config['hidden'])
            models = dict(fast=fast, world=world, slow=slow)
            for name, model in models.items():
                model.load_state_dict(saved[name]); model.eval().requires_grad_(False)
            hashes = lambda: {name: modules['fast'].state_hash(modules['fast'].numpy_state(model))
                              for name, model in models.items()}
            before = hashes()
            if any(before[name] != expected_hashes[name] for name in ('fast', 'world')):
                raise AssertionError('Worker fast/world weights differ from the fixed-target reference')
            episode.fast_model = fast
            for controller in episode.env.controllers:
                controller.navigator.model = fast
                controller.navigator.deterministic = True
            safety = []
            decisions = []
            status_path = output / 'progress' / (label + '.json')

            class Recorder(modules['graph_data'].GraphRecorder):
                def record_truth(self, frame): pass
                def save_input(self, graph): pass
                def step(self, frame):
                    self.prepare(frame)
                    # Match training: own ground history is refreshed every
                    # fast frame, not merely when the slow actor is queried.
                    graph = self.graph('ground_0', self.ground_observation())
                    self.predict(graph, 'ugv')
                    try:
                        result = self.base_step(frame)
                    finally:
                        for controller in self.env.controllers:
                            stats = controller.safety_stats
                            margin = stats.get('feasibility_margin')
                            safety.append(dict(frame=int(frame), owner=controller.drone_id,
                                intervention_m_s=float(stats.get('intervention', 0.)),
                                infeasible=bool(stats.get('infeasible', False)),
                                missing_links=list(stats.get('missing_links', ())),
                                margin_m_s=float(margin) if margin is not None and math.isfinite(margin) else None))
                    if frame % 20 == 0 or self.env.goal_reached:
                        p = np.asarray(self.env.obs[self.env.ground]['self']['position'][:2])
                        util.write_json(status_path, dict(state='running', label=label, steps=frame + 1,
                            mission_budget=budget, remaining_m=float(np.linalg.norm(p - episode.goal)),
                            slow_decisions=len(decisions), elapsed_seconds=round(time.monotonic() - started, 2)))
                    return result

            teacher = None if fresh_arguments else modules['validate'].runtime(repo)[2]
            recorder = Recorder(episode, world, fast, teacher, config)
            while not episode.finished:
                obs = recorder.slow_observation()
                action, logp, value = slow.act(obs, deterministic=True)
                origin = np.asarray(episode.env.obs[episode.env.ground]['self']['position'][:2]).copy()
                target = obs['targets'][action].copy()
                distance = float(np.linalg.norm(target - origin))
                reward, duration, terminal, truncated, record = episode.execute(obs, action)
                decisions.append(dict(record, origin=origin.tolist(), initial_subgoal_distance_m=distance,
                    valid_candidates=int(obs['mask'].sum()), selected_probability=math.exp(logp), value=value,
                    final_available=any(kind == 'final' and bool(obs['mask'][i])
                                        for i, kind in enumerate(obs['kinds'])),
                    singleton_mask=int(obs['mask'].sum()) == 1,
                    terminated=terminal, truncated=truncated))
            summary = episode.summary()
            after = hashes()
            if before != after: raise AssertionError('Frozen models changed during evaluation')
            if summary['goal_reached'] and summary['remaining_m'] > 2. + 1e-6:
                raise AssertionError('Arrival exceeded original two-metre tolerance')
            distances = [float(np.linalg.norm(np.asarray(p) - episode.goal)) for p in summary['trajectory']]
            result = dict(label=label, seed=case['seed'], kind=case['kind'], distance_m=case['direct_distance_m'],
                goal_reached=summary['goal_reached'], termination=summary['termination'], steps=summary['steps'],
                mission_budget=episode.mission_budget, remaining_m=summary['remaining_m'], min_remaining_m=min(distances),
                collisions=summary['collisions'], slow_decisions=len(decisions),
                singleton_mask_decisions=sum(row['singleton_mask'] for row in decisions),
                decision_contracts={name: sum(row.get('decision_contract', 'legacy_policy_choice') == name
                                              for row in decisions)
                                    for name in sorted({row.get('decision_contract', 'legacy_policy_choice')
                                                        for row in decisions})},
                target_changes=sum(row['changed'] for row in decisions),
                mean_subgoal_distance_m=float(np.mean([row['initial_subgoal_distance_m'] for row in decisions])),
                cbf_infeasible_decisions=sum(row['infeasible'] for row in safety),
                cbf_mean_intervention_m_s=float(np.mean([row['intervention_m_s'] for row in safety])),
                frozen_hashes=after, world_calls=recorder.world_calls, deterministic=True,
                hierarchy_config=summary.get('config'),
                option_steps=option_steps, elapsed_seconds=round(time.monotonic() - started, 2))
            util.write_json(output / 'episodes' / (label + '.json'), dict(case=case, result=result,
                trajectory=summary['trajectory'], decisions=decisions, cbf=safety, transport=recorder.network.stats))
            util.write_json(status_path, dict(state='complete', **result))
            return result
        finally:
            episode.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'run', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--checkpoint', default='final.pt',
                        help='Checkpoint in the selected run, or an absolute checkpoint path')
    parser.add_argument('--option-steps', type=int,
                        help='Override selected run option_steps; default uses its training configuration')
    parser.add_argument('--frozen', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1 or (args.option_steps is not None and args.option_steps < 4):
        parser.error('workers>=1 and option-steps>=4 required')
    repo, run, reference, output = (getattr(args, key).resolve() for key in ('repo', 'run', 'reference', 'output'))
    checkpoint = Path(args.checkpoint)
    checkpoint = checkpoint.resolve() if checkpoint.is_absolute() else run / 'checkpoints' / checkpoint
    if not args.frozen:
        if output.exists() and any(output.iterdir()): raise ValueError('Evaluation output is not empty')
        output.mkdir(parents=True, exist_ok=True)
        target = output / 'evaluate_slow_policy.py'; shutil.copy2(__file__, target)
        shutil.copy2(_helper_path, output / 'hierarchical_sources.py')
        os.execv(sys.executable, [sys.executable, '-I', '-B', '-u', str(target), *sys.argv[1:],
            '--repo', str(repo), '--run', str(run), '--reference', str(reference), '--output', str(output),
            '--checkpoint', str(checkpoint), '--frozen'])
    util = helper(reference)
    ref = json.loads((reference / 'manifest.json').read_text())
    if not ref['deterministic']: raise ValueError('Matched reference must use deterministic actions')
    if util.digest(ref['checkpoint']) != ref['checkpoint_sha256']: raise ValueError('Reference checkpoint changed')
    reference_run = Path(ref['run']).resolve()
    for name, expected in ref['source_sha256'].items():
        if util.digest(_sources.runtime_source(reference_run) / name) != expected:
            raise ValueError('Reference archived source changed: ' + name)
    manifest = json.loads((run / 'manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if util.digest(_sources.runtime_source(run) / name) != expected:
            raise ValueError('Archived source changed: ' + name)
    import torch
    torch.set_num_threads(1)
    modules = util.frozen_modules(run)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['manifest']['source_sha256'] != manifest['source_sha256']:
        raise ValueError('Checkpoint and selected run use different source snapshots')
    if saved['world_schema'] != modules['graph_world'].MODEL_SCHEMA:
        raise ValueError('Checkpoint world schema differs from selected runtime')
    if saved['manifest']['slow_schema'] != modules['graph_policy'].SLOW_SCHEMA:
        raise ValueError('Checkpoint slow schema differs from selected runtime')
    baseline_results = json.loads((reference / 'results.json').read_text())
    expected_hashes = {name: baseline_results[0]['frozen_hashes'][name] for name in ('fast', 'world')}
    if any(any(row['frozen_hashes'][name] != expected_hashes[name] for name in expected_hashes)
           for row in baseline_results):
        raise ValueError('Reference used inconsistent frozen fast/world weights')
    actual_hashes = {name: modules['fast'].state_hash({key: value.detach().cpu().numpy()
                     for key, value in saved[name].items()}) for name in expected_hashes}
    if actual_hashes != expected_hashes:
        raise ValueError('Matched evaluation requires exactly the reference fast/world weights; use a frozen --graph-init run')
    option_steps = args.option_steps if args.option_steps is not None else saved['config']['option_steps']
    for name in ('logs', 'episodes', 'progress'): (output / name).mkdir(exist_ok=True)
    cases = json.loads((reference / 'cases.json').read_text())
    util.write_json(output / 'cases.json', cases)
    util.write_json(output / 'manifest.json', dict(protocol='vtemp2_matched_learned_slow_full_budget_v2',
        reference=str(reference), run=str(run), checkpoint=str(checkpoint),
        checkpoint_sha256=util.digest(checkpoint), source_sha256=manifest['source_sha256'],
        source_package=_sources.source_package(_sources.runtime_source(run)),
        source_helper_sha256=util.digest(_helper_path),
        reference_source_sha256=ref['source_sha256'], matched_fast_world_hashes=actual_hashes,
        evaluator_sha256=util.digest(__file__), reference_evaluator_sha256=util.digest(reference / 'evaluate_fixed_target.py'),
        option_steps=option_steps, option_steps_overridden=args.option_steps is not None,
        evidence_interval=10, deterministic_uav=True, deterministic_slow=True,
        control='Frozen world/fast/slow; selected run learned slow actor and execution contract; physical planner/CBF retained; full mission budget.',
        matching='Same exact cases, fast/world parameter hashes, fast seed and deterministic UAV actions as the fixed-target reference.',
        difference='The selected run may use a repaired slow graph interface and execution contract. This compares the entire slow branch, including singleton action masks, not an isolated neural-weight ablation. No learning or collection cutoff.'))
    results = []; started = time.monotonic()
    def status(state, **extra):
        row = dict(state=state, pid=os.getpid(), completed=len(results), total=len(cases),
                   elapsed_seconds=round(time.monotonic() - started, 2), **extra)
        util.write_json(output / 'status.json', row); print(json.dumps(row), flush=True)
    status('running')
    try:
        jobs = [(str(repo), str(run), str(reference), str(checkpoint), str(output), case,
                 option_steps, expected_hashes) for case in cases]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(evaluate_case, job) for job in jobs]):
                result = future.result(); results.append(result)
                util.write_json(output / 'results.json', sorted(results, key=lambda row: row['label']))
                status('running', latest=result)
        summary = util.summarize(results)
        util.write_json(output / 'summary.json', summary); status('complete', summary=summary)
    except BaseException as error:
        status('failed', error=repr(error)); raise


if __name__ == '__main__':
    main()
