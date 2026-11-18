"""Frozen v9 three/four-drone evaluation, with auditable replay and layout control.

No training or production-code changes. Each job uses the historical model
construction RNG order, original task budget, observations and safety filter.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import concurrent.futures
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

REPO = (Path(__file__).resolve().parent / '../../..').resolve()
TRAINING = REPO / 'artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146'
REPLAY = REPO.parent / 'pre/v9_success_validation.html'
WAIT30 = REPO / 'artifacts/v9-start-delay-20261110/data'
SAFETY_FIELDS = ('ground_collisions', 'collisions', 'spacing_violation',
                 'chain_link_violation', 'out_of_bounds', 'communication_outage')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    temporary.replace(path)


def source_hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*.py'))}


def job(output, spec):
    output = Path(output)
    key = spec['key']
    started = time.monotonic()
    try:
        import numpy as np
        import torch
        sys.path.insert(0, str(output / 'runtime_sources'))
        from simulator import train_mappo as tm
        from simulator import mappo_experiment as me
        case = json.loads((output / 'cases.json').read_text())[str(spec['case'])]
        config = json.loads((output / 'config.json').read_text())
        config['fleet_sizes'] = [spec['fleet']]
        if spec.get('limit'):
            config['steps'] = spec['limit']
        torch.set_num_threads(1)
        torch.manual_seed(spec['sample_seed'])
        saved = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
        model = tm.ActorCritic.from_state_dict(saved['model'])
        model.eval()
        trajectory, frames, trace = [], [], []
        captured = {'max_hold_displacement_m': 0.0}
        original_experiment, original_env = tm.Experiment, me.MultiRobotEnv

        class PreservedLayoutEnv(original_env):
            def add_drone(self, x, y, z=1., **kwargs):
                index = len(self.drones)
                # All three original aircraft remain at their exact baseline
                # positions; append the fourth at (+27,+18) relative to vehicle.
                offsets = [(9., -18.), (9., 0.), (9., 18.), (27., 18.)]
                dx, dy = offsets[index]
                x, y = case['start'][0] + dx, case['start'][1] + dy
                if not (0 <= x <= 1000 and 0 <= y <= 1000):
                    raise ValueError('Preserved layout lies outside the world')
                return super().add_drone(x, y, z, **kwargs)

        class RecordingExperiment(original_experiment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured['env'] = self
                assert len(self.ids) == spec['fleet']
                np.testing.assert_allclose(self.start, case['start'], rtol=0, atol=1e-10)
                np.testing.assert_allclose(self.goal, case['goal'], rtol=0, atol=1e-10)
                self.record(0)

            def _ground_action(self, neighbors=None):
                # Match the archived wait30 experiment exactly: sensing,
                # communication, map fusion and planning continue normally.
                # There is no additional forced replan on departure.
                action = super()._ground_action(neighbors)
                if self.env.time < spec['hold_seconds']:
                    action = dict(action)
                    action['motion'] = (0., 0.)
                return action

            def record(self, step):
                positions = [np.asarray(self.obs[r]['self']['position'])[:2].tolist()
                             for r in self.ids + [self.ground]]
                trajectory.append(dict(step=step, positions=positions))
                if self.env.time <= spec['hold_seconds']:
                    displacement = float(np.linalg.norm(np.asarray(positions[-1])-self.start))
                    captured['max_hold_displacement_m'] = max(captured['max_hold_displacement_m'], displacement)
                    assert displacement < 1e-9, 'Vehicle moved during the initial survey period'
                if step % 6 == 0 or self.goal_reached:
                    planner = self.ground_controller.global_planner
                    frames.append(dict(step=step, ground_path=planner.get_remaining_path() or [],
                        using_graph=bool(planner.using_graph),
                        waiting_at_frontier=bool(planner.waiting_at_frontier)))

            def step(self, frame):
                result = super().step(frame)
                self.record(frame + 1)
                # Keep scalar measurements; full aircraft/vehicle trajectories
                # and actual planned waypoints are recorded separately.
                trace.append({k: float(v) for k, v in result[2].items()
                              if isinstance(v, (int, float, bool, np.number))
                              and np.isfinite(v)})
                return result

        def progress(value):
            write_json(output / 'data' / f'{key}.progress.json',
                       dict(status='running', wall_seconds=time.monotonic()-started, **spec, **value))

        tm.Experiment = RecordingExperiment
        if spec['layout'] == 'preserve_base':
            me.MultiRobotEnv = PreservedLayoutEnv
        try:
            with torch.inference_mode():
                rollout, summary = tm.run_episode(model, config, case['seed'],
                    tm.fixed_schedule(config, config['episodes']-1), True, progress=progress)
        finally:
            tm.Experiment, me.MultiRobotEnv = original_experiment, original_env

        assert all(row[0].shape[0] == spec['fleet'] for row in rollout)
        assert all(np.isfinite(row[0]).all() and np.isfinite(row[2]).all() for row in rollout)
        points = np.asarray([row['positions'][-1] for row in trajectory])
        segments = np.diff(points, axis=0)
        travel = float(np.linalg.norm(segments, axis=1).sum())
        np.testing.assert_allclose(travel, summary['ground_distance'], rtol=0, atol=1e-8)
        direction = (np.asarray(case['goal'])-case['start']) / case['distance']
        backward = float(-np.minimum(segments @ direction, 0).sum())
        moving = np.flatnonzero(np.linalg.norm(segments, axis=1) > 1e-6)
        safety = {k: summary[k] for k in SAFETY_FIELDS}
        reason = ('cbf_infeasible' if summary['safety_terminated'] else
                  'vehicle_arrival' if summary['goal_reached'] else
                  'budget_exhausted' if summary['budget_exhausted'] else 'other')
        parity = None
        if spec['fleet'] == 3 and spec['sample'] == 0:
            expected = {row['step']: np.asarray(row['positions']) for row in case['trajectory']}
            error = max(float(np.abs(np.asarray(row['positions'])-expected[row['step']]).max())
                        for row in trajectory)
            differences = {} if spec.get('limit') else {
                k: [summary.get(k), v] for k, v in case['summary'].items()
                if k != 'decision_ms' and (k not in summary or
                    not np.isclose(summary[k], v, rtol=0, atol=1e-8, equal_nan=True))}
            parity = dict(passed=error <= 5.01e-5 and not differences,
                          max_position_error_m=error, summary_differences=differences)
            if not parity['passed']:
                write_json(output / 'data' / f'{key}.parity_failure.json', parity)
                raise AssertionError('Three-drone baseline does not reproduce archived replay')
        landmarks = {}
        for step in (30, 60, 120, 180, 300, 450, 600):
            if len(trace) >= step:
                landmarks[str(step)] = {k: trace[step-1].get(k) for k in (
                    'ground_goal_distance', 'graph_task_connected', 'graph_frontier_goal_distance_m',
                    'graph_nodes', 'graph_components', 'frontier_progress')}
        brief = dict(**spec, status='completed', success=bool(summary['goal_reached']),
            strict_success=bool(summary['goal_reached'] and not summary['safety_terminated']
                                and not any(safety.values())),
            termination=reason, steps=int(summary['rollout_steps']),
            executed_steps=len(trace), budget_steps=summary['episode_budget_steps'],
            travel_m=travel, backward_m=backward,
            remaining_m=summary['ground_goal_distance'],
            first_motion_step=int(moving[0]+1) if len(moving) else None,
            max_hold_displacement_m=captured['max_hold_displacement_m'],
            first_connection_step=summary['graph_first_connection_step'],
            safety=safety, parity=parity, wall_seconds=time.monotonic()-started)
        write_json(output / 'data' / f'{key}.json', dict(
            **brief, summary=summary, trajectory=trajectory, frames=frames, trace=trace,
            landmarks=landmarks, initial_positions=trajectory[0]['positions'],
            actor_input_shape=list(rollout[0][0].shape), reference_length_m=case['reference_length']))
        write_json(output / 'data' / f'{key}.progress.json', brief)
        return brief
    except Exception:
        error = dict(**spec, status='error', traceback=traceback.format_exc(),
                     wall_seconds=time.monotonic()-started)
        write_json(output / 'data' / f'{key}.error.json', error)
        return error


def summarize(output, results):
    main = [r for r in results if r['layout'] == 'native' and r['status'] == 'completed']
    by_key = {(r['case'], r['sample'], r['fleet']): r for r in main}
    pairs = []
    for case, sample in sorted({(r['case'], r['sample']) for r in main}):
        a, b = by_key.get((case, sample, 3)), by_key.get((case, sample, 4))
        if a is None or b is None:
            continue
        both = a['strict_success'] and b['strict_success']
        pairs.append(dict(case=case, sample=sample, both_strict_success=both,
            three=a, four=b,
            travel_change_pct=100*(b['travel_m']/a['travel_m']-1) if both else None,
            time_change_s=b['steps']-a['steps'] if both else None))
    groups = {}
    for fleet in (3, 4):
        rows = [r for r in main if r['fleet'] == fleet]
        groups[str(fleet)] = dict(completed=len(rows), arrivals=sum(r['success'] for r in rows),
            strict_successes=sum(r['strict_success'] for r in rows),
            cbf_failures=sum(r['termination']=='cbf_infeasible' for r in rows),
            budget_exhausted=sum(r['termination']=='budget_exhausted' for r in rows),
            safety_sums={k: sum(r['safety'][k] for r in rows) for k in SAFETY_FIELDS})
    paired_success = [p for p in pairs if p['both_strict_success']]
    overall = dict(groups=groups, pairs=pairs,
        paired_strict_success_count=len(paired_success),
        paired_travel_change_pct=(100*(sum(p['four']['travel_m'] for p in paired_success) /
            sum(p['three']['travel_m'] for p in paired_success)-1) if paired_success else None),
        paired_time_change_pct=(100*(sum(p['four']['steps'] for p in paired_success) /
            sum(p['three']['steps'] for p in paired_success)-1) if paired_success else None),
        errors=[r for r in results if r['status']=='error'],
        layout_controls=[r for r in results if r['layout']=='preserve_base'])
    write_json(output / 'comparison.json', overall)
    with (output / 'results.csv').open('w', newline='') as stream:
        fields = ['case', 'sample', 'sample_seed', 'fleet', 'layout', 'status', 'success',
                  'strict_success', 'termination', 'steps', 'travel_m', 'backward_m',
                  'remaining_m', 'first_connection_step']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r['key']))
    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--hold-seconds', type=int, choices=(0,30), default=30)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    data = json.loads(re.search(r'<script id="replay-data" type="application/json">(.*?)</script>',
                              REPLAY.read_text(), re.S).group(1))
    cases = {str(n): data['cases'][n-1] for n in (2, 3, 4)}
    cases = {n: {k: c[k] for k in ('id','seed','sample_seed','start','goal','distance',
                                  'obstacles','reference','reference_length','summary','trajectory')}
             for n, c in cases.items()}
    baseline_sources = {}
    if args.hold_seconds == 30:
        for n, case in cases.items():
            source = WAIT30 / f'case{n}_wait30.json'
            expected = json.loads(source.read_text())
            assert expected['sample_seed'] == case['sample_seed']
            assert expected['hold_seconds'] == 30 and not expected['replan_at_departure']
            case['summary'], case['trajectory'] = expected['summary'], expected['trajectory']
            baseline_sources[str(source)] = digest(source)
    write_json(output / 'cases.json', cases)
    shutil.copytree(TRAINING / 'runtime_sources', output / 'runtime_sources',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('best.pt', 'config.json'):
        shutil.copy2(TRAINING / name, output / name)
    shutil.copy2(__file__, output / 'runner.py')
    config = json.loads((output / 'config.json').read_text())
    specs = []
    for sample in range(args.samples):
        for case in (2, 3, 4):
            c = cases[str(case)]
            seed = int(np.random.SeedSequence([config['seed'], c['seed'], sample]).generate_state(1)[0])
            if sample == 0:
                assert seed == c['sample_seed']
            variants = [(3, 'native'), (4, 'native')]
            if case == 4:
                variants.append((4, 'preserve_base'))
            for fleet, layout in variants:
                specs.append(dict(key=f'case{case}_r{sample}_n{fleet}_{layout}',
                    case=case, sample=sample, sample_seed=seed, fleet=fleet, layout=layout,
                    limit=args.limit, hold_seconds=args.hold_seconds))
    hashes = source_hashes(output / 'runtime_sources')
    provenance = dict(checkpoint_sha256=digest(output/'best.pt'),
        checkpoint_original=str(TRAINING/'best.pt'), original_config_sha256=digest(output/'config.json'),
        source_sha256=hashes, source_replay_sha256=digest(REPLAY),
        baseline_sources=baseline_sources,
        runner_sha256=digest(output/'runner.py'), jobs=specs, workers=args.workers,
        policy='frozen best576, stochastic inference, no updates',
        original_training_fleet=[3], changed_evaluation_config={'fleet_sizes': '[3] versus [4]'},
        startup_hold_seconds=args.hold_seconds,
        startup_protocol='Vehicle receives maps and plans while motion is suppressed for t<hold; '
            'no forced departure replan. Hold is included in the original total time budget.',
        layout_control='Case 4 native N=4 mirrors all UAV X offsets; preserve_base retains the '
            'original three positions and places UAV4 at ground+(27,18).',
        randomization='Original validation seed formula and model-construction order. Same seed '
            'does not mean identical per-agent noise after fleet size changes RNG consumption.',
        safety='Frozen 1.5m ground clearance, 15m UAV separation, 95m communication, 6m/s UAV speed.',
        metrics='Vehicle arrival <=2m; original full time budgets. Path/time comparison uses '
            'only pairs in which both fleets arrive safely. Failures remain in success counts.',
        caveat='Three previously selected successful scenes, three action seeds each; '
            'exploratory transfer evaluation, not an independent map generalization estimate.')
    write_json(output / 'provenance.json', provenance)
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = [pool.submit(job, str(output), spec) for spec in specs]
        for future in concurrent.futures.as_completed(pending):
            result = future.result()
            results.append(result)
            write_json(output / 'results.json', results)
            write_json(output / 'status.json', dict(status='running', completed=len(results), total=len(specs)))
            print(json.dumps({k: v for k, v in result.items() if k not in ('parity', 'safety')},
                             ensure_ascii=False), flush=True)
    summary = summarize(output, results)
    integrity = dict(runtime_unchanged=hashes==source_hashes(output/'runtime_sources'),
        checkpoint_unchanged=digest(output/'best.pt')==provenance['checkpoint_sha256'],
        original_checkpoint_unchanged=digest(TRAINING/'best.pt')==provenance['checkpoint_sha256'],
        original_config_unchanged=digest(TRAINING/'config.json')==provenance['original_config_sha256'],
        baseline_parity_passed=all(r.get('parity', {}).get('passed',False) for r in results
                                 if r['fleet']==3 and r['sample']==0 and r['status']=='completed'),
        no_execution_errors=not summary['errors'])
    write_json(output / 'verification.json', integrity)
    assert all(integrity.values()), integrity
    write_json(output / 'status.json', dict(status='completed', completed=len(results), total=len(specs)))


if __name__ == '__main__':
    main()
