"""Evaluate a frozen survey-point network with the native V9 aircraft on held-out kilometre tasks."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

import torch

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))


def survey_source(scripts, name):
    """Locate a dependency in the reorganized source tree for provenance."""
    if name in ('ground_policy.py', 'survey_fleet_clearance.py'):
        return scripts / ('survey' if name.startswith('survey_') else 'vtemp2_3') / name
    if name.startswith('survey_'):
        return scripts / 'survey' / name
    if name.startswith(('ground_evidence', 'ground_relay')):
        return scripts / 'shared' / 'simulator' / name
    return scripts / 'vtemp2_3' / 'navigation' / name

from vtemp2_3.frozen_navigation import CHECKPOINT_SHA, V9_RUN, runtime
from vtemp2_3.hierarchical_sources import atomic_json, file_sha256
from survey.train_survey_pretrain import load_survey_weights


def get_cases(data, survey_manifest):
    main_file = data/'scenes.json'
    far_file = data/'far-test'/'scenes.json'
    if (file_sha256(main_file) != survey_manifest['km_scenes_sha256']
            or file_sha256(far_file) != survey_manifest['far_scenes_sha256']):
        raise ValueError('Fleet cases must match the held-out lightweight evaluation')
    cases = json.loads(main_file.read_text())['test'] + json.loads(far_file.read_text())
    ids = [case['case_id'] for case in cases]
    if ids != survey_manifest['test'] + survey_manifest['far_test'] or len(set(ids)) != len(ids):
        raise ValueError('Unexpected held-out route IDs')
    return cases


def run_one(repo, checkpoint, case, options, out):
    from survey.survey_fleet_runtime import run_fleet_case
    label = case['case_id']
    progress_file = out/'progress'/f'{label}.json'
    started = time.monotonic()
    def progress(event):
        atomic_json(progress_file, dict(event, case_id=label, elapsed_wall_s=round(time.monotonic()-started, 2)))
    budget = math.ceil(3*case['direct_distance_m']+120)
    progress(dict(status='starting', budget_s=budget))
    with open(os.devnull, 'w') as sink, redirect_stdout(sink):
        torch.set_num_threads(1)
        runtime(repo)
        policy = load_survey_weights(checkpoint).eval().requires_grad_(False)
        with torch.inference_mode():
            result = run_fleet_case(repo, case, policy, fast_seed=case['seed'], budget_s=budget,
                                    progress=progress, **options)
    result.update(case_id=label, seed=case['seed'], direct_distance_m=case['direct_distance_m'],
                  time_budget_s=budget, elapsed_wall_s=round(time.monotonic()-started, 2))
    atomic_json(out/'episodes'/f'{label}.json', result)
    compact = {key:value for key,value in result.items() if not isinstance(value, (list,dict))}
    progress(dict(compact, status='complete'))
    return compact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=SOURCE.parents[3])
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--km-data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+', help='Optional subset of held-out route IDs')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--ground-interval-s', type=int, default=5)
    p.add_argument('--survey-check-s', type=int, default=5)
    p.add_argument('--survey-min-s', type=int, default=20)
    p.add_argument('--survey-timeout-s', type=int, default=60)
    args = p.parse_args()
    if min(args.workers,args.ground_interval_s,args.survey_check_s,args.survey_min_s) < 1 or args.survey_timeout_s < args.survey_min_s:
        p.error('Intervals must be positive and the survey timeout must cover minimum commitment')
    repo, checkpoint, data = args.repo.resolve(), args.checkpoint.resolve(), args.km_data.resolve()
    survey_manifest = json.loads((checkpoint.parent/'manifest.json').read_text())
    cases = get_cases(data, survey_manifest)
    if args.cases:
        known = {case['case_id']:case for case in cases}
        if len(set(args.cases)) != len(args.cases) or not set(args.cases) <= known.keys():
            p.error('Unknown or duplicated held-out route ID')
        cases = [known[key] for key in args.cases]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error('Use an empty output directory')
    for name in ('progress', 'episodes'):
        (out/name).mkdir()
    options = dict(ground_interval_s=args.ground_interval_s, survey_check_s=args.survey_check_s,
                   survey_min_s=args.survey_min_s, survey_timeout_s=args.survey_timeout_s)
    sources = [Path(__file__), SOURCE.parent/'vtemp2_3'/'frozen_navigation.py', *(survey_source(SOURCE.parent, name) for name in ('survey_fleet_runtime.py',
               'survey_fleet_observation.py','survey_fleet_metrics.py','survey_fleet_clearance.py','survey_fleet_drive.py','survey_fleet_phase.py','survey_pretrain_env.py','ground_evidence.py',
               'ground_policy.py','ground_relay.py','hierarchical_safety.py','hierarchical_runtime.py'))]
    expected = file_sha256(checkpoint)
    manifest = dict(protocol='vtemp2_frozen_v9_fleet_survey_km_v1', checkpoint=str(checkpoint),
                    checkpoint_sha256=expected, v9_checkpoint_sha256=CHECKPOINT_SHA, aircraft=3,
                    cases=[row['case_id'] for row in cases], schedule=options,
                    source_sha256={f.name:file_sha256(f) for f in sources},
                    data_sha256=survey_manifest['km_scenes_sha256'], far_data_sha256=survey_manifest['far_scenes_sha256'],
                    budget='ceil(3*direct_distance_m+120) native simulated seconds')
    atomic_json(out/'manifest.json',manifest)
    start = time.monotonic()
    rows = {}
    def status(state, **fields):
        atomic_json(out/'status.json', dict(state=state, completed=len(rows), total=len(cases),
                                            elapsed_wall_s=round(time.monotonic()-start,2), **fields))
    status('running')
    try:
        with ProcessPoolExecutor(min(args.workers,len(cases)), mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(run_one,repo,checkpoint,case,options,out):case['case_id'] for case in cases}
            for f in as_completed(futures):
                result = f.result()
                rows[result['case_id']] = result
                ordered = [rows[c['case_id']] for c in cases if c['case_id'] in rows]
                atomic_json(out/'evaluation.json', dict(cases=ordered, arrivals=sum(bool(r['goal_reached']) for r in ordered)))
                status('running')
                print(result['case_id'], result['end_reason'], result['sim_time_s'], result['remaining_m'], flush=True)
        if file_sha256(checkpoint)!=expected or file_sha256(repo/V9_RUN/'best.pt')!=CHECKPOINT_SHA:
            raise RuntimeError('Frozen checkpoint changed during fleet evaluation')
        status('complete')
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__=='__main__':
    main()
