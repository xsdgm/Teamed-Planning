"""Evaluate the continuous-angle history policy on held-out local and lake missions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))
from vtemp4.evaluate_vtemp4_waypoints import (
    LAKE_CASES, LOCAL_SEEDS, RETURN_RULE, aggregate, checkpoint_info, evaluate_arm, held_out_cases,
)
from vtemp2_3.frozen_navigation import V9_RUN
from vtemp2_3.hierarchical_sources import atomic_json, file_sha256
from vtemp4.train_vtemp4_formal import exclusive_run, source_digest
from vtemp4.vtemp4_curriculum import geometry_hash
from vtemp4.vtemp4_ring_contract import GOAL_VARIANTS, PROGRESS_DISTANCES
from vtemp4.vtemp4_stage_eval import bind_manifest


VARIANTS = (*GOAL_VARIANTS, 'periodic_unified_attention', 'periodic_heading_history', 'continuous_heading_history')


def run(args):
    checkpoint = args.checkpoint.resolve(strict=True)
    saved_manifest = json.loads((checkpoint.parent.parent / 'manifest.json').read_text())
    variant = args.policy_variant or saved_manifest['policy_variant']
    if variant not in VARIANTS:
        raise ValueError(f'Unsupported continuous-history policy variant: {variant}')
    distance = saved_manifest.get('progress_distance', 'euclidean')
    if getattr(args, 'progress_distance', None) not in (None, distance):
        raise ValueError('Requested evaluation progress distance differs from training')
    if args.lake_cases is None:
        args.lake_cases = () if distance == 'traversable' else LAKE_CASES
    if distance == 'traversable' and args.lake_cases:
        raise ValueError('Traversable progress distance is available only for synthetic circle obstacle maps')
    cases = held_out_cases(args)
    seeds = args.fast_seeds if args.fast_seeds is not None else list(range(20481001, 20481001 + len(cases)))
    if len(seeds) != len(cases) or any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError('Specify one uint32 fleet seed per selected map')
    info, _, manifest = checkpoint_info(checkpoint, variant, cases, args.pilot)
    frozen = args.repo / V9_RUN
    fleet = dict(v9_sha256=file_sha256(frozen / 'best.pt'), v9_config_sha256=file_sha256(frozen / 'config.json'),
                 v9_sources_sha256=source_digest(frozen / 'runtime_sources'))
    if any(manifest[k] != v for k, v in fleet.items()):
        raise ValueError('Frozen fleet inputs differ from the training manifest')
    plan = [dict(case_id=c['case_id'], fast_seed=s, geometry_sha256=geometry_hash(c),
                 budget_s=math.ceil(3*c['direct_distance_m'] + 120)) for c,s in zip(cases, seeds)]
    protocol = dict(format='vtemp4_continuous_history_evaluation_v1', policy_variant=variant, policy=info,
                    technical_pilot=args.pilot, split='development', fleet=fleet, cases=plan,
                    collector_seconds=args.collector_seconds, backtracking_definition=RETURN_RULE,
                    evaluator_sha256=file_sha256(Path(__file__)))
    bind_manifest(args.output / 'protocol.json', protocol)
    rows = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=min(args.workers, len(cases))) as pool:
        jobs = {pool.submit(evaluate_arm, args.repo, info, case, seed, args.output / case['case_id'],
                            args.collector_seconds): i for i, (case, seed) in enumerate(zip(cases, seeds))}
        for n, job in enumerate(as_completed(jobs), 1):
            i = jobs[job]
            rows[i] = job.result()
            atomic_json(args.output / 'progress.json', dict(completed=n, total=len(cases), case_id=cases[i]['case_id']))
    strata = {s: aggregate([r for r in rows if r['stratum'] == s]) for s in sorted({r['stratum'] for r in rows})}
    atomic_json(args.output / 'evaluation.json', dict(**protocol, overall=aggregate(rows), by_stratum=strata, results=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=SOURCE.parents[3])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--policy-variant', choices=VARIANTS, help='Defaults to the training manifest for the checkpoint')
    parser.add_argument('--scene-seeds', nargs='*', type=int, default=LOCAL_SEEDS)
    parser.add_argument('--lake-cases', nargs='*',
                        help='Additional lake missions (defaults to two development lakes for euclidean rewards)')
    parser.add_argument('--progress-distance', choices=PROGRESS_DISTANCES,
                        help='Assert the checkpoint uses this vehicle progress distance')
    parser.add_argument('--fast-seeds', nargs='+', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pilot', action='store_true', help='Allow a technical test of a nonfinal checkpoint')
    parser.add_argument('--collector-seconds', type=int)
    args = parser.parse_args()
    if args.workers < 1 or (args.collector_seconds is not None and (args.collector_seconds < 1 or not args.pilot)):
        parser.error('Use positive worker counts and limit the horizon only for a technical pilot')
    args.repo, args.output = args.repo.resolve(), args.output.resolve()
    with exclusive_run(args.output):
        run(args)
