"""Train or evaluate continuous lookahead turns on kilometre missions with the frozen V9 fleet."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys

import torch

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[3]
sys.path.insert(0, str(SOURCE.parent))
from vtemp2_3.frozen_navigation import CHECKPOINT_SHA, runtime
from vtemp2_3.hierarchical_sources import atomic_json
from vtemp2_3.train_slow_policy import atomic_checkpoint
from vtemp4.vtemp4_curriculum import (
    BRIDGE_GEOMETRY, FIELD_DATA, KM_DATA, LAKE_DATA, LAKE_GEOMETRY, balanced_cases, geometry_hash,
    load_curriculum, runtime_case, source_id, stratum_counts,
)

FORMAT = 'vtemp4_ego_attention_lookahead_v1'


def training_curriculum(field_override):
    # Preserve older invocations which explicitly supplied a bridge catalogue.
    if field_override is not None and (field_override / 'scenes.json').is_file():
        data = json.loads((field_override / 'scenes.json').read_text())
        if data['ppo_train'] and all(row.get('kind') != 'procedural_wooded_lake' for row in data['ppo_train']):
            return 'mixed'
    return 'lakes'


def run_episode(repo, model, case, clock, seed, collector_seconds, rollout):
    from vtemp4.vtemp4_runtime import run_heading_case

    budget = math.ceil(3 * case['direct_distance_m'] + 120)
    with open(os.devnull, 'w') as quiet, redirect_stdout(quiet), torch.inference_mode():
        record = run_heading_case(repo, runtime_case(case), model, fast_seed=seed, budget_s=budget,
                                  collector_limit_s=collector_seconds, rollout=rollout, **clock)
    record.update(navigation_task=case['navigation_task'], terrain_family_id=case['terrain_family_id'],
                  geometry_sha256=geometry_hash(case), fast_seed=seed)
    return record


class AlwaysHold:
    def act(self, observation, *, deterministic, generator):
        # The native first hold initializes the world bearing to the goal from the starting pose.
        return False, 0., 0., 0.


def summarize(record):
    keys = ('case_id', 'navigation_task', 'terrain_family_id', 'geometry_sha256', 'seed', 'fast_seed',
            'direct_distance_m', 'end_reason', 'remaining_m', 'goal_reached', 'sim_time_s', 'reward',
            'travelled_m', 'turn_count')
    row = {key: record[key] for key in keys}
    turns = [event for event in record['heading_reviews'] if event['turn']]
    row.update(decisions=len(record['heading_reviews']), left_turns=sum(t['angle_rad'] > 0. for t in turns),
               right_turns=sum(t['angle_rad'] < 0. for t in turns),
               first_turn_s=turns[0]['time_s'] if turns else None,
               first_turn_rad=turns[0]['angle_rad'] if turns else None)
    return row


def metrics(rows, expected):
    completed = sum(r['end_reason'] != 'collector_limit' for r in rows)
    arrivals = sum(r['goal_reached'] for r in rows)
    decisions = sum(r['decisions'] for r in rows)
    turns = sum(r['turn_count'] for r in rows)
    result = dict(reported_missions=len(rows), expected_missions=expected, completed_missions=completed,
                  arrivals=arrivals, success_rate=arrivals / expected if completed == expected and expected else None,
                  decisions=decisions, turn_count=turns, turn_rate=turns / decisions if decisions else None,
                  left_turns=sum(r['left_turns'] for r in rows), right_turns=sum(r['right_turns'] for r in rows),
                  mean_remaining_m=sum(r['remaining_m'] for r in rows) / len(rows) if rows else None)
    if rows and 'deterministic_turns_on_sampled_observations' in rows[0]:
        result['deterministic_turns_on_sampled_observations'] = sum(
            r['deterministic_turns_on_sampled_observations'] for r in rows)
    return result


def by_stratum(rows, expected):
    return {task: metrics([r for r in rows if r['navigation_task'] == task], count)
            for task, count in expected.items()}


@torch.inference_mode()
def deterministic_turns_on_rollout(model, rollout):
    count = 0
    for start in range(0, len(rollout), 16):
        distribution, _ = model([r['observation'] for r in rollout[start:start + 16]])
        turn, _ = distribution.action(deterministic=True, generator=None)
        count += int(turn.sum().item())
    return count


def train(args, out, model, cases, clock):
    from vtemp4.vtemp4_policy import update_policy

    rng = random.Random(args.seed)
    schedule = balanced_cases(cases, rng)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    training = []
    training_maps, training_sources, training_families = set(), set(), set()
    for update in range(1, args.updates + 1):
        episodes, summaries = [], []
        for episode in range(args.episodes_per_update):
            case = next(schedule)
            training_maps.add(geometry_hash(case))
            training_sources.add(source_id(case))
            training_families.add(case['terrain_family_id'])
            rollout = []
            seed = rng.randrange(2**31)
            record = run_episode(args.repo, model.eval(), case, clock, seed, args.collector_seconds, rollout)
            if not rollout:
                raise RuntimeError(f"No executed heading decision for {case['case_id']}")
            row = summarize(record)
            row['deterministic_turns_on_sampled_observations'] = deterministic_turns_on_rollout(model, rollout)
            record['deterministic_turns_on_sampled_observations'] = row['deterministic_turns_on_sampled_observations']
            atomic_json(out / 'episodes' / f'train-{update:04d}-{episode:03d}.json', record)
            summaries.append(row)
            episodes.append(rollout)
        ppo = update_policy(model.train(), optimizer, episodes)
        strata = by_stratum(summaries, stratum_counts(summaries))
        training.append(dict(update=update, episodes=summaries, by_stratum=strata, ppo=ppo))
        atomic_checkpoint(out / 'heading.pt', dict(format=FORMAT, model=model.state_dict(), clock=clock,
                          v9_sha256=CHECKPOINT_SHA, curriculum=args.curriculum, field_geometry=args.field_geometry,
                          dataset=args.dataset, training_maps=sorted(training_maps),
                          training_sources=sorted(training_sources), training_families=sorted(training_families),
                          completed_updates=update))
        atomic_json(out / 'training.json', training)
        print(f'Update {update}/{args.updates}: {sum(len(ep) for ep in episodes)} slow decisions', flush=True)


def evaluate(args, out, model, cases, clock):
    rows, holds, pairs = [], [], []
    expected = stratum_counts(cases)
    for i, case in enumerate(cases):
        seed = case['seed']
        record = run_episode(args.repo, model, case, clock, seed, args.collector_seconds, None)
        atomic_json(out / 'episodes' / f'{args.split}-{i:03d}.json', record)
        row = summarize(record)
        rows.append(row)
        report = dict(policy='deterministic_checkpoint', cases=rows, by_stratum=by_stratum(rows, expected),
                      **metrics(rows, len(cases)))
        if args.compare_always_hold:
            baseline = run_episode(args.repo, AlwaysHold(), case, clock, seed, args.collector_seconds, None)
            atomic_json(out / 'episodes' / f'{args.split}-{i:03d}-always-hold.json', baseline)
            hold = summarize(baseline)
            holds.append(hold)
            pairs.append(dict(case_id=case['case_id'], navigation_task=row['navigation_task'], fast_seed=seed,
                              geometry_sha256=geometry_hash(case),
                              checkpoint_minus_hold_remaining_m=row['remaining_m'] - hold['remaining_m'],
                              checkpoint_minus_hold_reward=row['reward'] - hold['reward']))
            report.update(always_hold=dict(cases=holds, by_stratum=by_stratum(holds, expected),
                                           **metrics(holds, len(cases))), paired_cases=pairs)
        atomic_json(out / 'evaluation.json', report)
        print(case['case_id'], record['end_reason'], flush=True)


def reject_training_overlap(checkpoint, cases):
    fields = (('training_maps', geometry_hash), ('training_sources', source_id),
              ('training_families', lambda c: c['terrain_family_id']))
    for key, identity in fields:
        if set(checkpoint.get(key, ())) & {identity(case) for case in cases}:
            raise ValueError(f'Evaluation routes overlap {key} for this checkpoint')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--repo', type=Path, default=REPO)
    shared.add_argument('--km-data', type=Path, help='Directory containing circle kilometre scenes.json')
    shared.add_argument('--field-data', type=Path, help='Directory containing audited detour/control scenes.json')
    shared.add_argument('--curriculum', choices=('lakes', 'mixed', 'circles'),
                        help='Train defaults to bilateral lakes; mixed selects historical bridge tasks. Evaluate uses checkpoint metadata.')
    shared.add_argument('--output', type=Path, required=True)
    shared.add_argument('--collector-seconds', type=int, help='Optional short rollout for integration checks')
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('train', parents=[shared])
    fit.add_argument('--updates', type=int, default=8)
    fit.add_argument('--episodes-per-update', type=int, help='Lake and bridge mixtures default to 4, circles to 3')
    fit.add_argument('--learning-rate', type=float, default=3e-4)
    fit.add_argument('--seed', type=int, default=43)
    fit.add_argument('--review-interval-s', type=int, default=15)
    fit.add_argument('--lookahead-m', type=float, default=200.)
    test = commands.add_parser('evaluate', parents=[shared])
    test.add_argument('--checkpoint', type=Path, required=True)
    test.add_argument('--split', choices=('development', 'test'), default='test')
    test.add_argument('--cases', nargs='+', help='Optional held-out case IDs')
    test.add_argument('--compare-always-hold', action='store_true', help='Pair with native hold on the same cases and seeds')
    args = parser.parse_args()
    if args.collector_seconds is not None and args.collector_seconds < 1:
        parser.error('--collector-seconds must be positive')
    args.repo = args.repo.resolve()
    training = args.command == 'train'
    if training:
        args.curriculum = args.curriculum or training_curriculum(args.field_data)
        if args.episodes_per_update is None:
            args.episodes_per_update = 3 if args.curriculum == 'circles' else 4
        if (min(args.updates, args.episodes_per_update, args.review_interval_s) < 1
                or not 0 < args.learning_rate <= 1 or not math.isfinite(args.lookahead_m) or args.lookahead_m <= 0):
            parser.error('Invalid PPO, review interval or lookahead setting')
        torch.manual_seed(args.seed)
        clock = dict(review_interval_s=args.review_interval_s, lookahead_m=args.lookahead_m)
    else:
        checkpoint_bytes = args.checkpoint.read_bytes()
        checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
        if (checkpoint.get('format') != FORMAT or checkpoint.get('v9_sha256') != CHECKPOINT_SHA
                or not isinstance(checkpoint.get('training_maps'), list)):
            parser.error('Expected a vtemp4 heading-policy checkpoint trained with the frozen V9 fleet')
        args.curriculum = args.curriculum or checkpoint.get('curriculum', 'circles')
        clock = checkpoint['clock']
    if args.curriculum not in ('lakes', 'mixed', 'circles') or args.curriculum == 'circles' and args.field_data is not None:
        parser.error('Field data requires a lake or bridge curriculum')
    args.field_geometry = {'lakes': LAKE_GEOMETRY, 'mixed': BRIDGE_GEOMETRY, 'circles': None}[args.curriculum]
    # Resolve the original data when it is available, including a custom training archive.
    # Explicit data and curriculum overrides remain available for independent evaluations.
    archived = {}
    if not training and args.curriculum == checkpoint.get('curriculum', 'circles'):
        archived = checkpoint.get('dataset', {})
    km = args.km_data or Path(archived.get('km_data') or args.repo / KM_DATA)
    field_default = args.repo / (LAKE_DATA if args.curriculum == 'lakes' else FIELD_DATA)
    field = args.field_data or Path(archived.get('field_data') or field_default)
    if args.km_data is None and not km.is_dir():
        km = args.repo / KM_DATA
    if args.field_data is None and not field.is_dir():
        field = field_default
    km = km.resolve()
    field = field.resolve() if args.field_geometry else None
    try:
        splits, hashes = load_curriculum(km, field, field_geometry=args.field_geometry or BRIDGE_GEOMETRY)
        if (not args.km_data and not args.field_data and archived.get('hashes') is not None
                and hashes != archived['hashes']):
            raise ValueError('Data fingerprints differ from the training checkpoint')
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    args.dataset = dict(km_data=str(km), field_data=str(field) if field else None, hashes=hashes)
    cases = splits['ppo_train' if training else args.split]
    if not training:
        if args.cases:
            by_id = {case['case_id']: case for case in cases}
            if len(set(args.cases)) != len(args.cases) or set(args.cases) - by_id.keys():
                parser.error('Expected unique held-out case IDs from the selected split')
            cases = [by_id[key] for key in args.cases]
        try:
            reject_training_overlap(checkpoint, cases)
        except ValueError as exc:
            parser.error(str(exc))
    runtime(args.repo)
    from vtemp4.vtemp4_policy import HeadingPolicy

    model = HeadingPolicy()
    if not training:
        model.load_state_dict(checkpoint['model'])
        model.eval().requires_grad_(False)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error('Use an empty output directory')
    (out / 'episodes').mkdir()
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = dict(format=FORMAT, options=options, clock=clock, gamma=1., cases=[c['case_id'] for c in cases],
                    stratum_counts=stratum_counts(cases), v9_sha256=CHECKPOINT_SHA, **hashes)
    if not training:
        manifest['checkpoint_sha256'] = checkpoint_sha
    atomic_json(out / 'manifest.json', manifest)
    if training:
        train(args, out, model, cases, clock)
    else:
        evaluate(args, out, model, cases, clock)


if __name__ == '__main__':
    main()
