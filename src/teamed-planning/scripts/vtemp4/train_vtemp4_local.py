"""Train a heading or fixed-waypoint policy on the same online kilometre-scale V9 missions."""
import argparse
from concurrent.futures import as_completed
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))

from vtemp2_3.frozen_navigation import CHECKPOINT_SHA, V9_RUN, runtime
from vtemp2_3.hierarchical_sources import atomic_json, file_sha256
from vtemp2_3.train_slow_policy import atomic_checkpoint
from vtemp4.train_vtemp4 import by_stratum
from vtemp4.train_vtemp4_formal import (
    CLOCK, FORMAL_FORMAT, checkpoint_path, exclusive_run, load_commits, publish_latest,
    restore_rng, rng_state, source_digest, source_tree_digest,
)
from vtemp4.vtemp4_curriculum import geometry_hash, source_id, stratum_counts
from vtemp4.vtemp4_local_curriculum import (
    CLEARANCE_M, CORRIDOR_HALF_WIDTH_M, DISTANCE_BAND_M, GENERATOR_SHA, WORLD_M, local_case,
)
from vtemp4.vtemp4_parallel import collect_episode, episode_pool, load_rollout
from vtemp4.vtemp4_policy import HeadingPolicy, update_policy
from vtemp4.vtemp4_ring_contract import (
    GOAL_VARIANTS, LSTM_POLICY_FORMATS, PERIODIC_ACTION_CONTRACTS, RING_POLICY_FORMATS, RING_WAYPOINT_FORMAT,
    PROGRESS_DISTANCES, new_run_settings, new_run_waypoint_bonus, periodic_policy_types, ring_action_contract,
    validate_progress_distance,
)
from vtemp4.vtemp4_runtime import DEFAULT_PROGRESS_REWARD_WEIGHT
from vtemp4.vtemp4_ring_history import ContinuousHistoryPolicy
from vtemp4.vtemp4_waypoint_policy import RingWaypointPolicy, update_waypoint_policy


CURRICULUM = 'online_v9_local_obstacles'


def manifest_for(args):
    fast = args.repo / V9_RUN
    if file_sha256(fast / 'best.pt') != CHECKPOINT_SHA:
        raise ValueError('Frozen V9 checkpoint differs')
    count = args.updates * args.episodes_per_update
    seeds = random.Random(args.seed).sample(range(2**31), 2 * count)
    variant = getattr(args, 'policy_variant', None) or 'heading'
    fmt = RING_POLICY_FORMATS[variant]
    interval = getattr(args, 'review_interval_s', None)
    weight = getattr(args, 'progress_reward_weight', None)
    bonus_ratio = getattr(args, 'waypoint_progress_bonus_ratio', None)
    distance = getattr(args, 'progress_distance', None)
    validate_progress_distance(variant, distance)
    if not getattr(args, 'resume', False):
        interval, weight = new_run_settings(variant, interval, weight)
        bonus_ratio = new_run_waypoint_bonus(variant, bonus_ratio, distance)
    if interval not in (15, 30) or (interval == 30 and variant not in GOAL_VARIANTS):
        raise ValueError('The 30-second review interval requires a goal-aware policy')
    clock = dict(CLOCK, review_interval_s=interval)
    if weight is not None and (variant not in GOAL_VARIANTS or not math.isfinite(weight) or weight <= 0):
        raise ValueError('Specify a positive finite progress reward weight for a goal-aware policy')
    if bonus_ratio is not None and (not math.isfinite(bonus_ratio) or bonus_ratio < 0
                                    or (bonus_ratio > 0 and variant not in LSTM_POLICY_FORMATS)):
        raise ValueError('A waypoint progress bonus requires a nonnegative finite ratio and the vehicle-graph LSTM')
    manifest = dict(
        format=FORMAL_FORMAT, policy_format=fmt, action_contract=ring_action_contract(fmt, review_interval_s=interval),
        policy_variant=variant, curriculum=CURRICULUM,
        training_stage=1, seed=args.seed, updates=args.updates, total_episodes=count,
        episodes_per_update=args.episodes_per_update, learning_rate=args.learning_rate,
        clock=clock, gamma=1., collector_seconds=None, field_data=None,
        sampling=dict(stage='local_obstacles_only', split='train', generation='online_per_episode',
                      terrain='synthetic_v9_circles', world_size_m=WORLD_M,
                      distance_band_m=list(DISTANCE_BAND_M), straight_line_must_be_blocked=True,
                      certified_corridor_half_width_m=CORRIDOR_HALF_WIDTH_M,
                      required_ground_clearance_m=CLEARANCE_M, generator_sha256=GENERATOR_SHA),
        v9_sha256=CHECKPOINT_SHA, v9_config_sha256=file_sha256(fast / 'config.json'),
        v9_sources_sha256=source_digest(fast / 'runtime_sources'), world_model_sha256=source_tree_digest(SOURCE.parent),
        python_version=sys.version,
        libraries={name: version(name) for name in ('torch', 'numpy', 'scipy', 'networkx', 'scikit-image')},
        plan=[dict(scene_seed=seeds[i], fast_seed=seeds[count+i]) for i in range(count)],
    )
    if weight is not None:
        manifest['progress_reward_weight'] = weight
    if distance is not None:
        manifest['progress_distance'] = distance
    if bonus_ratio or (distance == 'traversable' and bonus_ratio is not None):
        manifest['waypoint_progress_bonus_ratio'] = bonus_ratio
    return manifest


def collect_local(repo, path, digest, specification, folder, geometry_dir):
    """Generate the geometry inside the episode worker, then collect a complete rollout."""
    case = local_case(repo, specification['scene_seed'], split='train', geometry_dir=geometry_dir)
    row = collect_episode(repo, path, digest, case, specification['fast_seed'], folder)
    if (row['case_id'] != case['case_id'] or row['geometry_sha256'] != geometry_hash(case)
            or row['fast_seed'] != specification['fast_seed']
            or row['end_reason'] not in {'arrival', 'time_budget', 'safety_failure'}):
        raise ValueError(f'Unexpected online rollout at {folder}')
    return dict(row, scene_seed=specification['scene_seed'], source_id=source_id(case))


def run(args):
    out = args.output
    manifest_file = out / 'manifest.json'
    if args.resume:
        prior = json.loads(manifest_file.read_text())
        if getattr(args, 'policy_variant', None) is None:
            args.policy_variant = prior['policy_variant']
        if getattr(args, 'review_interval_s', None) is None:
            args.review_interval_s = prior['clock']['review_interval_s']
        if getattr(args, 'progress_reward_weight', None) is None:
            args.progress_reward_weight = prior.get('progress_reward_weight')
        elif (prior['policy_variant'] in GOAL_VARIANTS and 'progress_reward_weight' not in prior
              and args.progress_reward_weight == DEFAULT_PROGRESS_REWARD_WEIGHT):
            args.progress_reward_weight = None  # Early manifests left the weight of 2 implicit.
        if getattr(args, 'waypoint_progress_bonus_ratio', None) is None:
            args.waypoint_progress_bonus_ratio = prior.get('waypoint_progress_bonus_ratio')
        if getattr(args, 'progress_distance', None) is None or (
                args.progress_distance == 'euclidean' and 'progress_distance' not in prior):
            args.progress_distance = prior.get('progress_distance')
    manifest = manifest_for(args)
    if args.resume and prior != manifest:
        raise ValueError('Resume configuration, random schedule or runtime differs from manifest')
    if not args.resume:
        if any(item.name not in {'logs', 'sources'} for item in out.iterdir()):
            raise FileExistsError(f'Use a fresh training directory or --resume: {out}')
        atomic_json(manifest_file, manifest)
    manifest_sha = file_sha256(manifest_file)
    geometry_dir = out / 'geometry'
    geometry_dir.mkdir(exist_ok=True)
    (out / 'checkpoints').mkdir(exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    runtime(args.repo)
    variant = manifest['policy_variant']
    waypoint = manifest['policy_format'] == RING_WAYPOINT_FORMAT
    periodic = manifest['action_contract'] in PERIODIC_ACTION_CONTRACTS
    if waypoint:
        model = RingWaypointPolicy(use_history=variant == 'waypoint_history')
    elif variant == 'continuous_heading_history':
        model = ContinuousHistoryPolicy()
    elif periodic:
        policy_type, _ = periodic_policy_types(manifest['policy_format'])
        model = policy_type()
    else:
        model = HeadingPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history, state, digest = load_commits(out, manifest_sha)
    seen = {key: set() for key in ('training_maps', 'training_sources', 'training_families')}
    reward_settings = {k: manifest[k] for k in ('progress_reward_weight', 'waypoint_progress_bonus_ratio', 'progress_distance')
                       if k in manifest}

    def save(update, previous_digest, record):
        path = checkpoint_path(out, update)
        if path.exists():
            raise FileExistsError(path)
        atomic_checkpoint(path, dict(
            format=manifest['policy_format'], action_contract=manifest['action_contract'], policy_variant=variant,
            curriculum=CURRICULUM, training_stage=1, clock=manifest['clock'],
            completed_updates=update, v9_sha256=CHECKPOINT_SHA, formal_format=FORMAL_FORMAT,
            manifest_sha256=manifest_sha, previous_checkpoint_sha256=previous_digest,
            **reward_settings,
            dataset={'generator': 'vtemp4_local_curriculum.local_case', 'split': 'train'},
            model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(),
            update_record=record, **{key: sorted(values) for key, values in seen.items()},
        ))
        return file_sha256(path)

    if state is None:
        digest = save(0, None, None)
    else:
        if (state['format'] != manifest['policy_format'] or state.get('policy_variant', 'heading') != variant
                or state['clock'] != manifest['clock'] or state['action_contract'] != manifest['action_contract']):
            raise ValueError('Checkpoint and training manifest disagree on the policy, navigation clock or action contract')
        if state.get('waypoint_progress_bonus_ratio', 0.) != manifest.get('waypoint_progress_bonus_ratio', 0.):
            raise ValueError('Checkpoint and training manifest disagree on the waypoint progress bonus')
        if (state.get('progress_distance', 'euclidean') != manifest.get('progress_distance', 'euclidean')
                or state.get('progress_reward_weight', DEFAULT_PROGRESS_REWARD_WEIGHT)
                != manifest.get('progress_reward_weight', DEFAULT_PROGRESS_REWARD_WEIGHT)):
            raise ValueError('Checkpoint and training manifest disagree on progress distance or weight')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'])
        seen = {key: set(state[key]) for key in seen}
    completed = len(history)
    saved_history = out / 'training.json'
    if saved_history.exists():
        existing = json.loads(saved_history.read_text())
        if history[:len(existing)] != existing:
            raise ValueError('Published training history differs from committed checkpoints')
    atomic_json(saved_history, history)
    publish_latest(out, checkpoint_path(out, completed))
    batch_size = args.episodes_per_update
    status = dict(curriculum=CURRICULUM, policy_variant=variant, pid=os.getpid(), workers=args.workers,
                  completed_updates=completed, total_updates=args.updates, episodes_per_update=batch_size,
                  committed_episodes=completed * batch_size, total_episodes=manifest['total_episodes'])

    def report(phase, **extra):
        atomic_json(out / 'progress.json', dict(status, phase=phase, **extra))

    try:
        with episode_pool(args.workers) as pool:
            for update in range(completed + 1, args.updates + 1):
                start = (update - 1) * batch_size
                batch = manifest['plan'][start:start + batch_size]
                source = checkpoint_path(out, update - 1)
                locations = [out / 'episodes' / f'update-{update:04d}' / f'episode-{slot:02d}'
                             for slot in range(batch_size)]
                report('sampling', current_update=update, completed_episodes=0)
                print(f'Update {update}/{args.updates}: {len(batch)} fresh online missions', flush=True)
                jobs = {pool.submit(collect_local, args.repo, source, digest, entry, folder, geometry_dir): slot
                        for slot, (entry, folder) in enumerate(zip(batch, locations))}
                rows = [None] * batch_size
                for count, job in enumerate(as_completed(jobs), 1):
                    rows[jobs[job]] = job.result()
                    report('sampling', current_update=update, completed_episodes=count)
                hashes = {row['geometry_sha256'] for row in rows}
                if len(hashes) != batch_size or hashes.intersection(seen['training_maps']):
                    raise ValueError('Online mission generator repeated a training map')
                report('optimizing', current_update=update, completed_episodes=batch_size)
                optimize = update_waypoint_policy if waypoint else update_policy
                metrics = optimize(model.train(), optimizer, [load_rollout(folder) for folder in locations])
                if variant == 'heading':
                    with torch.no_grad():
                        metrics['max_possible_turn_logit'] = float(model.actor.bias[0] + model.actor.weight[0].abs().sum())
                seen['training_maps'].update(hashes)
                seen['training_sources'].update(row['source_id'] for row in rows)
                seen['training_families'].update(row['terrain_family_id'] for row in rows)
                if waypoint or periodic:
                    count_key = 'waypoint_replacements' if waypoint else 'waypoint_designations'
                    summary = {'random_circles': dict(completed_missions=len(rows),
                               arrivals=sum(row['goal_reached'] for row in rows),
                               mean_net_progress_m=float(np.mean([row['direct_distance_m'] - row['remaining_m'] for row in rows])),
                               **{count_key: sum(row[count_key] for row in rows)})}
                else:
                    summary = by_stratum(rows, stratum_counts(rows))
                record = dict(update=update, episodes=rows, by_stratum=summary, ppo=metrics)
                digest = save(update, digest, record)
                publish_latest(out, checkpoint_path(out, update))
                history.append(record)
                atomic_json(saved_history, history)
                status.update(completed_updates=update, committed_episodes=update * batch_size)
                report('committed', current_update=update, checkpoint_sha256=digest)
                print(f"Committed update {update}: {metrics['transitions']} slow decisions", flush=True)
        report('complete', checkpoint_sha256=digest)
    except BaseException as exc:
        report('interrupted', error=f'{type(exc).__name__}: {exc}')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=SOURCE.parents[3])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=25)
    parser.add_argument('--episodes-per-update', type=int, default=16)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--policy-variant', choices=tuple(RING_POLICY_FORMATS),
                        help='Defaults to heading for a new run, or to the saved variant on resume')
    parser.add_argument('--review-interval-s', type=int, choices=(15, 30),
                        help='Designation interval in simulation seconds (new vehicle-graph LSTM: 30, others: 15)')
    parser.add_argument('--progress-reward-weight', type=float,
                        help='Weight on normalized goal progress (new vehicle-graph LSTM: 10, other goal policies: 2)')
    parser.add_argument('--progress-distance', choices=PROGRESS_DISTANCES,
                        help='Goal-residual vehicle progress metric (default euclidean, or the saved metric on resume)')
    parser.add_argument('--waypoint-progress-bonus-ratio', type=float,
                        help='Maximum waypoint bonus fraction (default 0.10 for euclidean goal-residual, otherwise 0)')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (min(args.updates, args.episodes_per_update, args.workers) < 1 or not 0 <= args.seed < 2**32
            or 2 * args.updates * args.episodes_per_update > 2**31
            or not 0 < args.learning_rate <= 1 or not math.isfinite(args.learning_rate)):
        parser.error('Expected valid counts, random seed, and learning rate')
    args.repo = args.repo.resolve()
    args.output = args.output.resolve()
    with exclusive_run(args.output):
        run(args)


if __name__ == '__main__':
    main()
