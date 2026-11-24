"""Train a goal-aware waypoint policy on physical cores and evaluate held-out local maps."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vtemp4.run_vtemp4_formal import REPO, freeze, launch, snapshot_package
from vtemp4.vtemp4_ring_contract import (
    GOAL_VARIANTS, LSTM_POLICY_FORMATS, VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_VARIANT,
    PROGRESS_DISTANCES, new_run_settings, new_run_waypoint_bonus, validate_progress_distance,
)


WORKERS = 16


def select_cores(workers=WORKERS):
    if workers < 1:
        raise ValueError('A positive number of workers is required')
    selected = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        physical = tuple((topology / field).read_text().strip() for field in ('physical_package_id', 'core_id'))
        selected.setdefault(physical, cpu)
        if len(selected) == workers:
            break
    if len(selected) != workers:
        raise RuntimeError(f'Only {len(selected)} distinct physical cores are available, need {workers}')
    cpus = list(selected.values())
    os.sched_setaffinity(0, cpus)
    return cpus


def run(repo, out, resume, review_interval_s=None, progress_reward_weight=None, policy_variant=None,
        waypoint_progress_bonus_ratio=None, workers=WORKERS, progress_distance=None):
    logs = out / 'logs'
    package = snapshot_package(out / 'sources')
    if resume:
        if not package.is_dir() or not (out / 'manifest.json').is_file():
            raise ValueError('Resume requires the frozen sources and training manifest')
        manifest = json.loads((out / 'manifest.json').read_text())
        archived_variant = manifest['policy_variant']
        if policy_variant is not None and policy_variant != archived_variant:
            raise ValueError(f'Requested policy variant differs from the saved {archived_variant} variant')
        policy_variant = archived_variant
        archived_interval = manifest['clock']['review_interval_s']
        if review_interval_s is not None and review_interval_s != archived_interval:
            raise ValueError(f'Requested interval differs from the saved {archived_interval}-second interval')
        review_interval_s = archived_interval
        if progress_reward_weight is not None and progress_reward_weight != manifest.get('progress_reward_weight', 2.):
            raise ValueError('Requested progress reward weight differs from the saved run')
        progress_reward_weight = manifest.get('progress_reward_weight')
        if (waypoint_progress_bonus_ratio is not None and
                waypoint_progress_bonus_ratio != manifest.get('waypoint_progress_bonus_ratio', 0.)):
            raise ValueError('Requested waypoint progress bonus differs from the saved run')
        waypoint_progress_bonus_ratio = manifest.get('waypoint_progress_bonus_ratio')
        if progress_distance is not None and progress_distance != manifest.get('progress_distance', 'euclidean'):
            raise ValueError('Requested progress distance differs from the saved run')
        progress_distance = manifest.get('progress_distance')
    else:
        if policy_variant is None:
            policy_variant = VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_VARIANT
        if any(p != logs for p in out.iterdir()):
            raise FileExistsError(f'Expected a fresh training directory: {out}')
        validate_progress_distance(policy_variant, progress_distance)
        freeze(repo, out)
        package = snapshot_package(out / 'sources')
        review_interval_s, progress_reward_weight = new_run_settings(
            policy_variant, review_interval_s, progress_reward_weight)
        waypoint_progress_bonus_ratio = new_run_waypoint_bonus(policy_variant, waypoint_progress_bonus_ratio, progress_distance)
    cpus = select_cores(workers)

    def status(phase, **extra):
        tmp = logs / 'status.tmp'
        tmp.write_text(json.dumps(dict(phase=phase, policy_variant=policy_variant, review_interval_s=review_interval_s,
                                       waypoint_progress_bonus_ratio=waypoint_progress_bonus_ratio or 0.,
                                       progress_distance=progress_distance or 'euclidean',
                                       supervisor_pid=os.getpid(),
                                       workers=workers, cpu_ids=cpus, time=datetime.now(timezone.utc).isoformat(),
                                       **extra), indent=2) + '\n', encoding='utf-8')
        tmp.replace(logs / 'status.json')

    train_args = ['--output', out, '--policy-variant', policy_variant, '--workers', workers,
                  '--updates', 25, '--episodes-per-update', 16, '--seed', 43]
    # Historical 15 s snapshots have the same default and predate this option.
    if review_interval_s != 15 or policy_variant in LSTM_POLICY_FORMATS:
        train_args += ['--review-interval-s', review_interval_s]
    if progress_reward_weight is not None:
        train_args += ['--progress-reward-weight', progress_reward_weight]
    if progress_distance is not None:
        train_args += ['--progress-distance', progress_distance]
    # Older snapshots predate the option. A fresh residual run needs an explicit zero to disable its default.
    if waypoint_progress_bonus_ratio or (not resume and policy_variant == VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_VARIANT):
        train_args += ['--waypoint-progress-bonus-ratio', waypoint_progress_bonus_ratio]
    final = out / 'checkpoints/heading-update-0025.pt'
    eval_args = ['--output', out / 'evaluation', '--checkpoint', final, '--policy-variant', policy_variant,
                 '--lake-cases']
    if progress_distance is not None:
        eval_args += ['--progress-distance', progress_distance]
    if workers < 4:  # The evaluator defaults to four workers.
        eval_args += ['--workers', workers]
    for phase, script, args in (
            ('training', 'train_vtemp4_local.py', train_args),
            ('evaluation', 'evaluate_vtemp4_continuous_history.py', eval_args)):
        for attempt in range(2):
            restart = phase == 'training' and (resume or attempt)
            process = launch(package / script, args + (['--resume'] if restart else []), logs / f'{phase}.log', repo)
            status(phase, child_pid=process.pid, attempt=attempt + 1)
            code = process.wait()
            if code == 0:
                break
            if code < 0 or attempt or (phase == 'training' and not (out / 'manifest.json').is_file()):
                status('failed', failed_phase=phase, returncode=code)
                raise RuntimeError(f'{phase} exited with code {code}; see {logs / (phase + ".log")}')
    status('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=WORKERS, help='Physical CPU cores for training (default 16)')
    parser.add_argument('--policy-variant', choices=GOAL_VARIANTS,
                        help=f'Upper policy for a new run (default {VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_VARIANT}) or the saved variant on resume')
    parser.add_argument('--review-interval-s', type=int, choices=(15, 30),
                        help='Designation interval in simulation seconds (new vehicle-graph LSTM: 30, others: 15)')
    parser.add_argument('--progress-reward-weight', type=float,
                        help='Weight on normalized goal progress (new vehicle-graph LSTM: 10, other goal policies: 2)')
    parser.add_argument('--progress-distance', choices=PROGRESS_DISTANCES,
                        help='Goal-residual vehicle progress metric (default euclidean, or the saved metric on resume)')
    parser.add_argument('--waypoint-progress-bonus-ratio', type=float,
                        help='Maximum waypoint bonus fraction (default 0.10 for euclidean goal-residual, otherwise 0)')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('Expected at least one worker')
    out = args.output.resolve()
    logs = out / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.repo.resolve(), out, args.resume, args.review_interval_s, args.progress_reward_weight,
            args.policy_variant, args.waypoint_progress_bonus_ratio, args.workers, args.progress_distance)
