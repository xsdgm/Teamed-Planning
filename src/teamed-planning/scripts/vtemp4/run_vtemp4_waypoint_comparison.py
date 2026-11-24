"""Train three matched online navigation policies, then evaluate them on the same held-out missions."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vtemp4.run_vtemp4_formal import freeze, snapshot_package


REPO = Path(__file__).resolve().parents[4]
ARMS = ('heading', 'waypoint', 'waypoint_history')
STAGE = dict(seed=43, updates=25, episodes_per_update=16)
LOCAL_SEEDS = (198701, 198702)
FLEET_SEEDS = (20481001, 20481002, 20481003, 20481004)
LAKE_CASES = ('lake-dev-00-detour_left', 'lake-dev-01-detour_right')


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def run(repo, out, workers, resume):
    package = snapshot_package(out / 'sources')
    plan = dict(training=STAGE, arms=list(ARMS), workers_per_arm=workers,
                guidance=dict(radius_m=200., relative_to='vehicle_yaw_at_designation',
                              hold='fixed_world_coordinates_until_next_designation'),
                local_scene_seeds=list(LOCAL_SEEDS), lake_cases=list(LAKE_CASES), evaluation_fast_seeds=list(FLEET_SEEDS))
    if resume:
        if plan != json.loads((out / 'plan.json').read_text()) or not package.is_dir():
            raise ValueError('Comparison plan or frozen sources differ')
    else:
        if any(p.name != 'comparison.log' for p in out.iterdir()):
            raise FileExistsError(f'Use a fresh comparison directory or --resume: {out}')
        freeze(repo, out)
        package = snapshot_package(out / 'sources')
        write_json(out / 'plan.json', plan)
    processes = {}
    retries = set()
    completed = set()
    failed = {}

    def train(arm, restart):
        directory = out / arm
        logs = directory / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        snapshot = directory / 'sources'
        if not snapshot.exists():
            snapshot.symlink_to('../sources', target_is_directory=True)
        if snapshot.resolve() != package.parent.resolve():
            raise ValueError(f'Unexpected source snapshot for {arm}')
        args = [sys.executable, '-IBu', str(package / 'train_vtemp4_local.py'), '--repo', str(repo),
                '--output', str(directory), '--policy-variant', arm, '--workers', str(workers),
                '--episodes-per-update', str(STAGE['episodes_per_update']), '--updates', str(STAGE['updates']),
                '--seed', str(STAGE['seed'])]
        if restart:
            args.append('--resume')
        print(stamp(), arm, 'resuming' if restart else 'starting', flush=True)
        with (logs / 'train.log').open('a') as stream:
            return subprocess.Popen(args, cwd=repo, stdout=stream, stderr=subprocess.STDOUT)

    def status(phase):
        write_json(out / 'status.json', dict(phase=phase, time=stamp(), supervisor_pid=os.getpid(),
                   running={key: p.pid for key, p in processes.items()}, completed=sorted(completed), failed=failed))

    for arm in ARMS:
        processes[arm] = train(arm, resume and (out / arm / 'manifest.json').exists())
    status('training')
    while processes:
        for arm, process in list(processes.items()):
            code = process.poll()
            if code is None:
                continue
            del processes[arm]
            if code == 0:
                completed.add(arm)
            elif code > 0 and arm not in retries and (out / arm / 'manifest.json').is_file():
                retries.add(arm)
                processes[arm] = train(arm, True)
            else:
                failed[arm] = code
            status('training')
        if processes:
            time.sleep(10)
    if failed:
        status('failed')
        raise RuntimeError(f'Training failed: {failed}')
    status('evaluation')
    args = [sys.executable, '-IBu', str(package / 'evaluate_vtemp4_waypoints.py'), '--repo', str(repo),
            '--output', str(out / 'evaluation'), '--workers', str(min(6, len(FLEET_SEEDS) * len(ARMS))),
            '--scene-seeds', *map(str, LOCAL_SEEDS), '--lake-cases', *LAKE_CASES, '--fast-seeds', *map(str, FLEET_SEEDS)]
    for arm, flag in zip(ARMS, ('--heading-checkpoint', '--waypoint-checkpoint', '--history-checkpoint')):
        args.extend([flag, str(out / arm / 'checkpoints' / f"heading-update-{STAGE['updates']:04d}.pt")])
    with (out / 'evaluation.log').open('a') as log:
        eval_code = subprocess.call(args, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    if eval_code:
        failed['evaluation'] = eval_code
        status('failed')
        raise RuntimeError(f'Evaluation exited with code {eval_code}')
    status('complete')
    print(stamp(), 'Paired evaluation complete.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers-per-arm', type=int, default=6)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers_per_arm < 1:
        parser.error('Worker count must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as resources:
        fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
        resources.callback(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.repo.resolve(), out, args.workers_per_arm, args.resume)


if __name__ == '__main__':
    main()
