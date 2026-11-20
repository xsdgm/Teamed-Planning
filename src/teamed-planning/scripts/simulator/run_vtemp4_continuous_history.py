"""Train the unified-attention policy on 16 physical cores and evaluate held-out local maps."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.run_vtemp4_formal import REPO, freeze, launch, snapshot_package


WORKERS = 16
VARIANT = 'periodic_unified_attention_goal'


def select_cores():
    selected = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        physical = tuple((topology / field).read_text().strip() for field in ('physical_package_id', 'core_id'))
        selected.setdefault(physical, cpu)
        if len(selected) == WORKERS:
            break
    if len(selected) != WORKERS:
        raise RuntimeError(f'Only {len(selected)} distinct physical cores are available, need {WORKERS}')
    cpus = list(selected.values())
    os.sched_setaffinity(0, cpus)
    return cpus


def run(repo, out, resume):
    logs = out / 'logs'
    package = snapshot_package(out / 'sources')
    if resume:
        if not package.is_dir() or not (out / 'manifest.json').is_file():
            raise ValueError('Resume requires the frozen sources and training manifest')
    else:
        if any(p != logs for p in out.iterdir()):
            raise FileExistsError(f'Expected a fresh training directory: {out}')
        freeze(repo, out)
        package = snapshot_package(out / 'sources')
    cpus = select_cores()

    def status(phase, **extra):
        tmp = logs / 'status.tmp'
        tmp.write_text(json.dumps(dict(phase=phase, policy_variant=VARIANT, supervisor_pid=os.getpid(),
                                       workers=WORKERS, cpu_ids=cpus, time=datetime.now(timezone.utc).isoformat(),
                                       **extra), indent=2) + '\n', encoding='utf-8')
        tmp.replace(logs / 'status.json')

    train_args = ['--output', out, '--policy-variant', VARIANT, '--workers', WORKERS,
                  '--updates', 25, '--episodes-per-update', 16, '--seed', 43]
    final = out / 'checkpoints/heading-update-0025.pt'
    for phase, script, args in (
            ('training', 'train_vtemp4_local.py', train_args),
            ('evaluation', 'evaluate_vtemp4_continuous_history.py',
             ['--output', out / 'evaluation', '--checkpoint', final, '--policy-variant', VARIANT, '--lake-cases'])):
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
    args = parser.parse_args()
    out = args.output.resolve()
    logs = out / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.repo.resolve(), out, args.resume)
