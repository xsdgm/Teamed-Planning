"""Run the 50-update mixed VTemp4 experiment and its paired development evaluations."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REPO = Path(__file__).resolve().parents[4]
STAGES = (10, 25, 50)


def freeze(repo, out):
    src = repo / 'src/teamed-planning/scripts'
    with tempfile.TemporaryDirectory(dir=out, prefix='.snapshot-') as tmp:
        tree = Path(tmp) / 'sources'
        pkg = tree / 'vtemp2_3'
        for family in ('shared', 'cadmm', 'mappo', 'vtemp', 'vtemp2_3', 'vtemp4', 'vtemp5', 'survey'):
            shutil.copytree(src / family, tree / family, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        backend = src / 'shared/simulator/wildscenes/wildscenes_map.py'
        shutil.copy2(backend, pkg / '_wildscenes_map_backend.py')
        tree.rename(out / 'sources')


def snapshot_package(sources):
    """Use the reorganized snapshot for new runs and the original layout on resume."""
    for candidate in (sources / 'vtemp4', sources / 'simulator' / 'vtemp4', sources / 'simulator', sources / 'world_model'):
        if (candidate / 'train_vtemp4_formal.py').is_file():
            return candidate
    return sources / 'world_model'


def launch(script, args, log, repo):
    with log.open('a', buffering=1) as stream:
        return subprocess.Popen([sys.executable, '-I', '-B', '-u', str(script), '--repo', str(repo), *map(str, args)],
                                cwd=repo, stdout=stream, stderr=subprocess.STDOUT)


def run(repo, out, resume):
    logs = out / 'logs'
    sources = out / 'sources'
    pkg = snapshot_package(sources)
    if resume:
        if not (out / 'manifest.json').is_file() or not pkg.is_dir():
            raise ValueError('Resume requires the original run manifest and source snapshot')
    else:
        if any(p != logs for p in out.iterdir()):
            raise FileExistsError(f'Use a new output directory, or --resume for {out}')
        freeze(repo, out)
        pkg = snapshot_package(sources)
    args = ['--output', out, '--workers', 16, '--episodes-per-update', 16, '--updates', 50, '--seed', 43,
            '--field-data', repo / 'artifacts/vtemp4-lake-curriculum-v1']
    train = launch(pkg / 'train_vtemp4_formal.py', args + (['--resume'] if resume else []), logs / 'train.log', repo)
    print(f'Training PID {train.pid}. Paired development evaluations queued for {STAGES}.', flush=True)
    pending = list(STAGES)
    development = out / 'development'
    evaluation = None
    train_retries = 0
    eval_retries = 0
    failed_stages = []
    while True:
        code = train.poll()
        if code not in (0, None):
            if code < 0 or not (out / 'manifest.json').is_file() or train_retries >= 1:
                if evaluation is not None:
                    evaluation.wait()
                raise RuntimeError(f'Trainer exited with code {code}. See {logs / "train.log"}')
            train_retries += 1
            print('Retrying from the latest committed training state.', flush=True)
            train = launch(pkg / 'train_vtemp4_formal.py', args + ['--resume'], logs / 'train.log', repo)
            code = None
        if evaluation is not None and evaluation.poll() is not None:
            if evaluation.returncode == 0:
                print(f'Completed development evaluation at update {pending[0]}.', flush=True)
                pending.pop(0)
                eval_retries = 0
            elif eval_retries < 1:
                eval_retries += 1
                print(f'Retrying development evaluation at update {pending[0]}.', flush=True)
            else:
                failed_stages.append(pending.pop(0))
                eval_retries = 0
            evaluation = None
        if pending and evaluation is None:
            stage = pending[0]
            ckpt = out / 'checkpoints' / f'heading-update-{stage:04d}.pt'
            if ckpt.exists():
                stage_eval = (pkg / 'vtemp4_stage_eval.py' if pkg.name == 'simulator'
                              else pkg / 'navigation/vtemp4_stage_eval.py')
                evaluation = launch(stage_eval,
                                    ['--checkpoint', ckpt, '--update', stage, '--output', development, '--workers', 2],
                                    logs / 'development.log', repo)
                print(f'Started paired development evaluation for update {stage}, PID {evaluation.pid}.', flush=True)
            elif code == 0:
                raise RuntimeError(f'Training ended without the scheduled checkpoint {ckpt}')
        if code == 0 and not pending:
            if failed_stages:
                raise RuntimeError(f'Development evaluation failed at updates {failed_stages}. See development.log')
            return
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    config = parser.parse_args()
    output = config.output.resolve()
    (output / 'logs').mkdir(parents=True, exist_ok=True)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    with (output / 'logs/supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(config.repo.resolve(), output, config.resume)
