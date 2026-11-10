"""Detached, logged installer for the explicitly authorized conda environment.

Wait for an existing torch install before taking over; never run two pip
writers against the environment. Closing the editor does not own this job.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'artifacts/world-model-gpu-setup'


def status(**fields):
    path = RUN / 'status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(pid=os.getpid(), updated_at=time.time(), **fields), indent=2) + '\n')
    temp.replace(path)


def installers():
    result = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode(errors='replace').strip('\0').split('\0')
            if len(args) >= 4 and args[1:4] == ['-m', 'pip', 'install']:
                if Path(args[0]).absolute() == Path(sys.executable).absolute():
                    result.append(int(path.parent.name))
        except (OSError, ValueError):
            continue
    return result


def torch_version():
    probe = subprocess.run([sys.executable, '-c', 'import torch; print(torch.__version__)'],
                           capture_output=True, text=True)
    return probe.stdout.strip() if probe.returncode == 0 else None


def run():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'install.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another installation supervisor already owns this job.', flush=True)
            return
        started = time.monotonic()
        while True:
            other = installers()
            if not other:
                break
            status(stage='waiting_for_existing_install', installer_pids=other)
            print(f'Waiting for existing pip installer: {other}', flush=True)
            if time.monotonic() - started > 7200:
                raise TimeoutError('Existing installer has not exited after two hours')
            time.sleep(10)
        if torch_version() != '2.14.1+cu130':
            status(stage='installing', python=sys.executable)
            command = [sys.executable, '-m', 'pip', 'install', '--timeout', '60', '--retries', '5',
                       'torch==2.14.1+cu130', '--index-url', 'https://download.pytorch.org/whl/cu130']
            subprocess.run(command, check=True)
        # A real allocation, matmul and synchronization, not just an import.
        status(stage='checking_cuda')
        probe = subprocess.run([sys.executable, '-c',
            'import json,torch; assert torch.cuda.is_available(); '
            'x=torch.ones((32,32),device="cuda"); y=x@x; torch.cuda.synchronize(); '
            'assert float(y[0,0])==32.; '
            'print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda, '
            'gpu=torch.cuda.get_device_name(0), matmul_ok=True)))'],
            capture_output=True, text=True, check=True)
        details = json.loads(probe.stdout.strip().splitlines()[-1])
        print(probe.stdout, flush=True)
        status(stage='complete', **details)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (RUN / 'install.log').open('a', buffering=1) as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        (RUN / 'supervisor.pid').write_text(str(process.pid) + '\n')
        print(json.dumps(dict(pid=process.pid, log=str(RUN / 'install.log'),
                              status=str(RUN / 'status.json'))), flush=True)
        return
    try:
        run()
    except Exception as exc:
        traceback.print_exc()
        status(stage='failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
