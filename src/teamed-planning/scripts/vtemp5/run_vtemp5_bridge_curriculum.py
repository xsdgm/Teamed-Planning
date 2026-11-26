"""Start bridge/lake PPO after a completed six-map circle evaluation passes."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vtemp2_3.hierarchical_sources import atomic_json, file_sha256
from vtemp5.circle_mastery import assess


def supervise(args):
    circle, bridge = args.stage1.resolve(), args.stage2.resolve()
    logs = bridge / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    manifest = circle / 'manifest.json'
    config = json.loads(manifest.read_text())
    expected = args.stage1_manifest_sha256
    if (file_sha256(manifest) != expected or config.get('curriculum') != 'circle'
            or len(config['evaluation']) != 6 or circle == bridge):
        raise ValueError('Expected the selected six-map circle experiment and a separate bridge directory')
    script = logs / 'sources/vtemp5/run_vtemp5_fleet.py'
    if not script.is_file():
        raise ValueError(f'Missing frozen fleet trainer: {script}')
    checkpoint = circle / 'checkpoints' / f"heading-{config['updates']:04d}.pt"
    training_cpus = sorted(os.sched_getaffinity(0))
    if set(training_cpus) & set(args.baseline_cpus):
        raise ValueError('Training CPUs overlap with the specified baseline CPUs')
    argv = [
        sys.executable, '-I', '-B', '-u', str(script), '--repo', config['repo'],
        '--output', str(bridge), '--init-checkpoint', str(checkpoint),
        '--world-model', config['world_model'], '--curriculum', 'bridge-lake',
        '--workers', config['workers'], '--episodes-per-update', config['episodes_per_update'],
        '--updates', config['updates'], '--eval-episodes', len(config['evaluation']),
        '--seed', args.seed, '--eval-seed', args.eval_seed, '--smoke-seconds', config['smoke_s'],
        '--epochs', config['epochs'], '--microbatch-size', config['microbatch'],
        '--learning-rate', config['learning_rate'], '--budget-seconds-per-metre', config['seconds_per_m'],
        '--baseline-workers', len(args.baseline_cpus), '--baseline-cpus', *args.baseline_cpus,
    ]
    plan = dict(stage1=str(circle), stage1_manifest_sha256=expected, stage2=str(bridge),
                required_arrivals_beyond_obstacle=4, heldout_maps=6, allowed_collisions=0,
                on_failed_gate='stop', training_cpus=training_cpus, baseline_cpus=args.baseline_cpus,
                argv=list(map(str, argv)))
    state = logs / 'curriculum_status.json'

    def report(phase, **values):
        atomic_json(state, dict(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(),
                                supervisor_pid=os.getpid(), **values))

    with (logs / 'curriculum.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_file = logs / 'curriculum_plan.json'
        if not plan_file.exists() and any(p.name != 'logs' for p in bridge.iterdir()):
            raise ValueError('New curriculum needs a fresh bridge output directory')
        if plan_file.is_file() and plan != json.loads(plan_file.read_text()):
            raise ValueError('This curriculum directory already has a different plan')
        atomic_json(plan_file, plan)
        if state.is_file() and json.loads(state.read_text())['phase'] in ('stopped_at_circle', 'complete'):
            return
        try:
            report('waiting_for_circle_evaluation')
            while True:
                progress = circle / 'progress.json'
                if progress.is_file() and json.loads(progress.read_text()).get('phase') == 'complete':
                    break
                time.sleep(15)
            if file_sha256(manifest) != expected:
                raise ValueError('The first-stage manifest has changed')
            decision = assess(circle)
            atomic_json(logs / 'circle_assessment.json', decision)
            if not decision['passed']:
                report('stopped_at_circle', gate=decision)
                return
            if Path(decision['checkpoint']) != checkpoint or file_sha256(checkpoint) != decision['checkpoint_sha256']:
                raise ValueError('The bridge warm-start differs from the evaluated checkpoint')
            bridge_progress = bridge / 'progress.json'
            if bridge_progress.is_file() and json.loads(bridge_progress.read_text()).get('phase') == 'complete':
                existing = json.loads((bridge / 'manifest.json').read_text())
                if existing.get('actor_sha256') != decision['checkpoint_sha256']:
                    raise ValueError('Completed bridge run used a different initial actor')
                report('complete', gate=decision)
                return
            command = plan['argv'] + (['--resume'] if (bridge / 'manifest.json').is_file() else [])
            with (logs / 'training.log').open('a') as stream:
                child = subprocess.Popen(command, cwd=config['repo'], stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
                report('bridge_training', gate=decision, trainer_pid=child.pid)
                atomic_json(logs / 'launch.json', dict(pid=child.pid, argv=command, cpu_affinity=training_cpus,
                            baseline_cpus=args.baseline_cpus, actor_sha256=decision['checkpoint_sha256'],
                            started_at=datetime.now(timezone.utc).isoformat()))
                rc = child.wait()
            if rc or json.loads(bridge_progress.read_text()).get('phase') != 'complete':
                raise RuntimeError(f'Bridge trainer did not complete successfully (exit {rc})')
            report('complete', gate=decision)
        except Exception as exc:
            report('error', detail=str(exc))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1', type=Path, required=True)
    parser.add_argument('--stage2', type=Path, required=True)
    parser.add_argument('--stage1-manifest-sha256', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--eval-seed', type=int, required=True)
    parser.add_argument('--baseline-cpus', type=int, nargs='+', required=True)
    supervise(parser.parse_args())
