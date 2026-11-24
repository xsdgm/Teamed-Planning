"""Wait for audited kilometre geometry, launch once, then monitor training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))
from vtemp2_3.hierarchical_sources import atomic_json as write_json, source_paths, source_package
from vtemp2_3.scene_manifest import file_sha256

READY = 'verified_kilometer_tasks_available'
REQUIRED_SOURCES = {'train_hierarchical.py', 'scene_manifest.py', 'wildscenes_tasks.py',
                    'wildscenes_runtime.py', '_wildscenes_map_backend.py', 'wildscenes_workflow.py'}
INTEGER_OPTIONS = {'warmup_updates', 'fast_updates', 'slow_updates', 'joint_updates', 'workers',
    'episodes_per_update', 'steps', 'option_steps', 'stride', 'seed', 'jitter', 'hidden', 'horizon',
    'wm_epochs', 'wm_batch_size', 'distill_epochs', 'ppo_epochs', 'ppo_minibatch', 'slow_epochs',
    'slow_bc_epochs', 'slow_bc_batch_size', 'eval_every', 'eval_scenes', 'replay_episodes'}
OTHER_OPTIONS = {'device', 'gpu_memory_fraction', 'loss', 'random_slow_probability',
                 'mission_distances_m', 'warmup_distances_m', 'complete_missions',
                 'freeze_fast_world', 'validation_split'}


class WorkflowBlocked(ValueError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_assets(plan):
    """Recheck immutable inputs before readiness polling and immediately before launch."""
    for key in ('repo', 'workflow_dir', 'preparation_dir', 'source_dir', 'graph_init',
                'slow_init', 'readiness_verification'):
        if not isinstance(plan.get(key), str) or not Path(plan[key]).is_absolute():
            raise WorkflowBlocked(key + ' must be an absolute path')
    source = Path(plan['source_dir'])
    expected = plan['source_sha256']
    files = source_paths(source)
    required = (REQUIRED_SOURCES - {'train_hierarchical.py'}) | {
        'train_graph.py' if source_package(source) == 'vtemp2' else 'train_hierarchical.py'}
    if not isinstance(expected, dict) or files.keys() != expected.keys() or not required <= files.keys():
        raise WorkflowBlocked('Frozen Python source inventory changed or is incomplete')
    for name, sha in expected.items():
        if file_sha256(files[name]) != sha:
            raise WorkflowBlocked('Frozen source SHA-256 changed: ' + name)
    for key in ('graph_init', 'slow_init', 'readiness_verification'):
        if file_sha256(plan[key]) != plan[key + '_sha256']:
            raise WorkflowBlocked(key + ' SHA-256 changed')
    readiness = read_json(plan['readiness_verification'])
    if (readiness.get('smoke_complete') is not True or readiness.get('all_models_updated') is not True
            or type(readiness.get('tests_passed')) is not int or readiness['tests_passed'] < 1):
        raise WorkflowBlocked('Readiness must prove completed smoke, all-model updates and passing tests')
    config = plan['training_config']
    if not isinstance(config, dict) or set(config) - INTEGER_OPTIONS - OTHER_OPTIONS:
        raise WorkflowBlocked('Training config contains unsupported command options')
    if config.get('complete_missions') is not True or config.get('freeze_fast_world', False) is not False:
        raise WorkflowBlocked('Formal workflow requires complete missions and unfrozen fast/world updates')
    if config.get('validation_split', 'heldout_tasks') != 'heldout_tasks':
        raise WorkflowBlocked('This workflow explicitly uses heldout_tasks validation')
    for key in ('warmup_updates', 'fast_updates', 'slow_updates', 'joint_updates', 'seed'):
        if type(config.get(key)) is not int or config[key] < 0:
            raise WorkflowBlocked(key + ' must be a nonnegative integer')
    for key in ('episodes_per_update', 'eval_scenes', 'workers'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise WorkflowBlocked(key + ' must be a positive integer')
    if (config['warmup_updates'] + config['joint_updates'] == 0
            or config['fast_updates'] + config['joint_updates'] == 0
            or config['slow_updates'] + config['joint_updates'] == 0):
        raise WorkflowBlocked('Training stages must update world, fast and slow models')
    for key in ('warmup_distances_m', 'mission_distances_m'):
        values = config.get(key)
        if (not isinstance(values, list) or len(values) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
                or not 0 < values[0] <= values[1]):
            raise WorkflowBlocked('Invalid explicit distance range: ' + key)
    if config['mission_distances_m'][0] < 1000.:
        raise WorkflowBlocked('Formal kilometre missions cannot be shortened below 1000m')
    if any(a > b for a, b in zip(config['warmup_distances_m'], config['mission_distances_m'])):
        raise WorkflowBlocked('Warmup range exceeds the formal mission range')


def preparation_ready(plan):
    path = Path(plan['preparation_dir']) / 'preparation-status.json'
    if not path.exists():
        return None, dict(stage='status_not_created')
    status = read_json(path)
    state = status.get('stage', status.get('state', status.get('status')))
    if state == 'failed':
        raise WorkflowBlocked('Map preparation failed: ' + str(status.get('error', status)))
    if state != 'complete':
        return None, status
    audit_path = Path(status.get('audit_path', Path(plan['preparation_dir']) / 'K-03-kilometer-audit.json'))
    if read_json(audit_path).get('status') != READY:
        raise WorkflowBlocked('Completed map audit has no verified kilometre tasks; distances remain unchanged')
    return audit_path.resolve(), status


def endpoint_sensor_acceptance(plan):
    """Geometric sensor-support filtering only; no policy rollout selection."""
    forests = {}

    def accepted(case):
        from vtemp2_3.wildscenes_runtime import load_forest, sensor_support_at
        key = (case['map_path'], case['map_sha256'])
        if key not in forests:
            forests[key] = load_forest(case, plan['repo'])
        return bool(sensor_support_at(forests[key], case['start'], resolution=.5,
                                      shape=(16, 16), clearance=1.5)['supported'])

    return accepted


def prepare_manifest(plan, audit_path):
    """Sample once; a restart reuses only an intact manifest and sampling record."""
    if SOURCE.resolve() != Path(plan['source_dir']).resolve():
        raise WorkflowBlocked('Run the workflow from the declared frozen source_dir')
    from vtemp2_3.train_hierarchical import build_scene_cases, curriculum_ranges
    from vtemp2_3.wildscenes_tasks import load_verified_routes, sample_task_manifest
    directory = Path(plan['workflow_dir'])
    manifest_path, audit_output = directory / 'scene-manifest.json', directory / 'task-sampling-audit.json'
    record_path = directory / 'sampling-record.json'
    config = dict(plan['training_config'], validation_split='heldout_tasks')
    if any(path.exists() for path in (manifest_path, audit_output, record_path)):
        if not all(path.exists() for path in (manifest_path, audit_output, record_path)):
            raise WorkflowBlocked('Incomplete prior sampling transaction; automatic resampling is forbidden')
        record = read_json(record_path)
        if (record['scene_manifest_sha256'] != file_sha256(manifest_path)
                or record['sampling_audit_sha256'] != file_sha256(audit_output)
                or record['map_audit_sha256'] != file_sha256(audit_path)):
            raise WorkflowBlocked('Saved task manifest or geometry audit changed')
        manifest = read_json(manifest_path)
    else:
        routes = load_verified_routes(audit_path)
        manifest, sampling_audit = sample_task_manifest(routes, curriculum_ranges(config),
            config['episodes_per_update'], config['mission_distances_m'], config['eval_scenes'],
            config['seed'], case_acceptance=endpoint_sensor_acceptance(plan), return_audit=True)
        sampling_audit['endpoint_sensor_filter'] = dict(endpoints=['start'],
            shape=[16, 16], resolution_m=.5, clearance_m=1.5, selection='geometry_only_no_policy_rollouts')
        manifest = build_scene_cases(dict(config, scene_manifest=manifest))
        write_json(manifest_path, manifest)
        write_json(audit_output, sampling_audit)
        write_json(record_path, dict(scene_manifest_sha256=file_sha256(manifest_path),
            sampling_audit_sha256=file_sha256(audit_output), map_audit_sha256=file_sha256(audit_path),
            validation_split='heldout_tasks', created_at=now()))
        return manifest_path
    build_scene_cases(dict(config, scene_manifest=manifest))
    return manifest_path


def training_command(plan, run, manifest):
    entry = 'train_graph.py' if source_package(plan['source_dir']) == 'vtemp2' else 'train_hierarchical.py'
    command = ['nice', '-n', '5', sys.executable, '-I', '-B', '-u',
               str(Path(plan['source_dir']) / entry), '--repo', plan['repo'], '--output', str(run),
               '--graph-init', plan['graph_init'], '--slow-init', plan['slow_init'],
               '--scene-manifest', str(manifest), '--validation-split', 'heldout_tasks', '--complete-missions']
    for key, value in sorted(plan['training_config'].items()):
        if key in ('complete_missions', 'freeze_fast_world', 'validation_split'):
            continue
        flag = {'mission_distances_m': '--mission-distances', 'warmup_distances_m': '--warmup-distances'}.get(
            key, '--' + key.replace('_', '-'))
        command.extend([flag, *[str(item) for item in (value if isinstance(value, list) else [value])]])
    return command


def process_identity(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(') ', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, ValueError, TypeError):
        return None


def publish_pointers(plan, launch):
    directory = Path(plan['workflow_dir'])
    for name in ('vtemp2-graph-latest.json', 'vtemp2-wildscenes-formal-latest.json'):
        path = Path(plan['repo']) / 'artifacts' / name
        backup = directory / ('previous-' + name)
        if not backup.exists():
            write_json(backup, dict(existed=path.exists(), content=read_json(path) if path.exists() else None))
        write_json(path, dict(launch, kind='wildscenes_integrated_kilometer_formal',
            workflow_dir=str(directory), validation_split='heldout_tasks',
            validation_scope='Unseen endpoint tasks on potentially seen maps; no spatial-independence claim.'))
    launch['pointers_published'] = True
    write_json(directory / 'launch.json', launch)


def write_status(plan, state, **extra):
    write_json(Path(plan['workflow_dir']) / 'workflow-status.json',
               dict(state=state, pid=os.getpid(), updated_at=now(), **extra))


def monitor(plan, launch, poll_seconds=30):
    status_path = Path(launch['run']) / 'status.json'
    while True:
        trainer = read_json(status_path) if status_path.exists() else {}
        state = trainer.get('state')
        if not launch.get('pid') and trainer.get('pid'):
            # Recover only an already-created run's own status after an
            # interrupted Popen bookkeeping step. Never create another run.
            launch.update(pid=trainer['pid'], process_start_ticks=process_identity(trainer['pid']))
            write_json(Path(plan['workflow_dir']) / 'launch.json', launch)
        details = dict(run=launch['run'], log=launch['log'], trainer_pid=launch.get('pid'), trainer=trainer)
        if state in ('complete', 'failed', 'stopped'):
            write_status(plan, state, **details)
            return state
        identity = process_identity(launch.get('pid'))
        if identity is None or identity != launch.get('process_start_ticks'):
            raise WorkflowBlocked('Recorded trainer is absent or PID was reused; automatic relaunch is forbidden')
        if state == 'running' and trainer.get('pid') != launch.get('pid'):
            raise WorkflowBlocked('Running trainer status does not match the recorded process')
        if state == 'running' and not launch.get('pointers_published'):
            publish_pointers(plan, launch)
        write_status(plan, 'training' if state == 'running' else 'waiting_for_trainer', **details)
        time.sleep(poll_seconds)


def launch_once(plan, manifest):
    directory = Path(plan['workflow_dir'])
    run = Path(plan['repo']) / 'artifacts' / ('vtemp2-wildscenes-formal-' +
        datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8])
    log = run.with_suffix('.launch.log')
    launch = dict(run=str(run), log=str(log), pid=None, command=training_command(plan, run, manifest),
                  created_at=now(), state='launch_intent', pointers_published=False)
    # A durable intent closes the crash window that could otherwise spawn two
    # trainers. An ambiguous interrupted launch is blocked, never retried.
    write_json(directory / 'launch.json', launch)
    with log.open('ab') as stream:
        child = subprocess.Popen(launch['command'], cwd=plan['repo'], stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
    launch.update(pid=child.pid, process_start_ticks=process_identity(child.pid), state='launched', launched_at=now())
    write_json(directory / 'launch.json', launch)
    write_status(plan, 'launched', run=str(run), log=str(log), trainer_pid=child.pid)
    return launch


def run_workflow(plan_path, poll_seconds=30):
    plan_path = Path(plan_path).resolve(strict=True)
    plan = read_json(plan_path)
    directory = Path(plan['workflow_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'workflow.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WorkflowBlocked('Another workflow process holds this workflow lock') from error
        try:
            record_path = directory / 'plan-record.json'
            plan_sha = file_sha256(plan_path)
            if record_path.exists() and read_json(record_path)['sha256'] != plan_sha:
                raise WorkflowBlocked('Workflow plan changed; existing workflow cannot be repurposed')
            if not record_path.exists():
                write_json(record_path, dict(path=str(plan_path), sha256=plan_sha, created_at=now()))
            validate_assets(plan)
            launch_path = directory / 'launch.json'
            if launch_path.exists():
                return monitor(plan, read_json(launch_path), poll_seconds)
            while True:
                if file_sha256(plan_path) != plan_sha:
                    raise WorkflowBlocked('Workflow plan changed while waiting')
                validate_assets(plan)
                audit, preparation = preparation_ready(plan)
                if audit is not None:
                    break
                write_status(plan, 'waiting_for_geometry', preparation=preparation,
                             mission_distances_m=plan['training_config']['mission_distances_m'])
                time.sleep(poll_seconds)
            write_status(plan, 'validating_tasks', map_audit=str(audit))
            manifest = prepare_manifest(plan, audit)
            validate_assets(plan)
            return monitor(plan, launch_once(plan, manifest), poll_seconds)
        except (OSError, ValueError, KeyError, TypeError) as error:
            write_status(plan, 'blocked', reason=str(error), error_type=type(error).__name__)
            return 'blocked'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    try:
        state = run_workflow(args.plan)
    except WorkflowBlocked as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(0 if state == 'complete' else 2)


if __name__ == '__main__':
    main()
