"""The persistent workflow gates immutable inputs and never launches twice."""
import fcntl
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from world_model import wildscenes_workflow as workflow


def fixture(directory):
    root = Path(directory)
    repo, source, preparation, output = [root / name for name in ('repo', 'source', 'preparation', 'workflow')]
    for path in (repo / 'artifacts', source, preparation, output):
        path.mkdir(parents=True)
    for name in workflow.REQUIRED_SOURCES:
        (source / name).write_text('# frozen ' + name + '\n')
    graph, slow = root / 'graph.pt', root / 'slow.pt'
    graph.write_bytes(b'graph-checkpoint')
    slow.write_bytes(b'slow-checkpoint')
    readiness = root / 'readiness.json'
    workflow.write_json(readiness, dict(smoke_complete=True, all_models_updated=True, tests_passed=135))
    plan = dict(repo=str(repo), source_dir=str(source), preparation_dir=str(preparation), workflow_dir=str(output),
        graph_init=str(graph), slow_init=str(slow), readiness_verification=str(readiness),
        source_sha256={path.name: workflow.file_sha256(path) for path in source.glob('*.py')},
        training_config=dict(warmup_updates=8, fast_updates=8, slow_updates=64, joint_updates=32,
            episodes_per_update=4, eval_scenes=9, warmup_distances_m=[100., 200.], mission_distances_m=[1000., 1000.],
            seed=300, device='cpu', workers=2, complete_missions=True))
    for key in ('graph_init', 'slow_init', 'readiness_verification'):
        plan[key + '_sha256'] = workflow.file_sha256(plan[key])
    plan_path = root / 'plan.json'
    workflow.write_json(plan_path, plan)
    return plan, plan_path


class WorkflowTests(unittest.TestCase):
    def test_waits_for_geometry_then_launches_one_explicit_integrated_command(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, plan_path = fixture(directory)
            audit = Path(plan['preparation_dir']) / 'audit.json'
            manifest = Path(plan['workflow_dir']) / 'scene-manifest.json'
            with patch.object(workflow, 'preparation_ready', side_effect=[(None, {'stage': 'downloading'}),
                        (audit, {'stage': 'complete'})]), patch.object(workflow, 'prepare_manifest', return_value=manifest), \
                    patch.object(workflow.time, 'sleep') as sleep, patch.object(workflow, 'process_identity', return_value='42'), \
                    patch.object(workflow.subprocess, 'Popen', return_value=SimpleNamespace(pid=1234)) as popen, \
                    patch.object(workflow, 'monitor', return_value='complete'):
                self.assertEqual(workflow.run_workflow(plan_path), 'complete')
            sleep.assert_called_once_with(30)
            popen.assert_called_once()
            command = popen.call_args.args[0]
            self.assertEqual(command[:3], ['nice', '-n', '5'])
            self.assertIn('--complete-missions', command)
            self.assertNotIn('--freeze-fast-world', command)
            self.assertEqual(command[command.index('--validation-split') + 1], 'heldout_tasks')
            index = command.index('--mission-distances')
            self.assertEqual([float(v) for v in command[index + 1:index + 3]], [1000., 1000.])
            self.assertEqual(command[command.index('--slow-init') + 1], plan['slow_init'])
            self.assertEqual(popen.call_args.kwargs['env'], workflow.os.environ.copy())
            launch = workflow.read_json(Path(plan['workflow_dir']) / 'launch.json')
            self.assertEqual(launch['pid'], 1234)

    def test_preparation_failure_or_missing_kilometer_audit_blocks_without_sampling(self):
        for state in ('failed', 'complete'):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                plan, path = fixture(directory)
                preparation = Path(plan['preparation_dir'])
                workflow.write_json(preparation / 'preparation-status.json', dict(stage=state, error='geometry unavailable'))
                workflow.write_json(preparation / 'K-03-kilometer-audit.json', dict(status='no_verified_kilometer_task'))
                with patch.object(workflow, 'prepare_manifest') as sample, patch.object(workflow.subprocess, 'Popen') as popen:
                    self.assertEqual(workflow.run_workflow(path), 'blocked')
                sample.assert_not_called()
                popen.assert_not_called()
                self.assertEqual(workflow.read_json(Path(plan['workflow_dir']) / 'workflow-status.json')['state'], 'blocked')
                self.assertEqual(plan['training_config']['mission_distances_m'], [1000., 1000.])

    def test_readiness_and_source_checkpoint_hash_changes_block(self):
        for key in ('graph_init', 'slow_init', 'readiness_verification', 'source_dir'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                plan, path = fixture(directory)
                changed = Path(plan[key])
                if key == 'source_dir':
                    changed /= 'train_hierarchical.py'
                changed.write_bytes(changed.read_bytes() + b'changed')
                with patch.object(workflow.subprocess, 'Popen') as popen:
                    self.assertEqual(workflow.run_workflow(path), 'blocked')
                popen.assert_not_called()
        for values in (dict(smoke_complete=False), dict(all_models_updated=False), dict(tests_passed=0), dict(tests_passed=True)):
            with self.subTest(values=values), tempfile.TemporaryDirectory() as directory:
                plan, _ = fixture(directory)
                evidence = workflow.read_json(plan['readiness_verification'])
                evidence.update(values)
                workflow.write_json(plan['readiness_verification'], evidence)
                plan['readiness_verification_sha256'] = workflow.file_sha256(plan['readiness_verification'])
                with self.assertRaisesRegex(workflow.WorkflowBlocked, 'Readiness'):
                    workflow.validate_assets(plan)

    def test_existing_launch_only_monitors_and_never_prepares_or_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, path = fixture(directory)
            launch = dict(run=str(Path(plan['repo']) / 'artifacts' / 'existing'), log='/tmp/existing.log', pid=123)
            workflow.write_json(Path(plan['workflow_dir']) / 'launch.json', launch)
            with patch.object(workflow, 'monitor', return_value='complete') as monitor, \
                    patch.object(workflow, 'preparation_ready') as preparation, \
                    patch.object(workflow, 'prepare_manifest') as sample, patch.object(workflow.subprocess, 'Popen') as popen:
                self.assertEqual(workflow.run_workflow(path), 'complete')
            monitor.assert_called_once_with(plan, launch, 30)
            preparation.assert_not_called()
            sample.assert_not_called()
            popen.assert_not_called()

    def test_launch_intent_survives_popen_failure_and_prevents_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, path = fixture(directory)
            with patch.object(workflow.subprocess, 'Popen', side_effect=OSError('cannot spawn')):
                with self.assertRaises(OSError):
                    workflow.launch_once(plan, Path(plan['workflow_dir']) / 'scene-manifest.json')
            launch = workflow.read_json(Path(plan['workflow_dir']) / 'launch.json')
            self.assertEqual(launch['state'], 'launch_intent')
            self.assertIsNone(launch['pid'])
            with patch.object(workflow.subprocess, 'Popen') as popen:
                self.assertEqual(workflow.run_workflow(path), 'blocked')
            popen.assert_not_called()

    def test_sampling_transaction_reuses_manifest_and_never_resamples_changed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, _ = fixture(directory)
            audit = Path(plan['preparation_dir']) / 'audit.json'
            workflow.write_json(audit, dict(status=workflow.READY))
            manifest = dict(train=[{'seed': 1}], validation=[{'seed': 2}])
            with patch.object(workflow, 'SOURCE', Path(plan['source_dir'])), \
                    patch('world_model.wildscenes_tasks.load_verified_routes', return_value=['verified']) as load, \
                    patch('world_model.wildscenes_tasks.sample_task_manifest', return_value=(manifest, {'candidate_rejections': 0})) as sample, \
                    patch('world_model.train_hierarchical.build_scene_cases', side_effect=lambda config: config['scene_manifest']) as build:
                first = workflow.prepare_manifest(plan, audit)
                self.assertEqual(workflow.prepare_manifest(plan, audit), first)
                load.assert_called_once()
                sample.assert_called_once()
                self.assertEqual(len(sample.call_args.args[1]), 112)
                self.assertEqual(sample.call_args.args[2:5], (4, [1000., 1000.], 9))
                self.assertIs(sample.call_args.kwargs['return_audit'], True)
                self.assertTrue(callable(sample.call_args.kwargs['case_acceptance']))
                self.assertEqual(build.call_count, 2)
                workflow.write_json(first, dict(train=[], validation=[]))
                with self.assertRaisesRegex(workflow.WorkflowBlocked, 'changed'):
                    workflow.prepare_manifest(plan, audit)
                sample.assert_called_once()

    def test_endpoint_sensor_filter_uses_geometry_and_caches_map(self):
        case = dict(map_path='/maps/forest.npz', map_sha256='declared', start=[0., 0.], goal=[1000., 0.])
        forest = object()
        with patch('world_model.wildscenes_runtime.load_forest', return_value=forest) as load, \
                patch('world_model.wildscenes_runtime.sensor_support_at', create=True,
                      side_effect=[{'supported': True}, {'supported': False}]) as support:
            accept = workflow.endpoint_sensor_acceptance({'repo': '/repo'})
            self.assertTrue(accept(case))
            self.assertFalse(accept(case))
            load.assert_called_once_with(case, '/repo')
            self.assertEqual(support.call_args_list[1].args, (forest, [0., 0.]))
            self.assertEqual(support.call_args.kwargs, dict(resolution=.5, shape=(16, 16), clearance=1.5))

    def test_only_live_running_trainer_updates_pointers_and_preserves_previous_values(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, _ = fixture(directory)
            run = Path(plan['repo']) / 'artifacts' / 'training'
            run.mkdir()
            status = run / 'status.json'
            workflow.write_json(status, dict(state='running', pid=123))
            previous = dict(run='older-run', pid=10)
            workflow.write_json(Path(plan['repo']) / 'artifacts' / 'vtemp2-graph-latest.json', previous)
            launch = dict(run=str(run), log=str(run) + '.log', pid=123, process_start_ticks='42')
            def finish(_seconds):
                workflow.write_json(status, dict(state='complete', pid=123))
            with patch.object(workflow, 'process_identity', return_value='42'), patch.object(workflow.time, 'sleep', side_effect=finish):
                self.assertEqual(workflow.monitor(plan, launch), 'complete')
            backup = workflow.read_json(Path(plan['workflow_dir']) / 'previous-vtemp2-graph-latest.json')
            self.assertEqual(backup, dict(existed=True, content=previous))
            pointer = workflow.read_json(Path(plan['repo']) / 'artifacts' / 'vtemp2-wildscenes-formal-latest.json')
            self.assertEqual(pointer['run'], str(run))
            self.assertEqual(pointer['validation_split'], 'heldout_tasks')
            self.assertEqual(workflow.read_json(Path(plan['workflow_dir']) / 'workflow-status.json')['state'], 'complete')

    def test_lock_excludes_a_second_workflow_without_overwriting_status(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, path = fixture(directory)
            status = Path(plan['workflow_dir']) / 'workflow-status.json'
            workflow.write_json(status, dict(state='existing-supervisor'))
            with (Path(plan['workflow_dir']) / 'workflow.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(workflow.WorkflowBlocked, 'holds this workflow lock'):
                    workflow.run_workflow(path)
            self.assertEqual(workflow.read_json(status)['state'], 'existing-supervisor')

    def test_stale_running_status_or_reused_pid_cannot_publish_pointers(self):
        for identity in (None, 'different-start-tick'):
            with self.subTest(identity=identity), tempfile.TemporaryDirectory() as directory:
                plan, _ = fixture(directory)
                run = Path(plan['repo']) / 'artifacts' / 'stale-run'
                run.mkdir()
                workflow.write_json(run / 'status.json', dict(state='running', pid=123))
                launch = dict(run=str(run), log=str(run) + '.log', pid=123, process_start_ticks='42')
                with patch.object(workflow, 'process_identity', return_value=identity), \
                        patch.object(workflow, 'publish_pointers') as publish:
                    with self.assertRaisesRegex(workflow.WorkflowBlocked, 'absent or PID was reused'):
                        workflow.monitor(plan, launch)
                publish.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
