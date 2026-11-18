"""Repair evaluation pins its inputs and never turns diagnostics into success rates."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from world_model.evaluate_safety_repairs import (freeze_inputs, digest, lean_recorder_type,
    parse_args, select_cases, summarize, validate_checkpoint, safety_row, write_json, runtime_contract)


class RepairEvaluationTests(unittest.TestCase):
    def test_snapshot_pins_checkpoint_and_backend_without_mutating_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'scripts' / 'world_model'
            source.mkdir(parents=True)
            (source / '__init__.py').write_text('')
            (source / 'evaluate_safety_repairs.py').write_text('# evaluation\n')
            backend = source.parent / 'simulator'
            backend.mkdir()
            (backend / 'wildscenes_map.py').write_text('# backend\n')
            checkpoint = root / 'latest.pt'
            checkpoint.write_bytes(b'old checkpoint')
            before = digest(checkpoint)
            output = root / 'evaluation'
            entry = freeze_inputs(output, checkpoint, source)
            self.assertEqual(entry.parent, output / 'runtime_sources' / 'world_model')
            manifest = json.loads((output / 'input-manifest.json').read_text())
            self.assertEqual(manifest['checkpoint_sha256'], before)
            self.assertEqual(digest(checkpoint), before)
            self.assertEqual(entry.read_text(), '# evaluation\n')
            self.assertEqual((entry.parent / '_wildscenes_map_backend.py').read_text(), '# backend\n')
            checkpoint.write_bytes(b'next checkpoint')
            (source / 'evaluate_safety_repairs.py').write_text('# changed\n')
            self.assertEqual(digest(output / 'input.pt'), before)
            self.assertEqual(digest(entry), manifest['source_sha256']['evaluate_safety_repairs.py'])
            with self.assertRaisesRegex(ValueError, 'not empty'):
                freeze_inputs(output, checkpoint, source)

    def test_snapshot_rejects_checkpoint_changed_during_copy(self):
        import shutil
        original_copy = shutil.copy2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            (source / 'evaluate_safety_repairs.py').write_text('# evaluation')
            (source / '_wildscenes_map_backend.py').write_text('# backend')
            checkpoint = root / 'latest.pt'
            checkpoint.write_bytes(b'old')

            def changing_copy(src, dst):
                result = original_copy(src, dst)
                if Path(src) == checkpoint:
                    checkpoint.write_bytes(b'new')
                return result

            with patch('world_model.evaluate_safety_repairs.shutil.copy2', changing_copy):
                with self.assertRaisesRegex(ValueError, 'changed while copying'):
                    freeze_inputs(root / 'output', checkpoint, source)

    def test_baseline_snapshot_injects_only_evaluator_and_records_original_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'old_source'
            source.mkdir()
            (source / 'hierarchy.py').write_text('# old controller')
            (source / '_wildscenes_map_backend.py').write_text('# old backend')
            evaluator = root / 'new_evaluator.py'
            evaluator.write_text('# new evaluation harness')
            checkpoint = root / 'checkpoint.pt'
            checkpoint.write_bytes(b'weights')
            entry = freeze_inputs(root / 'output', checkpoint, source, evaluator=evaluator)
            self.assertEqual(entry.read_text(), evaluator.read_text())
            self.assertEqual((entry.parent / 'hierarchy.py').read_text(), '# old controller')
            self.assertFalse((source / 'evaluate_repairs.py').exists())
            self.assertEqual(runtime_contract({})['communication_reserve_m'], 0.)

    def test_selection_preserves_case_values_order_and_original_indices(self):
        cases = [dict(seed=i + 10, start=[i, 0], goal=[1000 + i, 0]) for i in range(9)]
        config = dict(scene_manifest=dict(validation=cases))
        selected = select_cases(config, [6, 0, 3])
        self.assertEqual(selected, [(6, cases[6]), (0, cases[0]), (3, cases[3])])
        self.assertIs(selected[0][1], cases[6])
        self.assertEqual(len(select_cases(config)), 9)
        for indices in ([0, 0], [-1], [9], []):
            with self.assertRaises(ValueError):
                select_cases(config, indices)
        with self.assertRaisesRegex(ValueError, 'fixed validation'):
            select_cases({})

    def test_lean_recorder_refreshes_causal_inputs_without_training_storage(self):
        class TrainingRecorder:
            def __init__(self):
                self.cbf_records = []
                self.inputs, self.transitions = [], []
                self.truth_states, self.truth_actions = {}, {}
                self.calls = []
                self.base_step = lambda frame: ('stepped', frame)

            def prepare(self, frame):
                self.calls.append(('prepare', frame))
                self.record_truth(frame)

            def record_truth(self, frame):
                self.truth_states[frame] = 'training truth'

            def save_input(self, graph):
                self.inputs.append(graph)

            def ground_observation(self):
                return 'received ground'

            def graph(self, owner, observation):
                self.calls.append((owner, observation))
                return 'graph'

            def predict(self, graph, kind):
                self.calls.append((graph, kind))

        callbacks = []
        recorder = lean_recorder_type(TrainingRecorder)(on_step=callbacks.append)
        for frame in range(5):
            self.assertEqual(recorder.step(frame), ('stepped', frame))
            recorder.save_input({'large': 'training tensor'})
            recorder.cbf_records.append({'large': 'received geometry'})
        self.assertEqual(callbacks, list(range(5)))
        self.assertEqual(recorder.calls[:3], [('prepare', 0), ('ground_0', 'received ground'), ('graph', 'ugv')])
        self.assertFalse(any((recorder.inputs, recorder.transitions, recorder.truth_states, recorder.truth_actions)))
        self.assertEqual(len(recorder.cbf_records), 0)
        with self.assertRaisesRegex(AssertionError, 'training transitions'):
            recorder.append_transition('data')

        def failure(frame):
            raise RuntimeError('safety failure')

        recorder.base_step = failure
        with self.assertRaisesRegex(RuntimeError, 'safety failure'):
            recorder.step(5)
        self.assertEqual(callbacks[-1], 5)

    def test_cutoff_never_reports_full_mission_success_rate(self):
        rows = [dict(goal_reached=True, truncated=False, termination='mission_arrival', collisions=0,
                     ground_safety_violations=0),
                dict(goal_reached=False, truncated=True, termination='rollout_limit', collisions=0,
                     ground_safety_violations=0)]
        result = summarize(rows, diagnostic=True)
        self.assertIsNone(result['success_rate'])
        self.assertEqual(result['successes'], 1)
        self.assertEqual(result['truncated_episodes'], 1)
        rows[1].update(truncated=False, termination='mission_budget')
        self.assertEqual(summarize(rows, diagnostic=False)['success_rate'], .5)

    def test_schema_mismatch_is_rejected_before_model_execution(self):
        modules = dict(graph_world=SimpleNamespace(MODEL_SCHEMA='world'),
                       graph_policy=SimpleNamespace(FAST_SCHEMA='fast', SLOW_SCHEMA='slow'))
        checkpoint = dict(world_schema='world', manifest=dict(fast_schema='fast', slow_schema='slow'))
        validate_checkpoint(checkpoint, modules)
        checkpoint['manifest']['slow_schema'] = 'old'
        with self.assertRaisesRegex(ValueError, 'schemas differ'):
            validate_checkpoint(checkpoint, modules)

    def test_nonfinite_margin_remains_explicit_and_serializable(self):
        controller = SimpleNamespace(drone_id='drone_0', safety_stats=dict(
            feasibility_margin=float('-inf'), infeasible=True, communication_guard_feasible=False))
        row = safety_row(4, controller)
        self.assertIsNone(row['margin_m_s'])
        self.assertEqual(row['margin_kind'], '-inf')
        self.assertTrue(row['infeasible'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'row.json'
            write_json(path, row)
            self.assertEqual(json.loads(path.read_text()), row)

    def test_defaults_control_memory_and_cutoff_must_be_positive(self):
        args = parse_args(['--checkpoint', '/tmp/checkpoint.pt', '--output', '/tmp/result'])
        self.assertEqual(args.workers, 2)
        self.assertIsNone(args.max_steps)
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            parse_args(['--checkpoint', '/tmp/checkpoint.pt', '--output', '/tmp/result', '--max-steps', '0'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
