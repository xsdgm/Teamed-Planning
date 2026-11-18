"""Declared phase endpoints and effective publication metadata remain reproducible."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from world_model import train_hierarchical as trainer


class ExecutionProtocolTests(unittest.TestCase):
    def config(self, extra=()):
        arguments = ['train_hierarchical.py', '--output', '/tmp/protocol-test', '--fresh-all',
                     '--warmup-updates', '0', '--fast-updates', '4', '--slow-updates', '16',
                     '--joint-updates', '8', '--episodes-per-update', '16', *extra]
        with patch.object(sys, 'argv', arguments):
            return trainer.make_initial_config(trainer.parse_args())

    def test_declared_phase_endpoint_keeps_full_plan_and_stops_at_last_matching_batch(self):
        config = self.config(['--stop-after-phase', 'slow'])
        before = copy.deepcopy(config)
        self.assertEqual(trainer.execution_schedule(config), ['fast'] * 4 + ['slow'] * 16)
        self.assertEqual(trainer.phase_schedule(config), ['fast'] * 4 + ['slow'] * 16 + ['joint'] * 8)
        self.assertEqual(len(trainer.training_batches(config)), 28)
        self.assertEqual(len(trainer.curriculum_ranges(config)), 28)
        self.assertEqual(config, before)
        self.assertTrue(trainer.evaluation_due(config, 20))

    def test_explicit_continuation_keeps_twelve_batches_but_executes_four_slow_batches(self):
        plan = [dict(phase='slow' if i < 4 else 'joint', episodes=16,
                     distance_range_m=[850., 1000.] if i < 4 else [1000., 1000.],
                     source_update_indices=[20 + i], evaluate_after=(i + 1) % 2 == 0)
                for i in range(12)]
        config = dict(warmup_updates=0, fast_updates=0, slow_updates=4, joint_updates=8,
                      episodes_per_update=16, update_plan=plan, stop_after_phase='slow')
        before = copy.deepcopy(config)
        self.assertEqual(trainer.execution_schedule(config), ['slow'] * 4)
        self.assertEqual(len(trainer.phase_schedule(config)), 12)
        self.assertEqual(sum(count for _, count in trainer.training_batches(config)[:4]), 64)
        self.assertEqual(sum(count for _, count in trainer.training_batches(config)), 192)
        self.assertTrue(trainer.evaluation_due(config, 4))
        self.assertEqual(config, before)

    def test_absent_endpoint_is_an_error_and_omitting_endpoint_retains_full_schedule(self):
        config = self.config()
        self.assertEqual(trainer.execution_schedule(config), trainer.phase_schedule(config))
        with self.assertRaisesRegex(ValueError, 'absent'):
            trainer.execution_schedule(dict(config, stop_after_phase='warmup'))

    def test_optimizer_protocol_is_json_safe_and_records_actual_groups_and_slow_defaults(self):
        parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in range(3)]
        optimizer = torch.optim.Adam([dict(params=[parameters[0]], lr=3e-4),
                                      dict(params=[parameters[1]], lr=2e-4)], betas=(.8, .9), eps=1e-6)
        slow = torch.optim.Adam([parameters[2]], lr=4e-4)
        config = self.config(['--slow-epochs', '7', '--ppo-epochs', '3'])
        protocol = trainer.optimization_protocol(config, dict(fast_actor=optimizer, slow=slow))
        restored = json.loads(json.dumps(protocol, allow_nan=False))
        groups = restored['optimizers']['fast_actor']['parameter_groups']
        self.assertEqual([group['lr'] for group in groups], [3e-4, 2e-4])
        self.assertEqual(groups[0]['betas'], [.8, .9])
        self.assertEqual(groups[0]['eps'], 1e-6)
        self.assertTrue(all('params' not in group for group in groups))
        self.assertEqual(restored['slow_ppo']['gamma'], .999)
        self.assertEqual(restored['slow_ppo']['gae_lambda'], .95)
        self.assertEqual(restored['slow_ppo']['epochs'], 7)
        self.assertEqual(restored['fast_ppo']['epochs'], 3)
        self.assertTrue(restored['phase_targets']['slow']['world'])

    def test_phase_endpoint_finishes_validation_and_keeps_each_update_checkpoint(self):
        repo = Path(__file__).resolve().parents[5]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'tiny-run'
            output.mkdir()
            arguments = ['train_hierarchical.py', '--output', str(output), '--repo', str(repo),
                '--fresh-all', '--frozen', '--warmup-updates', '0', '--fast-updates', '1',
                '--slow-updates', '1', '--joint-updates', '1', '--stop-after-phase', 'fast',
                '--episodes-per-update', '1', '--eval-scenes', '1', '--workers', '1', '--hidden', '16',
                '--steps', '4', '--option-steps', '4', '--stride', '1', '--no-complete-missions',
                '--wm-epochs', '1', '--ppo-epochs', '1', '--slow-epochs', '1',
                '--warmup-distances', '20', '30', '--mission-distances', '40', '50']
            with patch.object(sys, 'argv', arguments), patch.object(trainer.signal, 'signal'), \
                    contextlib.redirect_stdout(io.StringIO()):
                trainer.main()
            status = json.loads((output / 'status.json').read_text())
            self.assertEqual(status['state'], 'complete')
            self.assertEqual(status['phase'], 'finished')
            self.assertEqual(status['completed_updates'], 1)
            self.assertIsNone(status['pending_evaluation'])
            self.assertTrue((output / 'metrics' / 'validation-001.json').exists())
            self.assertTrue((output / 'checkpoints' / 'update-001.pt').exists())
            self.assertFalse((output / 'checkpoints' / 'update-002.pt').exists())
            final = torch.load(output / 'checkpoints' / 'final.pt', map_location='cpu', weights_only=True)
            self.assertEqual(final['completed_updates'], 1)
            self.assertIsNone(final['pending_evaluation'])
            self.assertEqual(final['manifest']['execution_scope']['effective_updates'], 1)
            self.assertEqual(final['manifest']['execution_scope']['full_plan_updates'], 3)
            self.assertEqual(final['manifest']['execution_scope']['effective_training_tasks'], 1)
            self.assertEqual([row['evaluation'] for row in final['records']], [False, True])


if __name__ == '__main__':
    unittest.main(verbosity=2)
