"""Fresh slow learning transfers perception/control weights, never old learner state."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vtemp2_3 import train_hierarchical as trainer
from vtemp2_3.graph_fast_policy import numpy_state, state_hash


def fresh_arguments(output='/tmp/fresh-slow-test', graph='/tmp/source.pt'):
    return ['train_hierarchical.py', '--output', str(output), '--fresh-slow',
            '--graph-init', str(graph), '--freeze-fast-world', '--warmup-updates', '1',
            '--fast-updates', '0', '--slow-updates', '1', '--joint-updates', '0',
            '--episodes-per-update', '2', '--eval-scenes', '2', '--hidden', '16']


def task_plan():
    return [dict(phase=phase, episodes=2, distance_range_m=bounds,
                 source_update_indices=[index], evaluate_after=True)
            for index, (phase, bounds) in enumerate([('warmup', [30., 60.]), ('slow', [100., 200.])])]


def checkpoint_hash(state):
    return state_hash({name: value.detach().cpu().numpy() for name, value in state.items()})


class FreshSlowTests(unittest.TestCase):
    def parse(self, arguments):
        with patch.object(sys, 'argv', arguments):
            return trainer.parse_args()

    def test_requires_graph_initialization_and_frozen_fast_world(self):
        valid = fresh_arguments()
        self.assertTrue(self.parse(valid).fresh_slow)
        for omitted in ('--graph-init', '--freeze-fast-world'):
            arguments = list(valid)
            index = arguments.index(omitted)
            del arguments[index:index + (2 if omitted == '--graph-init' else 1)]
            with self.subTest(omitted=omitted), patch('sys.stderr'), self.assertRaises(SystemExit):
                self.parse(arguments)

    def test_rejects_inherited_slow_learner_and_fast_or_joint_optimization(self):
        for extra in [
            ['--slow-init', '/tmp/old-slow.pt'], ['--resume'],
            ['--continue-from', '/tmp/old.pt', '--continuation-plan', '/tmp/continuation.json'],
            ['--fast-updates', '1'], ['--joint-updates', '1'], ['--slow-updates', '0']]:
            with self.subTest(extra=extra), patch('sys.stderr'), self.assertRaises(SystemExit):
                self.parse(fresh_arguments() + extra)

    def test_external_update_plan_cannot_override_resume_or_continuation(self):
        base = ['train_hierarchical.py', '--output', '/tmp/run', '--update-plan', '/tmp/plan.json']
        for extra in [['--resume'],
                      ['--continue-from', '/tmp/old.pt', '--continuation-plan', '/tmp/rebatch.json']]:
            with self.subTest(extra=extra), patch('sys.stderr'), self.assertRaises(SystemExit):
                self.parse(base + extra)

    def test_initial_config_embeds_validated_bare_plan_and_survives_source_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            expected = task_plan()
            path.write_text(json.dumps(expected))
            args = self.parse(fresh_arguments() + ['--update-plan', str(path)])
            config = trainer.make_initial_config(args)
            self.assertEqual(config['update_plan'], expected)
            self.assertIsInstance(config['update_plan'], list)
            self.assertEqual(config['training_mode'], 'fresh_slow')
            self.assertEqual(trainer.phase_schedule(config), ['warmup', 'slow'])
            self.assertEqual(trainer.training_batches(config), [(0, 2), (2, 2)])
            self.assertEqual(trainer.curriculum_ranges(config), [[30., 60.], [100., 200.]])
            json.dumps(config, allow_nan=False)
            path.write_text('[]')
            self.assertEqual(config['update_plan'], expected)
            self.assertNotIn('continue_from', config)

    def test_bad_plan_fails_before_scene_collection(self):
        malformed = [dict(update_plan=task_plan()), []]
        duplicate = task_plan()
        duplicate[1]['source_update_indices'] = [0]
        malformed.append(duplicate)
        oversized = task_plan()
        oversized[0]['episodes'] = 3
        malformed.append(oversized)
        no_boundary_validation = task_plan()
        no_boundary_validation[0]['evaluate_after'] = False
        malformed.append(no_boundary_validation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plan.json'
            args = self.parse(fresh_arguments() + ['--update-plan', str(path)])
            for value in malformed:
                with self.subTest(plan=value):
                    path.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):
                        trainer.make_initial_config(args)

    def test_initial_checkpoint_copies_fast_world_and_resets_every_slow_learner_state(self):
        torch.set_num_threads(1)
        repo = Path(__file__).resolve().parents[5]
        parent_fast = trainer.make_graph_fast(repo)
        parent_world = trainer.GraphWorldModel(16, 4)
        parent_slow = trainer.GraphSlowActorCritic(hidden=16)
        with torch.no_grad():
            for model in (parent_fast, parent_world):
                for parameter in model.parameters():
                    parameter.add_(.0125)
            for parameter in parent_slow.parameters():
                parameter.fill_(.75)
        parent = dict(schema=trainer.TRAIN_SCHEMA, world_schema=trainer.MODEL_SCHEMA,
            manifest=dict(fast_schema=trainer.FAST_SCHEMA, slow_schema=trainer.SLOW_SCHEMA),
            fast=parent_fast.state_dict(), world=parent_world.state_dict(), slow=parent_slow.state_dict(),
            completed_updates=20, counters=dict(slow_optimizer_updates=99),
            records=[dict(label='old-scene')], replay_paths=['/old/replay/must-not-be-used.npz'])
        for key in ('actor_optimizer', 'critic_optimizer', 'world_optimizer', 'slow_optimizer'):
            parent[key] = dict(state={0: dict(step=torch.tensor(99.), exp_avg=torch.ones(1))}, param_groups=[])
        original_slow = trainer.GraphSlowActorCritic
        initial_slow = []

        def capture_slow(*args, **kwargs):
            model = original_slow(*args, **kwargs)
            initial_slow.append(numpy_state(model))
            return model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, output = root / 'parent.pt', root / 'fresh'
            torch.save(parent, checkpoint)
            output.mkdir()
            arguments = fresh_arguments(output, checkpoint) + ['--repo', str(repo), '--frozen']
            # Exercise the real initialization and checkpoint path. Stop at
            # pool construction before any worker or simulation can start.
            with patch.object(sys, 'argv', arguments), \
                    patch.object(trainer, 'GraphSlowActorCritic', side_effect=capture_slow), \
                    patch.object(trainer, 'ProcessPoolExecutor', side_effect=RuntimeError('initialization checked')), \
                    patch.object(trainer.signal, 'signal'), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'initialization checked'):
                    trainer.main()
            saved = torch.load(output / 'checkpoints' / 'initial.pt', map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint_hash(saved['fast']), state_hash(numpy_state(parent_fast)))
            self.assertEqual(checkpoint_hash(saved['world']), state_hash(numpy_state(parent_world)))
            self.assertEqual(checkpoint_hash(saved['slow']), state_hash(initial_slow[0]))
            self.assertNotEqual(checkpoint_hash(saved['slow']), state_hash(numpy_state(parent_slow)))
            for key in ('actor_optimizer', 'critic_optimizer', 'world_optimizer', 'slow_optimizer'):
                self.assertEqual(saved[key]['state'], {})
            self.assertEqual(saved['completed_updates'], 0)
            self.assertEqual(saved['policy_version'], 0)
            self.assertTrue(all(value == 0 for value in saved['counters'].values()))
            self.assertEqual(saved['records'], [])
            self.assertEqual(saved['replay_paths'], [])
            self.assertEqual(saved['manifest']['training_mode'], 'fresh_slow')
            self.assertEqual(saved['manifest']['source_package'], 'world_model')
            for name in ('fast', 'world', 'slow'):
                self.assertEqual(saved['manifest']['initialization']['initial_model_sha256'][name],
                                 checkpoint_hash(saved[name]))
            self.assertTrue(saved['config']['freeze_fast_world'])
            self.assertEqual([group['lr'] for group in saved['actor_optimizer']['param_groups']],
                             [1e-5, 3e-4])
            self.assertEqual([group['lr'] for group in saved['manifest']['optimization']['optimizers']
                              ['fast_actor']['parameter_groups']], [1e-5, 3e-4])


if __name__ == '__main__':
    unittest.main(verbosity=2)
