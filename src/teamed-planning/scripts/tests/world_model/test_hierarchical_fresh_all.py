"""Three-stage fresh learning has no pretrained weight or learned-teacher path."""
import contextlib
from concurrent.futures import ProcessPoolExecutor
import io
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from world_model import train_hierarchical as trainer
from world_model import hierarchical_policy as policy
from world_model.graph_fast_policy import numpy_state, state_hash
from world_model.navigation_scenes import make_scene


def fresh_arguments(output='/tmp/fresh-all-test'):
    return ['train_hierarchical.py', '--output', str(output), '--fresh-all',
            '--warmup-updates', '0', '--fast-updates', '1', '--slow-updates', '1',
            '--joint-updates', '1', '--episodes-per-update', '2', '--eval-scenes', '2', '--hidden', '16']


def checkpoint_hash(state):
    return state_hash({name: value.detach().cpu().numpy() for name, value in state.items()})


def collect_without_checkpoint(job):
    """Spawn-worker contract: even initial simulator construction cannot load weights."""
    from world_model import graph_rollouts
    with patch.object(torch, 'load', side_effect=AssertionError('fresh worker read checkpoint')), \
            patch.object(graph_rollouts, 'TeacherBehavior', side_effect=AssertionError('learned teacher used')):
        return graph_rollouts.collect_graph(job)


class FreshAllTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.repo = Path(__file__).resolve().parents[5]

    def parse(self, arguments):
        with patch.object(sys, 'argv', arguments):
            return trainer.parse_args()

    def test_fresh_all_rejects_any_initialization_resume_or_freezing(self):
        args = self.parse(fresh_arguments())
        config = trainer.make_initial_config(args)
        self.assertEqual(config['training_mode'], 'fresh_all_staged')
        self.assertEqual(trainer.phase_schedule(config), ['fast', 'slow', 'joint'])
        for extra in [
            ['--fast-init', '/tmp/old.pt'], ['--graph-init', '/tmp/old.pt'],
            ['--slow-init', '/tmp/old.pt'], ['--resume'],
            ['--continue-from', '/tmp/old.pt', '--continuation-plan', '/tmp/plan.json'],
            ['--freeze-fast-world'], ['--fresh-slow']]:
            with self.subTest(extra=extra), patch('sys.stderr'), self.assertRaises(SystemExit):
                self.parse(fresh_arguments() + extra)

    def test_fresh_all_requires_all_three_stages_and_no_imitation_warmup(self):
        for extra in [['--warmup-updates', '1'], ['--fast-updates', '0'],
                      ['--slow-updates', '0'], ['--joint-updates', '0']]:
            with self.subTest(extra=extra), patch('sys.stderr'), self.assertRaises(SystemExit):
                self.parse(fresh_arguments() + extra)

    def test_world_updates_in_every_stage_and_policy_optimizer_matrix_is_explicit(self):
        config = trainer.make_initial_config(self.parse(fresh_arguments()))
        for phase, fast, slow in [('fast', True, False), ('slow', False, True), ('joint', True, True)]:
            self.assertEqual(trainer.phase_training_targets(config, phase),
                dict(world=True, fast_ppo=fast, slow_ppo=slow,
                     slow_imitation=False, fast_distillation=False))
        with self.assertRaises(ValueError):
            trainer.phase_training_targets(config, 'warmup')
        with self.assertRaises(ValueError):
            trainer.phase_training_targets(dict(config, freeze_fast_world=True), 'slow')
        # The separate frozen-fast/world mode retains its earlier semantics.
        self.assertEqual(trainer.phase_training_targets(dict(freeze_fast_world=True), 'slow'),
            dict(world=False, fast_ppo=False, slow_ppo=True, slow_imitation=False, fast_distillation=False))

    def test_random_fast_construction_reads_no_checkpoint_and_is_seed_reproducible(self):
        hashes = []
        with patch.object(torch, 'load', side_effect=AssertionError('fresh constructor read checkpoint')):
            for seed in (1729, 1729, 1730):
                torch.manual_seed(seed)
                model = policy.make_graph_fast(self.repo, random_init=True)
                hashes.append(state_hash(numpy_state(model)))
                np.testing.assert_allclose(model.base.log_std.detach().numpy(), -.5, atol=0.)
                self.assertEqual(float(model.base.value_mean), 0.)
                self.assertEqual(float(model.base.value_std), 1.)
                self.assertEqual(float(model.base.value_moments.abs().sum()), 0.)
        self.assertEqual(hashes[0], hashes[1])
        self.assertNotEqual(hashes[0], hashes[2])

    def test_all_initial_checkpoint_weights_come_from_fresh_constructors_and_learners_are_empty(self):
        captured = {}
        constructors = {name: getattr(trainer, attribute) for name, attribute in
            [('fast', 'make_graph_fast'), ('world', 'GraphWorldModel'), ('slow', 'GraphSlowActorCritic')]}

        def capture(name):
            def construct(*args, **kwargs):
                model = constructors[name](*args, **kwargs)
                captured[name] = numpy_state(model)
                return model
            return construct

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'fresh'
            output.mkdir()
            arguments = fresh_arguments(output) + ['--repo', str(self.repo), '--frozen']
            with patch.object(sys, 'argv', arguments), \
                    patch.object(trainer, 'make_graph_fast', side_effect=capture('fast')), \
                    patch.object(trainer, 'GraphWorldModel', side_effect=capture('world')), \
                    patch.object(trainer, 'GraphSlowActorCritic', side_effect=capture('slow')), \
                    patch.object(torch, 'load', side_effect=AssertionError('fresh main read checkpoint')), \
                    patch.object(trainer, 'ProcessPoolExecutor', side_effect=RuntimeError('initialization checked')), \
                    patch.object(trainer.signal, 'signal'), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'initialization checked'):
                    trainer.main()
            saved = torch.load(output / 'checkpoints' / 'initial.pt', map_location='cpu', weights_only=True)
            for name in ('fast', 'world', 'slow'):
                self.assertEqual(checkpoint_hash(saved[name]), state_hash(captured[name]))
                self.assertEqual(saved['manifest']['initialization']['initial_model_sha256'][name],
                                 state_hash(captured[name]))
            for name in ('actor_optimizer', 'critic_optimizer', 'world_optimizer', 'slow_optimizer'):
                self.assertEqual(saved[name]['state'], {})
            self.assertEqual(saved['completed_updates'], 0)
            self.assertEqual(saved['policy_version'], 0)
            self.assertTrue(all(value == 0 for value in saved['counters'].values()))
            self.assertEqual(saved['records'], [])
            self.assertEqual(saved['replay_paths'], [])
            self.assertFalse(saved['manifest']['initialization']['pretrained_weights_loaded'])
            self.assertFalse(saved['manifest']['initialization']['inherited_replay'])
            self.assertEqual(saved['manifest']['training_mode'], 'fresh_all_staged')
            self.assertFalse(saved['config']['freeze_fast_world'])
            self.assertEqual([group['lr'] for group in saved['actor_optimizer']['param_groups']],
                             [3e-4, 3e-4])
            self.assertEqual([group['lr'] for group in saved['manifest']['optimization']['optimizers']
                              ['fast_actor']['parameter_groups']], [3e-4, 3e-4])
            self.assertEqual(saved['manifest']['hierarchy_config']['option_steps'], saved['config']['option_steps'])
            self.assertIn('torch', saved['manifest']['environment'])
            for phase in ('fast', 'slow', 'joint'):
                self.assertEqual(saved['manifest']['phase_training_targets'][phase],
                                 trainer.phase_training_targets(saved['config'], phase))
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest, saved['manifest'])

    def test_spawn_collection_uses_random_fast_and_no_learned_teacher_or_checkpoint(self):
        torch.manual_seed(811)
        fast = policy.make_graph_fast(self.repo, random_init=True)
        world = trainer.GraphWorldModel(16, 4)
        slow = trainer.GraphSlowActorCritic(hidden=16)
        states = [numpy_state(model) for model in (fast, world, slow)]
        config = dict(hidden=16, horizon=4, option_steps=4, steps=4, stride=1,
                      loss=0., jitter=1, complete_missions=False, fresh_all=True,
                      training_mode='fresh_all_staged', random_slow_probability=0.)
        case = make_scene(58114, (100., 100.), 1000., 'open')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ('logs', 'data', 'episodes'):
                (output / name).mkdir()
            job = (str(self.repo), case, config, str(output), 'fresh-worker', *states, 0, 'fast', False)
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
                result = pool.submit(collect_without_checkpoint, job).result(timeout=60)
            for name, state in zip(('fast', 'world', 'slow'), states):
                self.assertEqual(result[name + '_sha256'], state_hash(state))
            with np.load(result['wm_path'], allow_pickle=False) as data:
                self.assertFalse(data['teacher_valid'].any())
                self.assertTrue(data['actor_valid'].any())
            with np.load(result['ppo_path'], allow_pickle=False) as data:
                self.assertEqual(len(data['rewards']), 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
