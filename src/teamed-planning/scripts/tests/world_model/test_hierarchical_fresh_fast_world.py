"""Fresh fast/world learning isolates rule subgoals from the untrained slow actor."""
import contextlib
from functools import wraps
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from world_model import graph_rollouts
from world_model import train_hierarchical as trainer
from world_model.graph_fast_policy import numpy_state, state_hash
from world_model.navigation_scenes import make_scene


def fresh_arguments(output='/tmp/fresh-fast-world-test'):
    return ['train_hierarchical.py', '--output', str(output), '--fresh-fast-world',
            '--warmup-updates', '0', '--fast-updates', '1', '--slow-updates', '0',
            '--joint-updates', '0', '--episodes-per-update', '1', '--eval-scenes', '1',
            '--hidden', '16']


def checkpoint_hash(state):
    return state_hash({name: value.detach().cpu().numpy() for name, value in state.items()})


class FreshFastWorldTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.repo = Path(__file__).resolve().parents[5]

    def parse(self, arguments):
        with patch.object(sys, 'argv', arguments):
            return trainer.parse_args()

    def config(self, extra=()):
        return trainer.make_initial_config(self.parse(fresh_arguments() + list(extra)))

    def test_fresh_fast_world_defaults_to_geometry_while_existing_modes_default_to_learned(self):
        config = self.config()
        self.assertEqual(config['training_mode'], 'fresh_fast_world')
        self.assertEqual(config['validation_slow_controller'], 'geometry')
        self.assertEqual(trainer.phase_schedule(config), ['fast'])
        old_arguments = ['train_hierarchical.py', '--output', '/tmp/legacy']
        fresh_all_arguments = old_arguments + ['--fresh-all', '--warmup-updates', '0',
            '--fast-updates', '1', '--slow-updates', '1', '--joint-updates', '1']
        for arguments in (old_arguments, fresh_all_arguments):
            with self.subTest(arguments=arguments):
                self.assertEqual(trainer.make_initial_config(self.parse(arguments))
                                 ['validation_slow_controller'], 'learned')

    def test_rejects_inherited_weights_learner_state_and_other_fresh_modes(self):
        for extra in [
            ['--fast-init', '/tmp/old.pt'], ['--graph-init', '/tmp/old.pt'],
            ['--slow-init', '/tmp/old.pt'], ['--resume'],
            ['--continue-from', '/tmp/old.pt', '--continuation-plan', '/tmp/plan.json'],
            ['--freeze-fast-world'], ['--fresh-slow'], ['--fresh-all']]:
            with self.subTest(extra=extra), patch('sys.stderr'), self.assertRaises(SystemExit):
                self.parse(fresh_arguments() + extra)

    def test_requires_exclusive_fast_stage_and_deterministic_rule_subgoals(self):
        for extra in [['--warmup-updates', '1'], ['--fast-updates', '0'],
                      ['--slow-updates', '1'], ['--joint-updates', '1'],
                      ['--validation-slow-controller', 'learned'],
                      ['--random-slow-probability', '.1']]:
            with self.subTest(extra=extra), patch('sys.stderr'), self.assertRaises(SystemExit):
                self.parse(fresh_arguments() + extra)
        self.assertEqual(self.config(['--validation-slow-controller', 'geometry'])
                         ['validation_slow_controller'], 'geometry')

    def test_update_targets_exclude_all_slow_learning_and_distillation(self):
        config = self.config()
        self.assertEqual(trainer.phase_training_targets(config, 'fast'),
            dict(world=True, fast_ppo=True, slow_ppo=False, slow_imitation=False,
                 fast_distillation=False))
        for phase in ('warmup', 'slow', 'joint'):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                trainer.phase_training_targets(config, phase)

    def test_initial_checkpoint_is_new_random_models_and_empty_learner_state(self):
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
            initialization = saved['manifest']['initialization']
            for name in ('fast', 'world', 'slow'):
                self.assertEqual(checkpoint_hash(saved[name]), state_hash(captured[name]))
                self.assertEqual(initialization['initial_model_sha256'][name], state_hash(captured[name]))
            for name in ('actor_optimizer', 'critic_optimizer', 'world_optimizer', 'slow_optimizer'):
                self.assertEqual(saved[name]['state'], {})
            self.assertEqual(saved['completed_updates'], 0)
            self.assertEqual(saved['policy_version'], 0)
            self.assertTrue(all(value == 0 for value in saved['counters'].values()))
            self.assertEqual(saved['records'], [])
            self.assertEqual(saved['replay_paths'], [])
            self.assertFalse(initialization['pretrained_weights_loaded'])
            self.assertFalse(initialization['inherited_replay'])
            self.assertEqual(initialization['frozen_models'], ['slow'])
            self.assertEqual(initialization['trained_models'], ['fast', 'world'])
            self.assertEqual(saved['manifest']['training_mode'], 'fresh_fast_world')
            self.assertEqual(saved['manifest']['validation_slow_controller'], 'geometry')
            self.assertEqual(saved['config']['validation_slow_controller'], 'geometry')
            self.assertEqual([group['lr'] for group in saved['actor_optimizer']['param_groups']],
                             [3e-4, 3e-4])
            self.assertEqual(json.loads((output / 'manifest.json').read_text()), saved['manifest'])

    def make_states(self, *, road_topology=False):
        torch.manual_seed(191)
        return [numpy_state(model) for model in
                (trainer.make_graph_fast(self.repo, random_init=True),
                 trainer.GraphWorldModel(16, 4, road_topology=road_topology),
                 trainer.GraphSlowActorCritic(hidden=16))]

    def rollout_config(self, extra=()):
        return dict(self.config(extra), horizon=4, option_steps=4, steps=4, stride=1,
                    loss=0., jitter=1, complete_missions=False, random_slow_probability=0.)

    def test_fixed_target_reaches_all_fast_actors_from_frame_zero_in_both_splits(self):
        states = self.make_states(road_topology=True)
        config = dict(self.rollout_config(['--fixed-mission-target']), steps=8, ground_start_hold_seconds=30)
        self.assertTrue(config['fixed_mission_target'])
        self.assertFalse(self.config()['fixed_mission_target'])
        original_step = graph_rollouts.GraphRecorder.step
        experiment = graph_rollouts.runtime(self.repo, random_init=True)[0].Experiment
        original_init = experiment.__init__

        @wraps(original_init)
        def check_v9_initialization(env, *args, **kwargs):
            original_init(env, *args, **kwargs)
            self.assertTrue(env.mapping_until_connected)
            self.assertIsNone(kwargs.get('subgoal_radius'))
            self.assertIsNone(env.subgoals)
            self.assertIsNone(env.ground_controller.subgoals)
            np.testing.assert_array_equal(env.reward_task_region, env.ledger.task_mask(env.start, env.goal))
            starts.append(np.asarray(env.goal).copy())

        def check_target(recorder, frame):
            env = recorder.env
            goal = np.asarray(recorder.episode.case['goal'])
            np.testing.assert_allclose(recorder.episode.manager.target, goal)
            np.testing.assert_allclose(env.ground_action['broadcast']['target'], goal)
            positions = np.asarray([env.obs[rid]['self']['position'][:2] for rid in env.ids])
            result = original_step(recorder, frame)
            features = recorder.transitions[-1]['features']
            np.testing.assert_allclose(features[:, 4:6], (goal - positions) / 350., atol=1e-6)
            np.testing.assert_array_equal(features[:, 10], np.ones(3))
            frames.append(frame)
            archived_rewards.append(result[1])
            return result

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ('logs', 'data', 'episodes'):
                (output / name).mkdir()
            with patch.object(experiment, '__init__', check_v9_initialization), \
                    patch.object(graph_rollouts.GraphRecorder, 'step', check_target), \
                    patch.object(graph_rollouts, 'geometry_teacher_action', side_effect=AssertionError('geometry selector ran')), \
                    patch.object(graph_rollouts.GraphSlowActorCritic, 'act', side_effect=AssertionError('slow actor ran')):
                for evaluation, seed in [(False, 58114), (True, 58115)]:
                    with self.subTest(evaluation=evaluation):
                        case = make_scene(seed, (275., 325.), 1000., 'open')
                        label = 'validation' if evaluation else 'train'
                        frames, starts, archived_rewards = [], [], []
                        row = graph_rollouts.collect_graph((str(self.repo), case, config, str(output),
                                                            label, *states, 0, 'fast', evaluation))
                        self.assertGreater(row['direct_distance_m'], 200.)
                        np.testing.assert_allclose(starts, [case['goal']])
                        self.assertEqual(frames, list(range(row['executed_steps'])))
                        self.assertGreater(len(frames), config['option_steps'])
                        self.assertEqual(row['slow_controller'], 'fixed_target')
                        self.assertEqual(row['slow_decisions'], 0)
                        self.assertFalse(row['slow_ppo_eligible'])
                        with np.load(row['wm_path'], allow_pickle=False) as data:
                            self.assertEqual(len(data['road_topology']), row['wm_samples'])
                            self.assertTrue(np.all(data['road_topology'][:, 7] == 1))
                        self.assertIsNone(row['slow_path'])
                        self.assertFalse(list((output / 'data').glob(f'{label}-slow*')))
                        self.assertEqual(row['fast_deterministic'], evaluation)
                        self.assertEqual(row['fast_controller'], 'learned')
                        self.assertEqual(row['fast_reward_contract'], 'uav_remaining_route_mapping_v9')
                        self.assertEqual(row['startup_hold_max_displacement_m'], 0.)
                        self.assertEqual(row['transitions'], config['steps'])
                        with np.load(row['ppo_path'], allow_pickle=False) as ppo:
                            np.testing.assert_array_equal(ppo['rewards'],
                                np.asarray(archived_rewards[:config['steps']], dtype=np.float32))
                        self.assertGreater(row['wm_samples'], 0)
                        self.assertGreaterEqual(row['world_calls']['uav'], 3 * row['transitions'])
                        self.assertGreaterEqual(row['world_calls']['ugv'], row['transitions'])
                        rollout = json.loads((output / 'episodes' / f'{label}.json').read_text())['rollout']
                        self.assertEqual(len(rollout['decisions']), 1)
                        np.testing.assert_allclose(rollout['decisions'][0]['target'], case['goal'])

    def test_fixed_target_ppo_preserves_v9_rewards_and_handles_terminal_outcomes(self):
        states = self.make_states(road_topology=True)
        config = dict(self.rollout_config(['--fixed-mission-target']), complete_missions=True)
        case = make_scene(58114, (3., 3.), 1000., 'open')
        goal = np.asarray(case['goal'])
        direction = (np.asarray(case['start']) - goal) / case['direct_distance_m']
        original_episode = graph_rollouts.HierarchicalEpisode
        step_rewards = [4.25, -0.5, 0.375, 7.5]

        def scripted_episode(*args, **kwargs):
            episode = original_episode(*args, **kwargs)
            env = episode.env
            episode.mission_budget = 4
            physics = env.step
            previous_samples = None

            def step(frame):
                nonlocal previous_samples
                if frame == 3 and outcome == 'rejected_action':
                    raise episode.me.SafetyTermination(1, previous_samples)
                samples, _, diagnostics = physics(frame)
                previous_samples = samples
                ground = env.obs[env.ground]['self']
                position = np.array(ground['position'])
                position[:2] = goal + direction * distances[frame + 1]
                ground['position'] = position
                env.goal_reached = frame == 3 and outcome in ('mission_arrival', 'unsafe_arrival')
                env.trace[-1]['ground_collisions'] = int(frame == 2)
                if frame == 3 and outcome in ('safety_failure', 'unsafe_arrival'):
                    env.env.ground_safety_violations = getattr(env.env, 'ground_safety_violations', 0) + 1
                return samples, step_rewards[frame], diagnostics

            env.step = step
            return episode

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ('logs', 'data', 'episodes'):
                (output / name).mkdir()
            with patch.object(graph_rollouts, 'HierarchicalEpisode', scripted_episode):
                for outcome, termination, terminal_reward in [
                        ('mission_arrival', 'mission_arrival', step_rewards[-1] + 100.),
                        ('safety_failure', 'safety_failure', -100.),
                        ('unsafe_arrival', 'safety_failure', -100.),
                        ('rejected_action', 'safety_failure', -100.),
                        ('mission_budget', 'mission_budget', step_rewards[-1])]:
                    with self.subTest(outcome=outcome):
                        arrived = outcome in ('mission_arrival', 'unsafe_arrival')
                        distances = [3., 2.8, 3., 3., 1.95 if arrived else 3.]
                        row = graph_rollouts.collect_graph((str(self.repo), case, config, str(output),
                                                            outcome, *states, 0, 'fast', False))
                        self.assertEqual(row['termination'], termination)
                        with np.load(row['ppo_path'], allow_pickle=False) as fast:
                            np.testing.assert_array_equal(fast['rewards'], [*step_rewards[:-1], terminal_reward])
                            np.testing.assert_array_equal(fast['terminals'], [False, False, False, True])
                            self.assertEqual(float(fast['bootstrap']), 0.)

    def test_fixed_target_replaces_v9_reward_when_physics_rejects_unknown_terrain(self):
        from tests.world_model.test_hierarchical_wildscenes_runtime import case_for
        states = self.make_states(road_topology=True)
        config = dict(self.rollout_config(['--fixed-mission-target']), steps=8)
        original_episode = graph_rollouts.HierarchicalEpisode

        def drive_into_unknown(*args, **kwargs):
            episode = original_episode(*args, **kwargs)
            ground_action = episode._base_ground_action
            episode._base_ground_action = lambda neighbors=None: dict(ground_action(neighbors), motion=(1., 0.))
            return episode

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ('logs', 'data', 'episodes'):
                (output / name).mkdir()
            known = np.ones((200, 200), bool)
            known[96:104, 83:87] = False
            path = output / 'forest.npz'
            np.savez_compressed(path, known=known, occupied=np.zeros_like(known),
                                resolution_m=1., origin_xy=np.zeros(2))
            case = case_for(path)
            with patch.object(graph_rollouts, 'HierarchicalEpisode', drive_into_unknown), \
                    patch('world_model.wildscenes_runtime.filter_ground_motion',
                          side_effect=lambda action, *args, **kwargs: (action, {'intervened': False})):
                row = graph_rollouts.collect_graph((str(self.repo), case, config, str(output), 'unknown',
                                                    *states, 0, 'fast', True))
            self.assertEqual(row['termination'], 'safety_failure')
            self.assertEqual(row['raster_unknown_attempts'], 1)
            self.assertEqual(row['ground_safety_violations'], 1)
            with np.load(row['ppo_path'], allow_pickle=False) as ppo:
                self.assertTrue(ppo['terminals'][-1])
                self.assertEqual(float(ppo['rewards'][-1]), -100.)
                self.assertEqual(float(ppo['bootstrap']), 0.)

    def test_geometry_validation_is_deterministic_and_independent_of_slow_parameters(self):
        states = self.make_states()
        changed_slow = {key: np.asarray(value + np.float32(.5)).copy() if value.dtype.kind == 'f' else value.copy()
                        for key, value in states[2].items()}
        self.assertNotEqual(state_hash(states[2]), state_hash(changed_slow))
        case = make_scene(58114, (100., 100.), 1000., 'open')
        results, arrays = [], []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ('logs', 'data', 'episodes'):
                (output / name).mkdir()
            with patch.object(torch, 'load', side_effect=AssertionError('geometry rollout read checkpoint')), \
                    patch.object(graph_rollouts, 'TeacherBehavior', side_effect=AssertionError('learned teacher used')), \
                    patch.object(graph_rollouts.GraphSlowActorCritic, 'act', side_effect=AssertionError('slow actor used')), \
                    patch.object(graph_rollouts.GraphSlowActorCritic, 'value', side_effect=AssertionError('slow critic used')), \
                    patch.object(graph_rollouts.GraphSlowActorCritic, 'forward', side_effect=AssertionError('slow network used')):
                for index, slow in enumerate((states[2], changed_slow)):
                    torch.manual_seed(433 + index)
                    result = graph_rollouts.collect_graph((str(self.repo), case, self.rollout_config(),
                        str(output), f'geometry-{index}', states[0], states[1], slow, 0, 'fast', True))
                    results.append(result)
                    self.assertEqual(result['slow_controller'], 'geometry')
                    self.assertFalse(result['slow_ppo_eligible'])
                    self.assertTrue(result['fast_deterministic'])
                    self.assertEqual(result['fast_controller'], 'learned')
                    with np.load(result['ppo_path'], allow_pickle=False) as data:
                        arrays.append({key: data[key].copy() for key in data.files})
                    with np.load(result['wm_path'], allow_pickle=False) as data:
                        self.assertFalse(data['teacher_valid'].any())
                    with np.load(result['slow_path'], allow_pickle=False) as data:
                        self.assertFalse(data['ppo_eligible'].any())
                        np.testing.assert_array_equal(data['action'], data['teacher_action'])
                    with self.assertRaisesRegex(ValueError, 'cannot consume'):
                        trainer.slow_episode(result['slow_path'], require_ppo=True)
                # The training collector also bypasses the unused slow network.
                train = graph_rollouts.collect_graph((str(self.repo), case, self.rollout_config(),
                    str(output), 'geometry-train', *states, 0, 'fast', False))
                self.assertEqual(train['slow_controller'], 'geometry')
                self.assertFalse(train['slow_ppo_eligible'])
                self.assertFalse(train['fast_deterministic'])
                with self.assertRaisesRegex(ValueError, 'cannot consume'):
                    trainer.slow_episode(train['slow_path'], require_ppo=True)
            for key in ('termination', 'remaining_m', 'executed_steps', 'slow_decisions'):
                self.assertEqual(results[0][key], results[1][key])
            self.assertGreater(len(arrays[0]['actions']), 0)
            for key in arrays[0]:
                np.testing.assert_array_equal(arrays[0][key], arrays[1][key], err_msg=key)

    def test_fresh_all_default_validation_still_executes_learned_slow(self):
        states = self.make_states()
        config = dict(self.rollout_config(), fresh_fast_world=False, fresh_all=True,
                      training_mode='fresh_all_staged', validation_slow_controller='learned')
        case = make_scene(58114, (100., 100.), 1000., 'open')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ('logs', 'data', 'episodes'):
                (output / name).mkdir()
            with patch.object(graph_rollouts.GraphSlowActorCritic, 'act',
                              side_effect=RuntimeError('learned validation invoked')):
                with self.assertRaisesRegex(RuntimeError, 'learned validation invoked'):
                    graph_rollouts.collect_graph((str(self.repo), case, config,
                        str(output), 'legacy-validation', *states, 0, 'fast', True))

    def test_slow_ppo_loader_rejects_geometry_and_evaluation_but_retains_legacy_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'decisions.npz'
            base = dict(action=np.asarray([0]), duration=np.asarray([4]))
            for controller in (0, 1):
                np.savez(path, **base, controller_code=np.asarray([controller]),
                         ppo_eligible=np.asarray([False]))
                with self.subTest(controller=controller), self.assertRaisesRegex(ValueError, 'cannot consume'):
                    trainer.slow_episode(path, require_ppo=True)
                self.assertEqual(len(trainer.slow_episode(path)), 1)
            np.savez(path, **base, controller_code=np.asarray([1]), ppo_eligible=np.asarray([True]))
            self.assertEqual(len(trainer.slow_episode(path, require_ppo=True)), 1)
            np.savez(path, **base)
            self.assertEqual(len(trainer.slow_episode(path, require_ppo=True)), 1)

    def test_real_one_batch_updates_fast_world_only_and_finishes_geometry_validation(self):
        self.check_one_batch(fixed_target=False)

    def test_real_one_batch_updates_fast_world_only_and_finishes_fixed_target_validation(self):
        self.check_one_batch(fixed_target=True)

    def check_one_batch(self, fixed_target):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'tiny-run'
            output.mkdir()
            minimum_distance = '250' if fixed_target else '100'
            arguments = fresh_arguments(output) + ['--repo', str(self.repo), '--frozen',
                '--workers', '1', '--steps', '8', '--option-steps', '4', '--stride', '1',
                '--no-complete-missions', '--wm-epochs', '1', '--ppo-epochs', '1',
                '--warmup-distances', minimum_distance, '400', '--mission-distances', minimum_distance, '400']
            if fixed_target:
                arguments.append('--fixed-mission-target')
            with patch.object(sys, 'argv', arguments), patch.object(trainer.signal, 'signal'), \
                    patch.object(trainer, 'slow_update', side_effect=AssertionError('slow PPO ran')), \
                    patch.object(trainer, 'slow_behavior_clone', side_effect=AssertionError('slow BC ran')), \
                    patch.object(trainer, 'distill_fast', side_effect=AssertionError('teacher distillation ran')), \
                    contextlib.redirect_stdout(io.StringIO()):
                trainer.main()
            initial = torch.load(output / 'checkpoints' / 'initial.pt', map_location='cpu', weights_only=True)
            final = torch.load(output / 'checkpoints' / 'final.pt', map_location='cpu', weights_only=True)
            for name in ('fast', 'world'):
                self.assertNotEqual(checkpoint_hash(initial[name]), checkpoint_hash(final[name]), name)
            self.assertEqual(checkpoint_hash(initial['slow']), checkpoint_hash(final['slow']))
            self.assertEqual(final['slow_optimizer']['state'], {})
            self.assertGreater(final['counters']['fast_ppo_optimizer_updates'], 0)
            self.assertGreater(final['counters']['world_optimizer_updates'], 0)
            for key in ('slow_optimizer_updates', 'slow_imitation_optimizer_updates',
                        'slow_imitation_transitions', 'fast_distillation_optimizer_updates'):
                self.assertEqual(final['counters'][key], 0, key)
            status = json.loads((output / 'status.json').read_text())
            self.assertEqual((status['state'], status['phase']), ('complete', 'finished'))
            self.assertIsNone(final['pending_evaluation'])
            self.assertEqual(final['completed_updates'], 1)
            self.assertEqual([row['evaluation'] for row in final['records']], [False, True])
            if fixed_target:
                reported = json.loads((output / 'manifest.json').read_text())
                for manifest in (initial['manifest'], final['manifest'], reported):
                    self.assertEqual(manifest['optimization']['fast_ppo']['reward_contract'],
                                     'uav_remaining_route_mapping_v9')
            for row in final['records']:
                self.assertEqual(row['slow_controller'], 'fixed_target' if fixed_target else 'geometry')
                self.assertFalse(row['slow_ppo_eligible'])
                if fixed_target:
                    self.assertGreater(row['direct_distance_m'], 200.)
                    self.assertIsNone(row['slow_path'])
            self.assertTrue((output / 'metrics' / 'validation-001.json').exists())
            self.assertTrue((output / 'checkpoints' / 'update-001.pt').exists())
            metrics = json.loads((output / 'metrics' / 'update-001.json').read_text())
            self.assertGreater(metrics['world']['optimizer_steps'], 0)
            self.assertNotIn('slow_ppo', metrics)
            self.assertNotIn('slow_imitation', metrics)


if __name__ == '__main__':
    unittest.main(verbosity=2)
