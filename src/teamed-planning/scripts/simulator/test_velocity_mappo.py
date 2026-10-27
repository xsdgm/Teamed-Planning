"""Continuous-policy integration: no candidates, real 2D execution, PPO and resume."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from drone.mappo import ActorCritic, CandidateNavigator, GLOBAL_FEATURES
from drone.drone_navigator import DroneNavigator
from drone.velocity_mappo import VelocityActorCritic, VelocityNavigator, VELOCITY_FEATURES
from simulator.mappo_experiment import Experiment, update, velocity_task_reward
from simulator.train_mappo import train, validate_config, run_episode, fixed_schedule

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT/'src/teamed-planning/config/algorithm/mappo_training_v3_velocity_le350.json'


class VelocityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_config_keeps_three_drones_and_limits_every_scene(self):
        c = validate_config(json.loads(CONFIG.read_text()))
        self.assertEqual(c['task_distances'], [250, 300, 350])
        self.assertEqual(c['reward_mode'], 'velocity_mapping')
        self.assertEqual(c['fleet_sizes'], [3])
        self.assertEqual((c['time_penalty'], c['intervention_weight']), (.01, .05))
        # Reward v4: no corridor exploration, CBF margin instead of the chain terms,
        # route quality paid only by best-route gains during the improvement phase.
        # v5: goal-directed frontier progress, lighter margin penalty, decaying entropy.
        self.assertEqual(c['exploration_weight'], 0.)
        self.assertEqual(c['frontier_progress_weight'], 50.)
        self.assertEqual((c['entropy_coefficient'], c['entropy_coefficient_final']), (.003, .001))
        self.assertEqual((c['arrival_reward'], c['safety_failure_penalty']), (0., 100.))
        self.assertEqual(c['termination'], 'map')
        self.assertEqual(c['map_success_reward'],100.)
        self.assertEqual((c['path_quality_weight'],c['map_quality_reward']),(20.,0.))
        self.assertEqual((c['chain_risk_weight'],c['chain_outward_weight'],c['chain_missing_weight']),(0.,0.,0.))
        self.assertEqual((c['cbf_margin_weight'],c['cbf_margin_scale']),(.1,8.))
        self.assertEqual((c['map_improvement_steps'],c['map_target_quality']),(150,.95))
        self.assertTrue(c['mapping_reward_until_connected'])
        self.assertEqual((c['critic_epochs'],c['critic_minibatch_size'],c['value_normalization']),(8,512,True))
        for delta in [dict(task_distances=[400]), dict(subgoal_radius=200.),
                      dict(exploration_candidates=5), dict(init_checkpoint=3)]:
            with self.assertRaises(ValueError):
                validate_config(dict(c, **delta))

    def test_speed_disk_and_submaximum_commands(self):
        actions = np.array([[0., 0.], [.01, .02], [1., 1.], [1e6, -1e6]])
        velocities = VelocityActorCritic.velocity(actions)
        self.assertTrue(np.all(np.linalg.norm(velocities, axis=-1) <= 6.))
        np.testing.assert_array_equal(velocities[0], [0., 0.])
        self.assertLess(np.linalg.norm(velocities[1]), 1.)
        self.assertGreater(np.linalg.norm(velocities[2]), 1.)

    def test_reward_ignores_revisits_and_off_task_exploration(self):
        old = np.zeros((2,2), bool)
        seen = np.ones((2,2), bool)
        corridor = np.array([[True, False], [False, False]])
        reward, terms = velocity_task_reward(100., 90., old, seen, corridor)
        self.assertAlmostEqual(reward, .1+.2*4/400-.01)
        self.assertEqual(terms['task_exploration_area'], 4.)
        repeat, _ = velocity_task_reward(90., 90., seen, seen, corridor)
        self.assertEqual(repeat, -.01)
        outside = seen.copy(); outside[0,0] = False
        reward, _ = velocity_task_reward(90., 90., old, outside, corridor)
        self.assertEqual(reward, -.01)

    def test_same_weights_support_unseen_fleet_sizes_without_candidates(self):
        torch.manual_seed(17)
        model = VelocityActorCritic()
        before = {k:v.clone() for k,v in model.state_dict().items()}
        with patch.object(DroneNavigator, '_generate_candidates', side_effect=AssertionError('candidate called')), \
             contextlib.redirect_stdout(io.StringIO()):
            for count in (3, 4, 5, 6):
                env = Experiment(10021, model, stochastic=False, num_drones=count)
                try:
                    for step in range(8):
                        samples, _, _ = env.step(step)
                        self.assertEqual(len(samples), count)
                        self.assertTrue(all(s[0].shape == (VELOCITY_FEATURES,) for s in samples))
                    summary = env.summary()
                    self.assertEqual(summary['num_drones'], count)
                    self.assertEqual(summary['collisions'], 0.)
                    self.assertEqual(summary['chain_link_violation'], 0.)
                    xs = torch.tensor(np.array([s[0] for s in samples]))
                    self.assertEqual(model.distribution(xs).sample().shape, (count,2))
                    self.assertEqual(model.value(xs, state=torch.tensor(env.critic_state(8,100))).shape, ())
                finally:
                    env.env.close()
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))

    def test_neighbor_pool_is_permutation_invariant(self):
        nav = VelocityNavigator('drone_1', model=VelocityActorCritic(), total_drones=4)
        peers = {'drone_0':dict(position=[10.,10.,15.]),
                 'drone_2':dict(position=[40.,10.,15.]),
                 'drone_3':dict(position=[60.,10.,15.])}
        first, _ = nav.encode([25.,10.,15.], dict(nearby_drones=peers))
        second, _ = nav.encode([25.,10.,15.], dict(nearby_drones=dict(reversed(list(peers.items())))))
        np.testing.assert_array_equal(first, second)

    def test_observation_uses_received_information_only(self):
        model = VelocityActorCritic()
        nav = VelocityNavigator('drone_0', model=model, total_drones=3)
        obs = dict(time=0., occupancy_map=np.zeros((12,12)), sensor_valid=True,
                   ground_robots={'g':dict(position=[10.,10.,0.],target=[100.,10.])})
        a, _ = nav.encode([20.,10.,15.], obs)
        b, _ = nav.encode([20.,10.,15.], dict(obs, full_state={'secret':1e9},
                                           reference_route=[1e9], obstacles=[1e9]))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (VELOCITY_FEATURES,))
        unknown, _ = nav.encode([20.,10.,15.], dict(obs, sensor_valid=False))
        self.assertFalse(np.array_equal(a, unknown))

    def test_real_2d_no_candidate_calls_logprob_and_ppo(self):
        torch.manual_seed(83)
        model = VelocityActorCritic()
        config = validate_config(dict(action_mode='velocity', exploration_candidates=0,
            task_distances=[250], steps=32, initial_validation=False))
        with patch.object(DroneNavigator, '_generate_candidates', side_effect=AssertionError('candidate called')), \
             patch.object(CandidateNavigator, 'encode', side_effect=AssertionError('candidate encoded')), \
             contextlib.redirect_stdout(io.StringIO()):
            rollout, summary = run_episode(model, config, 10021, fixed_schedule(config, 0), True)
        self.assertEqual(len(rollout), 32)
        self.assertEqual(summary['valid_actions'], 0.)
        self.assertGreater(summary['executed_speed_mean'], 0.)
        self.assertEqual(summary['collisions'], 0.)
        self.assertEqual(summary['chain_link_violation'], 0.)
        for x, mask, actions, logp, reward, state in rollout:
            actual = model.distribution(torch.tensor(x)).log_prob(torch.tensor(actions)).detach().numpy()
            np.testing.assert_allclose(actual, logp, atol=1e-5)
        before = model.velocity_actor[-1].weight.detach().clone()
        _, diagnostics = update(model, torch.optim.Adam(model.parameters(),lr=3e-4),
            rollout, epochs=2, return_metrics=True)
        self.assertTrue(all(np.isfinite(v) for v in diagnostics.values()))
        self.assertFalse(torch.equal(before, model.velocity_actor[-1].weight))
        restored = ActorCritic.from_state_dict(model.state_dict())
        self.assertIsInstance(restored, VelocityActorCritic)

    def test_controller_preserves_small_velocity_before_cbf(self):
        model = VelocityActorCritic()
        with torch.no_grad():
            for p in model.velocity_actor.parameters():
                p.zero_()
            model.velocity_actor[-1].bias.copy_(torch.tensor([.01, .02]))
        with contextlib.redirect_stdout(io.StringIO()):
            env = Experiment(10021, model, stochastic=False)
            try:
                controller = env.controllers[0]
                with patch.object(controller, '_safe_simulation_velocity', wraps=controller._safe_simulation_velocity) as cbf:
                    env.step(0)
                command = np.asarray(cbf.call_args.args[0])
                np.testing.assert_allclose(command[:2], model.velocity([.01,.02]), atol=1e-7)
                self.assertIsNone(controller.navigator._current_target)
            finally:
                env.env.close()

    def test_real_parallel_checkpoint_stop_and_resume(self):
        base = validate_config(dict(action_mode='velocity', exploration_candidates=0,
            task_distances=[250], episodes=2, steps=8, validation_interval=2,
            snapshot_interval=2, validation_seeds=[20000], validation_samples=1,
            test_seeds=[30000], initial_validation=False))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'STOP').touch()
            with contextlib.redirect_stdout(io.StringIO()):
                train(base, root, workers=2)
            self.assertEqual(json.loads((root/'progress.json').read_text())['status'], 'stopped')
            (root/'STOP').unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                train(base, root, resume=True, workers=2)
            saved = torch.load(root/'latest.pt', map_location='cpu', weights_only=True)
            self.assertEqual(saved['action_mode'], 'velocity')
            self.assertEqual(saved['feature_version'], 2)
            self.assertEqual(saved['training_episodes'], 2)
            self.assertIsInstance(ActorCritic.from_state_dict(saved['model']), VelocityActorCritic)
            self.assertTrue((root/'best.pt').exists())
            self.assertEqual(json.loads((root/'progress.json').read_text())['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
