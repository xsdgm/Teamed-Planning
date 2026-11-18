"""Continuous-policy observations and closed-loop velocity control."""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from drone.drone_navigator import DroneNavigator
from drone.graph_attention_mappo import GAT_FEATURES, GraphAttentionActorCritic
from drone.velocity_mappo import VelocityNavigator, bounded_velocity
from simulator.mappo_experiment import Experiment


class VelocityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_speed_disk_and_submaximum_commands(self):
        actions = np.array([[0., 0.], [.01, .02], [1., 1.], [1e6, -1e6]])
        velocities = bounded_velocity(actions)
        self.assertTrue(np.all(np.linalg.norm(velocities, axis=-1) <= 6.))
        np.testing.assert_array_equal(velocities[0], [0., 0.])
        self.assertLess(np.linalg.norm(velocities[1]), 1.)
        self.assertGreater(np.linalg.norm(velocities[2]), 1.)


    def test_same_weights_support_unseen_fleet_sizes_without_candidates(self):
        torch.manual_seed(17)
        model = GraphAttentionActorCritic()
        before = {k:v.clone() for k,v in model.state_dict().items()}
        with patch.object(DroneNavigator, '_generate_candidates', side_effect=AssertionError('candidate called')), \
             contextlib.redirect_stdout(io.StringIO()):
            for count in (3, 4, 5, 6):
                env = Experiment(10021, model, stochastic=False, num_drones=count)
                try:
                    for step in range(8):
                        samples, _, _ = env.step(step)
                        self.assertEqual(len(samples), count)
                        self.assertTrue(all(s[0].shape == (GAT_FEATURES,) for s in samples))
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
        nav = VelocityNavigator('drone_1', model=GraphAttentionActorCritic(), total_drones=4)
        peers = {'drone_0':dict(position=[10.,10.,15.]),
                 'drone_2':dict(position=[40.,10.,15.]),
                 'drone_3':dict(position=[60.,10.,15.])}
        first, _ = nav.encode([25.,10.,15.], dict(nearby_drones=peers))
        second, _ = nav.encode([25.,10.,15.], dict(nearby_drones=dict(reversed(list(peers.items())))))
        np.testing.assert_array_equal(first, second)

    def test_observation_uses_received_information_only(self):
        model = GraphAttentionActorCritic()
        nav = VelocityNavigator('drone_0', model=model, total_drones=3)
        obs = dict(time=0., occupancy_map=np.zeros((12,12)), sensor_valid=True,
                   ground_robots={'g':dict(position=[10.,10.,0.],target=[100.,10.])})
        a, _ = nav.encode([20.,10.,15.], obs)
        b, _ = nav.encode([20.,10.,15.], dict(obs, full_state={'secret':1e9},
                                           reference_route=[1e9], obstacles=[1e9]))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (GAT_FEATURES,))
        unknown, _ = nav.encode([20.,10.,15.], dict(obs, sensor_valid=False))
        self.assertFalse(np.array_equal(a, unknown))


    def test_controller_preserves_small_velocity_before_cbf(self):
        model = GraphAttentionActorCritic()
        with torch.no_grad():
            for p in model.gat_actor.parameters():
                p.zero_()
            model.gat_actor[-1].bias.copy_(torch.tensor([.01, .02]))
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


if __name__ == '__main__':
    unittest.main()
