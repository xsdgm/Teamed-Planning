"""An initial CBF rejection is terminal without an invented physics step."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from common.graph_structure import TopologicalGraph
from drone.cbf_safety import CBFSafetyFilter
from drone.velocity_mappo import VelocityActorCritic
from simulator.mappo_experiment import Experiment
from simulator import train_mappo


class EmptyExperimentSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def experiment(self, vehicle_mapping=True):
        torch.manual_seed(19)
        with contextlib.redirect_stdout(io.StringIO()):
            env = Experiment(10021, VelocityActorCritic(), stochastic=False,
                world_size=120., start=(15., 60.), scene=((100., 60.), [], 1),
                reward_mode='velocity_mapping' if vehicle_mapping else 'cadmm_exploration',
                mapping_route_origin='vehicle' if vehicle_mapping else 'initial',
                subgoal_radius=None)
        self.addCleanup(env.env.close)
        return env

    def assert_finite_summary(self, summary):
        # The training JSONL writer rejects NaN and Infinity as well.
        json.dumps(summary, allow_nan=False)
        self.assertTrue(all(np.isfinite(value) for value in summary.values()))

    def test_empty_summary_preserves_initial_state_and_normal_schema(self):
        for vehicle_mapping in (False, True):
            with self.subTest(vehicle_mapping=vehicle_mapping):
                env = self.experiment(vehicle_mapping)
                position = np.asarray(env.obs[env.ground]['self']['position']).copy()
                with warnings.catch_warnings():
                    warnings.simplefilter('error', RuntimeWarning)
                    summary = env.summary()
                self.assert_finite_summary(summary)
                self.assertEqual(summary['episode_steps'], 0)
                for key in ('distance', 'ground_distance', 'reward', 'ground_progress',
                            'min_step_reward', 'max_step_reward', 'cbf_intervention'):
                    self.assertEqual(summary[key], 0., key)
                self.assertEqual(summary['ground_goal_distance'], 85.)
                self.assertEqual(summary['graph_frontier_goal_distance_m'], 85.)
                self.assertEqual(summary['min_separation'], 18.)
                self.assertEqual(summary['graph_first_connection_step'], -1.)
                self.assertEqual(env.trace, [])
                self.assertIsNone(env._graph_metrics)
                self.assertEqual(env.env.step_count, 0)
                self.assertEqual(env.env.time, 0.)
                np.testing.assert_array_equal(env.obs[env.ground]['self']['position'], position)

                # The first real step must retain the same summary contract.
                with contextlib.redirect_stdout(io.StringIO()):
                    env.step(0)
                after = env.summary()
                self.assert_finite_summary(after)
                self.assertEqual(set(summary), set(after))
                self.assertEqual(after['episode_steps'], 1)
                self.assertEqual(after['reward'], env.trace[0]['reward'])
                self.assertEqual(after['ground_distance'], env.trace[0]['ground_distance'])
                self.assertEqual(after['min_step_reward'], env.trace[0]['reward'])
                self.assertEqual(after['max_step_reward'], env.trace[0]['reward'])
                self.assertEqual(after['min_separation'], env.trace[0]['min_separation'])

    def test_empty_summary_measures_the_known_graph_without_recording_a_step(self):
        env = self.experiment()
        graph = TopologicalGraph('received')
        graph.add_node('vehicle', (15., 60.))
        graph.add_node('goal', (100., 60.))
        graph.add_edge('vehicle', 'goal')
        env.ground_controller.unified_graph = graph
        env.ledger.receive('fixture', 0., np.array([60., 60.]), np.zeros((60, 60)), 2.)
        summary = env.summary()
        self.assert_finite_summary(summary)
        self.assertEqual(summary['graph_task_connected'], 1.)
        self.assertEqual(summary['graph_task_path_length'], 85.)
        self.assertEqual(summary['graph_nodes'], 2.)
        self.assertEqual(summary['current_route_length_m'], 85.)
        self.assertEqual(summary['route_reference_length_m'], 85.)
        self.assertEqual(summary['route_quality'], 1.)
        self.assertEqual(summary['route_target_switches'], 0.)
        self.assertEqual(summary['route_length_delta_m'], 0.)
        np.testing.assert_array_equal([summary['route_origin_x'], summary['route_origin_y']], [15., 60.])
        np.testing.assert_array_equal([summary['route_target_x'], summary['route_target_y']], [100., 60.])
        self.assertEqual(env.trace, [])
        self.assertIsNone(env._graph_metrics)
        self.assertEqual(env.env.step_count, 0)

    def test_first_rejected_action_is_one_terminal_transition_and_zero_physics_steps(self):
        config_path = Path(__file__).resolve().parents[3]/'config/algorithm/mappo_training_v9.json'
        base = json.loads(config_path.read_text())
        scene = dict(start=(15., 60.), goal=(100., 60.), circles=[], attempts=1,
                     reference_length=85., reference_polyline_length=85.)
        original_filter = CBFSafetyFilter.filter

        def reject_first_action(safety_filter, *args, **kwargs):
            velocity, stats = original_filter(safety_filter, *args, **kwargs)
            stats['infeasible'] = True
            return velocity, stats

        config = train_mappo.validate_config(dict(base, steps=8, min_episode_steps=8))
        torch.manual_seed(53)
        model = train_mappo.make_model(config)
        with patch.object(train_mappo, 'generate_training_scene', return_value=scene), \
             patch.object(CBFSafetyFilter, 'filter', reject_first_action), \
             patch('simulator.mappo_experiment.MultiRobotEnv.step',
                   side_effect=AssertionError('Rejected action must not execute physics')) as physics, \
             warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            rollout, summary = train_mappo.run_episode(model, config, 60002,
                train_mappo.fixed_schedule(config, 0), stochastic=True)
        physics.assert_not_called()
        self.assert_finite_summary(summary)
        self.assertEqual(len(rollout), 1)
        self.assertEqual(summary['rollout_steps'], 1)
        self.assertEqual(summary['episode_steps'], 0)
        self.assertEqual(summary['safety_terminated'], 1.)
        self.assertEqual(summary['budget_exhausted'], 0.)
        self.assertEqual(summary['goal_reached'], 0.)
        self.assertEqual(summary['ground_goal_distance'], 85.)
        self.assertEqual(summary['ground_distance'], 0.)
        self.assertEqual(summary['distance'], 0.)
        self.assertEqual(len(rollout[0][2]), 3)
        self.assertEqual(summary['reward'], rollout[0][-2])
        self.assertLess(summary['reward'], 0.)
        self.assertIn('relay_packets_sent', summary)
        self.assertEqual(summary['reward'], -config['safety_failure_penalty'])


if __name__ == '__main__':
    unittest.main()
