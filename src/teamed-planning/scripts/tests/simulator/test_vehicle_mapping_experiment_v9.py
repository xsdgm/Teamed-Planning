"""Exercise v9 map reward timing through real physics and packet delivery."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from common.graph_structure import TopologicalGraph
from drone.graph_attention_mappo import GraphAttentionActorCritic
from simulator.mappo_experiment import Experiment, navigation_cost
from simulator.navigation_metrics import task_graph_metrics


class VehicleMappingExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def experiment(self, detour=False, subgoal_radius=200.):
        torch.manual_seed(19)
        with contextlib.redirect_stdout(io.StringIO()):
            env = Experiment(10021, GraphAttentionActorCritic(), stochastic=False,
                world_size=120., start=(15., 60.), scene=((100., 60.), [], 1),
                reward_mode='velocity_mapping', mapping_route_origin='vehicle',
                subgoal_radius=subgoal_radius, path_quality_weight=20., mapping_until_connected=True,
                frontier_progress_weight=45., exploration_weight=0., time_penalty=0.)
        self.addCleanup(env.env.close)
        # The real differential-drive physics advances the vehicle 0.75 m.
        env.ground_action['motion'] = (.75, 0.)
        graph = TopologicalGraph('received')
        points = dict(vehicle=(15., 60.), goal=(100., 60.), next_goal=(80., 80.), detour=(50., 85.))
        for key, value in points.items():
            graph.add_node(key, value)
        graph.add_edge('vehicle', 'next_goal')
        if detour:
            graph.add_edge('vehicle', 'detour')
            graph.add_edge('detour', 'goal')
        else:
            graph.add_edge('vehicle', 'goal')
        env.ground_controller.unified_graph = graph
        # Synthetic free evidence makes graph endpoint access independently
        # inspectable; these pixels are never sent to the policy as map truth.
        env.ledger.receive('fixture', 0, np.array([60., 60.]), np.zeros((60, 60)), 2.)
        return env, graph

    def metric_recorder(self, env):
        calls = []
        def measure(graph, start, target, connector):
            result = task_graph_metrics(graph, start, target, connector)
            calls.append(dict(graph=graph, start=np.asarray(start).copy(),
                target=np.asarray(target).copy(), result=dict(result),
                start_seen=connector(np.asarray(start), np.asarray(start))))
            return result
        return calls, measure

    def test_fixed_goal_without_subgoals_still_uses_current_vehicle_origin(self):
        env, graph = self.experiment(subgoal_radius=None)
        self.assertIsNone(env.subgoals)
        calls, measure = self.metric_recorder(env)
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(env, '_ground_action', return_value=dict(env.ground_action)), \
             patch('simulator.mappo_experiment.task_graph_metrics', side_effect=measure):
            _, reward, metrics = env.step(0)
        position = np.asarray(env.obs[env.ground]['self']['position'])[:2]
        np.testing.assert_allclose(position, (15.75, 60.))
        self.assertEqual(len(calls), 2)
        for call in calls:
            np.testing.assert_array_equal(call['start'], position)
            np.testing.assert_array_equal(call['target'], env.goal)
        self.assertEqual(metrics['route_target_changed'], 0.)
        self.assertEqual(metrics['route_quality_gain'], 0.)
        self.assertEqual(reward, 0.)
        self.assertAlmostEqual(metrics['route_reference_length_m'], np.linalg.norm(env.goal-position))
        state = env.critic_state(1, 50)
        self.assertAlmostEqual(float(state[5]), np.linalg.norm(env.goal-position)/env.initial_goal_distance, places=6)

    def test_movement_and_switch_use_frozen_reward_endpoints_and_active_display(self):
        env, graph = self.experiment()
        old_target = env.subgoals.target.copy()
        next_target = np.array([80., 80.])
        action = dict(env.ground_action)
        calls, measure = self.metric_recorder(env)
        def switch_target(neighbors):
            env.subgoals.target = next_target.copy()
            return action
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(env, '_ground_action', side_effect=switch_target), \
             patch('simulator.mappo_experiment.task_graph_metrics', side_effect=measure), \
             patch('simulator.mappo_experiment.navigation_cost', wraps=navigation_cost) as cost:
            _, reward, metrics = env.step(0)
        position = np.asarray(env.obs[env.ground]['self']['position'])[:2]
        np.testing.assert_allclose(position, (15.75, 60.))
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(cost.call_args_list), 3)
        for index, target in enumerate((old_target, old_target, next_target)):
            np.testing.assert_array_equal(calls[index]['start'], position)
            np.testing.assert_array_equal(calls[index]['target'], target)
            np.testing.assert_array_equal(cost.call_args_list[index].args[1], position)
            np.testing.assert_array_equal(cost.call_args_list[index].args[2], target)
        self.assertEqual(metrics['route_quality_gain'], 0.)
        self.assertEqual(metrics['frontier_progress_reward'], 0.)
        self.assertEqual(metrics['mapping_reward'], 0.)
        self.assertEqual(reward, 0.)
        self.assertEqual(metrics['route_target_changed'], 1.)
        self.assertEqual(metrics['graph_task_path_length'], calls[2]['result']['graph_task_path_length'])
        self.assertNotEqual(metrics['graph_task_path_length'], calls[1]['result']['graph_task_path_length'])
        np.testing.assert_array_equal([metrics['reward_route_target_x'], metrics['reward_route_target_y']], old_target)
        np.testing.assert_array_equal([metrics['route_target_x'], metrics['route_target_y']], next_target)
        self.assertAlmostEqual(metrics['ground_path_cost'], navigation_cost(graph, position, next_target))
        state = env.critic_state(1, 50)
        self.assertAlmostEqual(float(state[5]), np.linalg.norm(position-next_target)/env.initial_goal_distance, places=6)
        self.assertAlmostEqual(float(state[7]), min(metrics['ground_path_cost']*.05/env.initial_goal_distance, 2.), places=6)

    def test_endpoint_evidence_is_measured_before_and_after_real_packet_delivery(self):
        env, graph = self.experiment()
        # The current vehicle cell is initially unknown. The real middle UAV
        # observes it and its broadcast opens this endpoint during env.step.
        x, y = np.floor(np.array([15.75, 60.])/env.ledger.resolution).astype(int)
        env.ledger.seen[y, x] = False
        env.ledger.log_odds[y, x] = 0.
        calls, measure = self.metric_recorder(env)
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(env, '_ground_action', return_value=dict(env.ground_action)), \
             patch('simulator.mappo_experiment.task_graph_metrics', side_effect=measure):
            _, _, metrics = env.step(0)
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0]['graph'], graph)
        self.assertIs(calls[1]['graph'], graph)
        self.assertFalse(calls[0]['start_seen'])
        self.assertTrue(calls[1]['start_seen'])
        self.assertEqual(calls[0]['result']['graph_task_connected'], 0.)
        self.assertEqual(calls[1]['result']['graph_task_connected'], 1.)
        self.assertGreater(metrics['received_ground_packets'], 0.)
        self.assertGreater(metrics['route_first_quality_gain'], 0.)
        self.assertEqual(metrics['route_lengths_comparable'], 0.)
        self.assertEqual(metrics['mapping_reward'], 0.)

    def test_in_place_fusion_credits_only_the_remaining_route_shortening(self):
        env, graph = self.experiment(detour=True)
        calls, measure = self.metric_recorder(env)
        action = dict(env.ground_action)
        def add_shortcut(neighbors):
            graph.add_edge('vehicle', 'goal')
            return action
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(env, '_ground_action', side_effect=add_shortcut), \
             patch('simulator.mappo_experiment.task_graph_metrics', side_effect=measure):
            _, _, metrics = env.step(0)
        self.assertIs(calls[0]['graph'], calls[1]['graph'])
        before, after = (call['result']['graph_task_path_length'] for call in calls)
        self.assertGreater(before, after)
        reference = np.linalg.norm(calls[0]['target']-calls[0]['start'])
        expected_gain = reference/after-reference/before
        self.assertAlmostEqual(metrics['route_quality_gain'], expected_gain)
        self.assertAlmostEqual(metrics['path_shortening_reward'], 20.*expected_gain)
        self.assertAlmostEqual(metrics['route_length_delta_m'], before-after)
        self.assertEqual(metrics['route_first_quality_gain'], 0.)


if __name__ == '__main__':
    unittest.main()
