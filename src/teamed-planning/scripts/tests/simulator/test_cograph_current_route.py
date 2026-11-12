"""Regression coverage for moving-vehicle CoGraph map rewards and termination."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.graph_structure import TopologicalGraph
from drone.cograph_v2_mappo import CoGraphV2ActorCritic
from simulator.cograph_experiment import CoGraphExperiment, run_cograph_episode
from simulator.navigation_metrics import current_team_routes, paired_team_route_update
from simulator.navigation_metrics import multi_target_graph_metrics


def route_metrics(lengths, frontier=None):
    lengths = np.asarray(lengths, float)
    connected = np.isfinite(lengths)
    return dict(connected=connected, length=lengths,
                frontier=np.where(connected, 0., 100.) if frontier is None else np.asarray(frontier, float),
                nodes=1, edges=0)


class TeamRouteRewardTests(unittest.TestCase):
    def test_team_reference_is_lower_bound_for_shortest_available_team_route(self):
        # The nearest car does not have a map. The farther car's 100 m route
        # must not be normalized by its own 100 m straight-line distance.
        positions, goals = [[90., 0., 0.], [0., 0., 0.]], [[100., 0.]]
        routes = [route_metrics([np.inf], [10.]), route_metrics([100.])]
        summary = current_team_routes(routes, positions, goals)
        np.testing.assert_array_equal(summary['connected'], [True])
        np.testing.assert_allclose(summary['reference_length'], [10.])
        np.testing.assert_allclose(summary['length'], [100.])
        np.testing.assert_allclose(summary['quality'], [.1])

    def test_movement_and_switch_of_shortest_vehicle_cannot_create_map_gain(self):
        goals = [[100., 0.], [0., 100.]]
        for positions, lengths in (
                ([[0., 0.], [10., 0.]], [[140., 150.], [160., 170.]]),
                ([[0., 30.], [60., 0.]], [[110., 130.], [80., 150.]])):
            routes = [route_metrics(row) for row in lengths]
            update = paired_team_route_update(routes, routes, positions, goals)
            np.testing.assert_allclose(update['quality_gain'], 0.)
            np.testing.assert_allclose(update['frontier_gain'], 0.)
            np.testing.assert_allclose(update['length_delta_m'], 0.)

    def test_map_shortening_is_credited_even_when_the_shortest_vehicle_switches(self):
        positions, goals = [[0., 0.], [20., 0.]], [[100., 0.]]
        before = [route_metrics([140.]), route_metrics([160.])]
        after = [route_metrics([140.]), route_metrics([100.])]
        update = paired_team_route_update(before, after, positions, goals)
        np.testing.assert_allclose(update['quality_gain'], [80./100.-80./140.])
        np.testing.assert_allclose(update['length_delta_m'], [40.])
        np.testing.assert_array_equal(update['lengths_comparable'], [True])

    def test_degradation_and_reconnection_do_not_pay_positive_only_rewards(self):
        positions, goals = [[0., 0.]], [[100., 0.]]
        good, detour = [route_metrics([120.])], [route_metrics([160.])]
        lost = [route_metrics([np.inf], [40.])]
        updates = [paired_team_route_update(a, b, positions, goals)
                   for a, b in ((good, detour), (detour, lost), (lost, good))]
        self.assertLess(updates[0]['quality_gain'][0], 0.)
        self.assertLess(updates[1]['quality_gain'][0], 0.)
        self.assertLess(updates[1]['frontier_gain'][0], 0.)
        self.assertGreater(updates[2]['quality_gain'][0], 0.)
        np.testing.assert_allclose(sum(u['quality_gain'] for u in updates), 0., atol=1e-12)
        np.testing.assert_allclose(sum(u['frontier_gain'] for u in updates), 0., atol=1e-12)
        self.assertFalse(updates[1]['lengths_comparable'][0])
        self.assertFalse(updates[2]['lengths_comparable'][0])
        np.testing.assert_allclose(updates[2]['length_delta_m'], 0.)

    def test_frontier_uses_current_distance_and_zero_distance_stays_finite(self):
        before = [route_metrics([np.inf], [60.])]
        after = [route_metrics([np.inf], [40.])]
        update = paired_team_route_update(before, after, [[20., 0.]], [[100., 0.]])
        np.testing.assert_allclose(update['frontier_gain'], [.25])
        at_goal = current_team_routes([route_metrics([0.])], [[100., 0.]], [[100., 0.]])
        np.testing.assert_allclose(at_goal['quality'], [1.])
        np.testing.assert_allclose(at_goal['reference_length'], [0.])
        np.testing.assert_allclose(at_goal['progress'], [1.])


class CurrentRouteExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def experiment(self):
        scene = dict(start=[100., 100.], goals=[[250., 100.], [100., 250.]], attempts=1,
                     vehicle_starts=[[100., 100.], [100., 114.], [100., 86.]], circles=[],
                     reference_polyline_lengths=[200., 200.])
        torch.manual_seed(23)
        with contextlib.redirect_stdout(io.StringIO()):
            env = CoGraphExperiment(23, CoGraphV2ActorCritic().eval(), scene,
                vehicle_planner='decentralized', cograph_route_reward='paired_current',
                frontier_progress_weight=45., path_quality_weight=20.)
        self.addCleanup(env.env.close)
        graph = TopologicalGraph('fixture')
        for key, point in dict(start=(100., 100.), bend=(150., 150.),
                               goal_0=(250., 100.), goal_1=(100., 250.)).items():
            graph.add_node(key, point)
        graph.add_edge('start', 'bend')
        graph.add_edge('bend', 'goal_0')
        graph.add_edge('bend', 'goal_1')
        for vehicle in env.vehicles:
            vehicle.unified_graph = graph
            vehicle.graph_fusion.unified_graph = graph
        # Fixture map evidence is evaluator-only and never sent to the policy.
        env.ledger.seen[:] = True
        env.ledger.log_odds[:] = -3.
        env.vehicle_actions[0]['motion'] = (.75, 0.)
        return env, graph

    def metric_recorder(self):
        calls = []
        def measure(graph, start, goals, connector):
            result = multi_target_graph_metrics(graph, start, goals, connector)
            calls.append(dict(graph=graph, start=np.asarray(start).copy(), goals=np.asarray(goals).copy(),
                              metrics={k: v.copy() if isinstance(v, np.ndarray) else v
                                       for k, v in result.items()}))
            return result
        return calls, measure

    def test_physics_motion_uses_identical_current_endpoints_before_and_after_delivery(self):
        env, _ = self.experiment()
        actions = [dict(a) for a in env.vehicle_actions]
        calls, measure = self.metric_recorder()
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(env, '_vehicle_action', side_effect=lambda k: dict(actions[k])), \
             patch('simulator.cograph_experiment.multi_target_graph_metrics', side_effect=measure):
            _, reward, metrics = env.step(0)
        positions = np.asarray(env._vehicle_positions())
        np.testing.assert_allclose(positions[0][:2], [100.75, 100.])
        self.assertEqual(len(calls), 6)
        for k in range(3):
            for call in (calls[k], calls[k+3]):
                np.testing.assert_allclose(call['start'][:2], positions[k][:2])
                np.testing.assert_array_equal(call['goals'], env.goals)
        self.assertEqual(metrics['route_quality_gain'], 0.)
        self.assertEqual(metrics['frontier_progress_reward'], 0.)
        self.assertEqual(reward, 0.)
        expected_reference = np.linalg.norm(env.goals[None, :, :]-positions[:, None, :2], axis=2).min(axis=0)
        np.testing.assert_allclose(env.current_route_reference, expected_reference)
        self.assertAlmostEqual(metrics['route_reference_length_m'], expected_reference.mean())
        self.assertEqual(metrics['route_reference_is_lower_bound'], 1.)
        self.assertLess(env.summary()['graph_task_path_ratio'], 2.)
        self.assertEqual(env.summary()['targets_served'], 0.)

    def test_in_place_fusion_credits_only_same_frame_remaining_route_shortening(self):
        env, graph = self.experiment()
        actions = [dict(a) for a in env.vehicle_actions]
        calls, measure = self.metric_recorder()
        def fuse(k):
            if k == 0:
                graph.add_edge('start', 'goal_0')
            return dict(actions[k])
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(env, '_vehicle_action', side_effect=fuse), \
             patch('simulator.cograph_experiment.multi_target_graph_metrics', side_effect=measure):
            _, _, metrics = env.step(0)
        self.assertIs(calls[0]['graph'], calls[3]['graph'])
        before = np.min([c['metrics']['length'] for c in calls[:3]], axis=0)
        after = np.min([c['metrics']['length'] for c in calls[3:]], axis=0)
        self.assertGreater(before[0], after[0])
        expected = env.current_route_reference*(1./after-1./before)
        self.assertAlmostEqual(metrics['route_quality_gain'], expected.sum())
        self.assertAlmostEqual(metrics['path_quality_reward'], 20.*expected.mean())
        self.assertAlmostEqual(metrics['route_length_delta_m'], (before-after).mean())

    def test_new_endpoint_evidence_is_compared_against_the_pre_delivery_ledger(self):
        env, _ = self.experiment()
        # The UAV's real sensor packet opens the previously unknown start cell.
        x, y = np.floor(np.array([100.75, 100.])/env.ledger.resolution).astype(int)
        env.ledger.seen[y, x] = False
        env.ledger.log_odds[y, x] = 0.
        actions = [dict(a) for a in env.vehicle_actions]
        calls, measure = self.metric_recorder()
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(env, '_vehicle_action', side_effect=lambda k: dict(actions[k])), \
             patch('simulator.cograph_experiment.multi_target_graph_metrics', side_effect=measure):
            _, _, metrics = env.step(0)
        self.assertFalse(calls[0]['metrics']['connected'].any())
        self.assertTrue(calls[3]['metrics']['connected'].all())
        self.assertGreater(metrics['route_quality_gain'], 0.)
        self.assertGreater(metrics['frontier_progress_reward'], 0.)
        self.assertEqual(metrics['route_length_delta_m'], 0.)


class CurrentRouteTerminationTests(unittest.TestCase):
    def run_stub_episode(self, quality, improvement_steps=3):
        config_path = Path(__file__).resolve().parents[3]/'config/algorithm/cograph_v2.json'
        config = dict(json.loads(config_path.read_text()), cograph_route_reward='paired_current',
                      steps=8, min_episode_steps=1, map_stability_steps=1,
                      map_improvement_steps=improvement_steps, map_target_quality=.95)
        scene = dict(start=[0., 0.], goals=[[100., 0.], [0., 100.]], distances=[100., 100.],
                     reference_lengths=[100., 100.])
        class Env:
            def __init__(self, *args, **kwargs):
                self.goals = np.asarray(scene['goals'])
                self.ledger = type('Ledger', (), {'seen': np.zeros((1, 1))})()
                self.env = type('Physics', (), {'close': lambda self: None})()
                self.steps = 0
            def critic_state(self, *args):
                return np.zeros(16)
            def step(self, frame):
                self.steps += 1
                return [(np.zeros(1),)*4], 0., dict(chain_max_distance_ratio=.5, all_connected=1.,
                    chain_links_available=1., targets_connected=2., route_quality=quality,
                    route_quality_gain=0.)
            def summary(self):
                return dict(reward=0., route_quality=quality)
        with patch('simulator.cograph_experiment.generate_cograph_scene', return_value=scene), \
             patch('simulator.cograph_experiment.CoGraphExperiment', Env):
            return run_cograph_episode(None, config, 0, {}, False)

    def test_straight_line_quality_threshold_is_not_reported_as_optimality(self):
        rollout, summary = self.run_stub_episode(1.)
        self.assertEqual(len(rollout), 1)
        self.assertEqual(summary['map_completed'], 1.)
        self.assertEqual(summary['map_quality_target_reached'], 1.)
        self.assertEqual(summary['map_optimal'], 0.)
        self.assertGreater(summary['map_completion_bonus'], 0.)

    def test_motion_without_map_gain_does_not_extend_the_improvement_phase(self):
        rollout, summary = self.run_stub_episode(.8)
        self.assertEqual(len(rollout), 4)
        self.assertEqual(summary['map_plateau'], 1.)
        self.assertEqual(summary['map_optimal'], 0.)
        self.assertEqual(summary['map_quality_target_reached'], 0.)


if __name__ == '__main__':
    unittest.main()
