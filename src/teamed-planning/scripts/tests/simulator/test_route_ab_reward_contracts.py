"""Bounded v11 desired-behavior regressions. No training or source mutations.

Synthetic map updates isolate shaping from physical costs and completion bonuses.
Short real rollouts exercise actual sensing/control with only the scene replaced.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / 'artifacts/v11-fix-20261110/rewards'
sys.path.insert(0, str(ROOT / 'src/teamed-planning/scripts'))

from common.graph_structure import TopologicalGraph
from drone.cbf_safety import CBFSafetyFilter
from simulator import train_mappo as runner
from simulator.large_world_scenarios import CircleGeometry
from simulator.mappo_experiment import DeliveredEvidence, Experiment, navigation_cost
from simulator.route_ab_experiment import RouteABExperiment
from simulator.survey_reward_v10 import SurveyTask

CONFIG = ROOT / 'src/teamed-planning/config/algorithm/mappo_training_v11.json'
EVIDENCE = {}
REWARD_KEYS = ('mapping_reward', 'path_quality_reward', 'frontier_progress_reward', 'map_completion_bonus')
COST_KEYS = ('time_cost', 'intervention_cost', 'cbf_margin_cost', 'spacing_cost',
             'chain_risk_cost', 'chain_outward_cost', 'chain_missing_cost', 'safety_failure_cost')


def connector_fixture(obstacle=True):
    a, node, b = np.array([500., 501.]), np.array([508., 501.]), np.array([520., 501.])
    circles = [([504., 504.], 2.5)] if obstacle else []
    ledger = DeliveredEvidence(world=1000., resolution=2.)
    for frame, center in enumerate(([506., 500.], [518., 500.])):
        xy = (np.arange(12) + .5 - 6.) * 2.
        xx, yy = np.meshgrid(xy + center[0], xy + center[1])
        pixels = np.stack((xx, yy), axis=-1)
        occupancy = np.zeros((12, 12))
        for center_circle, radius in circles:
            occupancy[np.linalg.norm(pixels - center_circle, axis=-1) <= radius] = 1.
        ledger.receive('fixture', frame, np.asarray(center), occupancy, 2.)
    graph = TopologicalGraph('delivered')
    graph.add_node('n', node)
    graph.add_node('b', b)
    graph.add_edge('n', 'b')
    env = object.__new__(RouteABExperiment)
    env.survey_task = SurveyTask(tuple(a), tuple(b), 12., world_size=1000.)
    env.ledger = ledger
    env.ground_safety_distance = 1.
    env.ground_controller = SimpleNamespace(unified_graph=graph)
    geometry = CircleGeometry(circles, world=1000., clearance=1.)
    facts = dict(a=a.tolist(), node=node.tolist(), b=b.tolist(), circles=circles,
        endpoint_a_clear=bool(geometry.clear(a, a)), endpoint_b_clear=bool(geometry.clear(b, b)),
        graph_edge_clear=bool(geometry.clear(node, b)),
        connector_raw_free=bool(ledger.connector_clear(a, node)),
        connector_margin_free=bool(ledger.segment_clear(a, node, 1.)),
        connector_truth_clear=bool(geometry.clear(a, node)), metrics=env.route_metrics())
    return env, facts


def map_cycle(handover):
    def graph(xs):
        result = TopologicalGraph('received-evidence')
        for i, x in enumerate(xs):
            result.add_node(str(i), (x, 10.), dict(observation_count=100, terrain_type='TRAVERSABLE'))
        for i in range(1, len(xs)):
            result.add_edge(str(i - 1), str(i))
        return result

    states = [graph([10.]), graph(np.arange(10., 61., 10.)), graph(np.arange(10., 111., 10.))]
    env = object.__new__(RouteABExperiment)
    env.survey_task = SurveyTask((10., 10.), (110., 10.), 2., world_size=140.)
    env.ground_controller = SimpleNamespace(unified_graph=states[0])
    env.ledger = SimpleNamespace(segment_clear=lambda a, b, clearance: True)
    env.ground_safety_distance = 1.
    env._ab_frontier_best = 0.
    env.first_graph_connection_step = -1
    env.ab_handover = handover
    env.ab_route_weight, env.ab_quality_weight, env.ab_frontier_weight = 10., 20., 45.
    env.trace = []
    env.connectivity = lambda: (True, 50.)
    rows = []
    for frame, index in enumerate([1, 2, 0]):
        def base_step(instance, _frame):
            instance.ground_controller.unified_graph = states[index]
            instance.trace.append({})
            return [], 0., {}

        with patch.object(Experiment, 'step', base_step):
            _, reward, metrics = env.step(frame)
        rows.append(dict(frame=frame, graph_state=index, reward=reward,
            **{key: metrics[key] for key in ('mapping_reward', 'path_quality_reward',
                                             'frontier_progress_reward', 'graph_task_connected')}))
    return dict(handover=handover, rows=rows, net_reward=sum(row['reward'] for row in rows),
        ends_at_initial_state=env.ground_controller.unified_graph is states[0],
        state_costs=[navigation_cost(g, (10., 10.), (110., 10.)) for g in states])


def config():
    return runner.validate_config(dict(json.loads(CONFIG.read_text()), init_checkpoint=None,
        steps=5, min_episode_steps=5, map_stability_steps=2))


def hold_model(settings):
    torch.manual_seed(20261110)
    model = runner.make_model(settings)
    with torch.no_grad():
        model.gat_actor[-1].weight.zero_()
        model.gat_actor[-1].bias.zero_()
    return model


def fixture():
    return dict(start=(500., 500.), goal=(650., 500.), circles=[], attempts=1,
        survey_task=SurveyTask((600., 500.), (630., 500.), 2.),
        survey_anchored=False, survey_stationary=True, survey_envelope_m=224.)


def run_short():
    settings = config()
    trace = []
    original_step = RouteABExperiment.step

    def recorded_step(env, frame):
        samples, reward, metrics = original_step(env, frame)
        trace.append(dict(metrics))
        return samples, reward, metrics

    with patch('simulator.route_ab_experiment.make_survey_scene', return_value=fixture()), \
         patch.object(RouteABExperiment, 'step', recorded_step), \
         torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
        rollout, summary = runner.run_episode(hold_model(settings), settings, 120000,
            runner.fixed_schedule(settings, 0), stochastic=False)
    return rollout, summary, trace


class ConnectorTests(unittest.TestCase):
    def test_endpoint_connector_rejects_insufficient_clearance(self):
        _, facts = connector_fixture(True)
        EVIDENCE['unsafe_connector'] = facts
        self.assertTrue(facts['endpoint_a_clear'] and facts['endpoint_b_clear'] and facts['graph_edge_clear'])
        self.assertFalse(facts['connector_truth_clear'])
        self.assertFalse(facts['connector_margin_free'])
        self.assertEqual(facts['metrics']['graph_task_connected'], 0.,
            'A route with an endpoint connector failing the 1 m clearance contract must stay disconnected')

    def test_clear_endpoint_connector_positive_control(self):
        _, facts = connector_fixture(False)
        EVIDENCE['safe_connector'] = facts
        self.assertTrue(facts['connector_truth_clear'] and facts['connector_margin_free'])
        self.assertEqual(facts['metrics']['graph_task_connected'], 1.)
        self.assertAlmostEqual(facts['metrics']['graph_task_path_length'], 20.)

    def test_nondefault_clearance_rejects_nearby_received_obstacle(self):
        env, _ = connector_fixture(False)
        # Known free grid with a single received obstacle cell y in [504,506).
        env.ledger.seen[:] = True
        env.ledger.log_odds[:] = -1.
        env.ledger.log_odds[252, 252] = 1.
        env.ground_safety_distance = 1.
        self.assertEqual(env.route_metrics()['graph_task_connected'], 1.)
        env.ground_safety_distance = 3.1
        self.assertEqual(env.route_metrics()['graph_task_connected'], 0.)

    def test_zero_length_connector_checks_surrounding_clearance(self):
        env, _ = connector_fixture(False)
        env.ledger.seen[:] = True
        env.ledger.log_odds[:] = -1.
        env.ledger.log_odds[250, 251] = 1.
        point = np.array([501.5, 501.])
        self.assertTrue(env.ledger.connector_clear(point, point))
        self.assertFalse(env._route_connector_clear(point, point))

    def test_endpoint_cap_needs_received_free_evidence(self):
        env, _ = connector_fixture(False)
        env.ledger.seen[:] = True
        env.ledger.log_odds[:] = -1.
        # Obstacle is beyond the segment's end, outside the lateral strip.
        env.ledger.log_odds[250, 254] = 1.
        a, b = np.array([500., 501.]), np.array([507.5, 501.])
        self.assertTrue(env.ledger.segment_clear(a, b, 1.))
        self.assertFalse(env._route_connector_clear(a, b))

    def test_unknown_endpoint_margin_is_not_accepted_as_free(self):
        env, _ = connector_fixture(False)
        env.ledger.seen[:] = True
        env.ledger.log_odds[:] = -1.
        env.ledger.seen[250, 254] = False
        self.assertFalse(env._route_connector_clear([500., 501.], [507.5, 501.]))


class ShapingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enabled, cls.disabled = map_cycle(True), map_cycle(False)
        EVIDENCE['cycle_handover_enabled'] = cls.enabled
        EVIDENCE['cycle_handover_disabled'] = cls.disabled

    def test_closed_map_cycle_cannot_mint_positive_shaping_reward(self):
        self.assertTrue(self.enabled['ends_at_initial_state'])
        self.assertLessEqual(self.enabled['net_reward'], 1e-8,
            'Zero-cost received-map cycle ending at its initial graph must not earn net positive shaping reward')

    def test_disabled_handover_closed_cycle_positive_control(self):
        self.assertTrue(self.disabled['ends_at_initial_state'])
        self.assertAlmostEqual(self.disabled['net_reward'], 0.)

    def test_connection_reward_is_reversed_on_map_loss(self):
        rows = self.enabled['rows']
        self.assertGreater(rows[1]['mapping_reward'], 0.)
        self.assertLess(rows[-1]['mapping_reward'], 0.)
        self.assertAlmostEqual(sum(row['mapping_reward'] for row in rows), 0.)

    def test_handover_stops_mapping_reward_for_connected_path_improvement(self):
        env = object.__new__(RouteABExperiment)
        env.ab_handover = True
        connected = dict(graph_task_connected=1.)
        self.assertEqual(env._mapping_potential(connected, 200., 100.),
                         env._mapping_potential(connected, 100., 100.))
        env.ab_handover = False
        self.assertGreater(env._mapping_potential(connected, 100., 100.),
                           env._mapping_potential(connected, 200., 100.))

    def test_mapping_potential_is_bounded_for_low_confidence_routes(self):
        env = object.__new__(RouteABExperiment)
        env.ab_handover = True
        disconnected = dict(graph_task_connected=0.)
        for cost in (0., 100., 2000., 100000.):
            potential = env._mapping_potential(disconnected, cost, 100.)
            self.assertGreaterEqual(potential, -1.)
            self.assertLessEqual(potential, 0.)

    def test_quality_and_frontier_charge_lost_map_and_balance(self):
        rows = self.enabled['rows']
        for key in ('path_quality_reward', 'frontier_progress_reward'):
            with self.subTest(term=key):
                self.assertGreater(rows[1][key], 0.)
                self.assertLess(rows[2][key], 0.)
                self.assertAlmostEqual(sum(row[key] for row in rows), 0.)

    def test_per_step_shaping_accounting_is_signed_and_exact(self):
        for row in self.enabled['rows']:
            with self.subTest(frame=row['frame']):
                self.assertAlmostEqual(row['reward'], sum(row[key] for key in REWARD_KEYS[:3]))
        self.assertLess(self.enabled['rows'][-1]['reward'], 0.)


class EpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.rollout, cls.summary, cls.trace = run_short()
        EVIDENCE['short_real_episode'] = dict(summary=cls.summary, trace=cls.trace)

    def test_vehicle_arrival_does_not_finish_uav_mapping(self):
        summary = self.summary
        self.assertEqual(summary['goal_reached'], 1.)
        self.assertEqual(summary['route_completed'], 0.)
        self.assertEqual(summary['map_completion_bonus'], 0.)
        self.assertEqual(summary['budget_exhausted'], 1.)
        self.assertEqual(len(self.rollout), 5)
        self.assertTrue(all(row['goal_reached'] == 1. for row in self.trace))

    def test_episode_reward_matches_physical_costs_and_route_terms(self):
        summary = self.summary
        rewards = sum(summary[key] for key in REWARD_KEYS)
        costs = sum(summary[key] for key in COST_KEYS)
        self.assertAlmostEqual(summary['reward'], rewards - costs)
        self.assertAlmostEqual(summary['reward'], sum(row[-2] for row in self.rollout))
        self.assertNotIn('survey_confidence_reward', summary)

    def test_best_frontier_progress_reports_the_ab_task(self):
        running_best = 0.
        for frame, row in enumerate(self.trace):
            running_best = max(running_best, row['frontier_progress'])
            self.assertAlmostEqual(row['frontier_best_progress'], running_best, places=8,
                msg=f'frame {frame}: best progress must be the running maximum of the A-B progress')
        self.assertAlmostEqual(self.summary['frontier_best_progress'], running_best)

    def test_empty_trace_summary_reports_fixed_ab_route(self):
        settings, layout = config(), fixture()
        with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
            env = RouteABExperiment(120000, hold_model(settings),
                survey_task=layout['survey_task'], stochastic=False,
                world_size=1000., start=layout['start'],
                scene=(layout['goal'], [], 1), stop_on_ground_arrival=False,
                moving_ground=True, ground_stationary=True, communication_mode='backbone',
                safety_filter='cbf', reward_mode='velocity_mapping', mapping_route_origin='initial')
        self.addCleanup(env.env.close)
        self.assertEqual(env.trace, [])
        summary = env.summary()
        for key, value in env.route_metrics().items():
            self.assertEqual(summary[key], value, key)
        self.assertEqual(summary['graph_frontier_goal_distance_m'], 30.)
        self.assertEqual(summary['frontier_best_progress'], 0.)

    def test_cbf_rejection_has_negative_reward_and_no_physics_step(self):
        original_filter = CBFSafetyFilter.filter

        def reject(safety_filter, *args, **kwargs):
            velocity, stats = original_filter(safety_filter, *args, **kwargs)
            stats['infeasible'] = True
            return velocity, stats

        with patch.object(CBFSafetyFilter, 'filter', reject), \
             patch('simulator.mappo_experiment.MultiRobotEnv.step',
                   side_effect=AssertionError('CBF rejected commands must not execute physics')) as physics:
            rollout, summary, trace = run_short()
        EVIDENCE['cbf_rejection_episode'] = dict(summary=summary, trace=trace)
        physics.assert_not_called()
        self.assertEqual(len(rollout), 1)
        self.assertEqual(summary['reward'], -100.)
        self.assertEqual(summary['safety_failure_cost'], 100.)
        self.assertEqual(summary['safety_terminated'], 1.)
        self.assertEqual(summary['route_completed'], 0.)
        self.assertEqual(summary['map_completion_bonus'], 0.)
        self.assertEqual(summary['episode_steps'], 0.)


class JsonResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append(dict(test=test.id(), status='pass'))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append(dict(test=test.id(), status='fail', detail=self._exc_info_to_string(err, test)))

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append(dict(test=test.id(), status='error', detail=self._exc_info_to_string(err, test)))

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            status = 'fail' if issubclass(err[0], test.failureException) else 'error'
            self.records.append(dict(test=subtest.id(), status=status, detail=self._exc_info_to_string(err, test)))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    started = time.monotonic()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=JsonResult).run(suite)
    payload = dict(protocol='Desired-behavior regressions; no training, backward, optimizer, or source edits.',
        command='OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/bdi/miniconda3/envs/teamedplanning-2d/bin/python -I src/teamed-planning/scripts/tests/simulator/test_route_ab_reward_contracts.py',
        tests_run=result.testsRun, passed=sum(x['status'] == 'pass' for x in result.records),
        failed=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        seconds=time.monotonic()-started, tests=result.records)
    (OUT / 'test.log').write_text(stream.getvalue())
    (OUT / 'test_results.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    (OUT / 'evidence.json').write_text(json.dumps(EVIDENCE, indent=2, allow_nan=False) + '\n')
    print(stream.getvalue(), end='')
    print(json.dumps({key: payload[key] for key in ('tests_run', 'passed', 'failed', 'errors', 'seconds')}, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
