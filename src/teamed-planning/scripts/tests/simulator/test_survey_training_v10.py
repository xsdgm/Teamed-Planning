"""Integration contracts for local a-b survey with the true vehicle radio anchor."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from drone.cbf_safety import CBFSafetyFilter
from drone.mappo import ActorCritic
from drone.survey_mappo import SurveyActorCritic, SURVEY_FEATURES
from simulator import train_mappo
from simulator.survey_experiment import SurveyExperiment, run_survey_episode
from simulator.survey_reward_v10 import SurveyTask


CONFIG = Path(__file__).resolve().parents[3]/'config/algorithm/mappo_training_v10.json'


def config(**overrides):
    return train_mappo.validate_config(dict(json.loads(CONFIG.read_text()), **overrides))


def scene(task=None, goal=(650., 500.), anchored=False, stationary=True):
    return dict(start=(500., 500.), goal=goal, circles=[], attempts=1,
                survey_task=task or SurveyTask((506., 496.), (516., 504.), 2.),
                survey_anchored=anchored, survey_stationary=stationary,
                survey_envelope_m=224.)


def hold_model():
    model = SurveyActorCritic()
    with torch.no_grad():
        model.gat_actor[-1].weight.zero_()
        model.gat_actor[-1].bias.zero_()
    return model


def record(**overrides):
    return dict(dict(survey_completed=0., survey_confidence=.5, survey_coverage=.4,
        survey_a_equals_initial_c=0., survey_vehicle_stationary=1.,
        episode_steps=100., goal_reached=0., collisions=0., ground_collisions=0.,
        out_of_bounds=0., survey_disconnected_steps=0., safety_terminated=0.), **overrides)


class SurveyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def experiment(self):
        fixture = scene()
        with contextlib.redirect_stdout(io.StringIO()):
            env = SurveyExperiment(90000, hold_model(), survey_task=fixture['survey_task'],
                stochastic=False, world_size=1000., start=fixture['start'],
                scene=(fixture['goal'], [], 1), stop_on_ground_arrival=False,
                moving_ground=True, ground_stationary=True,
                communication_mode='backbone', safety_filter='cbf',
                reward_mode='velocity_mapping', mapping_route_origin='initial',
                route_reward_weight=0., exploration_weight=0.,
                path_quality_weight=0., frontier_progress_weight=0., time_penalty=.01)
        self.addCleanup(env.env.close)
        return env

    def run_short(self, fixture, **overrides):
        settings = config(**dict(dict(steps=5, min_episode_steps=5, map_stability_steps=2), **overrides))
        with patch('simulator.survey_experiment.make_survey_scene', return_value=fixture):
            rollout, summary = train_mappo.run_episode(hold_model(), settings, 90000,
                train_mappo.fixed_schedule(settings, 0), stochastic=False)
        return settings, rollout, summary

    def test_configuration_separates_survey_from_navigation_and_resume(self):
        settings = config()
        self.assertEqual(settings['reward_mode'], 'local_survey')
        self.assertEqual(settings['termination'], 'survey')
        self.assertEqual(settings['policy_architecture'], 'gat_survey_v10')
        self.assertEqual(train_mappo.feature_version(settings), 11)
        self.assertIn('local', train_mappo.reward_contract(settings))
        for change in ({'arrival_reward': 100.}, {'termination': 'arrival'},
                       {'route_reward_weight': 10.}, {'init_optimizer': True},
                       {'survey_a': [10., 20.]}, {'survey_target_coverage': 0.}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                config(**change)
        old_path = CONFIG.with_name('mappo_training_v9.json')
        old = train_mappo.validate_config(json.loads(old_path.read_text()))
        with self.assertRaises(ValueError):
            train_mappo.optimizer_config_changes(old, settings, allow_upgrade=True)

    def test_real_broadcast_and_actor_keep_a_b_and_vehicle_c_distinct(self):
        env = self.experiment()
        message = env.ground_action['broadcast']
        np.testing.assert_allclose(message['position'][:2], (500., 500.))
        np.testing.assert_allclose(message['survey']['a'], (506., 496.))
        np.testing.assert_allclose(message['survey']['b'], (516., 504.))
        self.assertNotEqual(message['survey']['a'], list(message['position'][:2]))
        with contextlib.redirect_stdout(io.StringIO()):
            env.step(0)
            positions = [np.asarray(env.obs[rid]['self']['position']) for rid in env.ids]
            samples, _, _ = env.step(1)
        for sample, position in zip(samples, positions):
            encoded = sample[0]
            self.assertEqual(encoded[-1], 1.)
            np.testing.assert_allclose(encoded[6:8], (np.array((500., 500.))-position[:2])/95., atol=1e-6)
            np.testing.assert_allclose(encoded[-SURVEY_FEATURES:-SURVEY_FEATURES+2],
                                       (np.array((506., 496.))-position[:2])/350., atol=1e-6)
        self.assertTrue(env.connectivity()[0])

    def test_local_information_pays_without_vehicle_delivery(self):
        env = self.experiment()
        initial = env.local_evidence.metrics()['survey_confidence']
        with patch.object(env.ledger, 'receive', return_value=False), contextlib.redirect_stdout(io.StringIO()):
            _, reward, metrics = env.step(0)
        self.assertFalse(env.ledger.seen.any())
        self.assertGreater(metrics['survey_confidence'], initial)
        self.assertGreater(metrics['survey_confidence_reward'], 0.)
        self.assertAlmostEqual(metrics['survey_confidence_reward'],
                               100.*(metrics['survey_confidence']-initial))
        self.assertAlmostEqual(reward, metrics['survey_confidence_reward']-.01)

    def test_survey_completes_before_vehicle_arrival_and_total_reward_balances(self):
        settings, rollout, summary = self.run_short(scene())
        self.assertEqual(summary['survey_completed'], 1.)
        self.assertEqual(summary['goal_reached'], 0.)
        self.assertEqual(summary['map_completion_bonus'], settings['map_success_reward'])
        self.assertEqual(summary['budget_exhausted'], 0.)
        self.assertLess(len(rollout), settings['steps'])
        self.assertAlmostEqual(summary['reward'], sum(row[-2] for row in rollout))
        costs = sum(summary[key] for key in ('time_cost', 'spacing_cost', 'intervention_cost',
            'chain_risk_cost', 'chain_outward_cost', 'chain_missing_cost', 'cbf_margin_cost'))
        self.assertAlmostEqual(summary['reward'],
            summary['survey_confidence_reward']+summary['map_completion_bonus']-costs)
        self.assertAlmostEqual(summary['survey_confidence_reward'], settings['survey_confidence_weight']*
            (summary['survey_confidence']-summary['survey_initial_confidence']))
        json.dumps(summary, allow_nan=False)

    def test_vehicle_arrival_alone_neither_completes_survey_nor_earns_bonus(self):
        distant = SurveyTask((600., 500.), (630., 500.), 2.)
        settings, rollout, summary = self.run_short(scene(distant, goal=(500., 500.)))
        self.assertEqual(summary['goal_reached'], 1.)
        self.assertEqual(summary['survey_completed'], 0.)
        self.assertEqual(summary['map_completion_bonus'], 0.)
        self.assertEqual(summary['survey_confidence_reward'], 0.)
        self.assertEqual(summary['budget_exhausted'], 1.)
        self.assertEqual(len(rollout), settings['steps'])
        self.assertLess(summary['reward'], 0.)

    def test_first_cbf_rejection_charges_once_without_physics_or_information_gain(self):
        original = CBFSafetyFilter.filter

        def reject(safety_filter, *args, **kwargs):
            velocity, stats = original(safety_filter, *args, **kwargs)
            stats['infeasible'] = True
            return velocity, stats

        with patch.object(CBFSafetyFilter, 'filter', reject), \
             patch('simulator.mappo_experiment.MultiRobotEnv.step',
                   side_effect=AssertionError('Rejected action must not execute physics')) as physics:
            settings, rollout, summary = self.run_short(scene())
        physics.assert_not_called()
        self.assertEqual(len(rollout), 1)
        self.assertEqual(summary['episode_steps'], 0.)
        self.assertEqual(summary['ground_distance'], 0.)
        self.assertEqual(summary['survey_confidence_reward'], 0.)
        self.assertEqual(summary['survey_confidence_gain'], 0.)
        self.assertEqual(summary['survey_completed'], 0.)
        self.assertEqual(summary['safety_terminated'], 1.)
        self.assertEqual(summary['reward'], -settings['safety_failure_penalty'])
        self.assertEqual(summary['reward'], rollout[-1][-2])
        json.dumps(summary, allow_nan=False)

    def test_worker_model_roundtrip_runs_a_real_survey_rollout(self):
        settings = config(steps=2, min_episode_steps=2)
        original = hold_model()
        model = ActorCritic.from_state_dict(copy.deepcopy(original.state_dict()))
        self.assertIsInstance(model, SurveyActorCritic)
        with patch('simulator.survey_experiment.make_survey_scene', return_value=scene()):
            rollout, summary = run_survey_episode(model, settings, 90000,
                train_mappo.fixed_schedule(settings, 0), stochastic=False)
        self.assertEqual(len(rollout), 2)
        self.assertGreater(summary['survey_confidence'], 0.)
        self.assertEqual(rollout[0][0].shape[0], 3)
        for sample in rollout:
            self.assertTrue(np.isfinite(sample[0]).all())
            self.assertTrue(np.isfinite(sample[-1]).all())

    def test_checkpoint_score_ignores_arrival_and_penalizes_disconnection(self):
        settings = config()
        mapped = train_mappo.score_validation([record(survey_completed=1.)], settings)
        arrived = train_mappo.score_validation([record(survey_completed=1., goal_reached=1.)], settings)
        failed = train_mappo.score_validation([record(survey_completed=1., survey_disconnected_steps=1.)], settings)
        self.assertEqual(mapped['score'], arrived['score'])
        self.assertEqual(mapped['success_metric'], 'survey_completed')
        self.assertGreater(mapped['score'], failed['score'])
        rejected = train_mappo.score_validation([record(survey_completed=1., safety_terminated=1.)], settings)
        self.assertGreater(mapped['score'], rejected['score'])

    def test_validation_separates_independent_tasks_from_anchor_and_motion_cases(self):
        rows = [record(survey_a_equals_initial_c=1., survey_vehicle_stationary=1.,
                       survey_completed=1., survey_confidence=.9),
                record(survey_a_equals_initial_c=1., survey_vehicle_stationary=0.,
                       survey_completed=0., survey_confidence=.2),
                record(survey_a_equals_initial_c=0., survey_vehicle_stationary=1.,
                       survey_completed=1., survey_confidence=.8),
                record(survey_a_equals_initial_c=0., survey_vehicle_stationary=0.,
                       survey_completed=0., survey_confidence=.4, survey_disconnected_steps=1.)]
        report = train_mappo.score_validation(rows, config())
        groups = report['groups']
        self.assertEqual(set(groups), {'anchored_stationary', 'anchored_moving',
                                       'independent_stationary', 'independent_moving'})
        self.assertTrue(all(group['count'] == 1 for group in groups.values()))
        self.assertEqual(groups['anchored_stationary']['survey_confidence'], .9)
        self.assertEqual(groups['independent_moving']['survey_confidence'], .4)
        self.assertEqual(groups['independent_moving']['disconnected_rate'], 1.)
        partial = train_mappo.score_validation(rows[:1], config())['groups']['independent_moving']
        self.assertEqual(partial['count'], 0)
        self.assertIsNone(partial['survey_completed'])
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
