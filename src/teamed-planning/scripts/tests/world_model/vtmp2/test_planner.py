"""World-model planning contracts: free rollout, information and action honesty."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from world_model.planner import (CHECKPOINT_FORMAT, DRIVE_ONLY, DRIVE_SURVEY, MAP_SHAPE,
                     STATE_FEATURES, WorldModelPlanner)


class PredictableModel(nn.Module):
    algorithm = 'vtmp2'

    def __init__(self, favored=1., random=False, corrupt=False):
        super().__init__()
        self.favored, self.random, self.corrupt = favored, random, corrupt
        self.encoder = nn.Flatten()
        self.observe_calls = 0
        self.inputs = []

    def initial(self, maps, states, sample=False):
        return states.new_zeros((len(states), 2)), states.new_zeros((len(states), 2))

    def observe(self, hidden, embedding, state, sample):
        self.observe_calls += 1
        return hidden.clone(), None

    def imagine(self, hidden, latent, action, maps, states, sample=False):
        self.inputs.append((hidden.clone(), latent.clone(), maps.clone(), states.clone(), action.clone()))
        states, maps = states.clone(), maps.clone()
        advantageous = (action[:, 1] * self.favored > 0).float()
        drive = (action[:, 2] > 0).float()
        states[:, 0] += drive * (.001 + .005 * advantageous)
        states[:, 8:10] += (1 - drive[:, None]) * (.005 + .02 * advantageous[:, None])
        if self.random and sample:
            states[:, 0] += torch.randn(len(states), device=states.device) * .001
        maps[:, 0] += .01
        if self.corrupt:
            states[:, 0] = float('nan')
        outcome = states.new_full((len(states), 4), -10.)
        outcome[:, 0] = 10.  # Ignored arrival head must not drive the selection.
        outcome[:, 3] = 0.   # Predicted five seconds per requested ten-second step.
        return hidden + 1, latent + 1, None, maps, states, outcome


def candidate(y, status='unverified', purpose='drive'):
    return SimpleNamespace(point=np.array([150., y]), status=status,
                           source='test', cost=160., purpose=purpose)


def observed(planner):
    maps = np.zeros(MAP_SHAPE, np.float32)
    states = np.zeros(STATE_FEATURES, np.float32)
    states[6] = .3
    planner.observe(maps, states, 0.)
    return maps, states


class PlannerTests(unittest.TestCase):
    def test_model_predictions_flip_distant_candidate_selection(self):
        points = [candidate(50), candidate(-50)]
        for favored, expected in ((1, 0), (-1, 1)):
            planner = WorldModelPlanner(PredictableModel(favored), horizon_steps=4)
            observed(planner)
            scores = planner.rank(points, [300., 0.])
            self.assertEqual(int(np.argmin(scores)), expected)

    def test_verified_and_unknown_candidates_use_identical_formula(self):
        planner = WorldModelPlanner(PredictableModel(), horizon_steps=2)
        observed(planner)
        scores = planner.rank([candidate(50, 'verified'), candidate(50, 'unverified')], [300., 0.])
        self.assertEqual(scores[0], scores[1])

    def test_free_rollout_does_not_assimilate_future_or_mutate_observation(self):
        model = PredictableModel()
        planner = WorldModelPlanner(model, horizon_steps=3, particles=2)
        maps, states = observed(planner)
        original_maps, original_states = maps.copy(), states.copy()
        planner.rank([candidate(50)], [300., 0.])
        self.assertEqual(model.observe_calls, 0)
        self.assertEqual(len(model.inputs), 3)
        self.assertAlmostEqual(float(model.inputs[2][0][0, 0]), 2.)
        self.assertAlmostEqual(float(model.inputs[2][1][0, 0]), 2.)
        self.assertAlmostEqual(float(model.inputs[2][2][0, 0, 0, 0]), .02, places=6)
        self.assertAlmostEqual(float(model.inputs[2][3][0, 0]), .012, places=6)
        np.testing.assert_array_equal(maps, original_maps)
        np.testing.assert_array_equal(states, original_states)
        np.testing.assert_array_equal(planner._maps.cpu().numpy()[0], original_maps)
        self.assertEqual(planner.last_diagnostics['candidates'][0]['predicted_elapsed_seconds'], 15.)
        self.assertEqual(planner.last_diagnostics['requested_horizon_seconds'], 30.)

    def test_random_reproducibility_order_invariance_and_rng_restoration(self):
        planner = WorldModelPlanner(PredictableModel(random=True), horizon_steps=3, particles=4)
        observed(planner)
        before = torch.random.get_rng_state().clone()
        first = planner.rank([candidate(50), candidate(-50)], [300., 0.])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(first, planner.rank([candidate(50), candidate(-50)], [300., 0.]))
        reversed_scores = planner.rank([candidate(-50), candidate(50)], [300., 0.])
        self.assertEqual(first, reversed_scores[::-1])

    def test_survey_contract_and_information_without_vehicle_progress(self):
        legacy = WorldModelPlanner(PredictableModel(), horizon_steps=2)
        observed(legacy)
        with self.assertRaisesRegex(ValueError, 'drive_survey_v1-trained'):
            legacy.rank([candidate(50, purpose='survey')], [300., 0.])
        planner = WorldModelPlanner(PredictableModel(), horizon_steps=3, action_contract=DRIVE_SURVEY)
        observed(planner)
        scores = planner.rank([candidate(50, purpose='survey'), candidate(-50, purpose='survey')], [300., 0.])
        self.assertLess(scores[0], scores[1])
        row = planner.last_diagnostics['candidates'][0]
        self.assertEqual(row['predicted_goal_progress_m'], 0.)
        self.assertGreater(row['predicted_graph_gain_proxy'], 0.)
        self.assertFalse(planner.last_diagnostics['information_is_observed'])
        self.assertFalse(planner.last_diagnostics['uncertainty_calibrated'])

    def test_map_hallucination_without_graph_count_gain_gets_no_information_reward(self):
        planner = WorldModelPlanner(PredictableModel(), horizon_steps=2)
        observed(planner)
        planner.rank([candidate(50)], [300., 0.])
        self.assertEqual(planner.last_diagnostics['candidates'][0]['predicted_graph_gain_proxy'], 0.)

    def test_survey_graph_uncertainty_is_penalized_even_when_vehicle_is_stationary(self):
        class UncertainSurveyModel(PredictableModel):
            def imagine(self, hidden, latent, action, maps, states, sample=False):
                result = list(super().imagine(hidden, latent, action, maps, states, sample))
                variation = states.new_tensor([-.04, .04]) * (action[:, 1] > 0)
                result[4][:, 8:10] = .1 + variation[:, None]
                return tuple(result)
        planner = WorldModelPlanner(UncertainSurveyModel(), horizon_steps=1, particles=2,
                                    action_contract=DRIVE_SURVEY)
        observed(planner)
        scores = planner.rank([candidate(50, purpose='survey'), candidate(-50, purpose='survey')],
                              [300., 0.])
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(planner.last_diagnostics['candidates'][0]['position_spread_m'], 0.)
        self.assertGreater(planner.last_diagnostics['candidates'][0]['graph_uncertainty_penalty'], 0.)

    def test_rejects_nonfinite_observations_candidates_and_model_outputs(self):
        planner = WorldModelPlanner(PredictableModel(), horizon_steps=1)
        maps, states = observed(planner)
        states[0] = np.nan
        with self.assertRaisesRegex(ValueError, 'states must be finite'):
            planner.observe(maps, states, 1.)
        bad = candidate(50)
        bad.point[0] = np.inf
        with self.assertRaises(ValueError):
            planner.rank([bad], [300., 0.])
        planner.model.corrupt = True
        with self.assertRaisesRegex(ValueError, 'non-finite prediction'):
            planner.rank([candidate(50)], [300., 0.])

    def test_belief_advances_only_at_macro_boundary_or_changed_target(self):
        model = PredictableModel()
        planner = WorldModelPlanner(model)
        maps, states = observed(planner)
        planner.observe(maps, states, 0., executed_target=[150., 50.])
        planner.observe(maps, states, 1., executed_target=[150., 50.])
        self.assertEqual(len(model.inputs), 0)
        planner.observe(maps, states, 10., executed_target=[150., 50.])
        self.assertEqual(len(model.inputs), 1)
        planner.observe(maps, states, 12., executed_target=[150., -50.])
        self.assertEqual(len(model.inputs), 2)
        self.assertEqual(planner._irregular_updates, 1)
        planner.observe(maps, states, 30., executed_target=[150., -50.])
        self.assertEqual(len(model.inputs), 2)
        self.assertEqual(planner._gap_resets, 1)
        planner.reset()
        with self.assertRaisesRegex(ValueError, 'observe before rank'):
            planner.rank([candidate(50)], [300., 0.])

    def test_same_time_command_notification_anchors_new_action_without_one_tick_lag(self):
        model = PredictableModel()
        planner = WorldModelPlanner(model, action_contract=DRIVE_SURVEY)
        maps, states = observed(planner)
        first, second = [150., 50.], [160., -50.]
        planner.observe(maps, states, 0., executed_target=first)
        states_at_switch = states.copy()
        states_at_switch[0] = .003
        # Runtime senses under A, then chooses B at this exact simulation time.
        planner.observe(maps, states_at_switch, 3., executed_target=first)
        planner.observe(maps, states_at_switch, 3., executed_target=second,
                        executed_purpose='survey')
        self.assertEqual(planner._anchor_time, 3.)
        self.assertEqual(len(model.inputs), 1)
        torch.testing.assert_close(model.inputs[0][4], torch.tensor([[.15, .05, 1.]]))
        torch.testing.assert_close(planner._anchor_target, torch.tensor([[.16, -.05, -1.]]))
        states_next_tick = states_at_switch.copy()
        states_next_tick[0] = .004
        planner.observe(maps, states_next_tick, 4., executed_target=second,
                        executed_purpose='survey')
        self.assertEqual(planner._anchor_time, 3.)
        self.assertEqual(len(model.inputs), 1)
        planner.observe(maps, states_next_tick, 13., executed_target=second,
                        executed_purpose='survey')
        self.assertEqual(len(model.inputs), 2)
        torch.testing.assert_close(model.inputs[1][4], torch.tensor([[.16, -.05, -1.]]))
        self.assertAlmostEqual(float(model.inputs[1][3][0, 0]), .003, places=6)
        self.assertEqual(planner._anchor_time, 13.)
        self.assertEqual(planner._gap_resets, 0)

    def test_checkpoint_contract_shape_and_finite_validation(self):
        from world_model.model import Vtmp2WorldModel
        model = Vtmp2WorldModel(hidden=8, latent=4)
        payload = dict(format=CHECKPOINT_FORMAT, model=model.state_dict(), epoch=2,
                       config=dict(algorithm='vtmp2', schema_version=2, hidden=8, latent=4),
                       rng=dict(numpy=np.random.get_state()))
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(payload, path)
            before = torch.random.get_rng_state().clone()
            planner = WorldModelPlanner.from_checkpoint(path, horizon_steps=1, particles=2)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            self.assertEqual(planner.action_contract, DRIVE_ONLY)
            self.assertEqual(len(planner.checkpoint_info['sha256']), 64)
            observed(planner)
            self.assertTrue(np.isfinite(planner.rank([candidate(50)], [300., 0.])).all())
            with self.assertRaisesRegex(ValueError, 'override checkpoint'):
                WorldModelPlanner.from_checkpoint(path, action_contract=DRIVE_SURVEY)
            survey_payload = dict(payload)
            survey_payload['config'] = dict(payload['config'], schema_version=3,
                                            action_contract=DRIVE_SURVEY,
                                            rollout_steps=4, rollout_weight=.25)
            torch.save(survey_payload, path)
            with self.assertRaisesRegex(ValueError, 'both drive and survey'):
                WorldModelPlanner.from_checkpoint(path)
            survey_payload['config']['observed_action_purposes'] = ['drive', 'survey']
            torch.save(survey_payload, path)
            survey_planner = WorldModelPlanner.from_checkpoint(path, horizon_steps=1)
            self.assertEqual(survey_planner.action_contract, DRIVE_SURVEY)
            self.assertFalse(survey_planner.checkpoint_info['legacy_one_step_training'])
            self.assertEqual(survey_planner.checkpoint_info['training_objective'],
                             'one_step_prior_plus_free_rollout')
            observed(survey_planner)
            self.assertTrue(np.isfinite(survey_planner.rank([candidate(50, purpose='survey')], [300., 0.])).all())
            for field, value in [('format', 'bad'), ('model', {'bad': torch.tensor(float('nan'))})]:
                broken = dict(payload)
                broken[field] = value
                torch.save(broken, path)
                with self.assertRaises(ValueError):
                    WorldModelPlanner.from_checkpoint(path)
            broken = dict(payload)
            broken['model'] = {'unexpected': torch.zeros(1)}
            torch.save(broken, path)
            with self.assertRaisesRegex(ValueError, 'state_dict'):
                WorldModelPlanner.from_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
