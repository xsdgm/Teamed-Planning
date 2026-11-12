"""Forward-only seed, configuration and compatibility checks for repaired v11."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator import train_mappo as training

CONFIG = Path(__file__).resolve().parents[3]/'config/algorithm/mappo_training_v11.json'


class RouteABTrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.settings = training.validate_config(json.loads(CONFIG.read_text()))

    def test_initial_stage_is_stationary_mixed_within_200m_and_from_scratch(self):
        self.assertTrue(self.settings['ground_stationary'])
        self.assertEqual(self.settings['survey_stationary_probability'], 1.)
        self.assertEqual(self.settings['task_distance_range'], [20., 200.])
        self.assertEqual(self.settings['route_ab_distance_bands'], [[20.,60.],[60.,120.],[120.,200.]])
        self.assertEqual(self.settings['survey_start_offset_range'], [10., 25.])
        self.assertIsNone(self.settings['init_checkpoint'])
        self.assertFalse(self.settings['init_optimizer'])
        self.assertEqual(self.settings['arrival_reward'], 0.)

    def test_old_v11_resume_is_rejected_but_current_contract_is_accepted(self):
        for saved in ({}, {'reward_contract':'uav_fixed_ab_route_with_vehicle_backbone_v11'}):
            with self.subTest(saved=saved), self.assertRaisesRegex(ValueError, 'contract changed'):
                training.validate_resume_reward_contract(saved, self.settings)
        training.validate_resume_reward_contract(
            {'reward_contract':training.reward_contract(self.settings)}, self.settings)

    def test_other_architecture_resume_contract_is_unchanged(self):
        settings = dict(self.settings, policy_architecture='gat_relay_attention_v8')
        training.validate_resume_reward_contract({}, settings)

    def test_validation_samples_match_across_serial_and_worker_entry_points(self):
        config = dict(self.settings, validation_seeds=[120000], validation_samples=2)
        torch.manual_seed(57)
        model = training.make_model(config).eval()
        state = model.state_dict()

        def first_noise(*args, **kwargs):
            return [], {'noise':torch.randn(8).tolist()}

        with tempfile.TemporaryDirectory() as directory, torch.no_grad(), \
                patch.object(training, 'run_episode', first_noise), \
                patch.object(training, 'score_validation', lambda records, config:records):
            ambient_state = torch.get_rng_state().clone()
            serial = training.validation(model, config)
            self.assertTrue(torch.equal(ambient_state, torch.get_rng_state()))
            workers = []
            for seed, index, sample_seed in training.validation_runs(config):
                _, summary, _ = training._rollout_task('validation', state, config,
                    seed, sample_seed, training.fixed_schedule(config, 799), directory, str(index))
                workers.append(summary['noise'])
            self.assertEqual([row['noise'] for row in serial], workers)
            self.assertNotEqual(workers[0], workers[1])

    def test_worker_reconstruction_preserves_frozen_model_output(self):
        torch.manual_seed(59)
        model = training.make_model(self.settings).eval()
        reference = {key:value.clone() for key,value in model.state_dict().items()}

        def check_weights(reconstructed, *args, **kwargs):
            for key, value in reconstructed.state_dict().items():
                torch.testing.assert_close(value, reference[key])
            return [], {'checked':True}

        with tempfile.TemporaryDirectory() as directory, torch.no_grad(), \
                patch.object(training, 'run_episode', check_weights):
            _, summary, _ = training._rollout_task('validation', reference, self.settings,
                120000, 12345, training.fixed_schedule(self.settings, 0), directory, 'weights')
        self.assertTrue(summary['checked'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
