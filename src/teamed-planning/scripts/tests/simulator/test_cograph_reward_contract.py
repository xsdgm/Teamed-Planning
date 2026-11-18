"""CoGraph's paired current-route objective is isolated from legacy training."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator.train_mappo import (episode_line, optimizer_config_changes,
                                   reward_contract, validate_config)

CONFIGS = Path(__file__).resolve().parents[3] / 'config' / 'algorithm'


def config(name, **overrides):
    return validate_config(dict(json.loads((CONFIGS / name).read_text()), **overrides))


class CoGraphRewardContractTests(unittest.TestCase):
    def test_new_v2_keeps_task_distribution_and_enables_paired_maps(self):
        current = config('cograph_v2.json')
        self.assertEqual(current['cograph_route_reward'], 'paired_current')
        self.assertEqual(current['policy_architecture'], 'cograph_v2')
        self.assertEqual(current['vehicle_planner'], 'decentralized')
        self.assertEqual(current['vehicle_motion'], 'link_keeping')
        self.assertEqual(current['task_distance_range'], [100., 400.])
        self.assertEqual(current['distance_sampling'], 'annulus_area')
        self.assertEqual(current['vehicle_count'], 3)
        self.assertEqual(current['fleet_sizes'], [9])
        self.assertEqual(current['target_count_range'], [2, 8])
        self.assertEqual(current['episodes'], 1600)
        self.assertEqual(current['episodes_per_batch'], 16)
        self.assertEqual(current['termination'], 'map')
        self.assertIsNone(current['subgoal_radius'])
        self.assertFalse(current['ground_stationary'])
        self.assertTrue(current['init_checkpoint'].endswith(
            'mappo-v8-mixed100to400-bridge15-20261030-142558/best.pt'))
        legacy = dict(current, cograph_route_reward='legacy')
        self.assertEqual(reward_contract(current),
                         reward_contract(legacy) + '_paired_current_route')

    def test_missing_field_retains_old_checkpoint_contract(self):
        historical = config('cograph_v2.json')
        historical.pop('cograph_route_reward')
        explicit = validate_config(dict(historical, cograph_route_reward='legacy'))
        self.assertEqual(validate_config(historical)['cograph_route_reward'], 'legacy')
        self.assertEqual(reward_contract(historical), reward_contract(explicit))
        self.assertEqual(optimizer_config_changes(historical, explicit), {})

    def test_reward_change_requires_separate_training_run(self):
        current = config('cograph_v2.json')
        legacy = dict(current, cograph_route_reward='legacy')
        for upgrade in (False, True):
            for before, after in ((legacy, current), (current, legacy)):
                with self.subTest(upgrade=upgrade, before=before['cograph_route_reward']):
                    with self.assertRaisesRegex(ValueError, 'Resume configuration differs'):
                        optimizer_config_changes(before, after, allow_upgrade=upgrade)

    def test_contract_cannot_silently_apply_to_v1_or_v9(self):
        for name in ('cograph_v1.json', 'mappo_training_v9.json'):
            with self.subTest(name=name):
                legacy = config(name)
                self.assertEqual(legacy['cograph_route_reward'], 'legacy')
                with self.assertRaisesRegex(ValueError, 'requires cograph_v2'):
                    validate_config(dict(legacy, cograph_route_reward='paired_current'))
        self.assertEqual(reward_contract(config('mappo_training_v9.json')),
                         'uav_remaining_route_mapping_v9')
        self.assertEqual(reward_contract(config('cograph_v1.json')),
                         'cograph_v1_multi_target_option_space_bridged')

    def test_invalid_modes_and_stationary_team_are_rejected(self):
        current = config('cograph_v2.json')
        for mode in (None, True, 'vehicle', 'current'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_config(dict(current, cograph_route_reward=mode))
        with self.assertRaisesRegex(ValueError, 'requires cograph_v2'):
            validate_config(dict(current, vehicle_planner='parked', ground_stationary=True,
                                 vehicle_motion='free'))
        with self.assertRaises(ValueError):
            validate_config(dict(current, mapping_route_origin='vehicle'))

    def test_paired_progress_line_uses_current_signed_metrics(self):
        summary = dict(reward_beta=1., reward_eta=0., confidence_area=12., task_share=.5,
            goal_reached=0., map_completed=1., route_length_delta_m=-3.4,
            frontier_progress=.65, route_new_best_saving_m=7., frontier_best_progress=.9)
        ppo = dict(approx_kl=0., entropy=0., explained_variance=0.)
        current = config('cograph_v2.json')
        line = episode_line(1, current, summary, 0., ppo)
        for text in ('map_done=1 ', 'route_delta_m=-3.4 ', 'frontier=0.65 '):
            self.assertIn(text, line)
        self.assertNotIn('route_saved_m=', line)
        self.assertNotIn('arrived=', line)
        line = episode_line(1, dict(current, cograph_route_reward='legacy'), summary, 0., ppo)
        for text in ('map_done=1 ', 'route_saved_m=7.0 ', 'frontier=0.90 '):
            self.assertIn(text, line)


if __name__ == '__main__':
    unittest.main()
