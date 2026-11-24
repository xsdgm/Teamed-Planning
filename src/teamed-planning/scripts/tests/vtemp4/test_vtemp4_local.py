"""Check the online obstacle stage's episode count, seed schedule and restart boundary."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vtemp4 import train_vtemp4_local as local
from vtemp4 import run_vtemp4_continuous_history as supervisor
from vtemp4.vtemp4_ring_contract import (
    BODY_ANGLE_GOAL_RING_CONTRACTS, NO_GRID_GOAL_VARIANT, PERIODIC_GOAL_RING_CONTRACTS,
    RING_UNIFIED_GOAL_NO_GRID_FORMAT, RING_VEHICLE_GRAPH_BODY_LSTM_FORMAT, VEHICLE_GRAPH_BODY_LSTM_VARIANT,
    RING_VEHICLE_GRAPH_GOAL_FORMAT, RING_VEHICLE_GRAPH_LSTM_FORMAT, VEHICLE_GRAPH_GOAL_VARIANT,
    VEHICLE_GRAPH_LSTM_VARIANT, GOAL_RESIDUAL_RING_CONTRACTS,
    RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT, VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_VARIANT,
)


class OnlineTrainingPlanTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        archive = root / local.V9_RUN
        (archive / 'runtime_sources').mkdir(parents=True)
        (archive / 'runtime_sources/simulator.py').write_text('# archived runtime\n')
        (archive / 'config.json').write_text('{}')
        (archive / 'best.pt').write_bytes(b'archived policy')
        expected = local.file_sha256(archive / 'best.pt')
        self.enterContext(patch.object(local, 'CHECKPOINT_SHA', expected))
        output = root / 'online'
        output.mkdir()
        self.args = SimpleNamespace(repo=root, output=output, seed=43, updates=25,
                                    episodes_per_update=16, workers=6, learning_rate=.0003, resume=False)

    def test_four_hundred_distinct_online_missions(self):
        first = local.manifest_for(self.args)
        self.assertEqual(first['training_stage'], 1)
        self.assertEqual(first['total_episodes'], 400)
        self.assertEqual(len(first['plan']), 400)
        scene_seeds = {entry['scene_seed'] for entry in first['plan']}
        fast_seeds = {entry['fast_seed'] for entry in first['plan']}
        self.assertEqual(len(scene_seeds), 400)
        self.assertEqual(len(fast_seeds), 400)
        self.assertTrue(scene_seeds.isdisjoint(fast_seeds))
        self.args.workers = 2
        self.assertEqual(local.manifest_for(self.args), first)

    def test_resume_rejects_another_seed_before_touching_training(self):
        path = self.args.output / 'manifest.json'
        local.atomic_json(path, local.manifest_for(self.args))
        original = path.read_bytes()
        self.args.seed += 1
        self.args.resume = True
        with patch.object(local, 'runtime') as simulator:
            with self.assertRaisesRegex(ValueError, 'Resume configuration'):
                local.run(self.args)
            simulator.assert_not_called()
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.args.output / 'checkpoints').exists())

    def test_thirty_second_goal_policy_keeps_the_paired_mission_plan(self):
        self.args.policy_variant = 'periodic_unified_attention_goal'
        baseline = local.manifest_for(self.args)
        self.args.review_interval_s = 30
        changed = local.manifest_for(self.args)
        self.assertEqual(baseline['clock'], dict(review_interval_s=15, lookahead_m=200.))
        self.assertEqual(changed['clock'], dict(review_interval_s=30, lookahead_m=200.))
        self.assertNotEqual(baseline['action_contract'], changed['action_contract'])
        self.assertEqual(baseline['plan'], changed['plan'])

    def test_progress_reward_is_bound_to_the_run(self):
        self.args.policy_variant = 'periodic_unified_attention_goal'
        self.args.review_interval_s = 30
        baseline = local.manifest_for(self.args)
        self.args.progress_reward_weight = 8.
        weighted = local.manifest_for(self.args)
        self.assertEqual(weighted, dict(baseline, progress_reward_weight=8.))
        local.atomic_json(self.args.output / 'manifest.json', weighted)
        self.args.resume = True
        self.args.progress_reward_weight = 2.
        with self.assertRaisesRegex(ValueError, 'Resume configuration'):
            local.run(self.args)

    def test_traversable_reward_pins_distance_and_uses_the_same_circle_missions(self):
        self.args.policy_variant = VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_VARIANT
        baseline = local.manifest_for(self.args)
        self.args.progress_distance = 'traversable'
        route = local.manifest_for(self.args)
        self.assertEqual(route, dict(baseline, progress_distance='traversable', waypoint_progress_bonus_ratio=0.))
        local.atomic_json(self.args.output / 'manifest.json', route)
        self.args.resume = True
        self.args.progress_distance = 'euclidean'
        with self.assertRaisesRegex(ValueError, 'Resume configuration'):
            local.run(self.args)
        self.args.progress_distance = None
        with patch.object(local, 'runtime', side_effect=RuntimeError('reached runtime setup')):
            with self.assertRaisesRegex(RuntimeError, 'reached runtime setup'):
                local.run(self.args)
        self.assertEqual(self.args.progress_distance, 'traversable')
        self.assertEqual(self.args.waypoint_progress_bonus_ratio, 0.)

    def test_route_supervisor_passes_reward_to_trainer_and_evaluation(self):
        out = self.args.output
        (out / 'logs').mkdir()
        with patch.object(supervisor, 'freeze'), patch.object(supervisor, 'select_cores', return_value=[1]), \
                patch.object(supervisor, 'launch') as launch:
            launch.return_value.pid = 9
            launch.return_value.wait.return_value = 0
            supervisor.run(self.args.repo, out, False, workers=1, progress_distance='traversable')
        train, evaluate = [call.args[1] for call in launch.call_args_list]
        for args in (train, evaluate):
            self.assertEqual(args[args.index('--progress-distance') + 1], 'traversable')
        self.assertEqual(train[train.index('--waypoint-progress-bonus-ratio') + 1], 0.)
        self.assertTrue(evaluate[evaluate.index('--lake-cases') + 1].startswith('--'))

    def test_full_history_defaults_and_explicit_clock_and_reward(self):
        self.args.policy_variant = VEHICLE_GRAPH_LSTM_VARIANT
        for interval, weight, expected_s, expected_weight in ((None, None, 30, 10.), (15, None, 15, 10.),
                                                              (None, 2., 30, 2.), (15, 8., 15, 8.)):
            with self.subTest(interval=interval, weight=weight):
                self.args.review_interval_s = interval
                self.args.progress_reward_weight = weight
                manifest = local.manifest_for(self.args)
                self.assertEqual(manifest['clock'], dict(review_interval_s=expected_s, lookahead_m=200.))
                self.assertEqual(manifest['action_contract'], PERIODIC_GOAL_RING_CONTRACTS[expected_s])
                self.assertEqual(manifest['progress_reward_weight'], expected_weight)

    def test_waypoint_bonus_preserves_circle_missions_and_pins_resume(self):
        self.args.policy_variant = VEHICLE_GRAPH_LSTM_VARIANT
        baseline = local.manifest_for(self.args)
        self.assertNotIn('waypoint_progress_bonus_ratio', baseline)
        self.args.waypoint_progress_bonus_ratio = 0.
        self.assertEqual(local.manifest_for(self.args), baseline)
        self.args.waypoint_progress_bonus_ratio = .10
        weighted = local.manifest_for(self.args)
        self.assertEqual(weighted, dict(baseline, waypoint_progress_bonus_ratio=.10))

        local.atomic_json(self.args.output / 'manifest.json', weighted)
        self.args.resume = True
        self.args.waypoint_progress_bonus_ratio = None
        with patch.object(local, 'runtime', side_effect=RuntimeError('reached runtime setup')):
            with self.assertRaisesRegex(RuntimeError, 'reached runtime setup'):
                local.run(self.args)
        self.assertEqual(self.args.waypoint_progress_bonus_ratio, .10)
        self.args.waypoint_progress_bonus_ratio = 0.
        with self.assertRaisesRegex(ValueError, 'Resume configuration'):
            local.run(self.args)

    def test_body_angle_arm_uses_the_same_circle_missions_clock_and_reward(self):
        self.args.waypoint_progress_bonus_ratio = .10
        for interval in (None, 15):
            with self.subTest(interval=interval):
                self.args.review_interval_s = interval
                self.args.policy_variant = VEHICLE_GRAPH_LSTM_VARIANT
                baseline = local.manifest_for(self.args)
                self.args.policy_variant = VEHICLE_GRAPH_BODY_LSTM_VARIANT
                manifest = local.manifest_for(self.args)
                self.assertEqual(manifest, dict(baseline, policy_variant=VEHICLE_GRAPH_BODY_LSTM_VARIANT,
                                                policy_format=RING_VEHICLE_GRAPH_BODY_LSTM_FORMAT,
                                                action_contract=BODY_ANGLE_GOAL_RING_CONTRACTS[interval or 30]))
                self.assertNotEqual(manifest['action_contract'], baseline['action_contract'])
        local.atomic_json(self.args.output / 'manifest.json', manifest)
        self.args.resume = True
        self.args.policy_variant = VEHICLE_GRAPH_LSTM_VARIANT
        with self.assertRaisesRegex(ValueError, 'Resume configuration'):
            local.run(self.args)

    def test_resume_full_history_without_archived_weight_preserves_manifest(self):
        self.args.policy_variant = VEHICLE_GRAPH_LSTM_VARIANT
        self.args.review_interval_s = 15
        old = local.manifest_for(self.args)
        old.pop('progress_reward_weight', None)
        path = self.args.output / 'manifest.json'
        local.atomic_json(path, old)
        original = path.read_bytes()
        self.args.policy_variant = self.args.review_interval_s = self.args.progress_reward_weight = None
        self.args.resume = True
        with patch.object(local, 'runtime', side_effect=RuntimeError('reached runtime setup')):
            with self.assertRaisesRegex(RuntimeError, 'reached runtime setup'):
                local.run(self.args)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.args.review_interval_s, 15)

    def test_residual_manifest_preserves_the_paired_plan_and_disambiguates_log_probabilities(self):
        for interval in (None, 15):
            self.args.review_interval_s = interval
            self.args.policy_variant = VEHICLE_GRAPH_BODY_LSTM_VARIANT
            self.args.waypoint_progress_bonus_ratio = .10
            body = local.manifest_for(self.args)
            self.args.policy_variant = VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_VARIANT
            self.args.waypoint_progress_bonus_ratio = None
            residual = local.manifest_for(self.args)
            self.assertEqual(residual, dict(body, policy_variant=self.args.policy_variant,
                                            policy_format=RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT,
                                            action_contract=GOAL_RESIDUAL_RING_CONTRACTS[interval or 30]))
        self.args.waypoint_progress_bonus_ratio = 0.
        zero_bonus = local.manifest_for(self.args)
        self.assertNotIn('waypoint_progress_bonus_ratio', zero_bonus)
        local.atomic_json(self.args.output / 'manifest.json', zero_bonus)
        self.args.resume = True
        self.args.policy_variant = self.args.waypoint_progress_bonus_ratio = None
        with patch.object(local, 'runtime', side_effect=RuntimeError('reached runtime setup')):
            with self.assertRaisesRegex(RuntimeError, 'reached runtime setup'):
                local.run(self.args)
        self.assertEqual(self.args.policy_variant, VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_VARIANT)
        self.args.policy_variant = VEHICLE_GRAPH_BODY_LSTM_VARIANT
        with self.assertRaisesRegex(ValueError, 'Resume configuration'):
            local.run(self.args)

    def test_supervisor_launches_the_requested_or_saved_variant(self):
        residual = VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_VARIANT
        body = VEHICLE_GRAPH_BODY_LSTM_VARIANT
        configurations = [(False, None, None, residual, .10), (False, None, 0., residual, 0.),
                          (False, body, None, body, None), (True, None, None, body, None),
                          (True, None, 0., residual, None), (True, None, .10, residual, .10)]
        for index, (resume, requested, bonus, expected, expected_bonus) in enumerate(configurations):
            with self.subTest(resume=resume, requested=requested, bonus=bonus):
                out = self.args.output / str(index)
                (out / 'logs').mkdir(parents=True)
                if resume:
                    (out / 'sources/world_model').mkdir(parents=True)
                    self.args.policy_variant = expected
                    self.args.waypoint_progress_bonus_ratio = bonus
                    local.atomic_json(out / 'manifest.json', local.manifest_for(self.args))
                with patch.object(supervisor, 'freeze'), patch.object(supervisor, 'select_cores', return_value=[1]), \
                        patch.object(supervisor, 'launch') as launch:
                    launch.return_value.pid = 9
                    launch.return_value.wait.return_value = 0
                    supervisor.run(self.args.repo, out, resume, policy_variant=requested,
                                   waypoint_progress_bonus_ratio=None if resume else bonus, workers=1)
                train, evaluate = [call.args[1] for call in launch.call_args_list]
                for args in (train, evaluate):
                    self.assertEqual(args[args.index('--policy-variant') + 1], expected)
                self.assertEqual('--resume' in train, resume)
                self.assertEqual(train[train.index('--review-interval-s') + 1], 30)
                self.assertEqual(train[train.index('--progress-reward-weight') + 1], 10.)
                if expected_bonus is None:
                    self.assertNotIn('--waypoint-progress-bonus-ratio', train)
                else:
                    self.assertEqual(train[train.index('--waypoint-progress-bonus-ratio') + 1], expected_bonus)

    def test_goal_variants_only_change_policy_identity_in_paired_manifests(self):
        self.args.progress_reward_weight = 8.
        for seconds in (15, 30):
            self.args.review_interval_s = seconds
            self.args.policy_variant = 'periodic_unified_attention_goal'
            baseline = local.manifest_for(self.args)
            for variant, fmt in ((NO_GRID_GOAL_VARIANT, RING_UNIFIED_GOAL_NO_GRID_FORMAT),
                                 (VEHICLE_GRAPH_GOAL_VARIANT, RING_VEHICLE_GRAPH_GOAL_FORMAT),
                                 (VEHICLE_GRAPH_LSTM_VARIANT, RING_VEHICLE_GRAPH_LSTM_FORMAT)):
                with self.subTest(review_interval_s=seconds, variant=variant):
                    self.args.policy_variant = variant
                    self.assertEqual(local.manifest_for(self.args), dict(baseline, policy_variant=variant,
                                                                        policy_format=fmt))

    def test_resume_reads_archived_policy_when_no_variant_was_requested(self):
        self.args.policy_variant = NO_GRID_GOAL_VARIANT
        self.args.review_interval_s = 30
        self.args.progress_reward_weight = 8.
        local.atomic_json(self.args.output / 'manifest.json', local.manifest_for(self.args))
        self.args.policy_variant = self.args.review_interval_s = self.args.progress_reward_weight = None
        self.args.resume = True
        with patch.object(local, 'runtime', side_effect=RuntimeError('reached runtime setup')):
            with self.assertRaisesRegex(RuntimeError, 'reached runtime setup'):
                local.run(self.args)
        self.assertEqual(self.args.policy_variant, NO_GRID_GOAL_VARIANT)
        self.assertEqual(self.args.review_interval_s, 30)
        self.assertEqual(self.args.progress_reward_weight, 8.)


if __name__ == '__main__':
    unittest.main()
