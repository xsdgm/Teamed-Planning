"""The v9 current-route reward must not inherit fixed-start completion rules."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator.train_mappo import (episode_line, fixed_schedule, optimizer_config_changes,
    reward_contract, run_episode, score_validation, validate_config)

CONFIGS = Path(__file__).resolve().parents[3] / 'config' / 'algorithm'


def config(version):
    return validate_config(json.loads((CONFIGS / f'mappo_training_{version}.json').read_text()))


def record(**overrides):
    return dict(dict(goal_reached=0., map_completed=0., episode_steps=300.,
        map_completion_step=50., graph_task_path_ratio=1.2, reward=0., collisions=0.,
        ground_collisions=0., chain_link_violation=0., route_efficiency=.8,
        safety_terminated=0.), **overrides)


class TrainingV9Tests(unittest.TestCase):
    def test_mixed_distance_config_and_legacy_default_are_separate(self):
        v8, v9 = config('v8'), config('v9')
        self.assertEqual(v8['mapping_route_origin'], 'initial')
        self.assertEqual(reward_contract(v8), 'uav_mapping_route_quality_and_backbone_v7_surprise')
        self.assertEqual(reward_contract(v9), 'uav_remaining_route_mapping_v9')
        self.assertEqual(v9['task_distance_range'], [100., 400.])
        self.assertEqual(v9['task_distances'], [])
        self.assertIsNone(v9['subgoal_radius'])
        self.assertEqual(v9['termination'], 'arrival')
        self.assertEqual(v9['distance_sampling'], 'annulus_area')
        # The current-position objective also supports optional rolling targets,
        # but the requested mixed-distance training keeps one fixed task goal.
        rolling = validate_config(dict(v9, subgoal_radius=200.))
        self.assertEqual(rolling['subgoal_radius'], 200.)
        with self.assertRaises(ValueError):
            optimizer_config_changes(v9, rolling)

    def test_remaining_route_rejects_fixed_start_termination_and_rewards(self):
        v9 = config('v9')
        for overrides in ({'termination': 'map'}, {'ground_stationary': True},
                          {'map_success_reward': 100.}, {'map_quality_reward': 20.},
                          {'map_improvement_steps': 150}, {'surprise_weight': .05}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_config(dict(v9, **overrides))
        with self.assertRaises(ValueError):
            validate_config(dict(config('v8'), subgoal_radius=200.))

    def test_reward_change_is_not_an_optimizer_resume_upgrade(self):
        for upgrade in (False, True):
            with self.subTest(upgrade=upgrade), self.assertRaises(ValueError):
                optimizer_config_changes(config('v8'), config('v9'), allow_upgrade=upgrade)

    def test_checkpoint_selection_requires_arrival_and_vehicle_safety(self):
        v9 = config('v9')
        mapped = score_validation([record(map_completed=1.)], v9)
        arrived = score_validation([record(goal_reached=1.)], v9)
        collided = score_validation([record(goal_reached=1., ground_collisions=1.)], v9)
        self.assertEqual(arrived['success_metric'], 'goal_reached')
        self.assertGreater(arrived['score'], mapped['score'])
        self.assertGreater(mapped['score'], collided['score'])
        self.assertEqual(score_validation([record(map_completed=1.)], config('v8'))['success_metric'],
                         'map_completed')

    def test_progress_line_uses_signed_remaining_route_and_arrival(self):
        summary=dict(reward_beta=1., reward_eta=0., confidence_area=12., task_share=.5,
            goal_reached=1., map_completed=0., route_length_delta_m=-3.4,
            frontier_progress=.65, route_new_best_saving_m=7., frontier_best_progress=.9)
        ppo=dict(approx_kl=0., entropy=0., explained_variance=0.)
        v9_line=episode_line(1,config('v9'),summary,0.,ppo)
        for text in ('arrived=1 ', 'route_delta_m=-3.4 ', 'frontier=0.65 '):
            self.assertIn(text,v9_line)
        self.assertNotIn('map_done=',v9_line)
        self.assertNotIn('route_saved_m=',v9_line)
        v8_line=episode_line(1,config('v8'),summary,0.,ppo)
        for text in ('map_done=0 ', 'route_saved_m=7.0 ', 'frontier=0.90 '):
            self.assertIn(text,v8_line)
        self.assertNotIn('route_delta_m=',v8_line)

    def test_rollout_uses_arrival_and_preserves_remaining_reference(self):
        """A complete map at frame 1 must keep running until vehicle frame 3."""
        class FakeExperiment:
            def __init__(self):
                self.ground = 3
                self.initial_goal_distance = 100.
                self.obs = {3: {'self': {'position': [0., 0., 0.]}}}
                self.ledger = SimpleNamespace(seen=np.zeros((2, 2), bool))
                self.env = SimpleNamespace(close=lambda: None)
                self.trace = []
                self.phases = []
                self.goal_reached = False
                self.done = False

            def critic_state(self, frame, budget, phase):
                self.phases.append(phase)
                return np.zeros(10)

            def step(self, frame):
                self.goal_reached = frame == 2
                metrics = dict(positions=[[0., 0., 15.]] * 3,
                    chain_max_distance_ratio=.5, graph_task_connected=1.,
                    chain_links_available=1., ground_goal_distance=100.-frame,
                    route_target_changed=float(frame == 1))
                self.trace.append(metrics)
                return [(np.zeros((2, 2)), np.ones(2), np.zeros(2), 0.)] * 3, 0., metrics

            def summary(self):
                return dict(reward=0., graph_task_connected=1., graph_task_path_length=10.,
                    route_quality=.2, route_reference_length_m=2., episode_steps=len(self.trace),
                    goal_reached=float(self.goal_reached))

        v9 = validate_config(dict(config('v9'), steps=4, map_stability_steps=1))
        fake = FakeExperiment()
        scene = dict(start=[0., 0.], goal=[100., 0.], circles=[], attempts=1,
                     reference_length=110., reference_polyline_length=108.)
        with patch('simulator.train_mappo.generate_training_scene', return_value=scene), \
             patch('simulator.train_mappo.Experiment', return_value=fake) as experiment:
            rollout, summary = run_episode(None, v9, 60000, fixed_schedule(v9, 0), False)
        self.assertEqual(len(rollout), 3)
        self.assertEqual(summary['reward'], v9['arrival_reward'])
        self.assertEqual(summary['map_completion_bonus'], 0.)
        self.assertEqual(summary['map_completed'], 0.)
        self.assertEqual(summary['route_efficiency'], .2)
        self.assertEqual(summary['route_reference_length_m'], 2.)
        self.assertEqual(summary['initial_route_reference_length_m'], 108.)
        self.assertEqual(experiment.call_args.kwargs['mapping_route_origin'], 'vehicle')
        self.assertIsNone(experiment.call_args.kwargs['subgoal_radius'])
        self.assertNotIn('route_reference_length', experiment.call_args.kwargs)
        # A target switch must restart the critic's graph-stability phase too.
        v9 = validate_config(dict(v9, map_stability_steps=2, subgoal_radius=200.))
        fake = FakeExperiment()
        with patch('simulator.train_mappo.generate_training_scene', return_value=scene), \
             patch('simulator.train_mappo.Experiment', return_value=fake):
            run_episode(None, v9, 60000, fixed_schedule(v9, 0), False)
        self.assertEqual(fake.phases, [0., .5, .5])


if __name__ == '__main__':
    unittest.main()
