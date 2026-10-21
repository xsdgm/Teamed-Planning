"""Large-world training uses actual long tasks and preserves safety boundaries."""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from drone.mappo import ActorCritic,GLOBAL_FEATURES
from simulator.large_world_scenarios import generate_training_scene
from simulator.mappo_experiment import SafetyTermination
from simulator.train_mappo import fixed_schedule,run_episode,score_validation,validate_config


class LargeTrainingTests(unittest.TestCase):
    def test_task_generator_is_deterministic_and_uses_long_random_poses(self):
        a=generate_training_scene(50001)
        b=generate_training_scene(50001)
        self.assertEqual(a,b)
        self.assertIn(a['distance'],[350,550,750])
        self.assertEqual(len(a['circles']),3438)
        start,goal=np.array(a['start']),np.array(a['goal'])
        self.assertAlmostEqual(float(np.linalg.norm(goal-start)),a['distance'])
        self.assertTrue(((start>=35)&(start<=965)).all())
        self.assertTrue(((goal>=35)&(goal<=965)).all())
        self.assertGreater(a['reference_length'],a['distance']-2.)
        self.assertGreater(abs(goal[1]-start[1]),1.)

    def scene(self):
        return dict(start=[100.,500.],goal=[850.,500.],circles=[],attempts=1,
                    reference_length=750.)

    def config(self):
        return validate_config(dict(world_size=1000.,steps=3,initial_validation=False))

    def test_budget_and_long_task_flow_into_real_environment(self):
        config=self.config();progress=[]
        torch.set_num_threads(1)
        with patch('simulator.train_mappo.generate_training_scene',return_value=self.scene()):
            rollout,summary=run_episode(ActorCritic(),config,50001,
                fixed_schedule(config,0),True,progress=progress.append)
        self.assertEqual(len(rollout),3)
        # Critic state: elapsed budget fraction, then remaining steps/1000.
        self.assertEqual([len(r[5]) for r in rollout],[GLOBAL_FEATURES]*3)
        np.testing.assert_allclose([r[5][0] for r in rollout],[0.,1/3,2/3],rtol=1e-6)
        np.testing.assert_allclose(rollout[0][5][1],.003,rtol=1e-6)
        self.assertTrue(all(np.isfinite(r[5]).all() for r in rollout))
        self.assertEqual(summary['task_distance_m'],750.)
        self.assertEqual(summary['episode_budget_steps'],3)
        self.assertEqual(summary['budget_exhausted'],1.)
        self.assertEqual(progress[0]['world_size'],1000.)
        self.assertLessEqual(summary['max_link_distance_m'],95.)

    def test_cbf_failure_records_terminal_action_without_executing_it(self):
        from simulator.mappo_experiment import Experiment
        config=self.config();original=Experiment.step;calls=[]
        def fail_second(env,frame):
            if frame==1:
                raise SafetyTermination(1,[c.navigator.last_sample for c in env.controllers])
            calls.append(frame)
            return original(env,frame)
        torch.manual_seed(1)
        with patch('simulator.train_mappo.generate_training_scene',return_value=self.scene()), \
             patch.object(Experiment,'step',fail_second):
            rollout,summary=run_episode(ActorCritic(),config,50001,fixed_schedule(config,0),True)
        self.assertEqual(calls,[0])
        self.assertEqual(len(rollout),2)
        self.assertEqual(rollout[-1][4],-10.)
        self.assertEqual(len(rollout[-1][5]),GLOBAL_FEATURES)
        self.assertEqual(summary['safety_terminated'],1.)
        self.assertEqual(summary['episode_steps'],1)
        self.assertEqual(summary['rollout_steps'],2)

    def test_arrival_ends_the_episode_with_the_terminal_reward(self):
        from simulator.mappo_experiment import Experiment
        config=validate_config(dict(world_size=1000.,steps=5,initial_validation=False,
            termination='arrival',arrival_reward=100.,subgoal_radius=200.))
        original=Experiment.step;rewards=[]
        def arrive_second(env,frame):
            result=original(env,frame)
            rewards.append(result[1])
            env.goal_reached=frame==1
            return result
        torch.manual_seed(1)
        with patch('simulator.train_mappo.generate_training_scene',return_value=self.scene()), \
             patch.object(Experiment,'step',arrive_second):
            rollout,summary=run_episode(ActorCritic(),config,50001,fixed_schedule(config,0),True)
        self.assertEqual(len(rollout),2)
        self.assertAlmostEqual(rollout[-1][4],rewards[-1]+100.)
        self.assertAlmostEqual(summary['reward'],sum(rewards)+100.)
        self.assertEqual((summary['goal_reached'],summary['budget_exhausted']),(1.,0.))
        self.assertEqual(summary['subgoals_selected'],1.)

    def test_arrival_validation_ranks_vehicle_arrival(self):
        common=dict(seed=1,sample=0,graph_task_path_ratio=1.,reward=0.,ground_collisions=0.,
                    collisions=0.,chain_link_violation=0.,safety_terminated=0.,graph_first_connection_step=10.)
        arrived=[dict(common,goal_reached=1.,graph_task_connected=0.,episode_steps=900)]
        connected=[dict(common,goal_reached=0.,graph_task_connected=1.,episode_steps=4000)]
        arrival=validate_config(dict(world_size=1000.,termination='arrival'))
        graph=validate_config(dict(world_size=1000.))
        self.assertGreater(score_validation(arrived,arrival)['score'],score_validation(connected,arrival)['score'])
        self.assertLess(score_validation(arrived,graph)['score'],score_validation(connected,graph)['score'])

    def test_subgoal_and_arrival_options_are_validated(self):
        for bad in (dict(world_size=1000.,termination='goal'),dict(termination='arrival'),
                    dict(world_size=1000.,subgoal_radius=10.),dict(world_size=1000.,arrival_reward=-1.),
                    dict(world_size=1000.,init_checkpoint=3)):
            with self.assertRaises(ValueError):
                validate_config(bad)

    def test_large_config_rejects_tasks_that_cannot_fit(self):
        with self.assertRaises(ValueError):
            validate_config(dict(world_size=1000.,task_distances=[950]))


if __name__=='__main__':unittest.main()
