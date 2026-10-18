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
from drone.mappo import ActorCritic
from simulator.large_world_scenarios import generate_training_scene
from simulator.mappo_experiment import SafetyTermination
from simulator.train_mappo import fixed_schedule,run_episode,validate_config


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
        self.assertEqual(rollout[-1][-1],-10.)
        self.assertEqual(summary['safety_terminated'],1.)
        self.assertEqual(summary['episode_steps'],1)
        self.assertEqual(summary['rollout_steps'],2)

    def test_large_config_rejects_tasks_that_cannot_fit(self):
        with self.assertRaises(ValueError):
            validate_config(dict(world_size=1000.,task_distances=[950]))


if __name__=='__main__':unittest.main()
