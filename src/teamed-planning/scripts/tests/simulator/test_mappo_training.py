"""Reward configuration and shared candidate controls."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from simulator.train_mappo import fixed_schedule, validate_config
from drone.mappo import CandidateNavigator


class TrainingTests(unittest.TestCase):

    def test_historical_reward_modes_are_rejected(self):
        for overrides in ({'reward_schedule':'curriculum'}, {'reward_mode':'legacy'}):
            with self.assertRaises(ValueError):
                validate_config(overrides)

    def test_seed_splits_do_not_overlap(self):
        with self.assertRaises(ValueError):
            validate_config(dict(validation_seeds=[50000]))

    def test_exploration_candidates_are_shared_and_bounded(self):
        obs=dict(time=20.,ground_robots={'g':dict(position=[15,60,0],target=[100,60])})
        nav = CandidateNavigator('drone_0', algorithm='cadmm_controlled', exploration_candidates=5,
                                 world_bounds=(0, 0, 120, 120), comm_range=95.)
        _, mask, candidates, _ = nav.encode((24., 60., 15.), obs)
        self.assertEqual(len(mask), 21)
        self.assertLessEqual(len(candidates), 21)
        with self.assertRaises(ValueError):
            CandidateNavigator('drone_0', algorithm='greedy')
        all_explore=CandidateNavigator('drone_0',exploration_candidates=20)
        self.assertEqual(len(all_explore.encode((24.,60.,15.),obs)[2]),21)


if __name__=='__main__':
    unittest.main()
