"""Collection regressions for changing a target during a prediction horizon."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from world_model import data


class FakeEnvironment:
    def __init__(self):
        self.env = SimpleNamespace(time=0.)
        self.subgoals = SimpleNamespace(target=np.array([100., 0.]), final=False,
            arrival=1., selected=1, target_status='unverified', last_reason='initial')
        self.ground_controller = SimpleNamespace(position=np.zeros(2),
            reconnaissance=SimpleNamespace(describe=lambda: {'planning_radius_m': 200.}))
        self.goal = np.array([500., 0.])
        self.goal_reached = False
        self.initial_goal_distance = 500.
        self.executed_targets = []

    def step(self, frame):
        self.executed_targets.append(self.subgoals.target.copy())
        self.env.time += 1.
        self.ground_controller.position[0] += .1
        if frame in (2, 7):
            self.subgoals.target = np.array([100., (frame+1)*10.])
            self.subgoals.selected += 1
            self.subgoals.last_reason = 'blocked'


class CollectionTests(unittest.TestCase):
    def test_zero_duration_cbf_terminal_preserves_last_executed_velocity(self):
        env = FakeEnvironment()

        def step(frame):
            if frame == 10:
                raise RuntimeError('CBF rejected before physics')
            env.env.time += 1.
            env.ground_controller.position[0] += .1

        env.step = step
        velocities = []

        def encode(controller, goal, now, velocity):
            velocities.append(np.asarray(velocity).copy())
            return (np.zeros((data.CHANNELS, data.SIDE, data.SIDE), np.float32),
                    np.zeros(data.FEATURES, np.float32))

        with TemporaryDirectory() as directory, \
             patch.object(data, 'provenance', return_value={}), \
             patch.object(data, 'build_environment', return_value=(env, RuntimeError)), \
             patch.object(data, 'encode', side_effect=encode):
            summary = data.collect_episode((13, 'test', 20, directory))
            with np.load(Path(directory)/'test_13.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['times'], [0., 10., 10.])
                self.assertEqual(saved['labels'][-1, 2], 1.)
                self.assertTrue(saved['terminated'][-1])
            np.testing.assert_allclose(velocities[-1], [.1, 0.])
            np.testing.assert_array_equal(velocities[-1], velocities[-2])
            self.assertEqual(summary['reason'], 'cbf_infeasible')

    def test_arrival_label_when_final_coordinate_was_still_unverified(self):
        env = FakeEnvironment()
        env.goal = env.subgoals.target.copy()
        env.ground_controller.position = np.array([98.4, 0.])

        def arrive(frame):
            env.env.time += 1.
            env.ground_controller.position[0] = 98.5
            env.goal_reached = True

        env.step = arrive
        observation = (np.zeros((data.CHANNELS, data.SIDE, data.SIDE), np.float32),
                       np.zeros(data.FEATURES, np.float32))
        with TemporaryDirectory() as directory, \
             patch.object(data, 'provenance', return_value={}), \
             patch.object(data, 'build_environment', return_value=(env, RuntimeError)), \
             patch.object(data, 'encode', return_value=observation):
            summary = data.collect_episode((12, 'test', 10, directory))
            with np.load(Path(directory)/'test_12.npz', allow_pickle=False) as saved:
                self.assertEqual(saved['labels'][0, 0], 1.)
                self.assertTrue(saved['terminated'][0])
                self.assertEqual(saved['target_status'][0], 'unverified')
            self.assertEqual(summary['reason'], 'arrival')

    def test_switch_splits_transition_after_executed_old_command(self):
        env = FakeEnvironment()
        observation = (np.zeros((data.CHANNELS, data.SIDE, data.SIDE), np.float32),
                       np.zeros(data.FEATURES, np.float32))
        with TemporaryDirectory() as directory, \
             patch.object(data, 'provenance', return_value={}), \
             patch.object(data, 'build_environment', return_value=(env, RuntimeError)), \
             patch.object(data, 'encode', return_value=observation):
            summary = data.collect_episode((11, 'test', 10, directory))
            with np.load(Path(directory)/'test_11.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['times'], [0., 3., 8., 10.])
                np.testing.assert_allclose(saved['elapsed'], [.3, .5, .2])
                np.testing.assert_array_equal(saved['target_switched'], [True, True, False])
                np.testing.assert_array_equal(saved['switch_reason'], ['blocked', 'blocked', ''])
                np.testing.assert_array_equal(saved['terminated'], [False, False, False])
                np.testing.assert_array_equal(saved['truncated'], [False, False, True])
                np.testing.assert_array_equal(saved['actions'][:, 2], [1., 1., 1.])
                # Every executed control step has exactly the label's target.
                for index, (begin, end) in enumerate(((0, 3), (3, 8), (8, 10))):
                    for target in env.executed_targets[begin:end]:
                        np.testing.assert_array_equal(saved['targets'][index], target)
                self.assertEqual(saved['maps'].shape[0], saved['actions'].shape[0]+1)
                self.assertEqual(saved['algorithm'].item(), 'vtmp2')
                self.assertEqual(saved['schema_version'].item(), 2)
            self.assertEqual(summary['steps'], 10)
            self.assertEqual(summary['transitions'], 3)


if __name__ == '__main__':
    unittest.main()
