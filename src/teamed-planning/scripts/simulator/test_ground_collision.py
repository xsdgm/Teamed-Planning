"""Vehicle-vehicle contact must keep 3D velocities (CoGraph v2 crash at ep18)."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.env import MultiRobotEnv, EnvConfig


class GroundCollisionTests(unittest.TestCase):
    def test_vehicle_contact_reverts_and_observations_still_build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env = MultiRobotEnv(EnvConfig(world_size=(1000., 1000.), dt=1.))
        a = env.add_ground_robot(100., 100., max_speed=1.)
        b = env.add_ground_robot(101.5, 100., theta=np.pi, max_speed=1.)
        env.add_drone(100., 120., 15.)
        env.reset()
        start = [r.position.copy() for r in env.ground_robots]
        # Head-on diff-drive (v, omega): 1.5 m apart, both advance 1 m -> 0.5 m < 2 x 0.3 m.
        obs, info = env.step({a: np.array([1., 0.]), b: np.array([1., 0.])})
        self.assertTrue(any(c['object'] == 'robot' for c in info['collisions']))
        for robot, p in zip(env.ground_robots, start):
            self.assertEqual(robot.velocity.shape, (3,))
            np.testing.assert_array_equal(robot.velocity, np.zeros(3))
            np.testing.assert_array_equal(robot.position, p)
        # The next observation pass computes relative velocities between all robots.
        obs, _ = env.step({})
        self.assertEqual(len(obs), 3)


if __name__ == '__main__':
    unittest.main()
