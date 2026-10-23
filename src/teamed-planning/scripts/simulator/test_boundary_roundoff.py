"""Boundary roundoff must not hide real violations or change clamp dynamics."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.env import MultiRobotEnv, EnvConfig, BOUNDARY_ROUNDOFF_TOLERANCE_M
from drone.cbf_safety import CBFSafetyFilter


class BoundaryRoundoffTests(unittest.TestCase):
    def make_env(self, kind='drone'):
        with contextlib.redirect_stdout(io.StringIO()):
            env = MultiRobotEnv(EnvConfig(world_size=(1000.,1000.), dt=1.))
        if kind == 'drone':
            env.add_drone(50.,50.,15.)
            robot = env.drones[0]
        else:
            env.add_ground_robot(50.,50.)
            robot = env.ground_robots[0]
        return env, robot

    def check(self, env):
        info = dict(out_of_bounds=[])
        env._check_boundaries(info)
        return info

    def test_roundoff_on_all_edges_is_recorded_separately(self):
        for kind in ('drone','ground'):
            env, robot = self.make_env(kind)
            for xy in ((-1e-14,50.), (50.,-1e-14),
                       (np.nextafter(1000.,np.inf),50.), (50.,np.nextafter(1000.,np.inf))):
                with self.subTest(kind=kind,xy=xy):
                    robot.position[:2] = xy
                    robot.velocity[:] = [1.,2.,0.]
                    info = self.check(env)
                    self.assertEqual(info['out_of_bounds'], [])
                    self.assertEqual(len(info['boundary_roundoff']), 1)
                    self.assertEqual(info['boundary_violations'], [])
                    np.testing.assert_array_equal(robot.position[:2], np.clip(xy,0.,1000.))
                    np.testing.assert_array_equal(robot.velocity,np.zeros(3))

    def test_real_overshoot_keeps_violation_and_stop(self):
        for kind in ('drone','ground'):
            env, robot = self.make_env(kind)
            for xy in ((-1e-6,50.), (1000.001,50.), (50.,-0.1), (50.,1001.)):
                with self.subTest(kind=kind,xy=xy):
                    robot.position[:2] = xy
                    robot.velocity[:] = [1.,2.,0.]
                    info = self.check(env)
                    self.assertEqual(info['out_of_bounds'], [robot.robot_id])
                    self.assertEqual(info['boundary_roundoff'], [])
                    self.assertGreater(info['boundary_violations'][0]['overshoot_m'], BOUNDARY_ROUNDOFF_TOLERANCE_M)
                    np.testing.assert_array_equal(robot.velocity,np.zeros(3))

    def test_threshold_and_mixed_corner_do_not_hide_real_violation(self):
        env, robot = self.make_env()
        for excess, numerical in ((BOUNDARY_ROUNDOFF_TOLERANCE_M,True),
                (np.nextafter(BOUNDARY_ROUNDOFF_TOLERANCE_M,np.inf),False)):
            robot.position[:2] = [-excess,50.]
            info = self.check(env)
            self.assertEqual(bool(info['boundary_roundoff']), numerical)
            self.assertEqual(bool(info['out_of_bounds']), not numerical)
        robot.position[:2] = [-1e-14,1001.]
        info = self.check(env)
        self.assertEqual(info['out_of_bounds'],[robot.robot_id])
        self.assertEqual(info['boundary_roundoff'],[])

    def test_in_bounds_motion_is_untouched(self):
        env, robot = self.make_env()
        for xy in ((0.,0.),(1000.,1000.),(1e-14,50.)):
            robot.position[:2] = xy
            robot.velocity[:] = [1.,2.,0.]
            info = self.check(env)
            self.assertEqual(info['out_of_bounds'],[])
            self.assertEqual(info['boundary_roundoff'],[])
            np.testing.assert_array_equal(robot.position[:2],xy)
            np.testing.assert_array_equal(robot.velocity,[1.,2.,0.])

    def test_nonfinite_position_is_not_treated_as_roundoff(self):
        env, robot = self.make_env()
        for value in (np.nan,np.inf,-np.inf):
            robot.position[:] = [value,50.,15.]
            with self.assertRaises(FloatingPointError):
                self.check(env)

    def test_cbf_roundoff_is_accepted_but_real_conflict_still_fails(self):
        filt = CBFSafetyFilter()
        _, stats = filt.filter([-2.,0.,0.],[-1e-14,50.,15.],{},1.,world_bounds=(0.,0.,1000.,1000.))
        self.assertFalse(stats['infeasible'])
        _, stats = filt.filter([0.,0.,0.],[0.,50.,15.],{'peer':[1.,50.,15.]},1.,
                              world_bounds=(0.,0.,1000.,1000.))
        self.assertTrue(stats['infeasible'])


if __name__ == '__main__':
    unittest.main()
