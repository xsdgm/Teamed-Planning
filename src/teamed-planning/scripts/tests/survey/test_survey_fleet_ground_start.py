from types import SimpleNamespace
import unittest

import numpy as np

from survey.survey_fleet_clearance import certified_start_connector, native_segment_clear
from survey.survey_fleet_observation import FleetSurveyAdapter, START_CONNECTOR, _clearance
from survey.survey_pretrain_env import ground_plan


class FleetGroundStartTests(unittest.TestCase):
    def setUp(self):
        adapter = FleetSurveyAdapter.__new__(FleetSurveyAdapter)
        self.adapter = adapter
        self.car = np.array([10.1, 10.2])
        self.tile = np.zeros((16, 16), np.int8)
        self.obs = dict(self=dict(position=self.car, orientation=0.), map_resolution=.5,
                        time_step=1., occupancy_map=self.tile, map_axes='yx')
        adapter.exp = SimpleNamespace(ground=0, obs={0:self.obs},
                                      ground_controller=SimpleNamespace(waypoint_threshold=1.))
        adapter.ep = SimpleNamespace(goal=np.array([25.5, 10.5]))
        adapter.cells = np.full((32, 32), -1, np.int8)
        # A coarse square 2 m above the ego lies in the footprint of its
        # coarse centre. The finer observation can still certify the true pose.
        adapter.cells[12, 10] = 1
        adapter.bounds = (np.array([0, 0]), np.array([31, 31]))
        adapter.observe_receipts = lambda: None
        adapter.connector_plans = 0

    def test_rejoins_known_grid_with_an_executable_native_waypoint(self):
        self.assertNotEqual(_clearance(self.adapter.cells)[10, 10], -1)
        target, reason = self.adapter.plan_ground()
        self.assertEqual(reason, START_CONNECTOR)
        self.assertGreater(np.linalg.norm(target-self.car), 1.)
        self.assertTrue(native_segment_clear(self.tile, self.car, .5, self.car, target))
        self.assertEqual(_clearance(self.adapter.cells)[tuple(np.floor(target).astype(int)[::-1])], -1)
        np.testing.assert_array_equal(self.adapter.ground_path_xy, np.stack((self.car,target)))

    def test_unknown_start_or_missing_sensor_holds(self):
        for missing in (True, False):
            self.obs['sensor_valid'] = not missing
            self.tile[7:9, 7:9] = -1
            target, reason = self.adapter.plan_ground()
            self.assertEqual(reason, 'uncertified_vehicle_clearance')
            np.testing.assert_array_equal(target, self.car)

    def load_isolated_pose(self):
        # Native and receiver snapshots at t=1585s of v9-local-train-534945890,
        # translated by (-843,-129) m. The two circular obstacles leave a diagonal gap.
        fused = ['.........######']*2 + ['........#######']*4 + [
            '.........######', '..........#####', '...##.......###', '#######.....###',
            '#######........', '############...', '############...', '############...', '##########.....']
        native = ['..........######']*3 + ['...........#####']*2 + [
            '............####', '..............##', '................', '................',
            '................', '#...............', '####............', '#####...........',
            '######..........', '######..........', '######..........']
        self.car[:] = [10.9498212508505, 10.89250795556276]
        self.adapter.cells[:] = 0
        self.adapter.cells[3:18, 3:18] = np.array([[1 if c == '#' else -1 for c in row] for row in fused])
        self.tile[:] = np.array([[c == '#' for c in row] for row in native])
        return np.array([121.7121773969574, 175.31925256410454])

    def test_isolated_free_start_rejoins_a_road_with_an_onward_route(self):
        waypoint = self.load_isolated_pose()
        cells = _clearance(self.adapter.cells)
        start = np.floor(self.car).astype(int)[::-1]
        goal = np.floor(waypoint).astype(int)[::-1]
        self.assertEqual(cells[tuple(start)], -1)
        self.assertEqual(ground_plan(cells, start, goal, 1., directional=True)[2], 'hold')
        nearest, _ = certified_start_connector(self.tile, self.car, .5, self.car, cells, [0., 0.],
                                               min_distance_m=1.)
        self.assertEqual(ground_plan(cells, nearest, goal, 1., directional=True)[2], 'hold')
        target, reason = self.adapter.plan_ground(waypoint)
        self.assertEqual(reason, START_CONNECTOR)
        self.assertTrue(native_segment_clear(self.tile, self.car, .5, self.car, target))
        cell = np.floor(target).astype(int)[::-1]
        self.assertGreater(len(ground_plan(cells, cell, goal, 1., directional=True)[0]), 1)
        frames = []
        self.adapter.observe_receipts = lambda: frames.append(1)
        repeated, reason = self.adapter.plan_ground(waypoint + [-25., 30.])
        np.testing.assert_equal(repeated, target)
        self.assertTrue(frames)
        self.assertEqual(self.adapter.isolated_start_recoveries, 1)
        self.obs.pop('occupancy_map')
        target, reason = self.adapter.plan_ground(waypoint)
        self.assertEqual(reason, 'hold')
        np.testing.assert_equal(target, self.car)

    def test_isolated_start_at_its_actual_goal_does_not_recover(self):
        self.load_isolated_pose()
        self.adapter.ep.goal = self.car.copy()
        target, reason = self.adapter.plan_ground()
        self.assertEqual(reason, 'known_goal_route')
        np.testing.assert_equal(target, self.car)

if __name__ == '__main__':
    unittest.main()
