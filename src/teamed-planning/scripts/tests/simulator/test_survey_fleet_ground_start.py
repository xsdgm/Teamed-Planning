from types import SimpleNamespace
import unittest

import numpy as np

from ground.survey_fleet_clearance import native_segment_clear
from simulator.survey_fleet_observation import FleetSurveyAdapter, START_CONNECTOR, _clearance


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

if __name__ == '__main__':
    unittest.main()
