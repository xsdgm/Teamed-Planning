from types import SimpleNamespace
import unittest

import numpy as np

from survey.survey_fleet_drive import GroundMotionGuard
from survey.survey_fleet_observation import FleetSurveyAdapter, SHARED_GUIDANCE, START_CONNECTOR
from tests.simulator.test_survey_fleet_drive import LocalVehicle


class DirectionalTargetTests(unittest.TestCase):
    def setUp(self):
        adapter = self.adapter = FleetSurveyAdapter.__new__(FleetSurveyAdapter)
        self.car = np.array([11.9, 10.5])
        sensor = dict(self=dict(position=self.car), map_resolution=.5,
                      occupancy_map=np.zeros((16, 16), np.int8), map_axes='yx')
        adapter.exp = SimpleNamespace(ground=0, obs={0:sensor},
                                      ground_controller=SimpleNamespace(waypoint_threshold=1.))
        adapter.ep = SimpleNamespace(goal=np.array([25.5, 10.5]))
        adapter.cells = np.full((32, 32), -1, np.int8)
        adapter.cells[:, 15] = 1
        adapter.bounds = (np.array([0, 0]), np.array([31, 31]))
        adapter.observe_receipts = lambda: None

    def test_reached_directional_endpoint_hands_off_and_keeps_observed_guard_path(self):
        marker = np.array([29.5, 10.5])
        target, reason = self.adapter.plan_ground(marker)
        self.assertEqual(reason, SHARED_GUIDANCE)
        np.testing.assert_array_equal(target, marker)
        np.testing.assert_array_equal(self.adapter.ground_guidance, marker)
        np.testing.assert_array_equal(self.adapter.ground_target, marker)
        np.testing.assert_equal(self.adapter.ground_path_xy, [[11.5, 10.5], [12.5, 10.5]])

        self.car[0] = 8.5
        next_marker = np.array([27.5, 11.5])
        target, reason = self.adapter.plan_ground(next_marker)
        self.assertEqual(reason, SHARED_GUIDANCE)
        np.testing.assert_array_equal(target, next_marker)
        self.assertGreater(np.linalg.norm(self.adapter.ground_path_xy[-1]-self.car), 1.)

        self.adapter.cells[:, 15] = -1
        target, reason = self.adapter.plan_ground(next_marker)
        self.assertEqual(reason, 'known_goal_route')
        np.testing.assert_array_equal(target, next_marker)
        self.adapter.cells[:, 15] = 1
        target, reason = self.adapter.plan_ground(next_marker)
        self.assertEqual(reason, 'directional_route')
        np.testing.assert_array_equal(target, self.adapter.ground_path_xy[-1])

    def test_current_cell_best_with_a_safe_neighbour_can_delegate_only_with_guidance(self):
        self.car[:] = [12.5, 10.5]
        target, reason = self.adapter.plan_ground(self.adapter.ep.goal)
        self.assertEqual(reason, SHARED_GUIDANCE)
        np.testing.assert_array_equal(target, self.adapter.ep.goal)
        np.testing.assert_array_equal(self.adapter.ground_path_xy, self.car[None, :])
        target, reason = self.adapter.plan_ground()
        self.assertIsNone(self.adapter.ground_guidance)
        self.assertEqual(reason, 'reachable_frontier')
        np.testing.assert_array_equal(target, self.adapter.ground_path_xy[-1])
        self.car[0] = 8.5
        target, reason = self.adapter.plan_ground(self.adapter.ep.goal)
        self.assertEqual(reason, 'directional_route')
        np.testing.assert_array_equal(target, self.adapter.ground_path_xy[-1])

    def attach_frontier_guard(self):
        adapter = self.adapter
        adapter.cells[:] = 0
        adapter.connector_plans = 0
        adapter.ground_path_xy = self.car[None, :].copy()
        obs = adapter.exp.obs[adapter.exp.ground]
        obs.update(time_step=1.)
        obs['self']['orientation'] = 0.
        adapter.exp.goal_reached = False
        adapter.exp.env = SimpleNamespace(time=1180.)
        adapter.exp.ground_controller.local_planner = LocalVehicle()
        adapter.exp.ground_controller.global_planner = SimpleNamespace(waiting_at_frontier=False)
        adapter.ep.holding = False
        guard = adapter.ep.motion_guard = GroundMotionGuard(adapter)
        return guard, obs

    def test_only_active_observed_detour_can_cross_a_gap_without_a_coarse_start(self):
        adapter = self.adapter
        guard, obs = self.attach_frontier_guard()
        guide = self.car + [20., 0.]
        target, reason = adapter.plan_ground(guide)
        self.assertEqual(reason, 'uncertified_vehicle_clearance')
        np.testing.assert_array_equal(target, self.car)

        guard._update_ground_memory(obs['occupancy_map'], self.car, obs['map_resolution'])
        guard.follow_reason = 'observed_frontier'
        guard.follow_origin = self.car.copy()
        guard.bypass_anchor = guide.copy()
        guard.target = self.car + [2., 0.]
        target, reason = adapter.plan_ground(guide)
        self.assertEqual(reason, START_CONNECTOR)
        self.assertGreater(np.linalg.norm(target-self.car), 1.)
        np.testing.assert_array_equal(adapter.ground_path_xy, np.stack((self.car, target)))
        np.testing.assert_array_equal(adapter.ground_guidance, guide)

        for command in (None, self.car - [20., 0.]):
            target, reason = adapter.plan_ground(command)
            self.assertEqual(reason, 'uncertified_vehicle_clearance')
            np.testing.assert_array_equal(target, self.car)
        obs['occupancy_map'][:] = -1
        target, reason = adapter.plan_ground(guide)
        self.assertEqual(reason, 'uncertified_vehicle_clearance')
        np.testing.assert_array_equal(target, self.car)

    def test_guard_executes_submetre_connector_if_the_native_waypoint_is_already_reached(self):
        # Native sensor at 490s of the 1228722198 closed-loop replay, translated
        # by (-1295, -1017) m. Only short southward departures passed sensing.
        guard, obs = self.attach_frontier_guard()
        self.car[:] = [10.0114530076448, 10.1389435681115]
        rows = ['................', '.............###'] + ['............####'] * 2 + [
            '##..........####', '###........#####'] + ['####........####'] * 3 + [
            '#####........###'] + ['####............'] * 3 + [
            '###.............', '##..............', '................']
        obs['occupancy_map'][:] = np.array([[c == '#' for c in row] for row in rows])
        coarse = ['.......###.', '.....######', '##.....####', '###...#####', '####..#####',
                  '####...####', '####...####', '####.....##', '###........', '##.........',
                  '...........']
        self.adapter.cells[5:16, 5:16] = np.array([[1 if c == '#' else -1 for c in row]
                                                   for row in coarse])
        guide = self.car + [-155.13, -123.29]
        self.adapter.ground_guidance = guide.copy()
        guard._update_ground_memory(obs['occupancy_map'], self.car, .5)
        guard.follow_reason = 'observed_frontier'
        guard.follow_origin = np.array([48.4573, 47.8314])
        guard.bypass_anchor = np.array([-106.3709, -78.5719])
        guard.target = guard.observed_grid.frontier = np.array([9.375, 8.875])
        target, reason = self.adapter.plan_ground(guide)
        self.assertEqual(reason, START_CONNECTOR)
        self.assertGreater(np.linalg.norm(target-self.car), 0.)
        self.assertLess(np.linalg.norm(target-self.car), 1.)
        np.testing.assert_allclose(target[0], self.car[0], atol=1.e-9)
        self.assertLess(target[1], self.car[1])
        obs['self']['orientation'] = -np.pi/2
        speed, _ = guard.motion((0., 0.))
        self.assertGreater(speed, 0.)
