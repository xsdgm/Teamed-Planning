from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest

import numpy as np

from shared.ground.ground_main import GroundRobotController
from survey.survey_fleet_observation import FleetSurveyAdapter, START_CONNECTOR
from survey.survey_fleet_runtime import SurveyFleetEpisode


class KnownRouteTargetTests(unittest.TestCase):
    def setUp(self):
        self.car = np.array([20.95, 20.5])
        self.goal = np.array([90.5, 20.5])
        controller = GroundRobotController('test-ground', waypoint_threshold=1.)
        sensor = dict(self=dict(position=self.car), occupancy_map=np.zeros((16, 16), np.int8),
                      map_resolution=.5, map_axes='yx')
        exp = SimpleNamespace(ground=0, ground_controller=controller, obs={0:sensor})
        ep = self.ep = SurveyFleetEpisode.__new__(SurveyFleetEpisode)
        ep.env, ep.goal = exp, self.goal
        ep.position = self.car.copy
        ep.manager = SimpleNamespace(selected=0, target=None)
        ep.ground_events = []
        adapter = self.adapter = ep.adapter = FleetSurveyAdapter.__new__(FleetSurveyAdapter)
        adapter.ep, adapter.exp = ep, exp
        adapter.cells = np.full((96, 96), -1, np.int8)
        adapter.bounds = np.array([0, 0]), np.array([95, 95])
        adapter.observe_receipts = lambda: None
        adapter.ground_target = adapter.last_ground_reason = None
        adapter.connector_plans = 0
        self.planner = controller.global_planner
        self.resets = []

    def decide(self, x, goal=None, *, wait=False):
        self.car[0] = x
        with redirect_stdout(StringIO()):
            self.ep._ground_decision(5*len(self.ep.ground_events), self.goal if goal is None else goal,
                                     wait_for_guidance=wait)
        if self.planner.current_path is None:
            self.resets.append(tuple(self.ep.manager.target))
            self.planner.current_path = self.adapter.ground_path_xy.tolist()
            self.planner.current_waypoint_idx = 1
        return tuple(self.ep.manager.target)

    def start_pair(self):
        self.assertEqual(self.decide(20.95), (45.5, 20.5))
        self.assertEqual(self.decide(21.05), (46.5, 20.5))

    def test_adjacent_alternation_keeps_the_active_certified_path_and_native_progress(self):
        self.start_pair()
        active_path = self.planner.current_path
        self.planner.current_waypoint_idx = 3
        for x in (20.95, 21.05, 20.95, 21.05):
            self.assertEqual(self.decide(x), (46.5, 20.5))
            self.assertIs(self.planner.current_path, active_path)
            self.assertEqual(self.planner.current_waypoint_idx, 3)
            np.testing.assert_array_equal(self.adapter.ground_path_xy[-1], self.ep.manager.target)
        self.assertEqual(len(self.resets), 2)

    def test_reached_point_is_replaced(self):
        self.start_pair()
        self.assertEqual(self.decide(46.5), (71.5, 20.5))
        self.assertTrue(self.ep.ground_events[-1]['previous_target_reached'])
        self.assertEqual(len(self.resets), 3)

    def test_new_shared_destination_cannot_reuse_previous_alternation(self):
        self.start_pair()
        self.assertEqual(self.decide(20.95, self.goal + [1., 0.]), (45.5, 20.5))
        self.assertEqual(len(self.resets), 3)

    def test_endpoints_must_still_share_the_new_certified_route(self):
        self.start_pair()
        # A new receiver route shares the horizontal prefix to x=45, then turns
        # down one row. The formerly active cell at x=46 is no longer certified.
        path = ([(20, x) for x in range(20, 46)] + [(21, x) for x in range(45, 91)] + [(20, 90)])
        cells = np.zeros_like(self.adapter.cells)
        cells[tuple(np.array(path).T)] = -1
        self.adapter._plan = lambda guidance: (np.zeros(2, int), cells, np.array(path), None,
                                               np.array([20, 90]), 'known_goal_route', None)
        self.assertEqual(self.decide(20.95), (45.5, 20.5))
        self.assertEqual(len(self.resets), 3)
        self.assertFalse(np.any(np.all(self.adapter.ground_path_xy == [46.5, 20.5], axis=1)))

    def test_hold_and_explicit_wait_override_the_unreached_point(self):
        self.start_pair()
        self.assertEqual(self.decide(20.95, wait=True), (20.95, 20.5))
        self.assertTrue(self.ep.holding)
        self.assertEqual(self.adapter.last_ground_reason, 'waiting_for_guidance')
        self.assertEqual(self.decide(20.95), (45.5, 20.5))
        self.assertFalse(self.ep.holding)
        # An isolated reachable cell is a true planning hold even with guidance.
        cell = np.array([20, 20]); cells = np.zeros_like(self.adapter.cells)
        cells[tuple(cell)] = -1
        self.adapter._plan = lambda guidance: (np.zeros(2,int), cells, cell[None,:], None,
                                               np.array([20, 90]), 'hold', None)
        self.assertEqual(self.decide(20.95), (20.95, 20.5))
        self.assertTrue(self.ep.holding)
        self.assertEqual(self.adapter.last_ground_reason, 'hold')

    def test_certified_connector_takes_precedence_then_releases(self):
        self.start_pair()
        self.adapter._isolated_exit = self.car + [2., 0.]
        connector = tuple(self.adapter._isolated_exit)
        self.assertEqual(self.decide(21.05), connector)
        self.assertEqual(self.adapter.last_ground_reason, START_CONNECTOR)
        self.assertFalse(self.ep.holding)
        self.assertEqual(self.decide(connector[0]), (48.5, 20.5))
        self.assertEqual(self.adapter.last_ground_reason, 'known_goal_route')
        self.assertEqual(len(self.resets), 4)

    def test_wait_starts_a_fresh_sequence_of_known_route_proposals(self):
        self.start_pair()
        self.decide(20.95, wait=True)
        self.assertEqual(self.decide(20.95), (45.5, 20.5))
        self.assertEqual(self.decide(21.05), (46.5, 20.5))
        self.assertEqual(self.decide(20.95), (46.5, 20.5))
