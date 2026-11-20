from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace as Box
import unittest
from unittest.mock import patch

import numpy as np

from ground.vtemp4_observation import ring_points, valid_ring_points
from simulator.vtemp4_waypoint_runtime import (
    RING_ANGLES, RingWaypointFleetEpisode, WaypointFleetEpisode, motion_history, waypoint_options,
)


def endpoints(position, extent=200.):
    angles = np.arange(24) * 2*np.pi/24
    return position + extent * np.column_stack((np.cos(angles), np.sin(angles)))


class WaypointRuntimeTests(unittest.TestCase):
    def ring_episode(self):
        ep = RingWaypointFleetEpisode.__new__(RingWaypointFleetEpisode)
        vehicle = np.array([320., 320.])
        ep.position = lambda: vehicle.copy()
        ep.goal = np.array([1100., 320.])
        ep.env = Box(env=Box(time=0))
        ep.case = {'world_size': 1200., 'direct_distance_m': 780.}
        ep.adapter = Box(own={'orientation': np.pi / 2}, ep=ep, resolution=1.,
                         cells=np.zeros((1200, 1200), np.int8), observe_receipts=lambda: None,
                         _plan=lambda *args: (None, None, None, None, None, 'directional_route', None),
                         last_ground_reason='directional_route')
        ep._native_filter = lambda *args: None
        ep.samples, ep.route_history = deque(maxlen=61), deque()
        ep.ground_events, ep.requests, ep.waypoint_reviews, ep.rollout = [], [], [], []
        ep.pending = ep.current_request = ep.last_review = ep._last_control_time = None
        ep.best_distance = ep.best_route = None
        ep.final_approach = ep.waiting_for_evidence = False
        ep.heading_return, ep.last_progress_s = 0., 0
        ep.limit = ep.budget = 400
        ep.ground_interval, ep.review_interval, ep.lookahead_m = 5, 15, 200.
        ep.generator, ep.torch = None, Box(inference_mode=nullcontext)
        def act(obs, **kwargs):
            replace = not obs['can_hold']
            return replace, 0 if replace else -1, 0., 0.
        ep.slow = Box(act=act)
        ep._ground_decision = lambda now, **kwargs: ep.ground_events.append(dict(time_s=now, **kwargs))
        return ep, vehicle

    def test_full_ring_uses_issue_yaw_and_omits_blocked_and_outside_endpoints(self):
        ep, car = self.ring_episode()
        car[:] = [70., 470.]
        ep.adapter.cells[670, 70] = 1
        points = ring_points(car, np.pi / 2, RING_ANGLES, 200.)
        valid = valid_ring_points(ep.adapter, points)
        np.testing.assert_allclose(np.linalg.norm(points-car, axis=1), 200.)
        np.testing.assert_allclose(points[[0,12,24,36]], [[70.,670.],[-130.,470.],[70.,270.],[270.,470.]])
        self.assertEqual(len(np.unique(np.round(points, 8), axis=0)), 48)
        self.assertFalse(valid[0])
        self.assertFalse(valid[12])
        self.assertTrue(valid[24] and valid[36])  # Unmeasured endpoints may guide the survey.

    @patch('simulator.vtemp4_waypoint_runtime.build_observation', side_effect=lambda *args, **kwargs: {})
    def test_hold_survives_rotation_and_ground_replans_until_explicit_issue(self, observation):
        ep, car = self.ring_episode()
        ep._after_fusion(None, None)
        first = ep.current_request
        np.testing.assert_allclose(first['target'], [320., 520.])
        self.assertEqual(first['origin_xy'], [320.,320.])
        self.assertEqual(first['vehicle_yaw_rad'], np.pi / 2)
        self.assertEqual(first['relative_angle_rad'], 0.)
        self.assertEqual(first['radius_m'], 200.)
        car[:] += [3., 5.]
        ep.adapter.own['orientation'] = np.pi
        for t in (5, 10, 15, 20, 25):
            ep.env.env.time = t
            ep._after_fusion(None, None)
            self.assertIs(ep.current_request, first)
            self.assertEqual(ep.ground_events[-1]['shared_marker'], [320., 520.])
        ep.adapter.cells[520,320] = 1  # delivered obstacle makes a fresh choice eligible
        ep.env.env.time = 30
        ep._after_fusion(None, None)
        second = ep.current_request
        self.assertEqual(second['id'], first['id']+1)
        np.testing.assert_allclose(second['target'], [123.,325.])
        self.assertEqual(second['origin_xy'], car.tolist())
        self.assertEqual(second['vehicle_yaw_rad'], np.pi)
        self.assertEqual(observation.call_args.args[1], np.pi)
        self.assertFalse(observation.call_args.kwargs['limit_by_goal'])
        ep.goal = car + [20., 0.]
        ep.env.env.time = 35
        ep._after_fusion(None, None)
        self.assertIs(ep.current_request, second)
        self.assertTrue(ep.ground_events[-1]['final_approach'])
        self.assertIsNone(ep.ground_events[-1]['guidance'])  # native final-goal local planning

    @patch('simulator.vtemp4_waypoint_runtime.build_observation', side_effect=lambda *args, **kwargs: {})
    def test_empty_ring_does_not_publish_car_position_or_replace_blocked_marker(self, observation):
        ep, car = self.ring_episode()
        ep.adapter.cells.fill(1)
        ep._after_fusion(None, None)
        self.assertIsNone(ep.current_request)
        self.assertTrue(ep.ground_events[-1]['wait_for_guidance'])
        self.assertIsNone(ep.waypoint_reviews[-1]['waypoint_id'])
        ep.adapter.cells.fill(0)
        ep.env.env.time = 15
        ep._after_fusion(None, None)
        first = ep.current_request
        self.assertEqual(first['id'], 0)
        ep.adapter.cells.fill(1)
        car[:] += [5., 5.]
        ep.env.env.time = 30
        ep._after_fusion(None, None)
        self.assertIs(ep.current_request, first)
        self.assertTrue(ep.ground_events[-1]['wait_for_guidance'])
        self.assertFalse(ep.waypoint_reviews[-1]['replace'])
        self.assertEqual(first['target'], [320.,520.])
        self.assertEqual(len(ep.requests), 1)

    def test_history_missing_ages_and_waypoint_approach_during_goal_retreat(self):
        goal, waypoint = np.array([100., 0.]), np.array([0., 80.])
        samples = [(t, np.array([0., t/2]), t/2) for t in range(31)]
        history, mask = motion_history(samples, 30, samples[-1][1], goal, waypoint, 100.)
        np.testing.assert_array_equal(mask, [False, False, True, True])
        self.assertLess(history[2, 2], 0.)  # moving away from final goal
        self.assertAlmostEqual(history[2, 3], .15)  # closing on the fixed waypoint
        self.assertAlmostEqual(history[2, 4], .15)
        np.testing.assert_array_equal(history[~mask], np.zeros((2,5)))
        recent, _ = motion_history(samples, 30, samples[-1][1], goal, waypoint, 100., issued_at=10)
        self.assertEqual(recent[2,3], 0.)
        self.assertAlmostEqual(recent[3,3], .075)

    def test_mask_excludes_observed_block_and_world_edges(self):
        car = np.array([80.5, 80.5])
        cells = np.full((180, 180), -1, np.int8)
        cells[:, 100:105] = 1  # all territory beyond is disconnected in received map
        adapter = Box(cells=cells, resolution=1.)
        points, valid = waypoint_options(adapter, car, endpoints(car), 200.)
        self.assertTrue(valid.any())
        self.assertTrue(np.all(points[valid, 0] < 99.))
        self.assertTrue(np.all((points[valid] >= 0.) & (points[valid] < 180.)))
        xy = points[valid].astype(int)
        self.assertTrue(np.all(cells[xy[:, 1], xy[:, 0]] != 1))

    def test_frozen_point_persists_when_car_approaches_it_while_goal_distance_grows(self):
        ep = WaypointFleetEpisode.__new__(WaypointFleetEpisode)
        vehicle = np.array([220.5, 220.5])
        ep.position = lambda: vehicle.copy()
        ep.env = Box(env=Box(time=0))
        ep.goal = np.array([580.5, 220.5])
        ep.case = {'direct_distance_m': 360.}
        ep.adapter = Box(cells=np.full((660, 660), -1, np.int8), resolution=1.,
                         observe_receipts=lambda: None, last_ground_reason='known_goal_route',
                         _plan=lambda point: (np.zeros(2),None,np.ones((40,2)),None,None,'known_goal_route',None))
        ep._native_filter = lambda graph, terrain: graph
        ep.limit = ep.budget = 400
        ep.ground_interval, ep.review_interval, ep.lookahead_m = 5, 15, 200.
        ep.samples, ep.waypoint_reviews, ep.requests, ep.rollout, ep.ground_events = deque(maxlen=61), [], [], [], []
        ep.route_history = deque()
        ep.pending = ep.current_request = ep.last_review = ep._last_control_time = None
        ep.best_distance = ep.best_route = None
        ep.final_approach = ep.waiting_for_evidence = False
        ep.heading_return, ep.last_progress_s = 0., 0
        ep.generator = None
        ep.torch = Box(inference_mode=nullcontext)
        actions = []
        def act(obs, **kwargs):
            action = (False, -1) if obs['can_hold'] else (True, 12)  # north, across goal axis
            actions.append(action)
            return (*action, 0., 0.)
        ep.slow = Box(act=act)
        ep._ground_decision = lambda *args, **kwargs: ep.ground_events.append({})
        with patch('simulator.vtemp4_waypoint_runtime.build_observation',
                   side_effect=lambda *args, **kwargs: {'targets': endpoints(vehicle)}):
            ep._after_fusion(None, None)
            request = ep.current_request
            target = request['target'].copy()
            for t in range(1, 31):
                vehicle[1] += .25
                ep.env.env.time = t
                ep._after_fusion(None, None)
        self.assertEqual(actions, [(True, 12), (False,-1), (False,-1)])
        self.assertIs(ep.current_request, request)
        self.assertEqual(ep.current_request['target'], target)
        self.assertEqual(len(ep.requests), 1)
        self.assertTrue(all(row['shared_marker'] == target for row in ep.ground_events))
        self.assertGreater(ep.waypoint_reviews[-1]['remaining_m'], ep.waypoint_reviews[0]['remaining_m'])
        self.assertLess(ep.waypoint_reviews[-1]['distance_to_waypoint_m'], ep.waypoint_reviews[0]['distance_to_waypoint_m'])
        self.assertEqual(len(ep.rollout), 2)
        ep._settle(30,ep.waypoint_reviews[-1]['remaining_m'],0.,'safety_failure')
        self.assertTrue(ep.rollout[-1]['terminated'])

    def test_reached_blocked_and_sustained_stall(self):
        ep = WaypointFleetEpisode.__new__(WaypointFleetEpisode)
        target, car = np.array([90., 50.]), np.array([50., 50.])
        ep.adapter = Box(cells=np.zeros((140, 140), np.int8), resolution=1.,
                         _plan=lambda point: (None,None,[0]*20,None,None,'directional_route',None))
        ep.best_distance, ep.best_route, ep.last_progress_s, ep.waiting_for_evidence = 40., None, 0, False
        ep.samples, ep.route_history, ep.current_request, ep.review_interval = deque(), deque(), {'issued_at_s': 0}, 15
        self.assertEqual(ep._review_reason(45,car,target),'tracking')
        self.assertEqual(ep._review_reason(60,car,target),'stalled')
        ep.adapter._plan = lambda point: (np.zeros(2),None,np.ones((30,2)),None,None,'known_goal_route',None)
        self.assertEqual(ep._review_reason(61,car,target),'tracking')
        self.assertEqual(ep._review_reason(119,car,target),'tracking')
        ep.adapter.cells[50,90] = 1
        self.assertEqual(ep._review_reason(120,car,target),'blocked')
        self.assertEqual(ep._review_reason(120,target,target),'reached')

    def test_closing_since_retreat_keeps_point_above_old_best_distance(self):
        ep = WaypointFleetEpisode.__new__(WaypointFleetEpisode)
        active = np.array([95.,50.])
        ep.samples = deque(((t,np.array([25.+(t-60)*.2,50.]),(t-60)*.2) for t in range(60,121)), maxlen=61)
        ep.current_request, ep.route_history = {'issued_at_s':20}, deque()
        ep.best_distance, ep.best_route, ep.last_progress_s, ep.waiting_for_evidence = 40., None, 50, False
        ep.adapter = Box(cells=np.zeros((150,150),np.int8),resolution=1.,
                         _plan=lambda target: (None,None,None,None,None,'directional_route',None))
        self.assertEqual(ep._review_reason(120,ep.samples[-1][1],active),'tracking')
        self.assertEqual(ep.last_progress_s, 120)
        # Brief retreat, then subpixel drift. Earlier closing windows must not
        # buy extra time at later reviews, or expire the waypoint at t=135.
        for t in range(121,180):
            x = 37. - min(3, t-120)*.25 + max(t-123,0)*.001
            ep.samples.append((t,np.array([x,50.]),0.))
            if t in (135,150,165,179):
                self.assertEqual(ep._review_reason(t,ep.samples[-1][1],active),'tracking')
                self.assertEqual(ep.last_progress_s, 120)
        ep.samples.append((180,ep.samples[-1][1],0.))
        self.assertEqual(ep._review_reason(180,ep.samples[-1][1],active),'stalled')

    def test_route_recovery_renews_clock_only_for_new_decreases(self):
        ep = WaypointFleetEpisode.__new__(WaypointFleetEpisode)
        active, car = np.array([90.5,50.5]), np.array([50.5,50.5])
        ep.samples = deque()
        ep.current_request, ep.route_history, ep.review_interval = {'issued_at_s':0}, deque(), 15
        ep.best_distance, ep.best_route, ep.last_progress_s, ep.waiting_for_evidence = 21., 40., 35, False
        points = np.repeat([[50,50]], 91, axis=0)
        points[-1] = [50,90]
        ep.adapter = Box(cells=np.zeros((145,145),np.int8),resolution=1.,
                         _plan=lambda target: (np.zeros(2),None,points,None,None,'known_goal_route',None))
        self.assertEqual(ep._review_reason(90,car,active), 'tracking')
        points = points[4:]  # 4 m recovered, still far above the historical best
        self.assertEqual(ep._review_reason(105,car,active), 'tracking')
        self.assertEqual(ep.last_progress_s, 105)
        for t in (120,135,150):
            self.assertEqual(ep._review_reason(t,car,active), 'tracking')
            self.assertEqual(ep.last_progress_s, 105)
        self.assertEqual(ep._review_reason(165,car,active), 'stalled')


if __name__ == '__main__':
    unittest.main()
