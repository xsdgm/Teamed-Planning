import unittest
from types import SimpleNamespace

import numpy as np

from survey.survey_fleet_clearance import native_segment_clear
from survey.survey_fleet_drive import GroundMotionGuard, REPEATED_GRAPH_WAYPOINT
from tests.simulator.test_survey_fleet_drive import LocalVehicle


class GraphHandoffTest(unittest.TestCase):
    def setUp(self):
        self.car = np.array([10., 10.])
        self.tile = np.zeros((16, 16), dtype=np.int8)
        self.planner = SimpleNamespace(waiting_at_frontier=True, using_graph=True,
            get_current_waypoint=lambda: (23., 16.))
        obs = dict(occupancy_map=self.tile, time_step=1., map_resolution=.5,
                   self=dict(position=self.car, orientation=0.))
        exp = SimpleNamespace(ground=0, goal_reached=False, obs={0: obs},
            ground_controller=SimpleNamespace(local_planner=LocalVehicle(), global_planner=self.planner,
                                              waypoint_threshold=1.))
        adapter = SimpleNamespace(now=0., exp=exp, ground_path_xy=self.car + [[0., 0.], [.5, 0.]],
                                  ep=SimpleNamespace(holding=False, goal=np.array([70.,30.])))
        self.guard = GroundMotionGuard(adapter)

    def wait(self):
        for t in range(3):
            self.guard.adapter.now = t
            motion = self.guard.motion((0., 0.))
        return motion

    def replay_graph_pursuit(self, *, switch_node=True, turn_command=False, block_route=False, forward_route=True):
        adapter = self.guard.adapter
        self.planner.waiting_at_frontier = False
        adapter.last_ground_reason = 'native_shared_guidance'
        far_node = self.car + [50., 0.]
        self.planner.get_current_waypoint = lambda: self.node
        for i, x in enumerate((10., 15., 20., 25., 20., 15., 10.)):
            self.car[0] = x
            self.node = self.car + [-8., 0.] if switch_node and i in (3, 4, 5) else far_node
            adapter.now = 5. * i
            adapter.ground_guidance = self.car + ([-200., 0.] if i != 4 or not turn_command else [0., 200.])
            steps = -np.arange(16) if forward_route else np.arange(16)
            adapter.ground_path_xy = self.car + np.column_stack((steps, np.zeros(16)))
            adapter.exp.obs[0]['self']['orientation'] = np.pi
            if block_route and i == 6:
                self.tile[7:10, 3] = -1
            action = self.guard.motion((.4, -.1))
            if i < 6:
                self.assertNotEqual(self.guard.follow_reason, REPEATED_GRAPH_WAYPOINT)
        return action

    def test_reselected_backward_node_after_an_excursion_follows_observed_route(self):
        speed, turn = self.replay_graph_pursuit()
        self.assertEqual(self.guard.follow_reason, REPEATED_GRAPH_WAYPOINT)
        self.assertGreater(speed, 0.)
        self.assertEqual(turn, 0.)
        self.assertLess(self.guard.target[0], self.car[0])
        self.assertTrue(native_segment_clear(self.tile, self.car, .5, self.car, self.guard.target))
        self.assertEqual(self.guard.events[-1]['graph_waypoint'], [60., 10.])
        self.guard.adapter.ground_guidance = self.car + [0., 200.]
        self.guard.adapter.now += 1
        self.guard.motion((0., 0.))
        self.assertIsNone(self.guard.follow_reason)
        self.assertIsNone(self.guard.target)

    def test_one_continuous_obstacle_detour_does_not_trigger_a_repeated_node(self):
        self.assertEqual(self.replay_graph_pursuit(switch_node=False), (.4, -.1))
        self.assertIsNone(self.guard.follow_reason)

    def test_actual_command_turn_invalidates_previous_graph_pursuit(self):
        self.assertEqual(self.replay_graph_pursuit(turn_command=True), (.4, -.1))
        self.assertIsNone(self.guard.follow_reason)

    def test_conflicting_fresh_sensor_prevents_a_return_to_the_coarse_route(self):
        self.replay_graph_pursuit(block_route=True)
        self.assertEqual(self.guard.invalid_start, 0)
        self.assertNotEqual(self.guard.follow_reason, REPEATED_GRAPH_WAYPOINT)
        self.assertFalse(any(e['reason'] == REPEATED_GRAPH_WAYPOINT for e in self.guard.events))

    def test_backward_grid_route_cannot_take_control_from_a_repeated_node(self):
        self.assertEqual(self.replay_graph_pursuit(forward_route=False), (.4, -.1))
        self.assertIsNone(self.guard.follow_reason)

    def move_in_commanded_direction(self, meters, elapsed):
        self.car[0] -= meters
        adapter = self.guard.adapter
        adapter.now += elapsed
        # Intervening 30-second command reviews can move the goal but keep the heading.
        adapter.ground_guidance = self.car + [-200., 0.]
        adapter.ground_path_xy = self.car + np.column_stack((-np.arange(16), np.zeros(16)))
        return self.guard.motion((.4, -.1))

    def test_known_failed_node_remains_rejected_beyond_prior_distance_and_time(self):
        self.replay_graph_pursuit()
        for meters, elapsed in [(21., 79), (9., 42)]:
            speed, turn = self.move_in_commanded_direction(meters, elapsed)
            self.assertGreater(speed, 0.)
            self.assertEqual(turn, 0.)
            self.assertEqual(self.guard.follow_reason, REPEATED_GRAPH_WAYPOINT)
            self.assertLess(self.guard.target[0], self.car[0])
        self.node = self.car + [-15., 0.]
        self.assertEqual(self.move_in_commanded_direction(0., 1), (.4, -.1))
        self.assertIsNone(self.guard.follow_reason)
        self.assertIsNone(self.guard.target)

    def test_unproductive_short_hops_release_even_while_node_is_still_requested(self):
        self.replay_graph_pursuit()
        self.assertEqual(self.move_in_commanded_direction(2., 120), (.4, -.1))
        self.assertIsNone(self.guard.follow_reason)

    def test_direct_confirmation_or_missing_native_evidence_releases_at_once(self):
        for attr in ("waiting_at_frontier", "using_graph"):
            with self.subTest(attr=attr):
                self.setUp()
                self.replay_graph_pursuit()
                setattr(self.planner, attr, attr == "waiting_at_frontier")
                self.move_in_commanded_direction(1., 1)
                self.assertIsNone(self.guard.follow_reason)

    def test_original_command_range_limits_intervention_with_rolling_goal(self):
        self.replay_graph_pursuit()
        self.assertEqual(self.move_in_commanded_direction(201., 95), (.4, -.1))
        self.assertIsNone(self.guard.follow_reason)

    def test_traces_only_a_locally_certified_part_of_rejected_graph_edge(self):
        self.wait()
        point = self.guard.target
        self.assertIsNotNone(point)
        self.assertGreater(np.linalg.norm(point-self.car), 1.)
        self.assertTrue(native_segment_clear(self.tile, self.car, .5, self.car, point))
        self.assertTrue(self.planner.waiting_at_frontier)
        self.assertFalse(native_segment_clear(self.tile, self.car, .5, self.car,
                                             self.planner.get_current_waypoint()))
        # Stop immediately if the next tile contradicts the persisted local leg.
        self.tile[:] = -1
        self.assertEqual(self.guard.motion((0., 0.)), (0., 0.))
        self.assertIsNone(self.guard.target)

    def test_observed_start_without_room_for_any_leg_keeps_waiting(self):
        self.tile[:] = -1
        self.tile[4:12, 4:12] = 0
        self.assertTrue(native_segment_clear(self.tile, self.car, .5, self.car, self.car))
        self.assertEqual(self.wait(), (0., 0.))
        self.assertIsNone(self.guard.target)

    def test_can_choose_observed_side_step_when_graph_bearing_is_blocked(self):
        self.tile[7:10, 12] = 1
        self.wait()
        self.assertIsNotNone(self.guard.target)
        self.assertGreater(np.linalg.norm(self.guard.target-self.car), .8)
        self.assertTrue(native_segment_clear(self.tile, self.car, .5, self.car,
                                             self.guard.target))
        self.assertFalse(native_segment_clear(self.tile, self.car, .5, self.car,
                                             self.car + [2., 1.]))

    def test_motion_veto_with_single_coarse_cell_can_take_safe_alternative(self):
        self.guard.adapter.ground_path_xy = self.car[None, :]
        self.planner.waiting_at_frontier = False
        self.tile[7:10, 12] = 1
        self.guard.motion((1., 0.))
        self.assertEqual(self.guard.vetoes, 1)
        self.assertIsNotNone(self.guard.target)
        self.assertTrue(native_segment_clear(self.tile, self.car, .5, self.car, self.guard.target))

    def measure_loop(self, start):
        center = self.car.copy()
        direction = self.guard.adapter.ground_guidance - center
        across = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
        for tick in range(80):
            self.guard.adapter.now = float(start + tick)
            prior = center + across * (7. * np.sin(np.pi * tick / 20.))
            self.guard._record_motion(prior, False, .6)
        self.guard.adapter.now = float(start + 80)
        self.guard.motion((.6, 0.))

    def first_loop(self):
        self.guard.adapter.ground_guidance = self.car + [200., 0.]
        self.planner.waiting_at_frontier = False
        self.measure_loop(0)
        self.assertEqual(self.guard.follow_reason, 'route_loop')
        self.assertIsNone(self.guard.observed_grid.frontier)

    def second_loop(self, offset=(0., 0.), direction=(200., 0.)):
        origin = self.car.copy()
        offset = np.asarray(offset)
        for tick in range(81, 200):
            self.guard.adapter.now = float(tick)
            self.guard._record_motion(origin + offset * ((tick - 80.) / 120.), False, .6)
        self.car[:] = origin + offset
        self.guard.adapter.now = 200.
        self.guard.motion((.6, 0.))
        self.assertIsNone(self.guard.follow_reason)
        self.guard.adapter.ground_guidance = self.car + direction
        for tick in range(201, 205):
            self.guard.adapter.now = float(tick)
            self.guard._record_motion(self.car, False, .6)
        self.measure_loop(205)

    def activate_observed_frontier(self):
        self.first_loop()
        self.second_loop()
        self.assertEqual(self.guard.follow_reason, 'observed_frontier')
        np.testing.assert_array_equal(self.guard.bypass_anchor, self.guard.adapter.ground_guidance)
        np.testing.assert_array_equal(self.guard.observed_grid.goal, self.guard.adapter.ground_guidance)

    def test_first_loop_leaves_the_observed_frontier_for_recurrent_failure(self):
        self.first_loop()
        self.assertIsNone(self.guard.bypass_anchor)

    def test_progress_during_the_first_detour_keeps_the_second_under_native_control(self):
        self.first_loop()
        self.second_loop(offset=(12., 0.))
        self.assertEqual(self.guard.follow_reason, 'route_loop')
        self.assertIsNone(self.guard.bypass_anchor)

    def test_new_command_direction_starts_a_separate_loop_history(self):
        self.first_loop()
        self.second_loop(direction=(0., 200.))
        self.assertEqual(self.guard.follow_reason, 'route_loop')
        self.assertIsNone(self.guard.bypass_anchor)

    def test_memory_persists_past_short_hops_until_the_shared_command_turns(self):
        self.activate_observed_frontier()
        self.car += [23., 0.]
        self.guard.adapter.now += 150.
        self.guard.adapter.ground_guidance = self.car + [200., 0.]
        self.guard.motion((.6, 0.))
        self.assertEqual(self.guard.follow_reason, 'observed_frontier')
        self.guard.adapter.ground_guidance = self.car + [0., 200.]
        self.guard.motion((.6, 0.))
        self.assertIsNone(self.guard.follow_reason)
        self.assertIsNone(self.guard.observed_grid.frontier)

    def test_coarse_hold_can_continue_only_on_a_fresh_certified_active_leg(self):
        guide = self.car + [200., 0.]
        self.assertIsNone(self.guard.observed_frontier_segment(guide))
        self.activate_observed_frontier()
        self.assertIsNone(self.guard.observed_frontier_segment(None))
        self.assertIsNone(self.guard.observed_frontier_segment(self.car + [0., 200.]))
        segment = self.guard.observed_frontier_segment(guide)
        self.assertIsNotNone(segment)
        self.assertGreater(np.linalg.norm(segment[1]-self.car), 1.)
        self.assertTrue(native_segment_clear(self.tile, self.car, .5, *segment))
        self.tile[8,8] = -1
        self.assertIsNone(self.guard.observed_frontier_segment(guide))

    def test_coarse_hold_retries_a_real_loop_but_releases_on_command_change(self):
        self.activate_observed_frontier()
        self.guard.adapter.ep.holding = True
        self.guard.adapter.last_ground_reason = 'uncertified_vehicle_clearance'
        self.assertEqual(self.guard.motion((.6, 0.)), (0., 0.))
        self.assertIsNone(self.guard.target)
        self.assertEqual(self.guard.follow_reason, 'observed_frontier')
        self.guard.adapter.ground_guidance = self.car + [0., 200.]
        self.assertEqual(self.guard.motion((.6, 0.)), (0., 0.))
        self.assertIsNone(self.guard.follow_reason)
