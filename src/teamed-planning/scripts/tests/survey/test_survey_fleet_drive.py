from types import SimpleNamespace
import unittest

import numpy as np
from survey.survey_fleet_drive import GroundMotionGuard


class LocalVehicle:
    @staticmethod
    def compute_velocity_command(car, yaw, target, local_map, resolution):
        delta = target-car
        error = np.arctan2(np.sin(np.arctan2(delta[1],delta[0])-yaw),
                           np.cos(np.arctan2(delta[1],delta[0])-yaw))
        return min(1., np.linalg.norm(delta)/3), np.clip(2*error,-1.,1.)


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.car = np.array([10.,10.])
        self.tile = np.zeros((16,16), np.int8)
        self.own = dict(position=self.car, orientation=0.)
        self.obs = dict(occupancy_map=self.tile, time_step=1., map_resolution=.5, self=self.own)
        self.adapter = SimpleNamespace(now=30., ep=SimpleNamespace(holding=False,goal=np.array([1000.,10.])),
            ground_path_xy=self.car+np.array([[0,0],[1,0],[2,0],[3,0]]),
            exp=SimpleNamespace(ground=0,goal_reached=False,obs={0:self.obs},
                                ground_controller=SimpleNamespace(local_planner=LocalVehicle(),
                                    global_planner=SimpleNamespace(waiting_at_frontier=False))))
        self.guard=GroundMotionGuard(self.adapter)

    def test_clear_nominal_command_is_preserved(self):
        self.assertEqual(self.guard.motion((.6, .4)), (.6, .4))
        self.assertIsNone(self.guard.target)

    def test_retained_local_detour_uses_the_same_certified_waypoint_across_ticks(self):
        self.tile[8,12] = 1
        speed, turn = self.guard.motion((.6,0.))
        self.assertEqual(speed, 0.)
        self.assertNotEqual(turn, 0.)
        first=self.guard.target.copy()
        self.adapter.ground_path_xy[-1] = [14,10]
        for _ in range(2):
            self.own['orientation'] += turn
            self.adapter.now += 1.
            speed,turn=self.guard.motion((.6,0.))
            np.testing.assert_array_equal(first,self.guard.target)
        self.assertEqual(self.guard.detours,1)
        self.assertGreater(speed,0.)

    def test_current_disk_contradiction_never_authorizes_translation(self):
        self.tile[8,8] = -1
        self.assertEqual(self.guard.motion((.6,.3)), (0.,.3))
        self.assertEqual(self.guard.invalid_start,1)
        self.assertIsNone(self.guard.target)

    def test_ground_hold_explicitly_cancels_existing_detour(self):
        self.guard.target = self.car + [2.,0.]
        self.adapter.ep.holding = True
        self.assertEqual(self.guard.motion((.6,0.)), (0.,0.))
        self.assertIsNone(self.guard.target)

    def test_continuous_frontier_wait_follows_confirmed_dijkstra_route(self):
        self.adapter.exp.ground_controller.global_planner.waiting_at_frontier = True
        for t in (1.,2.):
            self.adapter.now = t
            self.assertEqual(self.guard.motion((0.,0.)), (0.,0.))
        self.adapter.now = 3.
        speed, turn = self.guard.motion((0.,0.))
        self.assertGreater(speed,0.)
        self.assertEqual(turn,0.)
        self.assertEqual(self.guard.target_reason,'frontier_wait')
        np.testing.assert_array_equal(self.guard.target,self.adapter.ground_path_xy[2])
        # One confirmed local hop stays active between the 5s route refreshes.
        self.adapter.now = 4.
        self.adapter.exp.ground_controller.global_planner.waiting_at_frontier = False
        self.guard.motion((0.,1.))
        self.assertEqual(self.guard.detours,1)

    def test_frontier_wait_with_no_safe_forward_path_stays_still(self):
        self.adapter.exp.ground_controller.global_planner.waiting_at_frontier = True
        self.tile[8,12] = 1
        for t in (1.,2.,3.,4.):
            self.adapter.now = t
            self.assertEqual(self.guard.motion((0.,0.)),(0.,0.))
        self.assertIsNone(self.guard.target)

    def test_revisit_after_travel_and_stalled_goal_progress_triggers_loop_handoff(self):
        for tick in range(81):
            self.adapter.now = float(tick)
            car = self.car + [0, 7*np.sin(np.pi*tick/20.)]
            self.guard._record_motion(car,False,.6)
        self.assertEqual(self.guard._handoff_reason(car),'route_loop')

    def test_motion_progress_and_temporary_detours_keep_native_control(self):
        for tick in range(81):
            self.adapter.now = float(tick)
            car = self.car + [.5*tick, 4*np.sin(np.pi*tick/20.)]
            self.guard._record_motion(car,False,.6)
        self.assertIsNone(self.guard._handoff_reason(car))

if __name__=='__main__':
    unittest.main()
