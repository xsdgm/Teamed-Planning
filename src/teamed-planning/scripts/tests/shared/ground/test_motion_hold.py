"""A deliberate survey hold must not activate obstacle recovery on departure."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.ground.ground_main import GroundRobotController
from shared.common.graph_structure import TopologicalGraph


class MotionHoldTests(unittest.TestCase):
    def setUp(self):
        output=contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__,None,None,None)
        self.controller=GroundRobotController('ground_0',fusion_interval=5.)
        self.controller.set_goal((150.,50.))

    def action(self, time, held=None, neighbors=None):
        observation=dict(self=dict(position=[50.,50.,0.],orientation=0.),
            occupancy_map=np.zeros((16,16),np.uint8),map_resolution=.5,time_step=1.,
            full_state=dict(time=float(time)),neighbors=neighbors or [])
        if held is not None:
            observation['motion_held']=held
        return self.controller.get_action(observation)

    def test_hold_preserves_mission_and_planning_without_false_recovery(self):
        graph=TopologicalGraph('drone_0')
        graph.add_node('a',(51.,50.),dict(terrain_type='TRAVERSABLE',observation_count=20))
        graph.add_node('b',(55.,50.),dict(terrain_type='TRAVERSABLE',observation_count=20))
        graph.add_edge('a','b')
        peers=[dict(type='drone',broadcast=dict(drone_id='drone_0',position=[51.,50.,15.],
                                             graph=graph.to_message()))]
        for t in range(40):
            action=self.action(t,True,peers)
            self.assertEqual(action['motion'],(0.,0.))
            self.assertEqual(action['broadcast']['target'],[150.,50.])
            self.assertFalse(self.controller._allow_local_escape)
            self.assertFalse(self.controller._route_recovery)
            self.assertFalse(self.controller._route_loop)
        self.assertIsNotNone(self.controller.unified_graph)
        self.assertGreater(len(self.controller.unified_graph),0)
        self.assertIsNotNone(self.controller.global_planner.current_path)
        self.assertFalse(self.controller._failed_local_connections)
        self.action(40,False)
        self.assertFalse(self.controller._allow_local_escape)
        self.assertFalse(self.controller._route_recovery)

    def test_real_failure_to_move_after_release_still_triggers_recovery(self):
        for t in range(40):
            self.action(t,True)
        for t in range(40,72):
            self.action(t,False)
        self.assertTrue(self.controller._route_recovery)

    def test_absent_hold_flag_retains_existing_stall_detection(self):
        for t in range(32):
            self.action(t)
        self.assertTrue(self.controller._route_recovery)


if __name__=='__main__':
    unittest.main()
