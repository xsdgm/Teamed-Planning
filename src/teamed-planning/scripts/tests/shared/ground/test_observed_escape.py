"""Observed-pixel escape safety and the real r1 raster pocket."""
import contextlib
import io
import json
from pathlib import Path
import unittest
import numpy as np

from shared.ground.ground_main import GroundRobotController
from shared.ground.observed_grid import escape_waypoint, segment_clearance
from shared.common.graph_structure import TopologicalGraph
from shared.simulator.env import EnvConfig, MultiRobotEnv
from shared.simulator.wildscenes_legacy_scene import LegacyObstacleScene, PROTOCOL, install_legacy_environment


class ObservedEscapeTests(unittest.TestCase):
    def test_swept_distance_checks_the_middle_of_the_segment(self):
        lo, hi = np.array([[0., 0.]]), np.array([[1., 1.]])
        self.assertEqual(segment_clearance(np.array([-1., .5]), np.array([2., .5]), lo, hi), 0.)
        self.assertAlmostEqual(segment_clearance(np.array([-1., 1.5]), np.array([2., 1.5]), lo, hi), .5)

    def test_unknown_space_and_a_physically_enclosed_start_are_not_cleared(self):
        self.assertIsNone(escape_waypoint((0., 0.), (10., 0.), np.zeros((80, 80)), .1))
        semantic = np.ones((80, 80), np.uint8)
        semantic[:, 42] = 2
        self.assertIsNone(escape_waypoint((0., 0.), (10., 0.), semantic, .1))

    def test_escape_never_crosses_the_unknown_sensor_boundary(self):
        semantic = np.ones((8, 8), np.uint8)
        point = escape_waypoint((0., 0.), (10., 0.), semantic, .5)
        self.assertIsNotNone(point)
        self.assertTrue(np.all(np.abs(point)+.5 < 2.))

    def test_real_r1_stop_reaches_goal_with_only_local_sensing(self):
        self._replay_real_stop()

    def test_real_r1_stop_recovers_with_a_stale_aerial_entry(self):
        self._replay_real_stop(stale_entry=True)

    def test_real_r1_slow_progress_recovers_before_entering_the_raster_peak(self):
        self._replay_real_stop(stale_entry=True,slow_progress=True)

    def _replay_real_stop(self, stale_entry=False,slow_progress=False):
        fixture = json.loads((Path(__file__).parent/'fixtures/wildscenes_r1_raster_stop.json').read_text())
        scene = LegacyObstacleScene(dict(protocol=PROTOCOL, simplified_scene=True,
            unknown_space='free_by_simulation_assumption', origin_xy=[0., 0.],
            world_size_m=fixture['world_size_m'], geometry=dict(type='axis_aligned_rectangles',
                                                             rectangles=fixture['rectangles'])))
        with contextlib.redirect_stdout(io.StringIO()):
            env = MultiRobotEnv(EnvConfig(world_size=(scene.world_size_m,)*2, dt=1.,
                ground_occupancy_map_size=16, ground_occupancy_map_resolution=.5))
            self.addCleanup(env.close)
            install_legacy_environment(env, scene)
            start=(1169.0406932096266,326.74458108024) if stale_entry else fixture['start']
            heading=-1.8971435000315835 if stale_entry else 0.
            if slow_progress:
                start=(1167.9622943443421,326.17103407930813)
                heading=-1.2490457723984818
            rid = env.add_ground_robot(*start,theta=heading)
            controller = GroundRobotController('test', safety_distance=1., subgoal_radius=200.,
                world_bounds=(0., 0., scene.world_size_m, scene.world_size_m))
            controller.set_goal(tuple(fixture['goal']))
            if stale_entry:
                # The received route approaches the pocket from its blocked
                # eastern side, as in the full r1 trace. Its presence must
                # not repeatedly override an observed recovery to the west.
                graph=TopologicalGraph('archived')
                graph.add_node('entry',(1188.3836181553895,328.0557268490988))
                graph.add_node('goal',tuple(fixture['goal']))
                graph.add_edge('entry','goal',attributes={'confidence':1.})
                controller.unified_graph=graph
            if slow_progress:
                controller._drive_to(tuple(fixture['goal']))
                controller._local_route_goal=tuple(fixture['goal'])
                controller._local_route_best=35.388921934203275
                controller._local_route_progress_time=0.
                controller._local_route_progress_position=np.array([1163.4136456317894,321.7979397394162])
                controller._local_progress_time=20.
                controller._local_progress_position=np.array([1167.8212345038014,326.07413611897874])
                controller._allow_local_escape=True
                controller._local_escape_origin=np.array(start)
            obs = env.reset()[rid]
            bounds = fixture['sensor_bounds']
            recovered = False
            for step in range(400):
                previous = obs['self']['position'][:2].copy()
                self.assertTrue(bounds[0] <= previous[0]-4 <= previous[0]+4 <= bounds[2])
                self.assertTrue(bounds[1] <= previous[1]-4 <= previous[1]+4 <= bounds[3])
                obs.pop('obstacles', None)
                obs['full_state'] = {'time': float(step+(31 if slow_progress else 0))}
                action = controller.get_action(obs)
                if slow_progress and step==0:
                    self.assertTrue(controller._route_recovery)
                recovered |= controller._allow_local_escape
                observations, info = env.step({rid: action})
                obs = observations[rid]
                position = obs['self']['position'][:2]
                self.assertFalse(info['collisions'])
                self.assertTrue(scene.segment_clear(previous, position, .3))
                remaining = np.linalg.norm(position-fixture['goal'])
                if remaining <= 2.:
                    break
            self.assertTrue(recovered)
            self.assertLessEqual(remaining, 2.)
