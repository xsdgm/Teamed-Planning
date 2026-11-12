"""Regressions for observed WildScenes deadlocks, without policy checkpoints."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.graph_structure import TopologicalGraph
from ground.global_planner import GlobalPlanner
from ground.ground_main import GroundRobotController
from ground.local_planner import LocalPerception, LocalPlanner
from ground.subgoal_manager import SubgoalManager
from simulator.env import EnvConfig, MultiRobotEnv
from simulator.wildscenes_legacy_scene import LegacyObstacleScene, PROTOCOL


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_archived_stops_have_physically_clear_local_steps(self):
        fixtures = json.loads((Path(__file__).parent/'fixtures/wildscenes_stops.json').read_text())
        for case in fixtures['snapshots']:
            with self.subTest(case=case['case'], step=case['step']):
                env = MultiRobotEnv(EnvConfig(world_size=(case['world_size_m'],)*2,
                    ground_occupancy_map_size=16, ground_occupancy_map_resolution=.5))
                rid = env.add_ground_robot(*case['position'])
                for rect in case['rectangles']:
                    env.add_rect_obstacle(*rect)
                obs = env.reset()[rid]
                if case['coarse_self_occupied']:
                    self.assertEqual(obs['occupancy_map'][8, 8], 1)
                fine = obs['navigation_map']
                self.assertEqual(fine[fine.shape[0]//2, fine.shape[1]//2], 0)
                semantic = LocalPerception('test').process_occupancy_map(fine)
                waypoint = LocalPlanner('test').observed_connection_waypoint(
                    case['position'], case['goal'], semantic, obs['navigation_resolution'],allow_escape=True)
                self.assertIsNotNone(waypoint)
                scene = LegacyObstacleScene(dict(protocol=PROTOCOL, simplified_scene=True,
                    unknown_space='free_by_simulation_assumption', origin_xy=[0., 0.],
                    world_size_m=case['world_size_m'],
                    geometry=dict(type='axis_aligned_rectangles', rectangles=case['rectangles'])))
                self.assertTrue(scene.segment_clear(case['position'], waypoint, .3))
                controller = GroundRobotController('test')
                controller.set_goal(case['goal'])
                controller.get_action(obs)
                self.assertAlmostEqual(controller.resolution, .1)
                env.close()

    def test_fine_sensor_does_not_clear_a_real_obstacle_at_the_vehicle(self):
        env = MultiRobotEnv(EnvConfig(ground_occupancy_map_resolution=.5))
        rid = env.add_ground_robot(10., 10.)
        env.add_rect_obstacle(10., 10., .1, .1)
        obs = env.reset()[rid]
        semantic = LocalPerception('test').process_occupancy_map(obs['navigation_map'])
        self.assertIsNone(LocalPlanner('test').observed_connection_waypoint(
            (10., 10.), (12., 10.), semantic, obs['navigation_resolution']))
        env.close()

    def test_repeated_entry_checks_reuse_results_but_sensor_changes_do_not(self):
        controller=GroundRobotController('test',safety_distance=1.)
        semantic=np.ones((80,80),np.uint8)
        planner=controller.local_planner
        with patch.object(planner,'observed_connection_waypoint',
                          wraps=planner.observed_connection_waypoint) as compute:
            point=controller._local_waypoint((10.,10.),(30.,10.),semantic,.1)
            self.assertIsNotNone(point)
            for _ in range(100):
                self.assertEqual(controller._local_waypoint(
                    (10.,10.),(30.,10.),semantic.copy(),.1),point)
            self.assertEqual(compute.call_count,1)
            # Mutating the same sensor buffer must invalidate the free step.
            semantic[:]=2
            self.assertIsNone(controller._local_waypoint((10.,10.),(30.,10.),semantic,.1))
            self.assertEqual(compute.call_count,2)
            semantic[:]=1
            controller._local_waypoint((10.,10.),(30.,10.),semantic,.1)
            controller._local_waypoint((10.1,10.),(30.,10.),semantic,.1)
            controller._allow_local_escape=True
            controller._local_waypoint((10.1,10.),(30.,10.),semantic,.1)
            planner.safety_distance=1.5
            controller._local_waypoint((10.1,10.),(30.,10.),semantic,.1)
            self.assertEqual(compute.call_count,6)

    def test_stalled_local_motion_recovers_without_ignoring_obstacles(self):
        fixtures=json.loads((Path(__file__).parent/'fixtures/wildscenes_stops.json').read_text())
        case=next(row for row in fixtures['snapshots'] if row['case']=='V_01_control_200m_forward_r2_repair_r1')
        env=MultiRobotEnv(EnvConfig(world_size=(887.,887.),dt=1.,
            ground_occupancy_map_size=16,ground_occupancy_map_resolution=.5))
        self.addCleanup(env.close)
        rid=env.add_ground_robot(*case['position'])
        for rect in case['rectangles']:
            env.add_rect_obstacle(*rect)
        scene=LegacyObstacleScene(dict(protocol=PROTOCOL,simplified_scene=True,
            unknown_space='free_by_simulation_assumption',origin_xy=[0.,0.],world_size_m=887.,
            geometry=dict(type='axis_aligned_rectangles',rectangles=case['rectangles'])))
        controller=GroundRobotController('test')
        controller.set_goal(case['goal'])
        obs=env.reset()[rid];previous=np.array(case['position']);used_escape=False
        # This fixture contains only the old stop's local sensor window;
        # assess recovery there, not arrival outside the recorded footprint.
        for _ in range(60):
            action=controller.get_action(obs)
            used_escape|=controller._allow_local_escape
            observations,info=env.step({rid:action});obs=observations[rid]
            current=np.asarray(obs['self']['position'][:2])
            self.assertFalse(info['collisions'])
            self.assertTrue(scene.segment_clear(previous,current,.3))
            previous=current.copy()
            if np.linalg.norm(current-case['position'])>=1.:
                break
        self.assertTrue(used_escape)
        self.assertGreaterEqual(np.linalg.norm(previous-case['position']),1.)

    def test_fine_sensor_preserves_footprint_and_subpixel_obstacles(self):
        env = MultiRobotEnv(EnvConfig(ground_occupancy_map_resolution=.5))
        rid = env.add_ground_robot(10., 10.)
        env.add_rect_obstacle(11.055, 10., .01, .2)
        before = env.reset()[rid]['navigation_map'].copy()
        self.assertTrue(before.any())
        env.add_rect_obstacle(16., 10., .2, .2)
        np.testing.assert_array_equal(before, env.reset()[rid]['navigation_map'])
        self.assertEqual(before.shape, (80, 80))
        env.close()

    def test_final_approach_reacquires_goal_despite_failure_penalties(self):
        manager = SubgoalManager((100., 100.), stall=10.)
        manager.update((0., 100.), None, 0.)
        # Recreate a historical state that was already chasing a detour.
        manager.target=np.array([180.,100.]);manager.final=False
        manager.failures=[(manager.goal.copy(),11.)]
        target, changed = manager.update((90., 100.), None, 12.)
        self.assertTrue(changed)
        np.testing.assert_allclose(target, manager.goal)
        for time in (13., 30., 100., 1000.):
            target, _ = manager.update((90., 100.), None, time)
            np.testing.assert_allclose(target, manager.goal)
        self.assertTrue(manager.final)
        self.assertEqual(manager.arrivals, 0)  # 10m remaining is NOT success.

    def test_failed_goal_is_not_outscored_by_distant_temporary_targets(self):
        manager=SubgoalManager((100.,100.),stall=10.)
        manager.update((0.,100.),None,0.)
        for time in (11.,22.,33.):
            target,changed=manager.update((0.,100.),None,time)
            self.assertTrue(changed)
            np.testing.assert_allclose(target,manager.goal)
        self.assertEqual(manager.failed,3)

    def test_local_oscillation_is_not_treated_as_route_progress(self):
        controller=GroundRobotController('test');controller.set_goal((100.,60.))
        recovered=False
        for step in range(45):
            obs=dict(self=dict(position=np.array([15.+2.*np.sin(step),60.,0.]),orientation=0.),
                neighbors=[],occupancy_map=np.zeros((16,16),np.uint8),map_resolution=.5,
                time_step=1.,full_state=dict(time=float(step)))
            controller.get_action(obs)
            recovered|=controller._allow_local_escape
        self.assertTrue(recovered)

    def test_waiting_without_a_drive_goal_does_not_trigger_escape(self):
        controller=GroundRobotController('test')
        obs=dict(self=dict(position=np.array([15.,60.,0.]),orientation=0.),neighbors=[],
            occupancy_map=np.zeros((16,16),np.uint8),map_resolution=.5,time_step=1.)
        for step in (0.,15.,40.):
            controller.get_action(dict(obs,full_state=dict(time=step)))
            self.assertFalse(controller._allow_local_escape)
        controller.set_goal((100.,60.))
        controller.get_action(dict(obs,full_state=dict(time=41.)))
        self.assertFalse(controller._allow_local_escape)

    def test_stationary_failed_connector_is_rejected_far_from_previous_best_pose(self):
        controller=GroundRobotController('test');controller.set_goal((100.,60.))
        controller._local_route_goal=(100.,60.)
        controller._local_route_best=85.
        controller._local_route_progress_time=0.
        controller._local_route_progress_position=np.array([15.,60.])
        controller._local_progress_time=1.
        controller._local_progress_position=np.array([15.,80.])
        controller.global_planner.current_path=[(15.,80.),(17.,80.),(100.,60.)]
        controller.global_planner.current_waypoint_idx=1
        controller.global_planner.goal_position=(100.,60.)
        obs=dict(self=dict(position=np.array([15.,80.,0.]),orientation=0.),neighbors=[],
            occupancy_map=np.zeros((16,16),np.uint8),map_resolution=.5,time_step=1.,
            full_state=dict(time=31.))
        controller.get_action(obs)
        self.assertEqual(len(controller._failed_local_connections),1)
        np.testing.assert_allclose(controller._failed_local_connections[0][1],(17.,80.))

    def test_large_loop_is_detected_even_far_from_best_goal_distance(self):
        controller=GroundRobotController('test');controller.set_goal((100.,60.))
        controller._local_route_goal=(100.,60.)
        controller._local_route_best=60.
        controller._local_route_progress_time=0.
        controller._local_route_progress_position=np.array([40.,60.])
        loop_seen=False
        for step in range(50):
            position=np.array([15.+6.*np.cos(step*np.pi/10),80.+6.*np.sin(step*np.pi/10)])
            obs=dict(self=dict(position=np.r_[position,0.],orientation=0.),neighbors=[],
                occupancy_map=np.zeros((16,16),np.uint8),map_resolution=.5,time_step=1.,
                full_state=dict(time=float(step)))
            controller.get_action(obs)
            loop_seen|=controller._route_loop
        self.assertTrue(loop_seen)
        self.assertTrue(controller._failed_local_connections)

    def test_progress_does_not_reinstate_an_obsolete_backward_route(self):
        controller=GroundRobotController('test',safety_distance=1.)
        controller.set_goal((300.,40.));controller._drive_to((140.,40.))
        controller._route_recovery=True
        # Restore the handoff boundary seen in the real 1000m trace: recovery
        # improved the temporary-target distance by 6m, but the old graph
        # route still begins at an entry behind the vehicle.
        controller._route_recovery_best=100.
        controller._local_route_goal=(140.,40.)
        controller._local_route_best=94.
        controller._local_route_progress_time=100.
        controller._local_route_progress_position=np.array([46.,40.])
        controller._local_progress_time=100.
        controller._local_progress_position=np.array([46.,40.])
        controller._allow_local_escape=True
        controller._local_escape_origin=np.array([40.,40.])
        planner=controller.global_planner
        planner.current_path=[(40.,40.),(30.,40.),(140.,40.)]
        planner.current_waypoint_idx=1;planner.goal_position=(140.,40.);planner.using_graph=True
        obs=dict(self=dict(position=np.array([46.,40.,0.]),orientation=0.),neighbors=[],
            navigation_map=np.zeros((80,80),np.uint8),navigation_resolution=.1,
            map_resolution=.5,time_step=1.,full_state=dict(time=101.))
        action=controller.get_action(obs)
        self.assertGreater(planner.get_current_waypoint()[0],46.)
        self.assertGreater(action['motion'][0],0.)
        self.assertTrue(controller._route_recovery)

    def test_early_final_recovery_does_not_preempt_a_relay_target(self):
        controller=GroundRobotController('test',safety_distance=1.)
        controller.set_goal((300.,40.));controller._drive_to((140.,40.))
        controller._local_route_goal=(140.,40.)
        controller._local_route_best=94.
        controller._local_route_progress_time=0.
        controller._local_route_progress_position=np.array([46.,40.])
        controller._local_progress_time=20.
        controller._local_progress_position=np.array([46.,40.])
        obs=dict(self=dict(position=np.array([46.,40.,0.]),orientation=0.),neighbors=[],
            navigation_map=np.zeros((80,80),np.uint8),navigation_resolution=.1,
            map_resolution=.5,time_step=1.,full_state=dict(time=31.))
        action=controller.get_action(obs)
        self.assertFalse(controller._route_recovery)
        self.assertGreater(action['motion'][0],0.)

    def test_a_moving_detour_does_not_reduce_clearance_just_for_receding_from_goal(self):
        controller=GroundRobotController('test');controller.set_goal((100.,60.))
        for step in range(45):
            obs=dict(self=dict(position=np.array([15.,60.+step,0.]),orientation=np.pi/2),
                neighbors=[],occupancy_map=np.zeros((16,16),np.uint8),map_resolution=.5,
                time_step=1.,full_state=dict(time=float(step)))
            controller.get_action(obs)
            self.assertFalse(controller._allow_local_escape)

    def test_final_approach_does_not_override_probe_radius_after_large_detour(self):
        manager=SubgoalManager((500.,100.))
        manager.update((490.,100.),None,0.)
        self.assertTrue(manager.finishing)
        target,_=manager.update((200.,100.),None,1.)
        self.assertFalse(manager.finishing)
        self.assertLessEqual(np.linalg.norm(target-(200.,100.)),manager.radius)

    def test_near_entry_is_kept_and_only_explicit_gaps_are_marked(self):
        graph = TopologicalGraph('test')
        for name, xy in [('a', (.8, 0.)), ('b', (4., 0.)), ('c', (30., 0.)), ('d', (34., 0.))]:
            graph.add_node(name, xy)
        graph.add_edge('a', 'b'); graph.add_edge('c', 'd')
        planner = GlobalPlanner('test')
        route = planner._stitch_multi_component_path(
            [dict(nodes={'a', 'b'}), dict(nodes={'c', 'd'}, nearest_to_start=('c', (30., 0.)))],
            (0., 0.), (40., 0.), graph)
        self.assertEqual(route[1], (.8, 0.))
        self.assertEqual(planner.start_connection_idx, 1)
        self.assertEqual(planner.observed_connection_indices, {1, 3})
        planner.clear_path()
        self.assertEqual(planner.observed_connection_indices, set())

    def test_failed_local_connector_selects_a_different_graph_entry(self):
        graph=TopologicalGraph('test')
        for name,point in [('bad',(20.,0.)),('side',(20.,8.)),('end',(40.,0.))]:
            graph.add_node(name,point)
        graph.add_edge('bad','side');graph.add_edge('side','end')
        planner=GlobalPlanner('test')
        planner.entry_validator=lambda point:True
        planner.connection_validator=lambda start,end:tuple(end)!=(20.,0.)
        self.assertEqual(planner._entry_node((0.,0.),graph)[0],'side')
        route=planner._stitch_multi_component_path([dict(nodes=set(graph.graph.nodes))],
            (0.,0.),(50.,0.),graph)
        self.assertEqual(route[1],(20.,8.))

    def test_narrow_corridor_recovery_keeps_graph_and_motion_consistent(self):
        env=MultiRobotEnv(EnvConfig(world_size=(120.,120.),dt=1.,
            ground_occupancy_map_size=16,ground_occupancy_map_resolution=.5))
        self.addCleanup(env.close)
        rid=env.add_ground_robot(15.,60.)
        # A 1.4m corridor: the 1m normal centre clearance does not fit;
        # the explicit 0.5m recovery clearance and 0.3m body do fit.
        # Reach the world edges so exploring around the wall cannot bypass
        # the corridor: this test specifically exercises reduced clearance.
        env.add_rect_obstacle(22.,29.65,8.,59.3)
        env.add_rect_obstacle(22.,90.35,8.,59.3)
        graph=TopologicalGraph('test')
        for i,x in enumerate((15.,20.,26.,30.)):
            graph.add_node(str(i),(x,60.),dict(terrain_type='TRAVERSABLE',observation_count=20))
            if i:graph.add_edge(str(i-1),str(i))
        controller=GroundRobotController('test');controller.set_goal((30.,60.))
        obs=env.reset()[rid];escaped=False
        for _ in range(180):
            obs['neighbors']=[dict(type='drone',broadcast=dict(
                drone_id='source',position=[20.,60.,15.],graph=graph.to_message()))]
            action=controller.get_action(obs);escaped|=controller._allow_local_escape
            observations,info=env.step({rid:action});obs=observations[rid]
            self.assertFalse(info['collisions'])
            if np.linalg.norm(obs['self']['position'][:2]-(30.,60.))<=2.:
                break
        self.assertTrue(escaped)
        self.assertLessEqual(np.linalg.norm(obs['self']['position'][:2]-(30.,60.)),2.)

    def test_disconnected_route_crosses_only_locally_observed_free_gap(self):
        env = MultiRobotEnv(EnvConfig(world_size=(120., 120.), dt=1.,
            ground_occupancy_map_size=16, ground_occupancy_map_resolution=.5))
        rid = env.add_ground_robot(15., 60.)
        graph = TopologicalGraph('test')
        for name, xy in [('a', (15.8, 60.)), ('b', (20., 60.)), ('c', (46., 60.)), ('d', (50., 60.))]:
            graph.add_node(name, xy, dict(terrain_type='TRAVERSABLE', observation_count=20))
        graph.add_edge('a', 'b'); graph.add_edge('c', 'd')
        controller = GroundRobotController('test')
        controller.set_goal((52., 60.))
        obs = env.reset()[rid]
        for step in range(90):
            obs['neighbors'] = [dict(type='drone', broadcast=dict(
                drone_id='source', position=[20., 60., 15.], graph=graph.to_message()))]
            action = controller.get_action(obs)
            observations, info = env.step({rid: action})
            self.assertFalse(info['collisions'])
            obs = observations[rid]
            if np.linalg.norm(obs['self']['position'][:2]-(52., 60.)) <= 2.:
                break
        self.assertLess(np.linalg.norm(obs['self']['position'][:2]-(52., 60.)), 2.)
        env.close()


if __name__ == '__main__':
    unittest.main()
