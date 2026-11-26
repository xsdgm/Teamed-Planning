"""Regression tests against the actual frozen vehicle, including its temporary review planner."""
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class FrozenRuntimeTests(unittest.TestCase):
    def test_connected_ground_in_frozen_runtime(self):
        result = subprocess.run([sys.executable, '-IB', __file__, 'frozen'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


def frozen_tests():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vtemp2_3.frozen_navigation import runtime
    runtime(Path(__file__).resolve().parents[5])
    from common.graph_structure import TopologicalGraph
    from ground.global_planner import GlobalPlanner
    from ground.ground_main import GroundRobotController
    from vtemp4.vtemp4_connected_ground import ConnectedGroundPlanner, install_connected_ground
    from vtemp4.vtemp4_anchored_ground import AnchoredGroundPlanner, install_anchored_ground

    def roads(nodes, edges):
        graph = TopologicalGraph()
        for key, position in nodes.items():
            graph.add_node(key, position, {'observation_count': 8, 'terrain_type': 'TRAVERSABLE'})
        for a,b in edges:
            graph.add_edge(a,b)
        return graph

    start, goal = (10.,10.), (90.,10.)
    nodes = {'isolated':(10.5,10.), 'entry':(14.,13.), 'mid':(21.,13.),
             'end':(28.,13.), 'unconnected':goal}
    edges = [('entry','mid'),('mid','end')]

    def driver():
        controller = GroundRobotController('test',resolution=.5,world_bounds=(0.,0.,120.,120.))
        controller.set_goal(goal)
        return controller

    def observation(cells,t=1.):
        return {'self':{'position':np.array([*start,0.]),'orientation':0.}, 'neighbors':[],
                'occupancy_map':cells,'map_resolution':.5,'time_step':1.,'full_state':{'time':t}}

    class PlannerTests(unittest.TestCase):
        def test_disconnected_graph_ends_at_true_frontier(self):
            graph = roads(nodes,edges)
            planner = ConnectedGroundPlanner('test')
            planner.entry_validator = lambda xy: True
            route = planner.plan_path(start,goal,graph)
            self.assertEqual(route, [start,nodes['entry'],nodes['mid'],nodes['end']])
            self.assertEqual(planner.start_connection_idx, 1)
            self.assertFalse(planner.is_goal_reached(nodes['end']))
            for xy in route:
                planner.advance_waypoint(xy,1.)
            self.assertTrue(planner.waiting_at_frontier)
            self.assertTrue(planner.should_replan(nodes['end']))
            self.assertGreaterEqual(planner._compute_path_length(route),80.)

        def test_within_tolerance_entry_still_anchors_the_actual_road_edge(self):
            near = (10.25,10.)
            graph = roads({'a':near,'b':(30.,10.),'c':goal}, [('a','b'),('b','c')])
            planner = ConnectedGroundPlanner('test')
            planner.entry_validator = lambda xy: True
            route = planner.plan_path(start,goal,graph)
            planner.advance_waypoint(start,1.)
            self.assertEqual(route[1], near)
            self.assertEqual(planner.current_waypoint_idx, 2)
            self.assertEqual(planner.get_current_waypoint(),(30.,10.))

        def test_confidence_failure_reroutes_without_altering_the_observed_graph(self):
            xy = {'a':(11.,10.),'b':(20.,10.),'c':goal,'d':(18.,22.)}
            graph = roads(xy,[('a','b'),('b','c'),('a','d'),('d','c')])
            planner = ConnectedGroundPlanner('test',confidence_threshold=.5)
            planner.entry_validator = lambda point: point==xy['a']
            def evidence(a,b,d_bar):
                return .5 if {a,b}=={xy['a'],xy['b']} else .9
            with patch.object(graph,'_edge_confidence',evidence):
                self.assertEqual(planner.plan_path(start,goal,graph), [start,xy['a'],xy['d'],goal])
                self.assertTrue(graph.graph.has_edge('a','b'))
                self.assertEqual(graph.edge_confidence_from_positions(xy['a'],xy['b']),.5)

        def test_remote_components_and_weak_only_edges_are_not_used(self):
            planner=ConnectedGroundPlanner('test',confidence_threshold=.5)
            planner.entry_validator=lambda point:True
            distant=roads({'a':(40.,10.),'b':goal}, [('a','b')])
            self.assertEqual(planner.plan_path(start,goal,distant),[start])
            weak=roads(nodes,edges)
            with patch.object(weak,'_edge_confidence',return_value=.5):
                self.assertEqual(planner.plan_path(start,goal,weak),[start])
            self.assertEqual(weak.graph.number_of_edges(),2)

        def test_empty_graph_direct_near_goal_and_obstacle_stop(self):
            control, legacy = driver(), driver()
            install_connected_ground(control)
            free=np.zeros((16,16),int)
            self.assertEqual(control.get_action(observation(free))['motion'],
                             legacy.get_action(observation(free))['motion'])
            self.assertGreater(control.get_action(observation(free))['motion'][0],0.)
            control.unified_graph = roads({'a':(30.,10.)},[])
            control._drive_to((11.5,10.))
            self.assertGreater(control.get_action(observation(free))['motion'][0],0.)
            control._drive_to(goal)
            self.assertEqual(control.get_action(observation(np.ones_like(free)))['motion'],(0.,0.))

        def test_review_uses_same_fix_and_neighboring_legacy_controller_is_unaffected(self):
            controller, untouched = driver(),driver()
            graph = roads(nodes,edges)
            controller.unified_graph=graph
            planner=controller.global_planner
            bad=[start,nodes['isolated'],nodes['mid'],goal]
            planner.current_path,planner.goal_position = bad,goal
            planner.current_waypoint_idx,planner.start_connection_idx,planner.using_graph = 2,1,True
            install_connected_ground(controller)
            self.assertIs(controller.global_planner,planner)
            self.assertIs(controller.unified_graph,graph)
            self.assertIsNone(planner.current_path)
            self.assertEqual(controller.goal_position,goal)
            self.assertIs(type(untouched.global_planner),GlobalPlanner)
            # Force the native five-second review of an incumbent detour.
            planner.current_path=[start,(10.,100.),goal]
            planner.goal_position,planner.current_waypoint_idx,planner.start_connection_idx = goal,1,1
            planner.using_graph=True
            controller._last_route_review=0.
            candidates=[]
            original=ConnectedGroundPlanner._plan_with_graph
            def observe(instance,*args,**kwargs):
                candidates.append(instance)
                return original(instance,*args,**kwargs)
            with (patch.object(controller,'_receive_and_fuse_graphs',return_value=True),
                  patch.object(controller.graph_fusion,'get_unified_graph',return_value=graph),
                  patch.object(ConnectedGroundPlanner,'_plan_with_graph',observe)):
                controller.get_action(observation(np.zeros((16,16),int),t=10.))
            self.assertEqual(len(candidates),1)
            self.assertIsNot(candidates[0],planner)
            self.assertEqual(planner.current_path,[start,nodes['entry'],nodes['mid'],nodes['end']])
            install_connected_ground(controller)
            self.assertIsNotNone(planner.current_path)

        def test_periodic_episode_installs_planner_before_first_physics_action(self):
            from vtemp4.vtemp4_periodic_history import PeriodicHistoryFleetEpisode
            class GoalEpisode(PeriodicHistoryFleetEpisode):
                designate_near_goal=True

            case=dict(seed=17,case_id='connected-smoke',kind='v9_random',start=[500.,490.],goal=[630.,490.],
                      world_size=1000.,direct_distance_m=130.,circles=[])
            for mode, kind in [('vtemp1_connected', ConnectedGroundPlanner),
                               ('vtemp1_connected_anchored', AnchoredGroundPlanner)]:
                with self.subTest(mode=mode):
                    ep=GoalEpisode(Path(__file__).resolve().parents[6],case,SimpleNamespace(value=lambda obs:0.),
                                   fast_seed=24,budget_s=2670,collector_limit_s=2,review_interval_s=30,
                                   ground_control=mode)
                    try:
                        self.assertIs(type(ep.env.ground_controller.global_planner),kind)
                        self.assertEqual(ep.env.ground_action['broadcast']['target'],case['goal'])
                        while not ep.finished:
                            ep.step()
                        self.assertEqual(ep.fleet_summary()['ground_control'],mode)
                        self.assertEqual(ep.requests[0]['physics_broadcasts'],2)
                    finally:
                        ep.close()

        def test_anchored_raster_uses_the_same_map_for_vehicle_and_drone_sensors(self):
            from vtemp4.vtemp4_periodic_history import PeriodicHistoryFleetEpisode
            from vtemp2_3.wildscenes_runtime import load_forest

            class GoalEpisode(PeriodicHistoryFleetEpisode):
                designate_near_goal = True

            repo = Path(__file__).resolve().parents[5]
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'bridge.npz'
                blocked = np.zeros((200, 200), bool)
                blocked[:, 98:102] = True
                blocked[125:135, 98:102] = False
                np.savez(path, occupied=blocked, known=np.ones_like(blocked), resolution_m=1., origin_xy=[0.,0.])
                case = dict(seed=17, case_id='anchored-raster', kind='procedural_wooded_valley',
                            terrain='wildscenes_raster', source_unknown_space='blocked',
                            map_path=str(path), map_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            resolution_m=1., required_clearance_m=1.5, world_size=200.,
                            start=[90., 100.], goal=[150., 100.], direct_distance_m=60., circles=[])
                options = dict(fast_seed=24, budget_s=90, collector_limit_s=2, review_interval_s=30)
                actor = SimpleNamespace(value=lambda obs: 0.)
                for legacy in ('vtemp1', 'vtemp1_connected'):
                    with self.assertRaisesRegex(ValueError, 'random-circle'):
                        GoalEpisode(repo, case, actor, **options, ground_control=legacy)
                with self.assertRaisesRegex(ValueError, 'route distance'):
                    GoalEpisode(repo, case, actor, **options, ground_control='vtemp1_connected_anchored',
                                progress_distance='traversable')
                ep = GoalEpisode(repo, case, actor, **options, ground_control='vtemp1_connected_anchored')
                try:
                    self.assertEqual(ep.observation_protocol, 'wildscenes_source_raster')
                    self.assertIs(type(ep.env.ground_controller.global_planner), AnchoredGroundPlanner)
                    forest = load_forest(case, repo)
                    self.assertFalse(forest.segment_clear(case['start'], case['goal']))
                    self.assertTrue(forest.segment_clear([90., 130.], [150., 130.], 1.5))
                    for rid in (ep.env.ground, *ep.env.ids):
                        obs = ep.env.obs[rid]
                        tile = obs['occupancy_map']
                        sensor = forest.local_observation(obs['self']['position'], tile.shape, obs['map_resolution'])
                        np.testing.assert_array_equal(tile, np.where(sensor['known_mask'], sensor['occupancy_map'], -1))
                    while not ep.finished:
                        ep.step()
                    self.assertEqual(ep.requests[0]['physics_broadcasts'], 2)
                finally:
                    ep.close()

        def test_anchored_prefers_observed_entry_and_falls_back_when_near_road_is_unusable(self):
            pts={'near':(10.5,10.), 'near_end':(19.,14.), 'far':(21.,10.),
                 'far_end':(44.,10.), 'remote_goal':goal}
            cs=driver()
            cs.position=np.array(start)
            cs._last_semantic_map=np.ones((16,16),int)
            install_anchored_ground(cs)
            planner=cs.global_planner
            def plan(graph):
                planner.entry_validator=lambda xy: True
                return planner.plan_path(start,goal,graph)
            edges=[('near','near_end'),('far','far_end')]
            self.assertEqual(plan(roads(pts,edges)),[start,pts['near'],pts['near_end']])
            # An isolated close node cannot take precedence over usable roads.
            self.assertEqual(plan(roads(pts,[edges[1]]))[1],pts['far'])
            # Neither can a close component whose observed edges only move sideways.
            sideways=dict(pts,near_end=(10.5,19.))
            self.assertEqual(plan(roads(sideways,edges))[1],pts['far'])
            # The raw close endpoint is visible; its inflated cell lacks a full safe route.
            cs._last_semantic_map[8,11]=2
            self.assertEqual(plan(roads(pts,edges))[1],pts['far'])

        def test_anchored_factory_is_private_and_keeps_live_path(self):
            a,b=driver(),driver()
            install_connected_ground(a)
            install_connected_ground(b)
            incumbent=a.global_planner
            incumbent.current_path=[start,goal]
            install_anchored_ground(a)
            install_anchored_ground(a)
            self.assertIs(a.global_planner,incumbent)
            self.assertEqual(incumbent.current_path,[start,goal])
            make_a=a.get_action.__func__.__globals__['GlobalPlanner']
            make_b=b.get_action.__func__.__globals__['GlobalPlanner']
            self.assertIs(make_a('review').controller,a)
            self.assertIs(type(make_b('review')),ConnectedGroundPlanner)

    with redirect_stdout(io.StringIO()):
        result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(PlannerTests))
    return result.wasSuccessful()


if __name__ == '__main__':
    if sys.argv[1:]==['frozen']:
        sys.exit(0 if frozen_tests() else 1)
    unittest.main()
