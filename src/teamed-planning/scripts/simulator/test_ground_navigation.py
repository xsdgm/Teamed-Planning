"""Partial-map navigation regressions, independent of the UAV policy."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common.graph_structure import TopologicalGraph
from ground.ground_main import GroundRobotController
from ground.global_planner import GlobalPlanner
from ground.local_planner import LocalPlanner
from simulator.env import EnvConfig, MultiRobotEnv


def graph_from(points, edges):
    graph = TopologicalGraph('test')
    for node, point in points.items():
        graph.add_node(node, point, dict(terrain_type='TRAVERSABLE', observation_count=20))
    for first, second in edges:
        graph.add_edge(first, second)
    return graph


def navigate(graph, goal, obstacles=(), steps=200, update=None):
    env = MultiRobotEnv(EnvConfig(world_size=(120, 120), dt=1.,
        ground_occupancy_map_size=16, ground_occupancy_map_resolution=.5))
    rid = env.add_ground_robot(15., 60., max_speed=1.)
    for rectangle in obstacles:
        env.add_rect_obstacle(*rectangle)
    controller = GroundRobotController('test', fusion_interval=5.)
    controller.set_goal(goal)
    observation = env.reset()[rid]
    collisions = 0
    try:
        for frame in range(steps):
            if update is not None and frame == 5:
                graph = update
            observation['neighbors'] = [dict(type='drone', broadcast=dict(
                drone_id='test', position=[15., 60., 15.], graph=graph.to_message()))]
            observation['full_state'] = {'time': env.time}
            observation.pop('obstacles', None)
            action = controller.get_action(observation)
            observations, info = env.step({rid: action})
            observation = observations[rid]
            collisions += len(info['collisions'])
            remaining = np.linalg.norm(np.asarray(observation['self']['position'])[:2] - goal)
            if remaining <= 2.:
                return True, frame + 1, collisions
        return False, steps, collisions
    finally:
        env.close()


class GroundNavigationTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)

    def test_wall_entry_uses_accessible_alternative(self):
        graph = graph_from({'near': (18.5,60.), 'left': (11.,60.), 'ul': (11.,69.),
            'ur': (22.,69.), 'right': (22.,60.), 'goal': (30.,60.)},
            [('left','ul'), ('ul','ur'), ('ur','right'), ('right','goal'), ('near','right')])
        reached, steps, collisions = navigate(graph, (30.,60.), [(16.25,60.,.5,12.)])
        self.assertTrue(reached)
        self.assertLess(steps, 100)
        self.assertEqual(collisions, 0)

    def test_new_shortcut_replaces_old_route(self):
        points = {'a': (15.,60.), 'b': (15.,80.), 'c': (35.,80.), 'g': (35.,60.)}
        edges = [('a','b'), ('b','c'), ('c','g')]
        reached, steps, collisions = navigate(graph_from(points, edges), (35.,60.),
            steps=60, update=graph_from(points, edges + [('a','g')]))
        self.assertTrue(reached)
        self.assertLess(steps, 50)
        self.assertEqual(collisions, 0)

    def test_sparse_final_connector_advances_in_observed_free_space(self):
        graph = graph_from({'a': (15.,60.), 'b': (19.,60.), 'c': (23.,60.)},
                           [('a','b'), ('b','c')])
        reached, steps, collisions = navigate(graph, (30.,60.), steps=50)
        self.assertTrue(reached)
        self.assertEqual(collisions, 0)

    def test_repeated_unchanged_maps_do_not_restart_route(self):
        graph = graph_from({'a': (15.,60.), 'b': (15.,80.), 'c': (35.,80.), 'g': (35.,60.)},
                           [('a','b'), ('b','c'), ('c','g')])
        reached, steps, collisions = navigate(graph, (35.,60.), steps=150)
        self.assertTrue(reached)
        self.assertEqual(collisions, 0)

    def test_disconnected_entry_rejects_inaccessible_nearest_node(self):
        graph = graph_from({'blocked': (1.,0.), 'entry': (0.,4.), 'exit': (8.,4.)},
                           [('blocked','entry'), ('entry','exit')])
        planner = GlobalPlanner('test')
        planner.entry_validator = lambda point: tuple(point) != (1.,0.)
        route = planner._stitch_multi_component_path(
            [{'nodes': set(graph.graph.nodes)}], (0.,0.), (10.,4.), graph)
        self.assertEqual(tuple(route[1]), (0.,4.))

    def test_unreachable_leading_component_is_skipped(self):
        graph = graph_from({'side': (5.,8.), 'b': (10.,0.), 'c': (30.,0.)}, [('b','c')])
        planner = GlobalPlanner('test')
        planner.entry_validator = lambda point: tuple(point) != (5.,8.)
        route = planner._stitch_multi_component_path(
            [{'nodes': {'side'}}, {'nodes': {'b','c'}}], (0.,0.), (40.,0.), graph)
        self.assertEqual([tuple(p) for p in route], [(0.,0.), (10.,0.), (30.,0.), (40.,0.)])
        self.assertEqual(planner.start_connection_idx, 1)
        planner.entry_validator = lambda point: False
        self.assertIsNone(planner._stitch_multi_component_path(
            [{'nodes': {'side'}}, {'nodes': {'b','c'}}], (0.,0.), (40.,0.), graph))

    def test_robot_enclosed_by_inflation_escapes_with_reduced_margin(self):
        # Local map recorded where the vehicle waited forever (training scene 55426):
        # a 3.3 m gap between two trees. 1 = traversable, 2 = obstacle; row 0 = lowest y.
        rows = ['2222222111111111', '2222222111111111', '2222222211111111', '2222222111111122',
                '2222222111111222', '2222222111112222', '2222221111112222', '2222211111122222',
                '2222111111122222', '2111111111122222', '1111111111122222', '1111111111222222',
                '1111111111122222', '1111111111122222', '1111111111122222', '1111111111122222']
        semantic = np.array([[int(c) for c in row] for row in rows], np.uint8)
        position = np.array([105.9, 713.4])
        nodes = [(2.4, -6.77), (-1.88, 7.19), (-0.44, -8.53), (-5.04, 7.05)]
        strict = LocalPlanner('test', escape_distance=1.5)
        planner = LocalPlanner('test')
        blocked = planner.observed_blocked_cells(semantic, .5)[0]
        self.assertTrue(blocked[8, 8])  # Inside the full margin, on the ridge between trees.
        for offset in nodes:
            self.assertIsNone(strict.observed_connection_waypoint(position, position + offset, semantic, .5))
        steps = [planner.observed_connection_waypoint(position, position + offset, semantic, .5)
                 for offset in nodes]
        self.assertTrue(any(step is not None for step in steps))
        for step in steps:
            if step is None:
                continue
            cell = np.floor((np.asarray(step) - position) / .5 + 8).astype(int)
            self.assertEqual(semantic[cell[1], cell[0]], 1)
            self.assertLessEqual(np.linalg.norm(np.asarray(step) - position), 2. + 1e-9)

    def test_escape_margin_is_used_only_when_enclosed(self):
        planner = LocalPlanner('test')
        semantic = np.ones((16, 16), np.uint8)
        self.assertEqual(planner.observed_connection_waypoint(np.zeros(2), (10., 0.), semantic, .5), (2., 0.))
        # A wall 2 m ahead: the full 1.5 m margin already stops the robot here,
        # and it is not enclosed, so the reduced margin must not let it closer.
        semantic[:, 12:] = 2
        self.assertFalse(planner._enclosed(planner.observed_blocked_cells(semantic, .5)[0], (8, 8)))
        self.assertIsNone(planner.observed_connection_waypoint(np.zeros(2), (10., 0.), semantic, .5))

    def test_component_path_minimizes_length(self):
        graph = graph_from({'a': (0.,0.), 'b': (100.,0.), 'c': (10.,0.),
            'd': (20.,0.), 'z': (30.,0.), 'other': (35.,0.)},
            [('a','b'), ('b','z'), ('a','c'), ('c','d'), ('d','z')])
        planner = GlobalPlanner('test')
        path = planner.plan_path((0.,0.), (35.,0.), graph)
        self.assertAlmostEqual(planner._compute_path_length(path), 35.)

    def test_safety_margin_changes_obstacle_inflation(self):
        semantic = np.ones((32,32), np.uint8)
        semantic[16,20] = 2
        narrow, _ = LocalPlanner('test', safety_distance=.5).observed_blocked_cells(semantic,.5)
        wide, _ = LocalPlanner('test', safety_distance=1.5).observed_blocked_cells(semantic,.5)
        self.assertFalse(narrow[16,18])
        self.assertTrue(wide[16,18])

    def test_unknown_barrier_cannot_be_crossed(self):
        semantic = np.ones((16,16), np.uint8)
        semantic[:,9:] = 0
        waypoint = LocalPlanner('test').observed_connection_waypoint((0.,0.), (10.,0.), semantic,.5)
        self.assertTrue(waypoint is None or waypoint[0] < .5)

    def test_reset_accepts_new_episode_packet(self):
        controller = GroundRobotController('test', fusion_interval=5.)
        graph = graph_from({'a': (15.,60.)}, [])
        packet = dict(broadcast=dict(position=[15.,60.,15.], graph=graph.to_message()))
        self.assertTrue(controller._receive_and_fuse_graphs({'test': packet}, 10.))
        controller.reset()
        self.assertTrue(controller._receive_and_fuse_graphs({'test': packet}, 0.))


if __name__ == '__main__':
    unittest.main()
