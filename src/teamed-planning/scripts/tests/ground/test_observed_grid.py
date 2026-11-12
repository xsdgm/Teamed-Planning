"""Evidence bounds and terminal detours for accumulated local observations."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ground.observed_grid import ObservedGrid
from ground.ground_main import GroundRobotController
from common.graph_structure import TopologicalGraph
from simulator.env import EnvConfig, MultiRobotEnv
from simulator.wildscenes_legacy_scene import (
    LegacyObstacleScene, PROTOCOL, install_legacy_environment,
)


class ObservedGridTests(unittest.TestCase):
    def test_recentering_preserves_world_evidence_and_marks_new_cells_unknown(self):
        memory=ObservedGrid((100.,100.),resolution=.1,half_width=4.,center=(10.,10.))
        point=np.array([11.05,10.05]);cell=memory._cell(point)
        memory.cells[cell]=0;memory.visits[cell]=7
        memory.recenter((13.,10.))
        shifted=memory._cell(point)
        self.assertEqual(memory.cells[shifted],0)
        self.assertEqual(memory.visits[shifted],7)
        self.assertEqual(memory.cells[memory._cell((16.,10.))],-1)
        memory.recenter((50.,50.))
        self.assertTrue(np.all(memory.cells==-1))
        self.assertTrue(np.all(memory.visits==0))

    def test_nonterminal_pocket_recovers_without_repeated_graph_connectors(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env=MultiRobotEnv(EnvConfig(world_size=(100.,100.),dt=1.,
                ground_occupancy_map_size=16,ground_occupancy_map_resolution=.5))
            self.addCleanup(env.close)
            rid=env.add_ground_robot(20.,40.)
            rectangles=[[24.,40.,1.,13.],[20.,34.,8.,1.],[20.,46.,8.,1.]]
            for rect in rectangles:env.add_rect_obstacle(*rect)
            scene=LegacyObstacleScene(dict(protocol=PROTOCOL,simplified_scene=True,
                unknown_space='free_by_simulation_assumption',origin_xy=[0.,0.],world_size_m=100.,
                geometry=dict(type='axis_aligned_rectangles',rectangles=rectangles)))
            controller=GroundRobotController('test',safety_distance=1.,world_bounds=(0.,0.,100.,100.))
            controller.set_goal((70.,40.))
            obs=env.reset()[rid];recovered=False
            for step in range(450):
                previous=obs['self']['position'][:2].copy()
                obs.pop('obstacles',None);obs['full_state']={'time':float(step)}
                action=controller.get_action(obs);recovered|=controller._route_recovery
                observations,info=env.step({rid:action});obs=observations[rid]
                self.assertFalse(info['collisions'])
                self.assertTrue(scene.segment_clear(previous,obs['self']['position'][:2],.3))
                remaining=np.linalg.norm(obs['self']['position'][:2]-(70.,40.))
                if remaining<=2.:break
            self.assertTrue(recovered)
            self.assertLessEqual(remaining,2.)

    def test_recovery_does_not_hand_back_to_the_same_failed_graph_pocket(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env=MultiRobotEnv(EnvConfig(world_size=(100.,100.),dt=1.,
                ground_occupancy_map_size=16,ground_occupancy_map_resolution=.5))
            self.addCleanup(env.close)
            rid=env.add_ground_robot(20.,40.)
            rectangles=[[24.,40.,1.,13.],[20.,34.,8.,1.],[20.,46.,8.,1.]]
            for rect in rectangles:env.add_rect_obstacle(*rect)
            graph=TopologicalGraph('source')
            for name,point in [('entry',(18.,40.)),('end',(22.,40.)),('goal',(70.,40.))]:
                graph.add_node(name,point,dict(terrain_type='TRAVERSABLE',observation_count=20))
            graph.add_edge('entry','end')
            controller=GroundRobotController('test',safety_distance=1.,world_bounds=(0.,0.,100.,100.))
            controller.set_goal((70.,40.))
            obs=env.reset()[rid]
            recovered=False
            for step in range(500):
                obs.pop('obstacles',None);obs['full_state']={'time':float(step)}
                obs['neighbors']=[dict(type='drone',broadcast=dict(drone_id='source',
                    position=[20.,40.,15.],graph=graph.to_message()))]
                action=controller.get_action(obs);recovered|=controller._route_recovery
                observations,info=env.step({rid:action});obs=observations[rid]
                self.assertFalse(info['collisions'])
                remaining=np.linalg.norm(obs['self']['position'][:2]-(70.,40.))
                if remaining<=2.:break
            self.assertTrue(recovered)
            self.assertLessEqual(remaining,2.)

    def test_archived_terminal_detour_arrives_using_only_local_observations(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures' /
                              'wildscenes_terminal_detour.json').read_text())
        scene = LegacyObstacleScene(dict(
            protocol=PROTOCOL, simplified_scene=True,
            unknown_space='free_by_simulation_assumption', origin_xy=[0., 0.],
            world_size_m=fixture['world_size_m'],
            geometry=dict(type='axis_aligned_rectangles', rectangles=fixture['rectangles'])))
        bounds = fixture['sensor_bounds']
        with contextlib.redirect_stdout(io.StringIO()):
            env = MultiRobotEnv(EnvConfig(world_size=(scene.world_size_m,) * 2, dt=1.,
                ground_occupancy_map_size=16, ground_occupancy_map_resolution=.5))
            self.addCleanup(env.close)
            install_legacy_environment(env, scene)
            rid = env.add_ground_robot(*fixture['start'])
            controller = GroundRobotController('test', safety_distance=1., subgoal_radius=200.,
                world_bounds=(0., 0., scene.world_size_m, scene.world_size_m))
            controller.set_goal(tuple(fixture['goal']))
            obs = env.reset()[rid]
            used_recovery = False
            for step in range(300):
                position = np.asarray(obs['self']['position'][:2]).copy()
                # Every sensor pixel must have complete source geometry. A
                # route outside this fixture cannot pass through missing trees.
                self.assertTrue(bounds[0] <= position[0] - 4. <= position[0] + 4. <= bounds[2])
                self.assertTrue(bounds[1] <= position[1] - 4. <= position[1] + 4. <= bounds[3])
                obs.pop('obstacles', None)
                obs['full_state'] = {'time': float(step)}
                action = controller.get_action(obs)
                used_recovery |= controller._terminal_recovery
                observations, info = env.step({rid: action})
                obs = observations[rid]
                after = np.asarray(obs['self']['position'][:2])
                self.assertFalse(info['collisions'])
                self.assertTrue(scene.segment_clear(position, after, .3))
                remaining = np.linalg.norm(after - fixture['goal'])
                if remaining <= 2.:
                    break
            self.assertTrue(used_recovery)
            self.assertLessEqual(remaining, 2., 'The archived terminal pocket must not cycle indefinitely')

    def test_shifted_observation_requires_complete_free_coverage(self):
        memory = ObservedGrid((10., 10.), resolution=.1, half_width=10.)
        semantic = np.ones((8, 8), dtype=np.uint8)
        semantic[4, 4] = 2
        memory.observe((10.035, 10.025), semantic, .1)
        self.assertEqual(np.count_nonzero(memory.cells >= 0), 49)
        self.assertEqual(np.count_nonzero(memory.cells == 1), 4)
        self.assertEqual(memory.cells[memory._cell((15., 10.))], -1)
        before = memory.cells.copy()
        memory.observe((10.035, 10.025), np.zeros((8, 8), dtype=np.uint8), .1)
        np.testing.assert_array_equal(memory.cells, before)
        memory.observe((10.035, 10.025), np.ones((8, 8), dtype=np.uint8), .1)
        self.assertEqual(np.count_nonzero(memory.cells == 1), 0)

    def test_unknown_and_occupied_regions_never_become_a_route(self):
        memory = ObservedGrid((15., 10.), half_width=12.)
        semantic = np.ones((80, 80), dtype=np.uint8)
        semantic[:, 45:55] = 2
        memory.observe((10., 10.), semantic, .1)
        waypoint = memory.waypoint((10., 10.), .3)
        self.assertIsNotNone(waypoint)
        for point in memory.path:
            self.assertEqual(memory.cells[memory._cell(point)], 0)
            self.assertLess(point[0], 10.5)
        self.assertLess(waypoint[0], 10.5)

    def test_static_free_certificate_survives_shifted_occupied_pixel(self):
        memory = ObservedGrid((10., 10.), resolution=.1, half_width=10.)
        # An obstacle occupies x >= 10.2. With the aligned first observation,
        # the complete world cell [10.1, 10.2) is certified free.
        first = np.ones((8, 8), dtype=np.uint8)
        first[:, 6:] = 2
        memory.observe((10., 10.), first, .1)
        free_cell = memory._cell((10.15, 10.05))
        blocked_cell = memory._cell((10.25, 10.05))
        self.assertEqual(memory.cells[free_cell], 0)
        self.assertEqual(memory.cells[blocked_cell], 1)
        # Shifting the sensor makes pixel [10.15, 10.25) occupied; its overlap
        # with the previously certified free cell must not grow the obstacle.
        shifted = np.ones((8, 8), dtype=np.uint8)
        shifted[:, 5:] = 2
        memory.observe((10.05, 10.), shifted, .1)
        self.assertEqual(memory.cells[free_cell], 0)
        self.assertEqual(memory.cells[blocked_cell], 1)

    def test_terminal_detour_can_first_move_away_from_the_goal(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env = MultiRobotEnv(EnvConfig(world_size=(60., 60.), dt=1.,
                ground_occupancy_map_size=16, ground_occupancy_map_resolution=.5))
            self.addCleanup(env.close)
            rid = env.add_ground_robot(20., 20.)
            # U opening faces away from the goal. Nearest-to-goal local search
            # alone walks into its closed end and cannot complete the detour.
            env.add_rect_obstacle(24., 20., 1., 13.)
            env.add_rect_obstacle(20., 14., 8., 1.)
            env.add_rect_obstacle(20., 26., 8., 1.)
            controller = GroundRobotController('test', safety_distance=.5,
                                                world_bounds=(0., 0., 60., 60.))
            controller.set_goal((28., 20.))
            obs = env.reset()[rid]
            maximum_distance = 8.
            for step in range(320):
                # Match deployed ground observations: no simulator geometry.
                obs.pop('obstacles', None)
                obs['full_state'] = {'time': float(step)}
                action = controller.get_action(obs)
                observations, info = env.step({rid: action})
                obs = observations[rid]
                self.assertFalse(info['collisions'])
                remaining = np.linalg.norm(np.asarray(obs['self']['position'][:2])-(28., 20.))
                maximum_distance = max(maximum_distance, remaining)
                if remaining <= 2.:
                    break
            self.assertGreater(maximum_distance, 12.)
            self.assertLessEqual(remaining, 2.)
            controller.set_goal((40., 40.))
            self.assertIsNone(controller._terminal_map)


if __name__ == '__main__':
    unittest.main()
