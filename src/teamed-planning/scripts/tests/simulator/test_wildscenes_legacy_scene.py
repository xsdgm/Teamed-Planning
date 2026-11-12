"""Contracts for simplified WildScenes-derived native 2D obstacle scenes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from simulator.env import EnvConfig, MultiRobotEnv
from simulator.wildscenes_legacy_scene import (
    LegacyObstacleScene, export_scene, install_legacy_environment, rectangles_from_mask,
)


class LegacySceneTests(unittest.TestCase):
    def make_source(self, directory):
        occupied = np.zeros((80, 80), dtype=bool)
        occupied[20:22, 28:30] = True  # x=[7,7.5], y=[5,5.5]
        occupied[55, 11] = True
        known = np.ones_like(occupied)
        known[40:44, 40:44] = False
        source = Path(directory) / 'forest_map.npz'
        np.savez_compressed(source, occupied=occupied, known=known,
                            resolution_m=.25, origin_xy=[0., 0.],
                            native_origin_xy=[-100., 200.])
        metadata = dict(sequence='TEST', world_size_m=20., resolution_m=.25,
                        coordinate_scale=1., vehicle_envelope_height_m=2.)
        (Path(directory) / 'map_metadata.json').write_text(json.dumps(metadata))
        return source, occupied, known

    def assert_rectangles_exact(self, mask, resolution, origin):
        rectangles = np.asarray(rectangles_from_mask(mask, resolution, origin), float).reshape(-1, 4)
        self.assertTrue(np.isfinite(rectangles).all())
        self.assertTrue((rectangles[:, 2:] > 0).all())
        # Interior samples at several offsets check geometry without relying on
        # the rectangle extraction algorithm or raster-center sampling alone.
        y, x = np.indices(mask.shape)
        for offset in ((.13, .29), (.5, .5), (.87, .71)):
            points = np.stack((x + offset[0], y + offset[1]), axis=-1) * resolution + origin
            cover = np.zeros(mask.shape, dtype=int)
            for cx, cy, width, height in rectangles:
                cover += ((points[..., 0] > cx - width / 2) &
                          (points[..., 0] < cx + width / 2) &
                          (points[..., 1] > cy - height / 2) &
                          (points[..., 1] < cy + height / 2))
            np.testing.assert_array_equal(cover, mask.astype(int))
        self.assertAlmostEqual(float(np.prod(rectangles[:, 2:], axis=1).sum()),
                               float(mask.sum()) * resolution ** 2)
        return rectangles

    def test_rectangle_coordinates_use_xy_and_preserve_nonzero_origin(self):
        mask = np.zeros((7, 11), dtype=bool)
        mask[1:4, 5:9] = True
        rectangles = self.assert_rectangles_exact(mask, .25, (-3., 8.))
        np.testing.assert_allclose(rectangles, [[-1.25, 8.625, 1., .75]])

    def test_holes_disconnected_cells_and_empty_mask_are_exact(self):
        mask = np.zeros((9, 13), dtype=bool)
        mask[1:7, 1:8] = True
        mask[2:6, 2:7] = False  # Hole must not become an obstacle.
        mask[8, 11] = True
        mask[4:6, 10:12] = True
        rectangles = self.assert_rectangles_exact(mask, .5, (4., -2.))
        self.assertLess(len(rectangles), int(mask.sum()))
        self.assertEqual(rectangles_from_mask(np.zeros((4, 6), bool), .5), [])

    def test_export_records_exact_geometry_and_open_unknown_assumption(self):
        with tempfile.TemporaryDirectory() as directory:
            source, occupied, known = self.make_source(directory)
            original = source.read_bytes()
            output = Path(directory) / 'scene.json'
            result = export_scene(source, output)
            self.assertEqual(result, json.loads(output.read_text()))
            self.assertEqual(result['protocol'], 'wildscenes-derived-legacy-2d-v1')
            self.assertIs(result['simplified_scene'], True)
            self.assertEqual(result['unknown_space'], 'free_by_simulation_assumption')
            self.assertEqual(result['source']['map_sha256'], hashlib.sha256(original).hexdigest())
            self.assertEqual(result['source']['shape_yx'], [80, 80])
            self.assertEqual(result['source']['resolution_m'], .25)
            self.assertEqual(result['source']['native_origin_xy'], [-100., 200.])
            self.assertEqual(result['source']['metadata']['sequence'], 'TEST')
            self.assertEqual(result['geometry']['type'], 'axis_aligned_rectangles')
            self.assertIs(result['geometry']['exact_source_occupied'], True)
            stats = result['statistics']
            self.assertEqual(stats['source_occupied_cells'], int(occupied.sum()))
            self.assertEqual(stats['source_known_cells'], int(known.sum()))
            self.assertEqual(stats['source_unknown_cells'], int((~known).sum()))
            self.assertEqual(stats['added_occupied_cells'], 0)
            self.assertEqual(stats['removed_occupied_cells'], 0)
            self.assertAlmostEqual(stats['obstacle_area_m2'], float(occupied.sum()) * .25 ** 2)
            self.assertEqual(stats['rectangle_count'], len(result['geometry']['rectangles']))
            self.assertTrue(result['assumptions'])
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises((FileExistsError, ValueError)):
                export_scene(source, output)

    def test_scene_unknown_is_free_obstacles_and_outside_remain_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            source, occupied, known = self.make_source(directory)
            output = Path(directory) / 'scene.json'
            export_scene(source, output)
            scene = LegacyObstacleScene.from_json(output)
            self.assertEqual(scene.world_size_m, 20.)
            self.assertEqual(scene.rectangles.ndim, 2)
            self.assertEqual(scene.rectangles.shape[1], 4)
            points = np.array([[7.125, 5.125], [10.125, 10.125], [1., 1.],
                               [-.01, 1.], [20., 1.]])
            np.testing.assert_array_equal(scene.truth(points), [True, False, False, True, True])
            np.testing.assert_array_equal(scene.known_at(points), [True, True, True, False, False])
            self.assertTrue(scene.point_clear([10.125, 10.125], .3))
            self.assertFalse(scene.point_clear([.2, 3.], .3))
            self.assertFalse(scene.point_clear([7.125, 5.125], 0.))
            y, x = np.indices(occupied.shape)
            centers = np.stack((x + .5, y + .5), axis=-1) * .25
            np.testing.assert_array_equal(scene.truth(centers.reshape(-1, 2)), occupied.ravel())
            self.assertTrue(scene.known_at(centers.reshape(-1, 2)).all())

    def test_swept_collision_and_euclidean_corner_clearance(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, _ = self.make_source(directory)
            output = Path(directory) / 'scene.json'
            export_scene(source, output)
            scene = LegacyObstacleScene.from_json(output)
            self.assertTrue(scene.point_clear([6., 5.25], .3))
            self.assertTrue(scene.point_clear([9., 5.25], .3))
            self.assertFalse(scene.segment_clear([6., 5.25], [9., 5.25], .3))
            self.assertFalse(scene.segment_clear([6., 4.8], [9., 4.8], .3))
            self.assertTrue(scene.segment_clear([6., 4.6], [9., 4.6], .3))
            # Distance to the lower-left corner is sqrt(.2**2+.2**2).
            # AABB inflation would incorrectly reject the .25m case.
            self.assertTrue(scene.point_clear([6.8, 4.8], .25))
            self.assertFalse(scene.point_clear([6.8, 4.8], .3))
            self.assertFalse(scene.segment_clear([6., 6.], [8., 4.], 0.))

    def test_installed_observations_match_native_rectangles(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, _ = self.make_source(directory)
            output = Path(directory) / 'scene.json'
            export_scene(source, output)
            scene = LegacyObstacleScene.from_json(output)

            def environment():
                config = EnvConfig(world_size=(20., 20.), dt=.1, perception_radius=24.,
                    occupancy_map_size=12, occupancy_map_resolution=2.,
                    ground_occupancy_map_size=16, ground_occupancy_map_resolution=.5)
                env = MultiRobotEnv(config)
                env.add_ground_robot(5., 5.)
                env.add_drone(12., 12., 15.)
                return env

            manual, adapted = environment(), environment()
            for rectangle in scene.rectangles:
                manual.add_rect_obstacle(*rectangle)
            self.assertIs(install_legacy_environment(adapted, scene), adapted)
            self.assertIs(adapted._generate_occupancy_map.__func__, MultiRobotEnv._generate_occupancy_map)
            self.assertIs(adapted._get_observations.__func__, MultiRobotEnv._get_observations)
            for action in (None, {0: np.array([.2, 0.]), 1: np.array([.5, .25])}):
                baseline = manual.reset() if action is None else manual.step(action)[0]
                result = adapted.reset() if action is None else adapted.step(action)[0]
                for robot_id in baseline:
                    np.testing.assert_array_equal(result[robot_id]['occupancy_map'],
                                                  baseline[robot_id]['occupancy_map'])
                    np.testing.assert_array_equal(result[robot_id]['self']['position'],
                                                  baseline[robot_id]['self']['position'])
                    self.assertEqual(result[robot_id]['map_resolution'], baseline[robot_id]['map_resolution'])
                    self.assertEqual(result[robot_id]['map_axes'], 'yx')
                    self.assertNotIn('mapping_occupancy_map', result[robot_id])
                self.assertEqual(result[1]['occupancy_map'].shape, (12, 12))
                self.assertEqual(result[0]['occupancy_map'].shape, (16, 16))

    def test_collision_prefilter_matches_native_rollback_and_event_order(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, _ = self.make_source(directory)
            output = Path(directory) / 'scene.json'
            document = export_scene(source, output)
            # A long, thin rectangle exercises the spatial broad phase far
            # from an obstacle center; two overlapping blocks exercise native
            # obstacle ordering when the first hit restores the prior pose.
            document['geometry']['rectangles'].extend([
                [10., 17., 18., .25], [7.25, 5.25, .25, 1.],
            ])
            scene = LegacyObstacleScene(document)
            scenarios = [
                (6.4, 5.25, 0., 1.), (8.1, 5.25, np.pi, 1.),
                (7.25, 4.4, np.pi / 2, 1.), (7.25, 6.1, -np.pi / 2, 1.),
                (1.2, 16.2, np.pi / 2, .8), (18.8, 16.2, np.pi / 2, .8),
                (9., 10.5, 0., 3.), (7.25, 5.25, 0., 0.),
                (6.8, 4.8, 0., .1),
            ]
            for x, y, heading, speed in scenarios:
                with self.subTest(position=(x, y), heading=heading, speed=speed):
                    native = MultiRobotEnv(EnvConfig(world_size=(20., 20.), dt=1.,
                        occupancy_map_size=10, ground_occupancy_map_size=16))
                    adapted = MultiRobotEnv(EnvConfig(world_size=(20., 20.), dt=1.,
                        occupancy_map_size=10, ground_occupancy_map_size=16))
                    for env in (native, adapted):
                        env.add_ground_robot(x, y, theta=heading, max_speed=4.)
                        # Keep ground-ground collision behavior in the same
                        # call as the filtered obstacle checks.
                        env.add_ground_robot(12., 10.5, max_speed=0.)
                    for rectangle in scene.rectangles:
                        native.add_rect_obstacle(*rectangle)
                    install_legacy_environment(adapted, scene)
                    native.reset()
                    adapted.reset()
                    native_obs, native_info = native.step({0: np.array([speed, 0.])})
                    adapted_obs, adapted_info = adapted.step({0: np.array([speed, 0.])})
                    self.assertEqual(adapted_info['collisions'], native_info['collisions'])
                    for robot_id in native_obs:
                        np.testing.assert_array_equal(adapted_obs[robot_id]['self']['position'],
                                                      native_obs[robot_id]['self']['position'])
                        np.testing.assert_array_equal(adapted_obs[robot_id]['self']['velocity'],
                                                      native_obs[robot_id]['self']['velocity'])
                        np.testing.assert_array_equal(adapted_obs[robot_id]['occupancy_map'],
                                                      native_obs[robot_id]['occupancy_map'])
                    native_obstacle_hit = any(e['object'] == 'obstacle' for e in native_info['collisions'])
                    if native_obstacle_hit:
                        self.assertGreater(adapted.legacy_collision_attempts, 0)
                        self.assertTrue(any(e['native_endpoint_collision']
                                            for e in adapted.legacy_collision_events))

    def test_swept_tunneling_is_recorded_without_changing_native_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, _ = self.make_source(directory)
            output = Path(directory) / 'scene.json'
            export_scene(source, output)
            scene = LegacyObstacleScene.from_json(output)
            native = MultiRobotEnv(EnvConfig(world_size=(20., 20.), dt=1.,
                occupancy_map_size=10, ground_occupancy_map_size=16))
            adapted = MultiRobotEnv(EnvConfig(world_size=(20., 20.), dt=1.,
                occupancy_map_size=10, ground_occupancy_map_size=16))
            for env in (native, adapted):
                env.add_ground_robot(6., 5.25, max_speed=4.)
            for rectangle in scene.rectangles:
                native.add_rect_obstacle(*rectangle)
            install_legacy_environment(adapted, scene)
            native.reset()
            adapted.reset()
            baseline, native_info = native.step({0: np.array([3., 0.])})
            result, info = adapted.step({0: np.array([3., 0.])})
            np.testing.assert_array_equal(result[0]['self']['position'], baseline[0]['self']['position'])
            np.testing.assert_allclose(result[0]['self']['position'][:2], [9., 5.25])
            self.assertEqual(info['collisions'], native_info['collisions'])
            self.assertEqual(info['collisions'], [])
            self.assertEqual(adapted.legacy_collision_attempts, 1)
            event = info['legacy_collision_attempts'][0]
            self.assertEqual(event['before'], [6., 5.25])
            self.assertEqual(event['proposed'], [9., 5.25])
            self.assertFalse(event['native_endpoint_collision'])
            self.assertTrue(event['swept_collision_attempt'])
            adapted.reset()
            self.assertEqual(adapted.legacy_collision_attempts, 0)
            self.assertEqual(adapted.legacy_collision_events, [])


if __name__ == '__main__':
    unittest.main()
