"""Geometry and observation contracts for the real-map adapter."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from simulator.env import EnvConfig, MultiRobotEnv
from simulator.wildscenes_map import RasterForestMap, install_raster_environment


class RasterForestTests(unittest.TestCase):
    def make_map(self, shape=(16, 16), resolution=1., origin=(0., 0.)):
        occupied = np.zeros(shape, bool)
        known = np.ones(shape, bool)
        return occupied, known, resolution, origin

    def test_origin_and_unknown_lookup(self):
        occupied, known, resolution, origin = self.make_map(origin=(-4., 10.))
        occupied[2, 3] = True
        known[1, 2] = False
        forest = RasterForestMap(occupied, known, resolution, origin)
        points = [[-.5, 12.5], [-1.5, 11.5], [-3.5, 10.5], [12., 10.5]]
        np.testing.assert_array_equal(forest.truth(points), [True, True, False, True])
        np.testing.assert_array_equal(forest.known_at(points), [True, False, True, False])

    def test_swept_capsule_blocks_tunneling_corner_and_clearance(self):
        occupied, known, resolution, origin = self.make_map()
        occupied[5, 5] = True
        forest = RasterForestMap(occupied, known, resolution, origin)
        self.assertFalse(forest.segment_clear([3., 5.5], [8., 5.5], .3))
        self.assertFalse(forest.segment_clear([4., 6.], [6., 4.], 0.))
        self.assertFalse(forest.segment_clear([3., 4.8], [8., 4.8], .3))
        self.assertTrue(forest.segment_clear([3., 4.6], [8., 4.6], .3))
        self.assertFalse(forest.point_clear([.2, 3.], .3))
        self.assertTrue(forest.point_clear([.4, 3.], .3))

    def test_observation_pooling_preserves_small_and_unknown_cells(self):
        occupied, known, resolution, origin = self.make_map(shape=(32, 32), resolution=.5)
        occupied[7, 7] = True
        known[12, 12] = False
        forest = RasterForestMap(occupied, known, resolution, origin)
        view = forest.local_occupancy([8., 8.], 8, 2.)
        self.assertEqual(view.shape, (8, 8))
        self.assertEqual(view.dtype, np.uint8)
        self.assertEqual(view[1, 1], 1)
        self.assertEqual(view[3, 3], 1)
        self.assertEqual(int(view.sum()), 2)
        shifted = forest.local_occupancy([8.25, 8.25], 8, 2.)
        self.assertTrue(shifted[:, -1].all())
        self.assertTrue(shifted[-1, :].all())
        self.assertEqual(shifted[1, 1], 1)

    def test_unknown_crossing_and_npz_roundtrip(self):
        occupied, known, resolution, origin = self.make_map()
        known[:, 6] = False
        forest = RasterForestMap(occupied, known, resolution, origin)
        self.assertFalse(forest.segment_known([4., 8.], [9., 8.]))
        self.assertFalse(forest.segment_clear([4., 8.], [9., 8.]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'forest.npz'
            np.savez_compressed(path, occupied=occupied, known=known,
                resolution_m=resolution, origin_xy=origin, source_origin_xy=[100., 200.])
            loaded = RasterForestMap.from_npz(path)
            self.assertEqual(loaded.metadata['source_origin_xy'], [100., 200.])
            np.testing.assert_array_equal(loaded.blocked, forest.blocked)

    def test_local_known_mask_matches_shifted_cell_geometry(self):
        occupied, known, resolution, origin = self.make_map(
            shape=(12, 20), resolution=.25, origin=(-4., 10.))
        occupied[3, 9] = True
        known[6, 5] = False
        forest = RasterForestMap(occupied, known, resolution, origin)
        self.assertIsNone(forest._unknown_integral)
        position, shape, pixel = np.array([-1.35, 11.425]), (8, 12), .5
        observation = forest.local_observation(position, shape, pixel, (6., 4.))
        expected_occupied = np.ones(shape, dtype=np.uint8)
        expected_known = np.zeros(shape, dtype=bool)
        lower = position - np.array([shape[1], shape[0]]) * pixel / 2
        sy, sx = np.indices(known.shape)
        source_lo = np.stack((sx, sy), axis=-1) * resolution + origin
        source_hi = source_lo + resolution
        for row in range(shape[0]):
            for column in range(shape[1]):
                lo = lower + np.array([column, row]) * pixel
                hi = lo + pixel
                inside = (lo >= forest.origin_xy).all() and (hi <= forest.upper_xy).all()
                overlap = ((source_hi > lo).all(axis=-1) &
                           (source_lo < hi).all(axis=-1))
                expected_known[row, column] = inside and known[overlap].all()
                expected_occupied[row, column] = (
                    not inside or (occupied[overlap] | ~known[overlap]).any())
        np.testing.assert_array_equal(observation['occupancy_map'], expected_occupied)
        np.testing.assert_array_equal(observation['known_mask'], expected_known)
        self.assertTrue((observation['known_mask'] & (observation['occupancy_map'] == 1)).any())
        self.assertTrue((~observation['known_mask']).any())
        self.assertTrue((observation['occupancy_map'] == 0).any())
        cached = forest._unknown_integral
        self.assertFalse(cached.flags.writeable)
        forest.local_observation(position + .125, shape, pixel)
        self.assertIs(forest._unknown_integral, cached)

    def test_local_observation_rejects_invalid_geometry(self):
        forest = RasterForestMap(*self.make_map())
        for size in (0, -1, 2.5, np.nan, (2, 0), (2, 3.5), (2,)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                forest.local_observation([8., 8.], size, .5)
        for resolution in (0., -1., np.nan, np.inf):
            with self.subTest(resolution=resolution), self.assertRaises(ValueError):
                forest.local_observation([8., 8.], 4, resolution)
        for position in ([1.], [np.nan, 1.], [[1., 2.]]):
            with self.subTest(position=position), self.assertRaises(ValueError):
                forest.local_observation(position, 4, .5)
        with self.assertRaises(ValueError):
            forest.local_observation([8., 8.], 4, .5, 2.1)

    def test_fine_mapping_observation_preserves_original_sensor_and_ground(self):
        occupied, known, resolution, origin = self.make_map(shape=(128, 160), resolution=.25)
        occupied[20, 45] = True
        known[60, 100] = False
        forest = RasterForestMap(occupied, known, resolution, origin)

        def make_environment(mapping_resolution):
            config = EnvConfig(world_size=(40., 32.), dt=.5, perception_radius=24.,
                occupancy_map_size=12, occupancy_map_resolution=2.,
                ground_occupancy_map_size=16, ground_occupancy_map_resolution=.5)
            env = install_raster_environment(MultiRobotEnv(config), forest, mapping_resolution)
            env.add_ground_robot(18., 16.)
            env.add_drone(20., 15., 15.)
            return env

        def assert_original_equal(expected, actual):
            if isinstance(expected, dict):
                self.assertEqual(set(expected), set(actual))
                for key in expected:
                    assert_original_equal(expected[key], actual[key])
            elif isinstance(expected, list):
                self.assertEqual(len(expected), len(actual))
                for first, second in zip(expected, actual):
                    assert_original_equal(first, second)
            elif isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(expected, actual)
            else:
                self.assertEqual(expected, actual)

        extra = {'mapping_occupancy_map', 'mapping_known_mask', 'mapping_resolution', 'mapping_axes'}
        for mapping_resolution in (.5, .25):
            with self.subTest(mapping_resolution=mapping_resolution):
                original, adapted = make_environment(None), make_environment(mapping_resolution)
                for action in (None, {0: np.zeros(2), 1: np.array([.25, .375])}):
                    baseline = original.reset() if action is None else original.step(action)[0]
                    fine = adapted.reset() if action is None else adapted.step(action)[0]
                    assert_original_equal(baseline[0], fine[0])
                    assert_original_equal(baseline[1], {k: v for k, v in fine[1].items() if k not in extra})
                    self.assertEqual(set(fine[1]) - set(baseline[1]), extra)
                    self.assertEqual(fine[1]['occupancy_map'].shape, (12, 12))
                    self.assertEqual(fine[1]['map_resolution'], 2.)
                    n = int(24 / mapping_resolution)
                    self.assertEqual(fine[1]['mapping_occupancy_map'].shape, (n, n))
                    self.assertEqual(fine[1]['mapping_known_mask'].shape, (n, n))
                    self.assertEqual(fine[1]['mapping_resolution'], mapping_resolution)
                    self.assertEqual(fine[1]['mapping_axes'], 'yx')
                    self.assertEqual(fine[1]['mapping_occupancy_map'].dtype, np.uint8)
                    self.assertEqual(fine[1]['mapping_known_mask'].dtype, np.bool_)
                    self.assertTrue(np.isin(fine[1]['mapping_occupancy_map'], [0, 1]).all())
                    # A source-sized obstacle and an unknown patch both remain
                    # blocked; only the latter invalidates known support.
                    pos = fine[1]['self']['position'][:2]
                    lo = pos - 12.
                    for point, is_known in (([11.375, 5.125], True), ([25.125, 15.125], False)):
                        column, row = np.floor((np.asarray(point) - lo) / mapping_resolution).astype(int)
                        self.assertEqual(fine[1]['mapping_occupancy_map'][row, column], 1)
                        self.assertEqual(fine[1]['mapping_known_mask'][row, column], is_known)

    def test_mapping_adapter_rejects_invalid_resolution_before_install(self):
        forest = RasterForestMap(*self.make_map())
        for resolution in (0., -1., np.nan, np.inf, .7, 25.):
            env = MultiRobotEnv(EnvConfig(perception_radius=24., occupancy_map_size=12,
                occupancy_map_resolution=2., ground_occupancy_map_size=16))
            with self.subTest(resolution=resolution), self.assertRaises(ValueError):
                install_raster_environment(env, forest, resolution)
            self.assertFalse(getattr(env, '_forest_adapter_installed', False))

    def test_environment_records_proposal_and_preserves_robot_collision(self):
        occupied, known, resolution, origin = self.make_map(shape=(64, 64), resolution=.5)
        known[:, 20] = False
        forest = RasterForestMap(occupied, known, resolution, origin)
        cfg = EnvConfig(world_size=(32., 32.), dt=1., perception_radius=24.,
            occupancy_map_size=12, occupancy_map_resolution=2.,
            ground_occupancy_map_size=16, ground_occupancy_map_resolution=.5)
        env = install_raster_environment(MultiRobotEnv(cfg), forest)
        robot_id = env.add_ground_robot(9., 16., max_speed=3.)
        observation = env.reset()[robot_id]
        self.assertEqual(observation['occupancy_map'].shape, (16, 16))
        _, info = env.step({robot_id: np.array([3., 0.])})
        np.testing.assert_allclose(env.ground_robots[0].position[:2], [9., 16.])
        self.assertEqual(env.raster_collision_attempts, 1)
        self.assertEqual(env.raster_unknown_attempts, 1)
        self.assertEqual(info['raster_collision_attempts'][0]['proposed'], [12., 16.])
        other = env.add_ground_robot(9.2, 16.)
        _, info = env.step({robot_id: np.zeros(2), other: np.zeros(2)})
        self.assertTrue(any(event['object'] == 'robot' for event in info['collisions']))


if __name__ == '__main__':
    unittest.main()
