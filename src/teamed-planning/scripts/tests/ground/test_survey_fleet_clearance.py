"""Continuous native-pixel geometry for connecting an off-centre ground vehicle."""
import unittest

import numpy as np
from scipy.optimize import minimize_scalar

from ground.survey_fleet_clearance import (
    _sweep_hits_boxes, certified_start_connector, native_segment_clear,
)


class NativeClearanceTests(unittest.TestCase):
    def test_previous_1100m_stall_has_a_safe_departure_beyond_controller_threshold(self):
        from simulator.survey_fleet_observation import _clearance
        from world_model.wildscenes_runtime import pool_received_tile

        # Frozen native 16x16 ground sensor in v9-km-test-000's final stalled pose.
        car = np.array([1225.5953636040017, 439.39900436198116])
        occupied_prefix_per_row = [0, 0, 0, 0, 0, 0, 0, 1, 3, 4, 4, 5, 5, 5, 6, 5]
        tile = (np.arange(16)[None, :] < np.array(occupied_prefix_per_row)[:, None]).astype(np.int8)
        origin = np.floor(car) - 10
        cells = np.zeros((21, 21), np.int8)
        ids, values = pool_received_tile(tile, car, .5, 1., cells.shape, origin,
                                         occupied_overrides_unknown=True)
        cells.flat[ids] = values
        native_coarse = _clearance(cells)
        self.assertNotEqual(native_coarse[10, 10], -1)
        cell, path = certified_start_connector(tile, car, .5, car, native_coarse, origin, min_distance_m=1.)
        self.assertEqual(native_coarse[tuple(cell)], -1)
        np.testing.assert_array_equal(path, np.stack((car, [1226.5, 438.5])))
        self.assertGreater(np.linalg.norm(path[1] - car), 1.)

    def test_cell_containing_vehicle_can_fail_while_another_certified_center_is_reachable(self):
        car = np.array([10., 10.])
        tile = np.zeros((16, 16), np.int8)  # raw tile covers [6,14] x [6,14]
        tile[10, 11] = 1  # known obstacle rectangle [11.5,12] x [11,11.5]
        grid = np.zeros((20, 20), np.int8)
        grid[9, 9] = -1
        self.assertTrue(native_segment_clear(tile, car, .5, car, car))
        self.assertFalse(native_segment_clear(tile, car, .5, car, [10.5, 10.5]))
        cell, path = certified_start_connector(tile, car, .5, car, grid, [0., 0.])
        np.testing.assert_array_equal(cell, [9, 9])
        np.testing.assert_array_equal(path, [[10., 10.], [9.5, 9.5]])
        # Even when the host coarse cell is flagged free, its native sweep fails.
        grid[10, 10] = -1
        cell, _ = certified_start_connector(tile, car, .5, car, grid, [0., 0.])
        np.testing.assert_array_equal(cell, [9, 9])
        grid[9, 9] = 0
        self.assertIsNone(certified_start_connector(tile, car, .5, car, grid, [0., 0.]))

    def test_obstacle_and_unknown_tangent_to_interior_of_segment_block_departure(self):
        # Endpoints each have full clearance; the interior just touches this pixel.
        start, end, center = [4., 6.], [8., 6.], [6., 6.]
        for obstacle in (-1, 1):
            with self.subTest(obstacle=obstacle):
                tile = np.zeros((16, 16), np.int8)
                tile[11, 8] = obstacle  # x in [6,6.5], y in [7.5,8]
                self.assertTrue(native_segment_clear(tile, center, .5, start, start))
                self.assertTrue(native_segment_clear(tile, center, .5, end, end))
                self.assertFalse(native_segment_clear(tile, center, .5, start, end))
                self.assertTrue(native_segment_clear(tile, center, .5, [4., 5.99], [8., 5.99]))

    def test_unknown_at_car_or_missing_coverage_keeps_car_stationary(self):
        tile = np.zeros((16, 16), np.int8)
        grid = -np.ones((22, 22), np.int8)
        # The car is 1m from the coarse world boundary. A simulated local sensor
        # may extend beyond that boundary, but cannot authorize moving outside.
        self.assertIsNone(certified_start_connector(tile, [1., 8.], .5, [1., 8.], grid, [0., 0.]))
        self.assertFalse(native_segment_clear(tile, [8., 8.], .5, [8., 8.], [10.6, 8.]))
        tile[8, 8] = -1
        self.assertIsNone(certified_start_connector(tile, [8., 8.], .5, [8., 8.], grid, [0., 0.]))
        self.assertIsNone(certified_start_connector(None, [8., 8.], .5, [8., 8.], grid, [0., 0.]))

    def test_bypass_reached_threshold_uses_only_known_free_certified_centres(self):
        # Deliberately use a nonzero crop origin and an asymmetric native tile.
        car = np.array([110.5, 70.5])
        tile = np.zeros((16, 24), np.int8)
        grid = np.zeros((20, 20), np.int8)
        grid[10, [10, 11, 12]] = -1
        args = (tile, car, .5, car, grid, [100., 60.])
        cell, _ = certified_start_connector(*args)
        np.testing.assert_array_equal(cell, [10, 10])
        cell, path = certified_start_connector(*args, min_distance_m=1.)
        np.testing.assert_array_equal(cell, [10, 12])
        np.testing.assert_array_equal(path[-1], [112.5, 70.5])
        self.assertIsNone(certified_start_connector(*args, min_distance_m=2.))

    def test_random_sweeps_agree_with_independent_convex_distance_minimization(self):
        rng = np.random.default_rng(98124)
        for _ in range(400):
            start, end, lower = rng.uniform(-2., 2., size=(3, 2))
            side, radius = rng.uniform(.05, 1.5, size=2)
            def distance_squared(t):
                point = start + t * (end - start)
                delta = point - np.clip(point, lower, lower + side)
                return float(delta @ delta)
            interior = minimize_scalar(distance_squared, bounds=(0., 1.), method='bounded')
            reference = min(interior.fun, distance_squared(0.), distance_squared(1.))
            self.assertEqual(_sweep_hits_boxes(start, end, lower[None, :], side, radius),
                             reference <= radius ** 2)
