from unittest import TestCase

import numpy as np

from vtemp4.vtemp4_route_distance import RouteDistanceField, _grid_distances


class RouteDistanceTests(TestCase):
    @staticmethod
    def field(circles, *, goal=(48., 24.), world=60.):
        return RouteDistanceField(dict(goal=goal, world_size=world, circles=circles))

    def test_open_world_uses_metric_distance_including_subcell_positions(self):
        field = self.field([], goal=(30.25, 25.75))
        self.assertEqual(field.distance_m((27.25, 21.75)), 5.)
        self.assertEqual(field.distance_m(field.goal), 0.)
        self.assertIsNone(field.distance_m((1., 12.)))

    def test_detour_awards_actual_progress_even_while_direct_distance_grows(self):
        # The only way past the vertical wall is to its north.
        wall = [([30., y], 2.) for y in range(0, 39, 3)]
        field = self.field(wall)
        initial, around = (24., 24.), (24., 32.)
        self.assertGreater(np.linalg.norm(field.goal - around), np.linalg.norm(field.goal - initial))
        self.assertGreater(field.distance_m(initial) - field.distance_m(around), 5.)
        self.assertEqual(field.distance_m(initial) - field.distance_m(initial), 0.)
        self.assertEqual(field.distance_m(field.goal), 0.)

    def test_disconnected_regions_and_clearance_edges(self):
        # The two obstacle rows leave a gap for a point, but not for the vehicle.
        wall = [([30., y], 2.) for y in range(0, 25, 3)] + [([30., y], 2.) for y in range(30, 61, 3)]
        field = self.field(wall)
        self.assertIsNone(field.distance_m((20., 27.)))
        self.assertIsNone(field.distance_m((30., 27.)))
        self.assertIsNone(field.distance_m((33.49, 24.)))
        # This legal pose is closer to the circle than the conservative cell padding.
        self.assertIsNotNone(field.distance_m((33.51, 24.)))
        self.assertIsNotNone(field.distance_m((34.15, 24.25)))

    def test_diagonal_requires_both_flanking_cells(self):
        values = _grid_distances(np.eye(2, dtype=bool), 1., np.array([3]), np.array([0.]))
        self.assertTrue(np.isinf(values[0, 0]))
        full = _grid_distances(np.ones((2, 2), bool), 1., np.array([3]), np.array([0.]))
        self.assertAlmostEqual(full[0, 0], np.sqrt(2.))

    def test_routed_potential_keeps_metric_progress_at_fractional_positions(self):
        field = self.field([([12., 40.], 1.)], goal=(41.5, 20.5))
        self.assertAlmostEqual(field.distance_m((30.75, 20.5)), 10.75)
        self.assertAlmostEqual(field.distance_m((31., 20.5)), 10.5)
        self.assertAlmostEqual(field.distance_m((40.5, 20.5)), 1.)
