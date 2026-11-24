"""Small metric mazes exercise task selection without scaling their geometry."""
import unittest

import numpy as np
from scipy.sparse.csgraph import dijkstra

from shared.simulator.select_wildscenes_cases import grid_graph, planning_mask, select_cases
from shared.simulator.wildscenes_map import RasterForestMap


class SelectCasesTests(unittest.TestCase):
    def maze(self):
        occupied = np.zeros((36, 44), bool)
        occupied[0, :] = occupied[-1, :] = True
        occupied[:, 0] = occupied[:, -1] = True
        occupied[1:29, 14:16] = True
        occupied[7:35, 28:30] = True
        known = np.ones_like(occupied)
        # A map hole must not become a traversable shortcut.
        known[12:15, 18:23] = False
        return RasterForestMap(occupied, known, .5, origin_xy=(20., 20.))

    def test_diagonal_corner_cut_is_forbidden(self):
        free = np.array([[True, False], [False, True]])
        graph, _ = grid_graph(free, 1.)
        self.assertFalse(np.isfinite(dijkstra(graph, indices=0)[1]))

    def test_metric_routes_are_exactly_safe_and_reverse_is_same_map(self):
        forest = self.maze()
        result = select_cases(forest, 'TEST', [12., 16.], clearance=.3,
                              formation_margin=0., max_cases=3)
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(len(result['cases']), 3)
        forward, reverse, longer = result['cases']
        self.assertEqual(forward['start'], reverse['goal'])
        self.assertEqual(forward['goal'], reverse['start'])
        self.assertEqual(forward['reference_points'], reverse['reference_points'][::-1])
        self.assertFalse(result['metadata']['forward_reverse_are_independent_maps'])
        for case in result['cases']:
            path = np.asarray(case['reference_points'])
            measured = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            self.assertAlmostEqual(measured, case['reference_length_m'])
            self.assertGreaterEqual(measured, case['requested_length_m'] - 1e-8)
            self.assertGreaterEqual(measured, case['straight_distance_m'] - 1e-8)
            self.assertTrue(all(forest.segment_clear(a, b, .3) for a, b in zip(path, path[1:])))
            self.assertTrue(forest.known_at(path).all())
            self.assertTrue(forest.point_clear(case['start'], 1.))
            self.assertTrue(forest.point_clear(case['goal'], 1.))
        self.assertGreater(longer['reference_length_m'], forward['reference_length_m'])

    def test_insufficient_length_is_reported_without_scaling_or_fallback(self):
        forest = self.maze()
        result = select_cases(forest, 'TEST', [1000.], clearance=1.5, formation_margin=0.)
        self.assertEqual(result['cases'], [])
        self.assertEqual(result['status'], 'insufficient_connected_length')
        self.assertLess(result['metadata']['maximum_found_reference_length_m'], 1000.)
        self.assertEqual(result['metadata']['clearance_m'], 1.5)
        self.assertFalse(result['metadata']['clearance_automatically_reduced'])
        self.assertEqual(forest.resolution_m, .5)

    def test_conservative_center_clearance_and_coarse_grid(self):
        forest = self.maze()
        safe, resolution, x, y = planning_mask(forest, .3, planning_resolution=1.)
        self.assertEqual(resolution, 1.)
        rows, cols = np.nonzero(safe)
        self.assertTrue(all(forest.point_clear([x[c], y[r]], .3) for r, c in zip(rows, cols)))

    def test_endpoint_clearance_cannot_be_automatically_relaxed(self):
        occupied = np.ones((20, 80), bool)
        occupied[8:12, 2:78] = False
        forest = RasterForestMap(occupied, np.ones_like(occupied), .5)
        result = select_cases(forest, 'NARROW', [10.], clearance=.3, formation_margin=0.)
        self.assertEqual(result['cases'], [])
        self.assertEqual(result['metadata']['endpoint_clearance_m'], 1.)
        self.assertIn('endpoint', result['metadata']['reason'])
        explicit = select_cases(forest, 'NARROW', [10.], clearance=.3, formation_margin=0., endpoint_clearance=.3)
        self.assertEqual(len(explicit['cases']), 2)

    def test_default_route_and_endpoint_clearance_are_one_metre(self):
        occupied = np.zeros((30, 60), bool)
        forest = RasterForestMap(occupied, np.ones_like(occupied), .5)
        result = select_cases(forest, 'DEFAULT', [10.], formation_margin=0.)
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['metadata']['clearance_m'], 1.)
        self.assertEqual(result['metadata']['endpoint_clearance_m'], 1.)
        for case in result['cases']:
            self.assertTrue(all(forest.segment_clear(a, b, 1.)
                for a, b in zip(case['reference_points'], case['reference_points'][1:])))
        replay = select_cases(forest, 'REPLAY', [10.], clearance=1.5, formation_margin=0.)
        self.assertEqual(replay['metadata']['clearance_m'], 1.5)
        self.assertEqual(replay['metadata']['endpoint_clearance_m'], 1.5)

    def test_smaller_component_can_supply_the_longer_route(self):
        occupied = np.ones((100, 210), bool)
        occupied[5:45, 5:45] = False  # More area, but short diameter.
        occupied[60:70, 65:205] = False  # Less area, much longer corridor.
        forest = RasterForestMap(occupied, np.ones_like(occupied), .5)
        result = select_cases(forest, 'TWO', [50.], clearance=.3, formation_margin=0.)
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(len(result['cases']), 2)
        self.assertEqual(result['metadata']['selected_component_size_rank'], 2)
        self.assertGreaterEqual(result['metadata']['components_examined'], 2)
        self.assertGreaterEqual(result['cases'][0]['reference_length_m'], 50.)


if __name__ == '__main__':
    unittest.main()
