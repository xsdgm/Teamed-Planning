"""Road-edge history must represent measured, directional passages only."""
import unittest

import numpy as np

from vtemp4.vtemp4_road_traversal import RoadTraversalMemory


class RoadTraversalTest(unittest.TestCase):
    xy = [[0., 0.], [8., 0.], [16., 0.], [100., 100.]]
    adjacent = [[0, 1, -1], [1, 0, 2], [2, 1, 3], [3, 2, -1]]
    valid = [True, True, True, False]

    def test_counts_and_ages_are_directional_causal_and_stable_while_stopped(self):
        path = [[0, 0], [2, 0], [4, 0], [6, 0], [8, 0], [8, 0], [8, 0],
                [6, 0], [4, 0], [2, 0], [0, 0], [2, 0], [4, 0], [6, 0], [8, 0],
                [10, 0], [12, 0], [14, 0], [16, 0]]
        memory = RoadTraversalMemory()

        def read(t):
            # Supply the whole file deliberately: future positions must be ignored.
            return np.expm1(memory.edge_features(t, path[t], path, self.xy, self.adjacent, self.valid))

        self.assertFalse(read(3).any())
        np.testing.assert_allclose(read(4)[0, 1], [1, 0, 0, 0])
        np.testing.assert_allclose(read(6)[0, 1], [1, 0, 2, 0])
        np.testing.assert_allclose(read(6)[1, 1], [0, 1, 0, 2])
        np.testing.assert_allclose(read(10)[0, 1], [1, 1, 6, 0])
        current = read(18)
        np.testing.assert_allclose(current[0, 1], [2, 1, 4, 8])
        np.testing.assert_allclose(current[1, 2], [1, 0, 0, 0])
        self.assertFalse(current[:, 0].any())  # Self loops
        self.assertFalse(current[2, 2].any())  # Invalid destination
        self.assertFalse(current[3].any())     # Invalid source

    def test_crossings_incomplete_passages_and_off_road_detours_are_unmatched(self):
        for path in (
                [[4, -3], [4, -1], [4, 1], [4, 3]],
                [[0, 0], [2, 0], [4, 0], [2, 0], [0, 0]],
                [[0, 0], [2, 0], [4, 5], [6, 0], [8, 0]],
                [[0, 0], [8, 0]]):
            with self.subTest(path=path):
                value = RoadTraversalMemory().edge_features(len(path) - 1, path[-1], path[:-1],
                                                            self.xy, self.adjacent, self.valid)
                self.assertFalse(value.any())

    def test_close_parallel_edges_leave_ambiguous_history_unknown(self):
        path = [[x, 0] for x in range(9)]
        xy = [[0, -.5], [8, -.5], [0, .5], [8, .5]]
        edges = [[1], [0], [3], [2]]
        result = RoadTraversalMemory().edge_features(8, path[-1], path[:-1], xy, edges, [True]*4)
        self.assertFalse(result.any())

    def test_later_observed_edges_can_recover_previous_travel_even_if_rows_reordered(self):
        path = [[x, 0] for x in range(9)]
        memory = RoadTraversalMemory()
        initial = memory.edge_features(8, path[-1], path[:-1], [], np.empty((0, 1)), [])
        self.assertEqual(initial.shape, (0, 1, 4))
        # Start node now has a larger array index than the destination.
        new = memory.edge_features(8, path[-1], path[:-1], [[8, 0], [0, 0]], [[1], [0]], [1, 1])
        np.testing.assert_allclose(np.expm1(new[:, 0, 0]), [0, 1])
        np.testing.assert_allclose(np.expm1(new[:, 0, 1]), [1, 0])


if __name__ == '__main__':
    unittest.main()
