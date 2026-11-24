import math
import unittest

import numpy as np

from vtemp4.vtemp4_curriculum import runtime_case
from vtemp5.vtemp5_circle_curriculum import _side_distances, circle_schedule


class CircleCurriculumTests(unittest.TestCase):
    def test_native_missions_block_the_centerline_and_fit_both_routes(self):
        for plan in circle_schedule('ppo_train', 40, 19):
            case = plan['case']
            native = runtime_case(case)
            self.assertNotIn('side_reference_distance_m', native)
            self.assertEqual(len(native['circles']), 1)
            point, radius = native['circles'][0]
            start, goal, center = map(np.asarray, (native['start'], native['goal'], point))
            distance = np.linalg.norm(goal - start)
            forward = (goal - start) / distance
            station = (center - start) @ forward
            offset = (center - start) @ [-forward[1], forward[0]]
            self.assertGreaterEqual(distance, 1000.)
            self.assertLess(abs(offset), radius)
            entry = station - math.sqrt(radius**2 - offset**2)
            self.assertAlmostEqual(case['barrier_distance_m'], entry)
            self.assertGreater(entry, 0.)
            self.assertLess(station + radius, distance)
            safe_radius = radius + native['required_clearance_m']
            for point in (center - safe_radius, center + safe_radius, start, goal):
                self.assertTrue((point >= native['required_clearance_m']).all())
                self.assertTrue((point <= native['world_size'] - native['required_clearance_m']).all())
            lengths = case['side_reference_distance_m']
            self.assertEqual(set(lengths), {'left', 'right'})
            self.assertEqual(lengths['left'] > lengths['right'], offset > 0)
            self.assertGreater(case['reference_distance_m'], distance)
            self.assertEqual(case['reference_distance_m'], min(lengths.values()))

    def test_split_seeds_are_disjoint_and_paired_maps_reproducible(self):
        train = circle_schedule('ppo_train', 32, 123)
        dev = circle_schedule('development', 16, 123)
        self.assertEqual(dev, circle_schedule('development', 40, 123)[:16])
        self.assertEqual(dev[0]['case']['barrier_distance_m'], 26.)
        for field in ('case_id', 'map_sha256', 'seed'):
            train_ids = {p['case'][field] for p in train}
            self.assertEqual(len(train_ids), len(train))
            self.assertTrue(train_ids.isdisjoint(p['case'][field] for p in dev))

    def test_reference_distance_for_a_centered_circle(self):
        side = _side_distances(1000., 500., 0., 100.)
        expected = 2 * math.sqrt(500.**2 - 100.**2) + 200. * math.asin(.2)
        for length in side.values():
            self.assertAlmostEqual(length, expected)
