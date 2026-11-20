"""Next-frame guard uses the measured XY-oriented 0.5 m raster."""
import unittest

import numpy as np

from ground.survey_fleet_clearance import native_segment_clear
from simulator.survey_fleet_phase import endpoint_phase_clear


class PhaseClearanceTests(unittest.TestCase):
    def test_subpixel_move_toward_obstacle_can_consume_next_frame_margin(self):
        tile = np.zeros((16, 16), np.int8)
        tile[8, 12] = 1
        start, end = np.array([4., 4.]), np.array([4.3, 4.])
        self.assertTrue(native_segment_clear(tile, start, .5, start, end))
        self.assertFalse(endpoint_phase_clear(tile, start, .5, start, end))
        self.assertTrue(endpoint_phase_clear(tile, start, .5, start, [3.7, 4.]))

    def test_translation_parallel_to_narrow_walls_has_no_lateral_padding(self):
        tile = np.zeros((16, 16), np.int8)
        tile[:4] = tile[12:] = 1
        self.assertTrue(endpoint_phase_clear(tile, [4., 4.], .5, [4., 4.], [4.69, 4.]))
        self.assertFalse(endpoint_phase_clear(tile, [4., 4.], .5, [4., 4.], [4.69, 4.1]))

    def test_no_motion_out_of_insufficient_current_clearance(self):
        tile = np.zeros((16, 16), np.int8)
        tile[8, 10] = -1
        car = [4., 4.]
        self.assertFalse(endpoint_phase_clear(tile, car, .5, car, [3., 4.]))
        self.assertFalse(endpoint_phase_clear(tile, car, .5, car, car))

    def test_accepted_translations_remain_safe_in_rerasterized_random_static_scenes(self):
        rng = np.random.default_rng(17722)
        rows, cols = np.indices((16, 16))
        accepted = 0

        def measure(car, circles):
            mask = np.zeros((16, 16), bool)
            for cx, cy, radius in circles:
                px, py = ((np.array([cx, cy]) - car + 4.) / .5).astype(int)
                # Match the archived simulator's circle-centre and radius quantizers.
                mask |= (cols - px) ** 2 + (rows - py) ** 2 <= (int(radius / .5) + 1) ** 2
            return mask.astype(np.int8)

        for _ in range(700):
            car = rng.uniform(10., 30., size=2)
            circles = np.column_stack((car + rng.uniform(-9., 9., (4, 2)), rng.uniform(.1, 4., 4)))
            move = rng.uniform(-1., 1., size=2)
            tile = measure(car, circles)
            if endpoint_phase_clear(tile, car, .5, car, car + move):
                target = car + move
                next_tile = measure(target, circles)
                self.assertTrue(native_segment_clear(next_tile, target, .5, target, target))
                accepted += 1
        self.assertGreater(accepted, 100)
