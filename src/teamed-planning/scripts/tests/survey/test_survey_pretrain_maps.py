"""Metric input and offline split checks for lightweight survey rollouts."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vtemp2_3.hierarchical_sources import file_sha256
from survey.survey_pretrain_maps import ARCHIVE, load_splits, load_survey_map
from vtemp2_3.wildscenes_runtime import load_forest


REPO = Path(__file__).resolve().parents[5]


class SurveyMapTests(unittest.TestCase):
    def test_circle_raster_includes_clearance_and_keeps_metric_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            geometry = Path(directory) / 'circles.npy'
            np.save(geometry, np.array([[125., 103., 2.]]))
            case = dict(case_id='test-v9', kind='v9_random', terrain='synthetic', world_size=320.,
                        circles_path=str(geometry), circle_count=1, circles_sha256=file_sha256(geometry),
                        start=[100.25, 100.75], goal=[150.6, 100.1])
            scene = load_survey_map(case)
        np.testing.assert_equal(scene.start, case['start'])
        np.testing.assert_equal(scene.goal, case['goal'])
        def free(point):
            x, y = ((np.array(point) - scene.origin) // scene.resolution).astype(int)
            return scene.free[y, x]
        self.assertFalse(free([125.5, 106.5]))
        self.assertTrue(free([125.5, 107.5]))

    @unittest.skipUnless((REPO / ARCHIVE / 'scenes.json').is_file(), 'Historical scenes not installed')
    def test_all_original_venman_endpoints_and_native_clearance(self):
        training, validation = load_splits(REPO)
        self.assertEqual({row['kind'] for row in training}, {'v9_random'})
        self.assertEqual([r['case_id'] for r in validation],
                         [f'venman-strict-obstacle-{index:02}' for index in range(1, 19)])
        forest = load_forest(validation[0], REPO)
        for case in validation:
            scene = load_survey_map(case)
            self.assertEqual(scene.resolution, .5)
            lo_x, lo_y = (scene.origin / scene.resolution).astype(int)
            height, width = scene.free.shape
            known = forest.known[lo_y:lo_y+height, lo_x:lo_x+width]
            self.assertFalse(np.any(scene.sensor_valid & ~known))
            self.assertTrue(np.all(scene.sensor_valid[scene.free]))
            self.assertTrue(np.array_equal(scene.start, case['start']))
            self.assertTrue(np.array_equal(scene.goal, case['goal']))
            cells = ((np.stack([scene.start, scene.goal]) - scene.origin) // scene.resolution).astype(int)
            connected = label(scene.free)[0]
            first, last = connected[cells[:, 1], cells[:, 0]]
            self.assertGreater(first, 0)
            self.assertEqual(first, last)
            for y, x in np.ndindex(3, 3):
                point = scene.start + [2.5 * x - 2.5, 2.5 * y - 2.5]
                ix, iy = ((point - scene.origin) // scene.resolution).astype(int)
                self.assertEqual(forest.point_clear(point, 1.5), scene.free[iy, ix])

    @unittest.skipUnless((REPO / ARCHIVE / 'scenes.json').is_file(), 'Historical scenes not installed')
    def test_unmeasured_venman_pixels_stay_unknown_after_survey_delivery(self):
        from survey.survey_pretrain_env import SurveyEnv

        _, validation = load_splits(REPO)
        case = validation[0]
        scene = load_survey_map(case)
        env = SurveyEnv(**scene.as_kwargs(), seed=case['seed'], max_steps=3)
        for _ in range(3):
            env.step(0)
            self.assertFalse(np.any(env.cells[~scene.sensor_valid]))
            self.assertFalse(np.any(np.isfinite(env.source_times[~scene.sensor_valid])))
            if env.done:
                break
