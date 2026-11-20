"""Disjoint metric-distance curriculum and repeatable V9 obstacle materialization."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator.survey_km_maps import GENERATOR, WORLD_M, _case_band
from simulator.survey_pretrain_maps import load_survey_map, make_km_far_test, make_km_splits

REPO = Path(__file__).resolve().parents[5]


class KilometreSurveyMapTests(unittest.TestCase):
    def test_train_curriculum_and_heldout_stratification(self):
        self.assertEqual(_case_band('warmup', 0, 1), (600., 900.))
        self.assertEqual([_case_band('ppo_train', i, 4) for i in range(4)],
                         [(1000., 1200.)]*2 + [(1200., 1500.)]*2)
        for split in ('development','test'):
            self.assertEqual([_case_band(split,i,2) for i in range(2)],
                             [(1000.,1200.), (1200.,1500.)])

    @unittest.skipUnless((REPO / GENERATOR).is_file(), 'Archived V9 map generator unavailable')
    def test_fixed_seed_metric_maps_resume_and_reject_mismatched_request(self):
        args = dict(seed=714915, warmup_cases=1, ppo_cases=1, dev_cases=1, test_cases=1, workers=2)
        with tempfile.TemporaryDirectory() as folder:
            splits = make_km_splits(REPO, folder, **args)
            rows = [row for group in splits.values() for row in group]
            self.assertEqual(len({r['seed'] for r in rows}),4)
            self.assertEqual(len({r['circles_sha256'] for r in rows}),4)
            for row in rows:
                grid=load_survey_map(row)
                dist=np.linalg.norm(grid.goal-grid.start)
                self.assertAlmostEqual(row['direct_distance_m'],dist)
                self.assertGreaterEqual(dist,row['distance_band_m'][0])
                self.assertLessEqual(dist,row['distance_band_m'][1])
                self.assertEqual(row['circle_count'],round(22/(80.*80.)*WORLD_M**2))
                coords=np.floor((np.stack((grid.start,grid.goal))-grid.origin)/grid.resolution).astype(int)
                a,b=label(grid.free)[0][coords[:,1],coords[:,0]]
                self.assertTrue(a>0 and a==b)
                self.assertNotIn('references',row)
            self.assertEqual(make_km_splits(REPO,folder,**args),splits)
            original_manifest = (Path(folder)/'scenes.json').read_bytes()
            far = make_km_far_test(REPO, folder, seed=args['seed'], existing=splits, workers=2)
            self.assertEqual(len(far), 2)
            self.assertTrue(all(1400. <= r['direct_distance_m'] <= 1500. for r in far))
            self.assertFalse({r['seed'] for r in rows} & {r['seed'] for r in far})
            self.assertEqual((Path(folder)/'scenes.json').read_bytes(), original_manifest)
            audit=json.loads((Path(folder)/'audit.json').read_text())
            self.assertEqual(len(audit['generator_sha256']),64)
            with self.assertRaisesRegex(ValueError, 'Different kilometre data request'):
                make_km_splits(REPO,folder,**dict(args,seed=args['seed']+1))


if __name__ == '__main__':
    unittest.main()
