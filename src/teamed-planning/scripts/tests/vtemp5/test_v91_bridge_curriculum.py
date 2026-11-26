"""Physical reset, split isolation, path-based crossing and reward-side oracle tests."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS/'vtemp5'))
sys.path.insert(0, str(SCRIPTS))
import v91_bridge_curriculum as bridge
from shared.simulator.wildscenes.wildscenes_map import RasterForestMap
from v91_km_cases import km_route_metrics


class BridgeCurriculumTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (bridge.REPO/'artifacts'/bridge.SUITE/'scenes.json').is_file():
            raise unittest.SkipTest('Archived river-valley suite is unavailable')
        cls.root, cls.table = bridge._archive(str(bridge.REPO))

    def test_counterfactual_family_leak_detected(self):
        for key in ('seed', 'terrain_family_id', 'map_sha256'):
            data = copy.deepcopy(self.table)
            data['development'][0][key] = data['ppo_train'][1][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                bridge._check_splits(data)

    def test_four_balanced_resets_use_exact_frozen_formation_and_native_clearance(self):
        scenes = bridge.curriculum_cases(bridge.REPO, 'ppo_train', 'mouth', 4, 19)
        self.assertEqual({(s['curriculum']['reverse'], s['curriculum']['offset_side']) for s in scenes},
                         {(False,-1), (False,1), (True,-1), (True,1)})
        self.assertEqual(scenes[0], bridge.bridge_curriculum_scene('mouth', 0, seed=19))
        for s in scenes:
            with self.subTest(case=s['case_id']):
                info = s['curriculum']
                self.assertEqual(np.sign(info['signed_lateral_m']), info['offset_side'])
                self.assertIn(s['seed'], {row['seed'] for row in self.table['ppo_train']})
                self.assertNotIn(s['seed'], {row['seed'] for row in self.table['development']+self.table['test']})
                self.assertEqual(s['max_steps'], int(np.ceil(6*s['reference_distance_m'] + 120)))
                self.assertLess(s['direct_distance_m'], 200.)
                self.assertGreaterEqual(s['reference_distance_m'] + 1e-5, s['direct_distance_m'])
                np.testing.assert_allclose(info['fleet_spawn_xy'], np.array(s['start']) + [[9,-18],[9,0],[9,18]])
                forest = RasterForestMap.from_npz(s['case']['map_path'])
                for point in [s['start'], s['goal'], *info['fleet_spawn_xy']]:
                    self.assertTrue(forest.point_clear(point, 1.5))
                with np.load(Path(s['case']['map_path']).with_name('landscape.npz')) as data:
                    canopy = data['canopy_height_m']
                for x,y in np.floor(info['fleet_spawn_xy']).astype(int):
                    self.assertLess(float(np.max(canopy[y-2:y+3, x-2:x+3])), 13.5)

    def test_single_deck_events_reject_water_shortcut_and_reverse_abort(self):
        scene = next(bridge.bridge_curriculum_scene('kilometre', i, seed=3) for i in range(18)
                     if self.table['ppo_train'][np.random.default_rng(3).permutation(18)[i]]['navigation_task'] != 'direct')
        self.assertTrue(1000. <= scene['direct_distance_m'] <= 1500.)
        path = np.asarray(json.loads(Path(scene['audit_path']).read_text())['reference_route_xy'])
        score = bridge.bridge_score(scene, path)
        self.assertTrue(score['reached_deck'] and score['crossed_bridge'] and score['goal_reached'])
        self.assertTrue(km_route_metrics(scene, path)['crossed_bridge'])
        self.assertLessEqual(score['first_deck_step'], score['first_cross_step'])
        # Interpolate a real point on the deck. An out-and-back trajectory cannot cross.
        _, _, reg, _, _ = bridge._terrain(scene['case']['map_path'], scene['case']['map_sha256'])
        i = score['first_deck_step']
        cells = bridge._raster_cells(path[i-1:i+1])
        deck = cells[reg[cells[:,1],cells[:,0]] == 0][0] + .5
        partial = np.vstack((path[:i], deck, path[:i][::-1]))
        back = bridge.bridge_score(scene, partial)
        self.assertTrue(back['reached_deck'])
        self.assertFalse(back['crossed_bridge'])
        self.assertIsNone(back['first_cross_step'])
        shortcut = bridge.bridge_score(scene, path[[0,-1]])
        self.assertFalse(shortcut['crossed_bridge'])
        self.assertIsNone(shortcut['first_deck_step'])
        potential = bridge.bridge_potential(scene, path[[0,-1]])
        np.testing.assert_allclose(potential, [-scene['reference_distance_m'], 0], atol=.005)
        self.assertGreater(-potential[0] - scene['direct_distance_m'], 100.)

    def test_projection_at_mask_edge_stays_on_same_bank_and_finite(self):
        regions = np.array([[1,1,-1,2,2,2], [1,1,0,2,2,2], [1,1,-1,2,2,2]])
        field = np.full(regions.shape, np.inf)
        field[0,0], field[1,3] = 102., 1.
        scene = dict(case=dict(map_path='unused', map_sha256='unused'), goal=[3.5,1.5])
        with patch.object(bridge, '_terrain', return_value=(None,None,regions,None,None)), \
             patch.object(bridge, '_potential_field', return_value=(field, 200.)):
            d = bridge.bridge_potential(scene, [[1.5,1.5],[4.5,1.5],[2.5,2.5],[10.,1.]])
        self.assertTrue(np.isfinite(d).all())
        self.assertLess(d[0], -102.)
        self.assertAlmostEqual(d[1], -2.)
        np.testing.assert_array_equal(d[2:], [-200.,-200.])


if __name__ == '__main__':
    unittest.main()
