"""Simplified scenes preserve source occupancy and explicitly change support."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from world_model.simplified_wildscenes_tasks import restore_map


class SimplifiedMapTests(unittest.TestCase):
    def source(self, root):
        path = root / 'source.npz'
        occupied = np.zeros((8, 8), bool); occupied[2:4, 1:3] = True
        known = occupied.copy(); known[0, 0] = True
        np.savez_compressed(path, occupied=occupied, known=known, resolution_m=np.array(.5), origin_xy=np.zeros(2))
        scene = dict(protocol='wildscenes-derived-legacy-2d-v1', simplified_scene=True,
            coordinate_scale=1., unknown_space='free_by_simulation_assumption',
            world_size_m=4., origin_xy=[0., 0.],
            source=dict(map_path=str(path), map_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        resolution_m=.5, shape_yx=[8, 8], metadata=dict(sequence='K-01')),
            geometry=dict(exact_source_occupied=True, rectangles=[[1., 1.5, 1., 1.]]),
            assumptions=['Unmeasured source space is simulation free, not confirmed real ground.'])
        scene_path = root / 'scene.json'; scene_path.write_text(json.dumps(scene))
        return scene_path, occupied, known

    def test_exact_obstacle_cells_and_explicit_all_known_simulation_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, occupied, known = self.source(root)
            contract = restore_map(scene, root / 'prepared')
            with np.load(contract['map_path'], allow_pickle=False) as archive:
                self.assertTrue(np.array_equal(archive['occupied'], occupied))
                self.assertTrue(archive['known'].all())
                self.assertEqual(str(archive['terrain_protocol']), 'wildscenes_simplified_raster')
                self.assertEqual(str(archive['source_scene_sha256']), hashlib.sha256(scene.read_bytes()).hexdigest())
                self.assertNotIn('terrain', archive.files)
            metadata = json.loads(Path(contract['metadata_path']).read_text())
            self.assertEqual(metadata['occupied_changed_cells'], 0)
            self.assertEqual(metadata['original_source_unknown_cells'], int((~known).sum()))
            self.assertEqual(contract['source_unknown_space'], 'free_by_simulation_assumption')
            self.assertEqual(contract['coordinate_scale'], 1.)
            with self.assertRaises(FileExistsError):
                restore_map(scene, root / 'prepared')

    def test_wrong_rectangles_and_unlabelled_unknown_assumptions_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, _, _ = self.source(root)
            document = json.loads(scene.read_text())
            document['geometry']['rectangles'][0][0] += .5
            scene.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'changed original occupied'):
                restore_map(scene, root / 'bad-geometry')
            document['unknown_space'] = 'blocked'; scene.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'explicitly simplified'):
                restore_map(scene, root / 'bad-protocol')


if __name__ == '__main__':
    unittest.main(verbosity=2)
