"""The new vtmp2 uses actual v9 simulation on isolated simplified terrain crops."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from tests.world_model.vtmp3.test_wildscenes import fixture
from tests.world_model.vtmp2 import test_decoupled as fixtures
from world_model.ab_environment import MappingConfig
from world_model.ab_wildscenes import WildScenesSource, freeze_manifest
from world_model.decoupled_observation import crop_origin
from world_model.decoupled_pipeline import terrain_labels
from world_model.decoupled_scenes import generate_scene, load_scene, geometry_for
from world_model.decoupled_training import load_dataset


class WildScenesTests(unittest.TestCase):
    def test_frozen_metric_geometry_certification_and_occupancy_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original, manifest = fixture(root)
            frozen = freeze_manifest(manifest, root/'terrain')
            provider = WildScenesSource(frozen, MappingConfig(world=600., clearance=1.5))
            record = generate_scene(provider, 3, 'train', root)
            original.write_text('original source changed after snapshot')
            scene, certificate = load_scene(record['scene_path'], record['scene_sha256'])
            self.assertTrue(certificate['valid'])
            self.assertEqual(scene['source_unknown_space'], 'free_by_simulation_assumption')
            geometry = geometry_for(scene)
            self.assertFalse(geometry.segment_clear((240., 250.), (260., 250.), 1.5))
            labels = terrain_labels(scene, crop_origin((250., 250.)))
            self.assertEqual(labels.shape, (64, 64))
            self.assertTrue(np.isin(labels, (0., 1.)).all())

    def test_tampered_reference_is_rechecked_independently(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); _, manifest = fixture(root)
            provider = WildScenesSource(manifest, MappingConfig(world=600., clearance=1.5))
            record = generate_scene(provider, 3, 'train', root)
            path = Path(record['scene_path'])
            document = json.loads(path.read_text())
            document['scene']['reference_points'][1] = [250., 250.]
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'clearance'):
                load_scene(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_evaluation_uses_test_region_even_when_file_group_is_train(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); _, manifest = fixture(root)
            provider = WildScenesSource(manifest, MappingConfig(world=600., clearance=1.5))
            record = generate_scene(provider, 3, 'train', root, region_split='test')
            scene, _ = load_scene(record['scene_path'], record['scene_sha256'])
            self.assertEqual(scene['split'], 'train')
            self.assertEqual(scene['terrain_split'], 'test')
            self.assertEqual(scene['terrain_origin_xy'], [0., 600.])

    def test_dataset_rejects_same_physical_region_with_distinct_seeds_and_ids(self):
        helper = fixtures.CollectionTrainingTests()
        with TemporaryDirectory() as directory:
            for split, seed, digest in (('train', 11, 'a'*64), ('val', 12, 'b'*64)):
                _, data = helper.collect(directory, split, seed, fixtures.FakeEnvironment(geometry=digest))
                data.update(terrain_backend=np.asarray('wildscenes_simplified'),
                    terrain_source_sha256=np.asarray('c'*64), terrain_origin_xy=np.zeros(2),
                    terrain_map_id=np.asarray('K-01'), terrain_region_id=np.asarray(str(seed)),
                    terrain_split=np.asarray(split), terrain_world_m=np.asarray(600.))
                np.savez_compressed(Path(directory)/f'{split}_{seed}.npz', **data)
            with self.assertRaisesRegex(ValueError, 'regions overlap'):
                load_dataset(directory)


if __name__ == '__main__':
    unittest.main()
