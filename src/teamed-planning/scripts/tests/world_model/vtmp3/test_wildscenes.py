"""WildScenes geometry, evidence isolation, spatial splits and frozen inputs."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from world_model.ab_environment import SurveyNavigationEnv, MappingConfig, make_task
from world_model.ab_pipeline import collect_episode
from world_model.ab_training import load_episodes
from world_model.ab_wildscenes import (PROTOCOL, SCENE_PROTOCOL, TERRAIN, WildScenesSource,
    WildScenesMapper, crop_geometry, digest, freeze_manifest, geometry_for, load_manifest,
    rasterize_rectangles)


def fixture(root):
    scene = root/'source.json'
    scene.write_text(json.dumps(dict(protocol=SCENE_PROTOCOL, simplified_scene=True,
        unknown_space='free_by_simulation_assumption', world_size_m=1200., origin_xy=[0., 0.],
        coordinate_scale=1., geometry=dict(type='axis_aligned_rectangles',
            rectangles=[[250., 250., .25, 200.], [850., 250., .25, 200.], [250., 850., .25, 200.]]))))
    spec = dict(protocol=PROTOCOL, world_size_m=600.,
        **{split: [dict(map_id='fixture', scene='source.json', origin_xy=origin)]
           for split, origin in [('train', [0., 0.]), ('val', [600., 0.]), ('test', [0., 600.])]})
    manifest = root/'manifest.json'
    manifest.write_text(json.dumps(spec))
    return scene, manifest


def config(**kwargs):
    return replace(MappingConfig(world=600., failure_probability=0., distance_failure_probability=0.,
        communication_failure_probability=0., delay_jitter=0.), **kwargs)


class WildScenesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_subpixel_rectangles_and_cell_boundaries_are_not_missed(self):
        occupied = rasterize_rectangles([[4., 5., .01, .01]], 20., 2.)
        self.assertTrue(occupied[2, 1])
        self.assertTrue(occupied[2, 2])
        self.assertEqual(int(occupied.sum()), 2)

    def test_crop_preserves_metres_and_physical_collision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, manifest = fixture(root)
            provider = WildScenesSource(manifest, config())
            scene = provider.scene(3, 'val')
            geometry, occupied = geometry_for(scene, config())
            self.assertEqual(geometry.world_size_m, 600.)
            self.assertTrue(np.array_equal(geometry.rectangles[0], [250., 250., .25, 200.]))
            self.assertFalse(geometry.segment_clear([240., 250.], [260., 250.], 1.))
            self.assertTrue(occupied.any())
            self.assertGreaterEqual(scene['distance'], 100.)
            self.assertLessEqual(scene['distance'], 400.)

    def test_spatial_overlap_is_rejected_even_with_distinct_region_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, manifest = fixture(root)
            spec = json.loads(manifest.read_text())
            spec['val'][0]['origin_xy'] = [599., 0.]
            spec['val'][0]['map_id'] = 'renamed_same_source'
            manifest.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                load_manifest(manifest)

    def test_frozen_sources_survive_original_changes_and_detect_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source, manifest = fixture(root)
            frozen = freeze_manifest(manifest, root/'frozen')
            source.write_text('changed original')
            spec = load_manifest(frozen)
            provider = WildScenesSource(frozen, config())
            provider.scene(1, 'train')
            Path(spec['train'][0]['scene']).write_text('changed frozen source')
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                load_manifest(frozen)

    def test_hidden_geometry_is_not_initial_observation_and_loss_returns_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source, manifest = fixture(root)
            item = WildScenesSource(manifest, config()).scene(7, 'train')
            item.update(start=[31., 101.], goal=[171., 101.])
            env = SurveyNavigationEnv(item, config())
            modified = json.loads(source.read_text())
            modified['geometry']['rectangles'].append([150., 150., 10., 10.])
            other = root/'other.json'; other.write_text(json.dumps(modified))
            changed = dict(item, terrain_path=str(other), terrain_source_sha256=digest(other.read_bytes()))
            other_env = SurveyNavigationEnv(changed, config())
            for first, second in zip(env.observe(), other_env.observe()):
                self.assertTrue(np.array_equal(first, second))
            before = env.received.cells.copy()
            cfg = config(communication_failure_probability=1.)
            _, occupied = geometry_for(item, cfg)
            mapper = WildScenesMapper(occupied, cfg, 9)
            packet = mapper.survey(0, make_task(item['start'], item['goal'], cfg), item['start'])
            self.assertEqual(packet.status, 'communication_failure')
            self.assertEqual(len(packet.cell_ids), 0)
            env.received.ingest(packet)
            self.assertTrue(np.array_equal(before, env.received.cells))

    def test_complete_survey_does_not_drive_through_rectangle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, manifest = fixture(root)
            item = WildScenesSource(manifest, config()).scene(3, 'train')
            item.update(start=[201., 251.], goal=[351., 251.])
            env = SurveyNavigationEnv(item, config())
            for _ in range(4):
                _, _, info = env.step([[201., 251.], [351., 251.]])
                self.assertEqual(info['status'], 'complete')
                self.assertEqual(info['survey_ground_displacement_m'], 0.)
                self.assertLess(env.position[0], 250.)
            self.assertFalse(env.done)

    def test_dataset_retains_backend_and_rejects_spatial_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); _, manifest = fixture(root)
            provider = WildScenesSource(manifest, config())
            for seed, split in [(1, 'train'), (2, 'val')]:
                scene = provider.scene(seed, split)
                data, _, _ = collect_episode(scene, config(), 2, 1000.)
                self.assertEqual(data['backend'].item(), WildScenesMapper.name)
                data['split'] = np.asarray(split)
                if split == 'val':
                    data['terrain_origin_xy'] = np.array([0., 0.])
                np.savez_compressed(root/f'{split}_{seed}.npz', **data)
            with self.assertRaisesRegex(ValueError, 'terrain regions overlap'):
                load_episodes(root)


if __name__ == '__main__':
    unittest.main()
