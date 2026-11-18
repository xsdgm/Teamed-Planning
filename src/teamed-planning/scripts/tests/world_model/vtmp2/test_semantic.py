"""Raw-data supervision, hidden-label isolation, and deployable semantic weights."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from simulator.train_world_model import build_command
from world_model.models import make_model
from world_model.semantic_data import (CHANNELS, CLASSES, CLASS_NAMES, FORMAT, NATIVE_IDS,
    atomic_json, crop_patch, encode_observation, load_scene, observation_mask, prepare_sequence, sha256)
from world_model.semantic_model import (ARCHITECTURE, CHECKPOINT_FORMAT, SCHEMA_VERSION,
    SemanticCompletionModel, completion_loss, load_checkpoint)
from world_model import semantic_training as training


def raw_fixture(root, name, boundary=16):
    sequence = root/name
    (sequence/'Clouds').mkdir(parents=True)
    (sequence/'Labels').mkdir()
    xyz = np.array([[x+.25, y+.25, (x+y)/100] for y in range(32) for x in range(32)], dtype='<f4')
    labels = np.where(xyz[:, 0] < boundary, 1, 12).astype('<i4')
    # A source-unlabelled corner must not become free or training supervision.
    labels[(xyz[:, 0] < 4) & (xyz[:, 1] < 4)] = 255
    xyz.tofile(sequence/'Clouds/1.bin')
    labels.tofile(sequence/'Labels/1.label')
    with (sequence/'poses3d.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter=' ')
        writer.writerow(['', 'timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
        writer.writerow(['2026-11-13', 1, 100, -50, 10, 1, 0, 0, 0])
    return sequence


def dataset_fixture(root):
    records = []
    for split, name, boundary in (('train', 'K-test', 16), ('val', 'V-test', 12)):
        sequence = raw_fixture(root/'raw', name, boundary)
        meta = prepare_sequence(sequence, root/'data'/name, resolution=1., minimum_points=1)
        records.append(dict(split=split, sequence=name, site=name.split('-')[0],
            map=str(root/'data'/name/'semantic_map.npz'), sha256=meta['map_sha256']))
    manifest = root/'data/manifest.json'
    atomic_json(manifest, dict(format=FORMAT, resolution_m=1., records=records, simulation_used=False))
    return manifest


class SemanticDataTests(unittest.TestCase):
    def test_original_semantics_pose_and_unlabelled_mask_survive_conversion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            seq = raw_fixture(root, 'K-test')
            meta = prepare_sequence(seq, root/'map', resolution=1., minimum_points=1)
            scene = load_scene(root/'map/semantic_map.npz')
            np.testing.assert_allclose(scene['origin_xy'], (100., -50.))
            self.assertEqual(meta['valid_cells'], 32*32-16)
            self.assertFalse(scene['valid'][0, 0])
            self.assertEqual(scene['probabilities'][NATIVE_IDS.index(1), 8, 8], 1.)
            self.assertEqual(scene['probabilities'][NATIVE_IDS.index(12), 8, 20], 1.)
            self.assertFalse(scene['height_valid'][8, 20])
            self.assertIn('water', meta['unsupported_classes'])
            self.assertFalse(meta['simulation_used'])

    def test_masks_and_encoder_do_not_leak_withheld_semantics_or_height(self):
        valid = np.ones((32, 32), bool); valid[:4, :4] = False
        observed = observation_mask(valid, np.random.default_rng(5))
        self.assertGreater(observed.sum(), 16)
        self.assertGreater((valid & ~observed).sum(), 16)
        p = np.zeros((CLASSES, 32, 32), np.float32); p[1] = 1
        heights = np.ones((32, 32), np.float32)
        first = encode_observation(p, observed, heights, valid)
        p[:, ~observed] = np.nan; heights[~observed] = np.nan
        second = encode_observation(p, observed, heights, valid)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first[:CLASSES, ~observed].sum(), 0.)
        self.assertEqual(first[CLASSES+2, ~observed].sum(), 0.)

    def test_padded_crop_has_no_fabricated_labels(self):
        scene = dict(probabilities=np.ones((CLASSES, 20, 20), np.float32)/CLASSES,
                     valid=np.ones((20, 20), bool), elevation=np.ones((20, 20)), height_valid=np.ones((20, 20), bool))
        cropped = crop_patch(scene, 0, 0, 32)
        self.assertEqual(cropped['valid'].sum(), 400)
        self.assertEqual(cropped['probabilities'][:, 20:].sum(), 0.)

    def test_site_isolation_and_fixed_validation_masks(self):
        with TemporaryDirectory() as directory:
            path = dataset_fixture(Path(directory))
            scenes, spec = training.load_manifest(path)
            dataset = training.SemanticPatches(scenes['val'], side=32, validation=True, masks_per_patch=2, seed=8)
            first = dataset[0]
            dataset.epoch = 10
            np.testing.assert_array_equal(first['maps'], dataset[0]['maps'])
            self.assertFalse(np.any(first['hidden_valid'].astype(bool) & first['maps'][CLASSES].astype(bool)))
            self.assertEqual(first['hidden_valid'][:4, :4].sum(), 0.)
            spec['records'][1]['site'] = 'K'
            atomic_json(path, spec)
            with self.assertRaisesRegex(ValueError, 'site disagree'):
                training.load_manifest(path)


class SemanticModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self):
        maps = torch.zeros(1, CHANNELS, 32, 32)
        maps[:, 1, :, :8] = 1.
        maps[:, CLASSES, :, :8] = 1.
        return maps

    def test_unobserved_inputs_are_ignored_and_received_evidence_stays_authoritative(self):
        model = SemanticCompletionModel(hidden=4, members=2)
        maps = self.inputs()
        original = maps.clone()
        altered = maps.clone(); altered[:, :CLASSES, :, 8:] = 100.
        altered[:, CLASSES+1:, :, 8:] = 10.
        first, second = model.predict(maps), model.predict(altered)
        self.assertTrue(torch.equal(first['semantics'], second['semantics']))
        self.assertTrue(torch.equal(maps, original))
        self.assertTrue(torch.equal(first['semantics'][:, 1, :, :8], torch.ones(1, 32, 8)))
        self.assertEqual(float(first['entropy'][:, :, :8].sum()), 0.)
        self.assertTrue(torch.allclose(first['semantics'].sum(1), torch.ones(1, 32, 32)))

    def test_loss_has_no_gradient_on_observed_or_source_unlabelled_cells(self):
        logits = torch.randn(2, 1, CLASSES, 16, 16, requires_grad=True)
        target = torch.zeros(1, CLASSES, 16, 16); target[:, 1] = 1
        mask = torch.zeros(1, 16, 16); mask[:, 4:12, 4:12] = 1
        loss = completion_loss(logits, target, mask, torch.ones(CLASSES))
        loss.backward()
        self.assertEqual(float(logits.grad[..., :4, :].abs().sum()), 0.)
        self.assertGreater(float(logits.grad[..., 4:12, 4:12].abs().sum()), 0.)

    def test_actual_training_checkpoint_reload_and_version_rejection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = dataset_fixture(root)
            v9 = root/'v9.pt'; v9.write_bytes(b'frozen test actor')
            original_hash = training.sha256
            with patch.object(training, 'sha256', side_effect=lambda p: training.V9_SHA256 if Path(p) == v9 else original_hash(p)):
                result = training.train(manifest, root/'training', epochs=2, side=32, samples_per_epoch=4,
                    validation_masks=2, batch_size=2, hidden=4, members=2, device='cpu', v9_checkpoint=v9)
            self.assertEqual(result['optimizer_steps'], 4)
            self.assertFalse(result['simulation_used'])
            self.assertFalse(result['navigation_validated'])
            model, document = load_checkpoint(result['checkpoint'])
            self.assertEqual(document['model_architecture'], ARCHITECTURE)
            self.assertEqual(document['v9_checkpoint_sha256'], training.V9_SHA256)
            self.assertTrue(torch.isfinite(model.predict(self.inputs())['semantics']).all())
            with (root/'training/metrics.jsonl').open() as stream:
                rows = [json.loads(line) for line in stream]
            self.assertNotEqual(rows[0]['train_loss'], rows[1]['train_loss'])
            self.assertEqual(v9.read_bytes(), b'frozen test actor')
            torch.save(dict(format='vtmp2-decoupled-terrain-v1'), root/'old.pt')
            with self.assertRaisesRegex(ValueError, 'schema-5'):
                load_checkpoint(root/'old.pt')

    def test_class_balancing_preserves_the_target_probability_optimum(self):
        probability = torch.arange(1, CLASSES+1, dtype=torch.float32)
        probability /= probability.sum()
        target = probability[None, :, None, None].expand(1, -1, 16, 16)
        logits = target.log()[None].clone().requires_grad_(True)
        weights = torch.linspace(.2, 5., CLASSES)
        completion_loss(logits, target, torch.ones(1, 16, 16), weights).backward()
        self.assertLess(float(logits.grad.abs().max()), 1e-7)

    def test_legacy_semantic_config_still_dispatches_without_simulation(self):
        config = Path(__file__).resolve().parents[4]/'config/algorithm/world_model_vtmp2_semantic_legacy.json'
        command = build_command(config)
        self.assertEqual(Path(command[3]).name, 'semantic_training.py')
        self.assertNotIn('--terrain', command)
        self.assertIsInstance(make_model(dict(algorithm='vtmp2', model_architecture=ARCHITECTURE, hidden=4, members=1)),
                              SemanticCompletionModel)


if __name__ == '__main__':
    unittest.main()
