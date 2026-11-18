"""Coarse labels, outward masks, probability validity and independent outputs."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import binary_fill_holes
import torch

from simulator.train_world_model import build_command
from tests.world_model.vtmp2.test_semantic import dataset_fixture
from world_model.models import make_model
from world_model.frontier_data import (CHANNELS, CLASSES, CLASS_NAMES, OBSERVED, DISTANCE, PRIOR,
    FrontierPatches, coarsen_scene, encode_observation, outward_mask)
from world_model.frontier_model import ARCHITECTURE, FrontierWorldModel, frontier_loss, load_checkpoint
from world_model import frontier_training as training


def coarse_fixture():
    p = np.zeros((CLASSES, 16, 16), np.float32)
    p[0, :8] = 1.; p[1, 8:] = 1.
    return dict(probabilities=p, valid=np.ones((16, 16), bool), support=np.ones((16, 16), np.float32),
                elevation=np.indices((16, 16))[0].astype(np.float32), height_valid=np.ones((16, 16), bool),
                resolution_m=16.)


class FrontierDataTests(unittest.TestCase):
    def test_coarse_regions_keep_partial_support_and_never_fill_unmeasured_space(self):
        p = np.zeros((14, 8, 8), np.float32)
        valid = np.zeros((8, 8), bool)
        p[11, 0, 0] = 1.; p[3, 0, 1] = 1.; p[9, 0, 2] = 1.
        valid[0, :3] = True
        source = dict(probabilities=p, valid=valid, elevation=np.zeros((8, 8)),
            height_valid=valid.copy(), resolution_m=2., origin_xy=np.array([10., 20.]),
            metadata=dict(sequence='K-test', map_sha256='a'*64))
        result = coarsen_scene(source, cell_size=4., minimum_support=.5)
        self.assertTrue(result['valid'][0, 0])
        np.testing.assert_allclose(result['probabilities'][:, 0, 0], [.5, .5, 0., 0., 0.])
        self.assertEqual(result['support'][0, 0], .5)
        self.assertFalse(result['valid'][0, 1])
        self.assertEqual(result['probabilities'][:, 0, 1].sum(), 0.)
        np.testing.assert_array_equal(result['origin_xy'], source['origin_xy'])

    def test_encoder_never_uses_hidden_semantics_support_or_elevation(self):
        s = coarse_fixture()
        observed = np.zeros((16, 16), bool); observed[:, :8] = True
        prior = np.array([.5, .5, 0., 0., 0.], np.float32)
        first, frontier = encode_observation(s['probabilities'], observed, class_prior=prior,
            support=s['support'], elevation=s['elevation'], height_valid=s['height_valid'])
        s['probabilities'][:, ~observed] = np.nan
        s['support'][~observed] = np.nan
        s['elevation'][~observed] = np.nan
        second, other = encode_observation(s['probabilities'], observed, class_prior=prior,
            support=s['support'], elevation=s['elevation'], height_valid=s['height_valid'])
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(frontier, other)
        self.assertTrue(frontier[:, 8:10].all())
        self.assertFalse(frontier[:, :8].any())
        self.assertFalse(frontier[:, 10:].any())

    def test_outward_mask_has_no_interior_holes_in_fully_measured_data(self):
        for seed in range(10):
            known = outward_mask(np.ones((16, 16), bool), np.random.default_rng(seed), 16., 32.)
            np.testing.assert_array_equal(binary_fill_holes(known), known)
            self.assertGreaterEqual(known.sum(), 6)

    def test_validation_mask_is_fixed_and_does_not_depend_on_hidden_classes(self):
        s = coarse_fixture(); prior = np.array([.5, .5, 0., 0., 0.], np.float32)
        dataset = FrontierPatches([s], prior=prior, validation=True, masks_per_patch=2, seed=8)
        first = dataset[0]
        dataset.epoch = 99
        np.testing.assert_array_equal(first['maps'], dataset[0]['maps'])
        other = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in s.items()}
        hidden = first['maps'][OBSERVED] == 0
        other['probabilities'][:, hidden] = 0
        other['probabilities'][2, hidden] = 1
        changed = FrontierPatches([other], prior=prior, validation=True, masks_per_patch=2, seed=8)[0]
        np.testing.assert_array_equal(first['maps'], changed['maps'])
        self.assertFalse(np.any(first['target_mask'].astype(bool) & (first['maps'][OBSERVED] > 0)))
        self.assertTrue(np.all(first['maps'][DISTANCE][first['target_mask'].astype(bool)] <= 1.))

    def test_missing_observations_produce_no_frontier(self):
        p = np.full((CLASSES, 16, 16), np.nan, np.float32)
        maps, frontier = encode_observation(p, np.zeros((16, 16), bool), class_prior=np.ones(CLASSES)/CLASSES)
        self.assertTrue(np.isfinite(maps).all())
        self.assertFalse(frontier.any())


class FrontierModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self, cell_size=16., frontier_m=32.):
        s = coarse_fixture(); known = np.zeros((16, 16), bool); known[:, :8] = True
        maps, _ = encode_observation(s['probabilities'], known, class_prior=np.array([.5, .5, 0., 0., 0.]),
                                     cell_size=cell_size, frontier_m=frontier_m)
        return torch.tensor(maps)[None]

    def test_initial_model_extends_structure_preserves_observations_and_abstains_outside_frontier(self):
        model = FrontierWorldModel(hidden=4, members=2)
        maps = self.inputs(); original = maps.clone()
        result = model.predict(maps)
        mask = result['prediction_mask']
        self.assertTrue(torch.allclose(result['semantics'].permute(0, 2, 3, 1)[mask],
            maps[:, PRIOR:PRIOR+CLASSES].permute(0, 2, 3, 1)[mask], atol=1e-6))
        self.assertEqual(float(result['semantics'][:, :, :, 10:].sum()), 0.)
        self.assertTrue(torch.equal(result['semantics'][:, :, :, :8], maps[:, :CLASSES, :, :8]))
        self.assertTrue(torch.equal(maps, original))
        self.assertNotIn('vehicle_target', result)
        self.assertNotIn('survey_target', result)
        self.assertTrue(result['semantics'][0, 0, 1, 8] > result['semantics'][0, 1, 1, 8])
        self.assertTrue(result['semantics'][0, 1, 14, 8] > result['semantics'][0, 0, 14, 8])

    def test_loss_only_updates_measured_outward_targets(self):
        logits = torch.randn(2, 1, CLASSES, 16, 16, requires_grad=True)
        target = torch.zeros(1, CLASSES, 16, 16); target[:, 0] = 1
        mask = torch.zeros(1, 16, 16); mask[:, :, 8:10] = 1
        support = torch.ones(1, 16, 16); support[:, :4] = 0
        frontier_loss(logits, dict(target=target, target_mask=mask, target_support=support), torch.ones(CLASSES)).backward()
        self.assertEqual(float(logits.grad[:, :, :, :, :8].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[:, :, :, :4].abs().sum()), 0.)
        self.assertGreater(float(logits.grad[:, :, :, 4:, 8:10].abs().sum()), 0.)

    def test_class_balanced_metric_does_not_reward_the_forest_majority(self):
        target = np.zeros((CLASSES, 1, 4)); target[0, 0, :3] = 1; target[1, 0, 3] = 1
        prediction = np.zeros_like(target); prediction[0] = 1
        metric = training.RegionMetrics(); metric.update(prediction, target, np.ones((1, 4), bool), np.ones((1, 4)))
        result = metric.result()
        self.assertAlmostEqual(result['brier'], .5)
        self.assertAlmostEqual(result['balanced_brier'], 1.)

    def test_real_updates_checkpoint_roundtrip_and_old_checkpoint_rejection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); manifest = dataset_fixture(root)
            v9 = root/'v9.pt'; v9.write_bytes(b'frozen test actor')
            original_hash = training.sha256
            with patch.object(training, 'sha256', side_effect=lambda p: training.V9_SHA256 if Path(p) == v9 else original_hash(p)):
                result = training.train(manifest, root/'run', v9_checkpoint=v9, epochs=2, side=16,
                    cell_size=2., frontier_m=4., samples_per_epoch=8, validation_masks=2,
                    batch_size=4, hidden=4, members=2, device='cpu')
            self.assertEqual(result['optimizer_steps'], 4)
            self.assertFalse(result['simulation_used'])
            self.assertFalse(result['ros_used'])
            model, document = load_checkpoint(result['checkpoint'])
            self.assertEqual(document['classes'], list(CLASS_NAMES))
            self.assertEqual(document['architecture']['frontier_m'], 4.)
            self.assertEqual(document['v9_checkpoint_sha256'], training.V9_SHA256)
            self.assertTrue(any(float(member.output.weight.abs().sum()) > 0 for member in model.predictors))
            self.assertTrue(torch.isfinite(model.predict(self.inputs(2., 4.))['semantics']).all())
            torch.save(dict(format='vtmp2-semantic-completion-v1'), root/'old.pt')
            with self.assertRaisesRegex(ValueError, 'schema-6'):
                load_checkpoint(root/'old.pt')

    def test_default_entry_and_factory_select_coarse_frontier(self):
        config = Path(__file__).resolve().parents[4]/'config/algorithm/world_model_vtmp2.json'
        command = build_command(config)
        self.assertEqual(Path(command[3]).name, 'frontier_training.py')
        self.assertIn('--data-manifest', command)
        self.assertNotIn('--terrain', command)
        self.assertIsInstance(make_model(dict(algorithm='vtmp2', model_architecture=ARCHITECTURE, hidden=4, members=1)),
                              FrontierWorldModel)


if __name__ == '__main__':
    unittest.main()
