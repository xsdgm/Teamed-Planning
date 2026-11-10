"""Dataset integrity, window boundaries and actual optimizer/resume checks."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from world_model.train import (CHANNELS, CHECKPOINT_SHA, FEATURES, SCHEMA_VERSION, SIDE,
                   EpisodeWindowDataset, Vtmp2WorldModel, evaluate, evaluate_rollout,
                   load_episodes, run_training)
from world_model.model import rollout_prediction_loss, sequence_loss


def episode_values(seed, split, length=3, *, zero_terminal=False):
    states = np.zeros((length+1, FEATURES), np.float32)
    states[:, 0] = np.arange(length+1)*.001
    states[:, 3] = 1.
    states[:, 6:8] = .8
    maps = np.zeros((length+1, CHANNELS, SIDE, SIDE), np.float32)
    maps[:, 0, 4, 4] = np.arange(length+1)/(length+1)
    actions = np.tile([.1, .2, 1.], (length, 1)).astype(np.float32)
    targets = actions[:, :2]*1000
    terminated = np.zeros(length, bool)
    truncated = np.zeros(length, bool)
    elapsed = np.ones(length, np.float32)
    if zero_terminal:
        terminated[-1] = True
        elapsed[-1] = 0
        states[-1] = states[-2]
        maps[-1] = maps[-2]
    else:
        truncated[-1] = True
    return dict(algorithm=np.asarray('vtmp2'), schema_version=np.asarray(SCHEMA_VERSION),
                checkpoint_sha256=np.asarray(CHECKPOINT_SHA), split=np.asarray(split), seed=np.asarray(seed),
                feasibility_validated=np.asarray(True), physical_clearance_m=np.asarray(1.5),
                scene_sha256=np.asarray(hashlib.sha256(f'{seed}/{split}'.encode()).hexdigest()),
                scene_geometry_sha256=np.asarray(hashlib.sha256(f'{seed}'.encode()).hexdigest()),
                maps=maps, states=states, actions=actions, targets=targets,
                labels=np.zeros((length, 3), np.float32), elapsed=elapsed,
                terminated=terminated, truncated=truncated,
                times=np.r_[0., np.cumsum(elapsed*10)], target_switched=np.zeros(length, bool),
                switch_reason=np.asarray(['']*length), target_status=np.asarray(['unverified']*length))


def write_episode(directory, seed, split, **kwargs):
    values = episode_values(seed, split, **kwargs)
    np.savez_compressed(Path(directory)/f'{split}_{seed}.npz', **values)
    return values


def write_mission_episode(directory, seed, split, purposes=('drive', 'survey')):
    values = episode_values(seed, split, length=len(purposes))
    values.update(schema_version=np.asarray(3), action_contract=np.asarray('drive_survey_v1'),
                  mission_purpose=np.asarray(purposes))
    values['actions'][:, 2] = np.where(values['mission_purpose'] == 'drive', 1., -1.)
    np.savez_compressed(Path(directory)/f'{split}_{seed}.npz', **values)
    return values


class DatasetTests(unittest.TestCase):
    def test_tail_is_masked_and_never_crosses_episode_boundary(self):
        first = episode_values(1, 'train', 3)
        second = episode_values(2, 'train', 1)
        second['states'][:, 0] = 9.
        dataset = EpisodeWindowDataset([first, second], 2)
        self.assertEqual(len(dataset), 3)
        tail = dataset[1]
        np.testing.assert_array_equal(tail['mask'], [1., 0.])
        np.testing.assert_array_equal(tail['states'][0], first['states'][2])
        np.testing.assert_array_equal(tail['states'][1], first['states'][3])
        np.testing.assert_array_equal(tail['states'][2], first['states'][3])
        np.testing.assert_array_equal(tail['actions'][1], [0., 0., 0.])
        self.assertEqual(dataset[2]['states'][0, 0], 9.)

    def test_cross_split_scene_seed_is_rejected(self):
        with TemporaryDirectory() as directory:
            write_episode(directory, 1, 'train')
            values = write_episode(directory, 1, 'val')
            # Keep the certificate distinct to exercise the independent seed gate.
            values['scene_geometry_sha256'] = np.asarray('a'*64)
            np.savez_compressed(Path(directory)/'val_1.npz', **values)
            with self.assertRaisesRegex(ValueError, 'Scene seed 1 appears twice'):
                load_episodes(directory)

    def test_cross_split_geometry_is_rejected_even_with_distinct_seed(self):
        with TemporaryDirectory() as directory:
            first = write_episode(directory, 1, 'train')
            second = write_episode(directory, 2, 'val')
            second['scene_geometry_sha256'] = first['scene_geometry_sha256']
            np.savez_compressed(Path(directory)/'val_2.npz', **second)
            with self.assertRaisesRegex(ValueError, 'physical scene occurs in both'):
                load_episodes(directory)

    def test_zero_duration_terminal_is_valid(self):
        with TemporaryDirectory() as directory:
            write_episode(directory, 1, 'train', zero_terminal=True)
            write_episode(directory, 2, 'val')
            episodes, manifest = load_episodes(directory)
            self.assertEqual(episodes['train'][0]['elapsed'][-1], 0.)
            self.assertEqual(len(manifest['episodes']), 2)

    def test_legacy_and_mission_datasets_expose_their_actual_contract(self):
        for mission in (False, True):
            with self.subTest(mission=mission), TemporaryDirectory() as directory:
                writer = write_mission_episode if mission else write_episode
                writer(directory, 1, 'train')
                writer(directory, 2, 'val')
                episodes, manifest = load_episodes(directory)
                self.assertEqual(manifest['schema_version'], 3 if mission else SCHEMA_VERSION)
                self.assertEqual(manifest['action_contract'], 'drive_survey_v1' if mission else 'legacy_drive_v1')
                self.assertEqual(manifest['observed_action_purposes'], ['drive', 'survey'] if mission else ['drive'])
                if mission:
                    np.testing.assert_array_equal(episodes['train'][0]['actions'][:, 2], [1., -1.])

    def test_mission_contract_mixed_schemas_and_missing_action_coverage_are_rejected(self):
        corruptions = {
            'sign': lambda v: v['actions'].__setitem__((1, 2), 1.),
            'purpose': lambda v: v.update(mission_purpose=np.asarray(['drive', 'unknown'])),
            'missing_purpose': lambda v: v.pop('mission_purpose'),
            'missing_contract': lambda v: v.pop('action_contract'),
            'wrong_contract': lambda v: v.update(action_contract=np.asarray('legacy_drive_v1')),
        }
        for name, corrupt in corruptions.items():
            with self.subTest(name=name), TemporaryDirectory() as directory:
                values = write_mission_episode(directory, 1, 'train')
                write_mission_episode(directory, 2, 'val')
                corrupt(values)
                np.savez_compressed(Path(directory)/'train_1.npz', **values)
                with self.assertRaises(ValueError):
                    load_episodes(directory)
        with TemporaryDirectory() as directory:
            write_episode(directory, 1, 'train')
            write_mission_episode(directory, 2, 'val')
            with self.assertRaisesRegex(ValueError, 'Cannot mix'):
                load_episodes(directory)
        with TemporaryDirectory() as directory:
            write_mission_episode(directory, 1, 'train', purposes=('drive', 'drive'))
            write_mission_episode(directory, 2, 'val')
            with self.assertRaisesRegex(ValueError, 'requires observed drive and survey'):
                load_episodes(directory)

    def test_bad_metadata_shapes_and_boundaries_are_rejected(self):
        corruptions = {
            'checkpoint': lambda v: v.update(checkpoint_sha256=np.asarray('0'*64)),
            'schema': lambda v: v.update(schema_version=np.asarray(1)),
            'feasibility': lambda v: v.update(feasibility_validated=np.asarray(False)),
            'clearance': lambda v: v.update(physical_clearance_m=np.asarray(.5)),
            'shape': lambda v: v.update(states=v['states'][:-1]),
            'nan': lambda v: v['maps'].__setitem__((0, 0, 0, 0), np.nan),
            'termination': lambda v: v['terminated'].__setitem__(0, True),
            'time': lambda v: v['times'].__setitem__(1, 100.),
        }
        for name, corrupt in corruptions.items():
            with self.subTest(name=name), TemporaryDirectory() as directory:
                values = write_episode(directory, 1, 'train')
                write_episode(directory, 2, 'val')
                corrupt(values)
                np.savez_compressed(Path(directory)/'train_1.npz', **values)
                with self.assertRaises(ValueError):
                    load_episodes(directory)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_validation_is_deterministic_and_counts_only_unmasked_steps(self):
        episodes = [episode_values(1, 'val', 3)]
        loader = DataLoader(EpisodeWindowDataset(episodes, 2), batch_size=2)
        model = Vtmp2WorldModel(hidden=8, latent=4)
        first = evaluate(model, loader, torch.device('cpu'))
        second = evaluate(model, loader, torch.device('cpu'))
        self.assertEqual(first, second)
        self.assertEqual(first['transitions'], 3)
        self.assertIn('persistence_state_mse_scaled', first)
        self.assertIn('persistence_map_mse_weighted', first)

    def test_free_rollout_uses_predictions_and_never_assimilates_future_labels(self):
        batch = next(iter(DataLoader(EpisodeWindowDataset([episode_values(1, 'train', 4)], 4))))
        model = Vtmp2WorldModel(hidden=8, latent=4)
        imagine = model.imagine
        inputs, outputs = [], []

        def record(*args, **kwargs):
            inputs.append((args[3].detach().clone(), args[4].detach().clone()))
            result = imagine(*args, **kwargs)
            outputs.append((result[3].detach().clone(), result[4].detach().clone()))
            return result

        with patch.object(model, 'observe', wraps=model.observe) as observe:
            with patch.object(model, 'imagine', side_effect=record):
                torch.manual_seed(5)
                first_loss = rollout_prediction_loss(model, batch, 4)
            self.assertEqual(observe.call_count, 1)
        first_inputs = inputs[:]
        for t in range(1, 4):
            self.assertTrue(torch.equal(inputs[t][0], outputs[t-1][0]))
            self.assertTrue(torch.equal(inputs[t][1], outputs[t-1][1]))
        changed = {key: value.clone() for key, value in batch.items()}
        changed['maps'][:, 1:] = .75
        changed['states'][:, 1:, 0] += .1
        inputs.clear()
        outputs.clear()
        with patch.object(model, 'imagine', side_effect=record):
            torch.manual_seed(5)
            changed_loss = rollout_prediction_loss(model, changed, 4)
        for old, new in zip(first_inputs, inputs):
            self.assertTrue(torch.equal(old[0], new[0]))
            self.assertTrue(torch.equal(old[1], new[1]))
        self.assertNotEqual(float(first_loss.detach()), float(changed_loss.detach()))
        first_loss.backward()
        for parameter in (model.features_head[-1].weight, model.memory.weight_ih):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_free_rollout_masks_padded_targets_and_reports_actual_durations(self):
        episode = episode_values(1, 'val', 3)
        episode['elapsed'][:] = [.2, .5, 1.]
        loader = DataLoader(EpisodeWindowDataset([episode, episode_values(2, 'val', 1)], 4), batch_size=2)
        batch = next(iter(loader))
        model = Vtmp2WorldModel(hidden=8, latent=4)
        changed = {key: value.clone() for key, value in batch.items()}
        changed['maps'][0, 4:] = 100.
        changed['maps'][1, 2:] = 100.
        changed['states'][0, 4:] = 100.
        changed['states'][1, 2:] = 100.
        changed['labels'][batch['mask'] == 0] = 1.
        torch.manual_seed(7)
        first = rollout_prediction_loss(model, batch, 4)
        torch.manual_seed(7)
        second = rollout_prediction_loss(model, changed, 4)
        self.assertEqual(float(first.detach()), float(second.detach()))
        report = evaluate_rollout(model, loader, torch.device('cpu'))
        self.assertEqual(report, evaluate_rollout(model, loader, torch.device('cpu')))
        self.assertEqual(report['horizons']['1']['samples'], 2)
        self.assertAlmostEqual(report['horizons']['1']['actual_elapsed_seconds'], 6.)
        self.assertEqual(report['horizons']['2']['samples'], 1)
        self.assertAlmostEqual(report['horizons']['2']['actual_elapsed_seconds'], 7.)
        self.assertEqual(report['horizons']['4']['samples'], 0)
        self.assertEqual(report['horizons']['8']['samples'], 0)
        self.assertIn('including_target_switches', report['action_protocol'])

    def test_rollout_training_options_reject_invalid_values(self):
        batch = next(iter(DataLoader(EpisodeWindowDataset([episode_values(1, 'train')], 4))))
        model = Vtmp2WorldModel(hidden=8, latent=4)
        for options in (dict(rollout_steps=0), dict(rollout_steps=True),
                        dict(rollout_weight=-1.), dict(rollout_weight=float('nan'))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                sequence_loss(model, batch, **options)

    def test_cpu_epoch_updates_parameters_then_resumes_with_strict_dataset(self):
        with TemporaryDirectory() as temporary:
            dataset = Path(temporary)/'data'
            dataset.mkdir()
            output = Path(temporary)/'training'
            write_episode(dataset, 1, 'train')
            write_episode(dataset, 2, 'val')
            options = dict(device='cpu', epochs=1, batch_size=2, sequence_length=2,
                           seed=17, hidden=8, latent=4, bootstrap=True)
            torch.manual_seed(options['seed'])
            initial = Vtmp2WorldModel(hidden=8, latent=4)
            with redirect_stdout(io.StringIO()):
                summary = run_training(dataset, output, **options)
            self.assertEqual(summary['optimizer_steps'], 1)
            self.assertFalse(summary['deployment_enabled'])
            checkpoint = torch.load(output/'latest.pt', map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['config']['rollout_steps'], 4)
            self.assertEqual(checkpoint['config']['rollout_weight'], .25)
            self.assertEqual(checkpoint['config']['schema_version'], SCHEMA_VERSION)
            self.assertEqual(checkpoint['config']['action_contract'], 'legacy_drive_v1')
            self.assertEqual(checkpoint['config']['observed_action_purposes'], ['drive'])
            report = json.loads((output/'rollout_validation.json').read_text())
            self.assertEqual(report['latest_epoch'], 1)
            self.assertEqual(report['best_epoch'], 1)
            self.assertFalse(torch.equal(checkpoint['model']['features_head.2.weight'],
                                         initial.state_dict()['features_head.2.weight']))
            self.assertGreater(checkpoint['history'][0]['max_gradient_norm'], 0.)
            self.assertTrue((output/'best.pt').is_file())
            options['epochs'] = 2
            with redirect_stdout(io.StringIO()):
                resumed = run_training(dataset, output, resume=True, **options)
            self.assertEqual(resumed['epoch'], 2)
            self.assertEqual(resumed['optimizer_steps'], 2)
            rows = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['epoch'] for row in rows], [1, 2])
            self.assertEqual(json.loads((output/'status.json').read_text())['stage'], 'complete')
            with self.assertRaisesRegex(ValueError, 'configuration/source hashes'):
                run_training(dataset, output, resume=True, rollout_weight=.5, **options)
            values = episode_values(2, 'val')
            values['maps'][0, 0, 1, 1] = .5
            np.savez_compressed(dataset/'val_2.npz', **values)
            with self.assertRaisesRegex(ValueError, 'dataset hashes/splits'):
                run_training(dataset, output, resume=True, **options)

    def test_mission_training_checkpoint_records_observed_purposes(self):
        with TemporaryDirectory() as temporary:
            dataset = Path(temporary)/'data'
            dataset.mkdir()
            output = Path(temporary)/'training'
            write_mission_episode(dataset, 1, 'train')
            write_mission_episode(dataset, 2, 'val')
            with redirect_stdout(io.StringIO()):
                run_training(dataset, output, device='cpu', epochs=1, batch_size=2,
                             sequence_length=2, hidden=8, latent=4)
            checkpoint = torch.load(output/'latest.pt', map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['config']['schema_version'], 3)
            self.assertEqual(checkpoint['config']['action_contract'], 'drive_survey_v1')
            self.assertEqual(checkpoint['config']['observed_action_purposes'], ['drive', 'survey'])


if __name__ == '__main__':
    unittest.main()
