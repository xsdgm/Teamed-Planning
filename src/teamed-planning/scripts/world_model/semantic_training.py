"""Offline original-WildScenes semantic completion. Never starts a simulator."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'world_model'

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch.utils.data import DataLoader, Dataset

from .hierarchical_sources import copy_source_files
from .semantic_data import (CHANNELS, CLASSES, CLASS_NAMES, FORMAT, NATIVE_IDS, atomic_json, augment_patch,
    crop_patch, encode_observation, load_scene, observation_mask, patch_coordinates, prepare_sequence, sha256)
from .semantic_model import (ARCHITECTURE, CHECKPOINT_FORMAT, SCHEMA_VERSION,
                             SemanticCompletionModel, completion_loss)

V9_SHA256 = '7ac3aedcc3876902773b2241f60e839eb3e3b5630ddeb3c8be660421f2c24198'


def load_manifest(path):
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    if document.get('format') != FORMAT or document.get('simulation_used') is not False:
        raise ValueError('Original measured semantic data required')
    scenes = {'train': [], 'val': []}
    seen, sites = set(), {'train': set(), 'val': set()}
    for record in document['records']:
        split = record['split']
        if split not in scenes:
            raise ValueError('Unsupported semantic split')
        source = (path.parent/record['map']).resolve()
        scene = load_scene(source, record['sha256'])
        sequence = scene['metadata']['sequence']
        if sequence != record['sequence'] or record['site'] != sequence.split('-')[0]:
            raise ValueError('Scene and manifest site disagree')
        source_hash = scene['metadata']['source_scans_sha256']
        if sequence in seen or record['sha256'] in seen or source_hash in seen:
            raise ValueError('Duplicate scene or train/validation leakage')
        seen.update((sequence, record['sha256'], source_hash))
        sites[split].add(record['site'])
        if not np.isclose(scene['resolution_m'], document['resolution_m']):
            raise ValueError('Mixed metric resolutions')
        scenes[split].append(scene)
    if not all(scenes.values()) or sites['train'] & sites['val']:
        raise ValueError('Separate nonempty forest locations required for train/validation')
    return scenes, document


class SemanticPatches(Dataset):
    def __init__(self, scenes, *, side=128, samples=1024, validation=False, masks_per_patch=4, seed=0):
        self.scenes, self.side, self.validation, self.seed = scenes, side, validation, seed
        self.epoch = 0
        self.patches = [(i, row, col) for i, scene in enumerate(scenes)
                        for row, col in patch_coordinates(scene['valid'], side)]
        if not self.patches:
            raise ValueError('No patches with enough real semantic support')
        self.samples = len(self.patches)*masks_per_patch if validation else samples
        if self.samples < 1:
            raise ValueError('Positive sample count required')

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0 if self.validation else self.epoch, index]))
        patch_index = index % len(self.patches) if self.validation else int(rng.integers(len(self.patches)))
        scene_index, row, col = self.patches[patch_index]
        scene = self.scenes[scene_index]
        if not self.validation:
            h, w = scene['valid'].shape
            rr = int(np.clip(row+rng.integers(-self.side//8, self.side//8+1), 0, max(0, h-self.side)))
            cc = int(np.clip(col+rng.integers(-self.side//8, self.side//8+1), 0, max(0, w-self.side)))
            if scene['valid'][rr:rr+self.side, cc:cc+self.side].sum() >= 128:
                row, col = rr, cc
        patch = crop_patch(scene, row, col, self.side)
        if not self.validation:
            patch = augment_patch(patch, rng)
        observed = observation_mask(patch['valid'], rng)
        maps = encode_observation(patch['probabilities'], observed, patch['elevation'], patch['height_valid'])
        return dict(maps=maps, target=patch['probabilities'], hidden_valid=(patch['valid'] & ~observed).astype(np.float32))


def training_prior(scenes):
    total = sum(scene['probabilities'].sum((1, 2), dtype=np.float64) for scene in scenes)
    prior = total/total.sum()
    weights = np.zeros(CLASSES, np.float32)
    support = prior > 0
    weights[support] = np.minimum(8., 1/np.sqrt(prior[support]))
    weights /= np.sum(weights*prior)
    return prior.astype(np.float32), weights


class Metrics:
    def __init__(self):
        self.cells, self.nll, self.brier = 0, 0., 0.
        self.confusion = np.zeros((CLASSES, CLASSES), np.int64)

    def update(self, probability, target, mask):
        mask = np.asarray(mask, bool)
        p, y = np.moveaxis(probability, 0, -1)[mask], np.moveaxis(target, 0, -1)[mask]
        self.cells += len(y)
        self.nll += float(-(y*np.log(np.maximum(p, 1e-8))).sum(dtype=np.float64))
        self.brier += float(((p-y)**2).sum(dtype=np.float64))
        index = y.argmax(1)*CLASSES+p.argmax(1)
        self.confusion += np.bincount(index, minlength=CLASSES**2).reshape(CLASSES, CLASSES)

    def result(self):
        intersection = self.confusion.diagonal()
        union = self.confusion.sum(0)+self.confusion.sum(1)-intersection
        present = union > 0
        per_class = {name: (float(intersection[i]/union[i]) if union[i] else None)
                     for i, name in enumerate(CLASS_NAMES)}
        return dict(cells=self.cells, nll=self.nll/max(1, self.cells), brier=self.brier/max(1, self.cells),
            dominant_class_accuracy=float(intersection.sum()/max(1, self.cells)),
            dominant_class_miou=float((intersection[present]/union[present]).mean()) if present.any() else None,
            per_class_iou=per_class, dominant_target_counts=self.confusion.sum(1).tolist())


@torch.no_grad()
def evaluate(model, loader, device, prior, resolution, *, baselines=False):
    model.eval()
    totals = {name: Metrics() for name in (('model', 'training_prior', 'nearest_observed') if baselines else ('model',))}
    bands = {name: Metrics() for name in ('0_10m', '10_30m', '30_60m', '60m_plus')}
    for batch in loader:
        maps = batch['maps'].to(device)
        predictions = model.predict(maps)['semantics'].cpu().numpy()
        for i, probability in enumerate(predictions):
            target, hidden = batch['target'][i].numpy(), batch['hidden_valid'][i].numpy().astype(bool)
            observed = batch['maps'][i, CLASSES].numpy().astype(bool)
            distance, nearest = distance_transform_edt(~observed, sampling=resolution, return_indices=True)
            totals['model'].update(probability, target, hidden)
            for key, low, high in (('0_10m', 0, 10), ('10_30m', 10, 30), ('30_60m', 30, 60), ('60m_plus', 60, np.inf)):
                bands[key].update(probability, target, hidden & (distance >= low) & (distance < high))
            if baselines:
                constant = np.broadcast_to(prior[:, None, None], target.shape)
                # Fixed train-prior smoothing avoids infinite log-loss when a
                # nearest column has no examples of a subsequently seen class.
                nearby = .95*batch['maps'][i, :CLASSES].numpy()[:, nearest[0], nearest[1]] + .05*constant
                totals['training_prior'].update(constant, target, hidden)
                totals['nearest_observed'].update(nearby, target, hidden)
    return {**{name: metric.result() for name, metric in totals.items()},
            'distance_bands': {name: metric.result() for name, metric in bands.items()}}


def save_checkpoint(path, document):
    temp = Path(path).with_suffix('.tmp')
    torch.save(document, temp)
    temp.replace(path)


def train(data_manifest, output, *, epochs=100, side=128, samples_per_epoch=1024,
          validation_masks=4, batch_size=8, hidden=24, members=3, device='cuda', seed=20261113,
          v9_checkpoint=None, workers=0):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'last.pt').exists() or (output/'metrics.jsonl').exists():
        raise FileExistsError('Training output already contains results')
    if min(epochs, samples_per_epoch, validation_masks, batch_size) < 1 or workers < 0:
        raise ValueError('Positive training counts required')
    if v9_checkpoint is None or sha256(v9_checkpoint) != V9_SHA256:
        raise ValueError('Expected frozen v9 deployment checkpoint')
    torch.set_num_threads(2)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    selected_device = torch.device(device)
    if selected_device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested CUDA is unavailable; no silent CPU fallback')
    scenes, manifest = load_manifest(data_manifest)
    prior, weights = training_prior(scenes['train'])
    train_set = SemanticPatches(scenes['train'], side=side, samples=samples_per_epoch, seed=seed)
    val_set = SemanticPatches(scenes['val'], side=side, validation=True, masks_per_patch=validation_masks, seed=seed+1)
    options = dict(batch_size=batch_size, num_workers=workers, pin_memory=selected_device.type == 'cuda')
    training = DataLoader(train_set, shuffle=False, **options)
    validation = DataLoader(val_set, shuffle=False, **options)
    model = SemanticCompletionModel(hidden, members).to(selected_device)
    # Initialize close to a measured prior, rather than a uniform guess over
    # absent classes. All spatial weights remain trainable.
    with torch.no_grad():
        for predictor in model.predictors:
            predictor.output.bias.copy_(torch.tensor(np.log(prior.clip(1e-6)), device=selected_device))
    class_weights = torch.tensor(weights, device=selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-5)
    config = dict(format=CHECKPOINT_FORMAT, schema_version=SCHEMA_VERSION, model_architecture=ARCHITECTURE,
        epochs=epochs, side=side, resolution_m=float(manifest['resolution_m']), samples_per_epoch=samples_per_epoch,
        validation_masks=validation_masks, batch_size=batch_size, architecture=model.configuration(),
        device=str(selected_device), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), seed=seed,
        data_manifest=str(Path(data_manifest).resolve()), data_manifest_sha256=sha256(data_manifest),
        training_sequences=[s['metadata']['sequence'] for s in scenes['train']],
        validation_sequences=[s['metadata']['sequence'] for s in scenes['val']],
        distinct_training_patch_origins=len(train_set.patches), distinct_validation_patch_origins=len(val_set.patches),
        validation_samples=len(val_set), train_class_prior=prior.tolist(), class_weights=weights.tolist(),
        classes=list(CLASS_NAMES), native_ids=list(NATIVE_IDS), input_channels=CHANNELS,
        unsupported_training_classes=[name for name, value in zip(CLASS_NAMES, prior) if value == 0],
        supervision='original measured columns withheld by contiguous masks; unmeasured cells ignored',
        masks='fixed for validation; seeded directional/rectangle masks, spatial jitter and rotations for training',
        nearest_baseline='95% nearest observed column + 5% training-only class prior',
        loss_probability_contract='class-weighted targets and log-weight-adjusted training logits; raw deployment softmax',
        input_semantics='labelled-point composition of vertical columns; not ground-surface traversability',
        height_normalization='observed ground mean elevation minus median observed elevation, divided by 10m, clipped to [-5,5]',
        vehicle_uav_state_training=False, action_dynamics_training=False, simulation_used=False,
        v9_checkpoint_sha256=V9_SHA256, v9_weights_trained=False, navigation_validated=False,
        parameters=sum(p.numel() for p in model.parameters()))
    atomic_json(output/'config.json', config)
    atomic_json(output/'dataset_manifest.json', manifest)
    atomic_json(output/'status.json', dict(stage='baseline_validation', pid=os.getpid(), epoch=0, epochs=epochs))
    baseline = evaluate(model, validation, selected_device, prior, manifest['resolution_m'], baselines=True)
    atomic_json(output/'baselines.json', baseline)
    best_loss, best_epoch, optimizer_steps, started = float('inf'), 0, 0, time.time()
    for epoch in range(1, epochs+1):
        train_set.epoch = epoch
        model.train()
        loss_sum = 0.
        for step, batch in enumerate(training, start=1):
            maps, targets, mask = (batch[k].to(selected_device, non_blocking=True) for k in ('maps', 'target', 'hidden_valid'))
            optimizer.zero_grad(set_to_none=True)
            loss = completion_loss(model(maps), targets, mask, class_weights)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite semantic training loss')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            optimizer_steps += 1
            loss_sum += float(loss.detach())
            if step == 1 or step % 16 == 0 or step == len(training):
                atomic_json(output/'status.json', dict(stage='training', pid=os.getpid(), epoch=epoch, epochs=epochs,
                    batch=step, batches=len(training), optimizer_steps=optimizer_steps, loss=loss_sum/step,
                    elapsed_seconds=time.time()-started))
        schedule.step()
        metrics = evaluate(model, validation, selected_device, prior, manifest['resolution_m'])
        score = metrics['model']['nll']
        improved = score < best_loss
        if improved:
            best_loss, best_epoch = score, epoch
        row = dict(epoch=epoch, optimizer_steps=optimizer_steps, train_loss=loss_sum/len(training),
            validation=metrics, best_epoch=best_epoch, best_validation_nll=best_loss,
            gradient_norm=float(gradient), elapsed_seconds=time.time()-started,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if selected_device.type == 'cuda' else None)
        with (output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        document = {**config, 'architecture': model.configuration(), 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': schedule.state_dict(),
            'epoch': epoch, 'optimizer_steps': optimizer_steps, 'validation': metrics,
            'best_epoch': best_epoch, 'best_validation_nll': best_loss,
            'baseline_validation': {key: baseline[key] for key in ('training_prior', 'nearest_observed')}}
        save_checkpoint(output/'last.pt', document)
        if improved:
            save_checkpoint(output/'best.pt', document)
        atomic_json(output/'status.json', dict(stage='training', pid=os.getpid(), **row, epochs=epochs))
        print(json.dumps(dict(stage='training', epoch=epoch, epochs=epochs, optimizer_steps=optimizer_steps,
            train_loss=row['train_loss'], validation_nll=score, best_epoch=best_epoch,
            prior_nll=baseline['training_prior']['nll'], nearest_nll=baseline['nearest_observed']['nll']), allow_nan=False), flush=True)
    result = dict(stage='complete', pid=os.getpid(), epochs=epochs, optimizer_steps=optimizer_steps,
        best_epoch=best_epoch, best_validation_nll=best_loss, checkpoint=str(output/'best.pt'),
        beats_prior_nll=best_loss < baseline['training_prior']['nll'],
        beats_nearest_nll=best_loss < baseline['nearest_observed']['nll'],
        simulation_used=False, navigation_validated=False, v9_weights_trained=False)
    atomic_json(output/'status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-manifest', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--v9-checkpoint', type=Path, required=True)
    parser.add_argument('--train-sequence', default='K-01')
    parser.add_argument('--val-sequence', default='V-01')
    parser.add_argument('--resolution', type=float, default=2.)
    for name, default in (('epochs', 100), ('side', 128), ('samples_per_epoch', 1024), ('validation_masks', 4),
                          ('batch_size', 8), ('hidden', 24), ('members', 3), ('seed', 20261113), ('workers', 0)):
        parser.add_argument('--'+name.replace('_', '-'), type=int, default=default)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.data_manifest is None and args.source_root is None:
        parser.error('--data-manifest or --source-root required; no simulator fallback')
    if args.train_sequence.split('-')[0] == args.val_sequence.split('-')[0]:
        parser.error('Separate forest sites required')
    output.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    sources = ('world_model/__init__.py', 'world_model/hierarchical_sources.py', 
               'world_model/semantic_data.py', 'world_model/terrain/semantic_model.py',
               'world_model/semantic_training.py', 'simulator/prepare_wildscenes.py')
    hashes = copy_source_files(source, output / 'sources', sources)
    atomic_json(output/'source_manifest.json', hashes)
    atomic_json(output/'launch.json', dict(pid=os.getpid(), command=sys.argv, simulation_used=False))
    try:
        data_manifest = args.data_manifest
        if data_manifest is None:
            records = []
            for split, sequence in (('train', args.train_sequence), ('val', args.val_sequence)):
                meta = prepare_sequence(args.source_root/sequence, output/'data'/sequence,
                    resolution=args.resolution,
                    progress=lambda row: atomic_json(output/'status.json', dict(pid=os.getpid(), **row)))
                records.append(dict(split=split, sequence=sequence, site=sequence.split('-')[0],
                    map=str(output/'data'/sequence/'semantic_map.npz'), sha256=meta['map_sha256']))
            data_manifest = output/'data/manifest.json'
            atomic_json(data_manifest, dict(format=FORMAT, source='original WildScenes3d', records=records,
                        resolution_m=args.resolution, simulation_used=False))
        kwargs = {key: getattr(args, key) for key in ('epochs', 'side', 'samples_per_epoch', 'validation_masks',
            'batch_size', 'hidden', 'members', 'device', 'seed', 'workers', 'v9_checkpoint')}
        train(data_manifest, output, **kwargs)
    except BaseException as exc:
        atomic_json(output/'status.json', dict(stage='failed', pid=os.getpid(), error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
