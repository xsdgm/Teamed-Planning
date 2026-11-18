"""Train coarse regional extrapolation from measured forest boundaries only."""
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
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader

from .frontier_data import (CHANNELS, CLASSES, CLASS_NAMES, DISTANCE, FORMAT, GROUPS, OBSERVED,
    PRIOR, REGIONAL_FORMAT, FrontierPatches, class_prior, coarsen_scene, load_regional_manifest)
from .frontier_model import (ARCHITECTURE, CHECKPOINT_FORMAT, SCHEMA_VERSION, FrontierWorldModel,
                             frontier_loss)
from .hierarchical_sources import copy_source_files
from .semantic_data import atomic_json, sha256
from .semantic_training import V9_SHA256, load_manifest, save_checkpoint

SOURCES = ('world_model/__init__.py', 'world_model/hierarchical_sources.py', 'world_model/terrain/frontier_data.py', 'world_model/terrain/frontier_model.py', 'world_model/frontier_training.py',
           'world_model/semantic_data.py', 'world_model/terrain/semantic_model.py', 'world_model/semantic_training.py',
           'simulator/prepare_wildscenes.py',
           'world_model/goose_data.py', 'world_model/goose_download.py', 'world_model/frontier_selective.py',
           'world_model/terrain/frontier_guard.py', 'world_model/frontier_assessment.py')


class RegionMetrics:
    def __init__(self):
        self.blocks = 0
        self.weight = self.nll = self.brier = 0.
        self.class_weight = np.zeros(CLASSES, np.float64)
        self.class_brier = np.zeros(CLASSES, np.float64)
        self.confusion = np.zeros((CLASSES, CLASSES), np.float64)
        self.targets = np.zeros(CLASSES, np.int64)
        self.predictions = np.zeros(CLASSES, np.int64)

    def update(self, probability, target, mask, support):
        mask = np.asarray(mask, bool)
        p, y = probability[:, mask].T, target[:, mask].T
        weight = np.asarray(support)[mask].astype(np.float64)
        if not len(y):
            return
        labels, pred = y.argmax(1), p.argmax(1)
        brier = ((p-y)**2).sum(1)
        self.blocks += len(y)
        self.weight += float(weight.sum())
        self.brier += float((brier*weight).sum())
        self.nll += float((-(y*np.log(p.clip(1e-8))).sum(1)*weight).sum())
        self.class_weight += np.bincount(labels, weights=weight, minlength=CLASSES)
        self.class_brier += np.bincount(labels, weights=weight*brier, minlength=CLASSES)
        self.targets += np.bincount(labels, minlength=CLASSES)
        self.predictions += np.bincount(pred, minlength=CLASSES)
        self.confusion += np.bincount(labels*CLASSES+pred, weights=weight, minlength=CLASSES**2).reshape(CLASSES, CLASSES)

    def result(self):
        intersection = self.confusion.diagonal()
        union = self.confusion.sum(0)+self.confusion.sum(1)-intersection
        supported = self.class_weight > 0
        balanced = float((self.class_brier[supported]/self.class_weight[supported]).mean()) if supported.any() else None
        return dict(blocks=self.blocks, label_support_weight=self.weight, nll=self.nll/max(self.weight, 1e-8),
            brier=self.brier/max(self.weight, 1e-8), balanced_brier=balanced,
            accuracy=float(intersection.sum()/max(self.weight, 1e-8)),
            miou=float((intersection[union > 0]/union[union > 0]).mean()) if (union > 0).any() else None,
            per_class_iou={name: float(intersection[i]/union[i]) if union[i] else None for i, name in enumerate(CLASS_NAMES)},
            per_class_brier={name: float(self.class_brier[i]/self.class_weight[i]) if supported[i] else None
                             for i, name in enumerate(CLASS_NAMES)},
            target_counts=self.targets.tolist(), prediction_counts=self.predictions.tolist())


@torch.no_grad()
def evaluate(model, loader, device, prior, *, baselines=False):
    model.eval()
    names = ('model', 'structure_extension', 'training_prior', 'observed_mean') if baselines else ('model',)
    totals = {name: RegionMetrics() for name in names}
    first, second, changes = RegionMetrics(), RegionMetrics(), RegionMetrics()
    for batch in loader:
        output = model.predict(batch['maps'].to(device))
        prediction = output['semantics'].cpu().numpy()
        eligible = output['prediction_mask'].cpu().numpy()
        for i, p in enumerate(prediction):
            target, mask, support = (batch[k][i].numpy() for k in ('target', 'target_mask', 'target_support'))
            mask = mask.astype(bool)
            if (mask & ~eligible[i]).any():
                raise AssertionError('Supervision outside predicted outward frontier')
            maps = batch['maps'][i].numpy()
            totals['model'].update(p, target, mask, support)
            distance = maps[DISTANCE]*model.frontier_m
            first.update(p, target, mask & (distance <= model.cell_size+1e-5), support)
            second.update(p, target, mask & (distance > model.cell_size+1e-5), support)
            known = maps[OBSERVED].astype(bool)
            _, nearest = distance_transform_edt(~known, return_indices=True)
            nearest_label = maps[:CLASSES, nearest[0], nearest[1]].argmax(0)
            changes.update(p, target, mask & (target.argmax(0) != nearest_label), support)
            if baselines:
                visible = .9*maps[:CLASSES, known].mean(1)+.1*prior
                totals['structure_extension'].update(maps[PRIOR:PRIOR+CLASSES], target, mask, support)
                totals['training_prior'].update(np.broadcast_to(prior[:, None, None], target.shape), target, mask, support)
                totals['observed_mean'].update(np.broadcast_to(visible[:, None, None], target.shape), target, mask, support)
    return {**{name: metric.result() for name, metric in totals.items()},
            'first_ring': first.result(), 'second_ring': second.result(), 'boundary_changes': changes.result()}


def prepare_data(manifest_path, output, cell_size, minimum_support):
    source_document = json.loads(Path(manifest_path).read_text())
    if source_document.get('format') == REGIONAL_FORMAT:
        scenes, manifest = load_regional_manifest(manifest_path, cell_size=cell_size,
                                                       minimum_support=minimum_support)
    else:
        originals, manifest = load_manifest(manifest_path)
        scenes = {split: [coarsen_scene(s, cell_size=cell_size, minimum_support=minimum_support) for s in group]
                  for split, group in originals.items()}
    records = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for split, group in scenes.items():
        for scene in group:
            path = output/(scene['sequence']+'.npz')
            if path.exists():
                raise FileExistsError(path)
            np.savez_compressed(path, **{k: v for k, v in scene.items() if k != 'metadata'})
            valid, p = scene['valid'], scene['probabilities']
            records.append(dict(split=split, sequence=scene['sequence'], file=str(path.resolve()), sha256=sha256(path),
                source_sha256=scene['source_sha256'], shape=list(valid.shape), valid_blocks=int(valid.sum()),
                dominant_class_counts=np.bincount(p.argmax(0)[valid], minlength=CLASSES).tolist(),
                class_mass=p.sum((1, 2), dtype=np.float64).tolist(), **scene['metadata']))
    result = dict(format=REGIONAL_FORMAT if source_document.get('format') == REGIONAL_FORMAT else FORMAT,
                  source_manifest=str(Path(manifest_path).resolve()),
                  source_manifest_sha256=sha256(manifest_path), records=records, simulation_used=False)
    result['source_groups'] = manifest.get('grouping', {name: list(group) for name, group in zip(CLASS_NAMES, GROUPS)})
    result['dataset'] = manifest.get('dataset', 'WildScenes')
    result['split_protocol'] = manifest.get('split_protocol', 'separate WildScenes sites')
    atomic_json(output/'manifest.json', result)
    return scenes, result


def train(data_manifest, output, *, v9_checkpoint, epochs=100, side=16, cell_size=16., frontier_m=32.,
          minimum_support=.125, samples_per_epoch=2048, validation_masks=8, batch_size=32,
          hidden=24, members=3, device='cuda', seed=20261114, patience=0, minimum_stratum_scenes=1,
          minimum_stratum_regions=1):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'metrics.jsonl').exists() or (output/'last.pt').exists():
        raise FileExistsError('Coarse frontier run already contains training results')
    if min(epochs, samples_per_epoch, validation_masks, batch_size) < 1:
        raise ValueError('Training counts must be positive')
    if patience < 0:
        raise ValueError('Patience must be nonnegative; zero disables early stopping')
    if sha256(v9_checkpoint) != V9_SHA256:
        raise ValueError('Frozen v9 checkpoint identity mismatch')
    torch.set_num_threads(2)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested CUDA unavailable; no silent fallback')
    atomic_json(output/'status.json', dict(stage='preparing_coarse_regions', pid=os.getpid()))
    scenes, manifest = prepare_data(data_manifest, output/'regions', cell_size, minimum_support)
    prior = class_prior(scenes['train'])
    training = FrontierPatches(scenes['train'], prior=prior, side=side, samples=samples_per_epoch,
                              frontier_m=frontier_m, seed=seed, minimum_stratum_scenes=minimum_stratum_scenes,
                              minimum_stratum_regions=minimum_stratum_regions)
    validation = FrontierPatches(scenes['val'], prior=prior, side=side, validation=True,
                                 masks_per_patch=validation_masks, frontier_m=frontier_m, seed=seed+1)
    train_loader = DataLoader(training, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(validation, batch_size=batch_size, num_workers=0)
    model = FrontierWorldModel(hidden, members, cell_size, frontier_m).to(device)
    weights = np.zeros(CLASSES, np.float32)
    supported = prior > 0
    weights[supported] = np.minimum(8., 1/np.sqrt(prior[supported]))
    weights /= np.sum(weights*prior)
    class_weights = torch.tensor(weights, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=3e-5)
    config = dict(format=CHECKPOINT_FORMAT, schema_version=SCHEMA_VERSION, model_architecture=ARCHITECTURE,
        architecture=model.configuration(), classes=list(CLASS_NAMES), input_channels=CHANNELS,
        source_groups=manifest['source_groups'], dataset=manifest['dataset'], split_protocol=manifest['split_protocol'],
        epochs=epochs, patience=patience, side=side, cell_size_m=cell_size, context_width_m=side*cell_size, frontier_m=frontier_m,
        minimum_support=minimum_support, samples_per_epoch=samples_per_epoch, validation_masks=validation_masks,
        batch_size=batch_size, seed=seed, device=str(device), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        class_prior=prior.tolist(), class_weights=weights.tolist(),
        unsupported_training_classes=[name for name, supported in zip(CLASS_NAMES, supported) if not supported],
        training_dominant_class_strata=[CLASS_NAMES[k] for k in sorted(training.strata)],
        minimum_stratum_scenes=minimum_stratum_scenes,
        minimum_stratum_regions=minimum_stratum_regions,
        training_sequences=[s['sequence'] for s in scenes['train']], validation_sequences=[s['sequence'] for s in scenes['val']],
        training_patch_origins=len(training.patches), validation_patch_origins=len(validation.patches),
        validation_samples=len(validation), data_manifest_sha256=manifest['source_manifest_sha256'],
        output_purpose='coarse outward terrain hypotheses for independent aerial information gathering; no vehicle targets',
        checkpoint_selection='validation class-balanced Brier only inside measured outward frontier',
        training_masks='directional half-plane/curved front; no interior random holes; no predicted-to-observed feedback',
        validation_mask_protocol='fixed and independent of hidden semantic classes',
        structural_prior='55% nearest observed + 35% local observed smoothing + 10% training class prior',
        loss='class-balanced targets with log-weight correction; measured support weights; frontier only',
        probability_calibrated=False, exploration_value_validated=False,
        mission_objective='establish a measured feasible navigation graph from current vehicle pose to final goal',
        training_stage='terrain pretraining only; graph-aware survey selection is a separate planner',
        graph_supervision_available=False, aerial_action_supervision_available=False,
        default_execution_policy='vtmp on measured graph; world-model advice disabled until navigation validation',
        simulation_used=False, ros_used=False, v9_weights_trained=False, v9_checkpoint_sha256=V9_SHA256,
        parameters=sum(p.numel() for p in model.parameters()))
    atomic_json(output/'config.json', config)
    atomic_json(output/'status.json', dict(stage='baseline_validation', pid=os.getpid(), epochs=epochs, epoch=0))
    baseline = evaluate(model, val_loader, device, prior, baselines=True)
    atomic_json(output/'baselines.json', baseline)
    best_score, best_epoch, updates, started = float('inf'), 0, 0, time.time()
    stop_reason = 'epoch_budget_completed'
    for epoch in range(1, epochs+1):
        training.epoch = epoch
        model.train()
        loss_sum = 0.
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch['maps'])
            loss = frontier_loss(logits, batch, class_weights)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite frontier training loss')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            loss_sum += float(loss.detach())
            if step == 1 or step % 16 == 0 or step == len(train_loader):
                atomic_json(output/'status.json', dict(stage='training', pid=os.getpid(), epoch=epoch, epochs=epochs,
                    batch=step, batches=len(train_loader), optimizer_steps=updates, train_loss=loss_sum/step,
                    best_epoch=best_epoch, best_balanced_brier=best_score if best_epoch else None,
                    elapsed_seconds=time.time()-started))
        scheduler.step()
        metrics = evaluate(model, val_loader, device, prior)
        score = metrics['model']['balanced_brier']
        if score is None or not np.isfinite(score):
            raise ValueError('No finite measured frontier validation score')
        improved = score < best_score
        if improved:
            best_score, best_epoch = score, epoch
        row = dict(epoch=epoch, optimizer_steps=updates, train_loss=loss_sum/len(train_loader),
            validation=metrics, best_epoch=best_epoch, best_balanced_brier=best_score,
            gradient_norm=float(gradient), elapsed_seconds=time.time()-started,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None)
        with (output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        checkpoint = {**config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'epoch': epoch, 'optimizer_steps': updates,
            'validation': metrics, 'best_epoch': best_epoch, 'best_balanced_brier': best_score,
            'baselines': {key: baseline[key] for key in ('structure_extension', 'training_prior', 'observed_mean')}}
        save_checkpoint(output/'last.pt', checkpoint)
        if improved:
            save_checkpoint(output/'best.pt', checkpoint)
        atomic_json(output/'status.json', dict(stage='training', pid=os.getpid(), epochs=epochs, **row))
        print(json.dumps(dict(stage='training', epoch=epoch, epochs=epochs, optimizer_steps=updates,
            train_loss=row['train_loss'], frontier_balanced_brier=score, best_epoch=best_epoch,
            structure_baseline_balanced_brier=baseline['structure_extension']['balanced_brier'],
            constant_baseline_balanced_brier=baseline['training_prior']['balanced_brier']), allow_nan=False), flush=True)
        if patience and epoch-best_epoch >= patience:
            stop_reason = 'validation_no_improvement'
            break
    result = dict(stage='complete', pid=os.getpid(), epochs=epoch, epochs_requested=epochs, stop_reason=stop_reason,
        optimizer_steps=updates, best_epoch=best_epoch,
        best_balanced_brier=best_score, checkpoint=str(output/'best.pt'),
        beats_structure_baseline=best_score < baseline['structure_extension']['balanced_brier'],
        beats_constant_baseline=best_score < baseline['training_prior']['balanced_brier'],
        simulation_used=False, ros_used=False, v9_weights_trained=False, exploration_value_validated=False,
        deployment_ready=False, default_execution_policy='existing_vtmp')
    atomic_json(output/'status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-manifest', type=Path, required=True)
    parser.add_argument('--v9-checkpoint', type=Path, required=True)
    for name, default in (('epochs', 100), ('side', 16), ('samples_per_epoch', 2048),
                          ('validation_masks', 8), ('batch_size', 32), ('hidden', 24), ('members', 3),
                          ('seed', 20261114), ('patience', 0), ('minimum_stratum_scenes', 1),
                          ('minimum_stratum_regions', 1)):
        parser.add_argument('--'+name.replace('_', '-'), type=int, default=default)
    parser.add_argument('--cell-size', type=float, default=16.)
    parser.add_argument('--frontier-m', type=float, default=32.)
    parser.add_argument('--minimum-support', type=float, default=.125)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    hashes = copy_source_files(source, output / 'sources', SOURCES)
    atomic_json(output/'source_manifest.json', hashes)
    atomic_json(output/'launch.json', dict(pid=os.getpid(), command=sys.argv, simulation_used=False, ros_used=False))
    try:
        options = vars(args).copy()
        options.pop('output'); options.pop('data_manifest')
        train(args.data_manifest, output, **options)
    except BaseException as exc:
        atomic_json(output/'status.json', dict(stage='failed', pid=os.getpid(), error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
