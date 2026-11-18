"""Calibrate coarse hypotheses on a separate group; allow explicit unknown.

Acceptance is an empirical operating point, not a safety certificate. A fixed
held-out test evaluates it once. Neither prediction nor abstention adds map edges.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'world_model'

import numpy as np
import torch
from torch.utils.data import DataLoader

from .frontier_data import CLASS_NAMES, FrontierPatches, load_regional_manifest
from .frontier_model import load_checkpoint
from .semantic_data import atomic_json, sha256


def rescale(probabilities, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite temperature required')
    logits = np.log(np.clip(probabilities, 1e-8, 1.))/temperature
    logits -= logits.max(-1, keepdims=True)
    p = np.exp(logits)
    return p/p.sum(-1, keepdims=True)


def selective_metrics(p, y, weight, threshold, allowed_classes=None):
    accepted = p.max(1) >= threshold
    if allowed_classes is not None:
        accepted &= np.isin(p.argmax(1), allowed_classes)
    denominator = weight.sum()
    selected_weight = weight[accepted].sum()
    error = p.argmax(1) != y.argmax(1)
    return dict(cells=len(y), accepted_cells=int(accepted.sum()),
        coverage=float(selected_weight/denominator),
        dominant_error=float((error[accepted]*weight[accepted]).sum()/selected_weight) if selected_weight else None,
        brier=float((((p[accepted]-y[accepted])**2).sum(1)*weight[accepted]).sum()/selected_weight) if selected_weight else None,
        per_class={name: dict(target_cells=int((y.argmax(1) == k).sum()),
            accepted_target_cells=int((accepted & (y.argmax(1) == k)).sum())) for k, name in enumerate(CLASS_NAMES)})


def fit(p, y, weight, *, max_error=.2, minimum_coverage=.2, minimum_cells=100):
    if not 0 < max_error < 1 or not 0 < minimum_coverage <= 1 or minimum_cells < 1:
        raise ValueError('Invalid acceptance operating point')
    temperatures = np.exp(np.linspace(np.log(.5), np.log(3.), 81))
    losses = [float((-(y*np.log(rescale(p, t).clip(1e-8))).sum(1)*weight).sum()/weight.sum()) for t in temperatures]
    temperature = float(temperatures[int(np.argmin(losses))])
    adjusted = rescale(p, temperature)
    allowed = np.flatnonzero(np.bincount(y.argmax(1), minlength=len(CLASS_NAMES)) >= minimum_cells).tolist()
    curve = [dict(threshold=float(t), **selective_metrics(adjusted, y, weight, t, allowed))
             for t in np.linspace(.2, .99, 80)]
    eligible = [row for row in curve if row['accepted_cells'] >= minimum_cells
                and row['coverage'] >= minimum_coverage and row['dominant_error'] <= max_error]
    best = max(eligible, key=lambda row:row['coverage']) if eligible else None
    return dict(format='vtmp2-selective-frontier-v1', temperature=temperature,
        threshold=best['threshold'] if best else 1.000001, operating_point_accepted=best is not None,
        max_calibration_error=max_error, minimum_coverage=minimum_coverage, minimum_cells=minimum_cells,
        validated_prediction_classes=allowed,
        calibration=best, calibration_curve=curve,
        limitation='Empirical correlated-region calibration; no finite-sample risk guarantee or flight validation')


@torch.no_grad()
def collect(model, dataset, device):
    probabilities, targets, weights = [], [], []
    for batch in DataLoader(dataset, batch_size=32, num_workers=0):
        output = model.predict(batch['maps'].to(device))
        p = output['semantics'].cpu().numpy().transpose(0, 2, 3, 1)
        y = batch['target'].numpy().transpose(0, 2, 3, 1)
        mask = batch['target_mask'].numpy().astype(bool)
        probabilities.append(p[mask]); targets.append(y[mask])
        weights.append(batch['target_support'].numpy()[mask])
    return np.concatenate(probabilities), np.concatenate(targets), np.concatenate(weights)


@torch.no_grad()
def predict(model, maps, calibration):
    if calibration.get('format') != 'vtmp2-selective-frontier-v1':
        raise ValueError('Unsupported abstention calibration')
    temperature, threshold = calibration['temperature'], calibration['threshold']
    if not 0 < temperature < 100 or not .2 <= threshold <= 1.000001:
        raise ValueError('Invalid abstention calibration values')
    original = model.predict(maps)
    raw = (original['semantics'].clamp_min(1e-8).log()/temperature).softmax(1)
    known, frontier = original['observed_mask'], original['prediction_mask']
    raw = torch.where(known[:, None], original['semantics'], raw)
    raw = torch.where((known | frontier)[:, None], raw, 0.)
    accepted = frontier & (raw.amax(1) >= threshold)
    allowed = calibration.get('validated_prediction_classes', list(range(len(CLASS_NAMES))))
    if any(type(k) is not int or not 0 <= k < len(CLASS_NAMES) for k in allowed):
        raise ValueError('Invalid calibrated class list')
    accepted &= torch.isin(raw.argmax(1), torch.tensor(allowed,device=maps.device,dtype=torch.long))
    return {**original, 'raw_semantics': raw, 'frontier_mask': frontier,
            'prediction_mask': accepted, 'unknown_mask': ~(known | accepted),
            'semantics': torch.where((known | accepted)[:, None], raw, 0.),
            'entropy': -(raw*raw.clamp_min(1e-8).log()).sum(1)*frontier}


def load_selective(checkpoint, calibration_path, *, device='cpu'):
    calibration = json.loads(Path(calibration_path).read_text())
    if (calibration.get('format') != 'vtmp2-selective-frontier-v1'
            or calibration.get('checkpoint_sha256') != sha256(checkpoint)
            or calibration.get('classes') != list(CLASS_NAMES)):
        raise ValueError('Abstention calibration does not belong to this checkpoint')
    model, config = load_checkpoint(checkpoint, device=device)
    if config.get('data_manifest_sha256') != calibration.get('data_manifest_sha256'):
        raise ValueError('Calibration data manifest mismatch')
    return model, config, calibration


def calibrate(checkpoint, data_manifest, output, *, device='cpu', masks=8, max_error=.2, minimum_coverage=.2):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    model, config = load_checkpoint(checkpoint, device=device)
    scenes, _ = load_regional_manifest(data_manifest, cell_size=config['cell_size_m'], minimum_support=config['minimum_support'])
    if not all(split in scenes for split in ('calibration', 'test')):
        raise ValueError('Separate calibration and test groups required')
    datasets = {split: FrontierPatches(scenes[split], prior=np.array(config['class_prior'], np.float32),
        side=config['side'], validation=True, masks_per_patch=masks, frontier_m=config['frontier_m'], seed=20261115)
        for split in ('calibration', 'test')}
    p, y, weight = collect(model, datasets['calibration'], device)
    result = fit(p, y, weight, max_error=max_error, minimum_coverage=minimum_coverage)
    result.update(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=sha256(checkpoint),
        data_manifest_sha256=sha256(data_manifest), classes=list(CLASS_NAMES), simulation_used=False,
        calibration_groups=[s['sequence'] for s in scenes['calibration']],
        test_groups=[s['sequence'] for s in scenes['test']])
    p, y, weight = collect(model, datasets['test'], device)
    result['test_raw'] = selective_metrics(p, y, weight, 0.)
    adjusted = rescale(p, result['temperature'])
    result['test_selective'] = selective_metrics(adjusted, y, weight, result['threshold'],result['validated_prediction_classes'])
    result['test_meets_operating_point'] = bool(result['operating_point_accepted']
        and result['test_selective']['coverage'] >= minimum_coverage
        and result['test_selective']['dominant_error'] is not None
        and result['test_selective']['dominant_error'] <= max_error)
    result['deployment_ready'] = False
    atomic_json(output, result)
    print(json.dumps({k:result[k] for k in ('temperature','threshold','operating_point_accepted',
        'test_raw','test_selective','test_meets_operating_point','deployment_ready')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--masks', type=int, default=8)
    parser.add_argument('--max-error', type=float, default=.2)
    parser.add_argument('--minimum-coverage', type=float, default=.2)
    calibrate(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
