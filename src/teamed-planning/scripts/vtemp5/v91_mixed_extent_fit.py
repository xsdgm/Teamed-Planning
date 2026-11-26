"""Train the V9.1 occupancy predictor from scratch on mixed terrain."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from v91_extent_network import FORMAT, THRESHOLDS, ObstaclePredictor, _classification, _report, _thresholds


TERRAINS = ('land', 'shore', 'bridge')
CHECKPOINT_METRIC = (
    'Average of the three terrain scores. Each score averages positive-frame extent IoU and '
    'positive-unseen IoU. Land additionally averages clear-frame specificity (area fraction). '
    'Bridge additionally averages the smaller of dry-mouth specificity and adjacent-water '
    'recall. Each statistic averages snapshots within a map first, then maps.'
)


def weight_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode() + b'\0')
        digest.update(str((array.dtype, array.shape)).encode() + b'\0')
        digest.update(array.tobytes())
    return digest.hexdigest()


def read_split(path):
    with np.load(path, allow_pickle=False) as data:
        x = data['x']
        y = data['presence'].reshape(-1)
        masks = data['mask']
        seeds = data['seed'].reshape(-1)
        kinds = data['kind'].astype(str).reshape(-1)
        mouth = data['entry_mouth'].astype(bool) | data['exit_mouth'].astype(bool)
        bank = data['entry_bank'].astype(bool) | data['exit_bank'].astype(bool)
    if (x.shape != (len(seeds), 5, 48, 48) or masks.shape != x[:, 0].shape
            or mouth.shape != masks.shape or bank.shape != masks.shape
            or kinds.shape != seeds.shape or y.shape != seeds.shape):
        raise ValueError(f'Unexpected observation or label dimensions in {path}')
    if (not np.isin(x, (0, 1)).all() or not np.isin(masks, (0, 1)).all()
            or set(np.unique(y)) != {0, 1} or np.any(masks[y == 0])):
        raise ValueError(f'Invalid grid, presence, or extent labels in {path}')
    if set(kinds) != {'land_large', 'land_clear', 'bridge', 'shore'}:
        raise ValueError(f'Expected both land strata, shore and bridge in {path}')
    if (np.any(mouth & masks.astype(bool)) or np.any(bank & ~masks.astype(bool))
            or np.any((mouth | bank)[kinds != 'bridge']) or not mouth.any() or not bank.any()):
        raise ValueError(f'Inconsistent bridge-region supervision in {path}')
    terrain = np.where(np.char.startswith(kinds, 'land'), 'land', kinds)
    weights = np.zeros(len(seeds), dtype=np.float32)
    for kind in TERRAINS:
        group = terrain == kind
        ids, idx, counts = np.unique(seeds[group], return_inverse=True, return_counts=True)
        weights[group] = len(seeds) / (len(TERRAINS) * len(ids) * counts[idx])
    return dict(x=torch.from_numpy(x).float(), y=torch.from_numpy(y).float(),
                mask=torch.from_numpy(masks).float(), mouth=torch.from_numpy(mouth),
                bank=torch.from_numpy(bank), weight=torch.from_numpy(weights), terrain=terrain,
                seeds=seeds)


def mean_over_pixels(loss, region):
    return (loss * region).sum((1, 2)) / region.sum((1, 2)).clamp_min(1)


def training_loss(outputs, y, target, mouth, bank, sample_weights, pos_weight):
    presence_logits, logits = outputs
    presence = F.binary_cross_entropy_with_logits(presence_logits, y, reduction='none')
    pixels = F.binary_cross_entropy_with_logits(logits, target, reduction='none', pos_weight=pos_weight)
    probs = logits.sigmoid()
    size = target.sum((1, 2))
    dice = (2 * (probs * target).sum((1, 2)) + 1) / (probs.sum((1, 2)) + size + 1)
    # The two sides of the segmentation boundary have comparable loss weight
    # even at a narrow opening. These truth masks never enter the predictor.
    mouth_loss = mean_over_pixels(F.softplus(logits), mouth)
    bank_loss = mean_over_pixels(F.softplus(-logits), bank)
    per_frame = presence + pixels.mean((1, 2)) + (1 - dice) * (size > 0) + .5 * (mouth_loss + bank_loss)
    return (per_frame * sample_weights).mean()


@torch.inference_mode()
def infer(model, x, batch_size, device):
    model.eval()
    ps, ms = [], []
    for batch in x.split(batch_size):
        p, m = model(batch.to(device))
        ps.append(p.sigmoid().cpu().numpy())
        ms.append(m.sigmoid().cpu().numpy())
    return np.concatenate(ps), np.concatenate(ms)


def macro(numerator, denominator, valid, seeds):
    ids = np.unique(seeds[valid])
    if not len(ids):
        return None
    values = numerator[valid] / denominator[valid]
    return float(np.mean([values[seeds[valid] == seed].mean() for seed in ids]))


def extent_score(pred, dataset):
    truth = dataset['mask'].numpy().astype(bool)
    unknown = ~(dataset['x'][:, :2].numpy().astype(bool).any(1))
    mouth, bank = dataset['mouth'].numpy(), dataset['bank'].numpy()
    seeds, terrain = dataset['seeds'], dataset['terrain']
    axes = (1, 2)
    hits = truth & pred
    combined = truth | pred
    regions = {'iou': (hits.sum(axes), combined.sum(axes), truth.any(axes)),
               'unknown_iou': ((hits & unknown).sum(axes), (combined & unknown).sum(axes),
                               (truth & unknown).any(axes))}
    scores = {}
    details = {}
    for kind in TERRAINS:
        subset = terrain == kind
        vals = {name: macro(num, den, valid & subset, seeds)
                for name, (num, den, valid) in regions.items()}
        if kind == 'land':
            dry = subset & ~truth.any(axes)
            cells = np.full(len(seeds), truth.shape[1] * truth.shape[2])
            vals['clear_specificity'] = macro((~pred).sum(axes), cells, dry, seeds)
        if kind == 'bridge':
            dry_acc = macro((~pred & mouth).sum(axes), mouth.sum(axes), subset & mouth.any(axes), seeds)
            wet_acc = macro((pred & bank).sum(axes), bank.sum(axes), subset & bank.any(axes), seeds)
            vals['bridge_boundary_min'] = min(dry_acc, wet_acc)
        scores[kind] = np.mean([v for v in vals.values() if v is not None])
        details[kind] = vals
        if kind == 'bridge':
            details[kind].update(dry_mouth_specificity=dry_acc, adjacent_water_recall=wet_acc)
    return float(np.mean(list(scores.values()))), details


def train(args):
    if (args.epochs < 1 or args.threads < 1 or args.batch_size < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError('Expected positive training parameters')
    out = args.data / 'model'
    if any((out / name).exists() for name in ('best.pt', 'metrics.json')):
        raise FileExistsError(f'Training output already exists in {out}')
    paths = [args.data / f'{name}.npz' for name in ('train', 'validation')]
    tr, val = map(read_split, paths)
    if set(tr['seeds']) & set(val['seeds']):
        raise ValueError('Training and validation share a map')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    shuffle = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    model = ObstaclePredictor(5).to(device)
    initial_hash = weight_digest(model.state_dict())
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    tensors = [tr[key].to(device) for key in ('x', 'y', 'mask', 'mouth', 'bank', 'weight')]
    x, _, mask, *_ = tensors
    pos_weight = ((mask.numel() - mask.sum()) / mask.sum()).sqrt().clamp(1., 10.)
    labels = val['y'].numpy().astype(bool)
    best_score, best_state, history = -math.inf, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for ids in torch.randperm(len(x), generator=shuffle).split(args.batch_size):
            ids = ids.to(device)
            optim.zero_grad(set_to_none=True)
            loss = training_loss(model(x[ids]), *(t[ids] for t in tensors[1:]), pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optim.step()
            batch_losses.append(loss.detach() * len(ids))
        p, m = infer(model, val['x'], args.batch_size, device)
        presence_threshold = max(THRESHOLDS, key=lambda t: _classification(labels, p, t)['balanced_accuracy'])
        balanced = _classification(labels, p, presence_threshold)['balanced_accuracy']
        extent_metric, threshold, details = max(
            ((score, t, detail) for t in THRESHOLDS for score, detail in [extent_score(m >= t, val)]),
            key=lambda candidate: candidate[0])
        score = .85 * extent_metric + .15 * balanced
        record = dict(epoch=epoch, train_loss=float(torch.stack(batch_losses).sum().item() / len(x)),
                      validation_score=score, extent_score=extent_metric, presence_balanced_accuracy=balanced,
                      mask_threshold=threshold, presence_threshold=presence_threshold, terrain=details)
        history.append(record)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best = copy.deepcopy(record)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f'Epoch {epoch}/{args.epochs}, mixed validation score {score:.4f}, best {best_score:.4f}', flush=True)
    if best_state is None:
        raise ValueError('No valid training epoch')
    last_hash = weight_digest(model.state_dict())
    best_hash = weight_digest(best_state)
    if best_hash == initial_hash:
        raise RuntimeError('Selected model equals the random initialization')
    model.load_state_dict(best_state)
    thresholds = (best['presence_threshold'], best['mask_threshold'])
    prior_presence = tr['y'].mean().item()
    prior_mask = tr['mask'].mean(0).numpy()
    masks = val['mask'].numpy().astype(bool)
    prior_thresholds = _thresholds(labels, np.full(len(labels), prior_presence), masks,
                                   np.broadcast_to(prior_mask, masks.shape))
    summary = _report((val['x'], val['y'], val['mask'], val['seeds']),
                      infer(model, val['x'], args.batch_size, device), thresholds,
                      prior_presence, prior_mask, prior_thresholds)
    checkpoint = dict(format=FORMAT, channels=5, grid_shape=[48, 48], state_dict=best_state,
                      presence_threshold=thresholds[0], mask_threshold=thresholds[1],
                      epoch=best['epoch'])
    metrics = dict(format=FORMAT, initialization='random_from_scratch', seed=args.seed,
                   initial_state_sha256=initial_hash, best_state_sha256=best_hash,
                   last_epoch_state_sha256=last_hash, epochs=args.epochs, best_epoch=best['epoch'],
                   device=str(device), threads=args.threads, batch_size=args.batch_size,
                   learning_rate=args.learning_rate, channel_count=5, grid_shape=[48, 48],
                   train_maps=len(np.unique(tr['seeds'])), train_frames=len(x),
                   best_validation=best, history=history, mask_threshold=thresholds[1],
                   presence_threshold=thresholds[0], train_prior_thresholds=list(prior_thresholds),
                   validation=summary, checkpoint_selection='.85 * terrain score + .15 * balanced presence accuracy. ' + CHECKPOINT_METRIC,
                   dataset_sha256={p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    out.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out / 'best.pt.tmp')
    (out / 'metrics.json.tmp').write_text(json.dumps(metrics, allow_nan=False, indent=2) + '\n')
    (out / 'best.pt.tmp').replace(out / 'best.pt')
    (out / 'metrics.json.tmp').replace(out / 'metrics.json')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--seed', type=int, default=951)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    result = train(parser.parse_args())
    print(f"Selected epoch {result['best_epoch']}; validation score {result['best_validation']['validation_score']:.4f}")
