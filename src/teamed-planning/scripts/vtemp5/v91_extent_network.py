"""Fit a vehicle-frame obstacle predictor on received evidence from held-out maps."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FORMAT = 'v91_local_large_obstacle_v1'
THRESHOLDS = sorted((.02, .05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9),
                    key=lambda value: abs(value - .5))


def conv_block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.SiLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.SiLU())


class ObstaclePredictor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local = conv_block(channels + 2, 12)
        self.mid = conv_block(12, 20)
        self.deep = conv_block(20, 32)
        self.decode_mid = conv_block(32 + 20, 16)
        self.decode_local = conv_block(16 + 12, 12)
        self.mask = nn.Conv2d(12, 1, 1)
        self.presence = nn.Sequential(nn.Linear(32 * 5, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        # Coordinates are fixed by the public car-frame grid and carry no terrain labels.
        h, w = x.shape[-2:]
        ys, xs = torch.meshgrid(torch.linspace(-1., 1., h, device=x.device),
                                torch.linspace(-1., 1., w, device=x.device), indexing='ij')
        xy = torch.stack((ys, xs)).expand(x.shape[0], -1, -1, -1)
        local = self.local(torch.cat((x, xy), dim=1))
        mid = self.mid(F.avg_pool2d(local, 2))
        deep = self.deep(F.avg_pool2d(mid, 2))
        context = torch.cat((F.adaptive_avg_pool2d(deep, 2).flatten(1), deep.amax(dim=(2, 3))), dim=1)
        up = F.interpolate(deep, size=mid.shape[2:], mode='bilinear', align_corners=False)
        up = self.decode_mid(torch.cat((mid, up), dim=1))
        up = F.interpolate(up, size=local.shape[2:], mode='bilinear', align_corners=False)
        mask = self.mask(self.decode_local(torch.cat((local, up), dim=1))).squeeze(1)
        return self.presence(context).squeeze(1), mask


def _dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        x = np.asarray(archive['x'], np.float32)
        y = np.asarray(archive['presence'], np.float32).reshape(-1)
        mask = np.asarray(archive['mask'], np.float32)
        seeds = np.asarray(archive['seed']).reshape(-1)
    if (x.ndim != 4 or len(x) == 0 or x.shape[1] < 2 or min(x.shape[2:]) < 8 or
            mask.shape != (len(x), *x.shape[2:]) or len(y) != len(x) or len(seeds) != len(x)):
        raise ValueError(f'Incompatible pilot observation or target shapes in {path}')
    if (not np.isfinite(x).all() or not np.isin(y, (0, 1)).all() or
            not np.isin(mask, (0, 1)).all() or np.any(mask[y == 0])):
        raise ValueError(f'Expected finite observations, binary targets, and empty masks on negative examples in {path}')
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), seeds


def _loss(outputs, targets, masks, pixel_weight):
    presence_logits, mask_logits = outputs
    classification = F.binary_cross_entropy_with_logits(presence_logits, targets)
    pixels = F.binary_cross_entropy_with_logits(mask_logits, masks, pos_weight=pixel_weight)
    truth_sum = masks.sum((1, 2))
    has_mask = truth_sum > 0
    probs = mask_logits.sigmoid()
    dice = (2 * (probs * masks).sum((1, 2)) + 1) / (probs.sum((1, 2)) + truth_sum + 1)
    overlap_loss = ((1 - dice) * has_mask).sum() / has_mask.sum().clamp_min(1)
    return classification + pixels + overlap_loss


@torch.inference_mode()
def _probabilities(model, x, batch_size):
    classified, segmented = [], []
    model.eval()
    for batch in x.split(batch_size):
        p, m = model(batch)
        classified.append(p.sigmoid().numpy())
        segmented.append(m.sigmoid().numpy())
    return np.concatenate(classified), np.concatenate(segmented)


def predict(checkpoint_path, x):
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if saved['format'] != FORMAT:
        raise ValueError('Incompatible world-model checkpoint')
    model = ObstaclePredictor(saved['channels'])
    model.load_state_dict(saved['state_dict'])
    x = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4 or x.shape[1] != saved['channels'] or list(x.shape[2:]) != saved['grid_shape']:
        raise ValueError('Observation shape differs from training')
    return _probabilities(model, x, 32)


def _ratio(a, b):
    return float(a / b) if b else None


def _classification(y, p, threshold):
    detected = p >= threshold
    tn, fp = int((~y & ~detected).sum()), int((~y & detected).sum())
    fn, tp = int((y & ~detected).sum()), int((y & detected).sum())
    recall, specificity = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, recall=recall, specificity=specificity,
                balanced_accuracy=(recall + specificity) / 2 if recall is not None and specificity is not None else None,
                brier=float(np.mean((p - y.astype(float)) ** 2)))


def _extent(mask, prediction, unseen, y):
    axes = (1, 2)
    hit, union = mask & prediction, mask | prediction
    pos = mask.any(axis=axes)
    unknown_pos = (mask & unseen).any(axis=axes)
    def positive_mean(valid, intersection, total):
        if not valid.any():
            return None
        return float(np.mean(intersection[valid].sum(axis=axes) / total[valid].sum(axis=axes)))
    return dict(iou_micro=_ratio(hit.sum(), union.sum()),
                positive_iou=positive_mean(pos, hit, union),
                positive_unknown_iou=positive_mean(unknown_pos, hit & unseen, union & unseen),
                positive_mask_frames=int(pos.sum()), positive_unknown_mask_frames=int(unknown_pos.sum()),
                negative_unknown_false_positive_rate=_ratio((prediction[~y] & unseen[~y]).sum(), unseen[~y].sum()))


def _thresholds(labels, probs, mask, pixel_probs):
    presence_t = max(THRESHOLDS, key=lambda t: _classification(labels, probs, t)['balanced_accuracy'])
    def micro_iou(t):
        predicted = pixel_probs >= t
        return _ratio((predicted & mask).sum(), (predicted | mask).sum()) or 0.
    return presence_t, max(THRESHOLDS, key=micro_iou)


def _report(dataset, predictions, thresholds, label_prior, mask_prior, prior_thresholds):
    x, labels, masks, seeds = dataset
    labels, masks = labels.numpy().astype(bool), masks.numpy().astype(bool)
    x = x.numpy()
    unseen = ~((x[:, 0] > 0) | (x[:, 1] > 0))
    p, m = predictions
    prior_p = np.full(len(labels), label_prior)
    prior_m = np.broadcast_to(mask_prior, masks.shape)
    return dict(maps=int(len(np.unique(seeds))), frames=len(x), positive_frames=int(labels.sum()),
                network=dict(presence=_classification(labels, p, thresholds[0]),
                             extent=_extent(masks, m >= thresholds[1], unseen, labels)),
                train_prior=dict(presence=_classification(labels, prior_p, prior_thresholds[0]),
                                 extent=_extent(masks, prior_m >= prior_thresholds[1], unseen, labels)),
                observed_occupied_only=dict(extent=_extent(masks, x[:, 1] > 0, unseen, labels)))


def train(train_path, validation_path, test_path, output_dir, *, epochs=50, seed=53, batch_size=16,
          learning_rate=0.001, threads=2):
    """Fit, select using validation, then score the frozen model once on independent test maps."""
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0 or threads < 1:
        raise ValueError('Training arguments must be positive')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'best.pt').exists() or (out / 'metrics.json').exists():
        raise FileExistsError(f'Training outputs already exist in {out}')
    paths = [Path(p) for p in (train_path, validation_path, test_path)]
    groups = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            groups.append(set(archive['seed'].ravel().tolist()))
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError('Scene seeds overlap between splits')
    tr, va = _dataset(paths[0]), _dataset(paths[1])
    if tr[0].shape[1:] != va[0].shape[1:] or any(len(torch.unique(d[1])) != 2 for d in (tr, va)):
        raise ValueError('Train and validation need matching observation shapes and both presence classes')
    if not tr[2].any() or not va[2].any():
        raise ValueError('Train and validation both need positive extent labels')
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    shuffle = torch.Generator().manual_seed(seed)
    x, y, masks, _ = tr
    model = ObstaclePredictor(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    weight = ((masks.numel() - masks.sum()) / masks.sum()).sqrt().clamp(1., 10.)
    best_loss, best_epoch, best_state = float('inf'), None, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for ids in torch.randperm(len(x), generator=shuffle).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(x[ids]), y[ids], masks[ids], weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(ids)
        model.eval()
        with torch.inference_mode():
            risk = sum(_loss(model(xb), yb, mb, weight).item() * len(xb)
                       for xb, yb, mb in zip(*(t.split(batch_size) for t in va[:3]))) / len(va[0])
        history.append(dict(epoch=epoch, train_loss=total / len(x), validation_loss=risk))
        if risk < best_loss:
            best_loss, best_epoch, best_state = risk, epoch, copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f'World model epoch {epoch}/{epochs}, validation loss {risk:.4f}', flush=True)
    if best_state is None:
        raise ValueError('No finite validation loss')
    model.load_state_dict(best_state)
    val_p, val_m = _probabilities(model, va[0], batch_size)
    val_labels, val_masks = va[1].numpy().astype(bool), va[2].numpy().astype(bool)
    thresholds = _thresholds(val_labels, val_p, val_masks, val_m)
    label_prior, mask_prior = y.mean().item(), masks.mean(0).numpy()
    prior_thresholds = _thresholds(val_labels, np.full(len(val_labels), label_prior), val_masks,
                                   np.broadcast_to(mask_prior, val_masks.shape))
    checkpoint = dict(format=FORMAT, channels=x.shape[1], grid_shape=list(x.shape[2:]), state_dict=best_state,
                      presence_threshold=thresholds[0], mask_threshold=thresholds[1], epoch=best_epoch)
    test = _dataset(paths[2])
    if test[0].shape[1:] != x.shape[1:]:
        raise ValueError('Test observation shape differs from training')
    metrics = dict(format=FORMAT, seed=seed, epochs=epochs, best_epoch=best_epoch, batch_size=batch_size,
                   learning_rate=learning_rate, threads=threads, presence_threshold=thresholds[0],
                   mask_threshold=thresholds[1], train_prior_thresholds=list(prior_thresholds),
                   channel_count=x.shape[1], grid_shape=list(x.shape[2:]),
                   train_maps=len(groups[0]), train_frames=len(x), history=history,
                   aggregation='Each snapshot is one example. Splits are disjoint by map seed. Positive unknown IoU is the mean over snapshots with at least one unobserved target cell.',
                   validation=_report(va, (val_p, val_m), thresholds, label_prior, mask_prior, prior_thresholds),
                   test=_report(test, _probabilities(model, test[0], batch_size), thresholds, label_prior,
                                mask_prior, prior_thresholds))
    metrics['dataset_sha256'] = {k: hashlib.sha256(path.read_bytes()).hexdigest()
                                for k, path in zip(('train', 'validation', 'test'), paths)}
    torch.save(checkpoint, out / 'best.pt.tmp')
    (out / 'metrics.json.tmp').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    (out / 'best.pt.tmp').replace(out / 'best.pt')
    (out / 'metrics.json.tmp').replace(out / 'metrics.json')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('train', 'validation', 'test', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    result = train(args.train, args.validation, args.test, args.output, epochs=args.epochs, seed=args.seed,
                   batch_size=args.batch_size, learning_rate=args.learning_rate, threads=args.threads)
    print(json.dumps({'best_epoch': result['best_epoch'], 'test': result['test']}, indent=2))
