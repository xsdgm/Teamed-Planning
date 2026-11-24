"""Train and evaluate the local obstacle predictor from subsequently delivered reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtemp5.vtemp5_env import rule_offset
from vtemp5.vtemp5_large import VTemp5LargeObstacleEnv
from vtemp5.vtemp5_world_model import (FEATURES, FORMAT, INPUT_SCALE, OUTPUT_CENTER,
                                          OUTPUT_SCALE, OUTPUTS, LocalWorldModel, fuse_measurements)


TRAIN_START = {'detour': 130000, 'bridge': 140000}
TEST_START = {'detour': 230000, 'bridge': 240000}


def collect(terrain, seeds):
    """Observe both sides on separate matched rollouts. Label only visited sides."""
    observations, labels = [], []
    counts = Counter()
    for seed in seeds:
        for first_side in (1, -1):
            env = VTemp5LargeObstacleEnv(seed, terrain=terrain)
            history = {}
            last = np.full(3, np.nan)
            while True:
                sample = env.training_example(first_side)
                if sample is not None:
                    x, arrived = (np.asarray(v, dtype=np.float32) for v in sample)
                    history.setdefault(x.tobytes(), x.copy())
                    known = np.isfinite(arrived)
                    last[known] = arrived[known]
                if env.done:
                    break
                obs = env.observe()
                if obs['detected'] and not obs['locked_side']:
                    obs['locked_side'] = first_side
                env.step(rule_offset(obs))
            counts['rollouts'] += 1
            counts['detected_rollouts'] += bool(history)
            counts['offset_reports'] += int(np.isfinite(last[0]))
            counts['thickness_reports'] += int(np.isfinite(last[1]))
            counts['open_reports'] += int(last[2] == 1)
            counts['blocked_reports'] += int(last[2] == 0)
            if np.isfinite(last).any():
                observations.extend(history.values())
                labels.extend([last.copy() for _ in history])
    if not observations:
        raise ValueError(f'{terrain}: no subsequently delivered supervision in the selected scenes')
    x, y = np.stack(observations), np.stack(labels)
    if not np.isfinite(y).any(axis=0).all():
        raise ValueError(f'{terrain}: training sample has a completely unlabelled output')
    return x, y, dict(counts)


def train(x, y, initial, epochs, seed, metadata):
    torch.manual_seed(seed)
    layers = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), 32), torch.nn.Tanh(),
                                 torch.nn.Linear(32, 3))
    if initial is not None:
        with torch.no_grad():
            for dst, src in zip(layers.parameters(), (initial.w1, initial.b1, initial.w2, initial.b2)):
                dst.copy_(torch.as_tensor(src))
    x = torch.tensor(x / INPUT_SCALE, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    y[:, :2] = (y[:, :2] - torch.tensor(OUTPUT_CENTER)) / torch.tensor(OUTPUT_SCALE)
    mask = torch.isfinite(y)
    y = torch.nan_to_num(y)
    optimizer = torch.optim.Adam(layers.parameters(), lr=0.003)
    losses = []
    for _ in range(epochs):
        last = []
        for ids in torch.randperm(len(x)).split(256):
            pred = layers(x[ids])
            geo = torch.square(pred[:, :2] - y[ids, :2])
            bce = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, 2:], y[ids, 2:],
                                                                       reduction='none')
            errors = torch.cat((geo, bce), dim=-1)
            valid = mask[ids]
            loss = ((errors * valid).sum(0) / valid.sum(0).clamp(min=1)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last.append(loss.item())
        losses.append(float(np.mean(last)))
    model = LocalWorldModel(*(p.detach().numpy() for p in layers.parameters()), metadata=metadata)
    return model, losses


def metrics(features, truth, prediction, before_verdict):
    observed_verdict = features[:, 8] != 0
    verdict_rows = ~observed_verdict if before_verdict else observed_verdict
    verdict_rows &= np.isfinite(truth[:, 2])
    classes = truth[verdict_rows, 2]
    probability = prediction[verdict_rows, 2]
    result = {'labelled_verdicts': int(len(classes))}
    if len(classes):
        result['brier'] = float(np.square(probability - classes).mean())
        result['accuracy_at_0_5'] = float(((probability >= .5) == (classes == 1)).mean())
        result['mean_p_open'] = float(probability.mean())
        for value, label in [(0, 'blocked'), (1, 'open')]:
            rows = classes == value
            result[label] = {'count': int(rows.sum()),
                             'mean_p_open': float(probability[rows].mean()) if rows.any() else None}
    return result


def evaluate(model, eval_data, train_mean):
    x, y, counts = eval_data
    methods = {'learned': model(x),
               'constant': fuse_measurements(np.broadcast_to(train_mean, y.shape), x)}
    result = {'episodes': counts, 'distinct_side_observations': len(x), 'methods': {}}
    for name, values in methods.items():
        errors = {}
        for d, (key, unseen) in enumerate([('offset', x[:, 5] == 0), ('thickness', x[:, 7] == 0)]):
            rows = unseen & np.isfinite(y[:, d])
            errors[key] = {'count': int(rows.sum()),
                           'mae_m': float(np.abs(values[rows, d] - y[rows, d]).mean()) if rows.any() else None}
        at_detection = np.isclose(x[:, 4], x[:, 2] / 2.) & (x[:, 5] == 0) & (x[:, 8] == 0)
        result['methods'][name] = {
            'before_geometry_measured': errors,
            'unknown_side': metrics(x, y, values, True),
            'after_verdict_delivery': metrics(x, y, values, False),
            'at_detection': metrics(x[at_detection], y[at_detection], values[at_detection], True),
        }
    return result


def mean_labels(data):
    return np.nanmean(data[1], axis=0)


def make_stage(args, terrain, previous, detour_data):
    start = TRAIN_START[terrain]
    train_data = collect(terrain, range(start, start + args.train_seeds))
    tests = collect(terrain, range(TEST_START[terrain], TEST_START[terrain] + args.eval_seeds))
    x, y = train_data[:2]
    seen = [[start, start + args.train_seeds]]
    if previous is not None:
        # Retain first-stage examples so the final model remains usable on both terrains.
        x = np.concatenate((x, detour_data[0]))
        y = np.concatenate((y, detour_data[1]))
        seen = previous.metadata['seen_seed_ranges'] + seen
    metadata = {'format': FORMAT, 'training_stage': terrain, 'seen_seed_ranges': seen,
                'features': list(FEATURES), 'outputs': list(OUTPUTS),
                'seed': args.seed, 'epochs': args.epochs,
                'target_source': 'Future delivered reports in the same rollout; other label dimensions masked.'}
    model, losses = train(x, y, previous, args.epochs, args.seed, metadata)
    out = args.output / f'world-{terrain}'
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / 'model.npz')
    loaded = LocalWorldModel.load(out / 'model.npz')
    np.testing.assert_allclose(loaded(tests[0]), model(tests[0]), rtol=0., atol=0.)
    report = {
        'metadata': metadata, 'train_rollouts': train_data[2], 'stage_training_samples': len(train_data[0]),
        'optimizer_samples_including_replay': len(x), 'loss_by_epoch': losses,
        'evaluation_seed_range': [TEST_START[terrain], TEST_START[terrain] + args.eval_seeds],
        'constant_training_reference': mean_labels(train_data).tolist(),
        'held_out': evaluate(loaded, tests, mean_labels(train_data)),
        'model_sha256': hashlib.sha256((out / 'model.npz').read_bytes()).hexdigest(),
    }
    if previous:
        start = TEST_START['detour']
        detour_test = collect('detour', range(start, start + args.eval_seeds))
        report['earlier_detour_held_out'] = evaluate(loaded, detour_test, mean_labels(detour_data))
    (out / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'{terrain} {len(x)} training samples, {len(tests[0])} held-out. {out}', flush=True)
    return model, train_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-seeds', type=int, default=128)
    parser.add_argument('--eval-seeds', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--seed', type=int, default=505)
    args = parser.parse_args()
    if not (1 <= args.train_seeds <= 10000 and 1 <= args.eval_seeds <= 10000 and args.epochs >= 1):
        parser.error('seed counts must lie in [1, 10000], epochs >= 1')
    torch.set_num_threads(1)
    first, data = make_stage(args, 'detour', None, None)
    make_stage(args, 'bridge', first, data)


if __name__ == '__main__':
    main()
