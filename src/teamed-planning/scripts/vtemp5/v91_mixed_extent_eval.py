"""Compare two frozen world models on identical held-out vehicle observations."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from run_v91_extent_pilot import CELL, GRID, REAR, digest, save_json
from v91_extent_network import predict
from v91_lake_cases import bridge_mask, make_lake_case, water_mask
from v91_lake_metrics import iou, mouth_metrics, ratio


EXTENT_METRICS = ('iou', 'unknown_iou', 'presence_recall', 'presence_specificity',
                  'unknown_false_positive_rate_on_empty_window')
MOUTH_METRICS = ('open_with_banks', 'open_path_through_mouth', 'banks_water_recall',
                'predicted_blocked_fraction')
BINARY_METRICS = {'presence_recall', 'presence_specificity', 'open_with_banks',
                  'open_path_through_mouth'}


def grid_centres(car, goal):
    forward = goal - car
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    right = np.array([forward[1], -forward[0]])
    rows, cols = np.indices((GRID, GRID))
    along = REAR + (GRID - rows - .5) * CELL
    across = (cols + .5 - GRID / 2) * CELL
    return car + along[..., None] * forward + across[..., None] * right


def checkpoint_info(path):
    weights = torch.load(path, map_location='cpu', weights_only=True)
    return dict(path=str(path.resolve()), sha256=digest(path),
                presence_threshold=float(weights['presence_threshold']),
                mask_threshold=float(weights['mask_threshold']))


def predictions(inputs, checkpoint):
    p, mask = predict(checkpoint['path'], inputs)
    if not (np.isfinite(p).all() and np.isfinite(mask).all()):
        raise ValueError('Checkpoint generated nonfinite predictions')
    return p >= checkpoint['presence_threshold'], mask >= checkpoint['mask_threshold']


def score_frame(mask, unseen, detected, prediction, scene, xy):
    positive = bool(mask.any())
    row = dict(iou=iou(mask, prediction, np.ones_like(mask)),
               unknown_iou=iou(mask, prediction, unseen),
               presence_recall=int(detected) if positive else None,
               presence_specificity=int(not detected) if not positive else None,
               unknown_false_positive_rate_on_empty_window=(
                   ratio((prediction & unseen).sum(), unseen.sum()) if not positive else None))
    if scene is not None and scene['kind'] == 'bridge':
        deck = bridge_mask(scene, xy)
        row['bridge_deck_predicted_blocked_fraction'] = ratio((deck & prediction).sum(), deck.sum())
        for end in ('entry', 'exit'):
            row.update(mouth_metrics(scene, xy, prediction, unseen, end))
    return row


def paired_metric(rows, name, rng, binary=False):
    """Pair frames, average within each map, then bootstrap paired map deltas."""
    by_seed = defaultdict(list)
    for row in rows:
        values = [row[arm].get(name) for arm in ('old', 'new')]
        if any(v is None for v in values):
            if values != [None, None]:
                raise ValueError(f'Unmatched denominator for {name}')
            continue
        by_seed[row['seed']].append(values)
    maps = len(by_seed)
    frames = sum(map(len, by_seed.values()))
    summary = dict(maps=maps, frames=frames, old=None, new=None,
                   delta_new_minus_old=None, paired_map_bootstrap_95=None)
    if maps:
        sample = np.array([np.mean(pairs, axis=0) for pairs in by_seed.values()])
        delta = sample[:, 1] - sample[:, 0]
        summary.update(old=float(sample[:, 0].mean()), new=float(sample[:, 1].mean()),
                       delta_new_minus_old=float(delta.mean()))
        if maps >= 2:
            draws = rng.choice(delta, size=(2000, maps)).mean(1)
            summary['paired_map_bootstrap_95'] = np.percentile(draws, [2.5, 97.5]).tolist()
    if binary:
        counts = np.zeros(2, dtype=int)
        for pairs in by_seed.values():
            counts += np.sum(pairs, axis=0, dtype=int)
        summary.update(old_hits=int(counts[0]), new_hits=int(counts[1]))
    return summary


def report_rows(rows):
    rng = np.random.default_rng(9271)
    result = {}
    for kind in sorted({r['terrain'] for r in rows}):
        group = [r for r in rows if r['terrain'] == kind]
        terrain = dict(maps=len({r['seed'] for r in group}), frames=len(group))
        result[kind] = terrain
        for name in EXTENT_METRICS:
            terrain[name] = paired_metric(group, name, rng, name in BINARY_METRICS)
        if kind != 'bridge':
            continue
        key = 'bridge_deck_predicted_blocked_fraction'
        terrain[key] = paired_metric(group, key, rng)
        for end in ('entry', 'exit'):
            fractions = end + '_deck_observed_fraction'
            eligible = [r for r in group if r['old'][end + '_eligible']]
            strata = dict(all=eligible,
                          unseen=[r for r in eligible if r['old'][fractions] == 0],
                          partial=[r for r in eligible if 0 < r['old'][fractions] < 1],
                          complete=[r for r in eligible if r['old'][fractions] == 1])
            terrain[end] = {
                stratum: {key: paired_metric(records, f'{end}_{key}', rng, key in BINARY_METRICS)
                          for key in MOUTH_METRICS}
                for stratum, records in strata.items()}
    return result


def score(path, checkpoints, legacy=False):
    with np.load(path, allow_pickle=False) as archive:
        x = archive['x']
        masks = archive['mask'].astype(bool)
        seeds = archive['seed']
        frames = archive['frame'] if 'frame' in archive else archive['step']
        if not np.array_equal(archive['presence'].astype(bool), masks.any(axis=(1, 2))):
            raise ValueError(f'Inconsistent presence and extent targets in {path}')
        if legacy:
            kinds = ['land'] * len(seeds)
        else:
            kinds = archive['kind'].astype(str)
            cars, goals = archive['car_xy'], archive['goal_xy']
    if len(set(zip(seeds, frames))) != len(seeds):
        raise ValueError(f'Duplicate map/frame in {path}')
    models = {arm: predictions(x, info) for arm, info in checkpoints.items()}
    scenes, rows = {}, []
    for i, (seed, frame, kind) in enumerate(zip(seeds, frames, kinds)):
        terrain = kind.split('_')[0]
        scene = xy = None
        if terrain in ('shore', 'bridge'):
            if seed not in scenes:
                scenes[seed] = make_lake_case(seed, terrain)
            scene = scenes[seed]
            xy = grid_centres(cars[i], goals[i])
            if not np.array_equal(water_mask(scene, xy), masks[i]):
                raise ValueError(f'Projected water mask differs on {seed=} {frame=}')
        elif terrain != 'land':
            raise ValueError(f'Unexpected terrain {kind!r}')
        row = dict(terrain=terrain, seed=int(seed), frame=int(frame))
        unseen = (x[i, 0] == 0) & (x[i, 1] == 0)
        for arm, (detected, predicted) in models.items():
            row[arm] = score_frame(masks[i], unseen, detected[i], predicted[i], scene, xy)
        rows.append(row)
    return dict(path=str(path.resolve()), sha256=digest(path), maps=len(set(seeds)),
                frames=len(rows), by_terrain=report_rows(rows),
                by_map={str(seed): report_rows([r for r in rows if r['seed'] == seed])
                        for seed in sorted(set(seeds))})


def check_splits(test, checkpoints, regression):
    def seeds(path):
        with np.load(path, allow_pickle=False) as archive:
            return set(archive['seed'].tolist())
    test_seeds = seeds(test)
    regression_seeds = seeds(regression) if regression else set()
    for folder in {test.parent, *(p.parent.parent for p in checkpoints)}:
        for name in ('train.npz', 'validation.npz'):
            path = folder / name
            if path.exists() and (test_seeds | regression_seeds) & seeds(path):
                raise ValueError(f'Test map overlaps {path}')
    if test_seeds & regression_seeds:
        raise ValueError('Regression and mixed test maps overlap')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('test', 'old', 'new', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--land-regression', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    check_splits(args.test, (args.old, args.new), args.land_regression)
    checkpoints = {name: checkpoint_info(path)
                   for name, path in [('old', args.old), ('new', args.new)]}
    result = dict(checkpoints=checkpoints, aggregation=(
        'Each mean weights eligible maps equally after averaging eligible frames within each map. '
        'IoU requires target cells in the specified domain. Binary hits are raw eligible-frame counts. '
        'Bridge probes and 4-connected mouth/bank criteria match v91_lake_metrics.mouth_metrics. '
        'Unseen/partial/complete describe observation coverage of the true mouth patch. '
        'Paired bootstrap percentiles resample maps with replacement (2000 draws).'),
        test=score(args.test, checkpoints))
    if args.land_regression:
        result['land_regression'] = score(args.land_regression, checkpoints, legacy=True)
    save_json(args.output, result)
    print(f'Scored {result["test"]["maps"]} held-out maps: {args.output}', flush=True)


if __name__ == '__main__':
    main()
