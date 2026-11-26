"""Evaluate received-evidence predictions on independent shore and bridge maps."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label
import torch

from run_v91_extent_pilot import encode
from v91_extent_network import FORMAT, ObstaclePredictor


@dataclass
class ModelBundle:
    model: ObstaclePredictor
    presence_threshold: float
    mask_threshold: float
    shape: tuple
    spatial_prior: np.ndarray | None


def load_model(checkpoint):
    path = Path(checkpoint)
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['format'] != FORMAT:
        raise ValueError('Expected a frozen V9.1 large-obstacle checkpoint')
    model = ObstaclePredictor(state['channels']).eval().requires_grad_(False)
    model.load_state_dict(state['state_dict'], strict=True)
    prior = None
    train_file = path.parent.parent / 'train.npz'
    metrics_path = path.parent / 'metrics.json'
    if train_file.is_file() and metrics_path.is_file():
        original = json.loads(metrics_path.read_text())
        with np.load(train_file, allow_pickle=False) as train:
            prior = train['mask'].mean(axis=0) >= original['train_prior_thresholds'][1]
    return ModelBundle(model, float(state['presence_threshold']), float(state['mask_threshold']),
                       (state['channels'], *state['grid_shape']), prior)


@torch.inference_mode()
def model_fields(bundle, snapshot):
    """Forward inference is isolated from the map generator and scoring masks."""
    x, world_xy = encode(snapshot)
    if x.shape != bundle.shape:
        raise ValueError(f'Expected observation shape {bundle.shape}, got {x.shape}')
    batch = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0)
    presence, extent = bundle.model(batch)
    return x, world_xy, float(presence.sigmoid().item()), extent[0].sigmoid().numpy()


def ratio(n, d):
    return float(n / d) if d else None


def iou(truth, prediction, domain):
    positive = truth & domain
    if not np.any(positive):
        return None
    return ratio((positive & prediction).sum(), ((truth | prediction) & domain).sum())


def mouth_metrics(scene, xy, prediction, unseen, end):
    """Probe the mouth and the true adjacent banks only when all are in view.

    The geometric probes select scoring pixels inside the model output window,
    irrespective of whether they were ever delivered to the car. A free route
    through the *mouth patch* is distinct from a drivable full bridge.
    """
    from v91_lake_cases import bank_near_mouth_mask, bridge_mouth_mask

    mouth = bridge_mouth_mask(scene, xy, end=end)
    bank = bank_near_mouth_mask(scene, xy, end=end)
    forward = np.asarray(scene['forward'])
    reference = np.asarray(scene['bridge_front'])
    if end == 'exit':
        reference = reference + scene['bridge_depth_m'] * forward
    delta = xy - reference
    along = delta @ forward
    side = delta @ np.asarray(scene['left'])
    banks = [bank & (side < 0), bank & (side >= 0)]
    near = mouth & (along <= -2.)
    far = mouth & (along >= 2.)
    eligible = (int(mouth.sum()) >= 4 and all(b.sum() >= 2 for b in banks)
                and near.any() and far.any())
    result = {f'{end}_eligible': int(eligible)}
    if not eligible:
        return result
    blocked = float(prediction[mouth].mean())
    bank_recall = [float(prediction[shoulder].mean()) for shoulder in banks]
    components, _ = label(mouth & ~prediction)   # face-adjacent cells, no diagonal shortcut
    near_ids = np.unique(components[near])
    far_ids = np.unique(components[far])
    spans = bool(np.any(np.intersect1d(near_ids, far_ids) > 0))
    result.update({
        f'{end}_predicted_blocked_fraction': blocked,
        f'{end}_unseen_blocked_fraction': ratio((prediction & mouth & unseen).sum(), (mouth & unseen).sum()),
        f'{end}_banks_water_recall': float(prediction[bank].mean()),
        f'{end}_deck_observed_fraction': float((~unseen)[mouth].mean()),
        f'{end}_open_path_through_mouth': int(spans),
        f'{end}_open_with_banks': int(spans and blocked < .5 and min(bank_recall) >= .5),
        f'{end}_cells': int(mouth.sum()),
        f'{end}_water_cells': int(bank.sum()),
    })
    return result


def score_fields(scene, snapshot, bundle, fields):
    """Score a previously computed prediction without resampling observations."""
    from v91_lake_cases import water_mask, bridge_mask
    x, xy, p, grid_prob = fields
    truth = np.asarray(water_mask(scene, xy), dtype=bool)
    pred = grid_prob >= bundle.mask_threshold
    unseen = (x[0] == 0) & (x[1] == 0)
    has_water = bool(truth.any())
    detected = p >= bundle.presence_threshold
    domain = np.ones_like(unseen)
    row = dict(seed=int(scene['seed']), kind=scene['kind'], frame=int(snapshot['frame']),
               water_in_window=int(has_water), presence_p=p, presence_detected=int(detected),
               presence_correct=int(detected == has_water), water_recall_if_present=int(detected) if has_water else None,
               specificity_if_dry=int(not detected) if not has_water else None,
               iou=iou(truth, pred, domain), unknown_iou=iou(truth, pred, unseen),
               observed_only_iou=iou(truth, x[1] > 0, domain),
               visible_fraction=float((~unseen).mean()),
               water_observed_fraction=ratio((truth & ~unseen).sum(), truth.sum()))
    if bundle.spatial_prior is not None:
        row['prior_iou'] = iou(truth, bundle.spatial_prior, domain)
        row['prior_unknown_iou'] = iou(truth, bundle.spatial_prior, unseen)
    if scene['kind'] == 'bridge':
        deck = bridge_mask(scene, xy)
        row['bridge_deck_predicted_blocked_fraction'] = ratio((pred & deck).sum(), deck.sum())
        for end in ('entry', 'exit'):
            row.update(mouth_metrics(scene, xy, pred, unseen, end))
    return row


def score_snapshots(scene, snapshots, model_bundle):
    """Return scalar per-frame records; score different control arms separately."""
    return [score_fields(scene, frame, model_bundle, model_fields(model_bundle, frame)) for frame in snapshots]


BASE_FIELDS = ('iou', 'unknown_iou', 'observed_only_iou', 'prior_iou', 'prior_unknown_iou',
               'presence_correct', 'water_recall_if_present', 'specificity_if_dry',
               'visible_fraction', 'water_observed_fraction')
BRIDGE_FIELDS = ('bridge_deck_predicted_blocked_fraction', *(
    f'{end}_{metric}' for end in ('entry', 'exit')
    for metric in ('predicted_blocked_fraction', 'unseen_blocked_fraction', 'banks_water_recall',
                   'deck_observed_fraction', 'open_path_through_mouth', 'open_with_banks')))


def _macro(cases, names, rng):
    summary = {}
    for name in names:
        values, frames = [], 0
        for case in cases:
            metrics = [row[name] for row in case if row.get(name) is not None]
            if metrics:
                values.append(np.mean(metrics))
                frames += len(metrics)
        if values:
            sample = np.asarray(values, dtype=float)
            # Cluster bootstrap over independent maps, after averaging frames per map.
            interval = None
            if len(sample) >= 2:
                means = rng.choice(sample, size=(2000, len(sample)), replace=True).mean(axis=1)
                interval = np.percentile(means, [2.5, 97.5]).tolist()
            summary[name] = dict(mean=float(sample.mean()), maps=len(sample), frames=frames,
                                 map_bootstrap_95=interval)
        else:
            summary[name] = dict(mean=None, maps=0, frames=0, map_bootstrap_95=None)
    return summary


def summarize_cases(cases):
    """Aggregate each independent map once. First/last refer to recorded snapshots."""
    if any(not case for case in cases):
        raise ValueError('Cannot summarize maps without observations')
    seeds = []
    for case in cases:
        if len({(row['seed'], row['kind']) for row in case}) != 1:
            raise ValueError('One trajectory must come from one map')
        if len(set(row['frame'] for row in case)) != len(case):
            raise ValueError('Duplicate frame within one trajectory')
        seeds.append(case[0]['seed'])
    if len(set(seeds)) != len(seeds):
        raise ValueError('Summarize each control arm independently; repeated map seeds found')
    rng = np.random.default_rng(9125)
    result = {}
    for kind in sorted({case[0]['kind'] for case in cases}):
        group = [sorted(case, key=lambda row:row['frame']) for case in cases if case[0]['kind'] == kind]
        fields = BASE_FIELDS + (BRIDGE_FIELDS if kind == 'bridge' else ())
        result[kind] = dict(maps=len(group), frames=sum(map(len, group)), all=_macro(group, fields, rng),
                            first=_macro([[case[0]] for case in group], fields, rng),
                            last=_macro([[case[-1]] for case in group], fields, rng))
    return result
