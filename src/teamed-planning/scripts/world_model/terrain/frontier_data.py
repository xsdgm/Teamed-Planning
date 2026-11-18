"""Coarse real-data regions and outward-only partial-observation tasks.

Labels originate from measured WildScenes columns or regional datasets such
as GOOSE. Unmeasured space and previous predictions supply no training labels.
The public encoder derives every feature from received observations only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .semantic_data import CLASS_NAMES as SOURCE_CLASSES
from .semantic_data import augment_patch, crop_patch, sha256, supported_patch_positions

FORMAT = 'wildscenes-coarse-frontier-v1'
REGIONAL_FORMAT = 'measured-coarse-regions-v1'
CLASS_NAMES = ('forest', 'open_ground', 'rock', 'water', 'other')
GROUPS = (('bush', 'tree-foliage', 'tree-trunk'),
          ('dirt', 'grass', 'gravel', 'mud', 'other-terrain'),
          ('rock',), ('water',), ('fence', 'log', 'other-object', 'structure'))
CLASSES = len(CLASS_NAMES)
OBSERVED = CLASSES
SUPPORT, HEIGHT, HEIGHT_VALID, DISTANCE = CLASSES+1, CLASSES+2, CLASSES+3, CLASSES+4
PRIOR = CLASSES+5
CHANNELS = PRIOR+CLASSES+2


def load_regional_manifest(path, *, cell_size, minimum_support):
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    if (document.get('format') != REGIONAL_FORMAT or document.get('simulation_used') is not False
            or document.get('classes') != list(CLASS_NAMES)
            or not np.isclose(document['cell_size_m'], cell_size)
            or not np.isclose(document['minimum_support'], minimum_support)):
        raise ValueError('Measured regional manifest/config mismatch')
    scenes = {split: [] for split in ('train', 'val', 'calibration', 'test')}
    groups, seen = {}, set()
    for record in document['records']:
        split, group = record['split'], record['group']
        if split not in scenes or not isinstance(group, str) or not group:
            raise ValueError('Invalid regional split/group')
        if group in groups and groups[group] != split:
            raise ValueError('Recording group leaked across splits')
        groups[group] = split
        identities = (record['sequence'], record['sha256'], record['source_sha256'])
        if any(x in seen for x in identities):
            raise ValueError('Duplicate scene/source across regional splits')
        seen.update(identities)
        source = (path.parent/record['map']).resolve()
        if sha256(source) != record['sha256']:
            raise ValueError('Regional map checksum mismatch')
        with np.load(source, allow_pickle=False) as data:
            scene = {key: data[key] for key in data.files}
        p, valid = scene['probabilities'], scene['valid']
        shape = valid.shape
        if (valid.dtype != bool or valid.ndim != 2 or p.shape != (len(CLASS_NAMES), *shape)
                or not np.isfinite(p).all() or (p < 0).any()
                or not np.allclose(p[:, valid].sum(0), 1., atol=1e-5)
                or np.any(p[:, ~valid] != 0)):
            raise ValueError('Invalid measured regional probabilities/mask')
        for key in ('support', 'elevation', 'height_valid'):
            if scene[key].shape != shape or not np.isfinite(scene[key]).all():
                raise ValueError('Invalid regional feature shape/value')
        support = scene['support']
        if ((support < 0).any() or (support > 1).any()
                or (support[valid] < minimum_support-1e-6).any()
                or scene['height_valid'].dtype != bool or (scene['height_valid'] & ~valid).any()
                or not np.isclose(float(scene['resolution_m']), cell_size)
                or scene['origin_xy'].shape != (2,) or not np.isfinite(scene['origin_xy']).all()
                or str(scene['sequence']) != record['sequence']
                or str(scene['source_sha256']) != record['source_sha256']):
            raise ValueError('Invalid regional support/identity')
        scene.update(sequence=record['sequence'], source_sha256=record['source_sha256'],
                     resolution_m=float(scene['resolution_m']), metadata=record['metadata'])
        scenes[split].append(scene)
    if not scenes['train'] or not scenes['val']:
        raise ValueError('Nonempty train and validation groups required')
    return {key: values for key, values in scenes.items() if values}, document


def coarsen_scene(source, *, cell_size=16., minimum_support=.125):
    resolution = float(source['resolution_m'])
    factor = int(round(cell_size/resolution))
    if (factor < 1 or not np.isclose(factor*resolution, cell_size)
            or not 0 < minimum_support <= 1):
        raise ValueError('Coarse cells must be integer multiples of source resolution')
    valid = source['valid']
    h, w = valid.shape
    hh, ww = math.ceil(h/factor), math.ceil(w/factor)

    def aggregate(values):
        padded = np.zeros((*values.shape[:-2], hh*factor, ww*factor), values.dtype)
        padded[..., :h, :w] = values
        return padded.reshape(*values.shape[:-2], hh, factor, ww, factor).sum(axis=(-3, -1))

    count = aggregate(valid.astype(np.float32))
    known = count >= max(1, math.ceil(factor*factor*minimum_support))
    grouped = np.stack([source['probabilities'][[SOURCE_CLASSES.index(name) for name in group]].sum(0)
                        for group in GROUPS])
    # Equal-area source columns contribute equally, rather than allowing dense
    # repeated scans or high tree canopies to contribute more point votes.
    probabilities = aggregate(grouped)/np.maximum(count, 1)[None]
    probabilities[:, ~known] = 0.
    height_count = aggregate(source['height_valid'].astype(np.float32))
    height_valid = (height_count > 0) & known
    elevation = aggregate(np.where(source['height_valid'], source['elevation'], 0.))/np.maximum(height_count, 1)
    elevation[~height_valid] = 0.
    return dict(probabilities=probabilities.astype(np.float32), valid=known,
        support=(count/(factor*factor)).astype(np.float32), elevation=elevation.astype(np.float32),
        height_valid=height_valid, resolution_m=float(cell_size), origin_xy=source['origin_xy'].copy(),
        sequence=source['metadata']['sequence'], source_sha256=source['metadata']['map_sha256'],
        metadata=dict(format=FORMAT, cell_size_m=float(cell_size), minimum_support=minimum_support,
            grouping={name: list(group) for name, group in zip(CLASS_NAMES, GROUPS)},
            label_definition='mean measured-column semantic composition within each coarse region',
            source_unmeasured='excluded; partial support remains explicit', simulation_used=False))


def class_prior(scenes):
    mass = sum(s['probabilities'].sum((1, 2), dtype=np.float64) for s in scenes)
    if mass.sum() <= 0:
        raise ValueError('No measured coarse terrain')
    return (mass/mass.sum()).astype(np.float32)


def encode_observation(probabilities, observed, *, class_prior, cell_size=16., frontier_m=32.,
                       support=None, elevation=None, height_valid=None):
    """Encode observations and structural continuation features, without labels.

Returned distance is measured between coarse cell centres. Predictions are
eligible only in the explicitly returned frontier mask, never the whole crop.
"""
    p, known = np.asarray(probabilities, np.float32), np.asarray(observed, bool)
    prior = np.asarray(class_prior, np.float32)
    if (known.ndim != 2 or p.shape != (CLASSES, *known.shape)
            or prior.shape != (CLASSES,) or not np.isfinite(prior).all()
            or (prior < 0).any() or not np.isclose(prior.sum(), 1.)
            or not np.isfinite(p[:, known]).all() or (p[:, known] < 0).any()
            or not np.allclose(p[:, known].sum(0), 1., atol=1e-5)
            or not np.isfinite([cell_size, frontier_m]).all() or not 0 < cell_size <= frontier_m):
        raise ValueError('Invalid received coarse semantic observation')
    maps = np.zeros((CHANNELS, *known.shape), np.float32)
    maps[:CLASSES, known] = p[:, known]
    maps[OBSERVED] = known
    maps[DISTANCE] = 2.
    maps[PRIOR:PRIOR+CLASSES] = prior[:, None, None]
    if support is None:
        support = known.astype(np.float32)
    support = np.asarray(support)
    if support.shape != known.shape or not np.isfinite(support[known]).all() or np.any((support[known] <= 0) | (support[known] > 1)):
        raise ValueError('Observed region support must be in (0,1]')
    maps[SUPPORT, known] = support[known]
    if elevation is not None and height_valid is not None:
        z, z_known = np.asarray(elevation), np.asarray(height_valid, bool)
        if z.shape != known.shape or z_known.shape != known.shape:
            raise ValueError('Height arrays must match the observation')
        z_known = z_known & known
        if not np.isfinite(z[z_known]).all():
            raise ValueError('Invalid observed elevation')
        if z_known.any():
            maps[HEIGHT, z_known] = np.clip((z[z_known]-np.median(z[z_known]))/10., -5., 5.)
        maps[HEIGHT_VALID] = z_known
    if not known.any():
        return maps, np.zeros(known.shape, bool)
    distance, nearest = distance_transform_edt(~known, sampling=cell_size, return_indices=True)
    nearest_p = maps[:CLASSES, nearest[0], nearest[1]]
    density = gaussian_filter(known.astype(np.float32), 1., mode='constant')
    smooth = np.stack([gaussian_filter(channel, 1., mode='constant') for channel in maps[:CLASSES]])
    smooth = np.divide(smooth, density[None], out=nearest_p.copy(), where=density[None] > 1e-6)
    continuation = .55*nearest_p+.35*smooth+.10*prior[:, None, None]
    continuation = np.maximum(continuation, 1e-6)
    continuation /= continuation.sum(0, keepdims=True)
    maps[PRIOR:PRIOR+CLASSES] = continuation
    maps[DISTANCE] = np.minimum(distance/frontier_m, 2.)
    yy, xx = np.indices(known.shape)
    maps[PRIOR+CLASSES] = np.clip((nearest[1]-xx)*cell_size/frontier_m, -2., 2.)
    maps[PRIOR+CLASSES+1] = np.clip((nearest[0]-yy)*cell_size/frontier_m, -2., 2.)
    frontier = ~known & (distance <= frontier_m+1e-6)
    return maps, frontier


def patch_positions(valid, side, stride, minimum_cells=16):
    if side < 8 or side % 4 or stride < 1:
        raise ValueError('Patch side must be a multiple of four, at least eight')
    return supported_patch_positions(valid, side, stride, minimum_cells)


def crop(scene, row, col, side):
    return crop_patch(scene, row, col, side,
                      fields=('probabilities', 'valid', 'support', 'elevation', 'height_valid'))


def outward_mask(valid, rng, cell_size, frontier_m):
    """A connected half-plane or gently curved front; no interior random holes."""
    yy, xx = np.indices(valid.shape)
    for _ in range(64):
        angle = rng.uniform(0, 2*np.pi)
        u = xx*np.cos(angle)+yy*np.sin(angle)
        v = -xx*np.sin(angle)+yy*np.cos(angle)
        if rng.random() < .5:
            u += rng.uniform(-1., 1.)*np.sin(v/3.)
        threshold = np.quantile(u[valid], rng.uniform(.35, .7))
        known = valid & (u <= threshold)
        distance = distance_transform_edt(~known, sampling=cell_size)
        target = valid & ~known & (distance <= frontier_m+1e-6)
        if known.sum() >= 6 and target.sum() >= 3:
            return known
    for projection in (xx, yy):
        for threshold in np.unique(projection[valid]):
            known = valid & (projection <= threshold)
            if known.sum() < 6:
                continue
            distance = distance_transform_edt(~known, sampling=cell_size)
            if (valid & ~known & (distance <= frontier_m+1e-6)).sum() >= 3:
                return known
    raise ValueError('No supported outward frontier in this patch')


class FrontierPatches:
    """Class-balanced training crops, fixed unbalanced validation frontiers."""
    def __init__(self, scenes, *, prior, side=16, samples=2048, validation=False,
                 masks_per_patch=8, seed=20261113, frontier_m=32., minimum_stratum_scenes=1,
                 minimum_stratum_regions=1):
        self.scenes, self.prior, self.side = scenes, prior, side
        self.validation, self.seed, self.epoch = validation, seed, 0
        self.frontier_m = frontier_m
        if minimum_stratum_scenes < 1 or minimum_stratum_regions < 1:
            raise ValueError('Positive stratum scene count required')
        self.cell_size = scenes[0]['resolution_m']
        self.patches = []
        for i, scene in enumerate(scenes):
            for row, col in patch_positions(scene['valid'], side, max(1, side//2)):
                patch = crop(scene, row, col, side)
                # Eligibility is measured support only; no semantic class is
                # used to filter validation windows.
                try:
                    outward_mask(patch['valid'], np.random.default_rng(seed+i+row*10000+col), self.cell_size, frontier_m)
                except ValueError:
                    continue
                self.patches.append((i, row, col))
        if not self.patches:
            raise ValueError('No measured frontier patches available')
        self.strata = {}
        for index, (i, r, c) in enumerate(self.patches):
            s = scenes[i]
            valid = s['valid'][r:r+side, c:c+side]
            labels = s['probabilities'][:, r:r+side, c:c+side].argmax(0)
            for label in np.unique(labels[valid]):
                self.strata.setdefault(int(label), []).append(index)
        if not validation:
            region_counts = sum(np.bincount(s['probabilities'].argmax(0)[s['valid']], minlength=CLASSES)
                                for s in scenes)
            self.strata = {label: indices for label, indices in self.strata.items()
                           if len({self.patches[index][0] for index in indices}) >= minimum_stratum_scenes
                           and region_counts[label] >= minimum_stratum_regions}
            if not self.strata:
                raise ValueError('No class has sufficient independent scan support for balancing')
        self.samples = len(self.patches)*masks_per_patch if validation else samples
        if self.samples < 1:
            raise ValueError('Positive sample count required')

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0 if self.validation else self.epoch, index]))
        desired = None
        if self.validation:
            selected = index % len(self.patches)
        else:
            desired = int(rng.choice(sorted(self.strata)))
            selected = int(rng.choice(self.strata[desired]))
        i, row, col = self.patches[selected]
        patch = crop(self.scenes[i], row, col, self.side)
        if not self.validation:
            patch = augment_patch(patch, rng)
        # Offline training may balance hidden target classes. Validation never
        # chooses masks using hidden semantics, and the encoder sees none.
        for _ in range(1 if self.validation else 16):
            known = outward_mask(patch['valid'], rng, self.cell_size, self.frontier_m)
            maps, frontier = encode_observation(patch['probabilities'], known, class_prior=self.prior,
                cell_size=self.cell_size, frontier_m=self.frontier_m, support=patch['support'],
                elevation=patch['elevation'], height_valid=patch['height_valid'])
            target_mask = patch['valid'] & frontier
            if desired is None or np.any((patch['probabilities'].argmax(0) == desired) & target_mask):
                break
        return dict(maps=maps, target=patch['probabilities'], target_mask=target_mask.astype(np.float32),
                    target_support=patch['support'], frontier_mask=frontier.astype(np.float32))
