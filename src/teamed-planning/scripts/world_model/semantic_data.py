"""Original WildScenes -> measured semantic BEV evidence, without a simulator.

Cells retain the distribution of labelled points in the vertical column, not
binary occupancy or certified vehicle traversability. Sky/unlabelled points
are excluded. No interpolation invents a label outside measured support.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'world_model'
from .hierarchical_sources import atomic_json, file_sha256 as sha256

FORMAT = 'wildscenes-semantic-columns-v1'
NATIVE_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14)
CLASS_NAMES = ('bush', 'dirt', 'fence', 'grass', 'gravel', 'log', 'mud',
               'other-object', 'other-terrain', 'rock', 'structure',
               'tree-foliage', 'tree-trunk', 'water')
GROUND_IDS = (1, 3, 4, 6, 8, 14)
CLASSES = len(CLASS_NAMES)
CHANNELS = CLASSES + 3  # semantic composition, observed, relative height, height-valid


def raw_reader():
    # Load only raw point/pose utilities. Importing the simulator package itself
    # would initialize unrelated environment code and is intentionally avoided.
    path = Path(__file__).resolve().parents[1]/'simulator/prepare_wildscenes.py'
    spec = importlib.util.spec_from_file_location('_semantic_wildscenes_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_sequence(sequence, output, *, resolution=2., minimum_points=3, progress=None):
    sequence, output = Path(sequence).resolve(), Path(output).resolve()
    if not np.isfinite(resolution) or resolution <= 0 or minimum_points < 1:
        raise ValueError('Positive resolution and minimum point count required')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    reader = raw_reader()
    pose_path = sequence/'poses3d.csv'
    pose_hash = sha256(pose_path)
    poses = reader.load_poses(pose_path)
    pairs = reader.scan_pairs(sequence, poses)
    lower, upper = np.full(2, np.inf), np.full(2, -np.inf)
    label_counts = np.zeros(256, np.int64)
    scans = []

    def report(stage, index):
        row = dict(stage=stage, sequence=sequence.name, scan=index+1, total_scans=len(pairs))
        atomic_json(output/'progress.json', row)
        if progress:
            progress(row)

    # First pass validates every scan and fixes the metric frame. Hashes are
    # retained to reject source changes during the rasterization pass.
    for i, (key, cp, lp) in enumerate(pairs):
        cloud_hash, label_hash = sha256(cp), sha256(lp)
        xyz, labels = reader.read_scan(key, cp, lp, poses)
        keep = np.isin(labels, NATIVE_IDS)
        if keep.any():
            lower = np.minimum(lower, xyz[keep, :2].min(0))
            upper = np.maximum(upper, xyz[keep, :2].max(0))
        label_counts += np.bincount(labels, minlength=256)
        scans.append(dict(timestamp=key, cloud=cp.name, label=lp.name,
                          cloud_sha256=cloud_hash, label_sha256=label_hash, points=len(labels)))
        if i % 100 == 0 or i == len(pairs)-1:
            report('validating_raw_scans', i)
    if not np.isfinite(lower).all():
        raise ValueError('No labelled terrain points')
    origin = np.floor(lower/resolution)*resolution
    width, height = (np.floor((upper-origin)/resolution).astype(int)+1).tolist()
    if width*height > 10_000_000:
        raise ValueError('Requested semantic raster exceeds ten million cells')
    counts = np.zeros((CLASSES, height*width), np.uint64)
    ground_sum, ground_count = np.zeros(height*width, np.float64), np.zeros(height*width, np.uint64)
    lookup = np.full(256, -1, np.int16)
    lookup[list(NATIVE_IDS)] = np.arange(CLASSES)
    for i, ((key, cp, lp), record) in enumerate(zip(pairs, scans)):
        if sha256(cp) != record['cloud_sha256'] or sha256(lp) != record['label_sha256']:
            raise ValueError('Raw scans changed during preparation')
        xyz, labels = reader.read_scan(key, cp, lp, poses)
        keep = lookup[labels] >= 0
        xyz, labels = xyz[keep], labels[keep]
        ij = np.floor((xyz[:, :2]-origin)/resolution).astype(np.int64)
        cells = ij[:, 1]*width+ij[:, 0]
        if ((ij < 0).any() or (ij[:, 0] >= width).any() or (ij[:, 1] >= height).any()):
            raise ValueError('Raw scan moved outside validated bounds')
        np.add.at(counts, (lookup[labels], cells), 1)
        ground = np.isin(labels, GROUND_IDS)
        np.add.at(ground_sum, cells[ground], xyz[ground, 2])
        np.add.at(ground_count, cells[ground], 1)
        if i % 100 == 0 or i == len(pairs)-1:
            report('building_semantic_evidence', i)
    total = counts.sum(0)
    valid = total >= minimum_points
    probabilities = np.divide(counts, total[None], out=np.zeros(counts.shape, np.float32), where=valid[None])
    height_valid = (ground_count >= minimum_points) & valid
    elevation = np.divide(ground_sum, ground_count, out=np.zeros_like(ground_sum), where=height_valid)
    path = output/'semantic_map.npz'
    if sha256(pose_path) != pose_hash:
        raise ValueError('Source poses changed during preparation')
    np.savez_compressed(path, probabilities=probabilities.reshape(CLASSES, height, width),
        valid=valid.reshape(height, width), elevation=elevation.astype(np.float32).reshape(height, width),
        height_valid=height_valid.reshape(height, width), point_count=total.reshape(height, width),
        origin_xy=origin, resolution_m=np.asarray(resolution), sequence=np.asarray(sequence.name))
    atomic_json(output/'source_scans.json', scans)
    metadata = dict(format=FORMAT, sequence=sequence.name, source=str(sequence),
        resolution_m=resolution, shape=[height, width], origin_xy=origin.tolist(),
        native_ids=list(NATIVE_IDS), classes=list(CLASS_NAMES), scans=len(scans),
        raw_points=int(label_counts.sum()), native_label_counts={str(k): int(label_counts[k]) for k in range(256) if label_counts[k]},
        class_point_counts={name: int(label_counts[native]) for name, native in zip(CLASS_NAMES, NATIVE_IDS)},
        unsupported_classes=[name for name, native in zip(CLASS_NAMES, NATIVE_IDS) if not label_counts[native]],
        valid_cells=int(valid.sum()), measured_area_m2=float(valid.sum()*resolution**2),
        semantic_definition='labelled-point composition of each vertical metric column; not traversability',
        unknown_source_cells='excluded_from_supervision', minimum_points=minimum_points,
        source_scans_sha256=sha256(output/'source_scans.json'), poses_sha256=pose_hash,
        map_sha256=sha256(path), simulation_used=False)
    atomic_json(output/'metadata.json', metadata)
    atomic_json(output/'progress.json', dict(stage='complete', sequence=sequence.name))
    return metadata


def load_scene(path, expected_sha=None):
    path = Path(path)
    metadata = json.loads((path.parent/'metadata.json').read_text())
    if (metadata.get('format') != FORMAT or metadata.get('native_ids') != list(NATIVE_IDS)
            or metadata.get('classes') != list(CLASS_NAMES)
            or metadata['map_sha256'] != (expected_sha or metadata['map_sha256'])
            or sha256(path) != metadata['map_sha256']):
        raise ValueError('Semantic map contract/hash mismatch')
    with np.load(path, allow_pickle=False) as data:
        scene = {key: data[key].copy() for key in data.files}
    p, valid = scene['probabilities'], scene['valid']
    if (p.shape != (CLASSES, *valid.shape) or not np.isfinite(p).all()
            or (p < 0).any() or not np.allclose(p.sum(0), valid, atol=1e-5)
            or scene['height_valid'].shape != valid.shape or scene['elevation'].shape != valid.shape
            or not np.isfinite(scene['elevation']).all() or (scene['height_valid'] & ~valid).any()
            or scene['sequence'].item() != metadata['sequence']):
        raise ValueError('Invalid semantic evidence arrays')
    scene['metadata'] = metadata
    return scene


def patch_coordinates(valid, side, stride=None, minimum_valid=128):
    if side < 16 or side % 8:
        raise ValueError('Patch side must be a multiple of eight, at least 16')
    stride = stride or side//2
    if stride < 1:
        raise ValueError('Positive stride required')
    return supported_patch_positions(valid, side, stride, minimum_valid)


def supported_patch_positions(valid, side, stride, minimum_valid):
    """Include the final edge-aligned patch even when stride does not divide the map."""
    height, width = valid.shape
    rows = sorted(set(range(0, max(1, height-side+1), stride)) | {max(0, height-side)})
    cols = sorted(set(range(0, max(1, width-side+1), stride)) | {max(0, width-side)})
    return [(row, col) for row in rows for col in cols
            if int(valid[row:row+side, col:col+side].sum()) >= minimum_valid]


def crop_patch(scene, row, col, side, *, fields=('probabilities', 'valid', 'elevation', 'height_valid')):
    result = {}
    for key in fields:
        source = scene[key]
        target = np.zeros((*source.shape[:-2], side, side), source.dtype)
        part = source[..., row:row+side, col:col+side]
        target[..., :part.shape[-2], :part.shape[-1]] = part
        result[key] = target
    return result


def augment_patch(patch, rng):
    """Apply the same random rotation and reflection to every field."""
    turns = int(rng.integers(4))
    flip = rng.random() < .5
    result = {}
    for key, value in patch.items():
        rotated = np.rot90(value, turns, axes=(-2, -1))
        result[key] = (np.flip(rotated, -1) if flip else rotated).copy()
    return result


def observation_mask(valid, rng, *, minimum_cells=16):
    """Hide a continuous directional region/rectangle, not random isolated pixels."""
    yy, xx = np.indices(valid.shape)
    for _ in range(24):
        theta = rng.uniform(0, 2*np.pi)
        projection = xx*np.cos(theta)+yy*np.sin(theta)
        fraction = rng.uniform(.25, .75)
        cut = np.quantile(projection[valid], fraction)
        hidden = projection > cut
        if rng.random() < .35:
            # A contiguous rectangle allows context on several sides.
            r, c = np.argwhere(valid)[rng.integers(valid.sum())]
            radius = rng.uniform(.2, .5)*min(valid.shape)
            hidden = (abs(xx-c) < radius) & (abs(yy-r) < radius)
        observed = valid & ~hidden
        if observed.sum() >= minimum_cells and (valid & hidden).sum() >= minimum_cells:
            return observed
    # The deterministic half-plane fallback also preserves spatial continuity.
    projection = xx+yy/(valid.shape[0]+1)
    observed = valid & (projection <= np.median(projection[valid]))
    if observed.sum() < minimum_cells or (valid & ~observed).sum() < minimum_cells:
        raise ValueError('Insufficient measured support for a completion task')
    return observed


def encode_observation(probabilities, observed, elevation=None, height_valid=None):
    """One training/deployment encoder; masked labels/heights never enter inputs."""
    probabilities = np.asarray(probabilities, np.float32)
    observed = np.asarray(observed, bool)
    if (probabilities.shape != (CLASSES, *observed.shape) or observed.ndim != 2
            or not np.isfinite(probabilities[:, observed]).all()
            or (probabilities[:, observed] < 0).any()
            or not np.allclose(probabilities[:, observed].sum(0), 1., atol=1e-5)):
        raise ValueError('Invalid observed semantic probabilities')
    maps = np.zeros((CHANNELS, *observed.shape), np.float32)
    maps[:CLASSES, observed] = probabilities[:, observed]
    maps[CLASSES] = observed
    if elevation is not None and height_valid is not None:
        elevation, height_valid = np.asarray(elevation), np.asarray(height_valid, bool)
        if elevation.shape != observed.shape or height_valid.shape != observed.shape:
            raise ValueError('Elevation and height-valid shapes must match observations')
        height_known = height_valid & observed
        if not np.isfinite(elevation[height_known]).all():
            raise ValueError('Nonfinite observed elevation')
        if height_known.any():
            reference = float(np.median(elevation[height_known]))
            maps[CLASSES+1, height_known] = np.clip((elevation[height_known]-reference)/10., -5., 5.)
        maps[CLASSES+2] = height_known
    return maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-sequence', default='K-01')
    parser.add_argument('--val-sequence', default='V-01')
    parser.add_argument('--resolution', type=float, default=2.)
    args = parser.parse_args()
    if args.train_sequence.split('-')[0] == args.val_sequence.split('-')[0]:
        parser.error('Use separate forest locations for training and validation')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started, records = time.time(), []
    try:
        for split, sequence in (('train', args.train_sequence), ('val', args.val_sequence)):
            def progress(row):
                print(json.dumps(dict(split=split, **row)), flush=True)
                atomic_json(output/'status.json', dict(split=split, **row))
            meta = prepare_sequence(args.source_root/sequence, output/sequence,
                                    resolution=args.resolution, progress=progress)
            records.append(dict(split=split, sequence=sequence, site=sequence.split('-')[0],
                map=str((output/sequence/'semantic_map.npz').resolve()), sha256=meta['map_sha256']))
        atomic_json(output/'manifest.json', dict(format=FORMAT, source='original WildScenes3d',
                    resolution_m=args.resolution, records=records, simulation_used=False))
        atomic_json(output/'status.json', dict(stage='complete', elapsed_seconds=time.time()-started))
    except BaseException as exc:
        atomic_json(output/'status.json', dict(stage='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
