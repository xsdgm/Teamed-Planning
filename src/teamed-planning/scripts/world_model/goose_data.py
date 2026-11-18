"""Convert real GOOSE point/label pairs into supported local regional targets.

Each scan is independent in its native sensor XY frame. No invented poses,
unmeasured ground, simulated aerial visibility or navigation edges are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'world_model'

import numpy as np

from .frontier_data import (CLASS_NAMES, REGIONAL_FORMAT as FORMAT,
                            load_regional_manifest as load_manifest, outward_mask)
from .semantic_data import atomic_json, sha256

GROUPS = {
    'forest': ('forest', 'bush', 'tree_crown', 'tree_trunk', 'tree_root', 'hedge'),
    'open_ground': ('snow', 'cobble', 'leaves', 'moss', 'asphalt', 'gravel', 'soil', 'low_grass',
                    'bikeway', 'pedestrian_crossing', 'road_marking', 'sidewalk'),
    'rock': ('rock',),
    'water': ('water',),
    'other': ('traffic_cone', 'obstacle', 'street_light', 'road_block', 'traffic_light', 'curb',
              'boom_barrier', 'rail_track', 'debris', 'crops', 'building', 'wall', 'fence',
              'guard_rail', 'bridge', 'tunnel', 'pole', 'traffic_sign', 'misc_sign', 'barrier_tape',
              'high_grass', 'wire', 'container', 'barrel', 'pipe'),
}
IGNORED = ('undefined', 'ego_vehicle', 'car', 'bicycle', 'person', 'bus', 'motorcycle', 'rider',
           'animal', 'truck', 'on_rails', 'caravan', 'trailer', 'kick_scooter', 'scenery_vegetation',
           'sky', 'outlier', 'heavy_machinery', 'military_vehicle')


def ontology(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    labels = {int(row['label_key']): row['class_name'] for row in rows}
    expected = set(IGNORED).union(*(set(group) for group in GROUPS.values()))
    if len(rows) != len(labels) or len(set(labels.values())) != len(rows) or set(labels.values()) != expected:
        raise ValueError('Official ontology differs from the reviewed 64-class mapping')
    lookup = np.full(max(labels)+1, -1, np.int16)
    for key, name in labels.items():
        for index, group in enumerate(CLASS_NAMES):
            if name in GROUPS[group]:
                lookup[key] = index
    return labels, lookup


def rasterize(points, native, lookup, *, cell_size=16., fine_size=2., extent=256.,
              minimum_support=.125, minimum_points=3):
    if (points.ndim != 2 or points.shape[1] != 4 or native.shape != (len(points),)
            or not 0 < minimum_support <= 1 or minimum_points < 1
            or not 0 < fine_size <= cell_size <= extent):
        raise ValueError('Invalid point/label shape or raster configuration')
    if len(native) and (native.min() < 0 or native.max() >= len(lookup)):
        raise ValueError('Unrecognized native label ID')
    factor, side = round(cell_size/fine_size), round(extent/fine_size)
    if not np.isclose(factor*fine_size, cell_size) or not np.isclose(side*fine_size, extent) or side % factor:
        raise ValueError('Metric grid sizes must divide evenly')
    assigned = lookup[native].astype(np.int64)
    keep = (np.isfinite(points[:, :3]).all(1) & (assigned >= 0)
            & (points[:, :2] >= -extent/2).all(1) & (points[:, :2] < extent/2).all(1))
    xy = np.floor((points[keep, :2]+extent/2)/fine_size).astype(np.int64)
    flat = xy[:, 1]*side+xy[:, 0]
    counts = np.bincount(assigned[keep]*side*side+flat,
                         minlength=len(CLASS_NAMES)*side*side).reshape(len(CLASS_NAMES), side, side)
    total = counts.sum(0)
    measured = total >= minimum_points
    fine = counts.astype(np.float32)/np.maximum(total, 1)[None]
    fine[:, ~measured] = 0
    coarse_side = side//factor
    aggregate = lambda a: a.reshape(*a.shape[:-2], coarse_side, factor, coarse_side, factor).sum(axis=(-3, -1))
    coverage = aggregate(measured.astype(np.float32))
    valid = coverage >= max(1, math.ceil(factor*factor*minimum_support))
    p = aggregate(fine)/np.maximum(coverage, 1)[None]
    p[:, ~valid] = 0
    # ZIPs do not provide gravity-aligned poses. Do not interpret raw sensor Z
    # as terrain elevation across platforms or create unsupported slope labels.
    result = dict(probabilities=p.astype(np.float32), valid=valid, support=coverage/(factor*factor),
                  elevation=np.zeros_like(coverage), height_valid=np.zeros_like(valid),
                  resolution_m=float(cell_size), origin_xy=np.array([-extent/2, -extent/2], np.float64))
    audit = dict(points=len(points), points_used=int(keep.sum()), fine_measured_columns=int(measured.sum()),
                 coarse_measured_regions=int(valid.sum()),
                 native_point_counts=np.bincount(native, minlength=len(lookup)).tolist())
    return result, audit


def prepare(raw_manifest, output, *, cell_size=16., fine_size=2., extent=256., minimum_support=.125):
    raw_manifest, output = Path(raw_manifest).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    raw = json.loads(raw_manifest.read_text())
    if raw.get('format') != 'goose-raw-subset-v1' or raw.get('simulation_used') is not False:
        raise ValueError('Verified real GOOSE source manifest required')
    output.mkdir(parents=True)
    records, discarded, audits, label_tables = [], [], [], {}
    for record in raw['records']:
        archive = record['archive']
        if archive not in label_tables:
            label_tables[archive] = ontology(raw_manifest.parent/archive/'goose_label_mapping.csv')
        labels, lookup = label_tables[archive]
        paths = {key: raw_manifest.parent/record[key] for key in ('cloud', 'label')}
        for key, path in paths.items():
            if sha256(path) != record[key+'_sha256']:
                raise ValueError('Raw point/label checksum mismatch')
        points = np.fromfile(paths['cloud'], dtype='<f4').reshape(-1, 4)
        native = (np.fromfile(paths['label'], dtype='<u4') & 0xffff).astype(np.int64)
        scene, audit = rasterize(points, native, lookup, cell_size=cell_size, fine_size=fine_size,
                                  extent=extent, minimum_support=minimum_support)
        sequence = archive+'__'+record['frame']
        audit.update(sequence=sequence, split=record['split'], scenario=record['scenario'])
        audits.append(audit)
        # Support-only eligibility; hidden semantic classes never filter held-out frames.
        try:
            if scene['valid'].sum() < 16:
                raise ValueError('Fewer than 16 measured coarse regions')
            outward_mask(scene['valid'], np.random.default_rng(0), cell_size, cell_size*2)
        except ValueError as exc:
            discarded.append(dict(sequence=sequence, split=record['split'], reason=str(exc)))
            continue
        source_hash = hashlib.sha256((record['cloud_sha256']+record['label_sha256']).encode()).hexdigest()
        scene.update(sequence=sequence, source_sha256=source_hash)
        path = output/'regions'/(sequence+'.npz')
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **scene)
        metadata = dict(format=FORMAT, dataset='GOOSE-Ex' if 'ex' in archive else 'GOOSE',
            cell_size_m=cell_size, minimum_support=minimum_support, grouping=GROUPS,
            label_definition='equal-area mean of measured fine-column static semantic compositions',
            coordinate_frame='single scan native sensor XY; not a globally registered map',
            ground_height='disabled without gravity-aligned sensor poses',
            source_unmeasured='excluded; no inferred tree positions, traversability or aerial visibility',
            simulation_used=False)
        records.append(dict(split=record['split'], group=record['group'], scenario=record['scenario'],
            sequence=sequence, map=str(path.relative_to(output)), sha256=sha256(path), source_sha256=source_hash,
            metadata=metadata, valid_blocks=int(scene['valid'].sum()),
            dominant_class_counts=np.bincount(scene['probabilities'].argmax(0)[scene['valid']], minlength=5).tolist(),
            class_mass=scene['probabilities'].sum((1, 2), dtype=np.float64).tolist()))
    manifest = dict(format=FORMAT, dataset='GOOSE + GOOSE-Ex official measured subset', classes=list(CLASS_NAMES),
        cell_size_m=cell_size, minimum_support=minimum_support, fine_size_m=fine_size, extent_m=extent,
        source_manifest=str(raw_manifest), source_manifest_sha256=sha256(raw_manifest),
        grouping=GROUPS, ignored_classes=list(IGNORED), split_protocol=raw['split_protocol'],
        simulation_used=False, graph_supervision_available=False, aerial_action_supervision_available=False,
        records=records)
    atomic_json(output/'manifest.json', manifest)
    by_split = {}
    for split in sorted({r['split'] for r in records}):
        selected = [r for r in records if r['split'] == split]
        by_split[split] = dict(frames=len(selected), valid_blocks=sum(r['valid_blocks'] for r in selected),
            dominant_class_counts=np.sum([r['dominant_class_counts'] for r in selected], axis=0).tolist(),
            class_mass=np.sum([r['class_mass'] for r in selected], axis=0).tolist())
    audit = dict(classes=list(CLASS_NAMES), splits=by_split, discarded=discarded,
        warning='Scan-relative regions are correlated; counts are not unique global map area. Classes do not certify drivability.',
        frames=audits, ontology_hashes={key: sha256(raw_manifest.parent/key/'goose_label_mapping.csv') for key in label_tables})
    readiness = {}
    for k, name in enumerate(CLASS_NAMES):
        counts = {split: dict(dominant_regions=values['dominant_class_counts'][k],
            frames_with_dominant_regions=sum(r['dominant_class_counts'][k] > 0 for r in records if r['split'] == split))
            for split, values in by_split.items()}
        # These are minimum audit counts, not a statistical proof of diversity.
        enough = all(counts.get(split, {}).get('dominant_regions', 0) >= regions and
                     counts.get(split, {}).get('frames_with_dominant_regions', 0) >= frames
                     for split, regions, frames in (('train',32,5),('val',16,3),('test',16,3)))
        readiness[name] = dict(splits=counts, minimum_coverage_gate_passed=enough)
    audit['class_readiness'] = readiness
    audit['all_classes_coverage_gate_passed'] = all(v['minimum_coverage_gate_passed'] for v in readiness.values())
    audit['coverage_thresholds'] = dict(train_dominant_regions=32,train_frames=5,
                                      val_test_dominant_regions=16,val_test_frames=3)
    audit['deployment_ready'] = False
    atomic_json(output/'audit.json', audit)
    print(json.dumps(dict(splits=by_split, discarded=len(discarded)), allow_nan=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cell-size', type=float, default=16.)
    parser.add_argument('--fine-size', type=float, default=2.)
    parser.add_argument('--extent', type=float, default=256.)
    parser.add_argument('--minimum-support', type=float, default=.125)
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == '__main__':
    main()
