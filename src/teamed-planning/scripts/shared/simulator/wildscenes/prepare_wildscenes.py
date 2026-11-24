"""Convert a complete WildScenes semantic sequence to a metric planar map.

This is a 2D semantic/geometric abstraction, not camera simulation or a
three-dimensional dynamics model.  Ground support is required for free space;
unobserved regions stay unknown.  Heights are used only to separate canopy
from obstacles in the ground vehicle's vertical envelope.
"""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation


LABELS = {0: 'bush', 1: 'dirt', 2: 'fence', 3: 'grass', 4: 'gravel',
          5: 'log', 6: 'mud', 7: 'other-object', 8: 'other-terrain',
          9: 'rock', 10: 'sky', 11: 'structure', 12: 'tree-foliage',
          13: 'tree-trunk', 14: 'water', 255: 'unlabelled'}
GROUND = (1, 3, 4, 6, 8, 14)
HAZARD = (6, 14)
SOLID = (2, 5, 7, 9, 11, 13)
VEGETATION = (0, 12)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def timestamp_key(value):
    """Match decimal timestamps without float rounding or trailing-zero drift."""
    key = format(Decimal(value), 'f')
    return key.rstrip('0').rstrip('.') if '.' in key else key


def load_poses(path):
    # Official files are space-separated, with a quoted human-readable
    # datetime index and an exact decimal timestamp matching cloud filenames.
    with Path(path).open(newline='') as f:
        records = list(csv.DictReader(f, delimiter=' '))
    required = {'timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'}
    if not records or not required.issubset(records[0]):
        raise ValueError(f'Unexpected WildScenes pose schema: {path}')
    poses = {}
    for row in records:
        key = timestamp_key(row['timestamp'])
        if key in poses:
            raise ValueError(f'Duplicate pose timestamp {key}')
        t = np.array([float(row[k]) for k in ('x', 'y', 'z')])
        q = np.array([float(row[k]) for k in ('qx', 'qy', 'qz', 'qw')])
        if not np.isfinite(t).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > .01:
            raise ValueError(f'Invalid pose {key}')
        poses[key] = (Rotation.from_quat(q).as_matrix(), t)
    return poses


def scan_pairs(sequence_dir, poses):
    clouds = {timestamp_key(p.stem): p for p in (sequence_dir/'Clouds').glob('*.bin')}
    labels = {timestamp_key(p.stem): p for p in (sequence_dir/'Labels').glob('*.label')}
    if set(clouds) != set(poses) or set(labels) != set(poses):
        raise ValueError(f'Incomplete sequence: {len(poses)} poses, {len(clouds)} clouds, '
                         f'{len(labels)} labels; missing clouds={len(set(poses)-set(clouds))}, '
                         f'missing labels={len(set(poses)-set(labels))}')
    return [(key, clouds[key], labels[key]) for key in poses]


def read_scan(key, cloud_path, label_path, poses):
    xyz = np.fromfile(cloud_path, dtype='<f4')
    classes = np.fromfile(label_path, dtype='<i4')
    if xyz.size % 3 or xyz.size//3 != len(classes):
        raise ValueError(f'Cloud/label count mismatch: {cloud_path}')
    xyz = xyz.reshape(-1, 3)
    if not np.isfinite(xyz).all():
        raise ValueError(f'Nonfinite point: {cloud_path}')
    if not np.isin(classes, list(LABELS)).all():
        raise ValueError(f'Unknown label in {label_path}: {np.unique(classes)}')
    r, t = poses[key]
    return xyz.astype(np.float64) @ r.T + t, classes


def prepare(sequence_dir, output, resolution=.5, interpolation=1.,
            vehicle_height=2., minimum_ground_points=3, minimum_obstacle_points=3):
    sequence_dir, output = Path(sequence_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    poses = load_poses(sequence_dir/'poses3d.csv')
    pairs = scan_pairs(sequence_dir, poses)
    trajectory_native = np.array([t for r, t in poses.values()])
    # Use all measured points for extent; do not truncate unseen margins or
    # change scale.  Sensor range is not used to invent free ground.
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    class_counts = np.zeros(256, dtype=np.int64)
    records, total_points = [], 0
    for i, (key, cp, lp) in enumerate(pairs):
        xyz, labels = read_scan(key, cp, lp, poses)
        lower = np.minimum(lower, xyz.min(0))
        upper = np.maximum(upper, xyz.max(0))
        class_counts += np.bincount(labels, minlength=256)
        total_points += len(xyz)
        records.append(dict(timestamp=key, points=len(xyz), cloud=cp.name,
                            cloud_sha256=digest(cp), label_sha256=digest(lp)))
        if i % 200 == 0:
            print(json.dumps(dict(phase='extent_and_integrity', scans=i+1, total=len(pairs))), flush=True)
    native_origin = np.floor((lower[:2]-40.)/resolution)*resolution
    world = math.ceil(float(np.max(upper[:2]-native_origin)+40.)/resolution)*resolution
    n = int(round(world/resolution))
    shape = (n, n)
    ground_sum = np.zeros(n*n, np.float64)
    ground_count = np.zeros(n*n, np.uint32)
    hazards = np.zeros(n*n, np.uint32)

    def indices(xyz):
        ij = np.floor((xyz[:, :2]-native_origin)/resolution).astype(np.int64)
        if (ij < 0).any() or (ij >= n).any():
            raise ValueError('Point outside computed map bounds')
        return ij[:, 1]*n + ij[:, 0]

    for i, (key, cp, lp) in enumerate(pairs):
        xyz, labels = read_scan(key, cp, lp, poses)
        ids = indices(xyz)
        ground = np.isin(labels, GROUND)
        np.add.at(ground_sum, ids[ground], xyz[ground, 2])
        np.add.at(ground_count, ids[ground], 1)
        np.add.at(hazards, ids[np.isin(labels, HAZARD)], 1)
        if i % 200 == 0:
            print(json.dumps(dict(phase='ground_support', scans=i+1, total=len(pairs))), flush=True)
    ground_count = ground_count.reshape(shape)
    ground_evidence = ground_count >= minimum_ground_points
    if not ground_evidence.any():
        raise ValueError('No ground support in sequence')
    z = np.divide(ground_sum.reshape(shape), ground_count,
                  out=np.full(shape, np.nan), where=ground_count>0)
    distance, nearest = distance_transform_edt(~ground_evidence, sampling=resolution, return_indices=True)
    supported = distance <= interpolation + 1e-9
    terrain = z[tuple(nearest)].astype(np.float32)
    terrain[~supported] = np.nan
    del distance, nearest, ground_sum
    obstacles = np.zeros(n*n, np.uint32)
    obstacle_by_label = {str(k): 0 for k in SOLID+VEGETATION}
    low_points = 0
    for i, (key, cp, lp) in enumerate(pairs):
        xyz, labels = read_scan(key, cp, lp, poses)
        ids = indices(xyz)
        height = xyz[:, 2] - terrain.ravel()[ids]
        # Solids in the ground vehicle's vertical envelope.  Low logs/rocks
        # count down to the ground; low bush/foliage above 0.35m blocks too.
        solid = np.isin(labels, SOLID) & (height >= -.15) & (height <= vehicle_height)
        vegetation = np.isin(labels, VEGETATION) & (height >= .35) & (height <= vehicle_height)
        low = (solid | vegetation) & np.isfinite(height)
        np.add.at(obstacles, ids[low], 1)
        low_points += int(low.sum())
        for label in SOLID+VEGETATION:
            obstacle_by_label[str(label)] += int((low & (labels == label)).sum())
        if i % 200 == 0:
            print(json.dumps(dict(phase='ground_obstacles', scans=i+1, total=len(pairs))), flush=True)
    occupied = ((obstacles.reshape(shape) >= minimum_obstacle_points) |
                (hazards.reshape(shape) >= minimum_ground_points))
    # Hazard observations are known occupied even without valid ground z.
    known = supported | occupied
    trajectory = trajectory_native.copy()
    trajectory[:, :2] -= native_origin
    map_path = output/'forest_map.npz'
    np.savez_compressed(map_path, occupied=occupied, known=known,
                        resolution_m=np.array(resolution), origin_xy=np.zeros(2),
                        terrain=terrain, native_origin_xy=native_origin,
                        ground_observed=ground_evidence, ground_count=ground_count,
                        obstacle_count=obstacles.reshape(shape),
                        acquisition_trajectory=trajectory)
    free = known & ~occupied
    metadata = dict(protocol='wildscenes-semantic-2d-v1', sequence=sequence_dir.name,
        data_source='https://data.csiro.au/collection/csiro:61541',
        official_code='https://github.com/csiro-robotics/WildScenes',
        license='CC BY-NC-SA 4.0', dimension='2D planar simulation',
        point_transform='p_native = R(qx,qy,qz,qw) @ p_sensor + t; p_sim_xy = p_native_xy - native_origin_xy',
        native_origin_xy=native_origin.tolist(), native_point_bounds=[lower.tolist(), upper.tolist()],
        world_size_m=world, resolution_m=resolution, shape=list(shape),
        coordinate_scale=1.0, scans=len(pairs), points=total_points,
        acquisition_pose_polyline_length_m=float(np.linalg.norm(np.diff(trajectory_native, axis=0), axis=1).sum()),
        acquisition_length_is_not_navigation_task_length=True,
        ground_labels={str(k): LABELS[k] for k in GROUND},
        obstacle_labels={str(k): LABELS[k] for k in SOLID+VEGETATION},
        hazard_labels={str(k): LABELS[k] for k in HAZARD},
        label_point_counts={str(k): int(class_counts[k]) for k in LABELS},
        low_obstacle_point_counts=obstacle_by_label, low_obstacle_points=low_points,
        ground_interpolation_max_m=interpolation, minimum_ground_points=minimum_ground_points,
        minimum_obstacle_points=minimum_obstacle_points, vehicle_envelope_height_m=vehicle_height,
        known_area_m2=float(known.sum()*resolution**2),
        directly_ground_observed_area_m2=float(ground_evidence.sum()*resolution**2),
        free_area_m2=float(free.sum()*resolution**2), occupied_area_m2=float(occupied.sum()*resolution**2),
        map_sha256=digest(map_path), poses_sha256=digest(sequence_dir/'poses3d.csv'),
        map_npz=map_path.name, scope_notes=[
            'All released labelled scans in this sequence were used; no synthetic obstacles or map scaling.',
            'Repeated scan observations are evidence counts, not independent measurements.',
            'Ground support may be interpolated only within the stated radius; all other cells remain unknown and blocked.',
            'Terrain height is used to filter canopy, not to simulate slopes, suspension or traction.',
            'Semantic labels and reconstructed poses are offline map evidence; policy receives only local binary maps.',
            'Drones retain fixed-height planar motion; no aerial canopy collision, optical occlusion, or RF attenuation model.',
            'Map is an approximation with occlusion, annotation and reconstruction errors, not a survey-certified collision mesh.'
        ])
    (output/'map_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    (output/'source_scans.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return metadata


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resolution', type=float, default=.5)
    p.add_argument('--ground-interpolation', type=float, default=1.)
    p.add_argument('--vehicle-height', type=float, default=2.)
    p.add_argument('--minimum-ground-points', type=int, default=3)
    p.add_argument('--minimum-obstacle-points', type=int, default=3)
    a = p.parse_args()
    if (a.resolution <= 0 or a.ground_interpolation < 0 or a.vehicle_height <= 0 or
            a.minimum_ground_points < 1 or a.minimum_obstacle_points < 1):
        p.error('Invalid map thresholds')
    prepare(a.sequence_dir, a.output, a.resolution, a.ground_interpolation,
            a.vehicle_height, a.minimum_ground_points, a.minimum_obstacle_points)
