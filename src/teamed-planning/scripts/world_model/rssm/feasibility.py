"""Offline physical-route certification for vtmp2 training scenes.

The reference polyline and obstacle truth stay in the scene manifest. They are
for task admission, never observations or target-selection inputs. Certification
uses continuous segment/circle distances, not navigation-graph connectivity.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from .hierarchical_sources import DEFAULT_REPO, file_sha256 as _digest


FROZEN = DEFAULT_REPO / 'artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146'
FROZEN_GROUND_CLEARANCE_M = 1.5
SCHEMA_VERSION = 1


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be finite and positive')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite and positive') from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return number


def _point(value, name):
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain two finite coordinates') from exc
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f'{name} must contain two finite coordinates')
    return point


def _seed(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError('seed must be a nonnegative integer')
    return int(value)


def _split(value):
    if not isinstance(value, str) or re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', value) is None:
        raise ValueError('split must be a simple alphanumeric label')
    return value


def certify_scene(scene, clearance=FROZEN_GROUND_CLEARANCE_M):
    """Return a fresh certificate or reject the entire scene with ValueError.

    The frozen vehicle uses 1.5m obstacle clearance. Callers may require more,
    but cannot silently lower it. Actual endpoints are included even though
    generated reference points are the centres of their containing grid cells.
    """
    clearance = _positive(clearance, 'clearance')
    if clearance < FROZEN_GROUND_CLEARANCE_M:
        raise ValueError('clearance cannot be below the frozen vehicle requirement of 1.5m')
    if not isinstance(scene, dict):
        raise ValueError('scene must be an object')
    required = ('seed', 'start', 'goal', 'world', 'distance', 'goals', 'references',
                'reference_resolution_m', 'circles')
    if any(key not in scene for key in required):
        raise ValueError('scene is missing required physical-reference fields')
    seed = _seed(scene['seed'])
    start, goal = _point(scene['start'], 'start'), _point(scene['goal'], 'goal')
    world = _positive(scene['world'], 'world')
    distance = _positive(scene['distance'], 'distance')
    resolution = _positive(scene['reference_resolution_m'], 'reference_resolution_m')
    if world <= 2 * clearance:
        raise ValueError('world has no interior satisfying the clearance')
    if not np.isclose(np.linalg.norm(goal-start), distance, rtol=1e-8, atol=1e-6):
        raise ValueError('distance does not match the actual start and goal')
    key = str(int(distance)) if distance.is_integer() else repr(distance)
    if (not isinstance(scene['goals'], dict) or not isinstance(scene['references'], dict)
            or key not in scene['goals'] or key not in scene['references']):
        raise ValueError('reference for the actual task distance is missing')
    if not np.allclose(_point(scene['goals'][key], 'reference goal'), goal, rtol=0., atol=1e-8):
        raise ValueError('reference goal differs from the actual task goal')
    route = scene['references'][key]
    if not isinstance(route, dict) or 'points' not in route:
        raise ValueError('reference points are missing')
    try:
        points = np.asarray(route['points'], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('reference must contain finite 2D points') from exc
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('reference must contain at least two finite 2D points')
    for actual, reference, name in ((start, points[0], 'start'), (goal, points[-1], 'goal')):
        grid_center = (np.floor(actual / resolution) + .5) * resolution
        if not (np.allclose(reference, actual, rtol=0., atol=1e-8)
                or np.allclose(reference, grid_center, rtol=0., atol=1e-8)):
            raise ValueError(f'reference does not terminate at the actual {name} or its grid cell')
    path = np.vstack((start, points, goal))
    boundary = float(np.minimum(path, world-path).min())
    if boundary < clearance:
        raise ValueError(f'reference crosses the required world-boundary clearance: {boundary:g}m')
    if not isinstance(scene['circles'], (list, tuple)):
        raise ValueError('circles must be a list')
    centers, radii = [], []
    for circle in scene['circles']:
        if not isinstance(circle, (list, tuple)) or len(circle) != 2:
            raise ValueError('each circle must contain a center and radius')
        centers.append(_point(circle[0], 'obstacle center'))
        radii.append(_positive(circle[1], 'obstacle radius'))
    centers, radii = np.asarray(centers), np.asarray(radii)
    obstacle_clearance = math.inf
    for first, second in zip(path, path[1:]):
        if not len(radii):
            break
        delta = second-first
        squared = float(delta @ delta)
        parameter = np.clip(((centers-first) @ delta) / max(squared, 1e-30), 0., 1.)
        distances = np.linalg.norm(centers-first-parameter[:, None]*delta, axis=1)-radii
        segment_clearance = float(distances.min())
        if not math.isfinite(segment_clearance) or segment_clearance <= clearance:
            raise ValueError(f'continuous reference segment violates obstacle clearance: {segment_clearance:g}m')
        obstacle_clearance = min(obstacle_clearance, segment_clearance)
    path_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    if not math.isfinite(path_length) or path_length < distance-1e-6:
        raise ValueError('invalid continuous reference length')
    if 'reference_polyline_length' in scene:
        claimed = _positive(scene['reference_polyline_length'], 'reference_polyline_length')
        if not np.isclose(claimed, path_length, rtol=1e-8, atol=1e-6):
            raise ValueError('reference_polyline_length does not match the complete physical route')
    return dict(valid=True, method='continuous_segment_circle_v1', seed=seed,
                clearance_m=clearance, path_length_m=path_length,
                min_clearance_m=min(boundary, obstacle_clearance),
                min_obstacle_clearance_m=None if not len(radii) else obstacle_clearance,
                boundary_clearance_m=boundary, segment_count=len(path)-1,
                reference_point_count=len(points))


def _atomic_json(path, document):
    """Publish a complete JSON file atomically, without overwriting a prior run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard link is an atomic, non-overwriting publication.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_certified_scene(path, expected_sha256=None, clearance=FROZEN_GROUND_CLEARANCE_M):
    """Read a manifest and independently recheck its complete physical route."""
    path = Path(path)
    contents = path.read_bytes()
    actual = hashlib.sha256(contents).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f'scene SHA256 mismatch: {path}')
    try:
        document = json.loads(contents)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError('invalid certified-scene JSON') from exc
    if (not isinstance(document, dict) or document.get('algorithm') != 'vtmp2'
            or document.get('schema_version') != SCHEMA_VERSION
            or 'scene' not in document or 'certificate' not in document):
        raise ValueError('invalid certified-scene schema')
    _split(document.get('split'))
    scene = document['scene']
    certificate = certify_scene(scene, clearance=clearance)
    if _seed(document.get('seed')) != certificate['seed']:
        raise ValueError('manifest and scene seed disagree')
    # Verify what was stored, then allow a caller to request stricter clearance.
    recorded = document['certificate']
    if not isinstance(recorded, dict) or 'clearance_m' not in recorded:
        raise ValueError('stored certificate is incomplete')
    original = certify_scene(scene, clearance=recorded['clearance_m'])
    if recorded != original:
        raise ValueError('stored certificate disagrees with independently checked geometry')
    return scene, certificate


def generate_certified_scene(seed, split, output_dir, frozen=FROZEN,
                             clearance=FROZEN_GROUND_CLEARANCE_M):
    """Generate one specified seed, certify, and write its offline scene file.

    An infeasible seed raises; it is not silently replaced and obstacles are
    never removed. Existing manifests are not overwritten.
    """
    seed, split = _seed(seed), _split(split)
    frozen = Path(frozen).resolve()
    source = frozen/'runtime_sources/simulator/large_world_scenarios.py'
    configuration = frozen/'config.json'
    source_digest, config_digest = _digest(source), _digest(configuration)
    config = json.loads(configuration.read_text())
    spec = importlib.util.spec_from_file_location('_vtmp2_frozen_scene_generator_'+source_digest, source)
    if spec is None or spec.loader is None:
        raise ValueError('cannot load the frozen scene generator')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scene = module.generate_training_scene(seed, config['task_distances'],
        world=config.get('world_size', 1000.), distance_range=config.get('task_distance_range'),
        distance_sampling=config.get('distance_sampling', 'linear'))
    if _digest(source) != source_digest or _digest(configuration) != config_digest:
        raise RuntimeError('frozen generator/config changed while selecting the scene')
    certificate = certify_scene(scene, clearance=clearance)
    if certificate['seed'] != seed:
        raise ValueError('scene generator returned a different seed')
    path = (Path(output_dir)/'scenes'/f'{split}_{seed}.json').resolve()
    document = dict(algorithm='vtmp2', schema_version=SCHEMA_VERSION, seed=seed, split=split,
        scene=scene, certificate=certificate, generator=str(source), generator_sha256=source_digest,
        config_sha256=config_digest)
    _atomic_json(path, document)
    return dict(seed=seed, split=split, scene_path=str(path), scene_sha256=_digest(path),
                certificate=certificate)
