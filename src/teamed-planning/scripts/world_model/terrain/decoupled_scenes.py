"""Simplified WildScenes setup/audit for the frozen v9 simulator.

Reuse the repository's spatial crop manifest and exact rectangle geometry.
Geometry stays outside both controllers. Original unmeasured source space is
free by the declared simulation assumption, not verified real traversability.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from .hierarchical_sources import file_sha256

TERRAIN = 'wildscenes_simplified'


@lru_cache(maxsize=1)
def geometry_adapter():
    # A standalone file import avoids putting the mutable simulator package in
    # sys.modules before build_environment imports and verifies the frozen v9.
    path = Path(__file__).resolve().parents[2]/'simulator/wildscenes_legacy_scene.py'
    spec = importlib.util.spec_from_file_location('_vtmp2_wildscenes_geometry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=8)
def _crop(path, sha, origin, world):
    adapter = geometry_adapter()
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != sha:
        raise ValueError('WildScenes source hash changed')
    source = adapter.LegacyObstacleScene(json.loads(content))
    origin = np.asarray(origin)
    if (origin < 0).any() or (origin+world > source.world_size_m).any():
        raise ValueError('WildScenes crop outside source bounds')
    lo, hi = np.maximum(source.lower, origin), np.minimum(source.upper, origin+world)
    keep = (hi > lo).all(1)
    rectangles = np.column_stack(((lo[keep]+hi[keep])/2-origin, hi[keep]-lo[keep]))
    return adapter.LegacyObstacleScene(dict(protocol=adapter.PROTOCOL, simplified_scene=True,
        unknown_space='free_by_simulation_assumption', world_size_m=world, origin_xy=[0., 0.],
        geometry=dict(type='axis_aligned_rectangles', rectangles=rectangles.tolist())))


def geometry_for(scene):
    return _crop(scene['terrain_path'], scene['terrain_source_sha256'],
                 tuple(scene['terrain_origin_xy']), scene['world'])


def geometry_hash(scene):
    keys = ('start', 'goal', 'world', 'terrain_source_sha256', 'terrain_origin_xy')
    return hashlib.sha256(json.dumps({key: scene[key] for key in keys},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def certify(scene):
    geometry = geometry_for(scene)
    path = np.asarray(scene['reference_points'], dtype=float)
    distance = float(np.linalg.norm(np.asarray(scene['goal'])-scene['start']))
    if not np.isfinite(distance) or not np.isclose(distance, scene['distance']):
        raise ValueError('WildScenes declared task distance disagrees with its endpoints')
    if 'task_distance_range_m' in scene:
        low, high = scene['task_distance_range_m']
        if not np.isfinite([low, high]).all() or not 0 < low <= distance <= high:
            raise ValueError('WildScenes endpoints violate the requested task-distance range')
    if (path.ndim != 2 or path.shape[1:] != (2,) or len(path) < 2
            or not np.isfinite(path).all()
            or not np.allclose(path[0], scene['start']) or not np.allclose(path[-1], scene['goal'])
            or not all(geometry.segment_clear(a, b, 1.5) for a, b in zip(path[:-1], path[1:]))):
        raise ValueError('WildScenes continuous reference violates 1.5m clearance')
    return dict(valid=True, method='continuous_segment_rectangle_v1', clearance_m=1.5,
                path_length_m=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()))


def generate_scene(provider, seed, split, output_dir, region_split=None, distance_range=(100., 400.)):
    # These imports belong only to offline setup, outside the v9 worker process.
    from scipy.ndimage import distance_transform_edt
    from simulator.large_world_scenarios import reference_route
    from .ab_wildscenes import geometry_for as source_geometry
    from .data import atomic_json, sha256
    scene = provider.scene(seed, region_split or split, distance_range=distance_range)
    scene['terrain_split'] = scene['split']
    scene['split'] = split
    _, occupied = source_geometry(scene, provider.config)
    resolution = provider.config.resolution
    free = distance_transform_edt(np.pad(~occupied, 1, constant_values=False))[1:-1, 1:-1]*resolution
    route = reference_route(free > 1.5+resolution*np.sqrt(2), scene['start'], scene['goal'], resolution)
    scene['reference_points'] = route['points']
    scene['reference_length'] = route['length']
    scene['attempts'] = 1
    scene['circles'] = []  # Rectangles are installed before the simulator reset.
    certificate = certify(scene)
    path = (Path(output_dir)/'scenes'/f'{split}_{seed}.json').resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    atomic_json(path, dict(algorithm='vtmp2', schema_version=4, terrain=TERRAIN,
                          seed=seed, split=split, scene=scene, certificate=certificate))
    return dict(seed=seed, split=split, scene_path=str(path), scene_sha256=sha256(path), certificate=certificate)


def load_scene(path, expected_sha256):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('WildScenes scene SHA256 mismatch')
    document = json.loads(content)
    if (document.get('algorithm') != 'vtmp2' or document.get('schema_version') != 4
            or document.get('terrain') != TERRAIN):
        raise ValueError('Invalid WildScenes vtmp2 scene contract')
    scene = document['scene']
    if document['seed'] != scene['seed'] or document['split'] != scene['split']:
        raise ValueError('WildScenes scene metadata disagrees')
    if file_sha256(scene['terrain_path']) != scene['terrain_source_sha256']:
        raise ValueError('Frozen WildScenes source hash mismatch')
    certificate = certify(scene)
    if document['certificate'] != certificate:
        raise ValueError('WildScenes route certificate changed')
    return scene, certificate


def regions_overlap(first, second):
    """Compare physical footprints, not task seeds, file names or region labels."""
    if not first.get('terrain_source_sha256') or not second.get('terrain_source_sha256'):
        return False
    if (first['terrain_source_sha256'] != second['terrain_source_sha256']
            and first.get('terrain_map_id') != second.get('terrain_map_id')):
        return False
    a, b = np.asarray(first['terrain_origin_xy']), np.asarray(second['terrain_origin_xy'])
    return bool((np.minimum(a+first['terrain_world_m'], b+second['terrain_world_m']) > np.maximum(a, b)).all())
