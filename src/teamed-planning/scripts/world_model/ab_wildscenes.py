"""Spatially separated, metric crops of WildScenes-derived simplified scenes.

Full geometry is private to setup, sensor simulation and the physical audit.
Unmeasured source space is assumed free by the legacy simulation protocol;
the vehicle still receives evidence only inside delivered survey footprints.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from simulator.wildscenes_legacy_scene import LegacyObstacleScene, PROTOCOL as SCENE_PROTOCOL
from .ab_environment import ProxyMapper

PROTOCOL = 'vtmp3-wildscenes-spatial-crops-v1'
TERRAIN = 'wildscenes_simplified'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_manifest(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if spec.get('protocol') != PROTOCOL:
        raise ValueError('Unsupported WildScenes crop manifest')
    world = float(spec['world_size_m'])
    if not np.isfinite(world) or world <= 470:
        raise ValueError('WildScenes crops must support 100-400m navigation tasks')
    regions, sources = [], {}
    for split in ('train', 'val', 'test'):
        if not spec.get(split):
            raise ValueError('Each WildScenes split requires at least one region')
        for item in spec[split]:
            source = (path.parent/item['scene']).resolve()
            content = source.read_bytes()
            sha = digest(content)
            if item.get('source_sha256', sha) != sha:
                raise ValueError('WildScenes source hash changed')
            doc = json.loads(content)
            if (doc.get('protocol') != SCENE_PROTOCOL or doc.get('simplified_scene') is not True
                    or doc.get('unknown_space') != 'free_by_simulation_assumption'
                    or doc.get('coordinate_scale', 1.) != 1.):
                raise ValueError('Expected a metric, explicitly simplified WildScenes scene')
            origin = np.asarray(item['origin_xy'], float)
            if (origin.shape != (2,) or not np.isfinite(origin).all() or (origin < 0).any()
                    or (origin+world > doc['world_size_m']).any()):
                raise ValueError('WildScenes crop is outside source bounds')
            map_id = item['map_id']
            if not isinstance(map_id, str) or not map_id:
                raise ValueError('WildScenes map_id must be nonempty')
            if map_id in sources and sources[map_id] != sha:
                raise ValueError('One physical map must use one source geometry')
            sources[map_id] = sha
            item.update(scene=str(source), source_sha256=sha)
            item['region_id'] = digest(json.dumps([sha, origin.tolist(), world]).encode())
            for other_split, other in regions:
                if split == other_split:
                    continue
                if map_id == other['map_id'] or sha == other['source_sha256']:
                    other_origin = np.asarray(other['origin_xy'])
                    if (np.minimum(origin+world, other_origin+world) > np.maximum(origin, other_origin)).all():
                        raise ValueError('WildScenes regions overlap across dataset splits')
            regions.append((split, item))
    return spec


def freeze_manifest(path, directory):
    spec = load_manifest(path)
    directory = Path(directory)
    directory.mkdir()
    copied = set()
    for split in ('train', 'val', 'test'):
        for item in spec[split]:
            name = item['source_sha256']+'.json'
            if name not in copied:
                content = Path(item['scene']).read_bytes()
                if digest(content) != item['source_sha256']:
                    raise ValueError('WildScenes source changed while freezing')
                (directory/name).write_bytes(content)
                copied.add(name)
            item['source_original_path'] = item['scene']
            item['scene'] = name
    target = directory/'manifest.json'
    target.write_text(json.dumps(spec, indent=2)+'\n')
    load_manifest(target)
    return target


def rasterize_rectangles(rectangles, world, resolution):
    """Mark every cell square touching a rectangle, including subpixel obstacles."""
    n = round(world/resolution)
    delta = np.zeros((n+1, n+1), dtype=np.int32)
    rectangles = np.asarray(rectangles, float).reshape(-1, 4)
    lower = rectangles[:, :2]-rectangles[:, 2:]/2
    upper = rectangles[:, :2]+rectangles[:, 2:]/2
    lo = np.maximum(0, np.ceil(lower/resolution).astype(int)-1)
    hi = np.minimum(n, np.floor(upper/resolution).astype(int)+1)
    np.add.at(delta, (lo[:, 1], lo[:, 0]), 1)
    np.add.at(delta, (hi[:, 1], lo[:, 0]), -1)
    np.add.at(delta, (lo[:, 1], hi[:, 0]), -1)
    np.add.at(delta, (hi[:, 1], hi[:, 0]), 1)
    return delta.cumsum(0).cumsum(1)[:-1, :-1] > 0


@lru_cache(maxsize=16)
def crop_geometry(path, sha, origin, world, resolution):
    content = Path(path).read_bytes()
    if digest(content) != sha:
        raise ValueError('Frozen WildScenes geometry hash mismatch')
    source = LegacyObstacleScene(json.loads(content))
    origin = np.asarray(origin, float)
    if (origin < 0).any() or (origin+world > source.world_size_m).any():
        raise ValueError('Crop outside WildScenes bounds')
    lo = np.maximum(source.lower, origin)
    hi = np.minimum(source.upper, origin+world)
    keep = (hi > lo).all(1)
    rectangles = np.column_stack(((lo[keep]+hi[keep])/2-origin, hi[keep]-lo[keep]))
    geometry = LegacyObstacleScene(dict(protocol=SCENE_PROTOCOL, simplified_scene=True,
        unknown_space='free_by_simulation_assumption', world_size_m=world, origin_xy=[0., 0.],
        geometry=dict(type='axis_aligned_rectangles', rectangles=rectangles.tolist())))
    occupied = rasterize_rectangles(rectangles, world, resolution)
    occupied.flags.writeable = False
    return geometry, occupied


def geometry_for(scene, config):
    return crop_geometry(scene['terrain_path'], scene['terrain_source_sha256'],
        tuple(scene['terrain_origin_xy']), config.world, config.resolution)


class WildScenesMapper(ProxyMapper):
    name = 'stochastic_ab_mapping_wildscenes_simplified_v1'

    def __init__(self, occupied, config, seed):
        super().__init__([], config, seed)
        self._occupied = occupied

    def _occupancy(self, centers):
        indices = np.floor(centers/self.config.resolution).astype(int)
        return self._occupied[indices[:, 1], indices[:, 0]].copy()


class WildScenesAudit:
    def __init__(self, geometry, clearance):
        self.geometry, self.clearance = geometry, clearance

    def clear(self, a, b):
        return self.geometry.segment_clear(a, b, self.clearance)


class WildScenesSource:
    def __init__(self, manifest, config):
        self.spec, self.config = load_manifest(manifest), config
        if self.spec['world_size_m'] != config.world:
            raise ValueError('WildScenes world differs from mapping configuration')
        self._sampling = {}

    def _region(self, item):
        if item['region_id'] not in self._sampling:
            geometry, occupied = crop_geometry(item['scene'], item['source_sha256'],
                tuple(item['origin_xy']), self.config.world, self.config.resolution)
            # Match the navigation grid's square/segment safety margin.
            free = np.pad(~occupied, 1, constant_values=False)
            distances = distance_transform_edt(free)[1:-1, 1:-1]*self.config.resolution
            safe = distances > self.config.clearance+self.config.resolution*np.sqrt(2.)
            components, _ = label(safe)
            yy, xx = np.indices(safe.shape)
            xy = np.stack(((xx+.5)*self.config.resolution, (yy+.5)*self.config.resolution), -1)
            interior = (xy.min(-1) >= 20.) & (xy.max(-1) <= self.config.world-20.)
            # Focus starts on measured obstacle neighbourhoods, not blank margins.
            obstacle_distances = distance_transform_edt(~occupied)*self.config.resolution
            starts = np.argwhere(safe & interior & (obstacle_distances >= 6.) & (obstacle_distances <= 40.))
            if not len(starts):
                raise ValueError('WildScenes region has no safe obstacle-neighbourhood starts')
            self._sampling[item['region_id']] = geometry, components, xy, starts, interior
        return self._sampling[item['region_id']]

    def scene(self, seed, split):
        if split not in ('train', 'val', 'test'):
            raise ValueError('Unknown WildScenes dataset split')
        items = self.spec[split]
        item = items[int(seed) % len(items)]
        geometry, components, xy, starts, interior = self._region(item)
        rng = np.random.default_rng(int(seed) ^ 0x5A17)
        for _ in range(256):
            y, x = starts[int(rng.integers(len(starts)))]
            start = xy[y, x]
            distance = np.linalg.norm(xy-start, axis=-1)
            targets = np.argwhere((components == components[y, x]) & interior
                                 & (distance >= 100.) & (distance <= 400.))
            if not len(targets):
                continue
            gy, gx = targets[int(rng.integers(len(targets)))]
            goal = xy[gy, gx]
            if not all(geometry.point_clear(p, self.config.clearance) for p in (start, goal)):
                raise RuntimeError('Conservative WildScenes task endpoints failed physical audit')
            return dict(seed=int(seed), world=self.config.world, start=start.tolist(), goal=goal.tolist(),
                terrain=TERRAIN, terrain_path=item['scene'], terrain_source_sha256=item['source_sha256'],
                terrain_origin_xy=item['origin_xy'], terrain_region_id=item['region_id'],
                terrain_map_id=item['map_id'], split=split, coordinate_scale=1.,
                distance=float(distance[gy, gx]), simplified_scene=True,
                task_feasibility='same connected component of conservative received-map-resolution truth grid',
                source_unknown_space='free_by_simulation_assumption')
        raise ValueError('No feasible 100-400m task in WildScenes region')
