"""Offline geometry for V9 survey training and the original 18 Venman tasks."""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from vtemp2_3.hierarchical_sources import file_sha256
from vtemp2_3.scene_manifest import load_v9_circles


ARCHIVE = Path('artifacts/v50-v9-random-wait30-prep')
HISTORICAL_VALIDATION = Path('artifacts/hierarchical-fast-mixed-retrain-par16-20261117/obstacle-test-wait30')
V9_SPLIT_SHA256 = 'ff004c34c8468d98c0c8b1b91dc6f50050a2911ab813a184efc510413978243f'
WILD_SPLIT_SHA256 = '9a9febcbfcf0ce44d9641203a43d32b2d2d58d3ea8ac28550c50759680194943'
CLEARANCE_M = 1.5
MARGIN_M = 50.


@dataclass
class SurveyMap:
    """Simulator-only grid; XY positions and crop origin use the source metre frame."""
    free: np.ndarray
    resolution: float
    origin: np.ndarray
    start: np.ndarray
    goal: np.ndarray
    sensor_valid: np.ndarray | None = None

    def as_kwargs(self):
        return dict(free=self.free, resolution=self.resolution, origin=self.origin,
                    start=self.start, goal=self.goal, sensor_valid=self.sensor_valid)


def load_splits(repo):
    """Bind the synthetic training pool and the independently evaluated fixed tasks."""
    repo = Path(repo).resolve()
    manifest = repo / ARCHIVE / 'scenes.json'
    validation_manifest = repo / HISTORICAL_VALIDATION / 'scenes.json'
    if file_sha256(manifest) != V9_SPLIT_SHA256 or file_sha256(validation_manifest) != WILD_SPLIT_SHA256:
        raise ValueError('Historical V9 or WildScenes task manifest digest differs')
    split = json.loads(manifest.read_text())
    validation = json.loads(validation_manifest.read_text())['validation']
    if split['validation'] != validation:
        raise ValueError('Training preparation differs from the original 18 validation tasks')
    for row in split['train']:
        row['circles_path'] = str(repo / ARCHIVE / 'geometry' / Path(row['circles_path']).name)
    raster_dir = repo / 'artifacts/wildscenes-v9-20261106/maps/V-01'
    for row in validation:
        for key in ('map_path', 'metadata_path'):
            row[key] = str(raster_dir / Path(row[key]).name)
    return split['train'], validation


def _crop(start, goal, resolution, dimensions):
    # Work area depends only on mission endpoints, never on an oracle route.
    lower = np.floor((np.minimum(start, goal) - MARGIN_M) / resolution).astype(int)
    upper = np.ceil((np.maximum(start, goal) + MARGIN_M) / resolution).astype(int)
    return np.maximum(lower, 0), np.minimum(upper, dimensions)


def _v9_free(case, start, goal):
    resolution = 1.
    world = float(case['world_size'])
    lo, hi = _crop(start, goal, resolution, np.array([world, world], dtype=int))
    circles = load_v9_circles(case)
    radii = circles[:, 2] + CLEARANCE_M
    intersects = np.all(circles[:, :2] + radii[:, None] >= lo, axis=1)
    intersects &= np.all(circles[:, :2] - radii[:, None] <= hi, axis=1)
    free = np.ones(tuple((hi - lo)[::-1]), dtype=bool)
    for (x, y, _), radius in zip(circles[intersects], radii[intersects]):
        x0, y0 = np.maximum(lo, np.floor([x-radius-1, y-radius-1]).astype(int))
        x1, y1 = np.minimum(hi, np.ceil([x+radius+1, y+radius+1]).astype(int))
        # Distance from each closed unit square to the circle's centre.
        dx = np.maximum(np.abs(np.arange(x0, x1) + .5 - x) - .5, 0.)
        dy = np.maximum(np.abs(np.arange(y0, y1) + .5 - y) - .5, 0.)
        free[y0-lo[1]:y1-lo[1], x0-lo[0]:x1-lo[0]] &= dy[:, None]**2 + dx**2 > radius**2
    xs = np.arange(lo[0], hi[0])
    ys = np.arange(lo[1], hi[1])
    free &= (ys[:, None] >= CLEARANCE_M) & (ys[:, None] + 1 <= world - CLEARANCE_M)
    free &= (xs >= CLEARANCE_M) & (xs + 1 <= world - CLEARANCE_M)
    return free, resolution, lo.astype(float)


@lru_cache(maxsize=1)
def _wild_grid(map_path, map_sha, metadata_path, metadata_sha):
    if file_sha256(map_path) != map_sha or file_sha256(metadata_path) != metadata_sha:
        raise ValueError('WildScenes map or metadata digest differs')
    meta = json.loads(Path(metadata_path).read_text())
    with np.load(map_path, allow_pickle=False) as arrays:
        resolution = float(arrays['resolution_m'])
        if (resolution != .5 or not np.array_equal(arrays['origin_xy'], [0., 0.])
                or meta['map_sha256'] != map_sha or meta['coordinate_scale'] != 1.):
            raise ValueError('Original metric V-01 raster required')
        known = arrays['known']
        occupied = arrays['occupied'] & known
        blocked = occupied | ~known
    # Distance from a source cell centre to the nearest point on a blocked
    # native pixel rectangle. Unknown pixels and points outside are blocked.
    radius = int(np.ceil(CLEARANCE_M/resolution + .5))
    offset = np.maximum(np.abs(np.arange(-radius, radius+1)) - .5, 0.) * resolution
    disk = offset**2 + offset[:, None]**2 <= CLEARANCE_M**2
    free = ~binary_dilation(blocked, structure=disk, border_value=1)
    # Unmeasured space forbids travel, but its absence of source evidence
    # cannot supply a training label or earn a blocked-road survey reward.
    sensor_valid = known & (free | binary_dilation(occupied, structure=disk))
    free.setflags(write=False)
    sensor_valid.setflags(write=False)
    return free, sensor_valid, resolution


def load_survey_map(case):
    """Return clearance-filtered truth for sensor simulation; keep endpoints verbatim."""
    start = np.asarray(case['start'], dtype=float)
    goal = np.asarray(case['goal'], dtype=float)
    sensor_valid = None
    if case['kind'] == 'v9_random' and case['terrain'] == 'synthetic':
        free, resolution, origin = _v9_free(case, start, goal)
    elif (case['terrain'] == 'wildscenes_raster' and case['sequence'] == 'V-01'
          and case['source_unknown_space'] == 'blocked' and case['required_clearance_m'] == CLEARANCE_M):
        source, valid, resolution = _wild_grid(case['map_path'], case['map_sha256'],
                                              case['metadata_path'], case['metadata_sha256'])
        lo, hi = _crop(start, goal, resolution, source.shape[::-1])
        origin = lo * resolution
        free = source[lo[1]:hi[1], lo[0]:hi[0]]
        sensor_valid = valid[lo[1]:hi[1], lo[0]:hi[0]]
    else:
        raise ValueError(f'Unsupported survey terrain {case["case_id"]}')
    cells = np.floor((np.stack([start, goal]) - origin) / resolution).astype(int)
    if (np.any(cells < 0) or np.any(cells >= free.shape[::-1])
            or not free[cells[:, 1], cells[:, 0]].all()):
        raise ValueError(f'Clearance raster excludes an original endpoint: {case["case_id"]}')
    return SurveyMap(free=free, resolution=resolution, origin=origin, start=start, goal=goal,
                     sensor_valid=sensor_valid)


def make_km_splits(repo, data_dir, *, seed, warmup_cases, ppo_cases, dev_cases, test_cases, workers=4):
    """Build independent V9 bridge, kilometre training, development and test maps."""
    from survey.survey_km_maps import make_km_splits as generate
    return generate(repo, data_dir, seed=seed, warmup_cases=warmup_cases, ppo_cases=ppo_cases,
                    dev_cases=dev_cases, test_cases=test_cases, workers=workers)


def make_km_far_test(repo, data_dir, *, seed, existing, workers=2):
    """Prepare the separate 1.4–1.5 km upper-range confirmation pair."""
    from survey.survey_km_maps import make_km_far_test as generate
    return generate(repo, data_dir, seed=seed, existing=existing, workers=workers)
