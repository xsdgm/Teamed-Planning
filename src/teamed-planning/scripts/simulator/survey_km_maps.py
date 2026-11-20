"""Materialize disjoint metre-scale survey tasks with the frozen V9 random-circle generator."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import fcntl
import json
import os

import numpy as np
from scipy.ndimage import label

from world_model.hierarchical_sources import atomic_json, file_sha256
from simulator.survey_pretrain_maps import CLEARANCE_M, load_splits, load_survey_map


GENERATOR = Path('artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146/'
                 'runtime_sources/simulator/large_world_scenarios.py')
GENERATOR_SHA = 'e9ed598886918da8291c6d55320ae3f68c2ba1153d7b456598f70c1005e4528f'
WORLD_M = 1650.
BRIDGE_M = (600., 900.)
KM_BANDS_M = ((1000., 1200.), (1200., 1500.))


@lru_cache(maxsize=2)
def _generator(path):
    if file_sha256(path) != GENERATOR_SHA:
        raise ValueError('Archived V9 map generator checksum mismatch')
    spec = spec_from_file_location('survey_archived_v9_large_world', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case_band(split, index, count):
    if split == 'warmup':
        return BRIDGE_M
    if split == 'ppo_train':
        return KM_BANDS_M[int(index >= (count + 1) // 2)]
    return KM_BANDS_M[index % 2]


def _make_case(job):
    generator_path, folder, seed, split_no, split, i, band, forbidden = job
    name = f'v9-km-{split}-{i:03d}'
    meta = folder / f'{name}.json'
    geometry = folder / f'{name}.npy'
    if meta.exists():
        row = json.loads(meta.read_text())
        if row['case_id'] != name or not geometry.exists() or file_sha256(geometry) != row['circles_sha256']:
            raise ValueError(f'Existing kilometre geometry disagrees with its record: {name}')
        row['circles_path'] = str(geometry)
        return row
    gen = _generator(generator_path)
    rejected = []
    for attempt in range(16):
        scene_seed = int(np.random.SeedSequence([seed, split_no, i, attempt]).generate_state(1)[0])
        if scene_seed in forbidden:
            rejected.append(dict(seed=scene_seed, reason='historical_seed'))
            continue
        scene = gen.generate_training_scene(scene_seed, world=WORLD_M, distance_range=band,
                                            distance_sampling='annulus_area')
        if not scene['reference_geometry_validated'] or scene['reference_geometric_clearance_m'] != CLEARANCE_M:
            raise ValueError('Archived generator did not certify the reference route')
        circles = np.asarray([[*point, radius] for point, radius in scene['circles']], dtype=np.float64)
        temp = geometry.with_suffix('.npy.tmp')
        with temp.open('wb') as stream:
            np.save(stream, circles, allow_pickle=False)
        os.replace(temp, geometry)
        row = dict(terrain='synthetic', kind='v9_random', case_id=name, seed=scene_seed,
                   world_size=WORLD_M, distance_band_m=list(band), start=scene['start'], goal=scene['goal'],
                   direct_distance_m=float(scene['distance']), reference_distance_m=float(scene['reference_polyline_length']),
                   circle_count=len(circles), circles_path=str(geometry), circles_sha256=file_sha256(geometry))
        grid = load_survey_map(row)
        rc = np.floor((np.stack((grid.start, grid.goal)) - grid.origin) / grid.resolution).astype(int)
        components, _ = label(grid.free)
        first, last = components[rc[:, 1], rc[:, 0]]
        if first == 0 or first != last:
            rejected.append(dict(seed=scene_seed, reason='disconnected_in_fixed_crop'))
            continue
        row.update(cropped_four_connected=True, generation_attempts=scene['attempts'],
                   geometry_rejections=rejected)
        atomic_json(meta, row)
        return row
    raise RuntimeError(f'No feasible V9 map in the fixed endpoint crop for {name}: {rejected}')


def make_km_splits(repo, data_dir, *, seed, warmup_cases, ppo_cases, dev_cases, test_cases, workers=4):
    """Generate or verify an identical cached request. Never expose reference paths to the planner."""
    repo, folder = Path(repo).resolve(), Path(data_dir).resolve()
    sizes = dict(warmup=warmup_cases, ppo_train=ppo_cases, development=dev_cases, test=test_cases)
    if min(*sizes.values(), workers) <= 0 or seed < 0:
        raise ValueError('Positive split sizes and worker count and nonnegative seed required')
    folder.mkdir(parents=True, exist_ok=True)
    gen_path = repo / GENERATOR
    _generator(gen_path)
    historical, wild = load_splits(repo)
    forbidden = frozenset(r['seed'] for r in historical + wild)
    request = dict(format='vtemp2_km_survey_maps_v1', seed=seed, splits=sizes,
                   generator_sha256=GENERATOR_SHA, world_size_m=WORLD_M, sampling='annulus_area',
                   bridge_m=list(BRIDGE_M), kilometres_m=[list(b) for b in KM_BANDS_M],
                   required_clearance_m=CLEARANCE_M)
    with (folder / '.generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request_path = folder / 'request.json'
        if request_path.exists():
            if request != json.loads(request_path.read_text()):
                raise ValueError('Different kilometre data request at this output path')
        else:
            if any(p.name != '.generation.lock' for p in folder.iterdir()):
                raise ValueError('Unidentified existing kilometre data directory')
            atomic_json(request_path, request)
        geometry = folder / 'geometry'
        geometry.mkdir(exist_ok=True)
        jobs = [(gen_path, geometry, seed, split_no, split, i, _case_band(split, i, n), forbidden)
                for split_no, (split, n) in enumerate(sizes.items()) for i in range(n)]
        # A resumed run verifies completed cases and regenerates any interrupted ones.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_make_case, jobs))
        seeds = {r['seed'] for r in rows}
        if len(seeds) != len(rows) or seeds & forbidden or len({r['circles_sha256'] for r in rows}) != len(rows):
            raise ValueError('Seed or map geometry overlap in kilometre splits')
        sets = {k: [] for k in sizes}
        for job, row in zip(jobs, rows):
            sets[job[4]].append(row)
        atomic_json(folder / 'scenes.json', sets)
        atomic_json(folder / 'audit.json', dict(**request, generator_path=str(GENERATOR),
                    density_circles_per_m2=_generator(gen_path).ORIGINAL_ACTIVE_DENSITY,
                    coordinates='metres in the original V9 square simulation frame',
                    partition='independent seeds; no overlap with 800 archived V9 or 18 WildScenes cases',
                    simulator='1m clearance-compatible grid cropped 50m beyond endpoint bounding box; 4-connected',
                    reference='scalar certified length for audit only; no oracle points exported',
                    scenes_sha256=file_sha256(folder / 'scenes.json')))
    return sets


def make_km_far_test(repo, data_dir, *, seed, existing, workers=2):
    """Two predeclared 1400–1500 m confirmations, separate from the first eight holdouts."""
    repo = Path(repo).resolve()
    folder = Path(data_dir).resolve() / 'far-test'
    folder.mkdir(parents=True, exist_ok=True)
    if workers < 1 or seed < 0:
        raise ValueError('Positive worker count and nonnegative seed required')
    current = [r for group in existing.values() for r in group]
    historical, wild = load_splits(repo)
    forbidden = frozenset(r['seed'] for r in current + historical + wild)
    request = dict(format='vtemp2_km_survey_far_test_v1', generator_sha256=GENERATOR_SHA,
                   seed=seed, split_code=4, count=2, distance_band_m=[1400.,1500.],
                   base_geometry=sorted(r['circles_sha256'] for r in current))
    with (folder / '.generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request_path = folder / 'request.json'
        if request_path.exists():
            if request != json.loads(request_path.read_text()):
                raise ValueError('Different kilometre far-test request at this output path')
        else:
            if any(p.name != '.generation.lock' for p in folder.iterdir()):
                raise ValueError('Unidentified existing kilometre far-test directory')
            atomic_json(request_path, request)
        gen_path = repo / GENERATOR
        _generator(gen_path)
        jobs=[(gen_path, folder, seed, 4, 'far-test', i, (1400.,1500.), forbidden) for i in range(2)]
        with ProcessPoolExecutor(max_workers=min(2, workers)) as pool:
            rows = list(pool.map(_make_case, jobs))
        digests = {r['circles_sha256'] for r in rows}
        seeds = {r['seed'] for r in rows}
        if (len(seeds) != 2 or len(digests) != 2 or seeds & forbidden
                or digests & {r['circles_sha256'] for r in current + historical}):
            raise ValueError('Kilometre far-test map or seed overlaps another split')
        atomic_json(folder / 'scenes.json', rows)
    return rows
