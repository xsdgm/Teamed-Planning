"""Generate seeded curriculum scenes and validate metric scene manifests."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import numpy as np

from .training_schedule import update_episode_counts, validate_update_plan

from .hierarchical_sources import DEFAULT_REPO, file_sha256

RASTER_SOURCE_ASSUMPTIONS = {
    'wildscenes_raster': 'blocked',
    'wildscenes_simplified_raster': 'free_by_simulation_assumption',
}


def feasible(case,resolution=4.,clearance=1.5):
    """Check connectivity offline with the vehicle clearance and grid resolution."""
    from scipy.ndimage import label

    n=int(np.ceil(case['world_size']/resolution))
    free=np.ones((n,n),bool)
    for point,radius in case['circles']:
        radius=radius+clearance+resolution*math.sqrt(2)/2
        x,y=np.asarray(point)/resolution
        reach=int(np.ceil(radius/resolution))+1
        x0,x1=max(0,int(x)-reach),min(n,int(x)+reach+1)
        y0,y1=max(0,int(y)-reach),min(n,int(y)+reach+1)
        yy,xx=np.mgrid[y0:y1,x0:x1]
        free[y0:y1,x0:x1]&=((xx+.5-x)**2+(yy+.5-y)**2>(radius/resolution)**2)
    components=label(free,np.array([[0,1,0],[1,1,1],[0,1,0]]))[0]
    start,goal=[np.floor(np.asarray(case[key])/resolution).astype(int) for key in ('start','goal')]
    if not all((p>=0).all() and (p<n).all() for p in (start,goal)):return False
    a,b=components[start[1],start[0]],components[goal[1],goal[0]]
    return bool(a and a==b)


def make_scene(seed,distance_range=(150.,300.),world_size=2000.,kind=None):
    rng=np.random.default_rng(seed)
    low,high=map(float,distance_range)
    if not 0<low<=high<=world_size-400:
        raise ValueError('Distances must fit inside surveyed bounds with a 200m endpoint margin')
    distance=float(rng.uniform(low,high))
    angle=float(rng.uniform(-np.pi,np.pi))
    direction=np.array([np.cos(angle),np.sin(angle)])
    normal=np.array([-direction[1],direction[0]])
    center=np.array([world_size/2,world_size/2])
    start=center-.5*distance*direction;goal=center+.5*distance*direction
    kind=kind or ('open','trees','wall_gap')[int(seed)%3]
    if kind not in ('open','trees','wall_gap'):raise ValueError('Unknown scene kind')
    circles=[]
    if kind=='trees':
        for _ in range(max(3,int(distance/20))):
            p=start+rng.uniform(45,max(46,distance-35))*direction+rng.uniform(-70,70)*normal
            circles.append((p.tolist(),float(rng.uniform(2,4))))
    elif kind=='wall_gap':
        for fraction in np.arange(.35,.9,max(.25,150/distance)):
            gap=float(rng.choice([-1,1])*rng.uniform(25,45))
            for offset in np.arange(-90,91,8):
                if abs(offset-gap)>17:
                    p=start+fraction*distance*direction+offset*normal
                    # Initial UAV deployment must never overlap setup obstacles.
                    if np.linalg.norm(p-start)>40 and np.linalg.norm(p-goal)>20:
                        circles.append((p.tolist(),4.5))
    case=dict(seed=int(seed),start=start.tolist(),goal=goal.tolist(),circles=circles,
              world_size=float(world_size),kind=kind,direct_distance_m=distance)
    if not feasible(case):
        raise ValueError(f'Seed {seed} failed offline scene feasibility; no silent scene replacement')
    return case


def describe_scene_assumptions(cases):
    """Persist source geometry assumptions separately from online knowledge."""
    protocols = sorted({(case.get('terrain', 'synthetic'), case.get('source_unknown_space'))
                        for rows in cases.values() for case in rows})
    simplified = any(terrain == 'wildscenes_simplified_raster' for terrain, _ in protocols)
    return dict(protocols=[dict(terrain=terrain, source_unknown_space=assumption)
                           for terrain, assumption in protocols],
        simplified_geometry=simplified,
        online_knowledge='Only local sensor and received evidence; simulator known masks do not reveal the global map.',
        simplified_source_assumption=('Legacy rectangle occupancy defines the simulator geometry. Source-unknown '
            'space is free by an explicit simulation assumption; this is not measured free-space evidence.') if simplified else None)


def load_v9_circles(case):
    """Read the exact archived V9 circle coordinates, identified by the NPY hash."""
    path = Path(case['circles_path'])
    if not path.is_absolute() or path.suffix != '.npy':
        raise ValueError('V9 circles_path must identify an absolute NPY file')
    if file_sha256(path) != case['circles_sha256']:
        raise ValueError('V9 obstacle geometry digest differs from the manifest')
    geometry = np.load(path, allow_pickle=False)
    count = case['circle_count']
    if (type(count) is not int or count <= 0 or geometry.shape != (count, 3)
            or geometry.dtype != np.float64 or not np.isfinite(geometry).all()):
        raise ValueError('V9 obstacle geometry must be a finite float64 [circle_count, 3] array')
    if (geometry[:, 2] <= 0).any():
        raise ValueError('V9 obstacle radii must be positive')
    return geometry


def validate_v9_case(case, bounds):
    required = {'seed', 'case_id', 'world_size', 'start', 'goal', 'circle_count', 'circles_sha256',
                'circles_path', 'direct_distance_m', 'reference_distance_m', 'kind', 'terrain'}
    if (not required <= case.keys() or case['terrain'] != 'synthetic' or case['kind'] != 'v9_random'
            or any(k in case for k in ('map_path','circles','reference_points','reference_route','reference_path'))):
        raise ValueError('Synthetic-to-WildScenes training requires a compact V9 random scene')
    world = _number(case, 'world_size')
    start, goal = _point(case, 'start'), _point(case, 'goal')
    direct = _number(case, 'direct_distance_m')
    if any((p < 0).any() or (p >= world).any() for p in (start, goal)):
        raise ValueError('V9 endpoints lie outside the simulation world')
    if not math.isclose(direct, np.linalg.norm(goal-start), abs_tol=1e-6, rel_tol=1e-8):
        raise ValueError('V9 direct distance differs from its endpoints')
    if not bounds[0]-1e-6 <= direct <= bounds[1]+1e-6 or _number(case,'reference_distance_m')+1e-6 < direct:
        raise ValueError('V9 task distances violate the training curriculum or reference lower bound')
    circles = load_v9_circles(case)
    if any((np.linalg.norm(circles[:, :2] - p, axis=1) <= circles[:, 2]).any() for p in (start, goal)):
        raise ValueError('V9 endpoint lies in an obstacle')


def _number(case, key):
    value = case.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'Raster case {key} must be a finite positive number')
    return float(value)


def _point(case, key):
    value = case.get(key)
    if (not isinstance(value, list) or len(value) != 2 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
        raise ValueError(f'Raster case {key} must contain two finite metric coordinates')
    return np.asarray(value, dtype=float)


def _sequence_group(sequence):
    match = re.fullmatch(r'([KV])-?0*[1-9][0-9]*', str(sequence), flags=re.IGNORECASE)
    if not match:
        raise ValueError('WildScenes sequence must identify a K or V acquisition')
    return 'Karawatha' if match.group(1).upper() == 'K' else 'Venman'


def validate_scene_manifest(manifest, config, ranges):
    """Check all cases against the declared curriculum and original raster files.

    The task lists are returned as JSON values, suitable for checkpoint config.
    Direct distance is measured from endpoints; reference distance is only a
    declared budget input, never a verified route length or policy input.
    Distinct sequences from the same park cannot establish a physical-map
    holdout. Explicit heldout_tasks mode instead checks task endpoint pairs.
    Endpoint clearance is checked here; connecting-route reachability is an
    external scene-audit responsibility, not established by this loader.
    """
    if not isinstance(manifest, dict) or set(manifest) != {'train', 'validation'}:
        raise ValueError('Scene manifest must contain exactly train and validation lists')
    episode_counts = (update_episode_counts(config) if config.get('update_plan') is not None else
                      [config['episodes_per_update']] * len(ranges))
    if len(ranges) != len(episode_counts):
        raise ValueError('Curriculum range count differs from the update schedule')
    if config.get('update_plan') is not None:
        declared_ranges = [row['distance_range_m'] for row in validate_update_plan(config)]
        if ranges != declared_ranges:
            raise ValueError('Curriculum ranges must match the explicit update_plan')
    task_ranges = [bounds for bounds, count in zip(ranges, episode_counts) for _ in range(count)]
    counts = dict(train=sum(episode_counts), validation=config['eval_scenes'])
    split_mode = config.get('validation_split', 'physical_map_group')
    if split_mode not in ('physical_map_group', 'heldout_tasks', 'synthetic_to_wildscenes'):
        raise ValueError('Unknown validation split protocol')
    for split, count in counts.items():
        if not isinstance(manifest[split], list) or len(manifest[split]) != count:
            raise ValueError(f'Scene manifest {split} must contain exactly {count} cases')
    # This also rejects non-JSON, nonfinite extras instead of serializing a
    # manifest that cannot later be loaded with weights_only=True.
    try:
        result = json.loads(json.dumps(manifest, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError('Scene manifest must contain finite JSON values') from error
    from .wildscenes_runtime import load_forest

    forests, checked_files, metadata_cache = {}, {}, {}
    groups = dict(train=set(), validation=set())
    hashes = dict(train=set(), validation=set())
    tasks = dict(train=[], validation=[])
    seen_seeds = set()
    required = {'terrain', 'map_path', 'map_sha256', 'map_group', 'world_size', 'start', 'goal',
                'circles', 'direct_distance_m', 'reference_distance_m', 'seed', 'kind',
                'resolution_m', 'required_clearance_m', 'source_unknown_space'}
    for split in ('train', 'validation'):
        for index, case in enumerate(result[split]):
            if not isinstance(case, dict):
                raise ValueError('Each scene must be a JSON object')
            if split_mode == 'synthetic_to_wildscenes':
                if split == 'train':
                    validate_v9_case(case, task_ranges[index])
                    seed = case['seed']
                    if type(seed) is not int or seed < 0 or seed in seen_seeds:
                        raise ValueError('V9 scene seeds must be unique nonnegative integers')
                    seen_seeds.add(seed)
                    continue
                if case.get('terrain') != 'wildscenes_raster':
                    raise ValueError('Synthetic-to-WildScenes validation requires original unknown-blocked raster maps')
            if not required <= set(case):
                raise ValueError(f'Raster case {split}[{index}] lacks required manifest fields')
            assumption = RASTER_SOURCE_ASSUMPTIONS.get(case['terrain'])
            if assumption is None or case['circles'] != []:
                raise ValueError('Raster cases require an explicit WildScenes raster protocol and no synthetic circles')
            if case['source_unknown_space'] != assumption:
                raise ValueError('Raster terrain and explicit source_unknown_space assumption disagree')
            if case.get('coordinate_scale', 1.) != 1.:
                raise ValueError('Raster case cannot rescale real metric coordinates')
            if any(key in case for key in ('reference_points', 'reference_path', 'reference_route')):
                raise ValueError('Reference route geometry must remain outside the training case manifest')
            if not isinstance(case['kind'], str) or not case['kind']:
                raise ValueError('Raster case kind must be a nonempty label')
            seed = case['seed']
            if type(seed) is not int or seed < 0 or seed in seen_seeds:
                raise ValueError('Raster task seeds must be unique nonnegative integers, excluding booleans')
            seen_seeds.add(seed)
            group = case['map_group']
            if group not in ('Karawatha', 'Venman'):
                raise ValueError('map_group must be Karawatha or Venman, not a sequence identity')
            groups[split].add(group)
            sha = case['map_sha256']
            if not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha):
                raise ValueError('map_sha256 must be the lowercase SHA-256 of the original raster')
            hashes[split].add(sha)
            if not isinstance(case['map_path'], str) or not Path(case['map_path']).is_absolute():
                raise ValueError('map_path must be an absolute NPZ path')
            path = Path(case['map_path']).resolve(strict=True)
            if path.suffix.lower() != '.npz':
                raise ValueError('map_path must identify an NPZ raster')
            if path not in checked_files:
                checked_files[path] = file_sha256(path)
            if checked_files[path] != sha:
                raise ValueError('Raster map SHA-256 differs from the declared map_sha256')
            case['map_path'] = str(path)
            sequences = [case['sequence']] if 'sequence' in case else []
            if case.get('metadata_path') is not None:
                if not isinstance(case['metadata_path'], str) or not Path(case['metadata_path']).is_absolute():
                    raise ValueError('metadata_path must be absolute when provided')
                metadata_path = Path(case['metadata_path']).resolve(strict=True)
                if metadata_path not in metadata_cache:
                    metadata_cache[metadata_path] = (file_sha256(metadata_path),
                        json.loads(metadata_path.read_text(encoding='utf-8')))
                metadata_sha, metadata = metadata_cache[metadata_path]
                if case.get('metadata_sha256', metadata_sha) != metadata_sha:
                    raise ValueError('Raster metadata SHA-256 changed since the manifest was saved')
                if not isinstance(metadata, dict) or metadata.get('map_sha256') != sha:
                    raise ValueError('Raster metadata does not bind to the declared map SHA-256')
                if metadata.get('coordinate_scale', 1.) != 1.:
                    raise ValueError('Raster metadata cannot rescale real metric coordinates')
                if metadata.get('source_unknown_space', assumption) != assumption:
                    raise ValueError('Raster metadata and declared source unknown-space assumption disagree')
                if metadata.get('terrain_protocol', case['terrain']) != case['terrain']:
                    raise ValueError('Raster metadata and declared terrain protocol disagree')
                if 'sequence' in metadata:
                    sequences.append(metadata['sequence'])
                case.update(metadata_path=str(metadata_path), metadata_sha256=metadata_sha)
            if any(_sequence_group(sequence) != group for sequence in sequences):
                raise ValueError('Sequence acquisition and physical map_group disagree')
            world = _number(case, 'world_size')
            resolution = _number(case, 'resolution_m')
            clearance = _number(case, 'required_clearance_m')
            if not math.isclose(clearance, 1.5, rel_tol=0., abs_tol=1e-9):
                raise ValueError('Raster tasks require the unchanged 1.5m clearance')
            start, goal = _point(case, 'start'), _point(case, 'goal')
            direct = _number(case, 'direct_distance_m')
            reference = _number(case, 'reference_distance_m')
            actual = float(np.linalg.norm(goal - start))
            if not math.isclose(actual, direct, rel_tol=1e-8, abs_tol=1e-6):
                raise ValueError('direct_distance_m does not match actual metric endpoints')
            if reference + 1e-6 < actual:
                raise ValueError('reference_distance_m cannot be shorter than direct distance')
            bounds = task_ranges[index] if split == 'train' else (config.get('validation_distances_m') or config['mission_distances_m'])
            if not bounds[0] - 1e-6 <= actual <= bounds[1] + 1e-6:
                raise ValueError(f'Raster case {split}[{index}] direct distance lies outside its curriculum range')
            forest_key = (path, case['terrain'], case.get('source_scene_path'), case.get('source_scene_sha256'))
            if forest_key not in forests:
                forests[forest_key] = load_forest(case, DEFAULT_REPO)
            forest = forests[forest_key]
            if (forest.metadata.get('terrain_protocol', case['terrain']) != case['terrain'] or
                    forest.metadata.get('source_unknown_space', assumption) != assumption):
                raise ValueError('Raster NPZ metadata cannot be relabeled with a different terrain or unknown-space protocol')
            if forest.metadata.get('coordinate_scale', 1.) != 1.:
                raise ValueError('Raster NPZ metadata cannot rescale metric coordinates')
            if case['terrain'] == 'wildscenes_simplified_raster':
                for key in ('source_scene_path', 'source_scene_sha256'):
                    if not case.get(key) or case[key] != forest.metadata.get(key):
                        raise ValueError('Simplified case provenance must match its raster metadata: ' + key)
            if case['terrain'] == 'wildscenes_simplified_raster' and not forest.known.all():
                raise ValueError('Simplified raster known masks must describe the complete simulated geometry')
            if not math.isclose(forest.resolution_m, resolution, rel_tol=0., abs_tol=1e-9):
                raise ValueError('Declared resolution differs from original raster resolution')
            if (forest.origin_xy < 0).any() or (forest.upper_xy > world + 1e-9).any():
                raise ValueError('Raster bounds do not fit the declared world without translation or scaling')
            if any((point < 0).any() or (point >= world).any() or not forest.point_clear(point, clearance)
                   for point in (start, goal)):
                raise ValueError('Raster endpoints violate original bounds or required clearance')
            native_origin = forest.metadata.get('native_origin_xy')
            if native_origin is not None:
                native_origin = np.asarray(native_origin, float)
                if native_origin.shape != (2,) or not np.isfinite(native_origin).all():
                    raise ValueError('Original raster coordinates require a finite native_origin_xy')
            tasks[split].append(dict(group=group, sha=sha, start=start, goal=goal, native_origin=native_origin))
    if split_mode == 'synthetic_to_wildscenes':
        return result
    if split_mode == 'physical_map_group':
        if groups['train'] & groups['validation']:
            raise ValueError('Train and validation must use disjoint physical map groups; sequences alone are insufficient')
        if hashes['train'] & hashes['validation']:
            raise ValueError('Train and validation cannot share a raster hash under different group labels')
    else:
        for train in tasks['train']:
            for validation in tasks['validation']:
                if train['group'] != validation['group']:
                    if train['sha'] == validation['sha']:
                        raise ValueError('Identical raster hashes cannot claim different physical map groups')
                    continue
                a, b, c, d = train['start'], train['goal'], validation['start'], validation['goal']
                if train['sha'] != validation['sha']:
                    if train['native_origin'] is None or validation['native_origin'] is None:
                        raise ValueError('Different rasters in one physical group need native_origin_xy to compare heldout tasks')
                    a, b = a + train['native_origin'], b + train['native_origin']
                    c, d = c + validation['native_origin'], d + validation['native_origin']
                same = np.linalg.norm(a - c) <= 2. and np.linalg.norm(b - d) <= 2.
                reverse = np.linalg.norm(a - d) <= 2. and np.linalg.norm(b - c) <= 2.
                if same or reverse:
                    raise ValueError('Heldout task repeats a training endpoint pair within 2m, including reversal')
    return result
