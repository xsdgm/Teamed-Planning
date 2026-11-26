"""Unscaled kilometre evaluation cases and privileged, post-rollout ground-path measures."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import label


REPO = Path(__file__).resolve().parents[4]
CIRCLE_RUN = 'vtemp5-fleet-circle-rejoin-formal-16w-eval4-20261124'
WATER_RUN = 'vtemp5-fleet-bridge-lake-after-circle-16w-20261124'


def _read(path):
    return json.loads(path.read_text())


def _scene(case, kind, fleet_seed, formal, seconds_per_m):
    start, goal = np.asarray(case['start']), np.asarray(case['goal'])
    direct = np.linalg.norm(goal-start)
    if not 1000 <= direct <= 1500 or not math.isclose(direct, case['direct_distance_m']):
        raise ValueError(f"Invalid kilometre course distance for {case['case_id']}")
    forward = (goal-start)/direct
    return dict(case=case, case_id=case['case_id'], kind=kind, seed=case['seed'],
                fleet_seed=fleet_seed, formal_eval=formal, attempts=1, start=case['start'],
                goal=case['goal'], circles=case['circles'], rects=[], world_size=case['world_size'],
                direct_distance_m=float(direct), reference_distance_m=case['reference_distance_m'],
                max_steps=math.ceil(seconds_per_m * direct + 120), forward=forward.tolist(),
                left=[float(-forward[1]), float(forward[0])])


def km_cases(repo=REPO):
    """Return six original circles and six single-deck/closed-lake development cases each."""
    root = Path(repo)/'artifacts'
    circle = _read(root/CIRCLE_RUN/'manifest.json')
    water = _read(root/WATER_RUN/'manifest.json')
    formal = {r['case_id']: r for r in water['evaluation']}
    result = []
    for row in circle['evaluation']:
        case = row['case']
        (center, radius), = case['circles']
        mission = np.array([case['world_size'], *case['start'], *case['goal'], *center, radius], dtype='<f8')
        if hashlib.sha256(mission.tobytes()).hexdigest() != case['map_sha256']:
            raise ValueError(f"Circle geometry digest mismatch for {case['case_id']}")
        result.append(_scene(case, 'circle', row['fleet_seed'], True, circle['seconds_per_m']))

    for kind, suite in [('bridge', 'vtemp4-detour-curriculum-v1'), ('shore', 'vtemp4-lake-curriculum-v1')]:
        folder = root/suite
        for case in _read(folder/'scenes.json')['development']:
            case = dict(case)
            path = (folder/case['map_path']).resolve()
            audit = _read(path.with_name('audit.json'))
            if (hashlib.sha256(path.read_bytes()).hexdigest() != case['map_sha256']
                or audit['map_sha256'] != case['map_sha256']
                or audit['certified_clearance_lower_bound_m'] < case['required_clearance_m']
                or not np.allclose(audit['reference_route_xy'][0], case['start'])
                or not np.allclose(audit['reference_route_xy'][-1], case['goal'])):
                raise ValueError(f"Invalid route certificate for {case['case_id']}")
            if not audit['bridge_closure_disconnects_endpoints' if kind == 'bridge' else 'lake_closed']:
                raise ValueError(f"Uncertified water geometry for {case['case_id']}")
            case['map_path'] = str(path)
            original = formal.get(case['case_id'])
            if original is not None and original['case']['map_sha256'] != case['map_sha256']:
                raise ValueError(f"Historical eval used a different map for {case['case_id']}")
            # Match the archived fleet on all historical maps. For extra layouts,
            # use one reproducible positive seed derived from the map identity.
            seed = original['fleet_seed'] if original else 1 + int(case['map_sha256'][:8], 16) % (2**31-1)
            scene = _scene(case, kind, seed, original is not None, water['seconds_per_m'])
            scene.update(audit_path=str(path.with_name('audit.json')), direct_line_hits_water=audit['direct_line_hits_water'])
            result.append(scene)
    if len(result) != 18 or len({r['case_id'] for r in result}) != len(result):
        raise ValueError('Expected six distinct development cases of each terrain')
    return result


@lru_cache(maxsize=2)
def _river_regions(path):
    """Label dry banks separately after removing the whole watercourse, including the deck."""
    path = Path(path)
    with np.load(path.with_name('landscape.npz')) as raster:
        cover = raster['landcover']
    with np.load(path) as raster:
        if not np.array_equal(raster['occupied'], np.isin(cover, [3, 4, 7, 8])):
            raise ValueError('Bridge labels disagree with native source occupancy')
    deck = cover == 6
    banks, _ = label((cover != 4) & ~deck)
    # River pixels become -1 and deck pixels remain 0.
    banks[cover == 4] = -1
    return banks


def _raster_cells(points):
    """Visit every half-open pixel traversed by a segment, even when poses are sparse."""
    samples = [points[0:1]]
    for a, b in zip(points[:-1], points[1:]):
        if np.array_equal(np.floor(a), np.floor(b)):
            continue
        times = [[0., 1.]]
        for axis in range(2):
            low, high = sorted((a[axis], b[axis]))
            edges = np.arange(math.floor(low)+1, math.ceil(high))
            if len(edges):
                times.append((edges-a[axis])/(b[axis]-a[axis]))
        times = np.unique(np.concatenate(times))
        samples.extend([a + ((times[1:] + times[:-1])*.5)[:, None]*(b-a), b[None]])
    return np.floor(np.vstack(samples)).astype(int)


def _bridge_pass(scene, path):
    banks = _river_regions(scene['case']['map_path'])
    start, goal = np.floor([scene['start'], scene['goal']]).astype(int)
    a, b = banks[start[1], start[0]], banks[goal[1], goal[0]]
    if a <= 0 or b <= 0 or a == b:
        raise ValueError('Original deck closure does not split the two dry banks')
    x, y = _raster_cells(path).T
    inside = (x >= 0) & (x < banks.shape[1]) & (y >= 0) & (y < banks.shape[0])
    regions = np.full(len(x), -2, dtype=int)
    regions[inside] = banks[y[inside], x[inside]]
    # Compress consecutive cells within the same bank or deck. Touching the deck,
    # turning back, moving through water or leaving the map cannot earn a crossing.
    regions = regions[np.r_[True, regions[1:] != regions[:-1]]]
    return bool(np.any((regions[:-2] == a) & (regions[1:-1] == 0) & (regions[2:] == b)))


def km_route_metrics(scene, path):
    """Score the recorded physical UGV trajectory only. Source geometry is for scoring."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or not len(path) or not np.isfinite(path).all():
        raise ValueError('Expected a finite, nonempty UGV path in world XY')
    start = np.asarray(scene['start'])
    if not np.allclose(path[0], start, atol=1e-4, rtol=0):
        raise ValueError('Missing initial UGV position')
    longitudinal = (path - start) @ scene['forward']
    lateral = (path - start) @ scene['left']
    result = dict(crossed_bridge=None, passed_obstacle=False,
                  max_abs_cross_track_m=float(np.abs(lateral).max()))
    if scene['kind'] == 'bridge':
        result['passed_obstacle'] = result['crossed_bridge'] = _bridge_pass(scene, path)
        return result
    if scene['kind'] == 'circle':
        (center, radius), = scene['circles']
        far = (np.asarray(center)-start) @ scene['forward'] + radius + scene['case']['required_clearance_m']
        passed = np.flatnonzero(longitudinal >= far)
        if len(passed):
            # An interpolated path through the circle cannot qualify as a bypass.
            points = path[:passed[0]+1]
            travel = np.diff(points, axis=0)
            length2 = np.einsum('ij,ij->i', travel, travel)
            along = np.einsum('ij,ij->i', np.asarray(center)-points[:-1], travel)
            t = np.divide(along, length2, out=np.zeros_like(along), where=length2 > 0)
            nearest = points[:-1] + np.clip(t, 0., 1.)[:, None]*travel
            result['passed_obstacle'] = bool((np.linalg.norm(nearest-center, axis=1) > radius).all())
        return result
    if scene['kind'] != 'shore':
        raise ValueError(f"Unknown terrain kind {scene['kind']}")
    audit = _read(Path(scene['audit_path']))
    gate = audit['lake_gate_station_m']
    crossings = np.flatnonzero((longitudinal[:-1] < gate) & (longitudinal[1:] >= gate))
    side = None
    for i in crossings:
        frac = (gate-longitudinal[i])/(longitudinal[i+1]-longitudinal[i])
        y = lateral[i] + frac*(lateral[i+1]-lateral[i])
        if y > audit['gate_left_shore_offset_m'] or y < audit['gate_right_shore_offset_m']:
            side = 'left' if y > audit['gate_left_shore_offset_m'] else 'right'
            if np.any(longitudinal[i+1:] >= audit['lake_exit_station_m']):
                result['passed_obstacle'] = True
                break
    result.update(mid_lake_shore_side=side, bypass_required=bool(audit['direct_line_hits_water']))
    return result
