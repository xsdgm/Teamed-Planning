"""Offline tasks sampled along already audited, metric WildScenes routes.

Only endpoints and the corresponding route length enter a task manifest.
Reference polylines remain in the offline audit. Validation here means unseen
endpoint pairs on a shared map, not unseen geography or independent routes.
This module never loads a policy, changes map occupancy, or launches training.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist

from .hierarchical_sources import file_sha256 as _file_sha256


DISTANCE_TOLERANCE_M = 1e-6
_CONTRACT_KEYS = ('terrain', 'map_path', 'map_sha256', 'metadata_path', 'metadata_sha256',
                  'world_size', 'kind', 'circles', 'resolution_m', 'required_clearance_m',
                  'source_unknown_space', 'map_group', 'sequence', 'coordinate_scale',
                  'source_scene_path', 'source_scene_sha256')


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be a finite positive number')
    return float(value)


def _count(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return value


def _bounds(values):
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError('Each distance range must contain MIN_M and MAX_M')
    low, high = [_positive(value, 'distance bound') for value in values]
    if low > high:
        raise ValueError('Distance range must have MIN_M <= MAX_M')
    return low, high


def _contract(value):
    if not isinstance(value, dict):
        raise ValueError('Audited route requires a map_contract dictionary')
    required = set(_CONTRACT_KEYS) - {'metadata_path', 'metadata_sha256', 'sequence',
                                     'coordinate_scale', 'source_scene_path', 'source_scene_sha256'}
    if not required <= set(value):
        raise ValueError('Audited route lacks required map-contract fields')
    result = {key: deepcopy(value[key]) for key in _CONTRACT_KEYS if key in value}
    if result['terrain'] not in ('wildscenes_raster', 'wildscenes_simplified_raster') or result['circles'] != []:
        raise ValueError('Audited tasks require a raster map and no synthetic circles')
    if not isinstance(result['map_path'], str) or not Path(result['map_path']).is_absolute():
        raise ValueError('Audited map_path must be absolute')
    if not isinstance(result['map_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', result['map_sha256']):
        raise ValueError('Audited map_sha256 is invalid')
    simplified = result['terrain'] == 'wildscenes_simplified_raster'
    expected_unknown = 'free_by_simulation_assumption' if simplified else 'blocked'
    if result['source_unknown_space'] != expected_unknown:
        raise ValueError('Unknown source space must remain blocked for real maps or explicitly use the simplified simulation assumption')
    if result.get('coordinate_scale', 1.) != 1.:
        raise ValueError('WildScenes coordinates cannot be rescaled')
    if simplified:
        if result.get('coordinate_scale') != 1. or not isinstance(result.get('source_scene_path'), str) or not Path(result['source_scene_path']).is_absolute():
            raise ValueError('Simplified tasks require an absolute original scene path and unit coordinate scale')
        if not isinstance(result.get('source_scene_sha256'), str) or not re.fullmatch('[0-9a-f]{64}', result['source_scene_sha256']):
            raise ValueError('Simplified tasks require the original scene SHA-256')
    if not math.isclose(_positive(result['required_clearance_m'], 'clearance'), 1.5, abs_tol=1e-9, rel_tol=0.):
        raise ValueError('Audited routes must retain the 1.5m clearance')
    _positive(result['world_size'], 'world_size')
    _positive(result['resolution_m'], 'resolution_m')
    if result['map_group'] not in ('Karawatha', 'Venman'):
        raise ValueError('map_group must name a physical WildScenes acquisition site')
    if not isinstance(result['kind'], str) or not result['kind']:
        raise ValueError('kind must be a nonempty label')
    return result


def _points(value, world):
    try:
        points = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('Reference route must contain finite metric XY coordinates') from error
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Reference route must contain at least two finite metric XY points')
    if (points < 0).any() or (points >= world).any():
        raise ValueError('Reference route exceeds the declared metric world')
    # Repeated observations at the same point have no travel length.
    keep = np.r_[True, np.any(np.diff(points, axis=0) != 0., axis=1)]
    points = points[keep]
    if len(points) < 2:
        raise ValueError('Reference route has zero length')
    return points


def _diameter(points):
    unique = np.unique(points, axis=0)
    if len(unique) <= 2:
        return float(np.linalg.norm(unique[-1] - unique[0]))
    try:
        hull = unique[ConvexHull(unique).vertices]
    except QhullError:
        # For a collinear route, extremes along any varying coordinate are
        # its diameter. Reject other Qhull failures instead of hiding them.
        if np.linalg.matrix_rank(unique - unique[0]) > 1:
            raise ValueError('Cannot establish the reference-route Euclidean diameter')
        axis = int(np.ptp(unique, axis=0).argmax())
        return float(np.linalg.norm(unique[unique[:, axis].argmax()] - unique[unique[:, axis].argmin()]))
    return max(float(cdist(hull[i:i + 256], hull).max()) for i in range(0, len(hull), 256))


def load_verified_routes(audit_path):
    """Load and bind the supervisor's exact-segment route audit to map files.

    The upstream audit, not this sampler, performs geometric segment checking.
    This loader verifies its flags, map/metadata hashes, endpoints and distances.
    Returned route dictionaries are accepted by ``sample_task_manifest``.
    """
    audit_path = Path(audit_path).resolve(strict=True)
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit.get('status') != 'verified_kilometer_tasks_available':
        raise ValueError('Map audit has no verified kilometre task; training cannot be scheduled')
    if audit.get('source_unknown_space') != 'blocked' or audit.get('required_clearance_m') != 1.5:
        raise ValueError('Map audit changed unknown-space or clearance rules')
    cases = audit.get('kilometer_cases', [])
    route_rows = audit.get('route_audits', [])
    if not cases or not route_rows:
        raise ValueError('Verified map audit lacks task or route evidence')
    route_by_id = {}
    for row in route_rows:
        if not isinstance(row.get('case_id'), str) or row['case_id'] in route_by_id:
            raise ValueError('Route audit case IDs must be unique strings')
        route_by_id[row['case_id']] = row
    if len({case.get('case_id') for case in cases}) != len(cases):
        raise ValueError('Verified kilometre case IDs must be unique')
    checked_maps, checked_metadata = {}, {}
    result = []
    for case in cases:
        contract = _contract(case)
        path = Path(contract['map_path']).resolve(strict=True)
        if path != Path(audit['map_path']).resolve() or contract['map_sha256'] != audit['map_sha256']:
            raise ValueError('Verified task map does not match its parent audit')
        if path not in checked_maps:
            checked_maps[path] = _file_sha256(path)
        if checked_maps[path] != contract['map_sha256']:
            raise ValueError('Audited raster SHA-256 changed')
        metadata_path = Path(contract.get('metadata_path', path.with_name('map_metadata.json'))).resolve(strict=True)
        if metadata_path not in checked_metadata:
            checked_metadata[metadata_path] = (_file_sha256(metadata_path),
                json.loads(metadata_path.read_text(encoding='utf-8')))
        metadata_sha, metadata = checked_metadata[metadata_path]
        if metadata.get('map_sha256') != contract['map_sha256'] or metadata.get('coordinate_scale') != 1.:
            raise ValueError('Map metadata must bind to the original metric raster without scaling')
        if contract.get('metadata_sha256', metadata_sha) != metadata_sha:
            raise ValueError('Audited map metadata SHA-256 changed')
        if not math.isclose(metadata['resolution_m'], contract['resolution_m'], abs_tol=1e-9, rel_tol=0.):
            raise ValueError('Map metadata and case resolution disagree')
        contract.update(metadata_path=str(metadata_path), metadata_sha256=metadata_sha)
        route = route_by_id.get(case.get('case_id'))
        if route is None or route.get('all_segments_clear') is not True or route.get('required_clearance_m') != 1.5:
            raise ValueError('Task lacks an exact-segment 1.5m clearance audit')
        points = _points(route['reference_points'], contract['world_size'])
        if not (np.allclose(points[0], case['start'], rtol=0., atol=1e-7) and
                np.allclose(points[-1], case['goal'], rtol=0., atol=1e-7)):
            raise ValueError('Reference route endpoints differ from the verified task')
        direct = float(np.linalg.norm(points[-1] - points[0]))
        reference = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if direct < 1000. or not math.isclose(direct, case['direct_distance_m'], rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError('Verified task must have actual kilometre endpoint separation')
        if not math.isclose(reference, case['reference_distance_m'], rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError('Reference distance disagrees with the audited polyline')
        result.append(dict(route_id=case['case_id'], map_contract=contract,
            reference_points=points.tolist(), all_segments_clear=True, required_clearance_m=1.5,
            audit_path=str(audit_path), audit_sha256=_file_sha256(audit_path)))
    return result


def _candidate(points, cumulative, start_arc, distance):
    """First exit from a metric circle along the remaining verified polyline."""
    index = min(int(np.searchsorted(cumulative, start_arc, side='right')) - 1, len(points) - 2)
    fraction = (start_arc - cumulative[index]) / (cumulative[index + 1] - cumulative[index])
    start = points[index] + fraction * (points[index + 1] - points[index])
    suffix = points[index + 1:]
    outside = np.flatnonzero(np.linalg.norm(suffix - start, axis=1) >= distance)
    if not len(outside):
        return None
    goal_index = index + 1 + int(outside[0])
    before = start if goal_index == index + 1 else points[goal_index - 1]
    before_arc = start_arc if goal_index == index + 1 else cumulative[goal_index - 1]
    segment = points[goal_index] - before
    offset = before - start
    a = float(segment @ segment)
    b = float(2. * (offset @ segment))
    c = float(offset @ offset - distance ** 2)
    t = (-b + math.sqrt(max(0., b * b - 4. * a * c))) / (2. * a)
    if not -1e-9 <= t <= 1. + 1e-9:
        raise ArithmeticError('Route/circle intersection escaped its audited segment')
    t = float(np.clip(t, 0., 1.))
    goal = before + t * segment
    goal_arc = float(before_arc + t * math.sqrt(a))
    return start, goal, goal_arc


def _duplicate(pair, previous, tolerance):
    if not previous:
        return False
    prior = np.asarray(previous)
    same = (np.linalg.norm(prior[:, 0] - pair[0], axis=1) <= tolerance) & (
            np.linalg.norm(prior[:, 1] - pair[1], axis=1) <= tolerance)
    reverse = (np.linalg.norm(prior[:, 0] - pair[1], axis=1) <= tolerance) & (
               np.linalg.norm(prior[:, 1] - pair[0], axis=1) <= tolerance)
    return bool((same | reverse).any())


def sample_task_manifest(routes, update_ranges_m, episodes_per_update,
                         validation_distances_m, validation_count, seed, *,
                         endpoint_tolerance_m=2., max_attempts_per_task=2000,
                         case_acceptance=None, validation_routes=None, return_audit=False):
    """Sample exact counts without relaxing distance, clearance or uniqueness.

    ``routes`` contains audited ``reference_points`` and a ``map_contract``.
    Use ``load_verified_routes`` for production input. Validation pairs are
    reserved first and all train/validation same/reverse pairs are excluded
    when both corresponding endpoints are within ``endpoint_tolerance_m``.
    A fixed [D,D] request targets D+1e-7 m so floating roundoff cannot undershoot
    its minimum; the public numerical distance tolerance remains 1e-6 m.
    Optional ``case_acceptance(case) -> bool`` may apply additional offline
    geometry/sensor-support checks. It receives a copy, must not evaluate policy
    performance, and shares the same bounded attempt budget; errors propagate.
    This is same-map heldout-task evaluation, not spatial independence.
    Supply ``validation_routes`` to reserve validation for a distinct map/site.
    """
    _count(episodes_per_update, 'episodes_per_update')
    _count(validation_count, 'validation_count')
    _count(max_attempts_per_task, 'max_attempts_per_task')
    if case_acceptance is not None and not callable(case_acceptance):
        raise ValueError('case_acceptance must be a callable returning a boolean, or None')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    tolerance = _positive(endpoint_tolerance_m, 'endpoint_tolerance_m')
    if not isinstance(update_ranges_m, (list, tuple)) or not update_ranges_m:
        raise ValueError('At least one training update range is required')
    ranges = [_bounds(bounds) for bounds in update_ranges_m]
    validation_range = _bounds(validation_distances_m)
    def prepare(source_routes):
        result = []
        for route in source_routes:
            if route.get('all_segments_clear') is not True or route.get('required_clearance_m') != 1.5:
                raise ValueError('Sampler accepts only previously audited 1.5m routes')
            contract = _contract(route.get('map_contract'))
            points = _points(route['reference_points'], contract['world_size'])
            cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            result.append(dict(route=route, contract=contract, points=points,
                cumulative=cumulative, diameter=_diameter(points)))
        if not result:
            raise ValueError('No audited routes available')
        return result
    prepared = dict(train=prepare(routes))
    prepared['validation'] = prepared['train'] if validation_routes is None else prepare(validation_routes)
    maxima = {split: max(route['diameter'] for route in items) for split, items in prepared.items()}
    for split, required_maximum in [('train', max(bounds[1] for bounds in ranges)),
                                    ('validation', validation_range[1])]:
        if maxima[split] + DISTANCE_TOLERANCE_M < required_maximum:
            raise ValueError(f'Audited route Euclidean diameter {maxima[split]:.6f}m cannot cover requested '
                             f'{required_maximum:.6f}m distance for {split}; route arc length cannot substitute')
    contracts = [item['contract'] for items in prepared.values() for item in items]
    terrains = {contract['terrain'] for contract in contracts}
    unknown_protocols = {contract['source_unknown_space'] for contract in contracts}
    if len(terrains) != 1 or len(unknown_protocols) != 1:
        raise ValueError('Do not mix real unknown-blocked and simplified simulation task protocols')
    groups = {split: {item['contract']['map_group'] for item in items} for split, items in prepared.items()}
    separate_groups = groups['train'].isdisjoint(groups['validation'])
    rng = np.random.default_rng(seed)
    total = len(ranges) * episodes_per_update
    manifest = dict(train=[], validation=[])
    segments, seen = [], {}
    acceptance_checks = dict(train=0, validation=0)
    acceptance_rejections = dict(train=0, validation=0)

    def draw(split, index, bounds):
        low, high = bounds
        eligible = [r for r in prepared[split] if r['diameter'] + DISTANCE_TOLERANCE_M >= low]
        rejected_by_geometry = 0
        for attempt in range(max_attempts_per_task):
            item = eligible[int(rng.integers(len(eligible)))]
            points, cumulative = item['points'], item['cumulative']
            reversed_route = bool(rng.integers(2))
            if reversed_route:
                points = points[::-1]
                cumulative = cumulative[-1] - cumulative[::-1]
            start_arc = float(rng.uniform(0., cumulative[-1]))
            desired = low + 1e-7 if low == high else float(rng.uniform(low, high))
            proposed = _candidate(points, cumulative, start_arc, desired)
            if proposed is None:
                continue
            start, goal, goal_arc = proposed
            direct = float(np.linalg.norm(goal - start))
            if direct < low or direct > high + DISTANCE_TOLERANCE_M:
                continue
            pair = np.stack([start, goal])
            map_hash = item['contract']['map_sha256']
            if _duplicate(pair, seen.get(map_hash, []), tolerance):
                continue
            task_seed = seed + index if split == 'train' else seed + total + index
            case = dict(deepcopy(item['contract']), start=start.tolist(), goal=goal.tolist(),
                direct_distance_m=direct, reference_distance_m=float(goal_arc - start_arc),
                seed=task_seed, case_id=f'wildscenes-{split}-{index:05d}')
            if case_acceptance is not None:
                acceptance_checks[split] += 1
                accepted = case_acceptance(deepcopy(case))
                if not isinstance(accepted, (bool, np.bool_)):
                    raise TypeError('case_acceptance must return a boolean geometry decision')
                if not accepted:
                    acceptance_rejections[split] += 1
                    rejected_by_geometry += 1
                    continue
            seen.setdefault(map_hash, []).append(pair)
            manifest[split].append(case)
            segments.append(dict(split=split, index=index, case_id=case['case_id'],
                route_id=item['route'].get('route_id'),
                route_audit_path=item['route'].get('audit_path'),
                route_audit_sha256=item['route'].get('audit_sha256'),
                reversed_route=reversed_route, arc_interval_m=[start_arc, goal_arc],
                direct_distance_m=direct, reference_distance_m=case['reference_distance_m'],
                requested_distance_range_m=list(bounds), sampling_attempts=attempt + 1,
                geometry_acceptance_rejections=rejected_by_geometry))
            return
        raise ValueError(f'Cannot sample distinct {split}[{index}] in endpoint-distance range '
                         f'{low:g}..{high:g}m after {max_attempts_per_task} attempts with '
                         f'{tolerance:g}m same/reverse endpoint exclusion; task capacity/acceptance '
                         f'insufficient ({rejected_by_geometry} case_acceptance geometry rejections); '
                         'no range or clearance was relaxed')

    for index in range(validation_count):
        draw('validation', index, validation_range)
    for update, bounds in enumerate(ranges):
        for episode in range(episodes_per_update):
            draw('train', update * episodes_per_update + episode, bounds)
    audit = dict(schema='vtemp2_audited_route_task_sampling_v1', seed=seed,
        validation_scope='heldout_map_groups' if separate_groups else 'heldout_tasks',
        coordinate_scale=1., terrain=next(iter(terrains)), source_unknown_space=next(iter(unknown_protocols)),
        required_clearance_m=1.5, endpoint_tolerance_m=tolerance,
        numerical_distance_tolerance_m=DISTANCE_TOLERANCE_M,
        exact_distance_target_offset_m=1e-7,
        update_ranges_m=[list(bounds) for bounds in ranges], episodes_per_update=episodes_per_update,
        training_tasks=total, validation_distances_m=list(validation_range), validation_tasks=validation_count,
        maximum_audited_route_euclidean_diameter_m=max(maxima.values()),
        split_route_euclidean_diameter_m=maxima, segments=segments,
        case_acceptance_enabled=case_acceptance is not None,
        geometry_acceptance_checks=acceptance_checks,
        geometry_acceptance_rejections=acceptance_rejections,
        limitation=('Training and validation use different physical map groups; simplified geometry, if selected, is a simulation assumption and not real unknown-space traversability.'
                    if separate_groups else 'Heldout endpoint pairs share maps and may share route segments with training. This is not geographic, spatial, route or sensor-footprint independence.'))
    return (manifest, audit) if return_audit else manifest
