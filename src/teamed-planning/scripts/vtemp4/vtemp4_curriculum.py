"""Load kilometre missions, audit their split, and schedule balanced VTemp4 rollouts."""
from collections import Counter
from itertools import cycle
import hashlib
import json
import math
from pathlib import Path

from vtemp2_3.hierarchical_sources import file_sha256


SPLITS = ('ppo_train', 'development', 'test')
# Alternate a long detour with a control even when a batch contains fewer than four cases.
STRATA = ('detour_left', 'random_circles', 'detour_right', 'direct')
FIELD_TASKS = {'detour_left', 'detour_right', 'direct'}
KM_DATA = Path('artifacts/vtemp2-survey-km-data-20261119')
FIELD_DATA = Path('artifacts/vtemp4-detour-curriculum-v1')
LAKE_DATA = Path('artifacts/vtemp4-lake-curriculum-v1')
BRIDGE_GEOMETRY = 'river_bridge'
LAKE_GEOMETRY = 'bilateral_lake'
RUNTIME_FIELDS = (
    'case_id', 'kind', 'terrain', 'seed', 'world_size', 'start', 'goal', 'direct_distance_m',
    'circles_path', 'circles_sha256', 'circle_count', 'circles', 'map_path', 'map_sha256',
    'coordinate_scale', 'resolution_m', 'required_clearance_m', 'source_unknown_space',
)


def geometry_hash(case):
    return case['map_sha256'] if case['terrain'] == 'wildscenes_raster' else case['circles_sha256']


def source_id(case):
    return f"{case['terrain']}:{case['seed']}"


def runtime_case(case):
    """The simulator owns the geometry. Class, family, bridge and reference route stay off-policy."""
    return {key: case[key] for key in RUNTIME_FIELDS if key in case}


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _verify_field_audit(case, map_path, field_geometry):
    audit = _read(map_path.with_name('audit.json'))
    identity = ('map_sha256', 'terrain_family_id', 'navigation_task', 'seed', 'start', 'goal', 'detour_offset_m')
    if any(audit.get(key) != case[key] for key in identity):
        raise ValueError(f"Field geometry audit differs from the manifest for {case['case_id']}")
    if field_geometry == LAKE_GEOMETRY:
        _verify_lake_audit(case, audit)
        return
    if case.get('kind') == 'procedural_wooded_lake' or 'bridge_closure_disconnects_endpoints' not in audit:
        raise ValueError(f"Expected a river-bridge audit in {case['case_id']}")
    task = case['navigation_task']
    detour = task != 'direct'
    if (not audit['bridge_closure_disconnects_endpoints'] or audit['direct_line_hits_water'] != detour
            or audit['certified_clearance_lower_bound_m'] < case['required_clearance_m']):
        raise ValueError(f"Uncertified crossing or clearance in {case['case_id']}")
    if detour:
        sx, sy = case['start']
        gx, gy = case['goal']
        bx, by = audit['bridge_center_xy']
        side = (gx - sx) * (by - sy) - (gy - sy) * (bx - sx)
        if case['detour_offset_m'] < 200. or side == 0. or (side > 0.) != (task == 'detour_left'):
            raise ValueError(f"Crossing offset or turn direction differs from route class in {case['case_id']}")


def _verify_lake_audit(case, audit):
    case_id = case['case_id']
    if (case.get('kind') != 'procedural_wooded_lake' or audit.get('lake_closed') is not True
            or audit.get('bridge_cell_count') != 0 or audit.get('water_component_count') != 1):
        raise ValueError(f'Expected a closed lake without a bridge in {case_id}')
    task = case['navigation_task']
    preferred = task.removeprefix('detour_')
    clearance = case['required_clearance_m']
    if (not math.isfinite(clearance) or clearance < 1.5 or audit.get('preferred_side') != preferred
            or audit.get('direct_line_hits_water') is not (task != 'direct')):
        raise ValueError(f'Lake route class or clearance differs from the mission in {case_id}')
    entry = audit.get('lake_entry_station_m', -1.)
    gate = audit.get('lake_gate_station_m', -1.)
    exit_station = audit.get('lake_exit_station_m', -1.)
    left = audit.get('gate_left_shore_offset_m', 0.)
    right = audit.get('gate_right_shore_offset_m', 0.)
    bound = audit.get('certified_clearance_lower_bound_m', 0.)
    if (not 0 < entry < gate < exit_station < case['direct_distance_m'] or not left > right
            or not math.isfinite(bound) or bound < clearance):
        raise ValueError(f'Invalid lake stations, shores, or route clearance in {case_id}')
    lengths = {}
    sx, sy = case['start']
    dx = (case['goal'][0] - sx) / case['direct_distance_m']
    dy = (case['goal'][1] - sy) / case['direct_distance_m']
    for side in ('left', 'right'):
        path = audit.get(f'{side}_reference_route_xy')
        length = audit.get(f'{side}_route_length_m', 0.)
        bound = audit.get(f'{side}_clearance_lower_bound_m', 0.)
        if (not math.isfinite(length) or length <= 0 or not math.isfinite(bound) or bound < clearance
                or not path or len(path) < 2 or path[0] != case['start'] or path[-1] != case['goal']):
            raise ValueError(f'Uncertified {side} path around the lake in {case_id}')
        actual = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        if not math.isclose(actual, length, rel_tol=1.e-5, abs_tol=.02):
            raise ValueError(f'Lake {side} path length differs from its certificate in {case_id}')
        # Intersect the certificate polyline with the lake's central cross-section.
        # A repeated copy of the same flank must not certify two options.
        local = [((x - sx) * dx + (y - sy) * dy, (y - sy) * dx - (x - sx) * dy) for x, y in path]
        crossings = [a + (b - a) * (gate - u) / (v - u) for (u, a), (v, b) in zip(local, local[1:])
                     if u < gate <= v]
        if not crossings or not (max(crossings) >= left if side == 'left' else min(crossings) <= right):
            raise ValueError(f'Lake path does not pass the {side} shore in {case_id}')
        lengths[side] = length
    if task == 'direct':
        forward_clearance = audit.get('direct_line_clearance_lower_bound_m', 0.)
        if not math.isfinite(forward_clearance) or forward_clearance < clearance or case['detour_offset_m'] != 0.:
            raise ValueError(f'Uncertified direct passage by the lake in {case_id}')
    else:
        other = 'left' if preferred == 'right' else 'right'
        if (not right < 0. < left or not case['detour_offset_m'] >= 100.
                or lengths[other] - lengths[preferred] < 40.):
            raise ValueError(f'Lake detour is too small or the preferred side is incorrect in {case_id}')


def _bound_case(row, directory, field, field_geometry=None):
    case = row.copy()
    distance = math.dist(case['start'], case['goal'])
    if distance < 1000 - 1e-6 or not math.isclose(distance, case['direct_distance_m']):
        raise ValueError(f"Expected kilometre-scale endpoints in {case['case_id']}")
    if field:
        if (case.get('terrain') != 'wildscenes_raster' or case.get('navigation_task') not in FIELD_TASKS
                or not isinstance(case.get('terrain_family_id'), str) or not case['terrain_family_id']):
            raise ValueError(f"Invalid field route class, terrain or source family in {case['case_id']}")
        key = 'map_path'
        candidates = [directory / case[key]]
    else:
        if case.get('terrain') != 'synthetic' or case.get('kind') != 'v9_random':
            raise ValueError(f"Expected archived random V9 geometry in {case['case_id']}")
        key = 'circles_path'
        name = Path(case[key]).name
        # Prefer assets next to the supplied manifest over stale absolute archive locations.
        candidates = [directory / 'geometry' / name, directory / name, directory / case[key]]
        case['navigation_task'] = 'random_circles'
        case['terrain_family_id'] = f"v9-random-seed-{case['seed']}"
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Geometry missing for {case['case_id']}: {case[key]}")
    if file_sha256(path) != geometry_hash(case):
        raise ValueError(f"Geometry digest differs from the manifest for {case['case_id']}")
    if field:
        _verify_field_audit(case, path, field_geometry)
    case[key] = str(path.resolve())
    return case


def load_curriculum(km_folder, field_folder=None, *, field_geometry=BRIDGE_GEOMETRY):
    """Return split records and manifest hashes. Check maps and their source families across all splits."""
    if field_geometry not in (BRIDGE_GEOMETRY, LAKE_GEOMETRY) or (field_geometry == LAKE_GEOMETRY and field_folder is None):
        raise ValueError(f'Expected a data directory for field geometry {field_geometry!r}')
    paths = {'scenes_sha256': km_folder / 'scenes.json'}
    circle_rows = _read(paths['scenes_sha256'])
    splits = {split: [_bound_case(row, km_folder, False) for row in circle_rows[split]] for split in SPLITS}
    far = km_folder / 'far-test' / 'scenes.json'
    if far.is_file():
        paths['far_scenes_sha256'] = far
        splits['test'].extend(_bound_case(r, far.parent, False) for r in _read(far))
    if field_folder is not None:
        path = field_folder / 'scenes.json'
        if not path.is_file():
            builder = 'build_vtemp4_lake_suite' if field_geometry == LAKE_GEOMETRY else 'build_vtemp4_detour_suite'
            script = f'src/teamed-planning/scripts/simulator/{builder}.py'
            raise FileNotFoundError(f'Missing {path}. In teamedplanning-2d, from the checkout run: python -I -B {script} --output "{field_folder}"')
        paths['field_scenes_sha256'] = path
        field_rows = _read(path)
        for split in SPLITS:
            rows = [_bound_case(row, field_folder, True, field_geometry) for row in field_rows[split]]
            if {row['navigation_task'] for row in rows} != FIELD_TASKS:
                raise ValueError(f'Expected left, right and direct field cases in {split}')
            splits[split].extend(rows)
    ids = set()
    for split, cases in splits.items():
        if not cases or not any(c['navigation_task'] == 'random_circles' for c in cases):
            raise ValueError(f'Expected random kilometre cases in {split}')
        for case in cases:
            if case['case_id'] in ids:
                raise ValueError(f"Duplicate case ID {case['case_id']}")
            ids.add(case['case_id'])
    for key in (geometry_hash, source_id, lambda c: c['terrain_family_id']):
        seen = set()
        for split, cases in splits.items():
            keys = {key(c) for c in cases}
            if keys & seen:
                raise ValueError(f'Map geometry or source family overlaps another split in {split}')
            seen.update(keys)
    hashes = {key: file_sha256(path) for key, path in paths.items()}
    if field_geometry == LAKE_GEOMETRY:
        audits = {c['case_id']: file_sha256(Path(c['map_path']).with_name('audit.json'))
                  for rows in splits.values() for c in rows if c['terrain'] == 'wildscenes_raster'}
        hashes['field_audits_sha256'] = hashlib.sha256(json.dumps(audits, sort_keys=True).encode()).hexdigest()
    return splits, hashes


def balanced_cases(cases, rng):
    """Cycle through all strata, drawing within each without replacement until its deck is exhausted."""
    groups = {task: [c for c in cases if c['navigation_task'] == task] for task in STRATA}
    decks = {task: [] for task in groups}
    for task in cycle(task for task in STRATA if groups[task]):
        if not decks[task]:
            decks[task] = groups[task].copy()
            rng.shuffle(decks[task])
        yield decks[task].pop()


def stratum_counts(cases):
    return dict(Counter(c['navigation_task'] for c in cases))
