"""Build native kilometre missions around closed lakes with two traversable shores."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as nd
from skimage.graph import MCP_Geometric

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vtemp4.build_vtemp4_detour_suite import rotate_xy
from vtemp4.generate_vtemp4_field import CLEARANCE, ROOT, SIDE, landscape, track_distance, write_json


FAMILIES = {
    'development': [820013, 820021],
    'test': [930011, 930037],
    'ppo_train': [710009, 710023, 710033, 710047, 710057, 710063],
}
PREFIX = {'development': 'dev', 'test': 'test', 'ppo_train': 'train'}
TASKS = ('detour_left', 'detour_right', 'direct')
LANDCOVER = ('meadow', 'open_woodland', 'closed_woodland', 'dense_scrub', 'lake',
             'dirt_track', 'unused', 'tree_trunk', 'rock')
ROUTE_CLEARANCE = 6.0
SOURCE = 'procedural_wooded_lake'


def route_on_shore(clearance, water, gate, start, goal, left, rotation):
    """Separate the two routes only on the transverse cut through the lake."""
    safe = clearance >= ROUTE_CLEARANCE
    shore_rows = np.flatnonzero(water[:, gate])
    if left:
        safe[:shore_rows[-1]+1, gate] = False
    else:
        safe[shore_rows[0]:, gate] = False
    a, b = (tuple(np.floor(point[::-1]).astype(int)) for point in (start, goal))
    planner = MCP_Geometric(np.where(safe, 1., np.inf))
    costs, _ = planner.find_costs([a], [b])
    if not np.isfinite(costs[b]):
        raise ValueError('Missing a wide, clearance-certified shoreline passage')
    path = np.asarray(planner.traceback(b), dtype=int)
    steps = np.diff(path, axis=0)
    corners = np.r_[True, np.any(steps[1:] != steps[:-1], axis=1), True]
    xy = [rotate_xy(point, rotation) for point in (path[corners, ::-1]+.5)]
    return float(costs[b]), float(clearance[tuple(path.T)].min()), xy


def audit_map(cover, water, start, goal, centre, rotation, task):
    occupied = np.isin(cover, [3, 4, 7, 8])
    # For an 8-connected centreline, subtract one half-diagonal to the
    # occupied square and a second one for points in between grid centres.
    clearance = nd.distance_transform_edt(np.pad(~occupied, 1))[1:-1, 1:-1] - np.sqrt(2.)
    if (water[0].any() or water[-1].any() or water[:, 0].any() or water[:, -1].any()
            or nd.label(water)[1] != 1 or not np.array_equal(water, nd.binary_fill_holes(water))
            or not np.array_equal(water, cover == 4) or (cover == 6).any()):
        raise ValueError('Expected a single closed lake with solid water and no crossing')
    gate = int(centre[0])
    ys = np.flatnonzero(water[:, gate])
    if len(ys) < 1 or not np.all(np.diff(ys) == 1):
        raise ValueError('Expected an uninterrupted transverse slice of the lake')
    # The row-edge values are the boundaries of continuous occupied squares.
    left_edge, right_edge = ys[-1]+1-start[1], ys[0]-start[1]
    lower_bound = max(0., min(left_edge, -right_edge))
    start_col, end_col = int(start[0]), int(goal[0])
    direct_water = water[int(start[1]), start_col:end_col+1]
    direct_bound = float(clearance[int(start[1]), start_col:end_col+1].min())
    direct_length = float(goal[0]-start[0])
    side_routes = {side: route_on_shore(clearance, water, gate, start, goal, side == 'left', rotation)
                   for side in ('left', 'right')}
    preferred = task.removeprefix('detour_')
    if task == 'direct':
        if direct_water.any() or direct_bound < ROUTE_CLEARANCE or lower_bound > 0:
            raise ValueError('Direct line intersects obstacles or lacks a wide passable corridor')
        length, bound = direct_length, direct_bound
        route = [rotate_xy(p, rotation) for p in (start, goal)]
    else:
        other = 'right' if preferred == 'left' else 'left'
        length, bound, route = side_routes[preferred]
        if (not direct_water.any() or lower_bound < 100. or side_routes[other][0] - length < 40.):
            raise ValueError(f'{task} lacks the required detour or shorter-side separation: '
                             f'offset={lower_bound:.1f}, left={side_routes["left"][0]:.1f}, '
                             f'right={side_routes["right"][0]:.1f}')
    wet_columns = np.flatnonzero(water.any(axis=0))
    first_hit = float(start_col + np.flatnonzero(direct_water)[0] - start[0]) if direct_water.any() else None
    audit = dict(start=rotate_xy(start, rotation), goal=rotate_xy(goal, rotation),
                 lake_center_xy=rotate_xy(centre, rotation), lake_gate_station_m=gate+.5-start[0],
                 lake_entry_station_m=float(wet_columns[0]-start[0]),
                 lake_exit_station_m=float(wet_columns[-1]+1-start[0]),
                 gate_left_shore_offset_m=float(left_edge), gate_right_shore_offset_m=float(right_edge),
                 barrier_distance_m=first_hit, detour_offset_m=float(lower_bound),
                 direct_distance_m=direct_length, reference_distance_m=length, reference_route_xy=route,
                 preferred_side=preferred, direct_line_hits_water=bool(direct_water.any()),
                 direct_line_clearance_lower_bound_m=direct_bound, required_clearance_m=CLEARANCE,
                 certified_clearance_lower_bound_m=bound, lake_closed=True, bridge_cell_count=0,
                 water_component_count=1, shore_passage_required_clearance_m=ROUTE_CLEARANCE,
                 water_area_m2=int(water.sum()))
    for side, (distance, safe_bound, xy) in side_routes.items():
        audit.update({f'{side}_route_length_m': distance, f'{side}_reference_route_xy': xy,
                      f'{side}_clearance_lower_bound_m': safe_bound})
    return occupied, audit


def make_family(root, split, index, seed):
    rng = np.random.default_rng(seed)
    family = f'lake-{PREFIX[split]}-{index:02d}'
    rotation = (index + {'ppo_train': 0, 'development': 1, 'test': 3}[split]) % 4
    y0 = rng.integers(705, 729) + .5
    start = np.array([rng.integers(105, 180)+.5, y0])
    goal = np.array([rng.integers(1260, 1335)+.5, y0])
    rx = rng.uniform(130., 181.) if index % 2 == 0 else rng.uniform(165., 220.)
    short = rng.uniform(124., 151.) if index % 2 == 0 else rng.uniform(191., 217.)
    shift = rng.uniform(59., 77.)
    ry = short+shift
    cx = np.floor(start[0] + rx + rng.uniform(139., 215.) + (index % 3)*64.)+.5
    phases = rng.uniform(0., 2*np.pi, 3)
    row, col = np.ogrid[:SIDE, :SIDE]
    x, y = col+.5, row+.5

    def shore(t, cy):
        # Both banks end at the same two headlands. Harmonic terms vary the
        # shoreline while preserving the exact central cut across each family.
        t = np.clip(t, -1., 1.)
        wave = 15.*(np.sin(3.1*t+phases[0])-np.sin(phases[0]))
        wave += 6.*(np.sin(7.9*t+phases[1])-np.sin(phases[1]))
        width = ry*np.sqrt(np.maximum(0., 1.-t*t))*(1.+.065*(np.sin(5.8*t+phases[2])-np.sin(phases[2])))
        return cy+wave+width, cy+wave-width

    # Reuse the wooded substrate and individual native tree footprints with its
    # river channel positioned entirely outside the map. Lakes are added below.
    hills = [[start, goal],
             [(1, y0-360), (200, y0-400), (470, y0-355), (670, y0-408), (1060, y0-370), (1438, y0-440)],
             [(30, y0+475), (345, y0+426), (620, y0+435), (935, y0+382), (1300, y0+419)]]
    elevation, substrate, canopy, tree_count = landscape(
        seed, channel=lambda _: SIDE+5000., lines=hills, endpoints=[start, goal], example_bridge=False)
    rows = []
    for task in TASKS:
        if task == 'direct':
            offset = (1 if seed % 4 == 1 else -1) * (ry+79.)
        else:
            offset = -shift if task == 'detour_left' else shift
        cy = y0+offset
        upper, lower = shore((x-cx)/rx, cy)
        water = (np.abs(x-cx) <= rx) & (y <= upper) & (y >= lower)
        t = np.linspace(-.98, .98, 33)
        upper, lower = shore(t, cy)
        tracks = [[start, goal]]
        for sign, edge in ((1, upper), (-1,lower)):
            margin = 18.+4.*np.sin(4.*t+phases[1])
            rim = np.column_stack((cx+rx*t, edge+sign*margin))
            tracks.append([start, [cx-rx-48., .65*rim[0,1]+.35*y0],
                           *rim, [cx+rx+62., .76*rim[-1,1]+.24*y0], goal])
        trail = track_distance(tracks)
        shore_dist = nd.distance_transform_edt(~water)
        # Meadow fringes and broad trail approaches can be driven across their
        # full width. Dirt is a surface class within this open terrain.
        meadow = (trail <= 15.) | (shore_dist < 29.+5.*np.sin(x/65.+y/93.+phases[2]))
        cover = substrate.copy()
        cover[meadow] = 0
        cover[trail <= 4.5] = 5
        cover[water] = 4
        crowns = canopy.copy()
        crowns[meadow | water] = 0.
        mix = np.clip(shore_dist/95., 0., 1.)
        mix = mix*mix*(3.-2.*mix)
        level = np.percentile(elevation[water], 25)
        relief = np.round(level + (elevation-level)*mix, 2).astype(np.float32)
        occupied, audit = audit_map(cover, water, start, goal, [cx,cy], rotation, task)
        folder = root/split/family/task
        folder.mkdir(parents=True, exist_ok=True)
        map_path = folder/'forest_map.npz'
        np.savez_compressed(map_path, occupied=np.rot90(occupied,rotation), known=np.ones_like(occupied),
                            resolution_m=1., origin_xy=[0.,0.], terrain=np.rot90(relief,rotation),
                            terrain_protocol='wildscenes_raster', source_unknown_space='blocked',
                            map_source=SOURCE, coordinate_scale=1.)
        np.savez_compressed(folder/'landscape.npz', landcover=np.rot90(cover,rotation),
                            canopy_height_m=np.rot90(crowns,rotation))
        digest = hashlib.sha256(map_path.read_bytes()).hexdigest()
        case = dict(case_id=f'{family}-{task}', seed=seed, kind=SOURCE, map_source=SOURCE,
                    terrain='wildscenes_raster', source_unknown_space='blocked', world_size=SIDE,
                    start=audit['start'], goal=audit['goal'], circles=[], map_path=map_path.name,
                    map_sha256=digest, coordinate_scale=1., resolution_m=1., required_clearance_m=CLEARANCE,
                    navigation_task=task, terrain_family_id=family, direct_distance_m=audit['direct_distance_m'],
                    reference_distance_m=audit['reference_distance_m'], detour_offset_m=audit['detour_offset_m'],
                    barrier_distance_m=audit['barrier_distance_m'])
        audit.update(seed=seed, terrain_family_id=family, map_sha256=digest, navigation_task=task,
                     rotation_quarter_turns=rotation, substrate_tree_count=tree_count, landcover=list(LANDCOVER))
        write_json(folder/'audit.json', audit)
        write_json(folder/'case.json', case)
        rows.append(dict(case, map_path=map_path.relative_to(root).as_posix()))
    return rows


def generate_suite(root, workers):
    seeds = [s for group in FAMILIES.values() for s in group]
    if len(seeds) != len(set(seeds)):
        raise ValueError('A base terrain seed occurs in multiple families')
    scenes = {split: [] for split in FAMILIES}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(split, pool.submit(make_family, root, split, i, seed))
                for split, seeds in FAMILIES.items() for i, seed in enumerate(seeds)]
        for split, job in jobs:
            rows = job.result()
            scenes[split].extend(rows)
            print(f"{rows[0]['terrain_family_id']} completed", flush=True)
    all_rows = [c for group in scenes.values() for c in group]
    if len(set(c['map_sha256'] for c in all_rows)) != len(all_rows):
        raise ValueError('Repeated lake map geometry')
    write_json(root/'suite.json', dict(format='vtemp4_lake_curriculum_v1', seeds=FAMILIES,
               topology='one closed lake; both shores traversable', landcover=list(LANDCOVER),
               detour_offset='guaranteed lateral deviation for a ground path at the central transverse slice',
               required_route_clearance_m=ROUTE_CLEARANCE,
               splits={s:len(group) for s,group in scenes.items()}))
    write_json(root/'scenes.json', scenes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/vtemp4-lake-curriculum-v1')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    generate_suite(args.output.resolve(), args.workers)
