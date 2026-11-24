"""Build split-disjoint, matched valley tasks for kilometre-scale heading decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vtemp4.generate_vtemp4_field import CLEARANCE, LANDCOVER, ROOT, SIDE, audit_geometry, landscape, write_json


# Independent river shapes, forest draws and endpoints across splits. Each seed's
# three deck placements are kept together, including in held-out evaluation.
FAMILIES = {
    'development': [510239, 510283],
    'test': [630439, 630443],
    'ppo_train': [410013, 410021, 410037, 410059, 410081, 410099],
}
PREFIXES = {'development': 'dev', 'test': 'test', 'ppo_train': 'train'}
TASKS = ('detour_left', 'detour_right', 'direct')


def rotate_xy(point, k):
    # np.rot90 on a [y,x] array with +y north maps (x,y) to (y,SIDE-x).
    x, y = point
    for _ in range(k):
        x, y = y, SIDE-x
    return [float(x), float(y)]


def family_geometry(seed, early, rotation):
    rng = np.random.default_rng(seed)
    start = np.array([rng.integers(100, 195), rng.integers(595, 801)]) + .5
    goal = np.array([rng.integers(1240, 1340) + .5, start[1] + rng.integers(-140,141)])
    direction = (goal-start) / np.linalg.norm(goal-start)
    normal = np.array([-direction[1], direction[0]])
    middle = start[0] + (rng.uniform(105,159) if early else rng.uniform(400,635))
    centre_y = start[1] + (middle-start[0])*direction[1]/direction[0]
    amp = rng.uniform([28,18,5], [70,42,12])
    periods = rng.uniform([150,65,24], [240,115,45])
    phases = rng.uniform(0, 2*np.pi, 3)

    def channel(y):
        return middle + sum(a*(np.sin(y/p+phase)-np.sin(centre_y/p+phase))
                            for a,p,phase in zip(amp,periods,phases))

    offsets = [rng.uniform(242,345), -rng.uniform(242,345), 0.]
    centres = []
    for offset in offsets:
        # Choose the physical thalweg point at the specified signed distance
        # from the infinite start-goal line.
        def residual(y):
            return ((channel(y)-start[0])*normal[0] + (y-start[1])*normal[1]) - offset
        y = brentq(residual, 20., SIDE-20.)
        centres.append(np.array([channel(y), y]))

    roads = [[start,goal]]
    for sign in (-1, 1):
        ys = np.linspace(min(c[1] for c in centres)-80, max(c[1] for c in centres)+80, 21)
        phase = rng.uniform(0, 2*np.pi)
        roads.append([(channel(y)+sign*(50.+10.*np.sin(y/75.+phase)), y) for y in ys])
    for c in centres[:2]:
        # Both banks have trails to all three potential crossings in every
        # counterfactual. Only the observed river/deck can distinguish them.
        roads.append([c-95.*direction,c+95.*direction])
    elevation, base, canopy, tree_count = landscape(seed, channel=channel, lines=roads,
                                                    endpoints=[start,goal], example_bridge=False)
    y, x = np.ogrid[:SIDE,:SIDE]
    lateral = (x+.5-start[0])*normal[0] + (y+.5-start[1])*normal[1]
    water = base == 4
    direct = np.linspace(start,goal,int(np.linalg.norm(goal-start)*5))
    ix, iy = np.floor(direct).astype(int).T
    hits = np.flatnonzero(water[iy,ix])
    if not len(hits):
        raise ValueError('Missing river along the centre line')
    barrier_distance = float(np.linalg.norm(direct[hits[0]]-start))
    if early and not 70. <= barrier_distance <= 170.:
        raise ValueError(f'Early encounter distance out of range: {barrier_distance:.1f}')
    fields = dict(elevation=np.rot90(elevation, rotation), canopy=np.rot90(canopy, rotation),
                  start=rotate_xy(start,rotation), goal=rotate_xy(goal,rotation), tree_count=tree_count,
                  barrier_distance=barrier_distance)
    for task,offset,c in zip(TASKS,offsets,centres):
        cover = base.copy()
        deck = water & (np.abs(lateral-offset) <= 6.)
        cover[deck] = 6
        minimum_offset = float(np.min(np.abs(lateral[deck])))
        if task != 'direct' and minimum_offset - np.sqrt(.5) < 200.:
            raise ValueError('Deck too close to centre line for a large detour')
        yield task, fields, np.rot90(cover, rotation), rotate_xy(c,rotation), minimum_offset


def make_family(root, split, index, seed):
    family = f'valley-{PREFIXES[split]}-{index:02d}'
    rotation = (index + {'development':1, 'test':3, 'ppo_train':0}[split]) % 4
    early = index % 2 == 0
    rows = []
    for task,world,cover,bridge,offset in family_geometry(seed,early,rotation):
        folder = root/split/family/task
        folder.mkdir(parents=True,exist_ok=True)
        occupied = np.isin(cover,[3,4,7,8])
        audit = audit_geometry(occupied,cover,start=world['start'],goal=world['goal'],
                               bridge_center=bridge,direct_route=task=='direct')
        audit.update(navigation_task=task, terrain_family_id=family,
                     detour_offset_m=offset,barrier_distance_m=world['barrier_distance'],
                     tree_count=world['tree_count'], rotation_quarter_turns=rotation, seed=seed,
                     landcover=list(LANDCOVER), ground_occupied_fraction=float(occupied.mean()))
        path = folder/'forest_map.npz'
        np.savez_compressed(path,occupied=occupied,known=np.ones(occupied.shape,bool),
                            resolution_m=1.,origin_xy=[0.,0.],terrain=world['elevation'],
                            terrain_protocol='wildscenes_raster',source_unknown_space='blocked',
                            map_source='procedural_wooded_valley',coordinate_scale=1.)
        np.savez_compressed(folder/'landscape.npz', landcover=cover, canopy_height_m=world['canopy'])
        case = dict(case_id=f'{family}-{task}', kind='procedural_wooded_valley',
                    terrain='wildscenes_raster',source_unknown_space='blocked',seed=seed,
                    map_source='procedural_wooded_valley',map_path=path.name,
                    map_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),world_size=SIDE,
                    start=world['start'],goal=world['goal'],direct_distance_m=audit['direct_distance_m'],
                    reference_distance_m=audit['reference_distance_m'],coordinate_scale=1.,
                    resolution_m=1.,required_clearance_m=CLEARANCE,circles=[],
                    navigation_task=task,terrain_family_id=family,detour_offset_m=offset,
                    barrier_distance_m=world['barrier_distance'])
        write_json(folder/'audit.json', dict(audit,map_sha256=case['map_sha256']))
        write_json(folder/'case.json',case)
        rows.append(dict(case,map_path=path.relative_to(root).as_posix()))
    return rows


def generate_suite(root):
    seeds = [s for split in FAMILIES.values() for s in split]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Terrain families must have disjoint seeds')
    scenes = {}
    for split,seeds in FAMILIES.items():
        scenes[split] = []
        for index,seed in enumerate(seeds):
            rows = make_family(root,split,index,seed)
            scenes[split].extend(rows)
            print(f"{rows[0]['terrain_family_id']}: {len(rows)} audited routes",flush=True)
    maps = [row['map_sha256'] for split in scenes.values() for row in split]
    if len(set(maps)) != len(maps):
        raise ValueError('Duplicate map geometry')
    write_json(root/'scenes.json',scenes)
    write_json(root/'suite.json',dict(format='vtemp4_detour_curriculum_v1',seeds=FAMILIES,
               task_order=list(TASKS), class_labels=list(LANDCOVER),
               detour_offset='minimum absolute perpendicular distance in metres from a deck pixel centre to the start-goal line',
               barrier_distance='arclength to the first intersection of the start-goal segment with the watercourse, including the available deck',
               splits={key:len(cases) for key,cases in scenes.items()},
               static_hydrology='single passable deck over a map-spanning impassable watercourse',
               known_geometry='stored as environment truth; agents use local observations and received messages'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/vtemp4-detour-curriculum-v1')
    generate_suite(parser.parse_args().output)
