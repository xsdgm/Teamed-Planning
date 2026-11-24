"""Create a metre-scale synthetic wooded river valley for the vtemp4 raster simulator."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as nd
from scipy.interpolate import PchipInterpolator
from skimage.graph import MCP_Geometric


SIDE = 1440
START = [140.5, 650.5]
GOAL = [1270.5, 675.5]
BRIDGE_Y = 930.5
CLEARANCE = 1.5
DEFAULT_SEED = 19723
ROOT = Path(__file__).resolve().parents[4]
LANDCOVER = ('meadow', 'open_woodland', 'closed_woodland', 'dense_scrub',
             'river', 'dirt_track', 'timber_bridge', 'tree_trunk', 'rock')


def river_x(y):
    return 716 + 84*np.sin(y/167 - .4) + 45*np.sin(y/83 + .6) + 11*np.sin(y/32 - .7)


def field_noise(rng, size):
    """Interpolate a smooth random field with a specified spatial correlation scale."""
    knots = max(4, int(np.ceil(SIDE / size)))
    coarse = rng.normal(size=(knots, knots)).astype(np.float32)
    return nd.zoom(coarse, SIDE/knots, order=3, mode='reflect', grid_mode=True)


def track_distance(lines):
    skeleton = np.zeros((SIDE, SIDE), bool)
    for line in lines:
        points = np.asarray(line, float)
        t = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        curve = PchipInterpolator(t, points, axis=0)(np.linspace(0, t[-1], int(t[-1]*4)+1))
        indices = np.floor(curve).astype(int).clip(0, SIDE-1)
        skeleton[indices[:, 1], indices[:, 0]] = True
    return nd.distance_transform_edt(~skeleton)


def plant_trees(rng, cover, trail_buffer, forest_score):
    """Place individual trunks and a separate overlapping crown-height layer."""
    density = np.interp(forest_score, [-2, -.45, .1, .7, 2], [.0005, .002, .009, .027, .042])
    sites = (rng.random(cover.shape) < density) & np.isin(cover, [0, 1, 2, 3]) & ~trail_buffer
    canopy = np.zeros(cover.shape, np.float32)
    for r, c in np.argwhere(sites):
        h = rng.uniform(6, 20) if cover[r, c] != 3 else rng.uniform(4, 11)
        radius = rng.uniform(.19, .34)*h
        cells = int(np.ceil(radius))
        y0, y1 = max(0,r-cells), min(SIDE,r+cells+1)
        x0, x1 = max(0,c-cells), min(SIDE,c+cells+1)
        yy, xx = np.ogrid[y0-r:y1-r, x0-c:x1-c]
        distance_sq = yy*yy + xx*xx
        crown = h*np.sqrt(np.maximum(0., 1.-distance_sq/radius**2))
        np.maximum(canopy[y0:y1,x0:x1], crown, out=canopy[y0:y1,x0:x1])
        cover[y0:y1,x0:x1][distance_sq <= (h/18)**2] = 7
    return np.round(canopy, 1), int(sites.sum())


def landscape(seed, *, channel=river_x, lines=None, endpoints=(START, GOAL), example_bridge=True):
    rng = np.random.default_rng(seed)
    y = np.arange(SIDE, dtype=np.float32)[:, None] + .5
    x = y.T
    bank = np.abs(x-channel(y))
    half_width = 17 + 5*np.sin(y/118+.4) + 3*np.sin(y/41-1.2)
    water = bank <= half_width
    valley_side = 1.-np.exp(-(bank/300.)**1.5)
    upland = field_noise(rng, 210)
    elevation = 204 + .008*y + 4*np.sin(y/560.) + valley_side*(24 + 12*upland + 3*field_noise(rng, 62))
    elevation = np.round(elevation, 2).astype(np.float32)

    forest_score = (.73*field_noise(rng, 175) + .37*field_noise(rng, 64)
                    + .18*field_noise(rng, 20) + .22*valley_side)
    cover = np.zeros((SIDE, SIDE), np.uint8)
    cover[forest_score > -.31] = 1
    cover[forest_score > .48] = 2
    understory = field_noise(rng, 22) + .4*field_noise(rng, 8)
    cover[(cover == 2) & (understory > 1.15)] = 3
    cover[(bank < half_width+24) & (understory > -.48)] = 3
    outcrops = .8*field_noise(rng, 31) + .42*field_noise(rng, 8)
    cover[(bank > 100) & (outcrops + .36*upland > 1.50)] = 8
    cover[water] = 4

    if lines is None:
        lines = example_tracks()
    tracks = track_distance(lines)
    cover[(tracks <= 4.5) & ~water] = 5
    if example_bridge:
        cover[water & (np.abs(y-BRIDGE_Y) <= 6.)] = 6
    buffer = (tracks < 7.5) | water
    # Leave room for the native fleet's spawn pattern at the trailheads.
    for point in endpoints:
        pad = np.hypot(x-point[0], y-point[1]) <= 25
        cover[pad & (cover != 5)] = 0
        buffer |= pad
    canopy, n_trees = plant_trees(rng, cover, buffer, forest_score)
    return elevation, cover, canopy, n_trees


def example_tracks():
    b = float(river_x(BRIDGE_Y))
    # Tracks on the two terraces meet the single deck; side spurs approach
    # the river without granting passage through water.
    return [
        [(1,585), (66,607), START, (240,685), (330,719), (381,801), (453,842),
         (495,877), (b-85,913), (b-42,BRIDGE_Y), (b+42,BRIDGE_Y), (b+87,925),
         (883,889), (974,831), (1040,731), (1144,699), GOAL, (1380,678), (1438,704)],
        [(240,685), (290,633), (388,603), (480,613), (560,618), (river_x(625)-39,625)],
        [(1040,731), (997,679), (915,664), (857,631), (river_x(620)+43,620)],
        [(974,831), (1055,886), (1144,939), (1178,1007), (1190,1115)],
        [(381,801), (354,881), (312,923), (225,953), (144,1010)],
    ]


def audit_geometry(occupied, cover, *, start=START, goal=GOAL, bridge_center=None, direct_route=False):
    """Certify a swept 1.5 m clearance along an 8-connected cell-centre path.

    Subtract the occupied cell's half diagonal and the maximum distance from
    any point along a step to its nearest endpoint. The padded map exterior is blocked.
    """
    start_rc, goal_rc = [tuple(np.floor(point[::-1]).astype(int)) for point in (start, goal)]
    clearance_bound = nd.distance_transform_edt(np.pad(~occupied, 1))[1:-1,1:-1] - np.sqrt(2.)
    safe = clearance_bound >= CLEARANCE
    planner = MCP_Geometric(np.where(safe, 1., np.inf))
    distance, _ = planner.find_costs(starts=[start_rc], ends=[goal_rc])
    length = float(distance[goal_rc])
    if not np.isfinite(length):
        raise ValueError('No clearance-certified route between endpoints')
    route = np.asarray(planner.traceback(goal_rc), dtype=int)
    route_y, route_x = route.T
    if not np.any(cover[route_y, route_x] == 6):
        raise ValueError('Reference route missed the crossing')
    # Close only deck pixels. Using raw free space with 8-neighbour connectivity is
    # conservative for proving there is no alternative ground crossing.
    components = nd.label(~occupied & (cover != 6), structure=np.ones((3,3)))[0]
    if components[start_rc] == components[goal_rc]:
        raise ValueError('Closing the bridge still allows a ground crossing')
    direct = np.floor(np.linspace(start, goal, 6000)).astype(int)
    hits_water = bool((cover[direct[:,1], direct[:,0]] == 4).any())
    if direct_route and (hits_water or not safe[direct[:,1], direct[:,0]].all()):
        raise ValueError('Direct route lacks the declared clearance')
    if not direct_route and not hits_water:
        raise ValueError('Direct route does not encounter the river')
    step = np.diff(route, axis=0)
    corners = np.r_[True, np.any(step[1:] != step[:-1], axis=1), True]
    return dict(start=start, goal=goal, bridge_center_xy=bridge_center or [float(river_x(BRIDGE_Y)), BRIDGE_Y],
                direct_distance_m=float(np.linalg.norm(np.subtract(goal, start))),
                reference_distance_m=length, required_clearance_m=CLEARANCE,
                certified_clearance_lower_bound_m=float(clearance_bound[route_y, route_x].min()),
                direct_line_hits_water=hits_water, bridge_closure_disconnects_endpoints=True,
                reference_method='shortest 8-neighbour route on conservative swept-clearance mask',
                reference_route_xy=(route[corners,::-1]+.5).tolist())


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def generate(output, seed):
    elevation, cover, canopy, n_trees = landscape(seed)
    occupied = np.isin(cover, [3, 4, 7, 8])
    audit = audit_geometry(occupied, cover)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'forest_map.npz'
    np.savez_compressed(path, occupied=occupied, known=np.ones(occupied.shape, bool),
                        resolution_m=1., origin_xy=[0.,0.], terrain=elevation,
                        terrain_protocol='wildscenes_raster', source_unknown_space='blocked',
                        map_source='procedural_wooded_valley', coordinate_scale=1.)
    np.savez_compressed(output/'landscape.npz', landcover=cover, canopy_height_m=canopy)
    case = dict(case_id='field-valley-crossing-v1', kind='procedural_wooded_valley',
                terrain='wildscenes_raster', source_unknown_space='blocked', seed=seed,
                map_source='procedural_wooded_valley', map_path=path.name,
                map_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), world_size=SIDE,
                start=START, goal=GOAL, direct_distance_m=audit['direct_distance_m'],
                reference_distance_m=audit['reference_distance_m'], coordinate_scale=1.,
                resolution_m=1., required_clearance_m=CLEARANCE, circles=[])
    write_json(output/'case.json', case)
    write_json(output/'audit.json', dict(audit, seed=seed, world_size=SIDE, resolution_m=1.,
               map_sha256=case['map_sha256'], tree_count=n_trees, landcover=list(LANDCOVER),
               blocked_classes=['dense_scrub','river','tree_trunk','rock'],
               ground_occupied_fraction=float(occupied.mean())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/vtemp4-field-valley-v1')
    args = parser.parse_args()
    generate(args.output, args.seed)
