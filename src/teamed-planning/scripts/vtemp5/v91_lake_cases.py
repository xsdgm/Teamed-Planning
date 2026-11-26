"""Independent local 2D shore/bridge test geometry for the frozen V9.1 fleet.

Water is a union of native static collision rectangles. The dry bridge deck is
an opening in a bank-to-bank water band. The simulator reports occupancy only.
Only the evaluator reads the water/deck classes and true return geometry.
"""
import math

import numpy as np


WORLD_M = 1000.
LOCAL_HORIZON_M = 24.
NAV_CLEARANCE_M = 1.5  # the planner setting for both arms; physical body radius is 0.3 m
CARDINAL = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])


def make_lake_case(seed, kind):
    """Make a held-out closed lake or a river with a single offset bridge deck."""
    seed = int(seed)
    if kind not in ('shore', 'bridge'):
        raise ValueError('Choose shore or bridge')
    rng = np.random.default_rng(seed)
    start = rng.uniform(365., 635., size=2)
    forward = CARDINAL[(seed // 2) % 4]
    left = np.array([-forward[1], forward[0]])
    goal = start + rng.uniform(128., 153.) * forward
    near = rng.uniform(20., 33.)
    depth = rng.uniform(27., 45.)
    phases = rng.uniform(0., 2*np.pi, 2)
    rects = []  # native (centre XY, full width X, full height Y)

    def strip(lo, hi, front, rear):
        middle = start + (front + rear) * .5 * forward + (lo + hi) * .5 * left
        wh = np.abs(forward) * (rear - front) + np.abs(left) * (hi - lo)
        rects.append((middle.tolist(), float(wh[0]), float(wh[1])))

    details = {}
    if kind == 'shore':
        # Irregular closed shoreline with varying thickness, two tips and a
        # shorter bypass side that alternates throughout the held-out seeds.
        span = rng.uniform(26., 40.)
        bias = (1 if seed % 2 else -1) * rng.uniform(7., 14.)
        edges = np.linspace(-span + bias, span + bias, int(np.ceil(2*span/3.)) + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            u = ((lo + hi) * .5 - bias) / span
            half = depth * .5 * math.sqrt(1. - u*u)
            meander = 4.*math.sin(3*u + phases[0]) + 2.*math.sin(6*u + phases[1])
            center = near + depth * .5 + meander
            strip(lo, hi, center - half, center + half)
    else:
        # The blocked band spans the full world transversely. Far from the
        # approach use long rectangles. Nonzero open deck overlaps both banks.
        gate = (1 if seed % 2 else -1) * rng.uniform(17., 26.)
        width = rng.uniform(11., 16.)
        details = dict(bridge_front=(start + near*forward + gate*left).tolist(),
                       bridge_center=(start + (near+depth/2)*forward + gate*left).tolist(),
                       bridge_width_m=float(width), bridge_depth_m=float(depth))
        bounds = (np.array([[0., 0.], [0., WORLD_M], [WORLD_M, 0.], [WORLD_M, WORLD_M]]) - start) @ left
        for sign in (-1, 1):
            deck_edge = gate + sign * width / 2
            far = bounds.min() if sign < 0 else bounds.max()
            distances = [0., 8., *np.arange(15., 93., 7.), abs(far-deck_edge)]
            for a, b in zip(distances[:-1], distances[1:]):
                lo, hi = sorted([deck_edge+sign*a, deck_edge+sign*b])
                coord = sign * (a+b) * .5
                # Planar at the deck and at its first 8 m of wet shoulders.
                blend = np.clip((abs(coord) - 8.) / 22., 0., 1.)
                phi = 3.2 * math.sin(coord/22 + phases[0])
                taper = 3.7 * math.sin(coord/31 + phases[1])
                strip(lo, hi, near + blend*phi, near + depth + blend*(phi+taper))
    return dict(seed=seed, kind=kind, attempts=1, start=start.tolist(), goal=goal.tolist(),
                forward=forward.tolist(), left=left.tolist(), circles=[], rects=rects, **details)


def install_terrain(experiment, scene):
    """Install the same test geometry in the native observation and physics engine."""
    if experiment.env.time != 0 or experiment.env.obstacles:
        raise ValueError('Install the held-out terrain once before the first action')
    for center, width, height in scene['rects']:
        experiment.env.add_rect_obstacle(*center, width, height)
    experiment.obs = experiment.env._get_observations()
    # All scenes start well clear of water, including the car's local footprint.
    # Its already prepared first action is consequently also valid.


def _points(world_xy):
    pts = np.asarray(world_xy, dtype=float)
    if pts.ndim < 1 or pts.shape[-1] != 2:
        raise ValueError('Expected world coordinates with final dimension 2')
    return pts


def _rect_bounds(scene):
    centers = np.array([r[0] for r in scene['rects']], dtype=float)
    half = np.array([r[1:] for r in scene['rects']], dtype=float) / 2.
    return centers-half, centers+half


def water_mask(scene, world_xy):
    """Blocked water cells at the queried world-pixel centres."""
    pts = _points(world_xy)
    result = np.zeros(pts.shape[:-1], dtype=bool)
    for lo, hi in zip(*_rect_bounds(scene)):
        result |= ((pts[..., 0] >= lo[0]) & (pts[..., 0] <= hi[0]) &
                   (pts[..., 1] >= lo[1]) & (pts[..., 1] <= hi[1]))
    return result


def _deck_coords(scene, xy):
    delta = xy - scene['bridge_front']
    return delta @ scene['forward'], delta @ scene['left']


def bridge_mask(scene, world_xy):
    """Full true traversable bridge deck, from the entry through the exit."""
    pts = _points(world_xy)
    if scene['kind'] != 'bridge':
        return np.zeros(pts.shape[:-1], bool)
    along, across = _deck_coords(scene, pts)
    return (along >= 0) & (along <= scene['bridge_depth_m']) & (np.abs(across) < scene['bridge_width_m']/2)


def _mouth_coords(scene, pts, end):
    along, across = _deck_coords(scene, pts)
    if end == 'exit':
        along -= scene['bridge_depth_m']
    elif end != 'entry':
        raise ValueError('Choose entry or exit')
    return along, np.abs(across)


def bridge_mouth_mask(scene, world_xy, end='entry'):
    """Dry entry/exit swath extending 4 m to either side of the bank line."""
    pts = _points(world_xy)
    if scene['kind'] != 'bridge':
        return np.zeros(pts.shape[:-1], bool)
    along, across = _mouth_coords(scene, pts, end)
    return (np.abs(along) <= 4.) & (across < scene['bridge_width_m']/2.)


def bank_near_mouth_mask(scene, world_xy, end='entry'):
    """The actual wet shoulders neighbouring the same mouth cross section."""
    pts = _points(world_xy)
    if scene['kind'] != 'bridge':
        return np.zeros(pts.shape[:-1], bool)
    along, across = _mouth_coords(scene, pts, end)
    half = scene['bridge_width_m']/2.
    return (np.abs(along) <= 4.) & (across >= half) & (across <= half+8.) & water_mask(scene, pts)


def true_rejoin_safe(scene, car, goal, horizon_m=LOCAL_HORIZON_M, clearance_m=NAV_CLEARANCE_M):
    """Evaluation only: continuous swept local segment clears all wet rectangles.

    This tests the next horizon, truncated at the goal. Truth is available to
    the evaluator only; it must never supply actions or detector thresholds.
    """
    car, goal = _points(car), _points(goal)
    if horizon_m <= 0 or clearance_m <= 0:
        raise ValueError('Horizon and clearance must be positive')
    vec = goal-car
    length = np.linalg.norm(vec)
    delta = vec * min(1., horizon_m/max(length, 1e-12))
    end = car + delta
    if (np.minimum(car, end) < clearance_m).any() or (np.maximum(car,end) > WORLD_M-clearance_m).any():
        return False
    lo, hi = _rect_bounds(scene)
    # Exact segment/rectangle intersection via the two axis slabs.
    tlo, thi = np.zeros(len(lo)), np.ones(len(lo))
    possible = np.ones(len(lo), bool)
    for coordinate in (0, 1):
        if abs(delta[coordinate]) < 1e-12:
            possible &= (lo[:, coordinate] <= car[coordinate]) & (car[coordinate] <= hi[:, coordinate])
        else:
            a = (lo[:, coordinate] - car[coordinate]) / delta[coordinate]
            b = (hi[:, coordinate] - car[coordinate]) / delta[coordinate]
            tlo = np.maximum(tlo, np.minimum(a,b))
            thi = np.minimum(thi, np.maximum(a,b))
    if np.any(possible & (tlo <= thi)):
        return False
    # For disjoint segments/rectangles the closest pair has a segment endpoint
    # or a rectangle corner. Check both for the requested radial clearance.
    for point in (car, end):
        if np.any(np.linalg.norm(point-np.clip(point,lo,hi),axis=1) <= clearance_m):
            return False
    corners = np.concatenate((lo, hi, np.column_stack((lo[:,0], hi[:,1])),
                              np.column_stack((hi[:,0], lo[:,1]))))
    t = np.clip((corners-car) @ delta / max(delta @ delta, 1e-24), 0., 1.)
    return bool(np.min(np.linalg.norm(corners - (car+t[:,None]*delta), axis=1)) > clearance_m)


def crossed_bridge(scene, car_path):
    """Front-to-rear deck crossing by the actual recorded vehicle polyline."""
    if scene['kind'] != 'bridge' or len(car_path) < 2:
        return False
    along, across = _deck_coords(scene, _points(car_path))
    half, depth = scene['bridge_width_m']/2., scene['bridge_depth_m']
    entered = False
    for a, b, y, z in zip(along[:-1], along[1:], across[:-1], across[1:]):
        if b < 0:
            entered = False
            continue
        if a <= 0 < b:
            entry_y = y + (z-y) * (-a)/(b-a)
            entered = abs(entry_y) < half
        if entered and a < depth <= b:
            exit_y = y + (z-y) * (depth-a)/(b-a)
            if abs(exit_y) < half:
                return True
        if 0 < b < depth and abs(z) >= half:
            entered = False
    return False
