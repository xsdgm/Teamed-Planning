"""Native kilometre missions with a single large obstacle and two open bypasses."""
import hashlib
import io
import math
from operator import index

import numpy as np


_WORLD_M = 1440.
_CLEARANCE_M = 1.5
_SPLITS = {'ppo_train': 0, 'development': 1}


def _side_distances(distance, station, offset, radius):
    """Shortest clearance-safe tangent–arc–tangent path on either side of the disk."""
    start_radius = math.hypot(station, offset)
    end_radius = math.hypot(distance - station, offset)
    straights = math.sqrt(start_radius**2 - radius**2) + math.sqrt(end_radius**2 - radius**2)
    # Angles are in the start-to-goal frame, with positive lateral distance to the left.
    start_angle = math.atan2(-offset, -station)
    end_angle = math.atan2(-offset, distance - station)
    tangent_angles = math.acos(radius / start_radius) + math.acos(radius / end_radius)
    return {name: straights + radius * ((sign * (start_angle - end_angle) - tangent_angles) % math.tau)
            for name, sign in (('left', 1), ('right', -1))}


def circle_schedule(split, count, seed):
    """Build deterministic, disjoint train/development plans for the native fleet runner."""
    count, seed = index(count), index(seed)
    if split not in _SPLITS or not 0 <= count <= 2**31 or seed < 0:
        raise ValueError('Expected a train/development split, nonnegative seed and valid scene count')
    split_id = _SPLITS[split]
    stream = np.random.default_rng([seed, split_id])
    first = 8 * int(stream.integers(2**28))
    plans = []
    for i in range(count):
        # Opposite parity keeps the two splits disjoint even for identical master seeds.
        scene_seed = 2 * ((first + i) % 2**31) + split_id
        rng = np.random.default_rng(scene_seed)
        distance = rng.uniform(1000., 1200.)
        yaw = rng.uniform(-math.pi, math.pi)
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        left = np.array([-forward[1], forward[0]])
        midpoint = _WORLD_M / 2 + rng.uniform(-24., 24., size=2)
        start, goal = midpoint - distance / 2 * forward, midpoint + distance / 2 * forward
        radius = rng.uniform(55., 95.)
        offset = rng.uniform(-.25, .25) * radius
        # Every eighth mission, including the first paired preflight case, starts
        # near the obstacle so the fleet can exercise early aerial reporting.
        entry = 26. if (scene_seed // 2) % 8 == 0 else rng.uniform(30., 110.)
        station = entry + math.sqrt(radius**2 - offset**2)
        center = start + station * forward + offset * left
        safe_radius = radius + _CLEARANCE_M
        # The clearance-inflated disk and both endpoints fit inside the eroded
        # rectangular world. By convexity both side-specific tangents fit there too.
        points = np.array([start, goal, center - safe_radius, center + safe_radius])
        if (points < _CLEARANCE_M).any() or (points > _WORLD_M - _CLEARANCE_M).any():
            raise ValueError('Circle or its clearance-safe bypasses exceed the mission boundary')
        lengths = _side_distances(distance, station, offset, safe_radius)
        case_id = f'vtemp5-circle-{split}-{scene_seed}'
        circle = np.array([[*center, radius]], dtype='<f8')
        circle_file = io.BytesIO()
        np.save(circle_file, circle, allow_pickle=False)
        mission = np.array([_WORLD_M, *start, *goal, *circle[0]], dtype='<f8')
        case = dict(case_id=case_id, terrain_family_id=case_id, seed=scene_seed,
                    kind='v9_random', terrain='synthetic', navigation_task='random_circles',
                    world_size=_WORLD_M, start=start.tolist(), goal=goal.tolist(),
                    circles=[[center.tolist(), float(radius)]], circle_count=1,
                    circles_sha256=hashlib.sha256(circle_file.getvalue()).hexdigest(),
                    map_sha256=hashlib.sha256(mission.tobytes()).hexdigest(),
                    direct_distance_m=float(distance), required_clearance_m=_CLEARANCE_M,
                    reference_distance_m=min(lengths.values()), side_reference_distance_m=lengths,
                    barrier_distance_m=float(entry))
        plans.append(dict(case=case, case_id=case_id, terrain='large_circle',
                          fleet_seed=int(rng.integers(1, 2**31))))
    return plans
