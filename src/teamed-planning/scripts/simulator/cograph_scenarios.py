"""CoGraph scenes: one 3-vehicle team, 2-8 targets around it in a 1000 m world.

Same obstacle density and reference-route checks as the single-target large
world. Reference routes are setup/evaluation truth only; they never enter a
policy or controller.

single_task (CoGraph v2 training): every target set is one task, i.e. the
drone network covers all targets at once from the team start
(ground.task_planner.drones_to_cover <= num_drones). Off, the scenes and their
random streams are exactly v1's.
"""
import math
import numpy as np
from scipy.ndimage import label
from simulator.large_world_scenarios import (ORIGINAL_ACTIVE_DENSITY, raster_free, reference_route,
    reference_geometry_clear, reference_polyline_length)
from ground.task_planner import drones_to_cover

# Vehicle team layout around the start (m); the nine-drone serpentine formation
# occupies up to 45 m beside it, so the start keeps START_MARGIN from the edge.
VEHICLE_OFFSETS = ((0., 0.), (0., 14.), (0., -14.))
START_MARGIN = 70.
TARGET_MARGIN = 35.
TARGET_SEPARATION = 40.


def sample_distance(rng, distance_range, sampling):
    if sampling == 'annulus_area':
        return float(np.sqrt(rng.uniform(distance_range[0]**2, distance_range[1]**2)))
    if sampling == 'linear':
        return float(rng.uniform(*distance_range))
    raise ValueError('Unknown task distance sampling mode')


def generate_cograph_scene(seed, target_count_range=(2, 8), distance_range=(100., 400.),
                           distance_sampling='annulus_area', vehicle_count=3, world=1000.,
                           density=ORIGINAL_ACTIVE_DENSITY, single_task=False, num_drones=9):
    """Targets at independent random distances/headings from the team start."""
    low, high = target_count_range
    if not (type(low) is int and type(high) is int and 1 <= low <= high <= 8):
        raise ValueError('target_count_range must be integers within 1..8')
    if not 0 < distance_range[0] < distance_range[1] < world/2:
        raise ValueError('Invalid target distance range')
    if not 1 <= vehicle_count <= len(VEHICLE_OFFSETS):
        raise ValueError(f'At most {len(VEHICLE_OFFSETS)} vehicles have a start layout')
    rng = np.random.default_rng([int(seed), 0xC06A])
    count = int(rng.integers(low, high+1))
    for attempt in range(1, 129):
        start = rng.uniform(START_MARGIN, world-START_MARGIN, 2)
        goals, distances = [], []
        for _ in range(count):
            for _ in range(200):
                distance = sample_distance(rng, distance_range, distance_sampling)
                heading = float(rng.uniform(-np.pi, np.pi))
                goal = start+distance*np.array([np.cos(heading), np.sin(heading)])
                if ((goal >= TARGET_MARGIN).all() and (goal <= world-TARGET_MARGIN).all()
                        and all(np.linalg.norm(goal-g) >= TARGET_SEPARATION for g in goals)
                        and (not single_task or drones_to_cover(start, goals+[goal]) <= num_drones)):
                    goals.append(goal); distances.append(distance)
                    break
            else:
                break
        if len(goals) < count:
            continue
        vehicles = [start+np.asarray(offset) for offset in VEHICLE_OFFSETS[:vehicle_count]]
        circles = []
        while len(circles) < round(density*world*world):
            p, r = rng.uniform(5., world-5., 2), float(rng.uniform(2., 5.))
            if (any(np.linalg.norm(p-v) <= r+10. for v in vehicles)
                    or any(np.linalg.norm(p-g) <= r+3.5 for g in goals)):
                continue
            circles.append((p, r))
        components, _ = label(raster_free(circles, world, 3.5))
        root = components[int(start[1]), int(start[0])]
        if not root or any(components[int(p[1]), int(p[0])] != root for p in [*vehicles, *goals]):
            continue
        reference = raster_free(circles, world, 1.5+math.sqrt(.5))
        try:
            routes = [reference_route(reference, start, goal) for goal in goals]
        except ValueError:
            continue
        if not all(reference_geometry_clear(circles, start, goal, route['points'], world)
                   for goal, route in zip(goals, routes)):
            continue
        scene = dict(seed=int(seed), world=world, start=start.tolist(),
            vehicle_starts=[v.tolist() for v in vehicles], goals=[g.tolist() for g in goals],
            distances=distances, target_count=count,
            circles=[(p.tolist(), r) for p, r in circles], attempts=attempt, density=density,
            reference_lengths=[route['length'] for route in routes],
            reference_polyline_lengths=[reference_polyline_length(start, goal, route['points'])
                                        for goal, route in zip(goals, routes)],
            reference_geometry_validated=True)
        if single_task:
            scene.update(single_task=True, task_drones=int(drones_to_cover(start, goals)))
        return scene
    raise RuntimeError(f'No feasible CoGraph scene for seed {seed}')
