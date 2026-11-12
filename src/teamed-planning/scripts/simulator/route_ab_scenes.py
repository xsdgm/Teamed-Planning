"""V11 setup-only solvability certificates and distance-stratified sampling.

Truth and witness paths are admission diagnostics, never policy observations.
The certificate is deliberately sufficient, not necessary: it proves a safe
ground route and a bounded-speed connected aerial observation sweep with a
stationary vehicle. It does not certify arbitrary later vehicle trajectories
or that a learned policy will discover the witness.
"""
import math
import numpy as np

from drone.communication import relative_neighborhood
from simulator.large_world_scenarios import CircleGeometry, raster_free, reference_route

FLIGHT_HEIGHT = 15.
LATERAL_OFFSET = 18.
OBSERVATION_RADIUS = 210.
WITNESS_SPEED = 1.
MAX_ATTEMPTS = 512


def validate_distance_bands(config):
    bands = config.get('route_ab_distance_bands')
    if bands is None:
        return
    if not isinstance(bands, (list, tuple)) or not bands:
        raise ValueError('route_ab_distance_bands must be a nonempty list of [low, high] pairs')
    previous = None
    for band in bands:
        if (not isinstance(band, (list, tuple)) or len(band) != 2
                or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float))
                       or not np.isfinite(v) for v in band)
                or not 0 < band[0] < band[1] <= 200.
                or (previous is not None and band[0] < previous)):
            raise ValueError('A-B bands must be ordered, nonoverlapping finite distances within 200m')
        previous = band[1]


def selected_distance_band(config, seed):
    """Choose once from the original episode seed, before any rejection."""
    bands = config.get('route_ab_distance_bands')
    if bands is None:
        return None, None
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xAB11]))
    index = int(rng.integers(len(bands)))
    return index, tuple(bands[index])


def _in_band(distance, band, final=False):
    return band[0] <= distance and (distance < band[1] or (final and distance <= band[1]))


def _formation(c, radius, angle):
    along = np.array([math.cos(angle), math.sin(angle)])
    across = np.array([-along[1], along[0]])
    # IDs preserve the real initial order: y=-18, 0, +18. Drone 1 observes
    # the path; drones 0 and 2 relay. Every pair stays at least 18m apart.
    xy = np.array([c + 2.*radius/3.*along - LATERAL_OFFSET*across,
                   c + radius*along,
                   c + radius/3.*along + LATERAL_OFFSET*across])
    return np.c_[xy, np.full(3, FLIGHT_HEIGHT)]


def observation_sweep(c, path):
    """An explicit 1s-step flight witness from the actual initial formation.

    Rotation increments are bounded; the simulator executes straight segments
    between frames. The validator checks those segments and the exact CBF
    half-space inequalities, not just the sampled endpoint geometry.
    """
    c = np.asarray(c, float)
    initial = np.c_[c + np.array([[9., -18.], [9., 0.], [9., 18.]]), np.full(3, FLIGHT_HEIGHT)]
    frames = [initial]
    centered = _formation(c, 0., 0.)
    for i in range(1, 10):
        frames.append(initial + i/9.*(centered-initial))
    radius, angle = 0., 0.
    observation_frames = []
    for point in path:
        vector = np.asarray(point)-c
        following_radius = float(np.linalg.norm(vector))
        desired = math.atan2(vector[1], vector[0]) if following_radius > 1e-9 else angle
        turn = (desired-angle+math.pi) % (2*math.pi)-math.pi
        # An upper bound on each drone's displacement during this polar arc.
        travel = abs(following_radius-radius) + (max(radius, following_radius)+LATERAL_OFFSET)*abs(turn)
        count = max(1, math.ceil(travel/WITNESS_SPEED), math.ceil(abs(turn)/.05))
        for i in range(1, count+1):
            t = i/count
            frames.append(_formation(c, radius+t*(following_radius-radius), angle+t*turn))
        radius, angle = following_radius, angle+turn
        observation_frames.append(len(frames)-1)
    return np.asarray(frames), observation_frames


def validate_sweep(frames, c, world, ground_speed):
    """Check the current simulator's sampled-data CBF inequalities directly.

    This duplicates the *inequalities*, not the optimizer: a witness already
    satisfying them needs no projection. Regression tests compare the returned
    witness controls with CBFSafetyFilter.filter itself.
    """
    ground = np.r_[np.asarray(c, float), 0.]
    if not np.isfinite(frames).all() or np.any(frames[:, :, :2] < 0) or np.any(frames[:, :, :2] > world):
        raise ValueError('Connected observation sweep leaves world bounds')
    speeds = np.linalg.norm(np.diff(frames, axis=0), axis=2)
    if speeds.max(initial=0.) > WITNESS_SPEED+1e-8:
        raise ValueError('Connected observation sweep exceeds its speed bound')
    worst_separation, max_link, worst_residual = float('inf'), 0., -float('inf')
    names = ('drone_0', 'drone_1', 'drone_2')
    for points, following in zip(frames, frames[1:]):
        velocity = following-points
        for i in range(3):
            for j in range(i):
                r, d = points[i]-points[j], velocity[i]-velocity[j]
                t = float(np.clip(-r@d/max(d@d, 1e-12), 0., 1.))
                worst_separation = min(worst_separation, float(np.linalg.norm(r+t*d)))
        for i, point in enumerate(points):
            peers = {names[j]: q for j, q in enumerate(points) if i != j and np.linalg.norm(q-point) <= 95.}
            visible = dict(peers)
            if np.linalg.norm(ground-point) <= 95.:
                visible['__ground__'] = ground
            required = relative_neighborhood(point, visible, 95.)
            for other in peers.values():
                r = point-other
                horizontal = float(np.linalg.norm(r[:2]))
                # Same fixed-height lower barrier and symmetric split as CBF.
                residual = -(r[:2]/horizontal)@velocity[i, :2]-(horizontal-15.25)/2.
                worst_residual = max(worst_residual, float(residual))
            for key in required:
                r = point-visible[key]
                distance = float(np.linalg.norm(r))
                max_link = max(max_link, distance)
                h = 94.75**2-distance**2
                bound = ((h-(6.+ground_speed)**2)/2.-np.linalg.norm(r[:2])*ground_speed
                         if key == '__ground__' else (h-4.*6.**2)/4.)
                worst_residual = max(worst_residual, float(r[:2]@velocity[i, :2]-bound))
    if worst_separation < 15.25-1e-8 or worst_residual > 1e-7:
        raise ValueError('Connected observation sweep violates a sampled-data CBF constraint')
    # For the radial relay construction, required links stay well inside radio
    # range. Check this independently so a topology/formation change fails closed.
    if max_link > 80.:
        raise ValueError('Connected observation sweep lost its radio reserve')
    return dict(min_continuous_uav_separation_m=worst_separation,
                max_required_link_m=max_link, max_cbf_constraint_residual=worst_residual,
                max_uav_speed_m_s=float(speeds.max(initial=0.)))


def certify_scene(config, scene, count=3):
    """Reject unless a safe ground path and connected observation sweep exist."""
    if count != 3:
        raise ValueError('V11 solvability witness currently supports exactly three UAVs')
    task = scene['survey_task']
    a, b, c = (np.asarray(p, float) for p in (task.a, task.b, scene['start']))
    world = float(config['world_size'])
    # Two-meter occupancy cells and endpoint strips require more room than a
    # point-sized car path. Reserve a full pixel diagonal beyond car clearance.
    clearance = float(config['ground_safety_distance']) + 2.*math.sqrt(2.)
    geometry = CircleGeometry(scene['circles'], world=world, clearance=clearance)
    if not geometry.clear(a, a) or not geometry.clear(b, b):
        raise ValueError('A-B endpoint lacks obstacle clearance for safe mapping')
    free = raster_free(scene['circles'], world=world, clearance=clearance+math.sqrt(.5))
    coordinates = np.arange(free.shape[0])+.5
    # All relay positions are convex combinations of C and the observer, plus
    # at most 18m perpendicular offset. This rectangle keeps them in the world.
    in_bounds = (coordinates >= LATERAL_OFFSET) & (coordinates <= world-LATERAL_OFFSET)
    free &= in_bounds[:, None] & in_bounds[None, :]
    free &= ((coordinates[None, :]-c[0])**2+(coordinates[:, None]-c[1])**2 <= OBSERVATION_RADIUS**2)
    if max(np.linalg.norm(a-c), np.linalg.norm(b-c)) > OBSERVATION_RADIUS:
        raise ValueError('A-B endpoints lie outside the connected observation construction')
    reference = reference_route(free, a, b)
    path = np.array([a, *reference['points'], b])
    if not all(geometry.clear(p, q) for p, q in zip(path, path[1:])):
        raise ValueError('A-B raster route fails continuous obstacle clearance')
    frames, observed_at = observation_sweep(c, path)
    facts = validate_sweep(frames, c, world, float(config['ground_speed']))
    length = float(np.linalg.norm(b-a))
    budget = min(config['steps'], max(config['min_episode_steps'],
        math.ceil(config['budget_multiplier']*length+config['budget_extra'])))
    # Evidence propagation takes one step per hop, fusion every five steps.
    # Reserve nominal delivery time and the configured stable-success window.
    # This is not a guarantee for every stochastic packet-loss realization.
    required_steps = len(frames)-1+20+config['map_stability_steps']
    if required_steps > budget:
        raise ValueError(f'Connected mapping witness needs {required_steps} steps, budget is {budget}')
    return dict(contract='stationary_three_uav_connected_sweep_v1',
        assumes_stationary_vehicle=True, moving_vehicle_guaranteed=False,
        guarantees_learned_policy_success=False, guarantees_all_channel_realizations=False,
        delivery_assumption='nominal timely delivery; arbitrary stochastic losses are not certified',
        ground_clearance_m=clearance, ground_path=path.tolist(),
        ground_path_length_m=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        observation_frame_indices=observed_at, flight_steps=len(frames)-1,
        delivery_margin_steps=20, stability_steps=config['map_stability_steps'],
        required_steps=required_steps, episode_budget_steps=budget, **facts)


def sample_certified_scene(config, seed, sampler, legacy_radio_test):
    validate_distance_bands(config)
    bands = config.get('route_ab_distance_bands')
    index, band = selected_distance_band(config, seed)
    explicit = config['survey_a'] is not None
    count = config['fleet_sizes'][seed % len(config['fleet_sizes'])]
    if count != 3:
        raise ValueError('V11 solvability witness currently supports exactly three UAVs')
    candidate_config = config
    if band is not None and not explicit and config['task_distance_range'] is not None:
        # Triangle inequality gives a cheap C-B proposal interval. This only
        # avoids generating obviously wrong-band worlds; final acceptance is
        # always based on the true A-B distance and the original chosen band.
        offset = config['survey_start_offset_range'][1]
        low = max(config['task_distance_range'][0], band[0]-offset)
        high = min(config['task_distance_range'][1], band[1]+offset)
        if low >= high:
            raise ValueError('Selected A-B band cannot fit the C-B proposal range')
        candidate_config = dict(config, task_distance_range=[low, high])
    last_reason = 'no candidate'
    for attempt in range(MAX_ATTEMPTS):
        candidate_seed = seed if attempt == 0 else int(np.random.SeedSequence([seed, 11, attempt]).generate_state(1)[0])
        try:
            scene = sampler(candidate_config, candidate_seed)
            task = scene['survey_task']
            length = float(np.linalg.norm(np.asarray(task.b)-task.a))
            if bands is not None:
                supported = (any(_in_band(length, interval, k == len(bands)-1) for k, interval in enumerate(bands))
                             if explicit else _in_band(length, band, index == len(bands)-1))
                if not supported:
                    raise ValueError('Actual A-B distance is outside the selected distance band')
            elif not legacy_radio_test(task.a, task.b, scene['start'], count):
                raise ValueError('Legacy a-b-c layout exceeds the conservative relay construction')
            certificate = certify_scene(config, scene, count)
        except ValueError as exc:
            last_reason = str(exc)
            if explicit:
                raise
            continue
        except RuntimeError as exc:
            # The upstream generator uses this one explicit exhaustion error;
            # retry a new deterministic world without masking programming bugs.
            if not str(exc).startswith('No feasible large scene for seed '):
                raise
            if explicit:
                raise ValueError(str(exc)) from exc
            last_reason = str(exc)
            continue
        scene.update(route_ab_feasibility=certificate, route_ab_distance_m=length,
            route_ab_distance_band_index=index if not explicit else None,
            route_ab_distance_band=list(band) if band is not None and not explicit else None,
            route_ab_scene_seed=int(candidate_seed), route_ab_sampling_attempts=attempt+1)
        return scene
    raise ValueError(f'Cannot certify A-B scene for fixed band {band} after {MAX_ATTEMPTS} attempts: {last_reason}')
