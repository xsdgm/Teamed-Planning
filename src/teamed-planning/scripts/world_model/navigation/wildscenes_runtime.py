"""Local-only WildScenes perception over the frozen v9 control runtime.

Truth is confined to simulator sensor generation and swept collision checks.
Signed local pixels retain unknown support through the original actor, graph
world model and delivered-evidence packets. Planning treats unknown as blocked.
The separately named simplified protocol retains the legacy simulator's
free-space assumption for unmeasured source regions; it never pre-populates
the robots' online observations or delivered-evidence ledger.
"""
from __future__ import annotations

import contextlib
import importlib.util
import math
from pathlib import Path
from types import MethodType

import numpy as np

from .navigation_evidence import EvidenceGrid

from .hierarchical_sources import PACKAGE_DIR, file_sha256 as _digest

_BACKENDS = {}
_FOREST_CACHE = None
RASTER_PROTOCOLS = {
    'wildscenes_raster': 'blocked',
    'wildscenes_simplified_raster': 'free_by_simulation_assumption',
}


def is_raster_case(case):
    """An explicit terrain contract; typos cannot become an empty simulator."""
    terrain = case.get('terrain', 'synthetic')
    if terrain == 'synthetic':
        return False
    if terrain not in RASTER_PROTOCOLS:
        raise ValueError(f'Unsupported terrain protocol: {terrain!r}')
    if case.get('source_unknown_space') != RASTER_PROTOCOLS[terrain]:
        raise ValueError(f'{terrain} requires source_unknown_space={RASTER_PROTOCOLS[terrain]!r}')
    return True


def load_map_backend(repo):
    """Load the snapshotted standalone adapter without importing simulator."""
    path = PACKAGE_DIR / '_wildscenes_map_backend.py'
    if not path.is_file():
        if PACKAGE_DIR.parent.name == 'runtime_sources':
            raise ValueError('Archived WildScenes run is missing its frozen raster backend')
        path = Path(repo).resolve() / 'src/teamed-planning/scripts/simulator/wildscenes_map.py'
    key = str(path.resolve())
    if key not in _BACKENDS:
        spec = importlib.util.spec_from_file_location('_vtemp2_wildscenes_backend', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BACKENDS[key] = module
    return _BACKENDS[key]


def load_forest(case, repo):
    """Check immutable geometry and metric endpoints before simulator setup."""
    global _FOREST_CACHE
    path = Path(case['map_path'])
    if not path.is_absolute() or not path.is_file():
        raise ValueError('WildScenes map_path must name an existing absolute NPZ path')
    if not is_raster_case(case):
        raise ValueError('WildScenes raster loading requires an explicit raster terrain protocol')
    clearance = float(case['required_clearance_m'])
    if not math.isfinite(clearance) or clearance != 1.5:
        raise ValueError('WildScenes requires the declared 1.5m ground clearance')
    digest = _digest(path)
    if digest != case['map_sha256']:
        raise ValueError('WildScenes map SHA256 changed')
    key = (str(path.resolve()), digest)
    if _FOREST_CACHE is None or _FOREST_CACHE[0] != key:
        _FOREST_CACHE = (key, load_map_backend(repo).RasterForestMap.from_npz(path))
    forest = _FOREST_CACHE[1]
    metadata = forest.metadata
    if (metadata.get('source_unknown_space', case['source_unknown_space']) != case['source_unknown_space']
            or metadata.get('terrain_protocol', case['terrain']) != case['terrain']):
        raise ValueError('WildScenes NPZ protocol differs from the declared case')
    if float(case.get('coordinate_scale', 1.)) != 1. or float(metadata.get('coordinate_scale', 1.)) != 1.:
        raise ValueError('WildScenes coordinates must retain the original scale 1.0')
    if case['terrain'] == 'wildscenes_simplified_raster':
        required = ('terrain_protocol', 'source_unknown_space', 'coordinate_scale',
                    'source_scene_path', 'source_scene_sha256')
        if any(name not in metadata for name in required):
            raise ValueError('Simplified WildScenes NPZ lacks explicit source/protocol provenance')
        if any(name not in case for name in ('coordinate_scale', 'source_scene_path', 'source_scene_sha256')):
            raise ValueError('Simplified WildScenes case lacks original scene provenance')
        if not bool(forest.known.all()):
            raise ValueError('Simplified WildScenes simulator support must be explicitly known everywhere')
        source = Path(case['source_scene_path'])
        if (not source.is_absolute() or not source.is_file()
                or metadata['source_scene_path'] != str(source)
                or metadata['source_scene_sha256'] != case['source_scene_sha256']
                or _digest(source) != case['source_scene_sha256']):
            raise ValueError('Simplified WildScenes original scene SHA256/path changed')
    if not math.isclose(forest.resolution_m, float(case['resolution_m']), rel_tol=0., abs_tol=1e-12):
        raise ValueError('WildScenes source resolution must remain at its original scale')
    world = float(case['world_size'])
    if (not math.isfinite(world) or (forest.origin_xy < 0).any()
            or (forest.upper_xy > world + 1e-9).any()):
        raise ValueError('WildScenes raster bounds must fit the declared world without rescaling')
    for name in ('start', 'goal'):
        point = np.asarray(case[name], float)
        if point.shape != (2,) or not forest.point_clear(point, clearance):
            raise ValueError(f'WildScenes {name} lacks declared raster free clearance')
    if case.get('circles'):
        raise ValueError('WildScenes raster cases must not add synthetic circle geometry')
    return forest


def pool_received_tile(occupancy, position, resolution, target_resolution, target_shape,
                       target_origin=(0., 0.), *, occupied_overrides_unknown=False):
    """Pool receiver-local rectangles. Free space requires complete known coverage.

    Pixel-centre voting is insufficient for a moving sensor: a shifted source
    pixel may touch a receiver cell without observing its remaining area.
    """
    pixels = np.asarray(occupancy)
    if pixels.ndim != 2:
        raise ValueError('Received raster evidence must be a two-dimensional local tile')
    h, w = pixels.shape
    target_origin = np.asarray(target_origin, float)
    if target_origin.shape != (2,) or not np.isfinite(target_origin).all():
        raise ValueError('Target evidence origin must be a finite XY pair')
    lower = np.asarray(position, float)[:2] - np.asarray([w, h]) * resolution / 2
    upper = lower + np.asarray([w, h]) * resolution
    start = np.maximum(0, np.floor((lower - target_origin) / target_resolution).astype(int))
    stop = np.minimum(np.asarray(target_shape)[::-1],
                      np.ceil((upper - target_origin) / target_resolution).astype(int))
    if np.any(stop <= start):
        return np.empty(0, np.int64), np.empty(0, np.int8)
    ys, xs = np.mgrid[start[1]:stop[1], start[0]:stop[0]]
    lo = target_origin + np.stack([xs, ys], -1) * target_resolution
    hi = lo + target_resolution
    full = (lo >= lower - 1e-9).all(-1) & (hi <= upper + 1e-9).all(-1)
    first = np.maximum(0, np.floor((lo - lower) / resolution + 1e-10).astype(int))
    last = np.minimum([w, h], np.ceil((hi - lower) / resolution - 1e-10).astype(int))
    known = np.isfinite(pixels) & (pixels >= 0)

    def any_intersection(mask):
        integral = np.pad(np.asarray(mask, np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        x0, y0, x1, y1 = first[..., 0], first[..., 1], last[..., 0], last[..., 1]
        return (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) > 0

    usable = full & ~any_intersection(~known)
    occupied = any_intersection(known & (pixels > .5))
    if occupied_overrides_unknown:
        usable |= occupied
    values = np.where(usable, np.where(occupied, 1, -1), 0)
    return (ys * target_shape[1] + xs).ravel(), values.astype(np.int8).ravel()


def world_sensor_tile(observation):
    """Return ego-centred 2 m occupancy and observed masks for the 12×12 world input.

    A known obstacle in any overlapping source cell establishes occupancy.
    A clear coarse cell requires its entire area to be observed and clear.
    """
    shape = (12, 12)
    resolution = float(observation['map_resolution'])
    pixels = np.asarray(observation['occupancy_map'])
    if pixels.ndim != 2 or not pixels.size or not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('World-model occupancy requires a nonempty local tile and positive resolution')
    if not observation.get('sensor_valid', True):
        return np.zeros(shape, np.float32), np.zeros(shape, bool)
    known = np.isfinite(pixels) & (pixels >= 0)
    if 'occupancy_known_mask' in observation:
        support = np.asarray(observation['occupancy_known_mask'], bool)
        if support.shape != pixels.shape:
            raise ValueError('Local occupancy and known mask shapes differ')
        known &= support
    if pixels.shape == shape and resolution == 2.:
        return (known & (pixels > .5)).astype(np.float32), known
    # Both grids share the sensor centre. Ego-relative coordinates avoid
    # rounding at world XY boundaries and keep the target footprint fixed.
    signed = np.where(known, (pixels > .5).astype(np.int8), np.int8(-1))
    ids, values = pool_received_tile(signed, (0., 0.), resolution,
        2., shape, target_origin=(-12., -12.), occupied_overrides_unknown=True)
    pooled = np.zeros(shape, np.int8)
    pooled.flat[ids] = values
    return (pooled == 1).astype(np.float32), pooled != 0


def receive_known_evidence(ledger, source, frame, position, occupancy, resolution, valid=True):
    """The archived packet interface with partial/unknown cells excluded."""
    key = (source, frame)
    if not valid or occupancy is None or key in ledger.received:
        return False
    ledger.received.add(key)
    ids, values = pool_received_tile(occupancy, position, resolution, ledger.resolution, ledger.seen.shape)
    use = values != 0
    target = ids[use]
    weights = values[use] * np.log(4.)
    ledger.log_odds.flat[target] += weights
    np.clip(ledger.log_odds, -3., 3., out=ledger.log_odds)
    ledger.seen.flat[target] = True
    return True


def local_known_belief(belief, observation, ledger):
    """Correct local aggregation without consulting any source raster truth."""
    cells = np.zeros(ledger.seen.shape, np.int8)
    cells[ledger.seen & (ledger.log_odds < 0)] = -1
    cells[ledger.seen & (ledger.log_odds > 0)] = 1
    ids, values = pool_received_tile(observation['occupancy_map'], observation['self']['position'],
        float(observation['map_resolution']), ledger.resolution, cells.shape)
    use = values != 0
    cells.flat[ids[use]] = values[use]
    result = dict(belief)
    result['grid'] = EvidenceGrid(cells, ledger.resolution)
    return result


def _known_semantics(processor):
    original = processor.process_occupancy_map

    def process(pixels):
        pixels = np.asarray(pixels)
        known = np.isfinite(pixels) & (pixels >= 0)
        result = original(np.where(known, pixels, 1))
        # Both archived planners use enum 0 for UNKNOWN. Keep that semantic
        # identity so the ground obstacle memory cannot invent observations.
        result[~known] = 0
        return result

    processor.process_occupancy_map = process
    processor.terrain_costs[0] = float('inf')


def local_safety_observation(forest, position, map_size=16, resolution_m=.5):
    """Generate native-aligned geometry within the simulator sensor footprint.

    Re-pooling obstacles onto a grid that moves with the robot can turn a
    certified endpoint into an occupied safety margin at the next frame.
    The safety controller instead receives original cell boundaries, but only
    for fully observed cells. Partial boundary cells and source-map exterior
    are unknown; no source array, global metadata or truth query escapes.
    """
    sizes = np.asarray(map_size, float)
    if sizes.ndim == 0:
        sizes = np.repeat(sizes, 2)
    pose = np.asarray(position, float)
    resolution_m = float(resolution_m)
    if (sizes.shape != (2,) or not np.isfinite(sizes).all() or (sizes <= 0).any()
            or (sizes != np.floor(sizes)).any() or pose.ndim != 1 or pose.size < 2
            or not np.isfinite(pose[:2]).all() or not math.isfinite(resolution_m)
            or resolution_m <= 0):
        raise ValueError('Invalid local safety sensor geometry')
    center = pose[:2]
    lower, upper = center - sizes[::-1] * resolution_m / 2, center + sizes[::-1] * resolution_m / 2
    first = np.floor((lower - forest.origin_xy) / forest.resolution_m + 1e-10).astype(int)
    stop = np.ceil((upper - forest.origin_xy) / forest.resolution_m - 1e-10).astype(int)
    ys, xs = np.mgrid[first[1]:stop[1], first[0]:stop[0]]
    cell_lower = forest.origin_xy + np.stack([xs, ys], -1) * forest.resolution_m
    full = ((cell_lower >= lower - 1e-10).all(-1)
            & (cell_lower + forest.resolution_m <= upper + 1e-10).all(-1))
    inside = (xs >= 0) & (ys >= 0) & (xs < forest.shape[1]) & (ys < forest.shape[0])
    usable = full & inside
    occupied, known = np.ones(xs.shape, np.uint8), np.zeros(xs.shape, bool)
    occupied[usable] = forest.occupied[ys[usable], xs[usable]]
    known[usable] = forest.known[ys[usable], xs[usable]]
    # Unknown occupancy stays opaque, including partial sensor-border cells.
    occupied[~known] = 1
    return dict(occupancy_map=occupied, known_mask=known,
                resolution_m=forest.resolution_m,
                origin_xy=(forest.origin_xy + first * forest.resolution_m).tolist(),
                map_axes='yx')


def _ground_local_geometry(observation, backend):
    """Build safety geometry from local sensor data only, never map truth."""
    position = np.asarray(observation['self']['position'], float)[:2]
    sensor_resolution = float(observation['map_resolution'])
    sensor_pixels = np.asarray(observation['occupancy_map'])
    axes = observation.get('map_axes', 'yx')
    if axes == 'xy':
        sensor_pixels = sensor_pixels.T
    elif axes != 'yx':
        raise ValueError('Unsupported local ground sensor axes')
    if (sensor_pixels.ndim != 2 or not sensor_pixels.size
            or not math.isfinite(sensor_resolution) or sensor_resolution <= 0):
        raise ValueError('Invalid local ground sensor geometry')
    span = np.asarray(sensor_pixels.shape[::-1]) * sensor_resolution
    sensor_lower, sensor_upper = position - span / 2, position + span / 2
    geometry = observation.get('ground_safety_map')
    if geometry is None:
        pixels = sensor_pixels
        known = np.isfinite(pixels) & (pixels >= 0)
        if 'occupancy_known_mask' in observation:
            mask = np.asarray(observation['occupancy_known_mask'], bool)
            known &= mask.T if axes == 'xy' else mask
        resolution, origin = sensor_resolution, sensor_lower
    else:
        if geometry.get('map_axes') != 'yx':
            raise ValueError('Native local safety geometry must use yx axes')
        pixels = np.asarray(geometry['occupancy_map'])
        known = np.asarray(geometry['known_mask'], bool).copy()
        resolution = float(geometry['resolution_m'])
        origin = np.asarray(geometry['origin_xy'], float)
        if (pixels.ndim != 2 or not pixels.size or known.shape != pixels.shape
                or not math.isfinite(resolution) or resolution <= 0
                or origin.shape != (2,) or not np.isfinite(origin).all()):
            raise ValueError('Invalid native local safety geometry')
        known &= np.isfinite(pixels) & (pixels >= 0)
        ys, xs = np.indices(pixels.shape)
        cell_lower = origin + np.stack([xs, ys], -1) * resolution
        # Enforce locality at the consumer too. Even an oversized supplied
        # tile cannot authorize travel outside this frame's sensor footprint.
        known &= ((cell_lower >= sensor_lower - 1e-10).all(-1)
                  & (cell_lower + resolution <= sensor_upper + 1e-10).all(-1))
    known &= bool(observation.get('sensor_valid', True))
    return backend.RasterForestMap(pixels > .5, known, resolution, origin)


def filter_ground_motion(action, observation, backend, *, clearance=1.5, max_speed=1.):
    """Limit a diff-drive translation using only its current local evidence.

    Frozen physics translates along the pre-step heading, then integrates the
    turn. Accordingly omega is preserved and this filter checks a straight
    swept capsule for the actual dt. Unknown/outside the sensor cannot certify
    a motion. No map path, forest instance or reference route is an argument.
    """
    motion = np.asarray(action['motion'], float)
    position = np.asarray(observation['self']['position'], float)[:2]
    heading = float(observation['self']['orientation'])
    dt = float(observation['time_step'])
    if (motion.shape != (2,) or not np.isfinite(motion).all() or not np.isfinite(heading)
            or not math.isfinite(dt) or dt <= 0 or not math.isfinite(max_speed) or max_speed <= 0):
        raise ValueError('Local ground filter requires a finite diff-drive action, pose, dt and speed')
    local = _ground_local_geometry(observation, backend)
    requested = float(np.clip(motion[0], -max_speed, max_speed))
    direction = np.asarray([math.cos(heading), math.sin(heading)])

    def clear(fraction):
        return local.segment_clear(position, position + requested * fraction * dt * direction, clearance)

    supported = local.point_clear(position, clearance)
    speed = requested
    if requested and not clear(1.):
        lower, upper = 0., 1.
        if supported:
            for _ in range(24):
                midpoint = (lower + upper) / 2
                if clear(midpoint):
                    lower = midpoint
                else:
                    upper = midpoint
        speed = requested * lower
        if abs(speed) < 1e-5:
            speed = 0.
    changed = abs(speed - requested) > 1e-9
    result = dict(action, motion=(speed, float(motion[1])))
    return result, dict(intervened=changed, requested_motion=motion.tolist(),
        executed_motion=list(result['motion']), dt=dt, heading=heading,
        position=position.tolist(), clearance_m=float(clearance),
        current_capsule_observed_free=bool(supported))


def sensor_support_at(forest, position, resolution=.5, shape=(16, 16), clearance=1.5):
    """Offline task precheck using the exact local ground sensor contract.

    Native cell boundaries avoid moving-grid pooling inflation. Source unknown
    cells and cells not fully within the sensor footprint remain unprovable.
    The source geometry comparison is diagnostic and never enters a policy.
    """
    position = np.asarray(position, float)
    if position.ndim != 1 or position.size < 2 or not np.isfinite(position).all():
        raise ValueError('Sensor support requires a finite position')
    position = position[:2]
    observed = local_safety_observation(forest, position, shape, resolution)
    known = np.asarray(observed['known_mask'], bool)
    local = type(forest)(observed['occupancy_map'], known,
                         observed['resolution_m'], observed['origin_xy'])
    supported = bool(local.point_clear(position, clearance))
    if supported:
        reason = 'locally_certified_clearance'
    elif not local.segment_known(position, position, clearance):
        reason = 'local_unknown_or_outside_within_clearance'
    else:
        reason = 'local_occupied_within_clearance'
    return dict(supported=supported, reason=reason,
        source_point_clear=bool(forest.point_clear(position, clearance)),
        known_fraction=float(known.mean()), sensor_resolution_m=float(resolution),
        sensor_shape=list(np.broadcast_to(np.asarray(shape, int), (2,))),
        safety_shape=list(known.shape), safety_resolution_m=observed['resolution_m'],
        clearance_m=float(clearance))


def validate_sensor_route_support(forest, reference_points, resolution=.5, shape=(16, 16),
                                  clearance=1.5, max_spacing=.25):
    """Offline diagnostic for a given route, never an automatic route filter.

    It can be stricter than source-map feasibility, and another nearby route
    may have better sensor support. Contiguous supported samples below are
    candidate subroutes; finite sampling does not certify the intervals in
    between or guarantee that the rule controller can track them.
    """
    points = np.asarray(reference_points, float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Sensor route support requires nonempty finite XY reference points')
    if max_spacing is not None and (not math.isfinite(max_spacing) or max_spacing <= 0):
        raise ValueError('Route sampling spacing must be finite and positive, or None')
    samples, distances = [points[0].copy()], [0.]
    cumulative = 0.
    for start, end in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(end - start))
        if length == 0:
            continue
        count = 1 if max_spacing is None else max(1, math.ceil(length / max_spacing))
        for index in range(1, count + 1):
            fraction = index / count
            samples.append(start + fraction * (end - start))
            distances.append(cumulative + fraction * length)
        cumulative += length
    diagnostics = [sensor_support_at(forest, point, resolution, shape, clearance) for point in samples]
    supported = np.asarray([item['supported'] for item in diagnostics], bool)
    subroutes = []
    begin = None
    for index, good in enumerate(np.r_[supported, False]):
        if good and begin is None:
            begin = index
        elif not good and begin is not None:
            end = index - 1
            subroutes.append(dict(first_sample=begin, last_sample=end,
                start_xy=samples[begin].tolist(), end_xy=samples[end].tolist(),
                start_distance_m=distances[begin], end_distance_m=distances[end],
                length_m=distances[end] - distances[begin]))
            begin = None
    unsupported = [dict(sample=index, distance_m=distances[index], position=samples[index].tolist(),
                        **diagnostics[index]) for index in np.flatnonzero(~supported).tolist()]
    return dict(diagnostic_only=True, continuous_clearance_certified=False,
        route_samples_supported=bool(supported.all()), start=diagnostics[0], goal=diagnostics[-1],
        usable_points=int(supported.sum()), unsupported_points=int((~supported).sum()),
        sampled_points=len(samples), reference_distance_m=cumulative, max_spacing_m=max_spacing,
        unsupported_samples=unsupported, sample_supported_subroutes=subroutes,
        limitation='Failure concerns this sampled reference route and sensor contract only; '
                   'it does not rule out alternative paths or certify the unsampled intervals.')


def install_observation_support(env, forest, clearance):
    """Local known masks plus a separate safety capsule around the 0.3m car."""
    original_observations, original_collisions = env._get_observations, env._handle_collisions
    env.ground_safety_violations = 0
    env.raster_safety_unknown_attempts = 0
    env.raster_safety_events = []

    def observations(instance):
        result = original_observations()
        ground_ids = {robot.robot_id for robot in instance.ground_robots}
        for robot_id, observation in result.items():
            raw = np.asarray(observation['occupancy_map'])
            local = forest.local_observation(observation['self']['position'], raw.shape,
                                             observation['map_resolution'])
            observation['occupancy_map'] = np.where(local['known_mask'],
                local['occupancy_map'], -1).astype(np.int8)
            observation['occupancy_known_mask'] = local['known_mask']
            if robot_id in ground_ids:
                observation['ground_safety_map'] = local_safety_observation(
                    forest, observation['self']['position'], raw.shape, observation['map_resolution'])
        return result

    def collisions(instance, info):
        for robot in instance.ground_robots:
            before, proposed = robot.prev_position[:2].copy(), robot.position[:2].copy()
            if not forest.segment_clear(before, proposed, clearance):
                unknown = not forest.segment_known(before, proposed, clearance)
                physical = not forest.segment_clear(before, proposed, robot.radius)
                instance.ground_safety_violations += 1
                instance.raster_safety_unknown_attempts += int(unknown)
                event = dict(step=int(instance.step_count) + 1, robot_id=robot.robot_id,
                    before=before.tolist(), proposed=proposed.tolist(), clearance_m=clearance,
                    physical_collision=physical, unknown_or_outside=unknown)
                instance.raster_safety_events.append(event)
                info.setdefault('ground_safety_violations', []).append(event)
                # The physical adapter runs after this guard; record the
                # rejected proposal before rollback so physical hits survive.
                if physical:
                    instance.raster_collision_attempts += 1
                    instance.raster_unknown_attempts += int(
                        not forest.segment_known(before, proposed, robot.radius))
                    instance.raster_collision_events.append(event)
                info.setdefault('collisions', []).append(dict(robot_id=robot.robot_id,
                    object='obstacle', obstacle_type='raster_safety', unknown_or_outside=unknown))
                robot.position = robot.prev_position.copy()
                robot.velocity = np.zeros(3)
        original_collisions(info)

    env._get_observations = MethodType(observations, env)
    env._handle_collisions = MethodType(collisions, env)
    return env


@contextlib.contextmanager
def raster_runtime(me, forest, backend, clearance=1.5):
    """Bind before Experiment initializes, restore globals on all exits."""
    originals = (me.MultiRobotEnv, me.DroneController, me.GroundRobotController, me.DeliveredEvidence)
    original_env, original_drone, original_ground, original_ledger = originals

    def environment(config):
        result = backend.install_raster_environment(original_env(config), forest)
        return install_observation_support(result, forest, clearance)

    def drone(*args, **kwargs):
        result = original_drone(*args, **kwargs)
        _known_semantics(result.semantic_processor)
        return result

    def ground(*args, **kwargs):
        result = original_ground(*args, **kwargs)
        _known_semantics(result.local_perception)
        avoid = result.local_planner._avoid_obstacles

        def conservative_avoid(linear, angular, heading, semantic_map, resolution):
            # Only this disposable control computation treats unknown as an
            # obstacle. The input, remembered evidence and broadcasts stay 0.
            blocked = np.where(np.asarray(semantic_map) == 0, 2, semantic_map)
            return avoid(linear, angular, heading, blocked, resolution)

        result.local_planner._avoid_obstacles = conservative_avoid
        return result

    class KnownEvidence(original_ledger):
        receive = receive_known_evidence

    try:
        me.MultiRobotEnv, me.DroneController = environment, drone
        me.GroundRobotController, me.DeliveredEvidence = ground, KnownEvidence
        yield
    finally:
        me.MultiRobotEnv, me.DroneController, me.GroundRobotController, me.DeliveredEvidence = originals
