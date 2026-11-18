"""Schema 4: received terrain and telemetry, with independent drive/survey commands.

The sparse terrain memory contains sensor samples only. A sampled coarse cell
is NOT a ground-route certificate. Navigation continues to use the original
local safety checks and received graph. Simulator audit payloads are ignored.
"""
from __future__ import annotations

import math
import numpy as np

SCHEMA_VERSION = 4
CHECKPOINT_FORMAT = 'vtmp2-decoupled-terrain-v1'
ACTION_CONTRACT = 'independent_drive_survey_v2'
ARCHITECTURE = 'terrain_dynamics_vtmp2'
VEHICLE_RUNTIME = 'vtmp_revisable_strategic_v1'
PACKET_KEY = 'world_model_observations'
SIDE = 64
HALF_EXTENT = 200.
RESOLUTION = 2 * HALF_EXTENT / SIDE
CHANNELS = 3  # observed occupied, observed mask, surveyed world boundary
GROUND_FEATURES = 14
DRONE_FEATURES = 12
ACTION_FEATURES = 6  # drive xy/mask, survey xy/mask; xy in global kilometres
MAX_PACKET_AGE = 30.


def point(value, dimensions=2):
    result = np.asarray(value, dtype=float)
    if result.shape != (dimensions,) or not np.isfinite(result).all():
        raise ValueError(f'Expected {dimensions} finite coordinates')
    return result.copy()


def action_vector(drive_target, survey_target):
    action = np.zeros(ACTION_FEATURES, np.float32)
    for offset, target in ((0, drive_target), (3, survey_target)):
        if target is not None:
            action[offset:offset+2] = point(target) / 1000.
            action[offset+2] = 1.
    return action


def crop_origin(position):
    return np.floor(point(position) / RESOLUTION).astype(int) - SIDE // 2


def cell_centers(origin):
    x, y = np.meshgrid(np.arange(SIDE), np.arange(SIDE))
    return (np.stack((x, y), -1) + np.asarray(origin) + .5) * RESOLUTION


def validate_packet(packet, now, drone_count):
    """Reject missing, stale, future or malformed packets before assimilation."""
    if not isinstance(packet, dict):
        return None
    try:
        source = packet['source']
        if source not in {f'drone_{i}' for i in range(drone_count)}:
            return None
        stamp = float(packet['time'])
        if not math.isfinite(stamp) or not 0 <= now - stamp <= MAX_PACKET_AGE:
            return None
        position = point(packet['position'], 3)
        velocity = point(packet['velocity'], 3)
        resolution = float(packet['resolution'])
        occupancy = np.asarray(packet['occupancy'])
        if (not math.isfinite(resolution) or resolution <= 0 or occupancy.ndim != 2
                or not 1 <= min(occupancy.shape) <= max(occupancy.shape) <= 128
                or not np.isin(occupancy, (0, 1)).all()):
            return None
        return dict(source=source, time=stamp, position=position, velocity=velocity,
                    resolution=resolution, occupancy=occupancy.astype(np.uint8),
                    valid=bool(packet.get('valid', False)))
    except (KeyError, ValueError, TypeError, OverflowError):
        return None


class ReceivedObservation:
    def __init__(self, world_bounds, drone_count=3):
        self.bounds = point(world_bounds, 4)
        if np.any(self.bounds[2:] <= self.bounds[:2]):
            raise ValueError('Invalid world bounds')
        if type(drone_count) is not int or not 0 <= drone_count <= 32:
            raise ValueError('drone_count must be an integer in [0, 32]')
        self.drone_count = drone_count
        self.cells = {}
        self.telemetry = {}
        self.received = set()
        self.now = None

    @property
    def features(self):
        return GROUND_FEATURES + DRONE_FEATURES * self.drone_count

    def add_sensor(self, position, occupancy, resolution):
        """Sample coarse world-cell centres covered by this actual sensor tile."""
        position = point(position)
        occupancy = np.asarray(occupancy)
        if (occupancy.ndim != 2 or min(occupancy.shape) < 1
                or not np.isin(occupancy, (0, 1)).all()
                or not np.isfinite(resolution) or resolution <= 0):
            raise ValueError('Invalid sensor tile')
        low = position - np.asarray(occupancy.shape[::-1]) * resolution / 2
        high = position + np.asarray(occupancy.shape[::-1]) * resolution / 2
        lo = np.ceil(low / RESOLUTION - .5).astype(int)
        hi = np.ceil(high / RESOLUTION - .5).astype(int)
        for y in range(lo[1], hi[1]):
            for x in range(lo[0], hi[0]):
                xy = (np.array([x, y]) + .5) * RESOLUTION
                if np.any(xy < self.bounds[:2]) or np.any(xy >= self.bounds[2:]):
                    continue
                col, row = np.floor((xy - low) / resolution).astype(int)
                value = int(occupancy[row, col])
                # Static conservative sensor memory: later free pixels cannot
                # erase a received obstacle. Predictions never enter this map.
                self.cells[x, y] = max(value, self.cells.get((x, y), 0))

    def receive(self, packets, now):
        now = float(now)
        if not math.isfinite(now) or (self.now is not None and now < self.now):
            raise ValueError('Observation time must be finite and nondecreasing')
        self.now, self.received = now, set()
        for raw in packets:
            packet = validate_packet(raw, now, self.drone_count)
            if packet is None:
                continue
            source = packet['source']
            previous = self.telemetry.get(source)
            if previous is not None and packet['time'] <= previous['time']:
                continue
            self.telemetry[source] = packet
            self.received.add(source)
            if packet['valid']:
                self.add_sensor(packet['position'][:2], packet['occupancy'], packet['resolution'])

    def raster(self, origin):
        origin = np.asarray(origin, dtype=int)
        maps = np.zeros((CHANNELS, SIDE, SIDE), np.float32)
        xy = cell_centers(origin)
        outside = np.any((xy < self.bounds[:2]) | (xy >= self.bounds[2:]), axis=-1)
        maps[2] = outside
        for row in range(SIDE):
            for col in range(SIDE):
                value = self.cells.get((col + int(origin[0]), row + int(origin[1])))
                if value is not None:
                    maps[0, row, col], maps[1, row, col] = value, 1.
        return maps

    def encode(self, position, heading, velocity, goal, drive_target=None, survey_target=None):
        position, velocity, goal = point(position), point(velocity), point(goal)
        if self.now is None or not math.isfinite(heading):
            raise ValueError('Receive an observation before encoding')
        states = np.zeros(self.features, np.float32)
        states[:8] = [*(position / 1000.), math.sin(heading), math.cos(heading),
                      *(velocity / 6.), *(goal / 1000.)]
        for offset, target in ((8, drive_target), (11, survey_target)):
            if target is not None:
                states[offset:offset+2] = (point(target) - position) / HALF_EXTENT
                states[offset+2] = 1.
        for index in range(self.drone_count):
            source = f'drone_{index}'
            packet = self.telemetry.get(source)
            row = states[GROUND_FEATURES + index*DRONE_FEATURES:GROUND_FEATURES + (index+1)*DRONE_FEATURES]
            row[6] = 1.  # Missing telemetry has maximum age, not a real zero state.
            if packet is None:
                continue
            age = self.now - packet['time']
            row[:3] = (packet['position'] - np.r_[position, 0.]) / HALF_EXTENT
            row[3:6] = packet['velocity'] / 6.
            row[6:] = [min(age/60., 1.), 1., float(source in self.received),
                       float(packet['valid']),
                       min(packet['occupancy'].shape)*packet['resolution']/2/HALF_EXTENT,
                       float(age <= MAX_PACKET_AGE)]
        origin = crop_origin(position)
        return self.raster(origin), states, origin

    def observed(self, target):
        return tuple(np.floor(point(target) / RESOLUTION).astype(int)) in self.cells


def motion_state(states, drone_count):
    """Ground and last-reported aircraft positions in absolute kilometres."""
    states = np.asarray(states)
    values = [states[..., :2]]
    for i in range(drone_count):
        start = GROUND_FEATURES + i*DRONE_FEATURES
        relative = states[..., start:start+3] * HALF_EXTENT / 1000.
        values.append(relative + np.concatenate((states[..., :2], np.zeros_like(states[..., :1])), -1))
    return np.concatenate(values, -1)
