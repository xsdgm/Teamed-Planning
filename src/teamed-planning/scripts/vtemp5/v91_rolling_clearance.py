"""Guard motion against quantisation of a vehicle-centred binary sensor image."""
import math
from itertools import product

import numpy as np


def _cell_shifts(displacement):
    """Possible relative shifts of a static raster feature as the sensor moves.

    For a translation of d pixels an unknown subpixel phase can yield either
    floor(d) or ceil(d) cells of relative shift. Check every combination that
    can occur together along this segment, including its intermediate poses.
    Each possible centre is within sqrt(2) pixels of the continuous one.
    """
    cuts = [0., 1.]
    for component in np.abs(displacement):
        if component > 1:
            cuts.extend(np.arange(1, component) / component)
    ordered = sorted(set(cuts))
    shifts = {(0, 0)}
    for lo, hi in zip(ordered, ordered[1:]):
        at_mid = displacement * ((lo + hi) * .5)
        lower = np.floor(at_mid).astype(int)
        upper = np.ceil(at_mid).astype(int)
        shifts.update(product((lower[0], upper[0]), (lower[1], upper[1])))
    return shifts


class _RollingGeometry:
    def __init__(self, geometry):
        self.native = geometry
        self.resolution_m = geometry.resolution_m
        self.support = {}

    def point_clear(self, point, clearance=0.):
        return self.native.point_clear(point, clearance)

    def segment_clear(self, start, end, clearance=0.):
        # Occupied squares, unknown squares and the edge have their original
        # native swept capsule. Phase checks only add vetoes.
        if not self.native.segment_clear(start, end, clearance):
            return False
        a, b = np.asarray(start, float)[:2], np.asarray(end, float)[:2]
        if np.array_equal(a, b):
            return True
        shifts = _cell_shifts((b - a) / self.resolution_m)
        for dx, dy in shifts:
            key = (float(a[0]), float(a[1]), dx, dy, float(clearance))
            if key not in self.support:
                probe = a + self.resolution_m * np.array([dx, dy])
                self.support[key] = self.native.point_clear(probe, clearance)
            if not self.support[key]:
                return False
        return True


def local_geometry(observation, backend):
    from vtemp2_3.wildscenes_runtime import _ground_local_geometry
    geometry = _ground_local_geometry(observation, backend)
    if observation.get('ground_safety_map') is not None:
        return geometry
    return _RollingGeometry(geometry)


def guard_motion(action, observation, backend, *, clearance=1.5, max_speed=1.):
    from vtemp2_3.wildscenes_runtime import filter_ground_motion
    guarded, event = filter_ground_motion(action, observation, backend,
                                          clearance=clearance, max_speed=max_speed)
    if observation.get('ground_safety_map') is not None:
        return guarded, event
    speed, omega = guarded['motion']
    if not speed:
        return guarded, event
    local = local_geometry(observation, backend)
    position = np.asarray(observation['self']['position'], float)[:2]
    heading = float(observation['self']['orientation'])
    step = speed * observation['time_step'] * np.array([math.cos(heading), math.sin(heading)])
    if local.segment_clear(position, position + step, clearance):
        return guarded, event
    lo, hi = 0., 1.
    for _ in range(24):
        mid = (lo + hi) / 2
        if local.segment_clear(position, position + mid * step, clearance):
            lo = mid
        else:
            hi = mid
    reduced = speed * lo
    if abs(reduced) < 1e-5:
        reduced = 0.
    guarded = dict(guarded, motion=(float(reduced), omega))
    return guarded, dict(event, intervened=True, executed_motion=list(guarded['motion']),
                         rolling_phase_intervention=True)
