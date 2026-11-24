"""Conservative raster adapter for real forest geometry.

Arrays use ``[y, x]`` with the lower-left cell at ``origin_xy``.  Unknown
and out-of-map space are blocked; no trajectory or global map is included
in controller observations.  The adapter retains the existing planar robot
and ideal local-map sensor models.  It does not add 3D foliage or RF physics.
"""
from pathlib import Path
from types import MethodType

import numpy as np


class RasterForestMap:
    def __init__(self, occupied, known, resolution_m, origin_xy=(0., 0.),
                 terrain=None, metadata=None):
        occupied, known = np.asarray(occupied), np.asarray(known)
        if occupied.ndim != 2 or not occupied.size or occupied.shape != known.shape:
            raise ValueError('occupied and known must be nonempty matching [y,x] arrays')
        if not np.isin(occupied, [0, 1]).all() or not np.isin(known, [0, 1]).all():
            raise ValueError('occupied and known must be binary')
        self.resolution_m = float(resolution_m)
        self.origin_xy = np.asarray(origin_xy, dtype=float).copy()
        if (not np.isfinite(self.resolution_m) or self.resolution_m <= 0 or
                self.origin_xy.shape != (2,) or not np.isfinite(self.origin_xy).all()):
            raise ValueError('invalid metric resolution or origin')
        self.occupied = occupied.astype(bool, copy=True)
        self.known = known.astype(bool, copy=True)
        self.blocked = self.occupied | ~self.known
        self.free = ~self.blocked
        self.shape = self.occupied.shape
        self.height_m, self.width_m = np.asarray(self.shape) * self.resolution_m
        self.upper_xy = self.origin_xy + [self.width_m, self.height_m]
        self.extent = [self.origin_xy[0], self.upper_xy[0],
                       self.origin_xy[1], self.upper_xy[1]]
        self.terrain = None if terrain is None else np.asarray(terrain).copy()
        if self.terrain is not None and self.terrain.shape != self.shape:
            raise ValueError('terrain must have the same [y,x] shape')
        self.metadata = dict(metadata or {})
        # Integral occupancy gives exact conservative pooling even when a
        # sensor grid is not aligned with the source raster.
        self._integral = np.pad(self.blocked.astype(np.int64).cumsum(0).cumsum(1),
                                ((1, 0), (1, 0)))
        # Only the optional mapping observation needs a separate validity
        # plane. Build its integral once on first use, keeping the original
        # binary-only adapter's memory footprint unchanged.
        self._unknown_integral = None
        for array in (self.occupied, self.known, self.blocked, self.free,
                      self.origin_xy, self.upper_xy, self._integral):
            array.flags.writeable = False

    @classmethod
    def from_npz(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            required = {'occupied', 'known', 'resolution_m', 'origin_xy'}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f'Missing raster fields: {sorted(missing)}')
            metadata = {}
            for key in data.files:
                if key in required | {'terrain'}:
                    continue
                value = data[key]
                # Diagnostics can contain several full-size rasters.  Do not
                # expand them into millions of Python scalar objects per worker.
                metadata[key] = (value.tolist() if value.size <= 1024 else
                                 dict(shape=list(value.shape), dtype=str(value.dtype),
                                      omitted_from_metadata=True))
            return cls(data['occupied'], data['known'], data['resolution_m'],
                       data['origin_xy'], data['terrain'] if 'terrain' in data else None,
                       metadata=metadata)

    def point_status(self, points):
        points = np.asarray(points, dtype=float)
        if points.shape == () or points.shape[-1] < 2:
            raise ValueError('points must have a final dimension of at least two')
        xy = points[..., :2]
        finite = np.isfinite(xy).all(axis=-1)
        indices = np.floor((np.where(np.isfinite(xy), xy, self.origin_xy) -
                            self.origin_xy) / self.resolution_m).astype(np.int64)
        inside = (finite & (indices[..., 0] >= 0) & (indices[..., 1] >= 0) &
                  (indices[..., 0] < self.shape[1]) & (indices[..., 1] < self.shape[0]))
        ix = np.clip(indices[..., 0], 0, self.shape[1] - 1)
        iy = np.clip(indices[..., 1], 0, self.shape[0] - 1)
        known = inside & self.known[iy, ix]
        occupied = inside & self.occupied[iy, ix]
        return dict(inside=inside, known=known, occupied=occupied,
                    blocked=~known | occupied)

    def truth(self, points):
        """Blocked-space truth for independent evaluator calibration only."""
        return self.point_status(points)['blocked']

    def known_at(self, points):
        return self.point_status(points)['known']

    def _pooling_window(self, position, map_size, resolution_m, footprint_m):
        """Compute local source-cell ranges shared by occupancy and validity."""
        try:
            sizes = np.asarray(map_size, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError('map_size must be a positive integer or (height,width)') from error
        if sizes.ndim == 0:
            sizes = np.repeat(sizes, 2)
        if (sizes.shape != (2,) or not np.isfinite(sizes).all() or
                (sizes <= 0).any() or (sizes != np.floor(sizes)).any()):
            raise ValueError('map_size must be a positive integer or (height,width)')
        shape = tuple(int(value) for value in sizes)
        resolution_m = float(resolution_m)
        pose = np.asarray(position, dtype=float)
        if pose.ndim != 1 or pose.size < 2:
            raise ValueError('invalid sensor pose or resolution')
        center = pose[:2]
        if not np.isfinite(center).all() or not np.isfinite(resolution_m) or resolution_m <= 0:
            raise ValueError('invalid sensor pose or resolution')
        span = np.array([shape[1], shape[0]], float) * resolution_m
        if footprint_m is not None:
            footprint = np.asarray(footprint_m, dtype=float)
            if (footprint.shape not in ((), (2,)) or not np.isfinite(footprint).all() or
                    (footprint <= 0).any() or
                    not np.allclose(span, footprint, rtol=0., atol=1e-9)):
                raise ValueError('sensor footprint does not match size times resolution')
        lower = center - span / 2
        x0 = lower[0] + np.arange(shape[1]) * resolution_m
        y0 = lower[1] + np.arange(shape[0]) * resolution_m
        x1, y1 = x0 + resolution_m, y0 + resolution_m
        # Epsilon is solely to avoid fictitious overlap from floating error
        # on exact raster boundaries, never a physical clearance allowance.
        eps = 1e-10
        ix0 = np.floor((x0 - self.origin_xy[0]) / self.resolution_m + eps).astype(int)
        iy0 = np.floor((y0 - self.origin_xy[1]) / self.resolution_m + eps).astype(int)
        ix1 = np.ceil((x1 - self.origin_xy[0]) / self.resolution_m - eps).astype(int)
        iy1 = np.ceil((y1 - self.origin_xy[1]) / self.resolution_m - eps).astype(int)
        outside = ((x0 < self.origin_xy[0] - eps) | (x1 > self.upper_xy[0] + eps))[None, :] | (
            (y0 < self.origin_xy[1] - eps) | (y1 > self.upper_xy[1] + eps))[:, None]
        ix0, ix1 = np.clip(ix0, 0, self.shape[1]), np.clip(ix1, 0, self.shape[1])
        iy0, iy1 = np.clip(iy0, 0, self.shape[0]), np.clip(iy1, 0, self.shape[0])
        return outside, ix0, iy0, ix1, iy1

    @staticmethod
    def _pooled_mask(integral, window):
        outside, ix0, iy0, ix1, iy1 = window
        counts = (integral[iy1[:, None], ix1[None, :]] -
                  integral[iy0[:, None], ix1[None, :]] -
                  integral[iy1[:, None], ix0[None, :]] +
                  integral[iy0[:, None], ix0[None, :]])
        return outside | (counts > 0)

    def _known_integral(self):
        if self._unknown_integral is None:
            # Signed counts keep the inclusion/exclusion arithmetic exact;
            # int32 suffices for normal maps and halves the additional cache.
            dtype = np.int32 if self.known.size <= np.iinfo(np.int32).max else np.int64
            unknown = (~self.known).astype(dtype)
            self._unknown_integral = np.pad(
                unknown.cumsum(0, dtype=dtype).cumsum(1, dtype=dtype), ((1, 0), (1, 0)))
            self._unknown_integral.flags.writeable = False
        return self._unknown_integral

    def local_occupancy(self, position, map_size=12, resolution_m=2., footprint_m=None):
        """Pool every intersecting source cell; unknown/outside is occupied.

        ``map_size`` may be an integer or (height,width). The source raster
        is never returned, only a fixed-size binary local observation.
        """
        window = self._pooling_window(position, map_size, resolution_m, footprint_m)
        return self._pooled_mask(self._integral, window).astype(np.uint8)

    def local_observation(self, position, map_size=12, resolution_m=2., footprint_m=None):
        """Return a binary local map and its separate known-support mask.

        A cell is known only if it lies inside the source bounds and every
        intersecting source cell is known. Known occupied cells stay valid;
        unknown cells remain blocked in the binary map.
        """
        window = self._pooling_window(position, map_size, resolution_m, footprint_m)
        return dict(
            occupancy_map=self._pooled_mask(self._integral, window).astype(np.uint8),
            known_mask=~self._pooled_mask(self._known_integral(), window))

    def _segment_hits(self, start, end, clearance, mask):
        """Exact Euclidean capsule-versus-cell-square test, including corners."""
        a, b = np.asarray(start, float)[:2], np.asarray(end, float)[:2]
        r = float(clearance)
        if r < 0 or not np.isfinite(r):
            raise ValueError('clearance must be finite and nonnegative')
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return True
        if ((np.minimum(a, b) - r < self.origin_xy).any() or
                (np.maximum(a, b) + r >= self.upper_xy).any()):
            return True
        low = np.maximum(0, np.floor((np.minimum(a, b) - r - self.origin_xy) /
                                    self.resolution_m).astype(int) - 1)
        high = np.minimum([self.shape[1], self.shape[0]],
                          np.floor((np.maximum(a, b) + r - self.origin_xy) /
                                   self.resolution_m).astype(int) + 2)
        ys, xs = np.nonzero(mask[low[1]:high[1], low[0]:high[0]])
        if not len(xs):
            return False
        lo = self.origin_xy + np.column_stack((xs + low[0], ys + low[1])) * self.resolution_m
        hi = lo + self.resolution_m
        delta = b - a
        # Segment intersection with each closed axis-aligned square.
        tmin, tmax = np.zeros(len(xs)), np.ones(len(xs))
        intersects = np.ones(len(xs), dtype=bool)
        for axis in (0, 1):
            if abs(delta[axis]) < 1e-15:
                intersects &= (a[axis] >= lo[:, axis]) & (a[axis] <= hi[:, axis])
            else:
                t0, t1 = (lo[:, axis] - a[axis]) / delta[axis], (hi[:, axis] - a[axis]) / delta[axis]
                tmin = np.maximum(tmin, np.minimum(t0, t1))
                tmax = np.minimum(tmax, np.maximum(t0, t1))
        if (intersects & (tmin <= tmax)).any():
            return True
        distance2 = np.minimum(
            np.sum(np.maximum(np.maximum(lo - a, a - hi), 0.) ** 2, axis=1),
            np.sum(np.maximum(np.maximum(lo - b, b - hi), 0.) ** 2, axis=1))
        length2 = float(delta @ delta)
        if length2 > 0:
            for corner in (lo, hi, np.column_stack((lo[:, 0], hi[:, 1])),
                           np.column_stack((hi[:, 0], lo[:, 1]))):
                t = np.clip(((corner - a) @ delta) / length2, 0., 1.)
                distance2 = np.minimum(distance2, np.sum((corner - a - t[:, None] * delta) ** 2, axis=1))
        return bool((distance2 <= r * r + 1e-12).any())

    def segment_clear(self, start, end, clearance=0.):
        return not self._segment_hits(start, end, clearance, self.blocked)

    def point_clear(self, point, clearance=0.):
        return self.segment_clear(point, point, clearance)

    def segment_known(self, start, end, clearance=0.):
        return not self._segment_hits(start, end, clearance, ~self.known)


def install_raster_environment(env, forest, mapping_resolution=None):
    """Bind the adapter before ``env.reset()`` / Experiment initialization.

    Existing ground-ground and legacy obstacle collision handling is retained.
    Proposed ground motion is swept against the raster before any rollback,
    so tunneling and attempts to enter unmapped space remain measurable.
    Optional fine mapping observations share the existing drone footprint and
    pose. They do not replace the policy's original sensor or ground sensing.
    """
    if getattr(env, '_forest_adapter_installed', False):
        raise ValueError('raster adapter is already installed on this environment')
    if mapping_resolution is not None:
        mapping_resolution = float(mapping_resolution)
        footprint = float(env.config.perception_radius)
        if (not np.isfinite(mapping_resolution) or mapping_resolution <= 0 or
                not np.isfinite(footprint) or footprint <= 0):
            raise ValueError('mapping resolution and drone footprint must be finite and positive')
        cells = footprint / mapping_resolution
        if not np.isfinite(cells) or cells < 1 or not np.isclose(cells, round(cells), rtol=0., atol=1e-9):
            raise ValueError('drone footprint must be an integer multiple of mapping resolution')
        mapping_size = int(round(cells))
        original_observations = env._get_observations
    original_collisions = env._handle_collisions
    env.raster_collision_attempts = 0
    env.raster_unknown_attempts = 0
    env.raster_collision_events = []

    def occupancy(instance, robot_position, robot_type='drone'):
        cfg = instance.config
        ground = robot_type == 'ground'
        size = cfg.ground_occupancy_map_size if ground else cfg.occupancy_map_size
        resolution = (cfg.ground_occupancy_map_resolution or cfg.occupancy_map_resolution
                      if ground else cfg.occupancy_map_resolution)
        footprint = cfg.ground_perception_radius if ground else cfg.perception_radius
        return forest.local_occupancy(robot_position, size, resolution, footprint)

    def collisions(instance, info):
        for robot in instance.ground_robots:
            before, proposed = robot.prev_position[:2].copy(), robot.position[:2].copy()
            if not forest.segment_clear(before, proposed, robot.radius):
                unknown = not forest.segment_known(before, proposed, robot.radius)
                instance.raster_collision_attempts += 1
                instance.raster_unknown_attempts += int(unknown)
                event = dict(step=int(instance.step_count) + 1, robot_id=robot.robot_id,
                             before=before.tolist(), proposed=proposed.tolist(),
                             clearance_m=float(robot.radius), unknown_or_outside=unknown)
                instance.raster_collision_events.append(event)
                info.setdefault('raster_collision_attempts', []).append(event)
                if unknown:
                    info.setdefault('raster_unknown_attempts', []).append(event)
                info.setdefault('collisions', []).append(dict(robot_id=robot.robot_id,
                    object='obstacle', obstacle_type='raster', unknown_or_outside=unknown))
                robot.position = robot.prev_position.copy()
                robot.velocity = np.zeros(3)
        original_collisions(info)

    def observations(instance):
        result = original_observations()
        for drone in instance.drones:
            observation = result[drone.robot_id]
            local = forest.local_observation(
                observation['self']['position'], mapping_size, mapping_resolution,
                instance.config.perception_radius)
            observation.update(mapping_occupancy_map=local['occupancy_map'],
                               mapping_known_mask=local['known_mask'],
                               mapping_resolution=mapping_resolution, mapping_axes='yx')
        return result

    env._generate_occupancy_map = MethodType(occupancy, env)
    env._handle_collisions = MethodType(collisions, env)
    if mapping_resolution is not None:
        env._get_observations = MethodType(observations, env)
    env._forest_adapter_installed = True
    return env
