"""Certify short ground connections on the vehicle's current native occupancy tile."""
from __future__ import annotations

import numpy as np


# At tangency, rounding must never authorize contact with an unknown or occupied cell.
_EPS_M = 1e-9


def _sweep_hits_boxes(start, end, lower, side, radius):
    """Test a segment against closed, axis-aligned squares thickened by radius."""
    if len(lower) == 0:
        return False
    upper = lower + side
    # Segment/box intersection needs all three separating axes: X, Y, and the
    # segment normal. An interior crossing need not contain a corner or endpoint.
    delta = end - start
    normal = np.array([-delta[1], delta[0]])
    centre_offset = lower + side / 2 - (start + end) / 2
    on_xy = np.all(np.abs(centre_offset) <= (np.abs(delta) + side) / 2 + _EPS_M, axis=1)
    normal_limit = (side / 2 + _EPS_M) * np.abs(normal).sum()
    if np.any(on_xy & (np.abs(centre_offset @ normal) <= normal_limit)):
        return True
    # For disjoint segments and rectangles, the minimum is attained at a
    # segment endpoint against a box edge, or a box corner against the segment.
    limit_sq = (radius + _EPS_M) ** 2
    for point in (start, end):
        offset = np.maximum(np.maximum(lower - point, point - upper), 0.)
        if np.any(np.einsum('ij,ij->i', offset, offset) <= limit_sq):
            return True
    length_sq = float(delta @ delta)
    if length_sq == 0.:
        return False
    corners = lower[:, None, :] + side * np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    relative = corners - start
    t = np.clip((relative @ delta) / length_sq, 0., 1.)
    residual = relative - t[..., None] * delta
    return bool(np.any(np.einsum('ijk,ijk->ij', residual, residual) <= limit_sq))


def _finite_xy(value):
    return value.shape == (2,) and np.isfinite(value).all()


def native_segment_clear(tile_yx, sensor_center_xy, sensor_resolution, car_xy, goal_xy,
                         clearance_m=1.5):
    """Require a continuous clearance disk wholly inside the car's native free pixels.

    The tile is ego-centred and XY-aligned in the world frame, indexed as YX.
    Only native value 0 denotes free. Every unknown or occupied pixel represents
    its entire closed physical rectangle, including its boundary.
    """
    tile = np.asarray(tile_yx)
    center, start, end = (np.asarray(p, dtype=float) for p in (sensor_center_xy, car_xy, goal_xy))
    if (tile.ndim != 2 or not tile.size or not all(_finite_xy(p) for p in (center, start, end))
            or not np.isfinite(sensor_resolution) or sensor_resolution <= 0.
            or not np.isfinite(clearance_m) or clearance_m < 0.):
        return False
    extent = np.asarray(tile.shape[::-1]) * sensor_resolution
    origin = center - extent / 2
    low = np.minimum(start, end) - clearance_m - _EPS_M
    high = np.maximum(start, end) + clearance_m + _EPS_M
    if np.any(low < origin) or np.any(high > origin + extent):
        return False
    first = np.maximum(0, np.floor((low - origin) / sensor_resolution).astype(int))
    stop = np.minimum(tile.shape[::-1], np.ceil((high - origin) / sensor_resolution).astype(int))
    crop = tile[first[1]:stop[1], first[0]:stop[0]]
    blocked_xy = np.argwhere(crop != 0)[:, ::-1] + first
    lower = origin + blocked_xy * sensor_resolution
    return not _sweep_hits_boxes(start, end, lower, sensor_resolution, clearance_m)


def certified_start_connector(tile_yx, sensor_center_xy, sensor_resolution, car_xy,
                              coarse_clearance_yx, coarse_origin_xy, coarse_resolution=1.,
                              clearance_m=1.5, *, min_distance_m=0.):
    """Return a start cell (row, col) and certified world-XY segment, or None.

    Only centres already labelled traversable (-1) on the supplied clearance
    grid are candidates. The native sensor additionally certifies the exact car
    pose, the connection and its endpoint using continuous swept clearance.
    """
    tile, grid = np.asarray(tile_yx), np.asarray(coarse_clearance_yx)
    car, center, origin = (np.asarray(p, dtype=float) for p in (car_xy, sensor_center_xy, coarse_origin_xy))
    if (grid.ndim != 2 or not grid.size or not _finite_xy(origin)
            or not np.isfinite(coarse_resolution) or coarse_resolution <= 0.
            or not np.isfinite(min_distance_m) or min_distance_m < 0.
            or not native_segment_clear(tile, center, sensor_resolution, car, car, clearance_m)):
        return None
    shape_xy = np.array(grid.shape[::-1])
    boundary_lo = origin + clearance_m + _EPS_M
    boundary_hi = origin + shape_xy * coarse_resolution - clearance_m - _EPS_M
    if np.any(car < boundary_lo) or np.any(car > boundary_hi):
        return None
    sensor_half = np.asarray(tile.shape[::-1]) * sensor_resolution / 2
    low = np.maximum(boundary_lo, center - sensor_half + clearance_m + _EPS_M)
    high = np.minimum(boundary_hi, center + sensor_half - clearance_m - _EPS_M)
    first = np.maximum(0, np.ceil((low - origin) / coarse_resolution - .5).astype(int))
    stop = np.minimum(shape_xy, 1 + np.floor((high - origin) / coarse_resolution - .5).astype(int))
    if np.any(stop <= first):
        return None
    crop = grid[first[1]:stop[1], first[0]:stop[0]]
    cells_xy = np.argwhere(crop == -1)[:, ::-1] + first
    points = origin + (cells_xy + .5) * coarse_resolution
    distances = np.sum((points - car) ** 2, axis=1)
    # Prefer the cell containing the pose when multiple centres are equidistant.
    containing = np.floor((car - origin) / coarse_resolution).astype(int)
    order = np.lexsort((np.any(cells_xy != containing, axis=1), distances))
    for index in order:
        if min_distance_m > 0. and distances[index] <= (min_distance_m + _EPS_M) ** 2:
            continue
        point = points[index]
        if native_segment_clear(tile, center, sensor_resolution, car, point, clearance_m):
            return cells_xy[index, ::-1].copy(), np.stack((car, point))
    return None
