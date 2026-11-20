"""Score native ground tracks against the water pixels of a finite lake after rollout."""
import numpy as np


def _dry_track(track, water, origin, resolution):
    # Split each 1 s line segment at every raster grid line. A midpoint tests the entire
    # open subsegment, including arbitrarily short passages through a water pixel.
    points = (track - origin) / resolution
    starts = points[:-1]
    deltas = np.diff(points, axis=0)
    segments = len(deltas)
    segment_ids = [np.arange(segments), np.arange(segments)]
    fractions = [np.zeros(segments), np.ones(segments)]
    for axis in (0, 1):
        low = np.floor(np.minimum(points[:-1, axis], points[1:, axis])).astype(int) + 1
        high = np.ceil(np.maximum(points[:-1, axis], points[1:, axis])).astype(int)
        count = np.maximum(high - low, 0)
        ids = np.repeat(np.arange(segments), count)
        offsets = np.arange(len(ids)) - (np.cumsum(count) - count)[ids]
        segment_ids.append(ids)
        fractions.append((low[ids] + offsets - starts[ids, axis]) / deltas[ids, axis])
    ids = np.concatenate(segment_ids)
    ts = np.concatenate(fractions)
    order = np.lexsort((ts, ids))
    ids, ts = ids[order], ts[order]
    same = (ids[:-1] == ids[1:]) & (ts[1:] > ts[:-1])
    mid = (ts[1:][same] + ts[:-1][same]) / 2
    i = ids[:-1][same]
    interior = starts[i] + deltas[i] * mid[:, None]
    vertices = np.vstack([points, interior])
    cells = np.floor(vertices).astype(int)
    if np.any((cells < 0) | (cells >= np.array(water.shape[::-1]))):
        raise ValueError('Ground trajectory leaves the lake evaluation raster')
    return not bool(water[cells[:, 1], cells[:, 0]].any())


def lake_metrics(track, cover, origin, resolution, start, goal, audit, direct):
    """Both sides count if a dry track crosses the midlake cut outside water then clears its far end."""
    track = np.asarray(track, dtype=float)
    if (track.ndim != 2 or track.shape[1] != 2 or not len(track) or not np.isfinite(track).all()
            or not np.allclose(track[0], start)):
        raise ValueError('Expected a finite ground trajectory beginning at the mission start')
    heading = np.subtract(goal, start).astype(float)
    heading /= np.linalg.norm(heading)
    normal = np.array([-heading[1], heading[0]])
    longitudinal = (track - start) @ heading
    lateral = (track - start) @ normal
    gate, far = audit['lake_gate_station_m'], audit['lake_exit_station_m']
    left, right = audit['gate_left_shore_offset_m'], audit['gate_right_shore_offset_m']
    if not 0 < gate < far < np.linalg.norm(np.subtract(goal, start)) or not right < left:
        raise ValueError('Invalid midlake shoreline or far-end station')
    dry = _dry_track(track, cover == 4, origin, resolution)
    side = side_at_exit = None
    gate_first = far_first = None
    first_side = None
    for i, (a, b) in enumerate(zip(longitudinal[:-1], longitudinal[1:])):
        if b < gate:
            side = None
        if a <= gate < b:
            fraction = (gate - a) / (b - a)
            offset = lateral[i] + fraction * (lateral[i + 1] - lateral[i])
            side = 'left' if offset > left else 'right' if offset < right else None
            if gate_first is None and side is not None:
                gate_first, first_side = float(i + fraction), side
        if far_first is None and a <= far < b:
            far_first = float(i + (far - a) / (b - a))
            side_at_exit = side
    completed = dry and far_first is not None and side_at_exit is not None
    result = dict(water_entered=not dry, reached_lake_far_end=completed,
                  far_end_station_first_s=far_first, lake_far_end_first_s=far_first if completed else None,
                  lake_central_cut_side=first_side,
                  lake_central_cut_first_s=gate_first,
                  max_lateral_deviation_m=float(np.abs(lateral).max()))
    if not direct:
        result.update(lake_bypassed=completed, lake_bypass_side=side_at_exit if completed else None,
                      lake_bypass_first_s=far_first if completed else None)
    return result
