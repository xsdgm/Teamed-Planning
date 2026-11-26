"""Encode local route hypotheses for the frozen vehicle policy's road graph."""
from dataclasses import dataclass
import math

import numpy as np


CLEARANCE_M = 1.5
CELL_M = 2.
# A cell can intersect the swept body even when its centre is off the route.
CELL_MARGIN_M = CLEARANCE_M + CELL_M / math.sqrt(2.)


@dataclass
class RouteEvidence:
    points: np.ndarray          # Three world-XY points after the car
    measured: np.ndarray        # Direct support for each individual point
    risk: float                 # Maximum unknown-cell risk along the swept route
    mean_risk: float            # Secondary ranking among equally risky bottlenecks
    fully_observed: bool        # Every intersecting cell was observed free
    source: str

    @property
    def confidence(self):
        return 1. - self.risk


def _covered(points, xy, world_size):
    if np.any(points < CLEARANCE_M) or np.any(points > world_size - CLEARANCE_M):
        return False
    # Include the full physical clearance in the rotated CNN footprint. The
    # grid axes need not coincide with world X/Y.
    axes = np.array([xy[0, 1] - xy[0, 0], xy[1, 0] - xy[0, 0]]) / CELL_M
    local = (points - xy[0, 0]) @ axes.T + CELL_M / 2.
    bounds = np.array([xy.shape[1], xy.shape[0]]) * CELL_M
    return bool(np.all(local >= CLEARANCE_M) and np.all(local <= bounds - CLEARANCE_M))


def evaluate_route(points, xy, car, free, occupied, *, world_size, probabilities=None):
    """Score a three-vertex hypothesis. Only receiver observations can establish free space."""
    points = np.asarray(points, dtype=float)
    if points.shape != (3, 2) or not np.isfinite(points).all():
        raise ValueError('Expected three finite world-XY route vertices')
    free, occupied = np.asarray(free, dtype=bool), np.asarray(occupied, dtype=bool)
    if (free.shape != xy.shape[:2] or occupied.shape != free.shape
            or (probabilities is not None and np.shape(probabilities) != free.shape)):
        raise ValueError('Local route evidence must match the physical grid')
    track = np.vstack((car, points))
    if not _covered(track, xy, world_size) or not (free | occupied).any():
        return None
    swept = np.zeros(free.shape, dtype=bool)
    for a, b in zip(track[:-1], track[1:]):
        delta = b - a
        along = np.clip(((xy - a) @ delta) / max(delta @ delta, 1.e-12), 0., 1.)
        swept |= np.linalg.norm(xy - a - along[..., None] * delta, axis=-1) <= CELL_MARGIN_M
    if not swept.any() or occupied[swept].any():
        return None
    unknown = swept & ~free
    measured = []
    for point in points:
        disk = np.linalg.norm(xy - point, axis=-1) <= CELL_MARGIN_M
        measured.append(bool(disk.any() and free[disk].all()))
    # Ranking by peak risk keeps a thin transverse wall from disappearing
    # into an average over tens of metres of observed free road.
    peak = mean = 0.
    if unknown.any():
        if probabilities is None:
            peak = .5
            mean = .5 * unknown.sum() / swept.sum()
        else:
            values = np.asarray(probabilities)[unknown]
            if not np.isfinite(values).all() or np.any((values < 0.) | (values > 1.)):
                raise ValueError('Unknown-cell obstacle probabilities must be finite and in [0, 1]')
            peak, mean = float(values.max()), float(values.sum(dtype=np.float64) / swept.sum())
            if peak == 1.:
                return None
    return RouteEvidence(points.copy(), np.asarray(measured), peak, mean, not bool(unknown.any()),
                         'observed' if probabilities is None else 'predicted')


def route_candidates(xy, car, goal, free, occupied, *, world_size, probabilities=None):
    """Return at most one left and right hypothesis; an empty result supplies no new road."""
    delta = np.asarray(goal) - car
    distance = np.linalg.norm(delta)
    if not distance:
        return {}
    # Before the first aircraft receipts, a clear 8 m-wide ego sensor alone
    # supplies no evidence about either 8–32 m branch. Real local blockers count.
    outer = np.linalg.norm(xy - car, axis=-1) >= 8.
    if not np.any(occupied) and not np.any(free & outer):
        return {}
    forward = delta / distance
    left = np.array([-forward[1], forward[0]])
    selected = {}
    for side in (1, -1):
        best = None
        for degrees in range(12, 109, 12):
            angle = math.radians(degrees)
            heading = math.cos(angle) * forward + side * math.sin(angle) * left
            points = np.array([car + 8. * heading, car + 20. * heading,
                               car + 20. * heading + 12. * forward])
            route = evaluate_route(points, xy, car, free, occupied, world_size=world_size,
                                   probabilities=probabilities)
            if route is None:
                continue
            rank = (route.risk, route.mean_risk, degrees)
            if best is None or rank < best:
                best = rank
                selected[side] = route
    return selected


def append_route_evidence(obs, candidates, car, goal, now):
    """Use the frozen topology, with corridor support and vertex provenance kept separate."""
    from vtemp5.vtemp5_fleet import add_candidate_roads

    if any(side not in (1, -1) for side in candidates):
        raise ValueError('Route sides must be +1 (left) or -1 (right)')
    start = len(obs['road_valid'])
    state = dict(routes={s: r.points for s, r in candidates.items()},
                 route_observed={s: r.measured for s, r in candidates.items()},
                 route_open_probability={s: r.confidence for s, r in candidates.items()},
                 side_status={s: 'unknown' for s in candidates}, evidence_time=None)
    add_candidate_roads(obs, state, car, goal, now)
    for route in candidates.values():
        rows = obs['road_nodes'][start:start + 3]
        # Column 2 represents support for traversing the candidate even for
        # measured vertices separated by unknown terrain. Column 6 retains
        # each vertex's real observation count. Per-cell times are unavailable.
        rows[:, 2] = route.confidence
        rows[:, 3] = 0.
        rows[:, 4] = 1.
        start += 3
