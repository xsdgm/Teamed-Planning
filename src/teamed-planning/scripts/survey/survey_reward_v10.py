"""Local UAV occupancy evidence for a fixed a--b survey task.

The task is a capsule on a world-aligned grid. Unknown cells stay in its fixed
denominator. Each UAV fuses only sensor packets it acquired or actually received;
neither a vehicle map nor simulator truth is an input. The team score is the
pointwise maximum of these local confidences, so copying an unchanged belief to
another UAV does not multiply its value.

The sensor likelihood (.8) is a simulation assumption, not a calibration claim.
Confidence measures certainty of occupied *or* free cells; obstacles are useful
survey information too. Signed changes retain penalties for contradictory data.
"""
from dataclasses import dataclass, field
import math
from numbers import Integral, Real

import numpy as np


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a positive finite number')
    result = float(value)
    if not math.isfinite(result) or result <= 0.:
        raise ValueError(f'{name} must be a positive finite number')
    return result


def _point(name, value, world_size):
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f'{name} must contain two finite coordinates')
    if ((point < 0.) | (point > world_size)).any():
        raise ValueError(f'{name} must lie inside the world bounds')
    return tuple(float(v) for v in point)


@dataclass(frozen=True)
class SurveyTask:
    """Immutable survey endpoints and width, independent of vehicle position c.

    A capsule extending beyond the world is clipped to the world grid. Its
    actual sampled area is exposed explicitly in ``area_m2``.
    """
    a: tuple
    b: tuple
    half_width: float
    world_size: float = 1000.
    resolution: float = 2.
    cells_xy: np.ndarray = field(init=False, repr=False, compare=False)
    cell_ids: np.ndarray = field(init=False, repr=False, compare=False)
    grid_size: int = field(init=False)

    def __post_init__(self):
        for name in ('half_width', 'world_size', 'resolution'):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        size = self.world_size / self.resolution
        if not math.isclose(size, round(size), rel_tol=0., abs_tol=1e-9):
            raise ValueError('world_size must be an integer multiple of resolution')
        if round(size) < 1:
            raise ValueError('resolution cannot exceed world_size')
        object.__setattr__(self, 'grid_size', int(round(size)))
        object.__setattr__(self, 'a', _point('a', self.a, self.world_size))
        object.__setattr__(self, 'b', _point('b', self.b, self.world_size))
        a, b = np.asarray(self.a), np.asarray(self.b)
        lo = np.maximum(0, np.floor((np.minimum(a, b)-self.half_width)/self.resolution).astype(int))
        hi = np.minimum(self.grid_size-1,
                        np.floor((np.maximum(a, b)+self.half_width)/self.resolution).astype(int))
        ix, iy = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
        cells = np.column_stack(((ix.ravel()+.5)*self.resolution,
                                 (iy.ravel()+.5)*self.resolution))
        delta = b-a
        along = np.clip((cells-a)@delta/max(float(delta@delta), 1e-12), 0., 1.)
        selected = np.linalg.norm(cells-a-along[:, None]*delta, axis=1) <= self.half_width+1e-9
        cells = cells[selected]
        ids = (iy.ravel()*self.grid_size+ix.ravel())[selected]
        if not len(ids):
            raise ValueError('Survey task contains no cell centres at this resolution')
        # meshgrid traversal makes ids strictly increasing, allowing searchsorted
        # instead of allocating an entire world-sized lookup for every task.
        cells.setflags(write=False)
        ids.setflags(write=False)
        object.__setattr__(self, 'cells_xy', cells)
        object.__setattr__(self, 'cell_ids', ids)

    @property
    def cell_count(self):
        return len(self.cell_ids)

    @property
    def area_m2(self):
        return self.cell_count*self.resolution**2

    def to_message(self):
        return dict(a=list(self.a), b=list(self.b), half_width=self.half_width)


class LocalSurveyEvidence:
    """Per-UAV local fusion with one vote per source/frame/world cell.

    ``packet`` has the existing sensor-audit fields: source, frame, position,
    occupancy (row Y, column X), resolution and optional valid. Calling code is
    responsible for feeding only the recipient's own or actually received
    packets. No inferred delivery is allowed. Packet source identity survives
    forwarding; repeated packets are ignored independently at each recipient.
    """
    SENSOR_RELIABILITY = .8
    LOG_ODDS_LIMIT = 3.

    def __init__(self, task, num_drones, confidence_threshold=.8):
        if not isinstance(task, SurveyTask):
            raise ValueError('task must be a SurveyTask')
        if isinstance(num_drones, bool) or not isinstance(num_drones, Integral) or num_drones < 1:
            raise ValueError('num_drones must be a positive integer')
        threshold = _positive('confidence_threshold', confidence_threshold)
        if threshold > 1.:
            raise ValueError('confidence_threshold must not exceed one')
        self.task = task
        self.num_drones = int(num_drones)
        self.confidence_threshold = threshold
        self.log_odds = np.zeros((self.num_drones, task.cell_count), dtype=np.float64)
        self.seen = np.zeros_like(self.log_odds, dtype=bool)
        self.received = [set() for _ in range(self.num_drones)]
        self._previous_confidence = 0.

    def ingest(self, drone_index, packet):
        """Fuse a locally available packet; return whether relevant cells changed."""
        if (isinstance(drone_index, bool) or not isinstance(drone_index, Integral)
                or not 0 <= drone_index < self.num_drones):
            raise ValueError('drone_index is outside this fleet')
        if not packet.get('valid', True) or packet.get('occupancy') is None:
            return False
        source, frame = packet['source'], packet['frame']
        if isinstance(frame, bool) or not isinstance(frame, Real) or not math.isfinite(frame) or frame < 0:
            raise ValueError('packet frame must be a finite nonnegative number')
        key = (str(source), float(frame))
        if key in self.received[drone_index]:
            return False
        occupancy = np.asarray(packet['occupancy'], dtype=float)
        if occupancy.ndim != 2 or not all(occupancy.shape):
            raise ValueError('packet occupancy must be a nonempty 2D grid')
        position = np.asarray(packet['position'], dtype=float)
        if position.ndim != 1 or len(position) not in (2, 3) or not np.isfinite(position).all():
            raise ValueError('packet position must have two or three finite coordinates')
        resolution = _positive('packet resolution', packet['resolution'])
        h, w = occupancy.shape
        xs = position[0]+(np.arange(w)+.5-w/2)*resolution
        ys = position[1]+(np.arange(h)+.5-h/2)*resolution
        ix, iy = np.meshgrid(np.floor(xs/self.task.resolution).astype(int),
                             np.floor(ys/self.task.resolution).astype(int))
        valid = (np.isfinite(occupancy) & (occupancy >= 0.) & (occupancy <= 1.)
                 & (ix >= 0) & (iy >= 0) & (ix < self.task.grid_size) & (iy < self.task.grid_size))
        ids = iy[valid]*self.task.grid_size+ix[valid]
        locations = np.searchsorted(self.task.cell_ids, ids)
        selected = locations < self.task.cell_count
        selected[selected] &= self.task.cell_ids[locations[selected]] == ids[selected]
        # Mark even irrelevant packets consumed: receiving the same old picture
        # later cannot change its original footprint or task membership.
        self.received[drone_index].add(key)
        if not selected.any():
            return False
        cells, inverse = np.unique(locations[selected], return_inverse=True)
        votes = np.bincount(inverse, weights=occupancy[valid][selected])
        counts = np.bincount(inverse)
        magnitude = math.log(self.SENSOR_RELIABILITY/(1.-self.SENSOR_RELIABILITY))
        update = np.where(votes/counts >= .5, magnitude, -magnitude)
        before = self.log_odds[drone_index, cells].copy()
        self.log_odds[drone_index, cells] = np.clip(before+update, -self.LOG_ODDS_LIMIT, self.LOG_ODDS_LIMIT)
        changed = bool((self.log_odds[drone_index, cells] != before).any()
                       or not self.seen[drone_index, cells].all())
        self.seen[drone_index, cells] = True
        return changed

    def local_confidence(self):
        """Return normalized occupancy certainty for each UAV and task cell."""
        return np.minimum(1., np.tanh(np.abs(self.log_odds)/2.)
                          /math.tanh(self.LOG_ODDS_LIMIT/2.))*self.seen

    def metrics(self):
        local = self.local_confidence()
        best = local.max(axis=0)
        return dict(survey_confidence=float(best.mean()),
                    survey_coverage=float(np.mean(best >= self.confidence_threshold)),
                    survey_observed_fraction=float(np.mean(self.seen.any(axis=0))),
                    survey_local_confidence_mean=float(local.mean()),
                    survey_local_coverage_mean=float(np.mean(local >= self.confidence_threshold)),
                    survey_area_m2=float(self.task.area_m2),
                    survey_cell_count=float(self.task.cell_count))

    def step_metrics(self):
        """Consume one signed potential difference; metric reads do not consume it."""
        metrics = self.metrics()
        confidence = metrics['survey_confidence']
        metrics['survey_confidence_gain'] = confidence-self._previous_confidence
        self._previous_confidence = confidence
        return metrics
