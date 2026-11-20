"""Received terrain evidence, candidate subgoal features and frozen-policy return estimates."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label


@dataclass
class EvidenceGrid:
    """Received observations only, with row-Y/column-X coordinates.

    Zero is unknown, -1 is observed free, +1 is observed occupied. Reachability
    is a four-neighbour raster statement within the supplied bounds, not a
    proof about routes outside the map or noisy real-world perception.
    """
    cells: np.ndarray
    resolution: float = 2.
    origin: tuple = (0., 0.)
    clearance: float = 1.5

    def __post_init__(self):
        self.cells = np.array(self.cells, dtype=np.int8, copy=True)
        if self.cells.ndim != 2 or not self.cells.size or self.cells.min() < -1 or self.cells.max() > 1:
            raise ValueError('cells must be a nonempty ternary grid')
        if not math.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError('resolution must be finite and positive')
        if not math.isfinite(self.clearance) or self.clearance < 0:
            raise ValueError('clearance must be finite and nonnegative')
        self.origin = tuple(np.asarray(self.origin, float))
        if len(self.origin) != 2 or not np.isfinite(self.origin).all():
            raise ValueError('origin must be a finite XY pair')
        radius = int(np.ceil(self.clearance / self.resolution))
        self.blocked, self.known_free = self.cells == 1, self.cells == -1
        if radius:
            # Conservative square-cell support, including diagonally adjacent cells.
            footprint = np.ones((2*radius+1, 2*radius+1), dtype=bool)
            self.blocked = binary_dilation(self.blocked, structure=footprint)
            self.known_free = binary_erosion(self.known_free, structure=footprint, border_value=0)
        self.optimistic = ~self.blocked
        structure = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=bool)
        self.free_components = label(self.known_free, structure)[0]
        self.possible_components = label(self.optimistic, structure)[0]

    def index(self, point):
        xy = np.floor((np.asarray(point, float)[:2] - self.origin) / self.resolution).astype(int)
        x, y = xy
        return (y, x) if 0 <= y < self.cells.shape[0] and 0 <= x < self.cells.shape[1] else None

    def sample(self, points):
        xy = np.floor((np.asarray(points) - self.origin) / self.resolution).astype(int)
        valid = ((xy[:,0] >= 0) & (xy[:,1] >= 0)
                 & (xy[:,0] < self.cells.shape[1]) & (xy[:,1] < self.cells.shape[0]))
        result = np.zeros(len(points), np.int8)
        result[valid] = self.cells[xy[valid,1], xy[valid,0]]
        return result, valid

    def region(self, point, radius):
        center = (np.asarray(point) - self.origin) / self.resolution
        x0, y0 = np.floor(center-radius/self.resolution-.5).astype(int)
        x1, y1 = np.ceil(center+radius/self.resolution+.5).astype(int)
        x0, y0 = max(x0,0), max(y0,0)
        x1, y1 = min(x1,self.cells.shape[1]-1), min(y1,self.cells.shape[0]-1)
        if x1 < x0 or y1 < y0:
            return np.array([], int), np.array([], int)
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
        inside = ((xx+.5-center[0])**2 + (yy+.5-center[1])**2 <= (radius/self.resolution)**2+1e-9)
        return yy[inside], xx[inside]

    def classify(self, start, target, arrival=25.):
        source, destination = self.index(start), self.index(target)
        if source is None or destination is None:
            return 'outside_bounds'
        yy, xx = self.region(target, arrival)
        if not len(yy):
            yy, xx = np.array([destination[0]]), np.array([destination[1]])
        possible = self.possible_components[source]
        if possible == 0:
            # Inconsistent current-pose evidence is not proof that every target
            # is unreachable. Let the ground safety layer resolve this state.
            return 'source_conflict'
        if not np.any(self.possible_components[yy,xx] == possible):
            return 'blocked_within_bounds'
        known = self.free_components[source]
        if known and np.any(self.free_components[yy,xx] == known):
            return 'known_reachable'
        return 'unknown'


class AttemptMemory:
    """Failed attempts are policy experience, never occupancy observations."""
    REASONS = {'timeout', 'control_stall', 'communication_wait'}

    def __init__(self):
        self.records = []

    def record(self, start, target, now, reason):
        if reason not in self.REASONS:
            raise ValueError('Only explicit execution-failure reasons belong in attempt memory')
        self.records.append(dict(start=np.asarray(start,float).copy(),
                                 target=np.asarray(target,float).copy(), time=float(now), reason=reason))

    def cost(self, start, target, now):
        # A temporary cost for substantially the same approach, not a 60 m
        # exclusion disk that blocks neighbouring entrances or future evidence.
        return sum(math.exp(-max(0.,now-r['time'])/180.) for r in self.records
                   if np.linalg.norm(np.asarray(start)-r['start']) <= 5.
                   and np.linalg.norm(np.asarray(target)-r['target']) <= 5.)


def candidates(position, goal, bounds=(0.,0.,1000.,1000.)):
    """Predeclared five-point pilot subset of the original 72 ring candidates."""
    position, goal = np.asarray(position,float), np.asarray(goal,float)
    heading = math.atan2(*(goal-position)[::-1])
    specifications = [(200.,0.), (200.*2/3,0.), (200./3,0.),
                      (200.*2/3,math.pi/6), (200.*2/3,-math.pi/6)]
    points = [position + radius*np.array([math.cos(heading+angle),math.sin(heading+angle)])
              for radius,angle in specifications]
    xmin,ymin,xmax,ymax = bounds
    return np.asarray([q for q in points if xmin+10 <= q[0] <= xmax-10 and ymin+10 <= q[1] <= ymax-10])


FEATURE_NAMES = ['target_forward', 'target_lateral', 'target_distance', 'remaining_distance',
                 'euclidean_progress', 'graph_nodes', 'graph_edges', 'nearest_node',
                 'known_reachable', 'blocked', 'unknown', 'source_conflict',
                 'target_free_fraction', 'target_blocked_fraction', 'target_unknown_fraction',
                 'recent_receipts', 'receipt_age', 'speed'] + [
                     f'corridor_{i}_{kind}' for i in range(8) for kind in ('free','blocked','unknown')]


def corridor_fractions(grid, position, targets, *, sample_bounds=None):
    """Return eight free/blocked/unknown triples. Optional XY bounds exclude exterior samples."""
    deltas = np.asarray(targets, dtype=float).reshape(-1, 2) - position
    normals = np.column_stack((-deltas[:, 1], deltas[:, 0]))
    normals /= np.maximum(np.linalg.norm(deltas, axis=1), 1e-9)[:, None]
    along = np.arange(8)[:, None] / 8. + np.linspace(0., 1./8., 5)
    across = np.linspace(-12., 12., 7)
    centres = position + along[None, :, :, None] * deltas[:, None, None, :]
    points = centres[:, :, :, None, :] + across[None, None, None, :, None] * normals[:, None, None, None, :]
    points = points.reshape(-1, 2)
    cells, valid = grid.sample(points)
    sections = cells.reshape(len(deltas), 8, 35)
    if sample_bounds is not None:
        low, high = np.asarray(sample_bounds, dtype=float).reshape(2, 2)
        valid &= np.all((points >= low) & (points < high), axis=1)
        inside = valid.reshape(sections.shape)
        count = inside.sum(axis=2)
        portions = np.stack([((sections == code) & inside).sum(axis=2)
                             for code in (-1, 1, 0)], axis=-1) / np.maximum(count[..., None], 1)
        portions[count == 0] = (0., 0., 1.)
        return portions.reshape(len(deltas), 24)
    return np.stack([(sections == code).mean(axis=2) for code in (-1, 1, 0)], axis=-1).reshape(len(deltas), 24)


def features(belief, target, *, corridor=None):
    """Candidate-conditioned features; no environment, hidden-map or critic input."""
    p, goal, q = (np.asarray(v,float) for v in (belief['position'],belief['goal'],target))
    grid = belief['grid']
    delta, direction = q-p, goal-p
    distance = float(np.linalg.norm(delta))
    heading = direction/max(np.linalg.norm(direction),1e-9)
    normal = np.array([-heading[1],heading[0]])
    status = grid.classify(p,q)
    yy,xx = grid.region(q,25.)
    local = grid.cells[yy,xx] if len(yy) else np.array([0])
    nodes = np.asarray(belief.get('graph_positions',[]),float).reshape(-1,2)
    nearest = min(200.,float(np.linalg.norm(nodes-q,axis=1).min())) if len(nodes) else 200.
    v = [delta@heading/200., delta@normal/200., distance/200., np.linalg.norm(goal-q)/1000.,
         (np.linalg.norm(direction)-np.linalg.norm(goal-q))/200.,
         np.log1p(len(nodes))/10., np.log1p(belief.get('graph_edges',0))/10.,nearest/200.,
         float(status=='known_reachable'),float(status in ('blocked_within_bounds','outside_bounds')),
         float(status=='unknown'),float(status=='source_conflict'),
         float(np.mean(local==-1)),float(np.mean(local==1)),float(np.mean(local==0)),
         belief.get('recent_receipts',0.),min(1.,belief.get('receipt_age',60.)/60.),belief.get('speed',0.)]
    v.extend(corridor_fractions(grid, p, [q])[0] if corridor is None else corridor)
    result = np.asarray(v,np.float32)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError('Invalid belief features')
    return result, status


class ReturnModel:
    """Ridge regression on measured closed-loop returns for candidate subgoals."""

    def __init__(self, regularization=10.):
        self.regularization = float(regularization)

    def fit(self, x, y):
        x,y = np.asarray(x,float),np.asarray(y,float)
        if x.ndim!=2 or len(x)!=len(y) or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Finite nonempty training data required')
        self.mean=x.mean(0)
        self.scale=np.maximum(x.std(0),1e-3)
        z=np.c_[(x-self.mean)/self.scale,np.ones(len(x))]
        penalty=np.eye(z.shape[1])*self.regularization
        penalty[-1,-1]=0.
        self.weights=np.linalg.solve(z.T@z+penalty,z.T@y)
        return self

    def predict(self, x):
        z=(np.asarray(x,float)-self.mean)/self.scale
        return np.c_[np.atleast_2d(z),np.ones(len(np.atleast_2d(z)))]@self.weights

    def save(self, path):
        np.savez(path,mean=self.mean,scale=self.scale,weights=self.weights,
                 regularization=self.regularization)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            model=cls(float(d['regularization']))
            model.mean=d['mean']; model.scale=d['scale']; model.weights=d['weights']
        return model


def select(predictions, statuses):
    permitted=np.array([s not in ('blocked_within_bounds','outside_bounds') for s in statuses])
    if not permitted.any():
        return None
    predictions=np.asarray(predictions,float)
    if len(predictions)!=len(permitted) or not np.isfinite(predictions).all():
        raise ValueError('Finite value required for every candidate')
    return int(np.argmax(np.where(permitted,predictions,-np.inf)))
