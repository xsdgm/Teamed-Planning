"""Observed local-map memory and continuous escape geometry.

Route memory and escape checks use only received local observations; unknown
space remains blocked and complete occupied pixel areas constrain each step.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class ObservedGrid:
    def __init__(self, goal, resolution=.1, half_width=40., world_bounds=None, center=None):
        self.goal = np.asarray(goal, float)
        self.resolution = float(resolution)
        center = self.goal if center is None else np.asarray(center, float)
        self.origin = np.floor((center-half_width)/resolution)*resolution
        size = int(np.ceil(2*half_width/resolution))
        self.cells = np.full((size, size), -1, dtype=np.int8)
        self.visits = np.zeros_like(self.cells, dtype=np.float32)
        self.world_bounds = world_bounds
        self.last_visit = None
        self.frontier = None
        self.path = []

    def recenter(self, position):
        """Shift by whole world cells, preserving overlapping observations."""
        position=np.asarray(position,float)
        extent=np.array(self.cells.shape[::-1])*self.resolution
        if np.all(np.abs(position-(self.origin+extent/2)) <= extent/4):
            return
        origin=np.floor((position-extent/2)/self.resolution)*self.resolution
        shift=np.rint((origin-self.origin)/self.resolution).astype(int)
        height,width=self.cells.shape
        sx0,sy0=max(0,shift[0]),max(0,shift[1])
        sx1,sy1=min(width,width+shift[0]),min(height,height+shift[1])
        cells=np.full_like(self.cells,-1);visits=np.zeros_like(self.visits)
        if sx1>sx0 and sy1>sy0:
            target=(slice(sy0-shift[1],sy1-shift[1]),slice(sx0-shift[0],sx1-shift[0]))
            source=(slice(sy0,sy1),slice(sx0,sx1))
            cells[target]=self.cells[source];visits[target]=self.visits[source]
        self.cells,self.visits,self.origin=cells,visits,origin
        self.path=[]

    def _cell(self, position):
        xy = np.floor((np.asarray(position)-self.origin)/self.resolution).astype(int)
        return int(xy[1]), int(xy[0])

    def observe(self, position, semantic_map, resolution):
        """Require full cell coverage; any overlapping occupied pixel blocks it.

        Moving sensor origins need area overlap, not nearest-pixel resampling.
        This memory is for static terrain. A completely free observation
        certifies the whole cell. A later occupied sensor pixel that merely
        overlaps it does not revoke that certificate: the pixel can contain an
        obstacle outside this cell. Live local sensing still checks execution.
        """
        position = np.asarray(position, float)
        h, w = semantic_map.shape
        lo = position - np.array([w, h])*resolution/2
        hi = lo + np.array([w, h])*resolution
        first = np.maximum(0, np.ceil((lo-self.origin)/self.resolution-1e-8).astype(int))
        last = np.minimum(self.cells.shape[::-1],
                          np.floor((hi-self.origin)/self.resolution+1e-8).astype(int))
        if np.any(last <= first):
            return
        xs, ys = np.arange(first[0], last[0]), np.arange(first[1], last[1])
        xlo = (self.origin[0]+xs*self.resolution-lo[0])/resolution
        ylo = (self.origin[1]+ys*self.resolution-lo[1])/resolution
        x0 = np.clip(np.floor(xlo+1e-8).astype(int), 0, w)
        y0 = np.clip(np.floor(ylo+1e-8).astype(int), 0, h)
        x1 = np.clip(np.ceil(xlo+self.resolution/resolution-1e-8).astype(int), 0, w)
        y1 = np.clip(np.ceil(ylo+self.resolution/resolution-1e-8).astype(int), 0, h)

        def counts(mask):
            integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            return (integral[np.ix_(y1, x1)]-integral[np.ix_(y0, x1)]
                    -integral[np.ix_(y1, x0)]+integral[np.ix_(y0, x0)])

        occupied = counts(np.isin(semantic_map, [2, 4])) > 0
        free = counts(~np.isin(semantic_map, [1, 3, 5])) == 0
        patch = self.cells[first[1]:last[1], first[0]:last[0]]
        patch[free] = 0
        patch[occupied & (patch != 0)] = 1
        if self.world_bounds is not None:
            bx0, by0, bx1, by1 = self.world_bounds
            outside_x = ((self.origin[0]+xs*self.resolution < bx0) |
                         (self.origin[0]+(xs+1)*self.resolution > bx1))
            outside_y = ((self.origin[1]+ys*self.resolution < by0) |
                         (self.origin[1]+(ys+1)*self.resolution > by1))
            patch[outside_y[:, None] | outside_x[None, :]] = 1
        if self.last_visit is None or np.linalg.norm(position-self.last_visit) >= .5:
            y, x = self._cell(position)
            radius = int(np.ceil(1./self.resolution))
            self.visits[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] += 1
            self.last_visit = position.copy()

    def waypoint(self, position, clearance):
        """Route through observed free space, or approach an unexplored boundary.

        Unlike choosing the reachable cell closest to the goal, a frontier may
        initially move away from the goal to get around an occluding obstacle.
        The returned short segment must still pass the live local controller.
        """
        self.path = []
        position = np.asarray(position, float)
        known_y, known_x = np.nonzero(self.cells >= 0)
        if not len(known_y):
            return None
        y0, x0 = max(0, known_y.min()-2), max(0, known_x.min()-2)
        y1, x1 = min(len(self.cells), known_y.max()+3), min(len(self.cells), known_x.max()+3)
        cells = self.cells[y0:y1, x0:x1]
        distances = distance_transform_edt(np.pad(cells == 0, 1))[1:-1, 1:-1]*self.resolution
        sy, sx = self._cell(position)
        sy -= y0; sx -= x0
        if not (0 <= sy < len(cells) and 0 <= sx < cells.shape[1]) or cells[sy, sx] != 0:
            return None
        # EDT measures occupied pixel centres; reserve their half diagonal.
        # Live local sensing checks execution again at the actual pose.
        cell_radius = self.resolution/np.sqrt(2)
        threshold = clearance + cell_radius
        if distances[sy, sx] <= threshold:
            # A vehicle can already be within an inflated margin when recovery
            # starts. Escape without reducing its measured clearance, keeping
            # at least the 0.3m body and cell-area discretization allowance.
            if distances[sy, sx] < .3 + cell_radius:
                return None
            threshold = distances[sy, sx] - 1e-8
        safe = distances > threshold
        ys, xs = np.nonzero(safe)
        count = len(ys)
        ids = np.full(safe.shape, -1, dtype=np.int32)
        ids[ys, xs] = np.arange(count)
        start = ids[sy, sx]
        visits = self.visits[y0:y1, x0:x1]
        sources, targets, weights = [], [], []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            ny, nx = ys+dy, xs+dx
            inside = (ny < len(safe)) & (nx >= 0) & (nx < safe.shape[1])
            a_y, a_x, b_y, b_x = ys[inside], xs[inside], ny[inside], nx[inside]
            valid = safe[b_y, b_x]
            if dy and dx:
                valid &= safe[a_y, b_x] & safe[b_y, a_x]
            a, b = ids[a_y[valid], a_x[valid]], ids[b_y[valid], b_x[valid]]
            cost = np.hypot(dx, dy)*self.resolution*(1+.03*np.minimum(
                visits[a_y[valid], a_x[valid]]+visits[b_y[valid], b_x[valid]], 50))
            sources.extend([a, b]); targets.extend([b, a]); weights.extend([cost, cost])
        graph = csr_matrix((np.concatenate(weights), (np.concatenate(sources),
                           np.concatenate(targets))), shape=(count, count))
        costs, parents = dijkstra(graph, indices=start, return_predecessors=True)
        points = self.origin + np.column_stack((xs+x0+.5, ys+y0+.5))*self.resolution
        gy, gx = self._cell(self.goal)
        gy -= y0; gx -= x0
        destination = ids[gy, gx] if 0 <= gy < len(safe) and 0 <= gx < safe.shape[1] else -1
        if destination < 0 or not np.isfinite(costs[destination]):
            unknown_distance = distance_transform_edt(cells >= 0)*self.resolution
            candidates = np.isfinite(costs) & (unknown_distance[ys, xs] <= clearance+1.)
            candidates &= np.linalg.norm(points-position, axis=1) > 1.
            if not candidates.any():
                self.frontier = None
                return None
            destination = -1
            if self.frontier is not None:
                fy, fx = self._cell(self.frontier)
                fy -= y0; fx -= x0
                if 0 <= fy < len(safe) and 0 <= fx < safe.shape[1]:
                    old = ids[fy, fx]
                    if old >= 0 and candidates[old]:
                        destination = old
            if destination < 0:
                score = costs + np.linalg.norm(points-self.goal, axis=1) + 3*visits[ys, xs]
                destination = np.argmin(np.where(candidates, score, np.inf))
                self.frontier = points[destination].copy()
        else:
            self.frontier = None
        route = []
        cursor = int(destination)
        while cursor != start:
            route.append(cursor)
            cursor = int(parents[cursor])
            if cursor < 0:
                return None
        route.reverse()
        self.path = [tuple(p) for p in points[route]]
        if not route:
            return tuple(self.goal) if np.linalg.norm(position-self.goal) < .2 else None
        waypoint = None
        for p in points[route]:
            distance = np.linalg.norm(p-position)
            if distance > 2.:
                break
            samples = np.linspace(position, p, max(2, int(distance/self.resolution*4)+1))
            ij = np.floor((samples-self.origin)/self.resolution).astype(int)
            if not safe[ij[:, 1]-y0, ij[:, 0]-x0].all():
                break
            if distance >= .1:
                waypoint = tuple(p)
        return waypoint


def segment_clearance(a, b, lower, upper):
    """Minimum distance from a closed segment to a union of rectangles."""
    if not len(lower):
        return float('inf')
    delta = b-a
    tmin, tmax = np.zeros(len(lower)), np.ones(len(lower))
    intersects = np.ones(len(lower), dtype=bool)
    for axis in (0, 1):
        if abs(delta[axis]) < 1e-15:
            intersects &= (a[axis] >= lower[:, axis]) & (a[axis] <= upper[:, axis])
        else:
            first = (lower[:, axis]-a[axis])/delta[axis]
            last = (upper[:, axis]-a[axis])/delta[axis]
            tmin = np.maximum(tmin, np.minimum(first, last))
            tmax = np.minimum(tmax, np.maximum(first, last))
    if np.any(intersects & (tmin <= tmax)):
        return 0.
    squared = np.minimum(
        np.sum(np.maximum(np.maximum(lower-a, a-upper), 0.)**2, axis=1),
        np.sum(np.maximum(np.maximum(lower-b, b-upper), 0.)**2, axis=1))
    length2 = float(delta @ delta)
    if length2:
        for corner in (lower, upper, np.column_stack((lower[:, 0], upper[:, 1])),
                       np.column_stack((upper[:, 0], lower[:, 1]))):
            t = np.clip((corner-a) @ delta / length2, 0., 1.)
            squared = np.minimum(squared, np.sum((corner-a-t[:, None]*delta)**2, axis=1))
    return float(np.sqrt(squared.min()))


def escape_waypoint(position, target, semantic, resolution, clearance=.5, body_radius=.3):
    """Exit an inflated pocket without reducing the current observed clearance.

    Treat complete occupied/unknown pixel squares as blocked. A point already
    inside the requested margin may leave only along a segment whose clearance
    never drops below its current value, and must improve it. The body and the
    sensor boundary remain hard constraints. No world geometry is consulted.
    """
    h, w = semantic.shape
    zero = np.zeros(2)
    extent = np.array([w, h])*resolution/2.
    rows, cols = np.nonzero(~np.isin(semantic, [1, 3, 5]))
    lower = np.column_stack((cols-w/2., rows-h/2.))*resolution
    upper = lower+resolution
    # Only pixels within the maximum 2m step plus its clearance matter.
    keep = np.linalg.norm(np.maximum(np.maximum(lower, -upper), 0.), axis=1) <= 2.+clearance
    lower, upper = lower[keep], upper[keep]
    current = min(segment_clearance(zero, zero, lower, upper), float(extent.min()))
    if current <= body_radius+1e-6:
        return None
    required = min(clearance, current)
    delta = np.asarray(target)-position
    bearing = np.arctan2(delta[1], delta[0])
    candidates = []
    for angle in bearing+np.arange(32)*np.pi/16:
        direction = np.array([np.cos(angle), np.sin(angle)])
        # A narrow raster pocket may permit a turn shorter than 0.25m.
        # Stay above the velocity controller's 0.1m arrival tolerance.
        for length in (.125, .1875, .25, .5, 1., 2.):
            offset = length*direction
            if np.any(np.abs(offset)+required >= extent):
                continue
            swept = segment_clearance(zero, offset, lower, upper)
            if swept+1e-9 < required:
                continue
            end_clearance = segment_clearance(offset, offset, lower, upper)
            if current < clearance and end_clearance < current+.02:
                continue
            # The constraints above enforce safe, improving clearance. Among
            # those exits follow the recovery route, rather than repeatedly
            # reversing toward the largest pixel-dependent clearance gain.
            score = (-float(np.linalg.norm(delta-offset)), min(end_clearance, clearance))
            candidates.append((score, offset))
    if not candidates:
        return None
    offset = max(candidates, key=lambda item: item[0])[1]
    return tuple(np.asarray(position)+offset)
