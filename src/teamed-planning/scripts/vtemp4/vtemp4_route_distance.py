"""Training-only remaining route length on the original V9 circle geometry."""
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from survey.survey_pretrain_maps import CLEARANCE_M
from vtemp2_3.scene_manifest import load_v9_circles


GRID_RESOLUTION_M = 1.


def _circle_pairs(case):
    if 'circles_path' in case:
        return [(row[:2], row[2]) for row in load_v9_circles(case)]
    pairs = []
    for row in case['circles']:
        xy, radius = (row[:2], row[2]) if len(row) == 3 else row
        pairs.append((np.asarray(xy, dtype=float), float(radius)))
    return pairs


def _grid_distances(free, resolution, goal_ids, goal_costs):
    """Undirected metric lattice plus one vertex for the subcell goal position."""
    height, width = free.shape
    ids = np.arange(free.size, dtype=np.int32).reshape(height, width)
    rows, cols, costs = [], [], []

    def add_edges(start, end, mask, length):
        sources = start[mask]
        rows.append(sources)
        cols.append(end[mask])
        costs.append(np.full(sources.size, length, dtype=np.float64))

    add_edges(ids[:, :-1], ids[:, 1:], free[:, :-1] & free[:, 1:], resolution)
    add_edges(ids[:-1], ids[1:], free[:-1] & free[1:], resolution)
    # Both flanking cells must also be free before crossing a grid corner.
    square = free[:-1, :-1] & free[:-1, 1:] & free[1:, :-1] & free[1:, 1:]
    diagonal = math.sqrt(2.) * resolution
    add_edges(ids[:-1, :-1], ids[1:, 1:], square, diagonal)
    add_edges(ids[:-1, 1:], ids[1:, :-1], square, diagonal)

    goal_id = free.size
    rows.append(np.full(len(goal_ids), goal_id, dtype=np.int32))
    cols.append(goal_ids)
    costs.append(goal_costs)
    graph = coo_matrix((np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(goal_id + 1, goal_id + 1)).tocsr()
    del rows, cols, costs, ids
    values = dijkstra(graph, directed=False, indices=goal_id)
    return values[:goal_id].reshape(free.shape)


class RouteDistanceField:
    """Fixed shortest-path potential for one episode. Coordinates and costs use metres.

    Cell centres and their joining edges have certified ground clearance. A continuous
    pose joins the lattice only on a physically clear straight segment. Missing access
    to the goal component is represented by None.
    """

    def __init__(self, case, *, resolution_m=GRID_RESOLUTION_M, clearance_m=CLEARANCE_M):
        # Resolve these helpers after the frozen V9 runtime has installed its modules.
        from shared.simulator.large_world_scenarios import CircleGeometry, raster_free

        circles = _circle_pairs(case)
        self.goal = np.asarray(case['goal'], dtype=float)
        self.resolution = float(resolution_m)
        world = float(case['world_size'])
        if not (math.isfinite(world) and 0 < self.resolution <= world
                and math.isfinite(clearance_m) and clearance_m >= 0):
            raise ValueError('Invalid route lattice size or vehicle clearance')
        self.geometry = CircleGeometry(circles, world=world, clearance=clearance_m)
        if (self.goal.shape != (2,) or not np.isfinite(self.goal).all()
                or not self.geometry.clear(self.goal, self.goal)):
            raise ValueError('Route goal lacks vehicle clearance')
        self.empty_world = not circles
        if self.empty_world:
            return

        # A circumscribed cell disk certifies the entire square, including its edges.
        free = raster_free(circles, world, clearance_m + self.resolution / math.sqrt(2.), self.resolution)
        self.shape = free.shape
        goal_ids, goal_costs = self._access(self.goal, free)
        self.distances = _grid_distances(free, self.resolution, goal_ids, goal_costs)
        self.reachable = np.isfinite(self.distances)

    def _anchors(self, point, usable, reach):
        # The surrounding cell centres include all four interpolation neighbours.
        x, y = np.floor(point / self.resolution).astype(int)
        height, width = self.shape
        x0, y0 = max(0, x - reach), max(0, y - reach)
        x1, y1 = min(width, x + reach + 1), min(height, y + reach + 1)
        rows, cols = np.nonzero(usable[y0:y1, x0:x1])
        rows, cols = rows + y0, cols + x0
        centres = (np.column_stack((cols, rows)) + .5) * self.resolution
        lengths = np.linalg.norm(centres - point, axis=1)
        return rows * width + cols, centres, lengths

    def _access(self, point, usable):
        for reach in (1, 4):
            ids, centres, lengths = self._anchors(point, usable, reach)
            clear = np.fromiter((self.geometry.clear(point, p) for p in centres), dtype=bool, count=len(ids))
            if clear.any():
                return ids[clear], lengths[clear]
        return np.empty(0, dtype=np.int32), np.empty(0)

    def distance_m(self, position):
        """Remaining distance or None for insufficient clearance or disconnected poses."""
        point = np.asarray(position, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError('Expected a finite vehicle position in XY metres')
        if not self.geometry.clear(point, point):
            return None
        direct = float(np.linalg.norm(point - self.goal))
        if self.empty_world:
            return direct
        if direct <= 2 * self.resolution and self.geometry.clear(point, self.goal):
            return direct
        for reach in (1, 4):
            ids, centres, lengths = self._anchors(point, self.reachable, reach)
            costs = self.distances.ravel()[ids] + lengths
            for index in np.argsort(costs):
                if self.geometry.clear(point, centres[index]):
                    return float(costs[index])
        return None

