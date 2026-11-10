"""Shortest paths on a graph made exclusively from received free evidence."""
import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt


class ReceivedMap:
    def __init__(self, config):
        self.config = config
        self.n = round(config.world/config.resolution)
        self.cells = np.full((self.n, self.n), -1, dtype=np.int8)
        self._received = set()

    def ingest(self, packet):
        ids = np.asarray(packet.cell_ids)
        occupied = np.asarray(packet.occupied)
        if ids.ndim != 1 or ids.shape != occupied.shape or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('Invalid delivered occupancy packet')
        if (ids < 0).any() or (ids >= self.n**2).any() or not np.isin(occupied, (0, 1)).all():
            raise ValueError('Delivered occupancy is outside the map contract')
        key = (packet.backend, packet.task_id)
        if packet.task_id >= 0 and key in self._received:
            return 0
        new = int(np.count_nonzero(self.cells.ravel()[np.unique(ids)] < 0))
        self.cells.ravel()[ids] = occupied.astype(np.int8)
        if packet.task_id >= 0:
            self._received.add(key)
        return new

    def safe_nodes(self):
        free = np.pad(self.cells == 0, 1, constant_values=False)
        clearance = distance_transform_edt(free)[1:-1, 1:-1]*self.config.resolution
        # Covers cell rectangles, between-node segments and endpoint access.
        return clearance > self.config.clearance+self.config.resolution*math.sqrt(2)

    def cell(self, point):
        xy = np.floor(np.asarray(point)/self.config.resolution).astype(int)
        return int(xy[1]), int(xy[0])

    def center(self, cell):
        y, x = cell
        return (np.array([x, y])+.5)*self.config.resolution

    def search(self, position, goal):
        """Dijkstra gives routes to the goal or the closest reachable frontier.

        Unknown cells have no graph nodes. There is no simulator/mapper handle
        here, and unsuccessful A/B surveys need not create connected routes.
        """
        safe = self.safe_nodes()
        source = self.cell(position)
        reachable = np.zeros_like(safe)
        if not (0 <= source[0] < self.n and 0 <= source[1] < self.n and safe[source]):
            return [np.asarray(position).copy()], reachable
        costs, previous, queue = {source: 0.}, {}, [(0., source)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != costs[node]:
                continue
            reachable[node] = True
            y, x = node
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nxt = y+dy, x+dx
                if not (0 <= nxt[0] < self.n and 0 <= nxt[1] < self.n and safe[nxt]):
                    continue
                if dx and dy and not (safe[y, x+dx] and safe[y+dy, x]):
                    continue
                candidate = cost+math.hypot(dy, dx)*self.config.resolution
                if candidate < costs.get(nxt, math.inf):
                    costs[nxt], previous[nxt] = candidate, node
                    heapq.heappush(queue, (candidate, nxt))
        goal_cell = self.cell(goal)
        if goal_cell in costs:
            target = goal_cell
        else:
            target = min(costs, key=lambda node: (float(np.linalg.norm(self.center(node)-goal)), costs[node]))
            if np.linalg.norm(self.center(target)-goal) >= np.linalg.norm(np.asarray(position)-goal):
                return [np.asarray(position).copy()], reachable
        path = [target]
        while path[-1] != source:
            path.append(previous[path[-1]])
        points = [np.asarray(position).copy(), *(self.center(node) for node in reversed(path))]
        if target == goal_cell:
            points.append(np.asarray(goal).copy())
        return points, reachable


def follow_path(points, budget_m):
    """A bounded kinematic graph follower; no instantaneous endpoint jumps."""
    result, traveled = [np.asarray(points[0]).copy()], 0.
    for point in points[1:]:
        delta = np.asarray(point)-result[-1]
        length = float(np.linalg.norm(delta))
        if length <= 1e-10:
            continue
        remaining = budget_m-traveled
        if length >= remaining:
            result.append(result[-1]+delta*(remaining/length))
            traveled = float(budget_m)
            break
        result.append(np.asarray(point).copy())
        traveled += length
    return result, traveled
