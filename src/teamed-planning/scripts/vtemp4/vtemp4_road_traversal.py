"""Associate measured vehicle passages with currently observed road edges."""
from array import array
from collections import defaultdict
import math

import numpy as np


TRAVERSAL_FEATURES = 4
_CELL_M = 8.
_MAX_OFFSET_M = 1.5


class RoadTraversalMemory:
    """Retain an indexed, chronological sequence of the vehicle's distinct 1 Hz positions."""

    def __init__(self):
        self._points = array('d')
        self._times = array('q')
        self._cells = defaultdict(list)
        self._last_tick = -1

    def _observe(self, now, position, trajectory):
        if now < self._last_tick or len(trajectory) < now:
            raise ValueError('Expected a contiguous measured vehicle trajectory through the review')
        for tick in range(self._last_tick + 1, now + 1):
            # At the review callback, the new pose is already measured; step()
            # may only append it to the trajectory after issuing the next action.
            x, y = position if tick == now else trajectory[tick]
            if self._points and x == self._points[-2] and y == self._points[-1]:
                continue
            cell = (math.floor(x / _CELL_M), math.floor(y / _CELL_M))
            self._cells[cell].append(len(self._times))
            self._points.extend((x, y))
            self._times.append(tick)
        self._last_tick = now

    def _passages(self, a, b, points):
        delta = b - a
        length = np.linalg.norm(delta)
        if length < 1e-4:
            return []
        # Keep the endpoint neighborhoods disjoint, even on short roads.
        tolerance = min(_MAX_OFFSET_M, .2 * length)
        low = np.floor((np.minimum(a, b) - tolerance) / _CELL_M).astype(int)
        high = np.floor((np.maximum(a, b) + tolerance) / _CELL_M).astype(int)
        ids = []
        for x in range(low[0], high[0] + 1):
            for y in range(low[1], high[1] + 1):
                ids.extend(self._cells.get((x, y), ()))
        if len(ids) < 3:
            return []
        ids = np.asarray(sorted(ids))
        rel = points[ids] - a
        # Require a continuous stretch of measured poses inside the edge's
        # narrow corridor. Touching both ends via another road is insufficient.
        projection = np.clip((rel @ delta) / length**2, 0., 1.)
        offset = rel - projection[:, None] * delta
        inside = np.einsum('ij,ij->i', offset, offset) <= tolerance**2
        ids, rel = ids[inside], rel[inside]
        if len(ids) < 3:
            return []
        near_a = np.einsum('ij,ij->i', rel, rel) <= tolerance**2
        rel_b = rel - delta
        near_b = np.einsum('ij,ij->i', rel_b, rel_b) <= tolerance**2
        gates = np.flatnonzero(near_a | near_b)
        if len(gates) < 2:
            return []
        previous, current = gates[:-1], gates[1:]
        # At least one interior observation corroborates each passage. A gap
        # in the chronological ids means the vehicle left the corridor.
        observed = ((near_a[previous] != near_a[current]) & (current - previous >= 2)
                    & (ids[current] - ids[previous] == current - previous))
        return [(int(ids[i]), int(ids[j]), int(near_b[i])) for i, j in zip(previous[observed], current[observed])]

    def edge_features(self, now, position, trajectory, road_world_xy, adjacency, road_valid):
        """Return log counts (forward, reverse) and log ages (forward, reverse) in neighbor-slot order.

        Zero count denotes no confirmed passage in that direction. Overlapping
        matches to different measured edges are ambiguous and receive no credit.
        """
        self._observe(now, position, trajectory)
        points = np.frombuffer(self._points, dtype=np.float64).reshape(-1, 2)
        xy = np.asarray(road_world_xy, dtype=float)
        neighbors = np.asarray(adjacency, dtype=int)
        valid = np.asarray(road_valid, dtype=bool)
        output = np.zeros((*neighbors.shape, TRAVERSAL_FEATURES), dtype=np.float32)
        slots = defaultdict(list)
        for source, slot in zip(*np.nonzero(valid[:, None] & (neighbors >= 0))):
            target = neighbors[source, slot]
            if target != source and valid[target]:
                edge = min(source, target), max(source, target)
                slots[edge].append((source, slot))

        passages = []
        for edge in slots:
            a, b = edge
            for start, end, direction in self._passages(xy[a], xy[b], points):
                passages.append((start, end, edge, direction))
        passages.sort()
        i = 0
        history = {}
        while i < len(passages):
            _, end, edge, direction = passages[i]
            j = i + 1
            while j < len(passages) and passages[j][0] < end:
                end = max(end, passages[j][1])
                j += 1
            if j == i + 1:
                counts, last_seen = history.setdefault(edge, ([0, 0], [now, now]))
                counts[direction] += 1
                last_seen[direction] = self._times[end]
            i = j
        for edge, (counts, last_seen) in history.items():
            forward, reverse = np.log1p(counts)
            forward_age, reverse_age = np.log1p(np.subtract(now, last_seen))
            for source, slot in slots[edge]:
                if source == edge[0]:
                    output[source, slot] = forward, reverse, forward_age, reverse_age
                else:
                    output[source, slot] = reverse, forward, reverse_age, forward_age
        return output
