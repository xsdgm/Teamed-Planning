"""vtmp2: revisable subgoals within the fleet's communication-limited survey range.

The default selector is a geometric heuristic, not a trained world-model policy.
``recon`` permits explicitly unverified exploration targets, while ``observed``
and ``local`` require a known route. Local sensing validates execution and does
not define the fleet's reconnaissance radius. No simulator truth is consumed.
"""
from dataclasses import dataclass
import heapq
import math

import networkx as nx
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree


@dataclass
class _Candidate:
    point: np.ndarray
    source: str
    status: str
    cost: float


class Vtmp2SubgoalManager:
    algorithm = 'vtmp2'

    def __init__(self, goal, radius=200., arrival=1., world_bounds=None,
                 visibility_mode='recon', recon_radius=None, safety_distance=1.,
                 stall=45., max_wait=120., min_hold=3., review_interval=5.,
                 switch_improvement=2., failure_ttl=90., failure_radius=5.,
                 directions=24):
        self.goal = self._point(goal)
        numbers = (radius, arrival, safety_distance, stall, max_wait, min_hold,
                   review_interval, switch_improvement, failure_ttl, failure_radius)
        if (not np.isfinite(numbers).all() or radius <= 0 or not 0 < arrival <= 1.
                or safety_distance <= 0 or not 0 < stall <= max_wait or min_hold < 0
                or review_interval <= 0 or switch_improvement < 0
                or failure_ttl <= 0 or failure_radius <= 0):
            raise ValueError('Invalid vtmp2 distances or timing options')
        if visibility_mode not in ('recon', 'observed', 'local'):
            raise ValueError('visibility_mode must be recon, observed or local')
        if type(directions) is not int or directions < 4:
            raise ValueError('directions must be an integer >= 4')
        self.radius, self.arrival = float(radius), float(arrival)
        self.safety_distance = float(safety_distance)
        self.visibility_mode, self.directions = visibility_mode, directions
        self.recon_radius = self.radius
        self.set_reconnaissance_range(self.radius if recon_radius is None else recon_radius)
        self.world_bounds = None if world_bounds is None else np.asarray(world_bounds, float)
        if self.world_bounds is not None and (self.world_bounds.shape != (4,)
                or not np.isfinite(self.world_bounds).all()
                or np.any(self.world_bounds[2:] <= self.world_bounds[:2])):
            raise ValueError('world_bounds must be finite xmin,ymin,xmax,ymax')
        self.stall, self.max_wait = float(stall), float(max_wait)
        self.min_hold, self.review_interval = float(min_hold), float(review_interval)
        self.switch_improvement = float(switch_improvement)
        self.failure_ttl, self.failure_radius = float(failure_ttl), float(failure_radius)
        self.target = None
        self.final = False
        self.selected = self.arrivals = self.failed = self.rejected_requests = 0
        self.last_reason, self.last_source, self.target_status = 'initial', 'hold', 'hold'
        self.failures = []
        self._observation = None
        self._obstacles = set()
        self._obstacle_tree = None
        self._obstacle_margin = self.safety_distance
        self._requested = self._forced_failure = None
        self._manual = False
        self._chosen_at = self._reviewed_at = self._progress_at = self._evidence_at = 0.
        self._best = float('inf')
        self._evidence_count = 0

    @staticmethod
    def _point(value):
        point = np.asarray(value, float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError('A point must contain two finite coordinates')
        return point.copy()

    @property
    def effective_radius(self):
        return min(self.radius, self.recon_radius)

    def set_reconnaissance_range(self, radius):
        """Update the survey radius supplied by the communication-chain model."""
        if not np.isfinite(radius) or radius < 0:
            raise ValueError('Reconnaissance radius must be finite and nonnegative')
        self.recon_radius = float(radius)

    def set_observation(self, position, semantic_map, resolution, blocked=None):
        """Supply the current vehicle observation; semantic 1/3/5 means free.

        ``blocked`` may supply the controller's already inflated safety mask.
        Unknown cells are never accepted as observed free. Occupied cells are
        retained in world coordinates, as the navigation task has static terrain.
        """
        center = self._point(position)
        semantic = np.asarray(semantic_map)
        if semantic.ndim != 2 or min(semantic.shape) < 1:
            raise ValueError('semantic_map must be a nonempty two-dimensional array')
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError('resolution must be positive and finite')
        resolution = float(resolution)
        free = np.isin(semantic, (1, 3, 5))
        radius = max(1, int(math.ceil(self.safety_distance / resolution)))
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        footprint = xx * xx + yy * yy <= radius * radius
        obstacle_mask = binary_dilation(~free, structure=footprint, border_value=0)
        local_mask = binary_dilation(~free, structure=footprint, border_value=1)
        if blocked is not None:
            supplied = np.asarray(blocked, bool)
            if supplied.shape != semantic.shape:
                raise ValueError('blocked mask must have the semantic-map shape')
            local_mask |= supplied
            # The artificial sensor boundary is not contradictory evidence
            # against a received graph edge that continues beyond the window.
            interior = np.zeros(semantic.shape, bool)
            if min(semantic.shape) > 2 * radius:
                interior[radius:-radius, radius:-radius] = True
            obstacle_mask |= supplied & interior
        old_count = len(self._obstacles)
        rows, columns = np.nonzero(np.isin(semantic, (2, 4)))
        height, width = semantic.shape
        for y, x in zip(rows, columns):
            point = center + np.array([x + .5 - width / 2., y + .5 - height / 2.]) * resolution
            self._obstacles.add(tuple(np.round(point, 6)))
        if len(self._obstacles) != old_count:
            self._obstacle_tree = cKDTree(np.asarray(sorted(self._obstacles)))
        self._obstacle_margin = max(self._obstacle_margin,
                                    self.safety_distance + resolution / math.sqrt(2.))
        self._observation = (center, resolution, local_mask, obstacle_mask)

    def request_target(self, target):
        """Queue a manual target; the next update validates it before adoption.

        An unsafe/out-of-range request is rejected without disturbing a valid
        current target. ``rejected_requests`` and ``last_reason`` expose this.
        """
        self._requested = self._point(target)

    def invalidate_target(self, reason='external_failure'):
        """Request immediate reconsideration, e.g. after execution failure."""
        self._forced_failure = str(reason)

    def _cell(self, point):
        center, resolution, blocked, _ = self._observation
        height, width = blocked.shape
        x, y = np.floor((point - center) / resolution + [width / 2., height / 2.]).astype(int)
        return (int(y), int(x)) if 0 <= x < width and 0 <= y < height else None

    def _cell_point(self, cell):
        center, resolution, blocked, _ = self._observation
        y, x = cell
        return center + np.array([x + .5 - blocked.shape[1] / 2.,
                                  y + .5 - blocked.shape[0] / 2.]) * resolution

    def _within(self, point, position):
        if np.linalg.norm(point - position) > self.effective_radius + 1e-8:
            return False
        if self.world_bounds is not None:
            if np.any(point < self.world_bounds[:2]) or np.any(point > self.world_bounds[2:]):
                return False
        return True

    def _known_blocked(self, points):
        points = np.atleast_2d(points)
        if self._obstacle_tree is not None:
            if np.any(self._obstacle_tree.query(points)[0] <= self._obstacle_margin):
                return True
        center, resolution, _, blocked = self._observation
        height, width = blocked.shape
        cells = np.floor((points-center)/resolution+[width/2., height/2.]).astype(int)
        inside = ((cells[:, 0] >= 0) & (cells[:, 0] < width)
                  & (cells[:, 1] >= 0) & (cells[:, 1] < height))
        return bool(np.any(blocked[cells[inside, 1], cells[inside, 0]]))

    def _segment_clear(self, first, second):
        resolution = self._observation[1]
        count = max(2, int(math.ceil(np.linalg.norm(second - first) / (resolution / 3.))) + 1)
        return not self._known_blocked(np.linspace(first, second, count))

    def _local_routes(self, position):
        if self._observation is None:
            return {}
        cell = self._cell(position)
        blocked = self._observation[2]
        if cell is None or blocked[cell] or self._known_blocked(position):
            return {}
        costs = {cell: float(np.linalg.norm(self._cell_point(cell) - position))}
        queue = [(costs[cell], cell)]
        height, width = blocked.shape
        resolution = self._observation[1]
        while queue:
            cost, (y, x) = heapq.heappop(queue)
            if cost > costs[y, x]:
                continue
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                yy, xx = y + dy, x + dx
                if not (0 <= yy < height and 0 <= xx < width) or blocked[yy, xx]:
                    continue
                if dy and dx and (blocked[y, xx] or blocked[yy, x]):
                    continue
                step = resolution * math.hypot(dx, dy)
                if cost + step < costs.get((yy, xx), float('inf')):
                    costs[yy, xx] = cost + step
                    heapq.heappush(queue, (cost + step, (yy, xx)))
        return costs

    @staticmethod
    def _trusted(attributes):
        # Historical graph edges carry only weight. Honor an explicit
        # withdrawal without inventing an absent confidence measurement.
        try:
            confidence = float(attributes.get('confidence', 1.))
        except (TypeError, ValueError):
            return False
        return math.isfinite(confidence) and confidence > 0.

    def _context(self, position, graph):
        local = self._local_routes(position)
        network = graph if isinstance(graph, nx.Graph) else getattr(graph, 'graph', None)
        evidence = 0 if not isinstance(network, nx.Graph) else len(network) + network.number_of_edges()
        positions, distances = {}, {}
        if local and self.visibility_mode != 'local' and isinstance(network, nx.Graph):
            safe = nx.Graph()
            for node, attributes in network.nodes(data=True):
                if not self._trusted(attributes):
                    continue
                point = np.asarray(attributes.get('position', ()), float)
                if point.ndim != 1 or point.size < 2:
                    continue
                point = point[:2]
                if not np.isfinite(point).all() or not self._within(point, position):
                    continue
                if self._known_blocked(point):
                    continue
                positions[node] = point
                safe.add_node(node)
            for a, b, attributes in network.edges(data=True):
                if (self._trusted(attributes) and a in positions and b in positions
                        and self._segment_clear(positions[a], positions[b])):
                    safe.add_edge(a, b, weight=float(np.linalg.norm(positions[a] - positions[b])))
            source = object()
            for node, point in positions.items():
                cell = self._cell(point)
                if cell in local and self._segment_clear(self._cell_point(cell), point):
                    safe.add_edge(source, node, weight=local[cell] + float(np.linalg.norm(point-self._cell_point(cell))))
            if source in safe:
                distances = nx.single_source_dijkstra_path_length(safe, source)
                distances.pop(source)
        return dict(local=local, positions=positions, distances=distances, evidence=evidence)

    def _proof(self, point, position, context):
        if not self._within(point, position) or self._observation is None or self._known_blocked(point):
            return None
        cell = self._cell(point)
        if cell in context['local']:
            return _Candidate(point.copy(), 'local', 'verified', context['local'][cell])
        for node, distance in context['distances'].items():
            if np.linalg.norm(context['positions'][node] - point) < 1e-6:
                return _Candidate(point.copy(), 'graph', 'verified', distance)
        if self.visibility_mode != 'recon' or cell is not None or not context['local']:
            return None
        # Only an initial safe direction is certified; the distant endpoint
        # remains unverified even if it is inside communication/survey range.
        delta = point - position
        distance = float(np.linalg.norm(delta))
        if distance <= self.arrival or not self._segment_clear(position, point):
            return None
        step = position + delta / distance * min(distance, 1.)
        if self._cell(step) not in context['local']:
            return None
        return _Candidate(point.copy(), 'recon_frontier', 'unverified', distance)

    def _suppressed(self, point, now):
        return any(now < until and np.linalg.norm(point - failed) <= self.failure_radius
                   for failed, until in self.failures)

    def _score(self, candidate):
        return float(np.linalg.norm(candidate.point - self.goal) + .15 * candidate.cost)

    def _candidates(self, position, context, now):
        candidates = []
        if not context['local']:
            return candidates
        for cell, cost in context['local'].items():
            point = self._cell_point(cell)
            if self._within(point, position) and not self._known_blocked(point):
                candidates.append(_Candidate(point, 'local', 'verified', cost))
        for node, cost in context['distances'].items():
            candidates.append(_Candidate(context['positions'][node].copy(), 'graph', 'verified', cost))
        final = self._proof(self.goal, position, context)
        if final is not None:
            candidates.append(final)
        if self.visibility_mode == 'recon':
            heading = math.atan2(*(self.goal - position)[::-1])
            angles = heading + np.arange(self.directions) * (2 * np.pi / self.directions)
            for fraction in (1., 2. / 3., 1. / 3.):
                for angle in angles:
                    point = position + self.effective_radius * fraction * np.array([math.cos(angle), math.sin(angle)])
                    proof = self._proof(point, position, context)
                    if proof is not None:
                        candidates.append(proof)
        return [c for c in candidates if (np.linalg.norm(c.point - position) > self.arrival + 1e-6
                or (c.status == 'verified' and np.linalg.norm(c.point-self.goal) < 1e-6))
                and not self._suppressed(c.point, now)]

    def _choose(self, position, context, now):
        candidates = self._candidates(position, context, now)
        if not candidates:
            return None
        reached = [c for c in candidates if c.status == 'verified'
                   and np.linalg.norm(c.point-self.goal) < 1e-6
                   and np.linalg.norm(c.point-position) <= self.arrival]
        if reached:
            return reached[0]
        # Prefer useful observed graph routes; otherwise ask the fleet to
        # survey a safe departure direction before falling back to local steps.
        useful = [c for c in candidates if c.status == 'verified'
                  and (c.source == 'graph' or np.linalg.norm(c.point-self.goal) < 1e-6)
                  and np.linalg.norm(c.point-self.goal) < np.linalg.norm(position-self.goal)-self.arrival]
        if useful:
            candidates = useful
        elif self.visibility_mode == 'recon':
            exploration = [c for c in candidates if c.status == 'unverified']
            if exploration:
                candidates = exploration
        return min(candidates, key=lambda c: (self._score(c), c.cost, *c.point))

    def _adopt(self, candidate, position, context, now, reason, manual=False):
        previous = self.target
        self.target = position.copy() if candidate is None else candidate.point.copy()
        self.last_source = 'hold' if candidate is None else candidate.source
        self.target_status = 'hold' if candidate is None else candidate.status
        self.final = (candidate is not None and candidate.status == 'verified'
                      and np.linalg.norm(self.target-self.goal) < 1e-6)
        changed = previous is None or not np.allclose(self.target, previous, atol=1e-6, rtol=0.)
        if candidate is not None and changed:
            self.selected += 1
        self.last_reason = reason if candidate is not None else 'no_safe_candidate'
        self._chosen_at = self._reviewed_at = self._progress_at = self._evidence_at = now
        self._best = float(np.linalg.norm(self.target-position))
        self._evidence_count = context['evidence']
        self._manual = manual
        return self.target.copy(), changed

    def _record_failure(self, now):
        if self.target is not None and self.target_status != 'hold':
            self.failures.append((self.target.copy(), now+self.failure_ttl))
            self.failed += 1

    def update(self, position, graph, now):
        """Return (target, changed), checking safety before any hold/hysteresis."""
        position = self._point(position)
        now = float(now)
        if not np.isfinite(now):
            raise ValueError('now must be finite')
        self.failures = [(point, until) for point, until in self.failures if now < until]
        context = self._context(position, graph)
        rejected = False
        if self._requested is not None:
            requested, self._requested = self._requested, None
            candidate = self._proof(requested, position, context)
            if candidate is not None and not self._suppressed(requested, now):
                return self._adopt(candidate, position, context, now, 'manual_request', manual=True)
            self.rejected_requests += 1
            rejected = True
        if self.target is None or self.target_status == 'hold':
            self._forced_failure = None
            return self._adopt(self._choose(position, context, now), position, context, now, 'initial')
        current = self._proof(self.target, position, context)
        reason, self._forced_failure = self._forced_failure, None
        withdrawn = (self.target_status == 'verified' and current is not None
                     and current.status != 'verified')
        if current is None or withdrawn or reason is not None:
            self._record_failure(now)
            return self._adopt(self._choose(position, context, now), position, context, now,
                               reason or 'target_invalid')
        self.last_source, self.target_status = current.source, current.status
        self.final = current.status == 'verified' and np.linalg.norm(self.target-self.goal) < 1e-6
        distance = float(np.linalg.norm(self.target-position))
        if distance <= self.arrival:
            if self.final:
                self.last_reason = 'final_reached'
                return self.target.copy(), False
            self.arrivals += 1
            return self._adopt(self._choose(position, context, now), position, context, now, 'arrived')
        if distance < self._best-.25:
            self._best, self._progress_at, self._evidence_at = distance, now, now
        if context['evidence'] > self._evidence_count:
            self._evidence_at = now
        self._evidence_count = context['evidence']
        if now-self._progress_at >= self.max_wait or now-self._evidence_at >= self.stall:
            self._record_failure(now)
            return self._adopt(self._choose(position, context, now), position, context, now, 'stalled')
        if (not self._manual and now-self._chosen_at >= self.min_hold
                and now-self._reviewed_at >= self.review_interval):
            self._reviewed_at = now
            candidate = self._choose(position, context, now)
            if candidate is not None and (self._score(current)-self._score(candidate) > self.switch_improvement
                    or (current.status == 'unverified' and candidate.status == 'verified'
                        and candidate.source == 'graph')):
                return self._adopt(candidate, position, context, now, 'better_candidate')
        self.last_reason = 'manual_rejected' if rejected else 'retained'
        return self.target.copy(), False

    def stats(self):
        return dict(subgoals_selected=float(self.selected), subgoal_arrivals=float(self.arrivals),
                    subgoal_failures=float(self.failed), vtmp2_rejected_requests=float(self.rejected_requests),
                    vtmp2_unverified_target=float(self.target_status == 'unverified'))
