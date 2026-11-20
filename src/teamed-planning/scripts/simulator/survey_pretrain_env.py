"""Lightweight active survey simulation with a receiver-only, optimized ground plan.

Coordinates are in metres. Array rows follow increasing world Y and columns X.
The supplied truth is a clearance-checked drivable-cell mask; models see only
local measurements and survey packets that have reached the ground receiver.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


LOCAL_RADIUS_M = 6.
SURVEY_RADIUS_M = 12.
GROUND_SPEED_M_S = 1.
UAV_SPEED_M_S = 6.
GROUND_LOOKAHEAD_M = 25.
MAX_CANDIDATES = 12
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def survey_policy_window(position, targets, ground_target, origin, resolution, shape):
    """Crop the receiver raster around the vehicle and its active decisions, in metres."""
    xy = np.vstack((position, targets, ground_target)) if ground_target is not None else np.vstack((position, targets))
    # 95 m covers the actor's candidate road neighbourhood. Align the origin
    # with its 8 m sampling lattice so unchanged road samples retain coordinates.
    stride = max(1, round(8. / resolution))
    lo = np.floor((xy.min(axis=0) - 95. - origin) / resolution).astype(int)
    hi = np.ceil((xy.max(axis=0) + 95. - origin) / resolution).astype(int) + 1
    lo = np.clip(lo // stride * stride, 0, shape[::-1])
    hi = np.clip(hi, 0, shape[::-1])
    return (slice(lo[1], hi[1]), slice(lo[0], hi[0]))


def ground_plan(cells, start_yx, goal_yx, resolution, *, directional=False):
    """Return the reachable frontier set and an optimized path, both in (row, col).

    Minimize physical path length on the four-connected confirmed-free graph
    to the goal when connected. Otherwise minimize remaining Euclidean distance
    plus half the shortest known path length over reachable frontier cells.
    """
    start, goal = np.asarray(start_yx, int), np.asarray(goal_yx, int)
    free = cells == -1
    if not free[tuple(start)]:
        raise ValueError('Ground origin must have a confirmed free measurement')
    coords = np.argwhere(free)
    labels = np.full(free.shape, -1, dtype=np.int32)
    labels[tuple(coords.T)] = np.arange(len(coords))
    # Generate both directions with one copy of each physical road adjacency.
    horizontal = free[:, :-1] & free[:, 1:]
    vertical = free[:-1, :] & free[1:, :]
    a = np.r_[labels[:, :-1][horizontal], labels[:-1, :][vertical]]
    b = np.r_[labels[:, 1:][horizontal], labels[1:, :][vertical]]
    indices, adj = np.r_[a, b], np.r_[b, a]
    graph = csr_matrix((np.ones(len(indices), dtype=np.int8), (indices, adj)),
                       shape=(len(coords), len(coords)))
    start_id = labels[tuple(start)]
    costs, prev = shortest_path(graph, method='D', indices=start_id,
                                unweighted=True, return_predecessors=True)
    frontier = np.flatnonzero(np.isfinite(costs) & binary_dilation(
        cells == 0, structure=CROSS)[tuple(coords.T)])
    remaining = np.linalg.norm(coords[frontier] - goal, axis=1)
    objective = (.5 * costs[frontier] + remaining) * resolution
    front_order = np.argsort(objective, kind='stable')
    goal_id = labels[tuple(goal)] if ((goal >= 0) & (goal < cells.shape)).all() else -1
    if goal_id >= 0 and np.isfinite(costs[goal_id]):
        dest, reason = goal_id, 'known_goal_route'
    elif directional:
        # A direction marker can land on an obstacle. Track the best reachable
        # stretch of road toward it, including when the entire area is observed.
        reachable = np.flatnonzero(np.isfinite(costs))
        remaining = np.linalg.norm(coords[reachable] - goal, axis=1)
        dest = reachable[np.argmin(remaining + .5 * costs[reachable])]
        reason = 'directional_route' if dest != start_id else 'hold'
    elif len(frontier):
        dest, reason = frontier[front_order[0]], 'reachable_frontier'
    else:
        dest, reason = start_id, 'hold'
    path = [dest]
    while path[-1] != start_id:
        path.append(prev[path[-1]])
    return coords[path[::-1]], coords[frontier[front_order]], reason


def _sample_corridors(cells, candidates, goal, resolution, max_count=MAX_CANDIDATES):
    """Build alternative hypothetical local corridors ahead of visible frontiers.

    Only unknown cells matter for an open question. Confirmed blocked cells
    invalidate a proposed corridor. These lookahead routes never authorize motion.
    """
    questions = []
    frontier_anchors = []
    for anchor in candidates:
        if all(np.linalg.norm(anchor - old) * resolution >= 9. for old in frontier_anchors):
            frontier_anchors.append(anchor)
            if len(frontier_anchors) == 5:
                break
    for anchor in frontier_anchors:
        direction = np.asarray(goal) - anchor
        if not direction.any():
            continue
        heading = math.atan2(direction[0], direction[1])
        for angle in (0., -40., 40., -80., 80., -125., 125.):
            theta = heading + math.radians(angle)
            line_length = min(24., max(6., np.linalg.norm(direction) * resolution))
            distance = np.linspace(0., line_length / resolution, math.ceil(line_length / resolution) * 2 + 1)
            coords = np.rint(anchor + distance[:, None] * [math.sin(theta), math.cos(theta)]).astype(int)
            # A diagonal pair can touch across the corner of two blocked cells.
            # Insert one deterministic cardinal bridge into the proposed path
            # before checking occupancy or defining the outstanding question.
            unique = coords[np.r_[True, np.any(np.diff(coords, axis=0), axis=1)]]
            diagonals = np.all(np.diff(unique, axis=0) != 0, axis=1)
            if diagonals.any():
                ends = np.r_[0, np.cumsum(1 + diagonals)]
                coords = np.empty((int(ends[-1])+1, 2), dtype=int)
                coords[ends] = unique
                coords[ends[1:][diagonals]-1] = np.column_stack((
                    unique[:-1, 0][diagonals], unique[1:, 1][diagonals]))
            else:
                coords = unique
            inside = ((coords >= 0) & (coords < cells.shape)).all(axis=1)
            coords = coords[:np.flatnonzero(~inside)[0]] if not inside.all() else coords
            if not len(coords):
                continue
            values = cells[tuple(coords.T)]
            blocked = np.flatnonzero(values == 1)
            if len(blocked):
                coords, values = coords[:blocked[0]], values[:blocked[0]]
            unknown = coords[values == 0]
            if not len(unknown):
                continue
            indices = np.ravel_multi_index(unknown.T, cells.shape)
            ordered = unknown[np.r_[True, indices[1:] != indices[:-1]]]
            # Aim at the middle of a gap: a single sensor footprint can cover
            # an entire short outstanding segment or part of a longer one.
            target = ordered[len(ordered) // 2]
            if any(np.linalg.norm(target - q[0]) * resolution < 6. for q in questions):
                continue
            questions.append((target.copy(), np.unique(indices)))
    if len(questions) <= max_count:
        return questions
    # Retain alternatives from different anchors instead of privileging the
    # first frontier and exhausting every candidate slot there.
    select = np.linspace(0, len(questions)-1, max_count).round().astype(int)
    return [questions[i] for i in select]


class SurveyEnv:
    def __init__(self, free, resolution, origin, start, goal, *, seed=0, max_steps=32,
                 sensor_valid=None, ground_replan_interval_s=5, max_time_s=None, local_window=False):
        self._free = np.asarray(free, dtype=bool)
        self._sensor_valid = None if sensor_valid is None else np.asarray(sensor_valid, dtype=bool)
        if self._sensor_valid is not None and self._sensor_valid.shape != self._free.shape:
            raise ValueError('Terrain sensor validity mask must match the source raster')
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=float)
        self.start, self.goal = (np.asarray(p, dtype=float).copy() for p in (start, goal))
        if self._free.ndim != 2 or not self._free.size or self.resolution <= 0 or max_steps < 1:
            raise ValueError('Expected nonempty free raster, positive resolution and episode limit')
        self.pos = self._index(self.start)
        self.dest = self._index(self.goal)
        for pt in (self.pos, self.dest):
            if (pt < 0).any() or (pt >= self._free.shape).any() or not self._free[tuple(pt)]:
                raise ValueError('Endpoint outside clearance-checked free raster')
            if self._sensor_valid is not None and not self._sensor_valid[tuple(pt)]:
                raise ValueError('Ground endpoint lacks measured source support')
        if type(ground_replan_interval_s) is not int or ground_replan_interval_s < 1:
            raise ValueError('Ground replanning interval must be a positive integer in seconds')
        if max_time_s is not None and (type(max_time_s) is not int or max_time_s < 1):
            raise ValueError('Simulation horizon must be a positive integer in seconds')
        self.max_time_s = max_time_s
        self.local_window = local_window
        self.ground_replan_interval_s = ground_replan_interval_s
        self.ground_replan_events = []
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.cells = np.zeros(self._free.shape, dtype=np.int8)
        self.source_times = np.full(self._free.shape, np.nan, dtype=np.float32)
        self.receipt_times = self.source_times.copy()
        self.time = self.steps = 0
        self.travelled = self.reward_sum = 0.
        self.useful = self.resolved = 0
        self.passable_answers = self.blocked_answers = 0
        self.done = False
        self.previous = self.issued = None
        self.last_survey_receipt_at = None
        self.evidence_received = False
        self.air_position = self.start.copy()
        self.velocity = np.zeros(2)
        self.orientation = math.atan2(*(self.goal-self.start)[::-1])
        self._obs = None
        self._seen_min = self.pos.copy()
        self._seen_max = self.pos.copy()
        self._map_generation = 0
        self._plan_key = None
        self._motion_credit = 0.
        self._local_sensor()
        self._replan_ground()

    def _index(self, xy):
        return np.floor((xy-self.origin)/self.resolution).astype(int)[::-1]

    def _xy(self, rc):
        return self.origin + (np.asarray(rc)[..., ::-1] + .5) * self.resolution

    @property
    def position(self):
        if self.travelled == 0.:
            return self.start
        if np.array_equal(self.pos, self.dest):
            return self.goal
        return self._xy(self.pos)

    def _disk(self, rc, radius):
        k = math.ceil(radius/self.resolution)
        rows, cols = self.cells.shape
        y, x = rc
        yy, xx = np.ogrid[max(0, y-k):min(rows, y+k+1), max(0, x-k):min(cols, x+k+1)]
        inside = (yy-y)**2 + (xx-x)**2 <= (radius/self.resolution)**2
        a, b = np.nonzero(inside)
        ids = (a + yy[0, 0]) * cols + b + xx[0, 0]
        # An unmeasured source cell may be unavailable for driving but cannot
        # be reported as a measured obstacle by either the ground or air sensor.
        return ids if self._sensor_valid is None else ids[self._sensor_valid.flat[ids]]

    def _local_sensor(self):
        indices = self._disk(self.pos, LOCAL_RADIUS_M)
        self._extend_known_bounds(indices)
        self.cells.flat[indices] = np.where(self._free.flat[indices], -1, 1)
        self.source_times.flat[indices] = self.time
        self.receipt_times.flat[indices] = self.time

    def _extend_known_bounds(self, indices):
        new = indices[self.cells.flat[indices] == 0]
        if len(new):
            y, x = np.divmod(new, self.cells.shape[1])
            self._seen_min = np.minimum(self._seen_min, (y.min(), x.min()))
            self._seen_max = np.maximum(self._seen_max, (y.max(), x.max()))
            self._map_generation += 1

    def _plan_on_receiver_map(self):
        key = (tuple(self.pos), self._map_generation)
        if self._plan_key != key:
            # All observed components are inside this rectangle. One extra
            # unknown cell on each side preserves exactly its frontier set.
            lo = np.maximum(self._seen_min - 1, 0)
            hi = np.minimum(self._seen_max + 2, self.cells.shape)
            cells = self.cells[lo[0]:hi[0], lo[1]:hi[1]]
            path, fronts, reason = ground_plan(cells, self.pos-lo, self.dest-lo, self.resolution)
            self._plan_result = path+lo, fronts+lo, reason
            self._plan_key = key
        return self._plan_result

    def _replan_ground(self):
        path, fronts, reason = self._plan_on_receiver_map()
        previous = self._ground_target if self.ground_replan_events else None
        previous_reached = None if previous is None else bool(np.array_equal(
            self.pos, self._index(previous)))
        cap = max(1, int(GROUND_LOOKAHEAD_M/self.resolution))
        self._path = path[:1+cap]
        self._path_index = 0
        self._ground_target = self._xy(self._path[-1])
        if np.array_equal(self._path[-1], self.dest):
            self._ground_target = self.goal.copy()
        elif len(self._path) == 1:
            self._ground_target = self.position.copy()
        self.ground_replan_events.append(dict(
            time_s=self.time, position=self.position.tolist(),
            previous_target=None if previous is None else previous.tolist(),
            previous_target_reached=previous_reached, target=self._ground_target.tolist(),
            reason=reason, latest_survey_receipt_s=self.last_survey_receipt_at))
        self._frontiers = fronts

    def _prepare(self):
        event = self.ground_replan_events[-1]
        # Survey decisions can ask about current frontiers between clock ticks,
        # but can never set the ground vehicle's active destination.
        if event['time_s'] == self.time:
            fronts, reason = self._frontiers, event['reason']
        else:
            _, fronts, reason = self._plan_on_receiver_map()
        questions = ([] if reason == 'known_goal_route'
                     else _sample_corridors(self.cells, fronts, self.dest, self.resolution))
        if not questions:
            # One legal no-request action when the current map has no unanswered
            # corridor. It lets the ground planner continue driving autonomously.
            questions = [(self.pos.copy(), np.zeros(0, dtype=int))]
        self._questions = questions
        self._targets = self._xy([q[0] for q in questions])
        positions = self.position
        extra_distance = (np.linalg.norm(self._targets-positions, axis=1)
                          + np.linalg.norm(self._targets-self.goal, axis=1)
                          - np.linalg.norm(positions-self.goal))
        self._route_weights = 1. / (1. + np.maximum(extra_distance, 0.) / GROUND_LOOKAHEAD_M)
        scores = []
        for xy, (_, gap) in zip(self._targets, questions):
            progress = np.linalg.norm(self.goal-positions) - np.linalg.norm(self.goal-xy)
            scores.append(progress/25. + min(1., len(gap)*self.resolution/16.)
                          - np.linalg.norm(xy-positions)/125.)
        window = np.s_[:, :]
        origin = self.origin
        if self.local_window:
            window = survey_policy_window(positions, self._targets, self._ground_target, self.origin,
                                          self.resolution, self.cells.shape)
            origin = self.origin + np.array([window[1].start, window[0].start]) * self.resolution
        remaining = 1. - self.steps/self.max_steps
        if self.max_time_s is not None:
            remaining = min(remaining, 1. - self.time/self.max_time_s)
        self._obs = dict(position=positions.copy(), goal=self.goal.copy(),
                         ground_target=self._ground_target.copy(), ground_path=self._xy(self._path[self._path_index:]),
                         ground_plan_reason=event['reason'], no_request=not len(questions[0][1]), targets=self._targets.copy(),
                         heuristic_scores=np.array(scores), mask=np.ones(len(questions), dtype=bool),
                         time=self.time, survey_target=self.previous, survey_issued_at=self.issued,
                         survey_evidence_received=self.evidence_received,
                         last_survey_receipt_at=self.last_survey_receipt_at, cells=self.cells[window].copy(),
                         source_times=self.source_times[window].copy(), receipt_times=self.receipt_times[window].copy(),
                         local_window=self.local_window, resolution=self.resolution, origin=origin.copy(),
                         velocity=self.velocity.copy(),
                         orientation=self.orientation, remaining_budget=max(0., remaining))

    def observe(self):
        if self._obs is None:
            self._prepare()
        return self._obs

    def _advance_ground(self):
        available = GROUND_SPEED_M_S + self._motion_credit
        moves = min(len(self._path)-1-self._path_index, int(available/self.resolution))
        for _ in range(moves):
            self._path_index += 1
            self.pos = self._path[self._path_index]
            self.travelled += self.resolution
            self._local_sensor()
        self._motion_credit = (available % self.resolution
                               if self._path_index + 1 < len(self._path) else 0.)

    def _receive(self, gap, ids, packet, observed_at):
        unresolved = not np.any(self.cells.flat[gap] == 1) and np.any(self.cells.flat[gap] == 0)
        new = ids[self.cells.flat[ids] == 0]
        # Credit only unique unknown cells in the selected planning gap. A
        # fresh observation of an already mapped road has zero novelty.
        novel = np.intersect1d(gap, new, assume_unique=True)
        if len(ids):
            fresh = ~(self.source_times.flat[ids] > observed_at)
            accepted = ids[fresh]
            self._extend_known_bounds(accepted)
            self.cells.flat[accepted] = packet[fresh]
            self.source_times.flat[accepted] = observed_at
            self.receipt_times.flat[accepted] = self.time
            self.last_survey_receipt_at = self.time
        answered = bool(unresolved and len(novel) and
                        (np.any(self.cells.flat[gap] == 1) or np.all(self.cells.flat[gap] == -1)))
        if len(ids):
            # Simulator packets carry explicit ternary terrain, so either
            # delivered blockage or confirmed free continuity answers the query.
            self.evidence_received = answered
        coverage = len(novel) / len(gap) if len(gap) else 0.
        return len(novel), answered, coverage

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode has finished')
        self.observe()
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(self._questions):
            raise ValueError('Invalid survey candidate')
        rc, gap = self._questions[action]
        target = self._targets[action]
        issued = self.time
        previous = self.position.copy()
        if len(gap):
            flight = max(1, math.ceil(np.linalg.norm(target-self.air_position)/UAV_SPEED_M_S))
            latency = int(self.rng.integers(1, 4))
            duration = flight + latency
            observed_at = issued + flight
            ids = self._disk(rc, SURVEY_RADIUS_M)
            packet = np.where(self._free.flat[ids], -1, 1)
            self.previous, self.issued = target.copy(), issued
            self.evidence_received = False
            self.air_position = target.copy()
        else:
            duration = 8
            ids = packet = np.zeros(0, dtype=int)
            observed_at = issued
        events_start = len(self.ground_replan_events)
        new_count, answered, coverage = 0, False, 0.
        receipt = None
        for _ in range(duration):
            self.time += 1
            self._advance_ground()
            if self.time == issued + duration and len(gap):
                # Evidence due at a clock tick is incorporated before planning.
                new_count, answered, coverage = self._receive(gap, ids, packet, observed_at)
                receipt = self.time
            if self.time % self.ground_replan_interval_s == 0:
                self._replan_ground()
            if np.array_equal(self.pos, self.dest) or self.time == self.max_time_s:
                break
        duration = self.time - issued
        self.velocity = (self.position-previous)/duration
        if np.any(self.velocity):
            self.orientation = math.atan2(*self.velocity[::-1])
        relevance = float(self._route_weights[action])
        reward = relevance * (coverage + float(answered)) - .001*duration
        self.steps += 1
        self.useful += new_count
        self.resolved += int(answered)
        answer = None
        if answered:
            if np.any(self.cells.flat[gap] == 1):
                answer = 'blocked'
                self.blocked_answers += 1
            else:
                answer = 'passable'
                self.passable_answers += 1
        self.reward_sum += reward
        self.done = np.array_equal(self.pos, self.dest) or self.time == self.max_time_s or self.steps >= self.max_steps
        info = dict(duration_s=duration, ground_target=self._obs['ground_target'].tolist(),
                    survey_target=target.tolist() if len(gap) else None, useful_new_cells=new_count,
                    resolved_question=answered, answer=answer, gap_fraction=float(coverage), route_relevance=relevance,
                    ground_plan_reason=self._obs['ground_plan_reason'], receipt_s=receipt,
                    ground_replans=len(self.ground_replan_events)-events_start,
                    ground_replan_events=self.ground_replan_events[events_start:],
                    source_s=observed_at if len(gap) and observed_at <= self.time else None)
        self._obs = None
        return float(reward), self.done, info

    def summary(self):
        arrived = bool(np.array_equal(self.pos, self.dest))
        end_reason = ('arrival' if arrived else 'time_budget' if self.time == self.max_time_s else
                      'decision_budget' if self.steps >= self.max_steps else 'running')
        return dict(goal_reached=arrived, end_reason=end_reason, max_time_s=self.max_time_s, steps=self.steps,
                    reward=float(self.reward_sum), initial_distance_m=float(np.linalg.norm(self.goal-self.start)),
                    remaining_m=float(np.linalg.norm(self.goal-self.position)),
                    useful_new_cells=self.useful, resolved_questions=self.resolved,
                    passable_answers=self.passable_answers, blocked_answers=self.blocked_answers,
                    sim_time_s=self.time, travelled_m=self.travelled,
                    ground_replan_interval_s=self.ground_replan_interval_s,
                    ground_replans=len(self.ground_replan_events),
                    early_ground_replans=sum(e['previous_target_reached'] is False
                                             for e in self.ground_replan_events),
                    ground_replan_events=list(self.ground_replan_events))
