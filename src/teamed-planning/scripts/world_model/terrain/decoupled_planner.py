"""Independent ground and aerial decisions on a private predicted terrain map.

Strategic targets may lie in unobserved terrain and can change before arrival.
The separate execution waypoint always has received/local route evidence.
Aerial tasks remain independent of both vehicle intention and execution.
"""
from __future__ import annotations

import heapq
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
import torch

from .decoupled_observation import (ACTION_CONTRACT, ARCHITECTURE, CHECKPOINT_FORMAT, DRONE_FEATURES,
    GROUND_FEATURES, HALF_EXTENT, MAX_PACKET_AGE, RESOLUTION, SIDE, action_vector,
    cell_centers, point)
from .vtmp2 import Vtmp2SubgoalManager, _Candidate



def distance_field(costs, sources):
    """Eight-connected Dijkstra, with no cutting through blocked corners."""
    height, width = costs.shape
    result = np.full_like(costs, np.inf, dtype=float)
    queue = []
    for row, col, distance in sources:
        if np.isfinite(costs[row, col]) and distance < result[row, col]:
            result[row, col] = distance
            heapq.heappush(queue, (float(distance), row, col))
    while queue:
        value, row, col = heapq.heappop(queue)
        if value > result[row, col]:
            continue
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            y, x = row+dy, col+dx
            if not (0 <= y < height and 0 <= x < width) or not np.isfinite(costs[y, x]):
                continue
            if dx and dy and (not np.isfinite(costs[row, x]) or not np.isfinite(costs[y, col])):
                continue
            step = RESOLUTION * math.hypot(dx, dy) * (costs[row, col]+costs[y, x])/2
            proposed = value+step
            if proposed < result[y, x]:
                result[y, x] = proposed
                heapq.heappush(queue, (proposed, y, x))
    return result


class TerrainPlanner:
    """Finite candidate optimization, not a claim of global/real-world optimality."""
    action_contract = ACTION_CONTRACT

    def __init__(self, model=None, *, drone_count=3, device='cpu', seed=0,
                 risk_weight=4., uncertainty_weight=2., survey_candidates=24):
        if (not np.isfinite([risk_weight, uncertainty_weight]).all()
                or min(risk_weight, uncertainty_weight) < 0
                or type(survey_candidates) is not int or not 4 <= survey_candidates <= 128):
            raise ValueError('Invalid planner weights or candidate count')
        self.device = torch.device(device)
        self.model = None if model is None else model.to(self.device).eval().requires_grad_(False)
        if model is not None and model.drone_count != drone_count:
            raise ValueError('Model and deployed fleet sizes differ')
        self.drone_count, self.seed = drone_count, int(seed)
        self.risk_weight, self.uncertainty_weight = risk_weight, uncertainty_weight
        self.survey_candidates = survey_candidates
        self.selector_mode = 'bootstrap_independent' if model is None else 'world_model_independent'
        self.training_vehicle_runtime = None
        self.needs_detour_information = False
        self.last_diagnostics = {}
        self.snapshot = None
        self._fields = {}
        self._costs = self._travel = self._travel_origin = None

    @classmethod
    def from_checkpoint(cls, path, **options):
        from .data import CHECKPOINT_SHA
        from .decoupled_model import DecoupledWorldModel
        checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
        if (not isinstance(checkpoint, dict) or checkpoint.get('format') != CHECKPOINT_FORMAT
                or checkpoint.get('action_contract') != ACTION_CONTRACT
                or checkpoint.get('model_architecture') != ARCHITECTURE
                or checkpoint.get('schema_version') != 4):
            raise ValueError('Expected a schema-4 independent-drive/survey terrain checkpoint')
        if checkpoint.get('v9_checkpoint_sha256') != CHECKPOINT_SHA:
            raise ValueError('World model was not trained with the required frozen v9 aircraft weights')
        config = checkpoint['architecture']
        with torch.random.fork_rng(devices=[]):
            model = DecoupledWorldModel(**config)
        if any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in checkpoint['model'].values()):
            raise ValueError('Non-finite checkpoint weights')
        model.load_state_dict(checkpoint['model'], strict=True)
        if options.get('drone_count', model.drone_count) != model.drone_count:
            raise ValueError('Checkpoint fleet size does not match deployment')
        options['drone_count'] = model.drone_count
        planner = cls(model, **options)
        planner.training_vehicle_runtime = checkpoint.get('vehicle_runtime', 'archived_v9_ground')
        return planner

    def observe(self, maps, states, origin, now):
        maps, states = np.asarray(maps, np.float32), np.asarray(states, np.float32)
        origin = np.asarray(origin)
        if (maps.shape != (3, SIDE, SIDE) or states.shape != (GROUND_FEATURES+DRONE_FEATURES*self.drone_count,)
                or not np.isfinite(maps).all() or not np.isfinite(states).all()
                or np.any((maps < 0) | (maps > 1)) or not math.isfinite(now)
                or origin.shape != (2,) or not np.isfinite(origin).all()
                or not np.array_equal(origin, np.floor(origin))):
            raise ValueError('Invalid received observation')
        self.snapshot = (maps.copy(), states.copy(), origin.astype(int), float(now))
        self._fields = {}
        self._costs = self._travel = self._travel_origin = None

    @torch.no_grad()
    def predict(self, actions):
        if self.snapshot is None:
            raise ValueError('Observe before planning')
        maps, states, _, _ = self.snapshot
        actions = np.asarray(actions, np.float32).reshape(-1, 6)
        if self.model is None:
            known = maps[1]
            occupancy = np.where(known, maps[0], .5)
            occupancy = np.where(maps[2], 1., occupancy)
            return dict(occupancy=np.repeat(occupancy[None], len(actions), 0),
                        entropy=np.repeat(((1-known)*(1-maps[2])*math.log(2))[None], len(actions), 0),
                        disagreement=np.zeros((len(actions), SIDE, SIDE)),
                        information=np.repeat(((1-known)*(1-maps[2]))[None], len(actions), 0))
        result = self.model.predict(
            torch.tensor(maps, device=self.device).unsqueeze(0).expand(len(actions), -1, -1, -1),
            torch.tensor(states, device=self.device).unsqueeze(0).expand(len(actions), -1),
            torch.tensor(actions, device=self.device), torch.full((len(actions),), 10., device=self.device))
        if any(not torch.isfinite(value).all() for value in result.values()):
            raise ValueError('Non-finite world-model prediction')
        return {name: value.cpu().numpy() for name, value in result.items()}

    def _planning_costs(self):
        if self._costs is not None:
            return self._costs
        maps = self.snapshot[0]
        prediction = self.predict([action_vector(None, None)])
        occupancy = prediction['occupancy'][0]
        costs = 1 + self.risk_weight*occupancy + self.uncertainty_weight*(
            prediction['entropy'][0]/math.log(2)+prediction['disagreement'][0])
        # These 6.25m cells contain point samples, not fully occupied squares.
        # Hard-blocking a sampled cell invents walls across narrow *observed*
        # routes. Use a high finite continuation cost; actual drive permission
        # still comes exclusively from the fine local grid/received graph.
        occupied_sample = (maps[1] > .5) & (maps[0] > .5)
        costs[occupied_sample] = np.maximum(costs[occupied_sample], 12.)
        costs[maps[2] > .5] = np.inf
        self._costs = costs
        return costs

    def _continuation(self, goal):
        key = tuple(point(goal))
        if key in self._fields:
            return self._fields[key]
        origin = self.snapshot[2]
        costs = self._planning_costs()
        xy = cell_centers(origin)
        col, row = np.floor(point(goal)/RESOLUTION).astype(int)-origin
        sources = []
        if 0 <= row < SIDE and 0 <= col < SIDE and np.isfinite(costs[row, col]):
            sources.append((row, col, float(np.linalg.norm(xy[row, col]-goal))))
        else:
            # Outside the prediction window the continuation remains an
            # explicitly optimistic Euclidean tail, not invented map evidence.
            for y in range(SIDE):
                for x in range(SIDE):
                    if y in (0, SIDE-1) or x in (0, SIDE-1):
                        sources.append((y, x, float(np.linalg.norm(xy[y, x]-goal))))
        field = distance_field(costs, sources)
        # Disconnected predictions receive a finite penalty for ranking;
        # only actual route proofs can authorize vehicle execution.
        tail = np.linalg.norm(xy-point(goal), axis=-1)
        self._fields[key] = np.where(np.isfinite(field), field, tail+4*HALF_EXTENT)
        return self._fields[key]

    def _sample_field(self, field, target):
        col, row = point(target)/RESOLUTION-self.snapshot[2]-.5
        return float(map_coordinates(field, [[row], [col]], order=1, mode='nearest')[0])

    def strategic_cost(self, target, position, goal):
        """Compare intentions on one fused map, regardless of route verification.

        This forward distance is a prediction, never a certificate for motion.
        Use the same risk-weighted prefix for known and unknown candidates so
        that tiny observed targets do not win just by omitting that risk cost.
        """
        if self.model is None:
            return float(np.linalg.norm(target-position)+np.linalg.norm(target-goal))
        costs = self._planning_costs()
        key = tuple(point(position))
        if self._travel_origin != key:
            origin = self.snapshot[2]
            col, row = np.floor(point(position)/RESOLUTION).astype(int)-origin
            if not (0 <= row < SIDE and 0 <= col < SIDE):
                raise ValueError('Vehicle position lies outside its prediction window')
            center = (np.array([col, row])+origin+.5)*RESOLUTION
            start_cost = float(np.linalg.norm(position-center)*costs[row, col])
            self._travel = distance_field(costs, [(row, col, start_cost)])
            tail = np.linalg.norm(cell_centers(origin)-position, axis=-1)+4*HALF_EXTENT
            self._travel = np.where(np.isfinite(self._travel), self._travel, tail)
            self._travel_origin = key
        return self._sample_field(self._travel, target)+self._sample_field(self._continuation(goal), target)

    def observed_occupied_sample(self, target):
        """Reject an endpoint exactly at a received occupied sample, not its whole cell."""
        if self.snapshot is None:
            return False
        maps, _, origin, _ = self.snapshot
        xy = cell_centers(origin)[(maps[1] > .5) & (maps[0] > .5)]
        return bool(len(xy) and np.any(np.linalg.norm(xy-point(target), axis=1) < 1e-6))

    def ground_cost(self, candidate, goal):
        if self.model is None:
            return float(candidate.cost + np.linalg.norm(candidate.point-goal))
        field = self._continuation(goal)
        return candidate.cost + self._sample_field(field, candidate.point)

    def choose_survey(self, position, goal, drive_target, radius, bounds, memory, now,
                      failures=(), current=None, switch_cost=2.):
        if radius <= 0 or not self.drone_count:
            return None
        position, goal = point(position), point(goal)
        angles = math.atan2(*(goal-position)[::-1])+np.arange(self.survey_candidates)*2*math.pi/self.survey_candidates
        candidates = [] if current is None else [point(current)]
        for index, angle in enumerate(angles):
            distance = radius*(1+(index % 3))/3
            target = position+distance*np.array([math.cos(angle), math.sin(angle)])
            if (np.any(target < bounds[:2]) or np.any(target >= bounds[2:])
                    or memory.observed(target)
                    or any(now < until and np.linalg.norm(target-failed) <= 12. for failed, until in failures)):
                continue
            if not any(np.allclose(target, existing) for existing in candidates):
                candidates.append(target)
        if not candidates:
            return None
        actions = [action_vector(drive_target, target) for target in candidates]
        prediction = self.predict(actions)
        _, _, origin, _ = self.snapshot
        xy = cell_centers(origin)
        # Information near plausible routes matters more than distant coverage.
        detour = np.linalg.norm(xy-position, axis=-1)+np.linalg.norm(xy-goal, axis=-1)-np.linalg.norm(goal-position)
        relevance = np.exp(-np.maximum(detour, 0)/60.)
        if self.needs_detour_information:
            # Lack of mission progress makes side/rear information useful too.
            # This is vehicle-observed progress, never a hidden reference route.
            relevance = .5+.5*relevance
        freshest = [packet['position'][:2] for packet in memory.telemetry.values()
                    if now-packet['time'] <= MAX_PACKET_AGE]
        scores = []
        for i, target in enumerate(candidates):
            # Prioritize predicted information near the requested survey. This
            # spatial utility proxy does not certify arrival within ten seconds.
            footprint = np.exp(-np.sum((xy-target)**2, axis=-1)/(2*18.**2))
            gain = prediction['information'][i]*prediction['entropy'][i]*relevance*footprint
            distance = min((np.linalg.norm(target-p) for p in freshest), default=np.linalg.norm(target-position))
            footprint_cells = 2*math.pi*18.**2/RESOLUTION**2
            score = -100*float(gain.sum())/footprint_cells + .1*float(distance)
            if current is not None and not np.allclose(target, current):
                score += switch_cost
            if self.model is None or self.needs_detour_information:
                # An explicit angular exploration cost diversifies bootstrap
                # data and avoids repeatedly surveying the same failing bearing.
                # Information is normalized to one footprint so this cost is
                # not drowned out by a sum over dozens of map pixels.
                score += 40*(1-math.cos(math.atan2(*(target-position)[::-1])-(self.seed+int(now//40))*math.pi/4))
            scores.append(score)
        selected = int(np.argmin(scores))
        self.last_diagnostics = dict(selector=self.selector_mode, survey_candidates=len(candidates),
            survey_costs=[float(value) for value in scores], survey_target=candidates[selected].tolist(),
            objective='route_weighted_predicted_information_minus_travel_cost',
            uncertainty_calibrated=False)
        return candidates[selected].copy()


class IndependentGroundManager(Vtmp2SubgoalManager):
    """Revisable intention plus a separate, observed execution waypoint.

    ``target`` need not be reached or even observed. ``execution_target`` is the
    waypoint the vehicle controller may currently follow. Losing a graph route
    can stop execution without invalidating an otherwise useful intention.
    """
    _revoke_on_route_loss = False
    _prefer_verified_targets = False

    def __init__(self, goal, *, planner, **options):
        # This mode controls the base class's *evidence query*. Strategic
        # candidate generation below independently includes unknown terrain.
        options['visibility_mode'] = 'observed'
        options.setdefault('min_hold', 0.)
        options.setdefault('review_interval', 5.)
        super().__init__(goal, **options)
        self.planner = planner
        self._visits = []
        self._mission_best = float('inf')
        self._mission_progress_at = self._mission_time = None
        self._loop_recovery = False
        self.execution_target = None
        self._execution_best = float('inf')
        self._execution_progress_at = 0.
        self._context_current = None
        self._strategy_position = None

    @property
    def effective_radius(self):
        # Survey radio reach is not a vehicle navigation restriction.
        return self.radius

    def _score(self, candidate):
        cost = self.planner.strategic_cost(candidate.point, self._strategy_position, self.goal)
        if self._loop_recovery and self._visits:
            # Soft, fading history costs discourage revisiting the same few
            # metres. They never revoke a route proof or forbid backtracking.
            positions = np.asarray([position for _, position in self._visits])
            ages = self._mission_time-np.asarray([stamp for stamp, _ in self._visits])
            squared = np.sum((positions-candidate.point)**2, axis=1)
            cost += float(2.*np.sum(np.exp(-ages/300.-squared/(2*4.**2))))
        return cost

    def _context(self, position, graph):
        self._context_current = super()._context(position, graph)
        return self._context_current

    def _proof(self, target, position, context):
        """Qualify an intention; only an actual route earns 'verified'."""
        if not self._within(target, position):
            return None
        if (self._obstacle_tree is not None
                and self._obstacle_tree.query(target)[0] <= self._obstacle_margin):
            return None
        if self.planner.observed_occupied_sample(target):
            return None
        verified = super()._proof(target, position, context)
        if verified is not None:
            return verified
        return _Candidate(point(target), 'predicted_intent', 'unverified',
                          float(np.linalg.norm(target-position)))

    def _candidates(self, position, context, now):
        candidates = super()._candidates(position, context, now)
        heading = math.atan2(*(self.goal-position)[::-1])
        proposals = [self.goal]
        if self.target is not None and self.target_status != 'hold':
            proposals.append(self.target)
        for fraction in (1., 2./3., 1./3.):
            for angle in heading+np.arange(self.directions)*2*math.pi/self.directions:
                proposals.append(position+self.radius*fraction*np.array([math.cos(angle), math.sin(angle)]))
        for target in proposals:
            candidate = self._proof(target, position, context)
            if candidate is not None:
                candidates.append(candidate)
        unique = {}
        for candidate in candidates:
            if ((np.linalg.norm(candidate.point-position) > self.arrival+1e-6
                    or np.linalg.norm(candidate.point-self.goal) < 1e-6)
                    and not self._suppressed(candidate.point, now)):
                unique.setdefault(tuple(np.round(candidate.point, 7)), candidate)
        return list(unique.values())

    def _update_execution(self, position, now, strategic_changed):
        context = self._context_current
        if (self.target is None or self.target_status == 'hold'
                or self.final and np.linalg.norm(self.target-position) <= self.arrival):
            self.execution_target = None
            return
        if self.execution_target is not None and not strategic_changed:
            proof = super()._proof(self.execution_target, position, context)
            distance = float(np.linalg.norm(self.execution_target-position))
            if distance < self._execution_best-.25:
                self._execution_best, self._execution_progress_at = distance, now
            if proof is not None and distance > self.arrival and now-self._execution_progress_at < self.stall:
                return
        # A strategic endpoint may be unknown even while a useful observed
        # prefix exists. Never pass that unknown endpoint to the motion stack.
        direct = super()._proof(self.target, position, context)
        if direct is not None and np.linalg.norm(direct.point-position) > self.arrival:
            # The chosen intention already has an observed route; pass that
            # endpoint to the original graph/local motion planner.
            chosen = direct
        else:
            candidates = [candidate for candidate in super()._candidates(position, context, now)
                          if candidate.status == 'verified']
            chosen = min(candidates, key=lambda c: (round(self.planner.ground_cost(c, self.target), 5),
                         float(np.linalg.norm(c.point-self.target)), *c.point), default=None)
        if chosen is None:
            self.execution_target = None
            return
        self.execution_target = chosen.point.copy()
        self._execution_best = float(np.linalg.norm(chosen.point-position))
        self._execution_progress_at = now

    def update(self, position, graph, now):
        position = self._point(position)
        self._strategy_position = position.copy()
        now = float(now)
        if not math.isfinite(now) or self._mission_time is not None and now < self._mission_time:
            raise ValueError('Mission time must be finite and nondecreasing')
        distance = float(np.linalg.norm(position-self.goal))
        if distance < self._mission_best-2.:
            self._mission_best, self._mission_progress_at = distance, now
        self._mission_time = now
        self._visits = [(stamp, point) for stamp, point in self._visits if now-stamp <= 600.]
        if not self._visits or now-self._visits[-1][0] >= 10.:
            self._visits.append((now, position.copy()))
        # Reaching many tiny temporary goals must not reset mission progress.
        self._loop_recovery = now-self._mission_progress_at >= 60.
        self.planner.needs_detour_information = self._loop_recovery
        result = super().update(position, graph, now)
        self._update_execution(position, now, result[1])
        self.planner.last_diagnostics['ground_loop_recovery'] = self._loop_recovery
        self.planner.last_diagnostics['execution_target'] = (None if self.execution_target is None
                                                            else self.execution_target.tolist())
        return result

    def _choose(self, position, context, now):
        candidates = self._candidates(position, context, now)
        if not candidates:
            return None
        reached = [candidate for candidate in candidates
                   if candidate.status == 'verified' and np.linalg.norm(candidate.point-self.goal) < 1e-6
                   and np.linalg.norm(candidate.point-position) <= self.arrival]
        if reached:
            return reached[0]
        # Known and unknown intentions share the same objective. Actual motion
        # is independently restricted by _update_execution's observed proof.
        chosen = min(candidates, key=lambda c: (round(self._score(c), 5),
                     float(np.linalg.norm(c.point-self.goal)), *c.point))
        self.planner.last_diagnostics['ground_target'] = chosen.point.tolist()
        self.planner.last_diagnostics['ground_cost'] = float(self._score(chosen))
        return chosen


class IndependentSurveyManager:
    def __init__(self, planner, *, radius=200., interval=10., timeout=120., improvement=2.):
        if (not np.isfinite([radius, interval, timeout, improvement]).all()
                or radius <= 0 or interval <= 0 or timeout < interval or improvement < 0):
            raise ValueError('Invalid independent survey options')
        self.planner, self.radius = planner, float(radius)
        self.interval, self.timeout, self.improvement = float(interval), float(timeout), float(improvement)
        self.target = None
        self.chosen_at = self.reviewed_at = -float('inf')
        self.failures = []
        self.selected = self.completed = 0

    def update(self, position, goal, drive_target, reconnaissance, bounds, memory, now):
        radius = min(self.radius, reconnaissance)
        self.failures = [(p, until) for p, until in self.failures if now < until]
        invalid = self.target is None or np.linalg.norm(self.target-position) > radius
        if self.target is not None and memory.observed(self.target):
            self.completed += 1
            invalid = True
        if self.target is not None and now-self.chosen_at >= self.timeout:
            self.failures.append((self.target.copy(), now+90.))
            invalid = True
        # Regular model reviews use a switching penalty; ground target changes
        # never reset the survey timer or automatically cancel an aerial task.
        if not invalid and now-self.reviewed_at < self.interval:
            return False
        if self.target is None and now-self.reviewed_at < self.interval:
            return False
        previous = self.target
        self.target = self.planner.choose_survey(position, goal, drive_target, radius,
            np.asarray(bounds), memory, now, self.failures,
            current=None if invalid else self.target, switch_cost=self.improvement)
        self.reviewed_at = now
        changed = ((previous is None) != (self.target is None) or
                   (previous is not None and self.target is not None and
                    not np.allclose(previous, self.target)))
        self.selected += int(changed and self.target is not None)
        if changed:
            self.chosen_at = now
        return changed
