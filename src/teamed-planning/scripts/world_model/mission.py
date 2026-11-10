"""Choose a drive or aerial-survey task using predictions, not hidden terrain.

An active survey target is broadcast to the aircraft while the vehicle holds.
It is not a vehicle waypoint and never acquires a drivable-route certificate
from a prediction. The original manager continues to screen drive candidates
against received graph evidence and the vehicle's actual local observations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import networkx as nx
import numpy as np

if __package__:
    from .vtmp2 import Vtmp2SubgoalManager
else:
    from vtmp2 import Vtmp2SubgoalManager


@dataclass
class MissionCandidate:
    point: np.ndarray
    source: str
    status: str
    cost: float
    purpose: str


class WorldModelMissionManager(Vtmp2SubgoalManager):
    """Rank mixed tasks; ``target`` is the broadcast task, not always a drive.

    ``planner.observe`` must be called by the runtime with received observations
    before ``update``. Survey tasks always set ``drive_target=None``: retaining
    an old drive command would introduce a hidden action absent from schema 3.
    ``changed`` includes a purpose change even when coordinates stay the same.
    """

    def __init__(self, goal, *, planner, candidate_limit=24, survey_limit=12,
                 survey_evidence_radius=6., survey_timeout=120., **options):
        if any(not callable(getattr(planner, method, None))
               for method in ('rank', 'observe', 'reset')):
            raise TypeError('Mission planning requires rank, observe and reset')
        if (type(candidate_limit) is not int or candidate_limit < 4
                or candidate_limit > 24 or type(survey_limit) is not int
                or not 1 <= survey_limit <= min(12, candidate_limit-1)):
            raise ValueError('Invalid mixed-task candidate budget')
        if (not math.isfinite(survey_evidence_radius) or survey_evidence_radius <= 0
                or not math.isfinite(survey_timeout) or survey_timeout <= 0):
            raise ValueError('Invalid survey evidence radius or timeout')
        super().__init__(goal, **options)
        self.planner = planner
        self.selector_mode = str(getattr(planner, 'selector_mode', 'world_model'))
        self.candidate_limit, self.survey_limit = candidate_limit, survey_limit
        self.survey_evidence_radius = float(survey_evidence_radius)
        self.survey_timeout = float(survey_timeout)
        self.purpose = 'drive'
        self.drive_target = None
        self.survey_completed = 0
        self._survey_failures = []
        self._survey_evidence_count = 0
        self._received_survey_points = None
        self.last_decision = {}
        self.planner.reset()

    def request_target(self, target):
        """Explicit manual vehicle target; safety checks remain mandatory."""
        super().request_target(target)
        self._requested_purpose = 'drive'

    def request_survey_target(self, target):
        """Request an aerial task, without commanding the vehicle to that point."""
        super().request_target(target)
        self._requested_purpose = 'survey'

    @staticmethod
    def _key(candidate):
        return (candidate.purpose, *np.round(candidate.point, 7).tolist())

    def _received_points(self, graph):
        network = graph if isinstance(graph, nx.Graph) else getattr(graph, 'graph', None)
        if not isinstance(network, nx.Graph):
            return np.empty((0, 2))
        points = []
        for _, attributes in network.nodes(data=True):
            if not self._trusted(attributes):
                continue
            point = np.asarray(attributes.get('position', ()), float)
            if point.ndim == 1 and point.size >= 2 and np.isfinite(point[:2]).all():
                points.append(point[:2].copy())
        return np.asarray(points).reshape(-1, 2)

    def _survey_observed(self, point, context):
        points = context['received_points']
        return bool(len(points) and np.any(
            np.linalg.norm(points-point, axis=1) <= self.survey_evidence_radius))

    def set_received_survey_graph(self, graph):
        """Snapshot actual received evidence before ground-route filtering.

        A node rejected for driving can still establish that an aerial survey
        delivered information there. These points never enter drive proofs.
        The runtime supplies a fresh snapshot each observation cycle.
        """
        self._received_survey_points = self._received_points(graph).copy()

    def survey_observed(self, point, graph):
        """Whether real received graph evidence covers this aerial task point.

        This says nothing about vehicle reachability. It also permits the
        collector to label a just-finished task after ``update`` chose another.
        """
        return self._survey_observed(self._point(point),
                                     {'received_points': self._received_points(graph)})

    def _nearby_evidence(self, point, context):
        points = context['received_points']
        return int(np.count_nonzero(np.linalg.norm(points-point, axis=1)
                                    <= 3*self.survey_evidence_radius))

    def _survey_suppressed(self, point, now):
        return any(now < until and np.linalg.norm(point-failed) <= self.failure_radius
                   for failed, until in self._survey_failures)

    def _survey_candidate(self, point, position, context, now):
        if (not self._within(point, position) or self.effective_radius <= 0
                or self._survey_observed(point, context)
                or self._survey_suppressed(point, now)):
            return None
        # Ground obstacles are not an aerial survey-reachability test. Actual
        # aircraft motion still passes through its existing communication/CBF
        # constraints; the envelope only defines candidate task coordinates.
        return MissionCandidate(point.copy(), 'aerial_survey', 'unverified',
                                float(np.linalg.norm(point-position)), 'survey')

    def _as_drive(self, candidate):
        return MissionCandidate(candidate.point.copy(), candidate.source,
                                candidate.status, candidate.cost, 'drive')

    def _current_candidate(self, position, context, now):
        if self.target is None or self.target_status == 'hold':
            return None
        if self.purpose == 'survey':
            return self._survey_candidate(self.target, position, context, now)
        candidate = super()._proof(self.target, position, context)
        if candidate is None or self._suppressed(self.target, now):
            return None
        if self.target_status == 'verified' and candidate.status != 'verified':
            return None
        return self._as_drive(candidate)

    def _diverse(self, candidates, position, limit):
        """Spatial sampling, never pre-sort by model/terminal goal benefit."""
        unique = {}
        for candidate in candidates:
            unique.setdefault(self._key(candidate), candidate)
        remaining = list(unique.values())
        if len(remaining) <= limit:
            return remaining
        selected = []
        heading = math.atan2(*(self.goal-position)[::-1])
        # Cycle angle and radius together to retain side/back tasks and each
        # distance scale even when the local graph contains many near nodes.
        for index in range(limit):
            angle = heading + (index % 8)*math.pi/4
            radius = self.effective_radius*((index % 3)+1)/3
            anchor = position+radius*np.array([math.cos(angle), math.sin(angle)])
            chosen = min(range(len(remaining)), key=lambda i:
                         (float(np.linalg.norm(remaining[i].point-anchor)),
                          self._key(remaining[i])))
            selected.append(remaining.pop(chosen))
        return selected

    def _mixed_candidates(self, position, context, now, current=None):
        drive = [self._as_drive(c) for c in super()._candidates(position, context, now)]
        survey = []
        heading = math.atan2(*(self.goal-position)[::-1])
        for radius in (self.effective_radius/3, self.effective_radius*2/3,
                       self.effective_radius):
            if radius <= self.arrival:
                continue
            for index in range(self.directions):
                angle = heading+index*2*math.pi/self.directions
                point = position+radius*np.array([math.cos(angle), math.sin(angle)])
                candidate = self._survey_candidate(point, position, context, now)
                if candidate is not None:
                    survey.append(candidate)
        reserved = [] if current is None else [current]
        key = None if current is None else self._key(current)
        drive = [c for c in drive if self._key(c) != key]
        survey = [c for c in survey if self._key(c) != key]
        available = self.candidate_limit-len(reserved)
        survey_count = min(self.survey_limit, available, len(survey))
        drive_count = min(available-survey_count, len(drive))
        # A lack of drive evidence must not suppress possible aerial tasks.
        # Conversely, keep the reserved current task even when it is off-ring.
        # Representation quotas only: both groups still compete by exactly the
        # same model score. Otherwise the outer sampling rings can evict every
        # observed route simply because those routes happen to be shorter.
        known = [c for c in drive if c.status == 'verified']
        unknown = [c for c in drive if c.status != 'verified']
        known_count = min(len(known), max(drive_count-len(unknown), (drive_count+1)//2))
        candidates = reserved+self._diverse(known, position, known_count)
        candidates += self._diverse(unknown, position, drive_count-known_count)
        candidates += self._diverse(survey, position, survey_count)
        return candidates

    def _select(self, position, context, now, current=None):
        candidates = self._mixed_candidates(position, context, now, current)
        if not candidates:
            self.last_decision = {'selector_mode': self.selector_mode,
                                  'candidate_count': 0, 'reason': 'no_candidate'}
            return None, None, None
        costs = np.asarray(self.planner.rank(candidates, self.goal.copy()), dtype=float)
        if costs.shape != (len(candidates),) or not np.isfinite(costs).all():
            raise ValueError('World-model task ranking must return one finite cost per candidate')
        index = min(range(len(candidates)), key=lambda i: (float(costs[i]), self._key(candidates[i])))
        current_cost = None
        if current is not None:
            current_cost = float(costs[next(i for i, c in enumerate(candidates)
                                           if self._key(c) == self._key(current))])
        self.last_decision = dict(selector_mode=self.selector_mode,
            candidate_count=len(candidates), selected_purpose=candidates[index].purpose,
            selected_target=candidates[index].point.tolist(), selected_cost=float(costs[index]),
            current_cost=current_cost)
        return candidates[index], float(costs[index]), current_cost

    def _adopt_mission(self, candidate, position, context, now, reason, manual=False):
        previous_purpose = self.purpose
        previous_drive = None if self.drive_target is None else self.drive_target.copy()
        previous_target = None if self.target is None else self.target.copy()
        self.purpose = 'drive' if candidate is None else candidate.purpose
        target, _ = super()._adopt(candidate, position, context, now, reason, manual=manual)
        self.drive_target = (candidate.point.copy() if candidate is not None
                             and candidate.purpose == 'drive' else None)
        if self.purpose == 'survey':
            self.final = False
            self.target_status = 'unverified'
            self._survey_evidence_count = self._nearby_evidence(self.target, context)
        changed = (previous_target is None or previous_purpose != self.purpose
                   or not np.allclose(previous_target, target, atol=1e-6, rtol=0.))
        if not changed:
            changed = ((previous_drive is None) != (self.drive_target is None)
                       or (previous_drive is not None and self.drive_target is not None
                           and not np.allclose(previous_drive, self.drive_target, atol=1e-6, rtol=0.)))
        # The base counter only sees coordinates, so count purpose-only changes.
        if (candidate is not None and previous_target is not None
                and previous_purpose != self.purpose
                and np.allclose(previous_target, target, atol=1e-6, rtol=0.)):
            self.selected += 1
        return target, changed

    def _record_failure(self, now):
        if self.purpose == 'survey':
            if self.target is not None and self.target_status != 'hold':
                self._survey_failures.append((self.target.copy(), now+self.failure_ttl))
                self.failed += 1
        else:
            super()._record_failure(now)

    def _replace(self, position, context, now, reason):
        candidate, _, _ = self._select(position, context, now)
        return self._adopt_mission(candidate, position, context, now, reason)

    def update(self, position, graph, now):
        position, now = self._point(position), float(now)
        if not math.isfinite(now):
            raise ValueError('now must be finite')
        self.failures = [(p, until) for p, until in self.failures if now < until]
        self._survey_failures = [(p, until) for p, until in self._survey_failures if now < until]
        context = super()._context(position, graph)
        context['received_points'] = (self._received_points(graph)
            if self._received_survey_points is None else self._received_survey_points.copy())

        goal = super()._proof(self.goal, position, context)
        if (goal is not None and goal.status == 'verified'
                and np.linalg.norm(self.goal-position) <= self.arrival):
            return self._adopt_mission(self._as_drive(goal), position, context, now,
                                       'final_reached')

        if self._requested is not None:
            point, self._requested = self._requested, None
            if getattr(self, '_requested_purpose', 'drive') == 'survey':
                candidate = self._survey_candidate(point, position, context, now)
            else:
                proof = super()._proof(point, position, context)
                candidate = (None if proof is None or self._suppressed(point, now)
                             else self._as_drive(proof))
            if candidate is not None:
                return self._adopt_mission(candidate, position, context, now,
                                           'manual_request', manual=True)
            self.rejected_requests += 1

        forced, self._forced_failure = self._forced_failure, None
        if self.target is None or self.target_status == 'hold':
            return self._replace(position, context, now, 'initial')
        if self.purpose == 'survey' and self._survey_observed(self.target, context):
            self.survey_completed += 1
            return self._replace(position, context, now, 'survey_observed')
        current = self._current_candidate(position, context, now)
        if current is None or forced is not None:
            self._record_failure(now)
            return self._replace(position, context, now, forced or 'target_invalid')

        self.last_source, self.target_status = current.source, current.status
        if self.purpose == 'drive':
            self.drive_target = current.point.copy()
            self.final = current.status == 'verified' and np.linalg.norm(self.target-self.goal) < 1e-6
            distance = float(np.linalg.norm(self.target-position))
            if distance <= self.arrival:
                if self.final:
                    self.last_reason = 'final_reached'
                    return self.target.copy(), False
                self.arrivals += 1
                return self._replace(position, context, now, 'arrived')
            if distance < self._best-.25:
                self._best, self._progress_at, self._evidence_at = distance, now, now
            if context['evidence'] > self._evidence_count:
                self._evidence_at = now
            self._evidence_count = context['evidence']
            expired = now-self._progress_at >= self.max_wait or now-self._evidence_at >= self.stall
        else:
            self.drive_target, self.final = None, False
            evidence = self._nearby_evidence(self.target, context)
            if evidence > self._survey_evidence_count:
                self._evidence_at = now
            self._survey_evidence_count = evidence
            expired = now-self._chosen_at >= self.survey_timeout or now-self._evidence_at >= self.stall
        if expired:
            self._record_failure(now)
            return self._replace(position, context, now,
                                 'survey_timeout' if self.purpose == 'survey' else 'stalled')

        if (not self._manual and now-self._chosen_at >= self.min_hold
                and now-self._reviewed_at >= self.review_interval):
            self._reviewed_at = now
            candidate, cost, current_cost = self._select(position, context, now, current)
            if candidate is not None and current_cost-cost > self.switch_improvement:
                return self._adopt_mission(candidate, position, context, now, 'better_prediction')
        self.last_reason = 'retained'
        return self.target.copy(), False

    def stats(self):
        result = super().stats()
        result.update(vtmp2_survey_task=float(self.purpose == 'survey'),
                      vtmp2_survey_completed=float(self.survey_completed),
                      vtmp2_vehicle_holding=float(self.drive_target is None))
        return result
