"""Optional survey advice with a bounded, nonblocking fallback to the baseline.

This adapter does not own the vehicle controller. vtmp continues choosing its
temporary vehicle target from measured evidence regardless of survey advice.
ROS wiring/flight benefit must be validated separately before enabling it.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import threading
import time

import numpy as np


@dataclass(frozen=True)
class AdvicePolicy:
    enabled: bool = False
    terrain_gate_passed: bool = False
    selective_gate_passed: bool = False
    graph_benefit_validated: bool = False
    timeout_s: float = .2
    max_unknown_fraction: float = .5
    minimum_expected_graph_gain: float = 0.

    def __post_init__(self):
        if (not np.isfinite(self.timeout_s) or self.timeout_s <= 0
                or not 0 <= self.max_unknown_fraction <= 1
                or not np.isfinite(self.minimum_expected_graph_gain) or self.minimum_expected_graph_gain < 0):
            raise ValueError('Invalid survey fallback policy')


class GuardedSurveyAdvice:
    """Poll one worker; never wait for model inference in the navigation loop.

``baseline(context)`` is the existing survey policy, not a model or a vehicle
target. ``propose(snapshot)`` returns a survey target, unknown_fraction and an
expected_graph_gain over the baseline at the same current vehicle/final goal.
The caller supplies an immutable snapshot with mission_id, graph_revision,
vehicle_position, final_goal and delivered map data. No privileged map truth.
``validate`` checks flight feasibility, goal-relevant graph gaps and visibility;
this class cannot certify those from coarse semantic probabilities.
"""
    def __init__(self, baseline, propose, validate, *, policy=None, clock=time.monotonic):
        self.baseline, self.propose, self.validate = baseline, propose, validate
        self.policy, self.clock = policy or AdvicePolicy(), clock
        self._lock = threading.Lock()
        self._job = None
        self.disabled_reason = None

    @staticmethod
    def identity(context):
        position = np.asarray(context['vehicle_position'], float)
        goal = np.asarray(context['final_goal'], float)
        if position.shape != (2,) or goal.shape != (2,) or not np.isfinite([position, goal]).all():
            raise ValueError('Finite current vehicle and final goal required')
        return (context['mission_id'], context['graph_revision'], tuple(position), tuple(goal))

    def choose(self, context):
        # Compute the baseline on the current live context on every call. A
        # stale cached vtmp target is not a fallback after map/mission changes.
        baseline = self.baseline(context)

        def fallback(reason):
            return dict(survey_target=baseline, source='baseline', fallback_reason=reason,
                        vehicle_policy='vtmp_on_measured_graph')

        p = self.policy
        if not (p.enabled and p.terrain_gate_passed and p.selective_gate_passed and p.graph_benefit_validated):
            return fallback('model_not_approved_for_navigation')
        if self.disabled_reason:
            return fallback(self.disabled_reason)
        identity = self.identity(context)
        with self._lock:
            job = self._job
            if job is not None:
                elapsed = (job.get('finished', self.clock())-job['started'])
                if elapsed > p.timeout_s:
                    # One daemon worker at most, no new jobs after a timeout.
                    # A tardy result is never used and cannot block vtmp.
                    self.disabled_reason = 'inference_timeout'
                    return fallback(self.disabled_reason)
                if 'finished' not in job:
                    return fallback('inference_pending')
                self._job = None
                if self.clock()-job['started'] > p.timeout_s:
                    return fallback('model_result_expired')
                if job['identity'] != identity:
                    return fallback('stale_mission_pose_or_graph')
                if 'error' in job:
                    self.disabled_reason = 'model_error: '+job['error']
                    return fallback(self.disabled_reason)
                try:
                    result = job['result']
                    target = np.asarray(result['survey_target'], float)
                    unknown = float(result['unknown_fraction'])
                    gain = float(result['expected_graph_gain'])
                    if target.shape != (2,) or not np.isfinite(target).all():
                        return fallback('invalid_survey_target')
                    if not np.isfinite(unknown) or not 0 <= unknown <= p.max_unknown_fraction:
                        return fallback('prediction_unknown')
                    if not np.isfinite(gain) or gain <= p.minimum_expected_graph_gain:
                        return fallback('no_expected_graph_gain_over_baseline')
                    if not self.validate(target.copy(), context, result):
                        return fallback('survey_not_validated')
                    return dict(survey_target=target.copy(), source='world_model_advice',
                                fallback_reason=None, vehicle_policy='vtmp_on_measured_graph')
                except Exception as exc:
                    return fallback('invalid_model_result: '+type(exc).__name__)
            snapshot = deepcopy(context)
            job = dict(identity=identity, started=self.clock())
            self._job = job

        def run():
            try:
                value = dict(result=self.propose(snapshot))
            except Exception as exc:
                value = dict(error=type(exc).__name__+': '+str(exc))
            with self._lock:
                job.update(value, finished=self.clock())

        threading.Thread(target=run, name='optional-world-model-survey', daemon=True).start()
        return fallback('inference_pending')
