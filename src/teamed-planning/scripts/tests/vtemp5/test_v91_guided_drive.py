"""Regression tests for same-tick waypoint execution on the frozen fleet."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


REPO = Path(__file__).resolve().parents[5]
SRC = REPO / 'src/teamed-planning/scripts/vtemp5'
sys.path.insert(0, str(SRC))
from v91_guided_drive import _publish_vehicle_target


class GuidedDriveTests(unittest.TestCase):
    def test_jitter_is_relative_to_the_last_accepted_point_and_preserves_clocks(self):
        car, previous = np.array([100.,100.]), np.array([300.,100.])
        c = SimpleNamespace(goal_position=tuple(previous), _local_route_best=188.,
                            _route_map=None, _local_route_progress_time=140.,
                            _drive_to=Mock(), _start_route_progress=Mock())
        for dy in (.01, .1, 1.8):
            keep, result = _publish_vehicle_target(c, previous, True, previous + [0.,dy], car, 180, False)
            np.testing.assert_equal(keep, previous)
            self.assertEqual(result, 'held')
        rolling = previous + [8.,1.]
        accepted, result = _publish_vehicle_target(c, previous, True, rolling, car, 210, False)
        np.testing.assert_equal(accepted, rolling)
        self.assertEqual(result, 'rolled')
        self.assertEqual(c._local_route_progress_time, 140.)
        self.assertEqual(c.goal_position, tuple(rolling))
        c._drive_to.assert_not_called()
        c._start_route_progress.assert_not_called()
        _publish_vehicle_target(c, accepted, False, rolling, car, 240, False)
        c._drive_to.assert_called_once_with(tuple(rolling))
        c._start_route_progress.assert_called_once()

    def test_moving_a_nearby_stopped_target_can_replan_in_place(self):
        c = SimpleNamespace(goal_position=None, _drive_to=Mock(), _start_route_progress=Mock())
        car = np.array([100.,100.])
        accepted, reason = _publish_vehicle_target(c, car, True, car + [0.,.6], car, 60, False)
        self.assertEqual(reason,'replanned')
        np.testing.assert_equal(accepted,[100.,100.6])
        accepted, reason = _publish_vehicle_target(c, accepted, True, accepted, car, 90, True)
        self.assertEqual(reason,'replanned')
        self.assertEqual(c._drive_to.call_count, 2)

    def test_observed_arm_executes_one_action_with_same_tick_survey_and_no_cnn(self):
        protocol = REPO/'artifacts/v91-frozen-km-paired-20261126/protocol.json'
        if not protocol.is_file():
            self.skipTest('Frozen integration checkpoints are unavailable')
        # A fresh process keeps previously imported live and archived modules
        # isolated when running this test as part of the full suite.
        subprocess.run([sys.executable, '-B', str(Path(__file__)), '--probe', str(protocol)],
                       cwd=REPO, check=True, timeout=90)


def integration_probe(protocol):
    from unittest.mock import patch
    import run_v91_km_eval as runner
    from v91_guided_drive import run_km_actor_arm
    from v91_km_cases import km_cases
    from v91_rejoin_guidance import RejoinGuidance

    runner.load_models(json.loads(Path(protocol).read_text())['weights'])
    case = next(x for x in km_cases() if x['case_id'] == 'lake-dev-00-detour_left')
    delivered = {}
    records = []
    original_record = RejoinGuidance.record_local

    def record(self, tick, local):
        records.append((tick,local))
        return original_record(self, tick, local)

    def review(self, env, snapshot, fields, actor, arm):
        t = snapshot['frame']
        assert arm == 'observed' and fields[2:] == (None,None)
        assert len(records) == t and (not t or records[-1][0] == t-1)
        assert snapshot['fleet_seed'] == case['fleet_seed']
        delivered[t] = snapshot['received_packets']
        car = snapshot['car_xy']
        # Request a stationary vehicle and a distinct aircraft task.
        survey = car + ([40.,0.] if t == 0 else [-40.,0.])
        self.commands.append(dict(issued_at_s=t, origin_xy=car.copy(), target=car.copy()))
        return dict(target_xy=car.tolist(), mode='wait_for_survey', intent_key='wait', actor_called=False,
                    survey_target_xy=survey.tolist())

    with patch.object(RejoinGuidance, 'review', review), patch.object(RejoinGuidance,'record_local',record), \
         patch('v91_guided_drive.model_fields', side_effect=AssertionError('Control arm called the CNN')):
        result = run_km_actor_arm(case, *runner.MODEL_STATE, 'observed', max_steps=31)
    rows, trace = result['reviews'], result['trace_motion']
    assert [r['applied_target_frames'] for r in rows] == [30,1]
    assert [r['step'] for r in rows] == [0,30]
    assert len(records) == 31 and len(trace) == 31 and len(result['car_path']) == 32
    assert all(s['review_step'] == (0 if s['step'] < 30 else 30) for s in trace)
    assert all(s['motion'] == (0.,0.) and s['distance_m'] == 0. for s in trace)
    assert all(s['vehicle_target'] == case['start'] for s in trace)
    assert all(s['broadcast_target'] == rows[s['step']//30]['survey_target_xy'] for s in trace)
    assert delivered[0] == 0 and delivered[30] > 0


if __name__ == '__main__':
    if sys.argv[1:2] == ['--probe']:
        integration_probe(sys.argv[2])
    else:
        unittest.main()
