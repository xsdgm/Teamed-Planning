from contextlib import nullcontext, redirect_stdout
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vtemp4.vtemp4_ring_heading import RingHeadingFleetEpisode


class ScriptedAngle:
    def __init__(self, angles):
        self.angles = iter(angles)

    def act(self, obs, **kwargs):
        angle = next(self.angles)
        return angle is not None, angle or 0., -.2, .3

    def value(self, obs):
        return .4


class RingHeadingTests(unittest.TestCase):
    def episode(self, actions):
        ep = RingHeadingFleetEpisode.__new__(RingHeadingFleetEpisode)
        own = {'orientation': math.pi / 2, 'position': np.array([500., 500.])}
        ep.adapter = SimpleNamespace(own=own, resolution=1., cells=np.full((1000,1000), -1, np.int8),
                                     last_ground_reason='', observe_receipts=lambda: None, ep=ep)
        ep.case = {'world_size': 1000., 'direct_distance_m': 400.}
        ep.position = lambda: own['position'].copy()
        ep.goal = np.array([900., 500.])
        ep.limit = ep.budget = 900
        ep.review_interval, ep.ground_interval, ep.lookahead_m = 15, 5, 200.
        ep.current_request = ep.pending = ep.last_review = ep._last_control_time = None
        ep.heading_reviews, ep.requests, ep.ground_events, ep.rollout = [], [], [], []
        ep._heading_observation = lambda: {}
        ep.slow = ScriptedAngle(actions)
        ep.generator = None
        ep.heading_return = ep.held_since = 0.
        ep.final_approach = False
        ep.env = SimpleNamespace(env=SimpleNamespace(time=0))
        ep.torch = SimpleNamespace(inference_mode=nullcontext)
        ep._native_filter = lambda g, s: g
        ep._ground_decision = lambda now, guidance, wait_for_guidance: ep.ground_events.append(
            dict(time_s=now, guidance=guidance, waiting=wait_for_guidance))
        return ep, own

    def test_hold_after_rotation_and_translation_then_reissue_using_new_pose(self):
        ep, own = self.episode([None, None, -math.pi/2])
        ep._after_fusion(None, None)
        first = ep.current_request
        np.testing.assert_allclose(first['target'], [500., 700.])
        for t in range(1, 30):
            ep.env.env.time = t
            own['orientation'] = -math.pi/2
            own['position'] += [1., 2.]
            ep._after_fusion(None, None)
            self.assertIs(ep.current_request, first)
        own['orientation'] = math.pi
        ep.env.env.time = 30
        ep._after_fusion(None, None)
        selected = ep.current_request
        np.testing.assert_allclose(selected['origin_xy'], own['position'])
        np.testing.assert_allclose(selected['target'], own['position'] + [0., 200.])
        self.assertEqual([r['changed'] for r in ep.heading_reviews], [True, False, True])
        self.assertEqual(selected['relative_angle_rad'], -math.pi/2)
        self.assertEqual(selected['issued_at_s'], 30)
        # The local controller enters final approach between high-level reviews.
        ep.env.env.time = 31
        own['position'] = ep.goal - [8., 0.]
        ep._after_fusion(None, None)
        self.assertIs(ep.current_request, selected)
        self.assertTrue(ep.ground_events[-1]['final_approach'])
        self.assertIsNone(ep.ground_events[-1]['guidance'])
        self.assertEqual(ep.ground_events[-1]['shared_marker'], selected['target'])

    def test_out_of_bounds_skips_issue_preserves_sample_and_never_shortens_radius(self):
        ep, own = self.episode([None, math.pi, 0.])
        own['position'], own['orientation'] = np.array([990., 700.]), 0.
        ep._after_fusion(None, None)
        self.assertIsNone(ep.current_request)
        self.assertTrue(ep.ground_events[-1]['waiting'])
        self.assertIsNone(ep.ground_events[-1]['shared_marker'])
        ep.env.env.time = 15
        ep._after_fusion(None, None)
        only = ep.current_request
        np.testing.assert_allclose(only['target'], [790., 700.])
        ep.env.env.time = 30
        ep._after_fusion(None, None)
        self.assertIs(ep.current_request, only)
        self.assertEqual(len(ep.requests), 1)
        self.assertTrue(ep.heading_reviews[-1]['turn'])
        self.assertEqual(ep.heading_reviews[-1]['reason'], 'invalid_proposal')
        self.assertEqual(ep.pending['logp'], -.2)

    def test_native_held_point_reaches_fleet_broadcast_and_ground_planner(self):
        result = subprocess.run([sys.executable, '-IB', __file__, 'native'], capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def native_check():
    root = Path(__file__).resolve().parents[6]
    case = dict(seed=17, case_id='fixed-ring-heading', kind='v9_random', start=[500., 500.],
                goal=[1350., 500.], world_size=1650., direct_distance_m=850., circles=[])
    with open(os.devnull, 'w') as sink, redirect_stdout(sink):
        ep = RingHeadingFleetEpisode(root, case, ScriptedAngle([.7, None, -.9]), fast_seed=24,
                                     budget_s=2670, collector_limit_s=32, rollout=[])
        try:
            while not ep.finished:
                ep.step()
            record = ep.fleet_summary()
        finally:
            ep.close()
    requests = record['shared_markers']
    assert [r['issued_at_s'] for r in requests] == [0, 30], requests
    for r in requests:
        angle = r['vehicle_yaw_rad'] + r['relative_angle_rad']
        np.testing.assert_allclose(r['target'], np.asarray(r['origin_xy']) + 200.*np.array([math.cos(angle), math.sin(angle)]))
    assert requests[0]['physics_broadcasts'] == 30 and requests[0]['heard_by']
    assert record['travelled_m'] > 0
    assert record['heading_reviews'][1]['reason'] == 'hold'
    for g in record['ground_replan_events']:
        assert g['shared_marker'] == requests[int(g['time_s'] >= 30)]['target']
    assert ep.rollout[-1]['truncated'] and ep.rollout[-1]['next_value'] == .4


if __name__ == '__main__':
    if sys.argv[1:] == ['native']:
        native_check()
    else:
        unittest.main()
