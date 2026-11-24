"""Test the goal override at the periodic decision boundary and on native broadcasts."""
from collections import deque
from contextlib import redirect_stdout
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vtemp4.vtemp4_ring_heading import RingHeadingFleetEpisode
from vtemp4.vtemp4_unified_attention import GoalAwareUnifiedAttentionEpisode
from vtemp4.vtemp4_vehicle_graph import VehicleGraphHistoryLSTMEpisode


class GoalDesignationTest(unittest.TestCase):
    def test_waypoint_bonus_uses_issued_point_and_real_interval_progress(self):
        # At 1 km, this angle puts a 200 m waypoint exactly 160 m nearer the goal.
        oblique = math.acos((1000.**2 + 200.**2 - 840.**2) / (2 * 1000. * 200.))
        cases = [
            ('oblique', .1, 1000., 995., oblique, .8, .004),
            ('straight', .1, 1000., 995., 0., 1., .005),
            ('stopped', .1, 1000., 1000., 0., 1., 0.),
            ('retreated', .1, 1000., 1005., 0., 1., 0.),
            ('pointed away', .1, 1000., 995., math.pi, 0., 0.),
            ('rule assigned final goal', .1, 190., 185., 0., 0., 0.),
            ('disabled', 0., 1000., 995., 0., 0., 0.),
        ]
        for name, ratio, before, after, angle, score, bonus in cases:
            with self.subTest(case=name):
                ep = VehicleGraphHistoryLSTMEpisode.__new__(VehicleGraphHistoryLSTMEpisode)
                ep.case = dict(direct_distance_m=1000.)
                ep.goal = np.array([1100., 100.])
                ep.progress_reward_weight, ep.waypoint_progress_bonus_ratio = 10., ratio
                ep.lookahead_m, ep.budget, ep.heading_return = 200., 3000, 0.
                ep.adapter = SimpleNamespace(own=dict(orientation=0.))
                ep._heading_observation = lambda: {'all_ring_endpoints_blocked': False}
                ep.requests, ep.heading_reviews, ep.rollout = [], [], []
                ep.pending = ep.current_request = ep.generator = None
                ep.torch, ep.slow = torch, Mock()
                ep.slow.act.return_value = (True, angle, -.4, .2)
                ep.slow.value.return_value = .2
                ep._choose_heading(0, ep.goal - [before, 0.], before)
                ep._settle(30, after, 0.)

                expected = 10. * (before - after) / 1000. - 30. / 3000. + bonus
                event, transition = ep.heading_reviews[0], ep.rollout[0]
                self.assertAlmostEqual(event['reward'], expected)
                self.assertEqual(transition['reward'], event['reward'])
                self.assertEqual(ep.heading_return, event['reward'])
                if ratio:
                    self.assertAlmostEqual(event['waypoint_progress_score'], score)
                    self.assertAlmostEqual(event['waypoint_progress_bonus'], bonus)
                else:
                    self.assertNotIn('waypoint_progress_bonus', event)
                    self.assertNotIn('waypoint_progress_score', event)

    def test_distance_checked_only_when_issuing_a_waypoint(self):
        ep = GoalAwareUnifiedAttentionEpisode.__new__(GoalAwareUnifiedAttentionEpisode)
        ep.env = SimpleNamespace(env=SimpleNamespace(time=0))
        ep.goal = np.array([500., 500.])
        ep.case = dict(world_size=1000., direct_distance_m=201.)
        ep.adapter = SimpleNamespace(own={}, ground_target=None, last_ground_reason='', observe_receipts=lambda: None)
        ep.position = lambda: ep.adapter.own['position']
        ep.samples = deque(maxlen=61)
        ep.slow = Mock()
        ep.slow.act.return_value = (True, .45, -.4, .2)
        ep.slow.value.return_value = .3
        ep.torch, ep.generator, ep.rollout = torch, None, []
        ep.limit = ep.budget = 500
        ep.pending = ep.last_review = ep.current_request = ep._last_control_time = None
        ep.requests, ep.heading_reviews, ep.ground_events = [], [], []
        ep.held_since = 0
        ep.final_approach = False
        ep.review_interval, ep.ground_interval, ep.lookahead_m, ep.heading_return = 15, 5, 200., 0.
        ep._native_filter = lambda graph, semantic: graph
        ep._ground_decision = lambda now, guidance: ep.ground_events.append(dict(time_s=now, guidance=guidance))

        # Enter the radius at t=2, stand exactly on it at t=15, exit at t=34,
        # then come back within range one second after the t=45 designation.
        knots = (0, 3, 15, 30, 35, 45, 46, 47)
        distances = (201., 199., 200., 195., 203., 200.001, 198., 196.)
        issued_targets = []
        outside_arcs = (np.array([[-.1, .6]]), False)
        with patch.object(RingHeadingFleetEpisode, '_heading_observation', side_effect=dict), \
                patch('simulator.vtemp4_periodic_history.legal_ring_arcs', return_value=outside_arcs) as arcs:
            for t in range(48):
                ep.env.env.time = t
                distance = np.interp(t, knots, distances)
                ep.adapter.own.update(position=ep.goal - [distance, 0.], orientation=.9 + .01*t)
                ep._after_fusion(None, None)
                issued_targets.append(tuple(ep.current_request['target']))
            ep._end('collector_limit')
        self.assertEqual(arcs.call_count, 2)
        self.assertEqual(ep.slow.act.call_count, 2)
        self.assertEqual([r['issued_at_s'] for r in ep.requests], [0, 15, 30, 45])
        self.assertEqual([r['id'] for r in ep.requests], [0, 1, 2, 3])
        self.assertEqual([r['designation_source'] for r in ep.requests],
                         ['policy_angle', 'final_goal', 'final_goal', 'policy_angle'])
        for t, xy in enumerate(issued_targets):
            self.assertEqual(xy, tuple(ep.requests[t // 15]['target']))
        for row, marker in zip(ep.rollout, ep.requests):
            t = marker['issued_at_s']
            rule = t in (15, 30)
            distance = np.interp(t, knots, distances)
            yaw = .9 + .01*t
            self.assertEqual(marker['radius_m'], distance if rule else 200.)
            self.assertAlmostEqual(marker['relative_angle_rad'], -yaw if rule else .45)
            self.assertEqual(row['policy_controlled'], not rule)
            if rule:
                self.assertEqual(marker['target'], ep.goal.tolist())
                self.assertEqual(row['observation']['angle_arcs'].shape, (0, 2))
                self.assertEqual(row['logp'], 0.)
            else:
                expected = np.array(marker['origin_xy']) + 200. * np.array([math.cos(yaw+.45), math.sin(yaw+.45)])
                np.testing.assert_allclose(marker['target'], expected)
            end_distance = np.interp(t + row['duration'], knots, distances)
            # The first choice is free. At t=45 the policy turns away from the last
            # rule-issued bearing (due east) by yaw + its sampled angle.
            cost_m = .5 + (yaw + .45) / math.pi if t == 45 else 0.
            expected_reward = 2. * (distance - end_distance - cost_m) / 201. - row['duration'] / ep.budget
            self.assertAlmostEqual(row['reward'], expected_reward)
        self.assertEqual([r['waypoint_id'] for r in ep.ground_events], [t // 15 for t in range(0, 48, 5)])

    def test_periodic_turn_cost_uses_world_bearings_and_keeps_sampled_actions(self):
        ep = GoalAwareUnifiedAttentionEpisode.__new__(GoalAwareUnifiedAttentionEpisode)
        ep.goal = np.array([1000., 0.])
        ep.case = dict(direct_distance_m=1000.)
        ep.budget, ep.lookahead_m, ep.heading_return = 120, 200., 0.
        ep.env = SimpleNamespace(env=SimpleNamespace(time=0))
        ep.adapter = SimpleNamespace(own={})
        ep.position = lambda: ep.adapter.own['position']
        ep._heading_observation = lambda: {'all_ring_endpoints_blocked': False}
        ep.requests, ep.heading_reviews, ep.rollout = [], [], []
        ep.pending = ep.current_request = ep.generator = None
        ep.torch, ep.slow = torch, Mock()
        # Issued bearings are +179, -181 (= +179), -179 and -89 degrees.
        yaws = [5, -135, -120, -30]
        angles = [math.radians(a) for a in (174, -46, -59, -59)]
        ep.slow.act.side_effect = [(True, a, -.4, .2) for a in angles]
        distances = []
        for i, yaw in enumerate(yaws):
            pos = np.array([0., 5. * i])
            ep.adapter.own.update(position=pos, orientation=math.radians(yaw))
            distances.append(float(np.linalg.norm(ep.goal - pos)))
            ep._choose_heading(30 * i, pos, distances[-1])
        ep.env.env.time = ep.budget
        ep._end('time_budget')
        distances.append(distances[-1])
        for i, delta_deg in enumerate((0., 0., 2., 90.)):
            row = ep.rollout[i]
            self.assertTrue(row['turn'])
            self.assertEqual(row['angle_rad'], angles[i])
            self.assertEqual(row['logp'], -.4)
            self.assertAlmostEqual(ep.heading_reviews[i]['switch_angle_rad'], math.radians(delta_deg))
            cost_m = .5 + delta_deg / 180. if delta_deg else 0.
            expected = 2. * (distances[i] - distances[i+1] - cost_m) / 1000. - 30. / 120.
            self.assertAlmostEqual(row['reward'], expected)
            self.assertEqual(row['terminated'], i == 3)
            self.assertFalse(row['truncated'])
        self.assertEqual(ep.rollout[-1]['next_value'], 0.)

    def test_frozen_v9_broadcasts_goal_on_every_tick(self):
        run = subprocess.run([sys.executable, '-IB', __file__, 'native'], capture_output=True, text=True, timeout=90)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


def native_check():
    from vtemp2_3.frozen_navigation import runtime
    from vtemp4.vtemp4_unified_attention import GoalAwareUnifiedAttentionPolicy
    repo = Path(__file__).resolve().parents[5]
    runtime(repo)
    torch.manual_seed(17)
    policy = GoalAwareUnifiedAttentionPolicy().eval()
    case = dict(seed=17, case_id='goal-designation', kind='v9_random', start=[500., 500.], goal=[660., 500.],
                world_size=1650., direct_distance_m=160., circles=[])
    with open(os.devnull, 'w') as sink, redirect_stdout(sink), torch.inference_mode():
        ep = GoalAwareUnifiedAttentionEpisode(repo, case, policy, fast_seed=24, budget_s=100, collector_limit_s=16)
        try:
            while not ep.finished:
                ep.step()
            summary = ep.fleet_summary()
        finally:
            ep.close()
    markers = summary['shared_markers']
    assert summary['end_reason'] == 'collector_limit'
    assert summary['final_goal_xy'] == case['goal']
    assert [m['issued_at_s'] for m in markers] == [0, 15]
    assert [m['physics_broadcasts'] for m in markers] == [15, 1]
    assert all(m['target'] == case['goal'] and m['designation_source'] == 'final_goal' for m in markers)
    assert all(e['shared_marker'] == case['goal'] for e in summary['ground_replan_events'])


if __name__ == '__main__':
    if sys.argv[1:] == ['native']:
        native_check()
    else:
        unittest.main()
