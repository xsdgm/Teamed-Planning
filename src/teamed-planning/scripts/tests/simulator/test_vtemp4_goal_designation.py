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
from simulator.vtemp4_ring_heading import RingHeadingFleetEpisode
from simulator.vtemp4_runtime import interval_reward
from simulator.vtemp4_unified_attention import GoalAwareUnifiedAttentionEpisode


class GoalDesignationTest(unittest.TestCase):
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
            expected_reward = interval_reward(distance, end_distance, 201., not rule, row['angle_rad'],
                                              row['duration'], ep.budget)
            self.assertAlmostEqual(row['reward'], expected_reward)
        self.assertEqual([r['waypoint_id'] for r in ep.ground_events], [t // 15 for t in range(0, 48, 5)])

    def test_frozen_v9_broadcasts_goal_on_every_tick(self):
        run = subprocess.run([sys.executable, '-IB', __file__, 'native'], capture_output=True, text=True, timeout=90)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


def native_check():
    from world_model.frozen_navigation import runtime
    from simulator.vtemp4_unified_attention import GoalAwareUnifiedAttentionPolicy
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
