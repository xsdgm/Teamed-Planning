import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock

import numpy as np

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from survey.survey_pretrain_env import ground_plan
from vtemp4.vtemp4_runtime import HeadingFleetEpisode, interval_reward, lookahead_marker


class HeadingFleetTests(unittest.TestCase):
    def test_tree_at_marker_keeps_safe_directional_route(self):
        cells = np.full((41, 41), -1, np.int8)
        cells[20, 20] = 1
        start, marker = (20, 5), (20, 20)
        route, _, reason = ground_plan(cells, start, marker, 1., directional=True)
        self.assertEqual(reason, 'directional_route')
        self.assertTrue((cells[tuple(route.T)] == -1).all())
        self.assertEqual(tuple(route[-1]), (20, 19))
        self.assertEqual(ground_plan(cells, start, marker, 1.)[2], 'hold')

    def test_marker_shortens_at_edge_along_same_bearing(self):
        position = np.array([970., 500.])
        goal = np.array([40., 50.])
        heading = math.radians(35.)
        marker = lookahead_marker(position, heading, goal, 200., 1000.)
        self.assertAlmostEqual(marker[0], 1000.)
        self.assertAlmostEqual(math.atan2(*(marker - position)[::-1]), heading)

    def test_detour_progress_telescopes_and_only_high_level_turn_costs(self):
        first = interval_reward(1200., 1230., 1200., True, -math.pi / 2, 15, 1200)
        second = interval_reward(1230., 2., 1200., False, 0., 585, 1200, 'arrival')
        self.assertAlmostEqual(first + second, 2. * (1198 - 1.) / 1200 - .5 + 10.)

    def test_turn_cost_on_one_kilometre_mission(self):
        hold = interval_reward(800., 800., 1000., False, 0., 15, 3000)
        self.assertAlmostEqual(hold, -15 / 3000)
        for degrees, penalty in ((-90, .002), (90, .002), (180, .003)):
            with self.subTest(degrees=degrees):
                turn = interval_reward(800., 800., 1000., True, math.radians(degrees), 15, 3000)
                self.assertAlmostEqual(hold - turn, penalty)

    def test_higher_progress_weight_only_scales_distance_change(self):
        for end_distance, reason, terminal_reward in ((780., None, 0.), (820., 'time_budget', 0.),
                                                     (800., 'safety_failure', -5.), (2., 'arrival', 10.)):
            with self.subTest(end_distance=end_distance, reason=reason):
                reward = interval_reward(800., end_distance, 1000., True, math.pi / 2, 30, 3000,
                                         reason, progress_reward_weight=8.)
                progress_reward = 8. * (800. - end_distance) / 1000.
                self.assertAlmostEqual(reward, progress_reward - .002 - 30 / 3000 + terminal_reward)

    def test_safety_before_first_physics_step_belongs_to_chosen_action(self):
        episode = HeadingFleetEpisode.__new__(HeadingFleetEpisode)
        episode.case = {'direct_distance_m': 1500.}
        episode.budget = 4500
        episode.rollout = []
        episode.heading_return = 0.
        episode.pending = dict(start=15, distance=1490., observation={}, turn=False, angle_rad=0.,
                               logp=0., value=.7, event={})
        episode._settle(15, 1490., 0., 'safety_failure')
        self.assertEqual(episode.rollout[0]['reward'], -5.)
        self.assertEqual(episode.rollout[0]['duration'], 0)
        self.assertTrue(episode.rollout[0]['terminated'])
        self.assertFalse(episode.rollout[0]['truncated'])
        episode._settle(15, 1490., 0., 'safety_failure')
        self.assertEqual(len(episode.rollout), 1)

    def test_route_gain_reaches_ppo_and_telescopes_through_unscorable_poses(self):
        episode = HeadingFleetEpisode.__new__(HeadingFleetEpisode)
        episode.case = {'direct_distance_m': 1000.}
        episode.progress_reward_weight = 10.
        episode.budget = 3000
        episode.rollout = []
        episode.heading_return = 0.
        episode._route_field = Mock()
        episode._route_field.distance_m.side_effect = [1060., None, 1080., 1080., 1100.]
        episode._route_last_scored_m = episode._route_last_raw_m = 1100.
        episode._route_unavailable_intervals = 0
        episode.position = lambda: [0., 0.]
        for i, gain in enumerate([40., 0., -20., 0., -20.]):
            # The straight-line goal distance increases in every interval.
            event = {}
            episode.pending = dict(start=30*i, distance=800.+i, observation={}, turn=False,
                                   angle_rad=0., logp=0., value=.3, event=event)
            reason = 'time_budget' if i == 4 else None
            episode._settle(30*(i+1), 801.+i, 0., reason)
            self.assertEqual(event['route_progress_m'], gain)
            self.assertAlmostEqual(episode.rollout[-1]['reward'], 10.*gain/1000.-30./3000.)
        self.assertTrue(episode.rollout[-1]['terminated'])
        self.assertEqual(episode._route_unavailable_intervals, 2)
        self.assertAlmostEqual(episode.heading_return, -150./3000.)

    def test_forced_continuous_turn_actually_guides_both_vehicle_and_aircraft(self):
        result = subprocess.run([sys.executable, '-I', '-B', __file__, 'native'], capture_output=True,
                                text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def native_check():
    import contextlib
    import os

    from vtemp4.vtemp4_runtime import run_heading_case

    class LeftThenHold:
        def __init__(self):
            self.calls = 0

        def act(self, obs, *, deterministic, generator):
            self.calls += 1
            turn = self.calls == 1
            return turn, math.radians(87.) if turn else 0., -.3, .75

        def value(self, obs):
            return .25

    repo = Path(__file__).resolve().parents[5]
    case = dict(seed=17, case_id='heading-km', kind='v9_random', start=[250., 250.], goal=[1350., 250.],
                world_size=1650., direct_distance_m=1100., circles=[])
    rollout = []
    with open(os.devnull, 'w') as quiet, contextlib.redirect_stdout(quiet):
        # Ends exactly on the third evaluation time. Only two decisions were executed.
        result = run_heading_case(repo, case, LeftThenHold(), fast_seed=24, budget_s=3420,
                                  collector_limit_s=30, rollout=rollout)
    assert result['end_reason'] == 'collector_limit' and result['sim_time_s'] == 30
    assert [r['time_s'] for r in result['heading_reviews']] == [0, 15]
    assert [r['duration'] for r in rollout] == [15, 15]
    assert rollout[-1]['truncated'] and not rollout[-1]['terminated'] and rollout[-1]['next_value'] == .25
    assert len(result['shared_markers']) >= 2
    for event in result['ground_replan_events']:
        offset = np.subtract(event['shared_marker'], event['position'])
        assert abs(math.atan2(offset[1], offset[0]) - math.radians(87.)) < 1e-7
    first = result['ground_replan_events'][0]
    assert first['target'][1] > first['position'][1] + 1., first
    marker = result['shared_markers'][0]
    assert marker['physics_broadcasts'] > 0 and marker['heard_by']
    assert result['travelled_m'] > 0. and result['adapter']['unique_received_tiles'] > 0
    assert abs(sum(row['reward'] for row in rollout) - result['reward']) < 1.e-8


if __name__ == '__main__':
    if sys.argv[1:] == ['native']:
        native_check()
    else:
        unittest.main()
