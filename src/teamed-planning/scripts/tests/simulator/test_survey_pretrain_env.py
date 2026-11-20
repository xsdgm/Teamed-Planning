"""Causal survey proxy, confirmed-route optimization and independent ground clock."""
import unittest

import numpy as np

from simulator.survey_pretrain_env import SurveyEnv, ground_plan, _sample_corridors


class SurveyPretrainEnvTests(unittest.TestCase):
    def make_env(self, **kwargs):
        return SurveyEnv(np.ones((90, 90), bool), 1., [0., 0.], [20.5, 20.5],
                         [70.5, 25.5], **kwargs)

    def test_shortest_known_route_detours_around_unknown(self):
        observed = np.zeros((8, 8), np.int8)
        observed[1:6, 1] = observed[1:6, 5] = -1
        observed[1, 1:6] = -1
        observed[4, 3] = 1
        path, _, reason = ground_plan(observed, (5, 1), (5, 5), 1.)
        self.assertEqual(reason, 'known_goal_route')
        self.assertEqual(len(path)-1, 12)
        self.assertTrue(np.all(observed[tuple(path.T)] == -1))
        np.testing.assert_array_equal(np.abs(np.diff(path, axis=0)).sum(1), 1)
        observed[1, 3] = 0
        path, _, reason = ground_plan(observed, (5, 1), (5, 5), 1.)
        self.assertEqual(reason, 'reachable_frontier')
        self.assertTrue(np.all(observed[tuple(path.T)] == -1))

    def test_hidden_changes_cannot_change_initial_policy_inputs(self):
        a = self.make_env()
        changed = np.ones((90, 90), bool)
        changed[30:60, 29:58] = False
        b = SurveyEnv(changed, 1., [0., 0.], [20.5, 20.5], [70.5, 25.5])
        for key in ('cells', 'source_times', 'ground_target', 'targets', 'heuristic_scores',
                    'ground_path', 'mask'):
            np.testing.assert_array_equal(a.observe()[key], b.observe()[key])

    def test_candidate_corridors_cannot_jump_obstructed_diagonal_corner(self):
        observed = np.zeros((64, 64), dtype=np.int8)
        anchor = np.array([22, 22])
        observed[22, 22] = -1
        observed[21, 22] = observed[23, 22] = observed[22, 21] = observed[22, 23] = 1
        self.assertEqual(_sample_corridors(observed, [anchor], np.array([40, 40]), 1.), [])

    def test_same_initial_question_accepts_an_observed_obstacle_or_passage(self):
        initial = self.make_env(seed=3).observe()
        y, x = np.floor(initial['targets'][0]).astype(int)[::-1]
        for free_at_probe, expected in [(True, 'passable'), (False, 'blocked')]:
            with self.subTest(expected=expected):
                free = np.ones((90, 90), bool)
                free[y, x] = free_at_probe
                env = SurveyEnv(free, 1., [0., 0.], [20.5, 20.5], [70.5, 25.5], seed=3)
                np.testing.assert_array_equal(initial['targets'], env.observe()['targets'])
                reward, _, result = env.step(0)
                self.assertEqual(result['answer'], expected)
                self.assertEqual(env.summary()[expected + '_answers'], 1)
                self.assertGreater(reward, 0.)
                self.assertLess(result['source_s'], result['receipt_s'])

    def test_unobserved_source_never_becomes_obstacle_or_free(self):
        valid = np.ones((90, 90), bool)
        valid[21, 21] = False
        valid[23:40, 27:40] = False
        env = self.make_env(sensor_valid=valid)
        env.step(0)
        self.assertTrue(np.all(env.cells[~valid] == 0))
        self.assertTrue(np.isnan(env.source_times[~valid]).all())

    def test_long_flight_replans_before_previous_arrival_and_receives_before_clock_tick(self):
        env = self.make_env(ground_replan_interval_s=5)
        # Stretch one physical survey flight across many ground planning ticks.
        obs = env.observe()
        target = obs['targets'][0]
        flight_seconds = 19
        env.air_position = target - [flight_seconds*6., 0.]
        class OneSecondLatency:
            def integers(self, low, high):
                return 1
        env.rng = OneSecondLatency()
        _, _, info = env.step(0)
        self.assertEqual(info['duration_s'], 20)
        events = info['ground_replan_events']
        self.assertEqual([e['time_s'] for e in events], [5, 10, 15, 20])
        self.assertIsNone(events[0]['latest_survey_receipt_s'])
        self.assertEqual(events[-1]['latest_survey_receipt_s'], 20)
        self.assertTrue(any(e['previous_target_reached'] is False for e in events))
        self.assertEqual(env.summary()['ground_replans'], 5)
        np.testing.assert_array_equal(env.observe()['ground_target'], events[-1]['target'])

    def test_short_known_goal_has_no_spurious_survey_reward(self):
        env = SurveyEnv(np.ones((15, 15), bool), 1., [0., 0.],
                        [5.5, 5.5], [8.5, 5.5], ground_replan_interval_s=2)
        self.assertTrue(env.observe()['no_request'])
        reward, done, info = env.step(0)
        self.assertTrue(done)
        self.assertTrue(env.summary()['goal_reached'])
        self.assertEqual(env.time, 3)
        self.assertEqual(info['ground_replans'], 1)
        self.assertIsNone(info['survey_target'])
        self.assertIsNone(info['receipt_s'])
        self.assertEqual(reward, -.003)

    def test_time_budget_stops_in_flight_without_receiving_future_packet(self):
        env = self.make_env(max_time_s=5, max_steps=20)
        env.air_position = np.array([150., 160.])
        _, done, info = env.step(0)
        self.assertTrue(done)
        self.assertEqual(env.time, 5)
        self.assertEqual(info['duration_s'], 5)
        self.assertIsNone(info['receipt_s'])
        self.assertEqual(info['useful_new_cells'], 0)
        self.assertEqual(env.summary()['end_reason'], 'time_budget')
        self.assertEqual([e['time_s'] for e in env.ground_replan_events], [0, 5])

    def test_local_actor_window_has_global_alignment_and_preserves_ground_plan(self):
        args = (np.ones((400, 1400), bool), 1., [11., 13.], [240.5, 200.5], [1320.5, 340.5])
        full = SurveyEnv(*args)
        local = SurveyEnv(*args, local_window=True)
        for _ in range(3):
            wide, small = full.observe(), local.observe()
            off = ((small['origin'] - wide['origin']) / wide['resolution']).astype(int)
            self.assertTrue((off % 8 == 0).all())
            self.assertLess(small['cells'].shape[1], wide['cells'].shape[1])
            h, w = small['cells'].shape
            np.testing.assert_array_equal(small['cells'], wide['cells'][off[1]:off[1]+h, off[0]:off[0]+w])
            for key in ('targets', 'goal', 'ground_target', 'ground_path'):
                np.testing.assert_array_equal(small[key], wide[key])
            self.assertEqual(full.step(0), local.step(0))


if __name__ == '__main__':
    unittest.main()
