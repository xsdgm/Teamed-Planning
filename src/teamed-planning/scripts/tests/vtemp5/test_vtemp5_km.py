"""Independent checks on the full trip, collision map and public obstacle reports."""
import math
from unittest import TestCase

from vtemp5.vtemp5_env import rule_offset
from vtemp5.vtemp5_km import VTemp5KilometerEnv


class KilometerNavigationTests(TestCase):
    def test_requested_distance_is_the_physical_start_to_goal_distance(self):
        for seed in (0, 3, 9):
            env = VTemp5KilometerEnv(seed, terrain='detour')
            view = env.observe()
            direct = math.dist(view['position'], view['goal'])
            self.assertGreaterEqual(direct, 1000.)
            self.assertLessEqual(direct, 1500.)
        env = VTemp5KilometerEnv(3, terrain='bridge', min_distance_m=2000., max_distance_m=2000.)
        view = env.observe()
        self.assertAlmostEqual(math.dist(view['position'], view['goal']), 2000.)

    def test_stopped_vehicle_times_out_with_the_full_distance_remaining(self):
        env = VTemp5KilometerEnv(29, terrain='bridge', step_m=0., horizon=4)
        first = env.observe()
        for tick in range(1, 5):
            view, _, done, info = env.step(0.)
            self.assertEqual(view['position'], first['position'])
            self.assertAlmostEqual(info['remaining_m'], math.dist(first['position'], first['goal']))
            self.assertEqual(info['travelled_m'], 0.)
            self.assertEqual(info['obstacles_passed'], 0)
            self.assertFalse(info['success'])
            self.assertEqual(done, tick == 4)

    def test_continuous_rule_drive_passes_two_obstacles_then_reaches_the_goal(self):
        # The open-side configurations exercise both easy passage and a necessary reversal.
        for terrain, blocked in [('detour', None), ('bridge', -1), ('bridge', 1)]:
            with self.subTest(terrain=terrain, blocked=blocked):
                env = VTemp5KilometerEnv(19, terrain=terrain, blocked_side=blocked,
                                         min_distance_m=1200., max_distance_m=1200.,
                                         wait_probability=0.)
                start = view = env.observe()
                first_wall_x = next(x for x in range(0, 1200, 2) if not env.terrain_traversable(x, 0.))
                travelled = 0.
                reported_ids = []
                passed = 0
                for tick in range(1, env.horizon + 1):
                    before = view
                    view, _, done, info = env.step(rule_offset(before))
                    segment = math.dist(before['position'], view['position'])
                    travelled += segment
                    self.assertLessEqual(segment, env.step_m + 1e-7)
                    self.assertEqual(view['elapsed_steps'], tick)
                    self.assertEqual(len(view['history']), tick)
                    last = view['history'][-1]
                    self.assertEqual(last['position_before'], before['position'])
                    self.assertEqual(last['position_after'], view['position'])
                    self.assertEqual(view['goal'], start['goal'])
                    self.assertAlmostEqual(info['remaining_m'], math.dist(view['position'], view['goal']))
                    if info['side_switch']:
                        self.assertEqual(before['side_status'][before['locked_side']], 'blocked')
                    self.assertFalse(info['unauthorized_switch'])
                    if view['detected'] and view['obstacle_id'] not in reported_ids:
                        reported_ids.append(view['obstacle_id'])
                        self.assertEqual(view['locked_side'], 0)
                        self.assertEqual(set(view['side_status'].values()), {'unknown'})
                    if info['obstacles_passed'] != passed:
                        self.assertEqual(info['obstacles_passed'], passed + 1)
                        self.assertFalse(done)
                        self.assertFalse(info['success'])
                        self.assertGreater(info['remaining_m'], info['arrival_radius_m'])
                        passed += 1
                    if done:
                        break
                self.assertTrue(info['success'])
                self.assertEqual(reported_ids, [1, 2])
                self.assertEqual(passed, 2)
                self.assertAlmostEqual(info['goal_distance_m'], 1200.)
                self.assertEqual(info['arrival_radius_m'], 2.)
                self.assertLessEqual(info['remaining_m'], 2.)
                self.assertFalse(env.terrain_traversable(first_wall_x, 0.))
                self.assertGreater(travelled, 1200.)
                # A collision slide can make a driven substep path longer than its chord.
                self.assertGreaterEqual(info['travelled_m'] + 1e-6, travelled)
                self.assertLessEqual(info['travelled_m'], env.step_m * tick + 1e-6)

    def test_opposite_bridge_locations_cannot_change_earlier_observations(self):
        options = dict(terrain='bridge', wait_probability=0.)
        pair = [VTemp5KilometerEnv(24, blocked_side=side, **options) for side in (1, -1)]
        views = [env.observe() for env in pair]
        while True:
            self.assertEqual(views[0], views[1])
            if views[0]['obstacle_id'] == 2:
                self.assertEqual(views[0]['side_status'], {1: 'unknown', -1: 'unknown'})
                break
            action = rule_offset(views[0])
            steps = [env.step(action) for env in pair]
            self.assertFalse(any(result[2] for result in steps), 'Did not receive second obstacle report')
            views = [result[0] for result in steps]

    def test_ground_geometry_blocks_the_centre_and_opens_only_real_passages(self):
        # Locate walls independently from vehicle sensors and obstacle counters.
        for terrain, closed in [('detour', None), ('bridge', 1), ('bridge', -1)]:
            env = VTemp5KilometerEnv(33, terrain=terrain, blocked_side=closed)
            gx = env.observe()['goal'][0]
            bands = []
            for x in range(0, int(gx), 2):
                if not env.terrain_traversable(x, 0.):
                    if not bands or x != bands[-1][-1] + 2:
                        bands.append([])
                    bands[-1].append(x)
            self.assertEqual(len(bands), 2)
            for index, band in enumerate(bands):
                self.assertGreater(band[-1] - band[0], 40.)
                x = band[len(band) // 2]
                open_sides = {side for side in (1, -1)
                              if any(env.terrain_traversable(x, side * y) for y in range(1, 225))}
                expected = {-closed} if terrain == 'bridge' and index == 1 else {1, -1}
                self.assertEqual(open_sides, expected)
