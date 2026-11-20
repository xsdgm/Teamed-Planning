import unittest

import numpy as np

from simulator.vtemp4_stage_eval import bridge_metrics, group_metrics
from simulator.vtemp4_lake_metrics import lake_metrics


class StageEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.cover = np.zeros((6, 8), dtype=np.uint8)
        self.cover[:, 3:5] = 4
        self.cover[1, 3:5] = 6
        self.start = [1.5, 1.5]
        self.goal = [6.5, 1.5]

    def scores(self, path):
        return bridge_metrics(path, self.cover, np.array([0., 0.]), 1., self.start, self.goal, [4., 1.5])

    def test_crossing_requires_traversing_deck_between_banks(self):
        crossing = self.scores([self.start, self.goal])
        self.assertTrue(crossing['river_crossed_via_bridge'])
        self.assertLess(crossing['bridge_deck_first_s'], crossing['opposite_bank_first_s'])
        self.assertFalse(crossing['water_entered'])
        deck_only = self.scores([self.start, [3.5, 1.5], self.start])
        self.assertIsNotNone(deck_only['bridge_deck_first_s'])
        self.assertFalse(deck_only['river_crossed_via_bridge'])
        through_water = self.scores([self.start, [3.5, 1.5], self.start, [1.5, 3.5], [6.5, 3.5]])
        self.assertFalse(through_water['river_crossed_via_bridge'])
        self.assertTrue(through_water['water_entered'])
        self.assertIsNotNone(through_water['opposite_bank_first_s'])

    def test_static_track_and_truncated_reporting(self):
        row = self.scores([self.start])
        self.assertIsNone(row['bridge_deck_first_s'])
        self.assertFalse(row['river_crossed_via_bridge'])
        row.update(navigation_task='detour_left', end_reason='collector_limit', goal_reached=False,
                   remaining_m=1000, turn_count=0, decisions=1, left_turns=0, right_turns=0)
        summary = group_metrics([row], [row])['detour_left']
        self.assertIsNone(summary['bridge_crossing_rate'])
        self.assertIsNone(summary['success_rate'])


class LakeEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.cover = np.zeros((17, 21), dtype=np.uint8)
        self.cover[5:12, 5:11] = 4
        self.start, self.goal = [1.5, 8.5], [19.5, 8.5]
        self.audit = dict(lake_gate_station_m=6.5, lake_exit_station_m=9.5,
                          gate_left_shore_offset_m=3.5, gate_right_shore_offset_m=-3.5)

    def score(self, *points, direct=False):
        return lake_metrics([self.start, *points], self.cover, np.zeros(2), 1.,
                            self.start, self.goal, self.audit, direct)

    def test_valid_bypass_on_either_side(self):
        for y, side in ((2.5, 'right'), (14.5, 'left')):
            row = self.score([2.5, y], [14., y])
            self.assertTrue(row['lake_bypassed'])
            self.assertEqual(row['lake_bypass_side'], side)
            self.assertLess(row['lake_central_cut_first_s'], row['lake_bypass_first_s'])
            self.assertFalse(row['water_entered'])

    def test_cut_alone_wrong_way_and_illegal_sweep(self):
        self.assertFalse(self.score([2.5, 14.5], [9., 14.5])['lake_bypassed'])
        recross = self.score([2.5, 14.5], [9., 14.5], [2.5, 14.5], [2.5, 2.5], [14., 2.5])
        self.assertEqual(recross['lake_bypass_side'], 'right')
        illegal = self.score([2.5, 14.5], [9., 14.5], [9., 8.5], self.goal)
        self.assertIsNotNone(illegal['far_end_station_first_s'])
        self.assertFalse(illegal['reached_lake_far_end'])
        self.assertTrue(illegal['water_entered'])
        self.assertFalse(illegal['lake_bypassed'])
        # A single long movement from dry to dry cuts only a tiny corner of a water cell.
        corner = self.score([4., 5.99], [6., 4.015])
        self.assertTrue(corner['water_entered'])

    def test_direct_control_counts_turns_and_deviation(self):
        self.cover.fill(0)
        self.cover[12:16, 5:11] = 4
        self.audit.update(gate_left_shore_offset_m=7.5, gate_right_shore_offset_m=3.5)
        row = self.score(self.goal, direct=True)
        self.assertTrue(row['reached_lake_far_end'])
        self.assertNotIn('lake_bypassed', row)
        self.assertEqual(row['max_lateral_deviation_m'], 0.)
        row.update(navigation_task='direct', end_reason='arrival', goal_reached=True, remaining_m=0.,
                   turn_count=2, decisions=6, left_turns=1, right_turns=1)
        group = group_metrics([row], [row])['direct']
        self.assertEqual(group['turns_on_direct_controls'], 2)
        self.assertNotIn('lake_bypass_rate', group)


if __name__ == '__main__':
    unittest.main()
