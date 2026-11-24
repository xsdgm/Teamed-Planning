import unittest
from types import SimpleNamespace

import numpy as np

from vtemp5.vtemp5_circle_pretrain_rollout import _teacher, pretrain_plans


class CircleImitationTests(unittest.TestCase):
    def episode(self, position):
        tracker = SimpleNamespace(res=1., anchor=np.array([70., 100.]), left=np.array([0., 1.]),
                                  forward=np.array([1., 0.]),
                                  _component=lambda cells, anchor, reach: np.array([[75., 100.], [89., 106.]]))
        routes = {1: np.array([[80., 120.], [120., 120.], [139., 100.]]),
                  -1: np.array([[80., 80.], [120., 80.], [139.5, 100.]])}
        state = dict(active=True, locked_side=1, routes=routes, side_status={1:'unknown', -1:'unknown'})
        adapter = SimpleNamespace(cells=-np.ones((300, 300), dtype=np.int8), resolution=1., tiles=3,
                                  own={'orientation': 0.})
        ep = SimpleNamespace(adapter=adapter, position=lambda: np.asarray(position),
                             obstacle_state=state, tracker=tracker)
        return ep, dict(goal=np.array([0., 1., 1.]), angle_arcs=np.array([[-np.pi, np.pi]]))

    def test_measured_shared_exit_reconnects_without_reversing_toward_merge_point(self):
        ep, obs = self.episode([150., 100.])
        angle, label = _teacher(ep, obs, had_active=True)
        self.assertEqual(label, 'return_to_goal')
        self.assertAlmostEqual(angle, 0.)
        self.assertEqual(ep.obstacle_state['locked_side'], 1)

    def test_foreground_or_unknown_corridor_does_not_authorize_rejoin(self):
        ep, obs = self.episode([76., 100.])
        self.assertEqual(_teacher(ep, obs, had_active=True)[1], 'maintain_side')
        ep, obs = self.episode([150., 100.])
        ep.adapter.cells[:, 157:] = 0
        self.assertIsNone(_teacher(ep, obs, had_active=True))
        ep.adapter.cells[:, 157:] = -1
        ep.tracker._component = lambda cells, anchor, reach: np.array([[160., 100.]])
        label = _teacher(ep, obs, had_active=True)
        self.assertTrue(label is None or label[1] != 'return_to_goal')

    def test_unobserved_forward_motion_and_absence_of_earlier_encounter(self):
        ep, obs = self.episode([150., 100.])
        ep.obstacle_state = dict(active=False)
        self.assertEqual(_teacher(ep, obs, had_active=False)[1], 'clear_forward')
        ep.adapter.tiles = 0
        self.assertIsNone(_teacher(ep, obs, had_active=False))

    def test_start_strata_have_distinct_geometry_and_begin_before_far_side(self):
        train = pretrain_plans('ppo_train', 6, 913)
        dev = pretrain_plans('development', 6, 913)
        self.assertEqual(len({row['case']['map_sha256'] for row in train + dev}), 12)
        self.assertEqual({row['start_kind'] for row in train}, {'approach', 'flank', 'exit_approach'})
        for row in train + dev:
            case = row['case']
            start = np.array(case['start'])
            axis = (case['goal'] - start) / case['direct_distance_m']
            center, radius = case['circles'][0]
            self.assertGreater(np.linalg.norm(start-center), radius + 1.5)
            self.assertLess((start-center) @ axis, radius)
