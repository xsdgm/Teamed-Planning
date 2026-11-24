import unittest

import numpy as np

from vtemp5.vtemp5_fleet import VTemp5FleetEpisode, add_candidate_roads
from vtemp5.vtemp5_policy import _road_graph


class FleetPolicyAdapterTests(unittest.TestCase):
    def test_native_roads_and_model_hypotheses_keep_separate_edges(self):
        position, goal = np.zeros(2), np.array([1200., 0.])
        route = np.array([[3., 1.], [25., 1.], [40., 0.]])
        public = dict(goal=goal, elapsed_steps=2, routes={1:route},
                      route_observed={1:[True, True, False]}, side_status={1:'open'},
                      route_open_probability={1:.75}, evidence_time=30.)
        obs = _road_graph(public, position)
        original = {k:v.copy() for k,v in obs.items()}
        n = len(obs['road_valid'])
        add_candidate_roads(obs, public, position, goal, 60.)
        for k in original:
            np.testing.assert_array_equal(obs[k][:n], original[k])
        self.assertEqual(obs['road_valid'][n:].tolist(), [True]*3+[False]*3)
        self.assertFalse(obs['road_entry_mask'][n:].any())
        self.assertEqual(obs['road_nodes'][n+2,4], 1.)
        self.assertAlmostEqual(obs['road_nodes'][n,3], np.log1p(30.), places=5)
        neighbors = obs['road_neighbors'][n:]
        self.assertTrue((neighbors[neighbors >= 0] >= n).all())
        self.assertFalse(obs['road_traversals'][n:].any())

    def test_no_report_keeps_candidate_slots_empty(self):
        p, g = np.zeros(2), np.array([1000.,0.])
        public = dict(goal=g, elapsed_steps=0, routes={}, route_observed={}, side_status={},
                      route_open_probability={}, evidence_time=None)
        obs = _road_graph(public, p)
        n = len(obs['road_valid'])
        add_candidate_roads(obs, public, p, g, 0.)
        self.assertEqual(obs['road_valid'].shape, (n+6,))
        self.assertFalse(obs['road_valid'].any())

    def test_reward_follows_issued_route_and_limits_metric_jumps(self):
        episode = VTemp5FleetEpisode.__new__(VTemp5FleetEpisode)
        episode.position = lambda: np.array([0., 4.])
        pending = dict(vehicle_origin=np.zeros(2), feasible_path=np.array([[0,0],[0,10],[80,10]]), event={})
        before, after = episode._reward_distances(pending, 1000.)
        self.assertAlmostEqual(before-after, 4.)
        self.assertEqual(pending['event']['feasible_route_progress_m'], 4.)
