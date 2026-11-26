import math
from pathlib import Path
import unittest

import numpy as np
import torch

from vtemp4.vtemp4_body_graph import body_frame_observation, GoalResidualBodyFramePolicy
from vtemp4.vtemp4_decision_graph_history import FullCommandHistory
from vtemp5.vtemp5_policy import _road_graph
from vtemp5.v91_route_evidence import route_candidates, evaluate_route, append_route_evidence


REPO = Path(__file__).resolve().parents[5]
ACTOR = REPO / 'artifacts/vtemp5-fleet-circle-rejoin-formal-16w-eval4-20261124/checkpoints/heading-0006.pt'


class RouteEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.car = np.array([200., 200.])
        self.goal = self.car + [900., 0.]
        rows, cols = np.indices((48, 48))
        self.xy = self.car + np.stack((71. - 2. * rows, 47. - 2. * cols), axis=-1)
        r = np.linalg.norm(self.xy - self.car, axis=-1)
        self.gap = (r >= 12.5) & (r <= 15.)
        self.free = ~self.gap
        self.occupied = np.zeros_like(self.free)

    def routes(self, probability):
        return route_candidates(self.xy, self.car, self.goal, self.free, self.occupied,
                                world_size=1440., probabilities=probability)

    def actor_inputs(self, candidates):
        obs = _road_graph(dict(routes={}, goal=self.goal, elapsed_steps=0), self.car)
        append_route_evidence(obs, candidates, self.car, self.goal, 0)
        obs.update(FullCommandHistory().observation(0, self.car, self.goal, 200., self.car[None], [], []))
        obs.update(vehicle=np.array([0., 0., 0., 0., 0., 1., 0.], dtype=np.float32),
                   goal=np.r_[self.goal / 1440., (self.goal - self.car) / 350.].astype(np.float32),
                   local_target=np.zeros(4, np.float32), local_target_valid=False,
                   angle_arcs=np.array([[-math.pi + 1.e-6, math.pi - 1.e-6]]))
        return body_frame_observation(obs)

    def test_gap_risk_survives_measured_vertices_and_body_frame_conversion(self):
        self.assertEqual(int(self.gap.sum()), 52)
        safe = self.routes(np.where(self.gap, .05, 0.))
        risky = self.routes(np.where(self.gap, .95, 0.))
        for side in (-1, 1):
            np.testing.assert_array_equal(safe[side].points, risky[side].points)
            self.assertTrue(risky[side].measured.all())
            self.assertFalse(risky[side].fully_observed)
            self.assertAlmostEqual(risky[side].risk, .95)
        a, b = self.actor_inputs(safe), self.actor_inputs(risky)
        np.testing.assert_allclose(a['road_nodes'][-6:, 2], .95)
        np.testing.assert_allclose(b['road_nodes'][-6:, 2], .05)
        np.testing.assert_array_equal(a['road_nodes'][:, 3:], b['road_nodes'][:, 3:])
        self.assertTrue((b['road_nodes'][-6:, 4] == 1.).all())
        self.assertTrue((b['road_nodes'][-6:, 6] > 0.).all())
        self.assertFalse(b['road_entry_mask'][-6:].any())
        for key in a.keys() - {'road_nodes'}:
            np.testing.assert_array_equal(a[key], b[key])

    @unittest.skipUnless(ACTOR.is_file(), 'Archived frozen PPO checkpoint is unavailable')
    def test_frozen_checkpoint_receives_and_responds_to_gap_risk(self):
        actor = GoalResidualBodyFramePolicy()
        actor.load_state_dict(torch.load(ACTOR, map_location='cpu', weights_only=True)['model'])
        actor.eval().requires_grad_(False)
        a = self.actor_inputs(self.routes(np.where(self.gap, .05, 0.)))
        b = self.actor_inputs(self.routes(np.where(self.gap, .95, 0.)))
        with torch.inference_mode():
            low = actor.act(a, deterministic=True)[1]
            high = actor.act(b, deterministic=True)[1]
        self.assertGreater(abs(low - high), 1.e-7)

    def test_prediction_cannot_override_observations_or_fill_the_observed_arm(self):
        risk = self.routes(None)
        self.assertEqual(risk[1].risk, .5)
        self.free[:] = True
        a = self.actor_inputs(self.routes(None))
        b = self.actor_inputs(self.routes(np.ones_like(self.xy[..., 0])))
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.occupied[:] = True
        self.assertFalse(self.routes(np.zeros_like(self.xy[..., 0])))

    def test_empty_unobserved_and_boundary_cases_remain_uncertified(self):
        self.free[:] = False
        self.assertFalse(self.routes(np.zeros_like(self.xy[..., 0])))
        self.free[:] = np.linalg.norm(self.xy - self.car, axis=-1) < 4.
        self.assertFalse(self.routes(None))
        empty = self.actor_inputs({})
        self.assertFalse(empty['road_valid'][-6:].any())
        self.assertFalse(empty['road_entry_mask'][-6:].any())
        self.free[:] = True
        points = self.car + np.array([[0., 8.], [0., 20.], [0., 32.]])
        self.assertIsNone(evaluate_route(points, self.xy, self.car, self.free, self.occupied, world_size=232.))
        outside_view = self.car + np.array([[8., 0.], [20., 0.], [71., 0.]])
        self.assertIsNone(evaluate_route(outside_view, self.xy, self.car, self.free, self.occupied, world_size=1440.))

    def test_measured_and_unmeasured_vertices_keep_distinct_counts(self):
        self.free[:] = np.linalg.norm(self.xy - self.car, axis=-1) < 11.5
        routes = self.routes(None)
        self.assertEqual(routes[1].measured.tolist(), [True, False, False])
        obs = self.actor_inputs(routes)
        np.testing.assert_array_equal(obs['road_nodes'][-6:, 6] > 0.,
                                      [True, False, False, True, False, False])
        self.assertFalse(obs['road_entry_mask'][-6:].any())


if __name__ == '__main__':
    unittest.main()
