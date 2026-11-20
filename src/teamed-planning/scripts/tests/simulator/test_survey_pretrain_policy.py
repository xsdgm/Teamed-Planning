import unittest

import numpy as np

from ground.ground_policy import TASK_INPUT_WIDTH
from world_model.navigation_evidence import EvidenceGrid, corridor_fractions
from simulator.survey_pretrain_policy import _received_roads, policy_observation


class SurveyObservationTest(unittest.TestCase):
    def test_batch_corridors_preserve_cell_samples(self):
        rng = np.random.default_rng(17)
        cells = rng.choice([-1, 0, 1], size=(32, 47))
        grid = EvidenceGrid(cells, resolution=.5, origin=(-17., 9.), clearance=0.)
        position = np.array([-10.75, 14.75])
        targets = np.array([position, [-7.25, 15.75], [-13.5, 25.5], [3., 3.]])
        result = corridor_fractions(grid, position, targets)
        for row, point in zip(result, targets):
            delta = point - position
            normal = np.array([-delta[1], delta[0]]) / max(np.linalg.norm(delta), 1e-9)
            expected = []
            for i in range(8):
                line = position + np.linspace(i/8., (i+1)/8., 5)[:,None] * delta
                points = line[:,None,:] + np.linspace(-12., 12., 7)[None,:,None] * normal
                values, _ = grid.sample(points.reshape(-1, 2))
                expected.extend(np.mean(values == code) for code in (-1, 1, 0))
            np.testing.assert_array_equal(row, expected)

    def test_kilometre_observation_keeps_world_goal_and_candidate_offsets(self):
        cells = np.zeros((900, 1200), np.int8)
        cells[500:540, 600:680] = -1
        origin = np.array([-870., 181.])
        position = origin + [666.5, 512.5]
        goal = position + [1200., 800.]
        targets = position + [[30., 0.], [10., 70.]]
        state = dict(cells=cells, source_times=np.where(cells, 200., np.nan),
                     resolution=1., origin=origin, time=370., position=position, goal=goal,
                     velocity=[1., 0.], orientation=0., remaining_budget=.91,
                     ground_target=position+[20., 0.], survey_target=position+[12., 40.],
                     survey_issued_at=350., survey_evidence_received=True, targets=targets)
        full = policy_observation(state)
        local = policy_observation(state, local_window=True)
        # All observed cells are in the crop. A grid-phase shift or a localised
        # final goal would change at least one road, task or candidate channel.
        np.testing.assert_array_equal(full['ground_inputs'], local['ground_inputs'])
        np.testing.assert_array_equal(full['features'], local['features'])
        np.testing.assert_allclose(local['ground_inputs'][5:7], (goal-position)/350.)

    def test_probes_across_a_wall_remain_legal_with_independent_drive_task(self):
        cells = np.full((60, 60), -1, np.int8)
        cells[:, 24:27] = 1
        state = dict(cells=cells, source_times=np.full(cells.shape, 2.), resolution=2.,
                     origin=[-10., 5.], time=10., position=[24., 65.], goal=[90., 65.],
                     velocity=[1., 0.], orientation=0., remaining_budget=.75,
                     ground_target=[34., 65.], survey_target=[84., 65.], survey_issued_at=5.,
                     survey_evidence_received=True, targets=[[87., 65.], [22., 64.]], mask=[True, False])
        observed = policy_observation(state)
        self.assertEqual(observed['features'].shape, (96, 52))
        self.assertEqual(observed['ground_inputs'].shape, (TASK_INPUT_WIDTH,))
        np.testing.assert_array_equal(np.flatnonzero(observed['mask']), [0])
        np.testing.assert_array_equal(np.flatnonzero(observed['present']), [0, 1])
        np.testing.assert_array_equal(observed['targets'][:2], state['targets'])
        np.testing.assert_allclose(observed['ground_inputs'][7:9], [10 / 350, 0])
        np.testing.assert_allclose(observed['ground_inputs'][-6:], [1, 1, 60 / 350, 0, np.log1p(5), 1])

    def test_sparse_graph_edges_do_not_cross_unseen_cells(self):
        cells = np.full((11, 17), -1, np.int8)
        cells[:, 2] = 0
        grid = EvidenceGrid(cells, resolution=2., origin=(-21., 13.))
        times = np.where(cells != 0, 3., np.nan)
        roads = _received_roads(grid, times)
        a, b = roads['graph_positions'][roads['graph_edge_index']]
        midpoint_x = grid.origin[0] + 2.5 * grid.resolution
        self.assertFalse(np.any((a[:, 0] - midpoint_x) * (b[:, 0] - midpoint_x) < 0))
        self.assertGreater(roads['graph_edges'], 0)
        np.testing.assert_array_equal(roads['graph_source_stamps'], np.full(len(roads['graph_positions']), 3.))
