"""Slow-policy map/task preservation, coordinate geometry and topology use."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vtemp2_3.navigation_evidence import EvidenceGrid, AttemptMemory
from vtemp2_3.hierarchical_runtime import HierarchyConfig, build_observation
from vtemp2_3.hierarchical_policy import GraphSlowActorCritic, SLOW_BASE_FEATURES, SLOW_GRAPH_FEATURES
from vtemp2_3.planning_graph import build_environment_graph, NODE_FEATURES, POSITION_SCALE


class SlowEnvironmentGraphTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)

    def belief(self, goal=(60., 30.), road_count=40):
        cells = np.zeros((60, 60), np.int8)
        cells[10:40, 10:40] = -1
        cells[15:35, 25] = 1
        roads = np.c_[np.linspace(15., 65., road_count), np.full(road_count, 28.)]
        edges = np.vstack((np.arange(max(0, road_count - 1)), np.arange(1, road_count)))
        return dict(position=np.array([30., 30.]), goal=np.asarray(goal), grid=EvidenceGrid(cells, 2.),
                    graph_positions=roads, graph_edge_index=edges, graph_edges=edges.shape[1],
                    graph_confidence=np.full(road_count, .8), bounds=(0., 0., 120., 120.))

    def robots(self):
        tokens = np.zeros((4, 64), np.float32)
        tokens[:2, -1] = 1.
        tokens[0, 58:60] = 1.
        tokens[1, 0] = 50. / POSITION_SCALE
        tokens[:2, 62] = 1.
        tokens[2:, :-1] = np.nan
        return tokens

    def graph(self, belief=None):
        belief = self.belief() if belief is None else belief
        return build_environment_graph(belief, np.array([45., 35.]), self.robots(),
                                       candidate_targets=np.array([[60., 30.], [30., 60.], [35., 25.]]))

    def inputs(self):
        x = torch.zeros((3, SLOW_GRAPH_FEATURES))
        robots = self.robots(); robots[2:, :-1] = 0.
        x[:, SLOW_BASE_FEATURES:] = torch.from_numpy(robots.ravel())
        present = torch.ones(3, dtype=torch.bool)
        return x, present, present.clone()

    def test_all_road_edges_obstacles_and_tasks_survive(self):
        belief = self.belief(); original = belief['grid'].cells.copy()
        graph = self.graph(belief); nodes = graph['nodes']
        self.assertEqual(int(nodes[:, 5].sum()), 40)  # No nearest-32 cut.
        self.assertEqual(int((nodes[:, 9] * nodes[:, 6]).sum()), int((original == 1).sum()))
        self.assertTrue(np.any((nodes[:, 10] > .5) & (nodes[:, 6] > .5)))
        np.testing.assert_allclose(nodes[1, :2], (belief['goal'] - belief['position']) / POSITION_SCALE)
        np.testing.assert_allclose(nodes[2, :2], (np.array([45., 35.]) - belief['position']) / POSITION_SCALE)
        self.assertEqual(int((graph['edge_types'] == 0).sum()), 78)
        np.testing.assert_array_equal(original, belief['grid'].cells)
        self.assertTrue(np.isfinite(nodes).all())
        # Robot node payloads come from WM predictions, not map or task state.
        np.testing.assert_allclose(nodes[nodes[:, 7] > .5, :2], self.robots()[:2, :2])

    def test_missing_topology_is_an_error_and_unknown_is_not_obstacle(self):
        belief = self.belief(); del belief['graph_edge_index']
        with self.assertRaisesRegex(ValueError, 'topology'):
            self.graph(belief)
        belief = self.belief(road_count=0)
        belief['grid'] = EvidenceGrid(np.zeros((60, 60), np.int8), 2.)
        graph = self.graph(belief)
        self.assertEqual(float(graph['nodes'][:, 9].sum()), 0.)
        self.assertTrue(np.all(graph['nodes'][:3, 10] == 1.))

    def test_topology_changes_policy_and_receives_gradient(self):
        graph = self.graph(); x, mask, present = self.inputs()
        model = GraphSlowActorCritic(hidden=24)
        # The production head starts at gain .01 for safe warm-up. Use unit
        # gain here so topology-dependent probability differences are above
        # float32 rounding before any learning has taken place.
        with torch.no_grad():
            model.actor_head.weight.mul_(100.)
        before, _ = model(x, mask, present, graph)
        disconnected = deepcopy(graph)
        keep = disconnected['edge_types'] != 0
        disconnected['edges'] = disconnected['edges'][:, keep]
        disconnected['edge_types'] = disconnected['edge_types'][keep]
        after, _ = model(x, mask, present, disconnected)
        self.assertGreater(float((before.probs - after.probs).abs().max().detach()), 1e-9)
        (-before.log_prob(torch.tensor(0))).backward()
        self.assertGreater(float(model.actor_encoder.environment.embed[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.actor_encoder.environment.messages[0][0].weight.grad.abs().sum()), 0.)

    def test_graph_permutation_padding_and_ragged_batch(self):
        graph = self.graph(); x, mask, present = self.inputs(); model = GraphSlowActorCritic(hidden=24)
        expected, value = model(x, mask, present, graph)
        permutation = np.random.default_rng(42).permutation(len(graph['nodes']))
        inverse = np.argsort(permutation)
        changed = deepcopy(graph)
        changed['nodes'] = graph['nodes'][permutation]
        changed['edges'] = inverse[graph['edges']]
        reordered, reordered_value = model(x, mask, present, changed)
        torch.testing.assert_close(expected.probs, reordered.probs, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(value, reordered_value)
        padded = deepcopy(graph)
        padded['nodes'] = np.r_[graph['nodes'], np.full((1, NODE_FEATURES), np.nan, np.float32)]
        padded['node_mask'] = np.r_[np.ones(len(graph['nodes']), bool), False]
        same, _ = model(x, mask, present, padded)
        torch.testing.assert_close(expected.probs, same.probs)
        batch, values = model(x[None].repeat(2, 1, 1), mask[None].repeat(2, 1), present[None].repeat(2, 1), [graph, padded])
        torch.testing.assert_close(batch.probs[0], batch.probs[1])
        torch.testing.assert_close(values[0], values[1])

    def test_final_heading_counterexample_is_distinguishable(self):
        # The old 52 candidate features are rotation-invariant, while the peer
        # stays east. Explicit goal/candidate global axes now disambiguate it.
        base = dict(position=np.array([500., 500.]), grid=EvidenceGrid(np.zeros((50, 50), np.int8), 20.),
                    graph_positions=np.empty((0, 2)), graph_edge_index=np.empty((2, 0), int),
                    graph_edges=0, bounds=(0., 0., 1000., 1000.))
        configs = []
        robots = self.robots(); robots[2:, :-1] = 0.
        for goal in ([650., 500.], [500., 650.]):
            belief = dict(base, goal=np.asarray(goal))
            obs = build_observation(belief, AttemptMemory(), HierarchyConfig())
            graph = build_environment_graph(belief, belief['position'], robots, candidate_targets=obs['targets'])
            x = torch.from_numpy(np.c_[obs['features'], np.broadcast_to(robots.ravel(), (len(obs['features']), 256))].astype(np.float32))
            configs.append((obs, graph, x))
        np.testing.assert_allclose(configs[0][0]['features'], configs[1][0]['features'], atol=1e-6)
        self.assertFalse(np.array_equal(configs[0][1]['candidate_xy'], configs[1][1]['candidate_xy']))
        model = GraphSlowActorCritic(hidden=24)
        distributions = [model(x, torch.from_numpy(obs['mask']), torch.from_numpy(obs['present']), graph)[0]
                         for obs, graph, x in configs]
        self.assertGreater(float((distributions[0].probs - distributions[1].probs).abs().max().detach()), 1e-8)

    def test_old_inputs_and_old_checkpoint_fail_explicitly(self):
        model = GraphSlowActorCritic(hidden=24); x, mask, present = self.inputs()
        with self.assertRaisesRegex(ValueError, 'environment graph'):
            model(x, mask, present)
        state = model.state_dict()
        state['actor_encoder.candidate.0.weight'] = state['actor_encoder.candidate.0.weight'][:, :-2]
        with self.assertRaisesRegex(ValueError, 'v1 checkpoint'):
            model.load_state_dict(state)


if __name__ == '__main__':
    unittest.main(verbosity=2)
