"""Received graph causality, temporal attention and graph dynamics supervision."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from world_model.graph_dynamics import (GraphWorldModel, NODE_COUNT, TOKEN_WIDTH, LOCAL_EVIDENCE_WIDTH,
                                ROAD_MODEL_SCHEMA, ROAD_SUMMARY_FIELDS, ROAD_TOPOLOGY_WIDTH,
                                belief_tokens, build_road_topology, infer, objective)


def fixture(batch_size=2):
    history = torch.zeros(batch_size, NODE_COUNT, 8, 14)
    history[..., 2] = .1
    history[..., 8] = 1.
    history[..., 9] = .2
    history[..., 10] = .3
    history[..., 11] = .25
    history[:, 0, :, 13] = 1.
    ego = torch.zeros(batch_size, NODE_COUNT, dtype=torch.bool)
    ego[:, 1] = True
    history[:, 1, :, 9] = 0.
    history[:, 1, :, 11] = 0.
    baseline_states = torch.zeros(batch_size, NODE_COUNT, 5, 4)
    baseline_states[..., 2] = .1
    baseline_states[..., 0] = torch.arange(5) * .03
    context = torch.zeros(batch_size, NODE_COUNT, 20)
    context[:, 0, 0] = -.2
    result = dict(history=history, history_mask=torch.ones(batch_size, NODE_COUNT, 8, dtype=torch.bool),
        context=context, baseline_states=baseline_states,
        baseline_actions=torch.zeros(batch_size, NODE_COUNT, 4, 2),
        node_mask=torch.ones(batch_size, NODE_COUNT, dtype=torch.bool),
        node_type=torch.tensor([[1, 0, 0, 0]]).expand(batch_size, -1).clone(), ego_mask=ego,
        state_observed=ego.clone(),
        local_evidence=torch.zeros(batch_size, LOCAL_EVIDENCE_WIDTH),
        state_mask=torch.ones(batch_size, NODE_COUNT, 5, dtype=torch.bool),
        action_mask=torch.ones(batch_size, NODE_COUNT, 4, dtype=torch.bool))
    result['baseline_actions'][..., 0] = .1
    result['target_states'] = baseline_states.clone()
    result['target_states'][:, 2:, :, 0] += .25
    result['target_actions'] = result['baseline_actions'].clone()
    result['target_actions'][:, 2:, :, 0] += .1
    return result


class GraphWorldTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.model = GraphWorldModel(32, 4)
        self.batch = fixture()

    def test_current_sensed_states_are_anchored_and_ground_retains_rule_prior(self):
        self.batch['state_observed'][:, 2] = True
        with torch.no_grad():
            self.model.initial.bias[:4] = 10.
            self.model.action.bias[:2] = 10.
            self.model.innovation.bias[:2] = 10.
        prediction = infer(self.model, self.batch)
        torch.testing.assert_close(prediction['states'][:, 1, 0], self.batch['baseline_states'][:, 1, 0])
        self.assertTrue((prediction['state_std'][:, 1, 0] == 0).all())
        torch.testing.assert_close(prediction['states'][:, 2, 0], self.batch['baseline_states'][:, 2, 0])
        self.assertTrue((prediction['state_std'][:, 2, 0] == 0).all())
        self.assertFalse(torch.equal(prediction['states'][:, 3, 0], self.batch['baseline_states'][:, 3, 0]))
        torch.testing.assert_close(prediction['states'][:, 0], self.batch['baseline_states'][:, 0])
        torch.testing.assert_close(prediction['actions'][:, 0], self.batch['baseline_actions'][:, 0])
        self.assertTrue((prediction['actions'][:, 1:].norm(dim=-1) <= 1.000001).all())
        self.assertTrue((prediction['state_std'][:, :, 1:, :2] >=
                         prediction['state_std'][:, :, :-1, :2]).all())

    def test_age_and_hops_explicitly_downweight_sender_messages(self):
        graph = self.model.graph
        with torch.no_grad():
            graph.query.weight.zero_()
            graph.key.weight.zero_()
            for parameter in graph.edge_bias.parameters():
                parameter.zero_()
        prediction = infer(self.model, self.batch)
        for metadata_index in (9, 11):
            with self.subTest(metadata_index=metadata_index):
                older = copy.deepcopy(self.batch)
                older['history'][:, 2, :, metadata_index] += .5
                after = infer(self.model, older)
                self.assertTrue((after['attention'][:, :, 2] < prediction['attention'][:, :, 2]).all())
                torch.testing.assert_close(after['attention'].sum(-1), torch.ones(2, NODE_COUNT))

    def test_padding_and_truth_cannot_change_known_predictions(self):
        self.batch['node_mask'][:, 3] = False
        self.batch['history_mask'][:, :, :3] = False
        before = infer(self.model, self.batch)
        altered = copy.deepcopy(self.batch)
        altered['history'][:, :, :3] = float('nan')
        altered['history'][:, 3] = float('nan')
        altered['context'][:, 3] = float('nan')
        altered['baseline_states'][:, 3] = float('nan')
        altered['baseline_actions'][:, 3] = float('nan')
        altered['target_states'].fill_(float('nan'))
        altered['target_actions'].fill_(1e20)
        after = infer(self.model, altered)
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        self.assertTrue((after['attention'][:, :, 3] == 0).all())
        self.assertTrue((after['states'][:, 3] == 0).all())
        self.assertTrue((belief_tokens(after, altered)[:, 3] == 0).all())

    def test_future_condition_cannot_change_present_or_earlier_predictions(self):
        before = infer(self.model, self.batch)
        self.batch['baseline_actions'][:, 2, 3, 0] = .7
        after = infer(self.model, self.batch)
        torch.testing.assert_close(before['states'][:, :, :4], after['states'][:, :, :4], rtol=0, atol=0)
        self.assertFalse(torch.equal(before['states'][:, 2, 4], after['states'][:, 2, 4]))

    def test_censored_nan_labels_do_not_poison_loss_or_gradients(self):
        self.batch['state_mask'][:, :, 3:] = False
        self.batch['action_mask'][:, :, 2:] = False
        prediction = infer(self.model, self.batch)
        before = objective(prediction, self.batch)
        self.batch['target_states'][:, :, 3:] = float('nan')
        self.batch['target_actions'][:, :, 2:] = float('nan')
        after = objective(prediction, self.batch)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        after.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_graph_training_reduces_loss_and_updates_message_network(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.005)
        before = float(objective(infer(self.model, self.batch), self.batch).detach())
        initial = self.model.graph.value.weight.detach().clone()
        for _ in range(16):
            optimizer.zero_grad()
            loss = objective(infer(self.model, self.batch), self.batch)
            loss.backward()
            optimizer.step()
        after = float(objective(infer(self.model, self.batch), self.batch).detach())
        self.assertLess(after, before)
        self.assertFalse(torch.equal(initial, self.model.graph.value.weight))

    def test_known_sensor_and_map_attributes_condition_future_dynamics(self):
        # Output heads start at the rule prior. Give them nonzero weights, as
        # happens after training, to test the complete evidence-to-motion path.
        with torch.no_grad():
            torch.nn.init.normal_(self.model.action.weight, std=.1)
            torch.nn.init.normal_(self.model.innovation.weight, std=.1)
        self.batch['local_evidence'][:, 144] = 1.
        self.batch['local_evidence'][:, 288:293] = torch.tensor([.1, .2, .8, .25, 1.])
        before = infer(self.model, self.batch)
        for attribute in (0, 288, 290):
            with self.subTest(attribute=attribute):
                changed = copy.deepcopy(self.batch)
                changed['local_evidence'][:, attribute] += .5
                after = infer(self.model, changed)
                self.assertFalse(torch.equal(before['states'][:, 1:, 1:], after['states'][:, 1:, 1:]))
                # Locally measured current states remain physical observations.
                torch.testing.assert_close(before['states'][:, 1, 0], after['states'][:, 1, 0], rtol=0, atol=0)
        loss = objective(before, self.batch)
        loss.backward()
        self.assertGreater(float(self.model.local_evidence[0].weight.grad.abs().sum()), 0.)

    def test_equal_degree_roads_with_different_connectivity_change_learned_predictions(self):
        ring = nx.cycle_graph(6)
        triangles = nx.Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
        xy = {i: (float(i), 0.) for i in ring}
        summaries = []
        for graph in (ring, triangles):
            nx.set_node_attributes(graph, xy, 'position')
            summaries.append(build_road_topology(SimpleNamespace(graph=graph), xy[0], xy[5]))
        self.assertEqual(dict(ring.degree()), dict(triangles.degree()))
        self.assertEqual(summaries[0].shape, (ROAD_TOPOLOGY_WIDTH,))
        connected = ROAD_SUMMARY_FIELDS.index('anchors_connected')
        self.assertEqual([row[connected] for row in summaries], [1., 0.])
        self.assertFalse(np.array_equal(summaries[0][-496:], summaries[1][-496:]))
        model = GraphWorldModel(32, 4, road_topology=True)
        self.assertEqual(model.schema, ROAD_MODEL_SCHEMA)
        with torch.no_grad():
            torch.nn.init.normal_(model.action.weight, std=.1)
        batch = fixture(2)
        batch['road_topology'] = torch.from_numpy(np.stack(summaries))
        prediction = infer(model, batch)
        self.assertFalse(torch.equal(prediction['actions'][0, 1:], prediction['actions'][1, 1:]))
        objective(prediction, batch).backward()
        self.assertGreater(float(model.road_encoder[0].weight.grad.abs().sum()), 0.)
        del batch['road_topology']
        with self.assertRaises(KeyError):
            infer(model, batch)
        infer(self.model, batch)

    def test_far_edges_and_disconnected_roads_remain_visible_beyond_32_nearest(self):
        graph = nx.path_graph(60)
        coordinates = {i: (10. * i, 0.) for i in graph}
        nx.set_node_attributes(graph, coordinates, 'position')
        wrapped = SimpleNamespace(graph=graph)
        near = list(range(32))
        complete = build_road_topology(wrapped, coordinates[0], coordinates[59], nearby_nodes=near)
        graph.remove_edge(45, 46)
        cut = build_road_topology(wrapped, coordinates[0], coordinates[59])
        # Exact near edges and degrees are identical. The full-map component
        # containing the ego loses the endpoint and its remote nodes.
        np.testing.assert_array_equal(complete[-496:], cut[-496:])
        fields = dict(zip(ROAD_SUMMARY_FIELDS, cut))
        self.assertEqual(fields['anchors_connected'], 0.)
        self.assertAlmostEqual(fields['anchor_component_fraction'], 46. / 60.)
        self.assertAlmostEqual(fields['nearest_reachable_to_target_m'], 140. / 95.)
        self.assertAlmostEqual(complete[ROAD_SUMMARY_FIELDS.index('anchor_path_m')], 590. / 95., places=5)
        graph.remove_edge(52, 53)
        changed = build_road_topology(wrapped, coordinates[0], coordinates[59])
        self.assertFalse(np.array_equal(changed[len(ROAD_SUMMARY_FIELDS):-496],
                                        cut[len(ROAD_SUMMARY_FIELDS):-496]))
        empty = build_road_topology(None, [0., 0.])
        np.testing.assert_array_equal(empty, np.zeros(ROAD_TOPOLOGY_WIDTH, np.float32))
        target_missing = build_road_topology(wrapped, coordinates[0])
        np.testing.assert_array_equal(target_missing[7:13], np.zeros(6, np.float32))

    def test_unknown_sensor_pixels_and_absent_map_nodes_cannot_affect_dynamics(self):
        with torch.no_grad():
            torch.nn.init.normal_(self.model.action.weight, std=.1)
        before = infer(self.model, self.batch)
        unknown = copy.deepcopy(self.batch)
        unknown['local_evidence'][:, :144] = float('nan')
        nodes = unknown['local_evidence'][:, 288:].reshape(-1, 32, 5)
        nodes[..., :4] = float('nan')
        after = infer(self.model, unknown)
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def test_token_layout_retains_age_hops_type_ego_and_source_history(self):
        before = self.batch['history'].clone()
        tokens = belief_tokens(infer(self.model, self.batch), self.batch)
        self.assertEqual(tuple(tokens.shape), (2, NODE_COUNT, TOKEN_WIDTH))
        torch.testing.assert_close(tokens[:, 0, 56:], torch.tensor([.2, .25, 1., 0., 1., .3, 0., 1.]).expand(2, -1))
        self.assertTrue((tokens[:, 1, 59] == 1).all())
        torch.testing.assert_close(tokens[:, 0, 0], torch.full((2,), .2))
        torch.testing.assert_close(self.batch['history'], before, rtol=0, atol=0)

    def test_missing_ego_and_missing_known_history_are_rejected(self):
        self.batch['ego_mask'].fill_(False)
        with self.assertRaisesRegex(ValueError, 'ego'):
            infer(self.model, self.batch)
        self.batch['ego_mask'][:, 1] = True
        self.batch['history_mask'][:, 2].fill_(False)
        with self.assertRaisesRegex(ValueError, 'history'):
            infer(self.model, self.batch)


if __name__ == '__main__':
    unittest.main(verbosity=2)
