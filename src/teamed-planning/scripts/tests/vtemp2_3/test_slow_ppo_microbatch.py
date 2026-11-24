"""Slow PPO microbatches preserve the full-batch objective and rollback."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vtemp2_3.hierarchical_policy import GraphSlowActorCritic, SLOW_BASE_FEATURES
from vtemp2_3.planning_graph import NODE_FEATURES
from vtemp2_3.slow_policy import SlowActorCritic, update


class SlowPPOMicrobatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(917)

    def graph(self, index):
        # Different graph sizes exercise ragged slicing across microbatches.
        count = 5 + index
        nodes = np.zeros((count, NODE_FEATURES), np.float32)
        nodes[:, :2] = np.c_[np.linspace(0., .3, count), np.full(count, .1)]
        nodes[:3, 2:5] = np.eye(3)
        nodes[3:, 5] = 1.
        forward = np.vstack((np.arange(count - 1), np.arange(1, count)))
        edges = np.concatenate((forward, forward[::-1]), axis=1)
        return dict(nodes=nodes, edges=edges,
                    edge_types=np.zeros(edges.shape[1], np.int64),
                    candidate_xy=np.array([[.3, .1], [.1, .3], [-.1, .1], [0., 0.]], np.float32))

    def episodes(self, model):
        rng = np.random.default_rng(531)
        transitions = []
        for index in range(7):
            x = rng.normal(0., .2, size=(4, model.features)).astype(np.float32)
            present = np.array([True, True, True, False])
            mask = np.array([True, True, index % 2 == 0, False])
            graph = None
            if isinstance(model, GraphSlowActorCritic):
                robots = np.zeros((4, 64), np.float32)
                robots[:2, -1] = 1.
                robots[:2, 62] = 1.
                robots[1, 0] = .2
                x[:, SLOW_BASE_FEATURES:] = robots.ravel()
                graph = self.graph(index)
            inputs = (torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(present))
            if graph is not None:
                inputs += (graph,)
            with torch.no_grad():
                distribution, value = model(*inputs)
            action = index % 2
            item = dict(features=x, mask=mask, present=present, action=action,
                        logp=float(distribution.log_prob(torch.tensor(action))),
                        value=float(value), reward=float((index % 3) - 1) * 2.,
                        duration=index + 1, next_value=.7,
                        terminated=index == 3, truncated=index == 6)
            if graph is not None:
                item['environment_graph'] = graph
            transitions.append(item)
        return [transitions[:4], transitions[4:]]

    def assert_tree_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_tree_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_uneven_microbatches_match_full_batch_update_and_losses(self):
        for graph_model in (False, True):
            original = GraphSlowActorCritic(hidden=8) if graph_model else SlowActorCritic(4, 8)
            episodes = self.episodes(original)
            full = copy.deepcopy(original)
            # Momentum state also catches accidentally stepping once per chunk.
            full_optimizer = torch.optim.SGD(full.parameters(), lr=.003, momentum=.9)
            expected = update(full, full_optimizer, episodes, epochs=3,
                              target_kl=.1, max_kl=.2, microbatch_size=7)
            self.assertEqual(expected['accepted_epochs'], 3)
            for size in (1, 3):
                with self.subTest(graph=graph_model, microbatch_size=size):
                    model = copy.deepcopy(original)
                    optimizer = torch.optim.SGD(model.parameters(), lr=.003, momentum=.9)
                    observed_batch_sizes = []
                    hook = model.register_forward_pre_hook(
                        lambda module, args: observed_batch_sizes.append(args[0].shape[0]))
                    try:
                        actual = update(model, optimizer, episodes, epochs=3,
                                        target_kl=.1, max_kl=.2, microbatch_size=size)
                    finally:
                        hook.remove()
                    self.assertTrue(observed_batch_sizes)
                    self.assertLessEqual(max(observed_batch_sizes), size)
                    for key in ('transitions', 'fast_attempts', 'accepted_epochs', 'rejected_epochs'):
                        self.assertEqual(actual[key], expected[key])
                    for key in ('kl', 'behaviour_logp_error', 'mean_return_target'):
                        self.assertAlmostEqual(actual[key], expected[key], delta=2e-6)
                    for actual_loss, expected_loss in zip(actual['losses'], expected['losses']):
                        for key in ('actor', 'critic', 'entropy'):
                            self.assertAlmostEqual(actual_loss[key], expected_loss[key], delta=2e-6)
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, full.state_dict()[key], atol=2e-7, rtol=2e-5)
                    for actual_state, expected_state in zip(optimizer.state.values(), full_optimizer.state.values()):
                        torch.testing.assert_close(actual_state['momentum_buffer'],
                                                   expected_state['momentum_buffer'], atol=2e-7, rtol=2e-5)

    def test_stale_sample_in_last_microbatch_prevents_any_update(self):
        for field, message in (('logp', 'Stale behaviour policy'), ('value', 'Stale behaviour critic')):
            with self.subTest(field=field):
                model = SlowActorCritic(4, 8)
                episodes = self.episodes(model)
                episodes[-1][-1][field] += .1
                before = copy.deepcopy(model.state_dict())
                optimizer = torch.optim.Adam(model.parameters(), lr=.003)
                with self.assertRaisesRegex(ValueError, message):
                    update(model, optimizer, episodes, microbatch_size=3)
                self.assert_tree_equal(model.state_dict(), before)
                self.assertFalse(optimizer.state)

    def test_kl_rejection_restores_existing_adam_state(self):
        model = GraphSlowActorCritic(hidden=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        warmup = update(model, optimizer, self.episodes(model), epochs=1,
                        target_kl=.1, max_kl=.2, microbatch_size=3)
        self.assertEqual(warmup['accepted_epochs'], 1)
        self.assertTrue(optimizer.state)
        optimizer.param_groups[0]['lr'] = .5
        before_model = copy.deepcopy(model.state_dict())
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        result = update(model, optimizer, self.episodes(model), epochs=2,
                        target_kl=1e-14, max_kl=2e-14, microbatch_size=3)
        self.assertEqual(result['accepted_epochs'], 0)
        self.assertEqual(result['rejected_epochs'], 1)
        self.assert_tree_equal(model.state_dict(), before_model)
        self.assert_tree_equal(optimizer.state_dict(), before_optimizer)


if __name__ == '__main__':
    unittest.main(verbosity=2)
