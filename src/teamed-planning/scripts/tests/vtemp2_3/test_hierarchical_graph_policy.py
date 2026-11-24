"""World-graph policy boundaries and actual gradient use of belief nodes."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vtemp2_3.hierarchical_policy import (BeliefGraphEncoder, GraphFastActorCritic,
    GraphSlowActorCritic, GRAPH_WIDTH, LOCAL_WIDTH, RAW_WIDTH,
    SLOW_BASE_FEATURES, SLOW_GRAPH_FEATURES)
from vtemp2_3.graph_dynamics import NODE_COUNT, TOKEN_WIDTH
from vtemp2_3.slow_policy import update
from vtemp2_3.planning_graph import NODE_FEATURES


def graph_tokens(batch=2):
    value = torch.randn(batch, NODE_COUNT, TOKEN_WIDTH) * .1
    value[..., -1] = 1.
    value[..., 58:60] = 0.
    value[:, 0, 58] = 1.
    value[:, 1, 59] = 1.
    return value


def environment_graphs(batch, candidates):
    result = []
    for _ in range(batch):
        nodes = np.zeros((3, NODE_FEATURES), np.float32)
        nodes[np.arange(3), np.arange(3) + 2] = 1.
        nodes[1, 0] = 1.; nodes[2, 1] = .5
        result.append(dict(nodes=nodes, edges=np.array([[0, 1, 0, 2], [1, 0, 2, 0]]),
            edge_types=np.full(4, 2), candidate_xy=np.random.default_rng(1).normal(size=(candidates, 2)).astype(np.float32)))
    return result


class LocalBase(nn.Module):
    """Small stand-in with the archived actor/CTDE interfaces."""
    def __init__(self):
        super().__init__()
        self.gat_actor = nn.Linear(RAW_WIDTH, 2)
        self.log_std = nn.Parameter(torch.full((2,), -.5))
        self.critic = nn.Linear(RAW_WIDTH, 1)
        self.register_buffer('value_std', torch.tensor(1.))

    def distribution(self, x):
        mean = self.gat_actor(x)
        return Independent(Normal(mean, self.log_std.exp().expand_as(mean)), 1)

    def value(self, x, state=None):
        return self.critic(x.mean(-2)).squeeze(-1)


class GraphPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_graph_attention_ignores_padding_and_node_order(self):
        model = BeliefGraphEncoder()
        nodes = graph_tokens()
        nodes[:, -1, -1] = 0.
        expected = model(nodes)
        np.testing.assert_allclose(model(nodes.flip(-2)).detach(), expected.detach(), atol=1e-6)
        nodes[:, -1, :-1] = float('nan')
        torch.testing.assert_close(model(nodes), expected)
        nodes[..., -1] = 0.
        nodes[..., :-1] = float('nan')
        torch.testing.assert_close(model(nodes), torch.zeros_like(expected))

    def test_fast_actor_cannot_read_raw_peer_or_relay_payloads(self):
        model = GraphFastActorCritic(LocalBase())
        raw = torch.randn(2, RAW_WIDTH)
        x = torch.cat((raw, graph_tokens().flatten(-2)), -1)
        changed = x.clone()
        changed[:, LOCAL_WIDTH:RAW_WIDTH] = 1e6
        torch.testing.assert_close(model.distribution(x).mean, model.distribution(changed).mean)
        local, nodes = model.split(x)
        torch.testing.assert_close(local[:, :LOCAL_WIDTH], raw[:, :LOCAL_WIDTH])
        self.assertTrue((local[:, LOCAL_WIDTH:] == 0).all())
        self.assertEqual(nodes.shape, (2, NODE_COUNT, TOKEN_WIDTH))
        changed = x.clone(); changed[:, 32] += 1.
        self.assertGreater(float((model.distribution(x).mean - model.distribution(changed).mean).abs().max().detach()), 0.)
        with self.assertRaises(ValueError):
            model.distribution(x[..., :-1])

    def test_fast_graph_encoder_learns_and_changes_actions(self):
        model = GraphFastActorCritic(LocalBase())
        x = torch.cat((torch.randn(2, RAW_WIDTH), graph_tokens().flatten(-2)), -1)
        before = model.actor_peers.embed[0].weight.detach().clone()
        optimizer = torch.optim.Adam([*model.actor_peers.parameters(), *model.actor_delta.parameters()], lr=1e-3)
        for _ in range(3):
            loss = (model.distribution(x).mean - torch.tensor([[.3, -.4], [-.2, .5]])).square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        self.assertFalse(torch.equal(before, model.actor_peers.embed[0].weight))
        absent = x.clone(); absent[:, RAW_WIDTH:] = 0.
        self.assertGreater(float((model.distribution(x).mean - model.distribution(absent).mean).abs().max().detach()), 0.)
        team = x[None].repeat(3, 1, 1)
        self.assertEqual(model.value(team).shape, (3,))
        actor = {id(p) for p in model.actor_parameters()}
        critic = {id(p) for p in model.critic_parameters()}
        self.assertFalse(actor & critic)

    def slow_inputs(self):
        x = torch.randn(2, 6, SLOW_GRAPH_FEATURES) * .1
        nodes = graph_tokens()
        x[..., SLOW_BASE_FEATURES:] = nodes.flatten(-2)[:, None, :]
        present = torch.ones(2, 6, dtype=torch.bool)
        present[:, -1] = False
        mask = present.clone(); mask[:, 3] = False
        return x, mask, present

    def test_slow_mask_permutation_and_missing_candidate_invariance(self):
        model = GraphSlowActorCritic()
        x, mask, present = self.slow_inputs()
        graphs = environment_graphs(len(x), x.shape[1])
        dist, value = model(x, mask, present, graphs)
        self.assertTrue((dist.probs[~mask] == 0).all())
        x[:, -1] = float('nan')
        same, same_value = model(x, mask, present, graphs)
        torch.testing.assert_close(dist.probs, same.probs)
        torch.testing.assert_close(value, same_value)
        permutation = torch.tensor([2, 0, 5, 4, 3, 1])
        reordered_graphs = [dict(g, candidate_xy=g['candidate_xy'][permutation]) for g in graphs]
        reordered, reordered_value = model(x[:, permutation], mask[:, permutation], present[:, permutation], reordered_graphs)
        torch.testing.assert_close(reordered.probs, dist.probs[:, permutation])
        torch.testing.assert_close(reordered_value, value)
        with self.assertRaises(ValueError):
            model(x, torch.zeros_like(mask), present, graphs)

    def test_slow_smdp_update_includes_graph_parameters(self):
        model = GraphSlowActorCritic(hidden=32)
        x, mask, present = self.slow_inputs()
        graphs = environment_graphs(len(x), x.shape[1])
        with torch.no_grad():
            dist, values = model(x, mask, present, graphs)
        transitions = []
        for i in range(2):
            action = i
            transitions.append(dict(features=x[i].numpy(), mask=mask[i].numpy(), present=present[i].numpy(),
                environment_graph=graphs[i],
                action=action, logp=float(dist.log_prob(torch.tensor([0, 1]))[i]), value=float(values[i]),
                reward=float(2 * i - 1), duration=i + 1, terminated=i == 1, truncated=False,
                next_value=float(values[i + 1]) if i == 0 else 0.))
        before = model.actor_encoder.graph.embed[0].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        result = update(model, optimizer, [transitions], epochs=2)
        self.assertGreater(result['accepted_epochs'], 0)
        self.assertEqual(result['fast_attempts'], 3)
        self.assertFalse(torch.equal(before, model.actor_encoder.graph.embed[0].weight))
        copy = GraphSlowActorCritic(hidden=32)
        copy.load_state_dict(model.state_dict())
        torch.testing.assert_close(copy(x, mask, present, graphs)[0].probs, model(x, mask, present, graphs)[0].probs)


if __name__ == '__main__':
    unittest.main(verbosity=2)
