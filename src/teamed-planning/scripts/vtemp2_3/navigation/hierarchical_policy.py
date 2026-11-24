"""UAV fast and UGV slow policies over receiver-local world-model graphs.

The archived map/sensor encoder is reused in the UAV branch. Its raw peer and
relay inputs are removed: robot information reaches the policy through the
world model's typed, time-aligned belief nodes. The UGV branch chooses a masked
candidate subgoal and leaves path planning and motor control to the rules.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical

from .graph_dynamics import NODE_COUNT, TOKEN_WIDTH
from .graph_fast_policy import ResidualFastActorCritic
from .hierarchical_runtime import SLOW_FEATURES
from .slow_policy import SlowActorCritic
from .planning_graph import NODE_FEATURES, EDGE_TYPES
from .frozen_navigation import runtime

FAST_SCHEMA = 'vtemp2_world_graph_fast_policy_v1'
SLOW_SCHEMA = 'vtemp2_world_environment_graph_slow_policy_v2'
GRAPH_FAST_SCHEMA = FAST_SCHEMA
GRAPH_SLOW_SCHEMA = SLOW_SCHEMA
RAW_WIDTH = 1826
LOCAL_WIDTH = 34 + 288 + 1088  # Ego state, local sensor tile and map graph.
GRAPH_WIDTH = NODE_COUNT * TOKEN_WIDTH
SLOW_BASE_FEATURES = len(SLOW_FEATURES)
SLOW_GRAPH_FEATURES = SLOW_BASE_FEATURES + GRAPH_WIDTH


class BeliefGraphEncoder(nn.Module):
    """One graph-attention block with explicit missing-node masking.

    There are no node-index embeddings. Robot type, original age, forwarding
    count, uncertainty and ego identity are attributes in each belief token.
    The output is invariant to a consistent permutation of the graph nodes.
    """
    def __init__(self, hidden=64, output=32):
        super().__init__()
        self.hidden = int(hidden)
        self.embed = nn.Sequential(nn.Linear(TOKEN_WIDTH - 1, hidden), nn.Tanh())
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.message = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh())
        self.readout = nn.Sequential(nn.Linear(2 * hidden, output), nn.Tanh())

    def forward(self, tokens):
        if tokens.shape[-2:] != (NODE_COUNT, TOKEN_WIDTH):
            raise ValueError('Wrong belief graph token shape')
        valid = tokens[..., -1] > .5
        # Mask before any affine operation, including NaN-filled padding.
        payload = torch.where(valid[..., None], tokens[..., :-1], 0.)
        encoded = self.embed(payload)
        scores = self.query(encoded) @ self.key(encoded).transpose(-2, -1)
        scores = scores / math.sqrt(self.hidden)
        scores = scores.masked_fill(~valid[..., None, :], -1e9)
        weights = scores.softmax(-1) * valid[..., None, :]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        incoming = weights @ self.value(encoded)
        updated = self.message(torch.cat((encoded, incoming), -1))
        updated = torch.where(valid[..., None], updated, 0.)
        pooled = updated.sum(-2) / valid.sum(-1, keepdim=True).clamp_min(1)
        ego = (payload[..., 59] > .5) & valid
        own = (updated * ego[..., None]).sum(-2) / ego.sum(-1, keepdim=True).clamp_min(1)
        result = self.readout(torch.cat((pooled, own), -1))
        return torch.where(valid.any(-1, keepdim=True), result, 0.)


class GraphFastActorCritic(ResidualFastActorCritic):
    def __init__(self, base):
        super().__init__(base, RAW_WIDTH, BeliefGraphEncoder)
        self.assigned_target = None

    def split(self, x):
        if x.shape[-1] == RAW_WIDTH:
            # Explicit absent-graph diagnostic, also useful during distillation.
            tokens = x.new_zeros((*x.shape[:-1], NODE_COUNT, TOKEN_WIDTH))
        elif x.shape[-1] == RAW_WIDTH + GRAPH_WIDTH:
            tokens = x[..., RAW_WIDTH:].reshape(*x.shape[:-1], NODE_COUNT, TOKEN_WIDTH)
        else:
            raise ValueError('Wrong world-graph fast-policy feature width')
        raw = x[..., :RAW_WIDTH].clone()
        # Never give the actor both raw multi-hop records and their estimates.
        raw[..., LOCAL_WIDTH:] = 0.
        return raw, tokens

    def encode_graph(self, base, position, observation):
        if self.assigned_target is not None:
            # Assign the target before encoding map nodes, which also use its relative position.
            # Ground location and link features still require normal reception.
            delta = self.assigned_target - position[:2]
            if base[10]:
                base[8:10] = (delta - base[6:8] * 95.) / 350.
            base[4:6] = delta / 350.
            base[10] = 1.
        return self.base.encode_graph(base, position, observation)


def make_graph_fast(repo, *, random_init=False):
    """Construct a graph policy, optionally without reading learned weights."""
    return GraphFastActorCritic(runtime(repo, random_init=True)[2] if random_init else runtime(repo)[2])


class EnvironmentGraphEncoder(nn.Module):
    """Sparse topology messages and candidate-conditioned map attention.

    Map edges carry relation types: a task/spatial relation cannot masquerade
    as a traversable road. Graph node processing happens once per state.
    """
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Sequential(nn.Linear(NODE_FEATURES, hidden), nn.Tanh())
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden + len(EDGE_TYPES) + 2, hidden), nn.Tanh())
            for _ in range(2)])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh()) for _ in range(2)])
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.tasks = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.Tanh())
        self.distance_scale = nn.Parameter(torch.tensor(0.))

    def forward(self, graph, query, candidate_xy):
        nodes = torch.as_tensor(graph['nodes'], device=query.device, dtype=query.dtype)
        if nodes.ndim != 2 or nodes.shape[1] != NODE_FEATURES or not len(nodes):
            raise ValueError('Environment nodes require nonempty N x 16 features')
        valid = torch.as_tensor(graph.get('node_mask', torch.ones(len(nodes), dtype=torch.bool)),
                                device=query.device, dtype=torch.bool)
        if valid.shape != (len(nodes),) or not bool(valid.any()):
            raise ValueError('Environment graph requires valid nodes')
        nodes = torch.where(valid[:, None], nodes, 0.)
        if not bool(torch.isfinite(nodes).all()):
            raise ValueError('Non-finite valid environment node')
        edges = torch.as_tensor(graph['edges'], device=query.device, dtype=torch.long)
        kinds = torch.as_tensor(graph['edge_types'], device=query.device, dtype=torch.long)
        if edges.ndim != 2 or edges.shape[0] != 2 or kinds.shape != (edges.shape[1],):
            raise ValueError('Environment edges and relation types disagree')
        if edges.numel() and (bool((edges < 0).any()) or bool((edges >= len(nodes)).any())):
            raise ValueError('Environment edge references a missing node')
        if kinds.numel() and (bool((kinds < 0).any()) or bool((kinds >= len(EDGE_TYPES)).any())):
            raise ValueError('Unknown environment edge relation')
        if edges.numel():
            keep = valid[edges[0]] & valid[edges[1]]
            edges, kinds = edges[:, keep], kinds[keep]
        encoded = self.embed(nodes)
        encoded = torch.where(valid[:, None], encoded, 0.)
        source, destination = edges
        onehot = torch.nn.functional.one_hot(kinds, len(EDGE_TYPES)).to(query.dtype)
        relative = nodes[source, :2] - nodes[destination, :2]
        for message, update in zip(self.messages, self.updates):
            incoming = torch.zeros_like(encoded)
            degree = encoded.new_zeros((len(nodes), 1))
            if edges.numel():
                payload = torch.cat((encoded[source], encoded[destination], onehot, relative), -1)
                incoming.index_add_(0, destination, message(payload))
                degree.index_add_(0, destination, torch.ones_like(destination[:, None], dtype=query.dtype))
            encoded = update(torch.cat((encoded, incoming / degree.clamp_min(1)), -1))
            encoded = torch.where(valid[:, None], encoded, 0.)
        scores = query @ self.key(encoded).T / math.sqrt(self.hidden)
        squared_distance = (candidate_xy[:, None] - nodes[None, :, :2]).square().sum(-1)
        scores = scores - torch.nn.functional.softplus(self.distance_scale) * squared_distance
        scores = scores.masked_fill(~valid[None], -torch.inf)
        attended = scores.softmax(-1) @ self.value(encoded)
        # Keep the final/current/ego nodes explicit even in a large road graph;
        # their importance must not vanish under global mean pooling.
        roles = nodes[:, 2:5].T * valid[None]
        if not bool((roles.sum(-1) > 0).all()):
            raise ValueError('Environment graph lacks ego/final/current task nodes')
        tasks = roles @ encoded / roles.sum(-1, keepdim=True).clamp_min(1)
        return attended, self.tasks(tasks.flatten()).expand(len(query), -1)


class CandidateGraphEncoder(nn.Module):
    """Fuse task/map/robot evidence in a common global-axis relative frame."""
    def __init__(self, hidden=64):
        super().__init__()
        self.graph = BeliefGraphEncoder(hidden, hidden)
        self.environment = EnvironmentGraphEncoder(hidden)
        self.candidate = nn.Sequential(nn.Linear(SLOW_BASE_FEATURES + 2, hidden), nn.Tanh())
        self.fusion = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh())

    def forward(self, x, environment_graph, present):
        single = x.ndim == 2
        if single:
            rows, present_rows = x[None], present[None]
            graphs = [environment_graph]
        else:
            rows, present_rows = x, present
            graphs = environment_graph
        if not isinstance(graphs, (list, tuple)) or len(graphs) != len(rows):
            raise ValueError('One environment graph per slow observation is required')
        outputs = []
        for row, valid, graph in zip(rows, present_rows, graphs):
            if not isinstance(graph, dict):
                raise ValueError('Slow v2 requires the observed environment/task graph')
            xy = torch.as_tensor(graph['candidate_xy'], device=x.device, dtype=x.dtype)
            if xy.shape != (len(row), 2):
                raise ValueError('Candidate XY and candidate features disagree')
            xy = torch.where(valid[:, None], xy, 0.)
            if not bool(torch.isfinite(xy).all()):
                raise ValueError('Non-finite valid candidate coordinates')
            candidate = self.candidate(torch.cat((row[:, :SLOW_BASE_FEATURES], xy), -1))
            first = int(valid.to(torch.int64).argmax())
            robots = row[first, SLOW_BASE_FEATURES:].reshape(NODE_COUNT, TOKEN_WIDTH)
            robot_context = self.graph(robots).expand(len(row), -1)
            local, tasks = self.environment(graph, candidate, xy)
            outputs.append(self.fusion(torch.cat((candidate, robot_context, local, tasks), -1)))
        result = torch.stack(outputs)
        return result[0] if single else result


class GraphSlowActorCritic(SlowActorCritic):
    """SMDP candidate actor/critic; compatible with policy.update rollbacks.

    Graph modules live under actor_encoder and critic_encoder so the existing
    gradient clipping, optimizer updates and checkpoint handling include them.
    """
    def __init__(self, features=SLOW_GRAPH_FEATURES, hidden=64):
        nn.Module.__init__(self)
        if int(features) != SLOW_GRAPH_FEATURES:
            raise ValueError('Wrong world-graph slow-policy feature width')
        self.features, self.hidden = int(features), int(hidden)
        self.actor_encoder = CandidateGraphEncoder(hidden)
        self.actor_head = nn.Linear(hidden, 1)
        self.critic_encoder = CandidateGraphEncoder(hidden)
        self.critic_head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.actor_head.weight, gain=.01)
        nn.init.zeros_(self.actor_head.bias)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        weight = state_dict.get('actor_encoder.candidate.0.weight')
        if weight is not None and weight.shape[-1] != SLOW_BASE_FEATURES + 2:
            raise ValueError('Slow v1 checkpoint lacks environment/task inputs; initialize and warm-start a v2 slow policy')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x, mask, present, environment_graph=None):
        if x.shape[-1] != self.features or x.shape[:-1] != mask.shape or mask.shape != present.shape:
            raise ValueError('Candidate feature and mask shapes disagree')
        if not bool(mask.any(-1).all()) or bool((mask & ~present).any()):
            raise ValueError('Every state needs a valid action; padding cannot be selectable')
        if environment_graph is None:
            raise ValueError('Slow v2 requires an environment graph with final/current task nodes')
        x = torch.where(present[..., None], x, 0.)
        logits = self.actor_head(self.actor_encoder(x, environment_graph, present)).squeeze(-1).masked_fill(~mask, -torch.inf)
        encoded = self.critic_encoder(x, environment_graph, present) * present[..., None]
        pooled = encoded.sum(-2) / present.sum(-1, keepdim=True).clamp_min(1)
        return Categorical(logits=logits), self.critic_head(pooled).squeeze(-1)
