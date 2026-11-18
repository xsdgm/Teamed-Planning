"""Receiver-local temporal graph dynamics shared by UAVs and the ground vehicle.

Each graph contains only locally known robot nodes. Original source age and hop
count affect message attention explicitly. The network jointly aligns received
UAV states to the current clock and predicts future motion, without querying an
actor before inference. Ground motion uses the supplied kinematic rule prior;
learned spread describes its possible error. Current local sensor/map evidence
conditions the neural dynamics as ego-node attributes. CBF attributes are
retained by the runtime graph, rather than predicted or fabricated here.

Units match wm_protocol: state positions relative to each source packet origin
in units of 20 m, velocity/actions in units of 6 m/s, and one-second transitions.
Predictive standard deviations are not certified safety bounds.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
import torch
from torch import nn
from torch.nn import functional as F

from .residual_world import bounded_velocity
from .telemetry import HISTORY_FEATURES, CONTEXT_FEATURES, POSITION_SCALE, SPEED_SCALE

MODEL_SCHEMA = 'vtemp2_temporal_robot_graph_world_v2'
ROAD_MODEL_SCHEMA = 'vtemp2_temporal_robot_graph_world_road_v3'
NODE_IDS = ('ground_0', 'drone_0', 'drone_1', 'drone_2')
NODE_COUNT = len(NODE_IDS)
HORIZON = 4
TOKEN_WIDTH = (HORIZON + 1) * 8 + HORIZON * 4 + 8
LOCAL_EVIDENCE_WIDTH = 12 * 12 * 2 + 32 * 5
INPUT_KEYS = ('history', 'history_mask', 'context', 'baseline_states',
              'baseline_actions', 'node_mask', 'node_type', 'ego_mask', 'state_observed',
              'local_evidence')
ROAD_INPUT_KEYS = (*INPUT_KEYS, 'road_topology')
ROAD_SUMMARY_FIELDS = ('nodes', 'edges', 'components', 'independent_cycles', 'anchor_component_fraction',
                       'largest_component_fraction', 'ego_to_anchor_m', 'has_target', 'ego_to_target_m',
                       'target_to_anchor_m', 'anchors_connected', 'anchor_path_m',
                       'nearest_reachable_to_target_m')
_ROAD_SECTORS = 8
_ROAD_RADII = (25., 95., 200.)
_ROAD_BINS = (1 + len(_ROAD_RADII)) * _ROAD_SECTORS
_ROAD_NEAR = np.triu_indices(32, k=1)
_ROAD_PAIRS = np.triu_indices(_ROAD_BINS)
ROAD_TOPOLOGY_WIDTH = len(ROAD_SUMMARY_FIELDS) + 4 * _ROAD_BINS + len(_ROAD_PAIRS[0]) + len(_ROAD_NEAR[0])


def build_road_topology(local_graph, own_xy, target_xy=None, *, nearby_nodes=None):
    """Encode all observed roads in a bounded tensor, keeping exact near-node edges.

    A nearest map node anchors each supplied coordinate. Anchor distances remain
    explicit so graph connectivity does not imply a traversable last-meter link.
    Sectors follow the supplied target bearing (global X when absent/coincident).
    The last radial band is unbounded. Every node and undirected edge contributes,
    including disconnected components and edges beyond the nearest 32 nodes.
    """
    result = np.zeros(ROAD_TOPOLOGY_WIDTH, np.float32)
    own = np.asarray(own_xy, dtype=float)[:2]
    direction = np.zeros(2) if target_xy is None else np.asarray(target_xy, dtype=float)[:2] - own
    distance = np.linalg.norm(direction)
    if own.shape != (2,) or not np.isfinite(own).all() or not np.isfinite(distance):
        raise ValueError('Road features require finite receiver and task coordinates')
    result[7:9] = (target_xy is not None, distance / 95.)
    if local_graph is None:
        return result
    graph = local_graph.graph
    if not len(graph):
        return result

    nodes = list(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    positions = np.asarray([graph.nodes[node]['position'][:2] for node in nodes], dtype=float)
    if positions.shape != (len(nodes), 2) or not np.isfinite(positions).all():
        raise ValueError('Received road nodes require finite XY positions')
    delta = positions - own
    ranges = np.linalg.norm(delta, axis=1)
    if nearby_nodes is None:
        near = sorted(range(len(nodes)), key=lambda i: (ranges[i], str(nodes[i])))[:32]
    else:
        near = [index[node] for node in nearby_nodes]
        if len(near) != min(32, len(nodes)) or len(set(near)) != len(near):
            raise ValueError('Expected the ordered nearest road nodes used by local_evidence')
    anchor = near[0]
    edges = np.asarray([(index[a], index[b]) for a, b in graph.edges], dtype=np.intp).reshape(-1, 2)
    a, b = edges.T
    lengths = np.linalg.norm(positions[a] - positions[b], axis=1)
    roads = coo_matrix((lengths, (a, b)), shape=(len(nodes), len(nodes))).tocsr()
    components, labels = connected_components(roads, directed=False)
    counts = np.bincount(labels)
    reachable = labels == labels[anchor]
    result[:7] = (np.log1p(len(nodes)), np.log1p(len(edges)), np.log1p(components),
                  np.log1p(len(edges) - len(nodes) + components), reachable.mean(),
                  counts.max() / len(nodes), ranges[anchor] / 95.)
    if target_xy is not None:
        to_target = np.linalg.norm(delta - direction, axis=1)
        closest = np.flatnonzero(to_target == to_target.min())
        goal_node = min(closest, key=lambda i: str(nodes[i]))
        result[9:13] = (to_target[goal_node] / 95., reachable[goal_node], 0.,
                        to_target[reachable].min() / 95.)
        if reachable[goal_node]:
            routes = dijkstra(roads, directed=False, indices=anchor)
            result[11] = routes[goal_node] / 95.

    bearing = math.atan2(direction[1], direction[0])
    angles = (np.arctan2(delta[:, 1], delta[:, 0]) - bearing) % (2 * np.pi)
    sectors = np.minimum((angles * (_ROAD_SECTORS / (2 * np.pi))).astype(int), _ROAD_SECTORS - 1)
    bins = np.searchsorted(_ROAD_RADII, ranges, side='right') * _ROAD_SECTORS + sectors
    bin_size = np.bincount(bins, minlength=_ROAD_BINS)
    bin_reachable = np.bincount(bins[reachable], minlength=_ROAD_BINS)
    bin_ranges = np.bincount(bins, weights=ranges / 95., minlength=_ROAD_BINS)
    # Sum incident road lengths and divide by two so every physical road
    # contributes its length once across both endpoints.
    bin_lengths = (np.bincount(bins[a], weights=lengths, minlength=_ROAD_BINS)
                   + np.bincount(bins[b], weights=lengths, minlength=_ROAD_BINS)) / 2.
    start = len(ROAD_SUMMARY_FIELDS)
    stop = start + 4 * _ROAD_BINS
    result[start:stop] = np.column_stack((np.log1p(bin_size), np.log1p(bin_reachable),
        bin_ranges / np.maximum(1, bin_size), np.log1p(bin_lengths / 95.))).ravel()
    first, second = np.minimum(bins[a], bins[b]), np.maximum(bins[a], bins[b])
    crossings = np.bincount(first * _ROAD_BINS + second, minlength=_ROAD_BINS**2).reshape(_ROAD_BINS, _ROAD_BINS)
    result[stop:stop + len(_ROAD_PAIRS[0])] = np.log1p(crossings[_ROAD_PAIRS])
    local_index = np.full(len(nodes), -1, np.intp)
    local_index[near] = np.arange(len(near))
    i, j = local_index[a], local_index[b]
    included = (i >= 0) & (j >= 0)
    adjacency = np.zeros((32, 32), bool)
    adjacency[np.minimum(i[included], j[included]), np.maximum(i[included], j[included])] = True
    result[-len(_ROAD_NEAR[0]):] = adjacency[_ROAD_NEAR]
    return result


def latest_history(history, history_mask):
    """Last valid item, including histories with arbitrary padding placement."""
    times = torch.arange(history.shape[-2], device=history.device)
    index = torch.where(history_mask.bool(), times, -1).amax(-1).clamp_min(0)
    return history.gather(-2, index[..., None, None].expand(*index.shape, 1, history.shape[-1])).squeeze(-2)


def sanitize_local_evidence(evidence):
    """Remove unknown raster/map payloads before they enter a neural layer.

    First 144 values are occupancy, next 144 are known masks. Remaining values
    are 32 received map nodes [relative x, relative y, confidence, degree, valid].
    The caller supplies receiver evidence, never an omniscient simulator map.
    """
    sensor, known = evidence[..., :144], evidence[..., 144:288]
    nodes = evidence[..., 288:].reshape(*evidence.shape[:-1], 32, 5)
    if not bool(torch.isfinite(known).all()) or not bool(torch.isfinite(nodes[..., -1]).all()):
        raise ValueError('Nonfinite local-evidence validity mask')
    sensor = torch.where(known > .5, sensor, 0.)
    nodes = torch.where((nodes[..., -1] > .5)[..., None], nodes, 0.)
    return torch.cat((sensor, known, nodes.flatten(-2)), -1)


class AgeAwareGraphAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        # Pairwise geometry, age difference, and sender/receiver entity type.
        self.edge_bias = nn.Sequential(nn.Linear(5, hidden // 2 or 1), nn.Tanh(),
                                       nn.Linear(hidden // 2 or 1, 1))
        self.age_penalty = nn.Parameter(torch.tensor(0.))
        self.hop_penalty = nn.Parameter(torch.tensor(0.))
        self.update = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, nodes, position, age, hop, kind, valid):
        logits = self.query(nodes) @ self.key(nodes).transpose(-1, -2) / math.sqrt(nodes.shape[-1])
        relative = position[:, None, :, :] - position[:, :, None, :]
        age_difference = (age[:, None, :] - age[:, :, None]).abs()[..., None]
        receiver_kind = kind[:, :, None, None].expand(-1, -1, nodes.shape[1], -1)
        sender_kind = kind[:, None, :, None].expand(-1, nodes.shape[1], -1, -1)
        edges = torch.cat((relative, age_difference, receiver_kind, sender_kind), -1)
        logits = logits + self.edge_bias(edges).squeeze(-1)
        # Nonnegative freshness penalties cannot learn to reverse their sign.
        logits = logits - F.softplus(self.age_penalty) * age[:, None, :]
        logits = logits - F.softplus(self.hop_penalty) * hop[:, None, :]
        logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
        weights = logits.softmax(-1) * valid[:, :, None]
        messages = weights @ self.value(nodes)
        output = torch.tanh(self.norm(nodes + self.update(messages)))
        return torch.where(valid[..., None], output, 0.), weights


class GraphWorldModel(nn.Module):
    def __init__(self, hidden=64, horizon=HORIZON, *, road_topology=False):
        super().__init__()
        if hidden < 1 or horizon < 1:
            raise ValueError('Positive hidden width and prediction horizon required')
        self.hidden, self.horizon = int(hidden), int(horizon)
        self.schema = ROAD_MODEL_SCHEMA if road_topology else MODEL_SCHEMA
        self.input_keys = ROAD_INPUT_KEYS if road_topology else INPUT_KEYS
        self.encoder = nn.GRUCell(HISTORY_FEATURES, hidden)
        self.kind = nn.Embedding(2, hidden)
        self.context = nn.Sequential(nn.Linear(hidden + CONTEXT_FEATURES + 1, hidden), nn.Tanh())
        self.local_evidence = nn.Sequential(nn.Linear(LOCAL_EVIDENCE_WIDTH, hidden), nn.Tanh())
        self.road_encoder = (nn.Sequential(nn.Linear(ROAD_TOPOLOGY_WIDTH, hidden), nn.Tanh())
                             if road_topology else None)
        self.graph = AgeAwareGraphAttention(hidden)
        self.initial = nn.Linear(hidden, 8)
        self.decoder = nn.GRUCell(11, hidden)
        self.action = nn.Linear(hidden, 4)
        self.innovation = nn.Linear(hidden, 4)
        for layer in (self.initial, self.action, self.innovation):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def scale(raw):
        return raw.clamp(-4., 2.).exp() * .2 + .01

    def forward(self, history, history_mask, context, baseline_states, baseline_actions,
                node_mask, node_type, ego_mask, state_observed, local_evidence, road_topology=None):
        if history.ndim != 4 or history.shape[1] != NODE_COUNT or history.shape[-1] != HISTORY_FEATURES:
            raise ValueError('Expected a batch of four-node received robot graphs')
        batch, count, steps, _ = history.shape
        expected = dict(history_mask=(batch, count, steps), context=(batch, count, CONTEXT_FEATURES),
            baseline_states=(batch, count, self.horizon + 1, 4),
            baseline_actions=(batch, count, self.horizon, 2),
            node_mask=(batch, count), node_type=(batch, count), ego_mask=(batch, count),
            state_observed=(batch, count), local_evidence=(batch, LOCAL_EVIDENCE_WIDTH))
        values = dict(history_mask=history_mask, context=context, baseline_states=baseline_states,
                      baseline_actions=baseline_actions, node_mask=node_mask, node_type=node_type,
                      ego_mask=ego_mask, state_observed=state_observed, local_evidence=local_evidence)
        if any(tuple(values[key].shape) != shape for key, shape in expected.items()):
            raise ValueError('Graph input dimensions disagree')
        if self.road_encoder is not None:
            if road_topology is None or road_topology.shape != (batch, ROAD_TOPOLOGY_WIDTH):
                raise ValueError('Road-aware world model requires receiver road topology for each graph')
            if not bool(torch.isfinite(road_topology).all()):
                raise ValueError('Nonfinite road topology feature')
        valid, ego = node_mask.bool(), ego_mask.bool()
        if not bool((ego.sum(-1) == 1).all()) or bool((ego & ~valid).any()):
            raise ValueError('Every graph requires exactly one known ego node')
        observed = state_observed.bool()
        if bool((ego & ~observed).any()) or bool((observed & ~valid).any()):
            raise ValueError('Observed states must be known and include the ego')
        history_valid = history_mask.bool() & valid[..., None]
        if bool((valid & ~history_valid.any(-1)).any()):
            raise ValueError('Known nodes require received history')
        if bool((valid & ((node_type != 0) & (node_type != 1))).any()):
            raise ValueError('Unknown robot type')
        # Sanitize before the neural layers, including padded NaN/Inf payloads.
        history = torch.where(history_valid[..., None], history, 0.)
        context = torch.where(valid[..., None], context, 0.)
        baseline_states = torch.where(valid[..., None, None], baseline_states, 0.)
        baseline_actions = torch.where(valid[..., None, None], baseline_actions, 0.)
        node_type = torch.where(valid, node_type, 0).long()
        local_evidence = sanitize_local_evidence(local_evidence)
        if any(not bool(torch.isfinite(value).all()) for value in
               (history, context, baseline_states, baseline_actions, local_evidence)):
            raise ValueError('Nonfinite received graph evidence')
        h = history.new_zeros((batch * count, self.hidden))
        for t in range(steps):
            proposed = self.encoder(history[:, :, t].reshape(batch * count, -1), h)
            h = torch.where(history_valid[:, :, t].reshape(-1, 1), proposed, h)
        h = h.reshape(batch, count, self.hidden) + self.kind(node_type)
        h = self.context(torch.cat((h, context, ego[..., None].to(history.dtype)), -1))
        h = h + self.local_evidence(local_evidence)[:, None, :]
        if self.road_encoder is not None:
            h = h + self.road_encoder(road_topology)[:, None, :]
        last = latest_history(history, history_valid)
        age, hop = last[..., 9].clamp_min(0.), last[..., 11].clamp_min(0.)
        position = baseline_states[..., 0, :2] * (POSITION_SCALE / 95.) - context[..., :2]
        h, attention = self.graph(h, position, age, hop, node_type.to(history.dtype), valid)
        initial = self.initial(h)
        learned_motion = valid & (node_type == 0)
        adjusted = baseline_states[..., 0, :] + initial[..., :4]
        adjusted = torch.cat((adjusted[..., :2], bounded_velocity(adjusted[..., 2:])), -1)
        current = torch.where((learned_motion & ~observed)[..., None], adjusted, baseline_states[..., 0, :])
        spread = self.scale(initial[..., 4:]) + .02 * (age * 30.)[..., None]
        # Current locally sensed pose/velocity are measurements. No learned
        # adjustment replaces them; future motion remains a prediction.
        spread = torch.where(observed[..., None], 0., spread)
        states, stds, actions, action_stds = [current], [spread], [], []
        ratio = SPEED_SCALE / POSITION_SCALE
        h = h.reshape(batch * count, self.hidden)
        for t in range(self.horizon):
            timestep = current.new_full((batch, count, 1), (t + 1) / self.horizon)
            inputs = torch.cat((current, baseline_states[..., t + 1, :],
                                baseline_actions[..., t, :], timestep), -1)
            h = self.decoder(inputs.reshape(batch * count, -1), h)
            correction = self.action(h).reshape(batch, count, 4)
            innovation = self.innovation(h).reshape(batch, count, 4)
            action = bounded_velocity(baseline_actions[..., t, :] + correction[..., :2])
            action = torch.where(learned_motion[..., None], action, baseline_actions[..., t, :])
            action_std = self.scale(correction[..., 2:])
            position = current[..., :2] + ratio * action + .1 * innovation[..., :2].tanh()
            predicted = torch.cat((position, action), -1)
            current = torch.where(learned_motion[..., None], predicted, baseline_states[..., t + 1, :])
            position_std = (spread[..., :2].square() + ratio**2 * action_std.square()
                            + (.05 * self.scale(innovation[..., 2:])).square()).sqrt()
            spread = torch.cat((position_std, action_std), -1)
            states.append(current)
            stds.append(spread)
            actions.append(action)
            action_stds.append(action_std)
        result = dict(states=torch.stack(states, 2), state_std=torch.stack(stds, 2),
                      actions=torch.stack(actions, 2), action_std=torch.stack(action_stds, 2))
        result = {key: torch.where(valid[..., None, None], value, 0.) for key, value in result.items()}
        result['attention'] = attention
        return result


def infer(model, batch):
    """Whitelist inputs: labels, absolute simulator state, and future data stay out."""
    return model(**{key: batch[key] for key in model.input_keys})


def planning_readout(prediction, batch, *, belief, current_target, candidate_targets):
    """Complete UGV belief graph: predicted robots plus received map and tasks.

    The dynamics head predicts moving robots. Observed static map topology and
    the fixed mission goal are carried through this readout, never regressed or
    replaced by predicted occupancy. All spatial fields share ego-relative
    global XY axes. Only receiver-local belief is accepted here.
    """
    from .planning_graph import build_environment_graph
    tokens = belief_tokens(prediction, batch)[0].detach().cpu().numpy()
    environment = build_environment_graph(belief, current_target, tokens,
                                         candidate_targets=candidate_targets)
    return tokens, environment


def gaussian_loss(mean, std, truth, mask):
    # Select before arithmetic so censored NaN labels cannot poison gradients.
    valid = mask.bool().unsqueeze(-1).expand_as(mean)
    if not bool(valid.any()):
        return mean.sum() * 0.
    target, predicted, spread = truth[valid], mean[valid], std[valid].clamp_min(1e-4)
    if not bool(torch.isfinite(target).all()):
        raise ValueError('Nonfinite supervised label inside its valid mask')
    return (.5 * ((target - predicted) / spread).square() + spread.log()
            + .5 * math.log(2 * math.pi)).mean()


def objective(prediction, batch):
    state_mask = batch['state_mask'].bool() & batch['node_mask'].bool()[..., None]
    state_mask = state_mask.clone()
    state_mask[..., 0] &= ~batch['state_observed'].bool()
    action_mask = batch['action_mask'].bool() & batch['node_mask'].bool()[..., None]
    return (gaussian_loss(prediction['states'], prediction['state_std'], batch['target_states'], state_mask)
            + .5 * gaussian_loss(prediction['actions'], prediction['action_std'],
                                  batch['target_actions'], action_mask))


def belief_tokens(prediction, batch):
    """Four-second graph readout for both policies; final channel is validity.

    Layout: states[5,4], state_std[5,4], actions[4,2], action_std[4,2],
    age/30, hops/8, is_ground, is_ego, task_known, task_age/30,
    state_observed, valid. Positions/spreads use ego-relative /95 m.
    Neither model inference nor this readout modifies original message clocks.
    """
    states = prediction['states']
    if states.shape[-2:] != (HORIZON + 1, 4):
        raise ValueError('Policy belief tokens require the four-second prediction horizon')
    valid = batch['node_mask'].bool()
    context = torch.where(valid[..., None], batch['context'], 0.)
    relative = states[..., :2] * (POSITION_SCALE / 95.) - context[..., None, :2]
    states = torch.cat((relative, states[..., 2:]), -1)
    std = prediction['state_std']
    std = torch.cat((std[..., :2] * (POSITION_SCALE / 95.), std[..., 2:]), -1)
    history_valid = batch['history_mask'].bool() & valid[..., None]
    history = torch.where(history_valid[..., None], batch['history'], 0.)
    last = latest_history(history, history_valid)
    metadata = torch.stack((last[..., 9], last[..., 11], batch['node_type'].to(states.dtype),
        batch['ego_mask'].to(states.dtype), last[..., 8], last[..., 10],
        batch['state_observed'].to(states.dtype), valid.to(states.dtype)), -1)
    result = torch.cat((states.flatten(-2), std.flatten(-2), prediction['actions'].flatten(-2),
                        prediction['action_std'].flatten(-2), metadata), -1)
    if result.shape[-1] != TOKEN_WIDTH:
        raise ValueError('Unexpected graph belief token width')
    return torch.where(valid[..., None], result, 0.)
