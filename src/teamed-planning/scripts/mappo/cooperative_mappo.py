"""Shared one-hop teammate features and navigation for relay-attention policies."""
import numpy as np
import torch
from torch import nn
from shared.drone.communication import ChainState
from mappo.velocity_mappo import VelocityNavigator
from mappo.graph_attention_mappo import (GraphEncoder,
    GRAPH_BLOCK, MAX_FLEET, prefix_width, encode_graph)

PEER_FEATURES = 8
PEER_BLOCK = MAX_FLEET*(PEER_FEATURES+1)
MAP_WIDTH = prefix_width(2)+GRAPH_BLOCK
RAW_WIDTH = MAP_WIDTH+PEER_BLOCK
ENCODED_WIDTH = prefix_width(2)+64+32


class CooperativeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.map = GraphEncoder(prefix_width(2))
        self.peer_query = nn.Linear(prefix_width(2)+64, 32)
        self.peer_key = nn.Linear(PEER_FEATURES, 32)
        self.peer_value = nn.Linear(PEER_FEATURES, 32)

    def forward(self, x):
        own = self.map(x[..., :MAP_WIDTH])
        peers = x[..., MAP_WIDTH:MAP_WIDTH+MAX_FLEET*PEER_FEATURES].reshape(*x.shape[:-1], MAX_FLEET, PEER_FEATURES)
        mask = x[..., -MAX_FLEET:] > 0
        logits = (self.peer_query(own).unsqueeze(-2)*self.peer_key(peers)).sum(-1)/np.sqrt(32.)
        weights = logits.masked_fill(~mask, -1e9).softmax(-1)*mask
        pooled = (weights.unsqueeze(-1)*self.peer_value(peers)).sum(-2)
        return torch.cat((own, pooled), -1)


class CooperativeCriticEncoder(nn.Module):
    def __init__(self, global_features, encoder, raw_width, encoded_width):
        super().__init__()
        self.encoder = encoder
        self.global_features = global_features
        self.raw_width, self.encoded_width = raw_width, encoded_width

    def forward(self, x):
        raw, width = self.raw_width, self.encoded_width
        valid = x[..., -1] > 0
        flat = x[..., :raw].reshape(-1, raw)
        ids = valid.reshape(-1).nonzero().squeeze(-1)
        values = self.encoder(flat[ids])
        encoded = values.new_zeros((len(flat), width)).index_copy(0, ids, values)
        pooled = encoded.reshape(*valid.shape, width).sum(-2)/valid.sum(-1, keepdim=True).clamp_min(1)
        if self.global_features:
            pooled = torch.cat((pooled, x[..., 0, raw:raw+self.global_features]), -1)
        return pooled


def encode_cooperative_graph(base, position, observation, prefix):
    tokens = np.zeros((MAX_FLEET, PEER_FEATURES), np.float32)
    valid = np.zeros(MAX_FLEET, np.float32)
    cooperation = observation.get('cooperation', {})
    peers = cooperation.get('peers', {})
    if len(peers) >= MAX_FLEET:
        raise ValueError('Cooperative graph supports at most 16 drones including self')
    for i, (key, q) in enumerate(peers.items()):
        delta = np.asarray(q)-position
        relative_velocity = cooperation['velocities'][key]
        tokens[i] = [*(delta/95.), *(relative_velocity[:2]/12.),
                     float(key in cooperation['required']), np.linalg.norm(delta)/95., 1.]
        valid[i] = 1.
    return np.r_[encode_graph(base, position, observation, prefix), tokens.ravel(), valid].astype(np.float32)


class CooperativeNavigator(VelocityNavigator):
    def make_chain_state(self):
        # No fleet ordering is needed even when device names are arbitrary.
        return ChainState(self.drone_id, 1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Reuse reception-only task caching; its legacy rank/endpoints are never encoded.
        self.chain = ChainState(self.drone_id, 1)
        self.required_links = ()

    def cooperation_features(self, state, p, obs):
        data = obs['cooperation']
        peers, required = data['peers'], data['required']
        ends = [np.asarray(peers[k]) for k in required if k in peers]
        if '__ground__' in required:
            ends.append(np.asarray(data['ground']))
        state[12:21] = 0.
        state[12] = float('__ground__' in required)
        state[13] = len(required)/MAX_FLEET
        state[14] = float(data['ground'] is not None)
        if ends:
            delta = np.asarray(ends)-p
            state[15:17] = delta[:, :2].mean(0)/95.
            state[17] = 1.
            farthest = np.argmax(np.linalg.norm(delta, axis=1))
            state[18:20] = delta[farthest, :2]/95.
            state[20] = 1.
        state[26] = float(not required)
        state[28:32] = 0.
        if peers:
            delta = np.asarray(list(peers.values()))-p
            state[28:30] = delta[:, :2].mean(0)/95.
            state[30] = np.linalg.norm(delta, axis=1).min()/95.
        state[31] = np.log1p(len(peers))/np.log(16.)
        state[33] = max((np.linalg.norm(q-p)/95. for q in ends), default=0.)

    def reset(self):
        super().reset()
        self.chain = ChainState(self.drone_id, 1)
        self.required_links = ()
