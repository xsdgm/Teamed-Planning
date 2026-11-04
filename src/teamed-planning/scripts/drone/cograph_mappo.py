"""CoGraph v1: multi-vehicle, multi-target cooperative navigation-graph building.

The drones build the option space: a delivered navigation graph from the
vehicle team to EVERY task target. Which vehicle drives to which target is a
vehicle-side decision (not part of this policy). Each vehicle broadcasts all
targets with its own route status (connected on its fused graph or not); the
drones see every target as an attention token, so they can split the work.

Everything else is v8: map GAT, one-hop teammate attention, the relay cache with
age/hop attention, the RNG backbone and the CBF. Vehicles are separate
backbone nodes keyed by their robot_id. The legacy goal slots carry the
drone's nearest still-unconnected target ("focus"), a deterministic feature of
the received target list, so v8 weights warm-start the policy.
"""
import numpy as np
import torch
from torch import nn
from drone.chain_state import ChainState
from drone.cooperative_mappo import CooperativeCriticEncoder, CooperativeNavigator
from drone.graph_attention_mappo import GraphAttentionActorCritic, MAX_FLEET, prefix_width
from drone.relay_attention_mappo import (RelayActorCritic, RelayEncoder, MODES,
    RAW_WIDTH as RELAY_WIDTH, ENCODED_WIDTH as RELAY_ENCODED)

POLICY_ARCHITECTURE = 'cograph_v1'
FEATURE_VERSION = 9
MAX_TARGETS = 8
# Token layout: 0-1 relative target position /350 m, 2 distance /350 m,
# 3 route connected (any fresh vehicle report), 4 this drone's focus target,
# 5 report age /10 s, 6 target distance from the nearest vehicle /350 m.
TARGET_FEATURES = 7
TARGET_WIDTH, TARGET_STATS = 32, 3
RAW_WIDTH = RELAY_WIDTH+MAX_TARGETS*(TARGET_FEATURES+1)
ENCODED_WIDTH = RELAY_ENCODED+TARGET_WIDTH+TARGET_STATS
# Training-only centralized state; see CoGraphExperiment.critic_state.
GLOBAL_FEATURES = 16


class MultiTargetTaskState(ChainState):
    """Reception-only mission cache for several vehicles (no fleet truth).

    Missions arrive directly or forwarded hop by hop in teammates' broadcasts
    with their original source time; stale ones expire after `timeout`.
    """
    def __init__(self, drone_id, timeout=10.):
        super().__init__(drone_id, 1, timeout)
        self.missions = {}
        self.self_position = None

    def update(self, observation):
        super().update(observation)
        for rid, task in observation.get('ground_robots', {}).items():
            if task.get('targets') is None or task.get('position') is None:
                continue
            source = float(task.get('source_time', self.now))
            stored = self.missions.get(rid)
            if -.01 <= self.now-source <= self.timeout and (stored is None or source >= stored[2]):
                self.missions[rid] = (np.asarray(task['position'], float)[:2],
                                      np.asarray(task['targets'], float).reshape(-1, 3), source)
        self.missions = {k: v for k, v in self.missions.items() if self.now-v[2] <= self.timeout}

    def view(self, position):
        """Merged targets (the freshest list), any-vehicle route status, focus index."""
        if not self.missions:
            return None
        fresh = max(self.missions.values(), key=lambda m: m[2])
        targets = fresh[1][:, :2]
        connected = np.zeros(len(targets), bool)
        for _, rows, _ in self.missions.values():
            if len(rows) == len(targets):
                connected |= rows[:, 2] > .5
        p = np.asarray(position, float)[:2]
        vehicles = np.asarray([m[0] for m in self.missions.values()])
        distance = np.linalg.norm(targets-p, axis=1)
        pending = np.flatnonzero(~connected)
        focus = int(pending[np.argmin(distance[pending])] if len(pending) else np.argmin(distance))
        return dict(targets=targets, connected=connected, focus=focus, age=self.now-fresh[2],
                    vehicle=vehicles[np.argmin(np.linalg.norm(vehicles-p, axis=1))])

    def task(self):
        if self.self_position is None:
            return None
        view = self.view(self.self_position)
        if view is None:
            return None
        return dict(position=tuple(view['vehicle']), target=tuple(view['targets'][view['focus']]))


class CoGraphNavigator(CooperativeNavigator):
    def make_chain_state(self):
        return MultiTargetTaskState(self.drone_id)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chain = MultiTargetTaskState(self.drone_id)

    def encode(self, position, observation):
        self.chain.self_position = np.asarray(position, float)[:2]
        return super().encode(position, observation)

    def cooperation_features(self, state, p, obs):
        """v7 role features with every received vehicle as a backbone endpoint."""
        data = obs['cooperation']
        peers, required, grounds = data['peers'], data['required'], data['grounds']
        ends = ([np.asarray(peers[k]) for k in required if k in peers]+
                [np.r_[np.asarray(grounds[k])[:2], 0.] for k in required if k in grounds])
        state[12:21] = 0.
        state[12] = float(any(k in grounds for k in required))
        state[13] = len(required)/MAX_FLEET
        state[14] = float(bool(grounds))
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
        obs['cograph'] = self.chain.view(p)

    def reset(self):
        super().reset()
        self.chain = MultiTargetTaskState(self.drone_id)


def target_tokens(position, view):
    tokens = np.zeros((MAX_TARGETS, TARGET_FEATURES), np.float32)
    valid = np.zeros(MAX_TARGETS, np.float32)
    if view is None:
        return tokens, valid
    if len(view['targets']) > MAX_TARGETS:
        raise ValueError(f'CoGraph v1 supports at most {MAX_TARGETS} targets')
    p = np.asarray(position, float)[:2]
    for i, target in enumerate(view['targets']):
        delta = target-p
        tokens[i] = [delta[0]/350., delta[1]/350., np.linalg.norm(delta)/350., float(view['connected'][i]),
                     float(i == view['focus']), min(view['age']/10., 3.),
                     np.linalg.norm(target-view['vehicle'])/350.]
        valid[i] = 1.
    return tokens, valid


class TargetAttention(nn.Module):
    """Masked attention over the received target set; empty set gives zero context."""
    def __init__(self, query_width, features=TARGET_FEATURES):
        super().__init__()
        self.query = nn.Linear(query_width, TARGET_WIDTH)
        self.key = nn.Linear(features, TARGET_WIDTH)
        self.value = nn.Linear(features, TARGET_WIDTH)

    def forward(self, query, tokens, valid):
        logits = (self.query(query).unsqueeze(-2)*self.key(tokens)).sum(-1)/np.sqrt(TARGET_WIDTH)
        weights = logits.masked_fill(~valid, -1e9).softmax(-1)*valid
        context = (weights.unsqueeze(-1)*self.value(tokens)).sum(-2)
        count = valid.float().sum(-1, keepdim=True)
        pending = (valid.float()*(1.-tokens[..., 3])).sum(-1, keepdim=True)
        stats = torch.cat((count/MAX_TARGETS, pending/count.clamp_min(1.), (pending > 0).float()), -1)
        return context, stats


class CoGraphEncoder(RelayEncoder):
    def __init__(self, mode='full'):
        super().__init__(mode)
        self.targets = TargetAttention(prefix_width(2)+64)

    def forward(self, x):
        relay = super().forward(x[..., :RELAY_WIDTH])
        tokens = x[..., RELAY_WIDTH:RELAY_WIDTH+MAX_TARGETS*TARGET_FEATURES].reshape(
            *x.shape[:-1], MAX_TARGETS, TARGET_FEATURES)
        valid = x[..., -MAX_TARGETS:] > 0
        context, stats = self.targets(relay[..., :prefix_width(2)+64], tokens, valid)
        return torch.cat((relay, context, stats), -1)


class CoGraphActorCritic(RelayActorCritic):
    policy_architecture = POLICY_ARCHITECTURE

    def __init__(self, global_features=GLOBAL_FEATURES, input_version=2, relay_attention='full'):
        if input_version != 2:
            raise ValueError('CoGraph requires actor_input_version=2')
        if relay_attention not in MODES:
            raise ValueError(f'relay_attention must be one of {MODES}')
        GraphAttentionActorCritic.__init__(self, global_features, input_version)
        self.feature_version = FEATURE_VERSION
        self.relay_attention = relay_attention
        self.gat_actor = nn.Sequential(CoGraphEncoder(relay_attention), nn.Linear(ENCODED_WIDTH, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2))
        self.critic = nn.Sequential(
            CooperativeCriticEncoder(global_features, CoGraphEncoder(relay_attention), RAW_WIDTH, ENCODED_WIDTH),
            nn.Linear(ENCODED_WIDTH+global_features, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.orthogonal_(self.gat_actor[-1].weight, gain=.01)
        nn.init.zeros_(self.gat_actor[-1].bias)

    @classmethod
    def from_state_dict(cls, state):
        mode = MODES[int(state['gat_actor.0.relay.mode'])]
        model = cls(state['critic.1.weight'].shape[1]-ENCODED_WIDTH, 2, mode)
        model.load_state_dict(state)
        return model

    def warm_start(self, v8_state):
        """Load a v8 relay-attention checkpoint: shared modules exactly; the new
        target context starts with zero actor/critic weights, so the initial
        actor equals v8 driven toward each drone's focus target. Critic global
        columns keep their fresh initialization (different global state)."""
        own = self.state_dict()
        copied, partial = [], []
        with torch.no_grad():
            for key, value in v8_state.items():
                if key not in own:
                    raise ValueError(f'Unexpected v8 parameter {key}')
                if own[key].shape == value.shape:
                    own[key].copy_(value)
                    copied.append(key)
                elif key in ('gat_actor.1.weight', 'critic.1.weight'):
                    own[key][:, RELAY_ENCODED:ENCODED_WIDTH] = 0.
                    own[key][:, :RELAY_ENCODED] = value[:, :RELAY_ENCODED]
                    partial.append(key)
                else:
                    raise ValueError(f'Shape mismatch for {key}: {tuple(value.shape)} vs {tuple(own[key].shape)}')
        fresh = sorted(set(own)-set(v8_state))
        return dict(copied=len(copied), partial=partial, fresh=fresh)

    def encode_graph(self, base, position, observation):
        relay = super().encode_graph(base, position, observation)
        tokens, valid = target_tokens(position, observation.get('cograph'))
        return np.r_[relay, tokens.ravel(), valid].astype(np.float32)
