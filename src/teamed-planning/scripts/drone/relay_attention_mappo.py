"""v8: v7 cooperative policy plus age/hop-aware attention over relayed state evidence.

Tokens are this device's own relay cache (drone/relay_cache.py): teammates and
the vehicle, directly sensed or not, with the ORIGINAL measurement age and the
hop count of the path that delivered them. Relative relevance is a masked
softmax whose logits subtract learned, type-specific age and hop penalties.
An absolute reliability gate rho = exp(-age/tau - lambda_h (h-1) - lambda_s sigma^2)
then scales the weights WITHOUT renormalising, so "everything is stale" shrinks
the context instead of remaining a convex combination. Count/reliability
summaries are exposed explicitly; an empty cache gives a zero context.

One-hop geometry, the RNG backbone and the CBF are exactly v7's. Device IDs
only match tokens to currently sensed peers; they are never embedded.
"""
import numpy as np
import torch
from torch import nn
from drone.cooperative_mappo import (CooperativeActorCritic, CooperativeCriticEncoder, CooperativeEncoder,
    RAW_WIDTH as COOPERATIVE_WIDTH, ENCODED_WIDTH as COOPERATIVE_ENCODED)
from drone.graph_attention_mappo import GraphAttentionActorCritic, MAX_FLEET, prefix_width

RELAY_TOKENS = MAX_FLEET
# Token layout: 0-2 relative source position, 3-5 source velocity and known flag,
# 6 age (s), 7 hop count, 8-9 type (drone/vehicle), 10 sigma/95 m, 11 source time
# unknown, 12 origin currently sensed one hop, 13-15 relative task goal and flag.
RELAY_FEATURES = 16
AGE, HOP, VEHICLE, SIGMA = 6, 7, 9, 10
CONTENT = [i for i in range(RELAY_FEATURES) if i not in (AGE, HOP)]
RAW_WIDTH = COOPERATIVE_WIDTH+RELAY_TOKENS*(RELAY_FEATURES+1)
TIME_FEATURES = 8
HEADS, HEAD_WIDTH = 2, 16
RELAY_STATS = 4
ENCODED_WIDTH = COOPERATIVE_ENCODED+HEADS*HEAD_WIDTH+RELAY_STATS
# Ablations of the v8 validation matrix: full = v8-D, age = v8-A, hop = v8-B,
# softmax = v8-C (age+hop in the logits, no absolute gate).
MODES = ('full', 'age', 'hop', 'softmax')
# Fixed gate parameters (design values, not validated on aircraft).
TAU_S = (10., 30.)  # drone_state, vehicle_state
HOP_DECAY = .1
SIGMA_DECAY = 1.    # per (sigma/95 m)^2


class TimeEncoding(nn.Module):
    """TGAT-style functional time encoding cos(w*dt + b), learnable w and b."""
    def __init__(self, width=TIME_FEATURES):
        super().__init__()
        self.frequency = nn.Parameter(torch.from_numpy(1./np.geomspace(1., 100., width)).float())
        self.phase = nn.Parameter(torch.zeros(width))

    def forward(self, age):
        return torch.cos(age.unsqueeze(-1)*self.frequency+self.phase)


class RelayAttention(nn.Module):
    def __init__(self, query_width, mode='full'):
        super().__init__()
        self.register_buffer('mode', torch.tensor(MODES.index(mode)))
        self.register_buffer('tau', torch.tensor(TAU_S))
        self.time = TimeEncoding()
        width = len(CONTENT)+TIME_FEATURES+1
        self.query = nn.Linear(query_width, HEADS*HEAD_WIDTH)
        # Separate key/value projections per payload type.
        self.key = nn.ModuleList(nn.Linear(width, HEADS*HEAD_WIDTH) for _ in range(2))
        self.value = nn.ModuleList(nn.Linear(width, HEADS*HEAD_WIDTH) for _ in range(2))
        start = float(np.log(np.expm1(.5)))  # softplus(start) = 0.5
        self.age_penalty = nn.Parameter(torch.full((2, HEADS), start))
        self.hop_penalty = nn.Parameter(torch.full((2, HEADS), start))

    def forward(self, query, tokens, valid):
        mode = MODES[int(self.mode)]
        use_age, use_hop, gate = mode != 'hop', mode != 'age', mode != 'softmax'
        age, hop = tokens[..., AGE], tokens[..., HOP]
        vehicle = tokens[..., VEHICLE] > .5
        kind = vehicle.long()
        log_hop = torch.log1p(hop.clamp_min(0.))
        time = self.time(age)
        z = torch.cat((tokens[..., CONTENT], time if use_age else torch.zeros_like(time),
                       (log_hop if use_hop else torch.zeros_like(log_hop)).unsqueeze(-1)), -1)
        shape = (*tokens.shape[:-1], HEADS, HEAD_WIDTH)
        key = torch.where(vehicle[..., None], self.key[1](z), self.key[0](z)).reshape(shape)
        value = torch.where(vehicle[..., None], self.value[1](z), self.value[0](z)).reshape(shape)
        q = self.query(query).reshape(*query.shape[:-1], 1, HEADS, HEAD_WIDTH)
        logits = (q*key).sum(-1)/np.sqrt(HEAD_WIDTH)
        tau = torch.where(vehicle, self.tau[1], self.tau[0])
        if use_age:
            logits = logits-nn.functional.softplus(self.age_penalty)[kind]*(age/tau)[..., None]
        if use_hop:
            logits = logits-nn.functional.softplus(self.hop_penalty)[kind]*log_hop[..., None]
        # Hard mask first (absent/expired entries never reach the cache encoding).
        weights = logits.masked_fill(~valid[..., None], -1e9).softmax(-2)*valid[..., None]
        rho = valid.float()
        if gate:
            exponent = SIGMA_DECAY*tokens[..., SIGMA].square()
            if use_age:
                exponent = exponent+age/tau
            if use_hop:
                exponent = exponent+HOP_DECAY*(hop-1.).clamp_min(0.)
            rho = rho*torch.exp(-exponent)
        context = ((weights*rho[..., None]).unsqueeze(-1)*value).sum(-3).reshape(*query.shape[:-1], -1)
        count = valid.float().sum(-1, keepdim=True)
        stats = torch.cat((torch.log1p(count)/np.log(RELAY_TOKENS+1.),
                           rho.sum(-1, keepdim=True)/count.clamp_min(1.),
                           rho.amax(-1, keepdim=True), (count == 0).float()), -1)
        return context, stats


class RelayEncoder(CooperativeEncoder):
    def __init__(self, mode='full'):
        super().__init__()
        self.relay = RelayAttention(prefix_width(2)+64, mode)

    def forward(self, x):
        cooperative = super().forward(x[..., :COOPERATIVE_WIDTH])
        tokens = x[..., COOPERATIVE_WIDTH:COOPERATIVE_WIDTH+RELAY_TOKENS*RELAY_FEATURES].reshape(
            *x.shape[:-1], RELAY_TOKENS, RELAY_FEATURES)
        valid = x[..., -RELAY_TOKENS:] > 0
        # Same query as the v7 peer branch: own state/sensor/map encoding.
        context, stats = self.relay(cooperative[..., :prefix_width(2)+64], tokens, valid)
        return torch.cat((cooperative, context, stats), -1)


class RelayActorCritic(CooperativeActorCritic):
    policy_architecture = 'gat_relay_attention_v8'

    def __init__(self, global_features=10, input_version=2, relay_attention='full'):
        if input_version != 2:
            raise ValueError('v8 requires actor_input_version=2')
        if relay_attention not in MODES:
            raise ValueError(f'relay_attention must be one of {MODES}')
        GraphAttentionActorCritic.__init__(self, global_features, input_version)
        self.feature_version = 8
        self.relay_attention = relay_attention
        self.gat_actor = nn.Sequential(RelayEncoder(relay_attention), nn.Linear(ENCODED_WIDTH, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2))
        self.critic = nn.Sequential(
            CooperativeCriticEncoder(global_features, RelayEncoder(relay_attention), RAW_WIDTH, ENCODED_WIDTH),
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

    def encode_graph(self, base, position, observation):
        cooperative = super().encode_graph(base, position, observation)
        tokens = np.zeros((RELAY_TOKENS, RELAY_FEATURES), np.float32)
        valid = np.zeros(RELAY_TOKENS, np.float32)
        p = np.zeros(3)
        p[:len(position)] = position
        data = observation.get('cooperation', {})
        sensed = set(data.get('peers', {}))
        # KeyError on purpose: v8 never silently acts without its relay cache.
        # The cache lists vehicle/task records first, then fresher and shorter
        # paths, so truncation never drops the only task record for distance.
        for i, row in enumerate(observation['relay'][:RELAY_TOKENS]):
            vehicle = row['payload_type'] == 'vehicle_state'
            q = np.zeros(3)
            q[:len(row['position'])] = row['position']
            delta = q-p
            velocity, goal = row['velocity'], row['goal']
            tokens[i] = [delta[0]/350., delta[1]/350., delta[2]/15.,
                         *((velocity[0]/6., velocity[1]/6.) if velocity is not None else (0., 0.)),
                         float(velocity is not None), row['age'], row['hop'],
                         float(not vehicle), float(vehicle), row['sigma']/95., float(row['time_unknown']),
                         float(data.get('ground') is not None) if vehicle else float(row['origin_id'] in sensed),
                         *(((goal[0]-p[0])/350., (goal[1]-p[1])/350.) if goal is not None else (0., 0.)),
                         float(goal is not None)]
            valid[i] = 1.
        return np.r_[cooperative, tokens.ravel(), valid].astype(np.float32)
