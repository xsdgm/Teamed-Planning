"""One receiver-side attention readout for periodic, continuous ring designations."""
import math

import numpy as np
import torch
from torch import nn

from vtemp2_3 import frozen_navigation
from vtemp2_3.ground_policy import (EGO_FEATURES, PEER_FEATURES, RELAY_FEATURES, ROAD_FEATURES,
                                  SENSOR_SIDE, TASK_BLOCKS, TASK_FEATURES, _blocks)
from vtemp4.vtemp4_observation import CONTEXT_SIZE, DIRECTION_COUNT, FEATURE_COUNT
from vtemp4.vtemp4_periodic_history import ArcAngleDistribution, PeriodicHistoryFleetEpisode
from vtemp4.vtemp4_policy import HeadingPolicy
from vtemp4.vtemp4_waypoint_runtime import HISTORY_AGES_S


EGO_WITHOUT_PIXELS = EGO_FEATURES - 2 * SENSOR_SIDE**2


class UnifiedAttentionEpisode(PeriodicHistoryFleetEpisode):
    """Expose known map-frame positions alongside receiver-relative measurements."""

    def _heading_observation(self):
        obs = super()._heading_observation()
        world = float(self.case['world_size'])

        def normalized(point):
            if point is None:
                return np.zeros(2, np.float32)
            return np.asarray(point, dtype=np.float32)[:2] / world

        guidance = self.current_request['target'] if self.current_request is not None else None
        obs.update(vehicle_world_xy=normalized(self.adapter.own['position']),
                   goal_world_xy=normalized(self.goal),
                   guidance_world_xy=normalized(guidance),
                   local_target_world_xy=normalized(self.adapter.ground_target))
        return obs


class GoalAwareUnifiedAttentionEpisode(UnifiedAttentionEpisode):
    """Broadcast the mission goal at reviews within the 200 m lookahead."""
    designate_near_goal = True

    def fleet_summary(self):
        return dict(super().fleet_summary(), final_goal_xy=self.goal.tolist())


class GoalAwareTerrainSummaryEpisode(GoalAwareUnifiedAttentionEpisode):
    def _heading_observation(self):
        obs = super()._heading_observation()
        # The local planner and corridor features use the full receiver map. Only
        # the packed upper-policy observation (also saved for PPO) drops the pixels.
        packed = obs['ground_inputs']
        obs['ground_inputs'] = np.concatenate((packed[:EGO_WITHOUT_PIXELS], packed[EGO_FEATURES:]))
        return obs


class UnifiedAttentionPolicy(nn.Module):
    """Fuse receiver evidence as typed tokens. Both heads read only the attended value."""

    act = HeadingPolicy.act
    value = HeadingPolicy.value
    input_blocks = TASK_BLOCKS

    def __init__(self, hidden=64):
        super().__init__()
        if frozen_navigation._RUNTIME is None:
            raise RuntimeError('Call frozen_navigation.runtime(repo) before constructing the policy')
        from mappo.graph_attention_mappo import GraphEncoder

        graph = GraphEncoder()
        self.gat1, self.gat2 = graph.layer1, graph.layer2
        self.patch_side = 4
        grid_side = SENSOR_SIDE // self.patch_side
        centers = (torch.arange(grid_side) + .5) / grid_side - .5
        y, x = torch.meshgrid(centers, centers, indexing='ij')
        self.register_buffer('patch_centers', torch.stack((x, y), -1).reshape(-1, 2), persistent=False)
        ages = torch.tensor(HISTORY_AGES_S, dtype=torch.float32)
        self.register_buffer('history_ages', ages / ages.max(), persistent=False)
        widths = dict(vehicle=7, goal=4, guidance=5, local_target=4,
                      context=CONTEXT_SIZE - 4 + 1 + len(TASK_FEATURES) + 1 + 2,
                      sensor=2 * self.patch_side**2 + 2, roads=ROAD_FEATURES + 32,
                      directions=FEATURE_COUNT + 2, peers=PEER_FEATURES, relay=RELAY_FEATURES,
                      history=6, arcs=5)
        self.projections = nn.ModuleDict({name: nn.Linear(width, hidden) for name, width in widths.items()})
        self.query = nn.Linear(2 * hidden, hidden)
        self.attention = nn.MultiheadAttention(hidden, 4, dropout=0., batch_first=True)
        self.readout = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, 2)
        self.critic = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.actor.weight, gain=.01)
        nn.init.zeros_(self.actor.bias)

    def _sensor_patches(self, ego):
        b = ego.shape[0]
        side = self.patch_side
        grid = SENSOR_SIDE // side
        # The planes are in YX order. Patch positions are in map-frame XY,
        # centred on the receiver, in the same 95 m units as road positions.
        pixels = ego[:, 11:].reshape(b, 2, grid, side, grid, side)
        patches = pixels.permute(0, 2, 4, 1, 3, 5).reshape(b, grid * grid, -1)
        extent = ego[:, 9:11]
        centers = self.patch_centers * extent[:, None]
        valid = (extent > 0.).all(-1, keepdim=True).expand(-1, grid * grid)
        return torch.cat((patches, centers), -1), valid

    def _road_tokens(self, data):
        valid = data['road_valid'].bool()
        nodes = data['roads'].masked_fill(~valid[..., None], 0.)
        indices = data['adjacency'].masked_fill(~valid[..., None], -1).long()
        xy = nodes[..., :2]
        states = self.gat1(nodes, indices, valid, xy)
        states = self.gat2(states, indices, valid, xy)
        return torch.cat((nodes, states), -1), valid

    def _arc_tokens(self, observations):
        arcs = [np.asarray(obs['angle_arcs']) for obs in observations]
        pad = np.zeros((len(arcs), max(map(len, arcs)), 2), dtype=np.float32)
        for row, intervals in zip(pad, arcs):
            row[:len(intervals)] = intervals
        lo, hi = torch.as_tensor(pad, device=self.actor.weight.device).unbind(-1)
        return torch.stack((lo.cos(), lo.sin(), hi.cos(), hi.sin(), (hi-lo) / math.tau), -1), hi > lo

    def _attend(self, observations, *, need_weights):
        device = self.actor.weight.device

        def batch(key, dtype=torch.float32):
            return torch.as_tensor(np.stack([obs[key] for obs in observations]), dtype=dtype, device=device)

        data = _blocks(batch('ground_inputs'), self.input_blocks)
        ego, ctx, task = data['ego'], batch('context'), data['task']
        blocked = batch('all_ring_endpoints_blocked').reshape(-1, 1)
        history = batch('history')
        ages = self.history_ages.expand(len(observations), -1)[..., None]
        # Preserve the presence flags even when no guidance, local waypoint
        # or survey payload is available for attention.
        survey = task[:, 2:].masked_fill(~task[:, 1:2].bool(), 0.)
        receipt_age = ctx[:, 9:10].masked_fill(~ctx[:, 8:9].bool(), 0.)
        context = torch.cat((ctx[:, :1], ctx[:, 4:9], receipt_age, task[:, :2], survey,
                             blocked, ego[:, 9:11]), -1)
        groups = {}
        masks = {}

        def add(name, values, valid=None):
            if values.ndim == 2:
                values = values[:, None, :]
            if valid is None:
                valid = torch.ones(values.shape[:2], dtype=torch.bool, device=device)
            groups[name] = self.projections[name](values.masked_fill(~valid[..., None], 0.)).tanh()
            masks[name] = valid

        add('vehicle', torch.cat((batch('vehicle_world_xy'), ego[:, :5]), -1))
        add('goal', torch.cat((batch('goal_world_xy'), ego[:, 5:7]), -1))
        add('guidance', torch.cat((batch('guidance_world_xy'), ctx[:, 1:4]), -1), ctx[:, :1].bool())
        add('local_target', torch.cat((batch('local_target_world_xy'), ego[:, 7:9]), -1), task[:, :1].bool())
        add('context', context)
        if 'sensor' in self.projections:
            add('sensor', *self._sensor_patches(ego))
        add('roads', *self._road_tokens(data))
        rows = batch('features')[:, :DIRECTION_COUNT]
        offsets = data['candidate_xy'][:, :DIRECTION_COUNT]
        add('directions', torch.cat((rows, offsets), -1), batch('present', torch.bool)[:, :DIRECTION_COUNT])
        add('peers', data['peers'], data['peers'][..., -1].bool())
        add('relay', data['relay'], data['relay_valid'].bool())
        add('history', torch.cat((history, ages), -1), batch('history_mask', torch.bool))
        add('arcs', *self._arc_tokens(observations))

        query = self.query(torch.cat((groups['vehicle'], groups['goal']), dim=-1))
        tokens = torch.cat(list(groups.values()), dim=1)
        valid = torch.cat(list(masks.values()), dim=1)
        pooled, weights = self.attention(query, tokens, tokens, key_padding_mask=~valid,
                                         need_weights=need_weights, average_attn_weights=False)
        state = self.readout(pooled[:, 0])
        if need_weights:
            counts = [tokens.shape[1] for tokens in groups.values()]
            return state, dict(zip(groups, weights.squeeze(2).split(counts, dim=-1)))
        return state, None

    def forward(self, observations):
        state, _ = self._attend(observations, need_weights=False)
        arcs = [obs['angle_arcs'] for obs in observations]
        return ArcAngleDistribution(self.actor(state), arcs), self.critic(state).squeeze(-1)

    @torch.no_grad()
    def attention_weights(self, observations):
        """Per-head weights for each token group, from the common attention softmax."""
        return self._attend(observations, need_weights=True)[1]


class GoalAwareUnifiedAttentionPolicy(UnifiedAttentionPolicy):
    def critic_values(self, observations):
        state, _ = self._attend(observations, need_weights=False)
        return self.critic(state).squeeze(-1)

    @torch.no_grad()
    def value(self, observation):
        return self.critic_values([observation]).item()


class GoalAwareTerrainSummaryPolicy(GoalAwareUnifiedAttentionPolicy):
    """Read terrain through the 24 directional summaries, without local image patches."""

    input_blocks = (('ego', (EGO_WITHOUT_PIXELS,)), *TASK_BLOCKS[1:])

    def __init__(self, hidden=64):
        # Initialize shared weights in the same order as the control for paired runs.
        super().__init__(hidden)
        del self.projections['sensor']
        del self.patch_centers
