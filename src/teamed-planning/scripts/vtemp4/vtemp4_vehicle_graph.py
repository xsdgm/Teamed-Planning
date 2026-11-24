"""Vehicle-centred decision graph for continuous, goal-conditioned waypoint selection."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from vtemp2_3.ground_policy import CANDIDATE_ROAD_RADIUS_M, ROAD_FEATURES, ground_sensor_tile, observed_road_inputs
from survey.survey_fleet_clearance import native_segment_clear
from vtemp4.vtemp4_policy import HeadingPolicy
from shared.simulator.ground_evidence import observed_fused_graph
from vtemp4.vtemp4_decision_graph_history import (
    COMMAND_FEATURES, DRIVE_FEATURES, DecisionGraphHistoryEpisode, FullCommandHistory,
)
from vtemp4.vtemp4_periodic_history import ArcAngleDistribution, legal_ring_arcs
from vtemp4.vtemp4_ring_contract import LSTM_PROGRESS_REWARD_WEIGHT, LSTM_REVIEW_INTERVAL_S
from vtemp4.vtemp4_road_traversal import RoadTraversalMemory, TRAVERSAL_FEATURES


GOAL_SCALE_M = 350.


class VehicleGraphGoalEpisode(DecisionGraphHistoryEpisode):
    def __init__(self, *args, **kwargs):
        self.road_travel = RoadTraversalMemory()
        super().__init__(*args, **kwargs)

    def _heading_observation(self):
        self.adapter.observe_receipts()
        car, now = self.position(), self.clock_s
        own = self.adapter.own
        yaw = float(own['orientation'])
        world = float(self.case['world_size'])
        local = self.adapter.ground_target
        anchors = [car, self.goal]
        if local is not None:
            anchors.append(local)
        if self.current_request is not None:
            anchors.append(self.current_request['target'])
        graph = observed_fused_graph(self.env.ground_controller.unified_graph, now)
        self.adapter.last_graph_edges = graph['graph_edges']
        nodes, adjacency, valid, xy = observed_road_inputs(graph, car, np.asarray(anchors), now)
        entries = np.zeros(len(valid), dtype=bool)
        ground = self.env.obs[self.env.ground]
        tile = ground_sensor_tile(ground)
        if tile is not None:
            for i in np.flatnonzero(np.linalg.norm(xy - car, axis=1) <= 15.):
                entries[i] = native_segment_clear(tile, car, ground['map_resolution'], car, xy[i])
        traversals = self.road_travel.edge_features(now, car, self.trajectory, xy, adjacency, valid)
        obs = self._decision_history_observation()
        obs.update(
            vehicle=np.asarray([*(car / world), *(np.asarray(own['velocity'])[:2] / 6.),
                                math.sin(yaw), math.cos(yaw), own['angular_velocity']], np.float32),
            goal=np.r_[self.goal / world, (self.goal - car) / GOAL_SCALE_M].astype(np.float32),
            local_target=(np.zeros(4, np.float32) if local is None else
                          np.r_[np.asarray(local) / world, (local - car) / GOAL_SCALE_M].astype(np.float32)),
            local_target_valid=local is not None, road_nodes=nodes, road_neighbors=adjacency.astype(np.int64),
            road_valid=valid.astype(bool), road_entry_mask=entries, road_traversals=traversals)
        # These two fields belong to the action sampler and episode audit. Neither
        # the actor encoder nor the critic reads them.
        if self._use_final_goal(np.linalg.norm(self.goal - car)):
            obs['angle_arcs'], obs['all_ring_endpoints_blocked'] = np.empty((0, 2)), False
        else:
            obs['angle_arcs'], obs['all_ring_endpoints_blocked'] = legal_ring_arcs(
                self.adapter, car, yaw, self.lookahead_m)
        return obs


class VehicleGraphGoalPolicy(nn.Module):
    """Read typed neighbours into one vehicle node; condition the actor on body-frame goal geometry."""

    act = HeadingPolicy.act
    vehicle_features = 7
    target_features = 4
    angle_distribution = ArcAngleDistribution

    def __init__(self, hidden=64):
        super().__init__()
        # Import after the worker has selected the verified V9 core.
        from mappo.graph_attention_mappo import AttentionLayer

        self.gat1 = AttentionLayer(ROAD_FEATURES + 2)
        self.road_query = nn.Linear(32, 32)
        self.road_key = nn.Linear(32, 32)
        self.edge_message = nn.Linear(2 + TRAVERSAL_FEATURES, 32, bias=False)
        self.road_output = nn.Linear(ROAD_FEATURES + 2 + 32 + 1, hidden)
        self.vehicle = nn.Linear(self.vehicle_features, hidden)
        self.local_target = nn.Linear(self.target_features, hidden)
        self.command = nn.Linear(COMMAND_FEATURES, hidden)
        self.drive = nn.Linear(DRIVE_FEATURES, hidden)
        self.predecessor = nn.Linear(hidden, hidden, bias=False)
        self.attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.goal_condition = nn.Linear(self.target_features, hidden)
        self.readout = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden + 3, 2)
        self.critic = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.actor.weight, gain=.01)
        nn.init.zeros_(self.actor.bias)

    def _roads(self, batch, goal_displacement):
        valid = batch('road_valid', torch.bool)
        nodes = batch('road_nodes').masked_fill(~valid[..., None], 0.)
        xy = nodes[..., :2]
        goal_from_node = goal_displacement[:, None, :] - xy * (CANDIDATE_ROAD_RADIUS_M / GOAL_SCALE_M)
        conditioned = torch.cat((nodes, goal_from_node), -1).masked_fill(~valid[..., None], 0.)
        adjacency = batch('road_neighbors', torch.long).masked_fill(~valid[..., None], -1)
        h = self.gat1(conditioned, adjacency, valid, xy)

        # Aggregate along measured road edges. Each directed neighbour slot carries
        # separate evidence of forward and reverse travel from the vehicle's own track.
        index = adjacency.clamp_min(0)
        b = torch.arange(len(h), device=h.device)[:, None, None]
        edge_valid = (adjacency >= 0) & valid[b, index]
        offsets = xy[b, index] - xy[:, :, None]
        edge = torch.cat((offsets, batch('road_traversals')), -1).masked_fill(~edge_valid[..., None], 0.)
        messages = h[b, index] + self.edge_message(edge)
        scores = (self.road_query(h)[:, :, None] * self.road_key(messages)).sum(-1) / math.sqrt(h.shape[-1])
        weights = scores.masked_fill(~edge_valid, -1e9).softmax(-1) * edge_valid
        h = (h + (weights[..., None] * messages).sum(-2)).tanh()
        # The vehicle can read every observed road; only certified near-field
        # segments have a direct physical connection to the vehicle.
        entry = (batch('road_entry_mask', torch.bool) & valid).to(h.dtype)
        return self.road_output(torch.cat((conditioned, h, entry[..., None]), -1)).tanh(), valid

    def _command_inputs(self, observations):
        device = self.actor.weight.device

        def pad(key, dtype, fill=0):
            values = [torch.as_tensor(obs[key], dtype=dtype, device=device) for obs in observations]
            return pad_sequence(values, batch_first=True, padding_value=fill)

        command = self.command(pad('command_nodes', torch.float32)).tanh()
        lengths = torch.tensor([len(obs['command_nodes']) for obs in observations], device=device)
        valid = torch.arange(command.size(1), device=device)[None] < lengths[:, None]
        if command.size(1) == 0:
            return command, valid
        owner = pad('drive_command', torch.long, -1)
        segments = self.drive(pad('drive_segments', torch.float32)).tanh()
        return command + self._drive_context(command, segments, owner), valid

    def _drive_context(self, command, segments, owner):
        # Every segment contributes only to the command that governed its travel.
        ids = owner.clamp_min(0).unsqueeze(-1)
        counts = torch.zeros_like(command[..., :1]).scatter_add(1, ids, (owner >= 0)[..., None].to(command.dtype))
        motion = torch.zeros_like(command).scatter_add(1, ids.expand_as(segments),
                                                       segments.masked_fill((owner < 0)[..., None], 0.))
        return motion / counts.clamp_min(1.)

    def _commands(self, observations):
        command, valid = self._command_inputs(observations)
        if command.size(1) == 0:
            return command, valid
        device = command.device
        previous = [torch.as_tensor(obs['command_predecessor'], dtype=torch.long, device=device)
                    for obs in observations]
        prev = pad_sequence(previous, batch_first=True, padding_value=-1)
        older = command[torch.arange(len(command), device=device)[:, None], prev.clamp_min(0)]
        return (command + self.predecessor(older).masked_fill((prev < 0)[..., None], 0.)).tanh(), valid

    def _history_neighbours(self, observations, car, need_weights):
        commands, valid = self._commands(observations)
        return commands, valid, None

    @staticmethod
    def _goal_geometry(own, goal):
        # The graph uses world-axis displacements; the actor also reads a bearing relative to the vehicle.
        delta = goal[:, 2:4]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        ux, uy = (delta / distance.clamp_min(1.e-8)[:, None]).unbind(-1)
        sy, cy = own[:, 4], own[:, 5]
        body_goal = torch.stack((cy * uy - sy * ux, cy * ux + sy * uy, distance), dim=-1)
        return delta, body_goal

    def _encode(self, observations, *, need_weights=False):
        device = self.actor.weight.device

        def batch(key, dtype=torch.float32):
            return torch.as_tensor(np.stack([obs[key] for obs in observations]), dtype=dtype, device=device)

        own, goal = batch('vehicle'), batch('goal')
        goal_displacement, body_goal = self._goal_geometry(own, goal)
        car = self.vehicle(own).tanh()[:, None]
        roads, road_mask = self._roads(batch, goal_displacement)
        commands, command_mask, temporal_weights = self._history_neighbours(observations, car, need_weights)
        local_mask = batch('local_target_valid', torch.bool)[:, None]
        local = self.local_target(batch('local_target').masked_fill(~local_mask, 0.)).tanh()[:, None]

        groups = dict(vehicle=car, roads=roads, commands=commands, local_target=local)
        valid = torch.cat((torch.ones_like(local_mask), road_mask, command_mask, local_mask), dim=1)
        neighbours = torch.cat(list(groups.values()), dim=1)
        message, weights = self.attention(car, neighbours, neighbours, key_padding_mask=~valid,
                                          need_weights=need_weights, average_attn_weights=False)
        state = self.readout(car[:, 0] + message[:, 0] + self.goal_condition(goal).tanh())

        if need_weights:
            widths = [group.shape[1] for group in groups.values()]
            weights = dict(zip(groups, weights.squeeze(2).split(widths, dim=-1)))
            if temporal_weights is not None:
                weights['command_summary'] = weights.pop('commands')
                weights['commands'] = temporal_weights
        return state, body_goal, weights

    def _action_distribution(self, params, body_goal, arcs):
        return self.angle_distribution(params, arcs)

    def forward(self, observations):
        state, body_goal, _ = self._encode(observations)
        params = self.actor(torch.cat((state, body_goal), dim=-1))
        distribution = self._action_distribution(params, body_goal, [obs['angle_arcs'] for obs in observations])
        return distribution, self.critic(state).squeeze(-1)

    def critic_values(self, observations):
        state, _, _ = self._encode(observations)
        return self.critic(state).squeeze(-1)

    @torch.no_grad()
    def value(self, observation):
        return self.critic_values([observation]).item()

    @torch.no_grad()
    def attention_weights(self, observations):
        return self._encode(observations, need_weights=True)[2]


class VehicleGraphHistoryLSTMEpisode(VehicleGraphGoalEpisode):
    command_history_type = FullCommandHistory

    def __init__(self, *args, review_interval_s=LSTM_REVIEW_INTERVAL_S,
                 progress_reward_weight=LSTM_PROGRESS_REWARD_WEIGHT, waypoint_progress_bonus_ratio=0., **kwargs):
        if not math.isfinite(waypoint_progress_bonus_ratio) or waypoint_progress_bonus_ratio < 0:
            raise ValueError('Waypoint progress bonus ratio must be finite and nonnegative')
        self.waypoint_progress_bonus_ratio = waypoint_progress_bonus_ratio
        super().__init__(*args, review_interval_s=review_interval_s,
                         progress_reward_weight=progress_reward_weight, **kwargs)


class VehicleGraphHistoryLSTMPolicy(VehicleGraphGoalPolicy):
    """Encode travel within each command, then attend to the ordered history of commands."""

    def __init__(self, hidden=64):
        super().__init__(hidden)
        del self.predecessor  # The recurrent state carries the preceding commands in this variant.
        self.drive_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.command_lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.temporal_attention = nn.MultiheadAttention(hidden, 4, batch_first=True)

    def _drive_context(self, command, segments, owner):
        # Observation rows are grouped by command, then ordered by their measured
        # start time. Batch all independent GRU sequences with fresh initial states.
        batch, n_commands, width = command.shape
        valid = owner >= 0
        group_ids = (owner + torch.arange(batch, device=owner.device)[:, None] * n_commands)[valid]
        if group_ids.numel() == 0:
            return torch.zeros_like(command)
        groups, row, lengths = torch.unique_consecutive(group_ids, return_inverse=True, return_counts=True)
        starts = lengths.cumsum(0) - lengths
        tick = torch.arange(len(row), device=row.device) - starts[row]
        padded = segments.new_zeros((len(groups), int(lengths.max()), width))
        padded[row, tick] = segments[valid]
        encoded, _ = self.drive_gru(padded)
        last = encoded[torch.arange(len(groups), device=row.device), lengths - 1]
        return torch.zeros_like(command).flatten(0, 1).index_copy(0, groups, last).reshape_as(command)

    def _commands(self, observations):
        commands, valid = self._command_inputs(observations)
        if commands.size(1) == 0:
            return commands, valid
        # Right padding only follows real commands. The one-way recurrence cannot
        # read it at any valid step. Start fresh for every observation and PPO replay.
        encoded, _ = self.command_lstm(commands)
        return encoded.masked_fill(~valid[..., None], 0.), valid

    def _history_neighbours(self, observations, car, need_weights):
        steps, valid = self._commands(observations)
        present = valid.any(dim=1, keepdim=True)
        if steps.size(1) == 0:
            weights = car.new_zeros((len(car), self.temporal_attention.num_heads, 0)) if need_weights else None
            return torch.zeros_like(car), present, weights
        # MultiheadAttention requires one unmasked key per row. In a mixed batch
        # empty rows use a zero placeholder, then both their output and weights are masked.
        available = valid.clone()
        available[:, :1] |= ~present
        summary, weights = self.temporal_attention(car, steps, steps, key_padding_mask=~available,
                                                   need_weights=need_weights, average_attn_weights=False)
        if need_weights:
            weights = weights.squeeze(2).masked_fill(~valid[:, None], 0.)
        return summary.masked_fill(~present[..., None], 0.), present, weights
