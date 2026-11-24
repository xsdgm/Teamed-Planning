"""PPO over persistent world waypoints using received spatial evidence and optional UGV history."""
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical

from vtemp4.vtemp4_observation import DIRECTION_COUNT
from vtemp4.vtemp4_policy import HeadingPolicy, update_policy


WAYPOINT_FORMAT = 'vtemp4_fixed_map_waypoint_v1'
WAYPOINT_COUNT = 2 * DIRECTION_COUNT
HISTORY_LENGTH = 4
HISTORY_FEATURES = 5


class WaypointDistribution:
    """Keep the current world point, or select one of the legal replacements."""

    def __init__(self, gate_logits, point_logits, point_mask, can_hold):
        self.point_mask = point_mask
        self.can_hold = can_hold
        self.can_replace = point_mask.any(dim=-1)
        if not bool((can_hold | self.can_replace).all()):
            raise ValueError('A waypoint review must allow hold or at least one possible destination')
        self.gate = Bernoulli(logits=gate_logits)
        self.flexible = self.can_replace & can_hold
        # Categorical must have a valid support on keep-only rows. The dummy
        # index contributes neither probability nor entropy to a keep action.
        support = point_mask.clone()
        support[:, 0] |= ~self.can_replace
        self.point = Categorical(logits=point_logits.masked_fill(~support, -torch.inf))

    def action(self, deterministic, generator=None):
        if deterministic:
            gate = self.gate.logits > 0.
            index = self.point.probs.argmax(dim=-1)
        else:
            device = self.point.probs.device
            rng_device = generator.device if generator is not None else device
            uniform = torch.rand(self.gate.probs.shape, device=rng_device, generator=generator)
            gate = uniform.to(device) < self.gate.probs
            index = torch.multinomial(self.point.probs.to(rng_device), 1, generator=generator).squeeze(-1).to(device)
        replace = self.can_replace & (~self.can_hold | gate)
        return replace, torch.where(replace, index, -1)

    def log_prob(self, replace, index):
        if index.is_floating_point() and not torch.equal(index, index.round()):
            raise ValueError('Waypoint indices must be integers')
        index = index.long()
        safe_index = index.clamp(0, self.point_mask.shape[-1] - 1)
        valid_point = self.point_mask.gather(1, safe_index[:, None]).squeeze(1)
        allowed = torch.where(replace, (index == safe_index) & valid_point, self.can_hold & (index == -1))
        if not bool(allowed.all()):
            raise ValueError('Stored waypoint action is invalid for its observation')
        gate_logp = self.gate.log_prob(replace.to(self.gate.logits.dtype))
        point_logp = self.point.log_prob(safe_index)
        return torch.where(self.flexible, gate_logp, 0.) + torch.where(replace, point_logp, 0.)

    def entropy(self):
        p_replace = torch.where(self.flexible, self.gate.probs, self.can_replace.to(self.gate.probs.dtype))
        return torch.where(self.flexible, self.gate.entropy(), 0.) + p_replace * self.point.entropy()


class MotionHistoryAttention(nn.Module):
    """Read four ordered motion intervals while excluding missing earlier intervals."""

    def __init__(self, hidden, width=32):
        super().__init__()
        self.token = nn.Linear(HISTORY_FEATURES, width)
        self.age = nn.Parameter(torch.empty(HISTORY_LENGTH, width))
        nn.init.normal_(self.age, std=.1)
        self.query = nn.Linear(hidden, width)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)

    def forward(self, state, history, mask):
        if history.shape != (len(state), HISTORY_LENGTH, HISTORY_FEATURES) or mask.shape != history.shape[:2]:
            raise ValueError('Expected four ordered vehicle movement samples and their validity mask')
        history = torch.where(mask[..., None], history, 0.)
        tokens = (self.token(history) + self.age).tanh()
        scores = (self.query(state)[:, None] * self.key(tokens)).sum(-1) / math.sqrt(tokens.shape[-1])
        # A completely padded window has an exact zero readout, including at reset.
        weights = scores.masked_fill(~mask, -1e9).softmax(dim=-1) * mask
        return (weights[..., None] * self.value(tokens)).sum(dim=1)


class WaypointPolicy(HeadingPolicy):
    """The two fixed-waypoint arms share all parameters and differ only in the history input."""

    def __init__(self, hidden=64, *, use_history):
        super().__init__(hidden)
        self.use_history = use_history
        self.history_attention = MotionHistoryAttention(hidden)
        self.waypoint_state = nn.Sequential(nn.Linear(hidden + 2 + 32, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, 1)
        self.point_key = nn.Linear(hidden + 2, hidden)
        self.point_query = nn.Linear(hidden, hidden, bias=False)
        self.point_score = nn.Linear(hidden, 1, bias=False)
        nn.init.orthogonal_(self.actor.weight, gain=.01)
        nn.init.constant_(self.actor.bias, -.7)
        nn.init.orthogonal_(self.point_score.weight, gain=.01)

    def _direction_features(self, directions):
        return directions.repeat_interleave(2, dim=1)

    def forward(self, observations):
        device = self.actor.weight.device

        def batch(key, dtype):
            return torch.as_tensor(np.stack([obs[key] for obs in observations]), dtype=dtype, device=device)

        spatial, directions = self._spatial_state(observations, include_recent_motion=False)
        if self.use_history:
            history = self.history_attention(spatial, batch('history', torch.float32), batch('history_mask', torch.bool))
        else:
            history = spatial.new_zeros((len(observations), self.history_attention.value.out_features))
        state = self.waypoint_state(torch.cat((spatial, batch('active_waypoint_delta', torch.float32), history), dim=-1))

        points = batch('waypoint_xy', torch.float32)
        mask = batch('waypoint_mask', torch.bool)
        if points.shape != (len(observations), WAYPOINT_COUNT, 2) or mask.shape != points.shape[:2]:
            raise ValueError('Expected 48 waypoint offsets and their validity mask')
        points = torch.where(mask[..., None], points, 0.)
        keys = self.point_key(torch.cat((self._direction_features(directions), points), dim=-1))
        scores = self.point_score((keys + self.point_query(state)[:, None]).tanh()).squeeze(-1)
        actions = WaypointDistribution(self.actor(state).squeeze(-1), scores, mask, batch('can_hold', torch.bool))
        return actions, self.critic(state).squeeze(-1)

    @torch.no_grad()
    def act(self, observation, deterministic=False, generator=None):
        policy, value = self([observation])
        replace, index = policy.action(deterministic, generator)
        return replace.item(), index.item(), policy.log_prob(replace, index).item(), value.item()


class RingWaypointPolicy(WaypointPolicy):
    """Score 48 yaw-relative directions using the 24 measured terrain sectors."""

    def _direction_features(self, directions):
        # Even indices align with a measured ray. Odd indices bisect its two neighbours.
        midpoints = (directions + directions.roll(-1, dims=1)) * .5
        return torch.stack((directions, midpoints), dim=2).flatten(1, 2)


def _waypoint_log_prob(policy, rows, values):
    replace = torch.tensor([row['replace'] for row in rows], dtype=torch.bool, device=values.device)
    indices = torch.tensor([row['point_index'] for row in rows], dtype=torch.long, device=values.device)
    return policy.log_prob(replace, indices)


def update_waypoint_policy(model, optimizer, episodes, **options):
    return update_policy(model, optimizer, episodes, action_log_prob=_waypoint_log_prob, **options)
