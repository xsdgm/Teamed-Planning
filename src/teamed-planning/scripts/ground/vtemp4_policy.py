"""Slow PPO policy for holding a map heading or turning through a continuous angle."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli

from ground.ground_policy import TASK_BLOCKS, _GroundEncoder, _blocks
from world_model.slow_policy import smdp_advantages
from ground.vtemp4_observation import CONTEXT_SIZE, DIRECTION_COUNT, FEATURE_COUNT


class HeadingDistribution:
    """A point mass for hold and a wrapped Cauchy density conditional on turn."""

    def __init__(self, output):
        turn_logit, location, concentration = output.unbind(-1)
        self.turn = Bernoulli(logits=turn_logit)
        self.location = math.pi * torch.tanh(location)
        self.rho = .99 * torch.sigmoid(concentration)

    def action(self, deterministic, generator):
        if deterministic:
            turn = self.turn.logits > 0.
            angle = self.location
        else:
            device = generator.device if generator is not None else self.location.device
            uniform = torch.rand((2, *self.location.shape), generator=generator, device=device)
            gate, sample = uniform.to(self.location).unbind(0)
            turn = gate < self.turn.probs
            # Inverse CDF of the centred wrapped Cauchy, followed by a rotation.
            u = math.pi * (sample - .5)
            offset = 2 * torch.atan2((1 - self.rho) * u.sin(), (1 + self.rho) * u.cos())
            angle = torch.remainder(self.location + offset + math.pi, 2 * math.pi) - math.pi
        return turn, torch.where(turn, angle, 0.)

    def log_prob(self, turn, angle):
        denominator = (1 - self.rho).square() + 4 * self.rho * ((angle - self.location) / 2).sin().square()
        angular_logp = torch.log1p(-self.rho.square()) - math.log(2 * math.pi) - denominator.log()
        return self.turn.log_prob(turn.to(angle.dtype)) + torch.where(turn, angular_logp, 0.)

    def entropy(self):
        angular_entropy = math.log(2 * math.pi) + torch.log1p(-self.rho.square())
        return self.turn.entropy() + self.turn.probs * angular_entropy


class EgoMapAttention(nn.Module):
    """One attention hop from observed road nodes and terrain sections to the UGV."""

    WIDTH = 32
    DISTANCE_SCALE_M = 50.

    def __init__(self, ego_features):
        super().__init__()
        self.query = nn.Linear(ego_features + CONTEXT_SIZE, self.WIDTH)
        self.road = nn.Linear(32 + 2, self.WIDTH)
        self.terrain = nn.Linear(3 + 2, self.WIDTH)
        self.key = nn.Linear(self.WIDTH, self.WIDTH, bias=False)
        # Each terrain node is located at the centre of its measured corridor section.
        self.register_buffer('section_midpoints', (torch.arange(8) + .5) / 8., persistent=False)

    def forward(self, ego_state, context, road_embeddings, data, rows):
        # VTemp2 packs XY offsets centred at the car in units of 95 metres.
        road_xy = data['roads'][..., :2] * 95.
        endpoint_xy = data['candidate_xy'] * 95.
        section_xy = endpoint_xy[:, :, None, :] * self.section_midpoints[None, None, :, None]
        sections = rows[..., 4:].reshape(*rows.shape[:2], 8, 3)
        road = self.road(torch.cat((road_embeddings, road_xy / self.DISTANCE_SCALE_M), dim=-1)).tanh()
        terrain = self.terrain(torch.cat((sections, section_xy / self.DISTANCE_SCALE_M), dim=-1)).tanh()
        nodes = torch.cat((road, terrain.flatten(1, 2)), dim=1)
        offsets = torch.cat((road_xy, section_xy.flatten(1, 2)), dim=1)
        # A corridor shortened to zero at a world boundary has no terrain extent.
        has_extent = endpoint_xy.ne(0.).any(-1)
        valid = torch.cat((data['road_valid'].bool(), has_extent.repeat_interleave(8, dim=1)), dim=1)
        ego = self.query(torch.cat((ego_state, context), dim=-1))
        content = (ego[:, None] * self.key(nodes)).sum(-1) / math.sqrt(self.WIDTH)
        scores = content - torch.linalg.vector_norm(offsets, dim=-1) / self.DISTANCE_SCALE_M
        weights = scores.masked_fill(~valid, -1e9).softmax(-1) * valid
        return (weights.unsqueeze(-1) * nodes).sum(1), weights


class HeadingPolicy(nn.Module):
    """Encode observed road topology, fleet messages and the ordered terrain corridors."""

    def __init__(self, hidden=64):
        super().__init__()
        self.encoder = _GroundEncoder(FEATURE_COUNT, hidden, task_context=True)
        self.direction = nn.Sequential(nn.Linear(hidden, 8), nn.Tanh())
        self.ego_map = EgoMapAttention(self.encoder.ego[0].out_features)
        self.context = nn.Sequential(nn.Linear(8 * DIRECTION_COUNT + CONTEXT_SIZE + self.ego_map.WIDTH, hidden),
                                     nn.Tanh())
        self.actor = nn.Linear(hidden, 3)
        self.critic = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.actor.weight, gain=.01)
        nn.init.zeros_(self.actor.bias)
        with torch.no_grad():
            self.actor.bias[0] = -.7  # Prefer hold initially, while exploring turns.

    def _spatial_state(self, observations, *, include_recent_motion=True):
        device = self.actor.weight.device

        def batch(key, dtype):
            return torch.as_tensor(np.stack([obs[key] for obs in observations]), dtype=dtype, device=device)

        rows = batch('features', torch.float32)[:, :DIRECTION_COUNT]
        data = _blocks(batch('ground_inputs', torch.float32), TASK_BLOCKS)
        data['candidate_xy'] = data['candidate_xy'][:, :DIRECTION_COUNT]
        # All directions, including unobserved corridors, have a fixed slot.
        directions, ego_state, roads = self.encoder(rows, data, return_ego_and_roads=True)
        context = batch('context', torch.float32)
        if not include_recent_motion:
            # Fixed-waypoint experiments supply past displacement and progress
            # only through their separately controlled motion-history branch.
            context[:, 4:7] = 0.
        nearby, _ = self.ego_map(ego_state, context, roads, data, rows)
        encoded = self.direction(directions).flatten(1)
        state = self.context(torch.cat((encoded, context, nearby), dim=-1))
        return state, directions

    def forward(self, observations):
        state, _ = self._spatial_state(observations)
        return HeadingDistribution(self.actor(state)), self.critic(state).squeeze(-1)

    @torch.no_grad()
    def act(self, observation, deterministic=False, generator=None):
        policy, value = self([observation])
        turn, angle = policy.action(deterministic, generator)
        # Float32 can round the half-turn to just above the double-precision pi.
        signed_angle = max(-math.pi, min(math.pi, angle.item()))
        return bool(turn.item()), signed_angle, policy.log_prob(turn, angle).item(), value.item()

    @torch.no_grad()
    def value(self, observation):
        return float(self([observation])[1].item())


def _heading_log_prob(policy, rows, values):
    turns = torch.tensor([row['turn'] for row in rows], dtype=torch.bool, device=values.device)
    angles = torch.tensor([row['angle_rad'] for row in rows], dtype=values.dtype, device=values.device)
    return policy.log_prob(turns, angles)


def update_policy(model, optimizer, episodes, *, epochs=3, microbatch_size=16,
                  clip=.2, gae_lambda=1., entropy_coef=.005, action_log_prob=_heading_log_prob):
    """Use complete-mission rewards; collector cutoffs retain their critic bootstrap."""
    if not episodes or any(not ep for ep in episodes):
        raise ValueError('Heading PPO requires nonempty episodes')
    if min(epochs, microbatch_size) < 1 or not 0 < clip < 1:
        raise ValueError('Invalid heading PPO update configuration')
    transitions = [row for episode in episodes for row in episode]
    # A safety termination can fire before the next physics tick. At gamma=1
    # its zero-time transition has exactly the same return as a single tick.
    targets = []
    for episode in episodes:
        timed = [dict(row, duration=1) if row['terminated'] and row['duration'] == 0 else row
                 for row in episode]
        targets.append(smdp_advantages(timed, gamma=1., gae_lambda=gae_lambda))
    device = model.actor.weight.device
    advantages = torch.from_numpy(np.concatenate([a for a, _ in targets])).to(device)
    returns = torch.from_numpy(np.concatenate([r for _, r in targets])).to(device)
    # Keep rule-directed intervals in each return, including terminal rewards.
    # Only intervals where the actor selected an angle enter the PPO objective.
    actor_indices = [i for i, row in enumerate(transitions) if row.get('policy_controlled', True)]
    forced_indices = [i for i, row in enumerate(transitions) if not row.get('policy_controlled', True)]
    if any(transitions[i]['logp'] != 0. for i in forced_indices):
        raise ValueError('A rule-directed waypoint must have log probability zero')
    if len(actor_indices) > 1:
        actor_advantages = advantages[actor_indices]
        scale = actor_advantages.std(unbiased=False)
        if scale > 1e-6:
            advantages = (advantages - actor_advantages.mean()) / scale

    batches = [(use_actor, indices[start:start + microbatch_size])
               for use_actor, indices in ((True, actor_indices), (False, forced_indices))
               for start in range(0, len(indices), microbatch_size)]
    max_error = 0.
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.
        for use_actor, indices in batches:
            rows = [transitions[i] for i in indices]
            observations = [row['observation'] for row in rows]
            if use_actor:
                policy, values = model(observations)
                previous = torch.tensor([row['logp'] for row in rows], dtype=values.dtype, device=device)
                logp = action_log_prob(policy, rows, values)
                if epoch == 0:
                    error = float((logp.detach() - previous).abs().max())
                    if not math.isfinite(error) or error > 2e-4:
                        raise ValueError(f'Stale heading PPO behaviour log probability (error {error:g})')
                    max_error = max(max_error, error)
                ratio = (logp - previous).exp()
                adv = advantages[indices]
                objective = torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
                actor_loss = -(objective + entropy_coef * policy.entropy()).sum() / len(actor_indices)
            else:
                values = model.critic_values(observations)
                actor_loss = 0.
            value_loss = (values - returns[indices]).square().sum() / len(transitions)
            loss = actor_loss + .5 * value_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite heading PPO loss')
            loss.backward()
            loss_total += float(loss.detach())
        nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
        optimizer.step()
    return dict(transitions=len(transitions), policy_transitions=len(actor_indices), epochs=epochs, loss=loss_total,
                behaviour_logp_error=max_error, mean_return_target=float(returns.mean()))
