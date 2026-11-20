"""Continuous vehicle-relative guidance with attention over measured UGV motion."""
import numpy as np
import torch
from torch import nn

from ground.vtemp4_policy import HeadingDistribution, HeadingPolicy
from simulator.vtemp4_ring_heading import RingHeadingFleetEpisode
from ground.vtemp4_waypoint_policy import MotionHistoryAttention
from simulator.vtemp4_waypoint_runtime import WaypointProgress, motion_history


class ContinuousHistoryPolicy(HeadingPolicy):
    """Fuse spatial evidence and time-labelled UGV motion before selecting a signed angle."""

    def __init__(self, hidden=64):
        super().__init__(hidden)
        self.history_attention = MotionHistoryAttention(hidden)
        width = self.history_attention.value.out_features
        self.fuse = nn.Sequential(nn.Linear(hidden + width, hidden), nn.Tanh())

    def _history_state(self, observations):
        spatial, _ = self._spatial_state(observations, include_recent_motion=False)

        def batch(name, dtype):
            return torch.as_tensor(np.stack([obs[name] for obs in observations]),
                                   dtype=dtype, device=spatial.device)

        history = self.history_attention(spatial, batch('history', torch.float32), batch('history_mask', torch.bool))
        return self.fuse(torch.cat((spatial, history), dim=-1))

    def forward(self, observations):
        state = self._history_state(observations)
        logits = self.actor(state)
        can_hold = torch.as_tensor([obs['can_hold'] for obs in observations], dtype=torch.bool, device=state.device)
        can_designate = torch.as_tensor([obs['can_designate'] for obs in observations], dtype=torch.bool, device=state.device)
        if not (can_hold | can_designate).all():
            raise ValueError('A navigation review must permit hold or designate')
        # Saturation gives forced gates probability exactly zero or one in this
        # dtype; finite logits keep their Bernoulli log-probability and entropy defined.
        limit = torch.finfo(logits.dtype).max
        gate = torch.where(can_hold, logits[:, 0], limit)
        gate = torch.where(can_designate, gate, -limit)
        distribution = HeadingDistribution(torch.cat((gate[:, None], logits[:, 1:]), dim=-1))
        return distribution, self.critic(state).squeeze(-1)


class ContinuousHistoryFleetEpisode(WaypointProgress, RingHeadingFleetEpisode):
    """Protect a point while approaching it, while keeping every new angle continuous."""

    def _after_fusion(self, graph, semantic_map):
        self._record_pose(self.clock_s, self.position())
        return super()._after_fusion(graph, semantic_map)

    def _heading_observation(self):
        obs = super()._heading_observation()
        now, position, target = self.clock_s, self.position(), self._target()
        self._record_pose(now, position)
        reason = self._review_reason(now, position, target)
        history, mask = motion_history(self.samples, now, position, self.goal, target, self.lookahead_m,
                                      issued_at=self.held_since)
        obs.update(history=history, history_mask=mask, review_reason=reason,
                   can_hold=reason in ('tracking', 'stalled'), can_designate=reason != 'tracking')
        return obs

    def _choose_heading(self, now, position, distance):
        super()._choose_heading(now, position, distance)
        event = self.heading_reviews[-1]
        reason = self.pending['observation']['review_reason']
        event['review_reason'] = reason
        if event['changed']:
            self._reset_progress(now, float(np.linalg.norm(self._target() - position)))
        self.waiting_for_evidence = event['reason'] == 'invalid_proposal' and reason != 'stalled'

    def _ground_decision(self, now, guidance=None, wait_for_guidance=False):
        super()._ground_decision(now, guidance=guidance, wait_for_guidance=(
            wait_for_guidance or (self.waiting_for_evidence and not self.final_approach)))
