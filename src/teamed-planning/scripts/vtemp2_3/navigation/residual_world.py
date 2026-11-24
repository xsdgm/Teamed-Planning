"""Causal, action-conditioned local robot dynamics with an actor prior.

Only received history/context and an open-loop actor prior enter forward().
Simulator truth is used exclusively by the masked supervised objective.
The diagonal Gaussian describes predictive spread, not certified confidence.
"""
from __future__ import annotations
import copy
import math
import numpy as np
import torch
from torch import nn
from .telemetry import HISTORY_FEATURES, CONTEXT_FEATURES, POSITION_SCALE, SPEED_SCALE

MODEL_SCHEMA = 'vtemp2_actor_residual_dynamics_v1'
INPUT_KEYS = ('history', 'history_mask', 'context', 'baseline_states', 'baseline_actions')


def bounded_velocity(value):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1.)


class ResidualWorldModel(nn.Module):
    def __init__(self, hidden=64, horizon=4):
        super().__init__()
        if hidden < 1 or horizon < 1:
            raise ValueError('Positive hidden width and prediction horizon required')
        self.hidden, self.horizon = hidden, horizon
        self.encoder = nn.GRUCell(HISTORY_FEATURES, hidden)
        self.context = nn.Sequential(nn.Linear(hidden + CONTEXT_FEATURES, hidden), nn.Tanh())
        self.initial = nn.Linear(hidden, 8)
        self.decoder = nn.GRUCell(11, hidden)
        self.action = nn.Linear(hidden, 4)
        self.innovation = nn.Linear(hidden, 4)
        # Start close to the declared nominal prior; retain positive variances.
        for layer in (self.initial, self.action, self.innovation):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def scale(raw):
        return raw.clamp(-4., 2.).exp() * .2 + .01

    def forward(self, history, history_mask, context, baseline_states, baseline_actions):
        if baseline_actions.shape[1:] != (self.horizon, 2):
            raise ValueError('Actor prior horizon differs from trained model')
        h = history.new_zeros((len(history), self.hidden))
        for t in range(history.shape[1]):
            proposed = self.encoder(history[:, t], h)
            h = torch.where(history_mask[:, t, None].bool(), proposed, h)
        h = self.context(torch.cat((h, context), -1))
        initial = self.initial(h)
        current = baseline_states[:, 0] + initial[:, :4]
        current = torch.cat((current[:, :2], bounded_velocity(current[:, 2:])), -1)
        spread = self.scale(initial[:, 4:])
        states, stds, actions, action_stds = [current], [spread], [], []
        ratio = SPEED_SCALE / POSITION_SCALE  # dt = 1 second
        for t in range(self.horizon):
            timestep = current.new_full((len(current), 1), (t + 1) / self.horizon)
            decoder_input = torch.cat((current, baseline_states[:, t + 1],
                                       baseline_actions[:, t], timestep), -1)
            h = self.decoder(decoder_input, h)
            correction = self.action(h)
            action = bounded_velocity(baseline_actions[:, t] + correction[:, :2])
            action_std = self.scale(correction[:, 2:])
            innovation = self.innovation(h)
            position = current[:, :2] + ratio * action + .1 * innovation[:, :2].tanh()
            current = torch.cat((position, action), -1)
            # Propagate old uncertainty; no invented future measurements.
            position_std = (spread[:, :2].square() + ratio**2 * action_std.square()
                            + (.05 * self.scale(innovation[:, 2:])).square()).sqrt()
            spread = torch.cat((position_std, action_std), -1)
            states.append(current); stds.append(spread)
            actions.append(action); action_stds.append(action_std)
        return dict(states=torch.stack(states, 1), state_std=torch.stack(stds, 1),
                    actions=torch.stack(actions, 1), action_std=torch.stack(action_stds, 1))


def gaussian_loss(mean, std, truth, mask):
    terms = .5 * ((truth - mean) / std).square() + std.log() + .5 * math.log(2 * math.pi)
    valid = mask.unsqueeze(-1).expand_as(terms)
    if not valid.any():
        return mean.sum() * 0.
    return terms[valid].mean()


def objective(prediction, batch):
    state = gaussian_loss(prediction['states'], prediction['state_std'],
                          batch['target_states'], batch['state_mask'])
    action = gaussian_loss(prediction['actions'], prediction['action_std'],
                           batch['target_actions'], batch['action_mask'])
    return state + .5 * action


def infer(model, batch):
    return model(**{key: batch[key] for key in INPUT_KEYS})


def infer_graph(model, graph):
    """Predict peers while preserving deterministic ego safety evidence.

    Adapter for the existing residual model, not a learned heterogeneous GNN.
    The model only receives INPUT_KEYS; graph metadata and truth labels are
    never neural inputs. Current CBF stays tied to received geometry at t.
    """
    from .safety_context import GRAPH_SCHEMA, validate_cbf_context
    if graph.get('schema') != GRAPH_SCHEMA or graph.get('phase') != 'observation':
        raise ValueError('World model requires an observation graph')
    owner, now = graph['ego_id'], graph['estimate_stamp']
    validate_cbf_context(graph['nodes'][owner]['cbf'], owner, now)
    result = copy.deepcopy(graph)
    peers = [node for name, node in result['nodes'].items() if name != owner]
    if peers:
        device = next(model.parameters(), torch.empty(0)).device
        batch = {key: torch.as_tensor(np.asarray([node['world_input'][key] for node in peers]),
                                     device=device) for key in INPUT_KEYS}
        with torch.inference_mode():
            prediction = {key: value.detach().cpu().numpy() for key, value in infer(model, batch).items()}
        for i, node in enumerate(peers):
            node['prediction'] = {key: value[i].copy() for key, value in prediction.items()}
            node['estimate_stamp'] = now
    result['phase'] = 'belief'
    return result
