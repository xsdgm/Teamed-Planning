"""Shared recurrent dynamics for successive world-model task variants.

Task modules supply dimensions and observation adapters. Parameter names stay
at their original level so existing vtmp2/vtmp3 state_dicts load unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RecurrentWorldModel(nn.Module):
    def __init__(self, *, channels, side, features, actions, outcomes, delta_scale,
                 hidden=128, latent=24, state_head_name='features_head'):
        super().__init__()
        self.hidden, self.latent = hidden, latent
        self._channels, self._side = channels, side
        self._state_head_name = state_head_name
        self.encoder = nn.Sequential(nn.Conv2d(channels, 16, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.SiLU(), nn.Conv2d(32, 32, 4, 2, 1),
            nn.SiLU(), nn.Flatten(), nn.Linear(32*(side//8)**2, 96), nn.SiLU())
        self.memory = nn.GRUCell(latent+actions, hidden)
        self.prior = nn.Linear(hidden, 2*latent)
        self.posterior = nn.Sequential(nn.Linear(hidden+96+features, 128), nn.SiLU(), nn.Linear(128, 2*latent))
        head = nn.Sequential(nn.Linear(hidden+latent, 128), nn.SiLU(), nn.Linear(128, features))
        self.add_module(state_head_name, head)
        self.map_head = nn.Sequential(nn.Linear(hidden+latent, 256), nn.SiLU(), nn.Linear(256, channels*side*side))
        self.outcome_head = nn.Sequential(nn.Linear(hidden+latent, 64), nn.SiLU(), nn.Linear(64, outcomes))
        self.register_buffer('delta_scale', torch.tensor(delta_scale))
        for layer in (head[-1], self.map_head[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def distribution(parameters):
        mean, raw = parameters.chunk(2, -1)
        return torch.distributions.Normal(mean, .1+F.softplus(raw))

    def observe_embedding(self, hidden, embedding, state, sample=False):
        posterior = self.distribution(self.posterior(torch.cat((hidden, embedding, state), -1)))
        return (posterior.rsample() if sample else posterior.mean), posterior

    def posterior_input(self, maps):
        return self.encoder(maps)

    def initial(self, maps, states, sample=False):
        hidden = states.new_zeros((len(states), self.hidden))
        latent, _ = self.observe(hidden, self.posterior_input(maps), states, sample)
        return hidden, latent

    def imagine(self, hidden, latent, action, maps, states, sample=False):
        hidden = self.memory(torch.cat((latent, action), -1), hidden)
        prior = self.distribution(self.prior(hidden))
        latent = prior.rsample() if sample else prior.mean
        joint = torch.cat((hidden, latent), -1)
        predicted_states = states+getattr(self, self._state_head_name)(joint)*self.delta_scale
        predicted_maps = (maps+self.map_head(joint).reshape(-1, self._channels, self._side, self._side)).clamp(0, 1)
        return hidden, latent, prior, predicted_maps, predicted_states, self.outcome_head(joint)
