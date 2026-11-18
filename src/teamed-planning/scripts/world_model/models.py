"""Shared recurrent dynamics, sequence training helpers and architecture selection.

Task variants supply observation adapters, dimensions and losses. Parameters
remain at their original level so existing vtmp2/vtmp3 state_dicts still load.
"""
import numpy as np
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


def padded_window(episode, start, length, transition_keys, *, observation_dtype=None):
    """Repeat the last observation and mask zero-padded transitions at the scene boundary."""
    count = min(length, len(episode['actions']) - start)
    result = {}
    for key in ('maps', 'states'):
        values = episode[key][start:start+count+1]
        padded = np.empty((length+1, *values.shape[1:]), dtype=observation_dtype or values.dtype)
        padded[:count+1] = values
        padded[count+1:] = values[-1]
        result[key] = padded
    for key in transition_keys:
        values = episode[key][start:start+count]
        padded = np.zeros((length, *values.shape[1:]), np.float32)
        padded[:count] = values
        result[key] = padded
    mask = np.zeros(length, np.float32)
    mask[:count] = 1
    result['mask'] = mask
    return result


def balanced_kl(posterior, prior):
    """Balance prior and posterior gradients, with free nats per latent dimension."""
    detached_posterior = torch.distributions.Normal(posterior.mean.detach(), posterior.stddev.detach())
    detached_prior = torch.distributions.Normal(prior.mean.detach(), prior.stddev.detach())
    kl = .8 * torch.distributions.kl_divergence(detached_posterior, prior)
    kl = kl + .2 * torch.distributions.kl_divergence(posterior, detached_prior)
    return kl.clamp_min(.1).mean(-1)


def imagine_sequence(model, initial_maps, initial_states, actions, steps, sample=False):
    """Roll out a sequence of recorded actions from one observation without assimilating future observations."""
    if type(steps) is not int or steps < 1:
        raise ValueError('rollout steps must be a positive integer')
    hidden, latent = model.initial(initial_maps, initial_states, sample=sample)
    maps, states = initial_maps, initial_states
    for t in range(min(steps, actions.shape[1])):
        hidden, latent, _, maps, states, outcome = model.imagine(
            hidden, latent, actions[:, t], maps, states, sample)
        yield maps, states, outcome


ARCHITECTURES = {'vtmp2': 'rssm_target_vtmp2', 'vtmp3': 'rssm_ab_vtmp3'}
ADDITIONAL_ARCHITECTURES = {'terrain_dynamics_vtmp2': 'vtmp2', 'semantic_completion_vtmp2': 'vtmp2',
                            'coarse_frontier_vtmp2': 'vtmp2', 'abstract_observation_vtmp2': 'vtmp2'}


def architecture_matches(algorithm, architecture):
    return (algorithm in ARCHITECTURES and (ARCHITECTURES[algorithm] == architecture
            or ADDITIONAL_ARCHITECTURES.get(architecture) == algorithm))


def make_model(config):
    algorithm = config.get('algorithm')
    architecture = config.get('model_architecture', ARCHITECTURES.get(algorithm))
    if architecture not in ARCHITECTURES.values() and architecture not in ADDITIONAL_ARCHITECTURES:
        raise ValueError('Unknown world-model architecture')
    if algorithm is not None and not architecture_matches(algorithm, architecture):
        raise ValueError('Algorithm and world-model architecture disagree')
    if architecture == 'abstract_observation_vtmp2':
        from .abstract_model import AbstractWorldModel
        return AbstractWorldModel(**{key: config.get(key, default) for key, default in
            (('hidden', 24), ('latent', 96), ('members', 3))})
    if architecture == 'coarse_frontier_vtmp2':
        from .frontier_model import FrontierWorldModel
        return FrontierWorldModel(**{key: config.get(key, default) for key, default in
            (('hidden', 24), ('members', 3), ('cell_size', 16.), ('frontier_m', 32.))})
    if architecture == 'semantic_completion_vtmp2':
        from .semantic_model import SemanticCompletionModel
        return SemanticCompletionModel(**{key: config.get(key, default)
            for key, default in (('hidden', 24), ('members', 3))})
    if architecture == 'terrain_dynamics_vtmp2':
        from .decoupled_model import DecoupledWorldModel
        return DecoupledWorldModel(**{key: config.get(key, default)
            for key, default in (('drone_count', 3), ('hidden', 24), ('members', 3))})
    options = {key: config.get(key, default) for key, default in (('hidden', 128), ('latent', 24))}
    if any(type(value) is not int or value < 1 for value in options.values()):
        raise ValueError('Model dimensions must be positive integers')
    if architecture == 'rssm_target_vtmp2':
        from .model import Vtmp2WorldModel
        return Vtmp2WorldModel(**options)
    from .ab_model import ABWorldModel
    return ABWorldModel(**options)
