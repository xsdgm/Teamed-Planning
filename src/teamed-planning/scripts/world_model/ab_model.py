"""Action-conditioned recurrent model of survey reception AND navigation."""
import torch
from torch.nn import functional as F

from .ab_environment import ACTIONS, CHANNELS, DELTA_SCALE, FEATURES, SIDE


from .rssm import RecurrentWorldModel


class ABWorldModel(RecurrentWorldModel):
    algorithm = 'vtmp3'
    model_architecture = 'rssm_ab_vtmp3'

    def __init__(self, hidden=128, latent=24):
        super().__init__(channels=CHANNELS, side=SIDE, features=FEATURES,
                         actions=ACTIONS, outcomes=5, delta_scale=DELTA_SCALE,
                         hidden=hidden, latent=latent, state_head_name='state_head')

    def observe(self, hidden, maps, states, sample=False):
        return self.observe_embedding(hidden, self.encoder(maps), states, sample)

    def posterior_input(self, maps):
        return maps


WorldModel = ABWorldModel


def prediction_loss(model, pm, ps, outcome, maps, states, before_maps, labels):
    learnable = model.delta_scale > 0
    state = (((ps-states)[:, learnable]/model.delta_scale[learnable])**2).mean(-1)
    weights = 1+4*(maps > 0).float()+10*(abs(maps-before_maps) > .01).float()
    grid = ((pm-maps).square()*weights).mean((1, 2, 3))
    continuous = F.smooth_l1_loss(outcome[:, :3], labels[:, :3], reduction='none').mean(-1)
    events = F.binary_cross_entropy_with_logits(outcome[:, 3:], labels[:, 3:], reduction='none').mean(-1)
    return state+5*grid+continuous+.2*events


def sequence_loss(model, batch, rollout_steps=4, rollout_weight=.25, sample=True):
    maps, states, actions, labels, mask = (batch[k] for k in ('maps', 'states', 'actions', 'outcomes', 'mask'))
    hidden, latent = model.initial(maps[:, 0], states[:, 0], sample)
    losses = []
    for t in range(actions.shape[1]):
        hidden, _, prior, pm, ps, outcome = model.imagine(hidden, latent, actions[:, t], maps[:, t], states[:, t], sample)
        loss = prediction_loss(model, pm, ps, outcome, maps[:, t+1], states[:, t+1], maps[:, t], labels[:, t])
        latent, posterior = model.observe(hidden, maps[:, t+1], states[:, t+1], sample)
        detached_posterior = torch.distributions.Normal(posterior.mean.detach(), posterior.stddev.detach())
        detached_prior = torch.distributions.Normal(prior.mean.detach(), prior.stddev.detach())
        kl = .8*torch.distributions.kl_divergence(detached_posterior, prior)
        kl += .2*torch.distributions.kl_divergence(posterior, detached_prior)
        losses.append(loss+.01*kl.clamp_min(.1).mean(-1))
    loss = (torch.stack(losses, 1)*mask).sum()/mask.sum().clamp_min(1)
    if rollout_weight:
        hidden, latent = model.initial(maps[:, 0], states[:, 0], sample)
        pm, ps = maps[:, 0], states[:, 0]
        rollout, valid = [], mask.cumprod(1)
        for t in range(min(rollout_steps, actions.shape[1])):
            hidden, latent, _, pm, ps, outcome = model.imagine(hidden, latent, actions[:, t], pm, ps, sample)
            rollout.append(prediction_loss(model, pm, ps, outcome, maps[:, t+1], states[:, t+1], maps[:, t], labels[:, t]))
        m = valid[:, :len(rollout)]
        loss += rollout_weight*(torch.stack(rollout, 1)*m).sum()/m.sum().clamp_min(1)
    return loss
