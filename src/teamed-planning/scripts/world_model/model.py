"""vtmp2 action-conditioned RSSM; predicts outcomes for supplied targets."""
import math

import torch
from torch.nn import functional as F

if __package__:
    from .data import CHANNELS, FEATURES, SIDE, DELTA_SCALE
    from .models import RecurrentWorldModel
else:
    from data import CHANNELS, FEATURES, SIDE, DELTA_SCALE
    from models import RecurrentWorldModel


class Vtmp2WorldModel(RecurrentWorldModel):
    algorithm = 'vtmp2'
    model_architecture = 'rssm_target_vtmp2'

    def __init__(self, hidden=128, latent=24):
        super().__init__(channels=CHANNELS, side=SIDE, features=FEATURES,
                         actions=3, outcomes=4, delta_scale=DELTA_SCALE,
                         hidden=hidden, latent=latent, state_head_name='features_head')

    def observe(self, hidden, embedding, state, sample):
        return self.observe_embedding(hidden, embedding, state, sample)


# Preserve imports used by early experiments; checkpoint parameter names match.
WorldModel = Vtmp2WorldModel


def prediction_loss(model, predicted_maps, predicted_states, outcome, maps, states, previous_maps, labels, elapsed):
    # Static surveyed goal coordinates are copied exactly, never learned.
    learnable = model.delta_scale > 0
    error = (predicted_states[:, learnable] - states[:, learnable]) / model.delta_scale[learnable]
    state_loss = error.square().mean(-1)
    # Do not allow the abundance of blank map pixels to hide change errors.
    weights = 1 + 4 * ((maps > 0) | (previous_maps > 0)).float() + 10 * (abs(maps-previous_maps) > .01).float()
    map_loss = ((predicted_maps - maps).square() * weights).mean((1, 2, 3))
    event_loss = F.binary_cross_entropy_with_logits(outcome[:, :3], labels, reduction='none').mean(-1)
    duration_loss = (outcome[:, 3].sigmoid() - elapsed).square()
    return state_loss + 5 * map_loss + .1 * event_loss + .1 * duration_loss


def imagine_sequence(model, initial_maps, initial_states, actions, steps, sample=False):
    """Roll out recorded actions from one observation, without future assimilation.

    Actions may change their target and purpose between transitions. Completion
    means drive arrival or survey evidence received during that transition,
    never eventual ground reachability of the target.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError('rollout steps must be a positive integer')
    hidden, latent = model.initial(initial_maps, initial_states, sample=sample)
    maps, states = initial_maps, initial_states
    for t in range(min(steps, actions.shape[1])):
        hidden, latent, _, maps, states, outcome = model.imagine(
            hidden, latent, actions[:, t], maps, states, sample)
        yield maps, states, outcome


def rollout_prediction_loss(model, batch, rollout_steps):
    """One free-running anchor per window; ground truth is used only as a label."""
    maps, states = batch['maps'], batch['states']
    valid = batch['mask'].cumprod(1)
    losses, masks = [], []
    for t, (pm, ps, outcome) in enumerate(imagine_sequence(
            model, maps[:, 0], states[:, 0], batch['actions'], rollout_steps, sample=True)):
        if not valid[:, t].any():
            break
        losses.append(prediction_loss(model, pm, ps, outcome, maps[:, t+1], states[:, t+1],
                                      maps[:, t], batch['labels'][:, t], batch['elapsed'][:, t]))
        masks.append(valid[:, t])
    if not losses:
        return states.new_zeros(())
    mask = torch.stack(masks, 1)
    return (torch.stack(losses, 1) * mask).sum() / mask.sum().clamp_min(1)


def sequence_loss(model, batch, rollout_steps=4, rollout_weight=.25):
    if type(rollout_steps) is not int or rollout_steps < 1:
        raise ValueError('rollout_steps must be a positive integer')
    if not math.isfinite(rollout_weight) or rollout_weight < 0:
        raise ValueError('rollout_weight must be finite and nonnegative')
    maps, states = batch['maps'], batch['states']
    batch_size, observations = states.shape[:2]
    embeddings = model.encoder(maps.flatten(0, 1)).reshape(batch_size, observations, -1)
    hidden = states.new_zeros((batch_size, model.hidden))
    latent, _ = model.observe(hidden, embeddings[:, 0], states[:, 0], True)
    losses = []
    for t in range(observations - 1):
        hidden, _, prior, pm, ps, outcome = model.imagine(hidden, latent, batch['actions'][:, t],
                                                        maps[:, t], states[:, t], True)
        # Crucially, compute prior predictions BEFORE assimilating observation t+1.
        loss = prediction_loss(model, pm, ps, outcome, maps[:, t+1], states[:, t+1],
                               maps[:, t], batch['labels'][:, t], batch['elapsed'][:, t])
        latent, posterior = model.observe(hidden, embeddings[:, t+1], states[:, t+1], True)
        detached_posterior = torch.distributions.Normal(posterior.mean.detach(), posterior.stddev.detach())
        detached_prior = torch.distributions.Normal(prior.mean.detach(), prior.stddev.detach())
        kl = .8 * torch.distributions.kl_divergence(detached_posterior, prior)
        kl = kl + .2 * torch.distributions.kl_divergence(posterior, detached_prior)
        losses.append(loss + .01 * kl.clamp_min(.1).mean(-1))
    losses = torch.stack(losses, 1)
    loss = (losses * batch['mask']).sum() / batch['mask'].sum().clamp_min(1)
    if rollout_weight:
        loss = loss + rollout_weight * rollout_prediction_loss(model, batch, rollout_steps)
    return loss
