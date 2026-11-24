"""Shared UAV policy with causal peer predictions and a separate CTDE critic.

The archived policy is copied, never changed in place. An absent prediction
has an explicit mask. Actor-prior queries use that absent view, so inference
does not recursively invoke the world model.
"""
from __future__ import annotations
import copy
import hashlib
import numpy as np
import torch
from torch import nn
from .frozen_navigation import runtime

FAST_SCHEMA = 'vtemp2_prediction_conditioned_shared_fast_v1'
HORIZON = 4
PEERS = 2  # This experiment has three UAVs sharing one actor.
TOKEN_CONTENT = (HORIZON + 1) * 8 + HORIZON * 4 + 4
TOKEN_WIDTH = TOKEN_CONTENT + 1
PREDICTION_WIDTH = PEERS * TOKEN_WIDTH


class PeerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(TOKEN_CONTENT, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh())

    def forward(self, tokens):
        mask = tokens[..., -1:].clamp(0, 1)
        # Mask before the MLP as well: missing payloads cannot affect outputs.
        value = self.net(torch.where(mask.bool(), tokens[..., :-1], 0.)) * mask
        return value.sum(-2) / mask.sum(-2).clamp_min(1.)


class ResidualFastActorCritic(nn.Module):
    """Adapt the archived actor and team critic with separate peer encoders."""

    input_version = 2
    feature_version = 8
    action_mode = 'velocity'

    def __init__(self, base, raw_width, encoder):
        super().__init__()
        self.raw_width = raw_width
        self.base = copy.deepcopy(base).requires_grad_(True)
        self.actor_peers = encoder()
        self.actor_delta = nn.Linear(32, 2)
        self.critic_peers = encoder()
        self.critic_delta = nn.Linear(32, 1)
        for layer in (self.actor_delta, self.critic_delta):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def encode_graph(self, *args, **kwargs):
        return self.base.encode_graph(*args, **kwargs)

    def velocity(self, *args, **kwargs):
        return self.base.velocity(*args, **kwargs)

    def distribution(self, x, mask=None):
        raw, tokens = self.split(x)
        old = self.base.distribution(raw)
        delta = .5 * self.actor_delta(self.actor_peers(tokens)).tanh()
        return torch.distributions.Independent(torch.distributions.Normal(old.mean + delta, old.stddev), 1)

    def value(self, x, mask=None, state=None):
        raw, tokens = self.split(x)
        peers = self.critic_peers(tokens).mean(-2)
        return self.base.value(raw, state=state) + self.critic_delta(peers).squeeze(-1) * self.base.value_std

    def actor_parameters(self):
        return [*self.base.gat_actor.parameters(), self.base.log_std,
                *self.actor_peers.parameters(), *self.actor_delta.parameters()]

    def critic_parameters(self):
        return [*self.base.critic.parameters(), *self.critic_peers.parameters(), *self.critic_delta.parameters()]


class PredictionActorCritic(ResidualFastActorCritic):
    def __init__(self, base):
        from mappo.relay_attention_mappo import RAW_WIDTH
        super().__init__(base, RAW_WIDTH, PeerEncoder)

    def split(self, x):
        if x.shape[-1] == self.raw_width:
            tokens = x.new_zeros((*x.shape[:-1], PEERS, TOKEN_WIDTH))
        elif x.shape[-1] == self.raw_width + PREDICTION_WIDTH:
            tokens = x[..., self.raw_width:].reshape(*x.shape[:-1], PEERS, TOKEN_WIDTH)
        else:
            raise ValueError('Wrong prediction-conditioned feature width')
        return x[..., :self.raw_width], tokens


def make_fast(repo):
    return PredictionActorCritic(runtime(repo)[2])


def numpy_state(model):
    """Copy model parameters into independent NumPy arrays for rollout workers."""
    return {key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()}


def load_numpy(model, state):
    model.load_state_dict({key: torch.from_numpy(value.copy()) for key, value in state.items()})


def state_hash(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        result.update(key.encode()); result.update(str(value.shape).encode()); result.update(value.tobytes())
    return result.hexdigest()


def prediction_token(prediction, index, origin, own_position, last, now):
    """Only predicted quantities and received packet metadata, never labels."""
    states = prediction['states'][index].copy()
    states[:, :2] = (states[:, :2] * 20. + origin - np.asarray(own_position[:2])) / 95.
    std = prediction['state_std'][index].copy(); std[:, :2] *= 20. / 95.
    ack = last['goal_stamp'] is not None
    metadata = [(now - last['stamp']) / 30., last['hop'] / 8.,
                (now - last['goal_stamp']) / 30. if ack else 0., float(ack)]
    return np.r_[states.ravel(), std.ravel(), prediction['actions'][index].ravel(),
                 prediction['action_std'][index].ravel(), metadata, 1.].astype(np.float32)


def graph_features(graph):
    """Read received CBF from the belief graph without changing checkpoint width."""
    from .safety_context import GRAPH_SCHEMA, CBF_FEATURE_INDEX, validate_cbf_context
    if graph.get('schema') != GRAPH_SCHEMA or graph.get('phase') != 'belief':
        raise ValueError('Actor requires a belief graph')
    owner, now = graph['ego_id'], graph['estimate_stamp']
    ego = graph['nodes'][owner]
    norm = validate_cbf_context(ego['cbf'], owner, now)
    raw = np.asarray(ego['actor_features'], dtype=np.float32).copy()
    if raw.ndim != 1 or len(raw) <= CBF_FEATURE_INDEX:
        raise ValueError('Missing legacy actor state in graph')
    if raw[CBF_FEATURE_INDEX] != np.float32(norm):
        raise ValueError('Actor CBF feature disagrees with received graph context')
    raw[CBF_FEATURE_INDEX] = norm
    peers = [node for name, node in graph['nodes'].items() if name != owner]
    if len(peers) > PEERS:
        raise ValueError('Fleet size exceeds the declared three-UAV experiment')
    suffix = np.zeros(PREDICTION_WIDTH, np.float32)
    for i, node in enumerate(peers):
        if node['estimate_stamp'] != now:
            raise ValueError('Stale peer prediction at actor decision')
        prediction = {key: np.asarray(value)[None] for key, value in node['prediction'].items()}
        suffix[i * TOKEN_WIDTH:(i + 1) * TOKEN_WIDTH] = prediction_token(
            prediction, 0, node['origin'], ego['position'], node['packet'], now)
    return np.r_[raw, suffix].astype(np.float32)


def ppo_batch(transitions, steps):
    """Use the first unused behavior value to bootstrap a truncated rollout."""
    rows = transitions[:steps]
    if not rows:
        raise ValueError('No fast behavior transitions')
    if rows[-1]['terminals']:
        bootstrap = 0.
    elif len(transitions) > steps:
        bootstrap = transitions[steps]['values']
    else:
        raise ValueError('Truncated fast rollout lacks next-state bootstrap')
    return {**{key: np.asarray([row[key] for row in rows]) for key in rows[0]},
            'bootstrap': np.asarray(bootstrap, np.float32)}


def advantages(rewards, values, bootstrap, terminals, gamma=.995, lam=.97):
    rewards, values = np.asarray(rewards), np.asarray(values)
    result = np.zeros(len(rewards), np.float32); running = 0.
    for t in reversed(range(len(rewards))):
        future = bootstrap if t == len(rewards) - 1 else values[t + 1]
        continuation = 1. - float(terminals[t])
        delta = rewards[t] + gamma * continuation * future - values[t]
        running = delta + gamma * lam * continuation * running
        result[t] = running
    return result, result + values


def ppo_update(model, batches, actor_optimizer, critic_optimizer, device, rng, *,
               epochs=4, minibatch=32, max_kl=.02):
    """On-policy team GAE, latent actions, separate actor/critic optimizers.

    Actor epochs exceeding the analytic KL limit roll back both weights and
    Adam moments. Time-limit truncation bootstraps; physical terminal does not.
    """
    fields = ('features', 'actions', 'logp', 'states')
    arrays = {key: np.concatenate([b[key] for b in batches]) for key in fields}
    gae = [advantages(b['rewards'], b['values'], float(b['bootstrap']), b['terminals']) for b in batches]
    adv = np.concatenate([a for a, _ in gae]); returns = np.concatenate([r for _, r in gae])
    adv = (adv - adv.mean()) / max(float(adv.std()), 1e-6)
    data = {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}
    advantage = torch.as_tensor(adv, device=device)[:, None]
    target = torch.as_tensor(returns, device=device)
    count = len(target)
    initial = numpy_state(model)
    old_mean, old_std = [], []
    discrepancy = 0.
    with torch.no_grad():
        for begin in range(0, count, minibatch):
            sl = slice(begin, begin + minibatch)
            d = model.distribution(data['features'][sl])
            discrepancy = max(discrepancy, float((d.log_prob(data['actions'][sl]) - data['logp'][sl]).abs().max()))
            old_mean.append(d.mean); old_std.append(d.stddev)
    if discrepancy > 5e-4:
        raise ValueError(f'Stale or inconsistent behavior policy: log probability error {discrepancy}')
    old_mean, old_std = torch.cat(old_mean), torch.cat(old_std)
    actor_parameters = model.actor_parameters()
    critic_parameters = model.critic_parameters()
    accepted = 0; rejected = 0; final_kl = 0.; actor_steps = 0; critic_steps = 0
    for _ in range(epochs):
        saved = [p.detach().clone() for p in actor_parameters]
        moments = copy.deepcopy(actor_optimizer.state_dict())
        order = rng.permutation(count); epoch_steps = 0
        for begin in range(0, count, minibatch):
            ids = order[begin:begin + minibatch]
            distribution = model.distribution(data['features'][ids])
            ratio = (distribution.log_prob(data['actions'][ids]) - data['logp'][ids]).exp()
            objective = torch.minimum(ratio * advantage[ids], ratio.clamp(.8, 1.2) * advantage[ids])
            loss = -objective.mean() - .001 * distribution.entropy().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite PPO actor loss')
            actor_optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, .5, error_if_nonfinite=True)
            actor_optimizer.step(); epoch_steps += 1
        kl_sum = 0.
        with torch.no_grad():
            for begin in range(0, count, minibatch):
                sl = slice(begin, begin + minibatch)
                d = model.distribution(data['features'][sl])
                old = torch.distributions.Normal(old_mean[sl], old_std[sl])
                new = torch.distributions.Normal(d.mean, d.stddev)
                kl_sum += float(torch.distributions.kl_divergence(old, new).sum(-1).sum())
        kl = kl_sum / (count * arrays['actions'].shape[1])
        if not np.isfinite(kl) or kl > max_kl:
            with torch.no_grad():
                for parameter, value in zip(actor_parameters, saved): parameter.copy_(value)
            actor_optimizer.load_state_dict(moments)
            for group in actor_optimizer.param_groups: group['lr'] *= .5
            rejected += 1
            break
        final_kl = kl; accepted += 1; actor_steps += epoch_steps
    value_scale = max(float(np.std(returns)), float(model.base.value_std), 1.)
    critic_loss = 0.
    for _ in range(epochs):
        order = rng.permutation(count)
        for begin in range(0, count, minibatch):
            ids = order[begin:begin + minibatch]
            value = model.value(data['features'][ids], state=data['states'][ids])
            loss = .5 * ((value - target[ids]) / value_scale).square().mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite PPO critic loss')
            critic_optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_parameters, .5, error_if_nonfinite=True)
            critic_optimizer.step(); critic_steps += 1; critic_loss = float(loss.detach())
    final = numpy_state(model)
    delta = lambda prefix: float(np.sqrt(sum(np.sum((final[k].astype(float) - initial[k])**2)
                                  for k in final if k.startswith(prefix))))
    return dict(transitions=count, agent_actions=count * arrays['actions'].shape[1],
                behavior_logp_max_error=discrepancy, accepted_actor_epochs=accepted,
                rejected_actor_epochs=rejected, actor_optimizer_steps=actor_steps,
                critic_optimizer_steps=critic_steps, analytic_kl=final_kl,
                critic_loss=critic_loss, mean_return=float(np.mean(returns)),
                base_actor_parameter_delta=delta('base.gat_actor.'),
                prediction_adapter_parameter_delta=delta('actor_'),
                critic_parameter_delta=delta('base.critic.'))
