"""Masked candidate actor, belief-value critic and duration-aware PPO.

Gamma is per fast control attempt. GAE lambda is per slow decision, so its
recursion uses gamma**duration * lambda. True terminals have no bootstrap;
collector truncations bootstrap but never leak advantages across resets.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence


class SlowActorCritic(nn.Module):
    def __init__(self, features, hidden=64):
        super().__init__()
        self.features=int(features);self.hidden=int(hidden)
        self.actor_encoder=nn.Sequential(nn.Linear(features,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.actor_head=nn.Linear(hidden,1)
        self.critic_encoder=nn.Sequential(nn.Linear(features,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.critic_head=nn.Sequential(nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        nn.init.orthogonal_(self.actor_head.weight,gain=.01)
        nn.init.zeros_(self.actor_head.bias)

    def forward(self,x,mask,present):
        if x.shape[-1]!=self.features or x.shape[:-1]!=mask.shape or mask.shape!=present.shape:
            raise ValueError('Candidate feature and mask shapes disagree')
        if not bool(mask.any(-1).all()) or bool((mask & ~present).any()):
            raise ValueError('Every state needs a valid action; padding cannot be selectable')
        logits=self.actor_head(self.actor_encoder(x)).squeeze(-1).masked_fill(~mask,-torch.inf)
        encoded=self.critic_encoder(x)*present[...,None]
        pooled=encoded.sum(-2)/present.sum(-1,keepdim=True).clamp_min(1)
        return Categorical(logits=logits),self.critic_head(pooled).squeeze(-1)

    @torch.no_grad()
    def act(self,observation,generator=None,deterministic=False):
        distribution,value=self(*tensors(observation))
        action=distribution.probs.argmax(-1) if deterministic else torch.multinomial(distribution.probs,1,generator=generator).squeeze(-1)
        return int(action),float(distribution.log_prob(action)),float(value)

    @torch.no_grad()
    def value(self,observation):
        return float(self(*tensors(observation))[1])


def tensors(observation):
    result = (torch.as_tensor(observation['features'],dtype=torch.float32),
            torch.as_tensor(observation['mask'],dtype=torch.bool),
            torch.as_tensor(observation['present'],dtype=torch.bool))
    if 'environment_graph' in observation:
        result += (observation['environment_graph'],)
    return result


def smdp_advantages(transitions,gamma,gae_lambda):
    if not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError('Invalid discount or GAE lambda')
    advantages=np.zeros(len(transitions),np.float64)
    carry=0.
    for index in reversed(range(len(transitions))):
        item=transitions[index]
        duration=item['duration']
        if type(duration) is not int or duration<1:
            raise ValueError('Each macro transition needs a positive integer duration')
        terminal=bool(item['terminated']);boundary=terminal or bool(item['truncated'])
        discount=gamma**duration
        next_value=0. if terminal else item['next_value']
        delta=item['reward']+discount*next_value-item['value']
        carry=delta+discount*gae_lambda*(0. if boundary else carry)
        advantages[index]=carry
    values=np.asarray([t['value'] for t in transitions])
    returns=advantages+values
    if not np.isfinite(returns).all() or not np.isfinite(advantages).all():
        raise ValueError('Non-finite SMDP targets')
    return advantages.astype(np.float32),returns.astype(np.float32)


def update(model,optimizer,episodes,*,gamma=.999,gae_lambda=.95,epochs=4,
           clip=.2,entropy_coef=.01,value_coef=.5,target_kl=.02,max_kl=.04,
           microbatch_size=16):
    """Accumulate the full-batch PPO gradient with bounded graph activations.

    Each epoch still performs one optimizer step and one full-batch KL check.
    Only the forward/backward working set is split; advantages, clipping and
    rollback retain their full-batch semantics.
    """
    if not episodes or any(not episode for episode in episodes):
        raise ValueError('PPO requires nonempty episodes')
    if not 0<target_kl<=max_kl or not 0<clip<1 or epochs<1:
        raise ValueError('Invalid PPO configuration')
    if type(microbatch_size) is not int or microbatch_size < 1:
        raise ValueError('microbatch_size must be a positive integer')
    transitions=[item for episode in episodes for item in episode]
    x=torch.as_tensor(np.stack([t['features'] for t in transitions]),dtype=torch.float32)
    mask=torch.as_tensor(np.stack([t['mask'] for t in transitions]),dtype=torch.bool)
    present=torch.as_tensor(np.stack([t['present'] for t in transitions]),dtype=torch.bool)
    has_graph = ['environment_graph' in t for t in transitions]
    if any(has_graph) and not all(has_graph):
        raise ValueError('Mixed slow observation schemas in PPO batch')
    inputs = (x, mask, present)
    if all(has_graph):
        inputs += ([t['environment_graph'] for t in transitions],)
    chunks = [slice(begin, min(begin + microbatch_size, len(transitions)))
              for begin in range(0, len(transitions), microbatch_size)]

    def chunk_inputs(chunk):
        return tuple(item[chunk] for item in inputs)

    actions=torch.tensor([t['action'] for t in transitions])
    old_logp=torch.tensor([t['logp'] for t in transitions])
    old_values=torch.tensor([t['value'] for t in transitions])
    targets=[smdp_advantages(episode,gamma,gae_lambda) for episode in episodes]
    advantages=torch.from_numpy(np.concatenate([a for a,_ in targets]))
    returns=torch.from_numpy(np.concatenate([r for _,r in targets]))
    advantages=(advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-6)
    references=[];error=0.
    with torch.no_grad():
        for chunk in chunks:
            reference,values=model(*chunk_inputs(chunk))
            chunk_error=float((reference.log_prob(actions[chunk])-old_logp[chunk]).abs().max())
            if chunk_error>1e-4 or not np.isfinite(chunk_error):
                raise ValueError(f'Stale behaviour policy: log probability mismatch {chunk_error:g}')
            error=max(error,chunk_error)
            value_error=float((values-old_values[chunk]).abs().max())
            if not np.isfinite(value_error) or value_error>1e-4:
                raise ValueError('Stale behaviour critic')
            references.append(Categorical(logits=reference.logits.detach().clone()))
            del reference,values
    accepted=0;rejected=0;last_kl=0.;losses=[]
    actor=list(model.actor_encoder.parameters())+list(model.actor_head.parameters())
    critic=list(model.critic_encoder.parameters())+list(model.critic_head.parameters())
    for _ in range(epochs):
        before=copy.deepcopy(model.state_dict());before_optimizer=copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)
        epoch_losses=dict(actor=0.,critic=0.,entropy=0.)
        for chunk in chunks:
            distribution,values=model(*chunk_inputs(chunk))
            ratio=(distribution.log_prob(actions[chunk])-old_logp[chunk]).exp()
            actor_loss=-torch.minimum(ratio*advantages[chunk],ratio.clamp(1-clip,1+clip)*advantages[chunk]).mean()
            critic_loss=(values-returns[chunk]).square().mean()
            entropy=distribution.entropy().mean()
            weight=(chunk.stop-chunk.start)/len(transitions)
            loss=(actor_loss+value_coef*critic_loss-entropy_coef*entropy)*weight
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Non-finite PPO loss')
            loss.backward()
            for key,value in (('actor',actor_loss),('critic',critic_loss),('entropy',entropy)):
                epoch_losses[key]+=float(value.detach())*weight
            del distribution,values,ratio,actor_loss,critic_loss,entropy,loss,value
        torch.nn.utils.clip_grad_norm_(actor,.5,error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(critic,.5,error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            measured=float('inf')
            if all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                try:
                    measured=0.
                    for chunk,reference in zip(chunks,references):
                        updated,updated_value=model(*chunk_inputs(chunk))
                        if not bool(torch.isfinite(updated_value).all()):
                            measured=float('inf')
                            break
                        measured+=float(kl_divergence(reference,updated).sum())/len(transitions)
                        del updated,updated_value
                except ValueError:
                    # Distribution validation rejects non-finite logits before
                    # a KL can be measured; restore the entire optimizer step.
                    measured=float('inf')
        if not np.isfinite(measured) or measured>max_kl:
            model.load_state_dict(before);optimizer.load_state_dict(before_optimizer)
            rejected+=1;break
        last_kl=measured;accepted+=1
        losses.append(epoch_losses)
        if measured>target_kl:break
    return dict(transitions=len(transitions),fast_attempts=sum(t['duration'] for t in transitions),
                accepted_epochs=accepted,rejected_epochs=rejected,kl=last_kl,behaviour_logp_error=error,
                mean_return_target=float(returns.mean()),losses=losses,microbatch_size=microbatch_size)
