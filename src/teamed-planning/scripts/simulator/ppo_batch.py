"""On-policy multi-episode PPO with independent GAE and reversible KL guards."""
import copy

import numpy as np
import torch
from torch.distributions import kl_divergence


def episode_advantages(rewards, values, discount, gae_lambda):
    """A complete episode has its own terminal; never bootstrap into another."""
    advantages=torch.zeros_like(rewards)
    last=torch.zeros((),dtype=rewards.dtype,device=rewards.device)
    for t in reversed(range(len(rewards))):
        next_value=values[t+1] if t+1<len(values) else 0.
        last=rewards[t]+discount*next_value-values[t]+discount*gae_lambda*last
        advantages[t]=last
    return advantages,advantages+values


def clip_gradients(model, max_norm=.5):
    """Clip actor and critic gradients separately; returns pre-clip norms.

    A shared norm lets ±100 terminal value errors (critic norm ~10) shrink the
    policy step ~20x; with Adam moments from earlier unclipped steps this
    froze the policy (KL ~1e-5) for the rest of a stage.
    """
    critic=list(model.critic.parameters())
    ids={id(p) for p in critic}
    actor=[p for p in model.parameters() if id(p) not in ids]
    return tuple(float(torch.nn.utils.clip_grad_norm_(params,max_norm,error_if_nonfinite=True))
                 for params in (actor,critic))


VALUE_STATISTICS_RATE=.1  # per batch: return statistics remember ~10 batches


def update_value_statistics(model, returns, rate=VALUE_STATISTICS_RATE):
    """PopArt: move the debiased EMA return mean/std and rescale the critic
    head so every unnormalized value prediction stays unchanged."""
    with torch.no_grad():
        returns=returns.double()
        moments=model.value_moments.double()*(1-rate)+rate*torch.stack(
            (torch.ones((),dtype=torch.float64),returns.mean(),returns.square().mean()))
        mean=moments[1]/moments[0]
        std=(moments[2]/moments[0]-mean**2).clamp_min(1e-4).sqrt()
        old_mean,old_std=model.value_mean.double(),model.value_std.double()
        head=model.critic[-1]
        head.weight.copy_(head.weight.double()*old_std/std)
        head.bias.copy_((head.bias.double()*old_std+old_mean-mean)/std)
        model.value_moments.copy_(moments);model.value_mean.copy_(mean);model.value_std.copy_(std)


def update_batch(model, optimizer, rollouts, epochs=4, entropy_coef=.01,
                 discount=.99, gae_lambda=.95, target_kl=.01, max_kl=.02,
                 max_backtracks=4, critic_epochs=0, critic_minibatch_size=512,
                 value_normalization=False, critic_seed=0):
    """All episodes must come from this exact frozen policy.

    KL is analytic in action-distribution space, averaged over agents within
    each state and over sampled states. It does not bound unseen states or
    replace CBF. Every rejected optimizer step restores parameters AND Adam
    moments/step counters before a smaller-step retry.

    critic_epochs>0: the guarded steps optimize the actor only; afterwards the
    critic alone takes shuffled minibatch steps on the fixed returns (PopArt-
    normalized when requested). Critic steps cannot change the policy or its KL.
    """
    if not (0<target_kl<=max_kl and np.isfinite(max_kl)):
        raise ValueError('Require finite 0 < target_kl <= max_kl')
    if type(epochs) is not int or epochs<1 or type(max_backtracks) is not int or max_backtracks<0:
        raise ValueError('Invalid PPO epochs/backtracking limit')
    if not rollouts or any(not rollout for rollout in rollouts):
        raise ValueError('A PPO batch requires nonempty complete episodes')
    if (type(critic_epochs) is not int or critic_epochs<0 or type(critic_minibatch_size) is not int
            or critic_minibatch_size<1 or (value_normalization and not critic_epochs)):
        raise ValueError('Invalid separate critic schedule')
    if critic_epochs and not value_normalization and (float(model.value_mean)!=0. or float(model.value_std)!=1.):
        raise ValueError('Normalized critic statistics require value_normalization')
    batches=[]
    with torch.no_grad():
        for rollout in rollouts:
            x=torch.tensor(np.array([r[0] for r in rollout]),dtype=torch.float32)
            mask=torch.tensor(np.array([r[1] for r in rollout]),dtype=torch.bool)
            actions=torch.tensor(np.array([r[2] for r in rollout]))
            old_logp=torch.tensor(np.array([r[3] for r in rollout]),dtype=torch.float32)
            rewards=torch.tensor([r[4] for r in rollout],dtype=torch.float32)
            states=None
            if model.global_features:
                if len(rollout[0])<6:raise ValueError('Missing centralized critic state')
                states=torch.tensor(np.array([r[5] for r in rollout]),dtype=torch.float32)
            reference=copy.deepcopy(model.distribution(x,mask))
            error=float((reference.log_prob(actions)-old_logp).abs().max())
            if not np.isfinite(error) or error>1e-4:
                raise ValueError(f'Stale/mismatched synchronous rollout: log probability error {error:g}')
            values=model.value(x,mask,states)
            adv,returns=episode_advantages(rewards,values,discount,gae_lambda)
            batches.append(dict(x=x,mask=mask,actions=actions,old_logp=old_logp,
                states=states,reference=reference,adv=adv,returns=returns,logp_error=error,
                old_values=values,pooled=model.critic_input(x,mask,states) if critic_epochs else None))
        all_adv=torch.cat([b['adv'] for b in batches])
        mean,std=all_adv.mean(),all_adv.std(unbiased=False).clamp_min(1e-6)
        for b in batches:b['adv']=(b['adv']-mean)/std
    steps=sum(len(b['adv']) for b in batches)

    def losses(b):
        dist=model.distribution(b['x'],b['mask'])
        ratio=(dist.log_prob(b['actions'])-b['old_logp']).exp()
        actor=-torch.minimum(ratio*b['adv'][:,None],ratio.clamp(.8,1.2)*b['adv'][:,None]).mean()
        if critic_epochs:
            return actor-entropy_coef*dist.entropy().mean()
        critic=(model.value(b['x'],b['mask'],b['states'])-b['returns']).square().mean()
        return actor+.5*critic-entropy_coef*dist.entropy().mean()

    def exact_kl():
        with torch.no_grad():
            return sum(float(kl_divergence(b['reference'],model.distribution(b['x'],b['mask'])).mean())
                       *len(b['adv'])/steps for b in batches)

    accepted=0;rejected=0;attempts=0;nonfinite_attempts=0;early_stop=False
    max_attempt_kl=0.;minimum_lr_scale=1.;gradient_norms=[];actor_norms=[];critic_norms=[]
    for epoch in range(epochs):
        model_before=copy.deepcopy(model.state_dict())
        optimizer_before=copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad()
        for b in batches:
            loss=losses(b)*len(b['adv'])/steps
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite PPO batch loss')
            loss.backward()
        actor_norm,critic_norm=clip_gradients(model)
        original_lrs=[g['lr'] for g in optimizer.param_groups]
        committed=False
        for retry in range(max_backtracks+1):
            if retry:
                model.load_state_dict(model_before)
                optimizer.load_state_dict(optimizer_before)
            scale=.5**retry
            for group,lr in zip(optimizer.param_groups,original_lrs):group['lr']=lr*scale
            optimizer.step();attempts+=1
            finite=all(bool(torch.isfinite(p).all()) for p in model.parameters())
            measured=exact_kl() if finite else float('inf')
            if np.isfinite(measured):max_attempt_kl=max(max_attempt_kl,measured)
            else:nonfinite_attempts+=1
            if np.isfinite(measured) and measured<=max_kl:
                for group,lr in zip(optimizer.param_groups,original_lrs):group['lr']=lr
                minimum_lr_scale=min(minimum_lr_scale,scale)
                gradient_norms.append(float(np.hypot(actor_norm,critic_norm)))
                actor_norms.append(actor_norm);critic_norms.append(critic_norm)
                accepted+=1;committed=True
                break
            rejected+=1
        if not committed:
            model.load_state_dict(model_before)
            optimizer.load_state_dict(optimizer_before)
            early_stop=True
            break
        if measured>=target_kl:
            early_stop=True
            break
    critic_steps=0;critic_phase_norms=[]
    if critic_epochs:
        returns=torch.cat([b['returns'] for b in batches]);inputs=torch.cat([b['pooled'] for b in batches])
        if value_normalization:
            update_value_statistics(model,returns)
        target=((returns-model.value_mean)/model.value_std).detach()
        critic=list(model.critic.parameters())
        generator=torch.Generator().manual_seed(int(critic_seed))
        for _ in range(critic_epochs):
            for index in torch.randperm(len(target),generator=generator).split(critic_minibatch_size):
                # Actor gradients stay None, so Adam leaves actor parameters/moments alone.
                optimizer.zero_grad(set_to_none=True)
                loss=(model.critic(inputs[index]).squeeze(-1)-target[index]).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite critic loss')
                loss.backward()
                critic_phase_norms.append(float(torch.nn.utils.clip_grad_norm_(critic,.5,error_if_nonfinite=True)))
                optimizer.step();critic_steps+=1
        optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        metrics=dict(policy_loss=0.,value_loss=0.,entropy=0.,approx_kl=0.,clip_fraction=0.,
                     active_decision_fraction=0.)
        final_values=[];returns=[]
        continuous=getattr(model,'action_mode',None)=='velocity'
        normalized_entropy=0.
        for b in batches:
            weight=len(b['adv'])/steps
            dist=model.distribution(b['x'],b['mask'])
            log_ratio=dist.log_prob(b['actions'])-b['old_logp'];ratio=log_ratio.exp()
            value=model.value(b['x'],b['mask'],b['states'])
            metrics['policy_loss']+=weight*float(-torch.minimum(ratio*b['adv'][:,None],
                ratio.clamp(.8,1.2)*b['adv'][:,None]).mean())
            metrics['value_loss']+=weight*float((value-b['returns']).square().mean())
            metrics['entropy']+=weight*float(dist.entropy().mean())
            metrics['approx_kl']+=weight*float(((ratio-1)-log_ratio).mean())
            metrics['clip_fraction']+=weight*float(((ratio-1).abs()>.2).float().mean())
            counts=b['mask'].sum(-1)
            active=torch.ones_like(counts,dtype=torch.bool) if continuous else counts>1
            metrics['active_decision_fraction']+=weight*float(active.float().mean())
            if not continuous and active.any():
                normalized_entropy+=weight*float((dist.entropy()/counts.float().log().clamp_min(1e-8))[active].mean())
            final_values.append(value);returns.append(b['returns'])
        final_values=torch.cat(final_values);returns=torch.cat(returns)
        old_values=torch.cat([b['old_values'] for b in batches])
        metrics.update(exact_kl=exact_kl(),max_attempt_kl=max_attempt_kl,
            epochs_completed=accepted,optimizer_attempts=attempts,kl_rejected_attempts=rejected,
            kl_early_stopped=float(early_stop),nonfinite_attempts=nonfinite_attempts,
            minimum_lr_scale=minimum_lr_scale,
            on_policy_logp_max_error=max(b['logp_error'] for b in batches),
            gradient_norm=float(np.mean(gradient_norms)) if gradient_norms else 0.,
            actor_gradient_norm=float(np.mean(actor_norms)) if actor_norms else 0.,
            # Separate critic: the pre-clip norm of its own minibatch steps.
            critic_gradient_norm=float(np.mean(critic_phase_norms if critic_epochs else critic_norms))
                if (critic_phase_norms if critic_epochs else critic_norms) else 0.,
            explained_variance=float(1.-(returns-final_values).var(unbiased=False)/returns.var(unbiased=False).clamp_min(1e-8)),
            # Prediction by the critic before it saw this batch.
            explained_variance_pre=float(1.-(returns-old_values).var(unbiased=False)/returns.var(unbiased=False).clamp_min(1e-8)),
            critic_updates=critic_steps,value_mean=float(getattr(model,'value_mean',0.)),
            value_std=float(getattr(model,'value_std',1.)),
            return_mean=float(returns.mean()),return_std=float(returns.std(unbiased=False)),
            return_min=float(returns.min()),return_max=float(returns.max()),batch_episodes=len(batches),batch_steps=steps)
        if continuous:metrics['action_std']=float(model.log_std.clamp(-5.,1.).exp().mean())
        else:metrics['normalized_entropy']=normalized_entropy
    if not all(np.isfinite(v) for v in metrics.values()):
        raise FloatingPointError('Non-finite PPO batch diagnostics')
    return metrics['policy_loss']+.5*metrics['value_loss']-entropy_coef*metrics['entropy'],metrics
