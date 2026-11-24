"""Warm-start the waypoint actor from legal actions of a public-observation route follower."""
import numpy as np
import torch

from vtemp5.vtemp5_env import VTemp5Env, rule_offset
from vtemp5.vtemp5_policy import policy_observation, relative_goal_bearing, wrap_angle


def collect_teacher(first_seed, episodes, settings, *, env_factory=VTemp5Env, terrain='local', evidence_reference=False):
    examples, records = [], []
    for index in range(episodes):
        # Paired runs have identical geometry and packet delays. Their first
        # probe chooses different sides, independently of the hidden crossing.
        pair = index // 2
        seed = first_seed + pair
        first_side = 1 if index % 2 == 0 else -1
        blockers = {'local': (1, -1, 0), 'detour': (0,), 'bridge': (1, -1)}[terrain]
        scene_block = blockers[pair % len(blockers)]
        env = env_factory(seed, blocked_side=scene_block, **settings)
        public = env.observe()
        record = dict(seed=seed, initial_probe_side=first_side, scene_blocked_side=scene_block,
                      steps=0, supervised_steps=0, refuted_steps=0, side_switches=0)
        while True:
            first_probe = public['detected'] and not public['locked_side']
            refuted = bool(public['locked_side'] and public['side_status'][public['locked_side']] == 'blocked')
            # Supply a tie-break solely to the teacher on the first ambiguous review.
            teacher = dict(public, locked_side=first_side) if first_probe else public
            angle = rule_offset(teacher)
            after, _, done, info = env.step(angle)
            if info['unauthorized_request']:
                raise ValueError('Teacher requested a prohibited side')
            if not first_probe:
                # Equally informed left/right initial probes cannot be unique targets.
                # All other labels are the actual issued, sampler-admissible angles.
                body = wrap_angle(info['executed_offset_rad'] + relative_goal_bearing(public))
                phase = 'refuted' if refuted else 'following'
                examples.append((policy_observation(public, evidence_reference=evidence_reference), body, phase))
                record['supervised_steps'] += 1
            record['steps'] += 1
            record['refuted_steps'] += int(refuted)
            record['side_switches'] += int(info['side_switch'])
            public = after
            if done:
                record['success'] = bool(info['success'])
                records.append(record)
                break
    if terrain != 'detour' and not any(row['side_switches'] for row in records):
        raise ValueError('No demonstrated switch after evidence; increase horizon or teacher episodes')
    return examples, records


@torch.inference_mode()
def collect_recovery(model, first_seed, episodes, settings, *, env_factory, evidence_reference, batch_size):
    """Label policy-visited closure states, including repeated requests to the disproved side."""
    examples, records = [], []
    for first in range(0, episodes, batch_size):
        active = []
        for i in range(first, min(first + batch_size, episodes)):
            seed = first_seed + i // 2
            blocked = (1, -1)[i % 2]
            env = env_factory(seed, blocked_side=blocked, **settings)
            record = dict(seed=seed, scene_blocked_side=blocked, refuted_labels=0, post_switch_labels=0,
                          wrong_requests=0, switched=False, visited_blocked_side=0)
            records.append(record)
            active.append((env, record))
        while active:
            public = [env.observe() for env, _ in active]
            obs = [policy_observation(state, evidence_reference=evidence_reference) for state in public]
            dist, _ = model(obs)
            _, body_angles = dist.action(True)
            pending = []
            for (env, row), state, encoded, body in zip(active, public, obs, body_angles.tolist()):
                side = state['locked_side']
                refuted = bool(side and state['side_status'][side] == 'blocked')
                continuing = row['switched'] and row['post_switch_labels'] < 10
                if refuted or continuing:
                    expert = rule_offset(state)
                    body_expert = wrap_angle(expert + relative_goal_bearing(state))
                    examples.append((encoded, body_expert, 'refuted' if refuted else 'following'))
                    if refuted:
                        row['visited_blocked_side'] = side
                        row['refuted_labels'] += 1
                    else:
                        row['post_switch_labels'] += 1
                # These rollouts always execute the current actor. A persistent wrong
                # command generates a new state and new command/drive-history tokens.
                offset = wrap_angle(body - relative_goal_bearing(state))
                row['wrong_requests'] += int(refuted and env.requested_side(offset) == side)
                _, _, done, info = env.step(offset)
                row['switched'] |= bool(info['side_switch'])
                if not done and row['refuted_labels'] < 12 and row['post_switch_labels'] < 10:
                    pending.append((env, row))
            active = pending
    if not any(row['refuted_labels'] for row in records):
        raise ValueError('No on-policy closure observed on recovery training seeds')
    return examples, records


def fit_teacher(model, examples, epochs, batch_size, lr, seed, *, refuted_angle_weight=0.):
    """Optimize the observed physical angles under the actor's masked circular density."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    refuted = np.flatnonzero([phase == 'refuted' for _, _, phase in examples])
    regular = np.flatnonzero([phase == 'following' for _, _, phase in examples])
    epoch_nll, refuted_mae_rad, epoch_objective = [], [], []
    model.train()
    for _ in range(epochs):
        # A refutation occurs only once per blocked episode. Give these decisions
        # a quarter of the teaching exposure instead of diluting them in tracking.
        extra_count = max(0, len(regular) // 3 - len(refuted)) if len(refuted) else 0
        repeated = rng.choice(refuted, size=extra_count, replace=True)
        order = rng.permutation(np.concatenate((regular, refuted, repeated)))
        total, total_nll, absolute_error, refuted_count = 0., 0., 0., 0
        for start in range(0, len(order), batch_size):
            batch = [examples[i] for i in order[start:start + batch_size]]
            dist, _ = model([obs for obs, _, _ in batch])
            angles = torch.tensor([angle for _, angle, _ in batch], dtype=dist.location.dtype)
            turns = torch.ones_like(angles, dtype=torch.bool)
            nll = -dist.log_prob(turns, angles).mean()
            loss = nll
            # Use the mode before legal-arc conditioning; changing concentration alone
            # cannot improve the teacher angle error at a confirmed closure.
            mask = torch.tensor([phase == 'refuted' for _, _, phase in batch], dtype=torch.bool)
            if mask.any():
                delta = dist.location[mask] - angles[mask]
                error = torch.atan2(delta.sin(), delta.cos())
                absolute_error += error.detach().abs().sum().item()
                refuted_count += len(error)
                if refuted_angle_weight:
                    huber = torch.nn.functional.huber_loss(error, torch.zeros_like(error),
                                                           reduction='sum', delta=1.)
                    loss = loss + refuted_angle_weight * huber / len(batch)
            total_nll += nll.detach().item() * len(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite teacher likelihood')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        epoch_nll.append(total_nll / len(order))
        epoch_objective.append(total / len(order))
        refuted_mae_rad.append(absolute_error / refuted_count if refuted_count else None)
    return dict(epoch_nll=epoch_nll, epoch_objective=epoch_objective, refuted_mae_rad=refuted_mae_rad)
