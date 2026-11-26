"""Collect on-policy vehicle branches and score the executed trajectory after simulation."""
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from run_v91_lake_test import jsonable, sha


_models = None
_actor_sha = None


def init_worker(weights, cpus=None):
    global _models, _actor_sha
    if cpus is not None:
        os.sched_setaffinity(0, cpus)
    if set(weights) != {'uav', 'world', 'actor'}:
        raise ValueError('Expected a frozen UAV, world model and initial vehicle policy')
    for name, weight in weights.items():
        if sha(weight['path']) != weight['sha256']:
            raise ValueError(f'{name} checkpoint digest differs from the evaluation protocol')
    # The UAV checkpoint binds shared simulator imports to its archived runtime.
    from v91_extent_collect import load_frozen
    uav_path = Path(weights['uav']['path'])
    uav, config = load_frozen(uav_path.parent, uav_path)
    from v91_frozen_actor_inputs import load_actor
    from v91_lake_metrics import load_model
    _models = (uav, config, load_model(weights['world']['path']), load_actor(weights['actor']['path']))
    _actor_sha = weights['actor']['sha256']


def _decisions(rollout, sampled):
    reviews = rollout['reviews']
    if len(sampled) != len(reviews) or not reviews or reviews[0]['step'] != 0:
        raise ValueError('Missing policy decisions at the start of a trajectory')
    executed = []
    for i, (row, review) in enumerate(zip(sampled, reviews)):
        step = row['step']
        end = reviews[i+1]['step'] if i + 1 < len(reviews) else rollout['last_step']
        duration = end - step
        if step != review['step'] or duration != review['applied_target_frames'] or not 0 <= duration <= 30:
            raise ValueError('Vehicle frames are not attributed to their sampled decision')
        if duration == 0:
            # Native safety may end the episode before the next prepared tick.
            # This proposal never reached physics. Credit the last executed action.
            if i + 1 != len(reviews) or rollout['termination'] not in ('safety_termination', 'ground_safety_violation'):
                raise ValueError('Unexecuted policy review outside a safety stop')
            continue
        allowed, action = review['candidate_count'], row['action']
        obs = row['observation']
        if (len(obs['branch_angles']) != max(1, allowed) or len(obs['branch_prior']) != len(obs['branch_angles'])
                or not 0 <= action < len(obs['branch_angles']) or action != review['action_index']
                or row['branch'] != review['selected_branch'] or row['policy_controlled'] != (allowed > 1)
                or not np.isfinite([row['angle_rad'], row['value'], row['logp']]).all()
                or not math.isclose(row['angle_rad'], float(obs['branch_angles'][action]), abs_tol=1e-7)
                or (not row['policy_controlled'] and row['logp'] != 0.)):
            raise ValueError('Sampled branch does not match its executable candidate set')
        executed.append(dict(row, duration=duration))
    if not executed or sum(row['duration'] for row in executed) != rollout['last_step']:
        raise ValueError('Expected positive physical time completely assigned to actions')
    return executed


def _settle(rows, potentials, route, termination, budget):
    if potentials.shape != (len(rows) + 1,) or not np.isfinite(potentials).all():
        raise ValueError('Invalid bridge potential at a review boundary')
    bonus = {'vehicle_arrival': 10., 'frame_limit': -2., 'safety_termination': -6., 'ground_safety_violation': -6.}
    for i, row in enumerate(rows):
        t, end = row['step'], row['step'] + row['duration']
        reward = (potentials[i+1] - potentials[i]) / 50. - row['duration'] / budget
        for key, amount in [('first_deck_step', 1.), ('first_cross_step', 5.)]:
            event = route.get(key)
            if event is not None and t < event <= end:
                reward += amount
        final = i + 1 == len(rows)
        # All collection calls run to an environment terminal, including expiry of
        # the mission's prescribed time budget. No collector cutoff is imposed.
        row.update(reward=float(reward + (bonus[termination] if final else 0.)),
                   terminated=final, truncated=False, next_value=0. if final else rows[i+1]['value'])


def _write_record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, default=jsonable, allow_nan=False, separators=(',', ':'))
    with NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
        f.write(data + '\n')
    os.replace(f.name, path)


def collect(scene, checkpoint_path, deterministic, sampling_seed, record_path=None):
    global _models, _actor_sha
    if _models is None:
        raise RuntimeError('Call init_worker before collecting')
    if scene['kind'] not in ('bridge', 'circle') or (scene['kind'] == 'circle' and not deterministic):
        raise ValueError('Bridge rollouts train the actor; circle rollouts measure retention')
    from v91_frozen_actor_inputs import load_actor
    from v91_guided_drive import run_km_actor_arm
    from v91_trainable_guidance import TrainableGuidance

    digest = sha(checkpoint_path)
    if digest != _actor_sha:
        actor = load_actor(checkpoint_path)
        if sha(checkpoint_path) != digest:
            raise ValueError('Vehicle checkpoint changed while loading')
        _models = (*_models[:3], actor)
        _actor_sha = digest
    guidance = TrainableGuidance(deterministic=deterministic, seed=sampling_seed)
    r = run_km_actor_arm(scene, *_models, arm='predicted', guidance=guidance)
    path = np.asarray(r['car_path'], float)
    steps, budget, termination = r['last_step'], scene['max_steps'], r['termination']
    if (steps < 1 or steps > budget or r['mission_budget_s'] != budget or not r['is_full_budget']
            or (termination == 'frame_limit' and steps != budget) or path.shape != (steps + 1, 2)
            or not np.isfinite(path).all() or not np.allclose(path[0], scene['start'], rtol=0, atol=1e-4)):
        raise ValueError('Incomplete physical mission trajectory')
    remaining = float(np.linalg.norm(path[-1] - scene['goal']))
    arrived = termination == 'vehicle_arrival'
    if (remaining <= 2.) != arrived or not math.isclose(remaining, r['goal_remaining_m'], abs_tol=1e-5):
        raise ValueError('Arrival disagrees with recorded final pose')
    rows = _decisions(r, guidance.transitions)
    boundaries = [row['step'] for row in rows] + [steps]
    if scene['kind'] == 'bridge':
        from v91_bridge_curriculum import bridge_potential, bridge_score
        route = bridge_score(scene, path)
        if route['goal_reached'] != arrived:
            raise ValueError('Bridge scorer disagrees with physical arrival')
        potential = bridge_potential(scene, path[boundaries])
        _settle(rows, potential, route, termination, budget)
        reward = sum(row['reward'] for row in rows)
    else:
        from v91_km_cases import km_route_metrics
        route = km_route_metrics(scene, path)
        reward = None
    summary = dict(case_id=scene['case_id'], kind=scene['kind'], stage=scene.get('curriculum_stage', 'kilometre'),
                   fleet_seed=scene['fleet_seed'], sampling_seed=int(sampling_seed), checkpoint_sha256=digest,
                   arm='predicted', termination=termination, entered_deck=route.get('entered_deck'),
                   crossed_bridge=route['crossed_bridge'], goal_reached=arrived, ground_collisions=r['ground_collisions'],
                   ground_safety_violations=r['ground_safety_violations'], last_step=steps, mission_budget_s=budget,
                   travelled_m=r['travelled_m'], remaining_m=remaining, reward_sum=reward,
                   policy_transitions=sum(int(row['policy_controlled']) for row in rows), transitions=len(rows),
                   uav_packets_received=r['uav_packets_received'], route=route)
    if record_path is not None:
        keys = ('step', 'mode', 'selected_branch', 'action_index', 'target_xy', 'survey_target_xy',
                'candidate_count', 'action_probabilities', 'applied_target_frames', 'received_packets')
        decisions = [{k: v[k] for k in keys} for v in r['reviews']]
        _write_record(record_path, dict(summary=summary, car_path=r['car_path'], decisions=decisions))
    return summary, [] if deterministic else rows
