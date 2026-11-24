"""Matched left/right evidence intervention for a frozen vtemp5 waypoint actor."""
import argparse
import hashlib
import json
import math
from functools import partial
from pathlib import Path
import sys

import torch

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))
from vtemp4.vtemp4_body_graph import GoalResidualBodyFramePolicy
from vtemp4.vtemp4_ring_contract import RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT
from vtemp5.vtemp5_env import VTemp5Env, rule_offset
from vtemp5.vtemp5_large import VTemp5LargeObstacleEnv
from vtemp5.vtemp5_world_model import LocalWorldModel
from vtemp5.vtemp5_policy import REVIEW_S, policy_observation, relative_goal_bearing, wrap_angle
from vtemp2_3.hierarchical_sources import atomic_json


def matched_prefix(seed, side, factory, large):
    """Steer along the near bank until both cars receive opposite crossing reports."""
    pair = [factory(seed, blocked_side=b) for b in (-side, side)]
    obs = pair[0].observe()
    prefix = []
    while not any(e.done for e in pair):
        if obs != pair[1].observe():
            return pair, prefix, 'different public observations before report'
        # The global x axis is the initial vehicle-to-goal direction on these maps.
        # A grazing survey can see the crossing before the car drives onto it.
        x, y = obs['position']
        gx, gy = obs['goal']
        if large:
            offset = rule_offset(dict(obs, locked_side=obs['locked_side'] or side))
        else:
            offset = wrap_angle(side * math.radians(72.) - math.atan2(gy-y, gx-x)) if obs['detected'] else 0.
        states = [env.step(offset)[0] for env in pair]
        obs, other = states
        prefix.append(dict(step=obs['elapsed_steps'], offset_rad=offset, position=obs['position']))
        if obs['position'] != other['position'] or obs['history'] != other['history']:
            return pair, prefix, 'different physical trajectories before handoff'
        results = [state['side_status'][side] for state in states]
        if results == ['open', 'blocked'] and all(state['locked_side'] == side for state in states):
            return pair, prefix, None
        if results != ['unknown', 'unknown']:
            return pair, prefix, 'crossing reports not paired'
    return pair, prefix, 'episode ended before crossing report'


@torch.inference_mode()
def finish(model, cases, *, evidence_reference=False):
    """Run the deterministic production action plus an unmasked preference diagnostic."""
    active = list(cases)
    while active:
        public = [env.observe() for env, _ in active]
        distribution, _ = model([policy_observation(obs, evidence_reference=evidence_reference) for obs in public])
        _, actions = distribution.action(True, None)
        next_active = []
        for (env, row), obs, angle, mode in zip(active, public, actions.tolist(), distribution.location.tolist()):
            goal_bearing = relative_goal_bearing(obs)
            offset = wrap_angle(angle - goal_bearing)
            if 'first_action' not in row:
                row['first_action'] = dict(
                    step=obs['elapsed_steps'] + 1, requested_side=env.requested_side(offset),
                    unmasked_requested_side=env.requested_side(wrap_angle(mode - goal_bearing)),
                    offset_rad=offset)
            after, _, done, info = env.step(offset)
            if info['side_switch']:
                row['switch_steps'].append(after['elapsed_steps'])
            row['unauthorized_switches'] += int(info['unauthorized_switch'])
            if done:
                row.update(success=bool(info['success']), detected=after['detected'],
                           final_step=after['elapsed_steps'], final_side=after['locked_side'])
            else:
                next_active.append((env, row))
        active = next_active


def evaluate(checkpoint, output, seed, count, calibration, world_model):
    if count < 1:
        raise ValueError('Expected at least one holdout seed')
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['format'] != RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT or saved['environment'] != 'vtemp5':
        raise ValueError('Expected a vtemp5 goal-relative checkpoint')
    seeds = range(seed, seed + count)
    excluded_ranges = [saved['train_seed_range'], *saved.get('prior_train_seed_ranges', []),
                       *saved.get('eval_seed_ranges', [])]
    manifest_path = checkpoint.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if set(manifest.get('eval_seeds', [])) & set(seeds):
        raise ValueError('Use fresh seeds outside the original evaluation set')
    settings = dict(manifest.get('environment', {}))
    parameters = json.loads(calibration.read_text())
    if parameters['schema'] != 'vtemp5_motion_calibration_v1' or parameters['parameters']['review_interval_s'] != REVIEW_S:
        raise ValueError('Expected vtemp5 calibration at the policy review interval')
    settings.update(step_m=parameters['parameters']['moving_interval_displacement_m'],
                    wait_probability=parameters['parameters']['stationary_interval_fraction'])
    terrain = saved.get('terrain', 'local')
    if terrain not in ('local', 'bridge'):
        raise ValueError('This test needs a bridge or local pilot policy')
    large = terrain == 'bridge'
    wm_digest = None
    if large:
        stored = manifest.get('world_model') or {}
        reference = world_model or (Path(stored['path']) if 'path' in stored else None)
        if reference is None:
            raise ValueError('Provide the frozen bridge predictor via --world-model')
        wm_digest = hashlib.sha256(reference.read_bytes()).hexdigest()
        if stored.get('sha256', wm_digest) != wm_digest:
            raise ValueError('World model differs from the one used for actor training')
        predictor = LocalWorldModel.load(reference)
        excluded_ranges += predictor.metadata['seen_seed_ranges']
        factory = partial(VTemp5LargeObstacleEnv, terrain='bridge', predictor=predictor, **settings)
    else:
        factory = partial(VTemp5Env, **settings)
    if any(max(seed, low) < min(seed + count, high) for low, high in excluded_ranges):
        raise ValueError('Use fresh seeds outside all ancestor training and evaluation ranges')
    model = GoalResidualBodyFramePolicy().eval()
    model.load_state_dict(saved['model'], strict=True)

    pairs, free, cases = [], [], []
    for s in seeds:
        for side in (1, -1):
            envs, prefix, error = matched_prefix(s, side, factory, large)
            row = dict(seed=s, starting_side=side, prefix=prefix, handoff_error=error)
            pairs.append(row)
            if error:
                continue
            for env, name in zip(envs, ('open', 'blocked')):
                arm = dict(switch_steps=[], unauthorized_switches=0)
                row[name] = arm
                cases.append((env, arm))
        arm = dict(seed=s, switch_steps=[], unauthorized_switches=0)
        free.append(arm)
        cases.append((factory(s), arm))
    for i in range(0, len(cases), 32):
        finish(model, cases[i:i+32], evidence_reference=saved.get('evidence_reference', False))
    matched = [p for p in pairs if not p['handoff_error']]
    for p in matched:
        side = p['starting_side']
        a, b = p['open'], p['blocked']
        p['first_action_correct_in_both'] = (a['first_action']['requested_side'] == side and
                                            b['first_action']['requested_side'] == -side)
        p['first_unmasked_correct_in_both'] = (a['first_action']['unmasked_requested_side'] == side and
                                               b['first_action']['unmasked_requested_side'] == -side)
        p['blocked_switch_delay_steps'] = (b['switch_steps'][0] - len(p['prefix'])) if b['switch_steps'] else None
    delays = [p['blocked_switch_delay_steps'] for p in matched if p['blocked_switch_delay_steps'] is not None]
    summary = dict(
        pairs=len(pairs), matched_pairs=len(matched),
        first_action_correct_in_both=sum(p['first_action_correct_in_both'] for p in matched),
        first_unmasked_correct_in_both=sum(p['first_unmasked_correct_in_both'] for p in matched),
        open_stayed=sum(not p['open']['switch_steps'] for p in matched),
        blocked_switched=len(delays), blocked_never_switched=len(matched)-len(delays),
        mean_blocked_switch_delay_steps=sum(delays)/len(delays) if delays else None,
        open_passed=sum(p['open']['success'] for p in matched),
        blocked_passed=sum(p['blocked']['success'] for p in matched),
        both_passed=sum(p['open']['success'] and p['blocked']['success'] for p in matched),
        unauthorized_switches=sum(p[k]['unauthorized_switches'] for p in matched for k in ('open', 'blocked')),
        free_episodes=count, free_passed=sum(p['success'] for p in free),
        free_passed_without_detection=sum(p['success'] and not p['detected'] for p in free))
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, dict(schema='vtemp5_matched_evidence_v1', checkpoint=str(checkpoint.resolve()),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
                settings=settings, terrain=terrain, world_model_sha256=wm_digest, seed_range=[seed, seed+count], summary=summary, pairs=pairs, free=free))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output JSON file')
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--world-model', type=Path, help='Frozen bridge predictor; defaults to actor manifest')
    parser.add_argument('--seed', type=int, default=3000000)
    parser.add_argument('--seeds', type=int, default=32)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output, args.seed, args.seeds, args.calibration, args.world_model)
