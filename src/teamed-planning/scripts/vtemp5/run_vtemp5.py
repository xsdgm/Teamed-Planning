"""Train the existing vehicle waypoint actor in the vtemp5 lightweight environment."""
import argparse
import hashlib
import json
import math
from functools import partial
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))

from vtemp4.vtemp4_policy import update_policy
from vtemp4.vtemp4_body_graph import GoalResidualBodyFramePolicy
from vtemp4.vtemp4_ring_contract import RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT
from vtemp5.vtemp5_env import VTemp5Env, rule_offset
from vtemp5.vtemp5_demonstrations import collect_recovery, collect_teacher, fit_teacher
from vtemp5.vtemp5_policy import REVIEW_S, policy_observation, relative_goal_bearing, wrap_angle
from vtemp2_3.hierarchical_sources import atomic_checkpoint, atomic_json


def new_record(seed):
    return dict(seed=seed, decisions=0, success=False, total_reward=0., progress_m=0., side_switches=0,
                unauthorized_switches=0, unauthorized_requests=0, locked_decisions=0,
                unmasked_mode_opposite=0, projected_actions=0, first_locked_blocked_evidence_step=None,
                post_evidence_delays_steps=[])


def record_mission_step(row, before, after, info, done):
    if 'obstacles' not in row:
        row['initial_distance_m'] = info['goal_distance_m']
        row['start_xy'], row['goal_xy'] = before['position'], before['goal']
        row['arrival_radius_m'] = info['arrival_radius_m']
        row['obstacles_passed'] = 0
        row['obstacles'] = [dict(id=i+1, first_report_step=None, first_side=None, reports=[], closure_step=None,
                                 switch_steps=[], passed_step=None) for i in range(info['obstacles_total'])]
    step = after['elapsed_steps']
    obstacle_id = before['obstacle_id'] or after['obstacle_id']
    if obstacle_id is not None:
        obstacle = row['obstacles'][obstacle_id-1]
        if obstacle['first_report_step'] is None:
            obstacle['first_report_step'] = step if not before['detected'] else before['elapsed_steps']
        if info['locked_side'] and obstacle['first_side'] is None:
            obstacle['first_side'] = info['locked_side']
        for kind, side in info['delivered']:
            report = dict(step=step, kind=kind, side=side)
            if kind == 'crossing' and after['obstacle_id'] == obstacle_id:
                report['verdict'] = after['side_status'][side]
            obstacle['reports'].append(report)
        if info['blocked_evidence_step'] is not None:
            obstacle['closure_step'] = info['blocked_evidence_step']
        if info['side_switch']:
            obstacle['switch_steps'].append(step)
    for i in range(row['obstacles_passed'], info['obstacles_passed']):
        row['obstacles'][i]['passed_step'] = step
    row['obstacles_passed'] = info['obstacles_passed']
    row['final_xy'], row['remaining_m'], row['travelled_m'] = after['position'], info['remaining_m'], info['travelled_m']
    if done:
        row['end_reason'] = 'goal' if info['success'] else 'time_budget'
        row['simulated_s'] = step * REVIEW_S


@torch.no_grad()
def collect(model, seeds, settings, kind, generator=None, *, env_factory=VTemp5Env, evidence_reference=False):
    """All sampler angles and behaviour log probabilities are in the vtemp4 body frame."""
    environments = [env_factory(seed, **settings) for seed in seeds]
    active = list(range(len(seeds)))
    public = [env.observe() for env in environments]
    records = [new_record(seed) for seed in seeds]
    rollouts = [[] for seed in seeds]
    while active:
        if kind in ('sampled', 'deterministic'):
            observations = [policy_observation(public[i], evidence_reference=evidence_reference) for i in active]
            dist, values = model(observations)
            turns, angles = dist.action(kind == 'deterministic', generator)
            logps = dist.log_prob(turns, angles)
            # location is the physical preferred angle before legal-arc conditioning.
            modes = dist.location.tolist()
        remaining = []
        for batch_index, i in enumerate(active):
            env, before, row = environments[i], public[i], records[i]
            locked = before['locked_side']
            goal_bearing = relative_goal_bearing(before)
            if kind in ('sampled', 'deterministic'):
                body_angle = angles[batch_index].item()
                offset = wrap_angle(body_angle - goal_bearing)
                free_offset = wrap_angle(modes[batch_index] - goal_bearing)
            else:
                offset = 0. if kind == 'straight' else rule_offset(before)
                free_offset = offset
            if locked and before['side_status'][locked] != 'blocked':
                row['locked_decisions'] += 1
                row['unmasked_mode_opposite'] += int(env.requested_side(free_offset) == -locked)
            after, reward, done, info = env.step(offset)
            row['decisions'] += 1
            row['total_reward'] += reward
            row['progress_m'] += info['progress_m']
            row['side_switches'] += int(info['side_switch'])
            row['unauthorized_switches'] += int(info['unauthorized_switch'])
            row['unauthorized_requests'] += int(info['unauthorized_request'])
            row['projected_actions'] += int(abs(wrap_angle(offset - info['executed_offset_rad'])) > 1.e-5)
            row['success'] = bool(info['success'])
            if locked and after['side_status'][locked] == 'blocked' and row['first_locked_blocked_evidence_step'] is None:
                row['first_locked_blocked_evidence_step'] = info['blocked_evidence_step']
            if info['side_switch'] and info['blocked_evidence_step'] is not None:
                row['post_evidence_delays_steps'].append(after['elapsed_steps'] - info['blocked_evidence_step'])
            if 'obstacles_total' in info:
                record_mission_step(row, before, after, info, done)
            if kind == 'sampled':
                rollout = rollouts[i]
                value = values[batch_index].item()
                if rollout:
                    rollout[-1]['next_value'] = value
                rollout.append(dict(observation=observations[batch_index], turn=True, angle_rad=body_angle,
                                    logp=logps[batch_index].item(), value=value, next_value=0.,
                                    duration=REVIEW_S, reward=float(reward), terminated=done, truncated=False))
            public[i] = after
            if not done:
                remaining.append(i)
        active = remaining
    return records, rollouts


def summarize(rows):
    steps = sum(r['decisions'] for r in rows)
    locked = sum(r['locked_decisions'] for r in rows)
    delays = [t for row in rows for t in row['post_evidence_delays_steps']]
    result = dict(episodes=len(rows), decisions=steps, arrivals=sum(row['success'] for row in rows),
                success_rate=float(np.mean([r['success'] for r in rows])),
                mean_return=float(np.mean([r['total_reward'] for r in rows])),
                mean_route_progress_m=float(np.mean([r['progress_m'] for r in rows])),
                locked_decisions=locked, unmasked_mode_opposite=sum(r['unmasked_mode_opposite'] for r in rows),
                unauthorized_switches=sum(r['unauthorized_switches'] for r in rows),
                unauthorized_requests=sum(r['unauthorized_requests'] for r in rows),
                side_switches=sum(r['side_switches'] for r in rows),
                projected_actions=sum(r['projected_actions'] for r in rows),
                episodes_with_blocked_evidence=sum(r['first_locked_blocked_evidence_step'] is not None for r in rows),
                blocked_evidence_without_switch=sum(r['first_locked_blocked_evidence_step'] is not None
                                                    and not r['post_evidence_delays_steps'] for r in rows),
                post_evidence_switches=len(delays),
                mean_post_evidence_delay_steps=float(np.mean(delays)) if delays else None)
    if 'obstacles' in rows[0]:
        distances = [r['initial_distance_m'] for r in rows]
        result.update(initial_distance_range_m=[min(distances), max(distances)],
                      mean_remaining_m=float(np.mean([r['remaining_m'] for r in rows])),
                      mean_travelled_m=float(np.mean([r['travelled_m'] for r in rows])),
                      timeouts=sum(r['end_reason'] == 'time_budget' for r in rows),
                      passed_by_obstacle=[sum(r['obstacles'][i]['passed_step'] is not None for r in rows)
                                          for i in range(len(rows[0]['obstacles']))])
    return result


def evaluate(model, seeds, settings, policy, batch_size, *, env_factory=VTemp5Env, evidence_reference=False):
    rows = []
    for start in range(0, len(seeds), batch_size):
        records, _ = collect(model, seeds[start:start + batch_size], settings, policy, env_factory=env_factory,
                             evidence_reference=evidence_reference)
        rows.extend(records)
    return dict(summary=summarize(rows), episodes=rows)


def run(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(args.seed)
    kilometer = args.terrain.startswith('km-')
    terrain_family = args.terrain.removeprefix('km-')
    settings = dict(step_m=args.step_m, wait_probability=args.wait_probability)
    calibration = None
    if args.calibration:
        raw = args.calibration.read_bytes()
        data = json.loads(raw)
        parameters = data['parameters']
        if data['schema'] != 'vtemp5_motion_calibration_v1' or parameters['review_interval_s'] != REVIEW_S:
            raise ValueError('Expected vtemp5 motion calibration at the 30-second review interval')
        settings.update(step_m=parameters['moving_interval_displacement_m'],
                        wait_probability=parameters['stationary_interval_fraction'])
        calibration = dict(path=str(args.calibration.resolve()), sha256=hashlib.sha256(raw).hexdigest())
    if not math.isfinite(settings['step_m']) or settings['step_m'] <= 0.:
        raise ValueError('Expected positive finite step distance')
    if kilometer:
        low = args.min_distance_m if args.min_distance_m is not None else 1000.
        high = args.max_distance_m if args.max_distance_m is not None else 1500.
        if not (math.isfinite(low) and math.isfinite(high) and 1000. <= low <= high):
            raise ValueError('Kilometer route distances must be finite, ordered and at least 1000 m')
        settings.update(min_distance_m=low, max_distance_m=high)
        if args.horizon is None:
            # Allow for two large detours plus an indirect approach to the terminal goal.
            args.horizon = math.ceil(1.6 * (high + 1000.) / settings['step_m'])
    elif args.min_distance_m is not None or args.max_distance_m is not None:
        raise ValueError('Route distance options apply to km-detour and km-bridge')
    if args.horizon is None:
        args.horizon = 40 if args.terrain == 'local' else 144
    settings['horizon'] = args.horizon
    if min(args.threads, args.episodes_per_update, args.eval_episodes, args.epochs, args.horizon,
           args.microbatch_size) < 1:
        raise ValueError('Expected positive training dimensions')
    if args.updates < 1 and not args.eval_only:
        raise ValueError('Training requires at least one PPO update')
    if args.eval_only and not args.init_checkpoint:
        raise ValueError('Use --init-checkpoint with --eval-only')
    eval_seeds = list(range(args.eval_seed, args.eval_seed + args.eval_episodes))
    env_factory = VTemp5Env
    wm_info = None
    if args.terrain == 'local':
        if args.world_model:
            raise ValueError('The local pilot uses its built-in route prior; use --terrain detour or bridge for a trained model')
    else:
        if not args.world_model:
            raise ValueError('Large terrain training and evaluation require a --world-model checkpoint')
        from vtemp5.vtemp5_large import VTemp5LargeObstacleEnv
        from vtemp5.vtemp5_world_model import LocalWorldModel
        predictor = LocalWorldModel.load(args.world_model)
        if not args.eval_only and predictor.metadata['training_stage'] != terrain_family:
            raise ValueError('The learned world model must match the requested terrain')
        env_type = VTemp5LargeObstacleEnv
        if kilometer:
            from vtemp5.vtemp5_km import VTemp5KilometerEnv
            env_type = VTemp5KilometerEnv
        env_factory = partial(env_type, terrain=terrain_family, predictor=predictor)
        wm_info = dict(path=str(args.world_model.resolve()), training_stage=predictor.metadata['training_stage'],
                       seen_seed_ranges=predictor.metadata['seen_seed_ranges'],
                       sha256=hashlib.sha256(args.world_model.read_bytes()).hexdigest())
    first_train_seed = args.seed
    train_count = 0 if args.eval_only else args.updates * args.episodes_per_update
    if args.teacher_episodes < 0 or args.teacher_epochs < 1 or (args.eval_only and args.teacher_episodes):
        raise ValueError('Positive teacher epochs and nonnegative demonstration count are required for training')
    if not math.isfinite(args.teacher_refuted_angle_weight) or args.teacher_refuted_angle_weight < 0:
        raise ValueError('Expected a finite nonnegative refuted-angle loss weight')
    teacher_lr = args.learning_rate if args.teacher_learning_rate is None else args.teacher_learning_rate
    if any(not math.isfinite(lr) or lr <= 0. for lr in (teacher_lr, args.learning_rate)):
        raise ValueError('Expected finite positive optimizer learning rates')
    teacher_seed = first_train_seed + train_count
    recovery_seed = teacher_seed + (args.teacher_episodes + 1) // 2
    train_end = recovery_seed + (args.recovery_episodes + 1) // 2
    if args.recovery_episodes < 0 or (args.recovery_episodes and
            (terrain_family != 'bridge' or not args.init_checkpoint or args.eval_only)):
        raise ValueError('Recovery labeling requires bridge training with a warm-start checkpoint')
    if any(first_train_seed <= seed < train_end for seed in eval_seeds):
        raise ValueError('Training and evaluation map seeds overlap')
    model = GoalResidualBodyFramePolicy()
    prior_ranges = list(wm_info['seen_seed_ranges']) if wm_info else []
    eval_ranges = [[args.eval_seed, args.eval_seed + args.eval_episodes]]
    ancestor = None
    evidence_reference = bool(args.evidence_reference)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        if checkpoint['format'] != RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT:
            raise ValueError('Checkpoint must have the existing goal-residual vehicle actor format')
        prior_ranges.extend(checkpoint.get('prior_train_seed_ranges', []))
        eval_ranges.extend(checkpoint.get('eval_seed_ranges', []))
        if 'train_seed_range' in checkpoint:
            prior_ranges.append(checkpoint['train_seed_range'])
        if args.evidence_reference is None:
            evidence_reference = checkpoint.get('evidence_reference', False)
        model.load_state_dict(checkpoint['model'], strict=True)
        ancestor = dict(path=str(args.init_checkpoint.resolve()),
                        sha256=hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest(),
                        terrain=checkpoint.get('terrain', 'unspecified'))
    eval_policy = partial(evaluate, env_factory=env_factory, evidence_reference=evidence_reference)
    # Half-open seed intervals propagate across world model, demonstrations and PPO.
    # A later stage must preserve the independence of the earlier holdout as well.
    for low, high in [*prior_ranges, [first_train_seed, train_end]]:
        for test_low, test_high in eval_ranges:
            if max(low, test_low) < min(high, test_high):
                raise ValueError('Training seeds overlap current or inherited evaluation seeds')
    args.output.mkdir(parents=True, exist_ok=False)
    options = {key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()}
    source_names = ['vtemp5_env.py', 'vtemp5_policy.py', 'run_vtemp5.py', 'vtemp5_demonstrations.py']
    if wm_info:
        source_names += ['vtemp5_large.py', 'vtemp5_world_model.py']
    if kilometer:
        source_names.append('vtemp5_km.py')
    sources = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in source_names}
    atomic_json(args.output / 'manifest.json', dict(options=options, sources_sha256=sources,
                environment=settings, terrain=args.terrain, world_model=wm_info, ancestor=ancestor, calibration=calibration,
                evidence_reference=evidence_reference,
                eval_seed_ranges=eval_ranges,
                policy='GoalResidualBodyFramePolicy', review_s=REVIEW_S, eval_seeds=eval_seeds,
                train_seed_range=[first_train_seed, first_train_seed + train_count],
                teacher_seed_range=[teacher_seed, recovery_seed], recovery_seed_range=[recovery_seed, train_end],
                prior_train_seed_ranges=prior_ranges,
                inference='deterministic constrained physical angle; unmasked mode is diagnostic only'))
    evaluation = {}
    if not args.eval_only:
        evaluation['initial'] = eval_policy(model.eval(), eval_seeds, settings, 'deterministic', args.episodes_per_update)
    for baseline in ('straight', 'route'):
        evaluation[baseline] = eval_policy(None, eval_seeds, settings, baseline, args.episodes_per_update)
    if args.teacher_episodes or args.recovery_episodes:
        start = perf_counter()
        examples, records, recovery_records = [], [], []
        if args.teacher_episodes:
            examples, records = collect_teacher(teacher_seed, args.teacher_episodes, settings,
                                               env_factory=env_factory, terrain=terrain_family,
                                               evidence_reference=evidence_reference)
        if args.recovery_episodes:
            extra, recovery_records = collect_recovery(model.eval(), recovery_seed, args.recovery_episodes, settings,
                                                    env_factory=env_factory, evidence_reference=evidence_reference,
                                                    batch_size=args.episodes_per_update)
            examples.extend(extra)
        losses = fit_teacher(model, examples, args.teacher_epochs, args.microbatch_size,
                             teacher_lr, args.seed, refuted_angle_weight=args.teacher_refuted_angle_weight)
        atomic_json(args.output / 'teacher.json', dict(
            terrain=args.terrain, source='rule_offset on delivered evidence; paired initial probes; first ambiguous probes excluded',
            sampler='masked physical-angle NLL; refuted examples oversampled to one quarter',
            examples=len(examples), **losses, refuted_angle_weight=args.teacher_refuted_angle_weight,
            refuted_huber_delta_rad=1., episodes=records, recovery_episodes=recovery_records,
            training_seconds=perf_counter()-start))
        atomic_checkpoint(args.output / 'teacher.pt', dict(format=RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT,
                          model=model.state_dict(), environment='vtemp5', terrain=args.terrain, world_model=wm_info,
                          eval_seed_ranges=eval_ranges, ancestor=ancestor, evidence_reference=evidence_reference,
                          train_seed_range=[teacher_seed, train_end],
                          prior_train_seed_ranges=prior_ranges, completed_updates=0))
        prior_ranges = [*prior_ranges, [teacher_seed, train_end]]
        evaluation['after_teacher'] = eval_policy(model.eval(), eval_seeds, settings, 'deterministic', args.episodes_per_update)
        atomic_json(args.output / 'evaluation.json', evaluation)
        print(f"teacher episodes={len(records)} samples={len(examples)} "
              f"demonstrated switches={sum(r['side_switches'] for r in records)}", flush=True)
    if not args.eval_only:
        history = []
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        start_time = perf_counter()
        for update in range(1, args.updates + 1):
            first = first_train_seed + (update - 1) * args.episodes_per_update
            seeds = list(range(first, first + args.episodes_per_update))
            start = perf_counter()
            rows, samples = collect(model.eval(), seeds, settings, 'sampled', generator, env_factory=env_factory,
                                    evidence_reference=evidence_reference)
            sampling_s = perf_counter() - start
            ppo = update_policy(model.train(), optimizer, samples, epochs=args.epochs, microbatch_size=args.microbatch_size)
            summary = summarize(rows)
            history.append(dict(update=update, episodes=rows, summary=summary, ppo=ppo, sampling_s=sampling_s,
                                elapsed_s=perf_counter()-start, sampling_decisions_per_s=summary['decisions']/sampling_s))
            atomic_checkpoint(args.output / 'heading.pt',
                              dict(format=RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT, model=model.state_dict(),
                                   environment='vtemp5', terrain=args.terrain, world_model=wm_info,
                                   eval_seed_ranges=eval_ranges, ancestor=ancestor, evidence_reference=evidence_reference,
                                   train_seed_range=[first_train_seed, first + len(seeds)],
                                   prior_train_seed_ranges=prior_ranges, completed_updates=update))
            atomic_json(args.output / 'training.json', history)
            print(f"update {update}/{args.updates}  decisions={summary['decisions']}  "
                  f"success={summary['arrivals']}/{len(rows)}  wall={perf_counter()-start_time:.1f}s", flush=True)
        evaluation['training_seconds'] = perf_counter() - start_time
    evaluation['trained'] = eval_policy(model.eval(), eval_seeds, settings, 'deterministic', args.episodes_per_update)
    atomic_json(args.output / 'evaluation.json', evaluation)
    print(json.dumps(evaluation['trained']['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New directory for this run')
    parser.add_argument('--evidence-reference', action='store_true', default=None,
                        help='Use the surviving public route as the local target on a delivered closure; inherited by default')
    parser.add_argument('--recovery-episodes', type=int, default=0,
                        help='On-policy bridge rollouts to label after closure for demonstration training')
    parser.add_argument('--teacher-episodes', type=int, default=0)
    parser.add_argument('--teacher-epochs', type=int, default=8)
    parser.add_argument('--updates', type=int, default=8)
    parser.add_argument('--episodes-per-update', type=int, default=8)
    parser.add_argument('--eval-episodes', type=int, default=16)
    parser.add_argument('--horizon', type=int, help='Maximum decisions (default 40 local, 144 large, distance-based on km routes)')
    parser.add_argument('--terrain', choices=('local', 'detour', 'bridge', 'km-detour', 'km-bridge'), default='local')
    parser.add_argument('--min-distance-m', type=float, help='Minimum kilometer mission start-to-goal distance (default 1000 m)')
    parser.add_argument('--max-distance-m', type=float, help='Maximum kilometer mission start-to-goal distance (default 1500 m)')
    parser.add_argument('--world-model', type=Path, help='Frozen local world model for the selected large terrain')
    parser.add_argument('--step-m', type=float, default=6.209)
    parser.add_argument('--wait-probability', type=float, default=.02889)
    parser.add_argument('--calibration', type=Path, help='Use motion parameters measured from past episodes')
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--eval-seed', type=int, default=1000000)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--microbatch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=3.e-4)
    parser.add_argument('--teacher-refuted-angle-weight', type=float, default=0.,
                        help='BC weight on the unmasked mode angular Huber loss after an observed closure')
    parser.add_argument('--teacher-learning-rate', type=float,
                        help='Demonstration optimizer step size (defaults to the PPO learning rate)')
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--eval-only', action='store_true')
    run(parser.parse_args())
