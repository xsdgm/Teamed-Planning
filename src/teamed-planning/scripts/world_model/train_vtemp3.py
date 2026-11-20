"""Train received-graph dynamics, UAV fast control and UGV subgoal control.

Stages: world pretraining plus fast-policy and slow-policy distillation;
frozen-world fast PPO with slow imitation; frozen-world/fast slow PPO;
alternating updates of all three models. Collection completes missions for
both policies; --steps limits only explicitly requested short diagnostics.
Every collection batch uses immutable, jointly versioned model snapshots.
With --fresh-all: short missions train world+fast, long missions train
world+slow with fast fixed, and long joint missions fine-tune all three.
With --fresh-fast-world only world and fast train. Collection and validation use
rule-based subgoals by default. With --fixed-mission-target they use the same
local temporary target throughout each episode. The slow model stays frozen.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import inspect
import math
import multiprocessing
import os
from pathlib import Path
import signal
import platform
import sys
import time

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))

import numpy as np
import torch
from torch.distributions import Independent, Normal, kl_divergence

from world_model.graph_fast_policy import numpy_state, state_hash, ppo_update
from world_model.hierarchical_policy import make_graph_fast, GraphSlowActorCritic, FAST_SCHEMA, SLOW_SCHEMA
from world_model.graph_dynamics import (GraphWorldModel, MODEL_SCHEMA, ROAD_MODEL_SCHEMA, infer, objective,
                               belief_tokens, latest_history)
from world_model.slow_policy import update as slow_update
from world_model.scene_manifest import describe_scene_assumptions, make_scene, validate_scene_manifest
from world_model.frozen_navigation import CHECKPOINT_SHA, V9_FAST_REWARD_CONTRACT, V9_RUN, digest, write_json
from world_model.hierarchical_sources import (atomic_checkpoint, runtime_source, source_hashes as hash_sources,
                                               source_package, snapshot_sources)

TRAIN_SCHEMA = 'vtemp2_received_graph_three_model_training_v2'
PHASES = ('warmup', 'fast', 'slow', 'joint')


def random_initialization(config):
    return bool(config.get('fresh_all', False) or config.get('fresh_fast_world', False))


def phase_training_targets(config, phase):
    """Declare which optimizers may run; collection always uses fixed weights."""
    if phase not in PHASES:
        raise ValueError('Unknown training phase: ' + str(phase))
    if config.get('fresh_fast_world', False):
        if phase != 'fast' or config.get('freeze_fast_world', False):
            raise ValueError('Fresh fast/world training requires an unfrozen fast-only schedule')
        return dict(world=True, fast_ppo=True, slow_ppo=False,
                    slow_imitation=False, fast_distillation=False)
    if config.get('fresh_all', False):
        if phase == 'warmup':
            raise ValueError('Fresh three-stage training has no separate imitation warmup')
        if config.get('freeze_fast_world', False):
            raise ValueError('Fresh three-stage training must update its world model')
        return dict(world=True, fast_ppo=phase in ('fast', 'joint'),
                    slow_ppo=phase in ('slow', 'joint'), slow_imitation=False,
                    fast_distillation=False)
    frozen = config.get('freeze_fast_world', False)
    return dict(world=phase in ('warmup', 'joint') and not frozen,
                fast_ppo=phase in ('fast', 'joint') and not frozen,
                slow_ppo=phase in ('slow', 'joint'),
                slow_imitation=phase in ('warmup', 'fast'),
                fast_distillation=phase == 'warmup' and not frozen)


def optimization_protocol(config, optimizers):
    """Record effective optimizer and objective settings for publication audits."""
    def defaults(function):
        return {name: parameter.default for name, parameter in inspect.signature(function).parameters.items()
                if parameter.default is not inspect.Parameter.empty}
    slow = defaults(slow_update)
    slow['epochs'] = config['slow_epochs']
    fast = defaults(ppo_update)
    fast.update(epochs=config['ppo_epochs'], minibatch=config['ppo_minibatch'],
                gamma=.995, gae_lambda=.97, clip=.2, entropy_coef=.001,
                gradient_norm_clip=.5)
    protocol = dict(optimizers={name: dict(type=type(optimizer).__name__,
                     parameter_groups=[{key: value for key, value in group.items() if key != 'params'}
                                       for group in optimizer.param_groups])
                  for name, optimizer in optimizers.items()},
                slow_ppo=dict(slow, actor_gradient_norm_clip=.5, critic_gradient_norm_clip=.5,
                              discount='gamma ** executed option duration; lambda per slow decision'),
                fast_ppo=fast,
                world=dict(epochs=config['wm_epochs'], batch_size=config['wm_batch_size'], gradient_norm_clip=1.),
                slow_imitation=dict(epochs=config['slow_bc_epochs'], batch_size=config['slow_bc_batch_size'],
                                    gradient_norm_clip=1.),
                fast_distillation=dict(epochs=config['distill_epochs'], gradient_norm_clip=1.),
                phase_targets={phase: phase_training_targets(config, phase)
                               for phase in dict.fromkeys(phase_schedule(config))})
    # Keep the JSON and checkpoint copies identical, including Adam beta lists.
    return json.loads(json.dumps(protocol, allow_nan=False))


def phase_schedule(config):
    if 'update_plan' in config:
        from world_model.training_schedule import validate_update_plan
        return [entry['phase'] for entry in validate_update_plan(config)]
    return [phase for phase in PHASES for _ in range(config[phase + '_updates'])]


def execution_schedule(config):
    """A declared comparison may finish at a phase boundary, after validation."""
    schedule = phase_schedule(config)
    end_phase = config.get('stop_after_phase')
    if end_phase is None:
        return schedule
    if end_phase not in schedule:
        raise ValueError('Requested final phase is absent from this training schedule')
    end = max(index for index, phase in enumerate(schedule) if phase == end_phase) + 1
    return schedule[:end]


def curriculum_ranges(config):
    """Deterministic scene ranges, retained in the run manifest for replay."""
    if 'update_plan' in config:
        from world_model.training_schedule import validate_update_plan
        return [list(entry['distance_range_m']) for entry in validate_update_plan(config)]
    short = np.asarray(config['warmup_distances_m'], float)
    final = np.asarray(config['mission_distances_m'], float)
    ranges = []
    slow_index = 0
    for phase in phase_schedule(config):
        if phase in ('warmup', 'fast'):
            bounds = short
        elif phase == 'slow':
            # At least the last slow-only batch sees the full requested range.
            fraction = (slow_index + 1) / max(config['slow_updates'], 1)
            bounds = short + fraction * (final - short)
            slow_index += 1
        else:
            bounds = final
        ranges.append([float(value) for value in bounds])
    return ranges


def training_batches(config):
    """Cumulative task offsets also support a partially filled first batch."""
    if 'update_plan' in config:
        from world_model.training_schedule import update_episode_counts
        counts = update_episode_counts(config)
    else:
        counts = [config['episodes_per_update']] * len(phase_schedule(config))
    offset = 0
    result = []
    for count in counts:
        result.append((offset, count))
        offset += count
    return result


def evaluation_due(config, completed):
    schedule = phase_schedule(config)
    if not 0 < completed <= len(schedule):
        return False
    if 'update_plan' in config:
        from world_model.training_schedule import validate_update_plan
        return bool(validate_update_plan(config)[completed - 1]['evaluate_after'])
    return (completed % config['eval_every'] == 0 or completed == len(schedule)
            or schedule[completed] != schedule[completed - 1])


def pending_checkpoint_evaluation(saved):
    """Legacy checkpoints may have stopped after an update but before its eval."""
    if 'pending_evaluation' in saved:
        return saved['pending_evaluation']
    completed, config = saved['completed_updates'], saved['config']
    if not evaluation_due(config, completed):
        return None
    version = config.get('policy_version_offset', 0) + completed
    evaluated = {row['label'] for row in saved.get('records', ())
                 if row.get('evaluation') and row.get('version') == version}
    if len(evaluated) >= config['eval_scenes']:
        return None
    return dict(version=version, phase=phase_schedule(config)[completed - 1])


def validate_scene_seeds(seeds, config):
    """Validate an explicit, disjoint scene list without replacing any seed."""
    if not isinstance(seeds, dict) or set(seeds) != {'train', 'validation'}:
        raise ValueError('Scene seeds must contain exactly train and validation lists')
    counts = dict(train=sum(count for _, count in training_batches(config)),
                  validation=config['eval_scenes'])
    seen = set()
    result = {}
    for split, count in counts.items():
        values = seeds[split]
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f'Scene seeds {split} must be a list of exactly {count} seeds')
        for seed in values:
            if type(seed) is not int or seed < 0:
                raise ValueError('Scene seeds must be nonnegative integers; booleans are invalid')
            if seed in seen:
                raise ValueError(f'Duplicate scene seed across train/validation lists: {seed}')
            seen.add(seed)
        result[split] = list(values)
    return result


def build_scene_cases(config):
    """Rebuild the declared curriculum from config alone, including on resume."""
    ranges = curriculum_ranges(config)
    if 'scene_manifest' in config:
        if 'scene_seeds' in config:
            raise ValueError('Scene manifests and scene seed lists are mutually exclusive')
        return validate_scene_manifest(config['scene_manifest'], config, ranges)
    seeds = config.get('scene_seeds')
    per_case_ranges = [bounds for bounds, (_, count) in zip(ranges, training_batches(config))
                       for _ in range(count)]
    if seeds is None:
        seeds = dict(train=[config['seed'] + i
                            for i in range(len(per_case_ranges))],
                     validation=[config['seed'] + 1000000 + i
                                 for i in range(config['eval_scenes'])])
    else:
        seeds = validate_scene_seeds(seeds, config)
    return dict(train=[make_scene(seed, per_case_ranges[i], config['world_size'])
                       for i, seed in enumerate(seeds['train'])],
                validation=[make_scene(seed, config['mission_distances_m'], config['world_size'])
                            for seed in seeds['validation']])


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    for key, value in result.items():
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            raise ValueError('Invalid numeric dataset field: ' + key)
    return result


def load_world(paths):
    chunks = [load_npz(path) for path in paths]
    if not chunks:
        raise ValueError('No received-graph supervision available')
    for chunk in chunks:
        # Historical datasets recorded a teacher for every valid UAV actor
        # sample; fresh runs explicitly mark teacher-free samples instead.
        if 'actor_valid' in chunk and 'teacher_valid' not in chunk:
            chunk['teacher_valid'] = chunk['actor_valid'].copy()
    if any(set(chunk) != set(chunks[0]) for chunk in chunks):
        raise ValueError('World dataset schema differs between rollouts')
    return {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}


def tensor_batch(data, indices, device):
    return {key: torch.as_tensor(value[indices], device=device) for key, value in data.items()}


def parameter_delta(before, model):
    after = numpy_state(model)
    return float(np.sqrt(sum(np.sum((after[key].astype(float) - before[key]) ** 2)
                             for key in before)))


def require_finite_model(model, label):
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise FloatingPointError('Nonfinite parameters or buffers in ' + label)


def require_finite_optimizer(optimizer, label):
    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        if isinstance(value, float):
            return math.isfinite(value)
        return True
    if not finite(optimizer.state_dict()):
        raise FloatingPointError('Nonfinite optimizer state in ' + label)


def world_update(model, optimizer, data, device, rng, *, epochs=2, batch_size=128):
    before = numpy_state(model)
    losses = []
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(data['history']))
        for begin in range(0, len(order), batch_size):
            batch = tensor_batch(data, order[begin:begin + batch_size], device)
            loss = objective(infer(model, batch), batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite graph-world objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            require_finite_model(model, 'graph world model')
            losses.append(float(loss.detach()))
    return dict(optimizer_steps=len(losses), mean_loss=float(np.mean(losses)),
                parameter_delta=parameter_delta(before, model), samples=len(data['history']))


def distill_fast(fast, world, optimizer, data, device, rng, *, epochs=2, batch_size=64):
    """Transfer archived teacher decisions using newly estimated belief graphs.

    Teacher distributions are labels from the actual UAV local observation.
    They never enter world inference. World predictions are recomputed after
    its supervised update and detached before optimizing the graph actor.
    """
    ids = np.flatnonzero(data['actor_valid'] & data.get('teacher_valid', data['actor_valid']))
    if not len(ids):
        raise ValueError('Warmup lacks UAV teacher decisions')
    before = numpy_state(fast)
    world.eval()
    fast.train()
    losses = []
    for _ in range(epochs):
        order = rng.permutation(ids)
        for begin in range(0, len(order), batch_size):
            batch = tensor_batch(data, order[begin:begin + batch_size], device)
            with torch.no_grad():
                tokens = belief_tokens(infer(world, batch), batch)
            x = torch.cat((batch['raw_features'].float(), tokens.flatten(1)), dim=-1)
            teacher = Independent(Normal(batch['teacher_mean'], batch['teacher_std']), 1)
            loss = kl_divergence(teacher, fast.distribution(x)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite teacher-to-graph-actor KL')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fast.actor_parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            require_finite_model(fast, 'distilled fast actor')
            losses.append(float(loss.detach()))
    return dict(optimizer_steps=len(losses), teacher_kl=float(np.mean(losses)),
                parameter_delta=parameter_delta(before, fast), samples=len(ids))


def slow_episode(path, *, require_ppo=False):
    arrays = load_npz(path)
    rows = []
    for i in range(len(arrays['action'])):
        row = {key: value[i].copy() if value[i].ndim else value[i].item()
               for key, value in arrays.items()}
        row['duration'] = int(row['duration'])
        row['action'] = int(row['action'])
        rows.append(row)
    if not rows:
        raise ValueError('No slow behavior decisions in rollout')
    if require_ppo and any(not row.get('ppo_eligible', True) for row in rows):
        raise ValueError('Slow PPO cannot consume geometry-controlled or evaluation decisions')
    graph_path = Path(path).with_suffix('.graphs.npz')
    if graph_path.exists():
        graphs = load_npz(graph_path)
        fields = ('nodes', 'edges', 'edge_types', 'candidate_xy')
        expected = {f'g{i}_{key}' for i in range(len(rows)) for key in fields}
        if set(graphs) != expected:
            raise ValueError('Slow environment graph sidecar does not match decisions')
        for i, row in enumerate(rows):
            row['environment_graph'] = {key: graphs[f'g{i}_{key}'] for key in fields}
    return rows


def slow_behavior_clone(model, optimizer, episodes, *, epochs=4, batch_size=16, rng=None):
    """Warm-start candidate selection from feasible receiver-local teachers.

    Graphs are separate per-state arrays rather than candidate-repeated data.
    Only the actor receives gradients; no PPO likelihoods or critic targets
    are reused after these supervised updates.
    """
    rows = [row for episode in episodes for row in episode]
    if not rows or epochs < 1 or batch_size < 1:
        raise ValueError('Slow imitation requires nonempty data and positive update counts')
    if any('environment_graph' not in row or 'teacher_action' not in row for row in rows):
        raise ValueError('Slow imitation requires environment graphs and teacher labels')
    for row in rows:
        action = int(row['teacher_action'])
        if not 0 <= action < len(row['mask']) or not row['mask'][action]:
            raise ValueError('Slow teacher selected a masked action')
    total_samples = len(rows)
    rows = [row for row in rows if np.count_nonzero(row['mask']) > 1]
    if not rows:
        # A forced single action supplies no imitation signal. In particular,
        # do not advance existing Adam momentum on a zero-information batch.
        return dict(optimizer_steps=0, samples=total_samples, informative_samples=0,
                    singleton_samples=total_samples, mean_cross_entropy=0., parameter_delta=0.)
    rng = np.random.default_rng(0) if rng is None else rng
    before = numpy_state(model)
    actor = [*model.actor_encoder.parameters(), *model.actor_head.parameters()]
    losses = []
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(rows))
        for begin in range(0, len(order), batch_size):
            selected = [rows[i] for i in order[begin:begin + batch_size]]
            x = torch.as_tensor(np.stack([row['features'] for row in selected]), dtype=torch.float32)
            mask = torch.as_tensor(np.stack([row['mask'] for row in selected]), dtype=torch.bool)
            present = torch.as_tensor(np.stack([row['present'] for row in selected]), dtype=torch.bool)
            target = torch.tensor([int(row['teacher_action']) for row in selected], dtype=torch.long)
            distribution, _ = model(x, mask, present, [row['environment_graph'] for row in selected])
            loss = -distribution.log_prob(target).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite slow imitation objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor, 1., error_if_nonfinite=True)
            optimizer.step()
            require_finite_model(model, 'distilled slow actor')
            losses.append(float(loss.detach()))
    return dict(optimizer_steps=len(losses), samples=total_samples, informative_samples=len(rows),
                singleton_samples=total_samples-len(rows),
                mean_cross_entropy=float(np.mean(losses)),
                parameter_delta=parameter_delta(before, model))


def validate_resume_schema(saved):
    manifest = saved.get('manifest', {})
    schema = ROAD_MODEL_SCHEMA if saved.get('config', {}).get('road_topology') else MODEL_SCHEMA
    if (saved.get('schema') != TRAIN_SCHEMA or saved.get('world_schema') != schema
            or manifest.get('fast_schema') != FAST_SCHEMA or manifest.get('slow_schema') != SLOW_SCHEMA):
        raise ValueError('Training checkpoint schema mismatch; use --graph-init to transfer only compatible fast/world weights')


def initialize_graph_modules(fast, world, saved):
    """Transfer graph dynamics and fast control without importing an old slow actor."""
    manifest = saved.get('manifest', {})
    if saved.get('world_schema') != getattr(world, 'schema', MODEL_SCHEMA) or manifest.get('fast_schema') != FAST_SCHEMA:
        raise ValueError('Graph initialization requires compatible fast/world schemas')
    fast.load_state_dict(saved['fast'], strict=True)
    world.load_state_dict(saved['world'], strict=True)
    require_finite_model(fast, 'initialized fast actor')
    require_finite_model(world, 'initialized world model')


def initialize_slow_module(slow, saved):
    """Warm-start only the compatible slow graph actor, with fresh optimizers."""
    if saved.get('manifest', {}).get('slow_schema') != SLOW_SCHEMA:
        raise ValueError('Slow initialization requires the current slow graph schema')
    slow.load_state_dict(saved['slow'], strict=True)
    require_finite_model(slow, 'initialized slow actor')


def restore_training_state(saved, models, optimizers, rng, device):
    """Restore all learner state after fresh model construction, including RNG."""
    for key, model in models.items():
        model.load_state_dict(saved[key], strict=True)
        require_finite_model(model, key)
    for key, optimizer in optimizers.items():
        optimizer.load_state_dict(saved[key])
        require_finite_optimizer(optimizer, key)
    torch.set_rng_state(saved['torch_rng'])
    rng.bit_generator.state = saved['numpy_rng']
    if device.type == 'cuda' and saved.get('cuda_rng') is not None:
        torch.cuda.set_rng_state(saved['cuda_rng'], device)


def verify_continuation_parent(saved, path, plan):
    """A new source snapshot may continue a proven parent, never relabel it."""
    validate_resume_schema(saved)
    path = Path(path).resolve(strict=True)
    parent = plan['parent']
    if Path(parent['checkpoint_path']).resolve() != path or parent['checkpoint_sha256'] != digest(path):
        raise ValueError('Continuation parent checkpoint path or SHA-256 mismatch')
    if parent['completed_updates'] != saved['completed_updates']:
        raise ValueError('Continuation plan refers to a different completed parent update')
    original_source = runtime_source(path.parent.parent)
    hashes = hash_sources(original_source)
    if hashes != saved['manifest']['source_sha256'] or parent['source_sha256'] != hashes:
        raise ValueError('Continuation parent source snapshot changed')
    backend = '_wildscenes_map_backend.py'
    if backend in hashes:
        current_backend = SOURCE / backend
        if not current_backend.is_file() and SOURCE.parent.name != 'runtime_sources':
            current_backend = SOURCE.parent / 'simulator' / 'wildscenes_map.py'
        if not current_backend.is_file() or digest(current_backend) != hashes[backend]:
            raise ValueError('Continuation cannot silently change or omit the parent raster backend')
    geometry = plan['audit'].get('geometry_proof_reference')
    if geometry is not None and digest(geometry['path']) != geometry['sha256']:
        raise ValueError('Continuation geometry audit changed')
    for key in saved.get('replay_paths', ()):
        if not Path(key).is_file():
            raise ValueError('Continuation parent replay dataset is missing: ' + str(key))


def summarize_missions(rows, complete_missions):
    """Keep short smoke tasks separate from measured kilometre outcomes."""
    kilometer = [row for row in rows if row.get('kilometer_task', False)]
    group_rows = {}
    for row in rows:
        group_rows.setdefault(row.get('map_group') or 'synthetic', []).append(row)
    groups = {group: dict(episodes=len(items), goals_reached=sum(bool(row['goal_reached']) for row in items),
                         success_rate=float(np.mean([bool(row['goal_reached']) for row in items])))
              for group, items in sorted(group_rows.items())}
    return dict(mission_episodes=len(rows),
        full_mission_episodes=sum(bool(row.get('complete_missions', complete_missions)) for row in rows),
        kilometer_episodes=len(kilometer),
        kilometer_successes=sum(bool(row.get('kilometer_success', False)) for row in kilometer),
        kilometer_success_rate=(float(np.mean([bool(row.get('kilometer_success', False)) for row in kilometer]))
                                if kilometer else None),
        by_map_group=groups,
        terrain_protocols=sorted({row.get('terrain', 'synthetic') for row in rows}),
        source_unknown_space_modes=sorted({row['source_unknown_space'] for row in rows
                                          if row.get('source_unknown_space') is not None}),
        **{key: sum(int(row.get(key, 0)) for row in rows) for key in
           ('collisions', 'ground_safety_violations', 'raster_collision_attempts', 'raster_unknown_attempts')})


def evaluate_world(model, data, device, batch_size=128):
    model.eval()
    predictions = []
    stds = []
    losses = []
    with torch.inference_mode():
        for begin in range(0, len(data['history']), batch_size):
            batch = tensor_batch(data, slice(begin, begin + batch_size), device)
            prediction = infer(model, batch)
            predictions.append(prediction['states'].cpu().numpy())
            stds.append(prediction['state_std'].cpu().numpy())
            losses.append((float(objective(prediction, batch)), len(batch['history'])))
    predicted = np.concatenate(predictions)
    std = np.concatenate(stds)
    valid = data['state_mask'].astype(bool) & data['node_mask'][..., None]
    truth = data['target_states']
    age = latest_history(torch.as_tensor(data['history']),
                         torch.as_tensor(data['history_mask'])).numpy()[..., 9] * 30.
    groups = [('all', data['node_mask'].astype(bool)),
              ('remote', data['node_mask'].astype(bool) & ~data['ego_mask'].astype(bool)),
              ('unobserved_remote', data['node_mask'].astype(bool) & ~data['ego_mask'].astype(bool)
                                    & ~data['state_observed'].astype(bool)),
              ('uav', data['node_mask'].astype(bool) & (data['node_type'] == 0)),
              ('ugv', data['node_mask'].astype(bool) & (data['node_type'] == 1)),
              ('age_below_5s', data['node_mask'].astype(bool) & (age < 5)),
              ('age_5_to_15s', data['node_mask'].astype(bool) & (age >= 5) & (age < 15)),
              ('age_at_least_15s', data['node_mask'].astype(bool) & (age >= 15))]
    result = dict(samples=len(predicted), horizons_seconds=list(range(model.horizon + 1)),
                  objective=sum(loss * size for loss, size in losses) / sum(size for _, size in losses))
    for name, selection in groups:
        group = {}
        mask = valid & selection[..., None]
        for baseline, values in [('learned', predicted), ('kinematic_prior', data['baseline_states'])]:
            squared = np.sum(((values[..., :2] - truth[..., :2]) * 20.) ** 2, axis=-1)
            group[baseline + '_position_rmse_m'] = [
                float(np.sqrt(squared[..., t][mask[..., t]].mean())) if mask[..., t].any() else None
                for t in range(model.horizon + 1)]
        group['valid_state_targets'] = int(mask.sum())
        coverage_mask = mask.copy()
        coverage_mask[..., 0] &= ~data['state_observed'].astype(bool)
        group['valid_uncertain_state_targets'] = int(coverage_mask.sum())
        group['position_marginal_90pct_coverage'] = (
            float((np.abs(predicted[..., :2] - truth[..., :2]) <= 1.644854 * std[..., :2])[coverage_mask].mean())
            if coverage_mask.any() else None)
        result[name] = group
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=SOURCE.parents[3])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.1)
    parser.add_argument('--resume', nargs='?', const='latest', default=None,
                        help='Resume latest.pt in this output, or a named checkpoint from this run')
    parser.add_argument('--continue-from', type=Path,
                        help='Continue complete learner state into a new run using an explicit task rebatch plan')
    parser.add_argument('--continuation-plan', type=Path,
                        help='Audited fixed-task continuation JSON; required with --continue-from')
    parser.add_argument('--update-plan', type=Path,
                        help='Fresh-run JSON list of fixed task batches and evaluation boundaries')
    parser.add_argument('--stop-after-phase', choices=PHASES,
                        help='Declared experiment endpoint: finish this phase and its validation, then exit normally')
    parser.add_argument('--scene-seeds', type=Path,
                        help='JSON train/validation seed lists; embedded in config for reproducible resume')
    parser.add_argument('--scene-manifest', type=Path,
                        help='JSON training/validation scene lists; requires explicit curriculum distances')
    parser.add_argument('--validation-split', choices=('physical_map_group', 'heldout_tasks', 'synthetic_to_wildscenes'),
                        default='physical_map_group',
                        help='Use physical map holdout, held-out tasks from known maps, or V9-to-strict-WildScenes transfer')
    parser.add_argument('--ground-start-hold-seconds', type=int, default=0,
                        help='Zero UGV motion before this simulation time; mapping and planning continue')
    parser.add_argument('--validation-distances', dest='validation_distances_m', type=float, nargs=2,
                        help='Explicit metric direct-distance interval for the held-out scene manifest')
    parser.add_argument('--validation-slow-controller', choices=('learned', 'geometry', 'fixed_target'),
                        help='Heldout UGV target source. Fixed-target training also requires fixed-target validation.')
    parser.add_argument('--fast-init', type=Path, help='Optional old joint checkpoint; only compatible base weights transfer')
    parser.add_argument('--graph-init', type=Path,
                        help='Transfer compatible graph fast/world weights; initialize the new slow actor independently')
    parser.add_argument('--slow-init', type=Path,
                        help='Transfer current-schema slow actor weights; may accompany --graph-init')
    parser.add_argument('--freeze-fast-world', action='store_true',
                        help='Keep transferred fast/world weights fixed throughout slow imitation and PPO')
    parser.add_argument('--fresh-slow', action='store_true',
                        help='Train a newly randomized slow policy with fresh optimizer state and frozen graph fast/world')
    parser.add_argument('--fresh-all', action='store_true',
                        help='Randomize all models without pretrained teachers; train world+fast, world+slow, then all three')
    parser.add_argument('--fresh-fast-world', action='store_true',
                        help='Randomize world and fast without teachers and train only world and fast')
    parser.add_argument('--fixed-mission-target', action='store_true',
                        help='With --fresh-fast-world, use the episode endpoint as the only local temporary target throughout training and validation')
    parser.add_argument('--complete-missions', action=argparse.BooleanOptionalAction, default=True,
                        help='Retain complete missions for both policies; --steps applies only when disabled for diagnostics')
    parser.add_argument('--mission-distances', dest='mission_distances_m', type=float, nargs=2,
                        default=[100., 200.], metavar=('MIN_M', 'MAX_M'))
    parser.add_argument('--warmup-distances', dest='warmup_distances_m', type=float, nargs=2,
                        default=[30., 60.], metavar=('MIN_M', 'MAX_M'))
    parser.add_argument('--random-slow-probability', type=float, default=0.,
                        help='Optional random behavior during geometry-teacher collection')
    for name, default in [('warmup-updates', 8), ('fast-updates', 8), ('slow-updates', 16), ('joint-updates', 64),
                          ('workers', 4), ('episodes-per-update', 4), ('steps', 256), ('option-steps', 80),
                          ('stride', 4), ('seed', 26111600), ('jitter', 3), ('hidden', 64),
                          ('horizon', 4), ('wm-epochs', 2), ('wm-batch-size', 128), ('distill-epochs', 2),
                          ('ppo-epochs', 4), ('ppo-minibatch', 16), ('slow-epochs', 4),
                          ('slow-bc-epochs', 32), ('slow-bc-batch-size', 16),
                          ('eval-every', 8), ('eval-scenes', 3), ('replay-episodes', 16)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--loss', type=float, default=.1)
    parser.add_argument('--frozen', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    needs_manifest = args.validation_split == 'synthetic_to_wildscenes' or args.validation_distances_m is not None
    if needs_manifest and args.scene_manifest is None:
        parser.error('Cross-domain validation and explicit validation distances require --scene-manifest')
    if args.ground_start_hold_seconds < 0:
        parser.error('--ground-start-hold-seconds must be nonnegative')
    if args.validation_distances_m is not None:
        lo, hi = args.validation_distances_m
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
            parser.error('--validation-distances requires 0 < MIN_M <= MAX_M')
    for phase in PHASES:
        if getattr(args, phase + '_updates') < 0:
            parser.error(phase + '_updates must be nonnegative')
    if not sum(getattr(args, phase + '_updates') for phase in PHASES):
        parser.error('At least one training phase must have updates')
    for name in ('workers', 'episodes_per_update', 'steps', 'option_steps', 'stride', 'hidden', 'horizon',
                 'wm_epochs', 'wm_batch_size', 'distill_epochs', 'ppo_epochs', 'ppo_minibatch', 'slow_epochs',
                 'slow_bc_epochs', 'slow_bc_batch_size', 'eval_every', 'eval_scenes', 'replay_episodes'):
        if getattr(args, name) < 1:
            parser.error(name + ' must be positive')
    if args.option_steps < 4 or args.horizon != 4 or args.jitter < 0 or not 0 <= args.loss < 1:
        parser.error('Require option_steps>=4, horizon=4 and valid transport settings')
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error('Invalid GPU memory fraction')
    if sum(value is not None for value in (args.resume, args.fast_init, args.graph_init)) > 1:
        parser.error('--resume, --fast-init and --graph-init are mutually exclusive')
    if (args.continue_from is None) != (args.continuation_plan is None):
        parser.error('--continue-from and --continuation-plan must be supplied together')
    if args.update_plan is not None and (args.resume is not None or args.continue_from is not None):
        parser.error('--update-plan is for fresh runs; resume and continuation use their saved plans')
    if sum((args.fresh_all, args.fresh_slow, args.fresh_fast_world)) > 1:
        parser.error('--fresh-all, --fresh-slow and --fresh-fast-world are mutually exclusive')
    if args.fixed_mission_target and not args.fresh_fast_world:
        parser.error('--fixed-mission-target requires --fresh-fast-world')
    if args.validation_slow_controller == 'fixed_target' and not args.fixed_mission_target and args.resume is None:
        parser.error('--validation-slow-controller fixed_target requires --fixed-mission-target')
    if args.fresh_fast_world:
        required_controller = 'fixed_target' if args.fixed_mission_target else 'geometry'
        if args.validation_slow_controller not in (None, required_controller):
            parser.error(f'This training target requires --validation-slow-controller {required_controller}')
        args.validation_slow_controller = required_controller
        if any(value is not None for value in (args.fast_init, args.graph_init, args.slow_init,
                                               args.resume, args.continue_from)):
            parser.error('--fresh-fast-world cannot inherit weights or previous learner state')
        if args.freeze_fast_world:
            parser.error('--fresh-fast-world must train its fast and world models')
        if args.warmup_updates or args.slow_updates or args.joint_updates or args.fast_updates < 1:
            parser.error('--fresh-fast-world requires positive fast updates and zero warmup/slow/joint updates')
        if args.random_slow_probability != 0.:
            parser.error('--fresh-fast-world requires --random-slow-probability 0')
    if args.validation_slow_controller is None:
        args.validation_slow_controller = 'learned'
    if args.fresh_slow:
        if args.graph_init is None or not args.freeze_fast_world:
            parser.error('--fresh-slow requires --graph-init and --freeze-fast-world')
        if any(value is not None for value in (args.slow_init, args.resume, args.continue_from)):
            parser.error('--fresh-slow cannot inherit a slow policy or previous learner state')
        if args.fast_updates or args.joint_updates or not args.slow_updates:
            parser.error('--fresh-slow requires positive slow updates and zero fast/joint updates')
    if args.fresh_all:
        if any(value is not None for value in (args.fast_init, args.graph_init, args.slow_init,
                                               args.resume, args.continue_from)):
            parser.error('--fresh-all cannot inherit weights or previous learner state')
        if args.fresh_slow or args.freeze_fast_world:
            parser.error('--fresh-all cannot combine with --fresh-slow or --freeze-fast-world')
        if args.warmup_updates or any(getattr(args, phase + '_updates') < 1 for phase in ('fast', 'slow', 'joint')):
            parser.error('--fresh-all requires zero warmup updates and positive fast, slow and joint updates')
    if args.continue_from is not None and any(value is not None for value in
            (args.resume, args.fast_init, args.graph_init, args.slow_init, args.scene_seeds, args.scene_manifest)):
        parser.error('--continue-from cannot combine with resume, weight initialization or external scene overrides')
    if args.resume is not None and args.scene_seeds is not None:
        parser.error('--resume uses saved scene seeds and cannot be combined with --scene-seeds')
    if args.scene_manifest is not None and args.scene_seeds is not None:
        parser.error('--scene-manifest and --scene-seeds are mutually exclusive')
    if args.resume is not None and (args.scene_manifest is not None or args.slow_init is not None):
        parser.error('--resume uses its saved scene manifest and slow actor; external overrides are forbidden')
    if args.scene_manifest is not None:
        for flag in ('--warmup-distances', '--mission-distances'):
            if not any(arg == flag or arg.startswith(flag + '=') for arg in sys.argv[1:]):
                parser.error('--scene-manifest requires explicit ' + flag)
    if args.freeze_fast_world and args.graph_init is None and args.resume is None and args.continue_from is None:
        parser.error('--freeze-fast-world requires --graph-init or a compatible --resume')
    for bounds in (args.mission_distances_m, args.warmup_distances_m):
        upper = math.inf if args.scene_manifest is not None else 1600.
        if not all(math.isfinite(value) for value in bounds) or not 0 < bounds[0] <= bounds[1] <= upper:
            parser.error('Mission ranges must be finite and satisfy 0 < MIN_M <= MAX_M (synthetic MAX_M <= 1600)')
    if any(a > b for a, b in zip(args.warmup_distances_m, args.mission_distances_m)):
        parser.error('Warmup distances cannot exceed the final curriculum distances')
    if not 0 <= args.random_slow_probability <= 1:
        parser.error('random-slow-probability must lie in [0, 1]')
    return args


def make_initial_config(args):
    """Embed a fresh task plan before validating its scene manifest."""
    config = {key: value for key, value in vars(args).items()
              if key not in ('repo', 'output', 'frozen', 'resume', 'fast_init', 'graph_init', 'slow_init',
                             'scene_seeds', 'scene_manifest', 'continue_from', 'continuation_plan', 'update_plan')}
    config.update(world_size=2000.)
    if args.update_plan is not None:
        from world_model.training_schedule import validate_update_plan
        config['update_plan'] = json.loads(args.update_plan.read_text(encoding='utf-8'))
        validate_update_plan(config)
    if args.fresh_slow:
        config['training_mode'] = 'fresh_slow'
    if args.fresh_all:
        config['training_mode'] = 'fresh_all_staged'
    if args.fresh_fast_world:
        config['training_mode'] = 'fresh_fast_world'
        if args.fixed_mission_target:
            from world_model.partial_observation import PARTIAL_OBSERVATION_PROTOCOL
            config['training_observation_protocol'] = PARTIAL_OBSERVATION_PROTOCOL
            config['road_topology'] = True
    return config


def snapshot(args):
    output, repo = args.output.resolve(), args.repo.resolve()
    if not args.frozen:
        target = output / 'runtime_sources' / 'world_model'
        if args.resume is not None:
            target = runtime_source(output)
            resume_path = (output / 'checkpoints' / 'latest.pt' if args.resume == 'latest'
                           else Path(args.resume).resolve())
            if resume_path.parent != output / 'checkpoints':
                raise ValueError('Resume checkpoint must belong to this output directory')
            # Reject legacy slow interfaces before re-entering their old code.
            validate_resume_schema(torch.load(resume_path, map_location='cpu', weights_only=True))
            # Re-enter the original source version even if the working tree has changed.
        else:
            if output.exists() and any(output.iterdir()):
                raise ValueError('Output is not empty; preserve the previous experiment')
            snapshot_sources(SOURCE, target,
                             include_backend=args.scene_manifest is not None or args.continue_from is not None)
        entrypoint = 'train_graph.py' if source_package(target) == 'vtemp2' else 'train_hierarchical.py'
        os.execv(sys.executable, [sys.executable, '-I', '-B', '-u', str(target / entrypoint),
                                 *sys.argv[1:], '--repo', str(repo), '--output', str(output), '--frozen'])
    for name in ('data', 'episodes', 'logs', 'metrics', 'checkpoints'):
        (output / name).mkdir(exist_ok=True)
    return output, repo


def main():
    args = parse_args()
    output, repo = snapshot(args)
    config = make_initial_config(args)
    saved = None
    continuation = None
    if args.resume is not None:
        resume_path = output / 'checkpoints' / 'latest.pt' if args.resume == 'latest' else Path(args.resume).resolve()
        if resume_path.parent != output / 'checkpoints':
            raise ValueError('Resume checkpoint must belong to this output directory')
        saved = torch.load(resume_path, map_location='cpu', weights_only=True)
        validate_resume_schema(saved)
        config = dict(saved['config'])
        if any(arg.split('=')[0] == '--validation-slow-controller' for arg in sys.argv[1:]):
            saved_controller = config.get('validation_slow_controller',
                                           'geometry' if config.get('fresh_fast_world') else 'learned')
            if saved_controller != args.validation_slow_controller:
                raise ValueError('Resume requires the saved validation target controller')
        # Only explicitly supplied resource choices may change on resume.
        for flag, key in [('--device', 'device'), ('--workers', 'workers'),
                          ('--gpu-memory-fraction', 'gpu_memory_fraction')]:
            if any(arg == flag or arg.startswith(flag + '=') for arg in sys.argv[1:]):
                config[key] = getattr(args, key)
    elif args.continue_from is not None:
        from world_model.training_schedule import validate_continuation_plan
        saved = torch.load(args.continue_from.resolve(), map_location='cpu', weights_only=True)
        continuation = json.loads(args.continuation_plan.read_text(encoding='utf-8'))
        verify_continuation_parent(saved, args.continue_from, continuation)
        config = validate_continuation_plan(continuation, saved)
        for flag, key in [('--device', 'device'), ('--workers', 'workers'),
                          ('--gpu-memory-fraction', 'gpu_memory_fraction')]:
            if any(arg == flag or arg.startswith(flag + '=') for arg in sys.argv[1:]):
                config[key] = getattr(args, key)
    elif args.scene_seeds is not None:
        config['scene_seeds'] = validate_scene_seeds(
            json.loads(args.scene_seeds.read_text(encoding='utf-8')), config)
    elif args.scene_manifest is not None:
        config['scene_manifest'] = validate_scene_manifest(
            json.loads(args.scene_manifest.read_text(encoding='utf-8')), config, curriculum_ranges(config))
    if args.stop_after_phase is not None:
        config['stop_after_phase'] = args.stop_after_phase
    execution_schedule(config)
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    device = torch.device(config['device'])
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable; no silent CPU fallback')
        torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'], device)
    elif device.type != 'cpu':
        raise ValueError('Only CPU and CUDA devices are supported')
    fast = (make_graph_fast(repo, random_init=True) if random_initialization(config)
            else make_graph_fast(repo)).to(device)
    graph_init = None
    if args.graph_init is not None:
        graph_init = torch.load(args.graph_init.resolve(), map_location='cpu', weights_only=True)
        config['road_topology'] = graph_init.get('world_schema') == ROAD_MODEL_SCHEMA
        if graph_init.get('config', {}).get('training_observation_protocol'):
            config['training_observation_protocol'] = graph_init['config']['training_observation_protocol']
    world = GraphWorldModel(config['hidden'], config['horizon'],
                            road_topology=config.get('road_topology', False)).to(device)
    # Existing SMDP PPO is CPU-only; its model is small and collected on CPU.
    slow = GraphSlowActorCritic(hidden=config['hidden'])
    initialization = (dict(teacher='none; rule-based UGV subgoals in the short stage',
                           pretrained_weights_loaded=False, model_initialization='fresh constructors')
                      if random_initialization(config) else
                      dict(teacher='archived_v9', archived_v9_sha256=CHECKPOINT_SHA))
    if args.fast_init is not None:
        old = torch.load(args.fast_init.resolve(), map_location='cpu', weights_only=True)
        base = {key[len('base.'):]: value for key, value in old['fast'].items() if key.startswith('base.')}
        fast.base.load_state_dict(base, strict=True)
        initialization.update(fast_init=str(args.fast_init.resolve()), fast_init_sha256=digest(args.fast_init),
                              transferred='base only; old prediction adapters are incompatible')
    if graph_init is not None:
        initialize_graph_modules(fast, world, graph_init)
        initialization.update(graph_init=str(args.graph_init.resolve()),
                              graph_init_sha256=digest(args.graph_init),
                              transferred='graph fast and world only')
    if args.slow_init is not None:
        old = torch.load(args.slow_init.resolve(), map_location='cpu', weights_only=True)
        initialize_slow_module(slow, old)
        initialization.update(slow_init=str(args.slow_init.resolve()), slow_init_sha256=digest(args.slow_init),
                              slow_transferred='current-schema slow graph actor weights only; fresh optimizer state')
    actor_optimizer = torch.optim.Adam([
        dict(params=[*fast.base.gat_actor.parameters(), fast.base.log_std],
             lr=3e-4 if random_initialization(config) else 1e-5),
        dict(params=[*fast.actor_peers.parameters(), *fast.actor_delta.parameters()], lr=3e-4)])
    critic_optimizer = torch.optim.Adam(fast.critic_parameters(), lr=1e-4)
    world_optimizer = torch.optim.Adam(world.parameters(), lr=3e-4)
    slow_optimizer = torch.optim.Adam(slow.parameters(), lr=3e-4)
    if args.fresh_slow:
        initialization.update(slow_initialization='random', learner_seed=config['seed'],
                              optimizer_initialization='fresh', inherited_replay=False,
                              frozen_models=['fast', 'world'],
                              initial_model_sha256={name: state_hash(numpy_state(model))
                                                    for name, model in [('fast', fast), ('world', world), ('slow', slow)]})
    if random_initialization(config):
        initialization.update(learner_seed=config['seed'], optimizer_initialization='fresh',
                              inherited_replay=False, models_initialized=['fast', 'world', 'slow'],
                              initial_model_sha256={name: state_hash(numpy_state(model))
                                                    for name, model in [('fast', fast), ('world', world), ('slow', slow)]})
    fixed_target = config.get('fixed_mission_target', False)
    if config.get('fresh_fast_world', False):
        slow.requires_grad_(False)
        initialization.update(teacher='none' if fixed_target else 'none; rule-based UGV subgoals in training and validation',
                              frozen_models=['slow'], trained_models=['fast', 'world'])
    source_hashes = hash_sources(SOURCE)
    from world_model.hierarchical_safety import control_contract
    manifest = dict(algorithm='vtemp3', schema=TRAIN_SCHEMA, world_schema=world.schema,
        control_safety=control_contract(),
        fast_schema=FAST_SCHEMA, slow_schema=SLOW_SCHEMA, initialization=initialization,
        source_sha256=source_hashes, source_package='world_model', entrypoint='train_hierarchical.py',
        training_mode=config.get('training_mode', 'staged'), device=str(device), slow_training_device='cpu',
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        architecture='Receiver-local temporal graph world model shared by UAVs and UGV; UAV graph fast actor; UGV graph slow candidate actor; UGV rule motion controller.',
        schedule='Warmup: supervised world plus fast distillation and slow geometry imitation. Fast: world frozen, fast PPO and slow geometry imitation. Slow: world/fast frozen, SMDP PPO. Joint: freeze all during collection, then update fast, slow and world separately. freeze_fast_world disables every fast/world update in all phases.',
        world_inputs='Only received/local histories, original source timestamps, hops, known geometry, task acknowledgements, local sensor occupancy with explicit known-pixel masks, and received map-node geometry/confidence attributes. True trajectories are masked supervision only. No actor-prior call before world inference.',
        actors='UAV sensor/map/own CBF context retained; direct raw peer/relay channels removed from fast base input and replaced by graph beliefs. UGV candidates use an explicit received environment graph with final/current task nodes, map structure and aligned candidate geometry.',
        cbf='Current CBF computed from actual received geometry before UAV action; deterministic graph attribute, never overwritten by estimated peer geometry. Final CBF filter remains in control loop.',
        slow='Candidate index maps to a subgoal. Rule UGV planner executes it and broadcasts it; UAVs can use the task only after receipt. SMDP PPO discounts by actual option duration, not decision count.',
        validation='Disjoint scene seeds; full-mission learned-slow heldout rollouts on the final distance range; state RMSE split by age and robot type. No heldout samples enter optimizers.',
        collection='Complete physical missions by default for both fast and slow policies, including true arrival/failure/deadline. steps is used only with the explicit --no-complete-missions diagnostic exception.',
        curriculum=dict(warmup_distances_m=config['warmup_distances_m'],
                        mission_distances_m=config['mission_distances_m'], update_ranges_m=curriculum_ranges(config)),
        limitations='Three UAVs and one UGV, dt=1s, synthetic 2km world. Four-second robot dynamics only; retained map evidence is not a learned future occupancy generator. Slow policy is trained from actual executions, not long-horizon imagined subgoal returns.')
    manifest['optimization'] = optimization_protocol(config, dict(
        fast_actor=actor_optimizer, fast_critic=critic_optimizer, world=world_optimizer, slow=slow_optimizer))
    manifest['environment'] = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        torch=str(torch.__version__), numpy=np.__version__, torch_threads=torch.get_num_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        thread_environment={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    from dataclasses import asdict, replace
    from world_model.hierarchical_runtime import HierarchyConfig
    manifest['hierarchy_config'] = asdict(replace(HierarchyConfig(), option_steps=config['option_steps'],
        ground_start_hold_seconds=config.get('ground_start_hold_seconds', 0)))
    schedule = execution_schedule(config)
    manifest['execution_scope'] = dict(final_phase=schedule[-1], effective_updates=len(schedule),
        full_plan_updates=len(phase_schedule(config)),
        effective_training_tasks=sum(count for _, count in training_batches(config)[:len(schedule)]),
        phase_endpoint_is_predeclared=True)
    if config.get('training_observation_protocol'):
        manifest['training_observation_protocol'] = config['training_observation_protocol']
        manifest['sensor_coverage'] = ('V9 random training maps use seed-defined aerial survey strips. '
            'The vehicle retains its local sensor. WildScenes uses the source map coverage.')
    manifest['phase_training_targets'] = {phase: phase_training_targets(config, phase)
                                          for phase in dict.fromkeys(schedule)}
    if config.get('fresh_all', False):
        manifest['schedule'] = ('Three models initialized by fresh constructors, without pretrained weights or learned teachers. '
            'Short fast stage: world supervised learning plus fast PPO; UGV follows rule-based subgoals and slow parameters stay fixed. '
            'Long slow stage: world supervised learning plus slow SMDP PPO, with fast fixed. '
            'Long joint stage: world supervised learning plus fast and slow PPO. '
            'All collection uses immutable model snapshots; optimizers run sequentially after complete batches.')
    if config.get('fresh_fast_world', False):
        target_protocol = ('Training and validation use one local temporary target per episode. '
                           if fixed_target else 'Training and validation use rule-based UGV subgoals. ')
        manifest['schedule'] = ('Fresh world and fast constructors without pretrained weights or learned teachers. '
            'Every batch trains the world model from supervised trajectories and the fast policy with PPO. '
            'The slow model is newly initialized, permanently frozen and never controls the UGV. '
            + target_protocol + 'Validation uses deterministic fast actions. All collection uses immutable '
            'model snapshots and optimizers run after complete batches.')
    if fixed_target:
        manifest['fixed_mission_target'] = True
        manifest['architecture'] = ('Receiver-local graph world model, UAV fast policy and rule-based UGV motion '
            'controller. The episode endpoint is the only task coordinate.')
        manifest['world_inputs'] = ('Each receiver knows its assigned local temporary target from the start. '
            'Robot histories, peer acknowledgements, map geometry and occupancy evidence retain receiver-local '
            'sensor and transport timestamps. True trajectories provide masked world-model supervision.')
        manifest['actors'] = ('Every UAV fast actor receives the assigned local temporary target from the first '
            'control step through termination. Ground position, peer state and map evidence retain their '
            'sensor and communication paths.')
        manifest['slow'] = 'Target assignment bypasses the geometric candidate selector and the frozen slow model.'
        manifest['collection'] = ('The UGV receives the same local temporary target for the entire episode and '
            'executes its local motion planner. UAV actions supply fast PPO data. Physical trajectories supply '
            'world-model supervision.')
        manifest['optimization']['fast_ppo'].update(
            reward_contract=V9_FAST_REWARD_CONTRACT,
            reward_config=str(V9_RUN / 'config.json'), reward_config_sha256=digest(repo / V9_RUN / 'config.json'),
            reward='Archived v9 vehicle-to-target mapping, route quality and frontier gains, step and CBF costs; '
                   'one arrival bonus, absorbing CBF failure, ordinary final step at the time limit.',
            raster_safety='WildScenes ground safety violations use the archived absorbing failure penalty.')
        manifest['execution_scope']['task'] = 'Reach the single local temporary target assigned for this episode.'
        manifest['hierarchy_config']['target_execution'] = 'One target held until episode termination or collection truncation.'
        manifest['limitations'] = ('Three UAVs and one UGV with dt=1s. The learned world model predicts '
            'four seconds of robot dynamics. Map evidence comes from sensing and communication.')
    if config.get('road_topology'):
        manifest['world_inputs'] += (' Any observed obstacle marks its 2 m cell occupied. Free cells require full '
            'known coverage. The input retains the exact edges among the closest 32 road nodes. All received '
            'nodes and roads contribute to spatial bins, component counts and target-relative graph distances.')
    if 'update_plan' in config:
        manifest['update_plan'] = config['update_plan']
    ranges = curriculum_ranges(config)
    batch_layout = training_batches(config)
    version_offset = config.get('policy_version_offset', 0)
    cases = build_scene_cases(config)
    scene_assumptions = describe_scene_assumptions(cases)
    config['scene_assumptions'] = scene_assumptions
    manifest['scene_assumptions'] = scene_assumptions
    validation_split = config.get('validation_split', 'physical_map_group') if 'scene_manifest' in config else 'synthetic_scene_seeds'
    validation_scope = ('Heldout navigation tasks; training maps may be seen. No spatial or cross-map independence claim.'
                        if validation_split == 'heldout_tasks' else
                        'Synthetic V9 random obstacle training; held-out original WildScenes raster tasks.'
                        if validation_split == 'synthetic_to_wildscenes' else
                        'Heldout physical map groups.' if validation_split == 'physical_map_group' else
                        'Independent synthetic scene seeds.')
    manifest.update(validation_split=validation_split, validation_scope=validation_scope)
    if 'scene_manifest' in config:
        manifest['validation'] = (validation_scope + ' Every task seed is unique; heldout_tasks rejects identical or reversed '
            'training endpoint pairs within 2m. Final-range heldout rollouts never enter optimizers. '
            'K sequences share Karawatha; V sequences share Venman.')
        manifest['scene_validation'] = ('Map identity, original metric scale, declared direct distance, curriculum range and '
            '1.5m endpoint clearance are checked. The scene loader does not establish a connecting route; '
            'route reachability requires separate geometry-audit evidence.')
        manifest['limitations'] = ('Three UAVs and one UGV, dt=1s, metric WildScenes 2D rasters; unknown space blocked. '
            'Declared reference distance sets the mission budget and is not a verified shortest-path length. '
            'Four-second learned robot dynamics; retained map evidence is not learned future occupancy.')
        if scene_assumptions['simplified_geometry']:
            manifest['limitations'] = ('Three UAVs and one UGV, dt=1s, WildScenes-derived simplified 2D rectangle rasters. '
                'Source-unknown space is free by explicit simulation assumption; simulator known masks are complete, '
                'while robots receive only local and communicated evidence. Reference distance sets the mission budget '
                'and is not a verified shortest-path length. Results apply to this simplified geometry protocol.')
    if validation_split == 'synthetic_to_wildscenes':
        manifest['validation'] = 'All training tasks use V9 random circle geometry. Evaluation uses fixed strict WildScenes routes only.'
        manifest['scene_validation'] = ('V9 circle geometry is bound to per-seed file hashes. '
            'Held-out routes use the original raster, exact metric scale, and 1.5 m continuous ground clearance. '
            'Reference route geometry is withheld from the policy.')
        manifest['limitations'] = ('Three UAVs and one UGV with dt=1s. WildScenes source-unknown terrain is impassable '
            'to the ground vehicle. Evaluation is cross-domain on a single held-out physical map.')
        manifest['curriculum']['validation_distances_m'] = (config.get('validation_distances_m') or config['mission_distances_m'])
        manifest['ground_start_hold_seconds'] = config.get('ground_start_hold_seconds', 0)
    validation_controller = config.get('validation_slow_controller', 'learned')
    manifest['validation_slow_controller'] = validation_controller
    training_controllers = {phase: 'geometry' if phase in ('warmup', 'fast') else 'learned'
                            for phase in dict.fromkeys(schedule)}
    if fixed_target:
        training_controllers = dict.fromkeys(schedule, 'fixed_target')
    manifest['controllers'] = dict(validation_slow=validation_controller, validation_fast='deterministic_learned',
        training_slow_by_phase=training_controllers)
    if fixed_target:
        validation_scope += ' Reach a single fixed local temporary target per episode with deterministic fast actions.'
        manifest['validation_scope'] = validation_scope
        manifest['validation'] = validation_scope + ' Held-out samples are excluded from training.'
        manifest['limitations'] += ' The task measures local temporary-target attainment only.'
    elif validation_controller == 'geometry':
        validation_scope += ' World/fast evaluation with a geometry UGV subgoal selector; not learned-slow evaluation.'
        manifest['validation_scope'] = validation_scope
        manifest['validation'] = (validation_scope + ' Geometry-only UGV subgoal selection with deterministic '
            'fast actions; evaluates world/fast with rule control, not learned slow-policy quality. '
            'Heldout samples never enter optimizers; geometry decisions are ineligible for slow PPO.')
    if continuation is not None:
        provenance = dict(schema=continuation['schema'], parent=continuation['parent'], audit=continuation['audit'],
                          plan_sha256=digest(args.continuation_plan), source_plan=str(args.continuation_plan.resolve()))
        config['continuation_provenance'] = provenance
        manifest.update(continuation=provenance, policy_version_offset=version_offset,
                        update_plan=config['update_plan'])
        manifest['initialization'] = dict(continued_from=continuation['parent'],
            transferred='Three models, four optimizers, Torch/NumPy/CUDA RNG, counters, records and world replay paths',
            parent_initialization=saved['manifest'].get('initialization'))
    completed = 0
    counters = dict(fast_ppo_optimizer_updates=0, fast_distillation_optimizer_updates=0,
                    world_optimizer_updates=0, slow_optimizer_updates=0, slow_imitation_optimizer_updates=0,
                    slow_imitation_transitions=0, slow_imitation_informative_transitions=0, training_env_steps=0,
                    fast_training_transitions=0)
    records, replay_paths = [], []
    pending_evaluation = None
    if saved is not None:
        if continuation is None and saved['manifest']['source_sha256'] != source_hashes:
            raise ValueError('Archived source snapshot changed since checkpoint')
        restore_training_state(saved, dict(fast=fast, world=world, slow=slow),
            dict(actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                 world_optimizer=world_optimizer, slow_optimizer=slow_optimizer), rng, device)
        completed = saved['completed_updates'] if continuation is None else 0
        counters = dict(saved['counters'])
        records, replay_paths = list(saved['records']), list(saved['replay_paths'])
        pending_evaluation = pending_checkpoint_evaluation(saved)
        if continuation is None:
            manifest = saved['manifest']
    write_json(output / 'config.json', config)
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'scenes.json', cases)
    started = time.time()
    stopped = []

    def stop_requested(signum, _frame):
        stopped.append(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)

    def status(phase, state='running', **extra):
        row = dict(state=state, phase=phase, pid=os.getpid(), device=str(device),
                   elapsed_seconds=round(time.time() - started, 2), completed_updates=completed,
                   policy_version=version_offset + completed, policy_version_offset=version_offset,
                   pending_evaluation=pending_evaluation,
                   total_updates=len(schedule), scenes_completed=len(records), **counters, **extra)
        write_json(output / 'status.json', row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def checkpoint(name):
        path = output / 'checkpoints' / name
        for label, optimizer in [('actor', actor_optimizer), ('critic', critic_optimizer),
                                 ('world', world_optimizer), ('slow', slow_optimizer)]:
            require_finite_optimizer(optimizer, label)
        atomic_checkpoint(path, dict(schema=TRAIN_SCHEMA, world_schema=world.schema,
            fast=fast.state_dict(), world=world.state_dict(), slow=slow.state_dict(),
            actor_optimizer=actor_optimizer.state_dict(), critic_optimizer=critic_optimizer.state_dict(),
            world_optimizer=world_optimizer.state_dict(), slow_optimizer=slow_optimizer.state_dict(),
            config=config, manifest=manifest, completed_updates=completed, counters=counters,
            policy_version=version_offset + completed, pending_evaluation=pending_evaluation,
            records=records, replay_paths=replay_paths, torch_rng=torch.get_rng_state(),
            numpy_rng=rng.bit_generator.state,
            cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None))

    from world_model.graph_rollouts import collect_graph
    try:
        if saved is None or continuation is not None:
            checkpoint('initial.pt')
            checkpoint('latest.pt')
        status('initializing')
        with ProcessPoolExecutor(max_workers=config['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
            def collect(split, selected, version, phase, evaluation=False):
                states = [numpy_state(model) for model in (fast, world, slow)]
                hashes = dict(zip(('fast_sha256', 'world_sha256', 'slow_sha256'), map(state_hash, states)))
                local_config = dict(config, random_slow_probability=0. if evaluation else config['random_slow_probability'])
                jobs = [(str(repo), case, local_config, str(output), f'{split}-{phase}-v{version:03d}-{i:04d}',
                         *states, version, phase, evaluation) for i, case in selected]
                status('collecting_' + split, training_stage=phase, expected_scenes=len(jobs))
                futures = [pool.submit(collect_graph, job) for job in jobs]
                result = []
                for future in as_completed(futures):
                    row = future.result()
                    if row['version'] != version or any(row[key] != value for key, value in hashes.items()):
                        raise ValueError('Rollout mixed model versions')
                    if fixed_target and (row['slow_controller'] != 'fixed_target' or row['slow_ppo_eligible']
                                         or row['slow_decisions'] or row['slow_path'] is not None):
                        raise ValueError('Fixed-target rollout unexpectedly used a slow selector or slow training data')
                    if fixed_target and row['fast_reward_contract'] != V9_FAST_REWARD_CONTRACT:
                        raise ValueError('Unexpected fixed-target fast reward')
                    result.append(row)
                    records.append(row)
                    write_json(output / 'collection.json', records)
                    status('collecting_' + split, training_stage=phase, latest_scene=row['label'])
                return sorted(result, key=lambda row: row['label'])

            def evaluate_pending():
                nonlocal pending_evaluation
                if pending_evaluation is None:
                    return
                version, phase = pending_evaluation['version'], pending_evaluation['phase']
                heldout = collect('validation', list(enumerate(cases['validation'])), version, phase, True)
                paths = [row['wm_path'] for row in heldout if row['wm_path']]
                evaluation = dict(version=version, stage=phase, episodes=heldout,
                    success_rate=float(np.mean([row['goal_reached'] for row in heldout])),
                    slow_controller=validation_controller, fast_controller='deterministic_learned',
                    slow_ppo_eligible=False,
                    validation_split=validation_split, validation_scope=validation_scope,
                    scene_assumptions=scene_assumptions,
                    world=evaluate_world(world, load_world(paths), device) if paths else None)
                evaluation.update(summarize_missions(heldout, config['complete_missions']))
                write_json(output / 'metrics' / f'validation-{version:03d}.json', evaluation)
                pending_evaluation = None
                checkpoint('latest.pt')

            # Resume any evaluation saved at an interrupted update boundary
            # before changing the corresponding policy version again.
            evaluate_pending()
            for iteration in range(completed, len(schedule)):
                if stopped:
                    break
                phase = schedule[iteration]
                targets = phase_training_targets(config, phase)
                start, count = batch_layout[iteration]
                version = version_offset + iteration
                selected = list(enumerate(cases['train'][start:start + count], start))
                collected = collect('train', selected, version, phase)
                # Finish a complete collection/update transaction before saving
                # SIGTERM. Never save new behavior labels with half-updated models.
                batches = [load_npz(row['ppo_path']) for row in collected]
                counters['training_env_steps'] += sum(row['executed_steps'] for row in collected)
                counters['fast_training_transitions'] += sum(len(batch['rewards']) for batch in batches)
                metrics = dict(stage=phase, version_before=version, version_after=version + 1,
                               local_update=iteration + 1, batch_episodes=count,
                               curriculum_distances_m=ranges[iteration], complete_missions=config['complete_missions'],
                               validation_split=validation_split, validation_scope=validation_scope,
                               scene_assumptions=scene_assumptions)
                metrics.update(summarize_missions(collected, config['complete_missions']))
                metrics['training_targets'] = targets
                if fixed_target:
                    metrics['slow_controller'] = 'fixed_target'
                if targets['fast_ppo']:
                    status('updating_fast', training_stage=phase)
                    fast.train()
                    result = ppo_update(fast, batches, actor_optimizer, critic_optimizer, device, rng,
                                        epochs=config['ppo_epochs'], minibatch=config['ppo_minibatch'])
                    counters['fast_ppo_optimizer_updates'] += result['actor_optimizer_steps']
                    metrics['fast_ppo'] = result
                if targets['slow_ppo']:
                    status('updating_slow', training_stage=phase)
                    slow.train()
                    before = numpy_state(slow)
                    result = slow_update(slow, slow_optimizer,
                                         [slow_episode(row['slow_path'], require_ppo=True) for row in collected],
                                         epochs=config['slow_epochs'])
                    result['parameter_delta'] = parameter_delta(before, slow)
                    counters['slow_optimizer_updates'] += result['accepted_epochs']
                    metrics['slow_ppo'] = result
                if targets['slow_imitation']:
                    status('distilling_slow', training_stage=phase)
                    result = slow_behavior_clone(slow, slow_optimizer,
                        [slow_episode(row['slow_path']) for row in collected],
                        epochs=config['slow_bc_epochs'], batch_size=config['slow_bc_batch_size'], rng=rng)
                    counters['slow_imitation_optimizer_updates'] += result['optimizer_steps']
                    counters['slow_imitation_transitions'] += result['samples']
                    counters['slow_imitation_informative_transitions'] += result['informative_samples']
                    metrics['slow_imitation'] = result
                replay_paths.extend(row['wm_path'] for row in collected if row['wm_path'])
                replay_paths = replay_paths[-config['replay_episodes']:]
                if targets['world']:
                    training = load_world(replay_paths)
                    status('updating_world', training_stage=phase)
                    result = world_update(world, world_optimizer, training, device, rng,
                                          epochs=config['wm_epochs'], batch_size=config['wm_batch_size'])
                    counters['world_optimizer_updates'] += result['optimizer_steps']
                    metrics['world'] = result
                    if targets['fast_distillation']:
                        status('distilling_fast', training_stage=phase)
                        result = distill_fast(fast, world, actor_optimizer, training, device, rng,
                                              epochs=config['distill_epochs'])
                        counters['fast_distillation_optimizer_updates'] += result['optimizer_steps']
                        metrics['fast_distillation'] = result
                for label, model in [('fast', fast), ('world', world), ('slow', slow)]:
                    require_finite_model(model, label)
                completed = iteration + 1
                if evaluation_due(config, completed):
                    pending_evaluation = dict(version=version_offset + completed, phase=phase)
                metrics.update(goals_reached=sum(row['goal_reached'] for row in collected),
                    mean_remaining_m=float(np.mean([row['remaining_m'] for row in collected])),
                    terminations={reason: sum(row['termination'] == reason for row in collected)
                                  for reason in sorted({row['termination'] for row in collected})})
                write_json(output / 'metrics' / f'update-{version + 1:03d}.json', metrics)
                checkpoint(f'update-{version + 1:03d}.pt')
                checkpoint('latest.pt')
                status('update_complete', training_stage=phase, latest_metrics=metrics)
                boundary = completed == len(schedule) or schedule[completed] != phase
                if boundary:
                    checkpoint(phase + '-complete.pt')
                if stopped:
                    break
                evaluate_pending()
        if stopped:
            checkpoint('latest.pt')
            status('stopped_at_batch_boundary', 'stopped', requested_signal=stopped[-1])
        else:
            checkpoint('final.pt')
            status('finished', 'complete')
    except BaseException as error:
        # latest.pt remains the last fully completed optimizer transaction.
        status('failed', 'failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
