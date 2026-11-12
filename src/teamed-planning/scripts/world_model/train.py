"""Train the vtmp2 world model on scene-separated, screened rollout episodes.

The caller screens physical scenes before collection. This module consumes
only the vehicle-observation NPZs, never the screening paths or obstacle truth.
Training a predictor does not automatically deploy it as a target selector.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

if __package__:
    from .data import (ALGORITHM, CHANNELS, CHECKPOINT_SHA, FEATURES, HORIZON,
                       MISSION_ACTION_CONTRACT, MISSION_SCHEMA_VERSION, SCHEMA_VERSION, SIDE)
    from .model import Vtmp2WorldModel, imagine_sequence, sequence_loss
    from .models import make_model
else:
    from data import (ALGORITHM, CHANNELS, CHECKPOINT_SHA, FEATURES, HORIZON,
                      MISSION_ACTION_CONTRACT, MISSION_SCHEMA_VERSION, SCHEMA_VERSION, SIDE)
    from model import Vtmp2WorldModel, imagine_sequence, sequence_loss


CHECKPOINT_FORMAT = 'vtmp2-world-model-training-v1'
LEGACY_ACTION_CONTRACT = 'legacy_drive_v1'
COMPLETION_EVENT_SEMANTICS = (
    'within_recorded_transition_drive_arrival_or_survey_evidence_not_eventual_ground_reachability')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_metrics(path, history):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in history))
    temporary.replace(path)


def _scalar(episode, name):
    array = episode[name]
    if array.shape != ():
        raise ValueError(f'{name} must be scalar')
    return array.item()


def load_episodes(dataset_dir, *, max_bytes=1024 ** 3):
    """Validate both splits and preload less than one GiB of rollout arrays."""
    root = Path(dataset_dir).resolve()
    episodes = {'train': [], 'val': []}
    manifest = []
    seeds = {}
    scenes = {}
    memory = 0
    dataset_contract = None
    observed_purposes = {split: set() for split in episodes}
    for split in episodes:
        for path in sorted(root.glob(split + '_*.npz')):
            try:
                with np.load(path, allow_pickle=False) as archive:
                    episode = {key: archive[key] for key in archive.files}
                if _scalar(episode, 'algorithm') != ALGORITHM:
                    raise ValueError('Expected algorithm=vtmp2')
                schema = _scalar(episode, 'schema_version')
                if type(schema) is not int or schema not in (SCHEMA_VERSION, MISSION_SCHEMA_VERSION):
                    raise ValueError(f'Expected schema_version={SCHEMA_VERSION} or {MISSION_SCHEMA_VERSION}')
                action_contract = (_scalar(episode, 'action_contract') if 'action_contract' in episode
                                   else LEGACY_ACTION_CONTRACT if schema == SCHEMA_VERSION else None)
                expected_contract = (MISSION_ACTION_CONTRACT if schema == MISSION_SCHEMA_VERSION
                                     else LEGACY_ACTION_CONTRACT)
                if action_contract != expected_contract:
                    raise ValueError(f'Schema {schema} requires action_contract={expected_contract}')
                contract = (schema, action_contract)
                if dataset_contract is not None and dataset_contract != contract:
                    raise ValueError('Cannot mix schema versions or action contracts in one dataset')
                dataset_contract = contract
                if _scalar(episode, 'checkpoint_sha256') != CHECKPOINT_SHA:
                    raise ValueError('Frozen actor checkpoint SHA does not match')
                if _scalar(episode, 'feasibility_validated') is not True:
                    raise ValueError('Physical scene feasibility has not been certified')
                clearance = _scalar(episode, 'physical_clearance_m')
                if not isinstance(clearance, (float, int)) or not math.isfinite(clearance) or clearance < 1.5:
                    raise ValueError('Physical path clearance must be at least 1.5m')
                scene_sha = _scalar(episode, 'scene_sha256')
                geometry_sha = _scalar(episode, 'scene_geometry_sha256') if 'scene_geometry_sha256' in episode else scene_sha
                for name, value in (('scene_sha256', scene_sha), ('scene_geometry_sha256', geometry_sha)):
                    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                        raise ValueError(f'{name} must be a SHA-256 hex digest')
                if geometry_sha in scenes and scenes[geometry_sha] != split:
                    raise ValueError('A physical scene occurs in both training and validation')
                scenes[geometry_sha] = split
                if _scalar(episode, 'split') != split:
                    raise ValueError('Filename split disagrees with episode split')
                seed = _scalar(episode, 'seed')
                if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
                    raise ValueError('Episode seed must be an integer')
                seed = int(seed)
                if seed in seeds:
                    raise ValueError(f'Scene seed {seed} appears twice ({seeds[seed]} and {split})')
                seeds[seed] = split
                actions = episode['actions']
                n = len(actions)
                if n < 1:
                    raise ValueError('An episode must contain a transition')
                shapes = dict(maps=(n+1, CHANNELS, SIDE, SIDE), states=(n+1, FEATURES),
                              actions=(n, 3), labels=(n, 3), elapsed=(n,),
                              terminated=(n,), truncated=(n,), times=(n+1,),
                              targets=(n, 2), target_switched=(n,), switch_reason=(n,),
                              target_status=(n,))
                if schema == MISSION_SCHEMA_VERSION:
                    shapes['mission_purpose'] = (n,)
                for key, shape in shapes.items():
                    if episode[key].shape != shape:
                        raise ValueError(f'{key} has shape {episode[key].shape}, expected {shape}')
                for key in ('maps', 'states', 'actions', 'labels', 'elapsed', 'times', 'targets'):
                    if not np.issubdtype(episode[key].dtype, np.number) or not np.isfinite(episode[key]).all():
                        raise ValueError(f'{key} must be finite numeric values')
                for key in ('terminated', 'truncated', 'target_switched'):
                    if not np.isin(episode[key], (0, 1)).all():
                        raise ValueError(f'{key} must be boolean')
                    episode[key] = episode[key].astype(bool)
                for key in ('maps', 'labels', 'elapsed'):
                    if np.any(episode[key] < -1e-6) or np.any(episode[key] > 1 + 1e-6):
                        raise ValueError(f'{key} must lie in [0, 1]')
                if episode['terminated'][:-1].any() or episode['truncated'][:-1].any():
                    raise ValueError('An episode cannot contain transitions after termination/truncation')
                if episode['terminated'][-1] == episode['truncated'][-1]:
                    raise ValueError('The last transition must be either terminated or truncated')
                durations = np.diff(episode['times'])
                if (np.any(durations < -1e-6)
                        or not np.allclose(durations / HORIZON, episode['elapsed'], atol=1e-5, rtol=0)):
                    raise ValueError('Episode times disagree with elapsed fractions')
                if np.any((durations <= 1e-8) & ~episode['terminated']):
                    raise ValueError('A zero-duration transition must be terminal')
                if not np.allclose(actions[:, :2], episode['targets'] / 1000., atol=1e-6, rtol=0):
                    raise ValueError('Action coordinates disagree with target coordinates')
                if schema == MISSION_SCHEMA_VERSION:
                    purposes = episode['mission_purpose']
                    if not np.isin(purposes, ('drive', 'survey')).all():
                        raise ValueError('mission_purpose must be drive or survey')
                    expected_sign = np.where(purposes == 'drive', 1., -1.)
                    if not np.allclose(actions[:, 2], expected_sign, atol=1e-6, rtol=0):
                        raise ValueError('Action sign disagrees with mission_purpose')
                    observed_purposes[split].update(purposes.tolist())
                else:
                    if not np.allclose(actions[:, 2], 1., atol=1e-6, rtol=0):
                        raise ValueError('Schema 2 actions request the full prediction horizon')
                    if 'mission_purpose' in episode and (episode['mission_purpose'].shape != (n,)
                            or not np.all(episode['mission_purpose'] == 'drive')):
                        raise ValueError('Schema 2 supports drive actions only')
                    observed_purposes[split].add('drive')
                if not np.allclose(episode['states'][:, 6:8], episode['states'][0, 6:8], atol=1e-6, rtol=0):
                    raise ValueError('The final mission goal changed within one episode')
                if not np.isin(episode['target_status'], ('verified', 'unverified', 'hold')).all():
                    raise ValueError('Unknown target status')
                memory += sum(value.nbytes for value in episode.values())
                if memory > max_bytes:
                    raise ValueError('Dataset exceeds the one-GiB preload budget')
                for key in ('maps', 'states', 'actions', 'labels', 'elapsed'):
                    episode[key] = episode[key].astype(np.float32, copy=False)
                episode['_seed'] = seed
                episodes[split].append(episode)
                manifest.append(dict(file=path.name, split=split, seed=seed, transitions=n,
                                     sha256=digest(path), scene_sha256=scene_sha,
                                     scene_geometry_sha256=geometry_sha,
                                     schema_version=schema, action_contract=action_contract,
                                     physical_clearance_m=float(clearance)))
            except (KeyError, ValueError, TypeError, OSError) as exc:
                raise ValueError(f'Invalid rollout {path.name}: {exc}') from exc
        if not episodes[split]:
            raise ValueError(f'No {split} episodes in {root}')
    if dataset_contract[0] == MISSION_SCHEMA_VERSION and observed_purposes['train'] != {'drive', 'survey'}:
        raise ValueError('Schema 3 training requires observed drive and survey actions')
    payload = dict(algorithm=ALGORITHM, schema_version=dataset_contract[0],
                   action_contract=dataset_contract[1],
                   observed_action_purposes=sorted(observed_purposes['train']),
                   split_action_purposes={split: sorted(values) for split, values in observed_purposes.items()},
                   checkpoint_sha256=CHECKPOINT_SHA, episodes=manifest)
    payload['sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return episodes, payload


class EpisodeWindowDataset(Dataset):
    """Non-overlapping within-scene windows with an explicitly masked tail."""
    def __init__(self, episodes, sequence_length=16):
        if type(sequence_length) is not int or sequence_length < 1:
            raise ValueError('sequence_length must be a positive integer')
        self.episodes = episodes
        self.sequence_length = sequence_length
        self.windows = [(index, start) for index, episode in enumerate(episodes)
                        for start in range(0, len(episode['actions']), sequence_length)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode_index, start = self.windows[index]
        episode = self.episodes[episode_index]
        length = min(self.sequence_length, len(episode['actions']) - start)
        result = {}
        for key in ('maps', 'states'):
            values = episode[key][start:start+length+1]
            padded = np.repeat(values[-1:], self.sequence_length+1, axis=0)
            padded[:length+1] = values
            result[key] = padded
        for key in ('actions', 'labels', 'elapsed'):
            values = episode[key][start:start+length]
            padded = np.zeros((self.sequence_length, *values.shape[1:]), np.float32)
            padded[:length] = values
            result[key] = padded
        result['mask'] = np.r_[np.ones(length, np.float32),
                               np.zeros(self.sequence_length-length, np.float32)]
        return result


def split_statistics(episodes):
    labels = np.concatenate([episode['labels'] for episode in episodes])
    elapsed = np.concatenate([episode['elapsed'] for episode in episodes])
    completion_name = 'mission_completed' if 'mission_purpose' in episodes[0] else 'arrived'
    return dict(episodes=len(episodes), transitions=len(labels),
                seeds=[episode['_seed'] for episode in episodes],
                event_positive_rates=dict(zip((completion_name, 'stalled', 'cbf_rejected'),
                                             labels.mean(0).astype(float).tolist())),
                completion_event_semantics=COMPLETION_EVENT_SEMANTICS,
                zero_duration_events=int(np.count_nonzero(elapsed == 0)),
                interrupted_fraction=float(np.mean(elapsed < 1)))


def _on_device(batch, device):
    return {key: value.to(device, non_blocking=False) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    """Deterministic one-step prior prediction, before future assimilation."""
    model.eval()
    names = ('state_mse_scaled', 'map_mse_weighted', 'event_bce', 'event_brier',
             'elapsed_mse', 'kl', 'position_squared_error_m',
             'persistence_state_mse_scaled', 'persistence_map_mse_weighted')
    totals = {name: 0. for name in names}
    count = 0.
    learnable = model.delta_scale > 0
    for batch in loader:
        batch = _on_device(batch, device)
        maps, states = batch['maps'], batch['states']
        batch_size, observations = states.shape[:2]
        embeddings = model.encoder(maps.flatten(0, 1)).reshape(batch_size, observations, -1)
        hidden = states.new_zeros((batch_size, model.hidden))
        latent, _ = model.observe(hidden, embeddings[:, 0], states[:, 0], False)
        for t in range(observations-1):
            mask = batch['mask'][:, t]
            if not mask.any():
                break
            hidden, _, prior, pm, ps, outcome = model.imagine(
                hidden, latent, batch['actions'][:, t], maps[:, t], states[:, t], False)
            future_maps, future_states = maps[:, t+1], states[:, t+1]
            weights = (1 + 4 * ((future_maps > 0) | (maps[:, t] > 0)).float()
                       + 10 * (abs(future_maps-maps[:, t]) > .01).float())
            state_error = (ps[:, learnable] - future_states[:, learnable]) / model.delta_scale[learnable]
            persistence_error = (states[:, t, learnable] - future_states[:, learnable]) / model.delta_scale[learnable]
            # Future observations may update the posterior only AFTER prediction.
            latent, posterior = model.observe(hidden, embeddings[:, t+1], future_states, False)
            kl = torch.distributions.kl_divergence(posterior, prior).clamp_min(.1).mean(-1)
            values = dict(
                state_mse_scaled=state_error.square().mean(-1),
                map_mse_weighted=((pm-future_maps).square()*weights).mean((1, 2, 3)),
                event_bce=F.binary_cross_entropy_with_logits(outcome[:, :3], batch['labels'][:, t], reduction='none').mean(-1),
                event_brier=(outcome[:, :3].sigmoid()-batch['labels'][:, t]).square().mean(-1),
                elapsed_mse=(outcome[:, 3].sigmoid()-batch['elapsed'][:, t]).square(),
                kl=kl,
                position_squared_error_m=((ps[:, :2]-future_states[:, :2])*1000).square().sum(-1),
                persistence_state_mse_scaled=persistence_error.square().mean(-1),
                persistence_map_mse_weighted=((maps[:, t]-future_maps).square()*weights).mean((1, 2, 3)))
            for name, value in values.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f'Non-finite validation metric: {name}')
                totals[name] += float((value*mask).sum())
            count += float(mask.sum())
    if not count:
        raise ValueError('Validation contains no unmasked transitions')
    result = {name: value/count for name, value in totals.items()}
    result['position_rmse_m'] = math.sqrt(result.pop('position_squared_error_m'))
    result['loss'] = (result['state_mse_scaled'] + 5*result['map_mse_weighted']
                      + .1*result['event_bce'] + .1*result['elapsed_mse'] + .01*result['kl'])
    result['transitions'] = int(count)
    return result


@torch.no_grad()
def evaluate_rollout(model, loader, device, horizons=(1, 2, 4, 8)):
    """Free predictions along recorded action sequences, from each window start.

    These sequences include target changes and interrupted transitions. Their
    recorded durations are reported explicitly; a horizon of four transitions
    is not necessarily forty seconds under one fixed candidate target.
    """
    horizons = tuple(sorted(set(horizons)))
    if not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError('rollout horizons must contain positive integers')
    model.eval()
    names = ('state_mse_scaled', 'map_mse_weighted', 'position_squared_error_m',
             'persistence_state_mse_scaled', 'persistence_map_mse_weighted',
             'persistence_position_squared_error_m', 'actual_elapsed_seconds',
             'predicted_elapsed_seconds', 'event_bce')
    totals = {h: dict.fromkeys(names, 0.) for h in horizons}
    counts = dict.fromkeys(horizons, 0)
    bounds = {h: [float('inf'), 0.] for h in horizons}
    learnable = model.delta_scale > 0
    for batch in loader:
        batch = _on_device(batch, device)
        maps, states = batch['maps'], batch['states']
        valid = batch['mask'].cumprod(1)
        actual_elapsed = batch['elapsed'].cumsum(1) * HORIZON
        predicted_elapsed = states.new_zeros(states.shape[0])
        for t, (pm, ps, outcome) in enumerate(imagine_sequence(
                model, maps[:, 0], states[:, 0], batch['actions'], max(horizons))):
            h = t + 1
            predicted_elapsed = predicted_elapsed + outcome[:, 3].sigmoid() * HORIZON
            if not valid[:, t].any():
                break
            if h not in totals:
                continue
            mask = valid[:, t]
            fm, fs = maps[:, h], states[:, h]
            weights = (1 + 4 * ((fm > 0) | (maps[:, 0] > 0)).float()
                       + 10 * (abs(fm-maps[:, 0]) > .01).float())
            values = dict(
                state_mse_scaled=(((ps[:, learnable]-fs[:, learnable]) /
                                   model.delta_scale[learnable]).square().mean(-1)),
                map_mse_weighted=((pm-fm).square()*weights).mean((1, 2, 3)),
                position_squared_error_m=((ps[:, :2]-fs[:, :2])*1000).square().sum(-1),
                persistence_state_mse_scaled=(((states[:, 0, learnable]-fs[:, learnable]) /
                                               model.delta_scale[learnable]).square().mean(-1)),
                persistence_map_mse_weighted=((maps[:, 0]-fm).square()*weights).mean((1, 2, 3)),
                persistence_position_squared_error_m=((states[:, 0, :2]-fs[:, :2])*1000).square().sum(-1),
                actual_elapsed_seconds=actual_elapsed[:, t],
                predicted_elapsed_seconds=predicted_elapsed,
                event_bce=F.binary_cross_entropy_with_logits(
                    outcome[:, :3], batch['labels'][:, t], reduction='none').mean(-1))
            for name, value in values.items():
                selected = value[mask.bool()]
                if not torch.isfinite(selected).all():
                    raise FloatingPointError(f'Non-finite rollout metric: {name}')
                totals[h][name] += float(selected.sum())
            durations = actual_elapsed[:, t][mask.bool()]
            bounds[h][0] = min(bounds[h][0], float(durations.min()))
            bounds[h][1] = max(bounds[h][1], float(durations.max()))
            counts[h] += int(mask.sum())
    result = {}
    for h in horizons:
        count = counts[h]
        if not count:
            result[str(h)] = dict(samples=0)
            continue
        row = {name: value/count for name, value in totals[h].items()}
        row['position_rmse_m'] = math.sqrt(row.pop('position_squared_error_m'))
        row['persistence_position_rmse_m'] = math.sqrt(row.pop('persistence_position_squared_error_m'))
        row.update(samples=count, actual_elapsed_min_seconds=bounds[h][0],
                   actual_elapsed_max_seconds=bounds[h][1])
        result[str(h)] = row
    return dict(initialization='window_start_observation_only',
                action_protocol='recorded_action_sequence_including_target_switches',
                event_semantics=COMPLETION_EVENT_SEMANTICS,
                horizons=result)


def _rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])


def _restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if device.type == 'cuda':
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume requires the same visible CUDA device count')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def _train(dataset_dir, output, *, device, epochs, batch_size, sequence_length, lr,
           seed, resume, bootstrap, hidden, latent, max_train_windows,
           rollout_steps, rollout_weight):
    started = time.time()
    state_path = output/'status.json'

    def status(stage, **fields):
        atomic_json(state_path, dict(algorithm=ALGORITHM, stage=stage, pid=os.getpid(),
                    updated_at=time.time(), bootstrap=bootstrap,
                    deployment_enabled=False, **fields))

    if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive integers')
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('lr must be finite and positive')
    if type(rollout_steps) is not int or rollout_steps < 1:
        raise ValueError('rollout_steps must be a positive integer')
    if not math.isfinite(rollout_weight) or rollout_weight < 0:
        raise ValueError('rollout_weight must be finite and nonnegative')
    if max_train_windows is not None and (type(max_train_windows) is not int or max_train_windows < 1):
        raise ValueError('max_train_windows must be a positive integer')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; no silent CPU fallback')
    if device.type not in ('cuda', 'cpu'):
        raise ValueError('Only cpu and cuda training devices are supported')
    if (output/'latest.pt').exists() and not resume:
        raise FileExistsError('Training checkpoint already exists; use resume=True or a new output directory')
    if resume and not (output/'latest.pt').is_file():
        raise FileNotFoundError('resume=True requires latest.pt')
    status('validating_dataset')
    episodes, manifest = load_episodes(dataset_dir)
    datasets = {split: EpisodeWindowDataset(values, sequence_length) for split, values in episodes.items()}
    statistics = {split: split_statistics(values) for split, values in episodes.items()}
    sources = {name: digest(Path(__file__).with_name(name))
               for name in ('train.py', 'model.py', 'models.py', 'data.py', 'runtime.py', 'vtmp2.py')}
    config = dict(algorithm=ALGORITHM, schema_version=manifest['schema_version'], device=str(device),
                  action_contract=manifest['action_contract'],
                  observed_action_purposes=manifest['observed_action_purposes'],
                  completion_event_semantics=COMPLETION_EVENT_SEMANTICS,
                  batch_size=batch_size, sequence_length=sequence_length, lr=lr,
                  weight_decay=1e-4, gradient_clip=1., seed=seed, bootstrap=bootstrap,
                  hidden=hidden, latent=latent, max_train_windows=max_train_windows,
                  rollout_steps=rollout_steps, rollout_weight=rollout_weight,
                  source_hashes=sources)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = (make_model(dict(algorithm=ALGORITHM, hidden=hidden, latent=latent)) if __package__
             else Vtmp2WorldModel(hidden=hidden, latent=latent)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config['weight_decay'])
    first_epoch, global_step, best_loss, best_epoch, history = 1, 0, float('inf'), 0, []
    if resume:
        checkpoint = torch.load(output/'latest.pt', map_location='cpu', weights_only=False)
        if checkpoint.get('format') != CHECKPOINT_FORMAT or checkpoint.get('config') != config:
            raise ValueError('Resume configuration/source hashes do not match the checkpoint')
        if checkpoint.get('dataset_sha256') != manifest['sha256']:
            raise ValueError('Resume dataset hashes/splits do not match the checkpoint')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        first_epoch = checkpoint['epoch'] + 1
        global_step, best_loss, best_epoch = checkpoint['global_step'], checkpoint['best_val_loss'], checkpoint['best_epoch']
        history = checkpoint['history']
        _restore_rng(checkpoint['rng'], device)
        write_metrics(output/'metrics.jsonl', history)
    atomic_json(output/'dataset_manifest.json', manifest)
    atomic_json(output/'dataset_statistics.json', statistics)
    atomic_json(output/'config.json', config)
    validation_loader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    if not resume:
        status('initial_validation', train_statistics=statistics['train'], val_statistics=statistics['val'])
        atomic_json(output/'initial_validation.json', evaluate(model, validation_loader, device))
    for epoch in range(first_epoch, epochs+1):
        model.train()
        generator = torch.Generator().manual_seed(seed+epoch)
        training = datasets['train']
        if max_train_windows is not None and max_train_windows < len(training):
            indices = torch.randperm(len(training), generator=generator)[:max_train_windows].tolist()
            training = Subset(training, indices)
        loader = DataLoader(training, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
        total_loss, transitions, largest_gradient = 0., 0., 0.
        epoch_started = time.time()
        status('training', epoch=epoch, epochs=epochs, batch=0, batches=len(loader),
               global_step=global_step, optimizer_steps=global_step)
        for index, batch in enumerate(loader, start=1):
            batch = _on_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = sequence_loss(model, batch, rollout_steps=rollout_steps, rollout_weight=rollout_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            weight = float(batch['mask'].sum())
            total_loss += float(loss.detach())*weight
            transitions += weight
            largest_gradient = max(largest_gradient, float(gradient))
            global_step += 1
            if index % 10 == 0 or index == len(loader):
                status('training', epoch=epoch, epochs=epochs, batch=index, batches=len(loader),
                       global_step=global_step, optimizer_steps=global_step,
                       train_loss=total_loss/transitions, max_gradient_norm=largest_gradient)
        status('validation', epoch=epoch, epochs=epochs, global_step=global_step)
        validation = evaluate(model, validation_loader, device)
        improved = validation['loss'] < best_loss
        if improved:
            best_loss, best_epoch = validation['loss'], epoch
        row = dict(epoch=epoch, global_step=global_step, optimizer_steps=global_step, train_loss=total_loss/transitions,
                   max_gradient_norm=largest_gradient, train_transitions=int(transitions),
                   val=validation, best_epoch=best_epoch, best_val_loss=best_loss,
                   epoch_seconds=time.time()-epoch_started)
        history.append(row)
        checkpoint = dict(format=CHECKPOINT_FORMAT, model=model.state_dict(), optimizer=optimizer.state_dict(),
                          config=config, epoch=epoch, global_step=global_step, best_val_loss=best_loss,
                          best_epoch=best_epoch, dataset_sha256=manifest['sha256'], dataset_manifest=manifest,
                          rng=_rng_state(device), history=history)
        atomic_checkpoint(output/'latest.pt', checkpoint)
        if improved:
            atomic_checkpoint(output/'best.pt', checkpoint)
        write_metrics(output/'metrics.jsonl', history)
        print(json.dumps(row, allow_nan=False), flush=True)
    status('rollout_validation', epoch=history[-1]['epoch'], global_step=global_step)
    latest_rollout = evaluate_rollout(model, validation_loader, device)
    if best_epoch == history[-1]['epoch']:
        best_rollout = latest_rollout
    else:
        best_checkpoint = torch.load(output/'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(best_checkpoint['model'])
        best_rollout = evaluate_rollout(model, validation_loader, device)
    rollout_validation = output/'rollout_validation.json'
    atomic_json(rollout_validation, dict(latest_epoch=history[-1]['epoch'], best_epoch=best_epoch,
                                       latest=latest_rollout, best=best_rollout))
    summary = dict(algorithm=ALGORITHM, stage='complete', bootstrap=bootstrap,
                   deployment_enabled=False, epoch=history[-1]['epoch'], global_step=global_step,
                   optimizer_steps=global_step,
                   best_epoch=best_epoch, best_val_loss=best_loss,
                   latest_checkpoint=str(output/'latest.pt'), best_checkpoint=str(output/'best.pt'),
                   rollout_validation=str(rollout_validation),
                   dataset_sha256=manifest['sha256'], wall_seconds=time.time()-started)
    atomic_json(output/'summary.json', summary)
    status('complete', **{key: value for key, value in summary.items()
                         if key not in ('algorithm', 'stage', 'bootstrap', 'deployment_enabled')})
    return summary


def run_training(dataset_dir, output_dir, *, device='cuda', epochs=100, batch_size=32,
                 sequence_length=16, lr=3e-4, seed=20261107, resume=False,
                 bootstrap=False, hidden=128, latent=24, max_train_windows=None,
                 rollout_steps=4, rollout_weight=.25):
    """Fit a predictor; epochs is the cumulative target, including resume epochs."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.train.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another trainer owns this output directory') from exc
        try:
            return _train(dataset_dir, output, device=device, epochs=epochs, batch_size=batch_size,
                          sequence_length=sequence_length, lr=lr, seed=seed, resume=resume,
                          bootstrap=bootstrap, hidden=hidden, latent=latent, max_train_windows=max_train_windows,
                          rollout_steps=rollout_steps, rollout_weight=rollout_weight)
        except Exception as exc:
            atomic_json(output/'status.json', dict(algorithm=ALGORITHM, stage='failed', pid=os.getpid(),
                        updated_at=time.time(), error=f'{type(exc).__name__}: {exc}'))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sequence-length', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--latent', type=int, default=24)
    parser.add_argument('--max-train-windows', type=int)
    parser.add_argument('--rollout-steps', type=int, default=4)
    parser.add_argument('--rollout-weight', type=float, default=.25)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--bootstrap', action='store_true', help='Explicitly label a small preliminary run')
    args = parser.parse_args()
    print(json.dumps(run_training(args.dataset, args.output, device=args.device, epochs=args.epochs,
          batch_size=args.batch_size, sequence_length=args.sequence_length, lr=args.lr, seed=args.seed,
          resume=args.resume, bootstrap=args.bootstrap, hidden=args.hidden, latent=args.latent,
          max_train_windows=args.max_train_windows, rollout_steps=args.rollout_steps,
          rollout_weight=args.rollout_weight), indent=2), flush=True)


if __name__ == '__main__':
    main()
