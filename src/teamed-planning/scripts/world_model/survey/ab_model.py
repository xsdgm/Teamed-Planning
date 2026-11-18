"""A/B survey dynamics, supervised training and checkpoint loading."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .ab_environment import ACTION_CONTRACT, ACTIONS, ALGORITHM, CHANNELS, DELTA_SCALE, FEATURES, SIDE
from .hierarchical_sources import atomic_json as _write_json, file_sha256
from .models import RecurrentWorldModel, balanced_kl, imagine_sequence, make_model, padded_window

CHECKPOINT_FORMAT = 'vtmp3-ab-world-model-v1'


class ABWorldModel(RecurrentWorldModel):
    algorithm = 'vtmp3'
    model_architecture = 'rssm_ab_vtmp3'

    def __init__(self, hidden=128, latent=24):
        super().__init__(channels=CHANNELS, side=SIDE, features=FEATURES,
                         actions=ACTIONS, outcomes=5, delta_scale=DELTA_SCALE,
                         hidden=hidden, latent=latent, state_head_name='state_head')

    def observe(self, hidden, maps, states, sample=False):
        return self.observe_embedding(hidden, self.encoder(maps), states, sample)

    def posterior_input(self, maps):
        return maps


WorldModel = ABWorldModel


def prediction_loss(model, pm, ps, outcome, maps, states, before_maps, labels):
    learnable = model.delta_scale > 0
    state = (((ps-states)[:, learnable]/model.delta_scale[learnable])**2).mean(-1)
    weights = 1+4*(maps > 0).float()+10*(abs(maps-before_maps) > .01).float()
    grid = ((pm-maps).square()*weights).mean((1, 2, 3))
    continuous = F.smooth_l1_loss(outcome[:, :3], labels[:, :3], reduction='none').mean(-1)
    events = F.binary_cross_entropy_with_logits(outcome[:, 3:], labels[:, 3:], reduction='none').mean(-1)
    return state+5*grid+continuous+.2*events


def sequence_loss(model, batch, rollout_steps=4, rollout_weight=.25, sample=True):
    maps, states, actions, labels, mask = (batch[k] for k in ('maps', 'states', 'actions', 'outcomes', 'mask'))
    hidden, latent = model.initial(maps[:, 0], states[:, 0], sample)
    losses = []
    for t in range(actions.shape[1]):
        hidden, _, prior, pm, ps, outcome = model.imagine(hidden, latent, actions[:, t], maps[:, t], states[:, t], sample)
        loss = prediction_loss(model, pm, ps, outcome, maps[:, t+1], states[:, t+1], maps[:, t], labels[:, t])
        latent, posterior = model.observe(hidden, maps[:, t+1], states[:, t+1], sample)
        losses.append(loss + .01 * balanced_kl(posterior, prior))
    loss = (torch.stack(losses, 1)*mask).sum()/mask.sum().clamp_min(1)
    if rollout_weight:
        rollout, valid = [], mask.cumprod(1)
        predictions = imagine_sequence(model, maps[:, 0], states[:, 0], actions, rollout_steps, sample)
        for t, (pm, ps, outcome) in enumerate(predictions):
            rollout.append(prediction_loss(model, pm, ps, outcome, maps[:, t+1], states[:, t+1], maps[:, t], labels[:, t]))
        m = valid[:, :len(rollout)]
        loss += rollout_weight*(torch.stack(rollout, 1)*m).sum()/m.sum().clamp_min(1)
    return loss


def atomic_json(path, document):
    _write_json(path, document)


def load_episodes(root):
    splits, seeds, geometries, manifest = {'train': [], 'val': []}, set(), {}, []
    terrain_regions = []
    for split in splits:
        for path in sorted(Path(root).glob(f'{split}_*.npz')):
            with np.load(path, allow_pickle=False) as source:
                episode = {key: source[key] for key in source.files}
            if episode['algorithm'].item() != ALGORITHM or episode['action_contract'].item() != ACTION_CONTRACT:
                raise ValueError('Wrong algorithm or action contract')
            if episode['split'].item() != split:
                raise ValueError('Dataset split mismatch')
            seed, geometry = int(episode['seed']), str(episode['geometry_sha256'].item())
            if seed in seeds or (geometry in geometries and geometries[geometry] != split):
                raise ValueError('Duplicate seed or physical geometry across dataset splits')
            seeds.add(seed); geometries[geometry] = split
            if 'terrain_region_id' in episode:
                source_sha = str(episode['terrain_source_sha256'].item())
                origin = np.asarray(episode['terrain_origin_xy'], float)
                world = float(episode['terrain_world_m'])
                if origin.shape != (2,) or not np.isfinite(origin).all() or not np.isfinite(world) or world <= 0:
                    raise ValueError('Invalid terrain region metadata')
                for old_split, old_sha, old_origin, old_world in terrain_regions:
                    if old_split != split and source_sha == old_sha and (
                            np.minimum(origin+world, old_origin+old_world) > np.maximum(origin, old_origin)).all():
                        raise ValueError('WildScenes terrain regions overlap across dataset splits')
                terrain_regions.append((split, source_sha, origin, world))
            n = len(episode['actions'])
            if not n:
                raise ValueError('Empty episode')
            shapes = dict(maps=(n+1, CHANNELS, SIDE, SIDE), states=(n+1, FEATURES),
                          actions=(n, ACTIONS), outcomes=(n, 5), terminated=(n,), truncated=(n,))
            for key, shape in shapes.items():
                if episode[key].shape != shape or not np.isfinite(episode[key]).all():
                    raise ValueError(f'Invalid {key} in {path.name}')
            if not np.isin(episode['outcomes'][:, 3:], (0, 1)).all():
                raise ValueError('Invalid event labels')
            if ((episode['maps'] < 0) | (episode['maps'] > 1)).any():
                raise ValueError('Map values outside [0, 1]')
            if not np.allclose(episode['states'][:, 2:4], episode['states'][0, 2:4]):
                raise ValueError('Final goal must be constant')
            if np.any(np.diff(episode['states'][:, 5]) < 0):
                raise ValueError('Elapsed time decreased')
            if episode['terminated'][:-1].any() or episode['truncated'][:-1].any():
                raise ValueError('Transitions after an episode boundary')
            if bool(episode['terminated'][-1]) == bool(episode['truncated'][-1]):
                raise ValueError('Last transition needs exactly one boundary flag')
            splits[split].append(episode)
            manifest.append(dict(path=path.name, sha256=file_sha256(path), seed=seed, split=split))
        if not splits[split]:
            raise ValueError(f'No {split} episodes')
    return splits, manifest


class Windows(Dataset):
    def __init__(self, episodes, length=8):
        self.episodes, self.length = episodes, length
        self.indices = [(i, start) for i, e in enumerate(episodes)
                        for start in range(0, len(e['actions']), max(1, length//2))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        ep, start = self.indices[index]
        episode = self.episodes[ep]
        return padded_window(episode, start, self.length, ('actions', 'outcomes'),
                             observation_dtype=np.float32)


@torch.no_grad()
def evaluate(model, episodes, device):
    model.eval()
    sums = dict(position_error_m=0., unchanged_position_error_m=0., map_mse=0., unchanged_map_mse=0.,
                duration_absolute_error_s=0., failure_brier=0.)
    samples = 0
    rollout = {1: [], 2: [], 4: []}
    for ep in episodes:
        maps, states, actions, labels = [torch.as_tensor(ep[k], dtype=torch.float32, device=device)
                                       for k in ('maps', 'states', 'actions', 'outcomes')]
        hidden, latent = model.initial(maps[:1], states[:1])
        for t in range(len(actions)):
            hidden, _, _, pm, ps, outcome = model.imagine(hidden, latent, actions[t:t+1], maps[t:t+1], states[t:t+1])
            sums['position_error_m'] += float(torch.linalg.vector_norm((ps[:, :2]-states[t+1:t+2, :2])*1000.))
            sums['unchanged_position_error_m'] += float(torch.linalg.vector_norm((states[t:t+1, :2]-states[t+1:t+2, :2])*1000.))
            sums['map_mse'] += float((pm-maps[t+1:t+2]).square().mean())
            sums['unchanged_map_mse'] += float((maps[t:t+1]-maps[t+1:t+2]).square().mean())
            sums['duration_absolute_error_s'] += float(abs(outcome[0, 2]-labels[t, 2])*200.)
            sums['failure_brier'] += float((outcome[0, 3].sigmoid()-labels[t, 3]).square())
            latent, _ = model.observe(hidden, maps[t+1:t+2], states[t+1:t+2])
            samples += 1
        # Free rollouts use future ground truth only to measure error.
        for start in range(0, len(actions), 4):
            predictions = imagine_sequence(model, maps[start:start+1], states[start:start+1],
                                           actions[start:start+4].unsqueeze(0), 4)
            for offset, (_, ps, _) in enumerate(predictions):
                if offset+1 in rollout:
                    target = states[start+offset+1:start+offset+2, :2]
                    rollout[offset+1].append((float(torch.linalg.vector_norm((ps[:, :2]-target)*1000.)),
                        float(torch.linalg.vector_norm((states[start:start+1, :2]-target)*1000.))))
    return dict(samples=samples, **{k: v/max(samples, 1) for k, v in sums.items()},
        free_rollout={str(k): dict(windows=len(v), position_error_m=float(np.mean([a for a, b in v])) if v else None,
            unchanged_position_error_m=float(np.mean([b for a, b in v])) if v else None) for k, v in rollout.items()})


def load_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != CHECKPOINT_FORMAT or checkpoint.get('action_contract') != ACTION_CONTRACT:
        raise ValueError('Checkpoint is not an A/B-conditioned vtmp3 model')
    model = make_model(dict(algorithm=checkpoint.get('algorithm', ALGORITHM),
        model_architecture=checkpoint.get('model_architecture', 'rssm_ab_vtmp3'), **checkpoint['architecture']))
    model.load_state_dict(checkpoint['model'], strict=True)
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('Non-finite checkpoint')
    return model.to(device).eval(), checkpoint


def train(dataset, output, *, epochs=50, device='cpu', model=None, seed=20261110,
          mapping_config=None, source_manifest_sha256=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    episodes, manifest = load_episodes(dataset)
    torch.manual_seed(seed)
    model = make_model(dict(algorithm=ALGORITHM)) if model is None else model
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loader = DataLoader(Windows(episodes['train']), batch_size=16, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    validation = DataLoader(Windows(episodes['val']), batch_size=16, shuffle=False, num_workers=0)
    atomic_json(output/'dataset_manifest.json', manifest)
    data_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best, steps, start = float('inf'), 0, time.time()
    metrics_path = output/'metrics.jsonl'
    for epoch in range(1, epochs+1):
        model.train(); losses, norms = [], []
        for raw in loader:
            batch = {k: v.to(device) for k, v in raw.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = sequence_loss(model, batch)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); steps += 1
            losses.append(float(loss.detach())); norms.append(float(norm))
        model.eval(); val_sum, val_count = 0., 0
        with torch.no_grad():
            for raw in validation:
                batch = {k: v.to(device) for k, v in raw.items()}
                n = float(batch['mask'].sum())
                val_sum += float(sequence_loss(model, batch, sample=False))*n; val_count += n
        val_loss = val_sum/max(val_count, 1.)
        improved = val_loss < best
        best = min(best, val_loss)
        row = dict(epoch=epoch, optimizer_steps=steps, train_loss=float(np.mean(losses)),
                   val_loss=val_loss, best_val_loss=best, grad_norm=float(np.mean(norms)))
        with metrics_path.open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        checkpoint = dict(format=CHECKPOINT_FORMAT, algorithm=ALGORITHM, action_contract=ACTION_CONTRACT,
            model_architecture=model.model_architecture,
            architecture=dict(hidden=model.hidden, latent=model.latent), model=model.state_dict(),
            optimizer=optimizer.state_dict(), torch_rng_state=torch.get_rng_state(),
            dataset_sha256=data_hash, mapping_config=mapping_config,
            source_manifest_sha256=source_manifest_sha256, **row)
        for filename in (['latest.pt', 'best.pt'] if improved else ['latest.pt']):
            temporary = output/(filename+'.tmp'); torch.save(checkpoint, temporary); temporary.replace(output/filename)
        atomic_json(output/'status.json', dict(stage='training', **row))
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(json.dumps(dict(stage='training', **row)), flush=True)
    parameter_delta = sum(float((v.detach().cpu()-before[k]).square().sum()) for k, v in model.state_dict().items())**.5
    best_model, checkpoint = load_model(output/'best.pt', device)
    validation_metrics = evaluate(best_model, episodes['val'], device)
    atomic_json(output/'validation.json', validation_metrics)
    result = dict(stage='complete', epochs=epochs, optimizer_steps=steps, parameter_delta_l2=parameter_delta,
                  best_epoch=checkpoint['epoch'], best_val_loss=best, wall_seconds=time.time()-start,
                  checkpoint=str(output/'best.pt'), validation=validation_metrics)
    atomic_json(output/'status.json', result)
    return best_model, result
