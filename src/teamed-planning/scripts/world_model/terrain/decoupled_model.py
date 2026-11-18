"""Terrain completion and action-conditioned information/motion prediction.

An ensemble represents disagreement between predictors; Bernoulli entropy and
ensemble spread are diagnostics, not calibrated collision/reachability bounds.
The persistent received map supplies history for this static-terrain model.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .decoupled_observation import (ACTION_CONTRACT, ARCHITECTURE, CHANNELS, CHECKPOINT_FORMAT,
                                    DRONE_FEATURES, GROUND_FEATURES, HALF_EXTENT, SCHEMA_VERSION,
                                    SIDE, VEHICLE_RUNTIME, motion_state)
from .hierarchical_sources import atomic_json, file_sha256


class _Predictor(nn.Module):
    def __init__(self, features, motions, hidden):
        super().__init__()
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, SIDE), torch.linspace(-1, 1, SIDE), indexing='ij')
        self.register_buffer('coordinates', torch.stack((xx, yy))[None])
        self.encoder = nn.Sequential(nn.Conv2d(CHANNELS+2, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1), nn.SiLU())
        self.state = nn.Sequential(nn.Linear(features, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.command = nn.Sequential(nn.Linear(features+7, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.terrain = nn.Sequential(nn.Conv2d(hidden+CHANNELS, hidden, 3, padding=1),
                                     nn.SiLU(), nn.Conv2d(hidden, 1, 1))
        self.information = nn.Sequential(nn.Conv2d(hidden+CHANNELS, hidden, 3, padding=1),
                                         nn.SiLU(), nn.Conv2d(hidden, 1, 1))
        self.motion = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, motions))
        nn.init.constant_(self.information[-1].bias, -4.)

    def forward(self, maps, states, actions, durations):
        encoded = self.encoder(torch.cat((maps, self.coordinates.expand(len(maps), -1, -1, -1)), 1))
        relative_states = states.clone()
        relative_states[:, :2] = 0.
        relative_states[:, 6:8] = (states[:, 6:8]-states[:, :2])*5
        relative_actions = actions.clone()
        for offset in (0, 3):
            relative_actions[:, offset:offset+2] = (actions[:, offset:offset+2]-states[:, :2])*5*actions[:, offset+2:offset+3]
        terrain = encoded + self.state(relative_states)[..., None, None]
        command = self.command(torch.cat((relative_states, relative_actions, durations[:, None]/10.), -1))
        future = encoded + command[..., None, None]
        terrain = self.terrain(torch.cat((F.interpolate(terrain, (SIDE, SIDE), mode='bilinear', align_corners=False), maps), 1))[:, 0]
        information = self.information(torch.cat((F.interpolate(future, (SIDE, SIDE), mode='bilinear', align_corners=False), maps), 1))[:, 0]
        return terrain, information, self.motion(future.mean((2, 3)))


class DecoupledWorldModel(nn.Module):
    algorithm = 'vtmp2'
    model_architecture = ARCHITECTURE

    def __init__(self, drone_count=3, hidden=24, members=3):
        super().__init__()
        if (type(drone_count) is not int or not 0 <= drone_count <= 32
                or type(hidden) is not int or not 4 <= hidden <= 256
                or type(members) is not int or not 1 <= members <= 8):
            raise ValueError('Invalid terrain-model dimensions')
        self.drone_count, self.hidden, self.members = drone_count, hidden, members
        self.features = GROUND_FEATURES + DRONE_FEATURES*drone_count
        self.predictors = nn.ModuleList(_Predictor(self.features, 2+3*drone_count, hidden)
                                        for _ in range(members))

    def configuration(self):
        return dict(algorithm=self.algorithm, model_architecture=self.model_architecture,
                    drone_count=self.drone_count, hidden=self.hidden, members=self.members)

    def forward(self, maps, states, actions, durations):
        predictions = [member(maps, states, actions, durations) for member in self.predictors]
        return {name: torch.stack([result[i] for result in predictions])
                for i, name in enumerate(('terrain_logits', 'information_logits', 'motion'))}

    @torch.no_grad()
    def predict(self, maps, states, actions, durations):
        outputs = self(maps, states, actions, durations)
        terrain = outputs['terrain_logits'].sigmoid()
        # Keep received evidence and the surveyed boundary authoritative. These
        # tensors are private planning copies and never update the real map.
        terrain = torch.where(maps[None, :, 1].bool(), maps[None, :, 0], terrain)
        terrain = torch.where(maps[None, :, 2].bool(), torch.ones_like(terrain), terrain)
        probability = terrain.mean(0)
        p = probability.clamp(1e-6, 1-1e-6)
        entropy = -(p*p.log()+(1-p)*(1-p).log())
        unknown = (1-maps[:, 1])*(1-maps[:, 2])
        return dict(occupancy=probability, disagreement=terrain.std(0, unbiased=False),
                    entropy=entropy*unknown,
                    information=outputs['information_logits'].sigmoid().mean(0)*unknown,
                    motion=outputs['motion'].mean(0))


def prediction_loss(outputs, batch, *, bootstrap=False):
    """Hidden terrain labels supervise completion, never enter model inputs."""
    mask = (1-batch['maps'][:, 1])*(1-batch['maps'][:, 2])
    losses = []
    for i in range(len(outputs['terrain_logits'])):
        terrain = F.binary_cross_entropy_with_logits(outputs['terrain_logits'][i], batch['terrain'], reduction='none')
        terrain = (terrain*mask).sum((1, 2))/mask.sum((1, 2)).clamp_min(1)
        # Only currently unknown, in-bounds cells can become new observations.
        gain = F.binary_cross_entropy_with_logits(outputs['information_logits'][i], batch['newly_observed'], reduction='none')
        gain = (gain*mask).sum((1, 2))/mask.sum((1, 2)).clamp_min(1)
        motion = ((outputs['motion'][i]-batch['motion']).square()*batch['motion_mask']).sum(-1)
        motion = motion/batch['motion_mask'].sum(-1).clamp_min(1)
        weight = torch.randint(0, 3, terrain.shape, device=terrain.device).float() if bootstrap else torch.ones_like(terrain)
        losses.append(((terrain+gain+.1*motion)*weight).sum()/weight.sum().clamp_min(1))
    return torch.stack(losses).mean()


class TransitionDataset(Dataset):
    def __init__(self, episodes):
        self.episodes = episodes
        self.indices = [(e, t) for e, episode in enumerate(episodes) for t in range(len(episode['actions']))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        e, t = self.indices[index]
        episode = self.episodes[e]
        return {key: episode[key][t].astype(np.float32) for key in
                ('maps', 'states', 'actions', 'durations', 'terrain', 'newly_observed', 'motion', 'motion_mask')}


def load_dataset(directory, drone_count=3, mmap_dir=None):
    from .data import CHECKPOINT_SHA
    from .decoupled_scenes import regions_overlap

    episodes = {'train': [], 'val': []}
    seeds, geometries, manifest = set(), {}, []
    vehicle_runtime = None
    total_bytes = 0
    for split in episodes:
        for path in sorted(Path(directory).glob(f'{split}_*.npz')):
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            for key, expected in (('algorithm', 'vtmp2'), ('schema_version', SCHEMA_VERSION),
                    ('action_contract', ACTION_CONTRACT), ('drone_count', drone_count),
                    ('checkpoint_sha256', CHECKPOINT_SHA), ('split', split), ('feasibility_validated', True)):
                if key not in data or data[key].shape != () or data[key].item() != expected:
                    raise ValueError(f'{path.name}: incompatible {key}')
            seed, geometry = data['seed'].item(), data['scene_geometry_sha256'].item()
            if (type(seed) is not int or seed in seeds or not isinstance(geometry, str)
                    or len(geometry) != 64 or any(c not in '0123456789abcdef' for c in geometry)
                    or geometry in geometries and geometries[geometry] != split):
                raise ValueError('Duplicate seed or scene leakage between training and validation')
            seeds.add(seed)
            geometries[geometry] = split
            n = len(data['actions'])
            shapes = dict(maps=(n+1, CHANNELS, SIDE, SIDE),
                states=(n+1, GROUND_FEATURES+drone_count*DRONE_FEATURES), origins=(n+1, 2),
                actions=(n, 6), durations=(n,), terrain=(n, SIDE, SIDE),
                newly_observed=(n, SIDE, SIDE), motion=(n, 2+3*drone_count),
                motion_mask=(n, 2+3*drone_count), times=(n+1,), terminated=(n,), truncated=(n,))
            if not n:
                raise ValueError('Empty collection episode')
            for key, shape in shapes.items():
                if key not in data or data[key].shape != shape or not np.isfinite(data[key]).all():
                    raise ValueError(f'{path.name}: invalid {key}')
            for key in ('maps', 'terrain', 'newly_observed', 'motion_mask'):
                if np.any((data[key] < 0) | (data[key] > 1)):
                    raise ValueError(f'{key} must be in [0, 1]')
            for key in ('terminated', 'truncated'):
                if not np.isin(data[key], (0, 1)).all() or data[key][:-1].any():
                    raise ValueError('Invalid episode termination markers')
            if bool(data['terminated'][-1]) == bool(data['truncated'][-1]):
                raise ValueError('Last transition must terminate or truncate')
            if (not np.allclose(np.diff(data['times']), data['durations'], atol=1e-6)
                    or np.any(data['durations'] < 0) or np.any(data['durations'] > 10)
                    or np.any((data['durations'] == 0) & ~data['terminated'].astype(bool))):
                raise ValueError('Transition duration disagrees with timestamps')
            for start, state_start in ((0, 8), (3, 11)):
                enabled = data['actions'][:, start+2]
                if (not np.isin(enabled, (0, 1)).all()
                        or np.any(data['actions'][enabled == 0, start:start+2] != 0)):
                    raise ValueError('Invalid independent action mask')
                recorded = data['states'][:-1, state_start:state_start+3]
                target = data['states'][:-1, :2]+recorded[:, :2]*HALF_EXTENT/1000.
                if (not np.array_equal(enabled, recorded[:, 2])
                        or not np.allclose(data['actions'][enabled == 1, start:start+2],
                                           target[enabled == 1], atol=2e-6, rtol=0)):
                    raise ValueError('Action disagrees with the independently recorded command')
            expected_motion = np.diff(motion_state(data['states'], drone_count), axis=0)/.01
            if not np.allclose(data['motion'], expected_motion, atol=1e-4, rtol=0):
                raise ValueError('Motion labels disagree with received states')
            if np.any(data['newly_observed']*data['maps'][:-1, 1] > 0):
                raise ValueError('Previously observed cells cannot count as new information')
            if not np.allclose(data['states'][:, 6:8], data['states'][0, 6:8]):
                raise ValueError('Final goal changed inside an episode')
            episode_bytes = sum(value.nbytes for value in data.values())
            total_bytes += episode_bytes
            if episode_bytes > 1024**3 or mmap_dir is None and total_bytes > 1024**3:
                raise ValueError('Preload exceeds one GiB; use memory-mapped training')
            if mmap_dir is not None:
                # NPZ is compressed and cannot be memory-mapped. Materialize
                # validated arrays once; shuffled mini-batches then page only
                # the rows they need instead of preloading the entire dataset.
                cache = Path(mmap_dir)/path.stem
                cache.mkdir(parents=True, exist_ok=False)
                for key in shapes:
                    target = cache/(key+'.npy')
                    np.save(target, data[key], allow_pickle=False)
                    data[key] = np.load(target, mmap_mode='r', allow_pickle=False)
            episodes[split].append(data)
            record = dict(file=path.name, split=split, seed=seed, transitions=n,
                geometry_sha256=geometry, sha256=file_sha256(path))
            runtime = data.get('vehicle_runtime', np.asarray('archived_v9_ground')).item()
            if (runtime not in ('archived_v9_ground', 'vtmp_with_observed_route_recovery', VEHICLE_RUNTIME)
                    or vehicle_runtime is not None and vehicle_runtime != runtime):
                raise ValueError('Cannot mix different vehicle runtime contracts in one training dataset')
            vehicle_runtime = record['vehicle_runtime'] = runtime
            backend = data.get('terrain_backend', np.asarray('circles')).item()
            if backend not in ('circles', 'wildscenes_simplified'):
                raise ValueError('Unsupported terrain backend')
            record['terrain_backend'] = backend
            if backend == 'wildscenes_simplified':
                for key in ('terrain_source_sha256', 'terrain_map_id', 'terrain_region_id', 'terrain_split', 'terrain_world_m'):
                    record[key] = data[key].item()
                origin = data['terrain_origin_xy']
                if (origin.shape != (2,) or not np.isfinite(origin).all() or (origin < 0).any()
                        or not math.isfinite(record['terrain_world_m']) or record['terrain_world_m'] <= 0
                        or record['terrain_split'] != split
                        or not isinstance(record['terrain_source_sha256'], str)
                        or len(record['terrain_source_sha256']) != 64
                        or any(c not in '0123456789abcdef' for c in record['terrain_source_sha256'])
                        or not isinstance(record['terrain_map_id'], str) or not record['terrain_map_id']):
                    raise ValueError('Invalid or mismatched WildScenes training region')
                record['terrain_origin_xy'] = origin.tolist()
                if any(item['split'] != split and regions_overlap(record, item) for item in manifest):
                    raise ValueError('WildScenes terrain regions overlap across dataset splits')
            manifest.append(record)
        if not episodes[split]:
            raise ValueError(f'No {split} episodes')
    return episodes, manifest


@torch.no_grad()
def evaluate(model, loader, device, prior):
    total_loss = count = error = baseline = cells = gain_error = 0.
    model.eval()
    for raw in loader:
        batch = {key: value.to(device) for key, value in raw.items()}
        output = model(batch['maps'], batch['states'], batch['actions'], batch['durations'])
        total_loss += float(prediction_loss(output, batch))*len(batch['maps'])
        count += len(batch['maps'])
        mask = (1-batch['maps'][:, 1])*(1-batch['maps'][:, 2])
        predicted = output['terrain_logits'].sigmoid().mean(0)
        error += float(((predicted-batch['terrain']).square()*mask).sum())
        baseline += float(((prior-batch['terrain']).square()*mask).sum())
        gain_error += float(((output['information_logits'].sigmoid().mean(0)-batch['newly_observed']).square()*mask).sum())
        cells += float(mask.sum())
    return dict(loss=total_loss/max(count, 1), unknown_terrain_brier=error/max(cells, 1),
                training_prior_brier=baseline/max(cells, 1), newly_observed_brier=gain_error/max(cells, 1),
                unknown_cells=int(cells), transitions=int(count))


def train_model(dataset_dir, output, *, epochs=20, batch_size=16, learning_rate=3e-4,
                hidden=24, members=3, drone_count=3, seed=20261112, device='cpu'):
    from .data import CHECKPOINT_SHA

    if (any(type(v) is not int or v < 1 for v in (epochs, batch_size))
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('Positive epochs, batch size and learning rate required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    episodes, manifest = load_dataset(dataset_dir, drone_count, mmap_dir=output/'cache')
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = DecoupledWorldModel(drone_count=drone_count, hidden=hidden, members=members).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train = DataLoader(TransitionDataset(episodes['train']), batch_size=batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(seed))
    val = DataLoader(TransitionDataset(episodes['val']), batch_size=batch_size)
    # The baseline prior is estimated on training labels only.
    positives = total = 0.
    for episode in episodes['train']:
        mask = (1-episode['maps'][:-1, 1])*(1-episode['maps'][:-1, 2])
        positives += float((episode['terrain']*mask).sum())
        total += float(mask.sum())
    prior = positives/max(total, 1)
    atomic_json(output/'dataset.json', manifest)
    config = dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                  seed=seed, prior=prior, device=str(device), **model.configuration())
    atomic_json(output/'config.json', config)
    history, best = [], float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        summed = count = 0.
        for raw in train:
            batch = {key: value.to(device) for key, value in raw.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch['maps'], batch['states'], batch['actions'], batch['durations'])
            loss = prediction_loss(outputs, batch, bootstrap=True)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            summed += float(loss.detach())*len(batch['maps'])
            count += len(batch['maps'])
        metrics = dict(epoch=epoch, train_loss=summed/max(count, 1),
                       validation=evaluate(model, val, device, prior))
        history.append(metrics)
        atomic_json(output/'metrics.json', history)
        checkpoint = dict(format=CHECKPOINT_FORMAT, schema_version=SCHEMA_VERSION,
            action_contract=ACTION_CONTRACT, model_architecture=ARCHITECTURE,
            algorithm='vtmp2', epoch=epoch, validation=metrics['validation'],
            architecture=dict(drone_count=drone_count, hidden=hidden, members=members),
            v9_checkpoint_sha256=CHECKPOINT_SHA,
            vehicle_runtime=manifest[0]['vehicle_runtime'],
            dataset=manifest,
            model={name: tensor.detach().cpu() for name, tensor in model.state_dict().items()})
        for name in ('last.pt', 'best.pt'):
            if name == 'best.pt' and metrics['validation']['loss'] >= best:
                continue
            temporary = output/(name+'.tmp')
            torch.save(checkpoint, temporary)
            temporary.replace(output/name)
        best = min(best, metrics['validation']['loss'])
    selected = min(history, key=lambda row: row['validation']['loss'])
    return dict(checkpoint=str(output/'best.pt'), epochs=epochs, best_epoch=selected['epoch'],
                best_validation_loss=best, validation=selected['validation'], v9_weights_trained=False)
