"""Stage 1: train a shared local world model; keep both control layers fixed.

The frozen v9 fast actor is also the actor prior. Slow tasks are sampled from
an explicitly recorded geometry/random mixture, not an updated slow policy.
This entry point does not yet feed predictions into either control policy.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))
import numpy as np
import torch
from vtemp2_3.scene_manifest import make_scene
from vtemp2_3.frozen_navigation import write_json, CHECKPOINT_SHA
from vtemp2_3.belief_rollouts import collect_scene
from vtemp2_3.residual_world import ResidualWorldModel, MODEL_SCHEMA, infer, objective
from vtemp2_3.telemetry import SCHEMA
from vtemp2_3.hierarchical_sources import atomic_checkpoint, snapshot_sources, source_hashes as hierarchical_source_hashes


def load_data(paths):
    chunks = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            chunks.append({k: data[k] for k in data.files})
    if not chunks:
        raise ValueError('Empty split')
    result = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    for key, value in result.items():
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite dataset field: ' + key)
    return result


def tensor_batch(data, ids, device):
    return {key: torch.as_tensor(value[ids], device=device) for key, value in data.items()}


def evaluate(model, data, device, batch_size=256):
    model.eval()
    states, spreads, actions = [], [], []
    total_loss = 0.
    with torch.inference_mode():
        for begin in range(0, len(data['history']), batch_size):
            batch = tensor_batch(data, slice(begin, begin + batch_size), device)
            prediction = infer(model, batch)
            total_loss += float(objective(prediction, batch)) * len(batch['history'])
            states.append(prediction['states'].cpu().numpy())
            spreads.append(prediction['state_std'].cpu().numpy())
            actions.append(prediction['actions'].cpu().numpy())
    predicted, std, action = np.concatenate(states), np.concatenate(spreads), np.concatenate(actions)
    truth, valid = data['target_states'], data['state_mask']
    cv = data['baseline_states'][:, :1].repeat(model.horizon + 1, axis=1)
    cv[:, :, :2] += np.arange(model.horizon + 1)[None, :, None] * cv[:, :, 2:] * (6. / 20.)
    results = dict(samples=len(predicted), objective=total_loss / len(predicted), horizons_seconds=list(range(model.horizon + 1)))
    for group, selection in [('all', np.ones(len(predicted), bool)),
                             ('ground', data['observer_ground']), ('drone', ~data['observer_ground']),
                             ('age_at_least_5s', data['frame'] - data['source_stamp'] >= 5)]:
        group_metrics = {}
        for name, values in [('learned', predicted), ('actor_prior', data['baseline_states']), ('constant_velocity', cv)]:
            errors = np.sum(((values[:, :, :2] - truth[:, :, :2]) * 20.)**2, axis=-1)
            group_metrics[name + '_position_rmse_m'] = [
                float(np.sqrt(errors[selection & valid[:, t], t].mean())) if (selection & valid[:, t]).any() else None
                for t in range(model.horizon + 1)]
        mask = valid & selection[:, None]
        if mask.any():
            coverage = np.abs(predicted[:, :, :2] - truth[:, :, :2]) <= 1.644854 * std[:, :, :2]
            group_metrics['position_marginal_90pct_coverage'] = float(coverage[mask].mean())
        results[group] = group_metrics
    mask = data['action_mask']
    for name, values in [('learned', action), ('actor_prior', data['baseline_actions'])]:
        error = np.sum(((values - data['target_actions']) * 6.)**2, -1)
        results[name + '_executed_action_rmse_mps'] = float(np.sqrt(error[mask].mean())) if mask.any() else None
    return results


def save_checkpoint(path, model, optimizer, config, epoch, updates, source_hashes):
    atomic_checkpoint(path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                    completed_epochs=epoch, optimizer_updates=updates,
                    schema=MODEL_SCHEMA, telemetry_schema=SCHEMA,
                    actor_checkpoint_sha256=CHECKPOINT_SHA, source_sha256=source_hashes,
                    torch_rng=torch.get_rng_state()))


def snapshot(args):
    output, repo = args.output.resolve(), args.repo.resolve()
    if not args.frozen:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Output exists; choose a new directory to preserve the run')
        target = output / 'runtime_sources' / 'world_model'
        snapshot_sources(SOURCE, target)
        arguments = sys.argv[1:]
        os.execv(sys.executable, [sys.executable, '-I', '-B', '-u', str(target / 'train_residual_world.py'),
                                 *arguments, '--repo', str(repo), '--output', str(output), '--frozen'])
    return output, repo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=SOURCE.parents[3])
    parser.add_argument('--output', type=Path, required=True)
    for name, default in [('workers', 4), ('train-scenes', 32), ('val-scenes', 6), ('test-scenes', 6),
                          ('wave-size', 8), ('steps', 240), ('horizon', 4), ('stride', 4),
                          ('option-steps', 24), ('epochs-per-wave', 15), ('batch-size', 256),
                          ('hidden', 64), ('seed', 15112026), ('jitter', 3)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--loss', type=float, default=.1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.1)
    parser.add_argument('--frozen', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = {k: v for k, v in vars(args).items() if k not in ('output', 'repo', 'frozen')}
    for key in ('workers', 'train_scenes', 'val_scenes', 'test_scenes', 'wave_size', 'steps',
                'horizon', 'stride', 'option_steps', 'epochs_per_wave', 'batch_size', 'hidden'):
        if config[key] < 1:
            parser.error(key + ' must be positive')
    if args.option_steps < 4 or args.jitter < 0 or not 0 <= args.loss < 1:
        parser.error('Invalid option or transport settings')
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error('Invalid GPU memory fraction')
    output, repo = snapshot(args)
    for directory in ('logs', 'data', 'episodes', 'metrics'):
        (output / directory).mkdir(exist_ok=True)
    write_json(output / 'config.json', config)
    source_hashes = hierarchical_source_hashes(SOURCE)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable; no silent CPU fallback')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    elif device.type != 'cpu':
        raise ValueError('Only CPU and CUDA are supported')
    model = ResidualWorldModel(args.hidden, args.horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    manifest = dict(stage=1, source_package='world_model', model_schema=MODEL_SCHEMA, telemetry_schema=SCHEMA, device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        parameters=sum(p.numel() for p in model.parameters()), source_sha256=source_hashes,
        actor_checkpoint_sha256=CHECKPOINT_SHA,
        training='Shared residual dynamics supervised by current and 1..H second future states/executed actions.',
        input='Per-receiver delivered histories, original timestamps, own local sensor/map proxies, frozen actor prior.',
        transport='Explicit experimental telemetry extension: own state, previous nominal action and received task ACK; bounded four-record packets, 2s period, 1..(1+jitter)s latency, independent loss; physical one-hop range links, multi-hop original stamps retained.',
        controls='v9 shared UAV actor frozen; slow targets 75% geometry, 25% random valid candidates; no control policy consumes model predictions in stage 1.',
        prior='Deterministic frozen actor queried on receiver-reconstructed remote view; missing CBF margin=0, unknown sensor pixels=-1; predict nominal motion, learn executed-motion residual.',
        distribution='Diagonal Gaussian predictive spread; empirical marginal coverage is reported, not guaranteed calibration.',
        boundaries='Truth is label-only. End-of-episode future labels are masked. Original timestamps survive forwarding. No speculative map writes.',
        split='Disjoint scene seeds across train/validation/test. Validation selects checkpoint; test used only at final evaluation.',
        limitations='Synthetic scenes, fixed fleet=3 and dt=1s; short rollout model is not a full environment/map/reward generator.')
    write_json(output / 'manifest.json', manifest)
    started = time.time(); records = []; updates = 0; epochs = 0; best = float('inf')
    rng = np.random.default_rng(args.seed)

    def status(phase, state='running', **extra):
        result = dict(state=state, phase=phase, pid=os.getpid(), device=str(device),
                      elapsed_seconds=round(time.time() - started, 2), scenes_completed=len(records),
                      samples_collected=sum(r['samples'] for r in records),
                      optimizer_updates=updates, completed_epochs=epochs, **extra)
        write_json(output / 'status.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    # Record every case before collection. No validation/test scene substitution.
    cases = {}
    for split, count, offset in [('train', args.train_scenes, 0), ('validation', args.val_scenes, 1000000),
                                 ('test', args.test_scenes, 2000000)]:
        cases[split] = [make_scene(args.seed + offset + i, (150., 300.), 2000.) for i in range(count)]
    if len({c['seed'] for group in cases.values() for c in group}) != sum(map(len, cases.values())):
        raise ValueError('Scene split overlap')
    write_json(output / 'scenes.json', cases)
    status('initializing')
    train_paths = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            def collect(split, indices):
                jobs = [(str(repo), cases[split][i], config, str(output), f'{split}-{i:04d}') for i in indices]
                futures = [pool.submit(collect_scene, job) for job in jobs]
                results = []
                for future in as_completed(futures):
                    result = future.result(); results.append(result); records.append(result)
                    write_json(output / 'collection.json', records)
                    status('collecting_' + split, latest_scene=result['label'])
                return [r['path'] for r in sorted(results, key=lambda x: x['label'])]

            validation = load_data(collect('validation', range(args.val_scenes)))
            # Baselines and untrained residual are evaluated before any update.
            write_json(output / 'metrics' / 'untrained-validation.json', evaluate(model, validation, device))
            for start in range(0, args.train_scenes, args.wave_size):
                train_paths += collect('train', range(start, min(start + args.wave_size, args.train_scenes)))
                training = load_data(train_paths)
                status('training', train_samples=len(training['history']))
                for _ in range(args.epochs_per_wave):
                    model.train(); losses = []
                    order = rng.permutation(len(training['history']))
                    for begin in range(0, len(order), args.batch_size):
                        batch = tensor_batch(training, order[begin:begin + args.batch_size], device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = objective(infer(model, batch), batch)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite world model objective')
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                        optimizer.step(); updates += 1; losses.append(float(loss.detach()))
                    epochs += 1
                    metrics = evaluate(model, validation, device)
                    metrics.update(epoch=epochs, optimizer_updates=updates,
                                   train_objective=float(np.mean(losses)), gradient_norm=float(norm),
                                   train_scenes=len(train_paths))
                    quality = float(np.mean(metrics['all']['learned_position_rmse_m']))
                    save_checkpoint(output / 'latest.pt', model, optimizer, config, epochs, updates, source_hashes)
                    if quality < best:
                        best = quality
                        save_checkpoint(output / 'best.pt', model, optimizer, config, epochs, updates, source_hashes)
                    write_json(output / 'metrics' / f'epoch-{epochs:04d}.json', metrics)
                    status('training', validation_mean_position_rmse_m=quality,
                           train_objective=metrics['train_objective'])
            test = load_data(collect('test', range(args.test_scenes)))
        saved = torch.load(output / 'best.pt', map_location=device, weights_only=True)
        model.load_state_dict(saved['model'])
        report = evaluate(model, test, device)
        report['selected_epoch'] = saved['completed_epochs']
        report['parameters_changed'] = any(not torch.equal(v.detach().cpu(), initial[k]) for k, v in model.state_dict().items())
        if not report['parameters_changed'] or updates == 0:
            raise AssertionError('Training made no parameter updates')
        write_json(output / 'test.json', report)
        status('finished', state='complete', best_checkpoint=str(output / 'best.pt'))
    except BaseException as exc:
        status('failed', state='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
