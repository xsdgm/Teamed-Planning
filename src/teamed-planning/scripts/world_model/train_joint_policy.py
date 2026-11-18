"""Stage 2: alternating on-policy fast MAPPO and supervised world-model updates.

Both are frozen for each collection batch and versioned together. Slow targets
use fixed received-map rules. Truth supervises dynamics/the centralized critic;
actors receive only local observations and causal per-receiver predictions.
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
from world_model.graph_fast_policy import make_fast, numpy_state, state_hash, ppo_update, FAST_SCHEMA
from world_model.belief_rollouts import collect_joint
from world_model.safety_context import GRAPH_SCHEMA, CBF_SCHEMA
from world_model.scene_manifest import make_scene
from world_model.train_residual_world import load_data, tensor_batch, evaluate
from world_model.residual_world import ResidualWorldModel, infer, objective, MODEL_SCHEMA
from world_model.frozen_navigation import digest, write_json, CHECKPOINT_SHA
from world_model.hierarchical_sources import snapshot_sources, source_hashes


def train_world_batch(model, optimizer, data, device, rng, epochs=2, batch_size=256):
    before = numpy_state(model); updates = 0; losses = []
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(data['history']))
        for begin in range(0, len(order), batch_size):
            batch = tensor_batch(data, order[begin:begin + batch_size], device)
            loss = objective(infer(model, batch), batch)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite world-model objective')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); updates += 1; losses.append(float(loss.detach()))
    after = numpy_state(model)
    delta = float(np.sqrt(sum(np.sum((after[k].astype(float) - before[k])**2) for k in before)))
    return dict(optimizer_steps=updates, mean_loss=float(np.mean(losses)), parameter_delta=delta,
                samples=len(data['history']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=SOURCE.parents[3])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warm-start', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.1)
    for name, default in [('workers', 4), ('updates', 96), ('episodes-per-update', 4), ('steps', 256),
                          ('option-steps', 24), ('stride', 4), ('seed', 26111500), ('jitter', 3),
                          ('ppo-epochs', 4), ('ppo-minibatch', 16), ('wm-epochs', 2),
                          ('eval-every', 8), ('eval-scenes', 3)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--loss', type=float, default=.1)
    parser.add_argument('--frozen', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ('workers', 'updates', 'episodes_per_update', 'steps', 'option_steps', 'stride',
                 'ppo_epochs', 'ppo_minibatch', 'wm_epochs', 'eval_every', 'eval_scenes'):
        if getattr(args, name) < 1: parser.error(name + ' must be positive')
    if args.option_steps < 4 or args.jitter < 0 or not 0 <= args.loss < 1:
        parser.error('Invalid option or transport settings')
    if not 0 < args.gpu_memory_fraction <= 1: parser.error('Invalid GPU memory fraction')
    output, repo, warm_path = args.output.resolve(), args.repo.resolve(), args.warm_start.resolve()
    if not args.frozen:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Output is not empty; preserve previous experiments')
        target = output / 'runtime_sources' / 'world_model'
        snapshot_sources(SOURCE, target)
        os.execv(sys.executable, [sys.executable, '-I', '-B', '-u', str(target / 'train_joint_policy.py'),
            *sys.argv[1:], '--repo', str(repo), '--output', str(output), '--warm-start', str(warm_path), '--frozen'])
    for directory in ('logs', 'data', 'episodes', 'metrics', 'checkpoints'):
        (output / directory).mkdir(exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ('repo', 'output', 'warm_start', 'frozen')}
    warm = torch.load(warm_path, map_location='cpu', weights_only=True)
    if warm['schema'] != MODEL_SCHEMA or warm['actor_checkpoint_sha256'] != CHECKPOINT_SHA:
        raise ValueError('Warm-start model/actor protocol mismatch')
    config.update(hidden=warm['config']['hidden'], horizon=warm['config']['horizon'],
                  random_slow_probability=.25, world_size=2000., mission_distances_m=[150., 300.])
    if config['horizon'] != 4: raise ValueError('Fast prediction tokens require a four-second horizon')
    write_json(output / 'config.json', config)
    torch.manual_seed(args.seed); torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available(): raise RuntimeError('Requested CUDA unavailable; no silent CPU fallback')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    elif device.type != 'cpu': raise ValueError('Only CPU and CUDA supported')
    fast = make_fast(repo).to(device)
    world = ResidualWorldModel(config['hidden'], config['horizon']).to(device)
    world.load_state_dict(warm['model'])
    actor_optimizer = torch.optim.Adam([
        dict(params=[*fast.base.gat_actor.parameters(), fast.base.log_std], lr=1e-5),
        dict(params=[*fast.actor_peers.parameters(), *fast.actor_delta.parameters()], lr=3e-4)])
    critic_optimizer = torch.optim.Adam(fast.critic_parameters(), lr=1e-4)
    world_optimizer = torch.optim.Adam(world.parameters(), lr=1e-4)
    rng = np.random.default_rng(args.seed)
    manifest = dict(stage=2, fast_schema=FAST_SCHEMA, world_schema=MODEL_SCHEMA,
        graph_adapter_schema=GRAPH_SCHEMA, cbf_context_schema=CBF_SCHEMA,
        warm_start=str(warm_path), warm_start_sha256=digest(warm_path), archived_v9_sha256=CHECKPOINT_SHA,
        source_package='world_model', source_sha256=source_hashes(SOURCE),
        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        schedule='Freeze paired actor/world versions during collection; update on-policy MAPPO, then supervised dynamics; deploy both next batch.',
        actor_inputs='Archived local map/relay features plus two masked peer tokens containing predicted 0..4s states, 1..4s executed actions, diagonal spread and original message/task ages.',
        cbf_context='Current received-geometry margin is computed at the real actor callback, carried unchanged on the ego graph node through world inference, and read at legacy feature 32. Raw value/status/time/geometry version are saved per decision. Predicted peer CBF remains unknown. This adapter is not the proposed full heterogeneous graph world model.',
        prior='Current behavior actor queried on a receiver-reconstructed remote view with absent remote prediction tokens. No recursive inference or access to the remote private observation.',
        reward='Archived velocity-mapping team reward (received-map/route gains, time, CBF intervention and margin costs), +100 final arrival, -100 infeasible CBF, -20 real mission deadline. No bonus for changing subgoals.',
        ppo='Latent nominal action log probabilities; actual CBF executed actions are dynamics labels only. Team GAE gamma=.995 lambda=.97; terminal zero bootstrap; truncated rollouts use a real next-state value. Actor analytic KL cap .02 with optimizer rollback.',
        slow='Fixed 75% geometry / 25% valid random task rule; evaluation uses geometry. Slow PPO is not updated in stage 2.',
        validation='Old phase-1 validation is a retention diagnostic only. Fresh coupled holdout rollouts at configured intervals assess the updated policy/world pair; no stage-2 test-set claims.',
        limitations='Three UAVs, dt=1s, 2km synthetic world, initial missions 150..300m. Not yet kilometer-length mission training or full environment/map/reward modeling.')
    write_json(output / 'manifest.json', manifest)
    train_count = args.updates * args.episodes_per_update
    cases = dict(train=[make_scene(args.seed + i) for i in range(train_count)],
                 validation=[make_scene(args.seed + 1000000 + i) for i in range(args.eval_scenes)])
    warm_cases_path = warm_path.parent / 'scenes.json'
    if warm_cases_path.exists():
        old_seeds = {c['seed'] for group in json.loads(warm_cases_path.read_text()).values() for c in group}
        if old_seeds & {c['seed'] for group in cases.values() for c in group}:
            raise ValueError('Joint rollout seeds overlap warm-start scenes')
    write_json(output / 'scenes.json', cases)
    warm_train_paths = sorted((warm_path.parent / 'data').glob('train-*.npz'))
    warm_val_paths = sorted((warm_path.parent / 'data').glob('validation-*.npz'))
    warm_training = load_data(warm_train_paths) if warm_train_paths else None
    retention = load_data(warm_val_paths) if warm_val_paths else None
    started = time.time(); completed = 0; actor_updates = 0; world_updates = 0; env_steps = 0
    records = []; replay_paths = []

    def status(phase, state='running', **extra):
        row = dict(state=state, phase=phase, pid=os.getpid(), device=str(device),
            elapsed_seconds=round(time.time() - started, 2), completed_updates=completed,
            actor_optimizer_updates=actor_updates, world_optimizer_updates=world_updates,
            training_env_steps=env_steps, scenes_completed=len(records), **extra)
        write_json(output / 'status.json', row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def checkpoint(name):
        path = output / 'checkpoints' / name
        temp = path.with_suffix('.tmp')
        torch.save(dict(fast=fast.state_dict(), world=world.state_dict(),
            actor_optimizer=actor_optimizer.state_dict(), critic_optimizer=critic_optimizer.state_dict(),
            world_optimizer=world_optimizer.state_dict(), config=config, manifest=manifest,
            completed_updates=completed, actor_optimizer_updates=actor_updates,
            world_optimizer_updates=world_updates, training_env_steps=env_steps,
            torch_rng=torch.get_rng_state(), numpy_rng=rng.bit_generator.state,
            cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None), temp)
        temp.replace(path)

    try:
        checkpoint('initial.pt')
        status('initializing')
        if retention is not None:
            write_json(output / 'metrics' / 'initial-retention.json', evaluate(world, retention, device))
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            def collect(split, selected, version, evaluation=False):
                fast_state, world_state = numpy_state(fast), numpy_state(world)
                local_config = dict(config, random_slow_probability=0. if evaluation else .25)
                jobs = [(str(repo), case, local_config, str(output), f'{split}-v{version:03d}-{i:04d}',
                         fast_state, world_state, version) for i, case in selected]
                status('collecting_' + split, collecting_version=version, expected_scenes=len(jobs))
                futures = [pool.submit(collect_joint, job) for job in jobs]
                result = []
                for future in as_completed(futures):
                    row = future.result()
                    if (row['version'] != version or row['fast_sha256'] != state_hash(fast_state)
                            or row['world_sha256'] != state_hash(world_state)):
                        raise ValueError('Collection mixed behavior versions')
                    result.append(row); records.append(row)
                    write_json(output / 'collection.json', records)
                    status('collecting_' + split, latest_scene=row['label'], collecting_version=version)
                return sorted(result, key=lambda x: x['label'])

            for iteration in range(args.updates):
                start = iteration * args.episodes_per_update
                selected = list(enumerate(cases['train'][start:start + args.episodes_per_update], start))
                collected = collect('train', selected, iteration)
                batches = []
                for row in collected:
                    with np.load(row['ppo_path'], allow_pickle=False) as data:
                        batches.append({k: data[k] for k in data.files})
                env_steps += sum(len(b['rewards']) for b in batches)
                status('updating_fast', collecting_version=iteration)
                fast.train()
                ppo = ppo_update(fast, batches, actor_optimizer, critic_optimizer, device, rng,
                                 epochs=args.ppo_epochs, minibatch=args.ppo_minibatch)
                actor_updates += ppo['actor_optimizer_steps']
                replay_paths.extend(row['wm_path'] for row in collected if row['wm_path'])
                replay_paths = replay_paths[-16:]
                if not replay_paths:
                    raise RuntimeError('No delivered telemetry labels; cannot claim joint training')
                training = load_data(replay_paths)
                if warm_training is not None:
                    ids = rng.choice(len(warm_training['history']),
                                     min(1024, len(warm_training['history'])), replace=False)
                    training = {key: np.concatenate((value, warm_training[key][ids]))
                                for key, value in training.items()}
                status('updating_world', collecting_version=iteration)
                wm = train_world_batch(world, world_optimizer, training, device, rng, args.wm_epochs)
                world_updates += wm['optimizer_steps']
                completed = iteration + 1
                metrics = dict(version_before=iteration, version_after=completed, ppo=ppo, world=wm,
                    predictions_consumed=sum(row['predicted_agent_steps'] for row in collected),
                    goals_reached=sum(row['goal_reached'] for row in collected),
                    mean_remaining_m=float(np.mean([row['remaining_m'] for row in collected])),
                    terminations={reason: sum(row['termination'] == reason for row in collected)
                                  for reason in sorted({row['termination'] for row in collected})})
                write_json(output / 'metrics' / f'update-{completed:03d}.json', metrics)
                checkpoint('latest.pt')
                status('update_complete', latest_metrics=metrics)
                if completed % args.eval_every == 0 or completed == args.updates:
                    checkpoint(f'update-{completed:03d}.pt')
                    if retention is not None:
                        write_json(output / 'metrics' / f'retention-{completed:03d}.json',
                                   evaluate(world, retention, device))
                    heldout = collect('validation', list(enumerate(cases['validation'])), completed, True)
                    paths = [r['wm_path'] for r in heldout if r['wm_path']]
                    metrics = dict(version=completed, episodes=heldout,
                        world=evaluate(world, load_data(paths), device) if paths else None)
                    write_json(output / 'metrics' / f'coupled-validation-{completed:03d}.json', metrics)
        status('finished', 'complete')
    except BaseException as error:
        status('failed', 'failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
