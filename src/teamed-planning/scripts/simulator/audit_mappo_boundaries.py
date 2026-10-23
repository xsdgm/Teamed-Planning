"""Replay a frozen training-log prefix and measure pre-clamp boundary errors.

Uses the run's saved source tree, policy versions, seeds and optimizer snapshots.
Intermediate policies are reconstructed offline in the original update order.
Never edits the training run or changes the simulator's clamp behavior.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def setup(source):
    sys.path.insert(0, str(source))
    torch.set_num_threads(1)


def compare_metrics(actual, expected, ignore=()):
    differences = {}
    for key, value in actual.items():
        if key in ignore or key not in expected or not isinstance(value, (int, float)):
            continue
        if not math.isclose(value, expected[key], rel_tol=1e-9, abs_tol=1e-9):
            differences[key] = dict(actual=value, expected=expected[key])
    return differences


def replay(job, model_path, config, output):
    from drone.mappo import ActorCritic
    from simulator.env import MultiRobotEnv
    from simulator.train_mappo import run_episode

    output = Path(output)
    started = time.monotonic()
    torch.manual_seed(job['sample_seed'])
    saved = torch.load(model_path, map_location='cpu', weights_only=True)
    model = ActorCritic.from_state_dict(saved['model'])
    model.train(job['kind'] == 'training')
    original = MultiRobotEnv._check_boundaries
    events = []

    def audit(env, info):
        size = np.asarray(env.config.world_size)
        ground_ids = {r.robot_id for r in env.ground_robots}
        for robot in env.ground_robots + env.drones:
            xy = robot.position[:2].copy()
            excess = float(max(0., np.max(-xy), np.max(xy-size)))
            if excess > 0:
                events.append(dict(step=env.step_count, robot_id=robot.robot_id,
                    kind='ground' if robot.robot_id in ground_ids else 'drone',
                    position=xy.tolist(), overshoot_m=excess))
        original(env, info)

    MultiRobotEnv._check_boundaries = audit
    live = output/'running'/f"{job['tag']}.json"
    try:
        rollout, summary = run_episode(model, config, job['seed'], job['schedule'], True,
            progress=lambda value: write_json(live, value))
    finally:
        MultiRobotEnv._check_boundaries = original
    differences = compare_metrics(summary, job['expected'], ignore=('decision_ms',))
    if len(events) != int(summary['out_of_bounds']):
        differences['event_count'] = dict(actual=len(events), expected=summary['out_of_bounds'])
    report = dict(tag=job['tag'], kind=job['kind'], policy_version=job['policy_version'],
        seed=job['seed'], sample_seed=job['sample_seed'], summary=summary,
        summary_differences=differences, events=events,
        max_overshoot_m=max((e['overshoot_m'] for e in events), default=0.),
        elapsed_seconds=time.monotonic()-started)
    write_json(output/'episodes'/f"{job['tag']}.json", report)
    if job['kind'] == 'training':
        # Local trusted artifact; includes numpy rollout arrays.
        torch.save(rollout, output/'rollouts'/f"{job['episode']:05d}.pt")
    live.unlink(missing_ok=True)
    return {k:v for k,v in report.items() if k not in ('events', 'summary')}


def reconstruct(episode, row, previous_path, output, config, anchor_path):
    from drone.mappo import ActorCritic
    from simulator.mappo_experiment import update

    previous = torch.load(previous_path, map_location='cpu', weights_only=True)
    model = ActorCritic.from_state_dict(previous['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    optimizer.load_state_dict(previous['optimizer'])
    rollout = torch.load(output/'rollouts'/f'{episode:05d}.pt',
                         map_location='cpu', weights_only=False)
    loss, metrics = update(model, optimizer, rollout, epochs=config['ppo_epochs'],
        entropy_coef=row['schedule']['entropy'], return_metrics=True,
        discount=config['discount'], gae_lambda=config['gae_lambda'])
    differences = compare_metrics(dict(loss=loss, **metrics), dict(loss=row['loss'], **row['ppo']))
    if differences:
        write_json(output/'updates'/f'{episode:05d}.json', dict(differences=differences))
        raise RuntimeError(f'Update {episode} does not match original optimization metrics')
    result = dict(episode=episode, metric_differences={}, anchor_max_difference=None)
    if anchor_path:
        anchor = torch.load(anchor_path, map_location='cpu', weights_only=True)
        delta = max(float((value-anchor['model'][key]).abs().max())
                    for key,value in model.state_dict().items())
        result['anchor_max_difference'] = delta
        if delta != 0.:
            write_json(output/'updates'/f'{episode:05d}.json', result)
            raise RuntimeError(f'Checkpoint {episode} not bitwise reproducible: {delta}')
    else:
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()),
                   output/'models'/f'{episode:05d}.pt')
    write_json(output/'updates'/f'{episode:05d}.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--through-episode', type=int, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    base, output = args.run.resolve(), args.output.resolve()
    output.mkdir(exist_ok=False, parents=True)
    for directory in ('episodes', 'rollouts', 'models', 'updates', 'running'):
        (output/directory).mkdir()
    source = base/'sources'
    setup(source)
    from simulator.train_mappo import validation_runs, fixed_schedule
    from drone.mappo import ActorCritic

    config = json.loads((base/'config.json').read_text())
    provenance = json.loads((base/'provenance.json').read_text())
    for name, digest in provenance['source_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Source snapshot changed: {name}')
    training = [json.loads(line) for line in (base/'training.jsonl').read_text().splitlines()
                if json.loads(line)['episode'] <= args.through_episode]
    validations = [json.loads(line) for line in (base/'validation.jsonl').read_text().splitlines()
                   if json.loads(line)['episode'] <= args.through_episode]
    assert [r['episode'] for r in training] == list(range(1,args.through_episode+1))
    write_json(output/'training_snapshot.json', training)
    write_json(output/'validation_snapshot.json', validations)
    write_json(output/'config.json', config)
    initial = Path(provenance['init_checkpoint']['path'])
    assert hashlib.sha256(initial.read_bytes()).hexdigest() == provenance['init_checkpoint']['sha256']
    state = torch.load(initial, map_location='cpu', weights_only=True)['model']
    model = ActorCritic.from_state_dict(state)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    torch.save(dict(model=state, optimizer=optimizer.state_dict()), output/'models/00000.pt')
    paths = {0:output/'models/00000.pt'}
    anchors = {}
    for path in base.glob('episode_*.pt'):
        version = int(path.stem.split('_')[1])
        if version <= args.through_episode:
            paths[version] = path
            anchors[version] = path
    jobs = []
    for row in training:
        version = row['episode']-1-row['policy_lag']
        assert 0 <= version < row['episode']
        jobs.append(dict(tag=f"train_{row['episode']:05d}",kind='training',episode=row['episode'],
            policy_version=version,seed=row['scene_seed'],schedule=row['schedule'],expected=row,
            sample_seed=int(np.random.SeedSequence([config['seed'],row['dispatch']-1]).generate_state(1)[0])))
    seeds = {(s,k):sample_seed for s,k,sample_seed in validation_runs(config)}
    for report in validations:
        for row in report['scenes']:
            jobs.append(dict(tag=f"val_{report['episode']:05d}_{row['seed']}_{row['sample']}",
                kind='validation',policy_version=report['episode'],seed=row['seed'],
                sample_seed=seeds[row['seed'],row['sample']],expected=row,
                schedule=fixed_schedule(config,config['episodes']-1)))
    # Prefer training rollouts to unlock intermediate policy versions.
    manifest = dict(run=str(base), through_episode=args.through_episode, workers=args.workers,
        started_utc=datetime.now(timezone.utc).isoformat(), source_hashes_verified=True,
        training_episodes=len(training), validation_episodes=sum(len(v['scenes']) for v in validations),
        training_boundary_events=sum(r['out_of_bounds'] for r in training),
        validation_boundary_events=sum(r['out_of_bounds'] for v in validations for r in v['scenes']),
        command=sys.argv)
    write_json(output/'manifest.json',manifest)
    started=time.monotonic()
    results, done_training, updated, tasks = {}, set(), {}, {}
    def progress(status='running', **extra):
        write_json(output/'progress.json',dict(status=status,pid=os.getpid(),
            elapsed_seconds=time.monotonic()-started,completed_replays=len(results),
            total_replays=len(training)+manifest['validation_episodes'],reconstructed_updates=len(updated),
            boundary_events=sum(r.get('boundary_events',0) for r in results.values()),
            max_overshoot_m=max((r['max_overshoot_m'] for r in results.values()),default=0.),
            active=[j['tag'] for j in tasks.values()],**extra))
    progress()
    try:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=setup, initargs=(source,)) as pool:
            while jobs or tasks or len(updated)<len(training):
                for episode in sorted(done_training-set(updated)):
                    if episode-1 not in paths:
                        continue
                    result = reconstruct(episode, training[episode-1],paths[episode-1],output,
                                         config,anchors.get(episode))
                    updated[episode] = result
                    if episode not in paths:
                        paths[episode] = output/'models'/f'{episode:05d}.pt'
                for job in jobs[:]:
                    if len(tasks) >= args.workers:
                        break
                    if job['policy_version'] in paths:
                        future=pool.submit(replay,job,paths[job['policy_version']],config,output)
                        tasks[future]=job
                        jobs.remove(job)
                progress()
                if not tasks:
                    if jobs or len(updated)<len(training):
                        raise RuntimeError('Unresolved policy dependencies')
                    break
                completed,_=wait(tasks,timeout=5,return_when=FIRST_COMPLETED)
                for future in completed:
                    job=tasks.pop(future)
                    result=future.result()
                    if result['summary_differences']:
                        raise RuntimeError(f"Replay {job['tag']} differs from original summary")
                    report=json.loads((output/'episodes'/f"{job['tag']}.json").read_text())
                    result['boundary_events']=len(report['events'])
                    results[job['tag']]=result
                    if job['kind']=='training':
                        done_training.add(job['episode'])
                    print(f"replayed {len(results)} {job['tag']} events={len(report['events'])} max={result['max_overshoot_m']:.3g}",flush=True)
        events=[]
        for path in (output/'episodes').glob('*.json'):
            report=json.loads(path.read_text())
            events.extend(dict(tag=report['tag'],**event) for event in report['events'])
        report=dict(**manifest,completed_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-started,all_summaries_matched=True,
            all_optimization_metrics_matched=True,verified_anchors=[v for v in updated.values()
                if v['anchor_max_difference'] is not None],events=len(events),
            max_overshoot_m=max((e['overshoot_m'] for e in events),default=0.),
            counts_above_m={str(t):sum(e['overshoot_m']>t for e in events)
                for t in (1e-12,1e-10,1e-8,1e-7,1e-6,1e-3)},
            ground_events=sum(e['kind']=='ground' for e in events),
            largest_events=sorted(events,key=lambda e:e['overshoot_m'],reverse=True)[:20])
        write_json(output/'report.json',report)
        progress('completed')
    except BaseException as exc:
        progress('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
