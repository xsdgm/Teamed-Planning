"""Schedule fleet PPO sampling and paired evaluation with one frozen simulation backend."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path

import torch


def collect_baseline(collect, cpus, *args):
    if cpus:
        os.sched_setaffinity(0, cpus)
        if os.sched_getaffinity(0) != set(cpus):
            raise ValueError(f'Could not assign all requested baseline CPUs: {cpus}')
    return collect(*args)


def save_baseline(futures, write_json, path):
    rows = [future.result() for future in futures]
    write_json(path, rows)
    return rows


def run(args, backend):
    out = args.output.resolve()
    protocol = backend.configure(args)
    manifest = out/'manifest.json'
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != protocol:
            raise ValueError('Use a fresh output, or --resume with the original inputs and source version')
    else:
        if args.resume or any(f.name not in ('logs',) for f in out.iterdir()):
            raise ValueError('Expected an empty run directory for a new experiment')
        backend.atomic_json(manifest,protocol)
    signature = backend.file_sha256(manifest)
    model, _ = backend.policy_from(protocol['actor'])
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    checkpoints = out/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    weights,digest = Path(protocol['actor']),protocol['actor_sha256']
    history = []
    worker_protocol = {k:protocol[k] for k in ('repo', 'ground_control', 'world_model', 'seconds_per_m',
                                                'arrival_radius_m', 'curriculum') if k in protocol}
    def report(phase, **info):
        backend.atomic_json(out/'progress.json',dict(phase=phase,pid=os.getpid(),completed_updates=len(history),**info))
        print(f'{phase} {json.dumps(info,ensure_ascii=False)}',flush=True)
    with ExitStack() as pools:
        pool = pools.enter_context(backend.episode_pool(args.workers))
        def batch(phase, plans, directory, limit=None):
            report(phase, directory=str(directory),episodes=len(plans),checkpoint_sha256=digest)
            jobs = {pool.submit(backend.collect,worker_protocol,plan,weights,digest,directory/f'episode-{i:02d}',phase,limit):i
                    for i,plan in enumerate(plans)}
            summaries = [None]*len(plans)
            for job in as_completed(jobs):
                summaries[jobs[job]] = job.result()
            return summaries
        smoke = batch('smoke',protocol['evaluation'][:1],out/'preflight',protocol['smoke_s'])
        backend.atomic_json(out/'preflight.json',dict(smoke=smoke,technical_smoke=True,manifest_sha256=signature))
        if args.preflight_only:
            report('preflight_complete')
            return
        first_episodes = None
        baseline_result = None
        baseline_workers = getattr(args, 'baseline_workers', 0)
        if baseline_workers:
            baseline_pool = pools.enter_context(backend.episode_pool(baseline_workers))
            cpus = getattr(args, 'baseline_cpus', None)
            baseline_futures = [
                baseline_pool.submit(collect_baseline, backend.collect, cpus, worker_protocol, plan,
                                     weights, digest, out/'baseline'/f'episode-{i:02d}', 'baseline', None)
                for i, plan in enumerate(protocol['evaluation'])
            ]
            # Write the initial-policy comparison as soon as its episodes finish.
            # The coordinator can publish PPO checkpoints independently.
            writer = pools.enter_context(ThreadPoolExecutor(max_workers=1))
            baseline_result = writer.submit(save_baseline, baseline_futures, backend.atomic_json, out/'baseline.json')
            report('baseline_dispatched', workers=baseline_workers, cpus=cpus,
                   episodes=len(baseline_futures), checkpoint_sha256=digest)
        elif (checkpoints/'heading-0001.pt').exists():
            baseline = batch('baseline',protocol['evaluation'],out/'baseline')
            backend.atomic_json(out/'baseline.json',baseline)
        else:
            # The initial baseline and first training rollouts use the exact same
            # frozen actor. Only the former use held-out maps and deterministic actions.
            baseline_jobs = [('baseline', i, plan, out/'baseline')
                             for i, plan in enumerate(protocol['evaluation'])]
            first_plans = protocol['train'][:args.episodes_per_update]
            train_dir = out/'training'/'update-0001'
            train_jobs = [('train', i, plan, train_dir) for i, plan in enumerate(first_plans)]
            jobs = {}
            baseline = [None] * len(baseline_jobs)
            first_episodes = [None] * len(first_plans)
            def initial_status():
                report('initial_baseline_and_ppo_sampling', update=1, checkpoint_sha256=digest,
                       baseline_dir=str(out/'baseline'), training_dir=str(train_dir),
                       baseline_completed=sum(row is not None for row in baseline), baseline_total=len(baseline),
                       training_completed=sum(row is not None for row in first_episodes), training_total=len(first_plans))
            initial_status()
            # Circle training fills the first wave with independent training missions.
            jobs_in_order = (train_jobs + baseline_jobs if args.curriculum == 'circle'
                             else [baseline_jobs[0], *train_jobs, *baseline_jobs[1:]])
            for phase, i, plan, directory in jobs_in_order:
                future = pool.submit(backend.collect,worker_protocol,plan,weights,digest,directory/f'episode-{i:02d}',phase,None)
                jobs[future] = phase, i
            for future in as_completed(jobs):
                phase, i = jobs[future]
                if phase == 'baseline':
                    baseline[i] = future.result()
                    if all(row is not None for row in baseline):
                        backend.atomic_json(out/'baseline.json',baseline)
                else:
                    first_episodes[i] = future.result()
                initial_status()
            # Reaching the optimizer requires every initial full-course outcome.
        def report_baseline(rows):
            report('baseline_complete', arrivals=sum(r['goal_reached'] for r in rows),
                   received_tiles=sum(r['received_tiles'] for r in rows),
                   online_world_predictions=sum(r['world']['prediction_calls'] for r in rows))
        if baseline_result is None:
            report_baseline(baseline)
        for update in range(1,args.updates+1):
            path = checkpoints/f'heading-{update:04d}.pt'
            if path.is_file():
                saved = torch.load(path,map_location='cpu',weights_only=True)
                if (saved['manifest_sha256'] != signature or saved['parent_sha256'] != digest
                        or saved['completed_updates'] != update):
                    raise ValueError('Fine-tune checkpoint lineage is inconsistent')
                model.load_state_dict(saved['model'],strict=True)
                optimizer.load_state_dict(saved['optimizer'])
                history.append(saved['update_record'])
            else:
                start = (update-1)*args.episodes_per_update
                plans = protocol['train'][start:start+args.episodes_per_update]
                directory = out/'training'/f'update-{update:04d}'
                episodes = first_episodes if update == 1 and first_episodes is not None else batch('train',plans,directory)
                rollouts = [backend.load_rollout(directory/f'episode-{i:02d}') for i in range(len(plans))]
                report('optimizing',update=update)
                before = model.actor.weight.detach().clone()
                metrics = backend.update_policy(model.train(),optimizer,rollouts,epochs=args.epochs,
                                        microbatch_size=args.microbatch_size)
                delta = float((model.actor.weight.detach()-before).abs().max())
                if not math.isfinite(delta) or delta <= 0:
                    raise FloatingPointError('Vehicle actor did not make a finite parameter update')
                record = dict(update=update,episodes=episodes,ppo=metrics,actor_max_weight_delta=delta)
                backend.atomic_checkpoint(path,dict(format=backend.FORMAT,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    environment='vtemp5_online_fleet',manifest_sha256=signature,parent_sha256=digest,
                    completed_updates=update,v9_sha256=backend.CHECKPOINT_SHA,world_sha256=protocol['world_sha256'],
                    update_record=record))
                history.append(record)
            weights,digest = path,backend.file_sha256(path)
            backend.atomic_json(out/'training.json',history)
            report('committed',update=update,checkpoint=str(path),checkpoint_sha256=digest)
        final = batch('paired_evaluation',protocol['evaluation'],out/'evaluation')
        if baseline_result is not None:
            baseline = baseline_result.result()
            report_baseline(baseline)
        backend.atomic_json(out/'evaluation.json',dict(baseline=baseline,trained=final,checkpoint_sha256=digest))
        report('complete',checkpoint=str(weights),arrivals=sum(e['goal_reached'] for e in final),episodes=len(final))
