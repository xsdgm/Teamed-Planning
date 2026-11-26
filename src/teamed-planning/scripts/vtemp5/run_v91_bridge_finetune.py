"""Fine-tune the frozen-checkpoint vehicle actor with V9.1 aircraft and extent inference."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import sys
import time

import torch
from torch import nn


SOLVER_VERSION = 'v91_candidate_bridge_ppo_v1'
SERVING = ('v91_guided_drive.py', 'v91_trainable_guidance.py', 'v91_rejoin_guidance.py',
           'v91_route_evidence.py', 'v91_rolling_clearance.py', 'v91_km_cases.py', 'v91_km_wild.py',
           'v91_frozen_actor_inputs.py', 'v91_extent_collect.py', 'v91_lake_drive.py',
           'v91_lake_metrics.py', 'v91_extent_network.py', 'v91_extent_scenes.py', 'v91_lake_cases.py',
           'run_v91_extent_pilot.py', 'run_v91_lake_test.py', 'v91_bridge_curriculum.py',
           'v91_bridge_rollout.py', 'run_v91_bridge_finetune.py')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_file(path, obj):
    from run_v91_lake_test import jsonable
    temp = path.with_suffix('.new')
    with temp.open('w') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False, default=jsonable)
        stream.write('\n')
    temp.replace(path)


def cpus(text):
    if text is None:
        return None
    values = set()
    for group in text.split(','):
        bounds = [int(x) for x in group.split('-')]
        if len(bounds) not in (1, 2) or min(bounds) < 0 or bounds[-1] < bounds[0]:
            raise argparse.ArgumentTypeError('Expected comma-separated CPU ids or ranges')
        values.update(range(bounds[0], bounds[-1] + 1))
    return sorted(values)


class BranchPPO(nn.Module):
    """Adapt the existing heading model to its deployed categorical candidate sampler."""
    def __init__(self, base):
        super().__init__()
        self.base = base

    @property
    def actor(self):
        return self.base.actor

    def forward(self, observations):
        from v91_trainable_guidance import branch_policy
        return branch_policy(self.base, observations)

    def critic_values(self, observations):
        return self.base.critic_values(observations)


def action_logp(dist, rows, values):
    indices = torch.tensor([row['action'] for row in rows], dtype=torch.long, device=values.device)
    return dist.log_prob(indices)


def optimize(policy, optimizer, episodes, epochs, microbatch):
    from vtemp4.vtemp4_policy import update_policy
    if not any(row['policy_controlled'] for ep in episodes for row in ep):
        raise ValueError('Rollout batch contained no selectable vehicle actions')
    previous = {k: t.detach().clone() for k, t in policy.base.named_parameters()}
    head_grads = []
    def watch(gradient):
        head_grads.append(float(gradient.detach().norm()))
    hook = policy.actor.weight.register_hook(watch)
    try:
        metrics = update_policy(policy, optimizer, episodes, epochs=epochs, microbatch_size=microbatch,
                                clip=.2, gae_lambda=.95, entropy_coef=.005, action_log_prob=action_logp)
    finally:
        hook.remove()
    with torch.no_grad():
        squared = {k: float((weight - previous[k]).double().square().sum())
                   for k, weight in policy.base.named_parameters()}
    metrics.update(parameter_delta_l2=math.sqrt(sum(squared.values())),
                   actor_head_delta_l2=math.sqrt(sum(v for k,v in squared.items() if k.startswith('actor.'))),
                   actor_head_max_microbatch_gradient=max(head_grads, default=0.))
    if not all(math.isfinite(float(v)) for v in metrics.values()) or metrics['actor_head_delta_l2'] <= 0:
        raise FloatingPointError('Vehicle actor head failed to receive a finite update')
    return metrics


def episodes_for_update(repo, number, args):
    from v91_bridge_curriculum import bridge_curriculum_scene
    output = []
    mixed = number > args.mouth_updates
    for i in range(args.episodes_per_update):
        stage = 'kilometre' if mixed and i < args.km_per_update else 'mouth'
        episode = (number-1) * args.episodes_per_update + i
        output.append(bridge_curriculum_scene(stage, episode, seed=args.seed, split='ppo_train', repo=repo))
    return output


def eval_scenes(repo, seed):
    from v91_bridge_curriculum import bridge_curriculum_scene
    from v91_km_cases import km_cases
    mouths = [bridge_curriculum_scene('mouth', i, seed=seed, split='development', repo=repo) for i in range(4)]
    targets = {'valley-dev-00-detour_right', 'valley-dev-01-detour_left',
               'vtemp5-circle-development-1545411713', 'vtemp5-circle-development-1545411715'}
    full = [s for s in km_cases(repo) if s['case_id'] in targets]
    if len(full) != len(targets):
        raise RuntimeError('Missing a fixed kilometre-scale evaluation case')
    return [*mouths, *full]


def case_spec(scene):
    return dict(case_id=scene['case_id'], kind=scene['kind'], start=list(scene['start']), goal=list(scene['goal']),
                seed=int(scene['seed']), fleet_seed=int(scene['fleet_seed']), max_steps=int(scene['max_steps']),
                direct_distance_m=float(scene['direct_distance_m']), map_sha256=scene['case']['map_sha256'])


def make_protocol(args, folder, weights, train, evaluation):
    source = folder/'sources'
    base = read(args.repo/'artifacts/v91-frozen-km-paired-20261126/protocol.json')
    if weights != base['weights']:
        raise ValueError('Expected the three weights of the original kilometre evaluation')
    for weight in weights.values():
        if digest(weight['path']) != weight['sha256']:
            raise ValueError(f"Checkpoint identity changed: {weight['path']}")
    frozen = Path(weights['uav']['path']).parent
    runtime = frozen/'runtime_sources'
    runtime_files = {str(p.relative_to(runtime)):digest(p) for p in sorted(runtime.rglob('*.py'))}
    runtime_sha = hashlib.sha256(json.dumps(runtime_files, sort_keys=True).encode()).hexdigest()
    if runtime_sha != base['uav_runtime_sources_sha256'] or digest(frozen/'config.json') != base['uav_config_sha256']:
        raise ValueError('Frozen fleet runtime differs from the reference evaluation')
    train_hash = {s['case']['map_sha256'] for group in train for s in group}
    eval_hash = {s['case']['map_sha256'] for s in evaluation}
    if train_hash & eval_hash:
        raise ValueError('Training and validation maps overlap')
    return dict(format=SOLVER_VERSION, weights=weights, optimized_model='vehicle PPO', source_sha256={
        f:digest(source/f) for f in SERVING}, frozen_fleet_runtime_sha256=runtime_sha,
        frozen_fleet_config_sha256=base['uav_config_sha256'], review_seconds=30, workers=args.workers,
        evaluation_workers=args.eval_workers, seed=args.seed, lr=args.lr, epochs=args.epochs,
        microbatch=args.microbatch, updates=args.updates, mouth_updates=args.mouth_updates,
        episodes_per_update=args.episodes_per_update, kilometres_per_mixed_update=args.km_per_update,
        training=[[case_spec(s) for s in group] for group in train], evaluation=[case_spec(s) for s in evaluation],
        reward=dict(route_progress_per_metre=.02, first_deck=1., first_opposite_bank_via_deck=5.,
                    goal=10., safety=-6., timeout=-2., time_cost_per_full_budget=-1.))


def aggregate(rows):
    grouped = {}
    for row in rows:
        key = row['stage'] if row['kind'] == 'bridge' else row['kind']
        group = grouped.setdefault(key, dict(episodes=0, entered_deck=0, crossed_bridge=0, arrivals=0,
                                            ground_collisions=0, ground_safety_violations=0))
        group['episodes'] += 1
        for name in ('entered_deck', 'crossed_bridge', 'ground_collisions', 'ground_safety_violations'):
            group[name] += int(row.get(name) or 0)
        group['arrivals'] += int(row['goal_reached'])
    return grouped


def stable_cp(folder, number):
    return folder/'checkpoints'/f'heading-{number:04d}.pt'


@contextmanager
def worker_pool(workers, weights, affinity):
    from v91_bridge_rollout import init_worker
    pool = ProcessPoolExecutor(workers, mp_context=get_context('spawn'),
                               initializer=init_worker, initargs=(weights,affinity))
    try:
        yield pool
    except BaseException:
        # A failed batch has no update to commit. Release this run's processes
        # immediately even if another job is in a full kilometre episode.
        for process in (pool._processes or {}).values():
            if process.is_alive():
                process.terminate()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def run(args):
    import fcntl
    from v91_bridge_rollout import collect
    from run_v91_lake_test import jsonable
    folder = args.output
    locks = (folder/'run.lock').open('a')
    fcntl.flock(locks.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    weights = read(args.repo/'artifacts/v91-frozen-km-paired-20261126/protocol.json')['weights']
    train = [episodes_for_update(args.repo, u, args) for u in range(1,args.updates+1)]
    evaluation = eval_scenes(args.repo, args.seed+1)
    protocol = make_protocol(args, folder, weights, train, evaluation)
    manifest = folder/'protocol.json'
    if manifest.exists():
        if read(manifest) != protocol:
            raise ValueError('Resume requires matching training protocol, sources, maps and weights')
    else:
        json_file(manifest, protocol)
    protocol_hash = digest(manifest)
    original = Path(weights['actor']['path'])
    checkpoint = original
    # Load the actual frozen runtime before importing its compatible actor class.
    from v91_extent_collect import load_frozen
    from v91_frozen_actor_inputs import load_actor
    fleet_path = Path(weights['uav']['path'])
    load_frozen(fleet_path.parent, fleet_path)
    torch.manual_seed(args.seed)
    policy = BranchPPO(load_actor(original).requires_grad_(True))
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    fmt = torch.load(original, map_location='cpu', weights_only=True)['format']
    completed = 0
    history = []
    for n in range(1,args.updates+1):
        candidate = stable_cp(folder,n)
        if not candidate.is_file():
            break
        snapshot = torch.load(candidate, map_location='cpu', weights_only=True)
        if snapshot['protocol_sha256'] != protocol_hash or snapshot['update'] != n:
            raise ValueError('Checkpoint protocol or update index mismatch')
        history.append(snapshot['summary'])
        completed, checkpoint = n, candidate
    if completed:
        policy.base.load_state_dict(snapshot['model'])
        optimizer.load_state_dict(snapshot['optimizer'])
    status = dict(completed_updates=completed, updates=args.updates, checkpoint=str(checkpoint))
    def report(phase, **fields):
        json_file(folder/'progress.json', dict(**status, phase=phase, timestamp=time.time(), **fields))
    def eval_submit(pool, label, actor_file):
        root = folder/label
        root.mkdir(exist_ok=True)
        futures = []
        for idx, scene in enumerate(evaluation):
            target = root/f'{idx:02d}-{scene["case_id"]}.json'
            if not target.is_file():
                futures.append(pool.submit(collect, scene, str(actor_file), True, args.seed + 90000 + idx, str(target)))
        return futures
    try:
        with ExitStack() as stack:
            baseline_pool = stack.enter_context(worker_pool(args.eval_workers,weights,args.eval_cpus))
            train_pool = stack.enter_context(worker_pool(args.workers,weights,args.train_cpus))
            baseline = eval_submit(baseline_pool, 'baseline', original)
            for n in range(completed+1,args.updates+1):
                scenes = train[n-1]
                report('sampling', current_update=n, completed_episodes=0, batch_episodes=len(scenes))
                jobs = {train_pool.submit(collect, scene, str(checkpoint), False, args.seed + n*1000+i):i
                        for i,scene in enumerate(scenes)}
                rows, transitions = [None]*len(scenes), [None]*len(scenes)
                for count,future in enumerate(as_completed(jobs),1):
                    idx = jobs[future]
                    rows[idx],transitions[idx] = future.result()
                    for check in baseline:
                        if check.done():
                            check.result()
                    report('sampling', current_update=n, completed_episodes=count, batch_episodes=len(scenes))
                report('optimizing', current_update=n)
                stats = optimize(policy, optimizer, transitions, args.epochs, args.microbatch)
                del transitions
                summary = json.loads(json.dumps(dict(update=n, behavior_checkpoint_sha256=digest(checkpoint),
                    ppo=stats, by_stage=aggregate(rows), episodes=rows), default=jsonable, allow_nan=False))
                checkpoint = stable_cp(folder,n)
                temp = checkpoint.with_suffix('.new')
                torch.save(dict(format=fmt, deployment_policy=SOLVER_VERSION, protocol_sha256=protocol_hash,
                                model=policy.base.state_dict(), optimizer=optimizer.state_dict(), update=n, summary=summary),temp)
                verified = torch.load(temp, map_location='cpu', weights_only=True)
                if any(not torch.equal(value, verified['model'][key]) for key,value in policy.base.state_dict().items()):
                    raise ValueError('Serialized checkpoint did not reproduce trained parameters')
                temp.replace(checkpoint)
                history.append(summary)
                json_file(folder/'training.json', history)
                status.update(completed_updates=n, checkpoint=str(checkpoint))
                report('updated', current_update=n, actor_head_delta_l2=stats['actor_head_delta_l2'])
                print(f'Committed update {n}: {stats}', flush=True)
            report('evaluating')
            final = eval_submit(train_pool, 'finetuned', checkpoint)
            for future in as_completed([*final,*baseline]):
                future.result()
        comparison = {}
        for label in ('baseline','finetuned'):
            record_files = sorted((folder/label).glob('*.json'))
            if len(record_files) != len(evaluation):
                raise RuntimeError('Incomplete deterministic evaluation')
            summaries=[read(f)['summary'] for f in record_files]
            if [r['case_id'] for r in summaries] != [s['case_id'] for s in evaluation]:
                raise RuntimeError('Evaluation case identity mismatch')
            expected = digest(original if label == 'baseline' else checkpoint)
            if any(r['checkpoint_sha256'] != expected for r in summaries):
                raise RuntimeError('Evaluation checkpoint identity mismatch')
            comparison[label] = dict(checkpoint_sha256=expected, by_stage=aggregate(summaries), episodes=summaries)
        if any(digest(w['path'])!=w['sha256'] for w in weights.values()):
            raise RuntimeError('A reference weight changed during the experiment')
        json_file(folder/'evaluation.json',comparison)
        json_file(folder/'training.json',history)
        report('complete')
    except BaseException as exc:
        report('error', detail=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--eval-workers', type=int, default=4)
    parser.add_argument('--train-cpus', type=cpus)
    parser.add_argument('--eval-cpus', type=cpus)
    parser.add_argument('--updates', type=int, default=6)
    parser.add_argument('--mouth-updates', type=int, default=4)
    parser.add_argument('--km-per-update', type=int, default=4)
    parser.add_argument('--episodes-per-update', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--microbatch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=91261126)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    args.repo, args.output = args.repo.resolve(), args.output.resolve()
    if (min(args.workers,args.eval_workers,args.updates,args.epochs,args.microbatch,args.episodes_per_update)<1
            or not 0 <= args.mouth_updates <= args.updates or not 0 <= args.km_per_update <= args.episodes_per_update
            or not math.isfinite(args.lr) or args.lr <= 0):
        parser.error('Invalid worker, stage, or PPO configuration')
    source = args.output/'sources'
    if Path(__file__).resolve().parent != source:
        if not args.resume:
            args.output.mkdir(parents=True,exist_ok=False)
            source.mkdir()
            (args.output/'checkpoints').mkdir()
            for name in SERVING:
                shutil.copyfile(Path(__file__).resolve().parent/name,source/name)
        if not (source/Path(__file__).name).is_file():
            parser.error('No archived trainer found')
        os.execv(sys.executable,[sys.executable,'-B','-u',str(source/Path(__file__).name),*sys.argv[1:]])
    run(args)


if __name__ == '__main__':
    main()
