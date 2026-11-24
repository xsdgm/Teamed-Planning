"""Pretrain circle-detour headings on short native three-aircraft, policy-in-the-loop trajectories."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import as_completed
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))
from vtemp5.run_vtemp5_fleet import DEFAULT_WORLD, FORMAT, GROUND_CONTROL, policy_from, validate_warm_start
from vtemp4.train_vtemp4_formal import exclusive_run, source_digest, source_tree_digest
from vtemp4.vtemp4_parallel import episode_pool
from vtemp2_3.frozen_navigation import CHECKPOINT_SHA, V9_RUN, runtime
from vtemp2_3.hierarchical_sources import atomic_checkpoint, atomic_json, file_sha256

EXPERIMENT = 'vtemp5_circle_pretraining_v1'
ENVIRONMENT = 'vtemp5_circle_pretraining'
TARGET_LABELS = ('maintain_side', 'return_to_goal')


def make_manifest(args):
    repo = args.repo.resolve()
    actor, world = (repo / args.init_checkpoint).resolve(), (repo / args.world_model).resolve()
    _, _, aircraft, fleet = runtime(repo)
    if fleet['num_drones'] != 3 or aircraft.training or any(p.requires_grad for p in aircraft.parameters()):
        raise ValueError('Expected the frozen three-aircraft runtime')
    from vtemp5.vtemp5_circle_pretrain_rollout import pretrain_plans
    archive = repo / V9_RUN
    fleet_identity = dict(v9_sha256=CHECKPOINT_SHA, v9_sources_sha256=source_digest(archive/'runtime_sources'),
                          v9_config_sha256=file_sha256(archive/'config.json'))
    world_sha = file_sha256(world)
    validate_warm_start(actor, world_sha, fleet_identity)
    train = pretrain_plans('ppo_train', args.updates * args.episodes_per_update, args.seed)
    evaluation = pretrain_plans('development', args.eval_episodes, args.eval_seed)
    plans = train + evaluation
    if (len({p['case']['map_sha256'] for p in plans}) != len(plans) or
            len({p['fleet_seed'] for p in plans}) != len(plans)):
        raise ValueError('Repeated maps or fleet seeds across the pretraining and evaluation plans')
    return dict(format=EXPERIMENT, repo=str(repo), actor=str(actor), actor_sha256=file_sha256(actor),
                world_model=str(world), world_sha256=world_sha, **fleet_identity,
                sources_sha256=source_tree_digest(SOURCE.parent), ground_control=GROUND_CONTROL,
                train=train, evaluation=evaluation, updates=args.updates, episodes_per_update=args.episodes_per_update,
                seed=args.seed, eval_seed=args.eval_seed, workers=args.workers, horizon_s=args.horizon_s,
                review_s=30, lookahead_m=200., teacher_probability=args.teacher_probability,
                final_teacher_probability=args.final_teacher_probability, learning_rate=args.learning_rate,
                epochs=args.epochs, batch_size=args.batch_size)


def freeze_example(example):
    """Store observations as tensors so their saved cache loads with weights_only=True."""
    obs = {}
    for key, value in example['observation'].items():
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(np.ascontiguousarray(value))
        elif isinstance(value, np.generic):
            value = value.item()
        obs[key] = value
    angle = float(example['angle_rad'])
    arcs = np.asarray(obs['angle_arcs'], dtype=np.float32)
    if (not np.isfinite(angle) or not np.any((arcs[:, 0] <= np.float32(angle)) & (np.float32(angle) <= arcs[:, 1]))):
        raise ValueError('Demonstration lies outside the legal action arcs')
    return dict(observation=obs, angle_rad=angle, label=example['label'],
                time_s=int(example['time_s']), actor_angle_rad=float(example['actor_angle_rad']),
                teacher_executed=bool(example['teacher_executed']))


def collect(repo, plan, actor, actor_sha, world, world_sha, horizon, probability, phase, directory):
    from vtemp5.vtemp5_circle_pretrain_rollout import collect_pretrain_episode
    directory.mkdir(parents=True, exist_ok=True)
    evidence_path, examples_path = directory/'episode.json', directory/'examples.pt'
    identity = dict(case_id=plan['case_id'], map_sha256=plan['case']['map_sha256'], fleet_seed=plan['fleet_seed'],
                    actor_sha256=actor_sha, world_sha256=world_sha, horizon_s=horizon,
                    teacher_probability=probability, phase=phase)
    if evidence_path.is_file():
        row = json.loads(evidence_path.read_text())
        if row['identity'] != identity or row['examples_sha256'] != file_sha256(examples_path):
            raise ValueError(f'Cached episode differs: {directory}')
        return row
    if file_sha256(actor) != actor_sha or file_sha256(world) != world_sha:
        raise ValueError('An input weight file changed during collection')
    with open(os.devnull, 'w') as sink, redirect_stdout(sink):
        summary, samples = collect_pretrain_episode(repo, plan, actor, world, horizon_s=horizon,
                                                     teacher_probability=probability, out_dir=directory)
    if summary['world']['unauthorized_switches'] or (phase != 'train' and summary['teacher_actions']):
        raise ValueError(f'Invalid side change or evaluation action attribution: {directory}')
    examples = [freeze_example(e) for e in samples]
    atomic_checkpoint(examples_path, examples)
    counts = dict(Counter(e['label'] for e in examples))
    row = dict(identity=identity, episode=summary, label_counts=counts, examples_sha256=file_sha256(examples_path))
    atomic_json(evidence_path, row)
    return row


def batch_angles(model, examples):
    distribution, _ = model([e['observation'] for e in examples])
    target = torch.tensor([e['angle_rad'] for e in examples], dtype=distribution.location.dtype)
    delta = distribution.location - target
    error = torch.atan2(delta.sin(), delta.cos())
    return distribution, target, error


def fit(model, optimizer, examples, config, rng):
    """Balance observed maneuver classes and supervise both physical heading and legal-angle density."""
    if not examples:
        raise ValueError('Training episodes produced no evidence-labelled policy decisions')
    counts = Counter(e['label'] for e in examples)
    before = model.actor.weight.detach().clone()
    steps = 0
    loss_sum = 0.
    model.train()
    for _ in range(config['epochs']):
        order = rng.permutation(len(examples))
        for start in range(0, len(order), config['batch_size']):
            batch = [examples[i] for i in order[start:start + config['batch_size']]]
            distribution, target, error = batch_angles(model, batch)
            # Equal total weight per class across an epoch; each sample remains on its own native observation.
            weights = target.new_tensor([len(examples) / (len(counts) * counts[e['label']]) for e in batch])
            nll = -distribution.log_prob(torch.ones_like(target, dtype=torch.bool), target)
            direction_loss = F.huber_loss(error, torch.zeros_like(error), reduction='none')
            loss = (weights * (nll + direction_loss)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite supervised objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            steps += 1
    delta = float((model.actor.weight.detach() - before).abs().max())
    if not math.isfinite(delta) or delta <= 0. or not all(torch.isfinite(x).all() for x in model.parameters()):
        raise FloatingPointError('Expected finite vehicle weights and a measurable actor update')
    return dict(optimizer_steps=steps, mean_loss=loss_sum/(config['epochs'] * len(examples)),
                actor_max_weight_delta=delta, samples=len(examples), labels=dict(counts))


@torch.no_grad()
def score(model, examples, batch_size):
    """Report errors using the deterministic legal action that a deployed actor would issue."""
    model.eval()
    errors = defaultdict(list)
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start+batch_size]
        dist, target, _ = batch_angles(model, batch)
        _, angle = dist.action(True)
        delta = angle - target
        distances = torch.rad2deg(torch.atan2(delta.sin(), delta.cos()).abs()).tolist()
        for entry, error in zip(batch, distances):
            errors[entry['label']].append(error)
    return {label:dict(n=len(vals), mean_error_deg=float(np.mean(vals)),
                       median_error_deg=float(np.median(vals)), within_30_deg=sum(x<=30. for x in vals))
            for label, vals in errors.items()}


def run(args):
    out = args.output.resolve()
    protocol = make_manifest(args)
    manifest = out/'manifest.json'
    if manifest.is_file():
        if not args.resume or json.loads(manifest.read_text()) != protocol:
            raise ValueError('Resuming requires the original protocol, actor, sources and seeds')
    else:
        if args.resume or any(f.name != 'logs' for f in out.iterdir()):
            raise ValueError('Expected a fresh pretraining output')
        atomic_json(manifest, protocol)
    signature = file_sha256(manifest)
    ckpts = out/'checkpoints'
    ckpts.mkdir(exist_ok=True)
    actor, actor_sha = Path(protocol['actor']), protocol['actor_sha256']
    model, _ = policy_from(actor)
    model.critic.requires_grad_(False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    history = []
    training_examples = []

    def report(phase, **details):
        status = dict(phase=phase, pid=os.getpid(), completed_updates=len(history), **details)
        atomic_json(out/'progress.json', status)
        print(json.dumps(status), flush=True)

    with episode_pool(args.workers) as pool:
        def batch(phase, plans, directory, weights, sha, probability):
            report(phase, directory=str(directory), episodes=len(plans), completed_episodes=0, checkpoint_sha256=sha)
            jobs = {pool.submit(collect, args.repo.resolve(), p, weights, sha, Path(protocol['world_model']),
                                protocol['world_sha256'], args.horizon_s, probability, phase,
                                directory/f'episode-{i:02}'):i for i,p in enumerate(plans)}
            rows = [None]*len(plans)
            totals = Counter()
            for completed, job in enumerate(as_completed(jobs), 1):
                row = job.result()
                rows[jobs[job]] = row
                totals.update(row['label_counts'])
                report(phase, directory=str(directory), episodes=len(plans), completed_episodes=completed,
                       label_counts=dict(totals), checkpoint_sha256=sha)
            examples = []
            for i in range(len(rows)):
                examples.extend(torch.load(directory/f'episode-{i:02}'/'examples.pt',
                                           map_location='cpu', weights_only=True))
            return rows, examples

        for update in range(1, args.updates+1):
            start = (update-1)*args.episodes_per_update
            directory = out/'training'/f'update-{update:04}'
            plans = protocol['train'][start:start+args.episodes_per_update]
            ratio = (update-1)/max(1, args.updates-1)
            probability = (1-ratio)*args.teacher_probability + ratio*args.final_teacher_probability
            rows, fresh = batch('train', plans, directory, actor, actor_sha, probability)
            training_examples.extend(fresh)
            checkpoint = ckpts/f'heading-{update:04}.pt'
            if checkpoint.is_file():
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if (saved['manifest_sha256'] != signature or saved['parent_sha256'] != actor_sha or
                        saved['completed_updates'] != update):
                    raise ValueError('Pretraining checkpoint lineage differs')
                model.load_state_dict(saved['model'], strict=True)
                optimizer.load_state_dict(saved['optimizer'])
                record = saved['update_record']
            else:
                report('optimizing', update=update, samples=len(training_examples))
                before = score(model, training_examples,args.batch_size)
                metrics = fit(model, optimizer, training_examples, protocol, np.random.default_rng([args.seed,update]))
                after = score(model, training_examples,args.batch_size)
                record = dict(update=update, cases=[p['case_id'] for p in plans], new_samples=len(fresh),
                              new_label_counts=dict(sum((Counter(r['label_counts']) for r in rows), Counter())),
                              fit=metrics, sample_error_before=before, sample_error_after=after)
                atomic_checkpoint(checkpoint,dict(format=FORMAT, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                 environment=ENVIRONMENT, manifest_sha256=signature, parent_sha256=actor_sha,
                                 completed_updates=update, v9_sha256=CHECKPOINT_SHA, world_sha256=protocol['world_sha256'],
                                 update_record=record))
            history.append(record)
            actor, actor_sha = checkpoint, file_sha256(checkpoint)
            atomic_json(out/'training.json', history)
            report('committed', checkpoint=str(actor), checkpoint_sha256=actor_sha, labels=record['fit']['labels'])

        # Both evaluation groups execute their respective frozen actor with teacher takeover disabled.
        # Evaluation is deliberately deferred until after the first usable training checkpoints exist.
        initial = Path(protocol['actor'])
        baseline, heldout_examples = batch('baseline',protocol['evaluation'],out/'baseline',
                                            initial,protocol['actor_sha256'],0.)
        atomic_json(out/'baseline.json', baseline)
        final, self_visited = batch('paired_evaluation',protocol['evaluation'],out/'evaluation',actor,actor_sha,0.)
        original, _ = policy_from(initial)
        evidence = dict(baseline=baseline, trained=final, checkpoint_sha256=actor_sha,
                        on_baseline_states=dict(before=score(original,heldout_examples,args.batch_size),
                                                after=score(model,heldout_examples,args.batch_size)),
                        on_trained_states=score(model,self_visited,args.batch_size))
        counts = Counter(e['label'] for e in training_examples)
        missing = [label for label in TARGET_LABELS if not counts[label]]
        evidence['missing_training_classes'] = missing
        atomic_json(out/'evaluation.json',evidence)
        report('complete' if not missing else 'complete_missing_coverage', checkpoint=str(actor),
               checkpoint_sha256=actor_sha, missing_training_classes=missing, label_counts=dict(counts))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=SOURCE.parents[3])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--init-checkpoint',type=Path,required=True)
    parser.add_argument('--world-model',type=Path,default=DEFAULT_WORLD)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--updates',type=int,default=4)
    parser.add_argument('--episodes-per-update',type=int,default=4)
    parser.add_argument('--eval-episodes',type=int,default=4)
    parser.add_argument('--horizon-s',type=int,default=360)
    parser.add_argument('--seed',type=int,default=12600000)
    parser.add_argument('--eval-seed',type=int,default=12700000)
    parser.add_argument('--teacher-probability',type=float,default=.65)
    parser.add_argument('--final-teacher-probability',type=float,default=.15)
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--epochs',type=int,default=6)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if min(args.workers,args.updates,args.episodes_per_update,args.eval_episodes,args.batch_size,args.epochs)<1:
        parser.error('Expected positive training dimensions')
    if min(args.seed,args.eval_seed)<0 or args.horizon_s<=60 or not math.isfinite(args.learning_rate) or args.learning_rate<=0:
        parser.error('Invalid seed, horizon or learning rate')
    if not all(0.<=x<=1. for x in (args.teacher_probability,args.final_teacher_probability)):
        parser.error('Teacher probabilities must be in [0, 1]')
    return args


if __name__ == '__main__':
    options = arguments()
    with exclusive_run(options.output):
        run(options)
