"""Synchronous, resumable mixed-curriculum PPO for the VTemp4 heading policy."""
import argparse
from concurrent.futures import as_completed
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))

from world_model.frozen_navigation import CHECKPOINT_SHA, V9_RUN, runtime
from world_model.hierarchical_sources import atomic_json, file_sha256, package_python_files
from world_model.train_slow_policy import atomic_checkpoint
from simulator.train_vtemp4 import FORMAT, by_stratum
from simulator.vtemp4_curriculum import (
    KM_DATA, LAKE_DATA, LAKE_GEOMETRY, STRATA, geometry_hash, load_curriculum, source_id, stratum_counts,
)
from ground.vtemp4_policy import HeadingPolicy, update_policy


CLOCK = dict(review_interval_s=15, lookahead_m=200.)
FORMAL_FORMAT = 'vtemp4_synchronous_full_missions_v1'


def sample_plan(cases, updates, batch_size, seed):
    """Balance each batch and each class deck, including across shuffles of small decks."""
    if min(updates, batch_size) < 1 or batch_size % len(STRATA):
        raise ValueError('Expected positive updates and a batch size divisible by the number of strata')
    groups = {task: [c for c in cases if c['navigation_task'] == task] for task in STRATA}
    if min(map(len, groups.values())) < batch_size // len(STRATA):
        raise ValueError('Expected enough distinct maps in each stratum for a full batch')
    rng = random.Random(seed)
    seeds = iter(rng.sample(range(2**31), updates * batch_size))
    decks = {task: [] for task in STRATA}
    plan = []
    for _ in range(updates):
        batch, selected = [], set()
        for slot in range(batch_size):
            task = STRATA[slot % len(STRATA)]
            if not decks[task]:
                fresh = groups[task].copy()
                rng.shuffle(fresh)
                # pop() consumes the end. Defer maps already used in this batch.
                fresh.sort(key=lambda c: c['case_id'] not in selected)
                decks[task] = fresh
            case = decks[task].pop()
            selected.add(case['case_id'])
            batch.append(dict(case_id=case['case_id'], fast_seed=next(seeds), navigation_task=task,
                              geometry_sha256=geometry_hash(case)))
        plan.append(batch)
    return plan


def source_digest(root):
    hashes = {p.as_posix(): file_sha256(root / p) for p in package_python_files(root)}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def source_tree_digest(root):
    """Fingerprint all packages in a VTemp4 source snapshot."""
    root = Path(root)
    hashes = {}
    for name in ('common', 'drone', 'ground', 'simulator', 'world_model'):
        package = root / name
        if not package.is_dir():
            raise ValueError(f'Missing frozen source package: {package}')
        hashes.update({(Path(name) / p).as_posix(): file_sha256(package / p)
                       for p in package_python_files(package)})
    if not hashes:
        raise ValueError(f'Missing frozen sources at {root}')
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def build_manifest(args, cases, hashes):
    fast = args.repo / V9_RUN
    plan = sample_plan(cases, args.updates, args.episodes_per_update, args.seed)
    if file_sha256(fast / 'best.pt') != CHECKPOINT_SHA:
        raise ValueError('Frozen V9 checkpoint differs')
    return dict(format=FORMAL_FORMAT, policy_format=FORMAT, seed=args.seed, updates=args.updates,
                km_data=str(args.km_data), field_data=str(args.field_data), field_geometry=LAKE_GEOMETRY,
                episodes_per_update=args.episodes_per_update, workers=args.workers, curriculum='lakes',
                clock=CLOCK, collector_seconds=None, gamma=1., learning_rate=args.learning_rate,
                v9_sha256=CHECKPOINT_SHA, v9_config_sha256=file_sha256(fast / 'config.json'),
                v9_sources_sha256=source_digest(fast / 'runtime_sources'), world_model_sha256=source_tree_digest(SOURCE.parent),
                libraries={name: version(name) for name in ('torch', 'numpy', 'scipy', 'networkx', 'scikit-image')},
                python_version=sys.version, stratum_counts=stratum_counts(cases), plan=plan, **hashes)


@contextmanager
def exclusive_run(out):
    out.mkdir(parents=True, exist_ok=True)
    fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another training process owns {out}') from exc
        yield
    finally:
        os.close(fd)


def checkpoint_path(out, update):
    return out / 'checkpoints' / f'heading-update-{update:04d}.pt'


def rng_state():
    algo, keys, pos, gauss, cached = np.random.get_state()
    return dict(python=random.getstate(), numpy=(algo, keys.tolist(), pos, gauss, cached),
                torch=torch.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    algo, keys, pos, gauss, cached = state['numpy']
    np.random.set_state((algo, np.array(keys, np.uint32), pos, gauss, cached))
    torch.set_rng_state(state['torch'])


def save_update(path, manifest_sha, previous_sha, update, model, optimizer, seen, record, dataset):
    if path.exists():
        raise FileExistsError(f'Immutable checkpoint already exists: {path}')
    atomic_checkpoint(path, dict(format=FORMAT, curriculum='lakes', field_geometry=LAKE_GEOMETRY,
                                 dataset=dataset, clock=CLOCK, completed_updates=update,
                                 v9_sha256=CHECKPOINT_SHA, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                 formal_format=FORMAL_FORMAT, manifest_sha256=manifest_sha,
                                 previous_checkpoint_sha256=previous_sha, rng=rng_state(), update_record=record,
                                 **{key: sorted(vals) for key, vals in seen.items()}))


def load_commits(out, manifest_sha):
    """Check the contiguous hash chain. A published checkpoint is the transaction boundary."""
    history = []
    last = None
    prior_sha = None
    for index, path in enumerate(sorted((out / 'checkpoints').glob('heading-update-*.pt'))):
        if path != checkpoint_path(out, index):
            raise ValueError(f'Gap in checkpoint history at update {index}')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if (state['formal_format'] != FORMAL_FORMAT or state['manifest_sha256'] != manifest_sha
                or state['completed_updates'] != index or state['previous_checkpoint_sha256'] != prior_sha):
            raise ValueError(f'Checkpoint chain differs at {path}')
        if index:
            history.append(state['update_record'])
        last = state
        prior_sha = file_sha256(path)
    return history, last, prior_sha


def publish_latest(out, path):
    temporary = out / '.heading-link.tmp'
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(path.relative_to(out))
    temporary.replace(out / 'heading.pt')


def run(args):
    from simulator.vtemp4_parallel import collect_episode, episode_pool, load_rollout

    out = args.output
    splits, hashes = load_curriculum(args.km_data, args.field_data, field_geometry=LAKE_GEOMETRY)
    dataset = dict(km_data=str(args.km_data), field_data=str(args.field_data), hashes=hashes)
    cases = {case['case_id']: case for case in splits['ppo_train']}
    manifest = build_manifest(args, list(cases.values()), hashes)
    manifest_file = out / 'manifest.json'
    if args.resume:
        if json.loads(manifest_file.read_text()) != manifest:
            raise ValueError('Resume requires the same configuration, schedule, data, software and environment')
    else:
        if any(f.name not in {'logs', 'sources'} for f in out.iterdir()):
            raise FileExistsError(f'Use a fresh output directory or --resume: {out}')
        atomic_json(manifest_file, manifest)
    manifest_sha = file_sha256(manifest_file)
    directory = out / 'checkpoints'
    directory.mkdir(exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    runtime(args.repo)
    model = HeadingPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    seen = {k: set() for k in ('training_maps', 'training_sources', 'training_families')}
    history, state, checkpoint_sha = load_commits(out, manifest_sha)
    if state is None:
        save_update(checkpoint_path(out, 0), manifest_sha, None, 0, model, optimizer, seen, None, dataset)
        checkpoint_sha = file_sha256(checkpoint_path(out, 0))
    else:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'])
        seen = {k: set(state[k]) for k in seen}
    completed = len(history)
    saved_history = out / 'training.json'
    if saved_history.exists():
        published = json.loads(saved_history.read_text())
        if history[:len(published)] != published:
            raise ValueError('Published training history differs from the checkpoint chain')
    atomic_json(saved_history, history)
    publish_latest(out, checkpoint_path(out, completed))
    status = dict(completed_updates=completed, total_updates=args.updates, workers=args.workers,
                  episodes_per_update=args.episodes_per_update, pid=os.getpid())

    def report(phase, **fields):
        atomic_json(out / 'progress.json', dict(status, phase=phase, **fields))

    try:
        with episode_pool(args.workers) as pool:
            for update in range(completed + 1, args.updates + 1):
                batch = manifest['plan'][update - 1]
                locations = [out / 'episodes' / f'update-{update:04d}' / f'episode-{slot:02d}'
                             for slot in range(len(batch))]
                path = checkpoint_path(out, update - 1)
                report('sampling', current_update=update, completed_episodes=0)
                print(f'Update {update}/{args.updates}: collecting {len(batch)} on-policy missions', flush=True)
                jobs = {pool.submit(collect_episode, args.repo, path, checkpoint_sha, cases[row['case_id']],
                                    row['fast_seed'], location, mode='train'): slot
                        for slot, (row, location) in enumerate(zip(batch, locations))}
                summaries = [None] * len(jobs)
                for count, job in enumerate(as_completed(jobs), 1):
                    slot = jobs[job]
                    row = job.result()
                    expected = batch[slot]
                    if (any(row[key] != expected[key] for key in expected)
                            or row['end_reason'] not in {'arrival', 'time_budget', 'safety_failure'}):
                        raise ValueError(f'Unexpected complete-episode result in {locations[slot]}')
                    summaries[slot] = row
                    report('sampling', current_update=update, completed_episodes=count)
                report('optimizing', current_update=update, completed_episodes=len(batch))
                rollouts = [load_rollout(location) for location in locations]
                metrics = update_policy(model.train(), optimizer, rollouts)
                del rollouts
                with torch.no_grad():
                    upper = model.actor.bias[0] + model.actor.weight[0].abs().sum()
                    metrics['max_possible_turn_logit'] = float(upper)
                for row in batch:
                    case = cases[row['case_id']]
                    seen['training_maps'].add(geometry_hash(case))
                    seen['training_sources'].add(source_id(case))
                    seen['training_families'].add(case['terrain_family_id'])
                record = dict(update=update, episodes=summaries, by_stratum=by_stratum(summaries, stratum_counts(batch)),
                              ppo=metrics)
                path = checkpoint_path(out, update)
                save_update(path, manifest_sha, checkpoint_sha, update, model, optimizer, seen, record, dataset)
                checkpoint_sha = file_sha256(path)
                publish_latest(out, path)
                history.append(record)
                atomic_json(saved_history, history)
                status['completed_updates'] = update
                report('committed', current_update=update, checkpoint_sha256=checkpoint_sha)
                print(f"Committed {path.name}, {metrics['transitions']} slow decisions", flush=True)
        report('complete', checkpoint_sha256=checkpoint_sha)
    except BaseException as exc:
        report('interrupted', error=f'{type(exc).__name__}: {exc}')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=SOURCE.parents[3])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--km-data', type=Path, help='Override the kilometre circle data directory')
    parser.add_argument('--field-data', type=Path, help='Override the audited bilateral lake data directory')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--updates', type=int, default=50)
    parser.add_argument('--episodes-per-update', type=int, default=16)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (min(args.workers, args.updates, args.episodes_per_update) < 1 or args.episodes_per_update % len(STRATA)
            or not 0 <= args.seed < 2**32 or not 0 < args.learning_rate <= 1 or not math.isfinite(args.learning_rate)):
        parser.error('Expected valid worker count, balanced batch size, seed, updates and learning rate')
    args.output, args.repo = args.output.resolve(), args.repo.resolve()
    args.km_data = args.km_data.resolve() if args.km_data else args.repo / KM_DATA
    args.field_data = args.field_data.resolve() if args.field_data else args.repo / LAKE_DATA
    with exclusive_run(args.output):
        run(args)


if __name__ == '__main__':
    main()
