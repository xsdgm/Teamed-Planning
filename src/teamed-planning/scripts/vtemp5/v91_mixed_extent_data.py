"""Collect map-disjoint mixed-terrain data with a frozen UAV and the native vehicle."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from run_v91_extent_pilot import CELL, CHANNELS, GRID, REAR, digest, encode, save_json
from v91_extent_collect import _world_cells, load_frozen
from v91_extent_scenes import make_case, rasterize_large
from v91_lake_cases import (bank_near_mouth_mask, bridge_mask, bridge_mouth_mask,
                            install_terrain, make_lake_case, water_mask)
from v91_lake_drive import _experiment


KINDS = ('land_large', 'land_clear', 'shore', 'bridge')
SPLIT_SEEDS = {'train': 710000, 'validation': 720000, 'test': 730000}
KIND_STRIDE = 2000
FRAMES = (1, 12, 32, 60, 90, 120, 160, 200)
_runtime = None


def initialize(run, checkpoint):
    global _runtime
    _runtime = load_frozen(run, checkpoint)


def map_plan(counts):
    # Consecutive seeds within each terrain balance left/right and the four
    # cardinal orientations in lake/bridge scenarios.
    return [(split, kind, SPLIT_SEEDS[split] + i * KIND_STRIDE + n)
            for split, count in counts.items()
            for n in range(count // len(KINDS)) for i, kind in enumerate(KINDS)]


def make_example(scene, kind, step, env, ground, path):
    # These are the only observations passed to the network. Simulator truth
    # first appears below encode(), solely for labelling its output pixels.
    car = path[-1]
    observed = dict(car_xy=car, goal_xy=env.goal, car_path=np.stack(path),
                    resolution=env.ledger.resolution, air_cells=_world_cells(env.ledger),
                    ground_cells=_world_cells(ground))
    x, xy = encode(observed)
    mask = water_mask(scene, xy) if kind in ('shore', 'bridge') else rasterize_large(scene, xy)
    empty = np.zeros_like(mask, dtype=np.uint8)
    out = dict(x=x, mask=mask.astype(np.uint8), presence=np.uint8(mask.any()), seed=scene['seed'],
               frame=step, step=step, kind=kind, car_xy=car.copy(), goal_xy=env.goal.copy(),
               received_packets=len(env.ledger.received))
    out['deck'] = bridge_mask(scene, xy).astype(np.uint8) if kind == 'bridge' else empty
    for end in ('entry', 'exit'):
        if kind == 'bridge':
            mouth = bridge_mouth_mask(scene, xy, end).astype(np.uint8)
            bank = bank_near_mouth_mask(scene, xy, end).astype(np.uint8)
            if (mouth & mask).any() or (bank & ~mask).any():
                raise ValueError(f'Inconsistent bridge truth, map {scene["seed"]} frame {step}')
        else:
            mouth = bank = empty
        out[end + '_mouth'] = mouth
        out[end + '_bank'] = bank
    return out


def simulate(kind, seed, frames):
    from mappo.mappo_experiment import DeliveredEvidence, SafetyTermination

    scene = (make_lake_case(seed, kind) if kind in ('shore', 'bridge')
             else make_case(seed, has_large=(kind == 'land_large')))
    model, config = _runtime
    sample_seed = int(np.random.SeedSequence([config['seed'], seed, 0]).generate_state(1)[0])
    torch.manual_seed(sample_seed)
    path, samples = [], []
    reason = 'frame_limit'
    with torch.inference_mode(), open(os.devnull, 'w') as sink, redirect_stdout(sink):
        env = _experiment(scene, seed, model, config)
        try:
            if kind in ('shore', 'bridge'):
                install_terrain(env, scene)
            ground = DeliveredEvidence(world=env.world_size, resolution=env.ledger.resolution)
            for step in range(frames[-1] + 1):
                car = env.obs[env.ground]
                if car['map_axes'] != 'yx':
                    raise ValueError('Expected native YX ground occupancy')
                position = np.asarray(car['self']['position'][:2], dtype=float).copy()
                path.append(position)
                ground.receive(env.ground, step, position, car['occupancy_map'], car['map_resolution'],
                               valid=car.get('sensor_valid', True))
                if step in frames:
                    samples.append(make_example(scene, kind, step, env, ground, path))
                if env.goal_reached:
                    reason = 'vehicle_arrival'
                    break
                if step == frames[-1]:
                    break
                try:
                    env.step(step)
                except SafetyTermination:
                    reason = 'safety_termination'
                    break
        finally:
            env.env.close()
    if not samples:
        raise RuntimeError(f'Map {seed} produced no sampled frames ({reason})')
    arrays = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
    return arrays, reason


def save_npz(path, arrays):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def worker(base, split, kind, seed, frames):
    arrays, reason = simulate(kind, seed, frames)
    save_npz(base / 'maps' / split / f'{seed}.npz',
             dict(**arrays, termination=np.asarray(reason)))
    return split, seed


def merge_split(base, split, jobs):
    chunks = {}
    summaries = []
    for _, kind, seed in [j for j in jobs if j[0] == split]:
        with np.load(base / 'maps' / split / f'{seed}.npz', allow_pickle=False) as data:
            assert np.all(data['seed'] == seed) and np.all(data['kind'] == kind)
            summaries.append(dict(seed=seed, kind=kind, samples=len(data['frame']),
                                  termination=str(data['termination'].item()),
                                  last_frame=int(data['frame'][-1]),
                                  last_received_packets=int(data['received_packets'][-1])))
            for key in data.files:
                if key != 'termination':
                    chunks.setdefault(key, []).append(data[key])
    save_npz(base / f'{split}.npz', {k: np.concatenate(values) for k, values in chunks.items()})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uav-run', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('episode_00768.pt'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--train-maps', type=int, default=160)
    parser.add_argument('--validation-maps', type=int, default=32)
    parser.add_argument('--test-maps', type=int, default=48)
    parser.add_argument('--frames', type=int, nargs='+', default=FRAMES)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    counts = dict(train=args.train_maps, validation=args.validation_maps, test=args.test_maps)
    if any(count < 4 or count % 4 or count > KIND_STRIDE * 4 for count in counts.values()):
        parser.error('Split map counts must be multiples of four between 4 and 8000')
    if args.workers < 1 or min(args.frames) < 1 or list(args.frames) != sorted(set(args.frames)):
        parser.error('Use positive workers and strictly increasing positive frames')
    base = args.output.resolve()
    run = args.uav_run.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint.is_file() else run / args.checkpoint
    sources = [Path(__file__).resolve(), *[Path(__file__).with_name(f'{name}.py') for name in
               ('run_v91_extent_pilot', 'v91_extent_scenes', 'v91_extent_collect', 'v91_lake_cases',
                'v91_lake_drive', 'v91_lake_metrics', 'v91_extent_network')]]
    protocol = dict(format='v91_mixed_extent_data_v1', uav_run=str(run), checkpoint=str(checkpoint),
                    checkpoint_sha256=digest(checkpoint), map_counts=counts, seed_starts=SPLIT_SEEDS,
                    terrain_order=list(KINDS), kind_stride=KIND_STRIDE, frames=list(args.frames),
                    channels=CHANNELS, cells=GRID, cell_m=CELL, rear_m=REAR,
                    supervision='Large land obstacle group extent or impassable water. A dry bridge is free.',
                    car_policy='Native ground planner, no world model in control or observation collection.',
                    source_sha256={f.name: digest(f) for f in sources})
    if args.resume:
        if json.loads((base / 'manifest.json').read_text()) != protocol:
            raise ValueError('Cannot resume a different collection protocol in this directory')
    else:
        base.mkdir(parents=True, exist_ok=False)
        (base / 'sources').mkdir()
        for src in sources:
            shutil.copy2(src, base / 'sources' / src.name)
        for split in counts:
            (base / 'maps' / split).mkdir(parents=True)
        save_json(base / 'manifest.json', protocol)
    jobs = map_plan(counts)
    missing = [job for job in jobs if not (base / 'maps' / job[0] / f'{job[2]}.npz').exists()]
    done = len(jobs) - len(missing)
    def progress(phase):
        save_json(base / 'progress.json', dict(phase=phase, maps_complete=done, maps_total=len(jobs)))
    progress('collecting')
    with ProcessPoolExecutor(args.workers, mp_context=get_context('spawn'), initializer=initialize,
                             initargs=(run, checkpoint)) as pool:
        futures = [pool.submit(worker, base, *j, tuple(args.frames)) for j in missing]
        for future in as_completed(futures):
            future.result()
            done += 1
            progress('collecting')
            if done % 16 == 0:
                print(f'Collected {done}/{len(jobs)} maps', flush=True)
    reports = {split: merge_split(base, split, jobs) for split in counts}
    save_json(base / 'map_reports.json', reports)
    progress('complete')


if __name__ == '__main__':
    main()
