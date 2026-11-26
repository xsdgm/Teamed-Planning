"""Simulate the frozen UAV/UGV fleet, learn obstacle extent and replay held-out scenes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import shutil

import numpy as np

from v91_extent_collect import collect_case, load_frozen
from v91_extent_scenes import make_case, rasterize_large


FRAMES = (1, 6, 12, 20, 32, 48, 64, 80)
GRID = 48
CELL = 2.
REAR = -24.
CHANNELS = ['observed_free', 'observed_occupied', 'received_from_air', 'observed_by_car', 'car_trajectory']
SPLITS = {'train': 410000, 'validation': 420000, 'test': 430000}
_runtime = None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def start_worker(run, checkpoint):
    global _runtime
    _runtime = load_frozen(run, checkpoint)


def encode(snapshot):
    """Use only observations already available at the vehicle at snapshot time."""
    car = snapshot['car_xy']
    forward = snapshot['goal_xy'] - car
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    right = np.array([forward[1], -forward[0]])
    rows, cols = np.indices((GRID, GRID))
    local_f = REAR + (GRID - rows - .5) * CELL
    local_r = (cols + .5 - GRID / 2) * CELL
    world_xy = car + local_f[..., None] * forward + local_r[..., None] * right
    ix, iy = np.floor(world_xy / snapshot['resolution']).astype(int).transpose(2, 0, 1)
    h, w = snapshot['air_cells'].shape
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    air = np.zeros((GRID, GRID), np.int8)
    ground = np.zeros_like(air)
    air[valid] = snapshot['air_cells'][iy[valid], ix[valid]]
    ground[valid] = snapshot['ground_cells'][iy[valid], ix[valid]]
    occupied = (air > 0) | (ground > 0)
    free = ((air < 0) | (ground < 0)) & ~occupied
    path = snapshot['car_path'] - car
    c = np.floor(path @ right / CELL + GRID / 2).astype(int)
    r = GRID - 1 - np.floor((path @ forward - REAR) / CELL).astype(int)
    inside = (r >= 0) & (r < GRID) & (c >= 0) & (c < GRID)
    trajectory = np.zeros_like(free)
    trajectory[r[inside], c[inside]] = True
    return np.stack([free, occupied, air != 0, ground != 0, trajectory]).astype(np.uint8), world_xy


def write_replay(path, scene, snapshots):
    positions = np.vstack((scene['start'], scene['goal'], snapshots[-1]['car_path'],
                           snapshots[-1]['drone_paths'].reshape(-1, 2)))
    res = snapshots[-1]['resolution']
    height, width = snapshots[-1]['air_cells'].shape
    lo = np.maximum(0, np.floor((positions.min(axis=0) - 32.) / res).astype(int))
    hi = np.minimum([width, height], np.ceil((positions.max(axis=0) + 32.) / res).astype(int))
    crop = np.s_[lo[1]:hi[1], lo[0]:hi[0]]
    fields = {key: np.stack([s[key] for s in snapshots]) for key in
              ('car_xy', 'drone_xy', 'drone_tiles', 'car_tile', 'received_packets')}
    for key in ('air_cells', 'ground_cells'):
        fields[key] = np.stack([s[key][crop] for s in snapshots])
    np.savez_compressed(path, seed=scene['seed'], steps=[s['frame'] for s in snapshots],
        goal=scene['goal'], start=scene['start'], car_path=snapshots[-1]['car_path'],
        drone_paths=snapshots[-1]['drone_paths'], crop_origin=lo * res, resolution=res,
        circles=np.array([[*xy, r] for xy, r in scene['circles']]),
        drone_tile_resolution=snapshots[0]['drone_tile_resolution'],
        car_tile_resolution=snapshots[0]['car_tile_resolution'],
        x=np.stack([encode(s)[0] for s in snapshots]), **fields)


def simulate(seed, demo_path, frames):
    scene = make_case(seed)
    requested = frames if demo_path is None else sorted(set(frames) | set(range(2, max(frames)+1, 2)))
    snapshots, reason = collect_case(scene, seed, *_runtime, requested)
    if not snapshots:
        raise RuntimeError(f'No valid observations for map {seed} ({reason})')
    if demo_path is not None:
        write_replay(demo_path, scene, snapshots)
    data = []
    for s in snapshots:
        if s['frame'] in frames:
            x, xy = encode(s)
            mask = rasterize_large(scene, xy)
            data.append((x, mask.astype(np.uint8), np.uint8(mask.any()), seed, s['frame']))
    metadata = dict(seed=seed, termination=reason, saved_samples=len(data),
                    last_sample_frame=snapshots[-1]['frame'], last_received_packets=snapshots[-1]['received_packets'],
                    travelled_m=float(np.linalg.norm(np.diff(snapshots[-1]['car_path'], axis=0), axis=1).sum()))
    return data, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uav-run', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--train-maps', type=int, default=96)
    parser.add_argument('--validation-maps', type=int, default=24)
    parser.add_argument('--test-maps', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    run = args.uav_run.resolve()
    ck = args.checkpoint.resolve() if args.checkpoint.is_file() else run / args.checkpoint
    counts = dict(train=args.train_maps, validation=args.validation_maps, test=args.test_maps)
    if any(n < 2 or n % 2 or n > 10000 for n in counts.values()) or args.workers < 1 or args.epochs < 1:
        parser.error('Use separate, even-sized sets of 2–10000 maps and positive workers/epochs')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'replays').mkdir()
    source = Path(__file__).resolve().parent
    (out/'sources').mkdir()
    for path in [Path(__file__).resolve(), *sorted(source.glob('v91_extent_*.py')), source/'v91_extent_replay.html']:
        shutil.copy2(path, out/'sources'/path.name)
    jobs = [(split, seed) for split, n in counts.items() for seed in range(SPLITS[split], SPLITS[split] + n)]
    demos = (SPLITS['test'], SPLITS['test']+1)
    manifest = dict(utc_start=datetime.now(timezone.utc).isoformat(), uav_run=str(run),
        checkpoint=str(ck), checkpoint_sha256=digest(ck), map_seed_starts=SPLITS, map_counts=counts,
        training_frames=FRAMES, demo_map_seeds=demos, workers=args.workers, epochs=args.epochs,
        input_channels=CHANNELS, ego_grid=dict(cells=GRID, cell_m=CELL, rear_m=REAR),
        target='Obstacle-group presence and its filled occupancy inside the car-relative 96 m window',
        supervision='Simulator truth is used for target labels and optional reference display only',
        car_policy='Native ground planner; this pilot collects its trajectories and trains only the world model')
    save_json(out/'manifest.json', manifest)
    rows = {split: [] for split in counts}
    reports = []
    with ProcessPoolExecutor(args.workers, mp_context=get_context('spawn'),
                             initializer=start_worker, initargs=(run, ck)) as pool:
        futures = {pool.submit(simulate, seed, out/'replays'/f'{seed}.npz' if seed in demos else None,
                               FRAMES): split for split, seed in jobs}
        for job in as_completed(futures):
            data, report = job.result()
            split = futures[job]
            rows[split].extend(data)
            reports.append(dict(split=split, **report))
            save_json(out/'progress.json', dict(stage='simulating', maps_complete=len(reports), maps_total=len(jobs)))
            if len(reports) % 16 == 0:
                print(f'Collected {len(reports)} / {len(jobs)} maps', flush=True)
    save_json(out/'trajectories.json', sorted(reports, key=lambda r: r['seed']))
    for split, samples in rows.items():
        samples.sort(key=lambda v: (v[3], v[4]))
        arrays = {key: np.stack(values) for key, values in zip(('x','mask','presence','seed','step'), zip(*samples))}
        np.savez_compressed(out/f'{split}.npz', **arrays)
    save_json(out/'progress.json', dict(stage='training_world_model', maps_complete=len(jobs), maps_total=len(jobs)))
    from v91_extent_network import train
    train(out/'train.npz', out/'validation.npz', out/'test.npz', out/'model', epochs=args.epochs)
    from v91_extent_replay import write_viewer
    write_viewer(out)
    save_json(out/'progress.json', dict(stage='complete', maps_complete=len(jobs), maps_total=len(jobs),
                                      utc_finish=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
