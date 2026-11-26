"""Run a previously trained V9.1 obstacle predictor on new closed-loop 2D simulations."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v91_extent_pilot import digest, save_json, write_replay
from v91_extent_collect import collect_case, load_frozen
from v91_extent_network import predict
from v91_extent_replay import write_viewer
from v91_extent_scenes import make_case


def simulate_maps(pilot, output, seeds, steps=80, stride=2):
    """Save paths, received observations, per-frame probabilities and an offline viewer."""
    pilot = Path(pilot).resolve()
    output = Path(output).resolve()
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError('Supply at least one distinct, nonnegative map seed')
    if steps < 1 or stride < 1:
        raise ValueError('Steps and recording stride must both be positive')
    info = json.loads((pilot / 'manifest.json').read_text())
    checkpoint = Path(info['checkpoint'])
    if digest(checkpoint) != info['checkpoint_sha256']:
        raise ValueError('The UAV checkpoint has changed since this world model was trained')
    weight = pilot / 'model' / 'best.pt'
    if not weight.is_file():
        raise FileNotFoundError(f'Trained world-model weights are required: {weight}')
    model, config = load_frozen(info['uav_run'], checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('replays', 'predictions', 'model'):
        (output / name).mkdir()
    saved_weight = output / 'model' / 'best.pt'
    shutil.copy2(weight, saved_weight)
    metrics = weight.with_name('metrics.json')
    if metrics.is_file():
        shutil.copy2(metrics, output / 'model' / metrics.name)
    frames = sorted({1, steps, *range(stride, steps + 1, stride)})
    manifest = dict(source_pilot=str(pilot), uav_run=info['uav_run'],
                    checkpoint=str(checkpoint), checkpoint_sha256=info['checkpoint_sha256'],
                    world_model_sha256=digest(saved_weight), seeds=list(seeds),
                    requested_steps=steps, recording_stride=stride, cases=[])
    manifest_path = output / 'manifest.json'
    save_json(manifest_path, manifest)
    for seed in seeds:
        scene = make_case(seed)
        snapshots, reason = collect_case(scene, seed, model, config, frames, include_terminal=True)
        if not snapshots:
            manifest['cases'].append(dict(seed=seed, termination=reason, saved_frames=0))
            save_json(manifest_path, manifest)
            continue
        replay = output / 'replays' / f'{seed}.npz'
        write_replay(replay, scene, snapshots)
        with np.load(replay, allow_pickle=False) as archive:
            presence, extent = predict(saved_weight, archive['x'])
            np.savez_compressed(output / 'predictions' / f'{seed}.npz', steps=archive['steps'],
                                presence_probability=presence, extent_probability=extent)
        last = snapshots[-1]
        travelled = np.linalg.norm(np.diff(last['car_path'], axis=0), axis=1).sum()
        manifest['cases'].append(dict(seed=seed, termination=reason, saved_frames=len(snapshots),
                                      last_step=last['frame'], travelled_m=float(travelled),
                                      goal_remaining_m=float(np.linalg.norm(last['goal_xy'] - last['car_xy'])),
                                      received_packets=last['received_packets']))
        save_json(manifest_path, manifest)
    if not any(case['saved_frames'] for case in manifest['cases']):
        raise RuntimeError(f'All simulations ended before the first sample; see {manifest_path}')
    return write_viewer(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', required=True, type=Path,
                        help='Completed world-model training folder')
    parser.add_argument('--output', required=True, type=Path,
                        help='New folder for the simulation results')
    parser.add_argument('--seed', required=True, action='append', type=int,
                        help='Map seed (repeatable): even seeds include a large obstacle, odd seeds only small ones')
    parser.add_argument('--steps', type=int, default=80, help='Maximum one-second simulation steps per map')
    parser.add_argument('--stride', type=int, default=2, help='Record a frame every N simulation steps')
    args = parser.parse_args()
    print(simulate_maps(args.pilot, args.output, args.seed, args.steps, args.stride))


if __name__ == '__main__':
    main()
