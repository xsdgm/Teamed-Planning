"""Package saved closed-loop simulations as a self-contained offline viewer."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np


def _pack_grid(grid):
    """Run lengths are unsigned 16-bit little endian, followed by one cell byte."""
    grid = np.ascontiguousarray(grid, dtype=np.uint8)
    flat = grid.ravel()
    changes = np.flatnonzero(np.diff(flat)) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(flat)]
    chunks = bytearray()
    for start, end in zip(starts, ends):
        remain = int(end - start)
        while remain:
            count = min(remain, 65535)
            chunks.extend((count & 255, count >> 8, int(flat[start])))
            remain -= count
    return [*grid.shape, base64.b64encode(chunks).decode('ascii')]


def _pack_evidence(grid):
    # Code 0 unknown, 1 observed free, 2 observed occupied.
    return _pack_grid(np.where(grid < 0, 1, np.where(grid > 0, 2, 0)))


def _points(values):
    return np.round(np.asarray(values, dtype=float), 3).tolist()


def _scene(path, checkpoint, mask_threshold):
    with np.load(path, allow_pickle=False) as data:
        scene = {key: _points(data[key]) for key in ('car_path', 'drone_paths', 'crop_origin',
                                                      'goal', 'start', 'circles')}
        scene.update(seed=int(data['seed'].item()), steps=data['steps'].astype(int).tolist(),
                     resolution=float(data['resolution'].item()),
                     drone_res=float(data['drone_tile_resolution'].item()),
                     car_res=float(data['car_tile_resolution'].item()))
        probs = masks = None
        if checkpoint.is_file():
            try:
                from .v91_extent_network import predict
            except ImportError:
                from v91_extent_network import predict
            probs, masks = predict(checkpoint, data['x'])
        scene['frames'] = []
        for k, step in enumerate(scene['steps']):
            frame = dict(car=_points(data['car_xy'][k]), drones=_points(data['drone_xy'][k]),
                         air=_pack_evidence(data['air_cells'][k]),
                         ground=_pack_evidence(data['ground_cells'][k]),
                         drone_tiles=[_pack_evidence(tile) for tile in data['drone_tiles'][k]],
                         car_tile=_pack_evidence(data['car_tile'][k]),
                         received=int(data['received_packets'][k]))
            if probs is not None:
                frame.update(presence=float(probs[k]), mask=_pack_grid(masks[k] >= mask_threshold))
            scene['frames'].append(frame)
    return scene


def write_viewer(run_dir):
    run_dir = Path(run_dir)
    paths = sorted((run_dir / 'replays').glob('*.npz'))
    if not paths:
        raise ValueError('No replay archives in this run')
    checkpoint = run_dir / 'model' / 'best.pt'
    metadata = run_dir / 'model' / 'metrics.json'
    if metadata.is_file():
        mask_threshold = json.loads(metadata.read_text())['mask_threshold']
    elif checkpoint.is_file():
        import torch
        mask_threshold = torch.load(checkpoint, map_location='cpu', weights_only=True)['mask_threshold']
    else:
        mask_threshold = .5
    scenes = [_scene(path, checkpoint, mask_threshold) for path in paths]
    payload = json.dumps(dict(scenes=scenes, mask_threshold=mask_threshold), separators=(',', ':'), allow_nan=False)
    template = Path(__file__).with_name('v91_extent_replay.html').read_text(encoding='utf-8')
    output = run_dir / 'simulator.html'
    temporary = run_dir / 'simulator.html.tmp'
    temporary.write_text(template.replace('__REPLAY_DATA__', payload), encoding='utf-8')
    temporary.replace(output)
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    print(write_viewer(parser.parse_args().run_dir))
