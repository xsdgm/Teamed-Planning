"""Audit physical map connectivity in each half-plane, independently of the policy."""
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vtemp5.vtemp5_large import VTemp5LargeObstacleEnv


def reachable(mask, start, stop_col, left):
    """Flood-fill from the car, allowing the centre line and just one side."""
    middle, start_col = start
    seen = np.zeros_like(mask)
    seen[start] = True
    queue = deque([(middle, start_col)])
    height, width = mask.shape
    low, high = (middle, height) if left else (0, middle + 1)
    while queue:
        y, x = queue.popleft()
        if x == stop_col:
            return True
        for iy, ix in ((y, x + 1), (y + 1, x), (y - 1, x), (y, x - 1)):
            if low <= iy < high and 0 <= ix < width and mask[iy, ix] and not seen[iy, ix]:
                seen[iy, ix] = True
                queue.append((iy, ix))
    return False


def inspect(terrain, seed):
    env = VTemp5LargeObstacleEnv(seed, terrain=terrain)
    # A 2 m grid resolves the bridge and spans both complete detour corridors.
    xs = np.arange(0, 251, 2)
    ys = np.arange(-224, 225, 2)
    mask = np.array([[env.terrain_traversable(x, y) for x in xs] for y in ys], dtype=bool)
    row = len(ys) // 2
    blocked = np.flatnonzero(~mask[row])
    if not len(blocked) or blocked[-1] >= len(xs) - 2:
        raise ValueError(f'Seed {seed}: expected a finite straight-line obstruction inside the audit grid')
    cross = (blocked[0] + blocked[-1]) // 2
    lateral = np.abs(ys[mask[:, cross]])
    if not len(lateral):
        raise ValueError(f'Seed {seed}: no traversable crossing in the audit extent')
    stop = blocked[-1] + 2
    sides = [name for name, left in [('left', True), ('right', False)]
             if reachable(mask, (row, 0), stop, left)]
    result = dict(terrain=terrain, seed=seed, reachable_sides=sides,
                  blocked_centreline_m=int(len(blocked) * 2),
                  nearest_clear_lateral_at_midpoint_m=int(min(lateral)))
    return result, (xs, ys, mask)


def plot(samples, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(1, len(samples), figsize=(8.4, 5.9), sharey=True, layout='constrained')
    for axis, (row, (xs, ys, grid)) in zip(axes, samples):
        detour = row['terrain'] == 'detour'
        axis.imshow(grid, cmap=ListedColormap(['#97724f' if detour else '#699ab5', '#f6f5ef']),
                    origin='lower', extent=(xs[0]-1, xs[-1]+1, ys[0]-1, ys[-1]+1), interpolation='nearest')
        axis.axhline(0, color='#363636', linestyle='--', linewidth=.8, label='Goal bearing (0°)')
        axis.plot(0, 0, 'o', color='#9a3434', clip_on=False, label='Vehicle start')
        axis.set_title('Stage 1 · detour on either side' if detour else 'Stage 2 · river with one bridge')
        axis.set_xlabel('Forward distance (m)')
        axis.set_xlim(-5, xs[-1])
        axis.legend(loc='lower right', fontsize=8)
    axes[0].set_ylabel('Left of initial goal bearing (m)')
    fig.savefig(path, dpi=155)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=8100000)
    parser.add_argument('--seeds', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('At least one map per stage is required')
    rows, samples, summary = [], [], {}
    for terrain in ('detour', 'bridge'):
        stage = []
        for seed in range(args.seed, args.seed + args.seeds):
            row, grid = inspect(terrain, seed)
            stage.append(row)
            if seed == args.seed:
                samples.append((row, grid))
        rows.extend(stage)
        summary[terrain] = dict(
            maps=len(stage),
            connected_sides=dict(Counter('+'.join(r['reachable_sides']) or 'neither' for r in stage)),
            min_centreline_blocked_m=min(r['blocked_centreline_m'] for r in stage),
            min_required_lateral_m=min(r['nearest_clear_lateral_at_midpoint_m'] for r in stage))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(grid_resolution_m=2, summary=summary, maps=rows), indent=2) + '\n')
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot(samples, args.plot)
    print(json.dumps(summary))
    if any(len(row['reachable_sides']) != (2 if row['terrain'] == 'detour' else 1) for row in rows):
        raise SystemExit('Unexpected number of connected sides on the physical map')


if __name__ == '__main__':
    main()
