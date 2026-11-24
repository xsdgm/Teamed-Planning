"""Measure one-second motion and 30-second decision outcomes from vtemp4 records."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[4]
FORMAL_RUN = Path('artifacts/vtemp4-goal-residual-route-circles-30s-r10-400ep-20261122')
REPAIR_RUN = Path('artifacts/vtemp4-ground-route-repair-20261122/diagnostics/stable-known-target')
FORMAL_UPDATES = (1, 8, 15, 22)
REPAIR_CASES = (1228722198, 1986560903, 534945890)
INTERVAL_S = 30
MOTION_THRESHOLD_M = 0.01


def distribution(values):
    if not len(values):
        return {'count': 0}
    q10, q50, q90 = np.quantile(values, [0.1, 0.5, 0.9])
    return {'count': len(values), 'p10': float(q10), 'p50': float(q50), 'p90': float(q90)}


class Measurements:
    def __init__(self):
        self.ends = Counter()
        self.full_intervals = 0
        self.stationary = 0
        self.partial = 0
        self.active_seconds = []
        self.full_interval_net = []
        self.active_interval_net = []
        self.active_interval_yaw = []

    def measure(self, episode):
        self.ends[episode['end_reason']] += 1
        trajectory = np.asarray(episode['trajectory_xy'], dtype=np.float64)
        if trajectory.shape != (int(episode['sim_time_s']) + 1, 2) or not np.isfinite(trajectory).all():
            raise ValueError('expected one finite XY trajectory sample per simulation second')
        reviews = episode['heading_reviews']
        previous_end = 0
        for i, review in enumerate(reviews):
            start = int(review['time_s'])
            seconds = int(review['duration_s'])
            end = start + seconds
            if start != previous_end or end >= len(trajectory) or not 0 < seconds <= INTERVAL_S:
                raise ValueError('unexpected decision timestamps')
            if not np.allclose(trajectory[start], review['position'], rtol=0, atol=1e-4):
                raise ValueError('trajectory disagrees with decision position')
            previous_end = end
            if seconds != INTERVAL_S:
                if i != len(reviews) - 1:
                    raise ValueError('partial decision interval before terminal step')
                self.partial += 1
                continue
            self.full_intervals += 1
            trace = trajectory[start:end + 1]
            metres_per_second = np.linalg.norm(np.diff(trace, axis=0), axis=1)
            self.active_seconds.extend(metres_per_second[metres_per_second > MOTION_THRESHOLD_M])
            net = float(np.linalg.norm(trace[-1] - trace[0]))
            self.full_interval_net.append(net)
            if metres_per_second.sum() <= MOTION_THRESHOLD_M:
                self.stationary += 1
                continue
            self.active_interval_net.append(net)
            if i + 1 < len(reviews):
                delta = reviews[i + 1]['vehicle_yaw_rad'] - review['vehicle_yaw_rad']
                self.active_interval_yaw.append(abs(math.degrees(math.remainder(delta, 2 * math.pi))))
        if previous_end != len(trajectory) - 1:
            raise ValueError('decision intervals do not cover the trajectory')

    def summary(self):
        return {
            'end_reason_counts': dict(sorted(self.ends.items())),
            'full_intervals': self.full_intervals,
            'excluded_partial_intervals': self.partial,
            'stationary_intervals': self.stationary,
            'stationary_interval_fraction': self.stationary / self.full_intervals,
            'active_speed_mps': distribution(self.active_seconds),
            'all_interval_displacement_m': distribution(self.full_interval_net),
            'moving_interval_displacement_m': distribution(self.active_interval_net),
            'moving_interval_abs_yaw_change_deg': distribution(self.active_interval_yaw),
        }


def source_files():
    # Complete update batches keep all outcomes, including stalled trajectories.
    for update in FORMAL_UPDATES:
        for index in range(16):
            yield 'formal', FORMAL_RUN / f'episodes/update-{update:04d}/episode-{index:02d}/episode.json'
    for seed in REPAIR_CASES:
        yield 'repair', REPAIR_RUN / f'episodes/v9-local-train-{seed}/episode.json'


def calibrate(repo):
    groups = {}
    sources = []
    for cohort, path in source_files():
        raw = (repo / path).read_bytes()
        episode = json.loads(raw)
        outcome = 'arrived' if episode['goal_reached'] else 'unreached'
        group = f'{cohort}_{outcome}'
        if group not in groups:
            groups[group] = Measurements()
        groups[group].measure(episode)
        sources.append({
            'cohort': cohort, 'case_id': episode['case_id'], 'end_reason': episode['end_reason'],
            'file': path.as_posix(), 'sha256': hashlib.sha256(raw).hexdigest(),
        })
    # The pilot motion parameters use all formal records. Keep the repaired controller
    # as an external contrast, since its outcome mix and motion policy differ.
    formal = Measurements()
    for group, measurements in groups.items():
        if group.startswith('formal_'):
            formal.ends.update(measurements.ends)
            formal.full_intervals += measurements.full_intervals
            formal.partial += measurements.partial
            formal.stationary += measurements.stationary
            formal.active_seconds.extend(measurements.active_seconds)
            formal.full_interval_net.extend(measurements.full_interval_net)
            formal.active_interval_net.extend(measurements.active_interval_net)
            formal.active_interval_yaw.extend(measurements.active_interval_yaw)
    stats = formal.summary()
    return {
        'schema': 'vtemp5_motion_calibration_v1',
        'parameters': {
            'review_interval_s': INTERVAL_S,
            'moving_interval_displacement_m': stats['moving_interval_displacement_m']['p50'],
            'stationary_interval_fraction': stats['stationary_interval_fraction'],
            'active_speed_mps': stats['active_speed_mps']['p50'],
            'moving_interval_abs_yaw_change_deg': stats['moving_interval_abs_yaw_change_deg']['p50'],
        },
        'method': {
            'formal_sample': 'All 16 episodes in each of the four fixed completed vtemp4 updates.',
            'formal_updates': list(FORMAL_UPDATES),
            'repair_sample': 'Three existing stable-known-target diagnostic cases, comparison only.',
            'weighting': 'Pooled one-second samples for speed and pooled complete 30-second decisions for interval measures.',
            'threshold_m': MOTION_THRESHOLD_M,
            'stationary': 'Total distance travelled within the complete interval at most threshold_m; a loop counts as moving.',
            'active_speed_mps': 'Per-second distance conditional on movement exceeding threshold_m.',
            'moving_interval_displacement_m': 'Endpoint separation conditional on the interval having movement.',
            'moving_interval_abs_yaw_change_deg': 'Wrapped absolute difference in recorded car yaw between decisions when both yaws are present.',
            'detection_delay': 'Historical records have episode packet totals and outbound command receipt. Obstacle detection and its inbound delivery are not timestamped here.',
        },
        'formal': stats,
        'by_cohort_and_outcome': {name: group.summary() for name, group in sorted(groups.items())},
        'sources': sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--output', type=Path, default=Path('artifacts/vtemp5-pilot/calibration.json'))
    args = parser.parse_args()
    document = calibrate(args.repo)
    output = args.output if args.output.is_absolute() else args.repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Wrote {len(document["sources"])} source records to {output}')


if __name__ == '__main__':
    main()
