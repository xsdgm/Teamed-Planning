"""Run a VTemp4 checkpoint and its frozen V9 aircraft on a procedural field raster."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world_model.frozen_navigation import CHECKPOINT_SHA, runtime
from world_model.hierarchical_sources import atomic_json
from simulator.train_vtemp4 import FORMAT
from ground.vtemp4_policy import HeadingPolicy
from simulator.vtemp4_runtime import HeadingFleetEpisode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, default=REPO / 'artifacts/vtemp4-field-valley-v1/case.json')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='Destination JSON for the trajectory and metrics')
    parser.add_argument('--seconds', type=int, help='Optional preview duration; omit for the full mission budget')
    args = parser.parse_args()
    if args.seconds is not None and args.seconds < 1:
        parser.error('--seconds must be positive')

    case = json.loads(args.case.read_text(encoding='utf-8'))
    case['map_path'] = str((args.case.parent / case['map_path']).resolve())
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
    if saved.get('format') != FORMAT or saved.get('v9_sha256') != CHECKPOINT_SHA:
        parser.error('Expected a VTemp4 heading checkpoint for the frozen V9 fleet')
    runtime(REPO)
    policy = HeadingPolicy()
    policy.load_state_dict(saved['model'])
    policy.eval().requires_grad_(False)
    budget = math.ceil(3 * case['direct_distance_m'] + 120)
    with torch.inference_mode():
        episode = HeadingFleetEpisode(REPO, case, policy, fast_seed=case['seed'], budget_s=budget,
                                     collector_limit_s=args.seconds, **saved['clock'])
        try:
            while not episode.finished:
                episode.step()
            summary = episode.fleet_summary()
            summary.update(map_sha256=case['map_sha256'], checkpoint_sha256=checkpoint_sha,
                           checkpoint_update=saved['completed_updates'], v9_sha256=CHECKPOINT_SHA,
                           ground_trajectory_xy=episode.trajectory, uav_samples_xy=episode.air_samples,
                           raster_ground_safety_attempts=episode.env.env.ground_safety_violations,
                           raster_collision_attempts=episode.env.env.raster_collision_attempts)
        finally:
            episode.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, summary)
    print(f"{case['case_id']}  {summary['sim_time_s']} s  {summary['end_reason']}  "
          f"{summary['remaining_m']:.1f} m to goal  {args.output}")


if __name__ == '__main__':
    main()
