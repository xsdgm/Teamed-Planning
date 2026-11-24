"""Paired local guidance intervention on an off-line, certified development route."""
import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vtemp2_3.frozen_navigation import CHECKPOINT_SHA
from vtemp2_3.hierarchical_sources import atomic_json
from vtemp4.vtemp4_runtime import HeadingFleetEpisode


def sample_route(vertices):
    """Resample the certificate at 1 m arc length for stable route projection."""
    vertices = np.asarray(vertices, float)
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    if len(lengths) == 0 or (lengths <= 0).any():
        raise ValueError("Expected a nondegenerate certificate polyline")
    arc = np.r_[0., np.cumsum(lengths)]
    samples = np.r_[np.arange(0., arc[-1], 1.), arc[-1]]
    return np.column_stack([np.interp(samples, arc, vertices[:, i]) for i in range(2)])


def choose_start(route, bridge_xy):
    """Choose the greatest 80 m directional disagreement on the approach, prior to any rollout."""
    crossing = int(np.argmin(np.linalg.norm(route - bridge_xy, axis=1)))
    indices = np.arange(0, crossing - 80)
    if not len(indices):
        raise ValueError("Expected at least 80 m of approach to the bridge")
    via = route[indices + 80] - route[indices]
    direct = route[-1] - route[indices]
    cosine = np.sum(via * direct, axis=1) / (np.linalg.norm(via, axis=1) * np.linalg.norm(direct, axis=1))
    idx = indices[np.argmin(cosine)]
    return int(idx), math.degrees(math.acos(np.clip(cosine.min(), -1., 1.)))


class Intervention:
    def __init__(self, route, use_certificate):
        self.route = route
        self.use_certificate = use_certificate

    def act(self, obs, *, deterministic, generator):
        if self.now < 15 or not self.use_certificate:
            return False, 0., 0., 0.
        i = int(np.argmin(np.linalg.norm(self.route - self.position, axis=1)))
        aim = self.route[min(i + 80, len(self.route) - 1)] - self.position
        delta = math.remainder(math.atan2(aim[1], aim[0]) - self.heading, 2 * math.pi)
        return (True, delta, 0., 0.) if abs(delta) >= math.radians(3.) else (False, 0., 0., 0.)


class DiagnosticEpisode(HeadingFleetEpisode):
    def _choose_heading(self, now, position, distance):
        # Used only by the explicitly labelled certificate controller. Both trials
        # otherwise share native ground sensing, packet delivery, planning and UAVs.
        self.slow.position = position.copy()
        self.slow.heading = self.heading
        self.slow.now = now
        super()._choose_heading(now, position, distance)


def run(case, route, seed, seconds, oracle):
    model = Intervention(route, oracle)
    with open(os.devnull, 'w') as sink, redirect_stdout(sink), torch.inference_mode():
        ep = DiagnosticEpisode(REPO, case, model, fast_seed=seed,
                               budget_s=math.ceil(3 * case['direct_distance_m'] + 120),
                               collector_limit_s=seconds)
        try:
            while not ep.finished:
                ep.step()
            record = ep.fleet_summary()
            track = np.array(ep.trajectory)
            if len(track) < 16:
                raise RuntimeError("Episode ended during the common initialization")
            q = route[min(80, len(route)-1)]
            d = np.linalg.norm(track - q, axis=1)
            moved = np.flatnonzero(d <= 10)
            baseline_vec = np.subtract(case['goal'], case['start']) / case['direct_distance_m']
            normal = np.array([-baseline_vec[1],baseline_vec[0]])
            lateral = np.abs((track - track[0]) @ normal)
            record.update(trajectory_xy=track.tolist(),
                          max_lateral_to_initial_bearing_m=float(lateral.max()),
                          route_exit_xy=q.tolist(), min_distance_to_exit_m=float(d.min()),
                          exit_first_within_10m_s=int(moved[0]) if len(moved) else None,
                          final_distance_to_exit_m=float(d[-1]))
        finally:
            ep.close()
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path, help='Case directory containing case.json and audit.json')
    p.add_argument('--seconds', type=int, default=300)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.seconds <= 15:
        p.error("The interval must include at least one post-initialization review")
    case = json.loads((args.folder / 'case.json').read_text())
    audit = json.loads((args.folder / 'audit.json').read_text())
    route = sample_route(audit['reference_route_xy'])
    idx, disagreement = choose_start(route, audit['bridge_center_xy'])
    start, goal = route[idx], np.asarray(case['goal'])
    if not np.allclose(route[-1], goal):
        raise ValueError("Certificate and scenario goals differ")
    case.update(start=start.tolist(),direct_distance_m=float(np.linalg.norm(start - goal)),
                reference_distance_m=float(np.linalg.norm(np.diff(route[idx:],axis=0),axis=1).sum()),
                map_path=str((args.folder / case['map_path']).resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(case_id=case['case_id'],fast_seed=case['seed'],map_sha256=case['map_sha256'],
                  v9_sha256=CHECKPOINT_SHA, window_seconds=args.seconds,
                  protocol='paired_local_guidance_intervention_v1', review_interval_s=15,
                  oracle_uses_offline_reference=True, mission_start_reset_xy=start.tolist(),
                  local_route_tangent_vs_goal_degrees=disagreement, reference_arc_at_reset_m=idx)
    for key, oracle in [('hold_world_heading',False),('offline_route_heading',True)]:
        result[key] = run(case,route[idx:],case['seed'],args.seconds,oracle)
    a,b = (np.asarray(result[key]['trajectory_xy']) for key in ['hold_world_heading','offline_route_heading'])
    if not np.array_equal(a[:16],b[:16]):
        raise RuntimeError("Paired trajectories differ prior to the first guidance intervention")
    result['common_initial_15s_identical'] = True
    atomic_json(args.output,result)
    print(args.output)


if __name__ == '__main__':
    main()
