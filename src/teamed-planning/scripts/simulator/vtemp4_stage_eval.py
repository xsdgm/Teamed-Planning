"""Paired, same-seed VTemp4 development evaluation for immutable training milestones."""
import argparse
from concurrent.futures import as_completed
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import label
import torch

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE.parent))
from world_model.frozen_navigation import CHECKPOINT_SHA, V9_RUN
from world_model.hierarchical_sources import atomic_json, file_sha256, package_python_files
from simulator.train_vtemp4 import FORMAT, metrics, reject_training_overlap
from simulator.train_vtemp4_formal import source_tree_digest
from simulator.vtemp4_curriculum import FIELD_DATA, KM_DATA, geometry_hash, load_curriculum, stratum_counts
from simulator.vtemp4_lake_metrics import lake_metrics


def bridge_metrics(track, cover, origin, resolution, start, goal, bridge_center):
    """Score the finished native 1 Hz ground trajectory on the original river raster."""
    track = np.asarray(track, dtype=float)
    if track.ndim != 2 or track.shape[1] != 2 or not len(track) or not np.allclose(track[0], start):
        raise ValueError('Expected a ground trajectory starting at the official start')
    banks, _ = label(~np.isin(cover, (4, 6)), structure=np.ones((3, 3), dtype=bool))

    def at(grid, points):
        ix, iy = np.floor((np.asarray(points) - origin) / resolution).astype(int).T
        if np.any((ix < 0) | (ix >= grid.shape[1]) | (iy < 0) | (iy >= grid.shape[0])):
            raise ValueError('Ground trajectory leaves the evaluation raster')
        return grid[iy, ix]

    near_bank, far_bank = at(banks, [start, goal])
    if not near_bank or not far_bank or near_bank == far_bank:
        raise ValueError('Expected two separate river banks with the deck removed')
    # Sample each native movement segment at <=1/4 cell just as in the field diagnostic.
    deltas = np.diff(track, axis=0)
    counts = np.maximum(1, np.ceil(np.linalg.norm(deltas, axis=1) / (.25 * resolution)).astype(int))
    steps = np.repeat(np.arange(len(counts)), counts)
    first = np.cumsum(counts) - counts
    fractions = (np.arange(len(steps)) - first[steps] + 1) / counts[steps]
    samples = np.vstack((track[0], track[steps] + fractions[:, None] * deltas[steps]))
    times = np.r_[0., steps + fractions]
    surfaces = at(cover, samples)
    land = at(banks, samples)
    on_deck = np.flatnonzero(surfaces == 6)
    on_far_bank = np.flatnonzero(land == far_bank)
    # A crossing must travel from the start bank over a continuous deck to the far bank.
    # A previous visit to the bridge followed by entering the water does not count.
    crossing = None
    land_samples = np.flatnonzero(land != 0)
    for back, ahead in zip(land_samples[:-1], land_samples[1:]):
        if (land[back] == near_bank and land[ahead] == far_bank and ahead > back + 1
                and np.all(surfaces[back+1:ahead] == 6)):
            crossing = float(times[ahead])
            break
    return dict(bridge_deck_first_s=float(times[on_deck[0]]) if len(on_deck) else None,
                opposite_bank_first_s=float(times[on_far_bank[0]]) if len(on_far_bank) else None,
                river_crossed_via_bridge=crossing is not None, bridge_crossing_s=crossing,
                water_entered=bool(np.any(surfaces == 4)),
                min_distance_to_bridge_center_m=float(np.linalg.norm(samples - bridge_center, axis=1).min()))


def geometry_scores(case, track):
    if case['terrain'] != 'wildscenes_raster':
        return {}
    path = Path(case['map_path'])
    audit = json.loads(path.with_name('audit.json').read_text())
    with np.load(path) as native, np.load(path.with_name('landscape.npz')) as landscape:
        if case['kind'] == 'procedural_wooded_lake':
            return lake_metrics(track, landscape['landcover'], native['origin_xy'], float(native['resolution_m']),
                                case['start'], case['goal'], audit, direct=case['navigation_task'] == 'direct')
        return bridge_metrics(track, landscape['landcover'], native['origin_xy'], float(native['resolution_m']),
                              case['start'], case['goal'], audit['bridge_center_xy'])


def group_metrics(rows, cases):
    groups = {}
    for task, count in stratum_counts(cases).items():
        selected = [row for row in rows if row['navigation_task'] == task]
        group = metrics(selected, count)
        if selected and 'reached_lake_far_end' in selected[0]:
            group.update(reached_lake_far_end=sum(r['reached_lake_far_end'] for r in selected),
                         tracks_entering_water=sum(r['water_entered'] for r in selected),
                         mean_max_lateral_deviation_m=sum(r['max_lateral_deviation_m'] for r in selected)/len(selected))
            if task == 'direct':
                group['turns_on_direct_controls'] = sum(r['turn_count'] for r in selected)
            else:
                bypasses = sum(r['lake_bypassed'] for r in selected)
                group.update(lake_bypasses=bypasses, via_left=sum(r['lake_bypass_side'] == 'left' for r in selected),
                             via_right=sum(r['lake_bypass_side'] == 'right' for r in selected),
                             lake_bypass_rate=bypasses/count if group['completed_missions'] == count else None)
        elif selected and 'river_crossed_via_bridge' in selected[0]:
            crossings = sum(row['river_crossed_via_bridge'] for row in selected)
            group.update(bridge_crossings=crossings,
                         bridge_crossing_rate=crossings / count if group['completed_missions'] == count else None,
                         reached_goal_bank=sum(row['opposite_bank_first_s'] is not None for row in selected),
                         mean_closest_bridge_center_m=(sum(r['min_distance_to_bridge_center_m'] for r in selected)
                                                       / len(selected) if selected else None))
        groups[task] = group
    return groups


def stage_report(cases, rows, update, checkpoint_sha):
    checkpoint_rows = [rows['evaluate', c['case_id']] for c in cases]
    hold_rows = [rows['always_hold', c['case_id']] for c in cases]
    pairs = []
    for policy, hold in zip(checkpoint_rows, hold_rows):
        if any(policy[k] != hold[k] for k in ('case_id', 'fast_seed', 'geometry_sha256', 'direct_distance_m')):
            raise ValueError('Paired development rollout identities differ')
        pair = dict(case_id=policy['case_id'], navigation_task=policy['navigation_task'],
                    fast_seed=policy['fast_seed'], geometry_sha256=policy['geometry_sha256'],
                    checkpoint_minus_hold_remaining_m=policy['remaining_m'] - hold['remaining_m'],
                    checkpoint_minus_hold_reward=policy['reward'] - hold['reward'],
                    checkpoint_minus_hold_arrival=int(policy['goal_reached']) - int(hold['goal_reached']))
        if 'river_crossed_via_bridge' in policy:
            pair.update(checkpoint_minus_hold_crossing=int(policy['river_crossed_via_bridge'])
                        - int(hold['river_crossed_via_bridge']),
                        checkpoint_minus_hold_closest_bridge_center_m=policy['min_distance_to_bridge_center_m']
                        - hold['min_distance_to_bridge_center_m'])
        if 'reached_lake_far_end' in policy:
            pair.update(checkpoint_minus_hold_far_end=int(policy['reached_lake_far_end'])
                        - int(hold['reached_lake_far_end']),
                        checkpoint_minus_hold_max_deviation_m=policy['max_lateral_deviation_m']
                        - hold['max_lateral_deviation_m'])
            if 'lake_bypassed' in policy:
                pair.update(checkpoint_minus_hold_bypass=int(policy['lake_bypassed']) - int(hold['lake_bypassed']),
                            checkpoint_bypass_side=policy['lake_bypass_side'], hold_bypass_side=hold['lake_bypass_side'])
            else:
                pair['extra_turns_on_direct_control'] = policy['turn_count'] - hold['turn_count']
        pairs.append(pair)
    return dict(update=update, checkpoint_sha256=checkpoint_sha, split='development',
                policy='deterministic_checkpoint', cases=checkpoint_rows, paired_cases=pairs,
                by_stratum=group_metrics(checkpoint_rows, cases), **metrics(checkpoint_rows, len(cases)),
                always_hold=dict(cases=hold_rows, by_stratum=group_metrics(hold_rows, cases),
                                 **metrics(hold_rows, len(cases))))


def bind_manifest(path, manifest):
    if path.is_file():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f'Development protocol or checkpoint changed at {path}')
    else:
        atomic_json(path, manifest)


def source_digest(root):
    files = {str(p): file_sha256(root/p) for p in package_python_files(root)}
    if not files:
        raise ValueError(f'Missing frozen sources at {root}')
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def evaluation_inputs(repo, checkpoint, state, field_data):
    from simulator.vtemp4_curriculum import LAKE_DATA

    training_sha = state.get('manifest_sha256')
    if training_sha is None:
        # Historic pilot weights can be evaluated on the current lake suite; use the
        # explicit historical suite when comparing bridge results.
        directory = field_data or repo/LAKE_DATA
        geometry = 'river_bridge' if directory.resolve() == (repo/FIELD_DATA).resolve() else 'bilateral_lake'
        return load_curriculum(repo/KM_DATA, directory, field_geometry=geometry)
    path = checkpoint.parent.parent/'manifest.json'
    raw_manifest = path.read_bytes()
    manifest = json.loads(raw_manifest)
    if (hashlib.sha256(raw_manifest).hexdigest() != training_sha
            or manifest.get('curriculum') != state.get('curriculum') or manifest.get('clock') != state.get('clock')):
        raise ValueError('Formal checkpoint and training manifest differ')
    geometry = manifest.get('field_geometry', 'river_bridge')
    if state.get('curriculum') == 'lakes' and geometry != 'bilateral_lake':
        raise ValueError('The formal lake run requires the bilateral-lake curriculum')
    directory = field_data or Path(manifest.get('field_data', repo/(LAKE_DATA if geometry == 'bilateral_lake' else FIELD_DATA)))
    km = Path(manifest.get('km_data', repo/KM_DATA))
    splits, hashes = load_curriculum(km, directory, field_geometry=geometry)
    if any(manifest.get(key) != value for key, value in hashes.items()):
        raise ValueError('Development input fingerprints differ from the formal training manifest')
    formal_runtime = {'v9_sha256': CHECKPOINT_SHA, 'v9_config_sha256': file_sha256(repo/V9_RUN/'config.json'),
                      'v9_sources_sha256': source_digest(repo/V9_RUN/'runtime_sources'),
                      'world_model_sha256': source_tree_digest(PACKAGE.parent)}
    if any(manifest.get(k) != v for k, v in formal_runtime.items()):
        raise ValueError('Development runtime differs from the formal training snapshot')
    return splits, hashes


def stage_eval(repo, checkpoint, update, out, workers, limit=None, field_data=None):
    from simulator.vtemp4_parallel import collect_episode, episode_pool

    payload = checkpoint.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if (state.get('format') != FORMAT or state.get('completed_updates') != update
            or state.get('v9_sha256') != CHECKPOINT_SHA or not isinstance(state.get('training_maps'), list)):
        raise ValueError('Expected an immutable VTemp4 checkpoint from the specified completed update')
    splits, datasets = evaluation_inputs(repo, checkpoint, state, field_data)
    cases = splits['development']
    reject_training_overlap(state, cases)
    identities = []
    for case in cases:
        identity = dict(case_id=case['case_id'], fast_seed=case['seed'], map_sha256=geometry_hash(case))
        if case['terrain'] == 'wildscenes_raster':
            folder = Path(case['map_path']).parent
            identity.update(landscape_sha256=file_sha256(folder/'landscape.npz'),
                            audit_sha256=file_sha256(folder/'audit.json'))
        identities.append(identity)
    protocol = dict(format='vtemp4_paired_development_v2', cases=identities, clock=state['clock'],
                    training_manifest_sha256=state.get('manifest_sha256'),
                    collector_seconds=limit, v9_sha256=CHECKPOINT_SHA,
                    v9_config_sha256=file_sha256(repo/V9_RUN/'config.json'),
                    v9_sources_sha256=source_digest(repo/V9_RUN/'runtime_sources'),
                    world_model_sources_sha256=source_tree_digest(PACKAGE.parent), **datasets)
    bind_manifest(out/'protocol.json', protocol)
    # One immutable checkpoint supplies the hold clock and provenance for every stage.
    # It is always independent of the current stage's action network.
    anchor_file = out/'hold-anchor.json'
    if not anchor_file.exists():
        atomic_json(anchor_file, dict(path=os.path.relpath(checkpoint, out), sha256=digest))
    anchor = json.loads(anchor_file.read_text())
    hold_path = (out/anchor['path']).resolve()
    if file_sha256(hold_path) != anchor['sha256']:
        raise ValueError('The original hold-baseline checkpoint has changed')
    dest = out/f'update-{update:04d}'
    dest.mkdir(exist_ok=True)
    bind_manifest(dest/'manifest.json', dict(update=update, sha256=digest, protocol_sha256=file_sha256(out/'protocol.json')))
    rows = {}
    jobs = {}
    with episode_pool(workers) as pool:
        for case in cases:
            for mode, source, sha, folder in [('evaluate', checkpoint, digest, dest),
                                                ('always_hold', hold_path, anchor['sha256'], out/'always-hold')]:
                work = folder/'episodes'/case['case_id']
                future = pool.submit(collect_episode, repo, source, sha, case, case['seed'], work,
                                     mode=mode, collector_seconds=limit)
                jobs[future] = mode, case, work
        for future in as_completed(jobs):
            mode, case, work = jobs[future]
            row = future.result()
            record = json.loads((work/'episode.json').read_text())
            row.update(geometry_scores(case, record['trajectory_xy']), collisions=record['collisions'])
            rows[mode, case['case_id']] = row
            atomic_json(dest/'progress.json', dict(completed=len(rows), total=len(jobs), last_case=case['case_id'],
                                                   last_policy=mode))
            print(f'Development {update}: {len(rows)}/{len(jobs)} {case["case_id"]} {mode}', flush=True)
    atomic_json(dest/'evaluation.json', stage_report(cases, rows, update, digest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--update', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Shared development output directory for all stages')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--field-data', type=Path, help='Optional relocated field suite with identical formal fingerprints')
    parser.add_argument('--collector-seconds', type=int, help='Optional truncation for isolated integration checks')
    args = parser.parse_args()
    if min(args.update, args.workers) < 1 or args.collector_seconds is not None and args.collector_seconds < 1:
        parser.error('Update, workers and any collector limit must be positive')
    out = args.output.resolve()
    out.mkdir(exist_ok=True, parents=True)
    # A restarted dispatcher can wait for a live stage safely. Training uses a separate lock.
    with (out/'.stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stage_eval(args.repo.resolve(), args.checkpoint.resolve(), args.update, out, args.workers,
                   args.collector_seconds, args.field_data.resolve() if args.field_data else None)


if __name__ == '__main__':
    main()
