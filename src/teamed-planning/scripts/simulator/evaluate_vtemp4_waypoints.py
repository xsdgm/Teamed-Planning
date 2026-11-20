"""Paired evaluation of three policies that issue vehicle-relative, fixed world waypoints."""
import argparse
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree
import torch

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))
from world_model.frozen_navigation import CHECKPOINT_SHA, V9_RUN
from world_model.hierarchical_sources import atomic_json, file_sha256
from simulator.train_vtemp4 import FORMAT, reject_training_overlap
from simulator.train_vtemp4_formal import checkpoint_path, exclusive_run, source_digest, source_tree_digest
from simulator.vtemp4_curriculum import KM_DATA, LAKE_DATA, LAKE_GEOMETRY, geometry_hash, load_curriculum
from simulator.vtemp4_local_curriculum import local_case
from simulator.vtemp4_ring_contract import (
    PERIODIC_ACTION_CONTRACTS, PERIODIC_GOAL_RING_ACTION_CONTRACT, RING_ACTION_CONTRACT,
    RING_HEADING_FORMAT, RING_POLICY_FORMATS,
    RING_RADIUS_M, RING_WAYPOINT_FORMAT, ring_action_contract,
)
from simulator.vtemp4_stage_eval import bind_manifest, geometry_scores


ARMS = ('heading', 'waypoint', 'waypoint_history')
LEGACY_WAYPOINT_FORMAT = 'vtemp4_fixed_map_waypoint_v1'
LOCAL_SEEDS = (198701, 198702)
LAKE_CASES = ('lake-dev-00-detour_left', 'lake-dev-01-detour_right')
RETURN_RULE = dict(return_radius_m=5., min_excursion_m=10., min_travel_m=20., min_elapsed_s=30,
                   overlapping_events=False)


def excursion_returns(track, rule=RETURN_RULE):
    """Find nonoverlapping returns after leaving a previously visited neighborhood."""
    xy = np.asarray(track, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError('Expected a finite 1 Hz ground trajectory')
    travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    spatial = cKDTree(xy)
    first = 0
    events = []
    for end in range(len(xy)):
        latest = end - rule['min_elapsed_s']
        if latest < first or travel[end] - travel[first] < rule['min_travel_m']:
            continue
        for start in sorted(spatial.query_ball_point(xy[end], rule['return_radius_m'])):
            if start < first or start > latest or travel[end] - travel[start] < rule['min_travel_m']:
                continue
            excursion = np.linalg.norm(xy[start:end + 1] - xy[start], axis=1).max()
            if excursion >= rule['min_excursion_m']:
                events.append(dict(start_s=int(start), returned_s=end, max_excursion_m=float(excursion)))
                first = end
                break
    return events


def checkpoint_contract(state, manifest, arm):
    expected_ring = RING_POLICY_FORMATS[arm]
    supported = {expected_ring}
    if arm in ARMS:
        supported.add(FORMAT if arm == 'heading' else LEGACY_WAYPOINT_FORMAT)
    fmt = state['format']
    if fmt not in supported or manifest.get('policy_format') != fmt:
        raise ValueError(f'Wrong checkpoint or manifest policy format for {arm}')
    default_variant = 'heading' if fmt == FORMAT else None
    if state.get('policy_variant', default_variant) != arm or manifest.get('policy_variant', default_variant) != arm:
        raise ValueError(f'Checkpoint or manifest variant differs from {arm}')
    contract = ring_action_contract(fmt) if fmt == expected_ring else None
    if state.get('action_contract') != contract or manifest.get('action_contract') != contract:
        raise ValueError(f'Action contract differs from policy format for {arm}')
    if state['clock'] != manifest['clock'] or (contract and state['clock']['lookahead_m'] != RING_RADIUS_M):
        raise ValueError(f'Checkpoint and manifest must match the action geometry and clock for {arm}')
    if contract in PERIODIC_ACTION_CONTRACTS and state['clock'].get('review_interval_s') != 15:
        raise ValueError('Periodic history checkpoints must use the 15-second designation clock')
    return contract


def checkpoint_info(path, arm, cases, pilot):
    path = path.resolve(strict=True)
    run = path.parent.parent
    snapshot = run / 'sources'
    modern = (snapshot / 'simulator' / 'vtemp4_parallel.py').is_file()
    sources = snapshot if modern else snapshot / 'world_model'
    manifest_path = run / 'manifest.json'
    blob = path.read_bytes()
    state = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
    if state.get('v9_sha256') != CHECKPOINT_SHA:
        raise ValueError(f'Wrong frozen fleet for {arm}: {path}')
    if not sources.is_dir() or not manifest_path.is_file():
        raise ValueError(f'Expected an immutable source snapshot and training manifest beside {path}')
    manifest = json.loads(manifest_path.read_text())
    sha = file_sha256(manifest_path)
    src_sha = source_tree_digest(sources) if modern else source_digest(sources)
    if state.get('manifest_sha256') != sha or manifest.get('world_model_sha256') != src_sha:
        raise ValueError(f'Source snapshot or training manifest hash mismatch for {arm}')
    contract = checkpoint_contract(state, manifest, arm)
    updates = state.get('completed_updates')
    if type(updates) is not int or updates < 0:
        raise ValueError(f'Missing completed update count for {arm}')
    batch = int(manifest['episodes_per_update'])
    count = updates * batch
    if not pilot and (updates != 25 or batch != 16 or count != 400 or len(state.get('training_maps', ())) != count):
        raise ValueError(f'Final comparison requires 25 completed updates and 400 distinct maps for {arm}')
    reject_training_overlap(state, cases)
    identity = dict(checkpoint=str(path), sha256=file_sha256(path), sources=str(sources),
                    sources_sha256=src_sha, manifest_sha256=sha, completed_updates=updates,
                    committed_episodes=count, weight_status='trained' if updates else 'untrained_initialization',
                    clock=state['clock'], action_contract=contract, policy_format=state['format'])
    return identity, state, manifest


def validate_arms(checkpoints, cases, repo, pilot):
    all_info = [checkpoint_info(path, arm, cases, pilot) for arm, path in zip(ARMS, checkpoints)]
    info, states, manifests = zip(*all_info)
    contracts = {item['action_contract'] for item in info}
    if len(contracts) != 1 or (not pilot and contracts != {RING_ACTION_CONTRACT}):
        raise ValueError('Use three fixed-ring checkpoints together, or three legacy checkpoints for a technical pilot')
    frozen = repo / V9_RUN
    fast_data = dict(v9_sha256=file_sha256(frozen / 'best.pt'), v9_config_sha256=file_sha256(frozen / 'config.json'),
                     v9_sources_sha256=source_digest(frozen / 'runtime_sources'))
    if fast_data['v9_sha256'] != CHECKPOINT_SHA or any(
        manifest.get(key) != val for manifest in manifests for key, val in fast_data.items()
    ):
        raise ValueError('Archived fleet inputs differ from the training provenance')
    if any(row['clock'] != info[0]['clock'] for row in info):
        raise ValueError('All arms must use the same review interval and lookahead')
    if not pilot:
        training_settings = ('seed', 'episodes_per_update', 'learning_rate', 'gamma', 'clock', 'total_episodes')
        if any(m.get(k) != manifests[0].get(k) for m in manifests[1:] for k in training_settings):
            raise ValueError('Arms used different training settings')
        if any(manifest.get('plan') != manifests[0].get('plan') for manifest in manifests[1:]):
            raise ValueError('Arms used different online training schedules')
        plan = manifests[0]['plan']
        if len(plan) != 400 or len({row['scene_seed'] for row in plan}) != 400:
            raise ValueError('Expected a 400-mission online training schedule')
        if any(set(state['training_maps']) != set(states[0]['training_maps']) for state in states[1:]):
            raise ValueError('Arms trained on different map geometry')
    initial = []
    for arm, item, state, manifest in zip(ARMS[1:], info[1:], states[1:], manifests[1:]):
        run = Path(item['checkpoint']).parent.parent
        path = checkpoint_path(run, 0)
        first = state if item['completed_updates'] == 0 else torch.load(path, map_location='cpu', weights_only=True)
        if first['completed_updates'] != 0 or first['manifest_sha256'] != item['manifest_sha256']:
            raise ValueError('Initial waypoint checkpoint belongs to another training run')
        checkpoint_contract(first, manifest, arm)
        item['initial_checkpoint_sha256'] = file_sha256(path)
        initial.append(first['model'])
    b, c = initial
    if b.keys() != c.keys() or not all(torch.equal(value, c[key]) for key, value in b.items()):
        raise ValueError('Initial B and C must share exactly the same parameter tensors')
    return dict(zip(ARMS, info)), fast_data


def held_out_cases(args):
    circle_seeds = args.scene_seeds
    names = args.lake_cases
    if len(set(circle_seeds)) != len(circle_seeds) or len(set(names)) != len(names):
        raise ValueError('Duplicate held-out case')
    cases = [local_case(args.repo, seed, split='development', geometry_dir=args.output / 'geometry')
             for seed in circle_seeds]
    if names:
        splits, _ = load_curriculum(args.repo / KM_DATA, args.repo / LAKE_DATA, field_geometry=LAKE_GEOMETRY)
        lakes = {case['case_id']: case for case in splits['development'] if case['terrain'] == 'wildscenes_raster'}
        if set(names) - lakes.keys():
            raise ValueError('Lake cases must belong to the registered development split')
        cases += [lakes[name] for name in names]
    if not cases:
        raise ValueError('Select at least one held-out mission')
    return cases


# Each subprocess imports its checkpoint's own snapshot in isolation. In particular, A
# uses the archived heading episode and collector from its training run.
WORKER = """
import json, sys
from pathlib import Path
request=json.load(sys.stdin)
sources = Path(request['sources'])
if (sources / 'simulator' / 'vtemp4_parallel.py').is_file():
    sys.path.insert(0, str(sources))
    from simulator.vtemp4_parallel import collect_episode
else:
    sys.path.insert(0, str(sources.parent))
    from world_model.vtemp4_parallel import collect_episode
collect_episode(Path(request['repo']), Path(request['checkpoint']), request['sha256'],
                request['case'], request['fast_seed'], Path(request['folder']),
                mode='evaluate', collector_seconds=request['limit'])
"""


def audit_ring_markers(record, track, goal_xy=None):
    contract = record.get('action_contract')
    use_final_goal = contract == PERIODIC_GOAL_RING_ACTION_CONTRACT
    if use_final_goal:
        goal = np.asarray(record.get('final_goal_xy'), dtype=float)
        if (goal.shape != (2,) or not np.isfinite(goal).all()
                or (goal_xy is not None and not np.array_equal(goal, goal_xy))):
            raise ValueError('Expected the final goal from the mission geometry')

    markers = record['shared_markers']
    times = [row['issued_at_s'] for row in markers]
    reviews = record.get('heading_reviews', record.get('waypoint_reviews'))
    changed_at = [row['time_s'] for row in reviews if row['changed']]
    if times != changed_at or times != sorted(set(times)):
        raise ValueError('A fixed-world marker may change only on an explicit policy update')
    for marker_id, (t, marker) in enumerate(zip(times, markers)):
        if type(t) is not int or not 0 <= t < len(track) or marker['id'] != marker_id:
            raise ValueError('Invalid fixed-world marker ID or issue time')
        origin = np.asarray(marker['origin_xy'], dtype=float)
        if origin.shape != (2,) or not np.allclose(origin, track[t], atol=1e-6, rtol=0.):
            raise ValueError('Marker origin must match the vehicle position on issue')
        radius = RING_RADIUS_M
        if use_final_goal:
            remaining = float(np.linalg.norm(goal - track[t]))
            close = remaining <= RING_RADIUS_M
            if marker.get('designation_source') != ('final_goal' if close else 'policy_angle'):
                raise ValueError('Marker designation source disagrees with the distance to the final goal on issue')
            if close:
                if not np.array_equal(marker['target'], goal):
                    raise ValueError('A goal within 200 m must be designated at its exact map coordinates')
                radius = remaining
        bearing = marker['vehicle_yaw_rad'] + marker['relative_angle_rad']
        expected = track[t] + radius * np.array([np.cos(bearing), np.sin(bearing)])
        if (not math.isclose(marker['radius_m'], radius, rel_tol=0., abs_tol=1e-6 if use_final_goal else 0.)
                or np.shape(marker['target']) != (2,) or not np.allclose(marker['target'], expected, atol=1e-6, rtol=0.)):
            raise ValueError('Marker radius or recorded body-relative bearing disagrees with its target on issue')

    for events, field in ((reviews, 'target'), (record['ground_replan_events'], 'shared_marker')):
        for row in events:
            active = bisect_right(times, row['time_s']) - 1
            expected_id = active if active >= 0 else None
            expected_target = markers[active]['target'] if active >= 0 else None
            if row['waypoint_id'] != expected_id or row[field] != expected_target:
                raise ValueError('Ground replan or policy review moved the last confirmed world marker')
    if contract in PERIODIC_ACTION_CONTRACTS:
        audit_periodic_broadcasts(record, reviews, markers, len(track) - 1)
    return len(markers)


def audit_periodic_broadcasts(record, reviews, markers, elapsed):
    # Native safety termination occurs before applying the prepared physics action.
    # A decision at that exact tick exists even though it has zero broadcasts.
    stop = elapsed + int(record['end_reason'] == 'safety_failure')
    expected_times = list(range(0, stop, 15))
    if [r['time_s'] for r in reviews] != expected_times or len(markers) != len(expected_times):
        raise ValueError('Expected a fresh confirmed waypoint at every active 15-second decision')
    for i, (review, marker) in enumerate(zip(reviews, markers)):
        if (record.get('action_contract') == PERIODIC_GOAL_RING_ACTION_CONTRACT
                and review.get('designation_source') != marker['designation_source']):
            raise ValueError('Periodic review and marker disagree on the designation source')
        if (not (review['turn'] and review['accepted'] and review['changed'])
                or not math.isclose(marker['relative_angle_rad'], review['angle_rad'], rel_tol=0., abs_tol=1e-7)
                or not math.isclose(marker['vehicle_yaw_rad'], review['vehicle_yaw_rad'], rel_tol=0., abs_tol=1e-7)):
            raise ValueError('Periodic marker does not match the continuous angle confirmed at its review')
        t = expected_times[i]
        broadcast_ticks = min(15, elapsed - t)
        if (marker['first_physics_broadcast_s'] != (t if broadcast_ticks else None)
                or marker['physics_broadcasts'] != broadcast_ticks):
            raise ValueError('Periodic marker must be broadcast on every successful physics tick until replaced')


def evaluate_arm(repo, info, case, seed, folder, limit):
    folder.mkdir(parents=True, exist_ok=True)
    request = dict(repo=str(repo), sources=info['sources'], checkpoint=info['checkpoint'], sha256=info['sha256'],
                   case=case, fast_seed=seed, folder=str(folder), limit=limit)
    env = dict(os.environ)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    with (folder / 'worker.log').open('a') as log:
        subprocess.run([sys.executable, '-I', '-B', '-c', WORKER], env=env, input=json.dumps(request),
                       text=True, stdout=log, stderr=log, check=True)
    record = json.loads((folder / 'episode.json').read_text())
    track = np.asarray(record['trajectory_xy'], dtype=float)
    start, goal = np.asarray(case['start']), np.asarray(case['goal'])
    if (record['fast_seed'] != seed or record['geometry_sha256'] != geometry_hash(case)
            or record['mode'] != 'evaluate' or record['collector_seconds'] != limit
            or record['budget_s'] != math.ceil(3 * case['direct_distance_m'] + 120)
            or record['checkpoint_sha256'] != info['sha256'] or not np.allclose(track[0], start)
            or len(track) != record['sim_time_s'] + 1):
        raise ValueError(f'Paired trajectory or checkpoint identity mismatch at {folder}')
    initial = float(np.linalg.norm(start - goal))
    final = float(np.linalg.norm(track[-1] - goal))
    if not math.isclose(record['remaining_m'], final, abs_tol=1e-5):
        raise ValueError('Terminal trajectory and simulator distance disagree')
    events = excursion_returns(track)
    row = dict(case_id=case['case_id'], fast_seed=seed, stratum=case['navigation_task'], record=str(folder/'episode.json'),
               end_reason=record['end_reason'], arrived=record['goal_reached'], truncated=record['end_reason']=='collector_limit',
               sim_s=record['sim_time_s'], budget_s=record['budget_s'], initial_distance_m=initial,
               terminal_distance_m=final, net_progress_m=initial-final, travelled_m=record['travelled_m'],
               backtracking_count=len(events), backtracking_events=events)
    if record.get('action_contract') != info['action_contract']:
        raise ValueError('Episode action contract differs from the checkpoint')
    if info['action_contract'] == RING_ACTION_CONTRACT or info['action_contract'] in PERIODIC_ACTION_CONTRACTS:
        row['verified_waypoint_issues'] = audit_ring_markers(record, track, goal)
    row.update(geometry_scores(case, track))
    return row


def aggregate(rows):
    truncated = sum(row['truncated'] for row in rows)
    arrivals = sum(row['arrived'] for row in rows)
    averages = {f'mean_{key}': sum(row[key] for row in rows) / len(rows) for key in (
        'initial_distance_m', 'terminal_distance_m', 'net_progress_m', 'travelled_m', 'backtracking_count')}
    return dict(cases=len(rows), truncated=truncated, observed_arrivals=arrivals,
                arrival_rate=None if truncated else arrivals / len(rows), **averages)


def paired_results(rows):
    pairs = []
    for index, reference in enumerate(rows['heading']):
        arms = {arm: rows[arm][index] for arm in ARMS}
        comparisons = {}
        for a, b in (('waypoint','heading'), ('waypoint_history','heading'), ('waypoint_history','waypoint')):
            x, y = arms[a], arms[b]
            comparisons[f'{a}_minus_{b}'] = dict(
                net_progress_m=x['net_progress_m']-y['net_progress_m'],
                backtracking_count=x['backtracking_count']-y['backtracking_count'],
                arrival=None if x['truncated'] or y['truncated'] else int(x['arrived'])-int(y['arrived']))
        pairs.append(dict(case_id=reference['case_id'], fast_seed=reference['fast_seed'], stratum=reference['stratum'],
                          arms=arms, differences=comparisons))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=SOURCE.parents[3])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--heading-checkpoint', type=Path, required=True)
    parser.add_argument('--waypoint-checkpoint', type=Path, required=True)
    parser.add_argument('--history-checkpoint', type=Path, required=True)
    parser.add_argument('--scene-seeds', nargs='*', type=int, default=LOCAL_SEEDS)
    parser.add_argument('--lake-cases', nargs='*', default=LAKE_CASES)
    parser.add_argument('--fast-seeds', nargs='+', type=int)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--collector-seconds', type=int)
    parser.add_argument('--pilot', '--allow-untrained', action='store_true',
                        help='Technical smoke test permitting untrained or unequal checkpoints')
    args = parser.parse_args()
    if args.workers < 1 or (args.collector_seconds is not None and args.collector_seconds < 1):
        parser.error('Workers and optional collection horizon must be positive')
    args.repo, args.output = args.repo.resolve(), args.output.resolve()
    with exclusive_run(args.output):
        run(args)


def run(args):
    cases = held_out_cases(args)
    seeds = args.fast_seeds if args.fast_seeds is not None else list(range(20481001, 20481001+len(cases)))
    if len(seeds) != len(cases) or any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError('Specify one uint32 fleet seed per selected map')
    checkpoints = (args.heading_checkpoint, args.waypoint_checkpoint, args.history_checkpoint)
    policies, fleet = validate_arms(checkpoints, cases, args.repo, args.pilot)
    plan = [dict(case_id=c['case_id'], stratum=c['navigation_task'], fast_seed=s, geometry_sha256=geometry_hash(c),
                 budget_s=math.ceil(3*c['direct_distance_m']+120), direct_distance_m=c['direct_distance_m'])
            for c,s in zip(cases,seeds)]
    protocol = dict(format='vtemp4_paired_waypoints_v2', comparison='technical_pilot' if args.pilot else 'matched_trained',
                    action_contract=policies['heading']['action_contract'],
                    split='development', collector_seconds=args.collector_seconds, backtracking_definition=RETURN_RULE,
                    evaluator_sha256=file_sha256(Path(__file__)), policies=policies, fleet=fleet, cases=plan)
    bind_manifest(args.output / 'protocol.json', protocol)
    rows = {arm: [None] * len(cases) for arm in ARMS}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(evaluate_arm, args.repo, policies[arm], case, seed,
                            args.output/arm/case['case_id'], args.collector_seconds): (arm,i)
                for i,(case,seed) in enumerate(zip(cases,seeds)) for arm in ARMS}
        for count, job in enumerate(as_completed(jobs),1):
            arm, i = jobs[job]
            rows[arm][i] = job.result()
            atomic_json(args.output/'progress.json', dict(completed=count, total=len(jobs), arm=arm, case_id=cases[i]['case_id']))
            print(f'{count}/{len(jobs)}: {arm} {cases[i]["case_id"]}', flush=True)
    summaries = {}
    for arm, results in rows.items():
        summaries[arm] = dict(overall=aggregate(results), by_stratum={
            task: aggregate([row for row in results if row['stratum']==task]) for task in sorted({r['stratum'] for r in results})})
    atomic_json(args.output/'evaluation.json', dict(**protocol, summaries=summaries, paired_cases=paired_results(rows)))


if __name__ == '__main__':
    main()
