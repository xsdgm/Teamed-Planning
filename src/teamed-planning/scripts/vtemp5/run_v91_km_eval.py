"""Paired, frozen-checkpoint navigation on the original kilometre-scale development maps."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from multiprocessing import get_context
from pathlib import Path
import runpy
import shutil
import sys
import time

import numpy as np

from run_v91_lake_test import sha, write_json


SCRIPTS = Path(__file__).resolve().parent
ARTIFACTS = SCRIPTS.parents[3] / 'artifacts'
ORIGINAL_CHECK = ARTIFACTS / 'v91-frozen-actor-paired-20261126' / 'protocol.json'
# Selected before evaluation. The two water maps per kind have different
# terrain families and opposite detour directions.
CASE_IDS = (
    'vtemp5-circle-development-1545411713', 'vtemp5-circle-development-1545411715',
    'lake-dev-00-detour_left', 'lake-dev-01-detour_right',
    'valley-dev-00-detour_right', 'valley-dev-01-detour_left',
)
ARMS = ('predicted', 'observed')
MODEL_STATE = None


def load_models(weights):
    global MODEL_STATE
    from v91_extent_collect import load_frozen
    uav_path = Path(weights['uav']['path'])
    uav, config = load_frozen(uav_path.parent, uav_path)
    from v91_lake_metrics import load_model
    from v91_frozen_actor_inputs import load_actor
    MODEL_STATE = (uav, config, load_model(weights['world']['path']), load_actor(weights['actor']['path']))


def trial(folder, scene, arm, *, guided=True):
    from v91_km_cases import km_route_metrics
    if guided:
        from v91_guided_drive import run_km_actor_arm
        driver = 'v91_guided_drive.py'
    else:
        from v91_km_drive import run_km_actor_arm
        driver = 'v91_km_drive.py'
    if Path(run_km_actor_arm.__code__.co_filename).resolve() != (folder/'sources'/driver).resolve():
        raise RuntimeError('Worker did not load the pinned vehicle controller')
    base = scene['case_id'] + '-' + arm + '.json'
    started = time.monotonic()

    def heartbeat(state):
        write_json(folder / 'live' / base, dict(**state, wall_seconds=time.monotonic() - started))

    rollout = run_km_actor_arm(scene, *MODEL_STATE, arm, on_progress=heartbeat)
    path = np.asarray(rollout['car_path'])
    goal = np.asarray(scene['goal'])
    if (path.shape != (rollout['last_step'] + 1, 2) or not np.isfinite(path).all()
            or not np.allclose(path[0], scene['start'], atol=1e-4, rtol=0)):
        raise RuntimeError('Executed starting point or trajectory is invalid')
    if rollout['termination'] == 'frame_limit' and rollout['last_step'] != scene['max_steps']:
        raise RuntimeError('Incomplete simulation budget')
    if (np.linalg.norm(path[-1] - goal) <= 2.) != (rollout['termination'] == 'vehicle_arrival'):
        raise RuntimeError('Arrival disagrees with the executed final pose')
    metrics = km_route_metrics(scene, path)
    record = dict(case_id=scene['case_id'], kind=scene['kind'], fleet_seed=scene['fleet_seed'],
                  arm=arm, direct_distance_m=scene['direct_distance_m'], route=metrics, rollout=rollout)
    write_json(folder / 'cases' / base, record)
    heartbeat(dict(step=rollout['last_step'], max_steps=scene['max_steps'], termination=rollout['termination'],
                   goal_remaining_m=rollout['goal_remaining_m'], travelled_m=rollout['travelled_m']))
    return base


def summarize(folder):
    groups, pairs = {}, []
    protocol = json.loads((folder/'protocol.json').read_text())
    for scene in protocol['cases']:
        case_id = scene['case_id']
        records = [json.loads((folder / 'cases' / f'{case_id}-{arm}.json').read_text()) for arm in ARMS]
        pred, obs = [r['rollout'] for r in records]
        positions = [np.asarray(r['car_path']) for r in (pred, obs)]
        shared = min(map(len, positions))
        divergence = np.linalg.norm(positions[0][:shared] - positions[1][:shared], axis=1)
        changed = np.flatnonzero(divergence > .01)
        pair = dict(case_id=case_id, kind=records[0]['kind'], direct_distance_m=records[0]['direct_distance_m'],
                    first_vehicle_pose_difference_over_1cm_s=int(changed[0]) if changed.size else None,
                    largest_same_time_pose_difference_m=float(divergence.max()))
        for record in records:
            r = record['rollout']
            compact = {k:r[k] for k in ('termination','last_step','goal_remaining_m','ground_collisions',
                                         'ground_safety_violations','ground_physical_collision_attempts',
                                         'boundary_violation_frames','travelled_m','actor_calls','uav_packets_received')}
            compact['route'] = record['route']
            changes = [review['changed_angle_rad'] for review in r['reviews']
                       if review.get('changed_angle_rad') is not None]
            compact['counterfactual_reviews'] = len(changes)
            compact['reviews_changing_heading_over_1deg'] = (
                sum(change > math.radians(1.) for change in changes) if changes else None)
            pair[record['arm']] = compact
            key = (record['kind'], record['arm'])
            group = groups.setdefault(key, dict(maps=0, arrivals=0, timeouts=0, safety_stops=0,
                                               ground_collisions=0, ground_safety_violations=0, passed_obstacles=0))
            group['maps'] += 1
            group['arrivals'] += int(r['termination'] == 'vehicle_arrival')
            group['timeouts'] += int(r['termination'] == 'frame_limit')
            group['safety_stops'] += int(r['termination'] in ('safety_termination', 'ground_safety_violation'))
            group['ground_collisions'] += r['ground_collisions']
            group['ground_safety_violations'] += r['ground_safety_violations']
            group['passed_obstacles'] += int(record['route']['passed_obstacle'])
        pairs.append(pair)
    write_json(folder / 'results.json', dict(paired_cases=pairs,
               by_terrain=[dict(kind=k[0], arm=k[1], **v) for k,v in groups.items()]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--terrain', choices=('circle','shore','bridge'), nargs='+',
                        help='Limit a new paired evaluation to the specified terrain types')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('Use at least one worker')
    folder = args.output.resolve()
    prior = json.loads((folder/'protocol.json').read_text()) if args.resume else None
    guided = prior is None or 'vehicle_guidance_version' in prior
    if args.resume:
        sys.path.insert(0,str(folder/'sources'))
    from v91_km_cases import km_cases
    scenes = {s['case_id']:s for s in km_cases(ARTIFACTS.parent)}
    selected = [scenes[case_id] for case_id in CASE_IDS]
    if args.resume:
        if args.terrain:
            parser.error('A resumed evaluation uses its saved terrain selection')
        selected = [scenes[row['case_id']] for row in prior['cases']]
    elif args.terrain:
        selected = [s for s in selected if s['kind'] in args.terrain]
    for scene in selected:
        distance = np.linalg.norm(np.asarray(scene['start']) - scene['goal'])
        if (not scene['formal_eval'] or distance < 1000 or not np.isclose(distance, scene['direct_distance_m'])
                or scene['max_steps'] != math.ceil(6. * distance + 120.)):
            raise ValueError(f"Invalid kilometre-scale formal case {scene['case_id']}")
    weights = json.loads(ORIGINAL_CHECK.read_text())['weights']
    def verify_weights():
        if any(sha(w['path']) != w['sha256'] for w in weights.values()):
            raise ValueError('Frozen model identity changed')
    verify_weights()
    sources = ['run_v91_km_eval.py','v91_km_cases.py','v91_km_wild.py','v91_km_drive.py',
               'v91_frozen_actor_drive.py','v91_frozen_actor_inputs.py','v91_lake_drive.py',
               'v91_extent_collect.py','v91_lake_metrics.py','run_v91_extent_pilot.py','v91_extent_network.py',
               'run_v91_lake_test.py']
    if guided:
        sources += ['v91_guided_drive.py', 'v91_route_evidence.py', 'v91_rejoin_guidance.py',
                    'v91_extent_scenes.py', 'v91_lake_cases.py']
        if prior is None or 'v91_rolling_clearance.py' in prior['source_sha256']:
            sources.append('v91_rolling_clearance.py')
    source_files = {name:SCRIPTS/name for name in sources}
    if args.resume:
        source_files = {name:folder/'sources'/name for name in sources}
    uav_root = Path(weights['uav']['path']).parent
    runtime = uav_root / 'runtime_sources'
    runtime_files = {str(p.relative_to(runtime)):sha(p) for p in sorted(runtime.rglob('*.py'))}
    protocol = dict(weights=weights, test_only=True, suite='fixed development maps', arms=list(ARMS),
                    interval_s=30, arrival_radius_m=2., seconds_per_m=6, budget_offset_s=120,
                    source_sha256={name:sha(path) for name,path in source_files.items()},
                    uav_runtime_sources_sha256=hashlib.sha256(json.dumps(runtime_files, sort_keys=True).encode()).hexdigest(),
                    uav_config_sha256=sha(uav_root / 'config.json'),
                    cases=[{k:s[k] for k in ('case_id','kind','fleet_seed','start','goal','world_size',
                                             'direct_distance_m','reference_distance_m','max_steps','case')}
                           for s in selected])
    if guided:
        protocol['vehicle_guidance_version'] = prior['vehicle_guidance_version'] if prior else 'v91_received_route_guidance_v3'
    if args.resume:
        from run_v91_lake_test import jsonable
        if prior != json.loads(json.dumps(protocol, default=jsonable)):
            raise ValueError('Prior evaluation inputs or sources differ')
    else:
        folder.mkdir(parents=True, exist_ok=False)
        (folder/'cases').mkdir()
        (folder/'live').mkdir()
        (folder/'sources').mkdir()
        for name, path in source_files.items():
            shutil.copy2(path, folder/'sources'/name)
            if sha(folder/'sources'/name) != protocol['source_sha256'][name]:
                raise RuntimeError(f'Source changed during snapshot: {name}')
        write_json(folder/'protocol.json', protocol)
    # Alternate terrain types between first jobs while preserving each pair.
    order = selected[::2] + selected[1::2]
    jobs = [(s, a) for s in order for a in ARMS]
    pending = [(s,a) for s,a in jobs if not (folder/'cases'/f"{s['case_id']}-{a}.json").exists()]
    done = len(jobs) - len(pending)
    def progress(phase):
        write_json(folder/'progress.json', dict(phase=phase, completed=done, total=len(jobs)))
    progress('running')
    if not args.resume:
        # Spawned workers inherit this search path. They run the exact files
        # hashed in the manifest even if the workspace is edited during a run.
        sys.path.insert(0,str(folder/'sources'))
    try:
        with ProcessPoolExecutor(args.workers, mp_context=get_context('spawn'),
                                 initializer=load_models, initargs=(weights,)) as pool:
            futures = [pool.submit(trial, folder, s, a, guided=guided) for s,a in pending]
            for future in as_completed(futures):
                print(future.result(), flush=True)
                done += 1
                progress('running')
        verify_weights()
        if args.resume:
            saved = runpy.run_path(str(folder/'sources'/'run_v91_km_eval.py'))
            saved['summarize'](folder)
        else:
            summarize(folder)
        progress('complete')
    except BaseException:
        progress('interrupted')
        raise
    finally:
        if guided:
            sys.path.remove(str(folder/'sources'))


if __name__ == '__main__':
    main()
