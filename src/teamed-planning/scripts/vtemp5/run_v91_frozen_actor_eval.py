"""Evaluate a frozen UAV, vehicle waypoint policy and world model on new paired maps."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import json
from multiprocessing import get_context
from pathlib import Path
import shutil

import numpy as np

from run_v91_lake_test import sha, write_json
from v91_extent_collect import load_frozen
from v91_extent_scenes import make_case
from v91_lake_cases import crossed_bridge, make_lake_case


ARMS = ('native', 'observed', 'predicted')
FIRST_SEED = {'land': 840000, 'shore': 850000, 'bridge': 860000}
STATE = None


def init_worker(uav_path, world_path, actor_path):
    global STATE
    uav, config = load_frozen(uav_path.parent, uav_path)
    from v91_frozen_actor_inputs import load_actor
    from v91_lake_metrics import load_model
    STATE = (uav, config, load_model(world_path), load_actor(actor_path))


def run_one(folder, kind, seed, arm, steps):
    from v91_frozen_actor_drive import run_actor_arm

    if kind == 'land':
        scene = dict(make_case(seed, has_large=True), kind=kind, seed=seed, rects=[])
    else:
        scene = make_lake_case(seed, kind)
    result = run_actor_arm(scene, seed, *STATE, arm, max_steps=steps, interval=30)
    path = np.asarray(result['car_path'])
    if len(path) != result['last_step'] + 1 or not np.isfinite(path).all():
        raise ValueError('Invalid executed path')
    # Terrain truth is only used here, after all actions have executed.
    result['verified_crossing'] = crossed_bridge(scene, path) if kind == 'bridge' else None
    distance = float(np.linalg.norm(np.asarray(scene['goal']) - scene['start']))
    write_json(folder / 'cases' / f'{kind}-{seed}-{arm}.json',
               dict(kind=kind, seed=seed, arm=arm, mission_distance_m=distance, rollout=result))
    return kind, seed, arm


def summarize(folder):
    rows = [json.loads(p.read_text()) for p in sorted((folder / 'cases').glob('*.json'))]
    out = {}
    for kind in FIRST_SEED:
        cases = {a: {r['seed']: r['rollout'] for r in rows if r['kind'] == kind and r['arm'] == a}
                 for a in ARMS}
        assert cases['native'].keys() == cases['predicted'].keys() == cases['observed'].keys()
        group = {}
        for arm, records in cases.items():
            values = list(records.values())
            ends = [r['termination'] for r in values]
            group[arm] = dict(maps=len(values), arrivals=ends.count('vehicle_arrival'),
                timeouts=ends.count('frame_limit'), safety_stops=ends.count('safety_termination'),
                ground_collisions=sum(r['ground_collisions'] for r in values),
                crossings=sum(r['verified_crossing'] is True for r in values) if kind == 'bridge' else None,
                maps_with_actor_calls=sum(r['actor_calls'] > 0 for r in values),
                actor_calls=sum(r['actor_calls'] for r in values),
                reviews_with_different_candidates=sum(e['candidate_input_changed'] for r in values
                                                      for e in r['reviews']),
                reviews_with_angle_change_over_one_degree=sum(e['changed_angle_rad'] > np.deg2rad(1)
                                                           for r in values for e in r['reviews']),
                moving_actor_target_frames=sum(r['moving_actor_target_frames'] for r in values),
                mean_final_distance_m=float(np.mean([r['goal_remaining_m'] for r in values])),
                mean_travelled_m=float(np.mean([r['travelled_m'] for r in values])))
        pairs = {}
        for a, b in combinations(ARMS, 2):
            changes = {}
            for seed in cases[a]:
                ra, rb = cases[a][seed], cases[b][seed]
                pa, pb = np.asarray(ra['car_path']), np.asarray(rb['car_path'])
                n = min(len(pa), len(pb))
                error = np.linalg.norm(pa[:n] - pb[:n], axis=1)
                diverged = np.flatnonzero(error > 1e-6)
                changes[seed] = dict(first_pose_change_s=int(diverged[0]) if len(diverged) else None,
                    largest_same_time_distance_m=float(error.max()),
                    arrival_time_delta_b_minus_a_s=(rb['last_step'] - ra['last_step']
                        if ra['termination'] == rb['termination'] == 'vehicle_arrival' else None))
            pairs[a + '_vs_' + b] = changes
        out[kind] = dict(arms=group, paired_trajectories=pairs)
    write_json(folder/'results.json', out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('uav', 'world', 'actor', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--maps-per-kind', type=int, default=6)
    parser.add_argument('--steps', type=int, default=480)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not (1 <= args.maps_per_kind <= 1000 and 1 <= args.steps <= 2000 and args.workers > 0):
        parser.error('Use positive worker and map counts and a supported step budget.')
    folder = args.output.resolve()
    weights = [getattr(args, k).resolve() for k in ('uav', 'world', 'actor')]
    src = Path(__file__).resolve().parent
    names = ('run_v91_frozen_actor_eval.py', 'v91_frozen_actor_drive.py', 'v91_frozen_actor_inputs.py',
             'v91_lake_drive.py', 'v91_lake_metrics.py', 'v91_lake_cases.py', 'v91_extent_collect.py',
             'v91_extent_scenes.py', 'run_v91_extent_pilot.py', 'run_v91_lake_test.py', 'v91_extent_network.py')
    protocol = dict(weights={key: {'path':str(p), 'sha256':sha(p)} for key,p in
                             zip(('uav','world','actor'),weights)}, arms=list(ARMS), seed_starts=FIRST_SEED,
                    maps_per_kind=args.maps_per_kind, steps=args.steps, review_interval_s=30,
                    test_only=True, source_sha256={name:sha(src/name) for name in names})
    if args.resume:
        if json.loads((folder/'protocol.json').read_text()) != protocol:
            raise ValueError('Incompatible evaluation protocol')
    else:
        folder.mkdir(parents=True, exist_ok=False)
        (folder/'cases').mkdir()
        (folder/'sources').mkdir()
        for name in names:
            shutil.copy2(src/name, folder/'sources'/name)
        write_json(folder/'protocol.json', protocol)
    jobs = [(kind, seed+i, arm) for i in range(args.maps_per_kind)
            for kind, seed in FIRST_SEED.items() for arm in ARMS]
    pending = [j for j in jobs if not (folder/'cases'/('%s-%d-%s.json'%j)).exists()]
    done = len(jobs) - len(pending)
    def progress(phase):
        write_json(folder/'progress.json', dict(phase=phase, completed=done, total=len(jobs)))
    progress('running')
    with ProcessPoolExecutor(args.workers, mp_context=get_context('spawn'),
                             initializer=init_worker, initargs=tuple(weights)) as pool:
        futures = [pool.submit(run_one, folder, *job, args.steps) for job in pending]
        for future in as_completed(futures):
            print('Finished', *future.result(), flush=True)
            done += 1
            progress('running')
    summarize(folder)
    progress('complete')


if __name__ == '__main__':
    main()
