"""Paired, held-out lake/bridge evaluation. Reuse fixed pilot weights without training."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import statistics

import numpy as np

from v91_extent_collect import load_frozen


ARMS = ('native', 'observed', 'predicted')
SEEDS = {'shore': 610000, 'bridge': 620000}
STATE = None


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, default=jsonable, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def start_worker(uav_run, checkpoint, weight):
    from v91_lake_metrics import load_model
    global STATE
    uav, cfg = load_frozen(uav_run, checkpoint)
    STATE = (uav, cfg, load_model(weight))


def physical_rejoin(scene, path):
    """Score executed turns in ground truth only after a whole trajectory has finished."""
    from v91_lake_cases import true_rejoin_safe

    route = np.asarray(path)
    start = np.asarray(scene['start'])
    goal = np.asarray(scene['goal'])
    axis = goal - start
    axis /= np.linalg.norm(axis)
    transverse = np.array([-axis[1], axis[0]])
    first_alignment = first_safe = first_deviation = None
    for step in range(max(0, len(route)-5)):
        car = route[step]
        if first_deviation is None and abs((car-start) @ transverse) >= 4.:
            first_deviation = step
        if first_deviation is None:
            continue
        goal_vec = goal - car
        motion = route[step+5] - car
        dist = np.linalg.norm(goal_vec)
        if (dist <= 2. or np.linalg.norm(motion) < 1. or
                motion @ goal_vec < np.cos(np.deg2rad(25)) * np.linalg.norm(motion) * dist):
            continue
        safe = bool(true_rejoin_safe(scene, car, goal, horizon_m=24.))
        if first_alignment is None:
            first_alignment = dict(step=step, safe=safe, car_xy=car)
        if safe:
            first_safe = step
            break
    return dict(first_deviation=first_deviation, first_alignment=first_alignment,
                first_safe_alignment=first_safe)


def run_case(output, kind, seed, arm, limit):
    from v91_lake_cases import make_lake_case, true_rejoin_safe, crossed_bridge
    from v91_lake_drive import run_arm
    from v91_lake_metrics import score_snapshots

    scene = make_lake_case(seed, kind)
    snapshots_at = (1, 20, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480) if arm == 'native' else ()
    summary, observations = run_arm(scene, seed, *STATE, arm, max_steps=limit, interval=5,
                                    snapshot_steps=tuple(s for s in snapshots_at if s<=limit))
    # Truth is strictly kept out of the controller and used here after roll-out.
    for review in summary['reviews']:
        if review['eligible']:
            review['truth_safe'] = bool(true_rejoin_safe(scene, review['car_xy'], scene['goal'],
                                                         horizon_m=24.))
    extent = score_snapshots(scene, observations, STATE[2]) if observations else []
    traj = summary['car_path']
    summary['movement_rejoin'] = physical_rejoin(scene, traj)
    summary['crossed_verified_deck'] = bool(crossed_bridge(scene, traj)) if kind == 'bridge' else None
    row = dict(kind=kind, seed=seed, arm=arm, rollout=summary, extent=extent)
    write_json(output / 'cases' / f'{kind}-{seed}-{arm}.json', row)
    return kind, seed, arm


def same_history(cases):
    counts = {}
    for gate in ('observed', 'predicted'):
        counts[gate] = dict(tp=0, fp=0, tn=0, fn=0)
    first = []
    for case in cases:
        events = [e for e in case['rollout']['reviews'] if e['eligible']]
        times = dict(seed=case['seed'])
        for gate in counts:
            hit = None
            for event in events:
                allowed, safe = event[gate+'_approve'], event['truth_safe']
                key = ('tp' if safe else 'fp') if allowed else ('fn' if safe else 'tn')
                counts[gate][key] += 1
                if hit is None and allowed and safe:
                    hit = event['step']
            times[gate] = hit
        first.append(times)
    both = [t['observed']-t['predicted'] for t in first
            if t['observed'] is not None and t['predicted'] is not None]
    return dict(common_native_opportunities=counts, first_safe_approval_per_map=first,
                paired_maps=len(both), predicted_earlier_s=both)


def movement_comparison(cases, alternate):
    by = {(c['seed'],c['arm']): c['rollout']['movement_rejoin'] for c in cases}
    deltas = []
    for seed in sorted({c['seed'] for c in cases}):
        a = by[(seed,alternate)]['first_safe_alignment']
        b = by[(seed,'predicted')]['first_safe_alignment']
        if a is not None and b is not None:
            deltas.append(dict(seed=seed, predicted_earlier_s=a-b))
    return deltas


def summarize(output):
    from v91_lake_metrics import summarize_cases
    rows = [json.loads(p.read_text()) for p in sorted((output/'cases').glob('*.json'))]
    result = {}
    for kind in SEEDS:
        cases = [r for r in rows if r['kind']==kind]
        native = [c for c in cases if c['arm']=='native']
        group = dict(extent=summarize_cases([c['extent'] for c in native]),
                     same_history=same_history(native),
                     paired_motion={a:movement_comparison(cases,a) for a in ('native','observed')})
        for arm in ARMS:
            maps=[c for c in cases if c['arm']==arm]
            records=[c['rollout'] for c in maps]
            firsts=[v['movement_rejoin']['first_alignment'] for v in records]
            group[arm] = dict(maps=len(maps), arrivals=sum(v['termination']=='vehicle_arrival' for v in records),
                endings={k:sum(v['termination']==k for v in records) for k in sorted({v['termination'] for v in records})},
                ground_collisions=sum(v['ground_collisions'] for v in records),
                verified_deck_crossings=sum(c['rollout']['crossed_verified_deck'] is True for c in maps) if kind=='bridge' else None,
                maps_with_applied_override=sum(any(e['applied'] for e in v['reviews']) for v in records),
                maps_with_executed_detour=sum(v['movement_rejoin']['first_deviation'] is not None for v in records),
                maps_with_alignment=sum(x is not None for x in firsts),
                maps_with_unsafe_first_alignment=sum(x is not None and not x['safe'] for x in firsts),
                maps_with_safe_alignment=sum(v['movement_rejoin']['first_safe_alignment'] is not None for v in records),
                mean_travelled_m=statistics.mean(v['travelled_m'] for v in records))
        result[kind]=group
    write_json(output/'results.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maps-per-kind',type=int,default=8)
    parser.add_argument('--steps',type=int,default=480)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not (1<=args.maps_per_kind<=1000 and 1<=args.steps<=480 and args.workers>=1):
        parser.error('Invalid map count, step budget, or worker count')
    pilot=args.pilot.resolve()
    manifest=json.loads((pilot/'manifest.json').read_text())
    checkpoint=Path(manifest['checkpoint'])
    weight=pilot/'model/best.pt'
    if manifest['checkpoint_sha256']!=sha(checkpoint):
        raise ValueError('Unexpected UAV checkpoint')
    output=args.output.resolve()
    files=[Path(__file__).resolve(), *sorted(Path(__file__).resolve().parent.glob('v91_lake_*.py'))]
    protocol=dict(pilot=str(pilot),uav=str(checkpoint),uav_sha256=sha(checkpoint),
                  world_model=str(weight),model_sha256=sha(weight),maps_per_kind=args.maps_per_kind,
                  seed_starts=SEEDS,arms=list(ARMS),steps=args.steps,simulation_dt_s=1,
                  observed_and_predicted_decisions_at_same_states='native shadow reviews, every 5 s subject to eligibility',
                  motion_measure='After first 4 m lateral offset, first 5 s net movement of >=1 m within 25 degrees of goal; safety scored independently over next 24 m in true geometry.',
                  source_hashes={p.name:sha(p) for p in files})
    output.mkdir(parents=True,exist_ok=args.resume)
    if args.resume:
        if json.loads((output/'protocol.json').read_text())!=protocol:
            raise ValueError('Refusing to mix runs with different code or protocols')
    else:
        (output/'cases').mkdir()
        (output/'sources').mkdir()
        for path in files: shutil.copy2(path,output/'sources'/path.name)
        write_json(output/'protocol.json',protocol)
    jobs=[(kind,SEEDS[kind]+idx,arm) for idx in range(args.maps_per_kind) for kind in SEEDS for arm in ARMS]
    todo=[job for job in jobs if not (output/'cases'/('%s-%d-%s.json'%job)).is_file()]
    completed=len(jobs)-len(todo)
    def status(phase):
        write_json(output/'progress.json', dict(phase=phase, completed=completed,total=len(jobs),
                  updated=datetime.now(timezone.utc).isoformat()))
    status('running')
    with ProcessPoolExecutor(args.workers,mp_context=get_context('spawn'),initializer=start_worker,
                             initargs=(manifest['uav_run'],checkpoint,weight)) as pool:
        futures=[pool.submit(run_case,output,*job,args.steps) for job in todo]
        for future in as_completed(futures):
            print('Finished',*future.result(),flush=True)
            completed+=1
            status('running')
    summarize(output)
    status('completed')


if __name__=='__main__':
    main()
