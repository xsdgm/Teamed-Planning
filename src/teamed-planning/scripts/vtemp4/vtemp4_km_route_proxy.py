"""VTemp4 with one probe-to-current-car survey route per request, on kilometre V9 cases."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import torch

from vtemp4.vtemp4_proxy_smoke import (ROOT, Policy, SurveyEnv, choose, ground_plan, load_survey_map,
                         reward_distances, stages, summary, update, _sample_corridors)

DATA = ROOT/'artifacts/vtemp2-survey-km-data-20261119'
PARENT = ROOT/'artifacts/vtemp4-proxy-smoke-20261119/best.pt'
LIMIT = 3000
SEED = 420
LOCAL_RADIUS = 80
RADIUS = 12
Y, X = np.ogrid[-RADIUS:RADIUS+1, -RADIUS:RADIUS+1]
FOOTPRINT = Y*Y+X*X <= RADIUS*RADIUS
SPLITS = json.loads((DATA/'scenes.json').read_text())
CASES = {r['case_id']:r for group in ('ppo_train','development','test') for r in SPLITS[group]}


@lru_cache(maxsize=3)
def terrain(case):
    row = dict(CASES[case])
    row['circles_path'] = str(DATA/'geometry'/Path(row['circles_path']).name)
    scene = load_survey_map(row)
    goal = np.floor((scene.goal-scene.origin)/scene.resolution).astype(int)[::-1]
    return scene, reward_distances(scene.free, goal)


def local_view(env):
    lo = np.maximum(env.pos-LOCAL_RADIUS, 0)
    hi = np.minimum(env.pos+LOCAL_RADIUS+1, env.cells.shape)
    return lo, env.cells[lo[0]:hi[0], lo[1]:hi[1]]


def confirmed_path(env, stage, lo, cells):
    path, _, reason = ground_plan(cells, env.pos-lo, stage-lo, 1.)
    if reason != 'known_goal_route':
        # A retained stage can leave the local planning window as the car turns.
        lo = np.maximum(env._seen_min-1, 0)
        hi = np.minimum(env._seen_max+2, env.cells.shape)
        cells = env.cells[lo[0]:hi[0], lo[1]:hi[1]]
        path, _, reason = ground_plan(cells, env.pos-lo, stage-lo, 1.)
    path = path+lo
    assert reason == 'known_goal_route' and np.array_equal(path[-1], stage)
    assert np.all(env.cells[tuple(path.T)] == -1)
    return path


def survey_route(received, target, car, lo):
    """Shortest path from probe point back to issuance car; no previously issued pose."""
    valid = received != 1
    costs = np.where(received == 0, 1.25, 1.)
    ids = np.arange(received.size).reshape(received.shape)
    hor = valid[:,1:] & valid[:,:-1]
    ver = valid[1:] & valid[:-1]
    a = np.r_[ids[:,1:][hor], ids[1:][ver]]
    b = np.r_[ids[:,:-1][hor], ids[:-1][ver]]
    weights = .5*(costs.flat[a]+costs.flat[b])
    graph = csr_matrix((np.r_[weights,weights], (np.r_[a,b], np.r_[b,a])),
                       shape=(received.size,received.size))
    start, end = (ids[tuple(p-lo)] for p in (target,car))
    distances, parent = dijkstra(graph, directed=True, indices=start, return_predecessors=True)
    if not np.isfinite(distances[end]):
        return None
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    points = np.column_stack(np.unravel_index(path[::-1], received.shape))+lo
    assert np.array_equal(points[0],target) and np.array_equal(points[-1],car)
    assert np.all(np.abs(np.diff(points,axis=0)).sum(axis=1) == 1)
    assert np.all(received[tuple((points-lo).T)] != 1)
    return points


def return_strip(env, route):
    lo = np.maximum(route.min(axis=0)-RADIUS, 0)
    hi = np.minimum(route.max(axis=0)+RADIUS+1, env.cells.shape)
    mask = np.zeros(hi-lo,dtype=bool)
    mask[tuple((route-lo).T)] = True
    covered = binary_dilation(mask,structure=FOOTPRINT)
    rr, cc = np.nonzero(covered)
    ids = np.ravel_multi_index((rr+lo[0],cc+lo[1]),env.cells.shape)
    # Terrain enters only at the sensor. Requested routes can expose blocked ground.
    obstructed = bool((~env._free[tuple(route.T)]).any())
    env._receive(np.empty(0,int),ids,np.where(env._free.flat[ids],-1,1),env.time)
    return obstructed


def inputs(env, points, stage, last_probe):
    pos, goal = env.pos, env.dest
    def local_fractions(p):
        y,x = p
        tile = env.cells[max(0,y-8):y+9,max(0,x-8):x+9]
        return [np.mean(tile == 0),np.mean(tile == 1)]
    goal_delta = goal-pos
    context = np.r_[goal_delta/max(160., np.linalg.norm(goal_delta)),
                    (last_probe-pos)/80., (stage-pos)/80., env.time/LIMIT, local_fractions(pos)]
    progress = np.linalg.norm(goal_delta)-np.linalg.norm(goal-points,axis=1)
    delta = points-pos
    feat = np.column_stack([delta/50.,progress/30.,np.linalg.norm(delta,axis=1)/50.,
                            np.linalg.norm(points-last_probe,axis=1)/60.,
                            np.linalg.norm(points-stage,axis=1)/50.,np.all(points==goal,axis=1),
                            [local_fractions(p) for p in points]])
    return context.astype('f'),feat.astype('f')


def episode(job):
    case, params, ground_mode, survey_mode, seed, trace = job
    scene, dist = terrain(case)
    env = SurveyEnv(**scene.as_kwargs(),max_time_s=LIMIT)
    assert env.resolution==1. and CASES[case]['direct_distance_m']>=1000
    net = Policy()
    net.load_state_dict(params)
    rng = np.random.default_rng(seed)
    initial_dist = dist[tuple(env.pos)]
    stage = env.pos.copy()
    last_probe = env.pos.copy()
    ground_at = survey_at = -100
    new_evidence = False
    requests = deliveries = stage_count = checks = blocked = 0
    lengths, history, routes = [], [], []
    while env.time<LIMIT and not np.array_equal(env.pos,env.dest):
        lo, cells = local_view(env)
        ground_due = np.array_equal(stage,env.pos) or env.time-ground_at>=60
        # A directly returned strip completes the task immediately; 20 s is the earliest retask.
        # If it reveals no new cells the scheduled 60 s reevaluation still applies.
        survey_due = survey_mode!='off' and (env.time-survey_at>=60 or
                        (env.time-survey_at>=20 and new_evidence))
        progress = .2*(1.-dist[tuple(env.pos)]/initial_dist)
        if ground_due or survey_due:
            _, fronts, reason = ground_plan(cells,env.pos-lo,env.dest-lo,1.)
            fronts = fronts+lo
            known_goal = reason=='known_goal_route'
        if ground_due:
            points = stages(env,fronts,known_goal)
            index = choose(net,ground_mode,0,inputs(env,points,stage,last_probe),0,rng,history,progress)
            stage, ground_at = points[index].copy(), env.time
            stage_count += 1
        if survey_due:
            survey_at = env.time
            new_evidence = False
            qs = [] if known_goal else _sample_corridors(cells,fronts-lo,env.dest-lo,1.)
            if qs:
                points = np.array([q[0]+lo for q in qs])
                rank = [(np.linalg.norm(env.dest-env.pos)-np.linalg.norm(env.dest-p))/25.
                        + min(1.,len(q[1])/16.)-np.linalg.norm(p-env.pos)/125. for p,q in zip(points,qs)]
                i = choose(net,survey_mode,1,inputs(env,points,stage,last_probe),int(np.argmax(rank)),
                           rng,history,progress)
                target = points[i]
                path = survey_route(cells,target,env.pos,lo)
                requests += 1
                last_probe = target.copy()
                if path is not None:
                    generation = env._map_generation
                    blocked += return_strip(env,path)
                    new_evidence = env._map_generation>generation
                    lengths.append(len(path)-1)
                    deliveries += 1
                    if trace:
                        routes.append(dict(t=env.time,car_xy=env.position.tolist(),
                                           probe_xy=env._xy(target).tolist(),
                                           path_probe_to_car_xy=np.vstack((env._xy(path[:-1]),env.position)).tolist()))
        env._path = confirmed_path(env,stage,lo,cells)[:26]
        env._path_index = 0
        checks += 1
        for _ in range(5):
            env.time += 1
            env._advance_ground()
            if np.array_equal(env.pos,env.dest):
                break
    success = bool(np.array_equal(env.pos,env.dest))
    reward = .2*(1.-dist[tuple(env.pos)]/initial_dist) + success*(1.-.3*env.time/LIMIT)
    row = dict(case=case,success=success,seconds=env.time,reward=float(reward),
               direct_m=CASES[case]['direct_distance_m'],driven_m=env.travelled,
               residual_route_m=float(dist[tuple(env.pos)]),stage_issues=stage_count,
               reachability_checks=checks,probe_requests=requests,probe_packets=deliveries,
               strip_length_sum_m=sum(lengths),routes_discovering_obstruction=blocked)
    samples = [(head,ctx,feat,action,lp,value,reward-prefix)
               for head,ctx,feat,action,lp,value,prefix in history]
    return row,samples,routes


def rollout(pool,net,cases,modes,seed,trace=False):
    state = {k:v.detach().clone() for k,v in net.state_dict().items()}
    jobs = [(c,state,*modes,seed+i,trace) for i,c in enumerate(cases)]
    return list(pool.map(episode,jobs))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    ap.add_argument('--workers',type=int,default=4)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    groups = {k:[r['case_id'] for r in SPLITS[k]] for k in ('ppo_train','development','test')}
    assert len({r['circles_sha256'] for r in CASES.values()}) == sum(map(len,groups.values()))
    assert all(1000<=np.linalg.norm(np.array(r['goal'])-r['start'])<=1500 for r in CASES.values())
    manifest = dict(groups=groups,dataset=str(DATA),baseline_checkpoint=str(PARENT),
                    baseline_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
                    seed=SEED,limit_s=LIMIT,grid_m=1,local_radius_m=LOCAL_RADIUS,
                    strip_radius_m=RADIUS,ground_replan_s=5,survey_min_s=20,survey_review_s=60,
                    training_updates=6,episodes_per_update=4,workers=args.workers)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    net = Policy()
    net.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=True)['state'])
    initial = {k:v.clone() for k,v in net.state_dict().items()}
    opt = torch.optim.Adam(net.parameters(),lr=.00075)
    order = list(rng.permutation(groups['ppo_train']))
    best = -math.inf
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for step in range(1,7):
            records = rollout(pool,net,order[4*(step-1):4*step],('train','train'),SEED+10*step)
            update(net,opt,[s for _,batch,_ in records for s in batch])
            stats = dict(step=step,train=summary([r for r,_,_ in records]))
            if step in (2,4,6):
                dev = rollout(pool,net,groups['development'],('learned','learned'),SEED)
                stats['dev'] = summary([r for r,_,_ in dev])
                if stats['dev']['mean_reward']>best:
                    best=stats['dev']['mean_reward']
                    torch.save(dict(state=net.state_dict(),step=step),out/'best.pt')
            with (out/'training.jsonl').open('a') as f:
                print(json.dumps(stats),file=f)
            print(json.dumps(stats),flush=True)
        cp = torch.load(out/'best.pt',map_location='cpu',weights_only=True)
        net.load_state_dict(cp['state'])
        delta = {name:math.sqrt(sum(float((v-initial[k]).square().sum()) for k,v in net.state_dict().items()
                                   if k.startswith(name))) for name in ('encoder','heads.0','heads.1')}
        assert min(delta.values())>0
        results = {}
        for label,modes in [('vtemp4',('learned','learned')),('rules',('rule','rule')),
                            ('no_probe',('learned','off')),('before_km_training',('learned','learned'))]:
            if label=='before_km_training':
                net.load_state_dict(initial)
            evaluations=rollout(pool,net,groups['test'],modes,SEED,trace=label=='vtemp4')
            rows=[r for r,_,_ in evaluations]
            results[label]=summary(rows)
            (out/f'{label}.json').write_text(json.dumps(rows,indent=2))
            if label=='vtemp4':
                with (out/'routes.jsonl').open('w') as f:
                    for row,_,routes in evaluations:
                        for route in routes:
                            print(json.dumps(dict(case=row['case'],**route)),file=f)
            print(label,results[label],flush=True)
    (out/'results.json').write_text(json.dumps(dict(test=results,selected_step=cp['step'],
                                                     parameter_change_l2=delta),indent=2))


if __name__=='__main__':
    main()
