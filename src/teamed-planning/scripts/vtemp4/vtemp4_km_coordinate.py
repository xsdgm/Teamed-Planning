"""Kilometre VTemp4: dense reachable ground targets and a continuous polar survey action."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.ndimage import binary_dilation, label
import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal

from vtemp4.vtemp4_km_route_proxy import (CASES, DATA, LIMIT, LOCAL_RADIUS, ROOT, SPLITS, SurveyEnv,
                            confirmed_path, local_view, return_strip, summary, survey_route, terrain)

SEED = 421
BASELINE = ROOT/'artifacts/vtemp4-km-route-proxy-20261119'
ANGLES = np.deg2rad(np.array([-135, -90, -45, 0, 45, 90, 135, 180]))
RADII = np.arange(8., 81., 8.)
FEATURES = 7


def receiver_context(env, stage, previous):
    delta = env.dest-env.pos
    bearing = math.atan2(delta[0],delta[1])
    headings = bearing+ANGLES
    directions = np.stack((np.sin(headings),np.cos(headings)),axis=-1)
    sample = np.rint(env.pos+directions[:,None,:]*RADII[None,:,None]).astype(int)
    inside = ((sample>=0)&(sample<env.cells.shape)).all(axis=2)
    clipped = np.clip(sample,0,np.asarray(env.cells.shape)-1)
    observed = np.where(inside,env.cells[tuple(clipped.transpose(2,0,1))],1)
    return np.r_[delta/max(1.,np.linalg.norm(delta)),np.linalg.norm(delta)/1500.,
                 (stage-env.pos)/80.,(previous-env.pos)/80.,env.time/LIMIT,
                 (observed==0).mean(axis=1),(observed==1).mean(axis=1)].astype('f')


def reachable_cells(received, car):
    tags,_ = label(received==-1)
    group = tags[tuple(car)]
    if group==0:
        raise AssertionError('Missing confirmed car cell')
    return tags==group


def ground_features(offsets, goal_delta):
    d = offsets[:,:2].astype('f')
    goal_delta = np.asarray(goal_delta,dtype='f')
    goal_distance = np.linalg.norm(goal_delta)
    heading = goal_delta/max(goal_distance,1.)
    side = (d[:,0]*heading[1]-d[:,1]*heading[0])/80.
    progress = goal_distance-np.linalg.norm(goal_delta-d,axis=1)
    return torch.from_numpy(np.column_stack((
        progress/30.,np.linalg.norm(d,axis=1)/80.,side,np.abs(side),
        np.all(d==goal_delta,axis=1),offsets[:,2],(d@heading)/80.)).astype('f'))


def metric_target(env,lo,cells,latent):
    # Actions are unconstrained normal draws in a goal-aligned polar frame.
    # The available distance along the ray is measured from the actual vehicle pose.
    delta = env.goal-env.position
    angle = math.atan2(delta[1],delta[0])+math.pi*math.tanh(float(latent[0]))
    direction = np.array([math.cos(angle),math.sin(angle)])
    lower = env.origin+lo[::-1]*env.resolution
    upper = lower+np.array(cells.shape[::-1])*env.resolution
    edges = np.where(direction>=0,upper,lower)
    limits = np.divide(edges-env.position,direction,out=np.full(2,np.inf),
                       where=np.abs(direction)>1.e-10)
    fraction = 1./(1.+math.exp(-float(np.clip(latent[1],-30.,30.))))
    distance = max(0.,limits.min()-1.e-5)*fraction
    xy = env.position+direction*distance
    rc = env._index(xy)
    assert np.all(rc>=lo) and np.all(rc<lo+cells.shape)
    return xy,rc


class CoordinatePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(24,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.ground = nn.Linear(32,FEATURES)
        self.survey = nn.Linear(32,2)
        self.log_std = nn.Parameter(torch.tensor([-.95,-.45]))
        self.value = nn.Linear(32,1)
        # Trainable initialization prefers forward progress and a known reachable final goal.
        # Continuous requests initially face the mission bearing at ~78% of local range.
        with torch.no_grad():
            self.ground.weight.mul_(.05)
            self.ground.bias.copy_(torch.tensor([4.,-.4,0.,-.4,10.,.2,0.]))
            self.survey.weight.mul_(.05)
            self.survey.bias.copy_(torch.tensor([0.,1.25]))

    def survey_distribution(self,z):
        return Independent(Normal(self.survey(z),self.log_std.clamp(-2.,.3).exp()),1)


@dataclass
class Decision:
    head: str
    context: np.ndarray
    action: object
    log_prob: float
    value: float
    reward_prefix: float
    offsets: object = None
    goal_delta: object = None


def choose_stage(net,env,stage,previous,lo,view,mask,train,history,prefix):
    points = np.argwhere(mask)
    relative = points+lo-env.pos
    frontier = binary_dilation(view==0)
    compact = np.column_stack((relative,frontier[tuple(points.T)])).astype(np.int8)
    assert np.array_equal(relative,compact[:,:2])
    ctx = receiver_context(env,stage,previous)
    z = net.encoder(torch.from_numpy(ctx))
    goal_delta = env.dest-env.pos
    logits = ground_features(compact,goal_delta) @ net.ground(z)
    if train and len(points)>1:
        dist = Categorical(logits=logits)
        action = dist.sample()
        history.append(Decision('ground',ctx,int(action),float(dist.log_prob(action)),
                                float(net.value(z).item()),prefix,compact,goal_delta))
    else:
        action = logits.argmax()
    chosen = points[int(action)]
    assert mask[tuple(chosen)] and view[tuple(chosen)]==-1
    return chosen+lo,len(points)


def choose_probe(net,env,stage,previous,lo,view,train,history,prefix):
    ctx = receiver_context(env,stage,previous)
    z = net.encoder(torch.from_numpy(ctx))
    dist = net.survey_distribution(z)
    draw = dist.sample() if train else dist.mean
    if train:
        history.append(Decision('survey',ctx,draw.numpy(),float(dist.log_prob(draw)),
                                float(net.value(z).item()),prefix))
    return metric_target(env,lo,view,draw.numpy())


@torch.no_grad()
def episode(job):
    case,weights,mode,seed,save_trace = job
    net = CoordinatePolicy()
    net.load_state_dict(weights)
    torch.manual_seed(seed)
    train = mode=='train'
    scene,dist = terrain(case)
    env = SurveyEnv(**scene.as_kwargs(),max_time_s=LIMIT)
    assert env.resolution==1. and CASES[case]['direct_distance_m']>=1000.
    initial = dist[tuple(env.pos)]
    stage = previous = env.pos.copy()
    ground_at = survey_at = -100
    fresh = False
    checks = packets = requested = blocked_targets = disconnected = 0
    stage_sizes,history,trace = [],[],[]
    while env.time<LIMIT and not np.array_equal(env.pos,env.dest):
        lo,view = local_view(env)
        ground_due = np.array_equal(stage,env.pos) or env.time-ground_at>=60
        probe_due = mode!='off' and (env.time-survey_at>=60 or (env.time-survey_at>=20 and fresh))
        prefix = .2*(1.-dist[tuple(env.pos)]/initial)
        if ground_due or probe_due:
            mask = reachable_cells(view,env.pos-lo)
        if ground_due:
            stage,n = choose_stage(net,env,stage,previous,lo,view,mask,train,history,prefix)
            ground_at = env.time
            stage_sizes.append(n)
            if save_trace:
                trace.append(dict(kind='ground',t=env.time,car=env.position.tolist(),
                                  xy=env._xy(stage).tolist(),legal_cells=n))
        if probe_due:
            fresh = False
            survey_at = env.time
            destination = env.dest-lo
            goal_known = np.all(destination>=0) and np.all(destination<view.shape) and mask[tuple(destination)]
            if not goal_known:
                xy,target = choose_probe(net,env,stage,previous,lo,view,train,history,prefix)
                previous = target.copy()
                requested += 1
                path = None
                if view[tuple(target-lo)]==1:
                    blocked_targets += 1
                else:
                    path = survey_route(view,target,env.pos,lo)
                    disconnected += path is None
                if path is not None:
                    generation = env._map_generation
                    return_strip(env,path)
                    packets += 1
                    fresh = env._map_generation>generation
                if save_trace:
                    polyline = None
                    if path is not None:
                        polyline = np.vstack((xy,env._xy(path),env.position)).tolist()
                    trace.append(dict(kind='survey',t=env.time,car=env.position.tolist(),
                                      xy=xy.tolist(),cell_xy=env._xy(target).tolist(),route=polyline))
        env._path = confirmed_path(env,stage,lo,view)[:26]
        env._path_index = 0
        checks += 1
        for _ in range(5):
            env.time += 1
            env._advance_ground()
            if np.array_equal(env.pos,env.dest):
                break
    success = bool(np.array_equal(env.pos,env.dest))
    reward = .2*(1.-dist[tuple(env.pos)]/initial)+success*(1.-.3*env.time/LIMIT)
    row = dict(case=case,success=success,seconds=env.time,reward=float(reward),driven_m=env.travelled,
               residual_route_m=float(dist[tuple(env.pos)]),direct_m=CASES[case]['direct_distance_m'],
               stage_issues=len(stage_sizes),min_legal=min(stage_sizes),max_legal=max(stage_sizes),
               reachability_checks=checks,probe_requests=requested,probe_packets=packets,
               known_blocked_targets=blocked_targets,no_route=disconnected)
    assert requested==packets+blocked_targets+disconnected
    return row,history,trace


def log_probs(net,head,rows,z):
    if head=='survey':
        actions = torch.from_numpy(np.stack([r.action for r in rows]))
        return net.survey_distribution(z).log_prob(actions)
    weights = net.ground(z)
    logps = []
    for i,r in enumerate(rows):
        logits = ground_features(r.offsets,r.goal_delta) @ weights[i]
        logps.append(logits[r.action]-torch.logsumexp(logits,dim=0))
    return torch.stack(logps)


def optimize(net,opt,episodes,rng):
    # Monte Carlo future team return for each actual head decision (undiscounted).
    for head in ('ground','survey'):
        rows = [(r,ep['reward']-r.reward_prefix) for ep,traj,_ in episodes for r in traj if r.head==head]
        if not rows:
            continue
        actions,returns = zip(*rows)
        returns = torch.tensor(returns,dtype=torch.float32)
        advantages = returns-torch.tensor([r.value for r in actions])
        advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-6)
        for _ in range(2):
            for indices in np.array_split(rng.permutation(len(rows)),math.ceil(len(rows)/48)):
                batch = [actions[i] for i in indices]
                z = net.encoder(torch.from_numpy(np.stack([r.context for r in batch])))
                logp = log_probs(net,head,batch,z)
                ratio = (logp-torch.tensor([r.log_prob for r in batch])).exp()
                advantage = advantages[indices]
                actor_loss = -torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
                critic_loss = (net.value(z).flatten()-returns[indices]).square().mean()
                opt.zero_grad()
                (actor_loss+.5*critic_loss).backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step()


def roll(pool,net,cases,mode,seed,trace=False):
    state = {k:v.detach().clone() for k,v in net.state_dict().items()}
    return list(pool.map(episode,[(c,state,mode,seed+i,trace) for i,c in enumerate(cases)]))


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True,exist_ok=False)
    groups = {k:[r['case_id'] for r in SPLITS[k]] for k in ('ppo_train','development','test')}
    old = json.loads((BASELINE/'manifest.json').read_text())
    assert groups==old['groups'] and LIMIT==old['limit_s'] and LOCAL_RADIUS==old['local_radius_m']
    assert len({r['circles_sha256'] for r in CASES.values()})==sum(map(len,groups.values()))
    assert all(1000.<=np.linalg.norm(np.array(r['goal'])-r['start'])<=1500. for r in CASES.values())
    manifest = dict(groups=groups,scene_manifest=str(DATA/'scenes.json'),seed=SEED,
                    baseline=str(BASELINE),epochs_over_train_maps=1,updates=6,workers=4,
                    init='new weights, learnable forward/known-goal bias and goal-oriented survey bearing',
                    receiver_context='goal, car-relative prior task points, remaining time and 8 measured ray summaries',
                    ground='categorical over every 4-connected confirmed free cell within local window',
                    survey='bounded goal-relative polar Gaussian, continuous XY then 1m sensor grid lookup',
                    time_limit_s=LIMIT,local_radius_m=LOCAL_RADIUS,ground_replan_s=5,survey_hold_review_s=[20,60],
                    shared_reward='0.2 normalized graph progress + arrival*(1-0.3*time/limit)',
                    sources={p:hashlib.sha256((Path(__file__).parent/p).read_bytes()).hexdigest()
                             for p in ('vtemp4_km_coordinate.py','vtemp4_km_route_proxy.py','vtemp4_proxy_smoke.py')})
    dump(out/'manifest.json',manifest)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    net = CoordinatePolicy()
    start = {k:v.clone() for k,v in net.state_dict().items()}
    opt = torch.optim.Adam(net.parameters(),lr=.0004)
    sequence=list(rng.permutation(groups['ppo_train']))
    best = -math.inf
    with ProcessPoolExecutor(max_workers=4) as pool:
        for step in range(1,7):
            batch = roll(pool,net,sequence[(step-1)*4:step*4],'train',SEED+step*10)
            optimize(net,opt,batch,rng)
            entry = dict(step=step,train=summary([r for r,_,_ in batch]))
            if step in (2,4,6):
                dev = roll(pool,net,groups['development'],'eval',SEED)
                entry['dev'] = summary([r for r,_,_ in dev])
                if entry['dev']['mean_reward']>best:
                    best = entry['dev']['mean_reward']
                    torch.save(dict(state=net.state_dict(),step=step),out/'best.pt')
            with (out/'training.jsonl').open('a') as f:
                print(json.dumps(entry),file=f)
            print(json.dumps(entry),flush=True)
        cp = torch.load(out/'best.pt',weights_only=True,map_location='cpu')
        net.load_state_dict(cp['state'])
        delta = {name:math.sqrt(sum(float((v-start[k]).square().sum()) for k,v in net.state_dict().items()
                                   if k.startswith(name))) for name in ('encoder','ground','survey')}
        assert min(delta.values())>0
        results = {}
        for name,mode in [('coordinate','eval'),('no_probe','off'),('untrained','eval')]:
            if name=='untrained':
                net.load_state_dict(start)
            batch = roll(pool,net,groups['test'],mode,SEED,trace=name=='coordinate')
            rows = [r for r,_,_ in batch]
            results[name] = summary(rows)
            dump(out/f'{name}.json',rows)
            if name=='coordinate':
                with (out/'decisions.jsonl').open('w') as f:
                    for row,_,records in batch:
                        for record in records:
                            print(json.dumps(dict(case=row['case'],**record)),file=f)
            print(name,results[name],flush=True)
    for name,old_name in [('candidate','vtemp4'),('rules','rules')]:
        rows=json.loads((BASELINE/f'{old_name}.json').read_text())
        assert [r['case'] for r in rows]==groups['test']
        results[name]=summary(rows)
    dump(out/'results.json',dict(test=results,selected_step=cp['step'],parameter_delta=delta))


if __name__=='__main__':
    main()
