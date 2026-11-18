"""Static GOOSE observation-mask tasks with abstract, goal-conditioned targets.

Actions reveal portions of already measured data, not simulated aerial views.
The graph is adjacency of observed open-ground semantic evidence. It is NOT a
clearance-certified vehicle graph. No physical flight or navigation rewards.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from torch.utils.data import Dataset

from .frontier_data import outward_mask, load_regional_manifest
from .semantic_data import atomic_json, sha256

FORMAT = 'goose-abstract-observation-v1'
STATE_NAMES = ('semantic_proxy_connected', 'reachable_goal_progress', 'semantic_path_efficiency',
               'corridor_observed_fraction', 'corridor_nonopen_fraction', 'reachable_fraction',
               'goal_cell_observed')
STATE_DIM, CHANNELS, SIDE = len(STATE_NAMES), 10, 16
UTILITY_WEIGHTS = np.array([4., 2., 1., .5, .25, 0., .1], np.float32)
TRAVEL_WEIGHT = .02
SPLITS = {'garching':'val', 'hohenbrunn':'calibration', 'putzbrunn':'calibration',
          'neuperlach':'calibration', 'solalinden':'test', 'siegertsbrunn':'test', 'aying':'test',
          'campus-neubiberg':'train', 'hoehenkirchner':'train', 'gooseex-all-sites-training-only':'train'}
CROSS = np.array([[0,1,0],[1,1,1],[0,1,0]])


def corridor(start, goal, side=SIDE):
    yy,xx=np.indices((side,side));points=np.stack((xx,yy),-1)
    reference=np.linalg.norm(np.asarray(goal)-start)
    return np.linalg.norm(points-start,axis=-1)+np.linalg.norm(points-goal,axis=-1)<=reference+2.


def evidence_state(probabilities, observed, start, goal):
    """Only observed labels enter this proxy; unknown is never an open node."""
    observed=np.asarray(observed,bool);p=np.asarray(probabilities)
    start,goal=np.asarray(start,int),np.asarray(goal,int)
    h,w=observed.shape
    if p.shape!=(5,h,w) or any(np.any(point<0) or point[0]>=w or point[1]>=h for point in (start,goal)):
        raise ValueError('Invalid evidence proxy dimensions/endpoints')
    open_cells=observed & (np.where(observed,p[1],0.)>=.5)
    components,_=label(open_cells,CROSS)
    component=components[start[1],start[0]]
    reachable=(components==component) if component else np.zeros_like(observed)
    reference=float(np.linalg.norm(goal-start))
    connected=bool(component and components[goal[1],goal[0]]==component)
    progress=quality=0.
    if reachable.any():
        yy,xx=np.nonzero(reachable)
        progress=float(np.clip(1.-np.min(np.hypot(xx-goal[0],yy-goal[1]))/max(reference,1e-6),0.,1.))
    if connected:
        distances={tuple(start):0};queue=deque([tuple(start)])
        while queue:
            x,y=queue.popleft()
            if (x,y)==tuple(goal):break
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                q=(x+dx,y+dy)
                if 0<=q[0]<w and 0<=q[1]<h and reachable[q[1],q[0]] and q not in distances:
                    distances[q]=distances[(x,y)]+1;queue.append(q)
        length=distances[tuple(goal)]
        quality=min(1.,reference/length) if length else 1.
    route=corridor(start,goal,w);denominator=max(1,route.sum())
    return np.array([connected,progress,quality,(observed&route).sum()/denominator,
        (observed&route&~open_cells).sum()/denominator,reachable.mean(),observed[goal[1],goal[0]]],np.float32)


def encode(probabilities, observed, support, start, goal):
    """Observed-only encoder. Hidden probabilities/support may even be NaN."""
    observed=np.asarray(observed,bool);h,w=observed.shape
    p=np.asarray(probabilities,np.float32)
    if (p.shape!=(5,h,w) or not np.isfinite(p[:,observed]).all()
            or (p[:,observed]<0).any() or not np.allclose(p[:,observed].sum(0),1.,atol=1e-5)):
        raise ValueError('Invalid measured semantic inputs')
    result=np.zeros((CHANNELS,h,w),np.float32)
    result[:5,observed]=p[:,observed];result[5]=observed
    result[6,observed]=np.asarray(support)[observed]
    if not np.isfinite(result).all():raise ValueError('Invalid observed support')
    result[7,start[1],start[0]]=1.;result[8,goal[1],goal[0]]=1.
    result[9]=corridor(start,goal,w)
    return result


def query_mask(action, side=SIDE):
    x,y,r=np.asarray(action,int)
    if not 0<=x<side or not 0<=y<side or r not in (1,2):
        raise ValueError('Invalid offline query window')
    yy,xx=np.indices((side,side))
    return (abs(xx-x)<=r)&(abs(yy-y)<=r)


def normalize_actions(actions):
    value=np.asarray(actions,np.float32).copy()
    value[...,:2]=value[...,:2]/(SIDE-1)*2-1
    value[...,2]/=2.
    return value


def query_costs(actions, start):
    """Geometric query displacement only, never a real aircraft energy label."""
    q=np.asarray(actions)[...,:2]
    first=np.linalg.norm(q[:,0]-start,axis=-1)
    second=first+np.linalg.norm(q[:,1]-q[:,0],axis=-1)
    return np.stack((first,second),-1).astype(np.float32)/(np.sqrt(2)*(SIDE-1))


def utility(states, before, costs):
    return ((states-before[...,None,None,:])*UTILITY_WEIGHTS).sum(-1)-TRAVEL_WEIGHT*costs


def candidates(observed, start, goal, rng, count=8):
    """Candidate construction uses known masks/endpoints, never hidden labels."""
    yy,xx=np.nonzero(~observed);unknown=np.stack((xx,yy),1)
    if not len(unknown):raise ValueError('No unknown query windows')
    delta=np.asarray(goal)-start
    desired=[start+f*delta for f in (.25,.5,.75,1.)]
    desired.extend((start-.5*delta,start+np.array([-delta[1],delta[0]])*.5))
    selected=[]
    for target in desired:
        order=np.argsort(np.linalg.norm(unknown-np.clip(target,0,SIDE-1),axis=1))
        for i in order:
            q=tuple(unknown[i])
            if q not in selected:selected.append(q);break
    for i in rng.permutation(len(unknown)):
        q=tuple(unknown[i])
        if q not in selected:selected.append(q)
        if len(selected)>=count:break
    if len(selected)<count:raise ValueError('Too few distinct candidate windows')
    rng.shuffle(selected)
    actions=[]
    for q in selected[:count]:
        q=np.array(q);direction=np.sign(goal-q).astype(int)
        second=q if rng.random()<.15 else np.clip(q+2*direction,0,SIDE-1)
        radius=int(rng.integers(1,3))
        actions.append([[int(q[0]),int(q[1]),radius],[int(second[0]),int(second[1]),radius]])
    return np.array(actions,np.int16)


def make_task(scene, rng, *, count=8):
    p,valid,support=scene['probabilities'],scene['valid'],scene['support']
    if valid.shape!=(SIDE,SIDE):raise ValueError('Abstract prototype expects 16x16 measured contexts')
    # Only the initially observed labels can determine the virtual vehicle anchor.
    for _ in range(16):
        known=outward_mask(valid,rng,8.,16.)
        open_points=np.argwhere(known&(p[1]>=.5))[:,::-1]
        if len(open_points):break
    else:raise ValueError('No observed open-ground semantic anchor')
    distances=np.linalg.norm(open_points-(SIDE-1)/2,axis=1)
    near=np.argsort(distances)[:min(5,len(distances))]
    start=open_points[int(rng.choice(near))]
    yy,xx=np.indices(valid.shape);reference=np.hypot(xx-start[0],yy-start[1])
    goal_cells=np.argwhere((~known)&(reference>=4)&(reference<=11))[:,::-1]
    if not len(goal_cells):raise ValueError('No eligible virtual goal outside observed region')
    goal=goal_cells[int(rng.integers(len(goal_cells)))]
    actions=candidates(known,start,goal,rng,count)
    before=evidence_state(p,known,start,goal)
    targets=[];next_masks=[];optimistic=[]
    for branch in actions:
        observed=known.copy();hypothetical=known.copy();optimistic_p=np.where(known[None],p,0.).copy()
        branch_targets=[];branch_masks=[];branch_upper=[]
        for action in branch:
            window=query_mask(action)
            observed |= window&valid  # Only real recorded support becomes evidence.
            branch_targets.append(evidence_state(p,observed,start,goal));branch_masks.append(observed.copy())
            newly_assumed=window&~hypothetical
            hypothetical |= window
            optimistic_p[:,newly_assumed]=0.;optimistic_p[1,newly_assumed]=1.
            branch_upper.append(evidence_state(optimistic_p,hypothetical,start,goal))
        targets.append(branch_targets);next_masks.append(branch_masks);optimistic.append(branch_upper)
    return dict(maps=encode(p,known,support,start,goal),known=known,start=start,goal=goal,
        task=np.concatenate((start,goal)).astype(np.float32)/(SIDE-1)*2-1,
        actions=normalize_actions(actions),actions_grid=actions,state=before,
        target=np.array(targets,np.float32),after_masks=np.array(next_masks,bool),
        optimistic=np.array(optimistic,np.float32),costs=query_costs(actions,start))


def load_sources(path):
    original,manifest=load_regional_manifest(path,cell_size=8.,minimum_support=.125)
    lookup={s['sequence']:s for group in original.values() for s in group}
    result={split:[] for split in ('train','val','calibration','test')}
    for record in manifest['records']:
        group=record['group']
        if group not in SPLITS:raise ValueError('Unreviewed recording group')
        scene=lookup[record['sequence']]
        scene={**scene,'group':group,'scenario':record['scenario']}
        result[SPLITS[group]].append(scene)
    return result,manifest


def prepare(path, output, *, tasks_per_scene=4, candidates_count=8, seed=20261114):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    scenes,source=load_sources(path);records=[];audit={};all_tasks={};dropped=[]
    for split,group in scenes.items():
        tasks=[]
        for i,scene in enumerate(group):
            for repetition in range(tasks_per_scene):
                rng=np.random.default_rng(np.random.SeedSequence([seed,SPLITS_ORDER.index(split),i,repetition]))
                try:task=make_task(scene,rng,count=candidates_count)
                except ValueError as exc:
                    dropped.append(dict(split=split,sequence=scene['sequence'],task=repetition,reason=str(exc)));continue
                task['scene']=i;tasks.append(task)
        if not tasks:raise ValueError(f'No supported abstract tasks in {split}')
        arrays={key:np.stack([x[key] for x in tasks]) for key in tasks[0]}
        destination=output/(split+'.npz');np.savez_compressed(destination,**arrays)
        all_tasks[split]=arrays
        score=utility(arrays['target'],arrays['state'],arrays['costs'])
        audit[split]=dict(scenes=len(group),tasks=len(tasks),candidate_pairs=int(np.prod(score.shape)),
            one_step_new_connections=int((arrays['target'][:,:,0,0]>arrays['state'][:,None,0]).sum()),
            informative_choice_tasks=int((np.ptp(score[:,:,0],axis=1)>.05).sum()),
            groups=sorted({s['group'] for s in group}))
        records.append(dict(split=split,file=destination.name,sha256=sha256(destination),
            scenes=[dict(sequence=s['sequence'],source_sha256=s['source_sha256'],group=s['group'],scenario=s['scenario']) for s in group]))
    document=dict(format=FORMAT,source_manifest=str(Path(path).resolve()),source_manifest_sha256=sha256(path),
        states=list(STATE_NAMES),channels=CHANNELS,side=SIDE,seed=seed,tasks_per_scene=tasks_per_scene,
        candidates=candidates_count,horizon=2,utility_weights=UTILITY_WEIGHTS.tolist(),travel_weight=TRAVEL_WEIGHT,
        split_groups=SPLITS,records=records,audit=audit,dropped=dropped,
        simulation_used=False,offline_mask_augmentation=True,real_aerial_actions=False,
        graph_is_vehicle_traversability=False,goal_endpoints='virtual within-scan tasks, sampled without hidden semantics',
        action_domain='offline windows revealing already measured GOOSE labels; no aerial FOV/occlusion assumption',
        evaluation_scope='reused public scenes; new grouped split and fresh weights; not an untouched blind navigation benchmark')
    atomic_json(output/'manifest.json',document)
    return scenes,all_tasks,document


SPLITS_ORDER=('train','val','calibration','test')


class AbstractTasks(Dataset):
    def __init__(self,scenes,arrays,*,augment=False,seed=0,include_future=True):
        self.scenes,self.arrays,self.augment,self.seed=scenes,arrays,augment,seed
        self.include_future=include_future
        self.epoch=0

    def __len__(self):return len(self.arrays['state'])

    def __getitem__(self,index):
        a={key:value[index].copy() for key,value in self.arrays.items()}
        scene=self.scenes[int(a['scene'])];mask=a['after_masks']
        future=np.empty((0,),np.float32)
        if self.include_future:
            future=np.zeros((*mask.shape[:2],CHANNELS,SIDE,SIDE),np.float32)
            future[:,:,:5]=np.where(mask[:,:,None],scene['probabilities'][None,None],0.)
            future[:,:,5]=mask;future[:,:,6]=np.where(mask,scene['support'][None,None],0.)
            future[:,:,7:]=a['maps'][None,None,7:]
        if self.augment:
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
            k=int(rng.integers(4));flip=rng.random()<.5
            a['maps']=np.rot90(a['maps'],k,axes=(-2,-1)).copy()
            if self.include_future:future=np.rot90(future,k,axes=(-2,-1)).copy()
            for _ in range(k):
                xy=a['actions'][...,:2].copy();a['actions'][...,0]=xy[...,1];a['actions'][...,1]=-xy[...,0]
                xy=a['task'].reshape(2,2).copy();a['task']=np.stack((xy[:,1],-xy[:,0]),1).reshape(4)
            if flip:
                a['maps']=np.flip(a['maps'],-1).copy()
                if self.include_future:future=np.flip(future,-1).copy()
                a['actions'][...,0]*=-1;a['task'][[0,2]]*=-1
        return dict(maps=a['maps'],task=a['task'],state=a['state'],actions=a['actions'],
                    target=a['target'],future_maps=future,costs=a['costs'],index=np.int64(index))


def baseline_features(arrays, step):
    n,k=arrays['actions'].shape[:2]
    repeat=lambda x:np.broadcast_to(x[:,None],(n,k,x.shape[-1]))
    return np.concatenate((np.ones((n,k,1)),repeat(arrays['state']),repeat(arrays['task']),
        arrays['actions'][:,:,step],arrays['costs'][:,:,step,None],
        arrays['optimistic'][:,:,step]-arrays['state'][:,None]),-1).astype(np.float64)


def load_prepared(directory, source_manifest):
    directory=Path(directory);doc=json.loads((directory/'manifest.json').read_text())
    if (doc.get('format')!=FORMAT or doc.get('source_manifest_sha256')!=sha256(source_manifest)
        or doc.get('split_groups')!=SPLITS or doc.get('real_aerial_actions') is not False
        or doc.get('graph_is_vehicle_traversability') is not False):
        raise ValueError('Abstract dataset provenance/scope mismatch')
    scenes,_=load_sources(source_manifest);arrays={}
    for record in doc['records']:
        path=directory/record['file'];split=record['split']
        if sha256(path)!=record['sha256']:raise ValueError('Abstract task checksum mismatch')
        if [x['sequence'] for x in record['scenes']]!=[x['sequence'] for x in scenes[split]]:
            raise ValueError('Scene indexing changed')
        with np.load(path,allow_pickle=False) as data:arrays[split]={key:data[key] for key in data.files}
    if set(arrays)!=set(SPLITS_ORDER):raise ValueError('Four separate splits required')
    return scenes,arrays,doc
