"""Reproducible hierarchical-navigation pilot with the real frozen v9 MAPPO closed loop.

Run with the existing teamedplanning-2d Python. Only the new return regressor
is fitted. Candidate experiments branch from identical complete snapshots;
after the injected decision they continue with the original vtemp manager.
This is a finite-horizon decision experiment, not a full-mission benchmark.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import inspect
import importlib
import importlib.abc
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'

SOURCE=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE.parent))
import numpy as np
from world_model.hierarchical_sources import (
    atomic_json,
    file_sha256 as digest,
    runtime_source,
    snapshot_sources,
    source_paths,
    source_hashes as hierarchical_source_hashes,
)
from world_model.navigation_evidence import EvidenceGrid, FEATURE_NAMES, ReturnModel, candidates, features, select

CHECKPOINT_SHA='7ac3aedcc3876902773b2241f60e839eb3e3b5630ddeb3c8be660421f2c24198'
V9_RUN=Path('artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146')
V9_FAST_REWARD_CONTRACT='uav_remaining_route_mapping_v9'
PROTOCOL='vtemp2_frozen_v9_single_decision_pilot_v1'
_RUNTIME=None


def _is_navigation_extension(package, name):
    if name.startswith(('vtemp4_', 'train_vtemp4', 'evaluate_vtemp4')):
        return package in {'ground', 'simulator'}
    if package == 'ground':
        return name in {'ground_policy', 'survey_fleet_clearance'}
    if package == 'simulator':
        return name in {'ground_evidence', 'ground_relay', 'survey_fleet_drive',
                        'survey_fleet_metrics', 'survey_fleet_observation', 'survey_fleet_phase',
                        'survey_fleet_runtime', 'survey_km_maps', 'survey_pretrain_env',
                        'survey_pretrain_maps', 'survey_pretrain_policy', 'train_survey_fleet',
                        'train_survey_pretrain', 'evaluate_survey_fleet'}
    return False


class _NavigationExtensionFinder(importlib.abc.MetaPathFinder):
    """Resolve navigation experiment extensions outside the archived runtime."""
    def __init__(self, extension_root, sources):
        self.extension_root=extension_root
        self.sources=sources

    def find_spec(self, fullname, path=None, target=None):
        parts=fullname.split('.')
        if len(parts)!=2 or not _is_navigation_extension(parts[0],parts[1]):
            return None
        if (self.sources/parts[0]/(parts[1]+'.py')).is_file():
            return None
        candidate=self.extension_root/parts[0]/(parts[1]+'.py')
        if candidate.is_file():
            return importlib.util.spec_from_file_location(fullname,candidate)
        return None


def write_json(path, data):
    atomic_json(path, data)


def v9_archive_config(repo):
    return json.loads((Path(repo).resolve()/V9_RUN/'config.json').read_text(encoding='utf-8'))


def runtime(repo, *, random_init=False):
    global _RUNTIME
    if _RUNTIME is not None and not random_init:
        return _RUNTIME
    run=Path(repo).resolve()/V9_RUN
    sources=run/'runtime_sources'
    extension_root=SOURCE.parent
    for name,module in list(sys.modules.items()):
        parts=name.split('.')
        if parts[0] in {'ground','drone','common','simulator'}:
            location=getattr(module,'__file__',None)
            if location and not Path(location).resolve().is_relative_to(sources):
                # Navigation experiments live beside the frozen core. Allow
                # only their own modules and the two lightweight package roots;
                # all actual v9 core modules must still come from the archive.
                extension=(parts[0] in {'ground','simulator'} and
                           (len(parts)==1 or (len(parts)==2 and
                            _is_navigation_extension(parts[0],parts[1]))) and
                           Path(location).resolve().is_relative_to(extension_root/parts[0]))
                if extension:
                    continue
                raise RuntimeError(f'Mutable runtime already imported: {name}')
    if not random_init and digest(run/'best.pt')!=CHECKPOINT_SHA:
        raise RuntimeError('Frozen v9 checkpoint digest differs')
    sys.path.insert(0,str(sources))
    # A vtemp4 import may have loaded these package objects already. Restrict
    # ordinary package lookup to the archive; the finder permits only the
    # navigation extensions to come from the experiment source tree.
    for name in ('ground', 'simulator'):
        package=importlib.import_module(name)
        package.__path__[:]=[str(sources/name)]
    sys.meta_path[:]=[finder for finder in sys.meta_path if not isinstance(finder,_NavigationExtensionFinder)]
    sys.meta_path.insert(0,_NavigationExtensionFinder(extension_root,sources))
    import torch
    from simulator import mappo_experiment as me
    from simulator import train_mappo as tm
    torch.set_num_threads(1)
    config=v9_archive_config(repo)
    if random_init:
        # Reuse only the immutable architecture and simulator. Construct every
        # learned parameter and value-normalization statistic from scratch;
        # even loading an archived teacher is forbidden for fresh training.
        if (config['policy_architecture'] != 'gat_relay_attention_v8'
                or config.get('actor_input_version', 2) != 2):
            raise ValueError('Fresh graph fast policy requires the pinned relay architecture')
        from drone.relay_attention_mappo import RelayActorCritic
        model=RelayActorCritic(global_features=tm.GLOBAL_FEATURES, input_version=2,
                              relay_attention=config['relay_attention'])
    else:
        model=tm.ActorCritic.from_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True)['model'])
    model.eval().requires_grad_(False)
    options={k:config[k] for k in inspect.signature(me.Experiment).parameters
             if k in config and k not in ('seed','model')}
    options.update(world_size=1000.,subgoal_radius=200.,subgoal_arrival=25.,
                   num_drones=3,stop_on_ground_arrival=False,
                   exploration_weight=tm.fixed_schedule(config,config['episodes']-1)['eta'])
    result=(me,torch,model,options)
    if not random_init:
        _RUNTIME=result
    return result


def scene(family,variant):
    rng=np.random.default_rng(1511000+100*family+variant)
    start=np.array([300.,450.+rng.uniform(-25,25)])
    goal=start+np.array([350.,rng.uniform(-30,30)])
    circles=[]
    if family==1:
        for offset in [(55.,0.),(85.,18.),(110.,-22.),(150.,8.)]:
            point=start+np.asarray(offset)+rng.uniform(-5,5,2)
            circles.append((point.tolist(),float(rng.uniform(3,6))))
    elif family==2:
        gap=(1 if variant%2 else -1)*float(rng.uniform(28,45))
        wall_x=float(rng.uniform(62,78))
        for y in np.arange(-84.,85.,8.):
            if abs(y-gap)>13:
                circles.append(((start+np.array([wall_x,y])).tolist(),4.5))
    return dict(family=['open','trees','wall_gap'][family],variant=variant,
                start=start.tolist(),goal=goal.tolist(),circles=circles,
                scene_seed=1511000+100*family+variant,
                split='train' if variant<4 else 'validation' if variant<6 else 'test')


def received_belief(env):
    """Explicit experimental receiver extension, using arrived audit pixels.

    Frozen UAV actors never consume these audit fields. Production use requires
    an equivalent real occupancy/coverage packet; this is not a claim that the
    archived ROS target selector already has this channel.
    """
    received=env.ledger
    cells=np.zeros(received.seen.shape,np.int8)
    cells[received.seen & (received.log_odds<0)]=-1
    cells[received.seen & (received.log_odds>0)]=1
    obs=env.obs[env.ground]
    p=np.asarray(obs['self']['position'][:2],float)
    local=np.asarray(obs['occupancy_map'])
    resolution=float(obs['map_resolution'])
    yy,xx=np.indices(local.shape)
    world=p+np.stack([xx+.5-local.shape[1]/2,yy+.5-local.shape[0]/2],-1)*resolution
    ij=np.floor(world/received.resolution).astype(int)
    valid=((ij>=0).all(-1)&(ij[...,0]<cells.shape[1])&(ij[...,1]<cells.shape[0]))
    flat=ij[...,1]*cells.shape[1]+ij[...,0]
    # Ground observations override older evidence; any occupied subpixel blocks
    # its coarse receiver cell. No hidden occupancy outside the local sensor.
    for key in np.unique(flat[valid]):
        cells.flat[key]=1 if np.any(local[valid & (flat==key)]!=0) else -1
    graph=env.ground_controller.unified_graph
    nodes=[] if graph is None else [graph.get_node_position(n)[:2] for n in graph.graph.nodes]
    times=list(env.ground_controller.last_fusion_per_drone.values())
    now=float(env.env.time)
    previous=np.asarray(env.trace[-2]['ground_position'][:2]) if len(env.trace)>1 and 'ground_position' in env.trace[-2] else p
    return dict(position=p,goal=env.goal.copy(),grid=EvidenceGrid(cells,received.resolution),
                graph_positions=nodes,graph_edges=0 if graph is None else graph.graph.number_of_edges(),
                recent_receipts=sum(now-t<=10. for t in times)/3,
                receipt_age=float(np.mean([now-t for t in times])) if times else 60.,
                speed=float(np.linalg.norm(p-previous)))


def force_candidate(env,target):
    p=np.asarray(env.obs[env.ground]['self']['position'][:2],float)
    manager=env.subgoals
    manager.target=np.asarray(target,float).copy()
    manager.origin=p.copy(); manager.final=False; manager.snapped=False
    manager.selected+=1
    manager._best=float(np.linalg.norm(manager.target-p))
    manager._nodes=manager._count(env.ground_controller.unified_graph,p,manager.target)
    manager._since=manager._approached=env.env.time
    env.ground_controller._drive_to(tuple(target))
    env.ground_action=env._ground_action()
    if not np.allclose(env.ground_action['broadcast']['target'],target):
        raise AssertionError('Target did not enter the real ground broadcast')


def clone_experiment(base,model,me):
    env=copy.deepcopy(base,{id(model):model})
    # deepcopy does not rebind Python function closures. The frozen runtime's
    # bridge callback captures its Experiment; bind it to THIS branch's ledger.
    fusion=env.ground_controller.graph_fusion
    if getattr(fusion,'bridge_distance',0.):
        fusion.bridge_check=lambda a,b: env.ledger.segment_clear(a,b,me.BRIDGE_CLEARANCE_M)
    return env


def branch_rollout(base,model,target,seed,warmup,horizon,me,torch):
    torch.manual_seed(seed)
    env=clone_experiment(base,model,me)
    force_candidate(env,target)
    previous=float(np.linalg.norm(np.asarray(env.obs[env.ground]['self']['position'][:2])-env.goal))
    initial=previous
    total=0.; collisions=0; uav_reward=0.; termination='horizon'; steps=0
    target_heard=False
    ground_trace=[]
    try:
        for offset in range(horizon):
            try:
                _,reward,_=env.step(warmup+offset)
            except me.SafetyTermination:
                total-=100.*.995**offset
                termination='cbf_infeasible'
                break
            steps+=1
            p=np.asarray(env.obs[env.ground]['self']['position'][:2],float)
            distance=float(np.linalg.norm(p-env.goal))
            hit=int(env.trace[-1].get('ground_collisions',0))
            collisions+=hit
            # Same final mission and reward definition for every candidate.
            # A fixed finite horizon is intentional; there is no terminal-value
            # bootstrap or claim of optimal long-horizon detour evaluation.
            r=(previous-distance)/10.-.01-10.*hit
            if env.goal_reached:r+=100.
            total+=.995**offset*r
            previous=distance
            uav_reward+=float(reward)
            ground_trace.append(p.tolist())
            for c in env.controllers:
                task=c.navigator.chain.task()
                if task and np.allclose(task['target'][:2],target): target_heard=True
            if env.goal_reached:
                termination='mission_arrival';break
        return dict(return_value=float(total),progress_m=initial-previous,
                    remaining_m=previous,steps=steps,termination=termination,
                    collisions=collisions,target_heard=target_heard,
                    subgoal_failures=int(env.subgoals.failed),uav_reward_sum=uav_reward,
                    ground_trace=ground_trace)
    finally:
        env.env.close()


def run_case(job):
    case,repo,output,warmup,horizon=job
    name=f"{case['family']}_{case['variant']}"
    destination=Path(output)/'cases'/f'{name}.json'
    me,torch,model,options=runtime(repo)
    torch.manual_seed(case['scene_seed'])
    options=dict(options,start=case['start'],scene=(case['goal'],case['circles'],1))
    start=time.monotonic()
    with (Path(output)/'logs'/f'{name}.log').open('w') as log,contextlib.redirect_stdout(log),torch.inference_mode():
        base=me.Experiment(case['scene_seed'],model,stochastic=True,**options)
        try:
            for frame in range(warmup):base.step(frame)
            belief=received_belief(base)
            qs=candidates(belief['position'],belief['goal'])
            data=[]
            for index,q in enumerate(qs):
                x,status=features(belief,q)
                results=[]
                for repetition in range(2 if case['split']=='test' else 1):
                    results.append(branch_rollout(base,model,q,case['scene_seed']+100000+repetition,
                                                  warmup,horizon,me,torch))
                data.append(dict(candidate=index,target=q.tolist(),features=x.tolist(),status=status,
                                 mean_return=float(np.mean([r['return_value'] for r in results])),rollouts=results))
            result=dict(case=case,decision_position=belief['position'].tolist(),
                        decision_time=float(base.env.time),known_cells=int(np.count_nonzero(belief['grid'].cells)),
                        graph_nodes=len(belief['graph_positions']),candidates=data,
                        wall_seconds=time.monotonic()-start)
            write_json(destination,result)
            return name,case['split'],result['wall_seconds']
        finally:
            base.env.close()


def assess(rows,model):
    decisions=[]
    for row in rows:
        options=row['candidates']
        x=np.asarray([q['features'] for q in options])
        actual=np.asarray([q['mean_return'] for q in options])
        predicted=model.predict(x)
        statuses=[q['status'] for q in options]
        choice=select(predicted,statuses)
        permitted=np.array([s not in ('blocked_within_bounds','outside_bounds') for s in statuses])
        # Candidate 0 is the unpenalized forward 200m winner at these explicit
        # reselection snapshots. Evidence-only chooses the first allowed ring.
        evidence_choice=int(np.flatnonzero(permitted)[0]) if permitted.any() else None
        best=int(np.argmax(actual))
        pairs=[(i,j) for i in range(len(actual)) for j in range(i) if abs(actual[i]-actual[j])>.05]
        accuracy=float(np.mean([(predicted[i]-predicted[j])*(actual[i]-actual[j])>0 for i,j in pairs])) if pairs else None
        decisions.append(dict(case=f"{row['case']['family']}_{row['case']['variant']}",selected=choice,
            geometric_selected=0,evidence_selected=evidence_choice,oracle_selected=best,
            selected_return=None if choice is None else float(actual[choice]),
            geometric_return=float(actual[0]),evidence_return=None if evidence_choice is None else float(actual[evidence_choice]),
            oracle_return=float(actual[best]),regret=None if choice is None else float(actual[best]-actual[choice]),
            pairwise_accuracy=accuracy,predicted_returns=predicted.tolist(),actual_returns=actual.tolist(),statuses=statuses))
    valid=[d for d in decisions if d['selected_return'] is not None]
    return dict(contexts=len(rows),decisions=decisions,
                mean_selected_return=float(np.mean([d['selected_return'] for d in valid])) if valid else None,
                mean_geometric_return=float(np.mean([d['geometric_return'] for d in valid])) if valid else None,
                mean_evidence_return=float(np.mean([d['evidence_return'] for d in valid])) if valid else None,
                mean_oracle_return=float(np.mean([d['oracle_return'] for d in valid])) if valid else None,
                mean_regret=float(np.mean([d['regret'] for d in valid])) if valid else None,
                wins=sum(d['selected_return']>d['geometric_return']+.05 for d in valid),
                losses=sum(d['selected_return']<d['geometric_return']-.05 for d in valid),
                ties=sum(abs(d['selected_return']-d['geometric_return'])<=.05 for d in valid),
                pairwise_accuracy=float(np.mean([d['pairwise_accuracy'] for d in valid if d['pairwise_accuracy'] is not None])) if any(d['pairwise_accuracy'] is not None for d in valid) else None)


def fit_and_report(output):
    rows=[json.loads(p.read_text()) for p in sorted((output/'cases').glob('*.json'))]
    splits={s:[r for r in rows if r['case']['split']==s] for s in ('train','validation','test')}
    train=splits['train']
    x=np.asarray([q['features'] for r in train for q in r['candidates']])
    y=np.asarray([q['mean_return'] for r in train for q in r['candidates']])
    fits=[]
    for alpha in (.1,1.,10.,100.):
        model=ReturnModel(alpha).fit(x,y)
        evaluation=assess(splits['validation'],model)
        fits.append((evaluation['mean_selected_return'],alpha,model,evaluation))
    _,alpha,model,_=max(fits,key=lambda row:row[0])
    model.save(output/'return_model.npz')
    report=dict(protocol=PROTOCOL,checkpoint_sha256=CHECKPOINT_SHA,mappo_retrained=False,
                value_model='standardized_ridge_regression_on_measured_returns',
                selected_regularization=alpha,
                validation_search=[dict(regularization=a,mean_return=s) for s,a,_,_ in fits],
                splits={s:assess(group,model) for s,group in splits.items()},
                contexts=len(rows),candidate_samples=sum(len(r['candidates']) for r in rows),
                physical_rollouts=sum(len(q['rollouts']) for r in rows for q in r['candidates']),
                target_delivery_count=sum(v['target_heard'] for r in rows for q in r['candidates'] for v in q['rollouts']),
                safety_terminations=sum(v['termination']=='cbf_infeasible' for r in rows for q in r['candidates'] for v in q['rollouts']),
                scope=['Single slow decision after identical warmup; original vtemp continuation.',
                       'Five predeclared ring candidates, not all 72 candidates.',
                       'Finite-horizon discounted final-goal progress and execution costs; no terminal-value bootstrap.',
                       'Only arrived evidence audit pixels plus ground local sensor reach the slow evaluator.',
                       'Small synthetic scene-separated pilot; no WildScenes or long-mission success claim.',
                       'Timeout is an execution outcome, never a physical-impossibility label.'])
    write_json(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=SOURCE.parents[3])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--horizon',type=int,default=80)
    parser.add_argument('--frozen',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    repo,output=args.repo.resolve(),args.output.resolve()
    if args.workers<1 or args.horizon<1 or args.warmup<1:
        parser.error('workers, horizon and warmup must be positive')
    if not args.frozen:
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise ValueError('Output already contains files; choose a new output or --resume')
        snapshot=runtime_source(output) if args.resume else output/'runtime_sources/world_model'
        if args.resume:
            frozen_paths = source_paths(snapshot)
            for name, path in source_paths(SOURCE).items():
                if name not in frozen_paths or digest(path)!=digest(frozen_paths[name]):
                    raise ValueError('Source changed; resume would mix different implementations')
        else:
            snapshot_sources(SOURCE,snapshot)
        os.execv(sys.executable,[sys.executable,'-I','-B',str(snapshot/'frozen_navigation.py'),
                 '--repo',str(repo),'--output',str(output),'--workers',str(args.workers),
                 '--warmup',str(args.warmup),'--horizon',str(args.horizon),'--frozen',
                 *(['--resume'] if args.resume else [])])
    for folder in ('cases','logs'):(output/folder).mkdir(parents=True,exist_ok=True)
    run=repo/'artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146'
    source_hashes={str(p.relative_to(run/'runtime_sources')):digest(p) for p in sorted((run/'runtime_sources').rglob('*.py'))}
    manifest=dict(protocol=PROTOCOL,source_package='world_model',checkpoint_sha256=digest(run/'best.pt'),
                  config_sha256=digest(run/'config.json'),frozen_source_hashes=source_hashes,
                  vtemp2_source_hashes=hierarchical_source_hashes(SOURCE),
                  warmup=args.warmup,horizon=args.horizon,features=FEATURE_NAMES,
                  reward='sum gamma^i * ((distance_to_G_before-distance_to_G_after)/10 - .01 - 10*collisions + 100*mission_arrival); CBF failure -100',
                  discount=.995,stochastic=True,mappo_retrained=False,
                  experiment='paired snapshot candidate rollouts; fixed vtemp continuation',
                  received_evidence_contract='Simulation-only slow receiver extension using actually arrived evidence_audit packets. No hidden map or live fleet truth input. ROS deployment needs equivalent packets.')
    if manifest['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('Checkpoint mismatch')
    if args.resume and json.loads((output/'manifest.json').read_text())!=manifest:
        raise ValueError('Resume manifest differs')
    write_json(output/'manifest.json',manifest)
    cases=[scene(family,variant) for family in range(3) for variant in range(8)]
    jobs=[(c,str(repo),str(output),args.warmup,args.horizon) for c in cases
          if not (output/'cases'/f"{c['family']}_{c['variant']}.json").exists()]
    write_json(output/'status.json',dict(status='running',completed=24-len(jobs),total=24))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(run_case,job):job[0] for job in jobs}
        completed=24-len(jobs)
        try:
            for future in as_completed(futures):
                name,split,seconds=future.result()
                completed+=1
                print(json.dumps(dict(completed=completed,total=24,case=name,split=split,seconds=round(seconds,1))),flush=True)
                write_json(output/'status.json',dict(status='running',completed=completed,total=24,last_case=name))
        except BaseException as error:
            write_json(output/'status.json',dict(status='failed',error=repr(error),completed=completed,total=24))
            for future in futures:future.cancel()
            raise
    report=fit_and_report(output)
    if digest(run/'best.pt')!=CHECKPOINT_SHA or any(digest(run/'runtime_sources'/name)!=sha for name,sha in source_hashes.items()):
        raise RuntimeError('Frozen assets changed during validation')
    write_json(output/'status.json',dict(status='complete',completed=24,total=24))
    print(json.dumps({k:v for k,v in report['splits']['test'].items() if k!='decisions'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
