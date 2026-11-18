"""Train/evaluate hierarchical PPO with frozen v9 UAV-MAPPO.

Example (from the repository):
  python -I -B src/teamed-planning/scripts/world_model/train_slow_policy.py --output artifacts/vtemp2-hrl
  python -I -B src/teamed-planning/scripts/world_model/train_slow_policy.py --output artifacts/vtemp2-hrl --resume --updates 120
  python -I -B src/teamed-planning/scripts/world_model/train_slow_policy.py --output artifacts/vtemp2-hrl-eval --evaluate artifacts/vtemp2-hrl/best.pt

The default curriculum reaches 800-1200m start/goal separations in a 2000m
world. Checkpoints contain only the slow policy and optimizer; the fast model
is loaded from the archived, SHA-verified v9 best576 checkpoint.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
SOURCE=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE.parent))
import numpy as np
import torch
from world_model.hierarchical_runtime import HierarchicalEpisode,HierarchyConfig,SLOW_FEATURES,MAX_CANDIDATES,SCHEMA
from world_model.slow_policy import SlowActorCritic,update
from world_model.scene_manifest import make_scene
from world_model.graph_fast_policy import numpy_state as worker_state
from world_model.frozen_navigation import CHECKPOINT_SHA,digest,write_json
from world_model.hierarchical_sources import (source_files, source_hashes as hierarchical_source_hashes,
    snapshot_sources, runtime_source)

FORMAT='vtemp2_hierarchical_ppo_v1'
DEFAULTS=dict(updates=100,episodes_per_update=4,workers=2,rollout_steps=6000,
              evaluation_episodes=4,evaluation_interval=5,seed=15112026,world_size=2000.,
              hidden=64,learning_rate=.0003,epochs=4,gae_lambda=.95,clip=.2,
              entropy_coef=.01,value_coef=.5,target_kl=.02,max_kl=.04,
              hierarchy=asdict(HierarchyConfig()),
              curriculum=[dict(start_update=0,distance=[150.,300.]),
                          dict(start_update=20,distance=[300.,700.]),
                          dict(start_update=60,distance=[800.,1200.])])


def validate_config(config):
    for name in ('updates','episodes_per_update','workers','rollout_steps','evaluation_episodes',
                 'evaluation_interval','hidden','epochs'):
        if type(config[name]) is not int or config[name]<1:raise ValueError(f'{name} must be a positive integer')
    HierarchyConfig(**config['hierarchy'])
    if not 0<config['learning_rate'] or not np.isfinite(config['learning_rate']):raise ValueError('Invalid learning rate')
    if not 0<=config['gae_lambda']<=1 or not 0<config['clip']<1:raise ValueError('Invalid PPO parameters')
    if not 0<config['target_kl']<=config['max_kl']:raise ValueError('Invalid KL limits')
    if config['entropy_coef']<0 or config['value_coef']<0:raise ValueError('Negative PPO loss coefficient')
    stages=config['curriculum']
    if not stages or stages[0]['start_update']!=0:raise ValueError('Curriculum must start at update zero')
    previous=-1
    for stage in stages:
        start=stage['start_update'];low,high=stage['distance']
        if type(start) is not int or start<=previous or not 0<low<=high<=config['world_size']-400:
            raise ValueError('Invalid curriculum stage or world extent')
        previous=start


def stage_for(config,index):
    return [s for s in config['curriculum'] if s['start_update']<=index][-1]


def frozen_provenance(repo):
    root=Path(repo)/'artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146'
    actual=digest(root/'best.pt')
    if actual!=CHECKPOINT_SHA:raise ValueError('Frozen v9 checkpoint hash differs')
    sources={str(p.relative_to(root/'runtime_sources')):digest(p) for p in sorted((root/'runtime_sources').rglob('*.py'))}
    return dict(checkpoint_sha256=actual,config_sha256=digest(root/'config.json'),sources=sources)


def contract(config,provenance):
    return dict(format=FORMAT,schema=SCHEMA,features=SLOW_FEATURES,max_candidates=MAX_CANDIDATES,
                hidden=config['hidden'],fast=provenance,
                hierarchy=config['hierarchy'],world_size=config['world_size'])


def load_checkpoint(path,expected):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('contract')!=expected:raise ValueError('Checkpoint observation, reward, dynamics or frozen-policy contract differs')
    return saved


def atomic_checkpoint(path,data):
    path=Path(path);temporary=path.with_suffix('.tmp')
    torch.save(data,temporary);temporary.replace(path)


def collect(job):
    repo,case,config,state,mode,output,label=job
    torch.set_num_threads(1)
    slow=SlowActorCritic(len(SLOW_FEATURES),config['hidden'])
    slow.load_state_dict({key:torch.from_numpy(value) for key,value in state.items()});slow.eval()
    generator=torch.Generator().manual_seed(case['seed']+config['seed']+71)
    log_path=Path(output)/'logs'/f'{label}.log'
    transitions=[]
    with log_path.open('w') as log,contextlib.redirect_stdout(log):
        env=HierarchicalEpisode(repo,case,HierarchyConfig(**config['hierarchy']),
                                fast_seed=case['seed']+config['seed'],rollout_limit=config['rollout_steps'])
        try:
            while not env.finished:
                observation=env.observation()
                action,logp,value=slow.act(observation,generator,deterministic=mode!='train')
                if mode=='geometry':
                    valid=np.flatnonzero(observation['mask'])
                    drive=[i for i in valid if observation['kinds'][i]!='hold']
                    pool=drive or list(valid)
                    p=env.belief()['position'];goal=env.goal
                    action=min(pool,key=lambda i:(round(float(np.linalg.norm(observation['targets'][i]-p)+np.linalg.norm(observation['targets'][i]-goal)),3),float(np.linalg.norm(observation['targets'][i]-goal))))
                reward,duration,terminated,truncated,record=env.execute(observation,action)
                next_value=0. if terminated else slow.value(env.observation())
                if mode=='train':
                    transitions.append(dict(features=observation['features'].copy(),mask=observation['mask'].copy(),
                        present=observation['present'].copy(),action=action,logp=logp,value=value,
                        reward=reward,duration=duration,next_value=next_value,terminated=terminated,truncated=truncated))
            summary=env.summary();summary['mode']=mode
            summary['case']=case
            write_json(Path(output)/'episodes'/f'{label}.json',summary)
            if transitions:
                np.savez_compressed(Path(output)/'rollouts'/f'{label}.npz',
                    **{key:np.asarray([r[key] for r in transitions]) for key in transitions[0]})
            # Avoid returning detailed traces through multiprocessing IPC.
            compact={k:v for k,v in summary.items() if k not in ('trajectory','decisions','case','config')}
            return transitions,compact
        finally:env.close()


def summaries(results):
    records=[r[1] for r in results]
    return dict(episodes=len(records),success_rate=float(np.mean([r['goal_reached'] for r in records])),
                mean_reward=float(np.mean([r['reward'] for r in records])),
                mean_discounted_reward=float(np.mean([r['discounted_reward'] for r in records])),
                mean_remaining_m=float(np.mean([r['remaining_m'] for r in records])),
                fast_steps=sum(r['steps'] for r in records),slow_decisions=sum(r['slow_decisions'] for r in records),
                safety_failures=sum(r['termination']=='safety_failure' for r in records),
                truncations=sum(r['truncated'] for r in records),collisions=sum(r['collisions'] for r in records),
                records=records)


def prepare_snapshot(args,config,repo,output):
    if args.frozen:return
    snapshot=runtime_source(output) if args.resume else output/'runtime_sources/world_model'
    if args.resume:
        if not snapshot.is_dir():raise ValueError('Resume requires this run\'s source snapshot')
        for p in source_files(SOURCE):
            if not (snapshot/p.name).exists() or digest(p)!=digest(snapshot/p.name):
                raise ValueError('Source changed; resume from the archived entry point or start a new run')
    else:
        if output.exists() and any(output.iterdir()):raise ValueError('Output exists; use a new directory or --resume')
        snapshot_sources(SOURCE,snapshot)
    request=output/'requested_config.json';write_json(request,config)
    command=[sys.executable,'-I','-B','-u',str(snapshot/'train_slow_policy.py'),'--repo',str(repo),
             '--output',str(output),'--config',str(request),'--frozen']
    if args.resume:command.append('--resume')
    if args.evaluate:command+=['--evaluate',str(args.evaluate.resolve())]
    if args.baseline:command.append('--baseline')
    os.execv(sys.executable,command)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo',type=Path,default=SOURCE.parents[3])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    for name in ('updates','episodes-per-update','rollout-steps','option-steps','workers','eval-episodes','seed'):
        parser.add_argument('--'+name,type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--evaluate',type=Path)
    parser.add_argument('--baseline',action='store_true',help='Evaluate a geometry selector with the same evidence and macro executor')
    parser.add_argument('--frozen',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.resume and args.evaluate:parser.error('Resume and evaluate are separate operations')
    if args.baseline and not args.evaluate:parser.error('--baseline requires --evaluate')
    config=json.loads(json.dumps(DEFAULTS))
    if args.resume and not args.config:
        config=json.loads((args.output/'config.json').read_text())
    if args.evaluate and not args.config:
        config=torch.load(args.evaluate,map_location='cpu',weights_only=True)['config']
    if args.config:
        supplied=json.loads(args.config.read_text())
        unknown=set(supplied)-set(config)
        if unknown:raise ValueError(f'Unknown config keys: {sorted(unknown)}')
        config.update({k:v for k,v in supplied.items() if k!='hierarchy'})
        config['hierarchy'].update(supplied.get('hierarchy',{}))
    for argument,key in [('updates','updates'),('episodes_per_update','episodes_per_update'),
                         ('rollout_steps','rollout_steps'),('workers','workers'),
                         ('eval_episodes','evaluation_episodes'),('seed','seed')]:
        value=getattr(args,argument)
        if value is not None:config[key]=value
    if args.option_steps is not None:config['hierarchy']['option_steps']=args.option_steps
    validate_config(config)
    repo,output=args.repo.resolve(),args.output.resolve()
    prepare_snapshot(args,config,repo,output)
    for directory in ('logs','episodes','rollouts','updates'):(output/directory).mkdir(parents=True,exist_ok=True)
    provenance=frozen_provenance(repo)
    expected=contract(config,provenance)
    hashes=hierarchical_source_hashes(SOURCE)
    torch.set_num_threads(1);torch.manual_seed(config['seed'])
    model=SlowActorCritic(len(SLOW_FEATURES),config['hidden'])
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    beginning=0;best_quality=(-1.,-float('inf'))
    if args.resume or args.evaluate:
        saved=load_checkpoint(output/'latest.pt' if args.resume else args.evaluate,expected)
        model.load_state_dict(saved['model'])
        if args.resume:
            previous=saved['config']
            for key in config:
                if key not in ('updates','workers') and config[key]!=previous[key]:raise ValueError(f'Resume changes {key}')
            if saved['source_sha256']!=hashes:raise ValueError('Resume source hashes differ')
            optimizer.load_state_dict(saved['optimizer']);beginning=saved['completed_updates']
            best_quality=tuple(saved['best_quality']);torch.set_rng_state(saved['torch_rng'])
    write_json(output/'config.json',config)
    write_json(output/'manifest.json',dict(contract=expected,source_package='world_model',source_sha256=hashes,
        fast_policy='frozen v9 best576; no fast-policy optimizer',algorithm='masked discrete PPO with SMDP GAE',
        evidence='Slow-only experimental receiver uses actually arrived audit occupancy and own local observations; real deployment requires equivalent evidence packets.',
        reward='Final-goal potential shaping gamma*Phi(next)-Phi(now), Phi=-distance/scale; time/collision cost; mission success/safety/budget terminals.',
        truncation='Collector limit bootstraps; mission budget and actual terminal events do not.',
        gamma_unit='fast control attempt',lambda_unit='slow decision',
        evaluation_checkpoint=None if args.evaluate is None else str(args.evaluate.resolve())))
    started=time.monotonic()
    write_json(output/'status.json',dict(status='running',phase='evaluation' if args.evaluate else 'training',completed_updates=beginning))
    try:
        with ProcessPoolExecutor(max_workers=config['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            if args.evaluate:
                index=max(0,saved['completed_updates']-1)
                stage=stage_for(config,index)
                cases=[make_scene(4100000+config['seed']+i,stage['distance'],config['world_size']) for i in range(config['evaluation_episodes'])]
                mode='geometry' if args.baseline else 'evaluation'
                jobs=[(str(repo),case,config,worker_state(model),mode,str(output),f'eval_{i:03d}') for i,case in enumerate(cases)]
                result=summaries(list(pool.map(collect,jobs)))
                write_json(output/'evaluation.json',result)
                print(json.dumps(dict(phase='evaluation',**{k:v for k,v in result.items() if k!='records'})),flush=True)
            else:
                if beginning>=config['updates']:raise ValueError('No additional updates requested')
                for index in range(beginning,config['updates']):
                    stage=stage_for(config,index)
                    cases=[make_scene(2100000+config['seed']+index*config['episodes_per_update']+i,
                                      stage['distance'],config['world_size']) for i in range(config['episodes_per_update'])]
                    jobs=[(str(repo),case,config,worker_state(model),'train',str(output),f'u{index+1:04d}_train_{i:03d}') for i,case in enumerate(cases)]
                    results=list(pool.map(collect,jobs))
                    before={k:v.detach().clone() for k,v in model.state_dict().items()}
                    model.train()
                    metrics=update(model,optimizer,[r[0] for r in results],gamma=config['hierarchy']['gamma'],
                        gae_lambda=config['gae_lambda'],epochs=config['epochs'],clip=config['clip'],
                        entropy_coef=config['entropy_coef'],value_coef=config['value_coef'],
                        target_kl=config['target_kl'],max_kl=config['max_kl'])
                    metrics['parameter_delta_l2']=float(torch.sqrt(sum((v-before[k]).square().sum() for k,v in model.state_dict().items())))
                    training=summaries(results);validation=None
                    if (index+1)%config['evaluation_interval']==0 or index+1==config['updates']:
                        cases=[make_scene(3100000+config['seed']+i,stage['distance'],config['world_size']) for i in range(config['evaluation_episodes'])]
                        jobs=[(str(repo),case,config,worker_state(model),'validation',str(output),f'u{index+1:04d}_val_{i:03d}') for i,case in enumerate(cases)]
                        validation=summaries(list(pool.map(collect,jobs)))
                    row=dict(update=index+1,stage=stage,ppo=metrics,training=training,validation=validation)
                    write_json(output/'updates'/f'{index+1:04d}.json',row)
                    quality=None if validation is None else (validation['success_rate'],validation['mean_discounted_reward'])
                    improved=quality is not None and quality>best_quality
                    if improved:best_quality=quality
                    saved=dict(contract=expected,config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),
                               completed_updates=index+1,best_quality=list(best_quality),torch_rng=torch.get_rng_state(),source_sha256=hashes)
                    atomic_checkpoint(output/'latest.pt',saved)
                    if improved:atomic_checkpoint(output/'best.pt',saved)
                    write_json(output/'status.json',dict(status='running',phase='training',completed_updates=index+1,
                                                        total_updates=config['updates'],elapsed_seconds=time.monotonic()-started))
                    print(json.dumps(dict(update=index+1,slow_decisions=training['slow_decisions'],fast_steps=training['fast_steps'],
                        ppo_accepted=metrics['accepted_epochs'],parameter_delta=metrics['parameter_delta_l2'],
                        train_success=training['success_rate'],validation_success=None if validation is None else validation['success_rate'])),flush=True)
        if frozen_provenance(repo)!=provenance:raise RuntimeError('Frozen assets changed during training')
        write_json(output/'status.json',dict(status='complete',phase='evaluation' if args.evaluate else 'training',
                    completed_updates=beginning if args.evaluate else config['updates'],elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        write_json(output/'status.json',dict(status='failed',error=repr(error)))
        raise


if __name__=='__main__':main()
