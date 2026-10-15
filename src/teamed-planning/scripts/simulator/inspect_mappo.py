"""Read-only checkpoint diagnostics on validation scenes; never train/resume."""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from drone.mappo import ActorCritic
from simulator.mappo_experiment import Experiment
from simulator.train_mappo import curriculum


def inspect(path,seed,stochastic,allow_unseen_peers=False,communication_mode=None,safety_filter=None):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=ActorCritic()
    model.load_state_dict(saved['model'])
    model.eval()
    config=saved['config']
    torch.manual_seed(12345+seed)
    choices=[]
    likelihood_errors=[]
    with contextlib.redirect_stdout(io.StringIO()):
        env=Experiment(seed,model,stochastic=stochastic,
            curriculum=curriculum(config,config['episodes']-1),
            exploration_candidates=config['exploration_candidates'],
            communication_mode=communication_mode or config.get('communication_mode','direct'),
            safety_filter=safety_filter or config.get('safety_filter','legacy'),
            moving_ground=config.get('moving_ground',False),reward_mode=config.get('reward_mode','legacy'),
            ground_speed=config.get('ground_speed',1.),path_reward_scale=config.get('path_reward_scale',100.))
        for controller in env.controllers:
            navigator=controller.navigator
            if allow_unseen_peers:
                # Diagnostic ablation only, NOT a production safety fix.
                # Isolate the full-fleet-visibility gate in direct upload mode.
                navigator._fleet_total=1
            original=navigator.encode
            def capture(position,observation,nav=navigator,encode=original):
                result=encode(position,observation)
                obs=result[3]
                tasks=[v for v in obs['ground_robots'].values()
                       if v.get('position') is not None and v.get('target') is not None]
                chain_mode=nav.communication_mode=='chain'
                if not (nav.chain.task() if chain_mode else tasks):
                    reason='missing_task'
                elif (bool(nav.chain.missing(position,nav.comm_range)) if chain_mode
                      else len(obs['nearby_drones'])<nav._fleet_total-1):
                    reason='missing_neighbor'
                elif not chain_mode and np.linalg.norm(np.asarray(position)-np.r_[tasks[0]['position'][:2],0.])>nav.comm_range:
                    reason='outside_ground_range'
                else:
                    reason='geometric_filter'
                nav._inspection_reason=reason
                return result
            navigator.encode=capture
        for frame in range(config['steps']):
            previous=np.array([env.obs[r]['self']['position'].copy() for r in env.ids])
            env.step(frame)
            current=np.array([env.obs[r]['self']['position'].copy() for r in env.ids])
            for i,c in enumerate(env.controllers):
                x,mask,action,old_logp=c.navigator.last_sample
                with torch.no_grad():
                    dist=model.distribution(torch.from_numpy(x),torch.from_numpy(mask))
                likelihood_errors.append(abs(float(dist.log_prob(torch.tensor(action)))-old_logp))
                requested=float(x[action,2]*100)
                actual=float(np.linalg.norm(current[i]-previous[i]))
                choices.append(dict(frame=frame,drone=i,action=int(action),valid_actions=int(mask.sum()),
                    hold=bool(action==0),forced=bool(mask.sum()==1),
                    missing_neighbor_forced=bool(mask.sum()==1 and c.navigator._inspection_reason=='missing_neighbor'),
                    geometric_forced=bool(mask.sum()==1 and c.navigator._inspection_reason=='geometric_filter'),
                    near_waypoint=bool(action!=0 and requested<1.),
                    voluntary_hold=bool(action==0 and mask.sum()>1),
                    moving=bool(actual>.1),requested=requested,actual=actual,
                    max_probability=float(dist.probs.max()),hold_probability=float(dist.probs[0]),
                    entropy=float(dist.entropy()),
                    normalized_entropy=float(dist.entropy()/max(np.log(mask.sum()),1e-8)),
                    selected_confidence=float(x[action,6]),
                    selected_local_node_fraction=float(x[action,7])))
        summary=env.summary()
        env.env.close()
    metrics={key:float(np.mean([r[key] for r in choices])) for key in
        ('hold','forced','near_waypoint','voluntary_hold','moving','valid_actions',
         'max_probability','hold_probability','entropy','normalized_entropy',
         'missing_neighbor_forced','geometric_forced')}
    metrics['max_log_probability_replay_error']=float(max(likelihood_errors))
    metrics['parameters_finite']=all(bool(torch.isfinite(p).all()) for p in model.parameters())
    return dict(checkpoint=str(path),episode=saved['training_episodes'],seed=seed,
        communication_mode=env.communication_mode,safety_filter=env.safety_filter,
        mode='sample' if stochastic else 'argmax',allow_unseen_peers=allow_unseen_peers,
        summary=summary,policy=metrics,choices=choices,
        trace=env.trace)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoints',nargs='+',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[20000,20001,20004])
    parser.add_argument('--modes',nargs='+',choices=['argmax','sample'],default=['argmax','sample'])
    parser.add_argument('--allow-unseen-peers',action='store_true',help='Diagnostic ablation only; changes visibility gate')
    parser.add_argument('--communication-mode',choices=['direct','chain'])
    parser.add_argument('--safety-filter',choices=['legacy','cbf'])
    parser.add_argument('--output',type=Path,default=Path('artifacts/mappo-inspection'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for path in args.checkpoints:
        for mode in args.modes:
            stochastic=mode=='sample'
            for seed in args.seeds:
                row=inspect(path,seed,stochastic,args.allow_unseen_peers,args.communication_mode,args.safety_filter)
                results.append(row)
                print(json.dumps(dict(episode=row['episode'],seed=seed,mode=row['mode'],
                    confidence=row['summary']['confidence_area'],**row['policy'])),flush=True)
                (args.output/'diagnostics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
