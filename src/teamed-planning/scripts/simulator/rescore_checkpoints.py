"""Re-score saved MAPPO checkpoints with the current validation protocol.

Uses the validation seeds, sample count and score of a training config
(default: the run configuration), sampling actions like training. --policy argmax
measures deterministic deployment instead. One JSON line per checkpoint.

Example:
  python simulator/rescore_checkpoints.py --run ../../../artifacts/<run> --every 100 --workers 24
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import multiprocessing
import numpy as np
import torch
from drone.mappo import ActorCritic
from simulator.train_mappo import (_worker_init, fixed_schedule, run_episode, score_validation,
                                   validate_config, validation_runs)


def _episode(path, seed, sample_seed, stochastic, protocol):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    config=dict(validate_config(saved['config']),**protocol)
    model=ActorCritic.from_state_dict(saved['model']);model.eval()
    torch.manual_seed(sample_seed)
    _,summary=run_episode(model,config,seed,fixed_schedule(config,config['episodes']-1),stochastic)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--checkpoints',nargs='*',help='file names in --run; default to episode snapshots and latest.pt, or best.pt when latest is absent')
    parser.add_argument('--every',type=int,default=1,help='keep episode_*.pt whose episode is a multiple')
    parser.add_argument('--config',type=Path,help='validation protocol override (default: the run configuration)')
    parser.add_argument('--samples',type=int,help='override validation_samples')
    parser.add_argument('--policy',choices=('sample','argmax'),default='sample')
    parser.add_argument('--seeds',type=int,nargs='+',help='override validation_seeds')
    parser.add_argument('--subgoal-radius',type=float,
                        help='override: relay subgoals at most this far ahead (ground/subgoal_manager.py); 0 disables')
    parser.add_argument('--termination',choices=('map','arrival'),
                        help="override: end at a stable task graph or at vehicle arrival")
    parser.add_argument('--chain-ranges',type=float,nargs=2,metavar=('GROUND','LINK'),
                        help='override: candidate chain slack ranges in m (95 95 disables)')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.every < 1:
        parser.error('--every must be a positive integer')
    names = args.checkpoints
    if not names:
        names = [p.name for p in sorted(args.run.glob('episode_*.pt'))
                 if int(p.stem.split('_')[1]) % args.every == 0]
        for name in ('latest.pt', 'best.pt'):
            if (args.run/name).is_file():
                names.append(name)
                break
    if not names:
        parser.error('No checkpoints found in the run directory')
    config_path = args.config or args.run/'config.json'
    if args.config or config_path.is_file():
        settings = json.loads(config_path.read_text(encoding='utf-8-sig'))
    else:
        settings = torch.load(args.run/names[0],map_location='cpu',weights_only=True)['config']
    reference=validate_config(settings)
    protocol=dict(validation_seeds=args.seeds or reference['validation_seeds'],
                  validation_samples=args.samples or reference['validation_samples'])
    # Without overrides each checkpoint keeps its own training protocol.
    if args.subgoal_radius is not None:
        protocol['subgoal_radius']=args.subgoal_radius or None
    if args.termination:
        protocol['termination']=args.termination
    if args.chain_ranges:
        protocol['chain_ground_range'],protocol['chain_link_range']=args.chain_ranges
    if args.policy=='argmax':
        protocol['validation_samples']=1  # Deterministic runs do not differ.
    output=args.output or args.run/f'rescore_{args.policy}.jsonl'
    jobs,configs={},{}
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=_worker_init) as pool:
        for name in names:
            saved=torch.load(args.run/name,map_location='cpu',weights_only=True)
            config=configs[name]=dict(validate_config(saved['config']),**protocol)
            for seed,k,sample_seed in validation_runs(config):
                jobs[pool.submit(_episode,str(args.run/name),seed,sample_seed,args.policy=='sample',protocol)]=(
                    name,saved['training_episodes'],seed,k)
        tic=time.monotonic();results={}
        for future,(name,episode,seed,k) in jobs.items():
            results.setdefault((name,episode),[]).append(dict(seed=seed,sample=k,**future.result()))
    with output.open('a',encoding='utf-8') as stream:
        for (name,episode),records in results.items():
            report=score_validation(records,configs[name])
            row=dict(checkpoint=name,episode=episode,policy=args.policy,protocol=protocol,
                     finished_utc=datetime.now(timezone.utc).isoformat(),
                     connected_rate=report['means'].get('graph_task_connected'),
                     safety_rate=report['means']['safety_terminated'],
                     arrival_rate=report['means'].get('goal_reached'),**report)
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            print(f"{name} episode={episode} score={report['score']:.1f} "
                  f"success={row['means'][row['success_metric']]:.2f} safety={row['safety_rate']:.2f}",flush=True)
    print(f'{len(jobs)} episodes in {time.monotonic()-tic:.0f}s -> {output}')


if __name__=='__main__':
    main()
