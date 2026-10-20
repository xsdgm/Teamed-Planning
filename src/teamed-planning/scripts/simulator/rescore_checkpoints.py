"""Re-score saved MAPPO checkpoints with the current validation protocol.

Uses the validation seeds, sample count and score of a training config
(default: the 1000 m config), sampling actions like training. --policy argmax
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

DEFAULT_CONFIG = Path(__file__).resolve().parents[2]/'config'/'algorithm'/'mappo_training_1000m.json'


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
    parser.add_argument('--checkpoints',nargs='*',help='file names in --run; default episode_*.pt and latest.pt')
    parser.add_argument('--every',type=int,default=1,help='keep episode_*.pt whose episode is a multiple')
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG,help='source of the validation protocol')
    parser.add_argument('--samples',type=int,help='override validation_samples')
    parser.add_argument('--policy',choices=('sample','argmax'),default='sample')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    reference=validate_config(json.loads(args.config.read_text(encoding='utf-8-sig')))
    protocol=dict(validation_seeds=reference['validation_seeds'],
                  validation_samples=args.samples or reference['validation_samples'])
    if args.policy=='argmax':
        protocol['validation_samples']=1  # Deterministic runs do not differ.
    names=args.checkpoints or [p.name for p in sorted(args.run.glob('episode_*.pt'))
                               if int(p.stem.split('_')[1])%args.every==0]+['latest.pt']
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
                     connected_rate=report['means']['graph_task_connected'],
                     safety_rate=report['means']['safety_terminated'],**report)
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            print(f"{name} episode={episode} score={report['score']:.1f} "
                  f"connected={row['connected_rate']:.2f} safety={row['safety_rate']:.2f}",flush=True)
    print(f'{len(jobs)} episodes in {time.monotonic()-tic:.0f}s -> {output}')


if __name__=='__main__':
    main()
