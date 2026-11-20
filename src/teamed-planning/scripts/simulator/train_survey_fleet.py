"""Fine-tune VTemp2 survey choices from real trajectories of the frozen V9 aircraft simulator."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

import torch

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))


def survey_source(scripts, name):
    """Locate a dependency in the reorganized source tree for provenance."""
    if name in ('ground_policy.py', 'survey_fleet_clearance.py'):
        return scripts / 'ground' / name
    if name.startswith(('ground_evidence', 'ground_relay', 'survey_')):
        return scripts / 'simulator' / name
    return scripts / 'world_model' / 'navigation' / name

from simulator.evaluate_survey_fleet import get_cases, run_one
from world_model.frozen_navigation import runtime, CHECKPOINT_SHA, V9_RUN
from world_model.hierarchical_sources import atomic_json, file_sha256
from world_model.slow_policy import update
from world_model.train_slow_policy import atomic_checkpoint
from simulator.train_survey_pretrain import KM_GAMMA, load_survey_weights, weights_contract


def collect(job):
    repo, weights, case, clock, limit, out = job
    from simulator.survey_fleet_runtime import run_fleet_case
    label = case['case_id']
    progress_file = out/'progress'/f'{label}.json'
    def progress(event):
        atomic_json(progress_file, dict(event, case_id=label))
    progress(dict(status='starting'))
    rollout = []
    with open(os.devnull, 'w') as sink, redirect_stdout(sink):
        torch.set_num_threads(1)
        runtime(repo)
        actor = load_survey_weights(weights).eval().requires_grad_(False)
        with torch.inference_mode():
            record = run_fleet_case(repo, case, actor, fast_seed=case['seed'],
                                    budget_s=math.ceil(3*case['direct_distance_m']+120), **clock,
                                    collector_limit_s=limit, rollout=rollout, progress=progress)
    record.update(case_id=label, direct_distance_m=case['direct_distance_m'], seed=case['seed'],
                  training_transitions=len(rollout), rollout_discounted_reward=sum(t['reward'] for t in rollout))
    atomic_json(out/'episodes'/f'{label}.json',record)
    compact = {k:v for k,v in record.items() if not isinstance(v, (dict,list))}
    progress(dict(compact,status='complete'))
    return rollout, compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=SOURCE.parents[3])
    parser.add_argument('--initialize-from', type=Path, required=True)
    parser.add_argument('--km-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=2)
    parser.add_argument('--episodes-per-update', type=int, default=3)
    parser.add_argument('--collector-seconds', type=int, default=180)
    parser.add_argument('--full-final-batch', action='store_true',
                        help='Collect the last training batch through the full native mission termination')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--test-cases', nargs='+', default=['v9-km-test-000','v9-km-far-test-000'])
    parser.add_argument('--learning-rate', type=float, default=.0001)
    args=parser.parse_args()
    if (min(args.updates,args.episodes_per_update,args.workers,args.collector_seconds)<1
            or not 0<args.learning_rate<=1):
        parser.error('Invalid training settings')
    repo, initial, data, out = [p.resolve() for p in (args.repo,args.initialize_from,args.km_data,args.output)]
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        parser.error('Use an empty output directory')
    base = json.loads((initial.parent/'manifest.json').read_text())
    test = {c['case_id']:c for c in get_cases(data,base)}
    if len(set(args.test_cases))!=len(args.test_cases) or not set(args.test_cases)<=test.keys():
        parser.error('Expected unique held-out case IDs')
    test = [test[key] for key in args.test_cases]
    pool_cases = json.loads((data/'scenes.json').read_text())['ppo_train']
    pool_ids = [c['case_id'] for c in pool_cases]
    # Older fleet checkpoints store the verified scene-file hash and the used
    # training subset. The complete split is in that same scene file.
    if 'ppo_train' in base:
        split_matches = pool_ids == base['ppo_train']
    else:
        split_matches = (base.get('format') == 'vtemp2_fleet_survey_ppo_v1'
                         and set(base['fleet_train_cases']).issubset(pool_ids))
    if not split_matches:
        raise ValueError('Fleet training split differs from the recorded kilometre split')
    middle = (len(pool_cases)+1)//2
    # Alternate the short and long kilometre bands in each training batch.
    cases = [c for i in range(middle) for c in (pool_cases[i:i+1]+pool_cases[middle+i:middle+i+1])]
    n=args.updates*args.episodes_per_update
    if n>len(cases): parser.error('Not enough distinct fleet training cases')
    cases=cases[:n]
    if {c['circles_sha256'] for c in cases} & {c['circles_sha256'] for c in test}:
        raise ValueError('Fleet training and evaluation must use disjoint maps')
    clock=dict(ground_interval_s=5,survey_check_s=5,survey_min_s=20,survey_timeout_s=60)
    train_dir=out/'train'
    for dest in (train_dir,out/'before',out/'after'):
        (dest/'progress').mkdir(parents=True)
        (dest/'episodes').mkdir()
    sources=[Path(__file__),SOURCE/'evaluate_survey_fleet.py',SOURCE/'train_survey_pretrain.py',
             SOURCE.parent/'world_model'/'frozen_navigation.py',
             *(survey_source(SOURCE.parent, f) for f in ('survey_fleet_runtime.py','survey_fleet_observation.py',
               'survey_fleet_metrics.py','survey_fleet_clearance.py','survey_fleet_drive.py','survey_fleet_phase.py','ground_relay.py','ground_evidence.py','ground_policy.py',
               'slow_policy.py','survey_pretrain_env.py','hierarchical_runtime.py','hierarchical_safety.py'))]
    provenance=dict(format='vtemp2_fleet_survey_ppo_v1', initial_checkpoint=str(initial),
                    initial_sha256=file_sha256(initial), v9_checkpoint_sha256=CHECKPOINT_SHA, aircraft=3,
                    ground_and_survey_clocks_s=clock, updates=args.updates, episodes_per_update=args.episodes_per_update,
                    collector_seconds=args.collector_seconds, full_final_batch=args.full_final_batch,
                    learning_rate=args.learning_rate, gamma=KM_GAMMA,
                    km_scenes_sha256=base['km_scenes_sha256'],far_scenes_sha256=base['far_scenes_sha256'],
                    ppo_train=pool_ids, test=base['test'],far_test=base['far_test'],evaluated_cases=args.test_cases,
                    fleet_train_cases=[c['case_id'] for c in cases], selection='final update; paired test not used for selection',
                    source_sha256={p.name:file_sha256(p) for p in sources})
    atomic_json(out/'manifest.json',provenance)
    start=time.monotonic()
    def status(phase,**kw):
        atomic_json(out/'status.json',dict(phase=phase,elapsed_wall_s=round(time.monotonic()-start,2),**kw))
    status('running')
    torch.set_num_threads(1)
    runtime(repo)
    model=load_survey_weights(initial)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    current=initial
    history=[]
    try:
        with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            # Start the full-budget initial comparison while collecting the first training batch.
            before = [pool.submit(run_one,repo,initial,c,clock,out/'before') for c in test]
            for i in range(args.updates):
                batch=cases[i*args.episodes_per_update:(i+1)*args.episodes_per_update]
                limit = None if args.full_final_batch and i+1 == args.updates else args.collector_seconds
                jobs=[(repo,current,c,clock,limit,train_dir) for c in batch]
                status('collecting',update=i+1)
                result=list(pool.map(collect,jobs))
                episodes=[episode for episode,_ in result if episode]
                if not episodes:
                    raise RuntimeError('No executed survey actions in fleet training batch')
                old={name:p.detach().clone() for name,p in model.named_parameters() if name.startswith('actor_')}
                metrics=update(model,optimizer,episodes,gamma=KM_GAMMA,epochs=2,target_kl=.015,max_kl=.03)
                delta=float(sum((p.detach()-old[name]).square().sum()
                                for name,p in model.named_parameters() if name in old).sqrt())
                metrics.update(actor_delta_l2=delta,
                               multi_choice_transitions=sum(int(t['mask'].sum()>1) for ep in episodes for t in ep))
                history.append(dict(update=i+1,ppo=metrics,episodes=[r for _,r in result]))
                atomic_json(out/'training.json',history)
                current=out/'fleet.pt'
                atomic_checkpoint(current,dict(contract=weights_contract(model.hidden),model=model.state_dict(),
                                               profile='fleet', completed_updates=i+1,
                                               initial_sha256=provenance['initial_sha256']))
                print('PPO',i+1,metrics['transitions'],'choices',metrics['multi_choice_transitions'],'actor delta',delta,flush=True)
            if not any(h['ppo']['actor_delta_l2']>0 for h in history):
                raise RuntimeError('The survey actor received no effective update')
            status('paired_evaluation')
            after=[pool.submit(run_one,repo,current,c,clock,out/'after') for c in test]
            for label,group in [('before',before),('after',after)]:
                rows=[f.result() for f in group]
                atomic_json(out/label/'evaluation.json',dict(cases=rows,arrivals=sum(bool(r['goal_reached']) for r in rows)))
                print(label,[(r['case_id'],r['end_reason'],r['remaining_m']) for r in rows],flush=True)
        if file_sha256(initial)!=provenance['initial_sha256'] or file_sha256(repo/V9_RUN/'best.pt')!=CHECKPOINT_SHA:
            raise RuntimeError('A source checkpoint changed during training')
        if any(file_sha256(p) != provenance['source_sha256'][p.name] for p in sources):
            raise RuntimeError('A fleet source file changed during training')
        status('complete',checkpoint_sha256=file_sha256(current))
    except BaseException as exc:
        status('failed',error=repr(exc))
        raise


if __name__=='__main__':
    main()
