"""Five annular distance stages with immutable endpoints and matched evaluations."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from drone.cbf_safety import CBFSafetyFilter
from drone.mappo import ActorCritic
from simulator.mappo_experiment import COMMUNICATION_RANGE_M, FLIGHT_HEIGHT_M
from simulator.train_mappo import (atomic_json, append_json, validate_config,
    fixed_schedule, run_episode, _worker_init, optimizer_config_changes)


def chain_radius(num_drones=3, margin=0.):
    """Ground-to-head horizontal reach plus same-altitude relay links."""
    radius = COMMUNICATION_RANGE_M-margin
    if num_drones < 2 or radius <= FLIGHT_HEIGHT_M:
        raise ValueError('Invalid chain geometry')
    return math.sqrt(radius**2-FLIGHT_HEIGHT_M**2)+(num_drones-1)*radius


def make_plan(base, settings):
    base = validate_config(base)
    if base['action_mode'] != 'velocity' or base['fleet_sizes'] != [3]:
        raise ValueError('This curriculum trains the shared velocity policy with three drones')
    if base['init_checkpoint'] is not None:
        raise ValueError('The first stage must start from random weights')
    edges = [float(settings['inner_radius_m']), chain_radius(), *settings['outer_stage_bounds_m']]
    if len(edges) != 6 or not all(a < b for a,b in zip(edges,edges[1:])):
        raise ValueError('Require five ordered annuli separated at the theoretical chain radius')
    if edges[-1] >= base['world_size']-70:
        raise ValueError('Annuli do not fit the configured world and initial formation margin')
    if type(settings['episodes_per_stage']) is not int or settings['episodes_per_stage'] < 1:
        raise ValueError('episodes_per_stage must be positive')
    eval_seeds = settings['evaluation_seeds']
    training_seeds = set(range(base['train_seed_start'],base['train_seed_start']+base['train_seed_count']))
    if not eval_seeds or len(set(eval_seeds)) != len(eval_seeds) or any(type(s) is not int for s in eval_seeds):
        raise ValueError('Evaluation seeds must be unique integers')
    if set(eval_seeds) & (training_seeds | set(base['test_seeds'])):
        raise ValueError('Distance evaluation must be held out from training and test seeds')
    if type(settings['evaluation_samples']) is not int or settings['evaluation_samples']<1:
        raise ValueError('evaluation_samples must be positive')
    return dict(schema_version=1,base_training_config=base,settings=settings,
        centre='initial_ground_vehicle_position_each_episode',
        distance_sampling='annulus_area',num_drones=3,
        communication_range_m=COMMUNICATION_RANGE_M,flight_height_m=FLIGHT_HEIGHT_M,
        theoretical_chain_radius_m=chain_radius(),
        cbf_margin_geometry_radius_m=chain_radius(margin=CBFSafetyFilter().margin),
        stages=[dict(stage=i+1,lower_m=lo,upper_m=hi,episodes=settings['episodes_per_stage'])
                for i,(lo,hi) in enumerate(zip(edges,edges[1:]))])


def stage_config(plan, stage, previous_checkpoint=None):
    config = dict(plan['base_training_config'])
    config.update(episodes=stage['episodes'],task_distances=[],
        task_distance_range=[stage['lower_m'],stage['upper_m']],
        max_task_distance=plan['stages'][-1]['upper_m'],distance_sampling='annulus_area',
        init_checkpoint=str(previous_checkpoint) if previous_checkpoint else None,
        init_optimizer=previous_checkpoint is not None,initial_validation=False,
        # Each ring has a distinct training action/scene sequence. Evaluation
        # seeds below deliberately do not depend on the trained stage.
        seed=plan['base_training_config']['seed']+stage['stage']-1,
        chain_ground_range=None,chain_link_range=None)
    return validate_config(config)


def preserve_checkpoint(source, destination):
    """Save once; verify copies before any next-stage training is allowed."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Immutable checkpoint already differs: {destination}')
    else:
        temporary=destination.with_suffix('.pt.tmp')
        shutil.copy2(source,temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
            raise IOError('Checkpoint copy checksum mismatch')
        temporary.replace(destination)
    return digest


def evaluation_job(checkpoint, config, seed, sample, sample_seed, stop_file):
    torch.manual_seed(sample_seed)
    model=ActorCritic.from_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
    model.eval()
    _,summary=run_episode(model,config,seed,fixed_schedule(config,config['episodes']-1),True,
                          stop=lambda:Path(stop_file).exists())
    return dict(seed=seed,sample=sample,**summary)


def distance_band_summary(band, rows, config):
    success_metric='map_completed' if config['reward_mode']=='velocity_mapping' else 'goal_reached'
    summary=dict(**band,n=len(rows),success_metric=success_metric,
        success_rate=float(np.mean([r[success_metric] for r in rows])),
        vehicle_arrival_rate=float(np.mean([r['goal_reached'] for r in rows])),
        map_completion_rate=float(np.mean([r['map_completed'] for r in rows])),
        safety_termination_rate=float(np.mean([r['safety_terminated'] for r in rows])),
        timeout_rate=float(np.mean([r['budget_exhausted'] for r in rows])),
        mean_remaining_m=float(np.mean([r['ground_goal_distance'] for r in rows])),
        mean_episode_steps=float(np.mean([r['episode_steps'] for r in rows])),
        boundary_violations=float(sum(r['out_of_bounds'] for r in rows)),
        boundary_roundoff_count=float(sum(r['boundary_roundoff_count'] for r in rows)))
    if all('route_new_best_saving_m' in r for r in rows):
        connected=[r for r in rows if r['graph_task_connected']]
        summary.update(connected_route_count=len(connected),
            mean_route_efficiency_connected=float(np.mean([r['route_efficiency'] for r in connected])) if connected else None,
            mean_route_excess_m_connected=float(np.mean([r['route_excess_m'] for r in connected])) if connected else None,
            mean_route_new_best_saving_m=float(np.mean([r['route_new_best_saving_m'] for r in rows])))
    if all('map_optimal' in r for r in rows):
        completed=[r for r in rows if r['map_completed']]
        summary.update(map_optimal_rate=float(np.mean([r['map_optimal'] for r in rows])),
            map_plateau_rate=float(np.mean([r['map_plateau'] for r in rows])),
            mean_improvement_steps_completed=float(np.mean([r['improvement_steps'] for r in completed])) if completed else None)
    return summary


def evaluate_distance_matrix(plan, stage, checkpoint, config, output, workers):
    """Compare a fixed stage-end policy on identical seeds in every ring."""
    folder=output/f"stage_{stage['stage']:02d}"
    records_path=folder/'distance_evaluation.jsonl'
    rows=[json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    completed={(r['band'],r['seed'],r['sample']) for r in rows}
    jobs=[]
    for band in plan['stages']:
        for seed in plan['settings']['evaluation_seeds']:
            for sample in range(plan['settings']['evaluation_samples']):
                if (band['stage'],seed,sample) not in completed:
                    jobs.append((band,seed,sample))
    count=len(rows)+len(jobs)
    stop=output/'STOP'
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=_worker_init) as pool:
        active={}
        while jobs or active:
            while jobs and len(active)<workers and not stop.exists():
                band,seed,sample=jobs.pop(0)
                eval_config=dict(config,task_distance_range=[band['lower_m'],band['upper_m']])
                sample_seed=int(np.random.SeedSequence([plan['base_training_config']['seed'],seed,sample]).generate_state(1)[0])
                future=pool.submit(evaluation_job,str(checkpoint),eval_config,seed,sample,sample_seed,str(stop))
                active[future]=band
            if not active:
                break
            done,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
            for future in done:
                band=active.pop(future)
                row=future.result()
                if not row['user_stopped']:
                    row.update(band=band['stage'],lower_m=band['lower_m'],upper_m=band['upper_m'])
                    append_json(records_path,row)
                    rows.append(row)
            atomic_json(output/'progress.json',dict(status='distance_evaluation',pid=os.getpid(),
                stage=stage['stage'],completed=len(rows),total=count,
                saved_checkpoint=str(checkpoint),updated_utc=datetime.now(timezone.utc).isoformat()))
    if stop.exists():
        return False
    by_band=[]
    success_metric='map_completed' if config['reward_mode']=='velocity_mapping' else 'goal_reached'
    for band in plan['stages']:
        sample_rows=[r for r in rows if r['band']==band['stage']]
        assert len(sample_rows)==len(plan['settings']['evaluation_seeds'])*plan['settings']['evaluation_samples']
        by_band.append(distance_band_summary(band,sample_rows,config))
    report=dict(trained_stage=stage['stage'],checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        evaluation_seeds=plan['settings']['evaluation_seeds'],success_metric=success_metric,
        samples_per_seed=plan['settings']['evaluation_samples'],bands=by_band)
    atomic_json(folder/'distance_summary.json',report)
    matrix=[json.loads(p.read_text()) for p in sorted(output.glob('stage_*/distance_summary.json'))]
    atomic_json(output/'distance_matrix.json',matrix)
    return True


def run(settings_path, output, workers=8, resume=False, through_stage=None, upgrade_optimizer=False):
    if upgrade_optimizer and not resume:
        raise ValueError('Optimizer upgrade requires resume')
    settings=json.loads(settings_path.read_text())
    base_path=(settings_path.parent/settings.pop('base_config')).resolve()
    plan=make_plan(json.loads(base_path.read_text()),settings)
    stage_limit=len(plan['stages']) if through_stage is None else through_stage
    if type(stage_limit) is not int or not 1<=stage_limit<=len(plan['stages']):
        raise ValueError('through_stage must select an existing stage')
    output.mkdir(parents=True,exist_ok=True)
    existing=output/'curriculum.json'
    if existing.exists():
        old_plan=json.loads(existing.read_text())
        if not resume or {k:v for k,v in old_plan.items() if k!='base_training_config'}!={
                k:v for k,v in plan.items() if k!='base_training_config'}:
            raise ValueError('Existing curriculum requires --resume with the same plan')
        changes=optimizer_config_changes(old_plan['base_training_config'],plan['base_training_config'],upgrade_optimizer)
        if changes:
            backup=output/'curriculum_before_optimizer_upgrade.json'
            if not backup.exists():shutil.copy2(existing,backup)
            atomic_json(output/'optimizer_upgrade.json',dict(changes=changes,
                upgraded_utc=datetime.now(timezone.utc).isoformat(),previous_curriculum=str(backup)))
    elif resume:
        raise ValueError('No existing curriculum to resume')
    atomic_json(existing,plan)
    if resume:
        (output/'STOP').unlink(missing_ok=True)
    trainer=Path(__file__).with_name('train_mappo.py')
    previous=None
    for stage in plan['stages'][:stage_limit]:
        if (output/'STOP').exists():
            break
        folder=output/f"stage_{stage['stage']:02d}"
        folder.mkdir(exist_ok=True)
        config=stage_config(plan,stage,previous)
        config_path=output/f"stage_{stage['stage']:02d}_config.json"
        atomic_json(config_path,config)
        final_checkpoint=output/f"stage_{stage['stage']:02d}_final.pt"
        if not (folder/'stage_complete.json').exists():
            progress_path=folder/'progress.json'
            prior=json.loads(progress_path.read_text()) if progress_path.exists() else {}
            if prior.get('status') != 'completed':
                command=[sys.executable,'-I','-u',str(trainer),'--config',str(config_path),
                         '--output',str(folder),'--workers',str(workers)]
                if (folder/'latest.pt').exists():
                    if not resume:
                        raise ValueError('Interrupted stage requires explicit resume')
                    (folder/'STOP').unlink(missing_ok=True)
                    command.append('--resume')
                    if upgrade_optimizer:command.append('--upgrade-optimizer')
                with (folder/'stdout.log').open('ab') as stdout,(folder/'stderr.log').open('ab') as stderr:
                    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
                    while child.poll() is None:
                        if (output/'STOP').exists():
                            (folder/'STOP').touch()
                        current=json.loads(progress_path.read_text()) if progress_path.exists() else {}
                        atomic_json(output/'progress.json',dict(status='training',pid=os.getpid(),
                            trainer_pid=child.pid,stage=stage['stage'],stages=len(plan['stages']),
                            through_stage=stage_limit,
                            lower_m=stage['lower_m'],upper_m=stage['upper_m'],
                            completed_episodes=current.get('completed_episodes',0),total_episodes=stage['episodes'],
                            cumulative_completed=(stage['stage']-1)*stage['episodes']+current.get('completed_episodes',0),
                            updated_utc=datetime.now(timezone.utc).isoformat()))
                        time.sleep(2)
                if child.returncode:
                    raise RuntimeError(f"Stage {stage['stage']} trainer exited {child.returncode}")
                if (output/'STOP').exists():
                    break
                if json.loads(progress_path.read_text())['status'] != 'completed':
                    raise RuntimeError('Stage stopped before completion; next ring was not started')
            checksum=preserve_checkpoint(folder/'latest.pt',final_checkpoint)
            if (folder/'best.pt').exists():
                preserve_checkpoint(folder/'best.pt',output/f"stage_{stage['stage']:02d}_best.pt")
            atomic_json(folder/'checkpoint_saved.json',dict(stage=stage['stage'],
                final_checkpoint=str(final_checkpoint),sha256=checksum,
                saved_utc=datetime.now(timezone.utc).isoformat()))
            if not evaluate_distance_matrix(plan,stage,final_checkpoint,config,output,workers):
                break
            atomic_json(folder/'stage_complete.json',dict(stage=stage['stage'],checkpoint=str(final_checkpoint),
                checkpoint_sha256=checksum,completed_utc=datetime.now(timezone.utc).isoformat()))
        if not final_checkpoint.exists():
            raise RuntimeError('Missing stage-end weights; refusing to expand')
        previous=final_checkpoint
    stopped=(output/'STOP').exists()
    status='stopped' if stopped else ('stage_limit_reached' if stage_limit<len(plan['stages']) else 'completed')
    atomic_json(output/'progress.json',dict(status=status,pid=os.getpid(),
        stage=stage['stage'],through_stage=stage_limit,updated_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--through-stage',type=int,default=None,
                        help='save/evaluate this stage and exit without starting the next ring')
    parser.add_argument('--upgrade-optimizer',action='store_true',
                        help='explicitly migrate only PPO sampling/update settings on resume')
    args=parser.parse_args()
    try:
        run(args.config.resolve(),args.output.resolve(),args.workers,args.resume,args.through_stage,args.upgrade_optimizer)
    except BaseException as exc:
        if args.output.exists():
            atomic_json(args.output/'progress.json',dict(status='failed',pid=os.getpid(),error=repr(exc),
                updated_utc=datetime.now(timezone.utc).isoformat()))
        raise
