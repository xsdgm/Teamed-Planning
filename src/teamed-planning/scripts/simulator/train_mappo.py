"""Persistent MAPPO training with fixed rewards and resumable checkpoints.

Training randomness, optimizer state and configuration are saved together.
Validation uses fixed reward weights and held-out seeds;
test seeds are never used for checkpoint selection.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import time
import traceback
import math
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from drone.mappo import ActorCritic
from simulator.mappo_experiment import Experiment, SafetyTermination, update
from simulator.large_world_scenarios import generate_training_scene
from simulator.scenarios import ENVIRONMENT_VERSION


DEFAULTS = dict(episodes=2000, steps=240, seed=42, learning_rate=3e-4,
    ppo_epochs=4, validation_interval=50, snapshot_interval=100,
    train_seed_start=10000, train_seed_count=8000,
    validation_seeds=[20000,20001,20002,20003,20004],
    test_seeds=[30000,30001,30002,30003,30004,30005,30006,30007,30008,30009],
    # Inert historical fields retained solely for v4 checkpoint config equality.
    exploration_candidates=5, drop=0., curriculum_start=.15, curriculum_end=.70,
    beta_early=.2, beta_late=2.5, eta_early=.6, eta_late=.05,
    entropy_early=.03,entropy_late=.005,communication_mode='chain',safety_filter='cbf',
    spacing_penalty_weight=0.,intervention_weight=0.,moving_ground=True,
    reward_mode='cadmm_exploration',ground_speed=1.,path_reward_scale=100.,
    reward_schedule='fixed',exploration_weight=.6,entropy_coefficient=.01,
    world_size=120.,task_distances=[350,550,750],budget_multiplier=3.,budget_extra=240,
    min_episode_steps=600,map_stability_steps=30,safety_failure_penalty=10.,
    discount=.99,gae_lambda=.95,initial_validation=True,
    environment_version=ENVIRONMENT_VERSION)


def fixed_schedule(config, completed):
    if config.get('reward_schedule', 'fixed') != 'fixed':
        raise ValueError('Historical curriculum rewards have been removed')
    progress = min(1., max(0., completed/max(config['episodes']-1,1)))
    return dict(progress=progress,focus=0.,beta=1.,eta=config['exploration_weight'],
                entropy=config['entropy_coefficient'])


def validate_config(config):
    unknown=set(config)-set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
    config = dict(DEFAULTS, **config)
    if config['environment_version'] != ENVIRONMENT_VERSION:
        raise ValueError('Environment version differs; start a separate corrected training run')
    if config['reward_schedule'] != 'fixed':
        raise ValueError('Invalid reward schedule')
    for name in ('exploration_weight','entropy_coefficient'):
        if not np.isfinite(config[name]) or config[name]<0:
            raise ValueError(f'{name} must be finite and nonnegative')
    for field in ('episodes','steps','ppo_epochs','validation_interval','snapshot_interval','train_seed_count'):
        if not isinstance(config[field],int) or config[field] < 1:
            raise ValueError(f'{field} must be a positive integer')
    if not 0 <= config['curriculum_start'] < config['curriculum_end'] <= 1:
        raise ValueError('Require 0 <= curriculum_start < curriculum_end <= 1')
    if not 0 <= config['drop'] <= 1 or not 0 <= config['exploration_candidates'] <= 20:
        raise ValueError('Invalid packet-loss probability or candidate count')
    if not config['validation_seeds'] or not config['test_seeds']:
        raise ValueError('Validation/test seeds must be non-empty')
    training=set(range(config['train_seed_start'],config['train_seed_start']+config['train_seed_count']))
    validation,test=set(config['validation_seeds']),set(config['test_seeds'])
    if training & validation or training & test or validation & test:
        raise ValueError('Training, validation and test scene seeds must be disjoint')
    for name in ('beta','eta','entropy'):
        if any(config[name+'_'+stage] < 0 for stage in ('early','late')):
            raise ValueError('Reward and entropy coefficients must be nonnegative')
    if config['learning_rate'] <= 0:
        raise ValueError('learning_rate must be positive')
    if config['communication_mode'] != 'chain' or config['safety_filter'] != 'cbf':
        raise ValueError('Current training requires chain communication and CBF safety')
    if config['spacing_penalty_weight']<0 or config['intervention_weight']<0:
        raise ValueError('Safety penalty weights must be nonnegative')
    if config['moving_ground'] is not True or config['reward_mode'] != 'cadmm_exploration':
        raise ValueError('Invalid ground mode or reward mode')
    if config['reward_mode']=='cadmm_exploration' and not config['moving_ground']:
        raise ValueError('CADMM reward requires the ground map receiver/controller')
    if not 0 < config['ground_speed'] <= 2 or not np.isfinite(config['path_reward_scale']) or config['path_reward_scale']<=0:
        raise ValueError('Invalid ground speed or path reward scale')
    if config['world_size'] not in (120.,1000.):
        raise ValueError('Supported training worlds are 120m and 1000m')
    if config['world_size']==1000. and (not config['task_distances'] or any(
        type(d) is not int or not 0<d<config['world_size']-70 for d in config['task_distances'])):
        raise ValueError('Invalid large-world task distances')
    for name in ('budget_multiplier','discount','gae_lambda','safety_failure_penalty'):
        if not np.isfinite(config[name]) or config[name]<=0:
            raise ValueError(f'{name} must be positive and finite')
    if config['discount']>1 or config['gae_lambda']>1:
        raise ValueError('Discount and GAE lambda cannot exceed 1')
    for name in ('budget_extra','min_episode_steps','map_stability_steps'):
        if type(config[name]) is not int or config[name]<1:
            raise ValueError(f'{name} must be a positive integer')
    if type(config['initial_validation']) is not bool:
        raise ValueError('initial_validation must be bool')
    return config


def atomic_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)


def atomic_checkpoint(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary)
    os.replace(temporary,path)


def append_json(path, value):
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush()


def run_episode(model, config, scene_seed, schedule, stochastic, progress=None, stop=None):
    large=config['world_size']==1000.
    scene=generate_training_scene(scene_seed,config['task_distances']) if large else None
    options={} if scene is None else dict(world_size=1000.,start=scene['start'],
        scene=(scene['goal'],scene['circles'],scene['attempts']),stop_on_ground_arrival=False)
    requested_budget=config['steps'] if scene is None else max(config['min_episode_steps'],
        math.ceil(config['budget_multiplier']*scene['reference_length']+config['budget_extra']))
    budget=min(config['steps'],requested_budget)
    with contextlib.redirect_stdout(io.StringIO()):
        env=Experiment(scene_seed,model,stochastic=stochastic,drop=config['drop'],
            exploration_weight=schedule['eta'],exploration_candidates=config['exploration_candidates'],
            communication_mode=config['communication_mode'],safety_filter=config['safety_filter'],
            spacing_penalty_weight=config['spacing_penalty_weight'],intervention_weight=config['intervention_weight'],
            moving_ground=config['moving_ground'],reward_mode=config['reward_mode'],
            ground_speed=config['ground_speed'],path_reward_scale=config['path_reward_scale'],**options)
        rollout=[];reason='budget';stable=0;max_link=0.
        try:
            for frame in range(budget):
                if stop is not None and stop() and rollout:
                    reason='user_stop';break
                try:
                    samples,reward,_=env.step(frame)
                except SafetyTermination as exc:
                    if not large:raise
                    # The proposed joint action transitions to an absorbing failure
                    # without executing infeasible velocities or relaxing safety.
                    samples=exc.samples;reward=-config['safety_failure_penalty']
                    rollout.append((*[np.array([sample[k] for sample in samples]) for k in range(4)],reward))
                    reason='cbf_infeasible';break
                rollout.append((*[np.array([sample[k] for sample in samples]) for k in range(4)],reward))
                points=np.asarray(env.trace[-1]['positions'])
                ground=np.asarray(env.obs[env.ground]['self']['position'])
                ground=np.r_[ground[:2],ground[2] if len(ground)>2 else 0.]
                max_link=max(max_link,float(np.linalg.norm(points[0]-ground)),
                    float(np.linalg.norm(points[1:]-points[:-1],axis=1).max()))
                stable=stable+1 if env.trace[-1]['graph_task_connected'] else 0
                if progress is not None and (frame%25==0 or frame+1==budget):
                    progress(dict(step=frame+1,budget=budget,scene_seed=scene_seed,
                        world_size=config['world_size'],task_distance_m=env.initial_goal_distance,
                        graph_connected=bool(stable),ground_remaining_m=env.trace[-1]['ground_goal_distance'],
                        delivered_area_m2=float(env.ledger.seen.sum()*4),max_link_distance_m=max_link))
                if large and stable>=config['map_stability_steps']:
                    reason='map_connected';break
                if not large and env.done:
                    reason='vehicle_arrival';break
            summary=env.summary()
            if reason=='cbf_infeasible':summary['reward']-=config['safety_failure_penalty']
            summary.update(safety_terminated=float(reason=='cbf_infeasible'),
                map_completed=float(reason=='map_connected'),budget_exhausted=float(reason=='budget'),
                user_stopped=float(reason=='user_stop'),episode_budget_steps=budget,
                task_distance_m=env.initial_goal_distance,max_link_distance_m=max_link,
                reference_length_m=scene['reference_length'] if scene else env.initial_goal_distance,
                rollout_steps=len(rollout))
        finally:
            env.env.close()
    return rollout,summary


def validation(model, config):
    # Use the same fixed weights for validation throughout training.
    schedule=fixed_schedule(config,config['episodes']-1)
    records=[]
    model.eval()
    for seed in config['validation_seeds']:
        _,summary=run_episode(model,config,seed,schedule,False)
        records.append(dict(seed=seed,**summary))
    averages={key:float(np.mean([row[key] for row in records])) for key in records[0] if key!='seed'}
    # Select navigation graphs by connectivity, time, efficiency and safety.
    delays=[r['graph_first_connection_step'] if r['graph_first_connection_step']>=0
            else config['steps'] for r in records]
    score=1000*averages['graph_task_connected']-float(np.mean(delays))/config['steps'] \
        -.1*averages['graph_task_path_ratio']+.001*averages['reward'] \
        -10000*averages['ground_collisions']-10000*averages['collisions'] \
        -10*averages['chain_link_violation']
    if config['world_size']==1000.:
        score-=1000*averages['safety_terminated']
    return dict(score=score,means=averages,scenes=records)


def train(config, output, resume=False):
    config=validate_config(config)
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    latest=output/'latest.pt'
    if latest.exists() and not resume:
        raise ValueError(f'{latest} already exists; use --resume or a new output directory')
    if resume and not latest.exists():
        raise FileNotFoundError(latest)
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    rng=np.random.default_rng(config['seed'])
    model=ActorCritic()
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    start,best_score,best_episode=0,None,None
    joint_steps=0
    if resume:
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        if validate_config(saved['config'])!=config:
            raise ValueError('Resume configuration differs from saved training configuration')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['scene_rng']
        torch.set_rng_state(saved['torch_rng'])
        start=saved['training_episodes']
        joint_steps=saved.get('joint_steps',start*config['steps'])
        best_score,best_episode=saved['best_score'],saved['best_episode']
        # Remove uncommitted log rows left by interruption between append and save.
        for name in ('training.jsonl','validation.jsonl'):
            path=output/name
            if path.exists():
                kept=[]
                for line in path.read_text(encoding='utf-8').splitlines():
                    try:
                        row=json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row['episode']<=start:
                        kept.append(line)
                path.write_text('\n'.join(kept)+ ('\n' if kept else ''),encoding='utf-8')
    atomic_json(output/'config.json',config)
    scripts=Path(__file__).resolve().parents[1]
    source_files=sorted(str(path.relative_to(scripts)).replace('\\','/')
                        for path in scripts.rglob('*.py'))
    source_hashes={name:hashlib.sha256((scripts/name).read_bytes()).hexdigest() for name in source_files}
    provenance=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,
                    source_sha256=source_hashes)
    atomic_json(output/('resume_provenance.json' if resume else 'provenance.json'),provenance)
    if not resume:
        for name in source_files:
            target=output/'sources'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(scripts/name,target)
    started=time.monotonic()
    def status(state, completed, **extra):
        atomic_json(output/'progress.json',dict(status=state,pid=os.getpid(),completed_episodes=completed,
            total_episodes=config['episodes'],joint_steps=joint_steps,
            updated_utc=datetime.now(timezone.utc).isoformat(),
            session_elapsed_seconds=round(time.monotonic()-started,2),
            best_validation_score=best_score,best_episode=best_episode,**extra))
    def checkpoint(completed):
        return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            scene_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            training_episodes=completed,joint_steps=joint_steps,config=config,feature_version=1,fov_radius=12.,
            exploration_candidates=config['exploration_candidates'],world_bounds=[0.,0.,config['world_size'],config['world_size']],
            communication_mode=config['communication_mode'],safety_filter=config['safety_filter'],
            moving_ground=config['moving_ground'],reward_mode=config['reward_mode'],
            environment_version=ENVIRONMENT_VERSION,
            ground_speed=config['ground_speed'],path_reward_scale=config['path_reward_scale'],
            exploration_weight=fixed_schedule(config,completed)['eta'],
            best_score=best_score,best_episode=best_episode)
    status('running',start)
    if not resume:
        atomic_checkpoint(latest,checkpoint(0))
    completed=start
    try:
        if not resume and config['initial_validation']:
            status('validating',0,reason='untrained baseline')
            atomic_json(output/'initial_validation.json',validation(model,config))
            status('running',0)
        for episode in range(start,config['episodes']):
            if (output/'STOP').exists():
                status('stopped',completed,reason='STOP file found; resume with --resume')
                return
            schedule=fixed_schedule(config,episode)
            scene_seed=int(rng.integers(config['train_seed_start'],
                                       config['train_seed_start']+config['train_seed_count']))
            model.train()
            tic=time.monotonic()
            if config['world_size']==1000.:
                def episode_progress(value):
                    atomic_json(output/'episode_progress.json',dict(episode=episode+1,
                        updated_utc=datetime.now(timezone.utc).isoformat(),**value))
                rollout,summary=run_episode(model,config,scene_seed,schedule,True,
                    progress=episode_progress,stop=lambda:(output/'STOP').exists())
            else:
                rollout,summary=run_episode(model,config,scene_seed,schedule,True)
            loss,ppo=update(model,optimizer,rollout,epochs=config['ppo_epochs'],
                entropy_coef=schedule['entropy'],return_metrics=True,
                discount=config['discount'],gae_lambda=config['gae_lambda'])
            completed=episode+1
            joint_steps+=len(rollout)
            record=dict(episode=completed,scene_seed=scene_seed,loss=loss,ppo=ppo,schedule=schedule,
                elapsed_seconds=round(time.monotonic()-tic,3),**summary)
            append_json(output/'training.jsonl',record)
            report=None
            if completed % config['validation_interval']==0 or completed==config['episodes']:
                # Validation status retains the latest committed episode number;
                # its updated checkpoint is written after validation succeeds.
                status('validating',completed-1,pending_episode=completed,last_training=record)
                report=validation(model,config)
                append_json(output/'validation.jsonl',dict(episode=completed,**report))
                if best_score is None or report['score']>best_score:
                    best_score,best_episode=report['score'],completed
                    atomic_checkpoint(output/'best.pt',checkpoint(completed))
            atomic_checkpoint(latest,checkpoint(completed))
            if completed % config['snapshot_interval']==0:
                atomic_checkpoint(output/f'episode_{completed:05d}.pt',checkpoint(completed))
            status('running',completed,last_training=record,last_validation=report)
            print(f'episode {completed}/{config["episodes"]} '
                f'beta={summary["reward_beta"]:.3f} eta={summary["reward_eta"]:.3f} '
                f'confidence={summary["confidence_area"]:.1f} '
                f'task_share={summary["task_share"]:.3f} '
                f'ground_m={summary.get("ground_distance",0.):.1f} '
                f'goal_remaining={summary.get("ground_goal_distance",0.):.1f} '
                f'cadmm_gain={summary.get("cadmm_gain",0.):.1f} loss={loss:.4f} '
                f'kl={ppo["approx_kl"]:.6f} entropy={ppo["entropy"]:.3f} '
                f'ev={ppo["explained_variance"]:.3f}',flush=True)
        status('completed',completed)
    except BaseException:
        status('failed',completed,error=traceback.format_exc())
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    train(config,args.output,args.resume)


if __name__=='__main__':
    main()
