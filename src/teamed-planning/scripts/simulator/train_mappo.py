"""Persistent MAPPO training with fixed rewards and resumable checkpoints.

Training randomness, optimizer state and configuration are saved together.
Validation uses fixed reward weights and held-out seeds and samples actions
like training (validation_samples runs per seed, fixed sample seeds), so it
scores the policy being trained; checkpoints record policy_mode='sample'.
Test seeds are never used for checkpoint selection.

Synchronous mode collects a complete batch under one frozen policy, then makes
one guarded PPO update. Legacy asynchronous mode remains available for old
experiments. Validation always uses the weights saved at its episode.
"""
import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing
import os
import shutil
import signal
import sys
import time
import traceback
import math
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from drone.mappo import ActorCritic, GLOBAL_FEATURES
from drone.velocity_mappo import VelocityActorCritic
from simulator.mappo_experiment import Experiment, SafetyTermination, update
from simulator.ppo_batch import update_batch
from simulator.surprise_reward import TaskMapEncoder, transition, SurpriseReward
from simulator.large_world_scenarios import generate_training_scene
from simulator.env import BOUNDARY_ROUNDOFF_TOLERANCE_M
from simulator.scenarios import ENVIRONMENT_VERSION


DEFAULTS = dict(episodes=2000, steps=240, seed=42, learning_rate=3e-4,
    ppo_epochs=4, validation_interval=50, snapshot_interval=100,
    train_seed_start=10000, train_seed_count=8000,
    validation_seeds=[20000,20001,20002,20003,20004],validation_samples=2,
    test_seeds=[30000,30001,30002,30003,30004,30005,30006,30007,30008,30009],
    # Inert historical fields retained solely for v4 checkpoint config equality.
    exploration_candidates=5, drop=0., curriculum_start=.15, curriculum_end=.70,
    beta_early=.2, beta_late=2.5, eta_early=.6, eta_late=.05,
    entropy_early=.03,entropy_late=.005,communication_mode='chain',safety_filter='cbf',
    spacing_penalty_weight=0.,intervention_weight=0.,moving_ground=True,
    reward_mode='cadmm_exploration',ground_speed=1.,path_reward_scale=100.,
    reward_schedule='fixed',exploration_weight=.6,entropy_coefficient=.01,
    world_size=120.,task_distances=[350,550,750],task_distance_range=None,budget_multiplier=3.,budget_extra=240,
    min_episode_steps=600,map_stability_steps=30,safety_failure_penalty=10.,
    discount=.99,gae_lambda=.95,initial_validation=True,
    # Relay subgoals (ground/subgoal_manager.py), episode end at vehicle arrival
    # instead of a stable task graph, its terminal reward, and initial weights.
    subgoal_radius=None,subgoal_arrival=25.,termination='map',arrival_reward=0.,init_checkpoint=None,
    # Candidate chain slack below the 95 m link limit (None: no slack).
    chain_ground_range=None,chain_link_range=None,
    # velocity_mapping: fixed start->goal map improvement and local chain risk;
    # stable map completion is independent of vehicle arrival/collisions.
    # route_graph reward: start->goal route cost decrease on the delivered graph
    # (vehicle-independent), a per-step time penalty, and on a stable connected
    # task graph a terminal reward scaled by reference/graph route length.
    # vehicle_route: the same, but the route cost is measured from the vehicle
    # as in cadmm_exploration (v2), without exploration reward.
    route_reward_weight=10.,time_penalty=0.,map_success_reward=0.,ground_stationary=False,
    action_mode='candidate',fleet_sizes=[3],environment_version=ENVIRONMENT_VERSION,
    max_task_distance=350.,distance_sampling='linear',init_optimizer=False,
    chain_soft_ratio=.85,chain_risk_weight=0.,chain_outward_weight=0.,chain_missing_weight=0.,
    path_quality_weight=0.,map_quality_reward=0.,
    # velocity_mapping: the start->goal map term stops once route quality pays.
    mapping_reward_until_connected=False,
    # velocity_mapping: penalty on the least CBF feasibility margin (m/s), and an
    # improvement phase after the first stable map (+100 then); it ends at
    # map_target_quality or after map_improvement_steps without a better route
    # (0: legacy end at the stable map).
    cbf_margin_weight=0.,cbf_margin_scale=8.,map_improvement_steps=0,map_target_quality=1.,
    # velocity_mapping v5: best-so-far progress of the start-connected delivered
    # graph toward the goal, paid up to this total per episode (0: none).
    frontier_progress_weight=0.,
    # Linear entropy decay from entropy_coefficient to this value (None: constant).
    entropy_coefficient_final=None,
    rollout_mode='asynchronous',ppo_target_kl=.01,ppo_max_kl=.02,ppo_max_backtracks=4,
    # Synchronous velocity critic trained apart from the KL-guarded actor step
    # (0: legacy shared loss), optionally on PopArt-normalized returns.
    critic_epochs=0,critic_minibatch_size=512,value_normalization=False,
    # v6: training-only task-map surprise; zero keeps the v5 rollout unchanged.
    policy_architecture='mlp',
    # v6.2: actor input version (1: v3-v6 features; 2: invalidity sensor plane,
    # centred altitude, own CBF margin and chain bend) and synchronous episodes
    # per PPO batch (None: one per worker, cut at snapshot/validation episodes).
    actor_input_version=1,episodes_per_batch=None,
    surprise_weight=0.,surprise_weight_final=0.,surprise_clip=3.,
    surprise_episode_budget=5.,surprise_learning_rate=3e-4,surprise_epochs=1,
    # v8 relayed state evidence: per-link loss, extra delay jitter (steps beyond
    # the minimum one), send period (steps), entries per envelope, and the
    # attention ablation (full=v8-D, age=v8-A, hop=v8-B, softmax=v8-C).
    relay_loss=0.,relay_delay_jitter=0,relay_period=1,relay_capacity=16,relay_attention='full',
    # Vehicle-side fusion joins graph components within this distance (m) when the
    # delivered occupancy shows a free corridor (0: off; at most the drones' 15 m edge).
    fusion_bridge_distance=0.)

COOPERATIVE_ARCHITECTURES=('gat_cooperative_v7','gat_relay_attention_v8')
GRAPH_ARCHITECTURES=('gat_map_v1',)+COOPERATIVE_ARCHITECTURES

OPTIMIZER_CONFIG_FIELDS={'rollout_mode','ppo_target_kl','ppo_max_kl','ppo_max_backtracks',
                         'critic_epochs','critic_minibatch_size','value_normalization'}


def optimizer_config_changes(previous, current, allow_upgrade=False):
    previous,current=validate_config(previous),validate_config(current)
    changes={k:dict(before=previous[k],after=current[k]) for k in current if previous[k]!=current[k]}
    if changes and (not allow_upgrade or not set(changes)<=OPTIMIZER_CONFIG_FIELDS):
        raise ValueError('Resume configuration differs; only explicit optimizer upgrades are permitted')
    return changes


def fixed_schedule(config, completed):
    if config.get('reward_schedule', 'fixed') != 'fixed':
        raise ValueError('Historical curriculum rewards have been removed')
    progress = min(1., max(0., completed/max(config['episodes']-1,1)))
    entropy=config['entropy_coefficient']
    if config.get('entropy_coefficient_final') is not None:
        entropy+=(config['entropy_coefficient_final']-entropy)*progress
    return dict(progress=progress,focus=0.,beta=1.,eta=config['exploration_weight'],
                entropy=entropy)


def validate_config(config):
    unknown=set(config)-set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
    config = dict(DEFAULTS, **config)
    if config['policy_architecture'] not in ('mlp',)+GRAPH_ARCHITECTURES:
        raise ValueError('Unknown policy_architecture')
    if config['policy_architecture'] in GRAPH_ARCHITECTURES and (config['action_mode']!='velocity'
            or config['rollout_mode']!='synchronous' or not config['critic_epochs']
            or not config['value_normalization']):
        raise ValueError('Map GAT requires synchronous velocity policy and separate PopArt critic')
    if config['policy_architecture'] in GRAPH_ARCHITECTURES and any(n>16 for n in config['fleet_sizes']):
        raise ValueError('Graph critic batches support at most 16 agents')
    if config['actor_input_version'] not in (1,2) or isinstance(config['actor_input_version'],bool):
        raise ValueError('actor_input_version must be 1 or 2')
    if config['actor_input_version']==2 and (config['action_mode']!='velocity' or config['safety_filter']!='cbf'):
        raise ValueError('Actor input version 2 requires the velocity policy with the CBF filter')
    batch=config['episodes_per_batch']
    if batch is not None:
        if type(batch) is not int or batch<1:
            raise ValueError('episodes_per_batch must be a positive integer or null')
        if config['rollout_mode']!='synchronous':
            raise ValueError('episodes_per_batch requires synchronous rollouts')
        if config['snapshot_interval']%batch or config['validation_interval']%batch:
            # Milestones must never cut a batch, or a short remainder batch returns.
            raise ValueError('snapshot_interval and validation_interval must be multiples of episodes_per_batch')
    if config['rollout_mode'] not in ('asynchronous','synchronous'):
        raise ValueError('Unknown rollout mode')
    if not (np.isfinite(config['ppo_target_kl']) and np.isfinite(config['ppo_max_kl'])
            and 0<config['ppo_target_kl']<=config['ppo_max_kl']):
        raise ValueError('Require finite 0 < ppo_target_kl <= ppo_max_kl')
    if type(config['ppo_max_backtracks']) is not int or not 0<=config['ppo_max_backtracks']<=12:
        raise ValueError('ppo_max_backtracks must be an integer in [0, 12]')
    if type(config['critic_epochs']) is not int or not 0<=config['critic_epochs']<=64:
        raise ValueError('critic_epochs must be an integer in [0, 64]')
    if type(config['critic_minibatch_size']) is not int or config['critic_minibatch_size']<1:
        raise ValueError('critic_minibatch_size must be a positive integer')
    if type(config['value_normalization']) is not bool or (config['value_normalization'] and not config['critic_epochs']):
        raise ValueError('value_normalization must be bool and requires critic_epochs')
    if config['critic_epochs'] and (config['rollout_mode']!='synchronous' or config['action_mode']!='velocity'):
        raise ValueError('Separate critic epochs require synchronous velocity training')
    if (isinstance(config['max_task_distance'],bool) or
            not isinstance(config['max_task_distance'],(int,float)) or
            not np.isfinite(config['max_task_distance']) or config['max_task_distance']<=0):
        raise ValueError('max_task_distance must be positive and finite')
    if config['distance_sampling'] not in ('linear','annulus_area'):
        raise ValueError('Unknown distance_sampling mode')
    if type(config['init_optimizer']) is not bool or (config['init_optimizer'] and not config['init_checkpoint']):
        raise ValueError('init_optimizer requires an initial checkpoint')
    if not config['fleet_sizes'] or any(type(n) is not int or n < 2 for n in config['fleet_sizes']):
        raise ValueError('fleet_sizes must contain integer sizes >=2')
    if len(set(config['fleet_sizes'])) != len(config['fleet_sizes']):
        raise ValueError('fleet_sizes must be unique')
    if config['action_mode'] not in ('candidate', 'velocity'):
        raise ValueError('action_mode must be candidate or velocity')
    if config['action_mode']=='velocity':
        if config['exploration_candidates'] != 0 or config['subgoal_radius'] is not None:
            raise ValueError('Velocity v3 uses no candidates or intermediate subgoals')
        if any(d > config['max_task_distance'] for d in config['task_distances']):
            raise ValueError('Velocity task distance exceeds max_task_distance')
    if config['environment_version'] != ENVIRONMENT_VERSION:
        raise ValueError('Environment version differs; start a separate corrected training run')
    if config['reward_schedule'] != 'fixed':
        raise ValueError('Invalid reward schedule')
    for name in ('exploration_weight','entropy_coefficient'):
        if not np.isfinite(config[name]) or config[name]<0:
            raise ValueError(f'{name} must be finite and nonnegative')
    final=config['entropy_coefficient_final']
    if final is not None and (isinstance(final,bool) or not isinstance(final,(int,float))
                              or not np.isfinite(final) or final<0):
        raise ValueError('entropy_coefficient_final must be null or finite and nonnegative')
    for field in ('episodes','steps','ppo_epochs','validation_interval','snapshot_interval','train_seed_count',
                  'validation_samples'):
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
    cooperative = config['policy_architecture'] in COOPERATIVE_ARCHITECTURES
    if config['communication_mode'] != ('backbone' if cooperative else 'chain') or config['safety_filter'] != 'cbf':
        raise ValueError('Training requires CBF and matching chain/v7 backbone communication')
    if cooperative and (config['actor_input_version']!=2 or config['drop']!=0.):
        raise ValueError('v7/v8 require input 2 and complete synchronous neighbor geometry (drop=0)')
    relay={k:config[k] for k in ('relay_loss','relay_delay_jitter','relay_period','relay_capacity','relay_attention')}
    if config['policy_architecture']!='gat_relay_attention_v8':
        if relay!={k:DEFAULTS[k] for k in relay}:
            raise ValueError('relay_* fields require policy_architecture gat_relay_attention_v8')
    elif (isinstance(relay['relay_loss'],bool) or not isinstance(relay['relay_loss'],(int,float))
            or not 0<=relay['relay_loss']<1
            or any(type(relay[k]) is not int for k in ('relay_delay_jitter','relay_period','relay_capacity'))
            or not 0<=relay['relay_delay_jitter']<=10 or not 1<=relay['relay_period']<=10
            or not 1<=relay['relay_capacity']<=64
            or relay['relay_attention'] not in ('full','age','hop','softmax')):
        raise ValueError('v8 relay: loss in [0,1), jitter 0..10, period 1..10, capacity 1..64 '
                         'steps/entries, attention full/age/hop/softmax')
    bridge=config['fusion_bridge_distance']
    if isinstance(bridge,bool) or not isinstance(bridge,(int,float)) or not 0<=bridge<=15:
        raise ValueError('fusion_bridge_distance must be 0 (off) up to the 15 m drone edge length')
    if config['spacing_penalty_weight']<0 or config['intervention_weight']<0:
        raise ValueError('Safety penalty weights must be nonnegative')
    if config['moving_ground'] is not True or config['reward_mode'] not in ('cadmm_exploration','route_graph','vehicle_route','velocity_task','velocity_mapping'):
        raise ValueError('Invalid ground mode or reward mode')
    if config['reward_mode']=='cadmm_exploration' and not config['moving_ground']:
        raise ValueError('CADMM reward requires the ground map receiver/controller')
    if not 0 < config['ground_speed'] <= 2 or not np.isfinite(config['path_reward_scale']) or config['path_reward_scale']<=0:
        raise ValueError('Invalid ground speed or path reward scale')
    if config['world_size'] not in (120.,1000.):
        raise ValueError('Supported training worlds are 120m and 1000m')
    distance_range = config['task_distance_range']
    if distance_range is not None:
        if (config['world_size'] != 1000. or not isinstance(distance_range, (list, tuple))
                or len(distance_range) != 2 or config['task_distances']
                or any(isinstance(d, bool) or not isinstance(d, (int,float)) or
                       not np.isfinite(d) for d in distance_range)
                or not 0 < distance_range[0] < distance_range[1] < config['world_size']-70):
            raise ValueError('Use task_distances=[] and a finite increasing task_distance_range')
        if config['action_mode']=='velocity' and distance_range[1] > config['max_task_distance']:
            raise ValueError('Velocity random task distance exceeds max_task_distance')
    if config['distance_sampling']=='annulus_area' and distance_range is None:
        raise ValueError('Annular sampling requires task_distance_range')
    if config['world_size']==1000. and distance_range is None and (not config['task_distances'] or any(
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
    if config['subgoal_radius'] is not None and (isinstance(config['subgoal_radius'],bool) or
            not isinstance(config['subgoal_radius'],(int,float)) or not 25<config['subgoal_radius']<1000):
        raise ValueError('subgoal_radius must be null or a distance between 25 and 1000 m')
    if (isinstance(config['subgoal_arrival'],bool) or not isinstance(config['subgoal_arrival'],(int,float))
            or not 0<config['subgoal_arrival']<(config['subgoal_radius'] or np.inf)):
        raise ValueError('subgoal_arrival must be positive and below subgoal_radius')
    if config['termination'] not in ('map','arrival'):
        raise ValueError('termination must be map or arrival')
    if config['termination']=='arrival' and config['world_size']!=1000.:
        raise ValueError('Arrival termination is defined for the 1000m world')
    if not np.isfinite(config['arrival_reward']) or config['arrival_reward']<0:
        raise ValueError('arrival_reward must be finite and nonnegative')
    if config['init_checkpoint'] is not None and not isinstance(config['init_checkpoint'],str):
        raise ValueError('init_checkpoint must be null or a checkpoint path')
    for name in ('route_reward_weight','time_penalty','map_success_reward',
                 'path_quality_weight','map_quality_reward'):
        if not np.isfinite(config[name]) or config[name]<0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if type(config['ground_stationary']) is not bool:
        raise ValueError('ground_stationary must be bool')
    if config['ground_stationary'] and (config['termination']!='map' or config['subgoal_radius'] is not None):
        raise ValueError('A stationary vehicle requires map termination and no subgoals')
    if config['map_success_reward'] and config['world_size']!=1000.:
        raise ValueError('map_success_reward uses the 1000m reference route')
    for name,low in (('chain_ground_range',0.),('chain_link_range',15.)):
        value=config[name]
        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float))
                                  or not low<value<=95.):
            raise ValueError(f'{name} must be null or in ({low:g}, 95] m')
    if not np.isfinite(config['chain_soft_ratio']) or not 0<config['chain_soft_ratio']<1:
        raise ValueError('chain_soft_ratio must be strictly between zero and one')
    for name in ('chain_risk_weight','chain_outward_weight','chain_missing_weight'):
        if not np.isfinite(config[name]) or config[name]<0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if config['reward_mode']=='velocity_mapping':
        if (config['action_mode']!='velocity' or config['world_size']!=1000.
                or config['termination']!='map' or config['arrival_reward']!=0.
                or config['subgoal_radius'] is not None):
            raise ValueError('Velocity mapping trains only map completion, with no arrival reward or subgoals')
    elif config['path_quality_weight'] or config['map_quality_reward']:
        raise ValueError('Physical route quality rewards require velocity_mapping')
    if type(config['mapping_reward_until_connected']) is not bool:
        raise ValueError('mapping_reward_until_connected must be bool')
    if config['mapping_reward_until_connected'] and (config['reward_mode']!='velocity_mapping'
                                                     or not config['path_quality_weight']):
        raise ValueError('Map-term hand-over requires the velocity_mapping route quality reward')
    if (not np.isfinite(config['cbf_margin_weight']) or config['cbf_margin_weight']<0
            or not np.isfinite(config['cbf_margin_scale']) or config['cbf_margin_scale']<=0):
        raise ValueError('cbf_margin_weight must be nonnegative and cbf_margin_scale positive')
    if type(config['map_improvement_steps']) is not int or config['map_improvement_steps']<0:
        raise ValueError('map_improvement_steps must be a nonnegative integer')
    if not 0<config['map_target_quality']<=1:
        raise ValueError('map_target_quality must be in (0, 1]')
    if (config['cbf_margin_weight'] or config['map_improvement_steps']) and config['reward_mode']!='velocity_mapping':
        raise ValueError('CBF margin and map improvement are velocity_mapping terms')
    if config['cbf_margin_weight']:
        # A permanent worst case of every per-step penalty (intervention <= 4)
        # must stay cheaper than a CBF failure, so failing is never attractive.
        worst=config['cbf_margin_weight']+config['time_penalty']+4*config['intervention_weight']
        if worst>=config['safety_failure_penalty']*(1-config['discount']):
            raise ValueError('Discounted worst-case step penalties must stay below the CBF failure penalty')
    if config['map_improvement_steps'] and (not config['path_quality_weight'] or config['map_quality_reward']):
        # Improvement is paid by route-quality gains; a final quality bonus would pay it twice.
        raise ValueError('Map improvement requires path_quality_weight and no map_quality_reward')
    if not np.isfinite(config['frontier_progress_weight']) or config['frontier_progress_weight']<0:
        raise ValueError('frontier_progress_weight must be finite and nonnegative')
    if config['frontier_progress_weight'] and (config['reward_mode']!='velocity_mapping'
            or config['frontier_progress_weight']>=config['map_success_reward']):
        # Progress pays at most its weight per episode; finishing the map must
        # stay worth more than any partial push toward the goal.
        raise ValueError('Frontier progress requires velocity_mapping and a weight below map_success_reward')
    for name in ('surprise_weight','surprise_weight_final','surprise_clip',
                 'surprise_episode_budget','surprise_learning_rate'):
        value=config[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if type(config['surprise_epochs']) is not int or not 1<=config['surprise_epochs']<=16:
        raise ValueError('surprise_epochs must be an integer in [1, 16]')
    if config['surprise_weight']:
        if (config['rollout_mode']!='synchronous' or config['reward_mode']!='velocity_mapping'
                or not config['critic_epochs'] or not config['value_normalization']):
            raise ValueError('Surprise requires synchronous velocity_mapping with a PopArt critic')
        if not (0<=config['surprise_weight_final']<=config['surprise_weight']
                and config['surprise_clip']>0 and config['surprise_learning_rate']>0
                and 0<config['surprise_episode_budget']<config['map_success_reward']
                and config['frontier_progress_weight']+config['surprise_episode_budget']<config['map_success_reward']):
            raise ValueError('Surprise must decay and progress plus its budget must stay below map success')
        if config['exploration_weight']:
            raise ValueError('v6 surprise replaces generic area exploration; exploration_weight must be zero')
    elif config['surprise_weight_final']:
        raise ValueError('surprise_weight_final requires surprise_weight')
    return config


def feature_version(config):
    """Feature contract: 1 candidate, 2/5 velocity MLP, 4/6 map GAT, 7 cooperative GAT, 8 relay attention."""
    if config['policy_architecture']=='gat_relay_attention_v8':
        return 8
    if config['policy_architecture']=='gat_cooperative_v7':
        return 7
    if config['action_mode']!='velocity':
        return 1
    gat=config['policy_architecture']=='gat_map_v1'
    if config['actor_input_version']==2:
        return 6 if gat else 5
    return 4 if gat else 2


def make_model(config):
    if config['policy_architecture']=='gat_relay_attention_v8':
        from drone.relay_attention_mappo import RelayActorCritic
        return RelayActorCritic(GLOBAL_FEATURES,config['actor_input_version'],config['relay_attention'])
    if config['policy_architecture']=='gat_cooperative_v7':
        from drone.cooperative_mappo import CooperativeActorCritic
        return CooperativeActorCritic(GLOBAL_FEATURES,config['actor_input_version'])
    if config['policy_architecture']=='gat_map_v1':
        from drone.graph_attention_mappo import GraphAttentionActorCritic
        return GraphAttentionActorCritic(GLOBAL_FEATURES,config['actor_input_version'])
    if config['action_mode']=='velocity':
        return VelocityActorCritic(GLOBAL_FEATURES,config['actor_input_version'])
    return ActorCritic(GLOBAL_FEATURES)


def optimizer_contract(config):
    if config['rollout_mode']!='synchronous':
        return 'legacy_async_v2'
    # v3: minibatch critic outside the actor's KL guard (optionally PopArt).
    return 'synchronous_kl_guard_v3' if config['critic_epochs'] else 'synchronous_kl_guard_v2'


def reward_contract(config):
    # v8 changes observations/transport only; its reward is v7's unchanged.
    if config['policy_architecture'] in COOPERATIVE_ARCHITECTURES:
        # Bridging changes which delivered maps count as connected (map success).
        return ('uav_mapping_route_quality_and_backbone_v7'+('_surprise' if config['surprise_weight'] else '')
                +('_bridged' if config['fusion_bridge_distance'] else ''))
    if config['reward_mode']!='velocity_mapping':
        return config['reward_mode']
    if config['surprise_weight']:
        return 'uav_mapping_route_quality_and_communication_v6_surprise'
    if config['frontier_progress_weight']:
        # v5: v4 terms plus best-so-far start-connected frontier progress toward the goal.
        return 'uav_mapping_route_quality_and_communication_v5'
    if config['cbf_margin_weight'] or config['map_improvement_steps']:
        # v4: CBF feasibility margin penalty and/or post-map improvement phase.
        return 'uav_mapping_route_quality_and_communication_v4'
    # v3: route quality alone pays for the route after its first connection.
    return ('uav_mapping_route_quality_and_communication_v3' if config['mapping_reward_until_connected']
            else 'uav_mapping_route_quality_and_communication_v2')


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
    scene=(generate_training_scene(scene_seed,config['task_distances'],
                                  distance_range=config['task_distance_range'],
                                  distance_sampling=config['distance_sampling']) if large else None)
    options={} if scene is None else dict(world_size=1000.,start=scene['start'],
        scene=(scene['goal'],scene['circles'],scene['attempts']),stop_on_ground_arrival=False)
    reference_length=(scene.get('reference_polyline_length',scene['reference_length']) if scene else None)
    if config['reward_mode']=='velocity_mapping':
        options['route_reference_length']=reference_length
    requested_budget=config['steps'] if scene is None else max(config['min_episode_steps'],
        math.ceil(config['budget_multiplier']*scene['reference_length']+config['budget_extra']))
    budget=min(config['steps'],requested_budget)
    termination=config['termination']
    with contextlib.redirect_stdout(io.StringIO()):
        env=Experiment(scene_seed,model,stochastic=stochastic,drop=config['drop'],
            exploration_weight=schedule['eta'],exploration_candidates=config['exploration_candidates'],
            communication_mode=config['communication_mode'],safety_filter=config['safety_filter'],
            spacing_penalty_weight=config['spacing_penalty_weight'],intervention_weight=config['intervention_weight'],
            moving_ground=config['moving_ground'],reward_mode=config['reward_mode'],
            ground_speed=config['ground_speed'],path_reward_scale=config['path_reward_scale'],
            subgoal_radius=config['subgoal_radius'],subgoal_arrival=config['subgoal_arrival'],
            route_reward_weight=config['route_reward_weight'],time_penalty=config['time_penalty'],
            ground_stationary=config['ground_stationary'],
            chain_soft_ratio=config['chain_soft_ratio'],chain_risk_weight=config['chain_risk_weight'],
            chain_outward_weight=config['chain_outward_weight'],chain_missing_weight=config['chain_missing_weight'],
            path_quality_weight=config['path_quality_weight'],
            mapping_until_connected=config['mapping_reward_until_connected'],
            cbf_margin_weight=config['cbf_margin_weight'],cbf_margin_scale=config['cbf_margin_scale'],
            frontier_progress_weight=config['frontier_progress_weight'],
            chain_ground_range=config['chain_ground_range'],
            chain_link_range=config['chain_link_range'],
            relay_loss=config['relay_loss'],relay_delay_jitter=config['relay_delay_jitter'],
            relay_period=config['relay_period'],relay_capacity=config['relay_capacity'],
            fusion_bridge_distance=config['fusion_bridge_distance'],
            num_drones=config['fleet_sizes'][scene_seed % len(config['fleet_sizes'])],**options)
        rollout=[];reason='budget';stable=0;max_link=0.;bonus=0.;quality_bonus=0.
        encoder=TaskMapEncoder(env) if config.get('surprise_weight',0.) else None
        surprise_transitions=[]
        improve=large and termination=='map' and config['map_improvement_steps']>0
        milestone=last_gain=-1  # frames of the first stable map and the latest best-route gain
        try:
            for frame in range(budget):
                if stop is not None and stop() and rollout:
                    reason='user_stop';break
                phase=min(stable/config['map_stability_steps'],1.)
                if milestone>=0:
                    # After the map: 1 + plateau clock, so the critic sees the time to termination.
                    phase=1.+min((frame-1-max(last_gain,milestone))/config['map_improvement_steps'],1.)
                state=env.critic_state(frame,budget,phase)
                before=encoder.encode(env) if encoder is not None else None
                try:
                    samples,reward,step_metrics=env.step(frame)
                except SafetyTermination as exc:
                    if not large:raise
                    # The proposed joint action transitions to an absorbing failure
                    # without executing infeasible velocities or relaxing safety.
                    samples=exc.samples;reward=-config['safety_failure_penalty']
                    rollout.append((*[np.array([sample[k] for sample in samples]) for k in range(4)],reward,state))
                    if encoder is not None:surprise_transitions.append(None)
                    reason='cbf_infeasible';break
                rollout.append((*[np.array([sample[k] for sample in samples]) for k in range(4)],reward,state))
                if encoder is not None:
                    surprise_transitions.append(transition(before,encoder.encode(env),
                        env.last_executed_velocities,step_metrics,frame))
                points=np.asarray(env.trace[-1]['positions'])
                ground=np.asarray(env.obs[env.ground]['self']['position'])
                ground=np.r_[ground[:2],ground[2] if len(ground)>2 else 0.]
                if config['communication_mode']=='backbone':
                    max_link=max(max_link,env.trace[-1]['chain_max_distance_ratio']*95.)
                else:
                    max_link=max(max_link,float(np.linalg.norm(points[0]-ground)),
                        float(np.linalg.norm(points[1:]-points[:-1],axis=1).max()))
                mapped=bool(env.trace[-1]['graph_task_connected'])
                if config['reward_mode']=='velocity_mapping':
                    mapped=mapped and bool(env.trace[-1]['chain_links_available']) and env.trace[-1]['chain_max_distance_ratio']<=1.
                stable=stable+1 if mapped else 0
                if progress is not None and (frame%25==0 or frame+1==budget):
                    progress(dict(step=frame+1,budget=budget,scene_seed=scene_seed,
                        world_size=config['world_size'],task_distance_m=env.initial_goal_distance,
                        graph_connected=bool(stable),ground_remaining_m=env.trace[-1]['ground_goal_distance'],
                        delivered_area_m2=float(env.ledger.seen.sum()*4),max_link_distance_m=max_link))
                if improve:
                    if env.trace[-1].get('route_quality_gain',0.)>0:last_gain=frame
                    if milestone<0 and stable>=config['map_stability_steps']:
                        # The stable map earns +100 now; the episode goes on to shorten the route.
                        milestone=frame;bonus=config['map_success_reward']
                        *head,reward,state=rollout[-1]
                        rollout[-1]=(*head,reward+bonus,state)
                    if milestone>=0:
                        if (stable>=config['map_stability_steps']
                                and env.trace[-1]['route_quality']>=config['map_target_quality']):
                            reason='map_optimal';break
                        if frame-max(last_gain,milestone)>=config['map_improvement_steps']:
                            reason='map_plateau';break
                elif large and termination=='map' and stable>=config['map_stability_steps']:
                    reason='map_connected'
                    # Fixed +100 completion and route quality are separate terms.
                    # Current quality matters even if an earlier route was better.
                    length=env.trace[-1]['graph_task_path_length']
                    if config['reward_mode']=='velocity_mapping':
                        efficiency=min(1.,reference_length/length) if length>0 else 0.
                        bonus=config['map_success_reward']
                        quality_bonus=config['map_quality_reward']*efficiency
                    else:
                        efficiency=min(1.,scene['reference_length']/max(length,1e-8))
                        bonus=config['map_success_reward']*efficiency
                    if bonus or quality_bonus:
                        *head,reward,state=rollout[-1]
                        rollout[-1]=(*head,reward+bonus+quality_bonus,state)
                    break
                if (env.done if not large else termination=='arrival' and env.goal_reached):
                    reason='vehicle_arrival'
                    if large and config['arrival_reward']:
                        # Terminal success reward on the arrival transition.
                        *head,reward,state=rollout[-1]
                        rollout[-1]=(*head,reward+config['arrival_reward'],state)
                    break
            summary=env.summary()
            if reason=='cbf_infeasible':summary['reward']-=config['safety_failure_penalty']
            if reason=='vehicle_arrival' and large:summary['reward']+=config['arrival_reward']
            if reason=='map_connected' or milestone>=0:summary['reward']+=bonus+quality_bonus
            if config['reward_mode']=='velocity_mapping':
                summary.update(uav_reward=summary['reward'],
                    map_completion_bonus=bonus,map_quality_bonus=quality_bonus,
                    route_reference_length_m=reference_length,
                    safety_failure_cost=config['safety_failure_penalty'] if reason=='cbf_infeasible' else 0.)
            if large:
                length=summary['graph_task_path_length']
                efficiency_reference=reference_length if config['reward_mode']=='velocity_mapping' else scene['reference_length']
                summary['route_efficiency']=(min(1.,efficiency_reference/length)
                                             if summary['graph_task_connected'] and length>0 else 0.)
            if improve:
                summary.update(map_completion_step=float(milestone+1 if milestone>=0 else -1),
                    map_optimal=float(reason=='map_optimal'),map_plateau=float(reason=='map_plateau'),
                    improvement_steps=float(len(rollout)-milestone-1 if milestone>=0 else 0))
            # A CBF failure after the map keeps map_completed; budget_exhausted means no map.
            summary.update(safety_terminated=float(reason=='cbf_infeasible'),
                map_completed=float(reason=='map_connected' or milestone>=0),
                budget_exhausted=float(reason=='budget' and milestone<0),
                user_stopped=float(reason=='user_stop'),episode_budget_steps=budget,
                task_distance_m=env.initial_goal_distance,max_link_distance_m=max_link,
                reference_length_m=scene['reference_length'] if scene else env.initial_goal_distance,
                rollout_steps=len(rollout))
        finally:
            env.env.close()
    if encoder is not None:
        assert len(surprise_transitions)==len(rollout)
        rollout=[(*row,item) for row,item in zip(rollout,surprise_transitions)]
    return rollout,summary


def validation_runs(config):
    """(scene seed, sample index, sample seed); identical for every checkpoint."""
    return [(seed,k,int(np.random.SeedSequence([config['seed'],seed,k]).generate_state(1)[0]))
            for seed in config['validation_seeds'] for k in range(config['validation_samples'])]


def validation(model, config):
    # Use the same fixed weights for validation throughout training.
    schedule=fixed_schedule(config,config['episodes']-1)
    records=[]
    model.eval()
    for seed,k,sample_seed in validation_runs(config):
        # Sample like training without disturbing the training random stream.
        with torch.random.fork_rng():
            torch.manual_seed(sample_seed)
            _,summary=run_episode(model,config,seed,schedule,True)
        records.append(dict(seed=seed,sample=k,**summary))
    return score_validation(records,config)


def score_validation(records, config):
    averages={key:float(np.mean([row[key] for row in records]))
              for key in records[0] if key not in ('seed','sample')}
    # Select navigation graphs by connectivity, time, efficiency and safety;
    # with arrival termination, by vehicle arrival and its time instead.
    if config['reward_mode']=='velocity_mapping':
        success=averages['map_completed']
        delays=[r.get('map_completion_step',r['episode_steps']) if r['map_completed'] else config['steps']
                for r in records]
    elif config['termination']=='arrival':
        success=averages['goal_reached']
        delays=[r['episode_steps'] if r['goal_reached'] else config['steps'] for r in records]
    else:
        success=averages['graph_task_connected']
        delays=[r['graph_first_connection_step'] if r['graph_first_connection_step']>=0
                else config['steps'] for r in records]
    score=1000*success-float(np.mean(delays))/config['steps'] \
        -.1*averages['graph_task_path_ratio']+.001*averages['reward'] \
        -10000*averages['collisions'] \
        -10*averages['chain_link_violation']
    if config['reward_mode']!='velocity_mapping':
        score-=10000*averages['ground_collisions']
    else:
        score+=20.*averages.get('route_efficiency',0.)
    if config['world_size']==1000.:
        score-=1000*averages['safety_terminated']
    return dict(score=score,means=averages,scenes=records,
        success_metric='map_completed' if config['reward_mode']=='velocity_mapping' else
                       ('goal_reached' if config['termination']=='arrival' else 'graph_task_connected'))


def _worker_init():
    # Rollout processes are single-threaded and exit together with the trainer.
    torch.set_num_threads(1)
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    if sys.platform.startswith('linux'):
        import ctypes
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)  # PR_SET_PDEATHSIG


def _rollout_task(kind, state, config, scene_seed, sample_seed, schedule, output, tag):
    """One episode on a frozen policy copy; validation returns no rollout.

    Validation samples actions exactly like training, from its fixed sample seed.
    """
    tic=time.monotonic()
    torch.manual_seed(sample_seed)
    model=ActorCritic.from_state_dict(state)
    model.train(kind=='train')
    output=Path(output)
    def progress(value):
        atomic_json(output/'running'/f'{tag}.json',dict(task=tag,
            updated_utc=datetime.now(timezone.utc).isoformat(),**value))
    rollout,summary=run_episode(model,config,scene_seed,schedule,True,
        progress=progress,stop=lambda:(output/'STOP').exists())
    return (rollout if kind=='train' else None),summary,time.monotonic()-tic


def _terminate(pool):
    for process in list((getattr(pool,'_processes',None) or {}).values()):
        process.terminate()


def episode_line(completed, config, summary, loss, ppo):
    return (f'episode {completed}/{config["episodes"]} '
        f'beta={summary["reward_beta"]:.3f} eta={summary["reward_eta"]:.3f} '
        f'confidence={summary["confidence_area"]:.1f} '
        f'task_share={summary["task_share"]:.3f} '
        f'map_done={summary.get("map_completed",0.):.0f} '
        f'route_quality={summary.get("route_efficiency",0.):.3f} '
        f'route_saved_m={summary.get("route_new_best_saving_m",0.):.1f} '
        f'frontier={summary.get("frontier_best_progress",0.):.2f} '
        f'chain_cost={sum(summary.get(k,0.) for k in ("chain_risk_cost","chain_outward_cost","chain_missing_cost")):.3f} '
        f'ground_m={summary.get("ground_distance",0.):.1f} '
        f'goal_remaining={summary.get("ground_goal_distance",0.):.1f} '
        f'cadmm_gain={summary.get("cadmm_gain",0.):.1f} loss={loss:.4f} '
        f'kl={ppo["approx_kl"]:.6f} entropy={ppo["entropy"]:.3f} '
        f'ev={ppo["explained_variance"]:.3f} ev_pre={ppo.get("explained_variance_pre",0.):.3f}'
        + (f' drift={ppo["policy_drift_speed"]:.2f}m/s coherence={ppo["policy_drift_coherence"]:.2f}'
           if 'policy_drift_speed' in ppo else ''))


def train(config, output, resume=False, workers=1, mp_context='spawn', upgrade_optimizer=False):
    if upgrade_optimizer and not resume:
        raise ValueError('Optimizer upgrade requires resume')
    if type(workers) is not int or workers<1:
        raise ValueError('workers must be a positive integer')
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
    model=make_model(config)
    initial_weights=None
    initial_optimizer=None
    if config['init_checkpoint'] is not None and not resume:
        # A new phase resets counters/RNG; curricula may retain Adam state.
        path=Path(config['init_checkpoint']).resolve()
        initial=torch.load(path,map_location='cpu',weights_only=True)
        if (initial.get('critic_global_features',0)!=GLOBAL_FEATURES or
                initial.get('feature_version')!=feature_version(config) or
                initial.get('action_mode','candidate')!=config['action_mode']):
            raise ValueError('init_checkpoint must use the current actor features and global-state critic')
        model.load_state_dict(initial['model'])
        if config['init_optimizer']:
            initial_optimizer=initial['optimizer']
        initial_weights=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             training_episodes=initial.get('training_episodes'),
                             optimizer_restored=config['init_optimizer'])
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    if initial_optimizer is not None:
        optimizer.load_state_dict(initial_optimizer)
        for group in optimizer.param_groups:
            group['lr']=config['learning_rate']
    surprise=SurpriseReward(config) if config['surprise_weight'] else None
    if surprise is not None and config['init_checkpoint'] is not None and not resume:
        if initial.get('surprise') is not None:
            surprise.load_state_dict(initial['surprise'])
    start,best_score,best_episode=0,None,None
    joint_steps=0;optimizer_batches=0;upgrade_changes={}
    if resume:
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        if saved.get('critic_global_features',0)!=GLOBAL_FEATURES or saved.get('policy_mode')!='sample':
            raise ValueError('Checkpoint predates the global-state critic and sampled validation; '
                             'start a separate training run')
        upgrade_changes=optimizer_config_changes(saved['config'],config,upgrade_optimizer)
        if upgrade_changes:
            backup=output/f"before_optimizer_upgrade_episode_{saved['training_episodes']:05d}.pt"
            if backup.exists() and backup.read_bytes()!=latest.read_bytes():
                raise ValueError('Optimizer-upgrade backup already differs')
            if not backup.exists():shutil.copy2(latest,backup)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['scene_rng']
        torch.set_rng_state(saved['torch_rng'])
        if surprise is not None:
            if saved.get('surprise') is None:raise ValueError('Missing surprise state in v6 checkpoint')
            surprise.load_state_dict(saved['surprise'])
        start=saved['training_episodes']
        optimizer_batches=saved.get('optimizer_batches',start)
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
                    workers=workers,source_sha256=source_hashes,init_checkpoint=initial_weights,
                    optimizer_contract=optimizer_contract(config),
                    optimizer_upgrade=upgrade_changes,starting_episode=start,
                    reward_contract=reward_contract(config),policy_architecture=config['policy_architecture'],
                    boundary_metric_version=2,boundary_roundoff_tolerance_m=BOUNDARY_ROUNDOFF_TOLERANCE_M)
    atomic_json(output/('resume_provenance.json' if resume else 'provenance.json'),provenance)
    if not resume or upgrade_changes:
        source_folder=f'sources_optimizer_upgrade_{start:05d}' if upgrade_changes else 'sources'
        for name in source_files:
            target=output/source_folder/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(scripts/name,target)
    started=time.monotonic()
    def status(state, completed, **extra):
        atomic_json(output/'progress.json',dict(status=state,pid=os.getpid(),completed_episodes=completed,
            total_episodes=config['episodes'],joint_steps=joint_steps,
            optimizer_batches=optimizer_batches,rollout_mode=config['rollout_mode'],
            updated_utc=datetime.now(timezone.utc).isoformat(),
            session_elapsed_seconds=round(time.monotonic()-started,2),
            best_validation_score=best_score,best_episode=best_episode,**extra))
    def checkpoint(completed):
        return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            surprise=surprise.state_dict() if surprise is not None else None,
            scene_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            training_episodes=completed,joint_steps=joint_steps,optimizer_batches=optimizer_batches,config=config,
            optimizer_contract=optimizer_contract(config),
            feature_version=feature_version(config),fov_radius=12.,
            policy_architecture=config['policy_architecture'],
            actor_input_version=config['actor_input_version'],
            action_mode=config['action_mode'],
            critic_global_features=GLOBAL_FEATURES,policy_mode='sample',
            exploration_candidates=config['exploration_candidates'],world_bounds=[0.,0.,config['world_size'],config['world_size']],
            communication_mode=config['communication_mode'],safety_filter=config['safety_filter'],
            moving_ground=config['moving_ground'],reward_mode=config['reward_mode'],
            environment_version=ENVIRONMENT_VERSION,
            reward_contract=reward_contract(config),
            ground_speed=config['ground_speed'],path_reward_scale=config['path_reward_scale'],
            exploration_weight=fixed_schedule(config,completed)['eta'],
            chain_ground_range=config['chain_ground_range'],chain_link_range=config['chain_link_range'],
            best_score=best_score,best_episode=best_episode)
    def parallel():
        nonlocal completed,joint_steps,best_score,best_episode
        running,pending=output/'running',output/'pending_validation'
        running.mkdir(exist_ok=True)
        pending.mkdir(exist_ok=True)
        for stale in running.glob('*.json'):
            stale.unlink()
        queue,tasks,snapshots,results=[],{},{},{}
        dispatched=completed
        validation_schedule=fixed_schedule(config,config['episodes']-1)
        def enqueue_validation(path):
            # Saved weights are reloaded, so later updates cannot alter them.
            snapshot=torch.load(path,map_location='cpu',weights_only=True)
            episode=snapshot['training_episodes']
            snapshots[episode],results[episode]=snapshot,{}
            queue.extend((episode,*run) for run in validation_runs(config))
        # Validations interrupted by STOP are repeated with their saved weights.
        for path in sorted(pending.glob('episode_*.pt')):
            enqueue_validation(path)
        if not resume and config['initial_validation']:
            atomic_checkpoint(pending/'episode_00000.pt',checkpoint(0))
            enqueue_validation(pending/'episode_00000.pt')
        def in_flight():
            return [dict(task=t['tag'],started_utc=t['started_utc']) for t in tasks.values()]
        def submit(pool):
            nonlocal dispatched
            started_utc=datetime.now(timezone.utc).isoformat()
            if queue:
                episode,seed,k,sample_seed=queue.pop(0)
                tag=f'validation_{episode:05d}_{seed}_{k}'
                future=pool.submit(_rollout_task,'validation',snapshots[episode]['model'],config,
                    seed,sample_seed,validation_schedule,str(output),tag)
                tasks[future]=dict(kind='validation',tag=tag,episode=episode,seed=seed,sample=k,
                                   started_utc=started_utc)
                return True
            if completed+sum(t['kind']=='train' for t in tasks.values())>=config['episodes']:
                return False
            schedule=fixed_schedule(config,dispatched)
            scene_seed=int(rng.integers(config['train_seed_start'],
                                       config['train_seed_start']+config['train_seed_count']))
            sample_seed=int(np.random.SeedSequence([config['seed'],dispatched]).generate_state(1)[0])
            dispatched+=1
            tag=f'train_{dispatched:05d}'
            state={key:value.detach().clone() for key,value in model.state_dict().items()}
            future=pool.submit(_rollout_task,'train',state,config,scene_seed,sample_seed,
                schedule,str(output),tag)
            tasks[future]=dict(kind='train',tag=tag,dispatch=dispatched,scene_seed=scene_seed,
                schedule=schedule,policy_episodes=completed,started_utc=started_utc)
            return True
        def finish(future):
            nonlocal completed,joint_steps,best_score,best_episode,optimizer_batches
            task=tasks.pop(future)
            (running/f'{task["tag"]}.json').unlink(missing_ok=True)
            rollout,summary,elapsed=future.result()
            if task['kind']=='validation':
                if summary.get('user_stopped'):
                    return
                episode=task['episode']
                results[episode][task['seed'],task['sample']]=summary
                runs=validation_runs(config)
                if len(results[episode])<len(runs):
                    return
                scenes=results.pop(episode)
                report=score_validation([dict(seed=seed,sample=k,**scenes[seed,k])
                                         for seed,k,_ in runs],config)
                snapshot=snapshots.pop(episode)
                if episode==0:
                    atomic_json(output/'initial_validation.json',report)
                else:
                    append_json(output/'validation.jsonl',dict(episode=episode,**report))
                    if best_score is None or report['score']>best_score:
                        best_score,best_episode=report['score'],episode
                        atomic_checkpoint(output/'best.pt',dict(snapshot,best_score=best_score,
                                                               best_episode=best_episode))
                        atomic_checkpoint(latest,checkpoint(completed))
                (pending/f'episode_{episode:05d}.pt').unlink()
                status('running',completed,workers=workers,in_flight=in_flight(),
                       last_validation=dict(episode=episode,**report))
                return
            loss,ppo=update(model,optimizer,rollout,epochs=config['ppo_epochs'],
                entropy_coef=task['schedule']['entropy'],return_metrics=True,
                discount=config['discount'],gae_lambda=config['gae_lambda'])
            completed+=1
            optimizer_batches+=1
            joint_steps+=len(rollout)
            # policy_lag counts updates applied after this episode's weights were copied.
            record=dict(episode=completed,scene_seed=task['scene_seed'],loss=loss,ppo=ppo,
                schedule=task['schedule'],elapsed_seconds=round(elapsed,3),dispatch=task['dispatch'],
                policy_lag=completed-1-task['policy_episodes'],**summary)
            append_json(output/'training.jsonl',record)
            atomic_checkpoint(latest,checkpoint(completed))
            if completed % config['validation_interval']==0 or completed==config['episodes']:
                atomic_checkpoint(pending/f'episode_{completed:05d}.pt',checkpoint(completed))
                enqueue_validation(pending/f'episode_{completed:05d}.pt')
            if completed % config['snapshot_interval']==0:
                atomic_checkpoint(output/f'episode_{completed:05d}.pt',checkpoint(completed))
            status('running',completed,workers=workers,in_flight=in_flight(),last_training=record)
            print(episode_line(completed,config,summary,loss,ppo)+
                  f' lag={record["policy_lag"]} seconds={elapsed:.0f}',flush=True)
        pool=ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context(mp_context),
                                 initializer=_worker_init)
        stopping=False
        try:
            while True:
                # STOP ends dispatch; running episodes return partial rollouts.
                stopping=stopping or (output/'STOP').exists()
                submitted=False
                while not stopping and len(tasks)<workers and submit(pool):
                    submitted=True
                if not tasks:
                    break
                if submitted:
                    status('running',completed,workers=workers,in_flight=in_flight())
                done,_=wait(list(tasks),timeout=5,return_when=FIRST_COMPLETED)
                for future in done:
                    finish(future)
        except BaseException:
            _terminate(pool)
            pool.shutdown(wait=False,cancel_futures=True)
            raise
        pool.shutdown()
        if stopping and (completed<config['episodes'] or queue):
            status('stopped',completed,reason='STOP file found; resume with --resume')
        else:
            status('completed',completed)
    def synchronous():
        """Frozen-policy batches; no update while any training rollout is running."""
        nonlocal completed,joint_steps,best_score,best_episode,optimizer_batches
        running,pending=output/'running',output/'pending_validation'
        running.mkdir(exist_ok=True);pending.mkdir(exist_ok=True)
        for path in running.glob('*.json'):path.unlink()
        pool=ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context(mp_context),
                                 initializer=_worker_init)

        def collect(kind,state,jobs,schedule):
            queue=list(jobs);tasks={};results=[]
            while queue or tasks:
                while queue and len(tasks)<workers and not (output/'STOP').exists():
                    task=queue.pop(0)
                    future=pool.submit(_rollout_task,kind,state,config,task['scene_seed'],
                        task['sample_seed'],schedule,str(output),task['tag'])
                    tasks[future]=task
                status('running',completed,workers=workers,phase=kind,
                    in_flight=[dict(task=t['tag'],policy_version=optimizer_batches) for t in tasks.values()])
                if not tasks:break
                done,_=wait(list(tasks),timeout=5,return_when=FIRST_COMPLETED)
                for future in done:
                    task=tasks.pop(future)
                    running.joinpath(task['tag']+'.json').unlink(missing_ok=True)
                    rollout,summary,elapsed=future.result()
                    results.append((task,rollout,summary,elapsed))
            return results, bool(queue) or (output/'STOP').exists() or any(r[2].get('user_stopped') for r in results)

        def evaluate(path):
            nonlocal best_score,best_episode
            snapshot=torch.load(path,map_location='cpu',weights_only=True)
            episode=snapshot['training_episodes']
            jobs=[dict(scene_seed=seed,sample=k,sample_seed=sample_seed,
                       tag=f'validation_{episode:05d}_{seed}_{k}') for seed,k,sample_seed in validation_runs(config)]
            results,aborted=collect('validation',snapshot['model'],jobs,fixed_schedule(config,config['episodes']-1))
            if aborted:return False
            scenes={(task['scene_seed'],task['sample']):summary for task,_,summary,_ in results}
            report=score_validation([dict(seed=seed,sample=k,**scenes[seed,k])
                                    for seed,k,_ in validation_runs(config)],config)
            if episode==0:atomic_json(output/'initial_validation.json',report)
            else:
                append_json(output/'validation.jsonl',dict(episode=episode,**report))
                if best_score is None or report['score']>best_score:
                    best_score,best_episode=report['score'],episode
                    atomic_checkpoint(output/'best.pt',dict(snapshot,best_score=best_score,best_episode=best_episode))
                    atomic_checkpoint(latest,checkpoint(completed))
            path.unlink()
            status('running',completed,workers=workers,last_validation=dict(episode=episode,**report))
            return True

        try:
            if not resume and config['initial_validation']:
                atomic_checkpoint(pending/'episode_00000.pt',checkpoint(0))
            for path in sorted(pending.glob('episode_*.pt')):
                if not evaluate(path):
                    status('stopped',completed,reason='STOP during validation; saved snapshot retained')
                    return
            while completed<config['episodes']:
                if (output/'STOP').exists():
                    status('stopped',completed,reason='STOP before next synchronous batch')
                    return
                # End batches at existing save/validation milestones, including
                # nonmultiples of workers, so 25/50/... snapshots keep their meaning.
                count=min(config['episodes_per_batch'] or workers,config['episodes']-completed,
                    config['snapshot_interval']-completed%config['snapshot_interval'],
                    config['validation_interval']-completed%config['validation_interval'])
                policy_version=optimizer_batches
                state={key:value.detach().clone() for key,value in model.state_dict().items()}
                schedule=fixed_schedule(config,completed)
                jobs=[]
                for i in range(count):
                    scene_seed=int(rng.integers(config['train_seed_start'],config['train_seed_start']+config['train_seed_count']))
                    sample_seed=int(np.random.SeedSequence([config['seed'],completed+i]).generate_state(1)[0])
                    jobs.append(dict(scene_seed=scene_seed,sample_seed=sample_seed,
                        episode=completed+i+1,tag=f'train_{completed+i+1:05d}'))
                results,aborted=collect('train',state,jobs,schedule)
                results.sort(key=lambda r:r[0]['episode'])
                if aborted:
                    # Administrative cuts are not task terminals. Do not train
                    # on the whole interrupted batch or preferentially keep its
                    # short failures; save diagnostics separately from training.
                    for task,_,summary,elapsed in results:
                        append_json(output/'discarded_rollouts.jsonl',dict(reason='user_stop_batch',
                            task=task['tag'],policy_version=policy_version,elapsed_seconds=elapsed,**summary))
                    atomic_checkpoint(latest,checkpoint(completed))
                    status('stopped',completed,reason='Interrupted batch discarded; completed weights saved')
                    return
                if len(results)!=count:raise RuntimeError('Incomplete synchronous batch')
                surprise_data=None
                if surprise is not None:
                    surprise_data=[[row[6] for row in r[1]] for r in results]
                    surprise.apply([r[1] for r in results],[r[2] for r in results],completed)
                loss,ppo=update_batch(model,optimizer,[r[1] for r in results],epochs=config['ppo_epochs'],
                    entropy_coef=schedule['entropy'],discount=config['discount'],gae_lambda=config['gae_lambda'],
                    target_kl=config['ppo_target_kl'],max_kl=config['ppo_max_kl'],
                    max_backtracks=config['ppo_max_backtracks'],critic_epochs=config['critic_epochs'],
                    critic_minibatch_size=config['critic_minibatch_size'],
                    value_normalization=config['value_normalization'],
                    critic_seed=int(np.random.SeedSequence([config['seed'],optimizer_batches]).generate_state(1)[0]))
                if surprise is not None:ppo.update(surprise.fit(surprise_data))
                optimizer_batches+=1
                for task,rollout,summary,elapsed in results:
                    completed+=1;joint_steps+=len(rollout)
                    record=dict(episode=completed,scene_seed=task['scene_seed'],loss=loss,ppo=ppo,
                        schedule=schedule,elapsed_seconds=round(elapsed,3),dispatch=task['episode'],
                        policy_lag=0,behavior_policy_version=policy_version,optimizer_batch=optimizer_batches,
                        batch_size=count,optimizer_contract=optimizer_contract(config),**summary)
                    append_json(output/'training.jsonl',record)
                atomic_checkpoint(latest,checkpoint(completed))
                if completed%config['snapshot_interval']==0:
                    atomic_checkpoint(output/f'episode_{completed:05d}.pt',checkpoint(completed))
                status('running',completed,workers=workers,last_training=record)
                print(episode_line(completed,config,record,loss,ppo)+
                    f' batch={optimizer_batches} n={count} lag=0 exact_kl={ppo["exact_kl"]:.6f}'
                    f' epochs={ppo["epochs_completed"]} rejected={ppo["kl_rejected_attempts"]}',flush=True)
                if completed%config['validation_interval']==0 or completed==config['episodes']:
                    path=pending/f'episode_{completed:05d}.pt'
                    atomic_checkpoint(path,checkpoint(completed))
                    if not evaluate(path):
                        status('stopped',completed,reason='STOP during validation; saved snapshot retained')
                        return
            status('completed',completed)
        except BaseException:
            _terminate(pool)
            pool.shutdown(wait=False,cancel_futures=True)
            raise
        finally:
            pool.shutdown()

    status('running',start)
    if not resume:
        atomic_checkpoint(latest,checkpoint(0))
    completed=start
    try:
        if config['rollout_mode']=='synchronous':
            synchronous()
            return
        if workers>1:
            parallel()
            return
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
            optimizer_batches+=1
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
            print(episode_line(completed,config,summary,loss,ppo),flush=True)
        status('completed',completed)
    except BaseException:
        status('failed',completed,error=traceback.format_exc())
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--upgrade-optimizer',action='store_true',
                        help='explicit resume-only migration of whitelisted PPO settings; keep a backup')
    parser.add_argument('--workers',type=int,default=1,
                        help='parallel rollout processes; 1 keeps the serial trainer')
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.workers>1:
        # Record the failure status and stop rollout processes on termination.
        signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    train(config,args.output,args.resume,workers=args.workers,upgrade_optimizer=args.upgrade_optimizer)


if __name__=='__main__':
    main()
