"""Persistent MAPPO training with fixed rewards and resumable checkpoints.

Training randomness, optimizer state and configuration are saved together.
Validation uses fixed reward weights and held-out seeds and samples actions
like training (validation_samples runs per seed, fixed sample seeds), so it
scores the policy being trained; checkpoints record policy_mode='sample'.
Test seeds are never used for checkpoint selection.

Synchronous mode collects a complete batch under one frozen policy, then makes
one guarded PPO update. Validation always uses the weights saved at its episode.
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
from simulator.mappo_experiment import Experiment, SafetyTermination
from common.navigation_config import DEFAULT_GROUND_CLEARANCE_M
from simulator.ppo_batch import update_batch
from simulator.large_world_scenarios import generate_training_scene
from simulator.env import BOUNDARY_ROUNDOFF_TOLERANCE_M
from simulator.scenarios import ENVIRONMENT_VERSION


DEFAULTS = dict(episodes=800, steps=4000, seed=53, learning_rate=3e-4,
    ppo_epochs=4, validation_interval=48, snapshot_interval=32,
    train_seed_start=50000, train_seed_count=8000,
    validation_seeds=[60002, 60006, 60011, 60000, 60005, 60004, 60014, 60020, 60024, 60010, 60007, 60008],validation_samples=2,
    test_seeds=[70000, 70001, 70002, 70003, 70004, 70005, 70006, 70007, 70008, 70009],
    # Keep mission and checkpoint schema fields used by retained policies.
    exploration_candidates=0, drop=0., curriculum_start=.15, curriculum_end=.70,
    beta_early=.2, beta_late=2.5, eta_early=.6, eta_late=.05,
    entropy_early=.03,entropy_late=.005,communication_mode='backbone',safety_filter='cbf',
    spacing_penalty_weight=0.,intervention_weight=0.05,moving_ground=True,
    reward_mode='velocity_mapping',ground_speed=1.,path_reward_scale=100.,
    # One obstacle clearance (m) for UAV graphs, UGV local planning and fusion bridges.
    # All versions default to 1 m; enclosed-state escape retains its 0.5 m margin.
    ground_safety_distance=DEFAULT_GROUND_CLEARANCE_M,
    reward_schedule='fixed',exploration_weight=0.0,entropy_coefficient=0.003,
    world_size=1000.0,task_distances=[],task_distance_range=[100.0, 400.0],budget_multiplier=3.,budget_extra=240,
    min_episode_steps=600,map_stability_steps=30,safety_failure_penalty=100.0,
    discount=0.995,gae_lambda=0.97,initial_validation=False,
    # Relay subgoals (ground/subgoal_manager.py), episode end at vehicle arrival
    # instead of a stable task graph, its terminal reward, and initial weights.
    subgoal_radius=None,subgoal_arrival=25.,termination='arrival',arrival_reward=100.0,init_checkpoint=None,
    # Compatibility fields present in existing policy checkpoints.
    chain_ground_range=None,chain_link_range=None,
    route_reward_weight=10.,time_penalty=0.01,map_success_reward=0.,ground_stationary=False,
    action_mode='velocity',fleet_sizes=[3],environment_version=ENVIRONMENT_VERSION,
    max_task_distance=750.0,distance_sampling='annulus_area',init_optimizer=False,
    chain_soft_ratio=.85,chain_risk_weight=0.,chain_outward_weight=0.,chain_missing_weight=0.,
    path_quality_weight=20.0,map_quality_reward=0.,
    # velocity_mapping: the start->goal map term stops once route quality pays.
    mapping_reward_until_connected=True,
    # v9 evaluates both delivered maps at the same current vehicle position and
    # rolling target. CoGraph retains the initial-origin schema.
    mapping_route_origin='vehicle',
    cbf_margin_weight=0.1,cbf_margin_scale=8.,map_improvement_steps=0,map_target_quality=1.,
    frontier_progress_weight=45.0,
    # Linear entropy decay from entropy_coefficient to this value (None: constant).
    entropy_coefficient_final=0.001,
    rollout_mode='synchronous',ppo_target_kl=.01,ppo_max_kl=.02,ppo_max_backtracks=4,
    # Synchronous velocity critic trained apart from the KL-guarded actor step
    # (0: legacy shared loss), optionally on PopArt-normalized returns.
    critic_epochs=8,critic_minibatch_size=512,value_normalization=True,
    policy_architecture='gat_relay_attention_v8',
    actor_input_version=2,episodes_per_batch=16,
    # Keep inert fields in the configuration schema for existing checkpoints.
    surprise_weight=0.,surprise_weight_final=0.,surprise_clip=3.,
    surprise_episode_budget=5.,surprise_learning_rate=3e-4,surprise_epochs=1,
    # v8 relayed state evidence: per-link loss, extra delay jitter (steps beyond
    # the minimum one), send period (steps), entries per envelope, and the
    # attention ablation (full=v8-D, age=v8-A, hop=v8-B, softmax=v8-C).
    relay_loss=0.1,relay_delay_jitter=2,relay_period=1,relay_capacity=16,relay_attention='full',
    # Vehicle-side fusion joins graph components within this distance (m) when the
    # delivered occupancy shows a free corridor (0: off; at most the drones' 15 m edge).
    fusion_bridge_distance=15.0,
    # CoGraph (policy_architecture cograph_v1): vehicles in the one team and the
    # inclusive range of task targets per episode (null for single-target runs).
    vehicle_count=1,target_count_range=None,
    # CoGraph v2: 'single_task' scenes put every target under the drone network
    # at once (one task; 'any' is v1's sampling); 'decentralized' vehicles each
    # claim, broadcast and drive to their own targets ('parked' is v1's stationary
    # team): a vehicle departs once its own graph reaches vehicle_depart_progress
    # of the way, and a target passes to a vehicle cheaper by vehicle_transfer_margin;
    # 'tree' budgets by the drones' coverage tree instead of the sum of target routes.
    scene_regime='any',vehicle_planner='parked',budget_reference='sum',
    vehicle_depart_progress=.5,vehicle_transfer_margin=.15,
    # 'link_keeping': each decentralized vehicle's onboard filter keeps its own
    # backbone links within vehicle_keep_range (m, 3D, inside the drones' CBF
    # edge) and yields to other vehicles (ground/vehicle_motion.py); 'free' is the
    # first v2 run's unfiltered drive.
    vehicle_motion='free',vehicle_keep_range=88.,
    # CoGraph v2: compare delivered maps at the same current vehicle positions.
    # Omitted fields preserve historical checkpoints and reward trajectories.
    cograph_route_reward='legacy',
    # v10: local UAV confidence over an independent a-b survey corridor.
    survey_a=None,survey_b=None,survey_vehicle_start=None,survey_half_width=12.,
    survey_resolution=2.,survey_start_offset_range=[20.,80.],survey_anchor_probability=.25,
    survey_stationary_probability=.5,survey_confidence_weight=100.,
    survey_cell_confidence=.8,survey_target_confidence=.85,survey_target_coverage=.9,
    # v11: choose an actual A->B distance band once per scene seed before
    # feasibility resampling. None retains the unrestricted legacy sampler.
    route_ab_distance_bands=None)

# cograph_v2 is v1 plus the learned per-drone target choice (drone/cograph_v2_mappo.py).
COGRAPH_ARCHITECTURES=('cograph_v1','cograph_v2')
# The drones' CBF keeps a link below comm range minus its margin (95 - 0.25 m).
COGRAPH_LINK_EDGE_M=94.75
RELAY_ARCHITECTURES=('gat_relay_attention_v8','gat_survey_v10','gat_route_ab_v11')+COGRAPH_ARCHITECTURES

OPTIMIZER_CONFIG_FIELDS={'rollout_mode','ppo_target_kl','ppo_max_kl','ppo_max_backtracks',
                         'critic_epochs','critic_minibatch_size','value_normalization'}


def optimizer_config_changes(previous, current, allow_upgrade=False):
    # A checkpoint without this field predates the shared 1 m default. Do not
    # silently change its physical contract when resuming under newer code.
    previous = dict(previous)
    previous.setdefault('ground_safety_distance', 1.5)
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
    # Historical CoGraph and survey checkpoints omit the fixed-start field.
    # Relay checkpoints with no origin describe the retired fixed-start task.
    if 'policy_architecture' in config and 'mapping_route_origin' not in config:
        config = dict(config, mapping_route_origin='initial')
    config = dict(DEFAULTS, **config)
    if (config['policy_architecture'] not in RELAY_ARCHITECTURES or
            (config['policy_architecture'] == 'gat_relay_attention_v8' and config['mapping_route_origin'] != 'vehicle')):
        raise ValueError('Training requires a v9, v10, v11, v11.2 or CoGraph policy configuration')
    if config['policy_architecture']=='gat_route_ab_v11':
        from simulator.route_ab_experiment import validate_route_ab_config
        validate_route_ab_config(config)
    elif config['policy_architecture']=='gat_survey_v10':
        from simulator.survey_experiment import validate_survey_config
        validate_survey_config(config)
    elif any(config[key]!=DEFAULTS[key] for key in DEFAULTS if key.startswith('survey_')):
        raise ValueError('survey_* settings require gat_survey_v10')
    if config['policy_architecture'] != 'gat_route_ab_v11' and config['route_ab_distance_bands'] is not None:
        raise ValueError('route_ab_distance_bands requires gat_route_ab_v11')
    clearance=config['ground_safety_distance']
    if (isinstance(clearance,bool) or not isinstance(clearance,(int,float))
            or not np.isfinite(clearance) or clearance<=0):
        raise ValueError('ground_safety_distance must be positive and finite')
    if config['cograph_route_reward'] not in ('legacy','paired_current'):
        raise ValueError('cograph_route_reward must be legacy or paired_current')
    if config['cograph_route_reward']=='paired_current' and (
            config['policy_architecture']!='cograph_v2' or config['vehicle_planner']!='decentralized'):
        raise ValueError('Paired current-route reward requires cograph_v2 with decentralized vehicles')
    if config['mapping_route_origin'] not in ('initial','vehicle'):
        raise ValueError('mapping_route_origin must be initial or vehicle')
    remaining_route=config['mapping_route_origin']=='vehicle'
    if config['action_mode'] != 'velocity' or config['rollout_mode'] != 'synchronous':
        raise ValueError('Training requires synchronous velocity policies')
    if config['actor_input_version'] != 2 or isinstance(config['actor_input_version'], bool):
        raise ValueError('Training requires actor_input_version 2')
    if config['value_normalization'] is not True:
        raise ValueError('Training requires a normalized critic')
    if any(n > 16 for n in config['fleet_sizes']):
        raise ValueError('Graph critic batches support at most 16 agents')
    batch=config['episodes_per_batch']
    if batch is not None:
        if type(batch) is not int or batch<1:
            raise ValueError('episodes_per_batch must be a positive integer or null')
        if config['snapshot_interval']%batch or config['validation_interval']%batch:
            # Milestones must never cut a batch, or a short remainder batch returns.
            raise ValueError('snapshot_interval and validation_interval must be multiples of episodes_per_batch')
    if not (np.isfinite(config['ppo_target_kl']) and np.isfinite(config['ppo_max_kl'])
            and 0<config['ppo_target_kl']<=config['ppo_max_kl']):
        raise ValueError('Require finite 0 < ppo_target_kl <= ppo_max_kl')
    if type(config['ppo_max_backtracks']) is not int or not 0<=config['ppo_max_backtracks']<=12:
        raise ValueError('ppo_max_backtracks must be an integer in [0, 12]')
    if type(config['critic_epochs']) is not int or not 1<=config['critic_epochs']<=64:
        raise ValueError('critic_epochs must be an integer in [1, 64]')
    if type(config['critic_minibatch_size']) is not int or config['critic_minibatch_size']<1:
        raise ValueError('critic_minibatch_size must be a positive integer')
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
    if config['exploration_candidates'] != 0 or (config['subgoal_radius'] is not None and not remaining_route):
        raise ValueError('Velocity policies use no candidates; rolling subgoals require v9')
    if any(d > config['max_task_distance'] for d in config['task_distances']):
        raise ValueError('Task distance exceeds max_task_distance')
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
    if config['communication_mode'] != 'backbone' or config['safety_filter'] != 'cbf' or config['drop'] != 0.:
        raise ValueError('Training requires the communication backbone, CBF and synchronous neighbour geometry')
    relay={k:config[k] for k in ('relay_loss','relay_delay_jitter','relay_period','relay_capacity','relay_attention')}
    if config['policy_architecture'] in COGRAPH_ARCHITECTURES:
        counts=config['target_count_range']
        if (not isinstance(counts,(list,tuple)) or len(counts)!=2 or any(type(n) is not int for n in counts)
                or not 1<=counts[0]<=counts[1]<=8):
            raise ValueError('CoGraph target_count_range must be two integers within 1..8')
        if type(config['vehicle_count']) is not int or not 1<=config['vehicle_count']<=3:
            raise ValueError('CoGraph v1 has a start layout for 1..3 vehicles')
        if (len(config['fleet_sizes'])!=1 or config['task_distance_range'] is None
                or config['reward_mode']!='velocity_mapping'):
            raise ValueError('CoGraph: one fleet size, a task_distance_range and velocity_mapping')
        if (config['scene_regime'] not in ('any','single_task')
                or config['vehicle_planner'] not in ('parked','decentralized')
                or config['budget_reference'] not in ('sum','tree')):
            raise ValueError("CoGraph scene_regime is any|single_task, vehicle_planner parked|decentralized, "
                             "budget_reference sum|tree")
        # v1 parks the team (the option space is built from the start); v2's
        # decentralized vehicles drive themselves, so they are not stationary.
        if config['ground_stationary']!=(config['vehicle_planner']=='parked'):
            raise ValueError('CoGraph: ground_stationary must be true exactly for the parked team')
        for name,low,high in (('vehicle_depart_progress',0.,1.),('vehicle_transfer_margin',0.,1.)):
            value=config[name]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not (low<value<=high if
                    name=='vehicle_depart_progress' else low<=value<high):
                raise ValueError(f'{name} out of range')
        if (config['vehicle_motion'] not in ('free','link_keeping')
                or (config['vehicle_motion']=='link_keeping' and config['vehicle_planner']!='decentralized')):
            raise ValueError('vehicle_motion is free, or link_keeping with decentralized vehicles')
        keep=config['vehicle_keep_range']
        if isinstance(keep,bool) or not isinstance(keep,(int,float)) or not 20.<keep<COGRAPH_LINK_EDGE_M:
            raise ValueError(f'vehicle_keep_range must lie in (20, {COGRAPH_LINK_EDGE_M}) m')
        if config['vehicle_motion']=='free' and keep!=DEFAULTS['vehicle_keep_range']:
            raise ValueError('vehicle_keep_range needs vehicle_motion link_keeping')
        if (config['vehicle_planner']=='parked' and (config['vehicle_depart_progress'],
                config['vehicle_transfer_margin'])!=(DEFAULTS['vehicle_depart_progress'],DEFAULTS['vehicle_transfer_margin'])):
            raise ValueError('vehicle_depart_progress/vehicle_transfer_margin need decentralized vehicles')
        if (config['surprise_weight'] or config['route_reward_weight'] or config['exploration_weight']
                or config['map_quality_reward'] or config['init_optimizer']):
            raise ValueError('CoGraph v1 has no surprise, route-cost, area or final-quality terms and '
                             'cannot restore an optimizer of another architecture')
    elif (config['target_count_range'] is not None or config['vehicle_count']!=1
            or (config['scene_regime'],config['vehicle_planner'],config['budget_reference'])!=('any','parked','sum')
            or (config['vehicle_depart_progress'],config['vehicle_transfer_margin'])
               !=(DEFAULTS['vehicle_depart_progress'],DEFAULTS['vehicle_transfer_margin'])
            or (config['vehicle_motion'],config['vehicle_keep_range'])
               !=(DEFAULTS['vehicle_motion'],DEFAULTS['vehicle_keep_range'])):
        raise ValueError('target_count_range/vehicle_count/scene_regime/vehicle_planner/budget_reference/'
                         'vehicle_* require a CoGraph policy_architecture')
    if (isinstance(relay['relay_loss'],bool) or not isinstance(relay['relay_loss'],(int,float))
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
    if config['moving_ground'] is not True or config['reward_mode'] not in ('velocity_mapping','local_survey'):
        raise ValueError('Invalid ground mode or reward mode')
    if config['reward_mode']=='local_survey' and config['policy_architecture']!='gat_survey_v10':
        raise ValueError('local_survey requires the survey policy')
    if not 0 < config['ground_speed'] <= 2 or not np.isfinite(config['path_reward_scale']) or config['path_reward_scale']<=0:
        raise ValueError('Invalid ground speed or path reward scale')
    if config['world_size'] != 1000.:
        raise ValueError('Training world size must be 1000 m')
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
    if config['termination'] not in ('map','arrival','survey'):
        raise ValueError('termination must be map, arrival or survey')
    if config['termination']=='survey' and config['policy_architecture']!='gat_survey_v10':
        raise ValueError('survey termination requires the survey policy')
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
    if config['ground_stationary'] and (config['termination'] not in ('map','survey') or config['subgoal_radius'] is not None):
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
    if remaining_route:
        if (config['reward_mode']!='velocity_mapping' or config['action_mode']!='velocity'
                or config['world_size']!=1000. or config['termination']!='arrival'
                or config['ground_stationary']
                or config['policy_architecture'] in COGRAPH_ARCHITECTURES):
            raise ValueError('v9 remaining-route mapping requires a moving vehicle, '
                             'velocity_mapping and arrival termination in the 1000m world')
        if (config['map_success_reward'] or config['map_quality_reward']
                or config['map_improvement_steps'] or config['surprise_weight']):
            raise ValueError('v9 has no fixed-start map completion, improvement or surprise rewards')
    elif config['reward_mode']=='velocity_mapping':
        if (config['action_mode']!='velocity' or config['world_size']!=1000.
                or config['termination']!='map' or config['arrival_reward']!=0.
                or config['subgoal_radius'] is not None):
            raise ValueError('Initial-origin velocity mapping trains only map completion, with no arrival reward or subgoals')
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
    if (config['cbf_margin_weight'] or config['map_improvement_steps']) and config['reward_mode'] not in ('velocity_mapping','local_survey'):
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
    completion_reward=config['arrival_reward'] if remaining_route else config['map_success_reward']
    if config['frontier_progress_weight'] and (config['reward_mode']!='velocity_mapping'
            or config['frontier_progress_weight']>=completion_reward):
        # Legacy progress is best-so-far per episode; v9 is a signed within-step
        # map change at frozen endpoints, with arrival as its completion event.
        raise ValueError('Frontier progress requires velocity_mapping and a weight below the completion reward')
    if config['surprise_weight'] != 0 or config['surprise_weight_final'] != 0:
        raise ValueError('Surprise rewards are unsupported in retained MAPPO policies')
    return config


def critic_features(config):
    """Width of the centralized critic state (CoGraph has its own multi-target state)."""
    if config['policy_architecture'] in COGRAPH_ARCHITECTURES:
        from drone.cograph_mappo import GLOBAL_FEATURES as COGRAPH_FEATURES
        return COGRAPH_FEATURES
    return GLOBAL_FEATURES


def feature_version(config):
    return {
        'gat_relay_attention_v8': 8,
        'cograph_v1': 9,
        'cograph_v2': 10,
        'gat_survey_v10': 11,
        'gat_route_ab_v11': 12,
    }[config['policy_architecture']]


def make_model(config):
    if config['policy_architecture']=='gat_route_ab_v11':
        from drone.survey_mappo import RouteABActorCritic
        return RouteABActorCritic(GLOBAL_FEATURES,config['actor_input_version'],config['relay_attention'])
    if config['policy_architecture']=='gat_survey_v10':
        from drone.survey_mappo import SurveyActorCritic
        return SurveyActorCritic(GLOBAL_FEATURES,config['actor_input_version'],config['relay_attention'])
    if config['policy_architecture']=='cograph_v2':
        from drone.cograph_v2_mappo import CoGraphV2ActorCritic
        return CoGraphV2ActorCritic(critic_features(config),config['actor_input_version'],config['relay_attention'])
    if config['policy_architecture']=='cograph_v1':
        from drone.cograph_mappo import CoGraphActorCritic
        return CoGraphActorCritic(critic_features(config),config['actor_input_version'],config['relay_attention'])
    if config['policy_architecture']=='gat_relay_attention_v8':
        from drone.relay_attention_mappo import RelayActorCritic
        return RelayActorCritic(GLOBAL_FEATURES,config['actor_input_version'],config['relay_attention'])
    raise ValueError(f"Unknown policy architecture: {config['policy_architecture']}")


def optimizer_contract(config):
    return 'synchronous_kl_guard_v3'


def reward_contract(config):
    if config['policy_architecture']=='gat_route_ab_v11':
        # r2: feasible scene certificates, clearance-aware endpoint joins,
        # signed handover potential and independent fresh vehicle observations.
        return 'uav_fixed_ab_route_with_vehicle_backbone_v11_r2'
    if config['policy_architecture']=='gat_survey_v10':
        return 'uav_local_ab_confidence_with_vehicle_backbone_v10'
    if config.get('mapping_route_origin','initial')=='vehicle':
        return 'uav_remaining_route_mapping_v9'
    if config['policy_architecture'] in COGRAPH_ARCHITECTURES:
        # v8 terms per target, averaged over targets; success = every target connected.
        bridged='_bridged' if config['fusion_bridge_distance'] else ''
        if (config['scene_regime'],config['vehicle_planner'],config['budget_reference'])==('any','parked','sum'):
            return 'cograph_v1_multi_target_option_space'+bridged
        # CoGraph v2: same per-target terms; the episode distribution, budget and
        # vehicle behaviour differ, so checkpoints are labelled apart from v1.
        return ('cograph_v2_task_option_space'
                +('_single_task' if config['scene_regime']=='single_task' else '')
                +('_decentralized_vehicles' if config['vehicle_planner']=='decentralized' else '')
                +('_tree_budget' if config['budget_reference']=='tree' else '')
                +('_vehicle_link_keeping' if config['vehicle_motion']=='link_keeping' else '')+bridged
                +('_paired_current_route' if config.get('cograph_route_reward','legacy')=='paired_current' else ''))
    raise ValueError('Unknown reward contract')


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
    if config['policy_architecture']=='gat_route_ab_v11':
        from simulator.route_ab_experiment import run_route_ab_episode
        return run_route_ab_episode(model,config,scene_seed,schedule,stochastic,progress,stop)
    if config['policy_architecture']=='gat_survey_v10':
        from simulator.survey_experiment import run_survey_episode
        return run_survey_episode(model,config,scene_seed,schedule,stochastic,progress,stop)
    if config['policy_architecture'] in COGRAPH_ARCHITECTURES:
        from simulator.cograph_experiment import run_cograph_episode
        return run_cograph_episode(model,config,scene_seed,schedule,stochastic,progress,stop)
    scene = generate_training_scene(scene_seed, config['task_distances'],
        distance_range=config['task_distance_range'], distance_sampling=config['distance_sampling'],
        ground_safety_distance=config['ground_safety_distance'])
    reference_length = scene.get('reference_polyline_length', scene['reference_length'])
    budget = min(config['steps'], max(config['min_episode_steps'],
        math.ceil(config['budget_multiplier']*scene['reference_length']+config['budget_extra'])))
    with contextlib.redirect_stdout(io.StringIO()):
        env=Experiment(scene_seed,model,stochastic=stochastic,drop=config['drop'],
            exploration_weight=schedule['eta'],exploration_candidates=config['exploration_candidates'],
            communication_mode=config['communication_mode'],safety_filter=config['safety_filter'],
            spacing_penalty_weight=config['spacing_penalty_weight'],intervention_weight=config['intervention_weight'],
            moving_ground=config['moving_ground'],reward_mode=config['reward_mode'],
            ground_speed=config['ground_speed'],path_reward_scale=config['path_reward_scale'],
            ground_safety_distance=config.get('ground_safety_distance',DEFAULT_GROUND_CLEARANCE_M),
            subgoal_radius=config['subgoal_radius'],subgoal_arrival=config['subgoal_arrival'],
            route_reward_weight=config['route_reward_weight'],time_penalty=config['time_penalty'],
            ground_stationary=config['ground_stationary'],
            chain_soft_ratio=config['chain_soft_ratio'],chain_risk_weight=config['chain_risk_weight'],
            chain_outward_weight=config['chain_outward_weight'],chain_missing_weight=config['chain_missing_weight'],
            path_quality_weight=config['path_quality_weight'],
            mapping_until_connected=config['mapping_reward_until_connected'],
            mapping_route_origin=config.get('mapping_route_origin','initial'),
            cbf_margin_weight=config['cbf_margin_weight'],cbf_margin_scale=config['cbf_margin_scale'],
            frontier_progress_weight=config['frontier_progress_weight'],
            chain_ground_range=config['chain_ground_range'],
            chain_link_range=config['chain_link_range'],
            relay_loss=config['relay_loss'],relay_delay_jitter=config['relay_delay_jitter'],
            relay_period=config['relay_period'],relay_capacity=config['relay_capacity'],
            fusion_bridge_distance=config['fusion_bridge_distance'],
            num_drones=config['fleet_sizes'][scene_seed % len(config['fleet_sizes'])],
            world_size=config['world_size'],start=scene['start'],
            scene=(scene['goal'],scene['circles'],scene['attempts']),stop_on_ground_arrival=False)
        rollout=[];reason='budget';stable=0;max_link=0.
        try:
            for frame in range(budget):
                if stop is not None and stop() and rollout:
                    reason='user_stop';break
                phase=min(stable/config['map_stability_steps'],1.)
                state=env.critic_state(frame,budget,phase)
                try:
                    samples,reward,_=env.step(frame)
                except SafetyTermination as exc:
                    # The proposed joint action transitions to an absorbing failure
                    # without executing infeasible velocities or relaxing safety.
                    samples=exc.samples
                    reward=-config['safety_failure_penalty']
                    rollout.append((*[np.array([sample[k] for sample in samples]) for k in range(4)],reward,state))
                    reason='cbf_infeasible';break
                rollout.append((*[np.array([sample[k] for sample in samples]) for k in range(4)],reward,state))
                max_link=max(max_link,env.trace[-1]['chain_max_distance_ratio']*95.)
                mapped=(bool(env.trace[-1]['graph_task_connected'])
                        and bool(env.trace[-1]['chain_links_available'])
                        and env.trace[-1]['chain_max_distance_ratio']<=1.)
                if env.trace[-1].get('route_target_changed',0.):
                    # A newly selected rolling target has its own stability clock.
                    stable=0
                stable=stable+1 if mapped else 0
                if progress is not None and (frame%25==0 or frame+1==budget):
                    progress(dict(step=frame+1,budget=budget,scene_seed=scene_seed,
                        world_size=config['world_size'],task_distance_m=env.initial_goal_distance,
                        graph_connected=bool(stable),ground_remaining_m=env.trace[-1]['ground_goal_distance'],
                        delivered_area_m2=float(env.ledger.seen.sum()*4),max_link_distance_m=max_link))
                if env.goal_reached:
                    reason='vehicle_arrival'
                    if config['arrival_reward']:
                        # Terminal success reward on the arrival transition.
                        *head,reward,state=rollout[-1]
                        rollout[-1]=(*head,reward+config['arrival_reward'],state)
                    break
            summary=env.summary()
            if reason=='cbf_infeasible':
                summary['reward']-=config['safety_failure_penalty']
            if reason=='vehicle_arrival':
                summary['reward']+=config['arrival_reward']
            summary.update(uav_reward=summary['reward'],map_completion_bonus=0.,map_quality_bonus=0.,
                initial_route_reference_length_m=reference_length,route_efficiency=summary['route_quality'],
                safety_failure_cost=config['safety_failure_penalty'] if reason=='cbf_infeasible' else 0.,
                safety_terminated=float(reason=='cbf_infeasible'),map_completed=0.,
                budget_exhausted=float(reason=='budget'),
                user_stopped=float(reason=='user_stop'),episode_budget_steps=budget,
                task_distance_m=env.initial_goal_distance,max_link_distance_m=max_link,
                reference_length_m=scene['reference_length'],
                rollout_steps=len(rollout))
        finally:
            env.env.close()
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
    if config['policy_architecture']=='gat_route_ab_v11':
        unsafe = np.mean([bool(r['safety_terminated'] or r['survey_disconnected_steps']
            or r['collisions'] or r['ground_collisions'] or r['out_of_bounds']) for r in records])
        return dict(score=float(1000.*averages['route_completed']+20.*averages['route_efficiency']
            -10000.*unsafe-averages['episode_steps']/config['steps']),
            means=averages, scenes=records, success_metric='route_completed',
            selection_contract='fixed_ab_route_and_real_vehicle_connectivity_v11')
    if config['policy_architecture']=='gat_survey_v10':
        unsafe=sum(bool(r['collisions'] or r['ground_collisions'] or r.get('out_of_bounds',0.)
                        or r.get('survey_disconnected_steps',0.) or r.get('safety_terminated',0.)) for r in records)
        groups={}
        for anchored in (True,False):
            for stationary in (True,False):
                key=('anchored' if anchored else 'independent')+('_stationary' if stationary else '_moving')
                rows=[r for r in records if bool(r['survey_a_equals_initial_c'])==anchored
                      and bool(r['survey_vehicle_stationary'])==stationary]
                group=dict(count=len(rows))
                for metric in ('survey_completed','survey_confidence','survey_coverage','safety_terminated'):
                    group[metric]=float(np.mean([r[metric] for r in rows])) if rows else None
                group['disconnected_rate']=float(np.mean([bool(r.get('survey_disconnected_steps',0.))
                                                         for r in rows])) if rows else None
                groups[key]=group
        return dict(score=float(-10000.*unsafe+1000.*averages['survey_completed']
                        +100.*averages['survey_confidence']+10.*averages['survey_coverage']
                        -averages['episode_steps']/config['steps']),
                    means=averages,scenes=records,groups=groups,success_metric='survey_completed',
                    selection_contract='local_ab_confidence_and_vehicle_connectivity_v10')
    # Select navigation graphs by connectivity, time, efficiency and safety;
    # with arrival termination, by vehicle arrival and its time instead.
    if config['termination']=='arrival':
        success=averages['goal_reached']
        delays=[r['episode_steps'] if r['goal_reached'] else config['steps'] for r in records]
    elif config['reward_mode']=='velocity_mapping':
        success=averages['map_completed']
        delays=[r.get('map_completion_step',r['episode_steps']) if r['map_completed'] else config['steps']
                for r in records]
    else:
        success=averages['graph_task_connected']
        delays=[r['graph_first_connection_step'] if r['graph_first_connection_step']>=0
                else config['steps'] for r in records]
    score=1000*success-float(np.mean(delays))/config['steps'] \
        -.1*averages['graph_task_path_ratio']+.001*averages['reward'] \
        -10000*averages['collisions'] \
        -10*averages['chain_link_violation']
    if config['reward_mode']!='velocity_mapping' or config['termination']=='arrival':
        score-=10000*averages['ground_collisions']
    if config['reward_mode']=='velocity_mapping':
        score+=20.*averages.get('route_efficiency',0.)
    if config['world_size']==1000.:
        score-=1000*averages['safety_terminated']
    return dict(score=score,means=averages,scenes=records,
        success_metric=('goal_reached' if config['termination']=='arrival' else
            ('map_completed' if config['reward_mode']=='velocity_mapping' else 'graph_task_connected')))


def validate_resume_reward_contract(saved, config):
    """Do not mix old v11 optimizer/history with the repaired task contract.

    Loading actor weights into a new run remains a separate supported action.
    Other architectures retain their existing resume compatibility checks.
    """
    if (config['policy_architecture'] == 'gat_route_ab_v11'
            and saved.get('reward_contract') != reward_contract(config)):
        raise ValueError('V11 scene/observation/reward contract changed; start a new run '
                         '(weights-only initialization is supported), not an old-run resume')


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
    model=ActorCritic.from_state_dict(state)
    model.train(kind=='train')
    output=Path(output)
    def progress(value):
        atomic_json(output/'running'/f'{tag}.json',dict(task=tag,
            updated_utc=datetime.now(timezone.utc).isoformat(),**value))
    # Reconstruction initializes layers before loading the frozen weights and
    # consumes RNG. Seed after it so serial and worker rollouts draw the same
    # action stream for the same sample seed.
    torch.manual_seed(sample_seed)
    rollout,summary=run_episode(model,config,scene_seed,schedule,True,
        progress=progress,stop=lambda:(output/'STOP').exists())
    return (rollout if kind=='train' else None),summary,time.monotonic()-tic


def _terminate(pool):
    for process in list((getattr(pool,'_processes',None) or {}).values()):
        process.terminate()


def episode_line(completed, config, summary, loss, ppo):
    if config['policy_architecture']=='gat_route_ab_v11':
        return (f'episode {completed}/{config["episodes"]} route_done={summary["route_completed"]:.0f} '
            f'route_quality={summary["route_quality"]:.3f} '
            f'mapping_reward={summary["mapping_reward"]:.3f} '
            f'quality_reward={summary["path_quality_reward"]:.3f} '
            f'frontier_reward={summary["frontier_progress_reward"]:.3f} '
            f'reward={summary["reward"]:.3f} steps={summary["episode_steps"]:.0f} '
            f'safety_stop={summary["safety_terminated"]:.0f} loss={loss:.4f} '
            f'kl={ppo["approx_kl"]:.6f}')
    if config['policy_architecture']=='gat_survey_v10':
        return (f'episode {completed}/{config["episodes"]} survey_done={summary["survey_completed"]:.0f} '
            f'confidence={summary["survey_confidence"]:.3f} coverage={summary["survey_coverage"]:.3f} '
            f'reward={summary["reward"]:.3f} steps={summary["episode_steps"]:.0f} '
            f'safety_stop={summary["safety_terminated"]:.0f} loss={loss:.4f} '
            f'kl={ppo["approx_kl"]:.6f} entropy={ppo["entropy"]:.3f}')
    if config.get('mapping_route_origin','initial')=='vehicle':
        completion=f'arrived={summary.get("goal_reached",0.):.0f}'
        route_change=f'route_delta_m={summary.get("route_length_delta_m",0.):.1f}'
        frontier=summary.get('frontier_progress',0.)
    elif config.get('cograph_route_reward','legacy')=='paired_current':
        completion=f'map_done={summary.get("map_completed",0.):.0f}'
        route_change=f'route_delta_m={summary.get("route_length_delta_m",0.):.1f}'
        frontier=summary.get('frontier_progress',0.)
    else:
        completion=f'map_done={summary.get("map_completed",0.):.0f}'
        route_change=f'route_saved_m={summary.get("route_new_best_saving_m",0.):.1f}'
        frontier=summary.get('frontier_best_progress',0.)
    return (f'episode {completed}/{config["episodes"]} '
        f'beta={summary["reward_beta"]:.3f} eta={summary["reward_eta"]:.3f} '
        f'confidence={summary["confidence_area"]:.1f} '
        f'task_share={summary["task_share"]:.3f} '
        f'{completion} '
        f'route_quality={summary.get("route_efficiency",0.):.3f} '
        f'{route_change} '
        f'frontier={frontier:.2f} '
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
        warm=None
        if (config['policy_architecture'] in ('gat_survey_v10','gat_route_ab_v11')
                and initial.get('policy_architecture')=='gat_relay_attention_v8'):
            if (initial.get('reward_contract')!='uav_remaining_route_mapping_v9'
                    or initial['config'].get('mapping_route_origin')!='vehicle'
                    or initial['config'].get('relay_attention')!=config['relay_attention']):
                raise ValueError('Survey warm start requires a compatible v9 checkpoint')
            warm=model.warm_start_actor(initial['model'])
        elif (config['policy_architecture'] in COGRAPH_ARCHITECTURES
                and initial.get('policy_architecture')=='gat_relay_attention_v8'):
            # Cross-architecture warm start: shared v8 modules, fresh target attention
            # (and, for v2, the target selector at its nearest-unconnected prior).
            if initial.get('relay_attention',initial['config'].get('relay_attention'))!=config['relay_attention']:
                raise ValueError('v8 warm start must use the same relay_attention mode')
            warm=model.warm_start(initial['model'])
        elif (initial.get('critic_global_features',0)!=critic_features(config) or
                initial.get('feature_version')!=feature_version(config) or
                initial.get('action_mode','candidate')!=config['action_mode']):
            raise ValueError('init_checkpoint must use the current actor features and global-state critic')
        else:
            model.load_state_dict(initial['model'])
        if config['init_optimizer']:
            initial_optimizer=initial['optimizer']
        initial_weights=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             training_episodes=initial.get('training_episodes'),
                             optimizer_restored=config['init_optimizer'],
                             source_architecture=initial.get('policy_architecture'),
                             warm_start=None if warm is None else dict(copied=warm['copied'],
                                 partial=warm['partial'],fresh=warm['fresh']))
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    if initial_optimizer is not None:
        optimizer.load_state_dict(initial_optimizer)
        for group in optimizer.param_groups:
            group['lr']=config['learning_rate']
    start,best_score,best_episode=0,None,None
    joint_steps=0;optimizer_batches=0;upgrade_changes={}
    if resume:
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        validate_resume_reward_contract(saved, config)
        if saved.get('critic_global_features',0)!=critic_features(config) or saved.get('policy_mode')!='sample':
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
            surprise=None,
            scene_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            training_episodes=completed,joint_steps=joint_steps,optimizer_batches=optimizer_batches,config=config,
            optimizer_contract=optimizer_contract(config),
            feature_version=feature_version(config),fov_radius=12.,
            policy_architecture=config['policy_architecture'],
            actor_input_version=config['actor_input_version'],
            action_mode=config['action_mode'],
            critic_global_features=critic_features(config),policy_mode='sample',
            exploration_candidates=config['exploration_candidates'],world_bounds=[0.,0.,config['world_size'],config['world_size']],
            communication_mode=config['communication_mode'],safety_filter=config['safety_filter'],
            moving_ground=config['moving_ground'],reward_mode=config['reward_mode'],
            mapping_route_origin=config['mapping_route_origin'],
            cograph_route_reward=config['cograph_route_reward'],
            environment_version=ENVIRONMENT_VERSION,
            reward_contract=reward_contract(config),
            ground_speed=config['ground_speed'],path_reward_scale=config['path_reward_scale'],
            exploration_weight=fixed_schedule(config,completed)['eta'],
            chain_ground_range=config['chain_ground_range'],chain_link_range=config['chain_link_range'],
            best_score=best_score,best_episode=best_episode)
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
                loss,ppo=update_batch(model,optimizer,[r[1] for r in results],epochs=config['ppo_epochs'],
                    entropy_coef=schedule['entropy'],discount=config['discount'],gae_lambda=config['gae_lambda'],
                    target_kl=config['ppo_target_kl'],max_kl=config['ppo_max_kl'],
                    max_backtracks=config['ppo_max_backtracks'],critic_epochs=config['critic_epochs'],
                    critic_minibatch_size=config['critic_minibatch_size'],
                    value_normalization=config['value_normalization'],
                    critic_seed=int(np.random.SeedSequence([config['seed'],optimizer_batches]).generate_state(1)[0]))
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
        synchronous()
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
                        help='number of rollout worker processes')
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.workers>1:
        # Record the failure status and stop rollout processes on termination.
        signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    train(config,args.output,args.resume,workers=args.workers,upgrade_optimizer=args.upgrade_optimizer)


if __name__=='__main__':
    main()
