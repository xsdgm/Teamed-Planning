"""Continuous high-level target control over the unchanged frozen v9 runtime."""
from __future__ import annotations

from dataclasses import asdict,dataclass
import contextlib
import math
import numpy as np

from .navigation_evidence import AttemptMemory,EvidenceGrid,FEATURE_NAMES,features
from .frozen_navigation import received_belief,runtime,v9_archive_config


MAX_CANDIDATES=96
EXTRA_FEATURES=['is_final','is_hold','attempt_cost','remaining_budget','previous_forward',
                'previous_lateral','previous_duration','previous_arrived','previous_timeout','previous_blocked']
SLOW_FEATURES=FEATURE_NAMES+EXTRA_FEATURES
SCHEMA='vtemp2_hierarchical_candidates_v2'


@dataclass(frozen=True)
class HierarchyConfig:
    radius: float=200.
    arrival: float=25.
    option_steps: int=80
    hold_steps: int=10
    minimum_steps: int=4
    evidence_interval: int=10
    gamma: float=.999
    time_cost: float=.02
    collision_cost: float=10.
    success_reward: float=100.
    safety_failure_cost: float=100.
    budget_failure_cost: float=20.
    potential_scale: float=200.
    budget_multiplier: float=4.
    budget_extra: int=240
    min_mission_budget: int=600
    adaptive_options: bool=True
    option_travel_multiplier: float=2.
    option_exploration_steps: int=30
    max_option_steps: int=600
    stall_steps: int=80
    stall_progress_m: float=1.
    final_approach_radius: float=50.
    ground_start_hold_seconds: int=0

    def __post_init__(self):
        if not 0<self.arrival<self.radius or not 0<self.gamma<=1:
            raise ValueError('Require 0 < arrival < radius and 0 < gamma <= 1')
        for name in ('option_steps','hold_steps','minimum_steps','evidence_interval','budget_extra','min_mission_budget',
                     'option_exploration_steps','max_option_steps','stall_steps'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.ground_start_hold_seconds) is not int or self.ground_start_hold_seconds < 0:
            raise ValueError('Ground startup hold must be a nonnegative number of whole seconds')
        if self.minimum_steps>min(self.option_steps,self.hold_steps):
            raise ValueError('minimum_steps exceeds an option duration')
        for name in ('time_cost','collision_cost','success_reward','safety_failure_cost',
                     'budget_failure_cost','potential_scale','budget_multiplier','option_travel_multiplier',
                     'stall_progress_m','final_approach_radius'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.potential_scale==0 or self.budget_multiplier==0:
            raise ValueError('Potential scale and budget multiplier must be positive')
        if type(self.adaptive_options) is not bool or self.option_travel_multiplier<=0 or self.stall_progress_m<=0:
            raise ValueError('Invalid adaptive option configuration')
        if self.max_option_steps<self.option_steps or self.stall_steps<self.minimum_steps:
            raise ValueError('Adaptive option limits are shorter than the minimum execution window')


def option_duration_limit(distance,kind,config,max_speed):
    """Physical travel budget; option_steps is a floor, not a forced replan clock."""
    if kind=='hold':return config.hold_steps
    if not config.adaptive_options:return config.option_steps
    if not math.isfinite(max_speed) or max_speed<=0:
        raise ValueError('A positive UGV speed limit is required')
    tolerance=2. if kind=='final' else config.arrival
    travel=max(0.,float(distance)-tolerance)/max_speed
    return min(config.max_option_steps,max(config.option_steps,
        math.ceil(config.option_travel_multiplier*travel)+config.option_exploration_steps))


def final_approach_available(belief,config,*,memory=None,now=0.):
    """A received known route authorizes finishing, never unknown free space.

    Connectivity may follow a detour. The original ground planner and physical
    safety layers still execute the route; this is not a straight-line command.
    """
    p,goal=np.asarray(belief['position']),np.asarray(belief['goal'])
    recent_failure=memory is not None and memory.cost(p,goal,now)>=.5
    return bool(config.adaptive_options and not recent_failure
                and np.linalg.norm(goal-p)<=config.final_approach_radius
                and belief['grid'].classify(p,goal,arrival=2.)=='known_reachable')


def geometry_teacher_action(observation,position,goal):
    """Received-evidence teacher for imitation; obey exactly the policy mask."""
    valid=[int(i) for i in np.flatnonzero(observation['mask'])]
    if not valid:raise ValueError('The teacher requires a valid candidate')
    kinds,statuses=observation['kinds'],observation['statuses']
    previous=observation.get('previous')
    failed_target=(np.asarray(previous['target']) if previous is not None
                   and previous['reason'] in ('control_stall','timeout','evidence_blocked') else None)
    retry=[i for i in valid if failed_target is not None
           and np.allclose(observation['targets'][i],failed_target)]
    final=[i for i in valid if kinds[i]=='final' and statuses[i]=='known_reachable' and i not in retry]
    if final:return final[0]
    p,goal=np.asarray(position),np.asarray(goal)
    # Continue a useful commitment, but allow a new plan after an explicit
    # stall, timeout, blockage or arrival. Attempt history is not occupancy.
    if previous is not None and previous['reason']=='option_budget':
        continuation=[i for i in valid if kinds[i]=='continuation']
        if continuation:return continuation[0]
    drive=[i for i in valid if kinds[i]!='hold' and i not in retry]
    if not drive:drive=[i for i in valid if kinds[i]!='hold']
    if not drive:return valid[0]
    def cost(i):
        q=observation['targets'][i]
        attempt=float(observation['features'][i,SLOW_FEATURES.index('attempt_cost')])
        return (round(float(np.linalg.norm(q-p)+np.linalg.norm(goal-q))+20.*attempt,3),
                float(np.linalg.norm(goal-q)))
    return min(drive,key=cost)


def mission_reward(before_distance,after_distance,config,*,collisions=0,success=False,safety_failure=False,budget_failure=False):
    """One immutable final-goal objective, independent of the chosen subgoal.

    Potential-based shaping uses terminal potential zero, including failures.
    A rollout cut is not terminal and retains its nonzero next-state potential.
    """
    if sum((bool(success),bool(safety_failure),bool(budget_failure)))>1:
        raise ValueError('Conflicting terminal events')
    terminal=success or safety_failure or budget_failure
    before=-float(before_distance)/config.potential_scale
    after=0. if terminal else -float(after_distance)/config.potential_scale
    base=(-config.time_cost-config.collision_cost*collisions
          +config.success_reward*success-config.safety_failure_cost*safety_failure
          -config.budget_failure_cost*budget_failure)
    shaping=config.gamma*after-before
    return float(base+shaping),dict(base=float(base),shaping=float(shaping))


class HeldTarget:
    """Ground-controller contract: only the slow policy changes this target."""
    def __init__(self,goal,position):
        self.goal=np.asarray(goal,float).copy();self.target=np.asarray(position,float).copy()
        self.origin=self.target.copy();self.final=False
        self.selected=self.arrivals=self.failed=0

    def update(self,position,graph,now):
        return self.target,False

    def stats(self):
        return dict(subgoals_selected=float(self.selected),subgoal_arrivals=float(self.arrivals),subgoal_failures=float(self.failed))


def build_observation(belief,memory,config,*,now=0.,remaining_budget=1.,previous=None):
    p=np.asarray(belief['position'],float);goal=np.asarray(belief['goal'],float)
    grid=belief['grid'];bounds=belief.get('bounds',(0.,0.,grid.cells.shape[1]*grid.resolution,grid.cells.shape[0]*grid.resolution))
    direction=goal-p;distance=float(np.linalg.norm(direction))
    heading=math.atan2(direction[1],direction[0]);unit=direction/max(distance,1e-9)
    normal=np.array([-unit[1],unit[0]])
    points=[];kinds=[];seen=set()

    def add(point,kind):
        point=np.asarray(point,float)[:2]
        if not np.isfinite(point).all():return
        if kind!='hold' and not (bounds[0]+10<=point[0]<=bounds[2]-10 and bounds[1]+10<=point[1]<=bounds[3]-10):return
        if kind not in ('final','hold') and np.linalg.norm(point-p)<=config.arrival+1:return
        key=tuple(np.round(point/2).astype(int))
        if key in seen or len(points)>=MAX_CANDIDATES-1:return
        seen.add(key);points.append(point.copy());kinds.append(kind)

    # Rotated 200 m tasks can round just above the radius in float64.
    if distance<=config.radius+1e-6:add(goal,'final')
    if previous is not None and np.linalg.norm(np.asarray(previous['target'])-p)<=config.radius:
        add(previous['target'],'continuation')
    for radius in (config.radius,config.radius*2/3,config.radius/3):
        for angle in heading+np.arange(24)*2*np.pi/24:
            q=p+radius*np.array([np.cos(angle),np.sin(angle)])
            cell=grid.index(q)
            if cell is not None and grid.blocked[cell]:
                yy,xx=grid.region(q,20.)
                source=grid.index(p)
                component=grid.free_components[source] if source is not None else 0
                valid=(grid.free_components[yy,xx]==component) if component else np.zeros(len(yy),bool)
                if valid.any():
                    safe=np.c_[xx[valid]+.5,yy[valid]+.5]*grid.resolution+grid.origin
                    q=safe[np.argmin(np.linalg.norm(safe-q,axis=1))]
            add(q,'ring')
    nodes=np.asarray(belief.get('graph_positions',[]),float).reshape(-1,2)
    if len(nodes):
        offsets=nodes-p;distances=np.linalg.norm(offsets,axis=1)
        sectors=np.floor((np.arctan2(offsets[:,1],offsets[:,0])+np.pi)/(np.pi/4)).astype(int)%8
        for sector in range(8):
            valid=np.flatnonzero((sectors==sector)&(distances>config.arrival+1)&(distances<=config.radius))
            if len(valid):
                i=valid[np.argmin(np.linalg.norm(nodes[valid]-goal,axis=1))]
                add(nodes[i],'mapped')
    # Explicit hold remains available even when all drive candidates are masked.
    points.append(p.copy());kinds.append('hold')
    x=np.zeros((MAX_CANDIDATES,len(SLOW_FEATURES)),np.float32)
    targets=np.zeros((MAX_CANDIDATES,2),np.float64)
    mask=np.zeros(MAX_CANDIDATES,bool);present=mask.copy();statuses=[]
    previous=previous or dict(target=p,duration=0,reason='initial')
    delta=np.asarray(previous['target'])-p
    for i,(q,kind) in enumerate(zip(points,kinds)):
        base,status=features(belief,q)
        status=grid.classify(p,q,arrival=2. if kind=='final' else config.arrival)
        hold=kind=='hold'
        # Keep state features consistent with the final goal's strict tolerance.
        base[8:12]=[status=='known_reachable',status in ('blocked_within_bounds','outside_bounds'),status=='unknown',status=='source_conflict']
        x[i]=np.r_[base,[kind=='final',hold,min(memory.cost(p,q,now),5.)/5.,remaining_budget,
                        delta@unit/config.radius,delta@normal/config.radius,
                        previous['duration']/(config.max_option_steps if config.adaptive_options else config.option_steps),previous['reason']=='arrival',
                        previous['reason'] in ('timeout','control_stall'),previous['reason']=='evidence_blocked']]
        targets[i]=q;present[i]=True
        mask[i]=hold or status not in ('blocked_within_bounds','outside_bounds')
        statuses.append(status)
    contract='policy_choice'
    final=[i for i,kind in enumerate(kinds) if kind=='final' and mask[i]]
    if final and final_approach_available(belief,config,memory=memory,now=now):
        # Apply the finishing contract before action sampling/log-probability
        # evaluation. PPO therefore learns the same masked policy we execute.
        mask[:]=False;mask[final[0]]=True;contract='known_final_approach'
    return dict(features=x,mask=mask,present=present,targets=targets,kinds=kinds,statuses=statuses,
                previous=previous,decision_contract=contract)


class HierarchicalEpisode:
    """Simulator adapter. No simulator truth is passed to the slow network."""
    def __init__(self,repo,case,config=None,*,fast_seed=0,rollout_limit=1800,random_init=False,
                 v9_fast_reward=False,training_observation_protocol=None):
        self.config=config or HierarchyConfig()
        self._repo=repo
        from .wildscenes_runtime import is_raster_case
        self.raster_case=is_raster_case(case)
        self.me,self.torch,self.fast_model,options=(runtime(repo,random_init=True) if random_init else runtime(repo))
        if v9_fast_reward:
            archived=v9_archive_config(repo)
            if (archived['reward_mode']!='velocity_mapping' or archived['mapping_route_origin']!='vehicle'
                    or archived['subgoal_radius'] is not None or archived['termination']!='arrival'):
                raise ValueError('Expected the archived v9 fixed-target vehicle-route reward')
            # The historical pilot requests rolling subgoals. The official v9 run
            # supplies a single goal and spells the mapping handover differently in its CLI.
            options=dict(options,subgoal_radius=archived['subgoal_radius'],
                         mapping_until_connected=archived['mapping_reward_until_connected'])
            self.v9_arrival_reward=float(archived['arrival_reward'])
            self.v9_safety_failure_penalty=float(archived['safety_failure_penalty'])
        self.torch.manual_seed(int(fast_seed))
        raster_scope=contextlib.nullcontext()
        self.observation_protocol='native'
        if self.raster_case:
            from .wildscenes_runtime import load_forest,load_map_backend,raster_runtime
            forest=load_forest(case,repo)
            raster_scope=raster_runtime(self.me,forest,load_map_backend(repo),case['required_clearance_m'])
            self.observation_protocol='wildscenes_source_raster'
        elif case.get('kind')=='v9_random' and training_observation_protocol is not None:
            from .partial_observation import PARTIAL_OBSERVATION_PROTOCOL,partial_v9_runtime
            if training_observation_protocol!=PARTIAL_OBSERVATION_PROTOCOL:
                raise ValueError('Unsupported synthetic aerial observation protocol')
            raster_scope=partial_v9_runtime(self.me,case['seed'])
            self.observation_protocol=training_observation_protocol
        if case.get('kind') == 'v9_random' and 'circles_path' in case:
            from .scene_manifest import load_v9_circles
            circles = [(row[:2], float(row[2])) for row in load_v9_circles(case)]
        else:
            circles = case['circles']
        with self.torch.inference_mode(),raster_scope:
            self.env=self.me.Experiment(case['seed'],self.fast_model,stochastic=True,
                **dict(options,world_size=case['world_size'],start=case['start'],scene=(case['goal'],circles,1)))
        from .hierarchical_safety import install_control_safety
        install_control_safety(self.env)
        self.safety_failure_details=[]
        self.case=case;self.goal=self.env.goal.copy()
        self.manager=HeldTarget(self.goal,case['start'])
        if v9_fast_reward:
            self.manager.target=self.goal.copy()
            self.manager.final=True
        else:
            self.env.subgoals=self.manager;self.env.ground_controller.subgoals=self.manager
        self.holding=False
        self.ground_motion_filter_interventions=0
        self.ground_motion_filter_events=[]
        self._base_ground_action=self.env._ground_action
        self.env._ground_action=self._ground_action
        self.memory=AttemptMemory();self.previous=None;self.attempts=0;self.frame=0
        self.rollout_limit=int(rollout_limit)
        if self.rollout_limit<1:raise ValueError('rollout_limit must be positive')
        budget_distance=float(case.get('reference_distance_m',case['direct_distance_m']))
        if not math.isfinite(budget_distance) or budget_distance<case['direct_distance_m']-1e-6:
            raise ValueError('Mission reference distance must be finite and at least the direct distance')
        self.mission_budget=max(self.config.min_mission_budget,
            math.ceil(self.config.budget_multiplier*budget_distance+self.config.budget_extra))
        self.finished=False;self.terminated=False;self.truncated=False
        self.trajectory=[list(case['start'])];self.decisions=[]
        self.startup_position=np.asarray(self.env.obs[self.env.ground]['self']['position'][:2],float).copy()
        self.startup_hold_steps=0
        self.max_startup_hold_displacement_m=0.
        self.first_motion_time_s=None
        self.total_reward=0.;self.total_base=0.;self.total_shaping=0.;self.collisions=0
        self.discounted_reward=0.
        self._ground_memory=np.zeros_like(self.env.ledger.seen,dtype=np.int8)
        self._belief_cache=None;self._belief_time=None

    def _ground_action(self,neighbors=None):
        action=self._base_ground_action(neighbors)
        # The original planner clears its goal on arrival, including a hold at
        # the current position. The hierarchy owns the task until option end.
        action['broadcast']['target']=self.manager.target.tolist()
        if self.holding:action['motion']=(0.,0.)
        if self.raster_case:
            from .wildscenes_runtime import filter_ground_motion,load_map_backend
            observation=self.env.obs[self.env.ground]
            if observation['time_step']!=self.env.env.config.dt:
                raise ValueError('Ground filter sensor dt differs from actual simulator physics')
            robot=next(robot for robot in self.env.env.ground_robots if robot.robot_id==self.env.ground)
            action,event=filter_ground_motion(action,observation,load_map_backend(self._repo),
                clearance=self.case['required_clearance_m'],max_speed=robot.max_speed)
            if event['intervened']:
                self.ground_motion_filter_interventions+=1
                self.ground_motion_filter_events.append(dict(event,time=float(self.env.env.time)))
        return action

    def belief(self):
        if self._belief_time==self.frame and self._belief_cache is not None:return self._belief_cache
        b=received_belief(self.env)
        if self.raster_case:
            from .wildscenes_runtime import local_known_belief
            b=local_known_belief(b,self.env.obs[self.env.ground],self.env.ledger)
        raw=b['grid'].cells.copy()
        # Retain locally sensed cells after they leave the vehicle footprint.
        # Disagreement with received aerial evidence remains unknown until a
        # fresh local observation resolves it; no timeout writes this memory.
        aerial=np.zeros_like(raw)
        aerial[self.env.ledger.seen & (self.env.ledger.log_odds<0)]=-1
        aerial[self.env.ledger.seen & (self.env.ledger.log_odds>0)]=1
        p=np.asarray(b['position']);obs=self.env.obs[self.env.ground]
        local=np.asarray(obs['occupancy_map']);r=float(obs['map_resolution'])
        yy,xx=np.indices(local.shape)
        world=p+np.stack([xx+.5-local.shape[1]/2,yy+.5-local.shape[0]/2],-1)*r
        ij=np.floor(world/self.env.ledger.resolution).astype(int)
        inside=((ij>=0).all(-1)&(ij[...,0]<raw.shape[1])&(ij[...,1]<raw.shape[0]))
        ids=np.unique((ij[...,1]*raw.shape[1]+ij[...,0])[inside])
        self._ground_memory.flat[ids]=raw.flat[ids]
        combined=aerial.copy()
        combined[(aerial==0)&(self._ground_memory!=0)]=self._ground_memory[(aerial==0)&(self._ground_memory!=0)]
        conflict=(aerial!=0)&(self._ground_memory!=0)&(aerial!=self._ground_memory)
        combined[conflict]=0
        combined.flat[ids]=raw.flat[ids]
        b['grid']=EvidenceGrid(combined,self.env.ledger.resolution)
        b['bounds']=(0.,0.,self.case['world_size'],self.case['world_size'])
        # Carry the complete received road topology, with the same insertion
        # order as received_belief.graph_positions. No simulator-map queries.
        graph=self.env.ground_controller.unified_graph
        nodes=[] if graph is None else list(graph.graph.nodes)
        index={node:i for i,node in enumerate(nodes)}
        edges=[] if graph is None else [(index[a],index[z]) for a,z in graph.graph.edges]
        if graph is not None and not graph.graph.is_directed():
            edges += [(z,a) for a,z in list(edges) if a!=z]
        b['graph_edge_index']=np.asarray(edges,np.int64).reshape(-1,2).T.copy()
        b['graph_confidence']=np.asarray([] if graph is None else
            [graph.node_confidence(node) for node in nodes],np.float32)
        self._belief_cache=b;self._belief_time=self.frame
        return b

    def observation(self):
        return build_observation(self.belief(),self.memory,self.config,now=self.env.env.time,
                                 remaining_budget=max(0.,1-self.attempts/self.mission_budget),previous=self.previous)

    def execute_fixed_target(self):
        """Issue this episode's single task directly and keep the local motion planner active."""
        observation = dict(targets=[self.goal], mask=[True], kinds=['final'],
                           statuses=['assigned'], decision_contract='fixed_mission_target')
        return self.execute(observation, 0)

    def execute(self,observation,action):
        if self.finished:raise RuntimeError('Episode already ended')
        action=int(action)
        if not 0<=action<MAX_CANDIDATES or not observation['mask'][action]:
            raise ValueError('Invalid slow action')
        q=observation['targets'][action].copy();kind=observation['kinds'][action]
        p=np.asarray(self.env.obs[self.env.ground]['self']['position'][:2],float)
        origin=p.copy();best=float(np.linalg.norm(q-p));initial=best
        target_changed=not np.allclose(q,self.manager.target,atol=1e-8)
        self.manager.target=q.copy();self.manager.origin=p.copy()
        self.manager.final=bool(np.linalg.norm(q-self.goal)<1e-8)
        self.manager.selected+=1
        self.holding=kind=='hold'
        with self.torch.inference_mode():
            current_goal=self.env.ground_controller.goal_position
            if current_goal is None or not np.allclose(current_goal,q,atol=1e-8):
                self.env.ground_controller._drive_to(tuple(q))
            self.env.ground_action=self.env._ground_action()
        if not np.allclose(self.env.ground_action['broadcast']['target'],q):
            raise AssertionError('Slow target did not enter the broadcast')
        reward=0.;duration=0;reason='option_budget';heard=False
        fixed_target=observation.get('decision_contract')=='fixed_mission_target'
        limit=self.mission_budget if fixed_target else option_duration_limit(
            initial,kind,self.config,float(self.env.ground_controller.max_linear_vel))
        initial_failures=len(self.memory.records)
        progress_anchor=initial;last_progress=0
        motion_anchor=p.copy();last_motion=0
        for offset in range(limit):
            previous_distance=float(np.linalg.norm(p-self.goal))
            failed=False
            safety_before=getattr(self.env.env,'ground_safety_violations',0)
            with self.torch.inference_mode():
                startup_hold = self.env.env.time < self.config.ground_start_hold_seconds
                if kind=='hold' or startup_hold:
                    self.env.ground_action['motion']=(0.,0.)
                if startup_hold:
                    self.startup_hold_steps += 1
                try:
                    self.env.step(self.frame);self.frame+=1
                except self.me.SafetyTermination:
                    from .hierarchical_safety import failure_diagnostics
                    self.safety_failure_details=failure_diagnostics(self.env)
                    failed=True
            raster_failed=getattr(self.env.env,'ground_safety_violations',0)>safety_before
            failed=failed or raster_failed
            duration+=1;self.attempts+=1
            p=np.asarray(self.env.obs[self.env.ground]['self']['position'][:2],float)
            displacement=float(np.linalg.norm(p-self.startup_position))
            if startup_hold:
                self.max_startup_hold_displacement_m=max(self.max_startup_hold_displacement_m,displacement)
                if displacement>1e-8:
                    raise AssertionError('Ground robot moved during its startup hold')
            elif self.first_motion_time_s is None and displacement>1e-8:
                self.first_motion_time_s=float(self.env.env.time)
            distance=float(np.linalg.norm(p-self.goal))
            hit=0 if failed and not raster_failed else int(self.env.trace[-1].get('ground_collisions',0))
            success=bool(self.env.goal_reached) and not failed
            deadline=self.attempts>=self.mission_budget and not (success or failed)
            r,terms=mission_reward(previous_distance,distance,self.config,collisions=hit,
                                  success=success,safety_failure=failed,budget_failure=deadline)
            reward+=self.config.gamma**offset*r
            self.discounted_reward+=self.config.gamma**(self.attempts-1)*r
            self.total_reward+=r;self.total_base+=terms['base'];self.total_shaping+=terms['shaping'];self.collisions+=hit
            self.trajectory.append(p.tolist())
            best=min(best,float(np.linalg.norm(q-p)))
            if progress_anchor-best>=self.config.stall_progress_m:
                progress_anchor=best;last_progress=duration
            if np.linalg.norm(p-motion_anchor)>=self.config.stall_progress_m:
                motion_anchor=p.copy();last_motion=duration
            for controller in self.env.controllers:
                task=controller.navigator.chain.task()
                if task and np.allclose(task['target'][:2],q):heard=True
            if success or failed or deadline:
                self.terminated=True;self.finished=True
                reason='mission_arrival' if success else 'safety_failure' if failed else 'mission_budget'
                break
            if self.attempts>=self.rollout_limit:
                self.truncated=True;self.finished=True;reason='rollout_limit';break
            if fixed_target:
                continue
            if kind!='hold' and not self.manager.final and duration>=self.config.minimum_steps and np.linalg.norm(q-p)<=self.config.arrival:
                self.manager.arrivals+=1;reason='arrival';break
            if kind!='hold' and duration%self.config.evidence_interval==0:
                belief=self.belief()
                status=belief['grid'].classify(p,q,2. if self.manager.final else self.config.arrival)
                if status in ('blocked_within_bounds','outside_bounds'):
                    reason='evidence_blocked';break
                if not self.manager.final and final_approach_available(belief,self.config,
                                                                      memory=self.memory,now=self.env.env.time):
                    reason='final_approach';break
            if (self.config.adaptive_options and kind!='hold'
                    and duration-last_progress>=self.config.stall_steps
                    and duration-last_motion>=self.config.stall_steps):
                # A safe detour can initially increase Euclidean distance.
                # Moving options retain their travel budget; only a genuine
                # lack of physical movement and approach is a control stall.
                reason='control_stall';break
        if reason=='option_budget' and kind!='hold' and initial-best<2.:
            reason='timeout'
            self.memory.record(origin,q,self.env.env.time,'timeout')
            self.manager.failed+=1
        elif reason=='control_stall':
            self.memory.record(p,q,self.env.env.time,'control_stall')
            self.manager.failed+=1
        self.previous=dict(target=q.copy(),duration=duration,reason=reason)
        record=dict(target=q.tolist(),kind=kind,action=action,duration=duration,reason=reason,
                    reward=reward,position=p.tolist(),remaining_m=float(np.linalg.norm(p-self.goal)),
                    target_heard=heard,changed=target_changed,attempt_records_added=len(self.memory.records)-initial_failures,
                    selected_status=observation['statuses'][action],planned_duration_limit=limit,
                    decision_contract=observation.get('decision_contract','external_target'))
        self.decisions.append(record)
        if not np.array_equal(self.env.goal,self.goal):raise AssertionError('Mission goal was changed')
        return float(reward),duration,self.terminated,self.truncated,record

    def summary(self):
        from .hierarchical_safety import control_contract
        return dict(seed=self.case['seed'],kind=self.case['kind'],direct_distance_m=self.case['direct_distance_m'],
                    steps=self.frame,control_attempts=self.attempts,slow_decisions=len(self.decisions),
                    observation_protocol=self.observation_protocol,
                    ground_start_hold_seconds=self.config.ground_start_hold_seconds,
                    startup_hold_control_attempts=self.startup_hold_steps,
                    startup_hold_max_displacement_m=self.max_startup_hold_displacement_m,
                    first_ground_motion_time_s=self.first_motion_time_s,
                    reward=self.total_reward,base_reward=self.total_base,shaping_reward=self.total_shaping,
                    discounted_reward=self.discounted_reward,
                    remaining_m=float(np.linalg.norm(np.asarray(self.trajectory[-1])-self.goal)),
                    goal_reached=bool(self.env.goal_reached) and not getattr(self.env.env,'ground_safety_violations',0),
                    collisions=self.collisions,mission_budget=self.mission_budget,
                    reference_distance_m=self.case.get('reference_distance_m',self.case['direct_distance_m']),
                    ground_safety_violations=int(getattr(self.env.env,'ground_safety_violations',0)),
                    ground_motion_filter_interventions=self.ground_motion_filter_interventions,
                    ground_motion_filter_events=self.ground_motion_filter_events,
                    raster_collision_attempts=int(getattr(self.env.env,'raster_collision_attempts',0)),
                    raster_unknown_attempts=int(max(getattr(self.env.env,'raster_unknown_attempts',0),
                        getattr(self.env.env,'raster_safety_unknown_attempts',0))),
                    terminated=self.terminated,truncated=self.truncated,
                    termination=self.decisions[-1]['reason'] if self.decisions else 'not_started',
                    control_safety=control_contract(),safety_failure_details=self.safety_failure_details,
                    attempts_recorded=len(self.memory.records),decisions=self.decisions,trajectory=self.trajectory,
                    frozen_fast_policy=True,config=asdict(self.config))

    def close(self):self.env.env.close()
