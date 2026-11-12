"""V10: independent a-b local-map survey, with the real vehicle as radio anchor.

Only each UAV's own observations and actually received evidence enter its map.
The team objective takes the spatial maximum of those local maps: forwarding
one observation to additional UAVs does not multiply its value. Vehicle delivery,
route length and vehicle arrival are diagnostics, not survey rewards.
"""
import contextlib
import io
import math

import numpy as np

from drone.communication import backbone_metrics
from simulator.large_world_scenarios import generate_training_scene, generate_large_scene
from simulator.mappo_experiment import Experiment, SafetyTermination
from simulator.survey_reward_v10 import SurveyTask, LocalSurveyEvidence


def validate_survey_config(config):
    if (config['reward_mode']!='local_survey' or config['termination']!='survey'
            or config['mapping_route_origin']!='initial' or config['world_size']!=1000.
            or config['action_mode']!='velocity' or config['actor_input_version']!=2
            or config['communication_mode']!='backbone' or config['safety_filter']!='cbf'
            or not config['moving_ground'] or config['drop']!=0.
            or config['subgoal_radius'] is not None or config['init_optimizer']):
        raise ValueError('Survey needs velocity/backbone/CBF, independent endpoints and a fresh optimizer')
    unused=('arrival_reward','route_reward_weight','path_quality_weight','frontier_progress_weight',
            'exploration_weight','map_quality_reward','map_improvement_steps',
            'surprise_weight','surprise_weight_final')
    if any(config[k] for k in unused) or config['mapping_reward_until_connected']:
        raise ValueError('Local survey cannot include navigation, arrival or surprise rewards')
    for key in ('survey_half_width','survey_resolution','survey_confidence_weight'):
        value=config[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
            raise ValueError(f'{key} must be positive and finite')
    for key in ('survey_anchor_probability','survey_stationary_probability',
                'survey_cell_confidence','survey_target_confidence','survey_target_coverage'):
        value=config[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<=value<=1:
            raise ValueError(f'{key} must be in [0,1]')
    if min(config[k] for k in ('survey_cell_confidence','survey_target_confidence','survey_target_coverage'))<=0:
        raise ValueError('Survey completion thresholds must be positive')
    offset=config['survey_start_offset_range']
    if (not isinstance(offset,(list,tuple)) or len(offset)!=2
            or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in offset)
            or not 0<offset[0]<=offset[1]<config['world_size']):
        raise ValueError('survey_start_offset_range must be a positive increasing pair')
    endpoints=(config['survey_a'],config['survey_b'],config['survey_vehicle_start'])
    if any(p is not None for p in endpoints):
        if any(p is None for p in endpoints):
            raise ValueError('An explicit simulator task needs survey_a, survey_b and survey_vehicle_start')
        for point in endpoints:
            if (not isinstance(point,(list,tuple)) or len(point)!=2
                    or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in point)):
                raise ValueError('Survey coordinates must be finite 2D points')
        SurveyTask(endpoints[0],endpoints[1],config['survey_half_width'],config['world_size'],config['survey_resolution'])
        c=np.asarray(endpoints[2])
        if (c<35.).any() or (c>config['world_size']-35.).any():
            raise ValueError('Vehicle start needs 35m world-boundary clearance for its initial fleet')


def make_survey_scene(config, seed):
    """Sample independent endpoints; obstacle truth is confined to scene setup."""
    rng=np.random.default_rng(np.random.SeedSequence([seed,0xAB10]))
    if config['survey_a'] is None:
        scene=generate_training_scene(seed,config['task_distances'],
            distance_range=config['task_distance_range'],distance_sampling=config['distance_sampling'],
            ground_safety_distance=config['ground_safety_distance'])
        c,b=np.asarray(scene['start']),np.asarray(scene['goal'])
        anchored=bool(rng.random()<config['survey_anchor_probability'])
        if anchored:
            a=c.copy()
        else:
            for _ in range(100):
                angle=rng.uniform(-np.pi,np.pi)
                a=c+rng.uniform(*config['survey_start_offset_range'])*np.array([np.cos(angle),np.sin(angle)])
                if (a>=config['survey_half_width']).all() and (a<=config['world_size']-config['survey_half_width']).all():
                    break
            else:
                raise ValueError('Cannot fit survey start inside world')
    else:
        a,b,c=(np.asarray(config[k],float) for k in ('survey_a','survey_b','survey_vehicle_start'))
        delta=b-c
        if np.linalg.norm(delta)<1e-6:
            delta=a-c
        length=float(np.linalg.norm(delta))
        layout=generate_large_scene(seed,(length,),start=c,
            heading=float(np.arctan2(delta[1],delta[0])),ground_safety_distance=config['ground_safety_distance'])
        scene=dict(layout,start=c.tolist(),goal=b.tolist())
        anchored=bool(np.linalg.norm(a-c)<1e-8)
    task=SurveyTask(a,b,config['survey_half_width'],config['world_size'],config['survey_resolution'])
    count=config['fleet_sizes'][seed % len(config['fleet_sizes'])]
    # A conservative necessary range bound, not a guarantee of CBF feasibility.
    envelope=.8*count*math.sqrt((95.-.25)**2-15.**2)
    if max(np.linalg.norm(a-c),np.linalg.norm(b-c))+task.half_width>envelope:
        raise ValueError('Survey corridor exceeds the configured fleet communication envelope')
    stationary=config['ground_stationary'] or bool(rng.random()<config['survey_stationary_probability'])
    scene.update(survey_task=task,survey_anchored=anchored,survey_stationary=stationary,
                 survey_envelope_m=envelope)
    return scene


class SurveyExperiment(Experiment):
    def __init__(self, *args, survey_task, survey_confidence_weight=100.,
                 survey_cell_confidence=.8, **kwargs):
        self.survey_task=survey_task
        self.survey_confidence_weight=float(survey_confidence_weight)
        super().__init__(*args, **kwargs)
        self.local_evidence=LocalSurveyEvidence(survey_task,len(self.ids),survey_cell_confidence)
        self._observe_local_maps(0)
        # Initial knowledge is available before the first decision, not earned
        # by its action. Freeze it as the potential baseline.
        self.survey_initial_confidence=self.local_evidence.step_metrics()['survey_confidence']

    def _observe_local_maps(self, observation_frame):
        for index,rid in enumerate(self.ids):
            received=self.obs[rid]
            self.local_evidence.ingest(index,dict(source=rid,frame=observation_frame,
                position=received['self']['position'],occupancy=received.get('occupancy_map'),
                resolution=received['map_resolution'],valid=received.get('sensor_valid',True)))
            for neighbor in received.get('neighbors',()):
                for packet in (neighbor.get('broadcast') or {}).get('evidence_audits',()):
                    self.local_evidence.ingest(index,packet)

    def _ground_action(self, neighbors=None):
        action=super()._ground_action(neighbors)
        if action.get('broadcast') is not None:
            action['broadcast']['survey']=self.survey_task.to_message()
            # Actual vehicle position is untouched. The survey goal is a separate
            # task; the controller's physical navigation still uses its own state.
            action['broadcast']['target']=list(self.survey_task.b)
        return action

    def connectivity(self):
        positions={c.drone_id:np.asarray(self.obs[rid]['self']['position'],float)
                   for c,rid in zip(self.controllers,self.ids)}
        ground=np.asarray(self.obs[self.ground]['self']['position'],float)
        if len(ground)==2:ground=np.r_[ground,0.]
        return backbone_metrics(positions,ground)

    def critic_state(self, frame, budget, stability=0.):
        metrics=self.local_evidence.metrics()
        c=np.asarray(self.obs[self.ground]['self']['position'])[:2]
        drones=[np.asarray(self.obs[rid]['self']['position']) for rid in self.ids]
        separation=min(np.linalg.norm(p-q) for i,p in enumerate(drones) for q in drones[i+1:])
        _,longest=self.connectivity()
        a,b=np.asarray(self.survey_task.a),np.asarray(self.survey_task.b)
        return np.array([frame/budget,(budget-frame)/1000.,metrics['survey_confidence'],
            metrics['survey_coverage'],stability,np.linalg.norm(b-c)/350.,
            np.linalg.norm(a-c)/350.,np.linalg.norm(b-a)/350.,longest/95.,separation/100.],np.float32)

    def step(self, frame):
        samples,base_reward,metrics=super().step(frame)
        # These are per-device buffers of OWN and ACTUALLY RECEIVED observations.
        # No global map union is injected into an agent's local evidence map.
        for index in range(len(self.ids)):
            for packet in self.audit_buffers[index].values():
                self.local_evidence.ingest(index,packet)
        # Charge the information obtained at the newly executed position now.
        # Its source/frame key matches the next step's broadcast for deduplication.
        self._observe_local_maps(frame+1)
        terms=self.local_evidence.step_metrics()
        terms['survey_confidence_reward']=self.survey_confidence_weight*terms['survey_confidence_gain']
        connected,longest=self.connectivity()
        terms.update(survey_connected=float(connected),survey_disconnected_steps=float(not connected),
                     survey_max_link_distance_m=float(longest))
        reward=base_reward+terms['survey_confidence_reward']
        metrics.update(terms,reward=float(reward),zero_reward=float(abs(reward)<1e-8))
        self.trace[-1].update(metrics)
        return samples,reward,metrics

    def summary(self):
        result=super().summary()
        result.update(self.local_evidence.metrics())
        result['survey_initial_confidence']=self.survey_initial_confidence
        for key in ('survey_confidence_gain','survey_confidence_reward','survey_disconnected_steps'):
            result[key]=float(sum(row.get(key,0.) for row in self.trace))
        a,b=self.survey_task.a,self.survey_task.b
        result.update(survey_a_x=a[0],survey_a_y=a[1],survey_b_x=b[0],survey_b_y=b[1],
            survey_half_width=self.survey_task.half_width,
            survey_vehicle_initial_x=float(self.start[0]),survey_vehicle_initial_y=float(self.start[1]))
        return result


def run_survey_episode(model, config, scene_seed, schedule, stochastic, progress=None, stop=None):
    scene=make_survey_scene(config,scene_seed)
    task=scene['survey_task']
    length=float(np.linalg.norm(np.asarray(task.b)-task.a))
    budget=min(config['steps'],max(config['min_episode_steps'],
        math.ceil(config['budget_multiplier']*length+config['budget_extra'])))
    with contextlib.redirect_stdout(io.StringIO()):
        env=SurveyExperiment(scene_seed,model,survey_task=task,
            survey_confidence_weight=config['survey_confidence_weight'],
            survey_cell_confidence=config['survey_cell_confidence'],
            world_size=config['world_size'],start=scene['start'],
            scene=(scene['goal'],scene['circles'],scene['attempts']),stop_on_ground_arrival=False,
            stochastic=stochastic,drop=config['drop'],exploration_weight=0.,
            exploration_candidates=0,communication_mode='backbone',safety_filter='cbf',
            moving_ground=True,ground_stationary=scene['survey_stationary'],
            ground_speed=config['ground_speed'],ground_safety_distance=config['ground_safety_distance'],
            reward_mode='velocity_mapping',mapping_route_origin='initial',
            route_reward_weight=0.,path_quality_weight=0.,frontier_progress_weight=0.,
            time_penalty=config['time_penalty'],intervention_weight=config['intervention_weight'],
            spacing_penalty_weight=config['spacing_penalty_weight'],
            cbf_margin_weight=config['cbf_margin_weight'],cbf_margin_scale=config['cbf_margin_scale'],
            chain_soft_ratio=config['chain_soft_ratio'],chain_risk_weight=config['chain_risk_weight'],
            chain_outward_weight=config['chain_outward_weight'],chain_missing_weight=config['chain_missing_weight'],
            relay_loss=config['relay_loss'],relay_delay_jitter=config['relay_delay_jitter'],
            relay_period=config['relay_period'],relay_capacity=config['relay_capacity'],
            fusion_bridge_distance=config['fusion_bridge_distance'],
            num_drones=config['fleet_sizes'][scene_seed % len(config['fleet_sizes'])])
        rollout=[];reason='budget';stable=0;bonus=0.;max_link=0.
        try:
            if not env.connectivity()[0]:
                raise ValueError('Survey requires an initially connected UAV/vehicle network')
            for frame in range(budget):
                if stop is not None and stop() and rollout:
                    reason='user_stop';break
                state=env.critic_state(frame,budget,min(stable/config['map_stability_steps'],1.))
                try:
                    samples,reward,metrics=env.step(frame)
                except SafetyTermination as exc:
                    samples=exc.samples
                    reward=-config['safety_failure_penalty']
                    rollout.append((*[np.array([s[k] for s in samples]) for k in range(4)],reward,state))
                    reason='cbf_infeasible';break
                unsafe=(metrics['collisions'] or metrics['ground_collisions'] or metrics['out_of_bounds']
                        or not metrics['survey_connected'])
                if unsafe:
                    reward-=config['safety_failure_penalty']
                    reason='physical_failure'
                complete=(metrics['survey_confidence']>=config['survey_target_confidence']
                          and metrics['survey_coverage']>=config['survey_target_coverage']
                          and metrics['survey_connected'])
                stable=stable+1 if complete and not unsafe else 0
                if stable>=config['map_stability_steps']:
                    reason='survey_completed';bonus=config['map_success_reward'];reward+=bonus
                rollout.append((*[np.array([s[k] for s in samples]) for k in range(4)],reward,state))
                max_link=max(max_link,metrics['survey_max_link_distance_m'])
                if progress is not None and (frame%25==0 or frame+1==budget):
                    progress(dict(step=frame+1,budget=budget,scene_seed=scene_seed,
                        world_size=config['world_size'],task_distance_m=length,
                        survey_confidence=metrics['survey_confidence'],survey_coverage=metrics['survey_coverage'],
                        survey_connected=bool(metrics['survey_connected']),max_link_distance_m=max_link))
                if unsafe or reason=='survey_completed':break
            summary=env.summary()
            failed=reason in ('cbf_infeasible','physical_failure')
            summary.update(reward=float(sum(row[-2] for row in rollout)),
                uav_reward=float(sum(row[-2] for row in rollout)),
                survey_completed=float(reason=='survey_completed'),map_completed=float(reason=='survey_completed'),
                map_completion_bonus=bonus,map_completion_step=float(len(env.trace) if bonus else -1),
                safety_terminated=float(failed),safety_failure_cost=config['safety_failure_penalty'] if failed else 0.,
                budget_exhausted=float(reason=='budget'),user_stopped=float(reason=='user_stop'),
                episode_budget_steps=budget,task_distance_m=length,reference_length_m=length,
                max_link_distance_m=max_link,rollout_steps=len(rollout),route_efficiency=0.,
                survey_a_equals_initial_c=float(scene['survey_anchored']),
                survey_vehicle_stationary=float(scene['survey_stationary']),
                survey_envelope_m=scene['survey_envelope_m'])
        finally:
            env.env.close()
    return rollout,summary
