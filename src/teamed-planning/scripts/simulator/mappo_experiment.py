"""Train/evaluate MAPPO in the existing 2D physics, mapper and controller.

Training entry point: simulator/train_mappo.py; validation: validate_mappo_2d.py
No simulator full_state, velocity or obstacle list enters the learner/reward.
"""
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from drone.drone_main import DroneController
from drone.drone_navigator import DroneNavigator
from drone.mappo import ActorCritic, CandidateNavigator
from simulator.env import EnvConfig, MultiRobotEnv
from ground.ground_main import GroundRobotController
from simulator.scenarios import generate_scene, ENVIRONMENT_VERSION
from simulator.navigation_metrics import task_graph_metrics


def navigation_cost(graph, start, goal):
    """Original CADMM uncertainty-weighted path cost and frontier proxy.

    Empty/unknown graph uses the same 1/epsilon off-graph penalty, rather
    than zero: the first delivered graph must not be punished for existing.
    """
    if graph is None or len(graph) == 0:
        return float(np.linalg.norm(np.asarray(goal)-start)/.05)
    cost=graph.path_cost(start,goal,epsilon=.05)
    return float(cost if np.isfinite(cost) else
                 DroneNavigator._frontier_proxy(graph,start,goal,.05,20.))


def cadmm_exploration_reward(cost_before, cost_after, old_seen, seen, weight,
                             path_scale=100., area_scale=400.):
    """Two task terms, computed solely from information delivered to ground."""
    gain=float(cost_before-cost_after)
    area=float(np.count_nonzero(seen & ~old_seen)*4.)
    return gain/path_scale+weight*area/area_scale, dict(
        cadmm_gain=gain,cadmm_reward=gain/path_scale,
        exploration_area=area,exploration_reward=weight*area/area_scale)


class DeliveredEvidence:
    """Reward-only evidence ledger. Each source/frame is accepted at most once.

    The binary simulator sensor has assumed reliability .8 (not calibrated on
    real images). Bounded signed log odds preserve losses from contradictions.
    Unknown cells have confidence zero, including unobserved free locations.
    """
    def __init__(self, world=120., resolution=2.):
        self.resolution = resolution
        self.n = int(world / resolution)
        self.log_odds = np.zeros((self.n, self.n), np.float32)
        self.seen = np.zeros_like(self.log_odds, bool)
        self.received = set()
        xy = (np.arange(self.n) + .5) * resolution
        self.xy = np.stack(np.meshgrid(xy, xy), -1)

    def confidence(self):
        return np.tanh(np.abs(self.log_odds) / 2) * self.seen

    def receive(self, source, frame, position, occupancy, resolution, valid=True):
        key = (source, frame)
        if not valid or occupancy is None or key in self.received:
            return False
        self.received.add(key)
        h, w = occupancy.shape
        # Simulator grids: row Y, column X. Use the actual received pixels only.
        xs = position[0] + (np.arange(w)+.5-w/2)*resolution
        ys = position[1] + (np.arange(h)+.5-h/2)*resolution
        ix, iy = np.meshgrid(np.floor(xs/self.resolution).astype(int),
                             np.floor(ys/self.resolution).astype(int))
        valid_cells = (ix >= 0) & (iy >= 0) & (ix < self.n) & (iy < self.n)
        ids = iy[valid_cells] * self.n + ix[valid_cells]
        # One vote per world cell per source frame, even at finer sensor scale.
        unique, inverse = np.unique(ids, return_inverse=True)
        votes = np.bincount(inverse, weights=occupancy[valid_cells].astype(float))
        count = np.bincount(inverse)
        weights = np.where(votes/count >= .5, 1., -1.) * np.log(4.)
        self.log_odds.flat[unique] += weights
        np.clip(self.log_odds, -3., 3., out=self.log_odds)
        self.seen.flat[unique] = True
        return True

    def task_mask(self, start, goal, car_radius=18., goal_radius=18., band=12.):
        d = np.asarray(goal)-start
        t = np.clip(np.sum((self.xy-start)*d, axis=-1)/max(np.dot(d,d), 1e-8), 0, 1)
        corridor = np.linalg.norm(self.xy-start-t[...,None]*d, axis=-1)
        return ((np.linalg.norm(self.xy-start, axis=-1) <= car_radius) |
                (np.linalg.norm(self.xy-goal, axis=-1) <= goal_radius) | (corridor <= band))

    def connector_clear(self, start, end):
        """Endpoint access must be supported by received free pixels."""
        points=np.linspace(start,end,max(2,int(np.linalg.norm(np.asarray(end)-start)/
                                                  (self.resolution/4))+1))
        cells=np.floor(points/self.resolution).astype(int)
        if not ((cells>=0)&(cells<self.n)).all():
            return False
        x,y=cells[:,0],cells[:,1]
        return bool((self.seen[y,x] & (self.log_odds[y,x]<0)).all())


def evidence_metrics(before, after, old_seen, seen, region, area=4.):
    """Read-only evidence diagnostics; these values do not define the reward."""
    delta = (after-before)*area
    return dict(task_gain=float(delta[region].sum()),
        outside_gain=float(delta[~region].sum()),
        outside_new=float(np.count_nonzero(seen & ~old_seen & ~region)*area),
        positive_task_gain=float(np.maximum(delta[region],0).sum()),
        positive_outside_gain=float(np.maximum(delta[~region],0).sum()),
        reward_focus=0.)


class SafetyTermination(RuntimeError):
    """Rejected joint action: no unsafe physics step has been executed."""
    def __init__(self, count, samples):
        super().__init__(f'CBF infeasible for {count} agents; simulation step was not executed')
        self.samples = samples


class Experiment:
    def __init__(self, seed, model, algorithm='mappo', stochastic=False,
                 drop=0.,
                 exploration_candidates=5, communication_mode='chain', safety_filter='cbf',
                 spacing_penalty_weight=0., intervention_weight=0., moving_ground=True,
                 reward_mode='cadmm_exploration', ground_speed=1., path_reward_scale=100.,exploration_weight=None,
                 world_size=120., start=(15.,60.), scene=None, stop_on_ground_arrival=True):
        if reward_mode != 'cadmm_exploration':
            raise ValueError('Only the current cadmm_exploration reward is supported')
        self.world_size=float(world_size)
        self.start=np.asarray(start,dtype=float)
        self.stop_on_ground_arrival=bool(stop_on_ground_arrival)
        self.rng = np.random.default_rng(seed)
        self.drop = drop
        self.communication_mode=communication_mode
        self.safety_filter=safety_filter
        self.spacing_penalty_weight=spacing_penalty_weight
        self.intervention_weight=intervention_weight
        self.moving_ground=moving_ground
        self.reward_mode=reward_mode
        self.path_reward_scale=path_reward_scale
        self.exploration_weight=float(exploration_weight) if exploration_weight is not None else .6
        self.audit_buffers=[{} for _ in range(3)]
        self.latest_delivered_frames={}
        self.env = MultiRobotEnv(EnvConfig(world_size=(self.world_size,self.world_size), dt=1.,
            perception_radius=24., communication_radius=95., occupancy_map_resolution=2.,
            occupancy_map_size=12, ground_occupancy_map_size=16 if moving_ground else 4,
            ground_occupancy_map_resolution=.5 if moving_ground else None))
        self.ground = self.env.add_ground_robot(*self.start,max_speed=ground_speed)
        self.goal, circles, self.scene_attempts = generate_scene(seed) if scene is None else scene
        self.goal=np.asarray(self.goal,dtype=float)
        # Rejection/feasibility checks are environment setup only. No grid or
        # reference route enters controllers, rewards, Actor or Critic.
        for position, radius in circles:
            self.env.add_circle_obstacle(*position, radius)
        self.ids, self.controllers = [], []
        bounds=(0.,0.,self.world_size,self.world_size)
        for i, (dx,dy) in enumerate(((9.,-18.),(9.,0.),(9.,18.))):
            x,y=self.start+np.asarray([dx,dy])
            self.ids.append(self.env.add_drone(x,y,15.,max_speed=6.))
            c = DroneController(f'drone_{i}', resolution=2., total_drones=3,
                communication_range=95., max_speed=6., broadcast_interval=1., fusion_interval=5.,
                communication_mode=communication_mode,safety_filter=safety_filter,
                graph_min_clearance=1.5)
            if algorithm != 'cadmm_original':
                c.navigator = CandidateNavigator(f'drone_{i}', total_drones=3, comm_range=95.,
                    min_separation=15., fov_radius=12., max_speed=6., model=model,
                    exploration_candidates=exploration_candidates, world_bounds=bounds,
                    communication_mode=communication_mode,
                    algorithm=algorithm, deterministic=not stochastic)
            else:
                c.navigator.fov_radius = 12.  # Same actual 24m sensor footprint.
            self.controllers.append(c)
            # Surveyed world bounds are part of the simulator task contract.
            c.graph_builder.world_bounds = bounds
            c.ground_max_speed=ground_speed if moving_ground else 0.
        self.obs = self.env.reset()
        self.ground_controller=GroundRobotController('ground_0',resolution=2.,
            perception_range=8.,max_linear_vel=ground_speed,max_angular_vel=1.,
            fusion_interval=5.,safety_distance=1.5)
        self.ground_controller.set_goal(tuple(self.goal))
        self.goal_reached=False
        self.ground_action=self._ground_action() if moving_ground else None
        self.initial_goal_distance=float(np.linalg.norm(self.goal-self.start))
        self.ledger = DeliveredEvidence(world=self.world_size)
        self.algorithm = algorithm
        self.trace = []
        self.first_graph_connection_step = -1
        self._graph_metric_source = None
        self._graph_metrics = None

    @property
    def done(self):
        return self.stop_on_ground_arrival and self.moving_ground and self.goal_reached

    def mission(self):
        # Fixed surveyed task source, broadcast by the ground robot; navigation
        # held stationary in this initial exploration-only experiment.
        return dict(robot_id='ground_0',position=[*self.start,0.],target=self.goal.tolist(),
                    planned_path=[],timestamp=self.env.time)

    def _ground_action(self,neighbors=None):
        received=dict(self.obs[self.ground])
        received['neighbors']=(neighbors if neighbors is not None else
            [n for n in received['neighbors'] if self.rng.random()>=self.drop])
        # Only time is retained from simulator diagnostics; no map truth,
        # fleet positions, velocities or obstacle lists reach the controller.
        received['full_state']={'time':self.env.time}
        received.pop('obstacles',None)
        action=self.ground_controller.get_action(received)
        if np.linalg.norm(np.asarray(received['self']['position'])[:2]-self.goal)<=2.:
            action['motion']=(0.,0.)
            self.goal_reached=True
        return action

    def critic_state(self, frame, budget, stability=0.):
        """Centralized critic input at decision time; training only, never acting.

        Remaining budget, delivered coverage, the ground's own path cost and
        task-graph status, and chain geometry. No obstacle truth or reference route.
        """
        ground=np.asarray(self.obs[self.ground]['self']['position'],float)[:2]
        drones=np.asarray([self.obs[r]['self']['position'] for r in self.ids],float)[:,:2]
        links=[np.linalg.norm(drones[0]-ground)]+[np.linalg.norm(drones[i]-drones[i-1])
                                                  for i in range(1,len(drones))]
        separation=min(np.linalg.norm(drones[i]-drones[j]) for i in range(len(drones)) for j in range(i))
        scale=max(self.initial_goal_distance,1.)
        # Path cost after the previous step's map update; empty map before the first step.
        cost=self.trace[-1]['ground_path_cost'] if self.trace else scale/.05
        connected=bool(self._graph_metrics and self._graph_metrics['graph_task_connected'])
        return np.array([frame/budget,(budget-frame)/1000.,self.ledger.seen.sum()*4/1e5,
            float(connected),stability,np.linalg.norm(ground-self.goal)/scale,scale/1000.,
            min(float(cost)*.05/scale,2.),max(links)/95.,separation/100.],np.float32)

    def step(self, frame):
        before, old_seen = self.ledger.confidence().copy(), self.ledger.seen.copy()
        old_ground=np.asarray(self.obs[self.ground]['self']['position']).copy()
        region = self.ledger.task_mask(old_ground[:2], self.goal)
        actions = {self.ground: self.ground_action if self.moving_ground else
                   {'motion':np.zeros(2), 'broadcast':self.mission()}}
        samples, decision_ms = [], []
        valid_counts, selected_distances, holds = [], [], []
        old = []
        interventions=[];observed_violations=[];infeasible=0
        for index,(rid,c) in enumerate(zip(self.ids,self.controllers)):
            received = dict(self.obs[rid])
            # Packet loss applies equally to all algorithms; no actor truth fields.
            received['neighbors'] = [n for n in received['neighbors']
                                     if self.rng.random() >= self.drop]
            old.append(np.asarray(received['self']['position']).copy())
            tic = time.perf_counter()
            action = c.get_action(received)
            decision_ms.append((time.perf_counter()-tic)*1000)
            target = c.navigator._current_target
            if target is not None and self.safety_filter!='cbf':
                distance = np.linalg.norm(np.asarray(target)-old[-1])
                speed = np.linalg.norm(action['motion'])
                if speed > distance:
                    action['motion'] *= distance/max(speed, 1e-8)
            # Never mutate a command after the CBF. Fail explicitly on an
            # infeasible filter rather than silently proceeding as if safe.
            stats=c.safety_stats
            if stats.get('infeasible',False):
                infeasible+=1
            interventions.append(float(stats.get('intervention',0.))**2/36.)
            received_distance=stats.get('min_received_separation')
            observed_violations.append(float(received_distance is not None and received_distance<15.-1e-5))
            actions[rid] = action
            if isinstance(action.get('broadcast'), dict):
                # Training-only audit payload travels with the real graph packet.
                # Never consumed by any navigator or included in ROS observations.
                action['broadcast']['evidence_audit'] = dict(source=rid, frame=frame,
                    position=old[-1].copy(), occupancy=received.get('occupancy_map'),
                    resolution=received['map_resolution'], valid=received.get('sensor_valid',True))
                if self.communication_mode=='chain':
                    # Audit transport follows actual received packets, one hop
                    # per simulation step. Reward ledger de-duplicates origins.
                    buffer=self.audit_buffers[index]
                    for neighbor in received['neighbors']:
                        for packet in (neighbor.get('broadcast') or {}).get('evidence_audits',[]):
                            if frame-packet['frame']<=30:
                                buffer[(packet['source'],packet['frame'])]=packet
                    packet=action['broadcast']['evidence_audit']
                    buffer[(rid,frame)]=packet
                    self.audit_buffers[index]={k:v for k,v in buffer.items() if frame-v['frame']<=30}
                    action['broadcast']['evidence_audits']=list(self.audit_buffers[index].values())
            if getattr(c.navigator, 'last_sample', None) is not None:
                samples.append(c.navigator.last_sample)
                features, mask, selected, _ = c.navigator.last_sample
                valid_counts.append(int(mask.sum()))
                selected_distances.append(float(features[selected,2]*100.))
                holds.append(float(selected == 0))
        if infeasible:
            raise SafetyTermination(infeasible,samples)
        self.obs, info = self.env.step(actions)
        ground_received=[n for n in self.obs[self.ground]['neighbors']
                         if self.rng.random()>=self.drop]
        ground_position=np.asarray(self.obs[self.ground]['self']['position']).copy()
        # Evaluate both maps at the SAME new position: vehicle progress alone
        # must not masquerade as an observation-induced CADMM improvement.
        cost_before=navigation_cost(self.ground_controller.unified_graph,ground_position[:2],self.goal)
        if self.moving_ground:
            self.ground_action=self._ground_action(ground_received)
        cost_after=navigation_cost(self.ground_controller.unified_graph,ground_position[:2],self.goal)
        # Delivery must be present in the actual ground receiver's neighbor list.
        delivered = ground_received
        delivered_ids = {n['id'] for n in delivered}
        for neighbor in delivered:
            message=neighbor.get('broadcast') or {}
            packets=message.get('evidence_audits',[]) if self.communication_mode=='chain' else [message.get('evidence_audit')]
            for packet in packets:
                if not packet:
                    continue
                self.ledger.receive(packet['source'], packet['frame'], packet['position'],
                    packet['occupancy'], packet['resolution'], valid=packet['valid'])
                if packet['valid']:
                    source=packet['source']
                    self.latest_delivered_frames[source]=max(packet['frame'],
                        self.latest_delivered_frames.get(source,-1))
        current = np.array([self.obs[r]['self']['position'] for r in self.ids])
        graph=self.ground_controller.unified_graph
        if self._graph_metrics is None or graph is not self._graph_metric_source or frame % 5 == 0:
            self._graph_metrics=task_graph_metrics(graph,self.start,self.goal,
                                                  self.ledger.connector_clear)
            self._graph_metric_source=graph
        if self._graph_metrics['graph_task_connected'] and self.first_graph_connection_step < 0:
            self.first_graph_connection_step=frame+1
        distance = np.linalg.norm(current-np.asarray(old), axis=1)
        metrics = evidence_metrics(before, self.ledger.confidence(), old_seen,
            self.ledger.seen, region)
        reward,terms=cadmm_exploration_reward(cost_before,cost_after,old_seen,
            self.ledger.seen,self.exploration_weight,self.path_reward_scale)
        metrics.update(terms)
        metrics.update(reward_beta=1.,reward_eta=self.exploration_weight)
        # Penalties use local received safety telemetry, not evaluator truth.
        spacing_cost=self.spacing_penalty_weight*max(observed_violations,default=0.)
        intervention_cost=self.intervention_weight*float(np.mean(interventions))
        reward-=spacing_cost+intervention_cost
        # Swept distances are independent evaluation, never learner inputs.
        sep = []
        for i in range(3):
            for j in range(i):
                r = old[i]-old[j]
                v = (current[i]-old[i])-(current[j]-old[j])
                t = np.clip(-np.dot(r,v)/max(np.dot(v,v),1e-8),0,1)
                sep.append(np.linalg.norm(r+t*v))
        # Independent evaluator checks configured chain edges and swept maxima.
        chain_distances=[max(np.linalg.norm(np.asarray(old[0])-old_ground),np.linalg.norm(current[0]-ground_position))]+[
            max(np.linalg.norm(np.asarray(old[i])-old[i-1]),np.linalg.norm(current[i]-current[i-1]))
            for i in range(1,3)]
        ages=[frame-self.latest_delivered_frames.get(rid,-1) for rid in self.ids]
        packet_age_limit=3 if self.communication_mode=='chain' else 1
        metrics.update(reward=float(reward), distance=float(distance.sum()),
            spacing_cost=spacing_cost,intervention_cost=intervention_cost,
            cbf_intervention=float(np.mean(interventions)),
            chain_link_violation=float(max(chain_distances)>95.+1e-5),
            min_separation=float(min(sep)),
            simultaneous_moving=float(np.all(distance > .1)),
            moving_fraction=float(np.mean(distance > .1)),
            spacing_violation=float(min(sep) < 15.-1e-5),
            collisions=float(min(sep) < 2.),
            communication_outage=float(any(age>packet_age_limit for age in ages)),
            evidence_source_coverage=float(np.mean(np.asarray(ages)<=packet_age_limit)),
            max_evidence_age=float(max(ages)),
            received_ground_packets=len(delivered_ids & set(self.ids)),
            fallback=sum(getattr(c.navigator,'fallbacks',0) for c in self.controllers),
            decision_ms=float(np.mean(decision_ms)),
            out_of_bounds=len(info['out_of_bounds']))
        metrics.update(ground_distance=float(np.linalg.norm(ground_position-old_ground)),
            ground_goal_distance=float(np.linalg.norm(ground_position[:2]-self.goal)),
            ground_collisions=float(len(info['collisions'])),goal_reached=float(self.goal_reached),
            ground_path_cost=cost_after)
        metrics.update(self._graph_metrics)
        metrics.update(valid_actions=float(np.mean(valid_counts)) if valid_counts else 0.,
            forced_hold_fraction=float(np.mean(np.asarray(valid_counts)==1)) if valid_counts else 0.,
            hold_fraction=float(np.mean(holds)) if holds else 0.,
            selected_waypoint_distance=float(np.mean(selected_distances)) if selected_distances else 0.,
            zero_reward=float(abs(reward)<1e-8))
        self.trace.append(dict(positions=current.tolist(),ground_position=ground_position.tolist(), **metrics))
        return samples, reward, metrics

    def summary(self):
        total = {k: float(sum(t[k] for t in self.trace)) for k in
                 ('task_gain','outside_gain','outside_new','positive_task_gain',
                  'positive_outside_gain','distance','reward','spacing_violation','collisions','out_of_bounds',
                  'cadmm_gain','cadmm_reward','exploration_area','exploration_reward','ground_distance','ground_collisions')}
        total['goal_reached']=float(self.goal_reached)
        total['ground_goal_distance']=self.trace[-1]['ground_goal_distance']
        total['ground_progress']=self.initial_goal_distance-total['ground_goal_distance']
        total['episode_steps']=len(self.trace)
        total['scene_attempts']=self.scene_attempts
        for key in self._graph_metrics:
            total[key]=self.trace[-1][key]
        total['graph_first_connection_step']=float(self.first_graph_connection_step)
        for key in ('valid_actions','forced_hold_fraction','hold_fraction','selected_waypoint_distance','zero_reward'):
            total[key]=float(np.mean([t[key] for t in self.trace]))
        positive=np.maximum([t['reward'] for t in self.trace],0.)
        total['first_minute_positive_share']=float(positive[:60].sum()/max(positive.sum(),1e-8))
        total['min_step_reward']=float(min(t['reward'] for t in self.trace))
        total['max_step_reward']=float(max(t['reward'] for t in self.trace))
        for k in ('simultaneous_moving','moving_fraction','communication_outage','decision_ms',
                  'reward_beta','reward_eta','reward_focus','evidence_source_coverage','max_evidence_age'):
            total[k] = float(np.mean([t[k] for t in self.trace]))
        total['task_share'] = total['positive_task_gain']/max(total['positive_task_gain']+total['positive_outside_gain'],1e-8)
        total['fallback'] = self.trace[-1]['fallback']
        total['coverage_area'] = float(self.ledger.seen.sum()*4)
        total['confidence_area'] = float(self.ledger.confidence().sum()*4)
        for key in ('spacing_cost','intervention_cost','chain_link_violation'):
            total[key]=float(sum(t[key] for t in self.trace))
        total['min_separation']=float(min(t['min_separation'] for t in self.trace))
        total['cbf_intervention']=float(np.mean([t['cbf_intervention'] for t in self.trace]))
        # Independent evaluator only: truth never enters rewards or network.
        seen = self.ledger.seen
        seen_positions = self.ledger.xy[seen]
        truth = np.zeros(len(seen_positions),bool)
        for obstacle in self.env.obstacles:
            truth |= np.linalg.norm(seen_positions-obstacle.position, axis=-1) <= obstacle.radius
        probability = 1/(1+np.exp(-self.ledger.log_odds[seen]))
        total['brier_error'] = float(np.mean((probability-truth)**2)) if seen.any() else 0.
        certainty = np.maximum(probability,1-probability)
        correct = ((probability >= .5) == truth).astype(float)
        ece = 0.
        for lower in np.arange(.5,1.,.05):
            selected = (certainty >= lower) & (certainty < lower+.05)
            if selected.any():
                ece += float(selected.mean()*abs(certainty[selected].mean()-correct[selected].mean()))
        total['calibration_error'] = ece
        return total


def update(model, optimizer, rollout, epochs=4, entropy_coef=.01, return_metrics=False,
           discount=.99, gae_lambda=.95):
    x = torch.tensor(np.array([s[0] for s in rollout]), dtype=torch.float32)
    mask = torch.tensor(np.array([s[1] for s in rollout]), dtype=torch.bool)
    actions = torch.tensor(np.array([s[2] for s in rollout]))
    old_logp = torch.tensor(np.array([s[3] for s in rollout]), dtype=torch.float32)
    rewards = torch.tensor([s[4] for s in rollout], dtype=torch.float32)
    states = None
    if model.global_features:
        if len(rollout[0]) < 6:
            raise ValueError('Rollout lacks the global critic state required by this model')
        states = torch.tensor(np.array([s[5] for s in rollout]), dtype=torch.float32)
    with torch.no_grad():
        values = model.value(x, mask, states)
        adv = torch.zeros_like(rewards)
        last = torch.tensor(0.)
        # Episodes end here: no cross-episode GAE or bootstrapping. A critic
        # given the remaining budget treats budget exhaustion as a true terminal.
        for t in reversed(range(len(rollout))):
            next_v = values[t+1] if t+1 < len(rollout) else 0.
            last = rewards[t] + discount*next_v - values[t] + discount*gae_lambda*last
            adv[t] = last
        returns = adv+values
        adv = (adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
    gradient_norms=[]
    for _ in range(epochs):
        dist = model.distribution(x, mask)
        ratio = (dist.log_prob(actions)-old_logp).exp()
        actor_loss = -torch.minimum(ratio*adv[:,None], ratio.clamp(.8,1.2)*adv[:,None]).mean()
        value_loss = (model.value(x, mask, states)-returns).square().mean()
        loss = actor_loss + .5*value_loss - entropy_coef*dist.entropy().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite PPO loss; training halted before optimizer update')
        optimizer.zero_grad()
        loss.backward()
        gradient_norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), .5)))
        optimizer.step()
    with torch.no_grad():
        final_dist=model.distribution(x,mask)
        log_ratio=final_dist.log_prob(actions)-old_logp
        final_ratio=log_ratio.exp()
        final_values=model.value(x,mask,states)
        final_policy=-torch.minimum(final_ratio*adv[:,None],
            final_ratio.clamp(.8,1.2)*adv[:,None]).mean()
        counts=mask.sum(-1)
        active=counts>1
        metrics=dict(policy_loss=float(final_policy),
            value_loss=float((final_values-returns).square().mean()),
            entropy=float(final_dist.entropy().mean()),
            normalized_entropy=float((final_dist.entropy()/counts.float().log().clamp_min(1e-8))[active].mean())
                if active.any() else 0.,
            approx_kl=float(((final_ratio-1)-log_ratio).mean()),
            clip_fraction=float(((final_ratio-1).abs()>.2).float().mean()),
            explained_variance=float(1.-(returns-final_values).var(unbiased=False)/
                                     returns.var(unbiased=False).clamp_min(1e-8)),
            gradient_norm=float(np.mean(gradient_norms)),
            return_mean=float(returns.mean()),return_std=float(returns.std(unbiased=False)),
            return_min=float(returns.min()),return_max=float(returns.max()),
            active_decision_fraction=float(active.float().mean()))
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError('Non-finite PPO diagnostic; halt before checkpointing')
    result=float(loss.detach())
    return (result,metrics) if return_metrics else result
