"""Train/evaluate MAPPO in the existing 2D physics, mapper and controller.

Training entry point: simulator/train_mappo.py; validation: simulator/train_mappo.py and rescore_checkpoints.py
No simulator full_state, velocity or obstacle list enters the learner/reward.
A scalar feasible reference length normalizes legacy fixed-start route quality.
V9 uses the current endpoints' straight-line lower bound, not an optimality
oracle. Reference paths and map truth never enter Actor/Critic observations.
"""
import sys
import time
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.drone.drone_main import DroneController
from cadmm.drone_navigator import DroneNavigator
from mappo.velocity_mappo import VelocityNavigator
from shared.simulator.env import EnvConfig, MultiRobotEnv, RelayChannel
from shared.ground.ground_main import GroundRobotController
from vtemp.subgoal_manager import DEFAULT_REVIEW_INTERVAL_S
from shared.simulator.scenarios import generate_scene, ENVIRONMENT_VERSION
from shared.simulator.navigation_metrics import task_graph_metrics, RouteQualityTracker, vehicle_route_update
from shared.drone.communication import RelayManager
from shared.common.navigation_config import DEFAULT_GROUND_CLEARANCE_M


COMMUNICATION_RANGE_M = 95.
FLIGHT_HEIGHT_M = 15.
COOPERATIVE_ARCHITECTURES = ('gat_relay_attention_v8', 'gat_survey_v10', 'gat_route_ab_v11')
RELAY_DEFAULTS = dict(relay_loss=0., relay_delay_jitter=0, relay_period=1, relay_capacity=16)
# Compatibility alias; mapping, bridges and driving share one default.
BRIDGE_CLEARANCE_M = DEFAULT_GROUND_CLEARANCE_M
# Per-step relay diagnostics (v8); evaluator truth is used only for the error.
RELAY_STEP_METRICS = ('relay_token_count', 'relay_vehicle_known', 'relay_remote_fraction',
                      'relay_age_s', 'relay_max_hop', 'relay_position_error_m')


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


def route_graph_reward(cost_before, cost_after, initial_cost, weight, time_penalty):
    """Vehicle-independent map reward: start->goal route cost decrease.

    The fixed-start route cost on the delivered graph is a potential: the
    episode sum telescopes to weight*(1 - final/initial cost), so a longer
    episode earns no more; the time penalty rewards building it sooner.
    """
    route=weight*float(cost_before-cost_after)/max(float(initial_cost),1e-8)
    return route-time_penalty, dict(route_reward=route, route_cost=float(cost_after))


def velocity_mapping_reward(cost_before, cost_after, initial_cost, old_seen, seen,
                            task_region, route_weight=10., exploration_weight=.2,
                            time_penalty=.01):
    """Map improvement at matched endpoints chosen by the caller.

    Legacy callers retain fixed start/goal across frames; v9 compares maps
    before/after delivery at one frozen current position and active target.
    """
    area=float(np.count_nonzero(seen & ~old_seen & task_region)*4.)
    mapping=route_weight*float(cost_before-cost_after)/max(initial_cost,1e-8)
    exploration=exploration_weight*area/400.
    return mapping+exploration-time_penalty, dict(mapping_reward=mapping,
        task_exploration_area=area,task_exploration_reward=exploration,
        time_cost=float(time_penalty))


def communication_chain_terms(stats, soft_ratio=.85, comm_range=95., max_speed=6.):
    """Local received geometry only; average over this agent's required links.

    Distance includes altitude on the vehicle link. Missing neighbors are
    penalized explicitly, never replaced with simulator fleet positions.
    Outward speed is this agent's nominal contribution, not a guessed peer
    velocity. The CBF's physical constraints remain unchanged.
    """
    distances=stats.get('required_link_distances_m',{})
    outward=stats.get('required_link_nominal_outward_m_s',{})
    missing=stats.get('missing_links',())
    count=len(distances)+len(missing)
    risks=[];stretch=[]
    for did,distance in distances.items():
        risk=float(np.clip((distance/comm_range-soft_ratio)/(1.-soft_ratio),0.,1.))
        risks.append(risk**2)
        stretch.append(risk*float(np.clip(outward.get(did,0.)/max_speed,0.,1.))**2)
    return dict(chain_risk=sum(risks)/max(count,1),
        chain_outward_risk=sum(stretch)/max(count,1),
        chain_missing_fraction=len(missing)/max(count,1),
        chain_links_available=float(count>0 and not missing),
        chain_max_distance_ratio=max(distances.values(),default=0.)/comm_range)


def cbf_margin_terms(margins, scale=8.):
    """Joint-action risk from the least CBF feasibility margin (m/s) among
    agents; any agent's infeasibility ends the episode. A missing required
    link counts as no margin, since its constraint is unknown to the filter."""
    least=min((m for m in margins if m is not None),default=np.inf)
    return dict(cbf_margin_risk=float(np.clip((scale-least)/scale,0.,1.))**2,
        # Finite for logs: -100 marks a missing link, +100 no constraint at all.
        cbf_margin_min_m_s=float(np.clip(least,-100.,100.)))


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

    def segment_clear(self, start, end, clearance):
        """Every cell within clearance (m) of the segment is received and free."""
        start,end=np.asarray(start,float)[:2],np.asarray(end,float)[:2]
        delta=end-start
        length=float(np.linalg.norm(delta))
        step=self.resolution/4
        along=np.linspace(0.,1.,max(2,int(length/step)+1))
        normal=np.array([-delta[1],delta[0]])/length if length>0 else np.zeros(2)
        across=np.linspace(-clearance,clearance,max(1,2*int(np.ceil(clearance/step))+1))
        points=(start+along[:,None]*delta)[:,None,:]+across[None,:,None]*normal
        cells=np.floor(points.reshape(-1,2)/self.resolution).astype(int)
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
    def __init__(self, seed, model, algorithm=None, stochastic=False,
                 drop=0.,
                 exploration_candidates=5, communication_mode='chain', safety_filter='cbf',
                 spacing_penalty_weight=0., intervention_weight=0., moving_ground=True,
                 reward_mode='cadmm_exploration', ground_speed=1., path_reward_scale=100.,exploration_weight=None,
                 world_size=120., start=(15.,60.), scene=None, stop_on_ground_arrival=True,
                 subgoal_radius=None, subgoal_arrival=25., chain_ground_range=None, chain_link_range=None,
                 route_reward_weight=10., time_penalty=0., ground_stationary=False,
                 num_drones=3,chain_soft_ratio=.85,chain_risk_weight=0.,
                 chain_outward_weight=0.,chain_missing_weight=0.,
                 path_quality_weight=0.,route_reference_length=None,mapping_until_connected=False,
                 cbf_margin_weight=0.,cbf_margin_scale=8.,frontier_progress_weight=0.,
                 relay_loss=0.,relay_delay_jitter=0,relay_period=1,relay_capacity=16,
                 fusion_bridge_distance=0., mapping_route_origin='initial',
                 ground_safety_distance=DEFAULT_GROUND_CLEARANCE_M, coordinated_safety=False,
                 subgoal_review_interval=DEFAULT_REVIEW_INTERVAL_S, guidance_strategy='vtemp1'):
        if type(num_drones) is not int or num_drones < 2:
            raise ValueError('At least two drones are required for the chain experiment')
        if reward_mode not in ('cadmm_exploration','velocity_mapping'):
            raise ValueError('Unsupported reward_mode')
        if mapping_route_origin not in ('initial', 'vehicle'):
            raise ValueError('mapping_route_origin must be initial or vehicle')
        self.mapping_route_origin = mapping_route_origin
        self.vehicle_route_mapping = mapping_route_origin == 'vehicle'
        if self.vehicle_route_mapping and (reward_mode != 'velocity_mapping' or not moving_ground
                                          or ground_stationary):
            raise ValueError('Vehicle-origin mapping requires a moving vehicle')
        if ground_stationary and (subgoal_radius is not None or not moving_ground):
            raise ValueError('A stationary vehicle keeps its map receiver and has no subgoals')
        if subgoal_radius is not None and not (subgoal_radius>0 and moving_ground):
            raise ValueError('subgoal_radius must be positive and requires a moving ground robot')
        if not 0<=fusion_bridge_distance<float('inf'):
            raise ValueError('fusion_bridge_distance must be a finite nonnegative distance')
        if (isinstance(ground_safety_distance,bool) or not isinstance(ground_safety_distance,(int,float))
                or not np.isfinite(ground_safety_distance) or ground_safety_distance<=0):
            raise ValueError('ground_safety_distance must be positive and finite')
        self.ground_safety_distance=float(ground_safety_distance)
        self.coordinated_safety=bool(coordinated_safety)
        self.coordinated_safety_events=[]
        if self.coordinated_safety and (communication_mode!='backbone' or drop!=0. or safety_filter!='cbf'):
            raise ValueError('Joint safety recovery requires synchronous backbone geometry and CBF')
        self.world_size=float(world_size)
        self.start=np.asarray(start,dtype=float)
        self.stop_on_ground_arrival=bool(stop_on_ground_arrival)
        self.rng = np.random.default_rng(seed)
        self.drop = drop
        architecture = getattr(model, 'policy_architecture', None)
        cooperative = architecture in COOPERATIVE_ARCHITECTURES
        if cooperative != (communication_mode=='backbone'):
            raise ValueError('Cooperative models require backbone communication and vice versa')
        if cooperative and (drop!=0. or safety_filter!='cbf'):
            raise ValueError('Backbone communication requires synchronous complete geometry and CBF')
        relay = dict(relay_loss=relay_loss,relay_delay_jitter=relay_delay_jitter,
                     relay_period=relay_period,relay_capacity=relay_capacity)
        # v8 delays/drops only relayed state evidence; one-hop geometry, maps
        # and the CBF use synchronous neighbor geometry.
        self.relay_channel=(RelayChannel(seed,relay_loss,relay_delay_jitter,relay_period)
                            if cooperative else None)
        if self.relay_channel is None and relay!=RELAY_DEFAULTS:
            raise ValueError('relay_* settings require the v8 relay attention policy')
        if type(relay_capacity) is not int or relay_capacity<1:
            raise ValueError('relay_capacity must be a positive integer')
        self.communication_mode=communication_mode
        self.safety_filter=safety_filter
        self.spacing_penalty_weight=spacing_penalty_weight
        self.intervention_weight=intervention_weight
        self.moving_ground=moving_ground
        self.reward_mode=reward_mode
        self.chain_soft_ratio=chain_soft_ratio
        self.chain_risk_weight=chain_risk_weight
        self.chain_outward_weight=chain_outward_weight
        self.chain_missing_weight=chain_missing_weight
        self.cbf_margin_weight=float(cbf_margin_weight)
        self.cbf_margin_scale=float(cbf_margin_scale)
        if self.cbf_margin_weight<0 or not self.cbf_margin_scale>0:
            raise ValueError('CBF margin penalty needs a nonnegative weight and a positive scale')
        self.frontier_progress_weight=float(frontier_progress_weight)
        if not np.isfinite(self.frontier_progress_weight) or self.frontier_progress_weight<0:
            raise ValueError('frontier_progress_weight must be finite and nonnegative')
        if self.frontier_progress_weight and reward_mode!='velocity_mapping':
            raise ValueError('Frontier progress is a velocity_mapping term')
        self._frontier_best=0.
        self.route_reward_weight=float(route_reward_weight)
        self.path_quality_weight=float(path_quality_weight)
        if self.path_quality_weight and route_reference_length is None and not self.vehicle_route_mapping:
            raise ValueError('Path quality reward requires a physical reference length')
        self.route_quality_tracker=(RouteQualityTracker(route_reference_length)
            if reward_mode=='velocity_mapping' and route_reference_length is not None
            and not self.vehicle_route_mapping else None)
        self.mapping_until_connected=bool(mapping_until_connected)
        if self.mapping_until_connected and (not self.path_quality_weight or
                (not self.vehicle_route_mapping and self.route_quality_tracker is None)):
            raise ValueError('Map-term hand-over requires the route quality reward')
        self.time_penalty=float(time_penalty)
        # The vehicle still receives and fuses maps but does not drive.
        self.ground_stationary=bool(ground_stationary)
        self.path_reward_scale=path_reward_scale
        self.exploration_weight=float(exploration_weight) if exploration_weight is not None else .6
        self.audit_buffers=[{} for _ in range(num_drones)]
        self.latest_delivered_frames={}
        self.env = MultiRobotEnv(EnvConfig(world_size=(self.world_size,self.world_size), dt=1.,
            perception_radius=24., communication_radius=COMMUNICATION_RANGE_M, occupancy_map_resolution=2.,
            occupancy_map_size=12, ground_occupancy_map_size=16 if moving_ground else 4,
            ground_occupancy_map_resolution=.5 if moving_ground else None))
        self.ground = self.env.add_ground_robot(*self.start,max_speed=ground_speed)
        self.goal, circles, self.scene_attempts = generate_scene(seed) if scene is None else scene
        self.goal=np.asarray(self.goal,dtype=float)
        # Rejection/feasibility checks are environment setup only. No grid or
        # reference route enters controllers or Actor/Critic observations.
        for position, radius in circles:
            self.env.add_circle_obstacle(*position, radius)
        self.ids, self.controllers = [], []
        bounds=(0.,0.,self.world_size,self.world_size)
        # A serpentine chain of 18m-spaced initial positions; N=3 retains the
        # historical layout. Reflect extra columns away from the world edge.
        offsets = [(9., -18.), (9., 0.), (9., 18.)]
        if num_drones != 3:
            sign = 1. if self.start[0] < self.world_size/2 else -1.
            offsets = [(sign*(9.+18.*(i//3)),
                        (-18.+18.*(i%3))*(1. if (i//3)%2==0 else -1.))
                       for i in range(num_drones)]
        for i, (dx,dy) in enumerate(offsets):
            x,y=self.start+np.asarray([dx,dy])
            if not (0. <= x <= self.world_size and 0. <= y <= self.world_size):
                raise ValueError('Initial fleet does not fit inside the world bounds')
            self.ids.append(self.env.add_drone(x,y,FLIGHT_HEIGHT_M,max_speed=6.))
            c = DroneController(f'drone_{i}', resolution=2., total_drones=num_drones,
                communication_range=COMMUNICATION_RANGE_M, max_speed=6., broadcast_interval=1., fusion_interval=5.,
                communication_mode=communication_mode,safety_filter=safety_filter,
                graph_min_clearance=self.ground_safety_distance)
            if getattr(model, 'action_mode', None) == 'velocity':
                navigator_type = VelocityNavigator
                if cooperative:
                    from mappo.cooperative_mappo import CooperativeNavigator
                    navigator_type = CooperativeNavigator
                c.navigator = navigator_type(f'drone_{i}', total_drones=num_drones, comm_range=COMMUNICATION_RANGE_M,
                    min_separation=15., model=model, deterministic=not stochastic,
                    max_speed=6., world_bounds=bounds)
            else:
                c.navigator.fov_radius = 12.  # Same actual 24m sensor footprint.
            if self.relay_channel is not None:
                c.relay = RelayManager(c.drone_id, capacity=relay_capacity)
            self.controllers.append(c)
            # Surveyed world bounds are part of the simulator task contract.
            c.graph_builder.world_bounds = bounds
            c.ground_max_speed=ground_speed if moving_ground and not ground_stationary else 0.
        self.obs = self.env.reset()
        # Both the ground planner and the aircraft receive the current direction.
        self.ground_controller=GroundRobotController('ground_0',resolution=2.,
            perception_range=8.,max_linear_vel=ground_speed,max_angular_vel=1.,
            fusion_interval=5.,safety_distance=self.ground_safety_distance,subgoal_radius=subgoal_radius,
            subgoal_arrival=subgoal_arrival, subgoal_review_interval=subgoal_review_interval,
            world_bounds=bounds, guidance_strategy=guidance_strategy)
        self.ground_controller.set_goal(tuple(self.goal))
        # The vehicle only originates its own state/task evidence; it does not forward.
        self.ground_relay=(RelayManager(self.ground_controller.robot_id, capacity=relay_capacity)
                           if self.relay_channel is not None else None)
        self.subgoals=self.ground_controller.subgoals
        self.goal_reached=False
        self.ground_action=self._ground_action() if moving_ground else None
        self.initial_goal_distance=float(np.linalg.norm(self.goal-self.start))
        self._initial_route_cost=self._route_cost=navigation_cost(None,self.start,self.goal)
        self.ledger = DeliveredEvidence(world=self.world_size)
        if fusion_bridge_distance:
            # The vehicle joins graph components only across corridors that the
            # occupancy delivered to it shows free; drones' graphs are unchanged.
            fusion=self.ground_controller.graph_fusion
            fusion.bridge_distance=float(fusion_bridge_distance)
            fusion.bridge_check=lambda a,b: self.ledger.segment_clear(a,b,self.ground_safety_distance)
        initial_target=(self.subgoals.target if self.vehicle_route_mapping
                        and self.subgoals is not None else self.goal)
        self.reward_task_region = self.ledger.task_mask(self.start, initial_target)
        self.algorithm = algorithm or ('mappo' if getattr(model, 'action_mode', None) == 'velocity' else 'cadmm')
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
        if self.ground_stationary:
            action['motion']=(0.,0.)
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
        if self.communication_mode=='backbone':
            from shared.drone.communication import backbone_metrics
            _, longest = backbone_metrics({str(i): np.r_[p, FLIGHT_HEIGHT_M] for i,p in enumerate(drones)},
                                          np.r_[ground, 0.], COMMUNICATION_RANGE_M)
            links = [longest]
        separation=min(np.linalg.norm(drones[i]-drones[j]) for i in range(len(drones)) for j in range(i))
        scale=max(self.initial_goal_distance,1.)
        # Path cost after the previous step's map update; empty map before the first step.
        cost=self.trace[-1]['ground_path_cost'] if self.trace else scale/.05
        target=self.goal
        if self.vehicle_route_mapping:
            target=self.goal if self.subgoals is None else self.subgoals.target
            if not self.trace:
                cost=navigation_cost(self.ground_controller.unified_graph,ground,target)
        connected=bool(self._graph_metrics and self._graph_metrics['graph_task_connected'])
        return np.array([frame/budget,(budget-frame)/1000.,self.ledger.seen.sum()*4/1e5,
            float(connected),stability,np.linalg.norm(ground-target)/scale,scale/1000.,
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
        interventions=[];observed_violations=[];infeasible=0;chain_terms=[];margins=[]
        relay_rows=[]
        if self.relay_channel is not None:
            self._relay_vehicle(frame, actions[self.ground])
        for index,(rid,c) in enumerate(zip(self.ids,self.controllers)):
            received = dict(self.obs[rid])
            # Packet loss applies equally to all algorithms; no actor truth fields.
            received['neighbors'] = [n for n in received['neighbors']
                                     if self.rng.random() >= self.drop]
            if self.relay_channel is not None:
                received['relay_envelopes'] = self.relay_channel.deliver(rid, frame)
            old.append(np.asarray(received['self']['position']).copy())
            tic = time.perf_counter()
            action = c.get_action(received)
            decision_ms.append((time.perf_counter()-tic)*1000)
            if self.relay_channel is not None:
                now=float(received['full_state']['time'])
                relay_rows.append((c.relay.tokens(now),{n['id'] for n in received['neighbors']}))
                if self.relay_channel.due(rid, frame):
                    # Sent now, readable no earlier than the next decision step.
                    self.relay_channel.transmit(c.relay.outgoing(now),
                        [n['id'] for n in received['neighbors'] if n['type']=='drone'], frame)
            target = c.navigator._current_target
            if target is not None and self.safety_filter!='cbf':
                distance = np.linalg.norm(np.asarray(target)-old[-1])
                speed = np.linalg.norm(action['motion'])
                if speed > distance:
                    action['motion'] *= distance/max(speed, 1e-8)
            # Only a second, explicit joint safety solve may replace these
            # commands below. Never clip or zero a verified command afterward.
            stats=c.safety_stats
            chain_terms.append(communication_chain_terms(stats,self.chain_soft_ratio,
                c.communication_range,c.max_speed))
            # No packet has arrived before the first step; later gaps are real losses.
            margins.append(float('-inf') if frame and stats.get('missing_links') else stats.get('feasibility_margin'))
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
                if self.communication_mode in ('chain','backbone'):
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
                if getattr(c.navigator, 'uses_velocity', False):
                    holds.append(float(np.linalg.norm(c.navigator.nominal_velocity) < .1))
                else:
                    valid_counts.append(int(mask.sum()))
                    selected_distances.append(float(features[selected,2]*100.))
                    holds.append(float(selected == 0))
        if infeasible and self.coordinated_safety:
            from shared.drone.cbf_safety import filter_joint
            commands,diagnostic=filter_joint({c.drone_id:c.safety_request for c in self.controllers})
            if commands is not None:
                self.coordinated_safety_events.append(dict(frame=frame,**diagnostic))
                for index,(rid,c) in enumerate(zip(self.ids,self.controllers)):
                    actions[rid]['motion']=commands[c.drone_id]
                    stats=c.safety_stats
                    stats['independent_cbf_infeasible']=stats['infeasible']
                    stats['independent_cbf_max_constraint_residual']=stats['max_constraint_residual']
                    stats['infeasible']=False
                    stats['coordinated_recovery']=True
                    stats['max_constraint_residual']=diagnostic['max_constraint_residual']
                    stats['feasibility_margin_scope']='independent_cbf_before_joint_recovery'
                    stats['joint_min_slack_m_s']=diagnostic['min_slack_m_s']
                    stats['intervention']=float(np.linalg.norm(commands[c.drone_id]-c._nominal_velocity))
                    interventions[index]=stats['intervention']**2/36.
                infeasible=0
        if infeasible:
            raise SafetyTermination(infeasible,samples)
        self.obs, info = self.env.step(actions)
        self.last_executed_velocities=np.asarray([actions[rid]["motion"] for rid in self.ids]).copy()
        ground_received=[n for n in self.obs[self.ground]['neighbors']
                         if self.rng.random()>=self.drop]
        ground_position=np.asarray(self.obs[self.ground]['self']['position']).copy()
        # Evaluate both maps at the SAME new position: vehicle progress alone
        # must not masquerade as an observation-induced CADMM improvement.
        # With relay subgoals both use the subgoal current before this update,
        # so a subgoal switch never appears as a map improvement.
        target=self.goal if self.subgoals is None else self.subgoals.target.copy()
        cost_before=navigation_cost(self.ground_controller.unified_graph,ground_position[:2],target)
        if self.vehicle_route_mapping:
            # Freeze BOTH endpoints after physics, before fusion/delivery. Comparing
            # successive timesteps would pay for driving or changing the subgoal.
            vehicle_before=task_graph_metrics(self.ground_controller.unified_graph,
                ground_position[:2],target,self.ledger.connector_clear)
        if self.moving_ground:
            self.ground_action=self._ground_action(ground_received)
        cost_after=navigation_cost(self.ground_controller.unified_graph,ground_position[:2],target)
        # Delivery must be present in the actual ground receiver's neighbor list.
        delivered = ground_received
        delivered_ids = {n['id'] for n in delivered}
        for neighbor in delivered:
            message=neighbor.get('broadcast') or {}
            packets=message.get('evidence_audits',[]) if self.communication_mode in ('chain','backbone') else [message.get('evidence_audit')]
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
        if self.vehicle_route_mapping:
            vehicle_after=task_graph_metrics(graph,ground_position[:2],target,self.ledger.connector_clear)
            vehicle_terms=vehicle_route_update(vehicle_before,vehicle_after,ground_position[:2],target)
            active_target=(self.goal if self.subgoals is None else self.subgoals.target).copy()
            target_changed=not np.array_equal(active_target,target)
            # The reward remains on the frozen target; observations and displayed
            # graph metrics follow the new active target immediately after a switch.
            self._graph_metrics=(task_graph_metrics(graph,ground_position[:2],active_target,
                self.ledger.connector_clear) if target_changed else vehicle_after)
            active_terms=vehicle_route_update(self._graph_metrics,self._graph_metrics,
                ground_position[:2],active_target)
            region=self.ledger.task_mask(ground_position[:2],target)
            self.reward_task_region=region
        elif (self.reward_mode=='velocity_mapping' or self._graph_metrics is None
                or graph is not self._graph_metric_source or frame % 5 == 0):
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
        if self.vehicle_route_mapping:
            route_before,route_after=cost_before,cost_after
            route_scale=max(vehicle_terms['route_reference_length_m'],1.)/.05
            self._route_cost=(navigation_cost(graph,ground_position[:2],active_target)
                              if target_changed else cost_after)
        else:
            # Legacy v8 fixed-start potential is preserved for existing runs.
            route_before=self._route_cost
            self._route_cost=navigation_cost(self.ground_controller.unified_graph,self.start,self.goal)
            route_after=self._route_cost
            route_scale=self._initial_route_cost
        _,route_terms=route_graph_reward(route_before,route_after,route_scale,
            self.route_reward_weight,self.time_penalty)
        metrics.update(route_terms)
        if self.reward_mode=='velocity_mapping':
            if self.vehicle_route_mapping:
                metrics.update(vehicle_terms)
                # Signed changes also charge a lost route. Reconnection/rebuilds
                # cannot repeatedly collect an unopposed first-connection bonus.
                handed_over=self.mapping_until_connected and bool(
                    vehicle_before['graph_task_connected'] or vehicle_after['graph_task_connected'])
            elif self.route_quality_tracker is not None:
                metrics.update(self.route_quality_tracker.update(self._graph_metrics))
            # Route quality alone pays connected routes, avoiding duplicate map
            # cost credit. Legacy hands over once; v9 decides per frozen pair.
            if not self.vehicle_route_mapping:
                handed_over=self.mapping_until_connected and self.route_quality_tracker.first_length is not None
            reward,focused_terms=velocity_mapping_reward(route_before,route_after,
                route_scale,old_seen,self.ledger.seen,region if self.vehicle_route_mapping else self.reward_task_region,
                0. if handed_over else self.route_reward_weight,self.exploration_weight,self.time_penalty)
            metrics.update(focused_terms)
            metrics['path_quality_reward']=self.path_quality_weight*metrics.get('route_quality_gain',0.)
            metrics['first_path_quality_reward']=self.path_quality_weight*metrics.get('route_first_quality_gain',0.)
            metrics['path_shortening_reward']=self.path_quality_weight*metrics.get('route_shortening_quality_gain',0.)
            reward+=metrics['path_quality_reward']
            if self.vehicle_route_mapping:
                gain=vehicle_terms['frontier_progress_gain']
                metrics.update(mapping_route_origin_vehicle=1.,
                    route_origin_x=float(ground_position[0]),route_origin_y=float(ground_position[1]),
                    route_target_x=float(active_target[0]),route_target_y=float(active_target[1]),
                    reward_route_target_x=float(target[0]),reward_route_target_y=float(target[1]),
                    route_target_changed=float(target_changed),
                    reward_route_quality_after=vehicle_terms['route_quality'],
                    reward_route_reference_length_m=vehicle_terms['route_reference_length_m'],
                    route_quality=active_terms['route_quality'],
                    route_reference_length_m=active_terms['route_reference_length_m'],
                    current_route_length_m=self._graph_metrics['graph_task_path_length'],
                    frontier_progress=active_terms['frontier_progress'],
                    frontier_progress_reward=self.frontier_progress_weight*gain)
            else:
                # Legacy v5: best-so-far progress from the fixed initial start.
                remaining=self._graph_metrics.get('graph_frontier_goal_distance_m',self.initial_goal_distance)
                progress=1. if self._graph_metrics['graph_task_connected'] else float(
                    np.clip(1.-remaining/max(self.initial_goal_distance,1e-8),0.,1.))
                gain=max(progress-self._frontier_best,0.)
                self._frontier_best=max(self._frontier_best,progress)
                metrics.update(frontier_progress=progress,frontier_best_progress=self._frontier_best,
                    frontier_progress_reward=self.frontier_progress_weight*gain)
            reward+=metrics['frontier_progress_reward']
        metrics.update(reward_beta=1.,reward_eta=self.exploration_weight)
        # Penalties use local received safety telemetry, not evaluator truth.
        spacing_cost=self.spacing_penalty_weight*max(observed_violations,default=0.)
        intervention_cost=self.intervention_weight*float(np.mean(interventions))
        reward-=spacing_cost+intervention_cost
        # A mean over local per-agent means keeps reward scale independent of
        # fleet size; actors still use only their own received observations.
        for key in ('chain_risk','chain_outward_risk','chain_missing_fraction'):
            metrics[key]=float(np.mean([t[key] for t in chain_terms]))
        metrics['chain_links_available']=float(all(t['chain_links_available'] for t in chain_terms))
        metrics['chain_max_distance_ratio']=max(t['chain_max_distance_ratio'] for t in chain_terms)
        metrics['chain_risk_cost']=self.chain_risk_weight*metrics['chain_risk']
        metrics['chain_outward_cost']=self.chain_outward_weight*metrics['chain_outward_risk']
        metrics['chain_missing_cost']=self.chain_missing_weight*metrics['chain_missing_fraction']
        metrics.update(cbf_margin_terms(margins,self.cbf_margin_scale))
        metrics['cbf_margin_cost']=self.cbf_margin_weight*metrics['cbf_margin_risk']
        reward-=sum(metrics[k] for k in ('chain_risk_cost','chain_outward_cost','chain_missing_cost',
                                         'cbf_margin_cost'))
        # Swept distances are independent evaluation, never learner inputs.
        sep = []
        for i in range(len(self.ids)):
            for j in range(i):
                r = old[i]-old[j]
                v = (current[i]-old[i])-(current[j]-old[j])
                t = np.clip(-np.dot(r,v)/max(np.dot(v,v),1e-8),0,1)
                sep.append(np.linalg.norm(r+t*v))
        # Independent evaluator checks configured chain edges and swept maxima.
        chain_distances=[max(np.linalg.norm(np.asarray(old[0])-old_ground),np.linalg.norm(current[0]-ground_position))]+[
            max(np.linalg.norm(np.asarray(old[i])-old[i-1]),np.linalg.norm(current[i]-current[i-1]))
            for i in range(1,len(self.ids))]
        if self.communication_mode=='backbone':
            from shared.drone.communication import backbone_metrics
            connected, _ = backbone_metrics({str(i): p for i,p in enumerate(current)}, ground_position,
                                             COMMUNICATION_RANGE_M)
            lookup = {c.drone_id:i for i,c in enumerate(self.controllers)}
            lookup.update({f'robot_{rid}':i for i,rid in enumerate(self.ids)})
            chain_distances = []
            for i,c in enumerate(self.controllers):
                for key in c.navigator.required_links:
                    before_q, after_q = ((old_ground,ground_position) if key=='__ground__'
                                          else (old[lookup[key]],current[lookup[key]]))
                    chain_distances.append(max(np.linalg.norm(old[i]-before_q),np.linalg.norm(current[i]-after_q)))
            metrics['backbone_connected'] = float(connected)
            metrics['backbone_required_links'] = [list(c.navigator.required_links) for c in self.controllers]
            metrics['chain_links_available'] *= float(connected)
            metrics['chain_max_distance_ratio'] = max(chain_distances,default=0.)/COMMUNICATION_RANGE_M
        ages=[frame-self.latest_delivered_frames.get(rid,-1) for rid in self.ids]
        packet_age_limit=len(self.ids) if self.communication_mode in ('chain','backbone') else 1
        metrics.update(reward=float(reward), distance=float(distance.sum()),
            spacing_cost=spacing_cost,intervention_cost=intervention_cost,
            cbf_intervention=float(np.mean(interventions)),
            chain_link_violation=float(max(chain_distances,default=0.)>95.+1e-5),
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
            out_of_bounds=len(info['out_of_bounds']),
            boundary_roundoff_count=len(info.get('boundary_roundoff', [])),
            boundary_roundoff_max_m=max((e['overshoot_m'] for e in info.get('boundary_roundoff', [])), default=0.),
            boundary_violation_max_m=max((e['overshoot_m'] for e in info.get('boundary_violations', [])), default=0.))
        metrics.update(ground_distance=float(np.linalg.norm(ground_position-old_ground)),
            ground_goal_distance=float(np.linalg.norm(ground_position[:2]-self.goal)),
            ground_collisions=float(len(info['collisions'])),goal_reached=float(self.goal_reached),
            ground_path_cost=self._route_cost if self.vehicle_route_mapping else cost_after)
        metrics.update(self._graph_metrics)
        metrics.update(valid_actions=float(np.mean(valid_counts)) if valid_counts else 0.,
            forced_hold_fraction=float(np.mean(np.asarray(valid_counts)==1)) if valid_counts else 0.,
            hold_fraction=float(np.mean(holds)) if holds else 0.,
            selected_waypoint_distance=float(np.mean(selected_distances)) if selected_distances else 0.,
            zero_reward=float(abs(reward)<1e-8))
        if self.relay_channel is not None:
            metrics.update(self._relay_metrics(relay_rows, old, old_ground))
        if getattr(self.controllers[0].navigator, 'uses_velocity', False):
            metrics['nominal_speed_mean'] = float(np.mean([
                np.linalg.norm(c.navigator.nominal_velocity) for c in self.controllers]))
            metrics['executed_speed_mean'] = float(np.mean([
                np.linalg.norm(actions[rid]['motion']) for rid in self.ids]))
        self.trace.append(dict(positions=current.tolist(),ground_position=ground_position.tolist(), **metrics))
        return samples, reward, metrics

    def _relay_vehicle(self, frame, action):
        """The vehicle wraps its own mission broadcast as vehicle_state evidence."""
        own=self.obs[self.ground]['self']
        message=action.get('broadcast') or {}
        position=np.asarray(own['position'],float)
        position=np.r_[position[:2],position[2] if len(position)>2 else 0.]
        target=message.get('target')
        now=float(message.get('timestamp',self.env.time))
        self.ground_relay.originate(now,'vehicle_state',position,own.get('velocity'),
                                    None if target is None else target[:2])
        if self.relay_channel.due('ground',frame):
            self.relay_channel.transmit(self.ground_relay.outgoing(now),
                [n['id'] for n in self.obs[self.ground]['neighbors'] if n['type']=='drone'],frame)

    def _relay_metrics(self, relay_rows, old, ground):
        """Evaluator-only relay diagnostics; truth here never reaches an actor."""
        truth={c.drone_id:np.asarray(p,float) for c,p in zip(self.controllers,old)}
        truth[self.ground_controller.robot_id]=np.r_[np.asarray(ground,float)[:2],
            np.asarray(ground,float)[2] if len(ground)>2 else 0.]
        index={c.drone_id:rid for c,rid in zip(self.controllers,self.ids)}
        rows=[row for tokens,_ in relay_rows for row in tokens]
        drones=[(row,sensed) for tokens,sensed in relay_rows for row in tokens
                if row['payload_type']=='drone_state']
        errors=[np.linalg.norm(row['position'][:2]-truth[row['origin_id']][:2]) for row in rows
                if row['origin_id'] in truth]
        return dict(relay_token_count=float(np.mean([len(tokens) for tokens,_ in relay_rows])),
            relay_vehicle_known=float(np.mean([any(r['payload_type']=='vehicle_state' for r in tokens)
                                               for tokens,_ in relay_rows])),
            # Teammate records whose origin is NOT a current one-hop neighbor.
            relay_remote_fraction=float(np.mean([index.get(row['origin_id']) not in sensed
                                                 for row,sensed in drones])) if drones else 0.,
            relay_age_s=float(np.mean([row['age'] for row in rows])) if rows else 0.,
            relay_max_hop=float(max((row['hop'] for row in rows),default=0)),
            relay_position_error_m=float(np.mean(errors)) if errors else 0.)

    def summary(self):
        # CBF can reject the very first joint action before any physics step.
        # Report the observed initial state without inventing a trace sample.
        if self.trace:
            last = self.trace[-1]
            graph_metrics = self._graph_metrics
            initial_separation = 0.
        else:
            ground = np.asarray(self.obs[self.ground]['self']['position'], float)[:2]
            target = self.goal if self.subgoals is None else self.subgoals.target
            graph_metrics = task_graph_metrics(self.ground_controller.unified_graph,
                ground if self.vehicle_route_mapping else self.start,
                target if self.vehicle_route_mapping else self.goal, self.ledger.connector_clear)
            last = dict(graph_metrics, ground_goal_distance=float(np.linalg.norm(ground-self.goal)),
                        fallback=0)
            if self.vehicle_route_mapping:
                last.update(vehicle_route_update(graph_metrics, graph_metrics, ground, target))
                last.update(mapping_route_origin_vehicle=1.,
                    current_route_length_m=graph_metrics['graph_task_path_length'],
                    route_origin_x=float(ground[0]), route_origin_y=float(ground[1]),
                    route_target_x=float(target[0]), route_target_y=float(target[1]))
            points = [np.asarray(self.obs[r]['self']['position'], float) for r in self.ids]
            initial_separation = min((float(np.linalg.norm(a-b)) for i,a in enumerate(points)
                                      for b in points[:i]), default=0.)
        def mean(key):
            return float(np.mean([t[key] for t in self.trace])) if self.trace else 0.
        total = {k: float(sum(t[k] for t in self.trace)) for k in
                 ('task_gain','outside_gain','outside_new','positive_task_gain',
                  'positive_outside_gain','distance','reward','spacing_violation','collisions','out_of_bounds',
                  'cadmm_gain','cadmm_reward','exploration_area','exploration_reward','ground_distance','ground_collisions',
                  'route_reward')}
        total['goal_reached']=float(self.goal_reached)
        total['ground_goal_distance']=last['ground_goal_distance']
        total['ground_progress']=self.initial_goal_distance-total['ground_goal_distance']
        total['episode_steps']=len(self.trace)
        total['scene_attempts']=self.scene_attempts
        total['num_drones']=len(self.ids)
        total['coordinated_safety_enabled']=self.coordinated_safety
        total['coordinated_safety_steps']=len(self.coordinated_safety_events)
        total['boundary_roundoff_count']=float(sum(t['boundary_roundoff_count'] for t in self.trace))
        for key in ('boundary_roundoff_max_m', 'boundary_violation_max_m'):
            total[key]=float(max((t[key] for t in self.trace), default=0.))
        if self.reward_mode=='velocity_mapping':
            for k in ('task_exploration_area', 'task_exploration_reward', 'time_cost'):
                total[k] = float(sum(t[k] for t in self.trace))
            total['mapping_reward']=float(sum(t['mapping_reward'] for t in self.trace))
            for key in ('path_quality_reward','first_path_quality_reward','path_shortening_reward',
                        'route_new_best_saving_m','frontier_progress_reward'):
                total[key]=float(sum(t.get(key,0.) for t in self.trace))
            total['frontier_best_progress']=float(last.get('frontier_best_progress',0.))
            for key in ('route_first_length_m','route_best_length_m','route_quality',
                        'route_best_quality','route_excess_m'):
                total[key]=float(last.get(key,0.))
            if self.vehicle_route_mapping:
                # No "best route" is comparable across moving origins/targets.
                for key in ('route_first_length_m','route_best_length_m','route_best_quality',
                            'route_excess_m','route_new_best_saving_m','frontier_best_progress'):
                    total.pop(key,None)
                for key in ('mapping_route_origin_vehicle','route_reference_length_m',
                            'route_reference_is_lower_bound','route_quality',
                            'current_route_length_m','route_origin_x','route_origin_y',
                            'route_target_x','route_target_y','frontier_progress'):
                    total[key]=float(last[key])
                total['route_target_switches']=float(sum(t['route_target_changed'] for t in self.trace))
                total['route_length_delta_m']=float(sum(t['route_length_delta_m'] for t in self.trace))
        for key in ('chain_risk_cost','chain_outward_cost','chain_missing_cost','cbf_margin_cost'):
            total[key]=float(sum(t[key] for t in self.trace))
        for key in ('chain_risk','chain_outward_risk','chain_missing_fraction','chain_links_available',
                    'cbf_margin_risk'):
            total[key]=mean(key)
        total['cbf_margin_min_m_s']=float(min((t['cbf_margin_min_m_s'] for t in self.trace), default=0.))
        total['chain_max_distance_ratio']=float(max((t['chain_max_distance_ratio'] for t in self.trace), default=0.))
        for key in graph_metrics:
            total[key]=last[key]
        total['graph_first_connection_step']=float(self.first_graph_connection_step)
        for key in ('valid_actions','forced_hold_fraction','hold_fraction','selected_waypoint_distance','zero_reward'):
            total[key]=mean(key)
        positive=np.maximum([t['reward'] for t in self.trace],0.)
        total['first_minute_positive_share']=float(positive[:60].sum()/max(positive.sum(),1e-8))
        total['min_step_reward']=float(min((t['reward'] for t in self.trace), default=0.))
        total['max_step_reward']=float(max((t['reward'] for t in self.trace), default=0.))
        for k in ('simultaneous_moving','moving_fraction','communication_outage','decision_ms',
                  'reward_beta','reward_eta','reward_focus','evidence_source_coverage','max_evidence_age'):
            total[k] = mean(k)
        total['task_share'] = total['positive_task_gain']/max(total['positive_task_gain']+total['positive_outside_gain'],1e-8)
        total['fallback'] = last['fallback']
        total['coverage_area'] = float(self.ledger.seen.sum()*4)
        total['confidence_area'] = float(self.ledger.confidence().sum()*4)
        for key in ('spacing_cost','intervention_cost','chain_link_violation'):
            total[key]=float(sum(t[key] for t in self.trace))
        total['min_separation']=float(min((t['min_separation'] for t in self.trace), default=initial_separation))
        total['cbf_intervention']=mean('cbf_intervention')
        if self.relay_channel is not None:
            for key in RELAY_STEP_METRICS:
                total[key]=mean(key)
            for key,value in self.relay_channel.stats.items():
                total[f'relay_packets_{key}']=float(value)
            for key in ('accepted','duplicate','stale','expired','future'):
                total[f'relay_entries_{key}']=float(sum(c.relay.stats[key] for c in self.controllers))
        if getattr(self.controllers[0].navigator, 'uses_velocity', False):
            total['nominal_speed_mean'] = mean('nominal_speed_mean')
            total['executed_speed_mean'] = mean('executed_speed_mean')
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
        if self.subgoals is not None:
            total.update(self.subgoals.stats())
        return total
