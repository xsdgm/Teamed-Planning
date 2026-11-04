"""CoGraph v1 environment: 3 vehicles + 9 drones, one team, 2-8 targets.

Training builds the option space only. The vehicles stay at the start as map
receivers and backbone anchors; parked within radio range of each other, they
pool the drone graphs any of them receives into one fused team graph, and each
broadcasts every target with its own route status. A target counts as served
once ANY vehicle's delivered graph connects that vehicle to it (the vehicle-side
assignment then has a choice). Reward terms are v8's, per target and averaged
over targets: best-so-far frontier progress and route quality, plus the same
time, CBF-margin and intervention penalties. Success: every target connected
with the backbone intact for map_stability_steps, then v8's improvement phase.

CoGraph v2 (vehicle_planner='decentralized'): the vehicles are not held still
and there is no leader. Each vehicle runs its own VehicleClaimAgent
(ground/vehicle_claims.py) on its own information: its position, its own fused
graph and the claims it receives from the other vehicles, directly within radio
range or forwarded by the drones. It claims a target, broadcasts the claim,
departs once its own graph has built part of the way, may hand the target over
mid-course to a clearly better placed vehicle, and on arrival broadcasts the
target as served and claims the next. Vehicles pool their fused graphs only
while in radio range of each other (direct exchange); a vehicle that leaves
keeps a copy and fuses on its own. The drones' CBF assumes the vehicle speed
whenever a vehicle may move.
"""
import contextlib
import copy
import io
import math
import time
import numpy as np
from drone.drone_main import DroneController
from drone.cograph_mappo import CoGraphNavigator, GLOBAL_FEATURES
from drone.cograph_v2_mappo import CoGraphV2Navigator
from drone.local_backbone import backbone_metrics_multi
from drone.relay_cache import RelayManager
from ground.ground_main import GroundRobotController
from ground.task_planner import coverage_tree
from ground.vehicle_claims import VehicleClaimAgent
from simulator.cograph_scenarios import generate_cograph_scene
from simulator.env import EnvConfig, MultiRobotEnv
from simulator.mappo_experiment import (COMMUNICATION_RANGE_M, FLIGHT_HEIGHT_M, BRIDGE_CLEARANCE_M,
    DeliveredEvidence, SafetyTermination, communication_chain_terms, cbf_margin_terms)
from simulator.navigation_metrics import multi_target_graph_metrics, RouteQualityTracker
from simulator.relay_channel import RelayChannel

WORLD_M = 1000.
SUM_KEYS = ('reward', 'distance', 'collisions', 'spacing_violation', 'chain_link_violation', 'out_of_bounds',
            'frontier_progress_reward', 'path_quality_reward', 'intervention_cost', 'cbf_margin_cost',
            'task_exploration_area', 'outside_new')
MEAN_KEYS = ('chain_links_available', 'cbf_margin_risk', 'cbf_intervention', 'moving_fraction', 'decision_ms',
             'nominal_speed_mean', 'executed_speed_mean', 'relay_token_count', 'relay_vehicle_known',
             'mission_known', 'backbone_connected')
EXTREME_KEYS = ('cbf_margin_min_m_s', 'chain_max_distance_ratio', 'min_separation', 'relay_max_hop', 'graph_nodes')


class CoGraphExperiment:
    def __init__(self, seed, model, scene, stochastic=False, num_drones=9, ground_speed=1.,
                 intervention_weight=0., chain_soft_ratio=.85, cbf_margin_weight=0., cbf_margin_scale=8.,
                 frontier_progress_weight=0., path_quality_weight=0., time_penalty=0.,
                 relay_loss=0., relay_delay_jitter=0, relay_period=1, relay_capacity=16,
                 fusion_bridge_distance=0., vehicle_planner='parked', claim_options=None):
        architecture = getattr(model, 'policy_architecture', None)
        if architecture not in ('cograph_v1', 'cograph_v2'):
            raise ValueError('CoGraphExperiment requires a CoGraph policy')
        # v2 drones choose their own target (learned); v1 drones take the nearest unconnected one.
        navigator = CoGraphV2Navigator if architecture == 'cograph_v2' else CoGraphNavigator
        if vehicle_planner not in ('parked', 'decentralized'):
            raise ValueError('vehicle_planner must be parked or decentralized')
        self.start = np.asarray(scene['start'], float)
        self.goals = np.asarray(scene['goals'], float)
        self.scene_attempts = scene['attempts']
        self.intervention_weight = float(intervention_weight)
        self.chain_soft_ratio = float(chain_soft_ratio)
        self.cbf_margin_weight, self.cbf_margin_scale = float(cbf_margin_weight), float(cbf_margin_scale)
        self.frontier_progress_weight = float(frontier_progress_weight)
        self.path_quality_weight = float(path_quality_weight)
        self.time_penalty = float(time_penalty)
        bounds = (0., 0., WORLD_M, WORLD_M)
        self.env = MultiRobotEnv(EnvConfig(world_size=(WORLD_M, WORLD_M), dt=1., perception_radius=24.,
            communication_radius=COMMUNICATION_RANGE_M, occupancy_map_resolution=2., occupancy_map_size=12,
            ground_occupancy_map_size=16, ground_occupancy_map_resolution=.5))
        self.vehicle_ids = [self.env.add_ground_robot(x, y, max_speed=ground_speed)
                            for x, y in scene['vehicle_starts']]
        for position, radius in scene['circles']:
            self.env.add_circle_obstacle(*position, radius)
        # The historical N!=3 serpentine of 18 m-spaced columns beside the vehicles.
        sign = 1. if self.start[0] < WORLD_M/2 else -1.
        self.ids, self.controllers = [], []
        for i in range(num_drones):
            x, y = self.start+np.asarray([sign*(9.+18.*(i//3)), (-18.+18.*(i % 3))*(1. if (i//3) % 2 == 0 else -1.)])
            self.ids.append(self.env.add_drone(x, y, FLIGHT_HEIGHT_M, max_speed=6.))
            c = DroneController(f'drone_{i}', resolution=2., total_drones=num_drones,
                communication_range=COMMUNICATION_RANGE_M, max_speed=6., broadcast_interval=1., fusion_interval=5.,
                communication_mode='backbone', safety_filter='cbf', graph_min_clearance=1.5)
            c.multi_vehicle = True
            c.navigator = navigator(f'drone_{i}', total_drones=num_drones, comm_range=COMMUNICATION_RANGE_M,
                min_separation=15., model=model, deterministic=not stochastic, max_speed=6., world_bounds=bounds)
            c.relay = RelayManager(c.drone_id, capacity=relay_capacity)
            c.graph_builder.world_bounds = bounds
            c.ground_max_speed = 0.  # v1 vehicles hold at the start
            self.controllers.append(c)
        self.obs = self.env.reset()
        self.ledger = DeliveredEvidence(world=WORLD_M)
        self.vehicles, self.vehicle_relays = [], []
        for k in range(len(self.vehicle_ids)):
            vehicle = GroundRobotController(f'ground_{k}', resolution=2., perception_range=8.,
                max_linear_vel=ground_speed, max_angular_vel=1., fusion_interval=5., safety_distance=1.5,
                world_bounds=bounds)
            if fusion_bridge_distance:
                vehicle.graph_fusion.bridge_distance = float(fusion_bridge_distance)
                vehicle.graph_fusion.bridge_check = lambda a, b: self.ledger.segment_clear(a, b, BRIDGE_CLEARANCE_M)
            if self.vehicles:
                # The team starts < 30 m apart, in radio range: it pools what any of them
                # receives into one fused graph, each drone graph once per interval
                # (v2 splits the pool when vehicles leave radio range, _split_pools).
                vehicle.graph_fusion = self.vehicles[0].graph_fusion
                vehicle.last_fusion_per_drone = self.vehicles[0].last_fusion_per_drone
            self.vehicles.append(vehicle)
            self.vehicle_relays.append(RelayManager(vehicle.robot_id, capacity=relay_capacity))
        self.relay_channel = RelayChannel(seed, relay_loss, relay_delay_jitter, relay_period)
        self.audit_buffers = [{} for _ in self.ids]
        count = len(self.goals)
        self.vehicle_status = np.zeros((len(self.vehicles), count), bool)
        self.ground_speed = float(ground_speed)
        self.agents = ([VehicleClaimAgent(v.robot_id, k, **(claim_options or {})) for k, v in enumerate(self.vehicles)]
                       if vehicle_planner == 'decentralized' else None)
        self.vehicle_travel, self.vehicle_spread_max, self.pool_splits = 0., 0., 0
        self.learned_choice = architecture == 'cograph_v2'
        self.distinct_foci = []
        self.vehicle_actions = [self._vehicle_action(k) for k in range(len(self.vehicles))]
        self.initial_distance = np.linalg.norm(self.goals-self.start, axis=1)
        self.trackers = [RouteQualityTracker(length) for length in scene['reference_polyline_lengths']]
        self.connected = np.zeros(count, bool)
        self.lengths = np.full(count, np.inf)
        self.frontier = self.initial_distance.copy()
        self.progress_best = np.zeros(count)
        self.quality = np.zeros(count)
        self.quality_best = np.zeros(count)
        self.first_connection = np.full(count, -1)
        region = np.zeros_like(self.ledger.seen)
        for goal in self.goals:
            region |= self.ledger.task_mask(self.start, goal)
        self.region = region
        self.trace = []

    def _vehicle_action(self, k):
        """Fuse what was delivered; broadcast every target with route status.
        v1 holds position; v2 drives toward the vehicle's own departed claim."""
        received = dict(self.obs[self.vehicle_ids[k]])
        received['full_state'] = {'time': self.env.time}
        received.pop('obstacles', None)
        action = self.vehicles[k].get_action(received)
        if self.agents is None:
            action['motion'] = (0., 0.)
        action['broadcast']['targets'] = [[*goal, float(status)] for goal, status
                                          in zip(self.goals.tolist(), self.vehicle_status[k])]
        return action

    def _received_vehicle_messages(self, k):
        """Vehicle broadcasts reaching vehicle k: direct neighbours, and the copies
        the drones forward hop by hop (each drone relays the freshest per vehicle)."""
        messages = []
        for neighbor in self.obs[self.vehicle_ids[k]]['neighbors']:
            message = neighbor.get('broadcast') or {}
            if neighbor['type'] in ('diff_drive', 'holonomic'):
                messages.append(message)
            elif neighbor['type'] == 'drone':
                messages.extend(m.get('broadcast') for m in (message.get('ground_missions') or {}).values())
        return messages

    def _decide(self, per_vehicle, positions):
        """CoGraph v2: every vehicle decides for itself (no leader, no central plan)."""
        now = float(self.env.time)
        for k, (vehicle, agent, metrics) in enumerate(zip(self.vehicles, self.agents, per_vehicle)):
            agent.receive(self._received_vehicle_messages(k))
            here = positions[k][:2]
            straight = np.maximum(np.linalg.norm(self.goals-here, axis=1), 1e-8)
            progress = np.where(metrics['connected'], 1., np.clip(1.-metrics['frontier']/straight, 0., 1.))
            decision = agent.decide(now, here, self.goals, vehicle.unified_graph, progress, metrics['connected'])
            if decision['goal'] is not None:
                goal = tuple(float(v) for v in decision['goal'])
                if vehicle.final_goal is None or not np.allclose(vehicle.final_goal, goal):
                    vehicle.set_goal(goal)
            elif vehicle.final_goal is not None:
                vehicle.stop()
            action = self.vehicle_actions[k]
            if vehicle.final_goal is None:
                # Stopped this cycle: never execute the motion computed before the stop.
                action['motion'] = (0., 0.)
            action['broadcast'].update(agent.message())
        moving = (any(v.goal_position is not None for v in self.vehicles)
                  or any(np.linalg.norm(a['motion']) > 0. for a in self.vehicle_actions))
        for c in self.controllers:
            c.ground_max_speed = self.ground_speed if moving else 0.

    def _split_pools(self, positions):
        """Vehicles exchange fused graphs directly only within radio range: a group
        that drifts out of range of the rest continues on its own copy."""
        groups = {}
        for k, vehicle in enumerate(self.vehicles):
            groups.setdefault(id(vehicle.graph_fusion), []).append(k)
        for members in groups.values():
            components, seen = [], set()
            for k in members:
                if k in seen:
                    continue
                component, stack = [], [k]
                seen.add(k)
                while stack:
                    a = stack.pop()
                    component.append(a)
                    for b in members:
                        if b not in seen and np.linalg.norm(positions[a][:2]-positions[b][:2]) <= COMMUNICATION_RANGE_M:
                            seen.add(b)
                            stack.append(b)
                components.append(sorted(component))
            for component in sorted(components)[1:]:
                source = self.vehicles[component[0]]
                fusion, last = copy.deepcopy(source.graph_fusion), dict(source.last_fusion_per_drone)
                for k in component:
                    self.vehicles[k].graph_fusion, self.vehicles[k].last_fusion_per_drone = fusion, last
                self.pool_splits += 1

    def _vehicle_positions(self):
        return [np.r_[np.asarray(self.obs[g]['self']['position'], float)[:2], 0.] for g in self.vehicle_ids]

    def critic_state(self, frame, budget, stability=0.):
        """Centralized critic input at decision time; training only, never acting."""
        drones = np.asarray([self.obs[r]['self']['position'] for r in self.ids], float)
        vehicles = {v.robot_id: q for v, q in zip(self.vehicles, self._vehicle_positions())}
        _, longest = backbone_metrics_multi({str(i): p for i, p in enumerate(drones)}, vehicles,
                                            COMMUNICATION_RANGE_M)
        separation = min(np.linalg.norm(drones[i]-drones[j]) for i in range(len(drones)) for j in range(i))
        pending = ~self.connected
        spread = np.linalg.norm(drones[:, :2]-self.start, axis=1)
        return np.array([frame/budget, (budget-frame)/1000., self.ledger.seen.sum()*4/1e5,
            self.connected.mean(), stability, self.progress_best.mean(), len(self.goals)/8.,
            self.initial_distance[pending].mean()/400. if pending.any() else 0.,
            self.initial_distance[pending].max()/400. if pending.any() else 0.,
            self.quality_best.mean(), longest/95., separation/100., spread.mean()/400., spread.max()/400.,
            self.frontier[pending].min()/400. if pending.any() else 0., pending.sum()/8.], np.float32)

    def _relay_vehicles(self, frame):
        for k, (vehicle, relay) in enumerate(zip(self.vehicles, self.vehicle_relays)):
            own = self.obs[self.vehicle_ids[k]]['self']
            message = self.vehicle_actions[k]['broadcast']
            relay.originate(float(message['timestamp']), 'vehicle_state', self._vehicle_positions()[k],
                            own.get('velocity'), None)
            if self.relay_channel.due(vehicle.robot_id, frame):
                self.relay_channel.transmit(relay.outgoing(float(message['timestamp'])),
                    [n['id'] for n in self.obs[self.vehicle_ids[k]]['neighbors'] if n['type'] == 'drone'], frame)

    def step(self, frame):
        old_seen = self.ledger.seen.copy()
        old_vehicles = self._vehicle_positions()
        actions = {g: a for g, a in zip(self.vehicle_ids, self.vehicle_actions)}
        self._relay_vehicles(frame)
        samples, decision_ms, old = [], [], []
        interventions, infeasible, chain_terms, margins, relay_rows = [], 0, [], [], []
        for index, (rid, c) in enumerate(zip(self.ids, self.controllers)):
            received = dict(self.obs[rid])
            received['relay_envelopes'] = self.relay_channel.deliver(rid, frame)
            old.append(np.asarray(received['self']['position']).copy())
            tic = time.perf_counter()
            action = c.get_action(received)
            decision_ms.append((time.perf_counter()-tic)*1000)
            now = float(received['full_state']['time'])
            relay_rows.append(c.relay.tokens(now))
            if self.relay_channel.due(rid, frame):
                self.relay_channel.transmit(c.relay.outgoing(now),
                    [n['id'] for n in received['neighbors'] if n['type'] == 'drone'], frame)
            # Never mutate a command after the CBF; an infeasible filter ends the episode.
            stats = c.safety_stats
            chain_terms.append(communication_chain_terms(stats, self.chain_soft_ratio, c.communication_range, c.max_speed))
            margins.append(float('-inf') if frame and stats.get('missing_links') else stats.get('feasibility_margin'))
            infeasible += bool(stats.get('infeasible', False))
            interventions.append(float(stats.get('intervention', 0.))**2/36.)
            actions[rid] = action
            if isinstance(action.get('broadcast'), dict):
                # Training-only audit payload; one hop per step, never read by a navigator.
                action['broadcast']['evidence_audit'] = dict(source=rid, frame=frame, position=old[-1].copy(),
                    occupancy=received.get('occupancy_map'), resolution=received['map_resolution'],
                    valid=received.get('sensor_valid', True))
                buffer = self.audit_buffers[index]
                for neighbor in received['neighbors']:
                    for packet in (neighbor.get('broadcast') or {}).get('evidence_audits', []):
                        if frame-packet['frame'] <= 30:
                            buffer[(packet['source'], packet['frame'])] = packet
                buffer[(rid, frame)] = action['broadcast']['evidence_audit']
                self.audit_buffers[index] = {k: v for k, v in buffer.items() if frame-v['frame'] <= 30}
                action['broadcast']['evidence_audits'] = list(self.audit_buffers[index].values())
            samples.append(c.navigator.last_sample)
        if infeasible:
            raise SafetyTermination(infeasible, samples)
        if self.learned_choice:
            chosen = {c.navigator.chain.choice for c in self.controllers} - {None}
            self.distinct_foci.append(len(chosen))
        self.obs, info = self.env.step(actions)
        self.last_executed_velocities = np.asarray([actions[rid]['motion'] for rid in self.ids]).copy()
        current = np.array([self.obs[r]['self']['position'] for r in self.ids])
        vehicles = self._vehicle_positions()
        if self.agents is not None:
            self._split_pools(vehicles)
        for k, gid in enumerate(self.vehicle_ids):
            self.vehicle_actions[k] = self._vehicle_action(k)
            for neighbor in self.obs[gid]['neighbors']:
                for packet in (neighbor.get('broadcast') or {}).get('evidence_audits', []):
                    self.ledger.receive(packet['source'], packet['frame'], packet['position'],
                                        packet['occupancy'], packet['resolution'], valid=packet['valid'])
        for vehicle in self.vehicles:
            vehicle.unified_graph = vehicle.graph_fusion.get_unified_graph()
        if self.agents is not None:
            self.vehicle_travel += float(sum(np.linalg.norm(a-b) for a, b in zip(vehicles, old_vehicles)))
            self.vehicle_spread_max = max(self.vehicle_spread_max, max(
                (float(np.linalg.norm(a-b)) for a in vehicles for b in vehicles), default=0.))
        # Per-vehicle option space on each vehicle's own delivered graph.
        per_vehicle = [multi_target_graph_metrics(v.unified_graph, q, self.goals, self.ledger.connector_clear)
                       for v, q in zip(self.vehicles, vehicles)]
        self.vehicle_status = np.asarray([m['connected'] for m in per_vehicle])
        for k, action in enumerate(self.vehicle_actions):
            action['broadcast']['targets'] = [[*goal, float(s)] for goal, s in zip(self.goals.tolist(), self.vehicle_status[k])]
        self.connected = self.vehicle_status.any(0)
        if self.agents is not None:
            self._decide(per_vehicle, vehicles)
        self.lengths = np.min([m['length'] for m in per_vehicle], axis=0)
        self.frontier = np.min([m['frontier'] for m in per_vehicle], axis=0)
        self.first_connection[(self.first_connection < 0) & self.connected] = frame+1
        count = len(self.goals)
        progress = np.where(self.connected, 1., np.clip(1.-self.frontier/np.maximum(self.initial_distance, 1e-8), 0., 1.))
        frontier_gain = np.maximum(progress-self.progress_best, 0.)
        self.progress_best = np.maximum(self.progress_best, progress)
        quality_gain = np.zeros(count)
        for m, tracker in enumerate(self.trackers):
            terms = tracker.update(dict(graph_task_connected=float(self.connected[m]),
                graph_task_path_length=float(self.lengths[m]) if self.connected[m] else 0.))
            quality_gain[m], self.quality[m] = terms['route_quality_gain'], terms['route_quality']
            self.quality_best[m] = terms['route_best_quality']
        metrics = dict(frontier_progress_reward=self.frontier_progress_weight*frontier_gain.sum()/count,
            path_quality_reward=self.path_quality_weight*quality_gain.sum()/count,
            route_quality_gain=float(quality_gain.sum()), route_quality=float(self.quality.mean()),
            targets_connected=float(self.connected.sum()), all_connected=float(self.connected.all()),
            frontier_best_progress=float(self.progress_best.mean()),
            vehicle_target_pairs_connected=float(self.vehicle_status.mean()),
            task_exploration_area=float(np.count_nonzero(self.ledger.seen & ~old_seen & self.region)*4.),
            outside_new=float(np.count_nonzero(self.ledger.seen & ~old_seen & ~self.region)*4.))
        metrics['intervention_cost'] = self.intervention_weight*float(np.mean(interventions))
        metrics['cbf_intervention'] = float(np.mean(interventions))
        metrics.update(cbf_margin_terms(margins, self.cbf_margin_scale))
        metrics['cbf_margin_cost'] = self.cbf_margin_weight*metrics['cbf_margin_risk']
        reward = (metrics['frontier_progress_reward']+metrics['path_quality_reward']-self.time_penalty
                  -metrics['intervention_cost']-metrics['cbf_margin_cost'])
        # Independent evaluator checks of the backbone and swept geometry.
        # Before the first broadcasts, received keys fall back to robot_<simulator id>.
        lookup = {c.drone_id: i for i, c in enumerate(self.controllers)}
        lookup.update({f'robot_{rid}': i for i, rid in enumerate(self.ids)})
        position = {v.robot_id: q for v, q in zip(self.vehicles, vehicles)}
        position.update({f'robot_{gid}': q for gid, q in zip(self.vehicle_ids, vehicles)})
        # v1 vehicles never move; v2 vehicles are checked over their swept step too.
        position_before = position if self.agents is None else dict(
            [(v.robot_id, q) for v, q in zip(self.vehicles, old_vehicles)]
            +[(f'robot_{gid}', q) for gid, q in zip(self.vehicle_ids, old_vehicles)])
        link_distances = []
        for i, c in enumerate(self.controllers):
            for key in c.navigator.required_links:
                before, after = ((position_before[key], position[key]) if key in position
                                 else (old[lookup[key]], current[lookup[key]]))
                link_distances.append(max(np.linalg.norm(old[i]-before), np.linalg.norm(current[i]-after)))
        connected, _ = backbone_metrics_multi({str(i): p for i, p in enumerate(current)}, position,
                                              COMMUNICATION_RANGE_M)
        sep = []
        for i in range(len(self.ids)):
            for j in range(i):
                r, v = old[i]-old[j], (current[i]-old[i])-(current[j]-old[j])
                t = np.clip(-np.dot(r, v)/max(np.dot(v, v), 1e-8), 0, 1)
                sep.append(np.linalg.norm(r+t*v))
        rows = [row for tokens in relay_rows for row in tokens]
        metrics.update(reward=float(reward), backbone_connected=float(connected),
            chain_links_available=float(all(t['chain_links_available'] for t in chain_terms) and connected),
            chain_max_distance_ratio=max(link_distances, default=0.)/COMMUNICATION_RANGE_M,
            chain_link_violation=float(max(link_distances, default=0.) > COMMUNICATION_RANGE_M+1e-5),
            min_separation=float(min(sep)), collisions=float(min(sep) < 2.),
            spacing_violation=float(min(sep) < 15.-1e-5),
            moving_fraction=float(np.mean(np.linalg.norm(current-np.asarray(old), axis=1) > .1)),
            distance=float(np.linalg.norm(current-np.asarray(old), axis=1).sum()),
            decision_ms=float(np.mean(decision_ms)), out_of_bounds=len(info['out_of_bounds']),
            graph_nodes=float(max(m['nodes'] for m in per_vehicle)),
            nominal_speed_mean=float(np.mean([np.linalg.norm(c.navigator.nominal_velocity) for c in self.controllers])),
            executed_speed_mean=float(np.mean(np.linalg.norm(self.last_executed_velocities, axis=1))),
            relay_token_count=float(np.mean([len(t) for t in relay_rows])),
            relay_vehicle_known=float(np.mean([any(r['payload_type'] == 'vehicle_state' for r in t) for t in relay_rows])),
            relay_max_hop=float(max((r['hop'] for r in rows), default=0)),
            mission_known=float(np.mean([c.navigator.chain.task() is not None for c in self.controllers])))
        self.trace.append(dict(positions=current.tolist(), **metrics))
        return samples, reward, metrics

    def summary(self):
        # A failure on the very first step leaves no executed step: report zeros.
        trace = self.trace or [dict.fromkeys(SUM_KEYS+MEAN_KEYS+EXTREME_KEYS, 0.)]
        total = {k: float(sum(t[k] for t in trace)) for k in SUM_KEYS}
        for k in MEAN_KEYS:
            total[k] = float(np.mean([t[k] for t in trace]))
        total['cbf_margin_min_m_s'] = float(min(t['cbf_margin_min_m_s'] for t in trace))
        total['chain_max_distance_ratio'] = float(max(t['chain_max_distance_ratio'] for t in trace))
        total['min_separation'] = float(min(t['min_separation'] for t in trace))
        total['relay_max_hop'] = float(max(t['relay_max_hop'] for t in trace))
        connected = self.connected
        straight = np.maximum(self.initial_distance, 1e-8)
        total.update(episode_steps=len(self.trace), scene_attempts=self.scene_attempts, num_drones=len(self.ids),
            num_vehicles=len(self.vehicles), target_count=len(self.goals),
            targets_connected=float(connected.sum()), targets_connected_fraction=float(connected.mean()),
            graph_task_connected=float(connected.all()),
            graph_task_path_ratio=float(np.mean(self.lengths[connected]/straight[connected])) if connected.any() else 0.,
            route_quality=float(self.quality.mean()), route_best_quality=float(self.quality_best.mean()),
            frontier_best_progress=float(self.progress_best.mean()),
            vehicle_target_pairs_connected=float(self.vehicle_status.mean()),
            target_first_connection_step_mean=float(self.first_connection[connected].mean()) if connected.any() else -1.,
            graph_first_connection_step=float(self.first_connection.max()) if connected.all() else -1.,
            graph_nodes=trace[-1]['graph_nodes'],
            coverage_area=float(self.ledger.seen.sum()*4), confidence_area=float(self.ledger.confidence().sum()*4),
            task_share=total['task_exploration_area']/max(total['task_exploration_area']+total['outside_new'], 1e-8),
            reward_beta=1., reward_eta=0., ground_distance=0., ground_goal_distance=0.,
            mean_target_distance_m=float(self.initial_distance.mean()),
            max_target_distance_m=float(self.initial_distance.max()))
        for key, value in self.relay_channel.stats.items():
            total[f'relay_packets_{key}'] = float(value)
        if self.learned_choice:
            total.update(drone_distinct_targets=float(np.mean(self.distinct_foci)) if self.distinct_foci else 0.,
                drone_target_decisions=float(np.mean([c.navigator.decisions for c in self.controllers])))
        if self.agents is not None:
            agents = self.agents
            total.update(ground_distance=self.vehicle_travel, vehicle_spread_max_m=self.vehicle_spread_max,
                vehicle_claims=float(sum(a.claims_made for a in agents)),
                vehicle_transfers=float(sum(a.transfers_out for a in agents)),
                vehicle_claim_conflicts=float(sum(a.conflicts for a in agents)),
                vehicle_switches=float(sum(a.switches for a in agents)),
                vehicle_departures=float(sum(a.departures for a in agents)),
                targets_served=float(len(set().union(*(a.served for a in agents)))),
                graph_pool_splits=float(self.pool_splits))
        return total


def episode_budget(config, scene):
    reference = sum(scene['reference_lengths'])
    if config.get('budget_reference', 'sum') == 'tree':
        # CoGraph v2: the graph the drones build at once, the coverage tree,
        # stretched by the scene's own obstacle detour (reference / straight).
        start, goals = np.asarray(scene['start'], float), np.asarray(scene['goals'], float)
        detour = reference/max(float(np.linalg.norm(goals-start, axis=1).sum()), 1e-8)
        reference = coverage_tree(start, goals)[1]*max(detour, 1.)
    return min(config['steps'], max(config['min_episode_steps'],
        math.ceil(config['budget_multiplier']*reference+config['budget_extra'])))


def run_cograph_episode(model, config, scene_seed, schedule, stochastic, progress=None, stop=None):
    """One CoGraph episode with v8's map termination and improvement phase."""
    scene = generate_cograph_scene(scene_seed, tuple(config['target_count_range']),
        tuple(config['task_distance_range']), config['distance_sampling'], config['vehicle_count'],
        single_task=config.get('scene_regime', 'any') == 'single_task', num_drones=config['fleet_sizes'][0])
    budget = episode_budget(config, scene)
    with contextlib.redirect_stdout(io.StringIO()):
        env = CoGraphExperiment(scene_seed, model, scene, stochastic=stochastic, num_drones=config['fleet_sizes'][0],
            ground_speed=config['ground_speed'], intervention_weight=config['intervention_weight'],
            chain_soft_ratio=config['chain_soft_ratio'], cbf_margin_weight=config['cbf_margin_weight'],
            cbf_margin_scale=config['cbf_margin_scale'], frontier_progress_weight=config['frontier_progress_weight'],
            path_quality_weight=config['path_quality_weight'], time_penalty=config['time_penalty'],
            relay_loss=config['relay_loss'], relay_delay_jitter=config['relay_delay_jitter'],
            relay_period=config['relay_period'], relay_capacity=config['relay_capacity'],
            fusion_bridge_distance=config['fusion_bridge_distance'], vehicle_planner=config.get('vehicle_planner', 'parked'),
            claim_options=dict(depart_progress=config.get('vehicle_depart_progress', .5),
                               transfer_margin=config.get('vehicle_transfer_margin', .15)))
        rollout, reason, stable, max_link, bonus = [], 'budget', 0, 0., 0.
        improve = config['map_improvement_steps'] > 0
        milestone = last_gain = -1
        try:
            for frame in range(budget):
                if stop is not None and stop() and rollout:
                    reason = 'user_stop'; break
                phase = min(stable/config['map_stability_steps'], 1.)
                if milestone >= 0:
                    phase = 1.+min((frame-1-max(last_gain, milestone))/config['map_improvement_steps'], 1.)
                state = env.critic_state(frame, budget, phase)
                try:
                    samples, reward, metrics = env.step(frame)
                except SafetyTermination as exc:
                    # The proposed joint action becomes an absorbing failure; nothing unsafe executes.
                    rollout.append((*[np.array([s[k] for s in exc.samples]) for k in range(4)],
                                    -config['safety_failure_penalty'], state))
                    reason = 'cbf_infeasible'; break
                rollout.append((*[np.array([s[k] for s in samples]) for k in range(4)], reward, state))
                max_link = max(max_link, metrics['chain_max_distance_ratio']*COMMUNICATION_RANGE_M)
                mapped = (metrics['all_connected'] and metrics['chain_links_available']
                          and metrics['chain_max_distance_ratio'] <= 1.)
                stable = stable+1 if mapped else 0
                if progress is not None and (frame % 25 == 0 or frame+1 == budget):
                    progress(dict(step=frame+1, budget=budget, scene_seed=scene_seed, world_size=WORLD_M,
                        target_count=len(env.goals), targets_connected=int(metrics['targets_connected']),
                        graph_connected=bool(stable), delivered_area_m2=float(env.ledger.seen.sum()*4),
                        max_link_distance_m=max_link))
                if improve:
                    if metrics['route_quality_gain'] > 0:
                        last_gain = frame
                    if milestone < 0 and stable >= config['map_stability_steps']:
                        milestone, bonus = frame, config['map_success_reward']
                        *head, last, row_state = rollout[-1]
                        rollout[-1] = (*head, last+bonus, row_state)
                    if milestone >= 0:
                        if stable >= config['map_stability_steps'] and metrics['route_quality'] >= config['map_target_quality']:
                            reason = 'map_optimal'; break
                        if frame-max(last_gain, milestone) >= config['map_improvement_steps']:
                            reason = 'map_plateau'; break
                elif stable >= config['map_stability_steps']:
                    reason, bonus = 'map_connected', config['map_success_reward']
                    *head, last, row_state = rollout[-1]
                    rollout[-1] = (*head, last+bonus, row_state)
                    break
            summary = env.summary()
            if reason == 'cbf_infeasible':
                summary['reward'] -= config['safety_failure_penalty']
            summary['reward'] += bonus
            completed = reason == 'map_connected' or milestone >= 0
            summary.update(uav_reward=summary['reward'], map_completion_bonus=bonus,
                safety_failure_cost=config['safety_failure_penalty'] if reason == 'cbf_infeasible' else 0.,
                safety_terminated=float(reason == 'cbf_infeasible'), map_completed=float(completed),
                budget_exhausted=float(reason == 'budget' and not completed),
                user_stopped=float(reason == 'user_stop'), episode_budget_steps=budget,
                task_distance_m=float(np.mean(scene['distances'])),
                reference_length_m=float(sum(scene['reference_lengths'])), max_link_distance_m=max_link,
                rollout_steps=len(rollout), route_efficiency=summary.get('route_quality', 0.) if completed else 0.,
                map_completion_step=float(milestone+1 if milestone >= 0 else -1),
                map_optimal=float(reason == 'map_optimal'), map_plateau=float(reason == 'map_plateau'),
                improvement_steps=float(len(rollout)-milestone-1 if milestone >= 0 else 0))
        finally:
            env.env.close()
    return rollout, summary
