"""V11 establishes a fixed a-b route using the v9 delivered-map rewards.

The vehicle stays at its real c for motion, broadcasts and CBF. Route evidence
comes from the actual vehicle receiver, matching v9's delivery contract.
"""
import contextlib
import io
import math
import numpy as np
from simulator.mappo_experiment import Experiment, SafetyTermination, navigation_cost
from simulator.navigation_metrics import task_graph_metrics, vehicle_route_update
from simulator.survey_experiment import SurveyExperiment, make_survey_scene as sample_survey_scene, validate_survey_config
from simulator.route_ab_scenes import sample_certified_scene, validate_distance_bands


def route_radio_feasible(a, b, c, count):
    """Sufficient straight-link relay construction, with 20% radio reserve.

    Count the drones needed along each of three possible two-edge trees.
    A drone occupies each task endpoint and intermediate relay positions.
    Using the smaller ground-air horizontal radius for every edge is conservative.
    This geometric certificate does not model obstacles or dynamic CBF feasibility.
    """
    radius = .8*math.sqrt((95.-.25)**2-15.**2)
    def links(p, q):
        return max(1, math.ceil(float(np.linalg.norm(np.asarray(p)-q))/radius))
    ca, cb, ab = links(c,a), links(c,b), links(a,b)
    return min(ca+ab, cb+ab, ca+cb) <= count


def make_survey_scene(config, seed):
    return sample_certified_scene(config, seed, sample_survey_scene, route_radio_feasible)


def validate_route_ab_config(config):
    validate_distance_bands(config)
    if config['reward_mode'] != 'velocity_mapping' or config['termination'] != 'map':
        raise ValueError('V11 requires route rewards and map completion')
    if config['arrival_reward'] or config['map_quality_reward'] or config['map_improvement_steps']:
        raise ValueError('V11 completes on stable a-b connectivity')
    # Reuse validation of the physical setup and task input, independently of
    # the route reward weights validated by the common training entry point.
    physical = dict(config, reward_mode='local_survey', termination='survey',
        route_reward_weight=0., path_quality_weight=0., frontier_progress_weight=0.,
        mapping_reward_until_connected=False)
    validate_survey_config(physical)


class RouteABExperiment(SurveyExperiment):
    def __init__(self, *args, route_reward_weight=10., path_quality_weight=20.,
                 frontier_progress_weight=45., mapping_until_connected=True, **kwargs):
        self.vehicle_target = None
        self.vehicle_target_selected_at = -100.
        self.vehicle_target_switches = 0
        self.vehicle_blocked_steps = 0
        self.ab_route_weight = route_reward_weight
        self.ab_quality_weight = path_quality_weight
        self.ab_frontier_weight = frontier_progress_weight
        self.ab_handover = mapping_until_connected
        self._ab_frontier_best = 0.
        super().__init__(*args, route_reward_weight=0., path_quality_weight=0.,
                         frontier_progress_weight=0., **kwargs)
        self._graph_metrics = self.route_metrics()
        self.first_graph_connection_step = -1

    def _select_vehicle_target(self, position):
        graph = self.ground_controller.unified_graph
        if self.ground_stationary or not hasattr(self,'ledger') or graph is None or not len(graph):
            return position.copy()
        reached = self.vehicle_target is None or np.linalg.norm(self.vehicle_target-position) <= 3.
        if not reached and self.vehicle_blocked_steps < 15:
            return self.vehicle_target
        if self.env.time-self.vehicle_target_selected_at < 10.:
            return position.copy() if reached else self.vehicle_target
        self.vehicle_target_selected_at = self.env.time
        candidates = []
        for _,data in graph.graph.nodes(data=True):
            point = np.asarray(data['position'],float)[:2]
            distance = np.linalg.norm(point-position)
            if 15. <= distance <= 60. and route_radio_feasible(
                    self.survey_task.a,self.survey_task.b,point,len(self.ids)):
                candidates.append(point)
        # Seeded sampling avoids a fixed preference for b. Every accepted goal
        # is connected to c in the map actually delivered to the vehicle.
        for index in self.rng.permutation(len(candidates))[:24]:
            point = candidates[index]
            if task_graph_metrics(graph,position,point,self._route_connector_clear)['graph_task_connected']:
                self.vehicle_target_switches += 1
                self.vehicle_blocked_steps = 0
                return point.copy()
        return position.copy()

    def _navigation_action(self, neighbors=None):
        received = dict(self.obs[self.ground])
        position = np.asarray(received['self']['position'],float)[:2]
        target = self._select_vehicle_target(position)
        if self.vehicle_target is None or not np.array_equal(target,self.vehicle_target):
            self.vehicle_target = target
            self.ground_controller.set_goal(tuple(target))
        received['neighbors'] = received['neighbors'] if neighbors is None else neighbors
        received['full_state'] = {'time':self.env.time}
        received.pop('obstacles',None)
        action = self.ground_controller.get_action(received)
        self.goal_reached = bool(np.linalg.norm(position-target) <= 3.)
        if self.ground_stationary or self.goal_reached:
            action['motion'] = (0.,0.)
        if action.get('broadcast') is not None:
            action['broadcast']['survey'] = self.survey_task.to_message()
            action['broadcast']['vehicle_navigation_target'] = target.tolist()
        return action

    def _ground_action(self, neighbors=None):
        action = self._navigation_action(neighbors)
        position = np.asarray(self.obs[self.ground]['self']['position'],float)[:2]
        # Differential drive translates at the CURRENT heading, then rotates.
        heading = self.ground_controller.heading
        direction = np.array([math.cos(heading),math.sin(heading)])
        speed, turn = action['motion']
        received = self.obs[self.ground]['neighbors'] if neighbors is None else neighbors
        ground_position = np.asarray(self.obs[self.ground]['self']['position'],float)
        anchors = [ground_position+np.asarray(n['relative_position'],float) for n in received
                   if n.get('relative_position') is not None]
        # Preserve the obstacle-aware planner's direction. Slow or wait when its
        # proposed move would leave the task relay envelope or nearest radio link.
        dt = self.env.config.dt
        for scale in (1., .5, .25, 0.):
            candidate = position+float(speed)*scale*dt*direction
            task_ok = route_radio_feasible(self.survey_task.a,self.survey_task.b,candidate,len(self.ids))
            radio_ok = any(np.linalg.norm(np.r_[candidate,0.]-q) <= 90. for q in anchors)
            if task_ok and radio_ok:
                action['motion'] = (float(speed)*scale,turn)
                self.vehicle_blocked_steps = self.vehicle_blocked_steps+1 if scale == 0. and speed != 0. else 0
                return action
        action['motion'] = (0.,turn)
        self.vehicle_blocked_steps += 1
        return action

    def _route_connector_clear(self, start, end):
        """Received free evidence covers the connector and its safety margin.

        The ledger scans a lateral strip. Extend both ends so the checked
        rectangle also covers the endpoint caps, including zero-length joins.
        This is conservative and uses no simulator obstacle truth.
        """
        start, end = np.asarray(start, float)[:2], np.asarray(end, float)[:2]
        delta = end-start
        length = float(np.linalg.norm(delta))
        direction = delta/length if length > 1e-8 else np.array([1., 0.])
        margin = self.ground_safety_distance
        return self.ledger.segment_clear(start-margin*direction,
            end+margin*direction, margin)

    def route_metrics(self):
        return task_graph_metrics(self.ground_controller.unified_graph,
            np.asarray(self.survey_task.a), np.asarray(self.survey_task.b), self._route_connector_clear)

    def _mapping_potential(self, metrics, cost, reference_length):
        """A bounded state potential; connected-map handover is reversible.

        When handover is enabled all connected graphs share potential zero.
        Losing connectivity therefore returns the previous mapping credit,
        while improving an already connected path earns only quality reward.
        """
        if self.ab_handover and metrics['graph_task_connected']:
            return 0.
        return -float(np.clip(cost/(max(reference_length, 1.)/.05), 0., 1.))

    def critic_state(self, frame, budget, stability=0.):
        state = super().critic_state(frame, budget, stability)
        metrics = self.route_metrics()
        terms = vehicle_route_update(metrics, metrics, self.survey_task.a, self.survey_task.b)
        state[2:4] = [metrics['graph_task_connected'], terms['route_quality']]
        return state

    def step(self, frame):
        a, b = np.asarray(self.survey_task.a), np.asarray(self.survey_task.b)
        before = self.route_metrics()
        cost_before = navigation_cost(self.ground_controller.unified_graph, a, b)
        # Parent navigation weights are zero: this yields only physical costs.
        # Bypass SurveyExperiment.step so local confidence earns no reward.
        first_connection = self.first_graph_connection_step
        samples, reward, metrics = Experiment.step(self, frame)
        after = self.route_metrics()
        cost_after = navigation_cost(self.ground_controller.unified_graph, a, b)
        terms = vehicle_route_update(before, after, a, b)
        reference = terms['route_reference_length_m']
        mapping = self.ab_route_weight*(
            self._mapping_potential(after, cost_after, reference)-
            self._mapping_potential(before, cost_before, reference))
        self._ab_frontier_best = max(self._ab_frontier_best, terms['frontier_progress'])
        terms.update(mapping_reward=mapping,
            frontier_best_progress=self._ab_frontier_best,
            path_quality_reward=self.ab_quality_weight*terms['route_quality_gain'],
            first_path_quality_reward=self.ab_quality_weight*terms['route_first_quality_gain'],
            path_shortening_reward=self.ab_quality_weight*terms['route_shortening_quality_gain'],
            frontier_progress_reward=self.ab_frontier_weight*terms['frontier_progress_gain'])
        reward += sum(terms[k] for k in ('mapping_reward','path_quality_reward','frontier_progress_reward'))
        connected, longest = self.connectivity()
        self._graph_metrics = after
        self.first_graph_connection_step = frame+1 if after['graph_task_connected'] and first_connection < 0 else first_connection
        metrics.update(after, **terms)
        metrics.update(reward=float(reward), zero_reward=float(abs(reward)<1e-8),
            route_origin_x=float(a[0]), route_origin_y=float(a[1]),
            route_target_x=float(b[0]), route_target_y=float(b[1]),
            survey_connected=float(connected), survey_disconnected_steps=float(not connected),
            survey_max_link_distance_m=float(longest))
        self.trace[-1].update(metrics)
        return samples, reward, metrics

    def summary(self):
        result = Experiment.summary(self)
        # Parent empty-trace summaries use the vehicle's C->B navigation task.
        # Keep route diagnostics tied to the fixed A->B task even when the
        # first action is rejected before a physical step is recorded.
        result.update(self.route_metrics())
        result.update(survey_disconnected_steps=sum(row.get('survey_disconnected_steps',0.) for row in self.trace))
        result['frontier_best_progress'] = self._ab_frontier_best
        a, b = self.survey_task.a, self.survey_task.b
        result.update(route_origin_x=a[0], route_origin_y=a[1], route_target_x=b[0], route_target_y=b[1])
        result.update(vehicle_target_switches=float(self.vehicle_target_switches),
            vehicle_navigation_target_x=float(self.vehicle_target[0]),
            vehicle_navigation_target_y=float(self.vehicle_target[1]))
        return result


def run_route_ab_episode(model, config, scene_seed, schedule, stochastic, progress=None, stop=None):
    scene=make_survey_scene(config,scene_seed)
    task=scene['survey_task']
    length=float(np.linalg.norm(np.asarray(task.b)-task.a))
    budget=min(config['steps'],max(config['min_episode_steps'],
        math.ceil(config['budget_multiplier']*length+config['budget_extra'])))
    with contextlib.redirect_stdout(io.StringIO()):
        env=RouteABExperiment(scene_seed,model,survey_task=task,
            survey_confidence_weight=config['survey_confidence_weight'],
            survey_cell_confidence=config['survey_cell_confidence'],
            world_size=config['world_size'],start=scene['start'],
            scene=(scene['goal'],scene['circles'],scene['attempts']),stop_on_ground_arrival=False,
            stochastic=stochastic,drop=config['drop'],exploration_weight=0.,
            exploration_candidates=0,communication_mode='backbone',safety_filter='cbf',
            moving_ground=True,ground_stationary=scene['survey_stationary'],
            ground_speed=config['ground_speed'],ground_safety_distance=config['ground_safety_distance'],
            reward_mode='velocity_mapping',mapping_route_origin='initial',
            route_reward_weight=config['route_reward_weight'],path_quality_weight=config['path_quality_weight'],
            frontier_progress_weight=config['frontier_progress_weight'],
            mapping_until_connected=config['mapping_reward_until_connected'],
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
                complete=bool(metrics['graph_task_connected'] and metrics['survey_connected'])
                stable=stable+1 if complete and not unsafe else 0
                if stable>=config['map_stability_steps']:
                    reason='route_completed';bonus=config['map_success_reward'];reward+=bonus
                rollout.append((*[np.array([s[k] for s in samples]) for k in range(4)],reward,state))
                max_link=max(max_link,metrics['survey_max_link_distance_m'])
                if progress is not None and (frame%25==0 or frame+1==budget):
                    progress(dict(step=frame+1,budget=budget,scene_seed=scene_seed,
                        world_size=config['world_size'],task_distance_m=length,
                        graph_connected=bool(metrics['graph_task_connected']),route_quality=metrics['route_quality'],
                        survey_connected=bool(metrics['survey_connected']),max_link_distance_m=max_link))
                if unsafe or reason=='route_completed':break
            summary=env.summary()
            failed=reason in ('cbf_infeasible','physical_failure')
            summary.update(reward=float(sum(row[-2] for row in rollout)),
                uav_reward=float(sum(row[-2] for row in rollout)),
                route_completed=float(reason=='route_completed'),map_completed=float(reason=='route_completed'),
                map_completion_bonus=bonus,map_completion_step=float(len(env.trace) if bonus else -1),
                safety_terminated=float(failed),safety_failure_cost=config['safety_failure_penalty'] if failed else 0.,
                budget_exhausted=float(reason=='budget'),user_stopped=float(reason=='user_stop'),
                episode_budget_steps=budget,task_distance_m=length,reference_length_m=length,
                max_link_distance_m=max_link,rollout_steps=len(rollout),route_efficiency=summary.get('route_quality',0.),
                survey_a_equals_initial_c=float(scene['survey_anchored']),
                survey_vehicle_stationary=float(scene['survey_stationary']),
                survey_envelope_m=scene['survey_envelope_m'])
        finally:
            env.env.close()
    return rollout,summary
