"""Received robot graphs shared by UAV fast and UGV slow control.

World inference never calls a remote actor. Original telemetry timestamps are
kept even when a separate, fresh local relative measurement anchors a node.
Map/sensor/task evidence stays in the graph's local attributes; estimates never
write back to the occupancy map. Simulator state is used only for labels.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .graph_fast_policy import load_numpy, numpy_state, ppo_batch, state_hash
from .safety_context import received_cbf_context, validate_cbf_context
from .graph_dynamics import (GraphWorldModel, NODE_IDS, NODE_COUNT, infer, belief_tokens,
                             planning_readout, build_road_topology)
from .hierarchical_policy import make_graph_fast, GraphSlowActorCritic
from .hierarchical_runtime import HierarchicalEpisode, HierarchyConfig, geometry_teacher_action
from .telemetry import (ReceivedHistory, TelemetryNetwork, packet, context_features,
                          HISTORY, HISTORY_FEATURES, CONTEXT_FEATURES)
from .wildscenes_runtime import world_sensor_tile
from .frozen_navigation import V9_FAST_REWARD_CONTRACT, runtime, write_json


class TeacherBehavior:
    """Warm-up behavior only; graph-policy distillation uses separate labels."""
    input_version = 2
    feature_version = 8
    action_mode = 'velocity'

    def __init__(self, base, raw_width):
        self.base, self.raw_width = base, raw_width

    def distribution(self, features, mask=None):
        return self.base.distribution(features[..., :self.raw_width])

    def encode_graph(self, *args, **kwargs):
        return self.base.encode_graph(*args, **kwargs)

    def velocity(self, *args, **kwargs):
        return self.base.velocity(*args, **kwargs)


class GraphRecorder:
    def __init__(self, episode, world, fast, teacher, config):
        self.episode, self.env = episode, episode.env
        self.world, self.fast, self.teacher = world, fast, teacher
        self.assigned_target = episode.goal.copy() if config.get('fixed_mission_target') else None
        self.fast.assigned_target = self.assigned_target
        self.horizon, self.stride = config['horizon'], config['stride']
        self.names = {self.env.ground: 'ground_0',
                      **{rid: c.drone_id for rid, c in zip(self.env.ids, self.env.controllers)}}
        self.ids = {name: rid for rid, name in self.names.items()}
        if set(self.ids) != set(NODE_IDS):
            raise ValueError('Graph experiment requires one UGV and three UAVs')
        self.controllers = dict(zip(self.env.ids, self.env.controllers))
        self.network = TelemetryNetwork(NODE_IDS, episode.case['seed'] + 19,
                                       loss=config['loss'], jitter=config['jitter'])
        self.own_history = {name: ReceivedHistory('__local__') for name in NODE_IDS}
        self.frame = -1
        self.inputs, self.transitions, self.cbf_records = [], [], []
        self.truth_states, self.truth_actions = {}, {}
        self.cbf, self.received = {}, {}
        self.world_calls = dict(uav=0, ugv=0)
        self.base_step = self.env.step
        self.env.step = self.step
        self.ground_received = None
        ground_action = self.env.ground_controller.get_action

        def receive_ground(observation):
            self.ground_received = observation
            return ground_action(observation)

        self.env.ground_controller.get_action = receive_ground
        for controller in self.env.controllers:
            name = controller.drone_id
            original = controller.navigator.encode

            def current_margin(observation, controller=controller, name=name):
                dt = observation.get('time_step')
                if dt is None or not np.isfinite(dt) or dt <= 0:
                    raise ValueError('Graph CBF requires a finite positive control step')
                stats = controller._cbf_filter((0., 0., 0.), observation, dt)[1]
                self.cbf[name] = received_cbf_context(controller, observation, stats)
                self.received[name] = observation
                return stats['feasibility_margin']

            controller._current_cbf_margin = current_margin

            def encode(position, observation, original=original, name=name):
                raw, mask = original(position, observation)
                norm = validate_cbf_context(self.cbf[name], name, self.frame)
                if np.float32(norm) != raw[32]:
                    raise ValueError('Current CBF changed between received graph and actor')
                graph = self.graph(name, self.received[name], raw)
                # Immutable local evidence, including CBF, accompanies the
                # predicted robot graph through to the actor feature adapter.
                graph['cbf'] = copy.deepcopy(self.cbf[name])
                tokens = self.predict(graph, 'uav')
                validate_cbf_context(graph['cbf'], name, self.frame)
                if self.teacher is None:
                    # Fresh training has no learned teacher. These typed
                    # placeholders keep the stored world-label schema stable;
                    # teacher_valid prevents their use as distillation labels.
                    teacher_mean, teacher_std = np.zeros(2, np.float32), np.ones(2, np.float32)
                else:
                    with torch.inference_mode():
                        dist = self.teacher.distribution(torch.from_numpy(np.asarray(raw, np.float32)))
                    teacher_mean, teacher_std = dist.mean.numpy().copy(), dist.stddev.numpy().copy()
                graph.update(raw_features=np.asarray(raw, np.float32).copy(),
                             teacher_mean=teacher_mean, teacher_std=teacher_std,
                             teacher_valid=self.teacher is not None,
                             actor_valid=True)
                self.save_input(graph)
                self.cbf_records.append(graph['cbf'])
                return np.r_[raw, tokens.ravel()].astype(np.float32), mask

            controller.navigator.encode = encode

    def own_packet(self, owner):
        rid = self.ids[owner]
        state = self.env.obs[rid]['self']
        if owner == 'ground_0':
            goal = self.episode.manager.target
            task_stamp = self.frame
            ground, nominal = state['position'], state['velocity']
        else:
            nav = self.controllers[rid].navigator
            task = nav.chain.task()
            goal = None if task is None else task['target']
            task_stamp = None if task is None else task['source_time']
            ground = None if task is None else task['position']
            nominal = nav.nominal_velocity
            if self.assigned_target is not None:
                goal = self.assigned_target
                task_stamp = 0. if task_stamp is None else task_stamp
        return packet(owner, self.frame, state['position'], state['velocity'], nominal,
                      goal=goal, goal_stamp=task_stamp, task_ground=ground, is_ground=owner == 'ground_0')

    def record_truth(self, frame):
        self.truth_states[frame] = {name: np.r_[self.env.obs[rid]['self']['position'][:2],
                                               self.env.obs[rid]['self']['velocity'][:2]].astype(np.float32)
                                    for rid, name in self.names.items()}

    def prepare(self, frame):
        if self.frame == frame:
            return
        if frame < self.frame or abs(self.env.env.time - frame) > 1e-6:
            raise ValueError('Graph experiment requires causal dt=1s decisions')
        self.frame = frame
        own = {name: self.own_packet(name) for name in NODE_IDS}
        links = {name: [self.names[n['id']] for n in self.env.obs[rid]['neighbors']
                        if n['id'] in self.names] for rid, name in self.names.items()}
        self.network.advance(own, links, frame)
        self.record_truth(frame)
        self.cbf.clear(); self.received.clear()

    def graph(self, owner, local, raw=None):
        """Construct solely from this receiver's evidence, never target objects."""
        own = self.own_packet(owner)
        rows = self.network.receivers[owner].rows(self.frame)
        histories = self.network.receivers[owner]
        self.own_history[owner].receive([own], self.frame)
        # The local task can change between a slow decision and the first fast
        # step at the same clock. Update only our local record; already-sent
        # packets and every receiver's remote evidence remain immutable.
        refreshed = dict(copy.deepcopy(own), received_at=float(self.frame))
        self.own_history[owner].latest[owner] = refreshed
        self.own_history[owner].histories[owner][-1] = refreshed
        direct = {self.names[n['id']]: n for n in local.get('neighbors', ()) if n['id'] in self.names}
        arrays = dict(history=np.zeros((NODE_COUNT, HISTORY, HISTORY_FEATURES), np.float32),
            history_mask=np.zeros((NODE_COUNT, HISTORY), bool),
            context=np.zeros((NODE_COUNT, CONTEXT_FEATURES), np.float32),
            baseline_states=np.zeros((NODE_COUNT, self.horizon + 1, 4), np.float32),
            baseline_actions=np.zeros((NODE_COUNT, self.horizon, 2), np.float32),
            node_mask=np.zeros(NODE_COUNT, bool), node_type=np.asarray([n == 'ground_0' for n in NODE_IDS], np.int64),
            ego_mask=np.asarray([n == owner for n in NODE_IDS], bool), state_observed=np.zeros(NODE_COUNT, bool),
            origins=np.zeros((NODE_COUNT, 2), np.float32), source_stamp=np.zeros(NODE_COUNT, np.float32))
        sensor_values, sensor_mask = world_sensor_tile(local)
        known = float(sensor_mask.mean())
        local_graph = (self.env.ground_controller.unified_graph if owner == 'ground_0'
                       else self.controllers[self.ids[owner]].current_graph)
        graph_nodes = 0 if local_graph is None else len(local_graph.graph.nodes)
        # Local graph attributes condition the neural world model, not merely
        # the downstream actor. Unknown pixels and map padding stay explicit.
        own_xy = np.asarray(own['position'][:2])
        map_tokens = np.zeros((32, 5), np.float32)
        nearby = []
        if local_graph is not None:
            nodes = list(local_graph.graph.nodes(data=True))
            nodes.sort(key=lambda item: (float(np.linalg.norm(np.asarray(item[1]['position'][:2]) - own_xy)),
                                          str(item[0])))
            nearby = nodes[:32]
            for index, (node, attrs) in enumerate(nearby):
                map_tokens[index] = np.r_[(np.asarray(attrs['position'][:2]) - own_xy) / 95.,
                    local_graph.node_confidence(node), local_graph.graph.degree(node) / 8., 1.]
        if 'road_topology' in self.world.input_keys:
            arrays['road_topology'] = build_road_topology(local_graph, own_xy, own['goal'],
                                                          nearby_nodes=[node for node, _ in nearby])
        arrays['local_evidence'] = np.r_[sensor_values.ravel(), sensor_mask.astype(np.float32).ravel(),
                                        map_tokens.ravel()].astype(np.float32)
        for i, name in enumerate(NODE_IDS):
            if name == owner:
                history, mask, origin, last = self.own_history[owner].encode(owner, self.frame)
                # Tasks can change at this frame after a slow decision. Own task
                # is known locally now; remote cached acknowledgements stay old.
                last = own
            elif name in rows:
                history, mask, origin, last = histories.encode(name, self.frame)
            elif name in direct:
                # A fresh relative sensor observation is a distinct evidence
                # source, not a refreshed forwarded packet.
                n = direct[name]
                p = np.asarray(own['position']) + np.asarray(n['relative_position'])
                v = np.asarray(own['velocity']) + np.asarray(n['relative_velocity'])
                last = packet(name, self.frame, p, v, v, is_ground=name == 'ground_0')
                temporary = ReceivedHistory('__sensor__'); temporary.receive([last], self.frame)
                history, mask, origin, last = temporary.encode(name, self.frame)
            else:
                continue
            age = self.frame - last['stamp']
            p = np.asarray(last['position'][:2]) + age * np.asarray(last['velocity'][:2])
            v = np.asarray(last['velocity'][:2], float)
            observed = name == owner or name in direct
            if name == owner:
                p, v = np.asarray(own['position'][:2]), np.asarray(own['velocity'][:2])
            elif name in direct:
                p = np.asarray(own['position'][:2]) + np.asarray(direct[name]['relative_position'][:2])
                v = np.asarray(own['velocity'][:2]) + np.asarray(direct[name]['relative_velocity'][:2])
            speed_limit = (float(self.env.ground_controller.max_linear_vel) if name == 'ground_0' else 6.)
            v = v * min(1., speed_limit / max(np.linalg.norm(v), 1e-8))
            # UGV low-level motion prior: observed world velocity, held over
            # this short horizon. Unknown turns/stops remain model uncertainty.
            for k in range(self.horizon + 1):
                q = np.clip(p + k * v, 0., self.episode.case['world_size'])
                arrays['baseline_states'][i, k] = np.r_[(q - origin) / 20., v / 6.]
            arrays['baseline_actions'][i] = v / 6.
            arrays['history'][i], arrays['history_mask'][i] = history, mask
            arrays['context'][i] = context_features(own, last, rows, self.frame, known, graph_nodes, [0., 0.])
            arrays['node_mask'][i], arrays['state_observed'][i] = True, observed
            arrays['origins'][i], arrays['source_stamp'][i] = origin, last['stamp']
        arrays.update(frame=np.int64(self.frame), observer_id=np.int64(NODE_IDS.index(owner)),
                      raw_features=np.zeros(self.fast.raw_width, np.float32) if raw is None
                                   else np.asarray(raw, np.float32).copy(),
                      teacher_mean=np.zeros(2, np.float32), teacher_std=np.ones(2, np.float32),
                      teacher_valid=False, actor_valid=False)
        return arrays

    def predict(self, graph, kind, environment=None):
        batch = {key: torch.from_numpy(np.asarray(graph[key]))[None] for key in self.world.input_keys}
        with torch.inference_mode():
            predicted = infer(self.world, batch)
            result = (belief_tokens(predicted, batch)[0].cpu().numpy() if environment is None
                      else planning_readout(predicted, batch, **environment))
        self.world_calls[kind] += 1
        return result

    def ground_observation(self):
        # Use the exact post-loss observation supplied to the ground controller.
        # At initialization no intercepted packet exists yet: keep local own
        # sensing, and leave neighbors unknown until actual receipt.
        if self.ground_received is not None:
            now = self.ground_received.get('full_state', {}).get('time')
            if now == self.frame:
                return self.ground_received
        local = dict(self.env.obs[self.env.ground])
        local['neighbors'] = []
        local['full_state'] = dict(time=float(self.frame))
        return local

    def save_input(self, graph):
        if self.frame % self.stride == 0:
            self.inputs.append({key: np.asarray(value).copy() for key, value in graph.items() if key != 'cbf'})

    def slow_observation(self, observation=None):
        self.prepare(self.episode.frame)
        observation = self.episode.observation() if observation is None else observation
        graph = self.graph('ground_0', self.ground_observation())
        tokens, environment = self.predict(graph, 'ugv', dict(
            belief=self.episode.belief(), current_target=self.episode.manager.target,
            candidate_targets=observation['targets']))
        self.save_input(graph)
        result = dict(observation)
        repeated = np.broadcast_to(tokens.ravel(), (len(observation['features']), tokens.size))
        result['features'] = np.c_[observation['features'], repeated].astype(np.float32)
        # Static observations and task identities survive the world-model
        # readout unchanged; only robot dynamics are predicted by the network.
        result['environment_graph'] = environment
        return result

    def append_transition(self, samples, state, reward, terminal):
        if len(samples) != 3:
            raise ValueError('Require all three actor proposals, including CBF-rejected steps')
        features = np.asarray([s[0] for s in samples], np.float32)
        with torch.inference_mode():
            value = float(self.fast.value(torch.from_numpy(features)[None], state=torch.from_numpy(state)[None])[0])
        self.transitions.append(dict(features=features, states=state,
            actions=np.asarray([s[2] for s in samples], np.float32),
            logp=np.asarray([s[3] for s in samples], np.float32), rewards=np.float32(reward),
            values=np.float32(value), terminals=bool(terminal)))

    def step(self, frame):
        self.prepare(frame)
        state = self.env.critic_state(frame, self.episode.mission_budget)
        # Ground graph is refreshed during the option, not just at slow actions.
        ground_graph = self.graph('ground_0', self.ground_observation())
        self.predict(ground_graph, 'ugv'); self.save_input(ground_graph)
        safety_before = getattr(self.env.env, 'ground_safety_violations', 0)
        try:
            result = self.base_step(frame)
        except self.episode.me.SafetyTermination as error:
            penalty = self.episode.v9_safety_failure_penalty if self.assigned_target is not None else 100.
            self.append_transition(error.samples, state, -penalty, True)
            raise
        samples, reward, _ = result
        self.record_truth(frame + 1)
        actions = {self.names[rid]: np.asarray(action[:2], np.float32)
                   for rid, action in zip(self.env.ids, self.env.last_executed_velocities)}
        actions['ground_0'] = (self.truth_states[frame + 1]['ground_0'][:2] -
                               self.truth_states[frame]['ground_0'][:2])
        self.truth_actions[frame] = actions
        safety_failure = getattr(self.env.env, 'ground_safety_violations', 0) > safety_before
        success = bool(self.env.goal_reached) and not safety_failure
        deadline = self.episode.attempts + 1 >= self.episode.mission_budget and not (success or safety_failure)
        if self.assigned_target is not None:
            # The archived collector stores Experiment.step's mapping and safety reward.
            # Arrival adds one bonus. A WildScenes raster violation is an absorbing safety failure.
            if safety_failure:
                reward = -self.episode.v9_safety_failure_penalty
            elif success:
                reward += self.episode.v9_arrival_reward
        else:
            reward += -100. if safety_failure else 100. if success else -20. if deadline else 0.
        self.append_transition(samples, state, reward, success or safety_failure or deadline)
        return result

    def world_dataset(self):
        samples = []
        for entry in self.inputs:
            row = dict(entry); frame = int(row['frame'])
            states = np.zeros((NODE_COUNT, self.horizon + 1, 4), np.float32)
            actions = np.zeros((NODE_COUNT, self.horizon, 2), np.float32)
            state_mask = np.zeros(states.shape[:-1], bool); action_mask = np.zeros(actions.shape[:-1], bool)
            for i, name in enumerate(NODE_IDS):
                if not row['node_mask'][i]:
                    continue
                for k in range(self.horizon + 1):
                    if frame + k in self.truth_states:
                        truth = self.truth_states[frame + k][name]
                        states[i, k] = np.r_[(truth[:2] - row['origins'][i]) / 20., truth[2:] / 6.]
                        state_mask[i, k] = True
                    if k < self.horizon and frame + k in self.truth_actions:
                        actions[i, k] = self.truth_actions[frame + k][name] / 6.
                        action_mask[i, k] = True
            row.update(target_states=states, target_actions=actions, state_mask=state_mask, action_mask=action_mask)
            samples.append(row)
        if not samples:
            raise ValueError('No real graph samples collected')
        return {key: np.asarray([row[key] for row in samples]) for key in samples[0]}

    def ppo_dataset(self, steps):
        return ppo_batch(self.transitions, steps)


def navigation_outcome(case, summary):
    """Measure final-goal navigation, never map size or accumulated travel."""
    direct = float(np.linalg.norm(np.asarray(case['goal'])[:2] - np.asarray(case['start'])[:2]))
    safety = {key: int(summary.get(key, 0)) for key in
              ('ground_safety_violations', 'raster_collision_attempts', 'raster_unknown_attempts')}
    clean = int(summary['collisions']) == 0 and not any(safety.values())
    arrived = (bool(summary['goal_reached']) and float(summary['remaining_m']) <= 2.
               and summary['termination'] == 'mission_arrival'
               and bool(summary['terminated']) and not bool(summary['truncated']))
    kilometer = direct >= 1000.
    return dict(terrain=case.get('terrain', 'synthetic'), map_group=case.get('map_group'),
                source_unknown_space=case.get('source_unknown_space'),
                map_sha256=case.get('map_sha256'), direct_distance_m=direct,
                reference_distance_m=float(case.get('reference_distance_m', direct)),
                mission_budget=summary.get('mission_budget'), **safety,
                ground_motion_filter_interventions=int(summary.get('ground_motion_filter_interventions', 0)),
                navigation_success=bool(arrived and clean), kilometer_task=kilometer,
                kilometer_success=bool(kilometer and arrived and clean))


def collect_graph(job):
    repo, case, config, output, label, fast_state, world_state, slow_state, version, phase, evaluation = job
    output = Path(output)
    torch.set_num_threads(1)
    fresh = bool(config.get('fresh_all', False) or config.get('fresh_fast_world', False))
    fixed_target = bool(config.get('fixed_mission_target', False))
    validation_controller = config.get('validation_slow_controller',
        'geometry' if config.get('fresh_fast_world', False) else 'learned')
    if validation_controller not in ('learned', 'geometry', 'fixed_target'):
        raise ValueError('Unknown validation slow controller')
    geometry_only = bool(config.get('fresh_fast_world', False)
                         or (evaluation and validation_controller == 'geometry'))
    geometry_behavior = geometry_only or (phase in ('warmup', 'fast') and not evaluation)
    slow_controller = ('geometry' if geometry_only or config.get('random_slow_probability', 0.) == 0.
                       else 'geometry_random_mixture') if geometry_behavior else 'learned'
    slow_ppo_eligible = not geometry_behavior and not evaluation
    if fixed_target:
        slow_controller = 'fixed_target'
        slow_ppo_eligible = False
    with (output / 'logs' / (label + '.log')).open('w') as log, contextlib.redirect_stdout(log):
        episode = HierarchicalEpisode(repo, case,
            replace(HierarchyConfig(), option_steps=config['option_steps'],
                    ground_start_hold_seconds=config.get('ground_start_hold_seconds', 0)),
            fast_seed=case['seed'] + 43, rollout_limit=config['steps'] + 1,
            random_init=fresh, v9_fast_reward=fixed_target,
            training_observation_protocol=config.get('training_observation_protocol'))
        if config.get('complete_missions', True):
            episode.rollout_limit = episode.mission_budget + 1
        try:
            fast = make_graph_fast(repo, random_init=fresh)
            load_numpy(fast, fast_state); fast.eval().requires_grad_(False)
            world = GraphWorldModel(config['hidden'], config['horizon'],
                                    road_topology=config.get('road_topology', False))
            load_numpy(world, world_state); world.eval().requires_grad_(False)
            if not fixed_target:
                slow = GraphSlowActorCritic(hidden=config['hidden'])
                load_numpy(slow, slow_state); slow.eval().requires_grad_(False)
            teacher = None if fresh else runtime(repo)[2]
            behavior = (TeacherBehavior(teacher, fast.raw_width)
                        if teacher is not None and phase == 'warmup' and not evaluation
                        and not config.get('freeze_fast_world', False)
                        else fast)
            episode.fast_model = fast
            for controller in episode.env.controllers:
                controller.navigator.model = behavior
                controller.navigator.deterministic = bool(evaluation)
            recorder = GraphRecorder(episode, world, fast, teacher, config)
            generator = torch.Generator().manual_seed(case['seed'] + 71)
            rng = np.random.default_rng(case['seed'] + 97)
            slow_rows = []
            if fixed_target:
                episode.execute_fixed_target()
                if not episode.finished:
                    raise AssertionError('The local task ended before arrival, failure or the time limit')
            while not episode.finished:
                obs = recorder.slow_observation()
                teacher_action = geometry_teacher_action(obs, episode.belief()['position'], episode.goal)
                if geometry_only:
                    # This evaluates/trains fast control under an explicit
                    # rule selector. The unused slow network cannot affect
                    # behavior or provide value/likelihood training targets.
                    action, logp, value = teacher_action, 0., 0.
                else:
                    action, logp, value = slow.act(obs, generator, deterministic=evaluation)
                if geometry_behavior and not geometry_only:
                    valid = np.flatnonzero(obs['mask'])
                    if not evaluation and rng.random() < config.get('random_slow_probability', 0.):
                        action = int(rng.choice(valid))
                    else:
                        action = teacher_action
                    # Teacher behavior is used only for BC, never PPO. Store
                    # the exact learner likelihood for reproducible diagnostics.
                    from .slow_policy import tensors
                    with torch.inference_mode():
                        distribution, _ = slow(*tensors(obs))
                        logp = float(distribution.log_prob(torch.tensor(action)))
                reward, duration, terminal, truncated, _ = episode.execute(obs, action)
                next_value = 0. if terminal or geometry_only else slow.value(recorder.slow_observation())
                slow_rows.append(dict(features=obs['features'], mask=obs['mask'], present=obs['present'],
                        environment_graph=obs['environment_graph'], teacher_action=teacher_action,
                        action=action, logp=logp, value=value, reward=reward, duration=duration,
                        terminated=terminal, truncated=truncated, next_value=next_value,
                        ppo_eligible=slow_ppo_eligible,
                        controller_code=0 if slow_controller == 'geometry' else
                                        2 if slow_controller == 'geometry_random_mixture' else 1))
            collection_models = [(fast, fast_state), (world, world_state)]
            if not fixed_target:
                collection_models.append((slow, slow_state))
            for model, state in collection_models:
                if state_hash(numpy_state(model)) != state_hash(state):
                    raise AssertionError('A policy/world version changed during graph rollout')
            # Keep the later mission states and successful terminal transition
            # in fast PPO too. Explicit short diagnostics retain the extra
            # bootstrap step; complete missions end at a real terminal.
            ppo_steps = len(recorder.transitions) if config.get('complete_missions', True) else config['steps']
            ppo = recorder.ppo_dataset(ppo_steps)
            ppo_path = output / 'data' / (label + '-ppo.npz'); np.savez_compressed(ppo_path, **ppo)
            wm = recorder.world_dataset()
            wm_path = output / 'data' / (label + '-wm.npz'); np.savez_compressed(wm_path, **wm)
            slow_path = None
            if slow_rows:
                slow_path = output / 'data' / (label + '-slow.npz')
                np.savez_compressed(slow_path, **{key: np.asarray([r[key] for r in slow_rows])
                    for key in slow_rows[0] if key != 'environment_graph'})
                np.savez_compressed(slow_path.with_suffix('.graphs.npz'), **{
                    f'g{i}_{key}': np.asarray(value)
                    for i, row in enumerate(slow_rows)
                    for key, value in row['environment_graph'].items()})
            summary = episode.summary()
            cbf_path = output / 'episodes' / (label + '-cbf.json')
            write_json(cbf_path, dict(decisions=recorder.cbf_records))
            result = dict(label=label, version=version, phase=phase, evaluation=evaluation,
                case_id=case.get('case_id'), circles_sha256=case.get('circles_sha256'),
                ground_start_hold_seconds=summary['ground_start_hold_seconds'],
                startup_hold_control_attempts=summary['startup_hold_control_attempts'],
                startup_hold_max_displacement_m=summary['startup_hold_max_displacement_m'],
                first_ground_motion_time_s=summary['first_ground_motion_time_s'],
                training_mode=config.get('training_mode', 'staged'),
                slow_controller=slow_controller, slow_ppo_eligible=slow_ppo_eligible,
                fast_controller='learned' if behavior is fast else 'archived_teacher',
                fast_deterministic=bool(evaluation),
                complete_missions=bool(config.get('complete_missions', True)),
                ppo_path=str(ppo_path), wm_path=str(wm_path), slow_path=None if slow_path is None else str(slow_path),
                cbf_context_path=str(cbf_path), transitions=len(ppo['rewards']), wm_samples=len(wm['history']),
                executed_steps=summary['steps'],
                slow_decisions=len(slow_rows), world_calls=recorder.world_calls,
                fast_sha256=state_hash(fast_state), world_sha256=state_hash(world_state), slow_sha256=state_hash(slow_state),
                termination=summary['termination'], goal_reached=summary['goal_reached'], remaining_m=summary['remaining_m'],
                fast_reward=float(ppo['rewards'].sum()), slow_reward=summary['reward'], collisions=summary['collisions'],
                **navigation_outcome(case, summary))
            if fixed_target:
                result['fast_reward_contract'] = V9_FAST_REWARD_CONTRACT
            write_json(output / 'episodes' / (label + '.json'),
                       dict(case=case, rollout=summary, transport=recorder.network.stats, collection=result))
            return result
        finally:
            episode.close()
