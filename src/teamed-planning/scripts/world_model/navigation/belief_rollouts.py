"""Collect supervised world-model examples and joint on-policy transitions from received evidence."""
from __future__ import annotations

import contextlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .belief_actor import ActorPrior
from .frozen_navigation import write_json
from .graph_fast_policy import graph_features, load_numpy, make_fast, numpy_state, ppo_batch, state_hash
from .hierarchical_runtime import HierarchicalEpisode, HierarchyConfig
from .residual_world import ResidualWorldModel, infer_graph
from .safety_context import CBF_SCHEMA, GRAPH_SCHEMA, observation_graph, received_cbf_context
from .telemetry import TelemetryNetwork, packet, context_features


class RolloutRecorder:
    def __init__(self, episode, seed, horizon=4, stride=4, loss=.1, jitter=3):
        self.episode = episode
        self.env = episode.env
        self.names = {self.env.ground: 'ground_0',
                      **{rid: c.drone_id for rid, c in zip(self.env.ids, self.env.controllers)}}
        self.controllers = dict(zip(self.env.ids, self.env.controllers))
        self.network = TelemetryNetwork(self.names.values(), seed, loss=loss, jitter=jitter)
        self.prior = ActorPrior(episode.fast_model, episode.case['world_size'], horizon)
        self.horizon, self.stride = horizon, stride
        self.inputs = []
        # These dictionaries never enter ActorPrior or the feature encoder.
        self.truth_states, self.truth_actions = {}, {}
        self.base_step = self.env.step
        self.env.step = self.step

    def own_packet(self, rid, frame):
        state = self.env.obs[rid]['self']
        ground = rid == self.env.ground
        if ground:
            goal = self.episode.manager.target
            stamp, task_ground, nominal = frame, state['position'], state['velocity']
        else:
            navigator = self.controllers[rid].navigator
            task = navigator.chain.task()
            # This source reports its own cached acknowledgement. Other robots
            # can use it only after transport; sending a task is not an ACK.
            goal = None if not task else task['target']
            stamp = None if not task else task['source_time']
            task_ground = None if not task else task['position']
            nominal = navigator.nominal_velocity
        return packet(self.names[rid], frame, state['position'], state['velocity'], nominal,
                      goal=goal, goal_stamp=stamp, task_ground=task_ground, is_ground=ground)

    def record_truth(self, frame):
        self.truth_states[frame] = {
            self.names[rid]: np.r_[self.env.obs[rid]['self']['position'][:2],
                                  self.env.obs[rid]['self']['velocity'][:2]].astype(np.float32)
            for rid in self.env.ids}

    def advance_network(self, frame):
        own_packets = {self.names[rid]: self.own_packet(rid, frame) for rid in self.names}
        links = {self.names[rid]: [self.names[n['id']] for n in self.env.obs[rid]['neighbors']
                                   if n['id'] in self.names] for rid in self.names}
        self.network.advance(own_packets, links, frame)
        self.record_truth(frame)
        return own_packets

    def peer_inputs(self, rid, own_packet, frame, *, sorted_targets=False):
        """Yield the actor prior and source packet for each received UAV."""
        owner = self.names[rid]
        receiver = self.network.receivers[owner]
        rows = receiver.rows(frame)
        local = self.env.obs[rid]
        sensor = dict(cells=local['occupancy_map'], resolution=local['map_resolution'],
                      center=local['self']['position'])
        graph = (self.env.ground_controller.unified_graph if rid == self.env.ground
                 else self.controllers[rid].current_graph)
        for target in sorted(rows) if sorted_targets else rows:
            if rows[target]['is_ground'] or target == owner:
                continue
            history, mask, origin, last = receiver.encode(target, frame)
            states, actions, known, std = self.prior.predict(own_packet, last, rows, frame, sensor, graph)
            context = context_features(own_packet, last, rows, frame, known,
                                       0 if graph is None else len(graph.graph.nodes), std)
            entry = dict(history=history, history_mask=mask, context=context,
                baseline_states=states, baseline_actions=actions, origin=origin,
                frame=frame, source_stamp=last['stamp'], observer_ground=rid == self.env.ground,
                observer=owner, target=target)
            yield entry, last

    def before(self, frame):
        own_packets = self.advance_network(frame)
        if frame % self.stride:
            return
        for rid, owner in self.names.items():
            for entry, _ in self.peer_inputs(rid, own_packets[owner], frame):
                self.inputs.append(entry)

    def step(self, frame):
        if abs(self.env.env.time - frame) > 1e-6:
            raise ValueError('World model requires the declared dt=1s rollout')
        self.before(frame)
        # If CBF rejects this step, do not manufacture a future transition.
        result = self.base_step(frame)
        self.record_truth(frame + 1)
        self.truth_actions[frame] = {self.names[rid]: np.asarray(action[:2], np.float32)
            for rid, action in zip(self.env.ids, self.env.last_executed_velocities)}
        return result

    def dataset(self):
        samples = []
        for entry in self.inputs:
            row = dict(entry); frame, target = row['frame'], row.pop('target')
            row['target_id'] = int(target.split('_')[-1])
            observer = row.pop('observer')
            row['observer_id'] = -1 if observer == 'ground_0' else int(observer.split('_')[-1])
            origin = row['origin']
            states = np.zeros((self.horizon + 1, 4), np.float32)
            actions = np.zeros((self.horizon, 2), np.float32)
            state_mask = np.zeros(self.horizon + 1, bool)
            action_mask = np.zeros(self.horizon, bool)
            for k in range(self.horizon + 1):
                if frame + k in self.truth_states:
                    truth = self.truth_states[frame + k][target]
                    states[k] = np.r_[(truth[:2] - origin) / 20., truth[2:] / 6.]
                    state_mask[k] = True
                if k < self.horizon and frame + k in self.truth_actions:
                    actions[k] = self.truth_actions[frame + k][target] / 6.
                    action_mask[k] = True
            row.update(target_states=states, target_actions=actions,
                       state_mask=state_mask, action_mask=action_mask)
            samples.append(row)
        if not samples:
            raise ValueError('No delivered telemetry yielded samples; never substitute truth')
        return {key: np.asarray([s[key] for s in samples]) for key in samples[0]}


def _run_collection_policy(episode, seed, random_probability):
    """Run the slow behavior policy shared by supervised and joint collection."""
    rng = np.random.default_rng(seed)
    while not episode.finished:
        observation = episode.observation()
        valid = np.flatnonzero(observation['mask'])
        driving = [int(i) for i in valid if observation['kinds'][i] != 'hold']
        if rng.random() < random_probability:
            action = int(rng.choice(valid))
        else:
            position = episode.belief()['position']
            targets = observation['targets']

            def route_cost(i):
                remaining = np.linalg.norm(targets[i] - episode.goal)
                total = np.linalg.norm(targets[i] - position) + remaining
                return round(float(total), 3), float(remaining)

            action = min(driving or valid, key=route_cost)
        episode.execute(observation, action)


def collect_scene(job):
    repo, case, config, output, label = job
    output = Path(output)
    with (output / 'logs' / (label + '.log')).open('w') as log, contextlib.redirect_stdout(log):
        episode = HierarchicalEpisode(repo, case,
            replace(HierarchyConfig(), option_steps=config['option_steps']),
            fast_seed=case['seed'] + 43, rollout_limit=config['steps'])
        before = {k: v.clone() for k, v in episode.fast_model.state_dict().items()}
        recorder = RolloutRecorder(episode, case['seed'] + 19,
            config['horizon'], config['stride'], config['loss'], config['jitter'])
        try:
            _run_collection_policy(episode, case['seed'] + 97, .25)
            arrays = recorder.dataset()
            for key, value in episode.fast_model.state_dict().items():
                if not episode.torch.equal(value, before[key]):
                    raise AssertionError('Frozen shared UAV actor changed during data collection')
            path = output / 'data' / (label + '.npz')
            np.savez_compressed(path, **arrays)
            summary = episode.summary()
            write_json(output / 'episodes' / (label + '.json'),
                       dict(case=case, rollout=summary, transport=recorder.network.stats,
                            samples=len(arrays['history']), fast_actor_unchanged=True))
            return dict(path=str(path), label=label, samples=len(arrays['history']),
                        steps=summary['steps'], termination=summary['termination'],
                        ground_samples=int(arrays['observer_ground'].sum()),
                        max_age=float(np.max(arrays['frame'] - arrays['source_stamp'])))
        finally:
            episode.close()


class JointRecorder(RolloutRecorder):
    def __init__(self, episode, world, seed, **kwargs):
        super().__init__(episode, seed, **kwargs)
        self.world = world
        self.transitions = []
        self.prepared_peers = {name: [] for name in self.names.values()}
        self.cbf_contexts = {}
        self.cbf_records = []
        self.decision_frame = None
        self.predicted_agent_steps = 0
        self.predicted_peers = 0
        for controller in self.env.controllers:
            nav = controller.navigator
            original = nav.encode
            name = controller.drone_id
            # Invoked by the original encoder after get_action updates local
            # state and after packet loss. No extra pre-reception CBF probe.
            def current_margin(observation, controller=controller, name=name):
                dt = observation.get('time_step')
                if dt is None or not np.isfinite(dt) or dt <= 0:
                    raise ValueError('CBF margin requires a positive time_step')
                stats = controller._cbf_filter((0., 0., 0.), observation, dt)[1]
                self.cbf_contexts[name] = received_cbf_context(controller, observation, stats)
                return stats['feasibility_margin']
            controller._current_cbf_margin = current_margin
            def encode(position, observation, original=original, name=name):
                features, mask = original(position, observation)
                graph = observation_graph(name, self.decision_frame, position, features,
                                          self.cbf_contexts[name], self.prepared_peers[name])
                belief = infer_graph(self.world, graph)
                actor_features = graph_features(belief)
                peers = len(belief['nodes']) - 1
                self.predicted_agent_steps += int(peers > 0)
                self.predicted_peers += peers
                self.cbf_records.append(belief['nodes'][name]['cbf'])
                return actor_features, mask
            nav.encode = encode

    def before(self, frame):
        own = self.advance_network(frame)
        self.decision_frame = frame
        self.cbf_contexts.clear()
        for rid, owner in self.names.items():
            self.prepared_peers[owner] = []
            if rid == self.env.ground and frame % self.stride:
                continue
            for entry, last in self.peer_inputs(rid, own[owner], frame, sorted_targets=True):
                if rid != self.env.ground:
                    self.prepared_peers[owner].append(dict(entry=entry, packet=last))
                if frame % self.stride == 0:
                    self.inputs.append(entry)

    def append_transition(self, samples, state, reward, terminal):
        if len(samples) != 3:
            raise ValueError('All three shared-actor proposals must be recorded, including a rejected CBF step')
        features = np.asarray([s[0] for s in samples], np.float32)
        with torch.inference_mode():
            value = float(self.episode.fast_model.value(torch.from_numpy(features)[None],
                                                       state=torch.from_numpy(state)[None])[0])
        self.transitions.append(dict(features=features,
            actions=np.asarray([s[2] for s in samples], np.float32),
            logp=np.asarray([s[3] for s in samples], np.float32), states=state,
            rewards=np.float32(reward), values=np.float32(value), terminals=bool(terminal)))

    def step(self, frame):
        if abs(self.env.env.time - frame) > 1e-6:
            raise ValueError('Joint model requires dt=1 second')
        self.before(frame)
        state = self.env.critic_state(frame, self.episode.mission_budget)
        try:
            result = self.base_step(frame)
        except self.episode.me.SafetyTermination as error:
            self.append_transition(error.samples, state, -100., True)
            raise
        samples, reward, _ = result
        self.record_truth(frame + 1)
        self.truth_actions[frame] = {self.names[rid]: np.asarray(action[:2], np.float32)
            for rid, action in zip(self.env.ids, self.env.last_executed_velocities)}
        success = bool(self.env.goal_reached)
        deadline = self.episode.attempts + 1 >= self.episode.mission_budget and not success
        self.append_transition(samples, state, reward + (100. if success else -20. if deadline else 0.),
                               success or deadline)
        return result

    def ppo_dataset(self, steps):
        return ppo_batch(self.transitions, steps)


def collect_joint(job):
    repo, case, config, output, label, fast_state, world_state, version = job
    output = Path(output)
    with (output / 'logs' / (label + '.log')).open('w') as log, contextlib.redirect_stdout(log):
        episode = HierarchicalEpisode(repo, case,
            replace(HierarchyConfig(), option_steps=config['option_steps']),
            fast_seed=case['seed'] + 43, rollout_limit=config['steps'] + 1)
        try:
            fast = make_fast(repo); load_numpy(fast, fast_state); fast.eval().requires_grad_(False)
            world = ResidualWorldModel(config['hidden'], config['horizon'])
            load_numpy(world, world_state); world.eval().requires_grad_(False)
            episode.fast_model = fast
            for controller in episode.env.controllers: controller.navigator.model = fast
            recorder = JointRecorder(episode, world, case['seed'] + 19, horizon=config['horizon'],
                                     stride=config['stride'], loss=config['loss'], jitter=config['jitter'])
            _run_collection_policy(episode, case['seed'] + 97, config.get('random_slow_probability', .25))
            if state_hash(numpy_state(fast)) != state_hash(fast_state):
                raise AssertionError('Behavior actor changed during collection')
            if state_hash(numpy_state(world)) != state_hash(world_state):
                raise AssertionError('World model changed during collection')
            ppo = recorder.ppo_dataset(config['steps'])
            ppo_path = output / 'data' / (label + '-ppo.npz')
            np.savez_compressed(ppo_path, **ppo)
            wm_path = None; samples = 0
            if recorder.inputs:
                wm = recorder.dataset(); samples = len(wm['history'])
                wm_path = output / 'data' / (label + '-wm.npz')
                np.savez_compressed(wm_path, **wm)
            summary = episode.summary()
            summary.update(frozen_fast_policy=False, behavior_frozen_during_rollout=True)
            cbf_path = output / 'episodes' / (label + '-cbf.json')
            write_json(cbf_path, dict(graph_schema=GRAPH_SCHEMA, cbf_schema=CBF_SCHEMA,
                                     decisions=recorder.cbf_records))
            result = dict(label=label, ppo_path=str(ppo_path), wm_path=str(wm_path) if wm_path else None,
                cbf_context_path=str(cbf_path), cbf_decisions=len(recorder.cbf_records),
                wm_samples=samples, transitions=len(ppo['rewards']), version=version,
                fast_sha256=state_hash(fast_state), world_sha256=state_hash(world_state),
                predicted_agent_steps=recorder.predicted_agent_steps,
                predicted_peers=recorder.predicted_peers, termination=summary['termination'],
                remaining_m=summary['remaining_m'], goal_reached=summary['goal_reached'],
                fast_reward=float(ppo['rewards'].sum()))
            write_json(output / 'episodes' / (label + '.json'),
                       dict(case=case, rollout=summary, transport=recorder.network.stats, collection=result))
            return result
        finally:
            episode.close()
