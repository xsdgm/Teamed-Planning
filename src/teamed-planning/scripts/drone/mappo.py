"""Shared candidate actor and legal-observation centralized critic (CPU capable)."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from drone.drone_navigator import DroneNavigator, NavigationPhase
from drone.chain_state import ChainState

FEATURES = 12
MAX_ACTIONS = 21
# Training-only centralized state (time, delivered map, task graph, chain);
# see Experiment.critic_state. The actor never receives these features.
GLOBAL_FEATURES = 10


def legal_observation(obs):
    """Copy the interface whitelist; never traverse raw broadcast dictionaries.

    Tasks and neighbors are reception-only: no additional mission cache is used.
    An empty path explicitly clears a formerly received path.
    """
    return dict(graph=obs.get('graph'), time=float(obs.get('time', 0)),
                ground_robots={str(k): {f: v[f] for f in
                    ('position', 'target', 'planned_path', 'source_time', 'direct') if f in v}
                    for k, v in obs.get('ground_robots', {}).items()},
                nearby_drones={str(k): {'position': v['position']}
                    for k, v in obs.get('nearby_drones', {}).items()
                    if v.get('position') is not None},
                neighbor_waypoints={str(k): tuple(v) for k, v in
                    obs.get('neighbor_waypoints', {}).items() if v is not None})


class ActorCritic(nn.Module):
    def __init__(self, global_features=0):
        super().__init__()
        self.global_features = int(global_features)
        self.actor = nn.Sequential(nn.Linear(FEATURES, 64), nn.Tanh(),
                                   nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        # Mean pooling is invariant to fleet order/size. The optional global
        # state is used only for training targets, never for acting.
        self.critic = nn.Sequential(nn.Linear(FEATURES+self.global_features, 64), nn.Tanh(),
                                    nn.Linear(64, 1))

    @classmethod
    def from_state_dict(cls, state):
        """Rebuild with the critic width stored in the weights (older checkpoints: 0)."""
        model = cls(global_features=state['critic.0.weight'].shape[1]-FEATURES)
        model.load_state_dict(state)
        return model

    def distribution(self, x, mask):
        return Categorical(logits=self.actor(x).squeeze(-1).masked_fill(~mask, -1e9))

    def value(self, x, mask, state=None):
        pooled = (x * mask[..., None]).sum(-2) / mask.sum(-1, keepdim=True).clamp_min(1)
        pooled = pooled.mean(-2)
        if self.global_features:
            if state is None:
                raise ValueError('This critic requires the global training state')
            pooled = torch.cat([pooled, state], -1)
        return self.critic(pooled).squeeze(-1)


class CandidateNavigator(DroneNavigator):
    """Controlled baselines and MAPPO share candidate generation and screening.

    First experiment uses direct upload, not an inferred relay chain. Swept
    separation uses received previous proposals and a bounded one-second motion.
    Missing expected neighbors forces hold, rather than assuming connectivity.
    """
    def __init__(self, *args, model=None, algorithm='mappo', checkpoint=None,
                 deterministic=None, max_speed=6., decision_dt=1.,
                 exploration_candidates=0, world_bounds=None,
                 communication_mode=None, fleet_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration_candidates = int(exploration_candidates)
        self.world_bounds = world_bounds
        self.communication_mode=communication_mode or 'direct'
        self.checkpoint_safety_filter='legacy'
        self.model = model
        policy_mode = 'argmax'
        if checkpoint:
            saved = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
            if saved.get('feature_version') != 1:
                raise ValueError('Unsupported MAPPO checkpoint feature version')
            self.model = ActorCritic.from_state_dict(saved['model'])
            # Act as the checkpoint was validated; older checkpoints used argmax.
            policy_mode = saved.get('policy_mode', 'argmax')
            self.fov_radius = float(saved.get('fov_radius', 12.))
            self.exploration_candidates = int(saved.get('exploration_candidates', 0))
            self.world_bounds = saved.get('world_bounds', world_bounds)
            self.communication_mode=communication_mode or saved.get('communication_mode','direct')
            self.checkpoint_safety_filter=saved.get('safety_filter','legacy')
            self.model.eval()
        if not 0 <= self.exploration_candidates <= 20:
            raise ValueError('exploration_candidates must be between 0 and 20')
        if policy_mode not in ('argmax', 'sample'):
            raise ValueError('policy_mode must be argmax or sample')
        self.algorithm = algorithm
        self.deterministic = policy_mode == 'argmax' if deterministic is None else bool(deterministic)
        self.max_speed, self.decision_dt = max_speed, decision_dt
        self.last_sample = None
        self.fallbacks = 0
        if self.communication_mode not in ('direct','chain'):
            raise ValueError('communication_mode must be direct or chain')
        self.chain=ChainState(self.drone_id,self._fleet_total if self.communication_mode=='chain' else 1,
                              fleet_ids=fleet_ids)
        self.recoveries=0
        self._recovery_target=None

    def encode(self, position, observation):
        obs = legal_observation(observation)
        p = np.asarray(position, dtype=float)
        self.chain.update(obs)
        self._recovery_target=None
        tasks = [v for v in obs['ground_robots'].values()
                 if v.get('position') is not None and v.get('target') is not None]
        if self.communication_mode=='chain':
            task=self.chain.task()
            tasks=[task] if task is not None else []
            if self.chain.missing(p,self.comm_range):
                self._recovery_target=self.chain.recovery(p,self.comm_range,self.min_separation,self.max_speed)
        candidates = [tuple(p)]
        mission = bool(tasks)
        graph = obs['graph']
        if mission:
            task = tasks[0]
            start, goal = np.asarray(task['position'][:2]), np.asarray(task['target'][:2])
            self.ground_robot_position = tuple(start)
            self.ground_robot_target = tuple(goal)
            self._planned_path = [tuple(w[:2]) for w in task.get('planned_path', [])]
            if self.exploration_candidates < 20:
                candidates += self._generate_candidates(graph, self._planned_path, tuple(goal),
                        tuple(p), p[2], max_candidates=20-self.exploration_candidates)
            # Local configuration and received position only; no unknown map
            # utility is inspected. Same optional set for all adapted baselines.
            if self.exploration_candidates:
                phase = sum(map(ord, self.drone_id)) * .37 + int(obs['time']//10) * .31
                for angle in np.linspace(0., 2*np.pi, self.exploration_candidates, endpoint=False)+phase:
                    q = p[:2] + 1.5*self.fov_radius*np.array([np.cos(angle),np.sin(angle)])
                    candidates.append((float(q[0]),float(q[1]),float(p[2])))
        else:
            start = goal = p[:2]
        x = np.zeros((MAX_ACTIONS, FEATURES), np.float32)
        mask = np.zeros(MAX_ACTIONS, bool)
        neighbors = obs['nearby_drones']
        nodes = [] if graph is None else list(graph.graph.nodes(data=True))
        node_positions = np.asarray([d['position'][:2] for _,d in nodes]).reshape(-1,2)
        node_confidences = np.asarray([graph.node_confidence(nid) for nid,_ in nodes])
        for a, w in enumerate(candidates):
            q = np.asarray(w)
            delta = q - p
            distance = np.linalg.norm(delta)
            end = p + delta * min(1., self.max_speed * self.decision_dt / max(distance, 1e-6))
            safe = mission and (not self.chain.missing(p,self.comm_range) if self.communication_mode=='chain'
                                else len(neighbors) >= self._fleet_total-1)
            if self.world_bounds is not None:
                xmin,ymin,xmax,ymax = self.world_bounds
                safe &= xmin <= q[0] <= xmax and ymin <= q[1] <= ymax
            # A convex ball contains the entire local linear motion segment.
            if self.communication_mode=='direct' or self.chain.rank==0:
                safe &= max(np.linalg.norm(p-np.r_[start,0.]),np.linalg.norm(end-np.r_[start,0.]))<=self.comm_range
            closest = self.comm_range
            for did, info in neighbors.items():
                other = np.asarray(info['position'], float)
                wp = np.asarray(obs['neighbor_waypoints'].get(did, other), float)
                movement = wp - other
                other_end = other + movement * min(1., self.max_speed * self.decision_dt /
                                                    max(np.linalg.norm(movement), 1e-6))
                r, v = p - other, (end - p) - (other_end - other)
                t = np.clip(-np.dot(r, v) / max(np.dot(v, v), 1e-8), 0., 1.)
                sep = np.linalg.norm(r + t * v)
                closest = min(closest, sep)
                safe &= sep >= self.min_separation
                if self.communication_mode=='chain' and did in self.chain.required:
                    safe &= max(np.linalg.norm(r),np.linalg.norm(end-other_end))<=self.comm_range
            # Hold always remains selectable as the documented fallback.
            mask[a] = bool(safe or a == 0)
            local = np.linalg.norm(node_positions-q[:2],axis=1) < self.fov_radius
            local_count = int(local.sum())
            confidence = float(node_confidences[local].mean()) if local_count else 0.
            direction = goal - start
            t = np.clip(np.dot(q[:2] - start, direction) / max(np.dot(direction, direction), 1e-8), 0, 1)
            corridor = np.linalg.norm(q[:2] - start - t * direction)
            x[a] = [delta[0]/100, delta[1]/100, distance/100,
                    np.linalg.norm(q[:2]-start)/100, np.linalg.norm(q[:2]-goal)/100,
                    corridor/100, confidence, min(local_count/20, 1),
                    closest/100, float(a == 0), float(mission), p[2]/100]
        return x, mask, candidates, obs

    def compute_next_target(self, current_position, observation_data):
        x, mask, candidates, obs = self.encode(current_position, observation_data)
        if mask.sum() == 1:
            self.fallbacks += 1
        xt, mt = torch.from_numpy(x), torch.from_numpy(mask)
        with torch.no_grad():
            if self.algorithm == 'mappo' and self.model is not None:
                try:
                    dist = self.model.distribution(xt, mt)
                    a = dist.logits.argmax() if self.deterministic else dist.sample()
                    index, logp = int(a), float(dist.log_prob(a))
                except (RuntimeError, ValueError):
                    index, logp = 0, 0.
                    self.fallbacks += 1
            elif self.algorithm == 'cadmm_controlled':
                # ADMM iterations use the identical received neighbor proposals.
                valid = [w for i, w in enumerate(candidates) if mask[i]]
                target = self._admm_select(valid, obs['graph'], self.ground_robot_position,
                    self.ground_robot_target, obs['neighbor_waypoints'],
                    {k: tuple(v['position']) for k,v in obs['nearby_drones'].items()}, current_position,
                    fixed_candidates=True
                    ) if mask.sum() > 1 and obs['graph'] is not None else candidates[0]
                index = min((i for i in range(len(candidates)) if mask[i]),
                            key=lambda i: np.linalg.norm(np.asarray(candidates[i])-target))
                logp = 0.
            elif self.algorithm == 'greedy':
                scores = (1-x[:,6]) - .2*x[:,2] - .1*x[:,5] - .2*x[:,9]
                index, logp = int(np.where(mask, scores, -1e9).argmax()), 0.
            else:
                index, logp = 0, 0.
                self.fallbacks += 1
        self.last_sample = (x, mask, index, logp)
        self._current_target = tuple(candidates[index])
        if self.communication_mode=='chain' and self._recovery_target is not None:
            self._current_target=self._recovery_target
            self.recoveries+=1
        self.phase = NavigationPhase.DISPATCHING if x[0,10] else NavigationPhase.DISCOVERING
        return self._current_target

    def reset(self):
        super().reset()
        self.fallbacks = 0
        self.last_sample = None
        if hasattr(self,'chain'):
            self.chain=ChainState(self.drone_id,self._fleet_total,fleet_ids=self.chain.fleet_ids)
            self.recoveries=0

    def get_stats(self):
        return dict(super().get_stats(),algorithm=self.algorithm,fallbacks=self.fallbacks,
                    communication_mode=self.communication_mode,recoveries=self.recoveries)
