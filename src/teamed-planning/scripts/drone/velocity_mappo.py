"""V3 continuous velocity policy for fixed-height 2D simulation.

No candidate generation, frontier search or waypoint selection. The sampled
action stored for PPO is a two-dimensional Gaussian latent. A fixed radial
transform maps it into the speed disk; the existing CBF follows that transform.
Log probabilities and entropy refer to the latent policy, never CBF outputs.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal
from drone.chain_state import ChainState
from drone.drone_navigator import DroneNavigator
from drone.mappo import GLOBAL_FEATURES, legal_observation

SENSOR_SIZE = 12
GRAPH_SIZE = 16
GRAPH_RADIUS = 96.
# Actor input versions. 1 (v3-v6): 32 state values and a sensor VALIDITY plane,
# which is exactly 1 on every simulated step; with Adam those 144 identical
# inputs act as a ~145x bias learning rate and drove a state-independent
# world-frame drift. 2 (v6.2): an INVALIDITY plane (0 when all pixels are
# valid), altitude centred on the 15 m flight height, plus the drone's own
# current CBF feasibility margin and its chain bend. Old indices are unchanged.
STATE_FEATURES = {1: 32, 2: 34}
INPUT_VERSIONS = tuple(STATE_FEATURES)
FLIGHT_HEIGHT = 15.


def velocity_features(input_version=1):
    return STATE_FEATURES[input_version] + 2*SENSOR_SIZE**2 + 2*GRAPH_SIZE**2


def input_version_for_width(width, prefix=0):
    """Actor input version from a first-layer width (minus any fixed suffix)."""
    for version in INPUT_VERSIONS:
        if width-prefix == velocity_features(version):
            return version
    raise ValueError(f'Unknown velocity actor input width {width}')


VELOCITY_FEATURES = velocity_features(1)


def _identity_value_statistics(module, state, prefix, *args):
    # Checkpoints from before value normalization load as identity statistics.
    state.setdefault(prefix+'value_mean', torch.zeros(()))
    state.setdefault(prefix+'value_std', torch.ones(()))
    state.setdefault(prefix+'value_moments', torch.zeros(3))


class VelocityActorCritic(nn.Module):
    action_mode = 'velocity'

    def __init__(self, global_features=GLOBAL_FEATURES, input_version=1):
        super().__init__()
        self.global_features = int(global_features)
        self.input_version = int(input_version)
        width = velocity_features(self.input_version)
        self.velocity_actor = nn.Sequential(nn.Linear(width, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2))
        self.log_std = nn.Parameter(torch.full((2,), -.5))
        self.critic = nn.Sequential(nn.Linear(width+self.global_features, 128),
            nn.Tanh(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.orthogonal_(self.velocity_actor[-1].weight, gain=.01)
        nn.init.zeros_(self.velocity_actor[-1].bias)
        # PopArt: value = critic(s)*std + mean. Identity unless the trainer
        # enables value normalization; moments are [weight, E[R], E[R虏]].
        self.register_buffer('value_mean', torch.zeros(()))
        self.register_buffer('value_std', torch.ones(()))
        self.register_buffer('value_moments', torch.zeros(3))
        self.register_load_state_dict_pre_hook(_identity_value_statistics)

    @classmethod
    def from_state_dict(cls, state):
        version = input_version_for_width(state['velocity_actor.0.weight'].shape[1])
        model = cls(state['critic.0.weight'].shape[1]-velocity_features(version), version)
        model.load_state_dict(state)
        return model

    def distribution(self, x, mask=None):
        return Independent(Normal(self.velocity_actor(x), self.log_std.clamp(-5., 1.).exp()), 1)

    def critic_input(self, x, mask=None, state=None):
        pooled = x.mean(-2)
        if self.global_features:
            if state is None:
                raise ValueError('Velocity critic requires the centralized training state')
            pooled = torch.cat((pooled, state), dim=-1)
        return pooled

    def value(self, x, mask=None, state=None):
        return self.critic(self.critic_input(x, mask, state)).squeeze(-1)*self.value_std+self.value_mean

    @staticmethod
    def velocity(action, max_speed=6.):
        # Smooth bijection R虏 -> open speed disk; zero also represents hold.
        a = np.asarray(action, dtype=np.float64)
        return max_speed*a/np.sqrt(1.+np.sum(a*a, axis=-1, keepdims=True))


class VelocityNavigator(DroneNavigator):
    uses_velocity = True

    def __init__(self, *args, model, deterministic=False, max_speed=6.,
                 world_bounds=(0., 0., 1000., 1000.), **kwargs):
        super().__init__(*args, **kwargs)
        self.model, self.deterministic = model, deterministic
        self.max_speed, self.world_bounds = max_speed, world_bounds
        self.chain = self.make_chain_state()
        self.last_sample = None
        self.fallbacks = 0
        self.nominal_velocity = np.zeros(3)
        self.previous_position = None
        self.previous_time = None
        # Input version 2: the controller sets a callable returning its own
        # CBF feasibility margin (m/s) for the CURRENT received geometry.
        self.cbf_margin_source = None

    def make_chain_state(self):
        return ChainState(self.drone_id, self._fleet_total)

    def encode(self, position, observation):
        obs = legal_observation(observation)
        if hasattr(self, 'cooperation_features'):
            obs['cooperation'] = observation['cooperation']
            if 'relay' in observation:
                # v8: decision-time snapshot of this device's own relay cache.
                obs['relay'] = observation['relay']
        self.chain.update(obs)
        p = np.asarray(position, float)
        version = getattr(self.model, 'input_version', 1)
        state = np.zeros(STATE_FEATURES[version], np.float32)
        xmin, ymin, xmax, ymax = self.world_bounds
        state[:4] = [(p[0]-xmin)/1000., (xmax-p[0])/1000.,
                     (p[1]-ymin)/1000., (ymax-p[1])/1000.]
        task = self.chain.task()
        if task is not None:
            ground = np.asarray(task['position'][:2])
            goal = np.asarray(task['target'][:2])
            state[4:6], state[6:8] = (goal-p[:2])/350., (ground-p[:2])/95.
            state[8:10] = (goal-ground)/350.
            state[10] = 1.
            state[11] = np.linalg.norm(p-np.r_[ground, 0.])/95.
        # Role features and two chain endpoints, never a fleet-sized one-hot
        # or an ID-dependent policy. The actor shape is identical for any N.
        state[12] = float(self.chain.predecessor == '__ground__')
        state[13] = float(self.chain.successor is None)
        state[14] = self.chain.rank/max(len(self.chain.fleet_ids)-1, 1)
        for i, did in enumerate((self.chain.predecessor, self.chain.successor)):
            endpoint = self.chain.visible.get(did)
            if endpoint is not None:
                state[15+3*i:17+3*i] = (endpoint[:2]-p[:2])/95.
                state[17+3*i] = 1.
        if self.previous_position is not None and obs['time'] > self.previous_time:
            state[21:23] = (p[:2]-self.previous_position[:2])/(obs['time']-self.previous_time)/self.max_speed
        state[23:25] = self.nominal_velocity[:2]/self.max_speed
        state[25] = min(obs['time']/4000., 1.)
        state[26] = len(self.chain.missing(p, self.comm_range))/2.
        state[27] = p[2]/15. if version == 1 else (p[2]-FLIGHT_HEIGHT)/FLIGHT_HEIGHT
        neighbors = [np.asarray(info['position'][:2])-p[:2]
                     for info in obs['nearby_drones'].values()]
        if neighbors:
            # Permutation-invariant pooling of the actually received set.
            state[28:30] = np.mean(neighbors, axis=0)/95.
            state[30] = min(np.linalg.norm(q) for q in neighbors)/95.
        state[31] = np.log1p(len(neighbors))/np.log(8.)
        if version >= 2:
            if self.cbf_margin_source is None:
                raise ValueError('Actor input version 2 requires the controller CBF margin')
            margin = self.cbf_margin_source()
            # No constraint at all is the safest case; -inf is an unusable link.
            state[32] = 1. if margin is None else float(np.clip(margin/self.max_speed, -1., 1.))
            ends = [self.chain.visible.get(did) for did in (self.chain.predecessor, self.chain.successor)]
            if all(end is not None for end in ends):
                # Cosine of the chain angle at this drone: -1 is a straight chain,
                # where two long links leave the CBF no admissible velocity.
                u, v = (np.asarray(end[:2])-p[:2] for end in ends)
                norm = np.linalg.norm(u)*np.linalg.norm(v)
                state[33] = float(u@v/norm) if norm > 1e-9 else 0.

        if hasattr(self, 'cooperation_features'):
            self.cooperation_features(state, p, obs)

        # Only actual local sensor pixels and the controller's received/fused
        # graph enter the actor. Unknown pixels have a separate validity plane
        # (version 1) or invalidity plane (version 2, zero when all are valid).
        sensor = np.zeros((2, SENSOR_SIZE, SENSOR_SIZE), np.float32)
        if version >= 2:
            sensor[1] = 1.
        raw = observation.get('occupancy_map')
        if raw is not None and observation.get('sensor_valid', True):
            raw = np.asarray(raw)
            if raw.shape != (SENSOR_SIZE, SENSOR_SIZE):
                raise ValueError('Velocity policy requires the 12x12 2D drone sensor')
            valid = np.isfinite(raw) & (raw >= 0)
            sensor[0] = np.where(valid, np.clip(raw, 0., 1.), 0.)
            sensor[1] = valid if version == 1 else ~valid
        spatial = np.zeros((2, GRAPH_SIZE, GRAPH_SIZE), np.float32)
        graph = obs['graph']
        if graph is not None:
            for node, data in graph.graph.nodes(data=True):
                xy = np.asarray(data['position'][:2])-p[:2]
                cell = np.floor((xy+GRAPH_RADIUS)/(2*GRAPH_RADIUS)*GRAPH_SIZE).astype(int)
                if np.all((cell >= 0) & (cell < GRAPH_SIZE)):
                    x, y = cell
                    spatial[0, y, x] += 1.
                    spatial[1, y, x] += graph.node_confidence(node)
            spatial[1] /= np.maximum(spatial[0], 1.)
            spatial[0] = np.minimum(spatial[0]/8., 1.)
        features = np.concatenate((state, sensor.ravel(), spatial.ravel()))
        if hasattr(self.model, 'encode_graph'):
            features = self.model.encode_graph(features, p, obs)
        return features, np.ones(1, dtype=bool)

    def compute_velocity(self, position, observation):
        features, mask = self.encode(position, observation)
        with torch.no_grad():
            dist = self.model.distribution(torch.from_numpy(features))
            action = dist.mean if self.deterministic else dist.sample()
            logp = float(dist.log_prob(action))
        action = action.numpy()
        self.nominal_velocity = np.r_[self.model.velocity(action, self.max_speed), 0.]
        self.last_sample = features, mask, action, logp
        self.previous_position = np.asarray(position, float).copy()
        self.previous_time = float(observation.get('time', 0.))
        self._current_target = None
        return tuple(self.nominal_velocity)

    def reset(self):
        super().reset()
        self.chain = self.make_chain_state()
        self.last_sample = None
        self.fallbacks = 0
        self.nominal_velocity = np.zeros(3)
        self.previous_position = None
        self.previous_time = None
