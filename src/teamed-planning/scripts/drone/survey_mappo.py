"""V10 survey-corridor and V11 fixed a-b route policies with a real vehicle anchor.

The v9 map, teammate and relay encoders retain their parameter names. Eight
additional task features describe a--b; the original prefix still describes
the real vehicle position c. Survey commands are read only from missions
actually received by this drone, with the same freshness limit as ChainState.
"""
import numpy as np
import torch
from torch import nn

from drone.cooperative_mappo import CooperativeCriticEncoder
from drone.relay_attention_mappo import (
    RelayActorCritic, RelayEncoder, MODES,
    RAW_WIDTH as RELAY_RAW_WIDTH, ENCODED_WIDTH as RELAY_ENCODED_WIDTH,
)


SURVEY_FEATURES = 8
RAW_WIDTH = RELAY_RAW_WIDTH + SURVEY_FEATURES
ENCODED_WIDTH = RELAY_ENCODED_WIDTH + SURVEY_FEATURES
TASK_TIMEOUT_S = 10.


def survey_features(position, observation):
    """Encode a fresh received command; never infer a from the vehicle c.

    Layout: relative a (2), relative b (2), b-a (2), half-width, valid.
    No command, expired command or future-dated command yields all zeros.
    Malformed commands raise rather than silently changing the requested area.
    """
    features = np.zeros(SURVEY_FEATURES, np.float32)
    now = float(observation.get('time', 0.))
    candidates = []
    for task in observation.get('ground_robots', {}).values():
        if task.get('survey') is None:
            continue
        stamp = float(task.get('source_time', now))
        if np.isfinite(stamp) and -.01 <= now-stamp <= TASK_TIMEOUT_S:
            candidates.append((stamp, bool(task.get('direct', True)), task['survey']))
    if not candidates:
        return features
    survey = max(candidates, key=lambda row: row[:2])[2]
    if not isinstance(survey, dict):
        raise ValueError('survey must contain a, b and half_width')
    try:
        a, b = np.asarray(survey['a'], float), np.asarray(survey['b'], float)
        width = float(survey['half_width'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('survey must contain finite a, b and positive half_width') from exc
    if (a.shape != (2,) or b.shape != (2,) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not np.isfinite(width) or width <= 0.
            or isinstance(survey['half_width'], bool)):
        raise ValueError('survey must contain finite 2D a, b and positive half_width')
    p = np.asarray(position, float)[:2]
    features[:] = [*((a-p)/350.), *((b-p)/350.), *((b-a)/350.), width/95., 1.]
    return features


class SurveyEncoder(RelayEncoder):
    def forward(self, x):
        relay = super().forward(x[..., :RELAY_RAW_WIDTH])
        survey = x[..., RELAY_RAW_WIDTH:RAW_WIDTH]
        # A masked command cannot leak coordinates even for padded training data.
        survey = survey * survey[..., -1:]
        return torch.cat((relay, survey), -1)


class SurveyActorCritic(RelayActorCritic):
    policy_architecture = 'gat_survey_v10'

    def __init__(self, global_features=10, input_version=2, relay_attention='full'):
        super().__init__(global_features, input_version, relay_attention)
        # Feature version 10 already belongs to the CoGraph v2 target selector.
        self.feature_version = 11
        self.register_buffer('survey_schema', torch.tensor(1))
        self.gat_actor = nn.Sequential(
            SurveyEncoder(relay_attention), nn.Linear(ENCODED_WIDTH, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2))
        self.critic = nn.Sequential(
            CooperativeCriticEncoder(global_features, SurveyEncoder(relay_attention),
                                     RAW_WIDTH, ENCODED_WIDTH),
            nn.Linear(ENCODED_WIDTH+global_features, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.orthogonal_(self.gat_actor[-1].weight, gain=.01)
        nn.init.zeros_(self.gat_actor[-1].bias)
        with torch.no_grad():
            self.gat_actor[1].weight[:, RELAY_ENCODED_WIDTH:].zero_()

    def encode_graph(self, base, position, observation):
        original = super().encode_graph(base, position, observation)
        return np.r_[original, survey_features(position, observation)].astype(np.float32)

    def warm_start_actor(self, source):
        """Migrate a v9 actor; keep the new objective's critic/PopArt untouched.

        Expanded task columns start at zero, so a nonempty survey command does
        not change the transferred policy's initial action distribution. The
        new columns remain trainable and immediately receive PPO gradients.
        """
        target = self.state_dict()
        actor_keys = [key for key in target if key == 'log_std' or key.startswith('gat_actor.')]
        copied, partial = [], []
        for key in actor_keys:
            if key not in source:
                raise ValueError(f'v9 checkpoint is missing actor parameter {key}')
            incoming = source[key]
            if key == 'gat_actor.1.weight':
                expected = (target[key].shape[0], RELAY_ENCODED_WIDTH)
                if incoming.shape != expected:
                    raise ValueError(f'Expected v9 first-layer shape {expected}, got {tuple(incoming.shape)}')
                extended = torch.zeros_like(target[key])
                extended[:, :RELAY_ENCODED_WIDTH] = incoming.detach()
                target[key] = extended
                partial.append(key)
            else:
                if incoming.shape != target[key].shape:
                    raise ValueError(f'Incompatible v9 actor parameter {key}')
                if key == 'gat_actor.0.relay.mode' and int(incoming) != int(target[key]):
                    raise ValueError('v9 warm start must use the same relay_attention mode')
                target[key] = incoming.detach().clone()
                copied.append(key)
        self.load_state_dict(target)
        return dict(copied=copied, partial=partial,
                    fresh=[key for key in target if key not in actor_keys])

    @classmethod
    def from_state_dict(cls, state):
        if int(state.get('survey_schema', -1)) != 1:
            raise ValueError('Unknown survey checkpoint schema')
        mode = MODES[int(state['gat_actor.0.relay.mode'])]
        global_features = state['critic.1.weight'].shape[1]-ENCODED_WIDTH
        model = cls(global_features, 2, mode)
        model.load_state_dict(state)
        return model


class RouteABActorCritic(SurveyActorCritic):
    """V11 fixed a-b route policy with a separate real vehicle anchor."""

    policy_architecture = 'gat_route_ab_v11'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_version = 12
        self.register_buffer('route_ab_schema', torch.tensor(1))

    @classmethod
    def from_state_dict(cls, state):
        if int(state.get('route_ab_schema', -1)) != 1:
            raise ValueError('Unknown a-b route checkpoint schema')
        return super().from_state_dict(state)
