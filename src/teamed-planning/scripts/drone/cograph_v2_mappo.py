"""CoGraph v2 drone policy: learned division of labour over the targets.

The drones' job is unchanged from v1: build the navigation graph from the
vehicles to every target. v2 lets each drone choose for itself which target's
corridor it works on. v1 fixed that choice with a rule (the nearest
still-unconnected target); here it is a learned discrete action.

Choice (TargetSelector): a categorical over the received targets, taken only at
decision points and held in between. The decision points are the first mission,
the chosen target becoming connected while others are still pending, and
COMMIT_S seconds after the last choice. The chosen target fills v8's goal slots,
so the warm-started v8 velocity trunk builds toward it.

No extra communication. What the other drones are building is already implicit
in the graphs this drone receives and fuses. Each target token therefore adds
how close this drone's own fused graph already reaches to that target
(graph_frontier). That exposes corridors others are extending far outside the
+/-96 m map window. The selector reads only token columns that do not depend on
the choice (every column except the focus flag), so its probabilities are the
same before and after the choice fills the goal slots, and PPO re-evaluates them
exactly from the stored features.

Initialisation: a prior logit -PRIOR_DISTANCE*distance/350 m - PRIOR_CONNECTED*connected
plus a zero residual. The greedy choice is then v1's nearest-unconnected rule,
and the sampled choice explores nearby alternatives. The velocity trunk
warm-starts from v8 exactly as in v1.

PPO: the action is [ax, ay, choice]. log_prob = velocity log-prob + decide x
choice log-prob, where decide (the last feature) is 1 only at a decision point.
Entropy and the analytic KL are combined the same way.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Distribution, Independent, Normal, kl_divergence, register_kl

from drone.cograph_mappo import (CoGraphActorCritic, CoGraphNavigator, MultiTargetTaskState, TargetAttention,
    MAX_TARGETS, GLOBAL_FEATURES, ENCODED_WIDTH, RELAY_WIDTH)
from drone.cooperative_mappo import CooperativeCriticEncoder
from drone.graph_attention_mappo import prefix_width
from drone.relay_attention_mappo import RelayActorCritic, RelayEncoder, MODES

POLICY_ARCHITECTURE = 'cograph_v2'
FEATURE_VERSION = 10
# Token layout: v1's seven columns (0-1 relative position /350 m, 2 distance /350 m,
# 3 connected, 4 focus flag, 5 report age /10 s, 6 distance from the nearest vehicle
# /350 m) and 7: closest approach of this drone's fused graph to the target /350 m.
TARGET_FEATURES = 8
FOCUS_COLUMN = 4
SELECT_COLUMNS = (0, 1, 2, 3, 5, 6, 7)
TOKENS_START = RELAY_WIDTH
VALID_START = TOKENS_START+MAX_TARGETS*TARGET_FEATURES
# ... valid flags, then the decide flag last.
RAW_WIDTH = VALID_START+MAX_TARGETS+1
COMMIT_S = 50.
PRIOR_DISTANCE, PRIOR_CONNECTED = 14., 10.
MASKED_LOGIT = -1e9
FRONTIER_CAP = 3.


def split(x):
    """Target tokens, their validity and the decide flag from raw v2 features."""
    tokens = x[..., TOKENS_START:VALID_START].reshape(*x.shape[:-1], MAX_TARGETS, TARGET_FEATURES)
    valid = x[..., VALID_START:VALID_START+MAX_TARGETS] > 0
    return tokens, valid, x[..., VALID_START+MAX_TARGETS]


def graph_frontier(graph, targets):
    """Least distance (m) from any node of this drone's fused graph to each target."""
    targets = np.asarray(targets, float).reshape(-1, 2)
    if graph is None or not len(graph):
        return np.full(len(targets), np.inf)
    points = np.asarray([data['position'][:2] for _, data in graph.graph.nodes(data=True)], float)
    return np.linalg.norm(points[:, None]-targets[None], axis=2).min(0)


def target_tokens(position, view):
    tokens = np.zeros((MAX_TARGETS, TARGET_FEATURES), np.float32)
    valid = np.zeros(MAX_TARGETS, np.float32)
    if view is None:
        return tokens, valid
    if len(view['targets']) > MAX_TARGETS:
        raise ValueError(f'CoGraph v2 supports at most {MAX_TARGETS} targets')
    p = np.asarray(position, float)[:2]
    frontier = view.get('graph_frontier')
    for i, target in enumerate(view['targets']):
        delta = target-p
        reach = FRONTIER_CAP if frontier is None else min(float(frontier[i])/350., FRONTIER_CAP)
        tokens[i] = [delta[0]/350., delta[1]/350., np.linalg.norm(delta)/350., float(view['connected'][i]),
                     float(i == view['focus']), min(view['age']/10., 3.),
                     np.linalg.norm(target-view['vehicle'])/350., reach]
        valid[i] = 1.
    return tokens, valid


class CoGraphV2Encoder(RelayEncoder):
    def __init__(self, mode='full'):
        super().__init__(mode)
        self.targets = TargetAttention(prefix_width(2)+64, TARGET_FEATURES)

    def forward(self, x):
        relay = super().forward(x[..., :RELAY_WIDTH])
        tokens, valid, _ = split(x)
        context, stats = self.targets(relay[..., :prefix_width(2)+64], tokens, valid)
        return torch.cat((relay, context, stats), -1)


class TargetSelector(nn.Module):
    """Per-target logits from choice-independent token columns and their mean."""
    def __init__(self, hidden=64):
        super().__init__()
        width = len(SELECT_COLUMNS)
        self.prior = nn.Linear(width, 1)
        self.residual = nn.Sequential(nn.Linear(2*width+1, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        with torch.no_grad():
            self.prior.weight.zero_()
            self.prior.bias.zero_()
            self.prior.weight[0, SELECT_COLUMNS.index(2)] = -PRIOR_DISTANCE
            self.prior.weight[0, SELECT_COLUMNS.index(3)] = -PRIOR_CONNECTED
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, tokens, valid):
        t = tokens[..., list(SELECT_COLUMNS)]
        weight = valid.to(t.dtype).unsqueeze(-1)
        count = weight.sum(-2).clamp_min(1.)
        context = torch.cat(((t*weight).sum(-2)/count, count/MAX_TARGETS), -1)
        context = context.unsqueeze(-2).expand(*t.shape[:-1], context.shape[-1])
        logits = self.prior(t).squeeze(-1)+self.residual(torch.cat((t, context), -1)).squeeze(-1)
        return logits.masked_fill(~valid, MASKED_LOGIT)


class SelectVelocity(Distribution):
    """Velocity latent and (at decision points only) the target choice."""
    arg_constraints = {}
    has_rsample = False

    def __init__(self, velocity, logits, decide):
        self.velocity = velocity
        self.choice = Categorical(logits=logits, validate_args=False)
        self.decide = decide.to(velocity.mean.dtype)
        super().__init__(batch_shape=velocity.batch_shape, validate_args=False)

    @property
    def mean(self):
        return self.velocity.mean

    def sample(self, sample_shape=torch.Size()):
        choice = self.choice.sample(sample_shape).unsqueeze(-1).to(self.velocity.mean.dtype)
        return torch.cat((self.velocity.sample(sample_shape), choice), -1)

    def log_prob(self, value):
        velocity = value[..., :2].to(self.velocity.mean.dtype)
        return self.velocity.log_prob(velocity)+self.decide*self.choice.log_prob(value[..., 2].round().long())

    def entropy(self):
        return self.velocity.entropy()+self.decide*self.choice.entropy()


@register_kl(SelectVelocity, SelectVelocity)
def _kl_select_velocity(p, q):
    return kl_divergence(p.velocity, q.velocity)+p.decide*kl_divergence(p.choice, q.choice)


class CoGraphV2ActorCritic(CoGraphActorCritic):
    policy_architecture = POLICY_ARCHITECTURE

    def __init__(self, global_features=GLOBAL_FEATURES, input_version=2, relay_attention='full'):
        super().__init__(global_features, input_version, relay_attention)
        self.feature_version = FEATURE_VERSION
        self.gat_actor = nn.Sequential(CoGraphV2Encoder(relay_attention), nn.Linear(ENCODED_WIDTH, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2))
        self.critic = nn.Sequential(
            CooperativeCriticEncoder(global_features, CoGraphV2Encoder(relay_attention), RAW_WIDTH, ENCODED_WIDTH),
            nn.Linear(ENCODED_WIDTH+global_features, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.orthogonal_(self.gat_actor[-1].weight, gain=.01)
        nn.init.zeros_(self.gat_actor[-1].bias)
        self.selector = TargetSelector()

    @classmethod
    def from_state_dict(cls, state):
        mode = MODES[int(state['gat_actor.0.relay.mode'])]
        model = cls(state['critic.1.weight'].shape[1]-ENCODED_WIDTH, 2, mode)
        model.load_state_dict(state)
        return model

    def selection_logits(self, x):
        tokens, valid, _ = split(x)
        return self.selector(tokens, valid)

    def distribution(self, x, mask=None):
        velocity = Independent(Normal(self.gat_actor(x), self.log_std.clamp(-5., 1.).exp()), 1)
        tokens, valid, decide = split(x)
        return SelectVelocity(velocity, self.selector(tokens, valid), decide)

    def encode_graph(self, base, position, observation):
        relay = RelayActorCritic.encode_graph(self, base, position, observation)
        tokens, valid = target_tokens(position, observation.get('cograph'))
        return np.r_[relay, tokens.ravel(), valid, float(observation.get('cograph_decide', 0.))].astype(np.float32)


class ChosenTargetState(MultiTargetTaskState):
    """v1's mission cache; the focus is this drone's own choice once it has one."""
    def __init__(self, drone_id, timeout=10.):
        super().__init__(drone_id, timeout)
        self.choice = None

    def view(self, position):
        view = super().view(position)
        if view is not None:
            view['nearest'] = view['focus']
            if self.choice is not None and self.choice < len(view['targets']):
                view['focus'] = self.choice
        return view


class CoGraphV2Navigator(CoGraphNavigator):
    def make_chain_state(self):
        return ChosenTargetState(self.drone_id)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chain = ChosenTargetState(self.drone_id)
        self.decide_flag, self.choice_time, self.decisions = 0., None, 0

    def cooperation_features(self, state, p, obs):
        super().cooperation_features(state, p, obs)
        view = obs['cograph']
        if view is not None:
            view['graph_frontier'] = graph_frontier(obs.get('graph'), view['targets'])
        obs['cograph_decide'] = self.decide_flag

    def due(self, view, now):
        choice = self.chain.choice
        if choice is None or choice >= len(view['targets']):
            return True
        if view['connected'][choice] and not view['connected'].all():
            return True
        return now-self.choice_time >= COMMIT_S

    def compute_velocity(self, position, observation):
        now = float(observation.get('time', 0.))
        self.decide_flag = 0.
        features, mask = self.encode(position, observation)
        view = self.chain.view(np.asarray(position, float)[:2])
        if view is not None and self.due(view, now):
            with torch.no_grad():
                logits = self.model.selection_logits(torch.from_numpy(features))
                choice = (int(logits.argmax()) if self.deterministic
                          else int(Categorical(logits=logits, validate_args=False).sample()))
            self.chain.choice, self.choice_time = choice, now
            self.decisions += 1
            self.decide_flag = 1.
            # The choice now fills the goal slots; the selector's columns are unchanged.
            features, mask = self.encode(position, observation)
        with torch.no_grad():
            dist = self.model.distribution(torch.from_numpy(features))
            velocity = dist.velocity.mean if self.deterministic else dist.velocity.sample()
            choice = float(self.chain.choice if self.chain.choice is not None else 0)
            action = torch.cat((velocity, velocity.new_tensor([choice])))
            logp = float(dist.log_prob(action))
        self.nominal_velocity = np.r_[self.model.velocity(velocity.numpy(), self.max_speed), 0.]
        self.last_sample = features, mask, action.numpy(), logp
        self.previous_position = np.asarray(position, float).copy()
        self.previous_time = now
        self._current_target = None
        return tuple(self.nominal_velocity)

    def reset(self):
        super().reset()
        self.chain = ChosenTargetState(self.drone_id)
        self.decide_flag, self.choice_time, self.decisions = 0., None, 0
