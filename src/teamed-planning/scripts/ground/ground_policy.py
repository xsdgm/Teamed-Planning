"""VTemp2 slow policy over vehicle observations and delivered V9 evidence.

Ground inputs use global XY axes centred at the UGV. Only the road topology
approved by the vehicle's observation adapter is eligible for graph encoding.
The bounded GAT uses up to 64 road nodes and seven actual neighbours per node,
plus an aggregation self-loop. Candidate locations are queries, not road edges.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from world_model.slow_policy import SlowActorCritic

ROAD_LIMIT, NEIGHBORS, FLEET_LIMIT = 64, 8, 16
SENSOR_SIDE = 16
CANDIDATE_ROAD_RADIUS_M = 95.
# Fixed width also preserves PPO's microbatch slicing and archived transitions.
CANDIDATE_LIMIT = 96
# dx, dy (95 m units), confidence, log1p(age seconds), age unknown,
# observed graph degree, normalized log observation count, distance from ego (95 m).
ROAD_FEATURES = 8
# Local one-hop relative XYZ, relative velocity XY, velocity known, range, valid.
PEER_FEATURES = 8
# V9 relay token layout is defined by drone.relay_attention_mappo.RelayAttention.
RELAY_FEATURES = 16
# Measured velocity XY, sin/cos yaw, yaw rate, goal and subgoal XY offsets,
# sensor width and height in metres. The remaining two planes hold occupied and unknown.
EGO_FEATURES = 11 + 2 * SENSOR_SIDE**2
BLOCKS = (
    ('ego', (EGO_FEATURES,)),
    ('roads', (ROAD_LIMIT, ROAD_FEATURES)),
    ('adjacency', (ROAD_LIMIT, NEIGHBORS)),
    ('road_valid', (ROAD_LIMIT,)),
    ('peers', (FLEET_LIMIT, PEER_FEATURES)),
    ('relay', (FLEET_LIMIT, RELAY_FEATURES)),
    ('relay_valid', (FLEET_LIMIT,)),
    ('candidate_xy', (CANDIDATE_LIMIT, 2)),
)
GROUND_INPUT_WIDTH = sum(math.prod(shape) for _, shape in BLOCKS)
TASK_FEATURES = ('ground_target_valid', 'survey_target_valid', 'survey_dx_350m', 'survey_dy_350m',
                 'survey_age_log_seconds', 'survey_evidence_received')
TASK_BLOCKS = BLOCKS + (('task', (len(TASK_FEATURES),)),)
TASK_INPUT_WIDTH = GROUND_INPUT_WIDTH + len(TASK_FEATURES)


def ground_sensor_tile(obs):
    """Decode one car sensor image to YX rows with -1 unknown, 0 free, 1 occupied."""
    raw = obs.get('occupancy_map')
    if raw is None or not obs.get('sensor_valid', True):
        return None
    raw = np.asarray(raw)
    if raw.ndim != 2 or not raw.size:
        raise ValueError('Expected a nonempty two-dimensional vehicle occupancy tile')
    known = np.isfinite(raw) & (raw >= 0)
    if 'occupancy_known_mask' in obs:
        mask = np.asarray(obs['occupancy_known_mask'], dtype=bool)
        if mask.shape != raw.shape:
            raise ValueError('Vehicle occupancy and known-mask shapes differ')
        known &= mask
    tile = np.where(known, raw > .5, -1).astype(np.int8)
    axes = obs.get('map_axes', 'yx')
    if axes == 'xy':
        return tile.T
    if axes != 'yx':
        raise ValueError('Invalid ground occupancy sensor axes')
    return tile


def _local_sensor(obs):
    tile = ground_sensor_tile(obs)
    if tile is None:
        sensor = np.zeros((2, SENSOR_SIDE, SENSOR_SIDE), np.float32)
        sensor[1] = 1.
        return sensor, (0., 0.)
    planes = np.stack((tile == 1, tile == -1)).astype(np.float32)
    if tile.shape == (SENSOR_SIDE, SENSOR_SIDE):
        sensor = planes
    else:
        sensor = F.adaptive_max_pool2d(torch.from_numpy(planes), SENSOR_SIDE).numpy()
    resolution = float(obs['map_resolution'])
    return sensor, (tile.shape[1] * resolution, tile.shape[0] * resolution)


def _road_inputs(belief, origin, anchors, now):
    xy = np.asarray(belief['graph_positions'], float).reshape(-1, 2)
    confidence = np.asarray(belief['graph_confidence'], float)
    stamps = np.asarray(belief['graph_source_stamps'], float)
    edges = np.asarray(belief['graph_edge_index'], dtype=np.int64)
    counts = np.asarray(belief['graph_observation_counts'], float)
    if (confidence.shape != (len(xy),) or stamps.shape != (len(xy),)
            or counts.shape != (len(xy),) or edges.ndim != 2 or edges.shape[0] != 2
            or (edges.size and (edges.min() < 0 or edges.max() >= len(xy)))):
        raise ValueError('Invalid received road graph topology or attributes')
    nodes = np.zeros((ROAD_LIMIT, ROAD_FEATURES), np.float32)
    adjacency = np.full((ROAD_LIMIT, NEIGHBORS), -1, np.float32)
    valid = np.zeros(ROAD_LIMIT, np.float32)
    if not len(xy):
        return nodes, adjacency, valid
    indices = np.arange(len(xy))
    if len(xy) > ROAD_LIMIT:
        # Retain measured roads near the current targets before filling by ego range.
        nearest = np.argmin(((anchors[:, None] - xy) ** 2).sum(-1), axis=1)
        by_range = np.argsort(((xy - origin) ** 2).sum(-1), kind='stable')
        indices = np.asarray(list(dict.fromkeys([*nearest, *by_range]))[:ROAD_LIMIT])
    lookup = np.full(len(xy), -1, dtype=np.int64)
    lookup[indices] = np.arange(len(indices))
    deltas = (xy[indices] - origin) / 95.
    degrees = np.bincount(edges[0], minlength=len(xy))
    age_known = np.isfinite(stamps[indices]) & (stamps[indices] >= 0.) & (stamps[indices] <= now)
    nodes[:len(indices), :2] = deltas
    nodes[:len(indices), 2] = np.clip(confidence[indices], 0., 1.)
    nodes[:len(indices), 3] = np.log1p(np.where(age_known, now - stamps[indices], 0.))
    nodes[:len(indices), 4] = ~age_known
    nodes[:len(indices), 5] = np.minimum(degrees[indices] / 8., 1.)
    nodes[:len(indices), 6] = np.clip(np.log1p(counts[indices]) / np.log(1001.), 0., 1.)
    nodes[:len(indices), 7] = np.linalg.norm(deltas, axis=1)
    valid[:len(indices)] = 1.
    outgoing = [[] for _ in indices]
    for source, target in lookup[edges].T:
        if source >= 0 and target >= 0 and source != target:
            dist = float(np.linalg.norm(deltas[source] - deltas[target]))
            outgoing[source].append((dist, target))
    for i, neighbors in enumerate(outgoing):
        neighbors.sort()
        kept = neighbors[:NEIGHBORS - 1]
        adjacency[i, 0] = i
        for j, (_, dest) in enumerate(kept, start=1):
            adjacency[i, j] = dest
    return nodes, adjacency, valid


def _communication_inputs(peers, relay_rows, position):
    if len(peers) > FLEET_LIMIT or len(relay_rows) > FLEET_LIMIT:
        raise ValueError('Vehicle peer/relay observation exceeds V9 capacity')
    direct = np.zeros((FLEET_LIMIT, PEER_FEATURES), np.float32)
    sensed = set()
    for i, peer in enumerate(peers):
        sensed.add(peer['origin_id'])
        delta = np.asarray(peer['relative_position'], float)
        velocity = peer.get('relative_velocity')
        known = velocity is not None and peer.get('velocity_known', True)
        direct[i] = [*(delta / 95.), *(np.asarray(velocity)[:2] / 12. if known else (0., 0.)),
                     float(known), np.linalg.norm(delta) / 95., 1.]
    relay = np.zeros((FLEET_LIMIT, RELAY_FEATURES), np.float32)
    valid = np.zeros(FLEET_LIMIT, np.float32)
    for i, row in enumerate(relay_rows):
        source = np.asarray(row['position'], float)
        delta = np.zeros(3)
        delta[:len(source)] = source
        delta -= position
        velocity, goal = row['velocity'], row['goal']
        vehicle = row['payload_type'] == 'vehicle_state'
        relay[i] = [delta[0]/350., delta[1]/350., delta[2]/15.,
                    *(np.asarray(velocity)[:2]/6. if velocity is not None else (0., 0.)),
                    float(velocity is not None), row['age'], row['hop'], float(not vehicle), float(vehicle),
                    row['sigma']/95., float(row['time_unknown']), float(row['origin_id'] in sensed),
                    *((np.asarray(goal)[:2]-position[:2])/350. if goal is not None else (0., 0.)),
                    float(goal is not None)]
        valid[i] = 1.
    return direct, relay, valid


def ground_observation(episode, candidate_obs, peers, relay_rows, *, task_context=None):
    """Pack receiver evidence and, when supplied, independently issued ground and survey targets.

    The legacy observation takes its driving target directly from the option manager.
    The task-aware observation uses the car's current position for all XY offsets.
    """
    obs = episode.env.obs[episode.env.ground]
    own = obs['self']
    position = np.zeros(3)
    position[:len(own['position'])] = own['position']
    origin = position[:2]
    sensor, extent = _local_sensor(obs)
    yaw = float(own['orientation'])
    goal = np.asarray(episode.goal)[:2]
    now = float(episode.env.env.time)
    task = ()
    survey_anchors = []
    if task_context is None:
        target = np.asarray(episode.manager.target)[:2]
    else:
        ground = task_context['ground_target']
        target = origin if ground is None else np.asarray(ground, dtype=float)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError('Ground target must be a finite XY point')
        survey = task_context['survey_target']
        issued_at = task_context['survey_issued_at']
        received = task_context['survey_evidence_received']
        if not isinstance(received, (bool, np.bool_)):
            raise ValueError('Survey evidence receipt must be a boolean')
        if survey is None:
            if issued_at is not None or received:
                raise ValueError('Survey feedback requires a previously issued target')
            task_features = [ground is not None, False, 0., 0., 0., False]
        else:
            survey = np.asarray(survey, dtype=float)
            if (survey.shape != (2,) or not np.isfinite(survey).all() or issued_at is None
                    or not 0. <= issued_at <= now):
                raise ValueError('Survey task requires a finite XY point and a valid issue time')
            task_features = [ground is not None, True, *((survey-origin)/350.),
                             np.log1p(now - issued_at), received]
            survey_anchors = [survey]
        task = (task_features,)
    ego = np.r_[np.asarray(own['velocity'])[:2]/6., np.sin(yaw), np.cos(yaw), own['angular_velocity'],
                (goal-origin)/350., (target-origin)/350., np.asarray(extent)/95., sensor.ravel()]
    targets = np.asarray(candidate_obs['targets'], float)
    present = np.asarray(candidate_obs['present'], bool)
    if targets.shape != (CANDIDATE_LIMIT, 2) or present.shape != (CANDIDATE_LIMIT,):
        raise ValueError('Expected padded vehicle candidate locations and presence mask')
    anchors = np.vstack((origin, target, goal, *survey_anchors, targets[present]))
    roads = _road_inputs(episode.belief(), origin, anchors, now)
    comm = _communication_inputs(peers, relay_rows, position)
    locations = np.where(present[:, None], (targets-origin)/95., 0.)
    packed = np.concatenate([np.ravel(block) for block in (ego, *roads, *comm, locations, *task)]).astype(np.float32)
    width = GROUND_INPUT_WIDTH if task_context is None else TASK_INPUT_WIDTH
    if packed.shape != (width,) or not np.isfinite(packed).all():
        raise ValueError('Nonfinite or malformed ground policy inputs')
    return dict(candidate_obs, ground_inputs=packed)


def _blocks(packed, blocks):
    sizes = [math.prod(shape) for _, shape in blocks]
    return {name: part.reshape(-1, *shape)
            for (name, shape), part in zip(blocks, packed.split(sizes, -1))}


class _GroundEncoder(nn.Module):
    """Use the archived V9 GAT, peer key/query/value projections and relay attention."""
    def __init__(self, features, hidden, task_context):
        super().__init__()
        # The entry point initializes the checked, frozen runtime before creating
        # models in the training process or spawned workers. Importing this file
        # for snapshots/schema inspection does not import a mutable drone package.
        from world_model import frozen_navigation
        if frozen_navigation._RUNTIME is None:
            raise RuntimeError('Call frozen_navigation.runtime(repo) before constructing the ground policy')
        from drone.cooperative_mappo import CooperativeEncoder
        from drone.relay_attention_mappo import RelayAttention

        coop = CooperativeEncoder()
        self.gat1, self.gat2 = coop.map.layer1, coop.map.layer2
        self.peer_query, self.peer_key, self.peer_value = coop.peer_query, coop.peer_key, coop.peer_value
        # V9 queries combine ego/sensor state with 64 graph readout features.
        query_width = coop.peer_query.in_features
        ego_width = EGO_FEATURES + (len(TASK_FEATURES) if task_context else 0)
        self.ego = nn.Sequential(nn.Linear(ego_width, query_width - 64), nn.Tanh())
        self.relay = RelayAttention(query_width, 'full')
        self.candidate = nn.Linear(features, 32)
        self.road_key = nn.Linear(32, 32)
        self.head = nn.Sequential(nn.Linear(features + 32 + 1 + query_width + 32 + 36, hidden),
                                  nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())

    def forward(self, x, data, *, return_ego_and_roads=False):
        nodes, indices = data['roads'], data['adjacency'].long()
        valid = data['road_valid'].bool()
        xy = nodes[..., :2]
        h = self.gat1(nodes, indices, valid, xy)
        h = self.gat2(h, indices, valid, xy)
        mean = h.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        maximum = h.masked_fill(~valid[..., None], -1e9).amax(1)
        maximum = torch.where(valid.any(1, keepdim=True), maximum, 0.)
        ego = torch.cat((data['ego'], data['task']), -1) if 'task' in data else data['ego']
        ego_state = self.ego(ego)
        query = torch.cat((ego_state, mean, maximum), -1)
        peers = data['peers']
        # The V9 one-hop branch uses scaled dot-product attention with a presence mask.
        peer_scores = (self.peer_query(query)[:, None] * self.peer_key(peers)).sum(-1)/math.sqrt(32.)
        sensed = peers[..., -1].bool()
        weights = peer_scores.masked_fill(~sensed, -1e9).softmax(-1) * sensed
        peer_context = (weights[..., None] * self.peer_value(peers)).sum(1)
        relay, stats = self.relay(query, data['relay'], data['relay_valid'].bool())
        # Candidate-specific readout over connected-road embeddings in vehicle coordinates.
        road_scores = self.candidate(x) @ self.road_key(h).transpose(-2, -1) / math.sqrt(32.)
        offset = data['candidate_xy'][:, :, None] - xy[:, None]
        distances_sq = offset.square().sum(-1)
        nearby = (distances_sq <= (CANDIDATE_ROAD_RADIUS_M / 95.) ** 2) & valid[:, None]
        road_weights = (road_scores - distances_sq).masked_fill(~nearby, -1e9).softmax(-1) * nearby
        road_context = road_weights @ h
        has_nearby = nearby.any(-1, keepdim=True).to(x.dtype)
        shared = torch.cat((query, peer_context, relay, stats), -1)
        directions = self.head(torch.cat((x, road_context, has_nearby,
                                          shared[:, None].expand(-1, x.shape[1], -1)), -1))
        if return_ego_and_roads:
            return directions, ego_state, h
        return directions



class GroundSlowActorCritic(SlowActorCritic):
    """Separate trainable vehicle actor and critic; frozen V9 UAV weights stay external."""
    def __init__(self, features, hidden=64, *, task_context=False):
        nn.Module.__init__(self)
        self.features, self.hidden = int(features), int(hidden)
        self.blocks = TASK_BLOCKS if task_context else BLOCKS
        self.input_width = TASK_INPUT_WIDTH if task_context else GROUND_INPUT_WIDTH
        self.actor_encoder = _GroundEncoder(features, hidden, task_context)
        self.critic_encoder = _GroundEncoder(features, hidden, task_context)
        self.actor_head = nn.Linear(hidden, 1)
        self.critic_head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.actor_head.weight, gain=.01)
        nn.init.zeros_(self.actor_head.bias)

    def forward(self, x, mask, present, ground_inputs):
        shape = x.shape[:-2]
        if (x.shape[-2:] != (CANDIDATE_LIMIT, self.features) or mask.shape != x.shape[:-1]
                or present.shape != mask.shape or ground_inputs.shape != (*shape, self.input_width)):
            raise ValueError('Ground policy input shapes disagree')
        if not bool(mask.any(-1).all()) or bool((mask & ~present).any()):
            raise ValueError('Every vehicle decision requires at least one legal, present candidate')
        data = _blocks(ground_inputs, self.blocks)
        rows = x.reshape(-1, CANDIDATE_LIMIT, self.features)
        logits = self.actor_head(self.actor_encoder(rows, data)).reshape(*shape, CANDIDATE_LIMIT)
        logits = logits.masked_fill(~mask, -torch.inf)
        encoded = self.critic_encoder(rows, data).reshape(*shape, CANDIDATE_LIMIT, self.hidden)
        pooled = (encoded * present[..., None]).sum(-2) / present.sum(-1, keepdim=True)
        return Categorical(logits=logits), self.critic_head(pooled).squeeze(-1)
