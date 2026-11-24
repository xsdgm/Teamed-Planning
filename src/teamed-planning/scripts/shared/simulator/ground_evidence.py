"""VTemp2 observations from the ground robot's local sensor and received maps."""
from __future__ import annotations

import math

import numpy as np

from vtemp2_3.ground_policy import ground_observation, ground_sensor_tile
from shared.simulator.ground_relay import GroundRelayObservation
from vtemp2_3.hierarchical_runtime import HierarchicalEpisode, build_observation
from vtemp2_3.navigation_evidence import EvidenceGrid
from vtemp2_3.wildscenes_runtime import pool_received_tile


LOCAL_MEMORY_RESOLUTION_M = 2.
# Both ends of the received road segment must lie in this disk around the survey target.
SURVEY_EVIDENCE_RADIUS_M = 25.


def remember_local_tile(cells, resolution, observation):
    """Retain local measurements without turning unsupported space into free space."""
    tile = ground_sensor_tile(observation)
    if tile is None:
        return
    ids, values = pool_received_tile(
        tile, observation['self']['position'], float(observation['map_resolution']),
        resolution, cells.shape, occupied_overrides_unknown=True)
    observed = values != 0
    cells.flat[ids[observed]] = values[observed]


def observed_fused_graph(graph, now):
    """Copy received topology and original observation stamps for the slow policy."""
    positions, confidence, counts, stamps, edges = [], [], [], [], []
    edge_count = 0
    if graph is not None:
        index = {}
        for node, attributes in graph.graph.nodes(data=True):
            index[node] = len(positions)
            positions.append(np.asarray(attributes['position'], dtype=float)[:2])
            confidence.append(graph.node_confidence(node))
            counts.append(attributes.get('observation_count', 1))
            received_frames = attributes.get('observation_frames', {}).values()
            stamps.append(max((float(frame) for frame in received_frames if 0 <= frame <= now),
                              default=np.nan))
        for start, end, attributes in graph.graph.edges(data=True):
            if attributes.get('bridged', False):
                # Frozen V9 may certify component bridges using its audit ledger.
                continue
            edge_count += 1
            edges.append((index[start], index[end]))
            if not graph.graph.is_directed():
                edges.append((index[end], index[start]))
    return dict(
        graph_positions=np.asarray(positions, dtype=float).reshape(-1, 2),
        graph_confidence=np.asarray(confidence, dtype=np.float32),
        graph_observation_counts=np.asarray(counts, dtype=np.int64),
        graph_source_stamps=np.asarray(stamps, dtype=float),
        graph_edge_index=np.asarray(edges, dtype=np.int64).reshape(-1, 2).T.copy(),
        graph_edges=edge_count,
    )


def survey_has_road_evidence(belief, target, issued_at, now, radius):
    """A received, observed road segment in the requested area, measured after dispatch."""
    edges = belief['graph_edge_index']
    if not edges.size:
        return False
    stamps = belief['graph_source_stamps']
    offsets = belief['graph_positions'] - target
    eligible = ((issued_at < stamps) & (stamps <= now)
                & (np.sum(offsets**2, axis=1) <= radius**2))
    start, end = edges
    return bool(np.any((start != end) & eligible[start] & eligible[end]))


class GroundPolicyEpisode(HierarchicalEpisode):
    """Use receiver evidence to generate, mask and reconsider slow policy actions."""

    def __init__(self, repo, case, config=None, *, fast_seed=0, rollout_limit=1800,
                 task_context=False):
        super().__init__(repo, case, config, fast_seed=fast_seed, rollout_limit=rollout_limit)
        self.task_context = task_context
        self._survey_target = None
        self._survey_issued_at = None
        self._survey_evidence_received = False
        self._last_shared_survey_target = None
        self._local_resolution = LOCAL_MEMORY_RESOLUTION_M
        self._world_size = float(case['world_size'])
        side = math.ceil(self._world_size / self._local_resolution)
        self._local_cells = np.zeros((side, side), dtype=np.int8)
        self._local_time = None
        try:
            self._relay_observation = GroundRelayObservation(self.env, seed=case['seed'])
        except BaseException:
            super().close()
            raise

    def _remember_ground(self, now):
        if now != self._local_time:
            remember_local_tile(self._local_cells, self._local_resolution,
                                self.env.obs[self.env.ground])
            self._local_time = now

    def record_survey_dispatch(self, target, issued_at):
        """Record a new outgoing survey request, independently of any drive command."""
        target = np.asarray(target, dtype=float)
        issued_at = float(issued_at)
        if (target.shape != (2,) or not np.isfinite(target).all()
                or not 0. <= issued_at <= self.env.env.time):
            raise ValueError('Expected a finite survey XY and an observed dispatch time')
        if self._survey_issued_at is not None:
            if issued_at < self._survey_issued_at:
                return
            if issued_at == self._survey_issued_at and np.array_equal(target, self._survey_target):
                return
        self._survey_target = target.copy()
        self._survey_issued_at = issued_at
        self._survey_evidence_received = False

    def _remember_shared_survey(self):
        # On the current single-target runtime, self.broadcast is the command the
        # simulator actually transmitted in the preceding physics step. Preparing
        # a command, driving to a point or repeating a broadcast is not a dispatch.
        sent = self.env.obs[self.env.ground]['self'].get('broadcast') or {}
        target = sent.get('target')
        if self.holding or not self.manager.selected or target is None:
            return
        if (self._last_shared_survey_target is None
                or not np.array_equal(target, self._last_shared_survey_target)):
            self.record_survey_dispatch(target, sent['timestamp'])
            self._last_shared_survey_target = np.asarray(target, dtype=float).copy()

    def _update_survey_evidence(self, belief, now):
        if self._survey_target is not None and not self._survey_evidence_received:
            self._survey_evidence_received = survey_has_road_evidence(
                belief, self._survey_target, self._survey_issued_at, now, SURVEY_EVIDENCE_RADIUS_M)

    def planning_task_context(self):
        now = float(self.env.env.time)
        self._update_survey_evidence(self.belief(), now)
        return dict(
            ground_target=self.manager.target.copy() if self.manager.selected else None,
            survey_target=None if self._survey_target is None else self._survey_target.copy(),
            survey_issued_at=self._survey_issued_at,
            survey_evidence_received=self._survey_evidence_received,
        )

    def _ground_action(self, neighbors=None):
        now = float(self.env.env.time)
        self._remember_ground(now)
        if self.task_context and neighbors is not None:
            self._remember_shared_survey()
        action = super()._ground_action(neighbors)
        # A later fusion can replace an earlier graph before the next slow
        # decision. Retain task evidence at the actual receiver's fusion time.
        if self.task_context and self._survey_target is not None and not self._survey_evidence_received:
            controller = self.env.ground_controller
            if now in controller.last_fusion_per_drone.values():
                self._update_survey_evidence(observed_fused_graph(controller.unified_graph, now), now)
        self._belief_cache = None
        return action

    def belief(self):
        now = float(self.env.env.time)
        if self._belief_time == now and self._belief_cache is not None:
            return self._belief_cache
        self._remember_ground(now)
        observation = self.env.obs[self.env.ground]['self']
        controller = self.env.ground_controller
        ages = [now - timestamp for timestamp in controller.last_fusion_per_drone.values()
                if 0 <= timestamp <= now]
        belief = dict(
            position=np.asarray(observation['position'], dtype=float)[:2].copy(),
            goal=self.goal.copy(),
            grid=EvidenceGrid(self._local_cells, resolution=self._local_resolution),
            bounds=(0., 0., self._world_size, self._world_size),
            recent_receipts=sum(age <= 10. for age in ages) / len(self.env.ids),
            receipt_age=float(np.mean(ages)) if ages else 60.,
            speed=float(np.linalg.norm(np.asarray(observation['velocity'], dtype=float)[:2])),
            **observed_fused_graph(controller.unified_graph, now),
        )
        self._belief_cache = belief
        self._belief_time = now
        return belief

    def observation(self):
        candidates = build_observation(
            self.belief(), self.memory, self.config, now=self.env.env.time,
            remaining_budget=max(0., 1. - self.attempts / self.mission_budget),
            previous=self.previous)
        received = self._relay_observation.observe()
        inputs = (self, candidates, received['one_hop_neighbors'], received['relay_tokens'])
        if getattr(self, 'task_context', False):
            return ground_observation(*inputs, task_context=self.planning_task_context())
        return ground_observation(*inputs)

    def close(self):
        try:
            self._relay_observation.close()
        finally:
            super().close()
