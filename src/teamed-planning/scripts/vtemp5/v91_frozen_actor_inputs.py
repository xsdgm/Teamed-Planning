"""Inference adapter for the frozen VTemp5 vehicle actor on V9.1 receiver data.

The CNN has no bridge/waypoint head. Two short, three-vertex route hypotheses
are derived from receiver occupancy, then packed using VTemp5's road-node
format. These hypotheses can inform the actor but never certify an entry.
"""
import math
from types import SimpleNamespace

import numpy as np
import torch


def load_actor(path):
    # The caller selects the matching frozen UAV runtime before these imports.
    from vtemp4.vtemp4_body_graph import GoalResidualBodyFramePolicy
    from vtemp4.vtemp4_ring_contract import RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT

    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['format'] != RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT:
        raise ValueError('Expected a goal-residual body-frame vehicle policy')
    actor = GoalResidualBodyFramePolicy()
    actor.load_state_dict(saved['model'], strict=True)
    if any(not p.isfinite().all() for p in actor.parameters()):
        raise ValueError('Invalid actor weights')
    return actor.eval().requires_grad_(False)


def _distance_to_segment(xy, a, b):
    delta = b - a
    t = np.clip(((xy - a) @ delta) / max(delta @ delta, 1e-9), 0., 1.)
    return np.linalg.norm(xy - a - t[..., None] * delta, axis=-1)


def _candidate_roads(xy, car, goal, free, occupied, probabilities):
    """Score identical left/right options for both arms; veto observed blockers."""
    forward = goal - car
    forward = forward / np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0]])
    unknown = ~(free | occupied)
    # Half the 2 m cell diagonal adds an allowance for grid quantisation.
    clearance = 1.5 + math.sqrt(2.)
    branch = {'observed': {}, 'predicted': {}}
    for side in (1, -1):
        for degrees in (12, 24, 36, 48, 60, 72, 84, 96, 108):
            angle = math.radians(degrees)
            heading = math.cos(angle) * forward + side * math.sin(angle) * left
            # Centrelines are short enough to stay inside the 96 m output window.
            route = np.array([car + 8. * heading, car + 20. * heading,
                              car + 20. * heading + 12. * forward])
            corridor = ((_distance_to_segment(xy, car, route[1]) <= clearance) |
                        (_distance_to_segment(xy, route[1], route[2]) <= clearance))
            if occupied[corridor].any():
                continue
            measured = np.array([free[np.linalg.norm(xy - pt, axis=-1) <= clearance].all() for pt in route])
            unknown_here = corridor & unknown
            count = int(corridor.sum())
            if count == 0:
                continue
            costs = {'observed': .5 * unknown_here.sum() / count,
                     'predicted': probabilities[unknown_here].sum() / count}
            for arm, cost in costs.items():
                # The tie breaker favours progress on the mission for equal risk.
                rank = (float(cost), degrees)
                if side not in branch[arm] or rank < branch[arm][side][0]:
                    branch[arm][side] = (rank, route, measured, 1. - float(cost))
    return branch


class EpisodeActorInputs:
    def __init__(self):
        from vtemp4.vtemp4_decision_graph_history import FullCommandHistory
        from vtemp4.vtemp4_road_traversal import RoadTraversalMemory

        self.commands = []
        self.replans = []
        self.local = None
        self.history = FullCommandHistory()
        self.traversals = RoadTraversalMemory()
        self.encounter = False

    def record_local(self, step, target):
        target = None if target is None else np.asarray(target, dtype=float).reshape(2).copy()
        if target is None and self.local is None or np.array_equal(target, self.local):
            return
        self.local = target
        self.replans.append({'time_s': step, 'target': target})

    def _base(self, env, snap, yaw, own):
        from shared.simulator.ground_evidence import observed_fused_graph
        from survey.survey_fleet_clearance import native_segment_clear
        from vtemp2_3.ground_policy import ground_sensor_tile, observed_road_inputs
        from vtemp4.vtemp4_ring_contract import RING_RADIUS_M
        from vtemp4.vtemp4_vehicle_graph import GOAL_SCALE_M

        car, goal, now = snap['car_xy'], snap['goal_xy'], snap['frame']
        history = self.history.observation(now, car, goal, RING_RADIUS_M, snap['car_path'], self.commands,
                                           self.replans)
        anchors = [car, goal]
        if self.local is not None:
            anchors.append(self.local)
        if self.commands:
            anchors.append(self.commands[-1]['target'])
        graph = observed_fused_graph(env.ground_controller.unified_graph, now)
        nodes, adjacency, valid, roads = observed_road_inputs(graph, car, np.asarray(anchors), now)
        obs = env.obs[env.ground]
        tile = ground_sensor_tile(obs)
        entries = np.zeros(len(valid), dtype=bool)
        if tile is not None:
            for i in np.flatnonzero(np.linalg.norm(roads-car, axis=1) <= 15.):
                entries[i] = native_segment_clear(tile, car, obs['map_resolution'], car, roads[i])
        traversal = self.traversals.edge_features(now, car, snap['car_path'], roads, adjacency, valid)
        world = snap['air_cells'].shape[0] * snap['resolution']
        local = np.zeros(4) if self.local is None else np.r_[self.local / world, (self.local-car)/GOAL_SCALE_M]
        return dict(**history, vehicle=np.array([*(car/world), *(np.asarray(own['velocity'])[:2]/6.),
                                                  math.sin(yaw), math.cos(yaw), own['angular_velocity']], np.float32),
                    goal=np.r_[goal/world, (goal-car)/GOAL_SCALE_M].astype(np.float32),
                    local_target=local.astype(np.float32), local_target_valid=self.local is not None,
                    road_nodes=nodes, road_neighbors=adjacency.astype(np.int64), road_valid=valid.astype(bool),
                    road_entry_mask=entries, road_traversals=traversal)

    @torch.inference_mode()
    def review(self, env, snapshot, fields, actor, arm):
        from vtemp4.vtemp4_body_graph import body_frame_observation
        from vtemp4.vtemp4_periodic_history import legal_ring_arcs
        from vtemp4.vtemp4_ring_contract import RING_RADIUS_M
        from vtemp5.vtemp5_fleet import add_candidate_roads

        if arm not in ('observed', 'predicted'):
            raise ValueError('Unknown frozen-actor evaluation arm')
        car, goal, now = snapshot['car_xy'], snapshot['goal_xy'], int(snapshot['frame'])
        own = env.obs[env.ground]['self']
        yaw = float(own['orientation'])
        x, xy, _, probability = fields
        free, occupied = x[0].astype(bool), x[1].astype(bool)
        delta = goal - car
        distance = float(np.linalg.norm(delta))
        forward = delta / max(distance, 1.e-9)
        clearance = 1.5 + math.sqrt(2.)
        corridor = _distance_to_segment(xy, car, car + min(48., distance) * forward) <= clearance
        near = _distance_to_segment(xy, car, car + min(24., distance) * forward) <= clearance
        if occupied[corridor].any():
            self.encounter = True
        elif free[near].all():
            self.encounter = False
        call = bool(distance > 2. and (self.encounter or distance > RING_RADIUS_M))
        angles = dict.fromkeys(('observed', 'predicted'), float(math.atan2(delta[1],delta[0])-yaw))
        counts = {'observed': 0, 'predicted': 0}
        changed = False
        if call:
            base = self._base(env, snapshot, yaw, own)
            air, ground = snapshot['air_cells'], snapshot['ground_cells']
            fused = np.where((air > 0) | (ground > 0), 1, np.minimum(air, ground)).astype(np.int8)
            # The original actor's legal-ring sampler uses a 1 m clearance kernel.
            res = float(snapshot['resolution'])
            if res != 2.:
                raise ValueError('Expected native 2 m evidence')
            adapter = SimpleNamespace(cells=fused.repeat(2,0).repeat(2,1), resolution=1.,
                                      ep=SimpleNamespace(case={'world_size': fused.shape[0] * res}))
            arcs, _ = legal_ring_arcs(adapter, car, yaw, RING_RADIUS_M)
            options = _candidate_roads(xy, car, goal, free, occupied, probability)
            observations = {}
            for name, candidates in options.items():
                obs = dict(base, angle_arcs=arcs)
                state = dict(routes={s: v[1] for s, v in candidates.items()},
                             route_observed={s: v[2] for s, v in candidates.items()},
                             route_open_probability={s: v[3] for s, v in candidates.items()},
                             side_status={s:'unknown' for s in candidates}, evidence_time=None)
                # Observed vertices lack native per-cell timestamps in these V9.1
                # snapshots. Candidate flags denote current fused-map support.
                add_candidate_roads(obs, state, car, goal, now)
                observations[name] = body_frame_observation(obs)
                counts[name] = len(candidates)
            changed = not np.array_equal(observations['observed']['road_nodes'], observations['predicted']['road_nodes'])
            # Use the exact single-observation deterministic inference used in formal
            # evaluation, including action decoding inside legal physical body arcs.
            angles['observed'] = actor.act(observations['observed'], deterministic=True)[1]
            angles['predicted'] = (actor.act(observations['predicted'], deterministic=True)[1] if changed
                                   else angles['observed'])
        angle = angles[arm]
        target = car + RING_RADIUS_M * np.array([math.cos(yaw+angle), math.sin(yaw+angle)]) if call else goal
        self.commands.append(dict(issued_at_s=now, origin_xy=car.copy(), target=target.copy()))
        other = 'observed' if arm == 'predicted' else 'predicted'
        return dict(target_xy=target.tolist(), source='policy_angle' if call else 'final_goal',
                    actor_called=call, candidate_count=counts[arm], candidate_input_changed=changed,
                    unknown_fraction=float((~(free | occupied))[corridor].mean()),
                    angle_rad=float(angle), other_angle_rad=float(angles[other]),
                    changed_angle_rad=float(abs(math.remainder(angle - angles[other], math.tau))))
