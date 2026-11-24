"""Adapt a public vtemp5 review to the existing goal-relative vehicle policy."""
import math

import numpy as np

from vtemp2_3.ground_policy import observed_road_inputs
from vtemp4.vtemp4_body_graph import body_frame_observation
from vtemp4.vtemp4_decision_graph_history import COMMAND_FEATURES, DRIVE_FEATURES
from vtemp4.vtemp4_ring_contract import RING_RADIUS_M
from vtemp4.vtemp4_road_traversal import TRAVERSAL_FEATURES
from vtemp4.vtemp4_vehicle_graph import GOAL_SCALE_M
from vtemp5.vtemp5_env import _target


REVIEW_S = 30


def wrap_angle(angle):
    return (angle + math.pi) % math.tau - math.pi


def body_arcs(goal_arcs, goal_bearing_in_body):
    """Rotate admissible arcs into the physical angle space in which PPO logs its action."""
    parts = []
    for low, high in goal_arcs:
        width = high - low
        if not 0 < width <= math.tau + 1.e-10:
            raise ValueError('Invalid goal-relative sector')
        if width >= math.tau - 1.e-12:
            return np.array([[-math.pi, math.pi]])
        start = wrap_angle(low + goal_bearing_in_body)
        end = start + width
        parts.append((start, min(end, math.pi)))
        if end > math.pi:
            parts.append((-math.pi, end - math.tau))
    merged = []
    for low, high in sorted(parts):
        if high - low < 1.e-12:
            continue
        if merged and low <= merged[-1][1] + 1.e-12:
            merged[-1][1] = max(high, merged[-1][1])
        else:
            merged.append([low, high])
    if not merged:
        raise ValueError('Empty admissible sector at an active decision')
    return np.asarray(merged, dtype=float)


def relative_goal_bearing(public):
    delta = np.asarray(public['goal']) - public['position']
    return wrap_angle(math.atan2(delta[1], delta[0]) - public['heading_rad'])


def _road_graph(public, position):
    points, measured, confidence, edges = [], [], [], []
    for side, route in public['routes'].items():
        if public['side_status'][side] == 'blocked':
            continue
        start = len(points)
        points.extend(route)
        observed = public['route_observed'][side]
        measured.extend(observed)
        # For large terrain the frozen world model supplies the chance that this
        # branch connects through. Measured vertices retain separate provenance.
        probability = public.get('route_open_probability', {}).get(side, .25)
        confidence.extend(1. if known else probability for known in observed)
        for i in range(start + 1, len(points)):
            edges.extend(((i - 1, i), (i, i - 1)))
    now = public['elapsed_steps'] * REVIEW_S
    measured = np.asarray(measured, dtype=bool)
    graph = dict(graph_positions=np.asarray(points).reshape(-1, 2),
                 graph_edge_index=np.asarray(edges, dtype=np.int64).reshape(-1, 2).T,
                 graph_confidence=np.asarray(confidence),
                 graph_source_stamps=np.where(measured, now, -1.),
                 graph_observation_counts=measured.astype(float))
    nodes, adjacent, valid, xy = observed_road_inputs(graph, position, np.array([position, public['goal']]), now)
    entries = np.zeros(len(valid), dtype=bool)
    # Only observed near-field vertices may be direct entries. Predicted segments
    # are explicitly marked by unknown source time, lower confidence and zero count.
    entries[:len(xy)] = (np.linalg.norm(xy - position, axis=1) <= 15.) & (nodes[:len(xy), 4] == 0)
    return dict(road_nodes=nodes, road_neighbors=adjacent.astype(np.int64), road_valid=valid.astype(bool),
                road_entry_mask=entries,
                road_traversals=np.zeros((*adjacent.shape, TRAVERSAL_FEATURES), dtype=np.float32))


def _command_history(history, position, goal):
    count = len(history)
    owners = np.arange(count, dtype=np.int64)
    if count == 0:
        return dict(command_nodes=np.empty((0, COMMAND_FEATURES), np.float32), command_predecessor=owners,
                    drive_segments=np.empty((0, DRIVE_FEATURES), np.float32), drive_command=owners)
    origin = np.asarray([row['position_before'] for row in history])
    end = np.asarray([row['position_after'] for row in history])
    waypoint = np.asarray([row['waypoint'] for row in history])
    valid_local = np.array([row['local_target'] is not None for row in history], dtype=bool)
    local = np.array([row['local_target'] if row['local_target'] is not None else position for row in history])
    duration = np.full(count, math.log1p(REVIEW_S))
    age = np.log1p((count - owners) * REVIEW_S)
    target_gain = np.linalg.norm(waypoint - origin, axis=1) - np.linalg.norm(waypoint - end, axis=1)
    goal_gain = np.linalg.norm(goal - origin, axis=1) - np.linalg.norm(goal - end, axis=1)
    start_offset = (origin - position) / RING_RADIUS_M
    end_offset = (end - position) / RING_RADIUS_M
    commands = np.column_stack((start_offset, (waypoint-position)/RING_RADIUS_M, end_offset, age, duration,
                                target_gain/RING_RADIUS_M, goal_gain/RING_RADIUS_M, owners == count-1))
    distance = np.asarray([row['distance_m'] for row in history])
    drives = np.column_stack((start_offset, end_offset, (local-position)/RING_RADIUS_M, valid_local,
                              age, duration, distance/RING_RADIUS_M))
    return dict(command_nodes=commands.astype(np.float32), command_predecessor=owners-1,
                drive_segments=drives.astype(np.float32), drive_command=owners)


def policy_observation(public, *, evidence_reference=False):
    position = np.asarray(public['position'], dtype=float)
    goal = np.asarray(public['goal'], dtype=float)
    yaw, history = public['heading_rad'], public['history']
    velocity = np.zeros(2)
    angular_velocity = 0.
    local = None
    if history:
        last = history[-1]
        velocity = (position - last['position_before']) / REVIEW_S
        local = last['local_target']
        if last['distance_m'] > 1.e-8:
            # The recorded command bearing may differ from the actual vehicle
            # heading after bank contact. Recover yaw from measured displacements.
            previous_yaw = 0.
            for earlier in reversed(history[:-1]):
                delta = np.asarray(earlier['position_after']) - earlier['position_before']
                if np.linalg.norm(delta) > 1.e-8:
                    previous_yaw = math.atan2(delta[1], delta[0])
                    break
            angular_velocity = wrap_angle(yaw - previous_yaw) / REVIEW_S
    # A delivered closure invalidates the previous interval's local target.
    # Only the active reference changes; command/drive history remains measured.
    side = public['locked_side']
    if evidence_reference and side and public['side_status'][side] == 'blocked':
        remaining = -side
        if public['side_status'][remaining] != 'blocked':
            local = _target(position, public['routes'][remaining])
    road = _road_graph(public, position)
    commands = _command_history(history, position, goal)
    obs = dict(**road, **commands,
               vehicle=np.array([0., 0., *(velocity/6.), math.sin(yaw), math.cos(yaw), angular_velocity]),
               goal=np.r_[0., 0., (goal-position)/GOAL_SCALE_M],
               local_target=np.r_[0., 0., np.zeros(2) if local is None else (np.asarray(local)-position)/GOAL_SCALE_M],
               local_target_valid=local is not None,
               angle_arcs=body_arcs(public['goal_arcs'], relative_goal_bearing(public)))
    return body_frame_observation(obs)
