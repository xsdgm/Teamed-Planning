"""Received-geometry CBF attributes for the world-model/actor graph adapter.

This is the safety-context slice of the graph interface, not a new learned
graph world model. Only the ego controller computes a CBF value. Peer values
remain unknown, and model predictions cannot overwrite the received value.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np

GRAPH_SCHEMA = 'vtemp2_cbf_graph_adapter_v1'
CBF_SCHEMA = 'vtemp2_received_cbf_context_v1'
CBF_FEATURE_INDEX = 32


def normalized_margin(margin, max_speed):
    """Preserve the archived actor convention, including a valid -inf result."""
    if not math.isfinite(max_speed) or max_speed <= 0:
        raise ValueError('CBF max_speed must be positive and finite')
    if margin is None:
        return 1.
    margin = float(margin)
    if math.isnan(margin) or margin == math.inf:
        raise ValueError('Invalid CBF margin; solver failures are not unconstrained')
    return float(np.clip(margin / max_speed, -1., 1.))


def _vector(value):
    values = np.asarray(value, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Invalid received CBF geometry')
    return values.tolist()


def received_cbf_context(controller, observation, stats):
    """Describe a probe of this controller's *actual* received observation.

    Called inside _current_cbf_margin, after get_action has updated the local
    controller. Never accepts simulator truth or another controller's state.
    full_state is used for its clock only, as in the archived runtime.
    """
    now = observation.get('time')
    if now is None:
        now = observation.get('full_state', {}).get('time')
    if now is None or not math.isfinite(float(now)) or now < 0:
        raise ValueError('CBF context requires the decision clock')
    dt = observation.get('time_step')
    if dt is None or not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError('CBF context requires a positive control step')
    nav = controller.navigator
    config = dict(max_speed=float(controller.max_speed), control_dt=float(dt),
        min_separation=float(nav.min_separation),
        communication_range=float(controller.communication_range),
        ground_max_speed=float(getattr(controller, 'ground_max_speed', 0.)),
        world_bounds=None if getattr(nav, 'world_bounds', None) is None
                     else _vector(nav.world_bounds),
        communication_mode=controller.communication_mode,
        multi_vehicle=bool(getattr(controller, 'multi_vehicle', False)),
        # Defaults of the immutable v9 CBFSafetyFilter and margin LP.
        constraint_margin_m=.25, barrier_gain=1., speed_polygon_sides=32)
    neighbors = []
    for neighbor in observation.get('neighbors', ()):
        if neighbor['type'] not in ('drone', 'diff_drive', 'holonomic'):
            continue
        broadcast = neighbor.get('broadcast') or {}
        # Relative sensor geometry and the broadcast timestamp are distinct.
        stamp = broadcast.get('timestamp')
        neighbors.append(dict(id=str(neighbor['id']), kind=neighbor['type'],
            source_id=str(broadcast.get('drone_id', neighbor['id'])),
            relative_position=_vector(neighbor['relative_position']),
            broadcast_stamp=None if stamp is None else float(stamp)))
    neighbors.sort(key=lambda row: (row['id'], row['source_id']))
    chain = getattr(nav, 'chain', None)
    task = None if chain is None else chain.task()
    task_evidence = None if task is None else dict(
        position=_vector(task['position']),
        source_stamp=None if task.get('source_time') is None else float(task['source_time']))
    geometry = dict(position=_vector(controller.position), neighbors=neighbors,
        required_links=sorted(str(x) for x in getattr(nav, 'required_links', ())),
        chain_required=sorted(str(x) for x in getattr(chain, 'required', ())),
        chain_visible=sorted(str(x) for x in getattr(chain, 'visible', ())),
        task=task_evidence,
        direct_ground=None if getattr(nav, 'ground_robot_position', None) is None
                      else _vector(nav.ground_robot_position), config=config)
    encoded = json.dumps(geometry, sort_keys=True, separators=(',', ':'), allow_nan=False)
    margin = stats['feasibility_margin']
    norm = normalized_margin(margin, config['max_speed'])
    negative_infinity = margin is not None and float(margin) == -math.inf
    status = ('unconstrained' if margin is None else
              'infeasible' if float(margin) < 0 else 'ok')
    return dict(schema=CBF_SCHEMA, owner=controller.drone_id,
        # JSON has no infinity. Preserve its meaning rather than serializing NaN.
        cbf_margin_raw=None if margin is None or negative_infinity else float(margin),
        raw_kind='negative_infinity' if negative_infinity else
                 'unconstrained' if margin is None else 'finite',
        cbf_margin_norm=norm, unit='m/s', has_constraints=margin is not None,
        calculation_status=status, computed_at=float(now),
        clipped=bool(margin is not None and abs(float(margin)) > config['max_speed']),
        geometry_source='received_local_geometry',
        geometry_version=hashlib.sha256(encoded.encode()).hexdigest(),
        constraint_config=config,
        # A new calculation does not refresh a task/broadcast source stamp.
        evidence=dict(local_observation_at=float(now),
            task_source_stamp=None if task_evidence is None else task_evidence['source_stamp'],
            neighbor_broadcasts=[dict(source_id=n['source_id'], source_stamp=n['broadcast_stamp'])
                                 for n in neighbors]),
        missing_links=sorted(str(x) for x in stats.get('missing_links', ())))


def validate_cbf_context(context, owner, now):
    if context.get('schema') != CBF_SCHEMA or context.get('owner') != owner:
        raise ValueError('CBF context belongs to a different schema or ego')
    if context.get('geometry_source') != 'received_local_geometry':
        raise ValueError('Estimated CBF margin cannot replace received geometry')
    if not math.isfinite(float(now)) or context.get('computed_at') != float(now):
        raise ValueError('Stale CBF context at actor decision')
    status, kind = context.get('calculation_status'), context.get('raw_kind')
    if status not in ('ok', 'infeasible', 'unconstrained'):
        raise ValueError('CBF calculation failed or is unknown')
    if not context.get('geometry_version'):
        raise ValueError('CBF geometry version is missing')
    raw = context.get('cbf_margin_raw')
    if kind == 'negative_infinity':
        if raw is not None or status != 'infeasible' or context.get('has_constraints') is not True:
            raise ValueError('Invalid infinite CBF margin encoding')
        raw = -math.inf
    elif kind == 'unconstrained':
        if raw is not None or status != 'unconstrained' or context.get('has_constraints') is not False:
            raise ValueError('Invalid unconstrained CBF context')
    elif kind == 'finite':
        if raw is None or not math.isfinite(float(raw)) or context.get('has_constraints') is not True:
            raise ValueError('Invalid finite CBF margin')
        if status != ('infeasible' if raw < 0 else 'ok'):
            raise ValueError('CBF status disagrees with its margin')
    else:
        raise ValueError('Unknown CBF margin encoding')
    expected = normalized_margin(raw, context['constraint_config']['max_speed'])
    if context.get('cbf_margin_norm') != expected:
        raise ValueError('CBF normalized value disagrees with its raw value')
    return expected


def observation_graph(owner, now, position, features, cbf, prepared_peers):
    """Retain legacy actor evidence while adding typed ego safety attributes."""
    validate_cbf_context(cbf, owner, now)
    nodes = {owner: dict(kind='uav', is_ego=True, position=np.asarray(position).copy(),
                        actor_features=np.asarray(features).copy(), cbf=copy.deepcopy(cbf))}
    edges = []
    for prepared in prepared_peers:
        entry, packet = prepared['entry'], prepared['packet']
        target = entry['target']
        if entry['observer'] != owner or entry['frame'] != now or target == owner or target in nodes:
            raise ValueError('Mixed receiver, time or duplicate peer in world graph')
        nodes[target] = dict(kind='uav', is_ego=False, cbf=None,
            world_input=copy.deepcopy(entry), packet=copy.deepcopy(packet),
            origin=np.asarray(entry['origin']).copy(), source_stamp=entry['source_stamp'])
        edges.append(dict(source=owner, target=target, kind='received_telemetry'))
    return dict(schema=GRAPH_SCHEMA, phase='observation', ego_id=owner,
                estimate_stamp=float(now), nodes=nodes, edges=edges)
