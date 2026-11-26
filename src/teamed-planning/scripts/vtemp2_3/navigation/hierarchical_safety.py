"""Versioned local safety repairs around the immutable v9 experiment.

Only a hierarchical episode's controller instances are adapted. Historical
modules, checkpoints, communication ranges and received-link selection stay
untouched. A preferred inner communication envelope never weakens the hard
constraints and cannot turn an infeasible hard problem into an executed step.
"""
from __future__ import annotations

import math
from types import MethodType

import numpy as np

from .hierarchical_cbf import CBFSafetyFilter, PROJECTION_VERSION


CONTROL_VERSION = 'vtemp2_local_safety_repair_v1'
COMMUNICATION_RESERVE_M = 5.


def control_contract():
    return dict(version=CONTROL_VERSION, cbf_projection=PROJECTION_VERSION,
        communication_reserve_m=COMMUNICATION_RESERVE_M,
        communication_guard='Prefer a feasible inner envelope; certify against unchanged hard constraints.',
        ground_geometry='Native aligned local safety raster; unchanged clearance and unknown-space rejection.',
        inputs='Received relative geometry and local sensor support only.')


def guarded_filter(velocity, position, peers, dt, *, min_separation, comm_range,
                   max_speed, reserve_m=COMMUNICATION_RESERVE_M, **geometry):
    """Prefer communication headroom without declaring inner failure terminal."""
    if not math.isfinite(reserve_m) or reserve_m < 0:
        raise ValueError('Communication reserve must be finite and nonnegative')
    hard = CBFSafetyFilter(min_separation, comm_range, max_speed)
    result, stats = hard.filter(velocity, position, peers, dt, **geometry)
    # Leave a nonempty distance annulus for nondefault controller settings too.
    reserve = min(reserve_m, (comm_range - min_separation - 2 * hard.margin) / 2)
    enabled = bool(reserve > 0 and geometry.get('required_links'))
    stats.update(communication_guard_enabled=enabled, communication_reserve_m=reserve,
                 communication_guard_feasible=None, communication_guard_applied=False,
                 communication_guard_certificate_status=None,
                 hard_solver_status=stats['solver_status'])
    if stats['infeasible'] or not enabled:
        return result, stats
    inner = CBFSafetyFilter(min_separation, comm_range - reserve, max_speed)
    preferred, inner_stats = inner.filter(velocity, position, peers, dt, **geometry)
    stats.update(communication_guard_feasible=not inner_stats['infeasible'],
                 communication_guard_solver_status=inner_stats['solver_status'])
    if inner_stats['infeasible']:
        return result, stats
    # Independently certify the actual command with the original envelope.
    certified, certificate = hard.filter(preferred, position, peers, dt, **geometry)
    # A point constructed on the speed circle may exceed its radius by an
    # ulp. The hard filter normalizes nominal input before certification; keep
    # that certified value and permit only the resulting tiny roundoff change.
    if (certificate['infeasible']
            or not np.allclose(certified, preferred, atol=1e-10, rtol=0.)):
        # The reserve is optional. A failed certificate must never replace a
        # proven hard-safe command or turn this optional preference terminal.
        stats.update(communication_guard_certificate_status='rejected',
                     communication_guard_certificate_delta=float(np.linalg.norm(certified - preferred)),
                     communication_guard_certificate_infeasible=bool(certificate['infeasible']))
        return result, stats
    nominal = np.asarray(velocity, float)[:2].copy()
    nominal *= min(1., max_speed / max(np.linalg.norm(nominal), 1e-12))
    stats.update(communication_guard_applied=bool(np.linalg.norm(certified - result) > 1e-9),
                 communication_guard_certificate_status='accepted',
                 intervention=float(np.linalg.norm(certified[:2] - nominal)),
                 max_constraint_residual=certificate['max_constraint_residual'],
                 solver_status=inner_stats['solver_status'],
                 candidate_count=inner_stats['candidate_count'])
    # The hard margin and original nominal outward speeds retain their feature
    # and reward definitions; only execution-dependent diagnostics change.
    return certified, stats


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (tuple, list, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _received_filter(self, velocity, observation, dt):
    """Preserve the frozen controller's received-geometry/link-selection rules."""
    peers = {}
    for neighbor in observation.get('neighbors', []):
        if neighbor['type'] == 'drone':
            did = (neighbor.get('broadcast') or {}).get('drone_id', f"robot_{neighbor['id']}")
            peers[did] = np.asarray(self.position) + np.asarray(neighbor['relative_position'])
    chain = getattr(self.navigator, 'chain', self._chain_state)
    grounds = None
    if self.communication_mode == 'backbone' and self.multi_vehicle:
        from drone.local_backbone import received_geometry_multi, relative_neighborhood
        peers, _, grounds = received_geometry_multi(self.position, observation)
        required = relative_neighborhood(self.position, {**peers, **grounds}, self.communication_range)
        ground = None
    elif self.communication_mode == 'backbone':
        from drone.local_backbone import received_geometry, relative_neighborhood
        peers, _, ground = received_geometry(self.position, observation)
        visible = dict(peers)
        if ground is not None:
            visible['__ground__'] = ground
        required = relative_neighborhood(self.position, visible, self.communication_range)
    elif self.communication_mode == 'chain' and chain is not None:
        required = chain.required
        task = chain.task()
        ground = task['position'] if task is not None and '__ground__' in chain.visible else None
    else:
        required = ('__ground__',)
        ground = self.navigator.ground_robot_position
    for neighbor in observation.get('neighbors', []):
        if neighbor['type'] in ('diff_drive', 'holonomic') and ground is not None:
            ground = np.asarray(self.position) + np.asarray(neighbor['relative_position'])
            break
    geometry = dict(required_links=tuple(sorted(required)), ground=ground, grounds=grounds,
        ground_max_speed=getattr(self, 'ground_max_speed', 0.),
        world_bounds=getattr(self.navigator, 'world_bounds', None), drone_id=self.drone_id)
    result, stats = guarded_filter(velocity, self.position, peers, dt,
        min_separation=self.navigator.min_separation, comm_range=self.communication_range,
        max_speed=self.max_speed, **geometry)
    # Retain only the latest action's geometry, not just a zero-nominal margin
    # probe. The episode persists this diagnostic when safety terminates it.
    self._last_cbf_diagnostic = _json_value(dict(owner=self.drone_id,
        computed_at=observation.get('time', observation.get('full_state', {}).get('time')),
        position=np.asarray(self.position).copy(), peers=peers, dt=dt,
        nominal_velocity=velocity, filtered_velocity=result, geometry=geometry,
        min_separation=self.navigator.min_separation, communication_range=self.communication_range,
        max_speed=self.max_speed, stats=stats, control_version=CONTROL_VERSION))
    return result, stats


def install_control_safety(experiment):
    for controller in experiment.controllers:
        controller._cbf_filter = MethodType(_received_filter, controller)


def failure_diagnostics(experiment):
    return [controller._last_cbf_diagnostic for controller in experiment.controllers
            if hasattr(controller, '_last_cbf_diagnostic')]
