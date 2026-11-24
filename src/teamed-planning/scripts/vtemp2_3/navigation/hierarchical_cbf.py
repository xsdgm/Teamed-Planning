"""Deterministic 2-D projection for the archived v9 CBF constraints.

The received geometry, barrier inequalities, speed disk and final 1e-7
acceptance tolerance are unchanged. Only the numerical projection changes:
enumerate the possible active boundaries of this two-dimensional convex set
instead of treating an unsuccessful SLSQP iterate as geometric infeasibility.
The historical 32-sided margin LP remains a conservative actor feature; its
sign is not used to decide whether the exact speed disk is feasible.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


PROJECTION_VERSION = 'vtemp2_exact_2d_cbf_v1'
ACCEPTANCE_TOLERANCE = 1e-7


def feasibility_margin(matrix, bound, speed, sides=32):
    """Preserve the archived conservative, normalized slack feature."""
    norms = np.linalg.norm(matrix, axis=1)
    live = norms > 1e-12
    if np.any(~live & (bound < 0)):
        return float('-inf')
    if not live.any():
        return None
    angles = np.arange(sides) * 2 * np.pi / sides
    polygon = np.c_[np.cos(angles), np.sin(angles)]
    rows = np.r_[np.c_[matrix[live] / norms[live, None], np.ones(live.sum())],
                 np.c_[polygon, np.zeros(sides)]]
    limits = np.r_[bound[live] / norms[live],
                   np.full(sides, speed * np.cos(np.pi / sides))]
    result = linprog([0., 0., -1.], A_ub=rows, b_ub=limits,
                     bounds=[(None, None)] * 3, method='highs')
    if result.status != 0:
        raise RuntimeError(f'CBF margin program failed: {result.message}')
    return float(-result.fun)


def project_velocity(desired, matrix, bound, speed):
    """Project onto ``matrix @ v <= bound, ||v|| <= speed``.

    A closest point in two dimensions is the desired point, a projection on
    one line/the circle, an intersection of two lines, or a line/circle
    intersection. Enumerating those cases also handles a feasible segment or
    singleton with no interior. Normalize only for constructing candidates;
    certify every result against the original inequalities and tolerance.
    Returns ``(None, diagnostics)`` when no admissible candidate exists.
    """
    desired = np.asarray(desired, dtype=float)
    matrix = np.asarray(matrix, dtype=float).reshape(-1, 2)
    bound = np.asarray(bound, dtype=float)
    if (desired.shape != (2,) or bound.shape != (len(matrix),)
            or not np.isfinite(desired).all() or not np.isfinite(matrix).all()
            or not np.isfinite(bound).all() or not np.isfinite(speed) or speed <= 0):
        raise ValueError('Projection requires finite 2-D constraints and a positive speed')

    def certified_candidate(v):
        v = np.asarray(v, dtype=float)
        if not np.isfinite(v).all():
            return None
        length = np.linalg.norm(v)
        if length > speed:
            # The legacy raw inequality tolerance must not enlarge the speed
            # disk: later nominal clipping would change a verified command.
            # Correct only roundoff in an analytically on-circle candidate,
            # then verify its original (unnormalized) halfspace inequalities.
            if length - speed > 8 * np.finfo(float).eps * max(1., speed):
                return None
            v = v * (np.nextafter(speed, 0.) / length)
        if np.linalg.norm(v) <= speed and np.all(matrix @ v <= bound + ACCEPTANCE_TOLERANCE):
            return v
        return None

    diagnostics = dict(solver=PROJECTION_VERSION, projection_solver=PROJECTION_VERSION,
                       solver_status='nominal_feasible',
                       candidate_count=1, acceptance_tolerance=ACCEPTANCE_TOLERANCE)
    initial = certified_candidate(desired)
    if initial is not None:
        return initial.copy(), diagnostics
    candidates = [np.zeros(2)]
    norm = np.linalg.norm(desired)
    if norm:
        candidates.append(desired * (speed / norm))

    # Long-double construction avoids cancellation in thin wedges; output is
    # still float64 and must pass the unchanged raw-coordinate acceptance test.
    wide = matrix.astype(np.longdouble)
    lengths = np.sqrt(np.sum(wide * wide, axis=1))
    live = lengths > 0
    normals = wide[live] / lengths[live, None]
    offsets = bound[live].astype(np.longdouble) / lengths[live]
    target = desired.astype(np.longdouble)
    radius = np.longdouble(speed)
    for normal, offset in zip(normals, offsets):
        foot = normal * offset
        candidates.append(target - normal * (np.dot(normal, target) - offset))
        if abs(offset) <= radius:
            tangent = np.array([-normal[1], normal[0]], dtype=np.longdouble)
            distance = np.sqrt(max(np.longdouble(0), radius * radius - offset * offset))
            candidates.extend((foot + distance * tangent, foot - distance * tangent))
    for i in range(len(normals)):
        for j in range(i):
            a, b = normals[i], normals[j]
            determinant = a[0] * b[1] - a[1] * b[0]
            if determinant != 0:
                candidates.append(np.array([
                    offsets[i] * b[1] - a[1] * offsets[j],
                    a[0] * offsets[j] - offsets[i] * b[0]]) / determinant)
    admissible = []
    for candidate in candidates:
        candidate = certified_candidate(candidate)
        if candidate is not None:
            admissible.append(candidate)
    diagnostics['candidate_count'] = len(candidates)
    diagnostics['solver_status'] = 'projected' if admissible else 'infeasible'
    if not admissible:
        return None, diagnostics
    result = min(admissible, key=lambda v: float(np.dot(v - desired, v - desired)))
    return result.copy(), diagnostics


class CBFSafetyFilter:
    """Drop-in v9 filter with the same physical constraints and diagnostics."""

    def __init__(self, min_separation=15., comm_range=95., max_speed=6., margin=.25, gain=1.):
        values = (min_separation, comm_range, max_speed, margin, gain)
        if (not np.isfinite(values).all() or min_separation <= 0
                or comm_range <= min_separation + 2 * margin or max_speed <= 0 or gain <= 0):
            raise ValueError('Invalid CBF distances, speed or gain')
        self.min_separation = float(min_separation)
        self.comm_range = float(comm_range)
        self.max_speed = float(max_speed)
        self.margin = float(margin)
        self.gain = float(gain)

    def filter(self, nominal, position, peers, dt, required_links=(), ground=None,
               world_bounds=None, drone_id='', ground_max_speed=0., grounds=None):
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError('CBF requires a positive finite control step')
        if not np.isfinite(ground_max_speed) or ground_max_speed < 0:
            raise ValueError('Ground speed bound must be finite and nonnegative')
        p, nominal = np.asarray(position, float), np.asarray(nominal, float)
        if (p.shape != (3,) or nominal.shape != (3,)
                or not np.isfinite(p).all() or not np.isfinite(nominal).all()):
            raise ValueError('CBF requires finite 3D position and nominal velocity')
        if abs(nominal[2]) > 1e-8:
            raise ValueError('This CBF implementation requires fixed flight height')
        desired = nominal[:2].copy()
        speed = self.max_speed
        desired *= min(1., speed / max(np.linalg.norm(desired), 1e-12))
        rows, limits = [], []
        link_distances, link_outward_speeds = {}, {}
        minimum = np.inf
        required = set(required_links)

        def upper(normal, limit):
            rows.append(np.asarray(normal, float))
            limits.append(float(limit))

        for did, other in peers.items():
            r = p - np.asarray(other, float)
            distance = float(np.linalg.norm(r))
            minimum = min(minimum, distance)
            horizontal = float(np.linalg.norm(r[:2]))
            clearance = np.sqrt(max(0., (self.min_separation + self.margin)**2 - r[2]**2))
            normal = (r[:2] / horizontal if horizontal > 1e-9 else
                      np.array([1. if str(drone_id) > str(did) else -1., 0.]))
            if clearance > 0:
                lower = min(self.gain, 1.) * (clearance - horizontal) / (2 * dt)
                upper(-normal, -lower)
            if did in required:
                link_distances[did] = distance
                link_outward_speeds[did] = float(np.dot(r[:2], desired) / max(distance, 1e-12))
                h = (self.comm_range - self.margin)**2 - distance**2
                upper(r[:2], (min(self.gain, 1.) * h / dt - 4 * speed**2 * dt) / 4)
        vehicles = {'__ground__': ground} if '__ground__' in required and ground is not None else {}
        vehicles.update({k: q for k, q in (grounds or {}).items() if k in required})
        for key, vehicle in vehicles.items():
            r = p - np.r_[np.asarray(vehicle, float)[:2], 0.]
            distance = float(np.linalg.norm(r))
            link_distances[key] = distance
            link_outward_speeds[key] = float(np.dot(r[:2], desired) / max(distance, 1e-12))
            h = (self.comm_range - self.margin)**2 - float(np.dot(r, r))
            upper(r[:2], (min(self.gain, 1.) * h / dt - (speed + ground_max_speed)**2 * dt) / 2
                  - np.linalg.norm(r[:2]) * ground_max_speed)
        if world_bounds is not None:
            xmin, ymin, xmax, ymax = world_bounds
            upper([-1, 0], (p[0] - xmin) / dt)
            upper([1, 0], (xmax - p[0]) / dt)
            upper([0, -1], (p[1] - ymin) / dt)
            upper([0, 1], (ymax - p[1]) / dt)
        matrix = np.asarray(rows, float).reshape(-1, 2)
        bound = np.asarray(limits, float)
        result, diagnostic = project_velocity(desired, matrix, bound, speed)
        infeasible = result is None
        # Preserve the rejected-command interface. Experiment must reject this
        # result before physics; zero is never claimed to be a safe fallback.
        if infeasible:
            result = np.zeros(2)
        return np.r_[result, 0.], dict(infeasible=infeasible,
            feasibility_margin=feasibility_margin(matrix, bound, speed),
            intervention=float(np.linalg.norm(result - desired)),
            min_received_separation=None if not np.isfinite(minimum) else minimum,
            max_constraint_residual=float(max(0., np.max(matrix @ result - bound))) if len(bound) else 0.,
            received_peers=len(peers), required_link_distances_m=link_distances,
            required_link_nominal_outward_m_s=link_outward_speeds,
            missing_links=sorted(required - set(peers)
                - ({'__ground__'} if ground is not None else set()) - set(grounds or ())),
            **diagnostic)
