"""Evaluation controls; no policy, reward, or safety-filter changes.

Random variates are keyed by physical participant and simulation time, so
adding an aircraft cannot consume another aircraft's action/channel noise.
"""
from contextlib import contextmanager
import hashlib
import json

import numpy as np


def event_seed(seed, *event):
    payload = json.dumps([int(seed), *event], separators=(',', ':'), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'little') & ((1 << 63) - 1)


def fleet_positions(start, count, world_size=1000.):
    """Keep the historical first three positions when appending UAV4.

    Reject out-of-world layouts instead of silently moving the baseline fleet.
    This experiment deliberately supports only the requested 3/4 comparison.
    """
    if count not in (3, 4):
        raise ValueError('Controlled comparison supports 3 or 4 aircraft')
    offsets = np.array([(9., -18.), (9., 0.), (9., 18.), (27., 18.)])[:count]
    points = np.asarray(start, float) + offsets
    if not np.isfinite(points).all() or np.any(points < 0.) or np.any(points > world_size):
        raise ValueError('Common initial layout does not fit the world')
    return points


@contextmanager
def action_stream(seed, aircraft, frame):
    import torch
    # The checkpoint runs on CPU. Restore the caller's RNG even on exceptions.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(event_seed(seed, 'action', str(aircraft), int(frame)))
        yield


def coupled_relay_class(base):
    class CoupledRelay(base):
        """Same channel distribution, keyed by sender/receiver/frame.

        The existing channel calls due immediately before each transmit.
        Explicitly check this contract instead of guessing packet provenance.
        """
        def __init__(self, seed, *args, **kwargs):
            super().__init__(seed, *args, **kwargs)
            self.control_seed = int(seed)
            self.send_context = None
            self.sent_events = set()

        def due(self, sender, frame):
            if sender not in self.phase:
                rng = np.random.default_rng(event_seed(self.control_seed, 'phase', str(sender)))
                self.phase[sender] = int(rng.integers(self.period))
            due = (frame + self.phase[sender]) % self.period == 0
            self.send_context = (sender, frame) if due else None
            return due

        def transmit(self, envelope, receivers, frame):
            if self.send_context is None or self.send_context[1] != frame:
                raise RuntimeError('Relay transmission has no matching due event')
            sender, _ = self.send_context
            for receiver in receivers:
                event = (sender, receiver, frame)
                if event in self.sent_events:
                    raise RuntimeError('Duplicate directed transmission in one frame')
                self.sent_events.add(event)
                rng = np.random.default_rng(event_seed(
                    self.control_seed, 'packet', str(sender), str(receiver), int(frame)))
                self.stats['sent'] += 1
                if rng.random() < self.loss:
                    self.stats['lost'] += 1
                    continue
                arrival = frame + 1 + int(rng.integers(self.delay_jitter + 1))
                self.order += 1
                packet = [dict(item, state=dict(item['state'])) for item in envelope]
                # Stable tie ordering: an added sender must not reorder common
                # senders' same-time packets at a receiver.
                order = (int(frame), str(sender), str(receiver))
                self.queues.setdefault(receiver, []).append((arrival, order, packet))
            self.send_context = None
    return CoupledRelay


def swept_clearance(first, second, circles):
    """Evaluator-only physical circle geometry, never fed to a controller."""
    circles = np.asarray(circles, float).reshape(-1, 3)
    if not len(circles):
        return float('inf')
    first, second = np.asarray(first, float), np.asarray(second, float)
    delta = second - first
    length2 = float(delta @ delta)
    t = (np.clip((circles[:, :2] - first) @ delta / length2, 0., 1.)
         if length2 else np.zeros(len(circles)))
    return float(np.min(np.linalg.norm(circles[:, :2] - first - t[:, None]*delta, axis=1)
                        - circles[:, 2]))


def minimum_constraint_speed(matrix, bound):
    """Exact 2D half-plane minimum norm, independent of SLSQP/the LP polygon.

    A minimum is zero, a projection on one boundary, or a two-line vertex.
    None means the half-planes themselves have no common point.
    """
    from itertools import combinations
    matrix, bound = np.asarray(matrix, float), np.asarray(bound, float)
    candidates = [np.zeros(2)]
    for row, limit in zip(matrix, bound):
        squared = float(row @ row)
        if squared > 1e-16:
            candidates.append(row * limit / squared)
    for i, j in combinations(range(len(bound)), 2):
        rows = matrix[[i, j]]
        if abs(np.linalg.det(rows)) > 1e-12:
            candidates.append(np.linalg.solve(rows, bound[[i, j]]))
    feasible = [v for v in candidates if np.all(matrix @ v <= bound + 1e-7)]
    return min((float(np.linalg.norm(v)) for v in feasible), default=None)


def loop_statistics(trajectory):
    points = np.asarray([row['positions'][-1] for row in trajectory], float)
    if len(points) < 151:
        return dict(repeating=False)
    recent = points[-151:]
    travel = float(np.linalg.norm(np.diff(recent, axis=0), axis=1).sum())
    repeated = [lag for lag in range(15, 61)
                if np.max(np.linalg.norm(recent[lag:] - recent[:-lag], axis=1)) < .05]
    return dict(repeating=bool(repeated and travel > 8.), periods=repeated,
                recent_travel_m=travel, extent_m=np.ptp(recent, axis=0).tolist())
