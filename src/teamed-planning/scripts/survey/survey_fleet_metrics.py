"""Exact V9 reward-region geometry with bounded temporary arrays for km runs."""
from types import MethodType

import numpy as np


def task_mask(ledger, start, goal, car_radius=18., goal_radius=18., band=12.):
    """Evaluate the archived circle/corridor tests inside their common bounds."""
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    delta = goal - start
    # The original ledger's world coordinates are cell centres. The extra
    # cell protects the crop at a floating-point boundary of a tangent circle.
    pad = max(car_radius, goal_radius, band, 0.) + ledger.resolution
    low = np.minimum(start, goal) - pad
    high = np.maximum(start, goal) + pad
    xs, ys = ledger.xy[0, :, 0], ledger.xy[:, 0, 1]
    x0, x1 = np.searchsorted(xs, (low[0], high[0]))
    y0, y1 = np.searchsorted(ys, (low[1], high[1]))
    mask = np.zeros(ledger.seen.shape, dtype=bool)
    if x1 == x0 or y1 == y0:
        return mask
    x, y = xs[None, x0:x1], ys[y0:y1, None]
    dx, dy = x - start[0], y - start[1]
    gx, gy = x - goal[0], y - goal[1]
    t = np.clip((dx*delta[0] + dy*delta[1]) / max(np.dot(delta, delta), 1e-8), 0., 1.)
    cx, cy = dx - t*delta[0], dy - t*delta[1]
    mask[y0:y1, x0:x1] = ((np.sqrt(dx*dx + dy*dy) <= car_radius)
                           | (np.sqrt(gx*gx + gy*gy) <= goal_radius)
                           | (np.sqrt(cx*cx + cy*cy) <= band))
    return mask


def use_bounded_task_mask(experiment):
    """Patch only this experiment's reward ledger; the frozen source remains immutable."""
    ledger = experiment.ledger
    ledger.task_mask = MethodType(task_mask, ledger)
