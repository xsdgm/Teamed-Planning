"""
EvidenceGrid — world-frame temporal vote accumulation for semantic segmentation.

Instead of using a single noisy frame, each new observation adds weighted evidence
to a persistent 2D grid in world coordinates.  When the graph builder requests an
occupancy map it reads the accumulated evidence, which smooths out frame-to-frame
jitter while still converging quickly on real obstacles.

Weighting rationale (asymmetric):
  obstacle observation  → +OBSTACLE_WEIGHT  (e.g. +1.0)
  free    observation   → −FREE_WEIGHT      (e.g. −0.5)

A cell needs twice as many free observations as obstacle observations to be cleared.
This is conservative and intentional: for ground-robot safety, a false-positive
(labelling navigable ground as obstacle) is much less dangerous than a false-negative
(labelling a tree as free).

Evidence is clamped to [−MAX_EVIDENCE, +MAX_EVIDENCE] to prevent any single cell
from becoming so confident it never updates.

Output: score > 0  → obstacle  |  score ≤ 0  → free (or unobserved)
"""

import numpy as np
from typing import Tuple, Optional


class EvidenceGrid:
    # ── Tunable constants ─────────────────────────────────────────────────────
    OBSTACLE_WEIGHT = 1.0     # score added per obstacle observation
    FREE_WEIGHT     = 0.5     # score subtracted per free observation
    MAX_EVIDENCE    = 5.0     # clamp range; ~5 strong obstacle obs = certainty
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self,
                 resolution: float = 0.5,
                 world_size: float = 200.0,
                 origin: Tuple[float, float] = (-100.0, -100.0)):
        """
        Args:
            resolution:  cell size in metres (should match the occ map resolution)
            world_size:  side length of the square grid in metres
            origin:      (x_min, y_min) world-frame corner of the grid
        """
        self.resolution = resolution
        self.origin_x, self.origin_y = origin
        n_cells = max(1, int(round(world_size / resolution)))
        # Float32 evidence accumulator — initialised to 0 (neutral / unobserved)
        self._evidence = np.zeros((n_cells, n_cells), dtype=np.float32)
        self._n = n_cells

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self,
               occ: np.ndarray,
               coverage_bounds: Tuple[float, float, float, float]) -> None:
        """Project occ map into the world-frame grid and accumulate evidence.

        Fully vectorised — no Python loops.  At a typical 58×104 occ map and
        10 Hz call rate this costs ~0.4 ms/s, which is negligible compared to
        ONNX inference (~150–300 ms/call).

        Args:
            occ:              uint8 array shape (H, W), 0=free, 1=obstacle.
                              Must already be yaw-corrected and axis-aligned with
                              world frame (rows→X, cols→Y).
            coverage_bounds:  (min_x, min_y, max_x, max_y) in world frame.
        """
        min_x, min_y, max_x, max_y = coverage_bounds
        h, w = occ.shape
        res = self.resolution

        # Grid-cell indices for rows (→X) and cols (→Y)
        gi = ((min_x + (np.arange(h, dtype=np.float32) + 0.5) * res
               - self.origin_x) / res).astype(np.int32)   # (H,)
        gj = ((min_y + (np.arange(w, dtype=np.float32) + 0.5) * res
               - self.origin_y) / res).astype(np.int32)   # (W,)

        ri_valid = (gi >= 0) & (gi < self._n)
        cj_valid = (gj >= 0) & (gj < self._n)

        if not (np.any(ri_valid) and np.any(cj_valid)):
            return   # footprint fully outside the grid (shouldn't happen normally)

        gi_v = gi[ri_valid]   # contiguous valid row indices
        gj_v = gj[cj_valid]  # contiguous valid col indices

        # Per-pixel weights for the valid sub-region (H', W')
        weights = np.where(occ[np.ix_(ri_valid, cj_valid)] == 1,
                           self.OBSTACLE_WEIGHT,
                           -self.FREE_WEIGHT).astype(np.float32)

        # Single slice assignment — O(H'×W') numpy BLAS, no Python loop
        self._evidence[gi_v[0]:gi_v[-1] + 1,
                       gj_v[0]:gj_v[-1] + 1] += weights

        np.clip(self._evidence, -self.MAX_EVIDENCE, self.MAX_EVIDENCE,
                out=self._evidence)

    def get_occ(self,
                coverage_bounds: Tuple[float, float, float, float],
                shape: Tuple[int, int],
                unobserved_as_free: bool = True) -> np.ndarray:
        """Read the accumulated evidence map for a given footprint.

        Args:
            coverage_bounds:    (min_x, min_y, max_x, max_y) in world frame.
            shape:              (H, W) of the output occ array.
            unobserved_as_free: if True, cells with zero evidence (never seen)
                                are returned as free; if False, as obstacle.
                                Free is optimistic but prevents the drone from
                                refusing to fly into unexplored territory.
        Returns:
            uint8 array shape (H, W), 0=free, 1=obstacle.
        """
        min_x, min_y, max_x, max_y = coverage_bounds
        h, w = shape
        res = self.resolution

        row_idx = np.arange(h, dtype=np.float32)
        col_idx = np.arange(w, dtype=np.float32)
        world_x = min_x + (row_idx + 0.5) * res
        world_y = min_y + (col_idx + 0.5) * res

        gi = ((world_x - self.origin_x) / res).astype(np.int32)   # (H,)
        gj = ((world_y - self.origin_y) / res).astype(np.int32)   # (W,)

        ri_valid = (gi >= 0) & (gi < self._n)
        cj_valid = (gj >= 0) & (gj < self._n)

        default = np.uint8(0) if unobserved_as_free else np.uint8(1)
        result = np.full((h, w), default, dtype=np.uint8)

        gi_v = gi[ri_valid]
        gj_v = gj[cj_valid]
        if gi_v.size and gj_v.size:
            scores = self._evidence[gi_v[0]:gi_v[-1] + 1,
                                    gj_v[0]:gj_v[-1] + 1]
            result[np.ix_(ri_valid, cj_valid)] = (scores > 0).astype(np.uint8)

        return result

    def reset(self) -> None:
        """Clear all accumulated evidence (call at mission start)."""
        self._evidence[:] = 0.0

    # ── Debug helpers ─────────────────────────────────────────────────────────

    def coverage_fraction(self) -> float:
        """Fraction of grid cells that have received at least one observation."""
        return float(np.count_nonzero(self._evidence)) / self._evidence.size

    def obstacle_fraction(self) -> float:
        """Fraction of observed cells currently classified as obstacle."""
        observed = self._evidence != 0
        if not np.any(observed):
            return 0.0
        return float(np.sum(self._evidence[observed] > 0)) / np.count_nonzero(observed)
