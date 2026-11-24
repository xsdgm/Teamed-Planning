"""Local mapping views and world-frame temporal occupancy evidence.

Fine mapping and policy views must describe the same centred footprint.  This module deliberately
has no imports from ``drone`` so an evaluation runner can load it by path and
wrap a frozen controller without importing mutable controller dependencies.
"""
from typing import NamedTuple, Tuple

import numpy as np


_MAPPING_FIELDS = frozenset(("mapping_occupancy_map", "mapping_resolution",
                             "mapping_known_mask", "mapping_axes"))


class LocalMapping(NamedTuple):
    occupancy: np.ndarray  # mapper convention: row X, column Y
    known: np.ndarray
    resolution: float
    footprint_xy: tuple


def _resolution(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive finite scalar") from None
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return result


def _axes(value, name):
    if value not in ("xy", "yx"):
        raise ValueError(f"{name} must be 'xy' or 'yx'")
    return value


def _binary_grid(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or not array.size:
        raise ValueError(f"{name} must be a nonempty 2D grid")
    try:
        valid = np.isfinite(array).all() and np.isin(array, (0, 1)).all()
    except TypeError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must contain finite binary values")
    return array


def validate_local_mapping(observation, policy_resolution=None):
    """Return an independent XY mapping view, or None for legacy observations.

    The original occupancy array and its metadata are never modified. Unknown
    mapping cells remain distinguishable from observed obstacles, and are also
    marked blocked in the returned binary occupancy.
    """
    present = _MAPPING_FIELDS.intersection(observation)
    if not present:
        return None
    if present != _MAPPING_FIELDS:
        raise ValueError("Incomplete local mapping observation; missing " +
                         ", ".join(sorted(_MAPPING_FIELDS - present)))
    occupancy = _binary_grid(observation["mapping_occupancy_map"], "mapping_occupancy_map")
    known = _binary_grid(observation["mapping_known_mask"], "mapping_known_mask")
    if occupancy.shape != known.shape:
        raise ValueError("mapping_known_mask must match mapping_occupancy_map shape")
    resolution = _resolution(observation["mapping_resolution"], "mapping_resolution")
    axes = _axes(observation["mapping_axes"], "mapping_axes")
    policy = np.asarray(observation.get("occupancy_map"))
    if policy.ndim != 2 or not policy.size:
        raise ValueError("Local mapping requires the original 2D occupancy_map")
    policy_axes = _axes(observation.get("map_axes", "yx"), "map_axes")
    policy_resolution = _resolution(observation.get("map_resolution", policy_resolution),
                                    "map_resolution")
    footprint = np.asarray(occupancy.shape if axes == "xy" else occupancy.shape[::-1]) * resolution
    policy_footprint = np.asarray(policy.shape if policy_axes == "xy" else policy.shape[::-1]) * policy_resolution
    if not np.allclose(footprint, policy_footprint, rtol=0., atol=1e-8):
        raise ValueError("Mapping and policy grids must cover the same centred metric footprint")
    if axes == "yx":
        occupancy, known = occupancy.T, known.T
    known = known.astype(bool, copy=True)
    occupancy = occupancy.astype(np.uint8, copy=True)
    occupancy[~known] = 1
    return LocalMapping(occupancy, known, resolution, tuple(float(x) for x in footprint))


def prepare_local_mapping(controller, observation):
    """Validate one frame and configure a new mapper before any graph is built.

    A controller cannot mix graph histories made with different resolutions or
    footprints, or silently fall back to the coarse sensor after opting in.
    ``controller.resolution`` retains the original policy/sensor resolution;
    ``local_mapping_resolution`` records the independent mapper resolution.
    """
    mapping = validate_local_mapping(observation, controller.resolution)
    contract = getattr(controller, "_local_mapping_contract", None)
    if mapping is None:
        if contract is not None:
            raise ValueError("Mapping observation is missing after local mapping was enabled")
        return None
    signature = (mapping.resolution, mapping.footprint_xy)
    if contract is not None:
        if signature != contract:
            raise ValueError("Mapping resolution or footprint cannot change within a controller run")
        return mapping
    if controller.current_graph is not None:
        raise ValueError("Local mapping must be enabled before constructing a graph")
    old = controller.graph_builder
    # Reconstruct rather than assigning resolution: the builder caches several
    # pixel-space parameters and owns its own semantic processor.
    controller.graph_builder = type(old)(
        drone_id=old.drone_id, resolution=mapping.resolution,
        node_spacing=old.node_spacing, min_clearance=old.min_clearance,
        max_edge_distance=old.max_edge_distance, k_neighbors=old.k_neighbors,
        world_bounds=old.world_bounds)
    controller.semantic_processor.resolution = mapping.resolution
    controller.local_mapping_resolution = mapping.resolution
    controller._local_mapping_contract = signature
    return mapping


def mapping_semantic_map(mapping, converter):
    """Convert the mapping view, retaining UNKNOWN (TerrainType value zero)."""
    semantic = np.asarray(converter(mapping.occupancy)).copy()
    if semantic.shape != mapping.occupancy.shape:
        raise ValueError("Semantic converter changed the local mapping grid shape")
    semantic[~mapping.known] = 0  # TerrainType.UNKNOWN; never traversable.
    return semantic


def with_local_mapping(base_controller):
    """Adapt a frozen controller class without changing its source or policy.

    Its parsed occupancy still feeds the original actor. Only the input to the
    semantic-map method is replaced, using this frame's validated mapping view.
    """
    if getattr(base_controller, "supports_local_mapping", False):
        return base_controller

    class LocalMappingController(base_controller):
        supports_local_mapping = True

        def _parse_simulator_observation(self, observation):
            self._current_local_mapping = prepare_local_mapping(self, observation)
            return super()._parse_simulator_observation(observation)

        def _generate_semantic_map(self, occupancy_map):
            mapping = getattr(self, "_current_local_mapping", None)
            converter = super()._generate_semantic_map
            return (converter(occupancy_map) if mapping is None else
                    mapping_semantic_map(mapping, converter))

    return LocalMappingController


class EvidenceGrid:
    """EvidenceGrid — world-frame temporal vote accumulation for semantic segmentation.

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
