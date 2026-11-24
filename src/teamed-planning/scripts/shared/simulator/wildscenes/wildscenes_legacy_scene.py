"""Export measured WildScenes obstacles into the original primitive 2D simulator.

This deliberately makes unmeasured space free. It is a simplified simulation
derived from recorded obstacle geometry, NOT a reconstruction of known drivable
terrain. Source occupied cell squares are retained exactly as merged rectangles.
The native simulator's local rasterizer, input shapes and dynamics are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
from scipy.spatial import cKDTree


PROTOCOL = "wildscenes-derived-legacy-2d-v1"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def rectangles_from_mask(mask, resolution_m, origin_xy=(0., 0.)):
    """Merge identical horizontal runs on successive rows, without filling gaps.

    Input axes are [y,x]; output is native [center_x,center_y,width,height].
    No dilation, size filtering, circle fitting, or coordinate rescaling occurs.
    """
    mask = np.asarray(mask)
    resolution_m = float(resolution_m)
    origin = np.asarray(origin_xy, dtype=float)
    if (mask.ndim != 2 or not mask.size or not np.isin(mask, [0, 1]).all()
            or not np.isfinite(resolution_m) or resolution_m <= 0
            or origin.shape != (2,) or not np.isfinite(origin).all()):
        raise ValueError("Expected a binary [y,x] mask, positive resolution and XY origin")
    active, bounds = {}, []
    for y, row in enumerate(mask.astype(bool, copy=False)):
        transitions = np.diff(np.r_[False, row, False].astype(np.int8))
        runs = list(zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)))
        current = set(runs)
        for run in active.keys() - current:
            bounds.append((run[0], active[run], run[1], y))
        active = {run: active.get(run, y) for run in runs}
    for run, y0 in active.items():
        bounds.append((run[0], y0, run[1], len(mask)))
    bounds.sort(key=lambda b: (b[1], b[0], b[3], b[2]))
    return [[float(origin[0] + (x0 + x1) * resolution_m / 2),
             float(origin[1] + (y0 + y1) * resolution_m / 2),
             float((x1 - x0) * resolution_m), float((y1 - y0) * resolution_m)]
            for x0, y0, x1, y1 in bounds]


def export_scene(map_path, output_path):
    """Write a self-contained, labelled scene; never overwrite an earlier export."""
    map_path, output_path = Path(map_path).resolve(), Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    with np.load(map_path, allow_pickle=False) as data:
        occupied, known = data["occupied"], data["known"]
        resolution = float(data["resolution_m"])
        origin = np.asarray(data["origin_xy"], dtype=float)
        native_origin = data["native_origin_xy"].tolist() if "native_origin_xy" in data else None
    if (occupied.ndim != 2 or occupied.shape != known.shape or not occupied.size
            or not np.isin(known, [0, 1]).all()):
        raise ValueError("Expected matching occupied/known binary rasters")
    if origin.shape != (2,) or not np.array_equal(origin, [0., 0.]):
        raise ValueError("The original simulator requires a zero XY origin")
    if occupied.shape[0] != occupied.shape[1]:
        raise ValueError("The original experiment requires a square world")
    rectangles = rectangles_from_mask(occupied, resolution, origin)
    metadata_path = map_path.with_name("map_metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    source_hash = digest(map_path)
    if metadata.get("map_sha256", source_hash) != source_hash:
        raise ValueError("Source map differs from its map_metadata.json hash")
    document = dict(
        protocol=PROTOCOL, simplified_scene=True,
        title="WildScenes-derived simplified 2D obstacle layout",
        unknown_space="free_by_simulation_assumption",
        world_size_m=float(occupied.shape[0] * resolution), origin_xy=origin.tolist(),
        coordinate_scale=1.0,
        source=dict(map_path=str(map_path), map_sha256=source_hash,
                    resolution_m=resolution, shape_yx=list(occupied.shape),
                    native_origin_xy=native_origin, metadata=metadata,
                    metadata_sha256=digest(metadata_path) if metadata_path.exists() else None),
        geometry=dict(type="axis_aligned_rectangles", rectangles=rectangles,
                      exact_source_occupied=True),
        statistics=dict(source_occupied_cells=int(np.count_nonzero(occupied)),
                        source_known_cells=int(np.count_nonzero(known)),
                        source_unknown_cells=int(np.count_nonzero(~known.astype(bool))),
                        obstacle_area_m2=float(np.count_nonzero(occupied) * resolution ** 2),
                        rectangle_count=len(rectangles), added_occupied_cells=0,
                        removed_occupied_cells=0),
        assumptions=[
            "Unmeasured regions are simulation free space, not verified real traversable ground.",
            "Only source occupied cells become obstacles; source unknown cells do not become obstacles.",
            "Occupied includes the source map's low vegetation, solid obstacles and mud/water hazards; rectangles are not individual tree detections.",
            "Obstacle squares and metre scale are exact relative to the source 2D raster, not to raw 3D surfaces.",
            "The source height/ground-support filters already excluded unsupported points and high canopy; conversion cannot recover them.",
            "Native ideal local perception and planar dynamics are retained, including their discretization and endpoint collision behavior.",
            "Acquisition paths and reference routes are not policy inputs; inherited task lengths are not new-scene shortest paths.",
            "Success in this simplified scenario is not evidence of navigation success in the original conservative WildScenes map or a real forest.",
        ],
    )
    # Validate the exported schema before publishing it.
    LegacyObstacleScene(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        json.dump(document, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    return document


class LegacyObstacleScene:
    """Primitive geometry and independent swept-clearance audit, never policy data."""

    def __init__(self, document):
        if (document.get("protocol") != PROTOCOL or document.get("simplified_scene") is not True
                or document.get("unknown_space") != "free_by_simulation_assumption"):
            raise ValueError("Expected an explicitly labelled WildScenes simplified scene")
        self.metadata = document
        self.world_size_m = float(document["world_size_m"])
        self.origin_xy = np.asarray(document["origin_xy"], dtype=float)
        if (not np.isfinite(self.world_size_m) or self.world_size_m <= 0
                or self.origin_xy.shape != (2,) or not np.array_equal(self.origin_xy, [0., 0.])):
            raise ValueError("Expected a finite positive square world at zero origin")
        if document["geometry"]["type"] != "axis_aligned_rectangles":
            raise ValueError("Expected axis aligned rectangle geometry")
        raw = document["geometry"]["rectangles"]
        self.rectangles = np.asarray(raw, dtype=float)
        if self.rectangles.size == 0:
            self.rectangles = np.empty((0, 4), dtype=float)
        if (self.rectangles.ndim != 2 or self.rectangles.shape[1] != 4
                or not np.isfinite(self.rectangles).all() or (self.rectangles[:, 2:] <= 0).any()):
            raise ValueError("Invalid rectangle coordinates or sizes")
        self.centers = self.rectangles[:, :2].copy()
        self.lower = self.centers - self.rectangles[:, 2:] / 2
        self.upper = self.centers + self.rectangles[:, 2:] / 2
        if (self.lower < -1e-9).any() or (self.upper > self.world_size_m + 1e-9).any():
            raise ValueError("Rectangle lies outside the world")
        self.half_diagonal = np.linalg.norm(self.rectangles[:, 2:] / 2, axis=1)
        self.native_radius = np.max(self.rectangles[:, 2:], axis=1) / 2
        self.maximum_radius = float(self.half_diagonal.max()) if len(self.rectangles) else 0.
        self.tree = cKDTree(self.centers)
        for array in (self.rectangles, self.origin_xy, self.centers, self.lower,
                      self.upper, self.half_diagonal, self.native_radius):
            array.flags.writeable = False

    @classmethod
    def from_json(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def nearby_indices(self, position, reach):
        """Conservative circle broad phase in stable native obstacle order."""
        xy = np.asarray(position, dtype=float)[:2]
        ids = np.asarray(self.tree.query_ball_point(xy, float(reach) + self.maximum_radius), dtype=int)
        if not len(ids):
            return ids
        distance = np.linalg.norm(self.centers[ids] - xy, axis=1)
        return np.sort(ids[distance <= float(reach) + self.half_diagonal[ids] + 1e-10])

    def known_at(self, points):
        xy = np.asarray(points, dtype=float)[..., :2]
        return np.isfinite(xy).all(axis=-1) & (xy >= 0).all(axis=-1) & (xy < self.world_size_m).all(axis=-1)

    def truth(self, points):
        xy = np.asarray(points, dtype=float)
        if xy.ndim < 1 or xy.shape[-1] < 2:
            raise ValueError("Points need an XY final dimension")
        xy = xy[..., :2]
        result = np.asarray(~self.known_at(xy)).copy()
        flat = result.reshape(-1)
        for i, point in enumerate(xy.reshape(-1, 2)):
            if flat[i]:
                continue
            ids = self.nearby_indices(point, 0.)
            flat[i] = bool(((point >= self.lower[ids]) & (point <= self.upper[ids])).all(axis=1).any())
        return result

    def segment_clear(self, start, end, clearance=0.):
        a, b = np.asarray(start, float)[:2], np.asarray(end, float)[:2]
        r = float(clearance)
        if not np.isfinite(r) or r < 0:
            raise ValueError("Clearance must be finite and nonnegative")
        if (a.shape != (2,) or b.shape != (2,) or not np.isfinite(a).all() or not np.isfinite(b).all()
                or (np.minimum(a, b) - r < 0).any()
                or (np.maximum(a, b) + r >= self.world_size_m).any()):
            return False
        delta = b - a
        ids = self.nearby_indices((a + b) / 2, np.linalg.norm(delta) / 2 + r)
        if not len(ids):
            return True
        lo, hi = self.lower[ids], self.upper[ids]
        tmin, tmax = np.zeros(len(ids)), np.ones(len(ids))
        intersects = np.ones(len(ids), dtype=bool)
        for axis in (0, 1):
            if abs(delta[axis]) < 1e-15:
                intersects &= (a[axis] >= lo[:, axis]) & (a[axis] <= hi[:, axis])
            else:
                t0, t1 = (lo[:, axis] - a[axis]) / delta[axis], (hi[:, axis] - a[axis]) / delta[axis]
                tmin = np.maximum(tmin, np.minimum(t0, t1))
                tmax = np.minimum(tmax, np.maximum(t0, t1))
        if (intersects & (tmin <= tmax)).any():
            return False
        distance2 = np.minimum(
            np.sum(np.maximum(np.maximum(lo - a, a - hi), 0.) ** 2, axis=1),
            np.sum(np.maximum(np.maximum(lo - b, b - hi), 0.) ** 2, axis=1))
        length2 = float(delta @ delta)
        if length2 > 0:
            for corner in (lo, hi, np.column_stack((lo[:, 0], hi[:, 1])),
                           np.column_stack((hi[:, 0], lo[:, 1]))):
                t = np.clip(((corner - a) @ delta) / length2, 0., 1.)
                distance2 = np.minimum(distance2, np.sum((corner - a - t[:, None] * delta) ** 2, axis=1))
        return not bool((distance2 <= r * r + 1e-12).any())

    def point_clear(self, point, clearance=0.):
        return self.segment_clear(point, point, clearance)


def install_legacy_environment(env, scene):
    """Load rectangles; accelerate only broad phases, retaining native sensing.

    Native endpoint collision handling still performs rollback. Swept attempted
    collisions are also recorded for evaluation, without changing that behavior.
    No high resolution mapping fields or source known-space masks are supplied.
    """
    if getattr(env, "_legacy_scene_installed", False) or env.obstacles:
        raise ValueError("Install once on an empty environment before reset")
    if not np.allclose(env.config.world_size, [scene.world_size_m] * 2, rtol=0., atol=1e-9):
        raise ValueError("Environment and scene bounds differ")
    for x, y, w, h in scene.rectangles:
        env.add_rect_obstacle(x, y, w, h)
    all_obstacles = tuple(env.obstacles)
    original_collisions, original_reset = env._handle_collisions, env.reset
    env.legacy_collision_attempts = 0
    env.legacy_collision_events = []

    def assert_geometry(instance):
        if len(instance.obstacles) != len(all_obstacles):
            raise ValueError("Loaded scene geometry is fixed; cannot add or remove obstacles")

    def nearby(instance, position, reach, slack=5.):
        assert_geometry(instance)
        ids = scene.nearby_indices(position, float(reach) + float(slack))
        # Match the original native prefilter exactly, including rectangle size
        # convention and ordering. The spatial index is only an acceleration.
        distance = np.linalg.norm(scene.centers[ids] - np.asarray(position)[:2], axis=1)
        ids = ids[distance <= float(reach) + scene.native_radius[ids] + float(slack)]
        return [all_obstacles[i] for i in ids]

    def collisions(instance, info):
        assert_geometry(instance)
        candidates, proposed = set(), []
        for robot in instance.ground_robots:
            before, after = robot.prev_position[:2].copy(), robot.position[:2].copy()
            proposed.append((robot, before, after))
            for xy in (before, after):
                candidates.update(scene.nearby_indices(xy, robot.radius).tolist())
        full = instance.obstacles
        offset = len(info.setdefault("collisions", []))
        try:
            instance.obstacles = [all_obstacles[i] for i in sorted(candidates)]
            original_collisions(info)
        finally:
            instance.obstacles = full
        native_ids = {event["robot_id"] for event in info["collisions"][offset:]
                      if event.get("object") == "obstacle"}
        for robot, before, after in proposed:
            swept_hit = not scene.segment_clear(before, after, robot.radius)
            native_hit = robot.robot_id in native_ids
            if swept_hit or native_hit:
                instance.legacy_collision_attempts += 1
                event = dict(step=int(instance.step_count) + 1, robot_id=robot.robot_id,
                             before=before.tolist(), proposed=after.tolist(),
                             clearance_m=float(robot.radius), native_endpoint_collision=native_hit,
                             swept_collision_attempt=swept_hit)
                instance.legacy_collision_events.append(event)
                info.setdefault("legacy_collision_attempts", []).append(event)

    def reset(instance):
        instance.legacy_collision_attempts = 0
        instance.legacy_collision_events = []
        return original_reset()

    env._nearby_obstacles = MethodType(nearby, env)
    env._handle_collisions = MethodType(collisions, env)
    env.reset = MethodType(reset, env)
    env._legacy_scene_installed = True
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True, help="Prepared forest_map.npz")
    parser.add_argument("--output", type=Path, required=True, help="New standalone scene JSON")
    args = parser.parse_args()
    result = export_scene(args.map, args.output)
    print(json.dumps(dict(output=str(args.output.resolve()), simplified_scene=True,
                         unknown_space=result["unknown_space"], world_size_m=result["world_size_m"],
                         statistics=result["statistics"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
