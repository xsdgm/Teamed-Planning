"""Online kilometre missions from the frozen V9 circle generator with local bypasses."""
import hashlib
import io
import json
import math
from operator import index
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.ndimage import label

from world_model.hierarchical_sources import atomic_json, file_sha256
from simulator.survey_km_maps import GENERATOR, GENERATOR_SHA, WORLD_M, _generator
from simulator.survey_pretrain_maps import CLEARANCE_M


DISTANCE_BAND_M = (1000., 1500.)
CORRIDOR_HALF_WIDTH_M = 35.
MAX_ATTEMPTS = 16
_SPLIT_CODES = {'train': 0, 'development': 1, 'test': 2}


def _local_route(gen, circles, start, goal):
    """Certify a blocked centreline and a safe bypass inside its narrow corridor."""
    distance = np.linalg.norm(goal - start)
    forward = (goal - start) / distance
    normal = np.array([-forward[1], forward[0]])
    xy = circles[:, :2] - start
    along = xy @ forward
    side = xy @ normal
    radii = circles[:, 2]
    touched = (along > 0.) & (along < distance) & (np.abs(side) <= radii)
    count = int(touched.sum())
    if count == 0:
        return 0

    # Only circles close enough to intersect the corridor can affect the certificate.
    # Inflate by a half-cell diagonal to check the continuous vehicle clearance.
    width = CORRIDOR_HALF_WIDTH_M
    inflated = CLEARANCE_M + math.sqrt(.5)
    nearby = ((np.abs(side) <= width + radii + inflated)
              & (along >= -width - radii - inflated)
              & (along <= distance + width + radii + inflated))
    subset = [(row[:2], row[2]) for row in circles[nearby]]
    free = gen.raster_free(subset, WORLD_M, inflated)
    # The rotated rectangle can protrude up to sqrt(2)*width along either world axis.
    pad = math.ceil(math.sqrt(2.) * width) + 1
    lo = np.maximum(0, np.floor(np.minimum(start, goal)).astype(int) - pad)
    hi = np.minimum(free.shape[::-1], np.ceil(np.maximum(start, goal)).astype(int) + pad)
    free = free[lo[1]:hi[1], lo[0]:hi[0]]
    dx = np.arange(lo[0], hi[0]) + .5 - start[0]
    dy = np.arange(lo[1], hi[1])[:, None] + .5 - start[1]
    lateral = normal[0] * dx + normal[1] * dy
    station = forward[0] * dx + forward[1] * dy
    free &= (np.abs(lateral) <= width - math.sqrt(.5))
    free &= (station >= -width + math.sqrt(.5)) & (station <= distance + width - math.sqrt(.5))
    components, _ = label(free)  # 4-neighbour paths only
    indices = np.floor([start, goal]).astype(int) - lo
    first, last = components[indices[:, 1], indices[:, 0]]
    return count if first and first == last else 0


def local_case(repo, scene_seed, *, split='train', geometry_dir=None):
    """Generate one reproducible mission at collection time; certificates stay off-policy."""
    scene_seed = index(scene_seed)
    if not 0 <= scene_seed < 2**32 or split not in _SPLIT_CODES:
        raise ValueError('Expected a uint32 scene seed and a train, development, or test split')
    name = f'v9-local-{split}-{scene_seed}'
    gen = _generator(Path(repo).resolve() / GENERATOR)
    if geometry_dir is not None:
        folder = Path(geometry_dir).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        meta = folder / f'{name}.json'
        geometry = folder / f'{name}.npy'
        if meta.is_file():
            case = json.loads(meta.read_text())
            if (case['case_id'] != name or case['generator_sha256'] != GENERATOR_SHA
                    or case['circles_sha256'] != file_sha256(geometry)):
                raise ValueError(f'Cached online geometry differs for {name}')
            return dict(case, circles_path=str(geometry))

    for attempt in range(MAX_ATTEMPTS):
        # Each split, requested seed and rejection attempt has its own native RNG stream.
        generator_seed = (scene_seed * len(_SPLIT_CODES) + _SPLIT_CODES[split]) * MAX_ATTEMPTS + attempt
        scene = gen.generate_training_scene(generator_seed, world=WORLD_M, distance_range=DISTANCE_BAND_M,
                                            distance_sampling='annulus_area')
        if not scene['reference_geometry_validated'] or scene['reference_geometric_clearance_m'] != CLEARANCE_M:
            raise ValueError('Expected a clearance-certified native V9 reference route')
        circles = np.asarray([[*point, radius] for point, radius in scene['circles']], dtype=np.float64)
        blocked = _local_route(gen, circles, np.asarray(scene['start']), np.asarray(scene['goal']))
        if not blocked:
            continue
        stream = io.BytesIO()
        np.save(stream, circles, allow_pickle=False)
        payload = stream.getvalue()
        case = dict(case_id=name, seed=scene_seed, kind='v9_random', terrain='synthetic',
                    terrain_family_id=name, navigation_task='random_circles', world_size=WORLD_M,
                    start=scene['start'], goal=scene['goal'], direct_distance_m=float(scene['distance']),
                    reference_distance_m=float(scene['reference_polyline_length']), circle_count=len(circles),
                    circles_sha256=hashlib.sha256(payload).hexdigest(), generator_seed=generator_seed,
                    generator_sha256=GENERATOR_SHA, generation_attempt=attempt + 1,
                    direct_line_obstacle_count=blocked, certified_corridor_half_width_m=CORRIDOR_HALF_WIDTH_M)
        if geometry_dir is None:
            return dict(case, circles=scene['circles'])
        tmp = geometry.with_name(f'.{name}.{uuid4().hex}.tmp')
        try:
            tmp.write_bytes(payload)
            tmp.replace(geometry)
        finally:
            tmp.unlink(missing_ok=True)
        # Commit metadata last so a killed generator can retry either stage.
        atomic_json(meta, case)
        return dict(case, circles_path=str(geometry))
    raise RuntimeError(f'No local-bypass V9 scenario within {MAX_ATTEMPTS} attempts for {name}')
