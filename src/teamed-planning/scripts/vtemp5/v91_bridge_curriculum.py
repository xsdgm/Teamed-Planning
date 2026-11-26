"""Bridge resets and privileged reward/score oracles. Never pass an oracle to a policy."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage as nd
from skimage.graph import MCP_Geometric

from v91_km_cases import REPO, _raster_cells, _river_regions

SUITE = 'vtemp4-detour-curriculum-v1'
FLEET_OFFSET_XY = np.array([(9, -18), (9, 0), (9, 18)])
NAV_CLEARANCE_M = 1.5


def _read(path):
    return json.loads(Path(path).read_text())


def _check_splits(table):
    """Sibling counterfactual decks share a family/seed and must stay together."""
    columns = ('seed', 'terrain_family_id', 'map_sha256', 'case_id')
    for i, split in enumerate(('ppo_train', 'development', 'test')):
        for other in ('ppo_train', 'development', 'test')[:i]:
            for key in columns:
                if {r[key] for r in table[split]} & {r[key] for r in table[other]}:
                    raise ValueError(f'Bridge split leak on {key}: {other}/{split}')


@lru_cache(maxsize=2)
def _archive(repo):
    root = Path(repo) / 'artifacts' / SUITE
    table = _read(root / 'scenes.json')
    _check_splits(table)
    return root, table


@lru_cache(maxsize=2)
def _terrain(path, digest):
    path = Path(path)
    audit = _read(path.with_name('audit.json'))
    if (hashlib.sha256(path.read_bytes()).hexdigest() != digest or audit['map_sha256'] != digest
            or not audit['bridge_closure_disconnects_endpoints']
            or audit['certified_clearance_lower_bound_m'] < NAV_CLEARANCE_M):
        raise ValueError('Invalid archived single-deck route certificate')
    with np.load(path) as data:
        if float(data['resolution_m']) != 1. or np.any(data['origin_xy'] != 0):
            raise ValueError('Expected the original 1 m raster and world origin')
        free = data['known'].astype(bool) & ~data['occupied'].astype(bool)
    with np.load(path.with_name('landscape.npz')) as data:
        canopy = data['canopy_height_m']
    regions = _river_regions(str(path))
    # On this native 1 m grid the archived Euclidean certificate requires nearest
    # occupied-centre distance >= 1.5 + sqrt(2); exactly the 5x5 square is excluded.
    safe = nd.minimum_filter(free, size=5, mode='constant', cval=0)
    # The three native MAPPO UAVs spawn at 15 m AGL. In addition to clear ground,
    # require 1.5 m of vertical headroom over this same 5x5 neighbourhood.
    air_safe = safe & (nd.maximum_filter(canopy, size=5, mode='constant', cval=100.) < 13.5)
    deck_y, deck_x = np.nonzero(regions == 0)
    deck_xy = np.column_stack((deck_x, deck_y))
    return safe, air_safe, regions, deck_xy, audit


def _formation_ok(air_safe, xy):
    all_xy = np.asarray(xy, dtype=int)[..., None, :] + FLEET_OFFSET_XY
    xs, ys = all_xy[..., 0], all_xy[..., 1]
    h, w = air_safe.shape
    valid = (xs >= 0) & (ys >= 0) & (xs < w) & (ys < h)
    return (valid & air_safe[ys.clip(0, h-1), xs.clip(0, w-1)]).all(axis=-1)


def _rc(xy):
    return tuple(np.floor(xy).astype(int)[::-1])


def _near_reset(terrain, row, variant, rng):
    safe, air_safe, regions, deck_xy, audit = terrain
    base = np.array(audit['start'])
    orig_goal = np.array(audit['goal'])
    axis = (orig_goal - base) / np.linalg.norm(orig_goal - base)
    reverse = bool(variant // 2)
    if reverse:
        axis = -axis
    left = np.array([-axis[1], axis[0]])
    side = 1 if variant % 2 == 0 else -1
    center = np.array(audit['bridge_center_xy'])
    # In the archive, deck side edges are parallel to the start-goal line.
    projection = (deck_xy + .5 - center) @ axis
    entry, exit = center + projection.min() * axis, center + projection.max() * axis
    bank_a, bank_b = [regions[_rc(p)] for p in ((orig_goal, base) if reverse else (base, orig_goal))]
    radius = 150
    origin = np.maximum(0, np.floor(center).astype(int) - radius)
    end = np.minimum(safe.shape[::-1], np.ceil(center).astype(int) + radius + 1)
    box = np.s_[origin[1]:end[1], origin[0]:end[0]]
    local = safe[box]
    goal_ref = exit + rng.uniform(23., 40.) * axis
    goal_cells = np.argwhere(local & (regions[box] == bank_b))[:, ::-1] + origin
    goal = goal_cells[np.argmin(np.linalg.norm(goal_cells + .5 - goal_ref, axis=1))] + .5
    # Running the same certified eight-neighbour planner inside the crop proves a
    # feasible crossing and supplies the short mission's true reference time budget.
    costs, _ = MCP_Geometric(np.where(local, 1., np.inf)).find_costs([_rc(goal-origin)])
    r, c = np.nonzero(np.isfinite(costs) & (regions[box] == bank_a))
    cells = np.column_stack((c, r)) + origin
    poses = cells + .5
    signed_offset = ((poses - entry) @ left) * side
    upstream = (entry - poses) @ axis
    fleet_ok = _formation_ok(air_safe, cells)
    valid = (signed_offset >= 6.) & (signed_offset <= 42.) & (upstream >= 13.) & (upstream <= 75.) & fleet_ok
    if not valid.any():
        # Dense canopy near a mouth can rule out every native three-aircraft formation.
        # Use the closest legal pose on the same assigned upstream side of this deck.
        valid = (signed_offset >= 6.) & (signed_offset <= 80.) & (upstream >= 13.) & (upstream <= 120.) & fleet_ok
    if not valid.any():
        raise ValueError(f'No certified three-UAV {row["case_id"]} reset for side {side} reverse {reverse}')
    target = entry - rng.uniform(23., 44.) * axis + side * rng.uniform(12., 27.) * left
    i = int(np.argmin(np.where(valid, np.linalg.norm(poses-target, axis=1), np.inf)))
    return poses[i], goal, float(costs[r[i], c[i]]), dict(
        reverse=reverse, offset_side=side, signed_lateral_m=float(signed_offset[i]*side),
        approach_before_deck_m=float(upstream[i]))


def bridge_curriculum_scene(stage, episode, *, seed=0, split='ppo_train', repo=REPO):
    """A deterministic counterbalanced episode on the original, unscaled terrain."""
    if stage not in ('mouth', 'kilometre') or split not in ('ppo_train', 'development', 'test'):
        raise ValueError('Unknown bridge curriculum stage or split')
    if int(episode) != episode or episode < 0 or int(seed) != seed or seed < 0:
        raise ValueError('Episode and seed must be nonnegative integers')
    episode, seed = int(episode), int(seed)
    root, table = _archive(str(Path(repo).resolve()))
    rows = table[split]
    cycle, slot = divmod(episode, len(rows))
    ordering = np.random.default_rng(seed).permutation(len(rows))
    row = rows[ordering[slot]]
    path = (root / row['map_path']).resolve()
    terrain = _terrain(str(path), row['map_sha256'])
    safe, air_safe, regions, _, audit = terrain
    if (row['terrain_family_id'] != audit['terrain_family_id'] or row['seed'] != audit['seed']
            or not np.array_equal(row['start'], audit['start']) or not np.array_equal(row['goal'], audit['goal'])):
        raise ValueError('Map identity or source endpoints do not match the archive')
    rng = np.random.default_rng(np.random.SeedSequence([seed, episode, 7919]))
    if stage == 'mouth':
        variant = (episode + cycle + seed) % 4
        start, goal, length, reset = _near_reset(terrain, row, variant, rng)
    else:
        start, goal = np.asarray(row['start']), np.asarray(row['goal'])
        length = float(audit['reference_distance_m'])
        reset = dict(reverse=False, offset_side=0)
    direct = float(np.linalg.norm(goal - start))
    if (not safe[_rc(start)] or not safe[_rc(goal)] or regions[_rc(start)] <= 0 or regions[_rc(goal)] <= 0
            or regions[_rc(start)] == regions[_rc(goal)] or not _formation_ok(air_safe, np.floor(start))
            or (stage == 'kilometre' and not 1000. <= direct <= 1500.)):
        raise ValueError(f'Unsafe/unqualified native reset on {row["case_id"]}')
    fleet_seed = int(rng.integers(1, 2**31))
    case_id = f'{row["case_id"]}-{stage}-{seed}-{episode}'
    case = dict(row, map_path=str(path), case_id=case_id, start=start.tolist(), goal=goal.tolist(),
                direct_distance_m=direct, reference_distance_m=length)
    axis = (goal-start)/direct
    return dict(case=case, case_id=case_id, kind='bridge', seed=row['seed'], fleet_seed=fleet_seed,
                formal_eval=False, attempts=1, start=case['start'], goal=case['goal'], circles=[], rects=[],
                world_size=row['world_size'], direct_distance_m=direct, reference_distance_m=length,
                max_steps=math.ceil(6 * (length if stage == 'mouth' else direct) + 120),
                forward=axis.tolist(), left=[float(-axis[1]), float(axis[0])], audit_path=str(path.with_name('audit.json')),
                curriculum_stage=stage, curriculum=dict(source_case_id=row['case_id'], split=split, **reset,
                    fleet_spawn_xy=(start + FLEET_OFFSET_XY).tolist()))


def curriculum_cases(repo, split, stage, count, seed):
    return [bridge_curriculum_scene(stage, i, seed=seed, split=split, repo=repo) for i in range(count)]


@lru_cache(maxsize=1)
def _potential_field(path, digest, goal_rc):
    safe, _, _, _, _ = _terrain(path, digest)
    distance, _ = MCP_Geometric(np.where(safe, 1., np.inf)).find_costs([goal_rc])
    finite = np.isfinite(distance)
    return distance.astype(np.float32), float(distance[finite].max() + 1.)


def bridge_potential(scene, xy):
    """Negative 1 m certified ground geodesic to the opposite-bank goal, in metres.

    Interpolate reachable cell centres in the same bank/deck. At the edge of
    the certificate, project only onto that same region. Water, outside and
    isolated components receive a finite worse-than-reachable potential.
    """
    case = scene['case']
    path, digest = case['map_path'], case['map_sha256']
    _, _, regions, _, _ = _terrain(path, digest)
    field, invalid_cost = _potential_field(path, digest, _rc(scene['goal']))
    xy = np.asarray(xy, float)
    if xy.shape[-1:] != (2,) or not np.isfinite(xy).all():
        raise ValueError('Expected finite world XY')
    h, w = field.shape
    out = np.full(xy.reshape(-1, 2).shape[0], invalid_cost, dtype=float)
    for i, point in enumerate(xy.reshape(-1, 2)):
        r, c = _rc(point)
        if not (0 <= r < h and 0 <= c < w) or regions[r, c] < 0:
            continue
        bank = regions[r, c]
        x, y = np.floor(point - .5).astype(int)
        fx, fy = point - .5 - [x, y]
        total, weight = 0., 0.
        for dx, dy, v in ((0,0,(1-fx)*(1-fy)), (1,0,fx*(1-fy)), (0,1,(1-fx)*fy), (1,1,fx*fy)):
            col, row = x+dx, y+dy
            if (v > 0 and 0 <= col < w and 0 <= row < h and regions[row,col] == bank and np.isfinite(field[row,col])):
                total += v * float(field[row,col])
                weight += v
        if weight:
            out[i] = total / weight
            continue
        for radius in (3, 12, 48):
            x0, y0 = max(0,c-radius), max(0,r-radius)
            sl = np.s_[y0:min(h,r+radius+1), x0:min(w,c+radius+1)]
            yy, xx = np.nonzero((regions[sl] == bank) & np.isfinite(field[sl]))
            if len(xx):
                xx, yy = xx+x0, yy+y0
                d = np.hypot(xx+.5-point[0], yy+.5-point[1])
                best = int(np.argmin(d))
                out[i] = float(field[yy[best], xx[best]]) + d[best]
                break
    values = -out.reshape(xy.shape[:-1])
    return float(values) if values.ndim == 0 else values


class BridgeProgress:
    """A reward-side monitor fed only executed UGV positions in time order."""
    def __init__(self, scene):
        case = scene['case']
        _, _, self.regions, self.deck, _ = _terrain(case['map_path'], case['map_sha256'])
        self.previous, self.goal = np.array(scene['start'], float), np.array(scene['goal'], float)
        self.bank_a = self.regions[_rc(self.previous)]
        self.bank_b = self.regions[_rc(self.goal)]
        self.last = self.bank_a
        self.approaching_deck = False
        self.entered_deck = self.crossed_bridge = False
        self.first_deck_step = self.first_cross_step = None
        self.step = 0

    def update(self, xy):
        """Each update advances one recorded trajectory step following the reset."""
        xy = np.asarray(xy, float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('Expected one finite physical car pose')
        self.step += 1
        newly_entered = newly_crossed = False
        if not np.array_equal(np.floor(self.previous), np.floor(xy)):
            cells = _raster_cells(np.array([self.previous, xy]))
            xx, yy = cells.T
            h, w = self.regions.shape
            valid = (xx >= 0) & (yy >= 0) & (xx < w) & (yy < h)
            labels = np.full(len(cells), -2, dtype=int)
            labels[valid] = self.regions[yy[valid], xx[valid]]
            for region in labels[np.r_[True, labels[1:] != labels[:-1]]]:
                if region == self.last:
                    continue
                if region == 0:
                    self.approaching_deck = self.last == self.bank_a
                    if not self.entered_deck:
                        self.first_deck_step = self.step
                        self.entered_deck = newly_entered = True
                else:
                    if region == self.bank_b and self.last == 0 and self.approaching_deck and not self.crossed_bridge:
                        self.first_cross_step = self.step
                        self.crossed_bridge = newly_crossed = True
                    self.approaching_deck = False
                self.last = region
        self.previous = xy.copy()
        remaining = float(np.linalg.norm(self.goal - xy))
        gap = np.maximum(0., np.maximum(self.deck - xy, xy - (self.deck + 1)))
        return dict(reached_deck=self.entered_deck, entered_deck=self.entered_deck, crossed_bridge=self.crossed_bridge,
                    new_deck=newly_entered, new_crossing=newly_crossed, first_deck_step=self.first_deck_step,
                    first_cross_step=self.first_cross_step, goal_reached=remaining<=2., goal_remaining_m=remaining,
                    deck_distance_m=float(np.sqrt(np.einsum('ij,ij->i', gap, gap).min())))


def bridge_score(scene, path):
    path = np.asarray(path, float)
    if (path.ndim != 2 or path.shape[1] != 2 or not len(path) or not np.isfinite(path).all()
            or not np.allclose(path[0], scene['start'], rtol=0, atol=1e-4)):
        raise ValueError('Expected the complete UGV track starting at reset')
    monitor = BridgeProgress(scene)
    if len(path) == 1:
        return monitor.update(path[0])
    for point in path[1:]:
        event = monitor.update(point)
    return event


bridge_curriculum_metrics = bridge_score
