"""Receiver-only terrain, known-road planning and survey decisions for a live V9 fleet."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from shared.simulator.ground_evidence import observed_fused_graph
from vtemp2_3.ground_policy import CANDIDATE_LIMIT, ground_observation, ground_sensor_tile
from vtemp2_3.hierarchical_runtime import SLOW_FEATURES
from vtemp2_3.navigation_evidence import EvidenceGrid, corridor_fractions, features
from survey.survey_fleet_clearance import certified_start_connector, certified_start_connectors, native_segment_clear
from survey.survey_pretrain_env import GROUND_LOOKAHEAD_M, ground_plan, _sample_corridors
from vtemp2_3.wildscenes_runtime import pool_received_tile


TERRAIN_MAX_AGE_S = 30.
START_CONNECTOR = 'certified_start_connector'
SHARED_GUIDANCE = 'native_shared_guidance'


def _road_degree(cells, cell):
    # Confirmed-free cells already include the car's full clearance at map edges.
    y, x = cell
    return sum(cells[y+dy, x+dx] == -1 for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
               if 0 <= y+dy < cells.shape[0] and 0 <= x+dx < cells.shape[1])


def pack_tile(tile):
    """Encode unknown, free and occupied in two bits per pixel, in native YX order."""
    codes = (np.asarray(tile, dtype=np.int8) + 1).astype(np.uint8).ravel()
    codes = np.pad(codes, (0, (-len(codes)) % 4)).reshape(-1, 4)
    return np.bitwise_or.reduce(codes << np.array([0, 2, 4, 6], np.uint8), axis=1).tobytes()


def unpack_tile(packet):
    codes = np.frombuffer(packet['pixels'], dtype=np.uint8)[:, None]
    values = ((codes >> np.array([0, 2, 4, 6], np.uint8)) & 3).ravel()
    return values[:math.prod(packet['shape'])].astype(np.int8).reshape(packet['shape']) - 1


class _TerrainRadio:
    """Extra local-map payloads follow actual UAV send ticks and eligible physical links."""
    def __init__(self, experiment, seed):
        self.experiment = experiment
        self.radio = experiment.relay_channel
        self.link = type(self.radio)(seed=int(seed) ^ 0x54494c45, loss=self.radio.loss,
                                     delay_jitter=self.radio.delay_jitter, period=self.radio.period)
        self.drone_ids = {c.drone_id:rid for c,rid in zip(experiment.controllers, experiment.ids)}
        self.cache = {rid:{} for rid in experiment.ids}
        self.original = self.radio.transmit
        self.previous = vars(self.radio).get('transmit')
        self.radio.transmit = self.transmit
        self.send_ticks = 0

    def transmit(self, envelope, receivers, frame):
        result = self.original(envelope, receivers, frame)
        rid = self.drone_ids.get(envelope[0]['sender_id']) if envelope else None
        if rid is None:
            return result
        exp = self.experiment
        obs = exp.obs[rid]  # sensor on the sending aircraft, before the map transport boundary
        now = float(exp.env.time)
        cache = self.cache[rid]
        for message in self.link.deliver(rid, frame):
            for item in message:
                p = item['state']
                old = cache.get(p['source'])
                if p['source'] != rid and (old is None or old['observed_at'] < p['observed_at']):
                    cache[p['source']] = p
        tile = ground_sensor_tile(obs)
        if tile is not None:
            cache[rid] = dict(source=rid, observed_at=now, position=tuple(obs['self']['position'][:2]),
                              resolution=float(obs['map_resolution']), pixels=pack_tile(tile),
                              shape=tile.shape, hops=0)
        # Only cache one actual measurement per source. Delayed packets retain source stamps.
        packets = [dict(state=dict(p, hops=p['hops']+1)) for p in cache.values()
                   if p['hops'] < len(self.drone_ids) and now - p['observed_at'] <= TERRAIN_MAX_AGE_S]
        links = list(receivers)
        if any(n['id'] == exp.ground for n in obs['neighbors']):
            links.append(exp.ground)
        if packets:
            self.link.transmit(packets, links, frame)
            self.send_ticks += 1
        return result

    def close(self):
        if self.radio.transmit == self.transmit:
            if self.previous is None:
                del self.radio.transmit
            else:
                self.radio.transmit = self.previous


class FleetSurveyAdapter:
    """Ground observations contain onboard measurements and packets delivered to this car."""
    resolution = 1.
    def __init__(self, episode, model):
        self.ep, self.model, self.exp = episode, model, episode.env
        n = math.ceil(episode.case['world_size']/self.resolution)
        self.cells = np.zeros((n, n), np.int8)
        self.stamps = np.full((n, n), -1., np.float32)
        self.bounds = None
        self.frame = -1
        self.terrain = _TerrainRadio(self.exp, episode.case['seed'])
        self.seen = set()
        self.tiles = self.indirect = self.air_new_cells = self.max_hops = 0
        self.last_received = None
        self.sources = set()
        self.last_choice = None
        self.last_survey_evidence = None
        self.survey_answered = False
        self.credit_cells = 0
        self.active_gap = np.empty(0, dtype=np.int64)
        self.dispatched_target = None
        self.dispatched_at = None
        self.dispatch_answered = False
        self.credited = np.empty(0, dtype=bool)
        self.ground_guidance = None
        self.ground_target = None
        self.ground_path_xy = None
        self.last_ground_reason = None
        self.connector_plans = 0
        self.isolated_start_recoveries = 0
        self._isolated_origin = None
        self._isolated_exit = None
        self._native_guidance = False
        self.last_graph_edges = 0

    @property
    def now(self):
        return float(self.exp.env.time)

    @property
    def own(self):
        return self.exp.obs[self.exp.ground]['self']

    def _ingest(self, tile, position, resolution, stamp, source):
        ids, values = pool_received_tile(tile, position, resolution, self.resolution,
                                         self.cells.shape, occupied_overrides_unknown=True)
        old = self.stamps.flat[ids]
        accepted = (values != 0) & ((stamp > old) | ((stamp == old) & (values > self.cells.flat[ids])))
        ids, values = ids[accepted], values[accepted]
        if not len(ids):
            return
        fresh = ids[self.cells.flat[ids] == 0]
        yy, xx = np.divmod(ids, self.cells.shape[1])
        lo, hi = np.array([yy.min(),xx.min()]), np.array([yy.max(),xx.max()])
        eligible = (source is not None and self.last_choice is not None and self.active_gap.size
                    and not self.survey_answered and stamp > self.last_choice['issued_at']
                    and self.now - stamp <= TERRAIN_MAX_AGE_S)
        if eligible:
            gap_rows, gap_cols = np.divmod(self.active_gap, self.cells.shape[1])
            low = np.maximum((gap_rows.min()-2, gap_cols.min()-2), 0)
            high = np.minimum((gap_rows.max()+3, gap_cols.max()+3), self.cells.shape)
            eligible = bool(np.all(lo < high) and np.all(hi >= low))
        if eligible:
            submap = self.cells[low[0]:high[0], low[1]:high[1]]
            at_gap = (gap_rows-low[0], gap_cols-low[1])
            before = _clearance(submap)[at_gap]
        self.cells.flat[ids], self.stamps.flat[ids] = values, stamp
        self.bounds = (lo,hi) if self.bounds is None else (np.minimum(self.bounds[0],lo),np.maximum(self.bounds[1],hi))
        if source is not None:
            self.air_new_cells += len(fresh)
        if not eligible or (before == 1).any() or (before == -1).all():
            return
        after = _clearance(submap)[at_gap]
        newly_answered = (before == 0) & (after != 0) & ~self.credited
        self.credited |= newly_answered
        self.credit_cells = int(self.credited.sum())
        verdict = 'blocked' if (after == 1).any() else ('passable' if (after == -1).all() else None)
        if verdict:
            self.survey_answered = self.dispatch_answered = True
        if verdict or (newly_answered.any() and self.last_survey_evidence is None):
            self.last_survey_evidence = dict(kind=verdict or 'partial', source=source, observed_at=float(stamp),
                                             received_at=self.now, target_xy=self.last_choice['target'].tolist())

    def observe_receipts(self):
        frame = int(self.exp.env.step_count)
        if self.frame == frame:
            return
        self.frame = frame
        ground = self.exp.obs[self.exp.ground]
        tile = ground_sensor_tile(ground)
        if tile is not None:
            self._ingest(tile, self.own['position'], ground['map_resolution'], self.now, None)
        for msg in self.terrain.link.deliver(self.exp.ground, frame):
            for item in msg:
                p = item['state']
                key = (p['source'], p['observed_at'])
                if key in self.seen:
                    continue
                if p['observed_at'] >= self.now or p['hops'] < 1:
                    raise ValueError('Impossible terrain packet receipt')
                self.seen.add(key)
                self.sources.add(p['source'])
                self.tiles += 1
                self.indirect += int(p['hops'] > 1)
                self.max_hops = max(self.max_hops, p['hops'])
                self.last_received = self.now
                self._ingest(unpack_tile(p), p['position'], p['resolution'], p['observed_at'], p['source'])

    def _plan(self, guidance=None):
        self.observe_receipts()
        car = np.asarray(self.own['position'])[:2]
        pos = np.floor(car/self.resolution).astype(int)[::-1]
        destination = self.ep.goal if guidance is None else guidance
        dest = np.floor(destination/self.resolution).astype(int)[::-1]
        span = self.bounds or (pos,pos)
        lo = np.maximum(np.minimum(span[0],pos)-28, 0)
        hi = np.minimum(np.maximum(span[1],pos)+29, self.cells.shape)
        view = self.cells[lo[0]:hi[0], lo[1]:hi[1]]
        cells = _clearance(view)
        p, g = pos-lo, dest-lo
        origin = getattr(self, '_isolated_origin', None)
        if origin is not None and not np.array_equal(pos, origin) and not np.array_equal(dest, origin):
            # After a subcell movement the isolated cell can temporarily reappear
            # as a dead end. Do not route right back into it during the handoff.
            old = origin - lo
            if ((old >= 0).all() and (old < cells.shape).all()
                    and _road_degree(cells, old) <= 1):
                cells[tuple(old)] = 1
        connector = None
        if cells[tuple(p)] != -1:
            ground = self.exp.obs[self.exp.ground]
            tile = ground_sensor_tile(ground)
            res = float(ground['map_resolution'])
            threshold = self.exp.ground_controller.waypoint_threshold
            entry = certified_start_connector(tile, car, res, car, cells, lo[::-1]*self.resolution,
                                               self.resolution, min_distance_m=threshold)
            if entry is None:
                return lo, cells, p[None], p[None], g, 'uncertified_vehicle_clearance', None
            p, connector = entry
        path, fronts, reason = ground_plan(cells, p, g, self.resolution, directional=guidance is not None)
        if (connector is None and len(path) == 1 and reason != 'known_goal_route'
                and _road_degree(cells, p) == 0):
            ground = self.exp.obs[self.exp.ground]
            tile = ground_sensor_tile(ground)
            entries = certified_start_connectors(
                tile, car, ground['map_resolution'], car, cells, lo[::-1]*self.resolution,
                self.resolution, min_distance_m=self.exp.ground_controller.waypoint_threshold)
            for entry, segment in entries:
                if _road_degree(cells, entry) == 0:
                    continue
                route, exits, status = ground_plan(cells, entry, g, self.resolution,
                                                   directional=guidance is not None)
                if len(route) > 1 or status == 'known_goal_route':
                    return lo, cells, route, exits, g, status, segment
        if connector is not None:
            # Choose a physically executable waypoint along the known route, within
            # this single native observation. Shortcuts through unknown are rejected.
            points = ((path + lo)[:, ::-1] + .5) * self.resolution
            half_span = np.asarray(tile.shape[::-1]) * res / 2
            for point in points[1:]:
                if np.any(np.abs(point-car) >= half_span):
                    break
                if np.linalg.norm(point-car) > threshold and native_segment_clear(tile, car, res, car, point):
                    connector = np.stack((car, point))
        return lo, cells, path, fronts, g, reason, connector

    def _drive_connector(self, connector):
        self._known_target_history = None
        self.ground_path_xy = connector
        self.ground_target = connector[-1].copy()
        self.last_ground_reason = START_CONNECTOR
        self.connector_plans += 1
        return self.ground_target.copy(), START_CONNECTOR

    def plan_ground(self, guidance=None):
        car = np.asarray(self.own['position'])[:2]
        self.ground_guidance = None if guidance is None else np.asarray(guidance, dtype=float).copy()
        if guidance is None:
            self._native_guidance = False
        ground = self.exp.obs[self.exp.ground]
        target = getattr(self, '_isolated_exit', None)
        if target is not None:
            self.observe_receipts()
            # Recheck the actual sensor on every replan. Commit long enough to
            # cross the subcell gap even if the coarse raster changes mid-turn.
            if (np.linalg.norm(target-car) > self.exp.ground_controller.waypoint_threshold
                    and native_segment_clear(ground_sensor_tile(ground), car, ground['map_resolution'], car, target)):
                return self._drive_connector(np.stack((car, target)))
            self._isolated_exit = None
        if getattr(self, '_isolated_origin', None) is not None:
            tile = ground_sensor_tile(ground)
            half_width = 0. if tile is None else min(tile.shape) * ground['map_resolution'] / 2
            origin_xy = (self._isolated_origin[::-1] + .5) * self.resolution
            if np.linalg.norm(car - origin_xy) > half_width:
                self._isolated_origin = None
        lo, cells, path, _, g, reason, connector = self._plan(guidance)
        if reason == 'uncertified_vehicle_clearance' and guidance is not None:
            # A continuous, ego-observed detour can cross a gap where the
            # coarser receiver grid has no certified cell centre to start from.
            guard = getattr(self.ep, 'motion_guard', None)
            if guard is not None:
                segment = guard.observed_frontier_segment(guidance)
                if segment is not None:
                    return self._drive_connector(segment)
        if connector is not None:
            pos = np.floor(car / self.resolution).astype(int)[::-1]
            if cells[tuple(pos-lo)] == -1 and _road_degree(cells, pos-lo) == 0:
                self._isolated_exit = connector[-1].copy()
                self._isolated_origin = pos.copy()
                self.isolated_start_recoveries = getattr(self, 'isolated_start_recoveries', 0) + 1
            return self._drive_connector(connector)
        count = min(len(path), 1+int(GROUND_LOOKAHEAD_M/self.resolution))
        candidate = path[count-1]+lo
        destination = self.ep.goal if guidance is None else np.asarray(guidance)
        last = getattr(self, '_known_target_history', None)
        if reason == 'known_goal_route' and connector is None and count < len(path):
            if (last is not None and self.last_ground_reason == 'known_goal_route'
                    and np.allclose(last[0], destination, rtol=0., atol=1e-7)):
                prior, before = last[1:3]
                if (before is not None and np.array_equal(candidate, before)
                        and np.abs(candidate-prior).sum() == 1):
                    # The prior unreached controller target must still lie on the
                    # same freshly observed route, at the same lookahead distance.
                    chosen = self.ground_target
                    oldcell = np.floor(chosen/self.resolution).astype(int)[::-1]
                    start = max(1, count-2)
                    matches = np.flatnonzero(np.all(path[start:count+1]+lo==oldcell,axis=1))
                    if (len(matches) and np.linalg.norm(chosen-car) > self.exp.ground_controller.waypoint_threshold
                            and np.allclose(chosen,(oldcell[::-1]+.5)*self.resolution,rtol=0.,atol=1e-7)):
                        count = start+int(matches[0])+1
                self._known_target_history = (destination.copy(),candidate.copy(),prior.copy())
            else:
                self._known_target_history = (destination.copy(),candidate.copy(),None)
        else:
            self._known_target_history = None
        path = path[:count]
        self.ground_path_xy = ((path+lo)[:,::-1]+.5)*self.resolution
        self.ground_target = self.ground_path_xy[-1].copy()
        if np.array_equal(path[-1], g) and (guidance is None or reason == 'known_goal_route'):
            self.ground_target = (self.ep.goal if guidance is None else guidance).copy()
        elif len(path) == 1:
            self.ground_target = np.asarray(self.own['position'][:2]).copy()
        # A directional search can end at the car or at an adjacent cell whose
        # centre the native driver already considers reached. Give the driver
        # the shared goal to route to, keeping only certified cells in its guard path.
        directional = guidance is not None and (
            reason == 'directional_route' or (reason == 'hold' and _road_degree(cells, path[0]) > 0))
        self._native_guidance = directional and (
            getattr(self, '_native_guidance', False)
            or np.linalg.norm(self.ground_target-car) <= self.exp.ground_controller.waypoint_threshold)
        if self._native_guidance:
            # Remain on the native road graph until the receiver can connect the
            # guidance point; otherwise the next cell crossing can undo this handoff.
            self.ground_target = np.asarray(guidance, dtype=float).copy()
            reason = SHARED_GUIDANCE
        self.last_ground_reason = reason
        return self.ground_target.copy(), reason

    def _survey_inputs(self):
        lo, cells, path, fronts, dest, reason, _ = self._plan()
        gaps = [] if reason == 'known_goal_route' else _sample_corridors(cells,fronts,dest,self.resolution)
        no_request = not gaps
        if no_request:
            gaps = [(path[0],np.empty(0, dtype=np.int64))]
        car = np.asarray(self.own['position'],dtype=float)[:2]
        points = (np.asarray([q for q,_ in gaps])[:,::-1]+lo[::-1]+.5)*self.resolution
        grid = EvidenceGrid(cells, self.resolution, lo[::-1]*self.resolution, clearance=0.)
        graph = observed_fused_graph(self.exp.ground_controller.unified_graph, self.now)
        self.last_graph_edges = graph['graph_edges']
        receipt_age = self.now-self.last_received if self.last_received is not None else 60.
        belief = dict(position=car, goal=self.ep.goal, grid=grid, speed=np.linalg.norm(self.own['velocity'][:2]),
                      receipt_age=receipt_age, recent_receipts=float(self.last_received is not None and receipt_age<=10), **graph)
        rows = np.zeros((CANDIDATE_LIMIT,len(SLOW_FEATURES)),np.float32)
        corridors = corridor_fractions(grid,car,points)
        for i,q in enumerate(points):
            base,_ = features(belief,q,corridor=corridors[i])
            rows[i,:len(base)] = base
        rows[:len(points),SLOW_FEATURES.index('remaining_budget')] = max(0.,1.-self.now/self.ep.rollout_limit)
        targets = np.zeros((CANDIDATE_LIMIT,2)); targets[:len(points)] = points
        mask = np.arange(CANDIDATE_LIMIT)<len(points)
        cand = dict(features=rows,targets=targets,present=mask.copy(),mask=mask)
        feedback = dict(ground_target=self.ground_target,
                        survey_target=self.dispatched_target, survey_issued_at=self.dispatched_at,
                        survey_evidence_received=self.dispatch_answered)
        comm = self.ep._relay_observation.observe()
        facade = SimpleNamespace(env=self.exp, goal=self.ep.goal, belief=lambda:belief)
        obs = ground_observation(facade,cand,comm['one_hop_neighbors'],comm['relay_tokens'],task_context=feedback)
        return obs,points,no_request,lo,cells.shape,gaps

    def survey_observation(self):
        return self._survey_inputs()[:3]

    def select_survey(self, *, deterministic=True, generator=None):
        obs,points,no_request,lo,shape,gaps = self._survey_inputs()
        index,logp,value = self.model.act(obs,generator=generator,deterministic=deterministic)
        q = None if no_request else points[index].copy()
        gap = gaps[index][1]
        yy,xx = np.unravel_index(gap,shape)
        self.active_gap = np.ravel_multi_index((yy+lo[0],xx+lo[1]),self.cells.shape)
        car = np.asarray(self.own['position'])[:2]
        extra = 0. if q is None else np.linalg.norm(q-car) + np.linalg.norm(self.ep.goal-q) - np.linalg.norm(self.ep.goal-car)
        if q is not None:
            if self.dispatched_target is None or not np.array_equal(q, self.dispatched_target):
                self.dispatched_at = self.now
                self.dispatch_answered = False
            self.dispatched_target = q.copy()
        else:
            self.dispatched_target = self.dispatched_at = None
            self.dispatch_answered = False
        self.credited = np.zeros(len(gap), dtype=bool)
        self.last_choice = dict(observation=obs,action=index,logp=logp,value=value,target=q,
                                issued_at=self.now,route_relevance=float(1/(1+max(extra,0.)/GROUND_LOOKAHEAD_M)),gap_count=len(gap))
        self.credit_cells=0; self.survey_answered=False; self.last_survey_evidence=None
        return q

    def survey_credit(self):
        choice = self.last_choice or {}
        fraction = self.credit_cells/max(len(self.active_gap),1)
        weight = choice.get('route_relevance',0.)
        return dict(useful_new_cells=self.credit_cells,gap_count=len(self.active_gap),gap_fraction=fraction,
                    resolved_question=self.survey_answered,route_relevance=weight,
                    answer=self.last_survey_evidence['kind'] if self.survey_answered else None,
                    reward_evidence=float(weight*(fraction+int(self.survey_answered))))

    def stats(self):
        return dict(connector_plans=self.connector_plans, isolated_start_recoveries=self.isolated_start_recoveries,
                    terrain_radio=dict(self.terrain.link.stats), unique_received_tiles=self.tiles,
                    multihop_tiles=self.indirect, max_tile_hops=self.max_hops,
                    received_terrain_sources=sorted(self.sources), new_air_cells=self.air_new_cells,
                    last_terrain_receipt_s=self.last_received, last_received_graph_edges=self.last_graph_edges,
                    last_survey_evidence=self.last_survey_evidence)

    def close(self):
        self.terrain.close()


def _clearance(cells):
    # A 1m-grid cell is drivable only if all cells within the car's 1.5m
    # centre-clearance neighbourhood were observed free. Unknown stays unknown.
    axis = np.maximum(np.abs(np.arange(-2,3))-.5,0.)
    disk = axis[:,None]**2 + axis**2 <= 1.5**2
    free = binary_erosion(cells == -1, structure=disk)
    blocked = binary_dilation(cells == 1, structure=disk)
    return np.where(blocked,1,np.where(free,-1,0)).astype(np.int8)
