"""VTemp5 local obstacle hypotheses from terrain delivered to a live vehicle.

World XY and native YX rasters are in metres. Positive side is left in the
vehicle-to-goal frame frozen at first detection, so driving around the obstacle
cannot reverse its identity. Predicted vertices serve the actor only.
"""
import math

import numpy as np
from scipy import ndimage


_SIDES = (1, -1)
_CONNECT8 = np.ones((3, 3), bool)


class ReceivedObstacleTracker:
    def __init__(self, model, shape, resolution):
        self.model = model
        self.res = float(resolution)
        if len(shape) != 2 or min(shape) < 2 or self.res <= 0:
            raise ValueError('Invalid receiver map')
        self.air = np.zeros(shape, np.int8)
        self.air_stamp = np.full(shape, -np.inf, np.float32)
        self.received_at = None
        self.anchor = None
        self.obstacle_count = self.switch_count = self.prediction_calls = 0
        self.encounter_id = None
        self._released = {}
        self.locked_side = 0
        self.state = self._empty()
        self._attached = None

    def _empty(self):
        return dict(active=False, obstacle_id=None, locked_side=0, side_status={s:'unknown' for s in _SIDES},
                    routes={}, route_observed={}, route_open_probability={}, evidence_time=self.received_at,
                    switch_count=self.switch_count, closure_evidence={})

    def ingest_air(self, ids, values, observed_at, received_at):
        """Ingest native pooled cells from an already delivered aircraft packet."""
        ids, values = np.asarray(ids, dtype=np.intp), np.asarray(values, dtype=np.int8)
        old = self.air_stamp.flat[ids]
        keep = (values != 0) & ((observed_at > old) | ((observed_at == old) & (values > self.air.flat[ids])))
        self.air_stamp.flat[ids[keep]] = observed_at
        self.air.flat[ids[keep]] = values[keep]
        if keep.any():
            self.received_at = float(received_at)

    def attach(self, adapter):
        """Observe copies of accepted radio payloads. The original planner still fuses each tile."""
        from vtemp2_3.wildscenes_runtime import pool_received_tile
        if self._attached is not None:
            if self._attached is not adapter:
                raise ValueError('A tracker belongs to one episode')
            return
        if adapter.cells.shape != self.air.shape or adapter.resolution != self.res:
            raise ValueError('Incompatible receiver raster')
        old_ingest = adapter._ingest

        def ingest(tile, position, resolution, stamp, source):
            result = old_ingest(tile, position, resolution, stamp, source)
            if source is not None:
                ids, values = pool_received_tile(tile, position, resolution, self.res, self.air.shape,
                                                  occupied_overrides_unknown=True)
                self.ingest_air(ids, values, stamp, adapter.now)
            return result

        adapter._ingest = ingest
        self._attached = adapter

    def _sample(self, cells, xy):
        ij = np.floor(xy / self.res).astype(np.intp)
        inside = ((ij >= 0) & (ij < np.array(cells.shape[::-1]))).all(-1)
        ij = np.clip(ij, 0, np.array(cells.shape[::-1])-1)
        return np.where(inside, cells[ij[...,1], ij[...,0]], 0)

    def _component(self, cells, point, reach):
        """Only the connected, air-observed occupied pixels around a measured hit."""
        ij = np.floor(point / self.res).astype(int)
        r = math.ceil(reach / self.res)
        lo = np.maximum(0, ij-r)
        hi = np.minimum(cells.shape[::-1], ij+r+1)
        key = np.s_[lo[1]:hi[1], lo[0]:hi[0]]
        labels, _ = ndimage.label((self.air[key] == 1) & (cells[key] == 1), _CONNECT8)
        index = labels[ij[1]-lo[1], ij[0]-lo[0]]
        if not index:
            return np.empty((0,2))
        y, x = np.nonzero(labels == index)
        return (np.column_stack((x,y)) + lo + .5) * self.res

    def _detect(self, car, goal, cells):
        distance = np.linalg.norm(goal-car)
        if distance <= 4 or self.received_at is None:
            return False
        forward = (goal-car) / distance
        left = np.array([-forward[1], forward[0]])
        samples = car + np.arange(0., min(distance, 130.), self.res)[:,None] * forward
        hits = (self._sample(self.air, samples) == 1) & (self._sample(cells, samples) == 1)
        # A small foreground stone should not mask a larger obstruction farther ahead.
        starts = np.flatnonzero(hits & ~np.r_[False, hits[:-1]])
        for i in starts:
            anchor = (np.floor(samples[i]/self.res) + .5) * self.res
            patch = self._component(cells, anchor, 52.)
            if len(patch) and np.ptp((patch-anchor) @ left) + self.res >= 24.:
                pixels = np.floor(patch / self.res).astype(int)
                self._identity_pixels = np.ravel_multi_index((pixels[:, 1], pixels[:, 0]), self.air.shape)
                if self._released:
                    # A turn back toward a previous obstacle can hit a different
                    # face. Match the connected reported material, not the ray
                    # intersection or the current vehicle-to-goal axes.
                    labels, _ = ndimage.label((self.air == 1) & (cells == 1), _CONNECT8)
                    x, y = np.floor(anchor / self.res).astype(int)
                    component = labels[y, x]
                    for encounter_id, old in self._released.items():
                        if np.any(labels.flat[old['pixels']] == component):
                            self._identity_pixels = np.union1d(self._identity_pixels, old['pixels'])
                            self.encounter_id = encounter_id
                            self.anchor, self.forward, self.left = old['axes']
                            self.locked_side = old['side']
                            self.verdict, self.denials, self.rear = old['evidence']
                            return True
                self.anchor, self.forward, self.left = anchor, forward, left
                self.obstacle_count += 1
                self.encounter_id = self.obstacle_count
                self.locked_side = 0
                self.verdict = {s:'unknown' for s in _SIDES}
                self.denials = {}
                self.rear = None
                return True
        return False

    def _certified_cut(self, cells, side, depth):
        """No path even with unknown cells open, 8-neighbour moves and the entire mission width.

        The upstream and downstream sets include *all* points on their respective
        cross-sections. A blocked sample or a local cut ending at the survey edge
        cannot disconnect this optimistic side excursion.
        """
        air = self.air
        if not any((line == 1).any() for line in (air[0], air[-1], air[:,0], air[:,-1])):
            return False
        h,w = cells.shape
        corners = np.array([[0,0],[w,0],[0,h],[w,h]]) * self.res
        limit = max(0., np.max((corners-self.anchor) @ (side*self.left))) + 3*self.res
        pad = math.sqrt(2.)*self.res
        back, front = -8.*self.res, max(12.*self.res, depth or 0.)
        poly = np.array([self.anchor+x*self.forward+y*side*self.left for x in (back-pad,front+pad)
                         for y in (-pad,limit)])
        lo = np.maximum(0,np.floor(poly.min(0)/self.res).astype(int)-1)
        hi = np.minimum([w,h],np.ceil(poly.max(0)/self.res).astype(int)+1)
        yy, xx = np.ogrid[lo[1]:hi[1],lo[0]:hi[0]]
        x,y = (xx+.5)*self.res-self.anchor[0], (yy+.5)*self.res-self.anchor[1]
        along = x*self.forward[0]+y*self.forward[1]
        across = (x*self.left[0]+y*self.left[1])*side
        band = (across >= -pad) & (along >= back-pad) & (along <= front+pad)
        near = band & (along <= back+pad)
        far = band & (along >= front-pad)
        if not near.any() or not far.any():
            return False
        key = np.s_[lo[1]:hi[1],lo[0]:hi[0]]
        optimistic = band & ~((air[key] == 1) & (cells[key] == 1))
        groups, _ = ndimage.label(optimistic, _CONNECT8)
        entrances = np.unique(groups[near & optimistic])
        return not np.isin(groups[far & optimistic], entrances).any()

    def _entry_connected(self, car, entry, cells):
        # A surveyed opening can be real while the car is still isolated from
        # its entrance. Only confirmed-free cells can establish that access.
        coords = np.floor(np.array([car, entry]) / self.res).astype(int)
        if np.any(coords < 0) or np.any(coords >= self.air.shape[::-1]):
            return False
        pad = math.ceil(32. / self.res)
        lo = np.maximum(0, coords.min(0) - pad)
        hi = np.minimum(self.air.shape[::-1], coords.max(0) + pad + 1)
        known = cells[lo[1]:hi[1], lo[0]:hi[0]] == -1
        labels, _ = ndimage.label(known)
        start, end = coords - lo
        group = labels[start[1], start[0]]
        return bool(group and labels[end[1], end[0]] == group)

    def _clear_goal_corridor(self, car, goal, cells):
        """Verify a contiguous swept footprint from the car towards its goal using received cells."""
        distance = np.linalg.norm(goal - car)
        if distance < 1.e-8:
            return False
        direction = (goal - car) / distance
        if direction @ self.forward <= 0.:
            return False
        # Check a local length at least as wide as the minimum large obstacle.
        # Require every cell touching the clearance envelope to be known free.
        length = min(distance, max(24., 6. * self.res))
        end = car + length * direction
        pad = 1.5 + np.sqrt(.5) * self.res
        lo = np.floor((np.minimum(car, end) - pad) / self.res).astype(int)
        hi = np.ceil((np.maximum(car, end) + pad) / self.res).astype(int)
        if (lo < 0).any() or (hi > cells.shape[::-1]).any():
            return False
        yy, xx = np.ogrid[lo[1]:hi[1], lo[0]:hi[0]]
        dx, dy = (xx + .5) * self.res - car[0], (yy + .5) * self.res - car[1]
        along = np.clip(dx * direction[0] + dy * direction[1], 0., length)
        corridor = (dx - along * direction[0])**2 + (dy - along * direction[1])**2 <= pad**2
        return bool(np.all(cells[lo[1]:hi[1], lo[0]:hi[0]][corridor] == -1))

    def _release(self, patch, now_s, reason):
        pixels = np.floor(patch / self.res).astype(int)
        self._released[self.encounter_id] = dict(
            pixels=np.union1d(self._identity_pixels,
                              np.ravel_multi_index((pixels[:, 1], pixels[:, 0]), self.air.shape)),
            axes=(self.anchor, self.forward, self.left), side=self.locked_side,
            evidence=(self.verdict, self.denials, self.rear))
        self.anchor = None
        self.locked_side = 0
        self.state = self._empty()
        self.state['release_evidence'] = dict(obstacle_id=self.encounter_id, reason=reason,
                                              review_at=float(now_s), received_at=self.received_at)
        return self.state

    def review(self, car_xy, goal_xy, cells, now_s):
        car, goal = np.asarray(car_xy,float), np.asarray(goal_xy,float)
        if cells.shape != self.air.shape or (self.received_at is not None and now_s < self.received_at):
            raise ValueError('Invalid receiver grid or time')
        if self.anchor is None and not self._detect(car,goal,cells):
            self.state = self._empty()
            return self.state
        r = self.res
        patch = self._component(cells,self.anchor,220.)
        xy = patch - self.anchor
        along, across = xy @ self.forward, xy @ self.left
        width = np.ptp(across)+r if len(across) else 0.
        seen_far = float(along.max()) if len(along) else 0.
        # A depth measurement requires observed contiguous material followed by
        # observed free terrain on the same forward ray.
        ticks = np.arange(0.,220.+r,r)
        spine = self._sample(cells, self.anchor + ticks[:,None]*self.forward)
        ends = np.flatnonzero(spine != 1)
        if len(ends) and ends[0] >= 1 and np.all(spine[ends[0]:ends[0]+2] == -1) and ends[0]+1 < len(spine):
            self.rear = ticks[ends[0]]
        car_along = (car-self.anchor) @ self.forward
        car_cross = (car-self.anchor) @ self.left
        rejoined = abs(car_cross) <= 6. and self._sample(cells, car) == -1
        if car_along > max(seen_far, self.rear or 0.) + 8. and (self.rear is not None or rejoined):
            reason = 'passed_measured_rear' if self.rear is not None else 'rejoined_observed_axis'
            return self._release(patch, now_s, reason)
        # Estimate the observed near-face slope from its leading edge (absolute
        # lateral displacement is the independent variable in the pretrained WM).
        bins = np.floor(across/(4*r)).astype(int)
        edges = [(float(np.mean(abs(across[bins==i]))), float(np.min(along[bins==i]))) for i in np.unique(bins)]
        slope = 0.
        if len(edges) >= 3 and np.ptp(np.array(edges)[:,0]) > 3*r:
            slope = float(np.clip(np.polyfit(*np.array(edges).T,1)[0],-.5,.5))
        features, gates = [], {}
        for side in _SIDES:
            lower = max(0.,float(np.max(side*across))) if len(across) else 0.
            gate = None
            if self.rear is not None:
                # A measured corridor must be explicitly free over its full depth
                # and a three-cell lateral footprint. Never certify an unknown gap.
                offsets = np.arange(r,220.+r,r)
                depth_ticks = np.arange(-3*r,self.rear+4*r,r)
                center = self.anchor + offsets[:,None]*side*self.left
                strip = center[:,None,:] + depth_ticks[None,:,None]*self.forward
                free = np.ones(len(offsets),bool)
                for shift in (-r,0.,r):
                    free &= (self._sample(cells,strip+shift*self.left) == -1).all(1)
                lanes = np.flatnonzero(free)
                if len(lanes):
                    first = lanes[0]
                    end = first
                    while end+1 < len(free) and free[end+1]:
                        end += 1
                    gate = float(offsets[(first+end)//2])
                    gates[side] = gate
                    entry = self.anchor - 3*r*self.forward + side*gate*self.left
                    if self.verdict[side] != 'blocked' and self._entry_connected(car, entry, cells):
                        self.verdict[side] = 'open'
            if self.verdict[side] != 'blocked' and self._certified_cut(cells,side,self.rear):
                self.verdict[side] = 'blocked'
                self.denials[side] = dict(reason='air_observed_occupied_cut_to_mission_boundary',
                                          received_at=self.received_at, review_at=float(now_s))
            features.append([0.,side,width,slope,lower,float(gate is not None),gate or 0.,self.rear or 0.,
                             {'unknown':0.,'open':1.,'blocked':-1.}[self.verdict[side]]])
        # Received occupancy carries no water-vs-land class. Average the two
        # trained conditional priors without reading scenario type or bridge truth.
        features = np.asarray(features,dtype=np.float32)
        both_types = np.tile(features,(2,1))
        both_types[2:,0] = 1.
        pred = np.asarray(self.model.predict_sides(both_types),float)
        if pred.shape != (4,3) or not np.isfinite(pred).all():
            raise ValueError('Invalid received-obstacle prediction')
        pred = (pred[:2]+pred[2:])*.5
        self.prediction_calls += 1
        routes, observed, probs = {}, {}, {}
        for (offset,depth,prob),side,feat in zip(pred,_SIDES,features):
            delta = gates.get(side,max(feat[4],np.clip(offset,10.,220.)))
            far = max(2*r, self.rear if self.rear is not None else slope*delta+np.clip(depth,10.,160.))
            entry = self.anchor-12.*self.forward+side*delta*self.left
            exit_xy = self.anchor+(far+5.)*self.forward+side*delta*self.left
            reconnect = self.anchor+(far+37.)*self.forward
            routes[side] = np.array([entry,exit_xy,reconnect])
            # Even a measured lane does not certify a template point 12 m before
            # or 5 m past it. Require a received free cell at that location too.
            observed[side] = [side in gates and self._sample(cells,entry)==-1,
                              side in gates and self._sample(cells,exit_xy)==-1,False]
            probs[side] = float(np.clip(prob,0.,1.))
        # The vehicle can rejoin goalward travel from the committed flank,
        # far from the axis on which the first obstacle report was received.
        # The estimated exit supplies a conservative checkpoint together with
        # the observed extent. The corridor itself always needs measured free space.
        exit_station = max((route[1] - self.anchor) @ self.forward for route in routes.values())
        if (len(patch) and car_along > max(exit_station, seen_far + 8.)
                and self._clear_goal_corridor(car, goal, cells)):
            return self._release(patch, now_s, 'observed_downstream_goal_corridor')
        self.state = dict(active=True,obstacle_id=self.encounter_id,locked_side=self.locked_side,
                          routes=routes,route_observed=observed,route_open_probability=probs,
                          side_status=self.verdict.copy(),evidence_time=self.received_at,
                          switch_count=self.switch_count,closure_evidence=self.denials.copy())
        return self.state

    def _axis(self, car_xy):
        """World bearing nearest to the left proposal rather than right proposal."""
        car = np.asarray(car_xy,float)
        unit = []
        joins = []
        for side in _SIDES:
            entry, exit_xy, join = self.state['routes'][side]
            along = (car-entry) @ self.forward
            misaligned = abs((car-entry) @ self.left) > 4.
            if (along < 1. and np.linalg.norm(entry-car) > 4.) or (along < 7. and misaligned):
                point = entry
            elif (car-exit_xy) @ self.forward < -1.:
                point = exit_xy
            else:
                point = join
                joins.append(join)
            v = point-car
            unit.append(v/max(1.e-9,np.linalg.norm(v)))
        # Two sub-cell variants of the same downstream merge have no distinct
        # route side. Preserve the side identity, but do not split headings on
        # a numerical difference between these two predicted points.
        if len(joins) == 2 and np.linalg.norm(joins[0] - joins[1]) <= np.sqrt(2.) * self.res:
            return None
        diff = unit[0]-unit[1]
        return math.atan2(diff[1],diff[0]) if np.linalg.norm(diff)>1.e-8 else None

    def constrain_arcs(self, body_angle_arcs, car_xy, yaw_rad):
        arcs = np.asarray(body_angle_arcs,float).reshape(-1,2)
        if not self.state['active']:
            return arcs
        side = self.locked_side
        if not side or self.verdict[side]=='blocked':
            available = [s for s in _SIDES if self.verdict[s]!='blocked']
            if len(available)!=1:
                return arcs if available else np.empty((0,2))
            side = available[0]
        axis = self._axis(car_xy)
        if axis is None:
            return arcs
        center = axis - yaw_rad + (math.pi if side<0 else 0.)
        center = (center+math.pi)%math.tau-math.pi
        intervals = []
        for lo,hi in arcs:
            for k in (-1,0,1):
                c = center+k*math.tau
                a,b = max(lo,c-math.pi/2),min(hi,c+math.pi/2)
                if b-a > 4.e-6:
                    intervals.append((a+1.e-6,b-1.e-6))
        return np.asarray(sorted(intervals),float).reshape(-1,2)

    def commit(self, world_heading_rad, car_xy):
        if not self.state['active']:
            return 0
        axis = self._axis(car_xy)
        if axis is None:
            return self.locked_side
        score = math.cos(world_heading_rad-axis)
        requested = self.locked_side if abs(score)<1.e-9 and self.locked_side else (1 if score>=0 else -1)
        old = self.locked_side
        if self.verdict[requested]!='blocked' and (not old or old==requested or self.verdict[old]=='blocked'):
            self.locked_side = requested
            self.switch_count += int(bool(old and old!=requested))
        self.state.update(locked_side=self.locked_side,switch_count=self.switch_count)
        return self.locked_side
