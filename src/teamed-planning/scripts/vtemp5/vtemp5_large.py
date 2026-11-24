"""Two large-obstacle local missions with delayed, directed terrain observations.

'detour' is one bounded land obstacle; 'bridge' is an unfordable river between
two headlands with exactly one bridge. Static physical geometry is accessible to
the simulator and evaluation only. The procedural observation model follows a
commanded survey swath along the selected bank. All learned inference uses the
delivered swath, mouth and crossing reports.
"""
import math

import numpy as np

from vtemp5.vtemp5_env import SIDES, VTemp5Env


class VTemp5LargeObstacleEnv(VTemp5Env):
    def __init__(self, seed, *, terrain, predictor=None, step_m=6.209, horizon=144,
                 wait_probability=.02889, blocked_side=None):
        if terrain not in ('detour', 'bridge'):
            raise ValueError("terrain must be detour or bridge")
        if terrain == 'detour' and blocked_side not in (None, 0):
            raise ValueError("the land obstacle is traversable on both sides")
        if terrain == 'bridge' and blocked_side == 0:
            raise ValueError("there is only one bridge")
        super().__init__(seed, step_m=step_m, horizon=horizon, wait_probability=wait_probability)
        self._terrain = terrain
        self._predictor = predictor
        size = self.rng.random()
        self._front = float(self.rng.uniform(70., 95.))
        self._slope = float(.05 + .11 * size + self.rng.uniform(-.015, .015))
        self._depth = float(52. + 34. * size + self.rng.uniform(-2., 7.))
        # For a land obstacle the candidate vehicle lane is 8 m outside its edge.
        # A closed river branch has a shore candidate in front of a headland.
        self._gate_y = {s: s * float(108. + 41. * size + self.rng.uniform(-7., 16.)) for s in SIDES}
        drawn_side = int(self.rng.choice(SIDES))
        self._blocked_side = 0 if terrain == 'detour' else (drawn_side if blocked_side is None else blocked_side)
        self._far = self._front + self._depth + self._slope * max(map(abs, self._gate_y.values()))
        self.goal = (float(self.rng.uniform(750., 950.)), 0.)
        self._coverage = dict.fromkeys(SIDES, 0.)
        self._visible_width = 0.
        self._visible_slope = 0.
        self._river_reported = False
        self._predicted_step = None
        self._route_cache = {}
        self._fit_cache = {}

    def terrain_traversable(self, x, y):
        """Physical static truth for collision and audits; unavailable in policy inputs."""
        if x < -80. or abs(y) > 225.:
            return False
        side = 1 if y >= 0 else -1
        lateral = abs(y)
        lane = abs(self._gate_y[side])
        if self._terrain == 'bridge' and lateral > lane + 22.:
            return False
        face = self._front + self._slope * lateral
        if not face <= x <= face + self._depth:
            return True
        if self._terrain == 'detour':
            return lateral >= lane - 8.
        return side != self._blocked_side and abs(lateral - lane) <= 8.

    def _driveable(self, x, y):
        if not self.terrain_traversable(x, y):
            return False
        face = self._front + self._slope * abs(y)
        return not (self.locked_side and face <= x <= face + self._depth and y * self.locked_side < 0.)

    def _features(self, side):
        known = self._observed_gates.get(side)
        verdict = {'open': 1., 'blocked': -1., 'unknown': 0.}[self.side_status[side]]
        return np.array([self._river_reported, side, self._visible_width, self._visible_slope,
                         self._coverage[side], known is not None, 0. if known is None else abs(known),
                         self._observed_far.get(side, 0.), verdict], dtype=np.float32)

    def training_example(self, side):
        """Labels are delayed observations. An unsurveyed value is missing, even at episode end."""
        if side not in SIDES:
            raise ValueError('invalid side')
        if not self.detected:
            return None
        status = self.side_status[side]
        target = [abs(self._observed_gates.get(side, math.nan)),
                  self._observed_far.get(side, math.nan),
                  {'open': 1., 'blocked': 0., 'unknown': math.nan}[status]]
        return self._features(side), np.asarray(target, dtype=np.float32)

    def _side_predictions(self):
        if self._predictor is None:
            # Prior for collecting the first supervised dataset before model fitting.
            return [(132., 74., .5)] * len(SIDES)
        features = np.stack([self._features(side) for side in SIDES])
        if hasattr(self._predictor, 'predict_sides'):
            predictions = self._predictor.predict_sides(features)
        else:
            # Custom single-side predictors retain their original calling convention.
            predict = getattr(self._predictor, 'predict', self._predictor)
            predictions = [predict(row) for row in features]
        predictions = np.asarray(predictions, dtype=float)
        if predictions.shape != (len(SIDES), 3) or not np.isfinite(predictions).all():
            raise ValueError('invalid learned local obstacle prediction')
        return predictions

    def _routes(self):
        if not self.detected:
            return {}
        if self._predicted_step != self.elapsed_steps:
            front = self._observed_front
            for side, (offset, depth, probability) in zip(SIDES, self._side_predictions()):
                offset = max(self._coverage[side], float(np.clip(offset, 15., 220.)))
                y = self._observed_gates.get(side, side * offset)
                depth = self._observed_far.get(side, float(np.clip(depth, 12., 150.)))
                status = self.side_status[side]
                self._fit_cache[side] = (float(np.clip(probability, 0., 1.))
                                         if status == 'unknown' else float(status == 'open'))
                far = front + self._visible_slope * abs(y) + depth
                self._route_cache[side] = [(front - 12., y), (far + 5., y), (far + 37., 0.)]
            self._predicted_step = self.elapsed_steps
        return dict(self._route_cache)

    def observe(self):
        obs = super().observe()
        obs['route_open_probability'] = self._fit_cache.copy()
        return obs

    def _survey(self, world_heading, side):
        if not self.detected:
            if not self._in_view((self._front, 0.), world_heading, 130., math.radians(80.)):
                return
            # A short central frontage is visible on this first scan. No endpoint,
            # obstacle thickness, or bridge side has been explored yet.
            width = min(65., max(24., 2.*(108. - math.dist(self.position, (self._front, 0.)))))
            key = ('front', 0)
            data = (self._front, width, self._slope, self._terrain == 'bridge')
        elif side not in self._observed_gates:
            start = self._coverage[side]
            edge = abs(self._gate_y[side])
            end = min(start + 26., edge)
            # The commanded survey only scans the next connected swath on this
            # side. The position of the actual edge determines when it is seen.
            midpoint = (start + end) / 2.
            probe = (self._front + self._slope * midpoint, side * midpoint)
            if not self._in_view(probe, world_heading, 92., math.radians(82.)):
                return
            key = ('mouth' if end >= edge else 'swath', side, end)
            data = end
        else:
            # This last survey spans the river cross-section at the candidate and
            # the remaining strip to the headland. The paired physical map has
            # the same headlands whether its bridge is placed left or right.
            near_side = self._gate_y[side]
            probe = (self._front + self._slope*abs(near_side) + self._depth/2., near_side)
            if not self._in_view(probe, world_heading, 105., math.radians(82.)):
                return
            key = ('crossing', side)
            data = ('blocked' if side == self._blocked_side else 'open', self._depth)
        if key not in self._sent:
            self._sent.add(key)
            self._packets.append((self.elapsed_steps + self._delay, key, data))

    def _deliver(self):
        delivered = []
        while self._packets and self._packets[0][0] <= self.elapsed_steps:
            _, key, data = self._packets.pop(0)
            kind, side = key[:2]
            delivered.append((kind, side))
            if kind == 'front':
                self._observed_front, self._visible_width, self._visible_slope, self._river_reported = data
                for s in SIDES:
                    self._coverage[s] = self._visible_width/2.
            elif kind in ('swath', 'mouth'):
                self._coverage[side] = data
                if kind == 'mouth':
                    self._observed_gates[side] = side * data
            else:
                self.side_status[side], self._observed_far[side] = data
                if self.side_status[side] == 'blocked':
                    self.blocked_evidence_step = self.elapsed_steps
        return delivered

    def step(self, offset_rad):
        obs, reward, done, info = super().step(offset_rad)
        # The new local task is hundreds of metres long. Retain time/success
        # terms and normalize the same signed progress by a fixed 400 m.
        reward -= 10. * info['progress_m'] * (1./100. - 1./400.)
        return obs, reward, done, info
