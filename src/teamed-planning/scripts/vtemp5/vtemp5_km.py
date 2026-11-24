"""Continuous kilometre navigation with two independently surveyed large obstacles."""
import math

import numpy as np

from vtemp5.vtemp5_env import VTemp5Env
from vtemp5.vtemp5_large import VTemp5LargeObstacleEnv


class _CourseObstacle(VTemp5LargeObstacleEnv):
    """A second encounter may start from any lateral position on the course."""

    def _survey(self, world_heading, side):
        sent_before = len(self._sent)
        super()._survey(world_heading, side)
        if len(self._sent) != sent_before:
            return  # The current directional sweep already acquired a report.
        if not self.detected:
            # From an offset approach the central face can be out of range while
            # the face directly ahead is visible. Measure a local patch there.
            key = ('front', 0)
            y = self.position[1]
            face = self._front + self._slope * abs(y)
            if (key in self._sent or self.terrain_traversable(face + 1., y)
                    or not self._in_view((face, y), world_heading, 105., math.radians(80.))):
                return
            distance = math.dist(self.position, (face, y))
            width = min(65., max(24., 2. * (108. - distance)))
            data = (face - self._slope * abs(y), width, self._slope, self._terrain == 'bridge')
        elif side not in self._observed_gates:
            # The commanded side's own edge can be seen directly without first
            # scanning every intermediate strip from the centre of the map.
            edge = abs(self._gate_y[side])
            key = ('mouth', side, edge)
            near = (self._front + self._slope * edge - 5., self._gate_y[side])
            if key in self._sent or (math.dist(self.position, near) > 26. and
                                      not self._in_view(near, world_heading, 55., math.radians(82.))):
                return
            data = edge
        else:
            return
        self._sent.add(key)
        self._packets.append((self.elapsed_steps + self._delay, key, data))


class VTemp5KilometerEnv(VTemp5Env):
    """Share one vehicle, command history and clock across the whole mission.

    Each obstacle owns its static geometry, pending packets and local belief. The
    only obstacle exposed in a policy observation is the currently reported one.
    """
    arrival_radius_m = 2.

    def __init__(self, seed, *, terrain='detour', predictor=None,
                 min_distance_m=1000., max_distance_m=1500., step_m=6.209,
                 horizon=650, wait_probability=.02889, blocked_side=None):
        if (not math.isfinite(min_distance_m) or not math.isfinite(max_distance_m)
                or not 1000. <= min_distance_m <= max_distance_m):
            raise ValueError('Expected an ordered kilometre distance range starting at 1000 m or more')
        if terrain not in ('detour', 'bridge'):
            raise ValueError('Expected detour or bridge curriculum')
        if terrain == 'detour' and blocked_side not in (None, 0):
            raise ValueError('Both sides of the land obstacles must be open')
        motion_seed, layout_seed, *obstacle_seeds = np.random.SeedSequence(seed).spawn(4)
        super().__init__(motion_seed, step_m=step_m, horizon=horizon, wait_probability=wait_probability)
        layout = np.random.default_rng(layout_seed)
        self.origin = self.position
        self.goal_distance_m = float(layout.uniform(min_distance_m, max_distance_m))
        self.goal = (self.goal_distance_m, 0.)
        self.travelled_m = 0.
        self.obstacles_passed = 0
        self._encounter_origin = self.origin
        self._scenes = []
        for i, fraction in enumerate((layout.uniform(.20, .28), layout.uniform(.60, .68))):
            scene = _CourseObstacle(obstacle_seeds[i], terrain='detour' if i == 0 else terrain,
                                    predictor=predictor, blocked_side=None if i == 0 else blocked_side)
            shift = self.goal_distance_m * fraction - scene._front
            scene._front += shift
            scene._far += shift
            self._scenes.append(scene)
        self._select_next_obstacle()

    def _select_next_obstacle(self):
        self._active = (self._scenes[self.obstacles_passed]
                        if self.obstacles_passed < len(self._scenes) else None)
        self.locked_side = 0
        self.blocked_evidence_step = None
        # Base step() and observe() consult these same delivered observations.
        self.side_status = self._active.side_status if self._active else {1: 'unknown', -1: 'unknown'}
        self._observed_gates = self._active._observed_gates if self._active else {}

    @property
    def detected(self):
        return self._active is not None and self._active.detected

    def _routes(self):
        return self._active._routes() if self.detected else {}

    def observe(self):
        public = super().observe()
        public['obstacle_id'] = self.obstacles_passed + 1 if self.detected else None
        public['route_open_probability'] = self._active._fit_cache.copy() if self.detected else {}
        return public

    def _survey(self, world_heading, side):
        if self._active is not None:
            self._active.position = self.position
            self._active.elapsed_steps = self.elapsed_steps
            self._active._survey(world_heading, side)

    def _deliver(self):
        scene = self._active
        if scene is None:
            return []
        was_detected = scene.detected
        scene.elapsed_steps = self.elapsed_steps
        delivered = scene._deliver()
        self.blocked_evidence_step = scene.blocked_evidence_step
        if not was_detected and scene.detected:
            self._encounter_origin = self.position
        if self.position[0] >= scene._far + 7.:
            self.obstacles_passed += 1
            self._select_next_obstacle()
        return delivered

    def _reference_path(self, route):
        if not self.locked_side:
            return [self.goal]
        return [self._encounter_origin, *route, self.goal]

    def terrain_traversable(self, x, y):
        """Collision truth for the entire course, including obstacles behind the car."""
        if x < -80. or x > self.goal[0] + 80. or abs(y) > 225.:
            return False
        for scene in self._scenes:
            face = scene._front + scene._slope * abs(y)
            if face <= x <= face + scene._depth and not scene.terrain_traversable(x, y):
                return False
        return True

    def _driveable(self, x, y):
        if not self.terrain_traversable(x, y):
            return False
        if self._active is not None and self.locked_side * y < 0.:
            face = self._active._front + self._active._slope * abs(y)
            return not face <= x <= face + self._active._depth
        return True

    def _arrived(self):
        return math.dist(self.position, self.goal) <= self.arrival_radius_m

    def _move(self, heading):
        # The goal is in the open final leg. If the selected straight command
        # crosses its arrival disk, stop at the closest point on that command.
        travel_m = self.step_m
        gx, gy = self.goal[0] - self.position[0], self.goal[1] - self.position[1]
        if math.hypot(gx, gy) <= self.step_m + self.arrival_radius_m:
            c, s = math.cos(heading), math.sin(heading)
            along = min(self.step_m, max(0., gx*c + gy*s))
            if math.hypot(gx - along*c, gy - along*s) <= self.arrival_radius_m:
                travel_m = along
        path_m = super()._move(heading, travel_m)
        self.travelled_m += path_m
        return path_m

    def step(self, offset_rad):
        passed = self.obstacles_passed
        obstacle_id = passed + 1 if self.detected else None
        public, _, done, info = super().step(offset_rad)
        # Normalize by this mission's initial distance; only physical goal
        # arrival receives the terminal reward.
        reward = 10.*info['progress_m'] / self.goal_distance_m - 1./self.horizon + 10.*info['success']
        info.update(obstacle_id=obstacle_id,
                    obstacle_passed_id=self.obstacles_passed if self.obstacles_passed > passed else None,
                    obstacles_passed=self.obstacles_passed, obstacles_total=len(self._scenes),
                    origin=self.origin, goal_distance_m=self.goal_distance_m,
                    remaining_m=math.dist(self.position, self.goal), travelled_m=self.travelled_m,
                    arrival_radius_m=self.arrival_radius_m)
        return public, reward, done, info

    def map_geometry(self):
        """Privileged map data for post-run route and collision audits."""
        return [dict(obstacle_id=i, terrain=scene._terrain, front_m=scene._front,
                     far_m=scene._far, slope=scene._slope, depth_m=scene._depth,
                     lanes_y_m=dict(scene._gate_y), blocked_side=scene._blocked_side)
                for i, scene in enumerate(self._scenes, 1)]
