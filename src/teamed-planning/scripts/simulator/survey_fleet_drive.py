"""One-step native clearance checks and short, receiver-local detours for the UGV."""
from collections import deque
import numpy as np

from ground.ground_policy import ground_sensor_tile
from ground.survey_fleet_clearance import native_segment_clear
from simulator.survey_fleet_phase import endpoint_phase_clear


class GroundMotionGuard:
    def __init__(self, adapter):
        self.adapter = adapter
        self.target = None
        self.recent = deque(maxlen=8)
        self.requested = self.vetoes = self.detours = self.invalid_start = 0
        self.detour_steps = 0
        self.events = []
        self.poses = deque(maxlen=421)
        self.travel = 0.
        self.wait_steps = 0
        self.follow_reason = None
        self.follow_origin = None
        self.follow_started = None
        self.target_reason = None

    def _safe_speed(self, tile, car, resolution, direction, speed, dt):
        if speed <= 0.:
            return 0.
        # Lattice-aligned steps preserve the native pixel phase in narrow areas.
        # Never increase the vehicle controller's requested translation.
        component = np.abs(direction[np.abs(direction) > 1.e-8])
        options = [speed]
        if component.size and np.ptp(component) < 1.e-8:
            aligned = resolution / (component[0] * dt)
            if aligned <= speed + 1.e-10:
                options.append(min(speed, aligned))
        options += [speed / 2, speed / 4]
        for candidate in options:
            end = car + direction * candidate * dt
            if endpoint_phase_clear(tile, car, resolution, car, end):
                return candidate
        return 0.

    def _choose(self, tile, car, res, dt, controller, *, route_only=False):
        path = self.adapter.ground_path_xy
        if path is None or len(path) < 2:
            return None
        nearest = np.argmin(np.linalg.norm(path - car, axis=1))
        guide = path[min(nearest + 5, len(path) - 1)]
        # Route points take precedence at equal cost. Neighbouring native-grid
        # alternatives allow a sideways move when the coarse path grazes a wall.
        points = list(path[nearest:nearest+7])
        for steps in (4, 3, 5, 2):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx or dy:
                        points.append(car + res*steps*np.array([dx, dy]))
        recent = [p for t,p in self.recent if self.adapter.now-t <= 45.]
        def score(point):
            penalty = sum(2.*max(0., 1.5 - np.linalg.norm(point-p)) for p in recent)
            return np.linalg.norm(point-guide) + penalty
        limit = np.asarray(tile.shape[::-1])*res / 2.
        ordered = (path[nearest+1:nearest+8][::-1] if route_only else sorted(points, key=score))
        for point in ordered:
            delta = point-car
            length = np.linalg.norm(delta)
            if length < .8 or np.any(np.abs(delta) + 1.5 >= limit):
                continue
            if not native_segment_clear(tile, car, res, car, point):
                continue
            heading = np.arctan2(delta[1], delta[0])
            speed, _ = controller.compute_velocity_command(car, heading, point, None, res)
            if self._safe_speed(tile, car, res, delta/length, speed, dt):
                return point.copy()
        return None

    def _track(self, tile, car, res, dt, yaw, controller):
        delta = self.target-car
        distance = np.linalg.norm(delta)
        if distance <= (.45 if self.target_reason in ('frontier_wait','route_loop') else .2):
            return None
        if not native_segment_clear(tile, car, res, car, self.target):
            return None
        bearing = np.arctan2(delta[1],delta[0])
        error = np.arctan2(np.sin(bearing-yaw), np.cos(bearing-yaw))
        speed, turn = controller.compute_velocity_command(car, yaw, self.target, None, res)
        if abs(error) > 1.e-8:
            # Same native proportional steering law with the V9 sampled-angle cap.
            return 0., float(np.clip(turn, -abs(error)/dt, abs(error)/dt))
        speed = self._safe_speed(tile, car, res, np.array([np.cos(yaw),np.sin(yaw)]), speed, dt)
        return (speed, 0.) if speed > 0. else None

    def _record_motion(self, car, waiting, speed):
        now = self.adapter.now
        step = np.linalg.norm(car-self.poses[-1][1]) if self.poses else 0.
        self.travel += step
        distance = float(np.linalg.norm(self.adapter.ep.goal-car))
        self.poses.append((now, car.copy(), self.travel, distance))
        self.wait_steps = self.wait_steps + 1 if waiting and not speed and step < .05 else 0

    def _handoff_reason(self, car):
        if self.wait_steps >= 3:
            return 'frontier_wait'
        if self.adapter.now % 5 or len(self.poses) < 61:
            return None
        now = self.adapter.now
        recent = [r for r in self.poses if now-r[0] <= 60]
        if recent[0][3] - min(r[3] for r in recent) >= 3.:
            return None
        # A legitimate detour may travel away from the goal. Declare a loop
        # only after returning to a measured earlier position by a long route.
        for t, prior, old_travel, _ in self.poses:
            if now-t < 40.:
                break
            if self.travel-old_travel >= 20. and np.linalg.norm(prior-car) < 2.:
                return 'route_loop'
        return None

    def motion(self, motion):
        exp = self.adapter.exp
        obs = exp.obs[exp.ground]
        car = np.asarray(obs['self']['position'], dtype=float)[:2]
        yaw = float(obs['self']['orientation'])
        dt, res = float(obs['time_step']), float(obs['map_resolution'])
        tile = ground_sensor_tile(obs)
        speed, turn = motion
        if exp.goal_reached or self.adapter.ep.holding:
            self.target = None
            self.follow_reason = None
            self.poses.clear()
            self.wait_steps = 0
            return 0., turn
        if not native_segment_clear(tile, car, res, car, car):
            self.invalid_start += 1
            self.target = None
            self.follow_reason = None
            self.poses.clear()
            self.wait_steps = 0
            return 0., turn
        controller = exp.ground_controller.local_planner
        waiting = exp.ground_controller.global_planner.waiting_at_frontier
        self._record_motion(car, waiting, speed)
        if self.target is not None:
            correction = self._track(tile, car, res, dt, yaw, controller)
            if correction is not None:
                self.detour_steps += 1
                return correction
            self.recent.append((self.adapter.now, self.target))
            self.target = None
        if self.follow_reason is not None and (
                (self.follow_reason == 'frontier_wait' and not waiting)
                or np.linalg.norm(car-self.follow_origin) >= 20.
                or self.adapter.now-self.follow_started >= 120.):
            self.follow_reason = None
            self.poses.clear()
            self.wait_steps = 0
        if self.follow_reason is None:
            self.follow_reason = self._handoff_reason(car)
            if self.follow_reason:
                self.follow_origin = car.copy()
                self.follow_started = self.adapter.now
        self.target_reason = self.follow_reason
        if self.follow_reason is not None:
            self.target = self._choose(tile, car, res, dt, controller, route_only=True)
        if self.target is None:
            if not speed:
                return motion
            self.requested += 1
            direction = np.array([np.cos(yaw),np.sin(yaw)])
            if endpoint_phase_clear(tile, car, res, car, car + speed*dt*direction):
                return motion
            self.vetoes += 1
            self.target_reason = 'clearance_veto'
            self.target = self._choose(tile, car, res, dt, controller)
        if self.target is None:
            return 0., turn
        self.detours += 1
        self.recent.append((self.adapter.now, car.copy()))
        self.events.append(dict(t=self.adapter.now, position=car.tolist(), target=self.target.tolist(),
                                reason=self.target_reason))
        correction = self._track(tile, car, res, dt, yaw, controller)
        if correction is None:
            self.target = None
            return 0., turn
        self.detour_steps += 1
        return correction

    def stats(self):
        return dict(native_motion_requests=self.requested, native_motion_vetoes=self.vetoes,
                    local_detours=self.detours, detour_control_steps=self.detour_steps,
                    uncertified_start_steps=self.invalid_start, detour_selections=self.events)
