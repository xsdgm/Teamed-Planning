"""One-step native clearance checks and short, receiver-local detours for the UGV."""
from collections import deque
import numpy as np

from vtemp2_3.ground_policy import ground_sensor_tile
from shared.ground.observed_grid import ObservedGrid
from survey.survey_fleet_clearance import native_segment_clear
from survey.survey_fleet_phase import endpoint_phase_clear


REPEATED_GRAPH_WAYPOINT = 'repeated_graph_waypoint'


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
        self.detour_direction = None
        self.observed_grid = None
        self.bypass_anchor = None
        self.graph_pursuits = deque(maxlen=85)
        self.pursuit_heading = None
        self.repeated_waypoint = None
        self.repeated_progress = None

    def _update_ground_memory(self, tile, car, res):
        guide = getattr(self.adapter, 'ground_guidance', None)
        if guide is None or tile is None:
            return None
        if self.observed_grid is None:
            case = getattr(self.adapter.ep, 'case', {})
            side = case.get('world_size')
            bounds = None if side is None else (0., 0., side, side)
            self.observed_grid = ObservedGrid(guide, resolution=res/2, half_width=44.,
                                              world_bounds=bounds, center=car)
        memory = self.observed_grid
        memory.recenter(car)
        # ObservedGrid takes semantic classes: unknown 0, traversable 1, obstacle 2.
        memory.observe(car, np.where(tile == 0, 1, np.where(tile == 1, 2, 0)), res)
        memory.goal = np.asarray(guide, dtype=float)
        return memory.goal

    def _repeat_without_progress(self, car, guide):
        previous = getattr(self, '_previous_loop', None)
        self._previous_loop = (car.copy(), guide.copy())
        if previous is None:
            return False
        origin, marker = previous
        axis = marker-origin
        current = guide-car
        if axis @ current <= 0.7*np.linalg.norm(axis)*np.linalg.norm(current):
            return False
        # If the previous native detour gained a vehicle sensor footprint in the
        # commanded direction, allow the native controller to handle this loop too.
        obs = self.adapter.exp.obs[self.adapter.exp.ground]
        tile = ground_sensor_tile(obs)
        span = min(tile.shape)*float(obs['map_resolution'])
        return (car-origin) @ axis < span*np.linalg.norm(axis)

    def _bypass_finished(self, car, guide):
        if guide is None:
            return True
        direction = self.bypass_anchor - self.follow_origin
        new_direction = guide-car
        # Release when the original commanded range has been covered, or a new
        # command diverges by more than 45 degrees. A same-bearing rolling target
        # leaves the chosen frontier intact through a local detour.
        return (np.dot(self.bypass_anchor-car, direction) <= 0.
                or np.dot(new_direction, direction) < np.linalg.norm(new_direction)
                * np.linalg.norm(direction) / np.sqrt(2.))

    def _frontier_target(self, tile, car, res, dt, controller, min_distance=.45):
        memory = self.observed_grid
        waypoint = memory.waypoint(car, 1.5)
        if waypoint is not None:
            point = np.asarray(waypoint)
            if (np.linalg.norm(point-car) > min_distance
                    and self._can_depart(tile, car, res, dt, controller, point)):
                return point
        guide = memory.frontier if memory.frontier is not None else memory.goal
        return self._choose(tile, car, res, dt, controller, graph_waypoint=guide,
                            min_distance=min_distance)

    def observed_frontier_segment(self, guidance):
        """Recheck a local hop when the coarse start cell cannot support the active detour."""
        if self.follow_reason != 'observed_frontier' or guidance is None:
            return None
        exp = self.adapter.exp
        obs = exp.obs[exp.ground]
        car = np.asarray(obs['self']['position'], dtype=float)[:2]
        if exp.goal_reached or self._bypass_finished(car, guidance):
            return None
        tile = ground_sensor_tile(obs)
        res, dt = float(obs['map_resolution']), float(obs['time_step'])
        if not native_segment_clear(tile, car, res, car, car):
            return None
        controller = exp.ground_controller.local_planner
        threshold = exp.ground_controller.waypoint_threshold
        point = self.target
        if (point is None or np.linalg.norm(point-car) <= (.2 if self.target_reason == 'observed_connector' else threshold)
                or not self._can_depart(tile, car, res, dt, controller, point)):
            point = self._frontier_target(tile, car, res, dt, controller, min_distance=threshold)
        if point is None:
            # The native goal tolerance can exceed the full certified length at
            # a narrow gap. The guard itself steers and rechecks this shorter hop.
            point = self._choose(tile, car, res, dt, controller, graph_waypoint=guidance,
                                 short_hops=True)
        if point is None:
            return None
        self.target = point.copy()
        self.target_reason = 'observed_connector'
        return np.stack((car, point))

    def _can_depart(self, tile, car, res, dt, controller, point):
        if not native_segment_clear(tile, car, res, car, point):
            return False
        delta = point-car
        heading = np.arctan2(delta[1], delta[0])
        speed, _ = controller.compute_velocity_command(car, heading, point, None, res)
        return self._safe_speed(tile, car, res, delta/np.linalg.norm(delta), speed, dt) > 0.

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

    def _choose(self, tile, car, res, dt, controller, *, route_only=False, graph_waypoint=None,
                min_distance=0., short_hops=False):
        path = self.adapter.ground_path_xy
        limit = np.asarray(tile.shape[::-1])*res / 2.
        if graph_waypoint is None:
            if path is None or len(path) < 2:
                return None
            nearest = np.argmin(np.linalg.norm(path - car, axis=1))
            guide = path[min(nearest + 5, len(path) - 1)]
            points = list(path[nearest:nearest+7])
        else:
            # A rejected graph edge supplies a bearing, never clearance.
            # Only the short segment inside the present vehicle tile is executable.
            guide = np.asarray(graph_waypoint, dtype=float)[:2]
            if guide.shape != (2,) or not np.isfinite(guide).all():
                return None
            direction = guide-car
            span = np.linalg.norm(direction)
            if span <= self.adapter.exp.ground_controller.waypoint_threshold:
                return None
            unit = direction/span
            extent = np.min((limit-1.5-res/2) / np.maximum(np.abs(unit),1.e-12))
            points = [car + unit*min(span,extent)] if extent > 0. else []
        # Route points take precedence at equal cost. Neighbouring native-grid
        # alternatives allow a sideways move when the coarse path grazes a wall.
        for steps in ((1.5, 1.) if short_hops else (4, 3, 5, 2)):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx or dy:
                        points.append(car + res*steps*np.array([dx, dy]))
        recent = [p for t,p in self.recent if self.adapter.now-t <= 45.]
        def score(point):
            penalty = sum(2.*max(0., 1.5 - np.linalg.norm(point-p)) for p in recent)
            if graph_waypoint is not None and self.detour_direction is not None:
                # Discourage an immediate reversal in the next sensor footprint.
                penalty += max(0., -np.dot(point-car,self.detour_direction))
            return np.linalg.norm(point-guide) + penalty
        ordered = (path[nearest+1:nearest+8][::-1] if route_only else sorted(points, key=score))
        for point in ordered:
            delta = point-car
            length = np.linalg.norm(delta)
            if length < (.2 if short_hops else .8) or length <= min_distance or np.any(np.abs(delta) + 1.5 >= limit):
                continue
            if self._can_depart(tile, car, res, dt, controller, point):
                if graph_waypoint is not None:
                    self.detour_direction = delta/length
                return point.copy()
        return None

    def _track(self, tile, car, res, dt, yaw, controller):
        delta = self.target-car
        distance = np.linalg.norm(delta)
        if distance <= (.45 if self.target_reason in ('frontier_wait','route_loop','observed_frontier',
                                                      REPEATED_GRAPH_WAYPOINT) else .2):
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

    def _observed_alternative(self, tile, car, res, dt, controller):
        native = self.adapter.exp.ground_controller
        planner = native.global_planner
        waypoint = planner.get_current_waypoint() if getattr(planner, 'using_graph', False) else None
        if waypoint is None:
            waypoint = getattr(native, 'goal_position', None)
        if waypoint is None:
            return None
        return self._choose(tile, car, res, dt, controller, graph_waypoint=waypoint)

    def _record_motion(self, car, waiting, speed):
        now = self.adapter.now
        step = np.linalg.norm(car-self.poses[-1][1]) if self.poses else 0.
        self.travel += step
        distance = float(np.linalg.norm(self.adapter.ep.goal-car))
        self.poses.append((now, car.copy(), self.travel, distance))
        self.wait_steps = self.wait_steps + 1 if waiting and not speed and step < .05 else 0

    def _forward_receiver_route(self, car, guide, margin):
        path = self.adapter.ground_path_xy
        return (guide is not None and getattr(self.adapter, 'last_ground_reason', None) == 'native_shared_guidance'
                and path is not None and len(path) > 1
                and np.linalg.norm(guide-car) - np.linalg.norm(guide-path[-1]) > margin)

    def _still_rejecting_waypoint(self, car, guide, footprint):
        if self._bypass_finished(car, guide) or not self._forward_receiver_route(car, guide, 0.):
            return False
        driver = self.adapter.exp.ground_controller
        planner = driver.global_planner
        current = planner.get_current_waypoint() if planner.using_graph and not planner.waiting_at_frontier else None
        if (current is None or np.linalg.norm(self.repeated_waypoint - current) > driver.waypoint_threshold
                or (self.repeated_waypoint - car) @ (guide - car) >= 0.):
            return False
        # The shared waypoint rolls every review. Measure actual vehicle gains
        # against the fixed command from the start of this intervention instead.
        time, remaining = self.repeated_progress
        if self.adapter.now - time >= 120.:
            distance = float(np.linalg.norm(self.bypass_anchor - car))
            if remaining - distance < footprint / 2:
                return False
            self.repeated_progress = (self.adapter.now, distance)
        return True

    def _repeated_graph_waypoint(self, car, guide, footprint):
        """Recognize an earlier pursuit of the same road node after driving out and back."""
        if guide is None:
            self.graph_pursuits.clear()
            return None
        direction = guide-car
        length = np.linalg.norm(direction)
        if length == 0.:
            self.graph_pursuits.clear()
            return None
        direction = direction / length
        if self.pursuit_heading is None or self.pursuit_heading @ direction < 1. / np.sqrt(2.):
            self.graph_pursuits.clear()
            self.pursuit_heading = direction
        if self.adapter.now % 5:
            return None
        driver = self.adapter.exp.ground_controller
        planner = driver.global_planner
        waypoint = None
        if (self.follow_reason is None and self.target is None and getattr(planner, 'using_graph', False)
                and not planner.waiting_at_frontier
                and getattr(self.adapter, 'last_ground_reason', None) == 'native_shared_guidance'):
            node = planner.get_current_waypoint()
            if node is not None:
                waypoint = np.asarray(node, dtype=float).copy()
        repeated = None
        if (waypoint is not None and (waypoint-car) @ direction < 0.
                and self._forward_receiver_route(car, guide, footprint / 2)):
            distance = np.linalg.norm(waypoint-car)
            closest = distance
            changed_node = False
            for prior, old_travel, old_waypoint in reversed(self.graph_pursuits):
                old_distance = np.linalg.norm(waypoint-prior)
                closest = min(closest, old_distance)
                if old_waypoint is None:
                    continue
                if np.linalg.norm(old_waypoint-waypoint) > driver.waypoint_threshold:
                    changed_node = True
                elif (changed_node and (old_waypoint-prior) @ direction < 0.
                      and self.travel-old_travel >= 20. and min(distance, old_distance)-closest >= footprint
                      and np.linalg.norm(prior-car) <= footprint / 2):
                    repeated = waypoint
                    break
        self.graph_pursuits.append((car.copy(), self.travel, waypoint))
        return repeated

    def _handoff_reason(self, car):
        if self.wait_steps >= 3:
            return 'frontier_wait'
        if self.adapter.now % 5 or len(self.poses) < 61:
            return None
        now = self.adapter.now
        recent = [r for r in self.poses if now-r[0] <= 60]
        guide = getattr(self.adapter, 'ground_guidance', None)
        distances = ([np.linalg.norm(r[1]-guide) for r in recent] if guide is not None
                     else [r[3] for r in recent])
        if distances[0] - min(distances) >= 3.:
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
        guide = self._update_ground_memory(tile, car, res)
        speed, turn = motion
        if exp.goal_reached or self.adapter.ep.holding:
            self.target = None
            self.graph_pursuits.clear()
            # An unconnected coarse start may acquire an executable leg at the
            # next replan. Stay still and retain only the established intent.
            retry = (not exp.goal_reached and self.follow_reason == 'observed_frontier'
                     and self.adapter.last_ground_reason == 'uncertified_vehicle_clearance'
                     and not self._bypass_finished(car, guide))
            if not retry:
                self.follow_reason = None
            self.detour_direction = None
            self.poses.clear()
            self.wait_steps = 0
            return 0., turn
        if not native_segment_clear(tile, car, res, car, car):
            self.invalid_start += 1
            self.target = None
            self.follow_reason = None
            self.detour_direction = None
            self.poses.clear()
            self.graph_pursuits.clear()
            self.wait_steps = 0
            return 0., turn
        controller = exp.ground_controller.local_planner
        waiting = exp.ground_controller.global_planner.waiting_at_frontier
        self._record_motion(car, waiting, speed)
        footprint = min(tile.shape) * res
        repeated_node = self._repeated_graph_waypoint(car, guide, footprint)
        if self.follow_reason == REPEATED_GRAPH_WAYPOINT and not self._still_rejecting_waypoint(car, guide, footprint):
            self.target = None
            self.follow_reason = None
            self.graph_pursuits.clear()
        if self.follow_reason == 'observed_frontier' and self._bypass_finished(car, guide):
            self.target = None
            self.follow_reason = None
            self.observed_grid.frontier = None
            self.poses.clear()
            self.wait_steps = 0
            self.detour_direction = None
        if self.target is not None:
            correction = self._track(tile, car, res, dt, yaw, controller)
            if correction is not None:
                self.detour_steps += 1
                return correction
            self.recent.append((self.adapter.now, self.target))
            self.target = None
        if self.follow_reason not in (None, 'observed_frontier', REPEATED_GRAPH_WAYPOINT) and (
                (self.follow_reason == 'frontier_wait' and not waiting)
                or np.linalg.norm(car-self.follow_origin) >= 20.
                or self.adapter.now-self.follow_started >= 120.):
            self.follow_reason = None
            self.detour_direction = None
            self.poses.clear()
            self.wait_steps = 0
        if self.follow_reason is None:
            self.follow_reason = REPEATED_GRAPH_WAYPOINT if repeated_node is not None else self._handoff_reason(car)
            if self.follow_reason:
                self.follow_origin = car.copy()
                self.follow_started = self.adapter.now
                if self.follow_reason == REPEATED_GRAPH_WAYPOINT:
                    self.bypass_anchor = guide.copy()
                    self.repeated_waypoint = repeated_node.copy()
                    self.repeated_progress = (self.adapter.now, float(np.linalg.norm(guide - car)))
                if self.follow_reason == 'route_loop' and guide is not None and self._repeat_without_progress(car, guide):
                    self.bypass_anchor = guide.copy()
                    self.observed_grid.frontier = None
                    self.follow_reason = 'observed_frontier'
        self.target_reason = self.follow_reason
        if self.follow_reason == 'observed_frontier':
            self.target = self._frontier_target(tile, car, res, dt, controller)
        elif self.follow_reason is not None:
            self.target = self._choose(tile, car, res, dt, controller, route_only=True)
            if self.target is None and self.follow_reason == REPEATED_GRAPH_WAYPOINT:
                self.follow_reason = self.target_reason = None
                self.graph_pursuits.clear()
            if self.target is None and self.follow_reason == 'frontier_wait':
                self.target = self._observed_alternative(tile, car, res, dt, controller)
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
                self.target = self._observed_alternative(tile, car, res, dt, controller)
        if self.target is None:
            return 0., turn
        self.detours += 1
        self.recent.append((self.adapter.now, car.copy()))
        event = dict(t=self.adapter.now, position=car.tolist(), target=self.target.tolist(), reason=self.target_reason)
        if repeated_node is not None and self.target_reason == REPEATED_GRAPH_WAYPOINT:
            event.update(graph_waypoint=repeated_node.tolist(), receiver_endpoint=self.adapter.ground_path_xy[-1].tolist())
        self.events.append(event)
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
