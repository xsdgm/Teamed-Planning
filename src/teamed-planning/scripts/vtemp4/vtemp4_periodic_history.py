"""Periodic continuous-angle designations on the 200 m ring, with motion-history attention."""
from collections import deque
import math

import numpy as np
import torch
from torch import nn

from survey.survey_fleet_observation import START_CONNECTOR, _clearance
from survey.survey_fleet_runtime import GOAL_TOLERANCE_M, _same
from vtemp4.vtemp4_observation import ring_points
from vtemp4.vtemp4_ring_contract import RING_PERIODIC_HISTORY_FORMAT, RING_UNIFIED_GOAL_FORMAT, ring_action_contract
from vtemp4.vtemp4_ring_heading import RingHeadingFleetEpisode
from vtemp4.vtemp4_ring_history import ContinuousHistoryPolicy
from vtemp4.vtemp4_runtime import FINAL_APPROACH_M
from vtemp4.vtemp4_waypoint_runtime import HISTORY_AGES_S, WaypointProgress, motion_history


def legal_ring_arcs(adapter, position, yaw, radius):
    """Compute continuous body-frame angle intervals from world and receiver grid crossings."""
    world = float(adapter.ep.case['world_size'])
    res = adapter.resolution
    position = np.asarray(position, dtype=float)
    # A ring's endpoint stays in the same cell between consecutive crossings
    # of x/y grid lines. Include the true boundary of a partial last world cell.
    crossings = []
    for axis, coord in enumerate(position):
        first = max(0, math.ceil((coord-radius)/res))
        last = min(math.floor(world/res), math.floor((coord+radius)/res))
        lines = np.r_[np.arange(first, last+1) * res, world]
        offset = (lines-coord)/radius
        offset = offset[np.abs(offset) <= 1.]
        if axis == 0:
            angle = np.arccos(offset)
            crossings.extend((angle, -angle))
        else:
            angle = np.arcsin(offset)
            crossings.extend((angle, math.pi-angle))
    relative = (np.concatenate(crossings)-yaw+math.pi) % math.tau - math.pi
    edges = np.unique(np.r_[-math.pi, relative, math.pi])
    mids = (edges[:-1]+edges[1:])*.5
    xy = ring_points(position, yaw, mids, radius)
    inside = ((xy >= 0.) & (xy < world)).all(axis=1)
    if not inside.any():
        raise ValueError('The mission map contains no 200 m ring arc from the vehicle')

    # Match native endpoint clearance, with a two-cell dilation halo. Unknown
    # cells can receive a survey request; local motion is always independently planned.
    indices = np.floor(xy[inside]/res).astype(int)
    lo = np.maximum(0, indices.min(axis=0)-2)
    hi = np.minimum(adapter.cells.shape[::-1], indices.max(axis=0)+3)
    clearance = _clearance(adapter.cells[lo[1]:hi[1], lo[0]:hi[0]])
    ix, iy = (indices-lo).T
    free = inside.copy()
    free[inside] = clearance[iy, ix] != 1
    # A fully observed blocked ring still supplies an in-bounds directional survey
    # request. This gives the UGV planner no permission to drive through a blockage.
    all_blocked = not free.any()
    allowed = inside if all_blocked else free
    boundaries = np.flatnonzero(np.diff(np.r_[False, allowed, False]))
    arcs = edges[boundaries].reshape(-1, 2)
    # Keep float32 rollout angles strictly within their endpoint cells. The
    # one-microradian margin corresponds to 0.2 mm at 200 m.
    arcs = arcs[(arcs[:, 1] - arcs[:, 0]) > 2e-6] + np.array([1e-6, -1e-6])
    if not len(arcs):
        raise ValueError('No numerically resolvable 200 m guidance arc')
    return arcs, all_blocked


class ArcAngleDistribution:
    """Wrapped Cauchy in cumulative admissible arc length, with a constant Jacobian."""

    def __init__(self, output, arc_lists):
        self.location = math.pi * output[:, 0].tanh()
        self.rho = .99 * output[:, 1].sigmoid()
        count = max(len(a) for a in arc_lists)
        padded = np.zeros((len(arc_lists), count, 2), dtype=float)
        for row, arcs in zip(padded, arc_lists):
            row[:len(arcs)] = arcs
        self.lower, upper = torch.as_tensor(padded, device=output.device).unbind(-1)
        self.widths = upper - self.lower
        self.ends = self.widths.cumsum(dim=1)
        self.starts = self.ends - self.widths
        self.total = self.ends[:, -1]

    @staticmethod
    def _at(table, index):
        return table.gather(1, index[:, None]).squeeze(1)

    def action(self, deterministic, generator=None):
        theta = self.location
        if not deterministic:
            device = generator.device if generator is not None else theta.device
            u = math.pi * (torch.rand(theta.shape, generator=generator, device=device).to(theta)-.5)
            delta = 2 * torch.atan2((1 - self.rho) * u.sin(), (1 + self.rho) * u.cos())
            theta = theta + delta
        length = ((theta.double()+math.pi) % math.tau) * (self.total/math.tau)
        index = (length[:, None] >= self.ends).sum(1).clamp_max(self.ends.shape[1]-1)
        angle = self._at(self.lower, index) + length - self._at(self.starts, index)
        return torch.ones_like(angle, dtype=torch.bool), angle.to(self.location.dtype)

    def log_prob(self, designate, angle):
        if not designate.all():
            raise ValueError('Periodic policy requires a new angle for every decision')
        a = angle.double()[:, None]
        # Allow the rounding of a single stored float32 endpoint.
        legal = (self.widths > 0.) & (a >= self.lower-2e-7) & (a <= self.lower+self.widths+2e-7)
        if not legal.any(dim=1).all():
            raise ValueError('Stored angle outside the admissible ring arcs')
        index = legal.long().argmax(dim=1)
        length = self._at(self.starts, index) + angle.double() - self._at(self.lower, index)
        theta = (length/self.total*math.tau-math.pi).to(self.location.dtype)
        denom = (1-self.rho).square() + 4*self.rho*((theta-self.location)/2).sin().square()
        # density_in_angle = density_in_latent * (2*pi / total_legal_radians)
        return torch.log1p(-self.rho.square()) - denom.log() - self.total.log().to(theta.dtype)

    def entropy(self):
        return self.total.log().to(self.location.dtype) + torch.log1p(-self.rho.square())


class PeriodicHistoryPolicy(ContinuousHistoryPolicy):
    """One continuous angle per decision, using spatial and temporal attention."""

    def __init__(self, hidden=64):
        super().__init__(hidden)
        self.actor = nn.Linear(hidden, 2)
        nn.init.orthogonal_(self.actor.weight, gain=.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, observations):
        state = self._history_state(observations)
        arcs = [obs['angle_arcs'] for obs in observations]
        return ArcAngleDistribution(self.actor(state), arcs), self.critic(state).squeeze(-1)


class PeriodicHistoryFleetEpisode(RingHeadingFleetEpisode):
    """Issue a shared world point on each periodic decision."""

    designate_near_goal = False
    waypoint_progress_bonus_ratio = 0.
    _native_drive_events = None
    _original_velocity_command = None

    def __init__(self, repo, case, model, *, review_interval_s=15, ground_control='vtemp4', **kwargs):
        if ground_control not in ('vtemp4', 'vtemp1', 'vtemp1_connected', 'vtemp1_connected_anchored'):
            raise ValueError(f'Unknown ground control {ground_control!r}')
        if ground_control != 'vtemp4':
            terrain = case.get('terrain', 'synthetic')
            circles = (terrain == 'synthetic' and case.get('kind') == 'v9_random'
                       and case.get('navigation_task', 'random_circles') == 'random_circles')
            field = (ground_control == 'vtemp1_connected_anchored' and terrain == 'wildscenes_raster'
                     and case.get('kind') in ('procedural_wooded_lake', 'procedural_wooded_valley'))
            if not (circles or field):
                raise ValueError('Expected a synthetic V9 random-circle case or an anchored lake/river raster')
            if field and kwargs.get('progress_distance') == 'traversable':
                raise ValueError('The circle-based route distance cannot measure progress on a lake/river raster')
        self.ground_control = ground_control
        fmt = RING_UNIFIED_GOAL_FORMAT if self.designate_near_goal else RING_PERIODIC_HISTORY_FORMAT
        ring_action_contract(fmt, review_interval_s)
        self.samples = deque(maxlen=max(HISTORY_AGES_S)+1)
        super().__init__(repo, case, model, review_interval_s=review_interval_s, **kwargs)

    def fleet_summary(self):
        return dict(super().fleet_summary(), ground_control=self.ground_control)

    def _capture_native_velocity(self, *args, **kwargs):
        waypoint = kwargs['target_waypoint'] if 'target_waypoint' in kwargs else args[2]
        self._native_local_target = np.asarray(waypoint, dtype=float).copy()
        return self._original_velocity_command(*args, **kwargs)

    def _ground_action(self, neighbors=None):
        if self.ground_control == 'vtemp4':
            return super()._ground_action(neighbors)
        if self._original_velocity_command is None:
            if self.ground_control == 'vtemp1_connected':
                from vtemp4.vtemp4_connected_ground import install_connected_ground
                install_connected_ground(self.env.ground_controller)
            if self.ground_control == 'vtemp1_connected_anchored':
                from vtemp4.vtemp4_anchored_ground import install_anchored_ground
                install_anchored_ground(self.env.ground_controller)
            planner = self.env.ground_controller.local_planner
            self._original_velocity_command = planner.compute_velocity_command
            planner.compute_velocity_command = self._capture_native_velocity
        self._native_local_target = None
        action = super()._ground_action(neighbors)
        # A review during this tick sees the preceding tick's executed target.
        # The frozen controller skips its velocity planner entirely when waiting.
        self.adapter.ground_target = self._native_local_target
        if self._native_drive_events is not None:
            xy = None if self._native_local_target is None else self._native_local_target.tolist()
            previous = self._native_drive_events[-1]['target'] if self._native_drive_events else None
            if xy != previous:
                self._native_drive_events.append(dict(time_s=self.clock_s, target=xy))
        return action

    def close(self):
        if self._original_velocity_command is not None:
            self.env.ground_controller.local_planner.compute_velocity_command = self._original_velocity_command
            self._original_velocity_command = None
        super().close()

    def _heading_observation(self):
        obs = super()._heading_observation()
        now, position = self.clock_s, self.position()
        WaypointProgress._record_pose(self, now, position)
        target = None if self.current_request is None else np.asarray(self.current_request['target'])
        history, mask = motion_history(self.samples, now, position, self.goal, target, self.lookahead_m,
                                      issued_at=self.held_since)
        if self._use_final_goal(float(np.linalg.norm(self.goal - position))):
            # No point on the outer ring is chosen at this review.
            arcs, all_blocked = np.empty((0, 2), dtype=float), False
        else:
            yaw = float(self.adapter.own['orientation'])
            arcs, all_blocked = legal_ring_arcs(self.adapter, position, yaw, self.lookahead_m)
        obs.update(history=history, history_mask=mask, angle_arcs=arcs, all_ring_endpoints_blocked=all_blocked)
        return obs

    def _use_final_goal(self, distance):
        return self.designate_near_goal and distance <= self.lookahead_m

    def _choose_heading(self, now, position, distance):
        obs = self._heading_observation()
        yaw = float(self.adapter.own['orientation'])
        use_goal = self._use_final_goal(distance)
        hold = bool(obs.get('wait_for_new_evidence', False))
        with self.torch.inference_mode():
            if hold:
                target = position.copy()
                angle, logp, value = 0., 0., self.slow.value(obs)
            elif use_goal:
                target = self.goal.copy()
                dx, dy = target - position
                angle = (math.atan2(dy, dx) - yaw + math.pi) % math.tau - math.pi
                logp, value = 0., self.slow.value(obs)
            else:
                designate, angle, logp, value = self.slow.act(obs, deterministic=self.rollout is None,
                                                             generator=self.generator)
                if not designate or not math.isfinite(angle) or abs(angle) > math.pi:
                    raise ValueError('Periodic decision requires a finite signed angle')
                target = ring_points(position, yaw, [angle], self.lookahead_m)[0]
        self._settle(now, distance, value)
        # Compare bearings at issue time so vehicle motion cannot rotate the previous command.
        switch_angle = 0.
        if self.current_request is not None:
            previous = self.current_request
            previous_bearing = previous['vehicle_yaw_rad'] + previous['relative_angle_rad']
            switch_angle = math.remainder(yaw + angle - previous_bearing, math.tau)
        # Each decision receives a distinct ID even if the car and the selected
        # angle happen to yield exactly the previous XY.
        self.current_request = dict(id=len(self.requests), issued_at_s=now, target=target.tolist(),
                                    origin_xy=position.tolist(), vehicle_yaw_rad=yaw,
                                    relative_angle_rad=angle, radius_m=0. if hold else (distance if use_goal else self.lookahead_m),
                                    physics_broadcasts=0, first_physics_broadcast_s=None, heard_by={})
        self.requests.append(self.current_request)
        event = dict(time_s=now, position=position.tolist(), remaining_m=distance, turn=True,
                     angle_rad=angle, vehicle_yaw_rad=yaw, switch_angle_rad=switch_angle, changed=True, accepted=True,
                     reason='scheduled_designation', waypoint_id=self.current_request['id'],
                     target=target.tolist(), all_ring_endpoints_blocked=obs['all_ring_endpoints_blocked'])
        if self.waypoint_progress_bonus_ratio:
            # Score the issued world point from its original vehicle pose.
            waypoint_gain = 0. if use_goal else distance - float(np.linalg.norm(target - self.goal))
            event['waypoint_progress_score'] = float(np.clip(waypoint_gain / self.lookahead_m, 0., 1.))
        self.heading_reviews.append(event)
        self.pending = dict(start=now, distance=distance, observation=obs, turn=True, angle_rad=angle,
                            turn_for_reward=(abs(switch_angle) > 1.e-6, switch_angle),
                            logp=float(logp), value=float(value), event=event)
        if self.designate_near_goal:
            source = 'final_goal' if use_goal else 'policy_angle'
            if hold:
                source = 'wait_for_new_evidence'
            self.current_request['designation_source'] = event['designation_source'] = source
            self.pending['policy_controlled'] = not (use_goal or hold)
        self.last_review = self.held_since = now
        self.previous_review_position = position.copy()

    def _issue_native_waypoint(self, now, position):
        request = self.current_request
        target = np.asarray(request['target'], dtype=float)
        previous = self.manager.target if self.manager.selected else None
        controller = self.env.ground_controller
        # _drive_to clears the native path. Repeating a broadcast, even with a
        # fresh command ID, must not restart a route or re-arm an arrived target.
        if not _same(previous, target):
            controller._drive_to(tuple(target))
        self.manager.target = target.copy()
        self.manager.origin = position.copy()
        self.manager.final = _same(target, self.goal)
        self.manager.selected += 1
        self.adapter.ground_guidance = target.copy()
        self.adapter.last_ground_reason = f'{self.ground_control}_shared_waypoint'
        self.ground_events.append(dict(
            time_s=now, position=position.tolist(), target=target.tolist(),
            previous_target=None if previous is None else previous.tolist(),
            previous_target_reached=None if previous is None else bool(
                np.linalg.norm(position-previous) <= controller.waypoint_threshold),
            reason=self.adapter.last_ground_reason, shared_marker=list(request['target']),
            final_approach=self.final_approach, waypoint_id=request['id']))

    def _after_fusion(self, graph, semantic_map):
        now = self.clock_s
        if self._last_control_time != now:
            self._last_control_time = now
            self._belief_cache = None
            self.adapter.observe_receipts()
            position = self.position()
            WaypointProgress._record_pose(self, now, position)
            distance = float(np.linalg.norm(position-self.goal))
            if now < self.limit and distance > GOAL_TOLERANCE_M:
                finishing = distance <= FINAL_APPROACH_M or (self.final_approach and distance <= self.lookahead_m)
                phase_changed = self.final_approach != finishing
                self.final_approach = finishing
                due = self.last_review is None or now-self.last_review >= self.review_interval
                if due:
                    self._choose_heading(now, position, distance)
                if self.ground_control != 'vtemp4':
                    if due:
                        self._issue_native_waypoint(now, position)
                elif due or phase_changed or now % self.ground_interval == 0:
                    marker = self.current_request['target']
                    self._ground_decision(now, guidance=None if finishing else np.asarray(marker))
                    self.ground_events[-1].update(shared_marker=list(marker), final_approach=finishing,
                                                  waypoint_id=self.current_request['id'])
        if self.ground_control != 'vtemp4':
            return self._original_filter(graph, semantic_map)
        fused = self._native_filter(graph, semantic_map)
        return None if self.adapter.last_ground_reason == START_CONNECTOR else fused
