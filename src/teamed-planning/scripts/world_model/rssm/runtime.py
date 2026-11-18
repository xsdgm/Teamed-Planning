"""Configured reconnaissance reach and adaptation of archived controllers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
import time

import numpy as np

from .vtmp2 import Vtmp2SubgoalManager


# Communication-constrained reconnaissance envelope for vtmp2.
#
# All distances are in metres. The caller supplies configured fleet size, radio
# range, fixed flight altitude and sensor footprint; this module never reads
# simulator fleet positions or obstacle truth. The envelope is centred on the
# vehicle's current position by the target manager.
#
# A connected backbone may branch, but its farthest member can have at most N
# radio hops from the vehicle. A straight deployment gives the geometric outer
# reach used here. The 3D ground-to-air hop is projected into the horizontal
# plane; equal-altitude air-to-air hops need no altitude correction. Half the
# square sensor width is its inscribed-circle radius, so no diagonal corner
# coverage is assumed.
#
# This is a deployment envelope, not proof of current visibility, free terrain,
# reachability by the vehicle, or a simultaneously feasible UAV manoeuvre. Live
# CBF, communication, world-boundary and received-map checks still apply.

@dataclass(frozen=True)
class ReconnaissanceEnvelope:
    """Configured reconnaissance reach, independent of simulator truth.

    ``deployment_ratio`` discounts each communication hop *before* the ground
    hop's height projection. It is a planning margin, not a CBF guarantee.
    ``theoretical_radius_m`` retains the CBF link margin but omits this extra
    discount; ``radius_m`` is the radius used for candidate planning.
    """

    drone_count: int = 3
    communication_range: float = 95.
    altitude: float = 15.
    footprint_width: float = 24.
    link_margin: float = .25
    deployment_ratio: float = .85

    def __post_init__(self):
        if (isinstance(self.drone_count, bool)
                or not isinstance(self.drone_count, Integral)
                or self.drone_count < 0):
            raise ValueError('drone_count must be a nonnegative integer')
        object.__setattr__(self, 'drone_count', int(self.drone_count))
        for name in ('communication_range', 'altitude', 'footprint_width',
                     'link_margin', 'deployment_ratio'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f'{name} must be a finite positive number')
            try:
                value = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f'{name} must be a finite positive number') from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a finite positive number')
            object.__setattr__(self, name, value)
        if self.link_margin >= self.communication_range:
            raise ValueError('link_margin must be smaller than communication_range')
        if self.deployment_ratio > 1:
            raise ValueError('deployment_ratio must lie in (0, 1]')
        # Reject unrepresentable ranges even when each input is finite.
        self._radius(self.communication_range - self.link_margin)

    def _radius(self, hop_range):
        if self.drone_count == 0 or hop_range < self.altitude:
            return 0.
        # Avoid squaring large lengths; both inputs have already been checked.
        horizontal_ground_hop = hop_range * math.sqrt(
            max(0., 1. - (self.altitude / hop_range) ** 2))
        try:
            radius = (horizontal_ground_hop
                      + (self.drone_count - 1) * hop_range
                      + self.footprint_width / 2.)
        except OverflowError as exc:
            raise ValueError('reconnaissance radius must remain finite') from exc
        if not math.isfinite(radius):
            raise ValueError('reconnaissance radius must remain finite')
        return float(radius)

    @property
    def theoretical_radius_m(self):
        """Geometric deployment reach after the configured radio-link margin."""
        return self._radius(self.communication_range - self.link_margin)

    @property
    def radius_m(self):
        """Conservative candidate radius after discounting every radio hop."""
        hop_range = (self.communication_range - self.link_margin) * self.deployment_ratio
        return self._radius(hop_range)

    def describe(self):
        """Return JSON-compatible parameters and distinct geometric/plan radii."""
        return dict(asdict(self), theoretical_radius_m=self.theoretical_radius_m,
                    planning_radius_m=self.radius_m,
                    interpretation='deployment_envelope_not_observed_or_traversable_space')


def make_vtmp2_controller(base_controller, *, reconnaissance=None,
                         max_target_distance=200., manager_options=None,
                         manager_factory=Vtmp2SubgoalManager, planner_factory=None):
    """Select once per control tick, AFTER current sensing and graph reception.

    The 200m default also bounds candidates to the world model's graph window.
    The envelope describes possible connected deployment, not map truth or a
    guarantee that the current aircraft can reach every point immediately.
    """
    envelope = reconnaissance or ReconnaissanceEnvelope()
    options = dict(manager_options or {})

    class Vtmp2GroundController(base_controller):
        high_level_algorithm = 'vtmp2'

        def set_goal(self, goal_pos):
            self.final_goal = tuple(goal_pos)
            self.reconnaissance = envelope
            planner_options = {}
            if planner_factory is not None:
                planner = planner_factory()
                planner.reset()
                planner_options['planner'] = planner
            self._vtmp2_previous_position = None
            self._vtmp2_previous_time = None
            self.subgoals = manager_factory(
                goal_pos, radius=max_target_distance, arrival=self.subgoal_arrival,
                world_bounds=self.world_bounds, recon_radius=envelope.radius_m,
                safety_distance=self.local_planner.safety_distance, **options, **planner_options)
            self.global_planner.clear_path()
            self.goal_position = None

        def get_action(self, observation):
            # The archived controller selects before parsing/fusion. Temporarily
            # suppress that callback and perform it at the post-fusion hook below.
            manager = self.subgoals
            self._vtmp2_manager = manager
            self.subgoals = None
            self._vtmp2_time = observation.get('full_state', {}).get('time', time.time())
            try:
                action = super().get_action(observation)
                if hasattr(manager, 'purpose'):
                    # The aerial information-gathering target is independent of
                    # the ground driving target. Preserve the normal transport:
                    # the drones learn this task only through received packets.
                    action['broadcast'] = dict(action.get('broadcast') or {})
                    action['broadcast'].update(
                        target=np.asarray(manager.target).tolist(),
                        task_purpose=manager.purpose,
                        vehicle_target=(None if manager.drive_target is None else
                                        np.asarray(manager.drive_target).tolist()))
                    if manager.purpose == 'survey' or manager.drive_target is None:
                        action['motion'] = (0., 0.)
                        action['broadcast']['planned_path'] = []
                return action
            finally:
                self.subgoals = manager
                self._vtmp2_manager = None

        def _receive_and_fuse_graphs(self, drones_data, current_time=None):
            fused = super()._receive_and_fuse_graphs(drones_data, current_time)
            if fused:
                self.unified_graph = self.graph_fusion.get_unified_graph()
            manager = self._vtmp2_manager
            if manager is not None:
                semantic = self._last_semantic_map
                graph = self._filter_observed_obstacles(self.unified_graph, semantic)
                blocked, _ = self.local_planner.observed_blocked_cells(semantic, self.resolution)
                manager.set_reconnaissance_range(self.reconnaissance.radius_m)
                manager.set_observation(self.position, semantic, self.resolution, blocked=blocked)
                if hasattr(manager, 'set_received_survey_graph'):
                    manager.set_received_survey_graph(self.unified_graph)
                planner = getattr(manager, 'planner', None)
                if planner is not None:
                    # Import lazily: data also imports this runtime adapter.
                    from .data import encode
                    now = float(self._vtmp2_time)
                    position = np.asarray(self.position, dtype=float)
                    velocity = np.zeros(2)
                    if self._vtmp2_previous_position is not None:
                        dt = now - self._vtmp2_previous_time
                        if dt > 0:
                            velocity = (position-self._vtmp2_previous_position)/dt
                    maps, states = encode(self, self.final_goal, now, velocity)
                    previous = getattr(manager, 'target', None)
                    planner.observe(maps, states, now, executed_target=previous,
                                    executed_purpose=getattr(manager, 'purpose', 'drive'))
                    self._vtmp2_previous_position = position.copy()
                    self._vtmp2_previous_time = now
                target, changed = manager.update(self.position, graph, self._vtmp2_time)
                if changed:
                    if planner is not None:
                        # Close the old macro action at the actual command
                        # boundary, before any physics executes the new task.
                        planner.observe(maps, states, now, executed_target=target,
                                        executed_purpose=getattr(manager, 'purpose', 'drive'))
                    drive_target = getattr(manager, 'drive_target', target)
                    if drive_target is None:
                        self.global_planner.clear_path()
                        self.goal_position = None
                    else:
                        self._drive_to(tuple(drive_target))
            return fused

    return Vtmp2GroundController
