"""Attach vtmp2 to a controller without editing its archived implementation."""
from __future__ import annotations

import time

import numpy as np

if __package__:
    from .reconnaissance import ReconnaissanceEnvelope
    from .vtmp2 import Vtmp2SubgoalManager
else:
    from reconnaissance import ReconnaissanceEnvelope
    from vtmp2 import Vtmp2SubgoalManager


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
                    if __package__:
                        from .data import encode
                    else:
                        from data import encode
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
