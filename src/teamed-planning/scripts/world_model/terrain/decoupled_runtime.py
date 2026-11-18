"""Adapters around frozen v9 flight and the original ground navigation stack.

Only explicit sensor/telemetry packets are added to the existing radio channel.
The adapter never consumes simulator audit ledgers or changes v9 weights/CBF.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np

from .decoupled_observation import (MAX_PACKET_AGE, PACKET_KEY, ReceivedObservation,
                                    point, validate_packet)
from .decoupled_planner import IndependentGroundManager, IndependentSurveyManager
from .runtime import ReconnaissanceEnvelope


@lru_cache(maxsize=1)
def ground_navigation_controller():
    """Load this run's vtmp vehicle stack without replacing frozen v9 modules.

The archived v9 simulator binds its original vehicle class at import time. The
independent adapter must use the current vtmp vehicle recovery implementation;
the v9 aircraft and CBF remain archived. Workers build environments serially.
"""
    scripts = Path(__file__).resolve().parents[2]
    saved = {name: value for name, value in sys.modules.items()
             if name == 'ground' or name.startswith('ground.')}
    config_name = 'common.navigation_config'
    old_config = sys.modules.get(config_name)
    old_path = list(sys.path)
    try:
        for name in saved:
            del sys.modules[name]
        if old_config is None:
            spec = importlib.util.spec_from_file_location(config_name, scripts/'common/navigation_config.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[config_name] = module
            spec.loader.exec_module(module)
        spec = importlib.util.spec_from_file_location('ground', scripts/'ground/__init__.py',
                                                      submodule_search_locations=[str(scripts/'ground')])
        package = importlib.util.module_from_spec(spec)
        sys.modules['ground'] = package
        spec.loader.exec_module(package)
        # Keep an importable identity for inspection/checkpoint provenance after
        # restoring the public ground namespace used by the archived runtime.
        module = sys.modules['ground.ground_main']
        sys.modules['_vtmp2_ground_navigation'] = module
        package.GroundRobotController.__module__ = '_vtmp2_ground_navigation'
        return package.GroundRobotController
    finally:
        for name in list(sys.modules):
            if name == 'ground' or name.startswith('ground.'):
                del sys.modules[name]
        sys.modules.update(saved)
        if old_config is None:
            sys.modules.pop(config_name, None)
        sys.path[:] = old_path


def sensor_tile(observation):
    tile = np.asarray(observation['occupancy_map'])
    axes = observation.get('occupancy_map_axes', 'yx')
    if axes == 'xy':
        return tile.T.copy()
    if axes != 'yx':
        raise ValueError('Unknown occupancy-map axes')
    return tile.copy()


def make_information_drone_controller(base_controller):
    class InformationDroneController(base_controller):
        def get_action(self, observation):
            # The original v9 actor and CBF produce the entire flight command.
            action = super().get_action(observation)
            now = float(observation.get('full_state', {}).get('time', time.time()))
            count = getattr(self, 'total_drones', 3)
            # Base versions keep fleet count on the navigator rather than on
            # the controller. Keep a bounded, one-packet-per-origin relay cache.
            count = getattr(self.navigator, 'total_drones', count)
            cache = dict(getattr(self, '_information_packets', {}))
            for neighbor in observation.get('neighbors', []):
                for raw in (neighbor.get('broadcast') or {}).get(PACKET_KEY, []):
                    packet = validate_packet(raw, now, count)
                    if packet is not None and (packet['source'] not in cache or
                            packet['time'] > cache[packet['source']]['time']):
                        cache[packet['source']] = packet
            own = dict(source=self.drone_id, time=now,
                position=point(observation['self']['position'], 3),
                velocity=point(observation['self']['velocity'], 3),
                resolution=float(observation['map_resolution']), occupancy=sensor_tile(observation),
                valid=bool(observation.get('sensor_valid', True)))
            cache[self.drone_id] = own
            self._information_packets = {key: value for key, value in cache.items()
                                         if 0 <= now-value['time'] <= MAX_PACKET_AGE}
            # No radio packet is invented when the original controller did not
            # broadcast. Existing delay, loss, range and relay timing still apply.
            if isinstance(action.get('broadcast'), dict):
                action['broadcast'] = dict(action['broadcast'])
                action['broadcast'][PACKET_KEY] = list(self._information_packets.values())
            return action
    return InformationDroneController


def make_decoupled_controller(base_controller, *, planner_factory, reconnaissance=None,
                              max_target_distance=200., manager_options=None):
    envelope = reconnaissance or ReconnaissanceEnvelope()
    options = dict(manager_options or {})

    class DecoupledGroundController(base_controller):
        high_level_algorithm = 'vtmp2'

        def set_goal(self, goal_pos):
            self.final_goal = tuple(goal_pos)
            self.reconnaissance = envelope
            if hasattr(self.local_planner, 'escape_distance'):
                # Current vtmp recovery supports a reduced-clearance mode. This
                # experiment retains its certified 1.5m margin during recovery.
                self.local_planner.escape_distance = self.local_planner.safety_distance
            self.world_planner = planner_factory()
            self.observation_memory = ReceivedObservation(self.world_bounds, envelope.drone_count)
            if self.world_planner.drone_count != envelope.drone_count:
                raise ValueError('World model and runtime fleet sizes differ')
            self.subgoals = IndependentGroundManager(goal_pos, planner=self.world_planner,
                radius=max_target_distance, arrival=1., world_bounds=self.world_bounds,
                safety_distance=self.local_planner.safety_distance, **options)
            self.survey = IndependentSurveyManager(self.world_planner, radius=max_target_distance)
            self.global_planner.clear_path()
            self.goal_position = None
            self.world_observation = None
            self._decoupled_manager = None

        @property
        def drive_target(self):
            manager = self._decoupled_manager or self.subgoals
            return None if manager.target_status == 'hold' else manager.target

        @property
        def execution_target(self):
            manager = self._decoupled_manager or self.subgoals
            return manager.execution_target

        def get_action(self, observation):
            manager = self.subgoals
            self._decoupled_manager, self.subgoals = manager, None
            self._decoupled_input = observation
            try:
                action = super().get_action(observation)
                broadcast = dict(action.get('broadcast') or {})
                broadcast.update(target=None if self.survey.target is None else self.survey.target.tolist(),
                    survey_target=None if self.survey.target is None else self.survey.target.tolist(),
                    vehicle_target=None if self.drive_target is None else self.drive_target.tolist(),
                    vehicle_waypoint=None if self.execution_target is None else self.execution_target.tolist(),
                    final_goal=list(self.final_goal), task_purpose='independent_information')
                action['broadcast'] = broadcast
                if self.execution_target is None:
                    action['motion'] = (0., 0.)
                return action
            finally:
                self.subgoals, self._decoupled_manager = manager, None
                self._decoupled_input = None

        def _receive_and_fuse_graphs(self, drones_data, current_time=None):
            fused = super()._receive_and_fuse_graphs(drones_data, current_time)
            manager = self._decoupled_manager
            if manager is None:
                return fused
            if fused:
                self.unified_graph = self.graph_fusion.get_unified_graph()
            raw = self._decoupled_input
            now = float(current_time)
            packets = [packet for neighbor in raw.get('neighbors', [])
                       for packet in (neighbor.get('broadcast') or {}).get(PACKET_KEY, [])]
            self.observation_memory.receive(packets, now)
            if raw.get('sensor_valid', True):
                self.observation_memory.add_sensor(self.position, sensor_tile(raw), self.resolution)
            velocity = np.asarray(raw['self'].get('velocity', (0., 0.)), dtype=float)[:2]
            maps, states, origin = self.observation_memory.encode(self.position, self.heading, velocity,
                self.final_goal, self.drive_target, self.survey.target)
            self.world_planner.observe(maps, states, origin, now)
            semantic = self._last_semantic_map
            graph = self._filter_observed_obstacles(self.unified_graph, semantic)
            blocked, _ = self.local_planner.observed_blocked_cells(semantic, self.resolution)
            manager.set_observation(self.position, semantic, self.resolution, blocked=blocked)
            manager.update(self.position, graph, now)
            # New information can enable or revoke an execution waypoint even
            # when the strategic target stays unchanged.
            waypoint = self.execution_target
            if waypoint is None:
                if self.goal_position is not None:
                    self.global_planner.clear_path()
                    self.goal_position = None
            elif self.goal_position is None or not np.allclose(waypoint, self.goal_position, atol=1e-6, rtol=0.):
                self._drive_to(tuple(waypoint))
            self.survey.update(np.asarray(self.position), np.asarray(self.final_goal), self.drive_target,
                self.reconnaissance.radius_m, self.world_bounds, self.observation_memory, now)
            # Collection records both independent high-level intentions. The
            # observed execution waypoint is a separate runtime decision.
            self.world_observation = self.observation_memory.encode(self.position, self.heading, velocity,
                self.final_goal, self.drive_target, self.survey.target)
            return fused
    return DecoupledGroundController
