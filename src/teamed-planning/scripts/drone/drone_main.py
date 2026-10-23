"""
Drone Main Controller
Integrates perception, mapping, navigation, and communication for drone operation
"""

import numpy as np
import time
import sys
import os
from typing import Dict, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone.semantic_segmentation import SemanticProcessor, TerrainType
from drone.skeleton_graph_builder import SkeletonGraphBuilder
from drone.graph_broadcaster import GraphBroadcaster, RobotDetector
from drone.drone_navigator import DroneNavigator
from common.message_definitions import SimpleMessageBroker
from common.graph_structure import TopologicalGraph
from common.timing import PerformanceTimer


class DroneController:
    """
    Main controller for drone agent.
    
    Coordinates:
    - Semantic perception
    - Topological graph building
    - Graph broadcasting to other drones and robots
    - Graph fusion with nearby drones
    """
    
    def __init__(self, drone_id: str, resolution: float = 0.5,
                 broadcast_interval: float = 5.0,
                 communication_range: float = 30.0,
                 fusion_interval: float = 30.0,
                 max_speed: float = 15.0,
                 waypoint_threshold: float = 10.0,
                 total_drones: int = 1,
                 dispatch_interval: float = 1.0,
                 navigation_algorithm: str = 'cadmm',
                 mappo_checkpoint: Optional[str] = None,
                 communication_mode: Optional[str] = None, safety_filter: Optional[str] = None,
                 graph_min_clearance: float = 3.0):
        """
        Initialize drone controller.

        Args:
            drone_id: Unique identifier for this drone
            resolution: Map resolution in metres per pixel (default 0.5 m/px)
            broadcast_interval: Seconds between graph broadcasts
            communication_range: Range for graph fusion with nearby drones (metres)
            fusion_interval: Minimum seconds between fusions with the same drone
            max_speed: Maximum speed in m/s
            waypoint_threshold: Distance to waypoint considered "reached" (metres)

        Note — perception_range is intentionally absent.  The occupancy map size
        is derived automatically from the drone's current altitude and camera HFOV
        inside drone_node._get_occupancy_map(), so the graph builder always
        receives a map that covers the actual camera footprint.
        """
        self.drone_id = drone_id
        self.communication_range = communication_range
        self.fusion_interval = fusion_interval
        self.resolution = resolution
        self.max_speed = max_speed
        self.broadcast_interval = broadcast_interval
        self.dispatch_interval = dispatch_interval
        self._last_dispatch_time = float('-inf')
        self._last_graph_broadcast_time = float('-inf')
        self._broadcast_graph = None
        if safety_filter not in (None,'legacy','cbf'):
            raise ValueError('safety_filter must be legacy or cbf')
        self.safety_filter=safety_filter or 'legacy'
        self.safety_stats={}
        self.communication_mode=communication_mode or 'direct'
        from drone.chain_state import ChainState
        self._chain_state=ChainState(drone_id,total_drones if self.communication_mode=='chain' else 1)

        # Initialize modules
        self.semantic_processor = SemanticProcessor(resolution=resolution)

        self.graph_builder = SkeletonGraphBuilder(
            drone_id=drone_id,
            resolution=resolution,
            node_spacing=10.0,    # graph node spacing in metres
            min_clearance=graph_min_clearance,
            max_edge_distance=15.0,
            k_neighbors=4,
            # perception_radius removed — derived from map shape at call time
        )
        
        self.navigator = DroneNavigator(
            drone_id=drone_id,
            total_drones=total_drones,
            waypoint_threshold=waypoint_threshold,
            comm_range=communication_range,
        )
        if navigation_algorithm != 'cadmm':
            from drone.mappo import CandidateNavigator
            if navigation_algorithm not in ('mappo', 'greedy', 'cadmm_controlled'):
                raise ValueError(f'Unknown navigation algorithm: {navigation_algorithm}')
            if navigation_algorithm == 'mappo' and not mappo_checkpoint:
                raise ValueError('MAPPO requires a trained mappo_checkpoint')
            self.navigator = CandidateNavigator(
                drone_id=drone_id, total_drones=total_drones,
                waypoint_threshold=waypoint_threshold, comm_range=communication_range,
                algorithm=navigation_algorithm, checkpoint=mappo_checkpoint,
                communication_mode=communication_mode,
                max_speed=max_speed, decision_dt=dispatch_interval)
        if communication_mode is None:
            self.communication_mode=getattr(self.navigator,'communication_mode','direct')
        if safety_filter is None:
            self.safety_filter=getattr(self.navigator,'checkpoint_safety_filter','legacy')
        self._chain_state=ChainState(drone_id,total_drones if self.communication_mode=='chain' else 1)
        
        # NOTE: GraphBroadcaster and RobotDetector are kept for future ground robot integration
        # Currently using unified dict-based broadcast in get_action()
        self.broadcaster = GraphBroadcaster(
            drone_id=drone_id,
            broadcast_interval=broadcast_interval
        )
        
        self.robot_detector = RobotDetector()
        
        # Performance timer
        self.timer = PerformanceTimer(drone_id)
        
        # State
        self.current_graph = None
        self.position = None
        self.last_fusion_stats = {}
        
        # Store other drones' graphs for fusion
        self.other_drone_graphs = {}  # {drone_id: TopologicalGraph}
        
        # Store last semantic map for visualization (debug)
        self._last_semantic_map = None
        self._last_semantic_map_position = None
    
    # NOTE: Waypoints are now automatically generated based on ground robot mission
    # No manual waypoint setting needed - drones discover mission and generate
    # vortex exploration patterns autonomously
    
    def _parse_simulator_observation(self, sim_obs: Dict) -> Dict:
        """
        Parse simulator observation into standardized format.
        
        Args:
            sim_obs: Raw observation from simulator
            
        Returns:
            Parsed observation dictionary with:
                - 'position': (x, y, z)
                - 'occupancy_map': Binary occupancy map from simulator
                - 'drones': Dict of {drone_id: {'position': (x,y,z)}}
                - 'ground_robots': Dict (future)
        """
        # Extract position
        pos_2d = sim_obs['self']['position']
        height = sim_obs['self']['height']
        position = (pos_2d[0], pos_2d[1], height)
        
        # Extract occupancy map (generated by simulator)
        occupancy_map = sim_obs.get('occupancy_map')
        if occupancy_map is None:
            raise ValueError(f"[{self.drone_id}] No occupancy map in simulator observation!")
        # The mapper consumes camera grids (row=X, column=Y). The 2D
        # simulator supplies world grids (row=Y, column=X); adapt once here.
        axes = sim_obs.get('map_axes', 'yx')
        if axes == 'yx':
            occupancy_map = occupancy_map.T
        elif axes != 'xy':
            raise ValueError(f"Unsupported occupancy map axes: {axes}")
        
        # Extract nearby drones from neighbors
        drones = {}
        ground_robots = {}
        neighbor_waypoints = dict(sim_obs.get('neighbor_waypoints', {}))
        
        neighbors = sim_obs.get('neighbors', [])
        for neighbor in neighbors:
            if neighbor['type'] == 'drone':
                # Get broadcast data from neighbor
                broadcast_msg = neighbor.get('broadcast')
                if broadcast_msg and isinstance(broadcast_msg, dict):
                    other_drone_id = broadcast_msg.get('drone_id')
                    other_position = broadcast_msg.get('position')
                    
                    if other_drone_id and other_position:
                        drones[other_drone_id] = {
                            'position': tuple(other_position),
                            'broadcast': broadcast_msg
                        }
                        waypoint = broadcast_msg.get('proposed_waypoint')
                        if waypoint is not None:
                            neighbor_waypoints[other_drone_id] = tuple(waypoint)
                        for robot_id, mission in broadcast_msg.get('ground_missions', {}).items():
                            ground_robots.setdefault(robot_id,dict(mission,direct=False))
            # Handle ground robots
            elif neighbor['type'] in ['diff_drive', 'holonomic']:
                # Get broadcast data from ground robot
                broadcast_msg = neighbor.get('broadcast')
                if broadcast_msg and isinstance(broadcast_msg, dict):
                    robot_id = broadcast_msg.get('robot_id')
                    robot_position = broadcast_msg.get('position')
                    robot_target = broadcast_msg.get('target')
                    if robot_id and robot_position:
                        ground_robots[robot_id] = {
                            'position': tuple(robot_position),
                            'target': tuple(robot_target) if robot_target else None,
                            'planned_path': broadcast_msg.get('planned_path', []),
                            'source_time': broadcast_msg.get('timestamp',sim_obs.get('full_state',{}).get('time',time.time())),
                            'direct': True,
                            'broadcast': broadcast_msg
                        }
        
        return {
            'position': position,
            'occupancy_map': occupancy_map,
            'drones': drones,
            'ground_robots': ground_robots,
            'neighbor_waypoints': neighbor_waypoints,
        }
    
    def _generate_semantic_map(self, occupancy_map: np.ndarray) -> np.ndarray:
        """
        Transform binary occupancy map to semantic map.
        Uses SemanticProcessor to adapt raw sensor data.
        
        In simulation: transforms occupancy grid from simulator
        In real-world: would transform RGB/depth via segmentation network
        
        Args:
            occupancy_map: Binary occupancy grid (0=free, 1=occupied)
            
        Returns:
            semantic_map: np.ndarray with TerrainType values
        """
        return self.semantic_processor.process_occupancy_map(occupancy_map)
    
    def _compute_velocity_to_target(self, target: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Compute velocity command to reach target waypoint.
        
        Args:
            target: Target (x, y, z) position
            
        Returns:
            (vx, vy, vz) velocity command
        """
        if self.position is None or target is None:
            return (0.0, 0.0, 0.0)
        
        # Compute direction to target
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        dz = target[2] - self.position[2]
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if distance < 1.0:
            # Close enough, stop
            return (0.0, 0.0, 0.0)
        
        # Compute velocity (proportional to distance, limited by max_speed)
        vx = (dx / distance) * self.max_speed
        vy = (dy / distance) * self.max_speed
        vz = (dz / distance) * self.max_speed
        
        return (vx, vy, vz)

    def _safe_simulation_velocity(self, velocity, observation):
        """Apply reciprocal separation bounds to the next simulator step.

        Uses physical relative positions, not delayed graph broadcasts. If
        spawned too close, agents separate at their available speed.
        """
        dt = observation.get('time_step')
        if dt is None or dt <= 0:
            if self.safety_filter=='cbf':
                raise ValueError('CBF simulator control requires time_step')
            return velocity
        if self.safety_filter=='cbf':
            from drone.cbf_safety import CBFSafetyFilter
            peers={}
            for neighbor in observation.get('neighbors',[]):
                if neighbor['type']=='drone':
                    did=(neighbor.get('broadcast') or {}).get('drone_id',f"robot_{neighbor['id']}")
                    peers[did]=np.asarray(self.position)+np.asarray(neighbor['relative_position'])
            chain=getattr(self.navigator,'chain',self._chain_state)
            if self.communication_mode=='chain' and chain is not None:
                required=chain.required
                task=chain.task()
                ground=task['position'] if task is not None and '__ground__' in chain.visible else None
            else:
                required=('__ground__',)
                ground=self.navigator.ground_robot_position
            self._nominal_velocity=np.asarray(velocity,float).copy()
            # Broadcast positions precede the last physical step. Direct
            # received relative geometry is the fresh CBF endpoint sample.
            for neighbor in observation.get('neighbors',[]):
                if neighbor['type'] in ('diff_drive','holonomic') and ground is not None:
                    ground=np.asarray(self.position)+np.asarray(neighbor['relative_position'])
                    break
            filtered,self.safety_stats=CBFSafetyFilter(self.navigator.min_separation,
                self.communication_range,self.max_speed).filter(velocity,self.position,peers,dt,
                required_links=required,ground=ground,
                ground_max_speed=getattr(self,'ground_max_speed',0.),
                world_bounds=getattr(self.navigator,'world_bounds',None),drone_id=self.drone_id)
            return tuple(filtered)
        v = np.asarray(velocity, dtype=float).copy()
        constraints = []
        for neighbor in observation.get('neighbors', []):
            if neighbor['type'] != 'drone':
                continue
            delta = -np.asarray(neighbor['relative_position'][:2], dtype=float)
            distance = float(np.linalg.norm(delta))
            if distance < 1e-8:
                sign = 1. if observation['self'].get('id', -1) > neighbor['id'] else -1.
                normal = np.array([sign, 0.])
            else:
                normal = delta / distance
            lower = (self.navigator.min_separation - distance) / (2. * dt)
            constraints.append((normal, lower))
        for _ in range(4):
            for normal, lower in constraints:
                shortage = lower - float(np.dot(v[:2], normal))
                if shortage > 0:
                    v[:2] += shortage * normal
            speed = float(np.linalg.norm(v))
            if speed > self.max_speed:
                v *= self.max_speed / speed
        return tuple(v)
    
    def get_action(self, simulator_obs: Dict, mode: str = 'sim') -> Dict:
        """
        Main method: processes observation and returns complete action.
        
        Args:
            simulator_obs: Raw observation from simulator (same format for both modes;
                           the ROS node constructs this dict from topic data)
            mode: 'sim'  — return velocity [vx, vy, vz] (original behaviour)
                  'real' — return target_waypoint (x, y, z) for local planner
            
        Returns:
            When mode='sim':
                {'motion': np.array([vx, vy, vz]), 'broadcast': dict}
            When mode='real':
                {'target_waypoint': (x,y,z) or None, 'broadcast': dict}
        """
        if mode=='real' and (self.safety_filter=='cbf' or getattr(self.navigator, 'uses_velocity', False)):
            raise NotImplementedError('CBF velocity filter is currently validated for fixed-height 2D simulation only')
        current_time = simulator_obs.get('full_state', {}).get('time', time.time())
        
        # Step 1: Parse simulator observation (adapter)
        with self.timer.measure("observation_parsing"):
            parsed_obs = self._parse_simulator_observation(simulator_obs)
            self.position = parsed_obs['position']
        
        # Step 2: Transform occupancy map to semantic map
        with self.timer.measure("semantic_map_generation"):
            semantic_map = self._generate_semantic_map(
                parsed_obs['occupancy_map']
            )
            # Store for visualization (debug)
            self._last_semantic_map = semantic_map
            self._last_semantic_map_position = self.position
            
        
        # Step 3: Build/update graph and perform fusion
        with self.timer.measure("graph_update"):
            # Extract graph data from nearby drones
            other_drones_data = {}
            for drone_id, drone_info in parsed_obs['drones'].items():
                broadcast = drone_info.get('broadcast')
                if (broadcast and 'graph' in broadcast and self.current_graph is not None
                        and self.current_graph.should_fuse_with_drone(
                            drone_id, current_time, self.fusion_interval)):
                    other_drones_data[drone_id] = TopologicalGraph.from_message(broadcast['graph'])
            
            # Build or update graph
            if self.current_graph is None:
                self.current_graph = self.graph_builder.build_graph(
                    semantic_map=semantic_map,
                    drone_position=self.position, observation_id=current_time
                ) if simulator_obs.get('sensor_valid', True) else TopologicalGraph(self.drone_id)
            elif simulator_obs.get('sensor_valid', True):
                self.graph_builder.update_graph(
                    existing_graph=self.current_graph,
                    semantic_map=semantic_map,
                    drone_position=self.position, observation_id=current_time
                )
        
        # Step 4: Perform fusion with nearby drones
        if other_drones_data and len(parsed_obs['drones']) > 0:
            with self.timer.measure("graph_fusion"):
                self.last_fusion_stats = self._check_and_fuse_with_nearby_drones(
                    parsed_obs['drones'],
                    other_drones_data,
                    current_time
                )
        
        # Step 5: Prepare observation data for navigator
        nav_obs = {
            'graph': self.current_graph,
            'nearby_drones': parsed_obs['drones'],
            'ground_robots': parsed_obs.get('ground_robots', {}),
            'neighbor_waypoints': parsed_obs.get('neighbor_waypoints', {}),
            'fusion_stats': self.last_fusion_stats,
            'time': current_time
        }
        if getattr(self.navigator, 'uses_velocity', False):
            nav_obs['occupancy_map'] = parsed_obs['occupancy_map']
            nav_obs['sensor_valid'] = simulator_obs.get('sensor_valid', True)
        self._chain_state.update(nav_obs)
        
        # Step 6: Compute navigation target
        with self.timer.measure("navigation"):
            if current_time - self._last_dispatch_time >= self.dispatch_interval:
                if getattr(self.navigator, 'uses_velocity', False):
                    self.navigator.compute_velocity(self.position, nav_obs)
                    target = None
                else:
                    target = self.navigator.compute_next_target(self.position, nav_obs)
                self._last_dispatch_time = current_time
            else:
                target = self.navigator._current_target
        
        # Step 7: Prepare broadcast data
        broadcast_data = None
        if self.current_graph is not None:
            with self.timer.measure("broadcasting"):
                if current_time - self._last_graph_broadcast_time >= self.broadcast_interval:
                    self._broadcast_graph = self.current_graph.to_message()
                    self._last_graph_broadcast_time = current_time
                broadcast_data = {
                    'graph': self._broadcast_graph,
                    'position': list(self.position),
                    'drone_id': self.drone_id,
                    'timestamp': current_time,
                    'node_count': len(self.current_graph),
                    'edge_count': self.current_graph.graph.number_of_edges(),
                    'proposed_waypoint': target,
                    'ground_missions': parsed_obs.get('ground_robots', {})
                }
        
        # Step 8: Return action according to mode
        if mode == 'real':
            return {
                'target_waypoint': target,
                'broadcast': broadcast_data
            }
        else:
            if getattr(self.navigator, 'uses_velocity', False):
                velocity = tuple(self.navigator.nominal_velocity)
            else:
                velocity = self._compute_velocity_to_target(target) if target else (0.0, 0.0, 0.0)
            # Arrival clipping must happen BEFORE the safety filter. Never
            # zero a CBF recovery command just because the selected action holds.
            dt=simulator_obs.get('time_step')
            if target is not None and dt is not None and dt>0:
                distance=np.linalg.norm(np.asarray(target)-self.position)
                speed=np.linalg.norm(velocity)
                if speed*dt>distance:
                    velocity=tuple(np.asarray(velocity)*distance/max(speed*dt,1e-9))
            velocity = self._safe_simulation_velocity(velocity, simulator_obs)
            return {
                'motion': np.array([velocity[0], velocity[1], velocity[2]]),
                'broadcast': broadcast_data
            }
        
    def process_observation(self, observation: Dict, 
                          other_drones_data: Optional[Dict[str, TopologicalGraph]] = None) -> Dict:
        """
        Process observation and build/update/fuse/broadcast graph.
        
        Args:
            observation: Dictionary containing:
                - 'position': (x, y, z) drone position
                - 'semantic_map': np.ndarray of terrain types
                - 'nearby_robots': List of (robot_id, (x, y)) tuples
                - 'drones': Dict of {drone_id: {'position': (x,y,z)}} for nearby drones
                - (optional) other sensor data
            other_drones_data: Optional dict of {drone_id: TopologicalGraph} from nearby drones
                
        Returns:
            Dictionary containing any messages to broadcast
        """
        current_time = time.time()
        self.position = observation['position']
        semantic_map = observation['semantic_map']
        nearby_robots = observation.get('nearby_robots', [])
        nearby_drones = observation.get('drones', {})
        
        # Step 1: Semantic processing and graph building/updating
        with self.timer.measure("semantic_processing"):
            # Build or update topological graph from semantic perception
            if self.current_graph is None:
                # First time: build new graph
                self.current_graph = self.graph_builder.build_graph(
                    semantic_map=semantic_map,
                    drone_position=self.position
                )
            else:
                # Subsequent times: incrementally update existing graph
                self.graph_builder.update_graph(
                    existing_graph=self.current_graph,
                    semantic_map=semantic_map,
                    drone_position=self.position
                )
        
        # Step 2: Check for nearby drones and perform fusion
        fusion_stats = {}
        if other_drones_data and len(nearby_drones) > 0:
            with self.timer.measure("graph_fusion"):
                fusion_stats = self._check_and_fuse_with_nearby_drones(
                    nearby_drones, 
                    other_drones_data,
                    current_time
                )
        
        self.last_fusion_stats = fusion_stats
        
        # Step 3: Prepare broadcast data with graph and position (unified for all robots)
        # NOTE: Old GraphBroadcaster system commented out - now using unified dict format
        # with self.timer.measure("broadcasting"):
        #     message = self.broadcaster.broadcast(
        #         graph=self.current_graph,
        #         current_time=current_time,
        #         nearby_robots=nearby_robots
        #     )
        
        broadcast_data = None
        if self.current_graph is not None:
            with self.timer.measure("broadcasting"):
                broadcast_data = {
                    'graph': self.current_graph.to_message(),
                    'position': list(self.position),
                    'drone_id': self.drone_id,
                    'timestamp': current_time,
                    'node_count': len(self.current_graph),
                    'edge_count': self.current_graph.graph.number_of_edges()
                }
        
        result = {
            'graph_built': True,
            'graph_nodes': len(self.current_graph),
            'graph_edges': self.current_graph.graph.number_of_edges(),
            'broadcast_data': broadcast_data,  # Unified broadcast format (dict) for all robots
            'fusion_stats': fusion_stats
        }
        
        return result
    
    def _check_and_fuse_with_nearby_drones(self, nearby_drones: Dict,
                                          other_drones_data: Dict[str, TopologicalGraph],
                                          current_time: float) -> Dict:
        """
        Check for nearby drones and fuse graphs if conditions are met.
        
        Args:
            nearby_drones: Dict of {drone_id: {'position': (x,y,z)}}
            other_drones_data: Dict of {drone_id: TopologicalGraph}
            current_time: Current timestamp
            
        Returns:
            Dictionary with fusion statistics
        """
        fusion_stats = {
            'drones_in_range': 0,
            'fusions_performed': 0,
            'total_nodes_merged': 0,
            'total_nodes_added': 0,
            'total_edges_added': 0
        }
        
        if self.current_graph is None:
            return fusion_stats
        
        # Check each nearby drone
        for other_drone_id, other_drone_info in nearby_drones.items():
            if other_drone_id == self.drone_id:
                continue
            
            other_position = other_drone_info.get('position', (0, 0, 0))
            
            # Calculate distance to other drone
            distance = np.sqrt((self.position[0] - other_position[0])**2 + 
                             (self.position[1] - other_position[1])**2)
            
            # Check if within communication range
            if distance <= self.communication_range:
                fusion_stats['drones_in_range'] += 1
                
                # Check if enough time has passed since last fusion
                if self.current_graph.should_fuse_with_drone(
                    other_drone_id, 
                    current_time, 
                    self.fusion_interval
                ):
                    # Get other drone's graph
                    other_graph = other_drones_data.get(other_drone_id)
                    
                    if other_graph is not None:
                        # Perform fusion
                        stats = self.current_graph.fuse_with_graph(
                            other_graph,
                            merge_distance_threshold=3.0,
                            current_time=current_time
                        )
                        
                        fusion_stats['fusions_performed'] += 1
                        fusion_stats['total_nodes_merged'] += stats['nodes_merged']
                        fusion_stats['total_nodes_added'] += stats['nodes_added']
                        fusion_stats['total_edges_added'] += stats['edges_added']
        
        return fusion_stats
    
    def get_timing_summary(self) -> str:
        """Get performance timing summary string."""
        return self.timer.get_summary_string()
    
    def print_timing_stats(self, summary: bool = True):
        """Print performance timing statistics."""
        self.timer.print_stats(summary=summary)
    
    def get_state(self) -> Dict:
        """
        Get current drone state.
        
        Returns:
            Dictionary with state information
        """
        return {
            'drone_id': self.drone_id,
            'position': self.position,
            'graph_nodes': len(self.current_graph) if self.current_graph else 0,
            'last_broadcast_time': self.broadcaster.last_broadcast_time
        }


# Example usage with simulator
if __name__ == "__main__":
    # This is just an example - actual usage is in test/test_multi_drone.py
    
    drone = DroneController(
        drone_id="drone_0",
        resolution=0.5,
        communication_range=100.0,
        max_speed=15.0
    )
    
    # NOTE: No manual waypoint setting needed!
    # Drones automatically discover ground robot mission from observations
    # and generate vortex exploration patterns
    
    # Simulated observation from simulator (with ground robot broadcasting target)
    simulator_obs = {
        'self': {
            'position': np.array([0.0, 0.0]),
            'height': 50.0,
            'type': 'drone'
        },
        'occupancy_map': np.zeros((60, 60)),  # 30m / 0.5m = 60 pixels
        'neighbors': [
            {
                'type': 'diff_drive',
                'relative_position': np.array([5.0, 5.0, 0.0]),
                'broadcast': {
                    'robot_id': 'robot_0',
                    'position': [5.0, 5.0],
                    'target': [500.0, 500.0]  # Ground robot's target
                }
            }
        ]
    }
    
    # Get action (handles everything: perception, mapping, fusion, navigation)
    action = drone.get_action(simulator_obs)
    
    print(f"Drone action: {action}")
    print(f"  Motion: {action['motion']}")
    print(f"  Broadcast: {action['broadcast'] is not None}")
    print(f"\nDrone state: {drone.get_state()}")
    print(f"Navigator stats: {drone.navigator.get_stats()}")
