"""
Ground Robot Main Controller
Integrates graph reception, fusion, planning, and navigation for ground robot operation
"""

import numpy as np
import time
import sys
import os
from typing import Dict, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ground.local_perception import LocalPerception
from ground.graph_receiver import GraphReceiver
from ground.graph_fusion import GraphFusion
from ground.global_planner import GlobalPlanner
from ground.local_planner import LocalPlanner
from common.graph_structure import TopologicalGraph
from common.timing import PerformanceTimer


class GroundRobotController:
    """
    Main controller for ground robot agent.
    
    Coordinates:
    - Local perception (occupancy → semantic map)
    - Graph reception from drones
    - Graph fusion with spatial merging
    - Hierarchical planning (global + local)
    - Navigation control
    """
    
    def __init__(self, robot_id: str,
                 resolution: float = 0.5,
                 perception_range: float = 8.0,
                 merge_threshold: float = 2.0,
                 max_speed: float = 1.0,
                 max_linear_vel: Optional[float] = None,
                 max_angular_vel: Optional[float] = None,
                 safety_distance: float = 1.5,
                 waypoint_threshold: float = 1.0,
                 replan_threshold: float = 5.0,
                 fusion_interval: float = 20.0,
                 edge_confidence_threshold: float = 0.0):
        """
        Initialize ground robot controller.
        
        Args:
            robot_id: Unique identifier for this robot
            resolution: Map resolution in meters per pixel (default 0.5)
            perception_range: Local perception range in meters (default 8m, LiDAR-like)
            merge_threshold: Distance threshold for graph fusion (default 2.0m)
            max_speed: Maximum speed for robot (m/s for linear, rad/s for angular) (default 1.0)
            max_linear_vel: Maximum linear velocity in m/s (overrides max_speed if provided)
            max_angular_vel: Maximum angular velocity in rad/s (overrides max_speed if provided)
            safety_distance: Safety distance to obstacles in meters (default 1.5)
            waypoint_threshold: Distance to waypoint for "reached" (default 1.0m)
            replan_threshold: Distance from path before replanning (default 5.0m)
            fusion_interval: Minimum time between fusing graphs from same drone (default 20.0s)
        """
        self.robot_id = robot_id
        self.resolution = resolution
        self.perception_range = perception_range
        self.fusion_interval = fusion_interval
        
        # Handle max_speed vs max_linear_vel/max_angular_vel
        self.max_linear_vel = max_linear_vel if max_linear_vel is not None else max_speed
        self.max_angular_vel = max_angular_vel if max_angular_vel is not None else max_speed
        
        # Initialize modules
        self.local_perception = LocalPerception(
            robot_id=robot_id,
            resolution=resolution
        )
        
        self.graph_receiver = GraphReceiver(robot_id=robot_id)
        
        self.graph_fusion = GraphFusion(
            robot_id=robot_id,
            merge_threshold=merge_threshold
        )
        
        self.global_planner = GlobalPlanner(
            robot_id=robot_id,
            replan_threshold=replan_threshold
        )
        
        self.local_planner = LocalPlanner(
            robot_id=robot_id,
            max_linear_vel=self.max_linear_vel,
            max_angular_vel=self.max_angular_vel,
            safety_distance=safety_distance
        )
        
        # Performance timer
        self.timer = PerformanceTimer(robot_id)
        
        # Uncertainty gate: the robot will not proceed along an edge whose
        # confidence is at or below this value.  Real graph edges (from
        # drone observation) always have confidence > 0; the fake edge
        # appended from the last graph node to the goal returns exactly
        # 0.0.  A threshold of 0.0 with <= blocks only the unobserved
        # fake edge.
        self.edge_confidence_threshold = edge_confidence_threshold

        # State
        self.position = None
        self.heading = None
        self.goal_position = None
        self.unified_graph = None
        
        # Statistics
        self.graphs_received = 0
        self.last_fusion_time = 0
        
        # Per-drone fusion tracking
        self.last_fusion_per_drone = {}  # {drone_id: timestamp}
        self._last_cooldown_print = {}   # {drone_id: timestamp} — throttle noisy prints
        
        # Communication state for visualization
        self.current_drone_communications = {}  # {drone_id: {'received': bool, 'fused': bool, 'last_fusion_time': float}}
        
        # Store last semantic map for visualization (debug)
        self._last_semantic_map = None
        self._last_semantic_map_position = None
    
    def set_goal(self, goal_pos: Tuple[float, float]):
        """
        Set navigation goal.
        
        Args:
            goal_pos: (x, y) goal position
        """
        self.goal_position = goal_pos
        print(f"[{self.robot_id}] Goal set to {goal_pos}")
    
    def set_target(self, target_pos: Tuple[float, float]):
        """
        Set navigation target (alias for set_goal).
        
        Args:
            target_pos: (x, y) target position
        """
        self.set_goal(target_pos)
    
    def _parse_simulator_observation(self, sim_obs: Dict) -> Dict:
        """
        Parse simulator observation into standardized format.
        
        Args:
            sim_obs: Raw observation from simulator
            
        Returns:
            Parsed observation dictionary with:
                - 'position': (x, y)
                - 'heading': theta (radians) [parsed from 'orientation']
                - 'occupancy_map': Binary occupancy map
                - 'drones': Dict of nearby drones with broadcasts
        """
        # Extract position and heading
        # Ground robots use 2D positions for planning (x, y only)
        pos_array = sim_obs['self']['position']
        position = (pos_array[0], pos_array[1])  # Extract only x, y
        heading = sim_obs['self'].get('orientation', 0.0)  # radians
        
        # Extract occupancy map
        occupancy_map = sim_obs.get('occupancy_map')
        if occupancy_map is None:
            raise ValueError(f"[{self.robot_id}] No occupancy map in simulator observation!")
        
        # Extract nearby drones from neighbors
        drones = {}
        neighbors = sim_obs.get('neighbors', [])
        
        for neighbor in neighbors:
            if neighbor['type'] == 'drone':
                # Get broadcast data from drone
                broadcast_msg = neighbor.get('broadcast')
                if broadcast_msg and isinstance(broadcast_msg, dict):
                    drone_id = broadcast_msg.get('drone_id')
                    drone_position = broadcast_msg.get('position')
                    
                    if drone_id and drone_position:
                        drones[drone_id] = {
                            'position': tuple(drone_position),
                            'broadcast': broadcast_msg
                        }
        
        return {
            'position': position,
            'heading': heading,
            'occupancy_map': occupancy_map,
            'drones': drones
        }
    
    def _receive_and_fuse_graphs(self, drones_data: Dict) -> bool:
        """
        Receive graphs from nearby drones and perform fusion.
        
        Args:
            drones_data: Dict of {drone_id: {'position': ..., 'broadcast': ...}}
            
        Returns:
            True if any graphs were received and fused
        """
        if not drones_data:
            # Clear communication state if no drones in range
            self.current_drone_communications.clear()
            return False
        
        current_time = time.time()
        graphs_fused = False
        
        # Reset communication state for this step
        self.current_drone_communications.clear()
        
        for drone_id, drone_info in drones_data.items():
            broadcast = drone_info.get('broadcast')
            if broadcast and 'graph' in broadcast:
                try:
                    # Deserialize graph
                    graph = TopologicalGraph.from_message(broadcast['graph'])
                    
                    # Validate graph
                    if len(graph) > 0:
                        # Mark as received
                        last_fusion_time = self.last_fusion_per_drone.get(drone_id, 0)
                        time_since_last_fusion = current_time - last_fusion_time
                        
                        # Check if we should fuse based on interval
                        should_fuse = time_since_last_fusion >= self.fusion_interval
                        
                        if should_fuse:
                            # Add to fusion
                            self.graph_fusion.add_drone_graph(drone_id, graph)
                            self.graphs_received += 1
                            self.last_fusion_per_drone[drone_id] = current_time
                            graphs_fused = True
                            
                            print(f"[{self.robot_id}] Fused graph from {drone_id}: "
                                  f"{len(graph)} nodes, {graph.graph.number_of_edges()} edges "
                                  f"(last fusion: {time_since_last_fusion:.1f}s ago)")
                            
                            # Update communication state
                            self.current_drone_communications[drone_id] = {
                                'received': True,
                                'fused': True,
                                'last_fusion_time': current_time
                            }
                        else:
                            # Received but not fused (cooldown) — print at most once per 10s
                            cooldown_remaining = self.fusion_interval - time_since_last_fusion
                            if current_time - self._last_cooldown_print.get(drone_id, 0) >= 10.0:
                                print(f"[{self.robot_id}] Received graph from {drone_id} "
                                      f"(cooldown: {cooldown_remaining:.1f}s remaining)")
                                self._last_cooldown_print[drone_id] = current_time
                            
                            # Update communication state
                            self.current_drone_communications[drone_id] = {
                                'received': True,
                                'fused': False,
                                'last_fusion_time': last_fusion_time
                            }
                            
                except Exception as e:
                    print(f"[{self.robot_id}] Error receiving graph from {drone_id}: {e}")
        
        if graphs_fused:
            self.last_fusion_time = current_time
        
        return graphs_fused
    
    def get_action(self, simulator_obs: Dict) -> Dict:
        """
        Main method: processes observation and returns complete action.
        
        get_action():
        1. Parse observation
        2. Process perception
        3. Receive & fuse graphs → returns graphs_fused=True/False
        4. Planning:
            - If graphs_fused OR no_path OR deviated:
                → Call planner (decides graph vs direct inside)
            - Else:
                → Advance waypoint
        5. Local planning
        6. Return action
        
        Args:
            simulator_obs: Raw observation from simulator
            
        Returns:
            Dictionary with:
                - 'motion': (linear_vel, angular_vel) velocity command
                - 'broadcast': None (ground robots don't broadcast yet)
        """
        current_time = time.time()
        
        # Step 1: Parse simulator observation
        with self.timer.measure("observation_parsing"):
            parsed_obs = self._parse_simulator_observation(simulator_obs)
            self.position = parsed_obs['position']
            self.heading = parsed_obs['heading']
        
        # Step 2: Process local perception (occupancy → semantic)
        with self.timer.measure("local_perception"):
            semantic_map = self.local_perception.process_occupancy_map(
                parsed_obs['occupancy_map']
            )
            # Store for visualization (debug)
            self._last_semantic_map = semantic_map
            self._last_semantic_map_position = self.position
        
        # Step 3: Receive and fuse graphs from drones
        with self.timer.measure("graph_reception_fusion"):
            graphs_received = self._receive_and_fuse_graphs(parsed_obs['drones'])
            
            if graphs_received:
                # Get updated unified graph
                self.unified_graph = self.graph_fusion.get_unified_graph()
        
        # Step 4: Global planning
        current_waypoint = None
        
        if self.goal_position is not None:
            with self.timer.measure("global_planning"):
                needs_replan = False
                
                # Check if we need to plan or replan
                if self.global_planner.current_path is None:
                    # Initial planning
                    needs_replan = True
                    print(f"[{self.robot_id}] Initial planning...")
                elif graphs_received and self.unified_graph is not None:
                    # New graph fused - re-evaluate planning strategy
                    needs_replan = True
                    print(f"[{self.robot_id}] New graph fused, re-evaluating route...")
                else:
                    # Check if goal reached
                    if self.global_planner.is_goal_reached(self.position):
                        print(f"[{self.robot_id}] Goal reached!")
                        self.global_planner.clear_path()
                        self.goal_position = None
                    else:
                        # Advance waypoint if reached
                        self.global_planner.advance_waypoint(self.position)
                        
                        # Check if should replan due to deviation
                        if self.global_planner.should_replan(self.position):
                            print(f"[{self.robot_id}] Deviated from path, replanning...")
                            needs_replan = True
                
                # Execute replanning if needed
                if needs_replan:
                    self.global_planner.plan_path(
                        start_pos=self.position,
                        goal_pos=self.goal_position,
                        graph=self.unified_graph
                    )
                    # path[0] is the start_pos anchor (robot is already there).
                    # Advance past it immediately so the first real destination is
                    # served on this control cycle, not the next one.
                    # Without this, every replan (triggered by new graph fusions)
                    # keeps the robot targeting its own position → zero velocity.
                    self.global_planner.advance_waypoint(self.position)
                
                # Get current waypoint
                current_waypoint = self.global_planner.get_current_waypoint()

                # Uncertainty gate: hold position if the upcoming edge is
                # through unexplored terrain (confidence below threshold).
                if (current_waypoint is not None
                        and self.unified_graph is not None
                        and self.position is not None):
                    conf = self.unified_graph.edge_confidence_from_positions(
                        self.position, current_waypoint)
                    if conf <= self.edge_confidence_threshold:
                        self.global_planner.waiting_at_frontier = True
                        current_waypoint = None
                    else:
                        self.global_planner.waiting_at_frontier = False
        
        # Step 5: Local planning (compute velocity command)
        with self.timer.measure("local_planning"):
            if current_waypoint is None:
                velocity_cmd = (0.0, 0.0)
            else:
                velocity_cmd = self.local_planner.compute_velocity_command(
                    current_pos=self.position,
                    current_heading=self.heading,
                    target_waypoint=current_waypoint,
                    local_semantic_map=semantic_map,
                    resolution=self.resolution
                )
        
        # Step 6: Return action with broadcast (for drone coordination)
        return {
            'motion': velocity_cmd,
            'broadcast': {
                'robot_id': self.robot_id,
                'position': list(self.position) if self.position else [0.0, 0.0],
                'target': list(self.goal_position) if self.goal_position else None,
                'timestamp': current_time
            }
        }
    
    def get_state(self) -> Dict:
        """
        Get current robot state.
        
        Returns:
            Dictionary with state information
        """
        state = {
            'robot_id': self.robot_id,
            'position': self.position,
            'heading': self.heading,
            'goal_position': self.goal_position,
            'has_graph': self.unified_graph is not None,
            'graphs_received': self.graphs_received
        }
        
        if self.unified_graph is not None:
            state['unified_graph_nodes'] = len(self.unified_graph)
            state['unified_graph_edges'] = self.unified_graph.graph.number_of_edges()
        
        if self.global_planner.current_path is not None:
            remaining = self.global_planner.get_remaining_path()
            state['waypoints_remaining'] = len(remaining) if remaining else 0
            state['current_waypoint'] = self.global_planner.get_current_waypoint()
            state['using_graph'] = self.global_planner.using_graph
        else:
            state['waypoints_remaining'] = 0
            state['current_waypoint'] = None
            state['using_graph'] = False
        
        return state
    
    def get_timing_summary(self) -> str:
        """Get performance timing summary string."""
        return self.timer.get_summary_string()
    
    def print_timing_stats(self, summary: bool = True):
        """Print performance timing statistics."""
        self.timer.print_stats(summary=summary)
    
    def reset(self):
        """Reset controller state."""
        self.position = None
        self.heading = None
        self.goal_position = None
        self.unified_graph = None
        self.graphs_received = 0
        self.global_planner.clear_path()
        self.graph_fusion.clear()
        print(f"[{self.robot_id}] Controller reset")


# Example usage
if __name__ == "__main__":
    # This is just an example - actual usage will be in test files
    
    robot = GroundRobotController(
        robot_id="robot_0",
        resolution=0.5,
        perception_range=8.0,
        max_speed=1.0
    )
    
    # Set navigation goal (can use set_goal or set_target)
    robot.set_target((100.0, 100.0))
    
    # Simulated observation from simulator
    simulator_obs = {
        'self': {
            'position': np.array([0.0, 0.0, 0.0]),  # 3D position (z=0 for ground)
            'orientation': 0.0,
            'type': 'diff_drive'
        },
        'occupancy_map': np.zeros((16, 16)),  # 8m / 0.5m = 16 pixels
        'neighbors': []  # List of neighbor dicts (drones, etc.)
    }
    
    # Get action (handles everything: perception, graph fusion, planning, control)
    action = robot.get_action(simulator_obs)
    
    print(f"Robot action: {action}")
    print(f"  Motion: {action['motion']}")
    print(f"  Broadcast: {action['broadcast']}")
    print(f"\nRobot state: {robot.get_state()}")
