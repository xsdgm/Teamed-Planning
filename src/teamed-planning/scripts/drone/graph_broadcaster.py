"""
Graph Broadcaster for Drones
Handles broadcasting of topological graphs to nearby ground robots
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.graph_structure import TopologicalGraph
from common.message_definitions import GraphBroadcastMessage, MessageParser


class GraphBroadcaster:
    """
    Manages broadcasting of topological graphs from drones.
    
    Features:
    - Periodic broadcasting
    - Proximity-based triggering
    - Message serialization and compression
    """
    
    def __init__(self, drone_id: str, broadcast_interval: float = 5.0,
                 broadcast_range: float = 100.0):
        """
        Initialize graph broadcaster.
        
        Args:
            drone_id: Unique identifier for this drone
            broadcast_interval: Time between broadcasts in seconds
            broadcast_range: Range within which to broadcast graphs (meters)
        """
        self.drone_id = drone_id
        self.broadcast_interval = broadcast_interval
        self.broadcast_range = broadcast_range
        
        self.last_broadcast_time = 0.0
        self.last_broadcast_graph = None
        
    def should_broadcast(self, current_time: float, 
                        nearby_robots: List[Tuple[str, Tuple[float, float]]]) -> bool:
        """
        Determine if drone should broadcast based on time and robot proximity.
        
        Args:
            current_time: Current simulation time
            nearby_robots: List of (robot_id, (x, y)) tuples for nearby robots
            
        Returns:
            True if should broadcast
        """
        # Check if enough time has passed since last broadcast
        time_since_broadcast = current_time - self.last_broadcast_time
        
        if time_since_broadcast >= self.broadcast_interval:
            # Check if there are any robots in range
            if len(nearby_robots) > 0:
                return True
        
        return False
    
    def create_broadcast_message(self, graph: TopologicalGraph) -> Optional[GraphBroadcastMessage]:
        """
        Create a broadcast message from a topological graph.
        
        Args:
            graph: TopologicalGraph to broadcast
            
        Returns:
            GraphBroadcastMessage or None if graph is empty
        """
        if len(graph) == 0:
            return None
        
        # Serialize graph
        graph_data = graph.to_message()
        
        # Create message
        message = GraphBroadcastMessage(
            sender_id=self.drone_id,
            graph_data=graph_data,
            coverage_bounds=graph.metadata.get('bounds'),
            node_count=len(graph),
            edge_count=graph.graph.number_of_edges()
        )
        
        return message
    
    def broadcast(self, graph: TopologicalGraph, current_time: float,
                 nearby_robots: List[Tuple[str, Tuple[float, float]]]) -> Optional[GraphBroadcastMessage]:
        """
        Attempt to broadcast graph if conditions are met.
        
        Args:
            graph: TopologicalGraph to broadcast
            current_time: Current simulation time
            nearby_robots: List of (robot_id, position) tuples
            
        Returns:
            GraphBroadcastMessage if broadcast occurred, None otherwise
        """
        if not self.should_broadcast(current_time, nearby_robots):
            return None
        
        message = self.create_broadcast_message(graph)
        
        if message is not None:
            self.last_broadcast_time = current_time
            self.last_broadcast_graph = graph
            
            print(f"[Drone {self.drone_id}] Broadcasting graph with "
                  f"{message.node_count} nodes to {len(nearby_robots)} robots")
        
        return message
    
    def get_robots_in_range(self, drone_position: Tuple[float, float, float],
                           all_robot_positions: Dict[str, Tuple[float, float]]) -> List[Tuple[str, Tuple[float, float]]]:
        """
        Find all robots within broadcast range.
        
        Args:
            drone_position: (x, y, z) position of drone
            all_robot_positions: Dictionary mapping robot_id to (x, y) position
            
        Returns:
            List of (robot_id, position) tuples for robots in range
        """
        nearby_robots = []
        
        for robot_id, robot_pos in all_robot_positions.items():
            # Compute horizontal distance (ignore z for ground robots)
            distance = np.sqrt((drone_position[0] - robot_pos[0])**2 + 
                             (drone_position[1] - robot_pos[1])**2)
            
            if distance <= self.broadcast_range:
                nearby_robots.append((robot_id, robot_pos))
        
        return nearby_robots


class RobotDetector:
    """
    Detects nearby ground robots for targeted broadcasting.
    
    In simulation, this uses ground truth positions.
    In real system, this could use visual detection or communication beacons.
    """
    
    def __init__(self, detection_range: float = 100.0):
        """
        Initialize robot detector.
        
        Args:
            detection_range: Maximum range for robot detection (meters)
        """
        self.detection_range = detection_range
    
    def detect_robots(self, drone_position: Tuple[float, float, float],
                     robot_positions: Dict[str, Tuple[float, float]]) -> List[str]:
        """
        Detect robots within range.
        
        Args:
            drone_position: (x, y, z) position of drone
            robot_positions: Dictionary mapping robot_id to (x, y) position
            
        Returns:
            List of robot IDs within detection range
        """
        detected_robots = []
        
        for robot_id, robot_pos in robot_positions.items():
            distance = np.sqrt((drone_position[0] - robot_pos[0])**2 + 
                             (drone_position[1] - robot_pos[1])**2)
            
            if distance <= self.detection_range:
                detected_robots.append(robot_id)
        
        return detected_robots
    
    def get_relative_positions(self, drone_position: Tuple[float, float, float],
                              robot_positions: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """
        Get relative positions of detected robots.
        
        Args:
            drone_position: (x, y, z) position of drone
            robot_positions: Dictionary mapping robot_id to (x, y) position
            
        Returns:
            Dictionary mapping robot_id to relative (dx, dy) position
        """
        detected = self.detect_robots(drone_position, robot_positions)
        relative_positions = {}
        
        for robot_id in detected:
            robot_pos = robot_positions[robot_id]
            relative_positions[robot_id] = (
                robot_pos[0] - drone_position[0],
                robot_pos[1] - drone_position[1]
            )
        
        return relative_positions
