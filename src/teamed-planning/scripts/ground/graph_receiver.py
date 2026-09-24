"""
Graph Receiver for Ground Robots
Handles receiving and deserializing topological graphs from drones
"""

import sys
import os
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.graph_structure import TopologicalGraph
from common.message_definitions import GraphBroadcastMessage, MessageParser, MessageType


class GraphReceiver:
    """
    Receives and processes topological graphs from drones.
    
    Features:
    - Message deserialization
    - Graph validation
    - Statistics tracking
    """
    
    def __init__(self, robot_id: str):
        """
        Initialize graph receiver.
        
        Args:
            robot_id: Unique identifier for this robot
        """
        self.robot_id = robot_id
        self.received_graphs = {}  # {drone_id: TopologicalGraph}
        self.reception_stats = {
            'total_received': 0,
            'total_nodes': 0,
            'total_edges': 0,
            'unique_drones': set()
        }
        
    def receive_message(self, message: GraphBroadcastMessage) -> Optional[TopologicalGraph]:
        """
        Receive and process a graph broadcast message.
        
        Args:
            message: GraphBroadcastMessage from a drone
            
        Returns:
            Deserialized TopologicalGraph or None if invalid
        """
        if not isinstance(message, GraphBroadcastMessage):
            print(f"[Robot {self.robot_id}] Received non-graph message")
            return None
        
        try:
            # Deserialize graph
            graph = TopologicalGraph.from_message(message.graph_data)
            
            # Validate graph
            if len(graph) == 0:
                print(f"[Robot {self.robot_id}] Received empty graph from {message.sender_id}")
                return None
            
            # Store graph
            self.received_graphs[message.sender_id] = graph
            
            # Update statistics
            self.reception_stats['total_received'] += 1
            self.reception_stats['total_nodes'] += message.node_count
            self.reception_stats['total_edges'] += message.edge_count
            self.reception_stats['unique_drones'].add(message.sender_id)
            
            print(f"[Robot {self.robot_id}] Received graph from {message.sender_id}: "
                  f"{message.node_count} nodes, {message.edge_count} edges")
            
            return graph
            
        except Exception as e:
            print(f"[Robot {self.robot_id}] Error deserializing graph: {e}")
            return None
    
    def get_received_graph(self, drone_id: str) -> Optional[TopologicalGraph]:
        """
        Get a previously received graph from a specific drone.
        
        Args:
            drone_id: ID of the drone
            
        Returns:
            TopologicalGraph or None if not received
        """
        return self.received_graphs.get(drone_id)
    
    def get_all_received_graphs(self) -> List[TopologicalGraph]:
        """
        Get all received graphs.
        
        Returns:
            List of TopologicalGraph objects
        """
        return list(self.received_graphs.values())
    
    def clear_old_graphs(self, max_age: float = 60.0):
        """
        Clear graphs older than max_age seconds.
        
        Args:
            max_age: Maximum age of graphs to keep (seconds)
        """
        import time
        current_time = time.time()
        
        to_remove = []
        for drone_id, graph in self.received_graphs.items():
            graph_time = graph.metadata.get('timestamp', 0)
            age = current_time - graph_time
            
            if age > max_age:
                to_remove.append(drone_id)
        
        for drone_id in to_remove:
            del self.received_graphs[drone_id]
            print(f"[Robot {self.robot_id}] Removed old graph from {drone_id}")
    
    def get_stats(self) -> dict:
        """
        Get reception statistics.
        
        Returns:
            Dictionary with reception statistics
        """
        stats = self.reception_stats.copy()
        stats['unique_drones'] = len(stats['unique_drones'])
        stats['current_graphs'] = len(self.received_graphs)
        return stats
