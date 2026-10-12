"""
Graph Fusion Module for Ground Robots
Merges topological graphs from multiple drones into a unified graph
Includes uncertainty handling (placeholder for future probabilistic fusion)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.graph_structure import TopologicalGraph


class GraphFusion:
    """
    Fuses topological graphs from multiple sources into a unified graph.
    
    Version 1: Simple spatial merging with distance-based thresholding
    Future versions will add:
    - Probabilistic merging with covariance tracking
    - Mahalanobis distance for node association
    - Graph consistency validation
    - SLAM-based uncertainty propagation
    """
    
    def __init__(self, robot_id: str, merge_threshold: float = 2.0):
        """
        Initialize graph fusion.
        
        Args:
            robot_id: ID of the ground robot
            merge_threshold: Distance threshold for merging nodes (meters)
                           Nodes closer than this are considered the same location
        """
        self.robot_id = robot_id
        self.merge_threshold = merge_threshold
        self.unified_graph = TopologicalGraph(drone_id=f"{robot_id}_unified")
        self.source_graphs = {}  # {drone_id: TopologicalGraph}
        self._merge_buckets = {}
        
        # Uncertainty handling (placeholder for future)
        self.uncertainty_enabled = False
        
    def add_drone_graph(self, drone_id: str, graph: TopologicalGraph):
        """
        Add or update a graph from a drone.
        
        Args:
            drone_id: ID of the drone that created the graph
            graph: TopologicalGraph to add
        """
        self.source_graphs[drone_id] = graph
        self._rebuild_unified_graph()
        
    def _rebuild_unified_graph(self):
        """
        Rebuild the unified graph from all source graphs.
        
        This is a simple version that rebuilds from scratch.
        More efficient incremental updates can be added later.
        """
        # Create new unified graph
        self.unified_graph = TopologicalGraph(drone_id=f"{self.robot_id}_unified")
        self._merge_buckets = {}
        
        # Track which nodes have been added
        # Maps (drone_id, original_node_id) -> unified_node_id
        node_mapping = {}
        
        # Maps unified_node_id -> list of (drone_id, original_node_id)
        unified_to_sources = {}
        
        # Add all nodes from all graphs, merging close ones
        for drone_id, graph in self.source_graphs.items():
            for node_id, node_data in graph.graph.nodes(data=True):
                position = node_data['position']
                
                # Find if there's already a nearby node in unified graph
                merged_node_id = self._find_merge_candidate(position)
                
                if merged_node_id is None:
                    # Create new node in unified graph
                    unified_node_id = f"unified_{len(self.unified_graph)}"
                    
                    # Use attributes from this node
                    attributes = {k: v for k, v in node_data.items() if k != 'position'}
                    self.unified_graph.add_node(unified_node_id, position, attributes)
                    self._merge_buckets.setdefault(self._merge_cell(position), []).append(unified_node_id)
                    
                    # Track mapping
                    node_mapping[(drone_id, node_id)] = unified_node_id
                    unified_to_sources[unified_node_id] = [(drone_id, node_id)]
                else:
                    # Merge with existing node
                    node_mapping[(drone_id, node_id)] = merged_node_id
                    unified_to_sources[merged_node_id].append((drone_id, node_id))
                    
                    # Update position as weighted average
                    self._update_merged_node_position(merged_node_id, position,
                                                     unified_to_sources[merged_node_id])
        
        # Add edges from all source graphs
        for drone_id, graph in self.source_graphs.items():
            for edge in graph.graph.edges(data=True):
                node1, node2, edge_data = edge
                
                # Get unified node IDs
                unified_node1 = node_mapping.get((drone_id, node1))
                unified_node2 = node_mapping.get((drone_id, node2))
                
                if unified_node1 and unified_node2 and unified_node1 != unified_node2:
                    # Add edge if it doesn't exist, or update weight if it does
                    if self.unified_graph.graph.has_edge(unified_node1, unified_node2):
                        # Average weights if edge already exists
                        existing_weight = self.unified_graph.graph[unified_node1][unified_node2].get('weight', 1.0)
                        new_weight = edge_data.get('weight', 1.0)
                        avg_weight = (existing_weight + new_weight) / 2.0
                        self.unified_graph.graph[unified_node1][unified_node2]['weight'] = avg_weight
                    else:
                        # Add new edge
                        weight = edge_data.get('weight')
                        attributes = {k: v for k, v in edge_data.items() if k != 'weight'}
                        self.unified_graph.add_edge(unified_node1, unified_node2, 
                                                   weight=weight, attributes=attributes)
        
        # Rebuild spatial index for efficient queries
        self.unified_graph.build_spatial_index()
        
        print(f"[GraphFusion] Unified graph: {len(self.unified_graph)} nodes, "
              f"{self.unified_graph.graph.number_of_edges()} edges from "
              f"{len(self.source_graphs)} drones")
    
    def _find_merge_candidate(self, position: Tuple[float, float]) -> Optional[str]:
        """
        Find a node in the unified graph that should be merged with this position.
        
        Args:
            position: (x, y) position to check
            
        Returns:
            Node ID of merge candidate, or None if no candidate found
        """
        if len(self.unified_graph) == 0:
            return None
        
        # Rebuilding a KD-tree after every inserted/averaged node is quadratic.
        # A threshold-sized spatial grid finds exactly the same nearby merge
        # candidates while supporting incremental position updates.
        cx, cy = self._merge_cell(position)
        best_node, best_distance = None, float('inf')
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for node_id in self._merge_buckets.get((cx + dx, cy + dy), []):
                    distance = float(np.linalg.norm(np.asarray(position) -
                                                   self.unified_graph.get_node_position(node_id)))
                    if distance <= self.merge_threshold and distance < best_distance:
                        best_node, best_distance = node_id, distance
        return best_node

    def _merge_cell(self, position):
        size = self.merge_threshold if self.merge_threshold > 0 else 1.
        return tuple(int(np.floor(value / size)) for value in position[:2])
    
    def _update_merged_node_position(self, node_id: str, new_position: Tuple[float, float],
                                     source_nodes: List[Tuple[str, str]]):
        """
        Update position of a merged node as weighted average.
        
        Args:
            node_id: Unified node ID
            new_position: New position to incorporate
            source_nodes: List of (drone_id, original_node_id) that contribute to this node
        """
        # For now, simple average (can be weighted by confidence later)
        current_pos = self.unified_graph.get_node_position(node_id)
        
        # Average with new position
        num_sources = len(source_nodes)
        avg_x = (current_pos[0] * (num_sources - 1) + new_position[0]) / num_sources
        avg_y = (current_pos[1] * (num_sources - 1) + new_position[1]) / num_sources
        
        # Update node position
        self.unified_graph.graph.nodes[node_id]['position'] = (avg_x, avg_y)
        old_cell, new_cell = self._merge_cell(current_pos), self._merge_cell((avg_x, avg_y))
        if old_cell != new_cell:
            self._merge_buckets[old_cell].remove(node_id)
            self._merge_buckets.setdefault(new_cell, []).append(node_id)
        
        # Invalidate spatial index
        self.unified_graph._spatial_index_valid = False
    
    def get_unified_graph(self) -> TopologicalGraph:
        """
        Get the unified graph.
        
        Returns:
            Unified TopologicalGraph
        """
        return self.unified_graph
    
    def get_path(self, start_pos: Tuple[float, float], 
                goal_pos: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Find path from start to goal using unified graph.
        
        Args:
            start_pos: (x, y) starting position
            goal_pos: (x, y) goal position
            
        Returns:
            List of (x, y) waypoints, or None if no path found
        """
        if len(self.unified_graph) == 0:
            print("[GraphFusion] No unified graph available")
            return None
        
        # Find nearest nodes to start and goal
        start_node = self.unified_graph.find_nearest_node(start_pos)
        goal_node = self.unified_graph.find_nearest_node(goal_pos)
        
        if start_node is None or goal_node is None:
            print("[GraphFusion] Cannot find nodes near start or goal")
            return None
        
        start_node_id, start_dist = start_node
        goal_node_id, goal_dist = goal_node
        
        print(f"[GraphFusion] Planning from {start_node_id} to {goal_node_id}")
        
        # Find path
        node_path = self.unified_graph.get_path(start_node_id, goal_node_id)
        
        if node_path is None:
            print("[GraphFusion] No path found in graph")
            return None
        
        # Convert node IDs to positions
        waypoints = []
        for node_id in node_path:
            pos = self.unified_graph.get_node_position(node_id)
            waypoints.append(pos)
        
        return waypoints
    
    def clear(self):
        """Clear all graphs."""
        self.unified_graph = TopologicalGraph(drone_id=f"{self.robot_id}_unified")
        self.source_graphs.clear()
    
    # ========== Uncertainty Handling Methods (Placeholder for Future) ==========
    
    def compute_merge_probability(self, pos1: Tuple[float, float],
                                  pos2: Tuple[float, float],
                                  cov1: Optional[np.ndarray] = None,
                                  cov2: Optional[np.ndarray] = None) -> float:
        """
        Compute probability that two nodes should be merged.
        
        Version 1: Returns 1.0 if within threshold, 0.0 otherwise
        Future: Use Mahalanobis distance with covariance matrices from broadcast
        
        Args:
            pos1: Position of first node
            pos2: Position of second node
            cov1: Covariance matrix of first node (future use)
            cov2: Covariance matrix of second node (future use)
            
        Returns:
            Probability of merge (0.0 to 1.0)
        """
        if not self.uncertainty_enabled:
            # Simple distance-based (already handled in _find_merge_candidate)
            distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            return 1.0 if distance <= self.merge_threshold else 0.0
        
        # Future: Mahalanobis distance with covariances
        # if cov1 is not None and cov2 is not None:
        #     combined_cov = cov1 + cov2
        #     diff = np.array(pos2) - np.array(pos1)
        #     inv_cov = np.linalg.inv(combined_cov)
        #     mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
        #     probability = scipy.stats.chi2.cdf(mahal_dist**2, df=2)
        #     return probability
        
        return 1.0
    
    def validate_graph_consistency(self, graph1: TopologicalGraph, 
                                   graph2: TopologicalGraph) -> bool:
        """
        Check consistency between two overlapping graphs.
        
        Version 1: Always returns True (no validation)
        Future: Check for conflicting edges, incompatible node positions
        
        Args:
            graph1: First TopologicalGraph
            graph2: Second TopologicalGraph
            
        Returns:
            True if graphs are consistent
        """
        if not self.uncertainty_enabled:
            return True
        
        # Future: Implement consistency checks
        # - Find overlapping nodes
        # - Check if edge existence is consistent
        # - Validate that node positions are within uncertainty bounds
        # - Check for loop closure consistency
        
        return True
    
    def estimate_node_uncertainty(self, node_sources: List[Tuple[str, str]]) -> float:
        """
        Estimate position uncertainty for a merged node.
        
        Version 1: Returns fixed small uncertainty
        Future: Compute based on source uncertainties and agreement
        
        Args:
            node_sources: List of (drone_id, original_node_id) contributing to node
            
        Returns:
            Estimated position uncertainty (meters)
        """
        if not self.uncertainty_enabled:
            # Nominal small uncertainty
            return 0.1
        
        # Future: If multiple drones observe same node with different positions,
        # compute uncertainty from variance
        # If node sources have covariances, propagate them
        
        return 0.1
    
    def get_stats(self) -> Dict:
        """
        Get fusion statistics.
        
        Returns:
            Dictionary with fusion statistics
        """
        return {
            'robot_id': self.robot_id,
            'unified_nodes': len(self.unified_graph),
            'unified_edges': self.unified_graph.graph.number_of_edges(),
            'source_graphs': len(self.source_graphs),
            'merge_threshold': self.merge_threshold,
            'uncertainty_enabled': self.uncertainty_enabled
        }
