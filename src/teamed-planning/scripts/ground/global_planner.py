"""
Global Planner for Ground Robots
Plans high-level paths using the topological graph from drones
Supports multi-component planning for disconnected graphs
"""

import numpy as np
from typing import List, Tuple, Optional, Set, Dict
import sys
import os
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.graph_structure import TopologicalGraph


class GlobalPlanner:
    """
    Global path planner using topological graph.
    
    Generates high-level waypoints for the robot to follow.
    The local planner handles obstacle avoidance between waypoints.
    
    Strategy:
    - If graph available and efficient: use graph-based planning
    - Otherwise: generate direct waypoints toward goal
    """
    
    def __init__(self, robot_id: str, replan_threshold: float = 5.0,
                 direct_waypoint_distance: float = 20.0,
                 graph_preference_threshold: float = 10.0):
        """
        Initialize global planner.
        
        Args:
            robot_id: Unique identifier for this robot
            replan_threshold: Distance from path before replanning (meters)
            direct_waypoint_distance: Distance for direct waypoints (meters)
            graph_preference_threshold: Prefer graph if path < threshold * direct_distance
        """
        self.robot_id = robot_id
        self.replan_threshold = replan_threshold
        self.direct_waypoint_distance = direct_waypoint_distance
        self.graph_preference_threshold = graph_preference_threshold
        
        self.current_path = None  # List of waypoints
        self.current_waypoint_idx = 0
        self.goal_position = None
        self.using_graph = False
        self.waiting_at_frontier = False
        self.start_connection_idx = None
        self.entry_validator = None

    def _entry_node(self, start_pos, graph):
        """Choose the nearest entry that local observations allow approaching."""
        if self.entry_validator is None:
            return graph.find_nearest_node(start_pos)
        candidates = sorted(
            ((node, float(np.linalg.norm(np.asarray(data['position']) - start_pos)))
             for node, data in graph.graph.nodes(data=True)),
            key=lambda item: (item[1], str(item[0])))
        for node, distance in candidates:
            if self.entry_validator(graph.get_node_position(node)):
                return node, distance
        return None
        
    def plan_path(self, start_pos: Tuple[float, float],
                 goal_pos: Tuple[float, float],
                 graph: Optional[TopologicalGraph] = None) -> Optional[List[Tuple[float, float]]]:
        """
        Plan a path from start to goal.
        
        Decides whether to use graph-based planning or direct waypoints.
        
        Args:
            start_pos: (x, y) starting position
            goal_pos: (x, y) goal position
            graph: TopologicalGraph to use for planning (optional)
            
        Returns:
            List of (x, y) waypoints, or None if no path found
        """
        self.start_connection_idx = None
        # Decide strategy: graph-based or direct
        if graph is not None and len(graph) > 0 and self.should_use_graph(start_pos, goal_pos, graph):
            # Use graph-based planning
            try:
                waypoints = self._plan_with_graph(start_pos, goal_pos, graph)
            except Exception as e:
                print(f"[Robot {self.robot_id}] Graph planning error (falling back to direct): {e}")
                waypoints = None
            if waypoints is not None:
                self.using_graph = True
                self.current_path = waypoints
                self.current_waypoint_idx = 0
                self.goal_position = goal_pos
                
                path_length = self._compute_path_length(waypoints)
                print(f"[Robot {self.robot_id}] Planned with GRAPH: {len(waypoints)} waypoints, "
                      f"{path_length:.1f}m total")
                return waypoints
        
        # Fallback: use direct waypoints
        waypoints = self.generate_direct_waypoints(start_pos, goal_pos)
        self.using_graph = False
        self.start_connection_idx = None
        self.waiting_at_frontier = False
        self.current_path = waypoints
        self.current_waypoint_idx = 0
        self.goal_position = goal_pos
        
        path_length = self._compute_path_length(waypoints)
        print(f"[Robot {self.robot_id}] Planned DIRECT (no graph or graph too inefficient): "
              f"{len(waypoints)} waypoints, {path_length:.1f}m total")
        if len(waypoints) > 0:
            print(f"[Robot {self.robot_id}]   First waypoint: ({waypoints[0][0]:.1f}, {waypoints[0][1]:.1f})")
        
        return waypoints
    
    def should_use_graph(self, start_pos: Tuple[float, float],
                        goal_pos: Tuple[float, float],
                        graph: TopologicalGraph) -> bool:
        """
        Decide whether to use graph for planning.
        
        Graph is used if:
        - Estimated graph path < threshold * direct_distance
        
        Args:
            start_pos: Starting position
            goal_pos: Goal position
            graph: Available topological graph
            
        Returns:
            True if should use graph
        """
        # Find nearest nodes
        start_node = self._entry_node(start_pos, graph)
        goal_node = graph.find_nearest_node(goal_pos)
        
        if start_node is None or goal_node is None:
            return False
        
        start_node_id, start_dist = start_node
        goal_node_id, goal_dist = goal_node
        
        # Estimate total graph distance
        graph_path_estimate = self._estimate_graph_path_length(
            start_pos, goal_pos, graph, start_node_id, goal_node_id, start_dist, goal_dist
        )
        
        # Direct distance
        direct_dist = np.sqrt((goal_pos[0] - start_pos[0])**2 + 
                             (goal_pos[1] - start_pos[1])**2)
        
        # Use graph if estimated path is within threshold
        use_graph = graph_path_estimate < direct_dist * self.graph_preference_threshold
        
        if use_graph:
            print(f"[Robot {self.robot_id}] Graph route: ~{graph_path_estimate:.1f}m vs "
                  f"direct: {direct_dist:.1f}m → Using GRAPH")
        else:
            print(f"[Robot {self.robot_id}] Graph route: ~{graph_path_estimate:.1f}m vs "
                  f"direct: {direct_dist:.1f}m → Using DIRECT")
        
        return use_graph
    
    def _estimate_graph_path_length(self, start_pos: Tuple[float, float],
                                    goal_pos: Tuple[float, float],
                                    graph: TopologicalGraph,
                                    start_node_id: str, goal_node_id: str,
                                    start_dist: float, goal_dist: float) -> float:
        """
        Estimate path length through graph (heuristic).
        
        Estimate = dist(start → start_node) + dist(start_node → goal_node) + dist(goal_node → goal)
        
        Args:
            start_pos: Start position
            goal_pos: Goal position
            graph: Topological graph
            start_node_id: Nearest node to start
            goal_node_id: Nearest node to goal
            start_dist: Distance from start to start_node
            goal_dist: Distance from goal to goal_node
            
        Returns:
            Estimated path length through graph
        """
        # Distance to/from graph nodes
        connection_dist = start_dist + goal_dist
        
        # Heuristic for path through graph: straight line between nodes
        start_node_pos = graph.get_node_position(start_node_id)
        goal_node_pos = graph.get_node_position(goal_node_id)
        
        graph_dist = np.sqrt((goal_node_pos[0] - start_node_pos[0])**2 + 
                            (goal_node_pos[1] - start_node_pos[1])**2)
        
        return connection_dist + graph_dist
    
    def _plan_with_graph(self, start_pos: Tuple[float, float],
                        goal_pos: Tuple[float, float],
                        graph: TopologicalGraph) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path using topological graph.
        
        Handles both connected and disconnected graphs using multi-component planning.
        Does NOT include start position - only destinations to navigate to.
        
        Args:
            start_pos: Starting position
            goal_pos: Goal position
            graph: Topological graph
            
        Returns:
            List of waypoints (destinations only) or None if planning fails
        """
        # Find nearest nodes in graph
        start_node = self._entry_node(start_pos, graph)
        goal_node = graph.find_nearest_node(goal_pos)
        
        if start_node is None or goal_node is None:
            print(f"[Robot {self.robot_id}] Cannot find nodes near start/goal")
            return None
        
        start_node_id, start_dist = start_node
        goal_node_id, goal_dist = goal_node
        
        # Try to find path using A*
        node_path = graph.get_path(start_node_id, goal_node_id)
        
        if node_path is not None:
            # Anchor the path at the robot's current position so that
            # _distance_to_path() measures perpendicular deviation from the
            # START→node0 segment, preventing immediate replan triggers.
            waypoints = [start_pos]

            for node_id in node_path:
                pos = graph.get_node_position(node_id)
                prev = waypoints[-1]
                if np.hypot(pos[0] - prev[0], pos[1] - prev[1]) > 0.5:
                    waypoints.append(pos)
                    if node_id == node_path[0]:
                        self.start_connection_idx = len(waypoints) - 1

            last = waypoints[-1]
            if np.hypot(goal_pos[0] - last[0], goal_pos[1] - last[1]) > 0.5:
                waypoints.append(goal_pos)

            print(f"[Robot {self.robot_id}] Found connected path through graph: {len(node_path)} nodes")
            return waypoints
        
        # No direct path - graph is disconnected
        # Try multi-component planning
        print(f"[Robot {self.robot_id}] Graph disconnected, attempting multi-component planning...")
        return self._plan_multi_component_path(start_pos, goal_pos, graph)
    
    def generate_direct_waypoints(self, start_pos: Tuple[float, float],
                                  goal_pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Generate waypoints along direct line to goal.
        
        Places waypoints every direct_waypoint_distance meters toward goal.
        Does NOT include start position - only destinations to navigate to.
        
        Args:
            start_pos: Starting position
            goal_pos: Goal position
            
        Returns:
            List of waypoints (destinations only, not including start)
        """
        # Anchor at current position (same reason as graph path: segment distance
        # check works correctly when the path starts exactly at the robot).
        waypoints = [start_pos]

        dx = goal_pos[0] - start_pos[0]
        dy = goal_pos[1] - start_pos[1]
        distance = np.sqrt(dx**2 + dy**2)

        if distance < self.direct_waypoint_distance:
            waypoints.append(goal_pos)
        else:
            # Generate intermediate waypoints
            num_waypoints = int(distance / self.direct_waypoint_distance)
            
            for i in range(1, num_waypoints + 1):
                t = (i * self.direct_waypoint_distance) / distance
                wp_x = start_pos[0] + t * dx
                wp_y = start_pos[1] + t * dy
                waypoints.append((wp_x, wp_y))
            
            # Add final goal
            waypoints.append(goal_pos)
        
        return waypoints
    
    def get_current_waypoint(self) -> Optional[Tuple[float, float]]:
        """
        Get the current waypoint to navigate to.
        
        Returns:
            (x, y) waypoint position or None if no path
        """
        if self.current_path is None or self.current_waypoint_idx >= len(self.current_path):
            return None
        
        return self.current_path[self.current_waypoint_idx] 
    
    def advance_waypoint(self, current_pos: Tuple[float, float],
                        arrival_threshold: float = 3.0) -> bool:
        """
        Check if robot has reached current waypoint and advance if so.
        
        Args:
            current_pos: Current robot position
            arrival_threshold: Distance to consider waypoint reached (meters)
            
        Returns:
            True if waypoint was advanced, False otherwise
        """
        current_waypoint = self.get_current_waypoint()
        
        if current_waypoint is None:
            return False
        
        # Check distance to waypoint
        distance = np.sqrt((current_pos[0] - current_waypoint[0])**2 + 
                          (current_pos[1] - current_waypoint[1])**2)
        
        if distance <= arrival_threshold:
            self.current_waypoint_idx += 1
            
            if self.current_waypoint_idx < len(self.current_path):
                next_waypoint = self.current_path[self.current_waypoint_idx]
                print(f"[Robot {self.robot_id}] Reached waypoint {self.current_waypoint_idx-1}, "
                      f"heading to waypoint {self.current_waypoint_idx}")
            else:
                print(f"[Robot {self.robot_id}] Reached goal!")
            
            return True
        
        return False
    
    def should_replan(self, current_pos: Tuple[float, float]) -> bool:
        """
        Check if robot should replan path.
        
        Args:
            current_pos: Current robot position
            
        Returns:
            True if should replan
        """
        if self.current_path is None:
            return False
        
        # Check if robot is too far from the planned path
        closest_dist = self._distance_to_path(current_pos)
        
        if closest_dist > self.replan_threshold:
            print(f"[Robot {self.robot_id}] Robot is {closest_dist:.1f}m from path, replanning")
            return True
        
        return False
    
    def _distance_to_path(self, position: Tuple[float, float]) -> float:
        """
        Compute minimum perpendicular distance from position to path SEGMENTS.

        Using point-to-segment (not point-to-waypoint) distance prevents false
        replan triggers when the robot is progressing between two waypoints that
        are both more than replan_threshold metres away.
        """
        if self.current_path is None or len(self.current_path) == 0:
            return float('inf')

        p = np.array(position[:2], dtype=float)
        min_dist = float('inf')

        # Single-waypoint path: fall back to point distance
        if len(self.current_path) == 1:
            wp = np.array(self.current_path[0][:2], dtype=float)
            return float(np.linalg.norm(p - wp))

        for i in range(len(self.current_path) - 1):
            a = np.array(self.current_path[i][:2],   dtype=float)
            b = np.array(self.current_path[i + 1][:2], dtype=float)
            ab = b - a
            ab_sq = float(np.dot(ab, ab))
            if ab_sq < 1e-6:
                dist = float(np.linalg.norm(p - a))
            else:
                t = float(np.dot(p - a, ab)) / ab_sq
                t = max(0.0, min(1.0, t))
                closest = a + t * ab
                dist = float(np.linalg.norm(p - closest))
            min_dist = min(min_dist, dist)

        return min_dist
    
    def _compute_path_length(self, waypoints: List[Tuple[float, float]]) -> float:
        """
        Compute total length of a path.
        
        Args:
            waypoints: List of (x, y) waypoints
            
        Returns:
            Total path length in meters
        """
        if len(waypoints) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(waypoints) - 1):
            dx = waypoints[i+1][0] - waypoints[i][0]
            dy = waypoints[i+1][1] - waypoints[i][1]
            total_length += np.sqrt(dx**2 + dy**2)
        
        return total_length
    
    def is_goal_reached(self, current_pos: Tuple[float, float],
                       threshold: float = 1.0) -> bool:
        """
        Check if robot has reached the goal.
        
        Args:
            current_pos: Current robot position
            threshold: Distance threshold for goal (meters)
            
        Returns:
            True if goal is reached
        """
        if self.goal_position is None:
            return False
        
        distance = np.sqrt((current_pos[0] - self.goal_position[0])**2 + 
                          (current_pos[1] - self.goal_position[1])**2)
        
        return distance <= threshold
    
    def get_remaining_path(self) -> Optional[List[Tuple[float, float]]]:
        """
        Get the remaining waypoints in the path.
        
        Returns:
            List of remaining waypoints or None
        """
        if self.current_path is None:
            return None
        
        return self.current_path[self.current_waypoint_idx:]
    
    def clear_path(self):
        """Clear current path."""
        self.current_path = None
        self.current_waypoint_idx = 0
        self.goal_position = None
        self.using_graph = False
        self.waiting_at_frontier = False
        self.start_connection_idx = None
    
    # ========== Multi-Component Planning Methods ==========
    
    def _plan_multi_component_path(self, start_pos: Tuple[float, float],
                                   goal_pos: Tuple[float, float],
                                   graph: TopologicalGraph) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path through disconnected graph components.
        
        Strategy:
        1. Detect connected components
        2. Find relevant components (near start, goal, or on path)
        3. Order components to traverse
        4. Stitch path: graph segments + direct gaps
        
        Args:
            start_pos: Starting position
            goal_pos: Goal position
            graph: Topological graph (potentially disconnected)
            
        Returns:
            List of waypoints or None if planning fails
        """
        # Detect connected components
        components = self._detect_graph_components(graph)
        
        if len(components) == 0:
            print(f"[Robot {self.robot_id}] No components found in graph")
            return None
        
        print(f"[Robot {self.robot_id}] Found {len(components)} disconnected components")
        
        # Find relevant components for this path
        print(f"[Robot {self.robot_id}] Filtering components by geometry and cost...")
        relevant_components = self._find_relevant_components(
            components, start_pos, goal_pos, graph
        )
        
        if len(relevant_components) == 0:
            print(f"[Robot {self.robot_id}] No relevant components found (all too far off path)")
            return None
        
        print(f"[Robot {self.robot_id}] {len(relevant_components)} relevant components found")
        
        # Build sequence of components to traverse
        print(f"[Robot {self.robot_id}] Ordering components by progress/cost ratio...")
        component_sequence = self._build_component_sequence(
            relevant_components, start_pos, goal_pos, graph
        )
        
        # Print sequence for debugging
        for i, comp in enumerate(component_sequence):
            print(f"[Robot {self.robot_id}]   Sequence {i+1}: {len(comp['nodes'])} nodes, "
                  f"entry dist={comp.get('dist_to_start', 0):.1f}m")
        
        # Stitch detailed path through components
        waypoints = self._stitch_multi_component_path(
            component_sequence, start_pos, goal_pos, graph
        )
        
        if waypoints and len(waypoints) > 0:
            print(f"[Robot {self.robot_id}] Multi-component path: {len(component_sequence)} components, "
                  f"{len(waypoints)} waypoints")
            return waypoints
        
        return None
    
    def _detect_graph_components(self, graph: TopologicalGraph) -> List[Set[str]]:
        """
        Detect connected components in graph using NetworkX.
        
        Args:
            graph: Topological graph
            
        Returns:
            List of sets, each set contains node IDs in a component
        """
        # Get undirected version for component detection
        undirected = graph.graph.to_undirected()
        
        # Find connected components
        components = list(nx.connected_components(undirected))
        
        return components
    
    def _find_relevant_components(self, components: List[Set[str]],
                                 start_pos: Tuple[float, float],
                                 goal_pos: Tuple[float, float],
                                 graph: TopologicalGraph) -> List[Dict]:
        """
        Find components relevant to the path.
        
        Uses two criteria:
        1. Geometric: Component is close to direct path line
        2. Cost-benefit: Detour through component is reasonable
        
        Args:
            components: List of component node sets
            start_pos: Start position
            goal_pos: Goal position
            graph: Topological graph
            
        Returns:
            List of component info dicts with nearest nodes and distances
        """
        relevant = []
        
        # Direct distance for reference
        direct_dist = np.sqrt((goal_pos[0] - start_pos[0])**2 + 
                             (goal_pos[1] - start_pos[1])**2)
        
        for comp_nodes in components:
            # Compute component centroid and find entry/exit points
            positions = [graph.get_node_position(nid) for nid in sorted(comp_nodes, key=str)]
            centroid = np.mean(positions, axis=0)
            
            # Find nearest node in component to start (entry point)
            nearest_to_start = None
            min_dist_to_start = float('inf')
            
            # Find nearest node in component to goal (exit point)
            nearest_to_goal = None
            min_dist_to_goal = float('inf')
            
            for node_id in sorted(comp_nodes, key=str):
                node_pos = graph.get_node_position(node_id)
                
                dist_to_start = np.sqrt((node_pos[0] - start_pos[0])**2 + 
                                       (node_pos[1] - start_pos[1])**2)
                dist_to_goal = np.sqrt((node_pos[0] - goal_pos[0])**2 + 
                                      (node_pos[1] - goal_pos[1])**2)
                
                if dist_to_start < min_dist_to_start:
                    min_dist_to_start = dist_to_start
                    nearest_to_start = (node_id, node_pos)
                
                if dist_to_goal < min_dist_to_goal:
                    min_dist_to_goal = dist_to_goal
                    nearest_to_goal = (node_id, node_pos)
            
            # Criterion 1: Perpendicular distance from component to direct path line
            dist_to_line = self._point_to_line_distance(
                centroid, start_pos, goal_pos
            )
            
            # Criterion 2: Detour ratio (cost-benefit)
            detour_dist = min_dist_to_start + min_dist_to_goal
            detour_ratio = detour_dist / direct_dist if direct_dist > 0 else float('inf')
            
            # Only include component if:
            # - Close to direct path line (< 100m perpendicular distance)
            # - Detour is reasonable (< 1.8x direct distance)
            PERPENDICULAR_THRESHOLD = 100.0  # meters
            DETOUR_RATIO_THRESHOLD = 1.8
            
            if dist_to_line < PERPENDICULAR_THRESHOLD and detour_ratio < DETOUR_RATIO_THRESHOLD:
                relevant.append({
                    'nodes': comp_nodes,
                    'centroid': centroid,
                    'nearest_to_start': nearest_to_start,
                    'nearest_to_goal': nearest_to_goal,
                    'dist_to_start': min_dist_to_start,
                    'dist_to_goal': min_dist_to_goal,
                    'dist_to_line': dist_to_line,
                    'detour_ratio': detour_ratio
                })
                
                print(f"[Robot {self.robot_id}]   Component: {len(comp_nodes)} nodes, "
                      f"dist_to_line={dist_to_line:.1f}m, detour_ratio={detour_ratio:.2f}x")
        
        return relevant
    
    def _point_to_line_distance(self, point: np.ndarray,
                               line_start: Tuple[float, float],
                               line_end: Tuple[float, float]) -> float:
        """
        Compute perpendicular distance from point to line segment.
        
        Args:
            point: Point coordinates (x, y)
            line_start: Line start point
            line_end: Line end point
            
        Returns:
            Perpendicular distance in meters
        """
        # Convert to numpy arrays
        p = np.array(point)
        a = np.array(line_start)
        b = np.array(line_end)
        
        # Vector from a to b
        ab = b - a
        ab_len = np.linalg.norm(ab)
        
        if ab_len < 0.001:  # Line is essentially a point
            return np.linalg.norm(p - a)
        
        # Vector from a to p
        ap = p - a
        
        # Project ap onto ab
        t = np.dot(ap, ab) / (ab_len ** 2)
        
        # Clamp t to [0, 1] to stay on line segment
        t = np.clip(t, 0.0, 1.0)
        
        # Find closest point on line segment
        closest = a + t * ab
        
        # Distance from point to closest point
        return np.linalg.norm(p - closest)
    
    def _build_component_sequence(self, components: List[Dict],
                                 start_pos: Tuple[float, float],
                                 goal_pos: Tuple[float, float],
                                 graph: TopologicalGraph) -> List[Dict]:
        """
        Order components to traverse from start to goal.

        Projects each component's centroid onto the start→goal axis and
        sorts by increasing projection.  This guarantees monotonic forward
        progress and eliminates the zigzag paths that the previous greedy
        approach produced when components were spread laterally.

        Args:
            components: List of relevant component dicts
            start_pos: Start position
            goal_pos: Goal position
            graph: Topological graph

        Returns:
            Ordered list of components to traverse
        """
        if len(components) <= 1:
            return components

        start_arr = np.array(start_pos)
        axis = np.array(goal_pos) - start_arr
        axis_len = float(np.linalg.norm(axis))
        if axis_len < 1e-6:
            return components
        axis_dir = axis / axis_len

        def _projection(comp):
            return float(np.dot(np.array(comp['centroid']) - start_arr, axis_dir))

        return sorted(components, key=_projection)
    
    def _stitch_multi_component_path(self, component_sequence: List[Dict],
                                    start_pos: Tuple[float, float],
                                    goal_pos: Tuple[float, float],
                                    graph: TopologicalGraph) -> List[Tuple[float, float]]:
        """
        Stitch together path through multiple components.
        
        Creates: start → [comp1 graph] → [direct gap] → [comp2 graph] → ... → goal
        
        Args:
            component_sequence: Ordered list of components
            start_pos: Start position
            goal_pos: Goal position
            graph: Topological graph
            
        Returns:
            List of waypoints
        """
        waypoints = [start_pos]
        current_pos = start_pos
        entered = False
        
        for i, comp in enumerate(component_sequence):
            # Find entry point (nearest node to current position)
            entry_node_id = None
            min_entry_dist = float('inf')
            
            for node_id in sorted(comp['nodes'], key=str):
                node_pos = graph.get_node_position(node_id)
                if (not entered and self.entry_validator is not None
                        and not self.entry_validator(node_pos)):
                    continue
                dist = np.sqrt((node_pos[0] - current_pos[0])**2 + 
                             (node_pos[1] - current_pos[1])**2)
                if dist < min_entry_dist:
                    min_entry_dist = dist
                    entry_node_id = node_id
            
            if entry_node_id is None:
                if not entered:
                    # Skip leading components the robot cannot reach from
                    # here; the route starts at the first reachable one.
                    continue
                return None

            # Find exit point (nearest node to next position)
            if i < len(component_sequence) - 1:
                # Next component exists - exit toward it
                next_comp = component_sequence[i + 1]
                next_target = next_comp['nearest_to_start'][1]
            else:
                # Last component - exit toward goal
                next_target = goal_pos
            
            exit_node_id = None
            min_exit_dist = float('inf')
            
            for node_id in sorted(comp['nodes'], key=str):
                node_pos = graph.get_node_position(node_id)
                dist = np.sqrt((node_pos[0] - next_target[0])**2 + 
                             (node_pos[1] - next_target[1])**2)
                if dist < min_exit_dist:
                    min_exit_dist = dist
                    exit_node_id = node_id
            
            # Add waypoint for entry (if not too close to current pos)
            entry_pos = graph.get_node_position(entry_node_id)
            if min_entry_dist > 1.0:  # Only add if > 1m away
                waypoints.append(entry_pos)
                if not entered:
                    self.start_connection_idx = len(waypoints) - 1
            entered = True
            
            # Path through component (if entry != exit)
            if entry_node_id != exit_node_id:
                # Create subgraph for this component
                subgraph = graph.graph.subgraph(comp['nodes'])
                
                # Find path within component
                try:
                    internal_path = nx.shortest_path(subgraph, entry_node_id, exit_node_id, weight='weight')
                    
                    # Add intermediate nodes (skip first as it's already added)
                    for node_id in internal_path[1:]:
                        node_pos = graph.get_node_position(node_id)
                        if node_pos is not None:
                            waypoints.append(node_pos)
                except (nx.NetworkXNoPath, nx.NodeNotFound, nx.exception.NetworkXError):
                    # No path within component — fall back to direct exit node
                    exit_pos = graph.get_node_position(exit_node_id)
                    if exit_pos is not None:
                        waypoints.append(exit_pos)
            
            # Update current position to exit
            current_pos = graph.get_node_position(exit_node_id)
        
        if not entered:
            return None
        # Add final goal
        last = waypoints[-1] if waypoints else start_pos
        if np.hypot(goal_pos[0] - last[0], goal_pos[1] - last[1]) > 0.5:
            waypoints.append(goal_pos)
        
        return waypoints
