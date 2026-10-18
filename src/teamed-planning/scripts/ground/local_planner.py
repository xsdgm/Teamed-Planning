"""
Local Planner for Ground Robots
Handles local obstacle avoidance and generates velocity commands
"""

import numpy as np
from typing import Tuple, Optional
import sys
import os
import heapq
from scipy.ndimage import binary_dilation, distance_transform_edt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone.semantic_segmentation import TerrainType


class LocalPlanner:
    """
    Local planner for obstacle avoidance and velocity control.
    
    Uses the robot's local semantic map to avoid obstacles
    while navigating toward global waypoints.
    """
    
    def __init__(self, robot_id: str, max_linear_vel: float = 1.0,
                 max_angular_vel: float = 1.0, safety_distance: float = 1.5):
        """
        Initialize local planner.
        
        Args:
            robot_id: Unique identifier for this robot
            max_linear_vel: Maximum linear velocity (m/s)
            max_angular_vel: Maximum angular velocity (rad/s)
            safety_distance: Minimum distance to obstacles (meters)
        """
        self.robot_id = robot_id
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.safety_distance = safety_distance

    def observed_connection_waypoint(self, current_pos, target, semantic_map,
                                     resolution):
        """Return a short collision-free step toward a navigation target.

        Only explicitly observed, traversable cells may be used. A distant
        entry is approached in increments inside the current sensor footprint;
        this method does not grant permission to traverse other graph gaps.
        """
        height, width = semantic_map.shape
        blocked, radius = self.observed_blocked_cells(semantic_map, resolution)
        start = (height // 2, width // 2)
        if blocked[start]:
            if semantic_map[start] not in (TerrainType.TRAVERSABLE.value,
                                           TerrainType.CLEARNING.value,
                                           TerrainType.DIFFICULT.value):
                return None
            free = np.isin(semantic_map, [TerrainType.TRAVERSABLE.value,
                                          TerrainType.CLEARNING.value,
                                          TerrainType.DIFFICULT.value])
            clearance = distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1]
            # If initialized inside the safety margin, allow escape without
            # reducing the existing clearance. Never enter occupied cells.
            blocked = ~free | (clearance < clearance[start])
            blocked[start] = False
        delta = np.asarray(target, dtype=float) - np.asarray(current_pos, dtype=float)
        distance = float(np.linalg.norm(delta))
        if distance < 0.1:
            return tuple(target)
        direction = delta / distance
        # Keep the entire footprint inside the observed map.
        limits = np.array([width, height]) * resolution / 2.0 - (radius + 1) * resolution
        scale = min(distance, float(np.min(limits / np.maximum(np.abs(direction), 1e-12))))
        if scale <= 0:
            return None
        endpoint = direction * scale
        goal = (int(np.floor(endpoint[1] / resolution + height / 2.0)),
                int(np.floor(endpoint[0] / resolution + width / 2.0)))
        samples = np.linspace(np.zeros(2), endpoint,
                              max(2, int(scale / resolution * 4) + 1))
        columns = np.floor(samples[:, 0] / resolution + width / 2.).astype(int)
        rows = np.floor(samples[:, 1] / resolution + height / 2.).astype(int)
        if not blocked[goal] and not np.any(blocked[rows, columns]):
            # Keep the exact bearing in free space. Snapping every control
            # cycle to cell centers causes alternating left/right corrections.
            return tuple(np.asarray(current_pos) + direction * min(scale, 2.))
        queue = [(0., start)]
        costs, parent = {start: 0.}, {}
        while queue:
            _, cell = heapq.heappop(queue)
            if cell == goal:
                break
            y, x = cell
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                next_cell = (y + dy, x + dx)
                ny, nx = next_cell
                if not (0 <= ny < height and 0 <= nx < width) or blocked[next_cell]:
                    continue
                if dy and dx and (blocked[y, nx] or blocked[ny, x]):
                    continue
                cost = costs[cell] + float(np.hypot(dx, dy))
                if cost >= costs.get(next_cell, float('inf')):
                    continue
                costs[next_cell], parent[next_cell] = cost, cell
                heuristic = float(np.hypot(goal[0] - ny, goal[1] - nx))
                heapq.heappush(queue, (cost + heuristic, next_cell))
        if goal not in costs:
            # A clipped straight-ahead endpoint may land inside an obstacle.
            # Approach the best reachable observed cell around it, rather
            # than repeatedly turning toward the blocked endpoint.
            def remaining_distance(cell):
                y, x = cell
                offset = np.array([(x + .5 - width / 2.) * resolution,
                                   (y + .5 - height / 2.) * resolution])
                return float(np.linalg.norm(delta - offset))
            goal = min(costs, key=remaining_distance)
            if goal == start or distance - remaining_distance(goal) < 0.01:
                return None
        path, cell = [], goal
        while cell != start:
            path.append(cell)
            cell = parent[cell]
        path.reverse()
        waypoint = None
        for y, x in path:
            offset = np.array([(x + .5 - width / 2.) * resolution,
                               (y + .5 - height / 2.) * resolution])
            # Follow at most 2m of the path, shortening at bends/obstacles.
            if waypoint is not None and np.linalg.norm(offset) > 2.:
                break
            samples = np.linspace(np.zeros(2), offset,
                                  max(2, int(np.linalg.norm(offset) / resolution * 4) + 1))
            columns = np.floor(samples[:, 0] / resolution + width / 2.).astype(int)
            rows = np.floor(samples[:, 1] / resolution + height / 2.).astype(int)
            if np.any(blocked[rows, columns]):
                break
            waypoint = tuple(np.asarray(current_pos) + offset)
        return waypoint

    def observed_blocked_cells(self, semantic_map, resolution, include_boundary=True):
        """Inflate non-traversable cells by the robot footprint and margin."""
        radius = max(1, int(np.ceil(self.safety_distance / resolution)))
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        footprint = xx * xx + yy * yy <= radius * radius
        blocked = ~np.isin(semantic_map, [TerrainType.TRAVERSABLE.value,
                                         TerrainType.CLEARNING.value,
                                         TerrainType.DIFFICULT.value])
        return binary_dilation(blocked, structure=footprint,
                               border_value=int(include_boundary)), radius
        
    def compute_velocity_command(self, 
                                current_pos: Tuple[float, float],
                                current_heading: float,
                                target_waypoint: Tuple[float, float],
                                local_semantic_map: Optional[np.ndarray] = None,
                                resolution: float = 0.5) -> Tuple[float, float]:
        """
        Compute velocity command to navigate toward target waypoint.
        
        Args:
            current_pos: Current (x, y) position
            current_heading: Current heading in radians
            target_waypoint: Target (x, y) waypoint
            local_semantic_map: Local semantic map (optional for obstacle avoidance)
            resolution: Resolution of semantic map (meters per pixel)
            
        Returns:
            (linear_velocity, angular_velocity) command
        """
        # Compute desired direction
        dx = target_waypoint[0] - current_pos[0]
        dy = target_waypoint[1] - current_pos[1]
        distance_to_target = np.sqrt(dx**2 + dy**2)
        
        if distance_to_target < 0.1:
            # Already at target
            return (0.0, 0.0)
        
        # Desired heading
        desired_heading = np.arctan2(dy, dx)
        
        # Heading error
        heading_error = self._normalize_angle(desired_heading - current_heading)
        
        # Angular velocity proportional to heading error
        angular_vel = np.clip(2.0 * heading_error, -self.max_angular_vel, self.max_angular_vel)

        # Linear velocity: smooth ramp from max (at 0° error) to 0 (at 90° error).
        # Only turn-in-place when more than 90° off — a 45° cutoff was too strict
        # and caused the robot to spin indefinitely on any slight misalignment.
        alignment_factor = max(0.0, 1.0 - abs(heading_error) / (np.pi / 2))
        distance_factor = min(1.0, distance_to_target / 3.0)
        linear_vel = self.max_linear_vel * alignment_factor * distance_factor
        
        # Check for obstacles if local map available
        if local_semantic_map is not None:
            linear_vel, angular_vel = self._avoid_obstacles(
                linear_vel, angular_vel, current_heading,
                local_semantic_map, resolution
            )
        
        return (linear_vel, angular_vel)
    
    def _avoid_obstacles(self, linear_vel: float, angular_vel: float,
                        current_heading: float, semantic_map: np.ndarray,
                        resolution: float) -> Tuple[float, float]:
        """
        Modify velocity commands to avoid obstacles.
        
        Args:
            linear_vel: Desired linear velocity
            angular_vel: Desired angular velocity
            current_heading: Current robot heading
            semantic_map: Local semantic map centered on robot
            resolution: Map resolution (meters per pixel)
            
        Returns:
            Modified (linear_velocity, angular_velocity)
        """
        # Simulator maps are aligned with world axes: columns increase along
        # +x and rows along +y. Rotate cell centers into the robot's frame;
        # heading zero faces +x, rather than the top of the image.
        height, width = semantic_map.shape
        rows, columns = np.nonzero(np.isin(
            semantic_map, [TerrainType.OBSTACLE.value, TerrainType.WATER.value]))
        dx = (columns + 0.5 - width / 2.0) * resolution
        dy = (rows + 0.5 - height / 2.0) * resolution
        cosine, sine = np.cos(current_heading), np.sin(current_heading)
        forward = dx * cosine + dy * sine
        lateral = -dx * sine + dy * cosine
        # Include the area of each occupied cell, including the safety boundary.
        cell_margin = resolution * (abs(cosine) + abs(sine)) / 2.0
        obstacle_detected = bool(np.any(
            (forward >= -cell_margin)
            & (forward <= self.safety_distance + cell_margin)
            & (np.abs(lateral) <= 0.5 + cell_margin)))
        
        if obstacle_detected:
            # Stop translation while turning away from the occupied corridor.
            linear_vel = 0.0
            
            # Increase turning to try to go around obstacle
            # (Simple reactive behavior - more sophisticated planning can be added)
            if angular_vel >= 0:
                angular_vel = max(angular_vel, 0.5)
            else:
                angular_vel = min(angular_vel, -0.5)
            angular_vel = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)
            
            print(f"[Robot {self.robot_id}] Obstacle detected, adjusting velocity")
        
        return (linear_vel, angular_vel)
    
    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-pi, pi].
        
        Args:
            angle: Angle in radians
            
        Returns:
            Normalized angle
        """
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def compute_pure_pursuit(self, current_pos: Tuple[float, float],
                            current_heading: float,
                            target_waypoint: Tuple[float, float],
                            lookahead_distance: float = 3.0,
                            local_semantic_map: Optional[np.ndarray] = None,
                            resolution: float = 0.5) -> Tuple[float, float]:
        """
        Pure pursuit controller for smoother path following.
        
        Args:
            current_pos: Current (x, y) position
            current_heading: Current heading in radians
            target_waypoint: Target (x, y) waypoint
            lookahead_distance: Lookahead distance for pure pursuit
            local_semantic_map: Local semantic map (optional for obstacle avoidance)
            resolution: Resolution of semantic map (meters per pixel)
            
        Returns:
            (linear_velocity, angular_velocity) command
        """
        # Vector to target
        dx = target_waypoint[0] - current_pos[0]
        dy = target_waypoint[1] - current_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 0.1:
            return (0.0, 0.0)
        
        # Desired heading
        desired_heading = np.arctan2(dy, dx)
        heading_error = self._normalize_angle(desired_heading - current_heading)
        
        # Pure pursuit curvature
        # Simplified version - full implementation would use lookahead point
        curvature = 2.0 * np.sin(heading_error) / lookahead_distance
        
        # Velocities
        linear_vel = self.max_linear_vel * min(1.0, distance / lookahead_distance)
        angular_vel = curvature * linear_vel
        
        # Clip angular velocity
        angular_vel = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)
        
        # Check for obstacles if local map available
        if local_semantic_map is not None:
            linear_vel, angular_vel = self._avoid_obstacles(
                linear_vel, angular_vel, current_heading,
                local_semantic_map, resolution
            )
        
        return (linear_vel, angular_vel)
