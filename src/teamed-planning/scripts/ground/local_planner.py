"""
Local Planner for Ground Robots
Handles local obstacle avoidance and generates velocity commands
"""

import numpy as np
from typing import Tuple, Optional
import sys
import os

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
        # Check area in front of robot
        height, width = semantic_map.shape
        center_x, center_y = width // 2, height // 2
        
        # Check a wedge in front of the robot
        check_distance_pixels = int(self.safety_distance / resolution)
        
        obstacle_detected = False
        
        # Simple forward check
        for dist in range(1, min(check_distance_pixels, height // 2)):
            # Check pixels in front
            for lateral_offset in range(-2, 3):  # Check 5 pixels wide
                check_x = center_x + lateral_offset
                check_y = center_y - dist  # Assuming robot faces "up" in map
                
                if 0 <= check_x < width and 0 <= check_y < height:
                    terrain = TerrainType(semantic_map[check_y, check_x])
                    
                    if terrain in [TerrainType.OBSTACLE, TerrainType.WATER]:
                        obstacle_detected = True
                        break
            
            if obstacle_detected:
                break
        
        if obstacle_detected:
            # Reduce linear velocity
            linear_vel *= 0.3
            
            # Increase turning to try to go around obstacle
            # (Simple reactive behavior - more sophisticated planning can be added)
            if angular_vel >= 0:
                angular_vel = max(angular_vel, 0.5)
            else:
                angular_vel = min(angular_vel, -0.5)
            
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
