"""
robots.py - Robot class definitions
Includes ground robots and drones
"""

import numpy as np
from typing import Dict, Optional


class GroundRobot:
    """
    Ground robot
    
    Supports two types:
    - diff_drive: Differential drive, control input [v, omega]
    - holonomic: Omnidirectional, control input [vx, vy, omega]
    """
    
    def __init__(self,
                 robot_id: int,
                 position: np.ndarray,
                 orientation: float = 0.0,
                 robot_type: str = 'diff_drive',
                 radius: float = 0.3,
                 max_speed: float = 1.0,
                 max_angular_speed: float = np.pi):
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.radius = radius
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        
        self.position = position.copy()
        self.orientation = orientation
        self.velocity = np.zeros(3)  # [vx, vy, vz] - ground robots have vz=0
        self.angular_velocity = 0.0
        
        self.prev_position = position.copy()
        self.prev_orientation = orientation
        
        self.init_position = position.copy()
        self.init_orientation = orientation
        
        # Broadcast communication (can be any type: dict, array, etc.)
        self.broadcast_message = None
    
    def step(self, action: np.ndarray, dt: float):
        """Execute one motion step"""
        self.prev_position = self.position.copy()
        self.prev_orientation = self.orientation
        
        if self.robot_type == 'diff_drive':
            self._diff_drive_step(action, dt)
        elif self.robot_type == 'holonomic':
            self._holonomic_step(action, dt)
        else:
            raise ValueError(f"Unknown robot type: {self.robot_type}")
    
    def _diff_drive_step(self, action: np.ndarray, dt: float):
        """Differential drive kinematics"""
        v, omega = action
        v = np.clip(v, -self.max_speed, self.max_speed)
        omega = np.clip(omega, -self.max_angular_speed, self.max_angular_speed)
        
        self.position[0] += v * np.cos(self.orientation) * dt
        self.position[1] += v * np.sin(self.orientation) * dt
        # position[2] stays 0 for ground robots
        self.orientation += omega * dt
        self.orientation = np.arctan2(np.sin(self.orientation), np.cos(self.orientation))
        
        self.velocity[0] = v * np.cos(self.orientation)
        self.velocity[1] = v * np.sin(self.orientation)
        self.velocity[2] = 0.0  # Ground robots don't move vertically
        self.angular_velocity = omega
    
    def _holonomic_step(self, action: np.ndarray, dt: float):
        """Holonomic kinematics"""
        vx, vy, omega = action
        speed = np.sqrt(vx**2 + vy**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            vx *= scale
            vy *= scale
        omega = np.clip(omega, -self.max_angular_speed, self.max_angular_speed)
        
        self.position[0] += vx * dt
        self.position[1] += vy * dt
        # position[2] stays 0 for ground robots
        self.orientation += omega * dt
        self.orientation = np.arctan2(np.sin(self.orientation), np.cos(self.orientation))
        
        self.velocity = np.array([vx, vy, 0.0])  # Ground robots don't move vertically
        self.angular_velocity = omega
    
    def set_broadcast(self, message):
        """Set broadcast message for this robot (supports any data type)"""
        self.broadcast_message = message
    
    def get_broadcast(self):
        """Get current broadcast message"""
        return self.broadcast_message
    
    def get_self_observation(self) -> Dict:
        """Get self observation (for agent use)"""
        return {
            'position': self.position.copy(),
            'orientation': self.orientation,
            'velocity': self.velocity.copy(),
            'angular_velocity': self.angular_velocity,
            'type': self.robot_type,
            'broadcast': self.get_broadcast()
        }
    
    def reset(self):
        """Reset to initial state"""
        self.position = self.init_position.copy()
        self.orientation = self.init_orientation
        self.velocity = np.zeros(3)  # [vx, vy, vz] - ground robots have vz=0
        self.angular_velocity = 0.0
        self.prev_position = self.init_position.copy()
        self.prev_orientation = self.init_orientation
        self.broadcast_message = None


class Drone:
    """
    Drone
    
    Simplified model: free movement in 3D space, no collision
    Control input: [vx, vy] or [vx, vy, vz]
    """
    
    def __init__(self,
                 robot_id: int,
                 position: np.ndarray,
                 height: float = 1.0,
                 max_speed: float = 2.0):
        self.robot_id = robot_id
        self.max_speed = max_speed
        
        # Ensure 3D position
        if len(position) == 2:
            self.position = np.array([position[0], position[1], height], dtype=float)
        else:
            self.position = position.copy()
        
        self.height = self.position[2]  # Height is the z-coordinate
        self.velocity = np.zeros(3)  # 3D velocity [vx, vy, vz]
        self.init_position = self.position.copy()
        self.init_height = self.height
        
        # Broadcast communication (can be any type: dict, array, etc.)
        self.broadcast_message = None
    
    def step(self, action: np.ndarray, dt: float):
        """
        Execute one motion step.
        
        Args:
            action: Control input [vx, vy] or [vx, vy, vz]
            dt: Time step
        """
        # Support both 2D [vx, vy] and 3D [vx, vy, vz] control
        if len(action) == 2:
            vx, vy = action
            vz = 0.0
        elif len(action) == 3:
            vx, vy, vz = action
        else:
            raise ValueError(f"Action must be 2D or 3D, got shape {action.shape}")
        
        # Limit horizontal speed
        speed = np.sqrt(vx**2 + vy**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            vx *= scale
            vy *= scale
        
        # Apply motion
        self.position[0] += vx * dt
        self.position[1] += vy * dt
        self.position[2] += vz * dt  # Update z-coordinate
        self.height = self.position[2]  # Keep height in sync
        self.velocity = np.array([vx, vy, vz])
    
    def set_broadcast(self, message):
        """Set broadcast message for this drone (supports any data type)"""
        self.broadcast_message = message
    
    def get_broadcast(self):
        """Get current broadcast message"""
        return self.broadcast_message
    
    def get_self_observation(self) -> Dict:
        """Get self observation"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),  # 3D velocity [vx, vy, vz]
            'height': self.height,
            'type': 'drone',
            'broadcast': self.get_broadcast()
        }
    
    def reset(self):
        """Reset to initial state"""
        self.position = self.init_position.copy()
        self.velocity = np.zeros(3)  # 3D velocity
        self.height = self.position[2]  # Height is z-coordinate
        self.broadcast_message = None
