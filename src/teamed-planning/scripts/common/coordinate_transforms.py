"""
Coordinate Transformation Utilities
Handles transformations between different coordinate frames
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class Pose2D:
    """2D pose representation (x, y, theta)."""
    x: float
    y: float
    theta: float = 0.0  # Orientation in radians
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, theta]."""
        return np.array([self.x, self.y, self.theta])
    
    def to_position(self) -> Tuple[float, float]:
        """Return only position as (x, y) tuple."""
        return (self.x, self.y)


@dataclass  
class Pose3D:
    """3D pose representation (x, y, z, roll, pitch, yaw)."""
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, z, roll, pitch, yaw]."""
        return np.array([self.x, self.y, self.z, self.roll, self.pitch, self.yaw])
    
    def to_position(self) -> Tuple[float, float, float]:
        """Return only position as (x, y, z) tuple."""
        return (self.x, self.y, self.z)
    
    def to_2d(self) -> Pose2D:
        """Project to 2D pose (x, y, yaw)."""
        return Pose2D(self.x, self.y, self.yaw)


class CoordinateTransform:
    """Utility class for coordinate transformations."""
    
    @staticmethod
    def transform_point_2d(point: Tuple[float, float], 
                          origin: Pose2D) -> Tuple[float, float]:
        """
        Transform a point from one coordinate frame to another in 2D.
        
        Args:
            point: (x, y) point in original frame
            origin: Pose of the new frame origin in the original frame
            
        Returns:
            (x, y) point in the new frame
        """
        px, py = point
        
        # Translate
        dx = px - origin.x
        dy = py - origin.y
        
        # Rotate
        cos_theta = np.cos(-origin.theta)
        sin_theta = np.sin(-origin.theta)
        
        x_new = dx * cos_theta - dy * sin_theta
        y_new = dx * sin_theta + dy * cos_theta
        
        return (x_new, y_new)
    
    @staticmethod
    def inverse_transform_point_2d(point: Tuple[float, float],
                                   origin: Pose2D) -> Tuple[float, float]:
        """
        Transform a point from local frame back to global frame in 2D.
        
        Args:
            point: (x, y) point in local frame
            origin: Pose of the local frame origin in the global frame
            
        Returns:
            (x, y) point in the global frame
        """
        px, py = point
        
        # Rotate
        cos_theta = np.cos(origin.theta)
        sin_theta = np.sin(origin.theta)
        
        x_rot = px * cos_theta - py * sin_theta
        y_rot = px * sin_theta + py * cos_theta
        
        # Translate
        x_global = x_rot + origin.x
        y_global = y_rot + origin.y
        
        return (x_global, y_global)
    
    @staticmethod
    def compute_relative_pose_2d(pose1: Pose2D, pose2: Pose2D) -> Pose2D:
        """
        Compute relative pose from pose1 to pose2.
        
        Args:
            pose1: First pose
            pose2: Second pose
            
        Returns:
            Relative pose of pose2 with respect to pose1
        """
        # Transform position
        dx = pose2.x - pose1.x
        dy = pose2.y - pose1.y
        
        cos_theta = np.cos(-pose1.theta)
        sin_theta = np.sin(-pose1.theta)
        
        x_rel = dx * cos_theta - dy * sin_theta
        y_rel = dx * sin_theta + dy * cos_theta
        
        # Compute relative orientation
        theta_rel = pose2.theta - pose1.theta
        
        # Normalize angle to [-pi, pi]
        theta_rel = np.arctan2(np.sin(theta_rel), np.cos(theta_rel))
        
        return Pose2D(x_rel, y_rel, theta_rel)
    
    @staticmethod
    def euclidean_distance_2d(point1: Tuple[float, float], 
                             point2: Tuple[float, float]) -> float:
        """
        Compute Euclidean distance between two 2D points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Euclidean distance
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def euclidean_distance_3d(point1: Tuple[float, float, float],
                             point2: Tuple[float, float, float]) -> float:
        """
        Compute Euclidean distance between two 3D points.
        
        Args:
            point1: First point (x, y, z)
            point2: Second point (x, y, z)
            
        Returns:
            Euclidean distance
        """
        return np.sqrt((point1[0] - point2[0])**2 + 
                      (point1[1] - point2[1])**2 + 
                      (point1[2] - point2[2])**2)
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        Normalize angle to [-pi, pi] range.
        
        Args:
            angle: Angle in radians
            
        Returns:
            Normalized angle in [-pi, pi]
        """
        return np.arctan2(np.sin(angle), np.cos(angle))
