"""
utils.py - Utility functions
Mainly collision detection
"""

import numpy as np


def check_collision_circle_circle(pos1: np.ndarray, r1: float,
                                  pos2: np.ndarray, r2: float) -> bool:
    """Check if two circles collide"""
    distance = np.linalg.norm(pos1 - pos2)
    return distance < (r1 + r2)


def check_collision_circle_rect(circle_pos: np.ndarray, circle_radius: float,
                                rect_pos: np.ndarray, rect_width: float, 
                                rect_height: float) -> bool:
    """Check if circle and rectangle collide (AABB method)"""
    half_w = rect_width / 2
    half_h = rect_height / 2
    
    rect_left = rect_pos[0] - half_w
    rect_right = rect_pos[0] + half_w
    rect_bottom = rect_pos[1] - half_h
    rect_top = rect_pos[1] + half_h
    
    closest_x = np.clip(circle_pos[0], rect_left, rect_right)
    closest_y = np.clip(circle_pos[1], rect_bottom, rect_top)
    closest_point = np.array([closest_x, closest_y])
    
    distance = np.linalg.norm(circle_pos - closest_point)
    return distance < circle_radius


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]"""
    return np.arctan2(np.sin(angle), np.cos(angle))


def angle_difference(angle1: float, angle2: float) -> float:
    """Calculate minimum difference between two angles"""
    diff = angle1 - angle2
    return normalize_angle(diff)


def rotate_vector(vector: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 2D vector"""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    return rotation_matrix @ vector
