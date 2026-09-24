"""
obstacles.py - Obstacle definitions
"""

import numpy as np


class CircleObstacle:
    """Circular obstacle"""
    
    def __init__(self, x: float, y: float, radius: float):
        self.type = 'circle'
        self.position = np.array([x, y], dtype=float)
        self.radius = radius
    
    def __repr__(self):
        return f"CircleObstacle(pos={self.position}, r={self.radius})"


class RectObstacle:
    """Rectangular obstacle"""
    
    def __init__(self, x: float, y: float, width: float, height: float):
        self.type = 'rect'
        self.position = np.array([x, y], dtype=float)
        self.width = width
        self.height = height
        self.size = (width, height)
    
    def get_corners(self):
        """Get four corner coordinates"""
        x, y = self.position
        w, h = self.width / 2, self.height / 2
        return np.array([
            [x - w, y - h],
            [x + w, y - h],
            [x + w, y + h],
            [x - w, y + h],
        ])
    
    def __repr__(self):
        return f"RectObstacle(pos={self.position}, size={self.width}x{self.height})"
