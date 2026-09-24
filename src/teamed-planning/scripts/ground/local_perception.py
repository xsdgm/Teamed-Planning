"""
Local Perception Module for Ground Robots
Processes local occupancy maps into semantic maps for obstacle avoidance
"""

import numpy as np
import sys
import os
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone.semantic_segmentation import TerrainType


class LocalPerception:
    """
    Processes local sensor data (occupancy map) into semantic representation.
    
    Ground robots observe their local environment (e.g., 8m radius with LiDAR)
    and convert it to semantic maps for obstacle avoidance.
    """
    
    def __init__(self, robot_id: str, resolution: float = 0.5):
        """
        Initialize local perception module.
        
        Args:
            robot_id: Unique identifier for this robot
            resolution: Map resolution in meters per pixel
        """
        self.robot_id = robot_id
        self.resolution = resolution
        
        # Terrain traversability costs (for future use)
        self.terrain_costs = {
            TerrainType.UNKNOWN: 1.0,
            TerrainType.TRAVERSABLE: 1.0,
            TerrainType.OBSTACLE: float('inf'),
            TerrainType.DIFFICULT: 2.5,
            TerrainType.WATER: float('inf'),
            TerrainType.CLEARNING: 0.5
        }
    
    def process_occupancy_map(self, occupancy_map: np.ndarray) -> np.ndarray:
        """
        Transform binary occupancy map to semantic map.
        
        Adapts raw sensor data (occupancy grid) to semantic representation.
        In simulation: transforms binary 0/1 map.
        In real-world: would integrate LiDAR point clouds and camera data.
        
        Args:
            occupancy_map: Binary occupancy grid (0=free, 1=occupied)
            
        Returns:
            semantic_map: Map with TerrainType values (TRAVERSABLE or OBSTACLE)
        """
        # Simple binary classification: occupied → obstacle, free → traversable
        semantic_map = np.where(
            occupancy_map == 1,
            TerrainType.OBSTACLE.value,
            TerrainType.TRAVERSABLE.value
        ).astype(np.uint8)
        
        return semantic_map
    
    def get_terrain_cost(self, terrain_type: TerrainType) -> float:
        """
        Get traversability cost for a terrain type.
        
        Args:
            terrain_type: Terrain type enum
            
        Returns:
            Cost value (higher = harder to traverse, inf = impassable)
        """
        return self.terrain_costs.get(terrain_type, 1.0)
    
    def is_traversable(self, terrain_type: TerrainType) -> bool:
        """
        Check if terrain type is traversable.
        
        Args:
            terrain_type: Terrain type enum
            
        Returns:
            True if traversable (cost < infinity)
        """
        cost = self.get_terrain_cost(terrain_type)
        return cost < float('inf')
