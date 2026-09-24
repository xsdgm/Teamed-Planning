"""
forest_generator.py - Forest scene generator
Randomly generate tree obstacles to simulate forest environment
"""

import numpy as np
from .obstacles import CircleObstacle, RectObstacle


class ForestGenerator:
    """Forest scene generator"""
    
    def __init__(self, world_size, seed=None):
        self.world_size = world_size
        if seed is not None:
            np.random.seed(seed)
    
    def generate_random_forest(self, density='medium', tree_size_range=(5, 15), clear_zones=None):
        """Generate randomly distributed forest"""
        obstacles = []
        area = self.world_size[0] * self.world_size[1]
        
        density_map = {
            'sparse': 5000, 'medium': 2000, 'dense': 1000, 'very_dense': 500
        }
        num_trees = int(area / density_map.get(density, 2000))
        
        print(f"Generating forest: {num_trees} trees, density={density}")
        
        clear_zones = clear_zones or []
        trees_gen = 0
        
        for _ in range(num_trees):
            x = np.random.uniform(0, self.world_size[0])
            y = np.random.uniform(0, self.world_size[1])
            
            if not self._in_clear_zone(np.array([x, y]), clear_zones):
                radius = np.random.uniform(*tree_size_range)
                obstacles.append(CircleObstacle(x, y, radius))
                trees_gen += 1
        
        # Groves
        num_groves = int(num_trees * 0.2)
        for _ in range(num_groves):
            x = np.random.uniform(0, self.world_size[0])
            y = np.random.uniform(0, self.world_size[1])
            
            if not self._in_clear_zone(np.array([x, y]), clear_zones):
                w = np.random.uniform(tree_size_range[1] * 2, tree_size_range[1] * 5)
                h = np.random.uniform(tree_size_range[1] * 2, tree_size_range[1] * 5)
                obstacles.append(RectObstacle(x, y, w, h))
        
        print(f"Generated {len(obstacles)} obstacles")
        return obstacles
    
    def _in_clear_zone(self, pos, clear_zones):
        """Check if in clear zone"""
        for cx, cy, r in clear_zones:
            if np.linalg.norm(pos - np.array([cx, cy])) < r:
                return True
        return False


def create_forest_scenario(world_size, scenario_type='medium', seed=None):
    """Quickly create forest scenario"""
    generator = ForestGenerator(world_size, seed)
    print(f"\nCreating {scenario_type} forest for {world_size[0]/1000:.1f}x{world_size[1]/1000:.1f}km world\n")
    
    if scenario_type in ['sparse', 'medium', 'dense']:
        return generator.generate_random_forest(density=scenario_type)
    else:
        return generator.generate_random_forest(density='medium')
