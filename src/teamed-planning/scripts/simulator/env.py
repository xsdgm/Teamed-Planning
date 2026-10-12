"""
env.py - Multi-robot simulation environment
Gym-like interface for reinforcement learning
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from .robots import GroundRobot, Drone
from .obstacles import CircleObstacle, RectObstacle
from .utils import check_collision_circle_circle, check_collision_circle_rect


@dataclass
class EnvConfig:
    """Environment configuration"""
    world_size: Tuple[float, float] = (10.0, 10.0)  # (width, height) in meters
    dt: float = 0.05  # Time step in seconds (configurable simulation frequency)
    obs_radius: float = 5.0  # Observation radius for obstacles in meters (deprecated, use perception_radius)
    perception_radius: float = None  # Drone perception range for rectangular area (width=height) in meters (if None, uses obs_radius)
    ground_perception_radius: float = None  # Ground robot perception range in meters (if None, uses 8.0m for LiDAR-like)
    communication_radius: float = None  # Communication radius for robot-to-robot in meters (if None, uses obs_radius)
    enable_collision: bool = True  # Enable collision detection
    # Occupancy map configuration
    occupancy_map_size: int = None  # Size of occupancy map in pixels for drones (if None, auto-calculated)
    ground_occupancy_map_size: int = None  # Size of occupancy map in pixels for ground robots (if None, auto-calculated)
    occupancy_map_resolution: float = 0.5  # Resolution in meters per pixel
    

class MultiRobotEnv:
    """
    Multi-robot simulation environment
    
    Usage:
        env = MultiRobotEnv(config)
        obs = env.reset()
        obs, info = env.step(actions)
    """
    
    def __init__(self, config: EnvConfig = None):
        self.config = config or EnvConfig()
        # Set default values for new parameters if not provided
        if self.config.perception_radius is None:
            self.config.perception_radius = self.config.obs_radius
        if self.config.ground_perception_radius is None:
            self.config.ground_perception_radius = 8.0  # Default 8m LiDAR-like range for ground robots
        if self.config.communication_radius is None:
            self.config.communication_radius = self.config.obs_radius
        
        # Auto-calculate occupancy map sizes from perception ranges
        if self.config.occupancy_map_size is None:
            # Drone perception is square: perception_radius × perception_radius
            # Map size = perception_radius / resolution (pixels)
            self.config.occupancy_map_size = int(self.config.perception_radius / self.config.occupancy_map_resolution)
            print(f"[Simulator] Auto-calculated drone occupancy_map_size: {self.config.occupancy_map_size} pixels "
                  f"({self.config.perception_radius}m × {self.config.perception_radius}m at {self.config.occupancy_map_resolution}m/pixel)")
        
        if self.config.ground_occupancy_map_size is None:
            # Ground robot perception is square: ground_perception_radius × ground_perception_radius
            self.config.ground_occupancy_map_size = int(self.config.ground_perception_radius / self.config.occupancy_map_resolution)
            print(f"[Simulator] Auto-calculated ground_robot occupancy_map_size: {self.config.ground_occupancy_map_size} pixels "
                  f"({self.config.ground_perception_radius}m × {self.config.ground_perception_radius}m at {self.config.occupancy_map_resolution}m/pixel)")
        
        self.ground_robots: List[GroundRobot] = []
        self.drones: List[Drone] = []
        self.obstacles: List = []
        self.time = 0.0
        self.step_count = 0
        self.renderer = None
        
    def add_ground_robot(self, x: float, y: float, theta: float = 0.0, 
                        robot_type: str = 'diff_drive', **kwargs) -> int:
        """Add ground robot"""
        robot_id = len(self.ground_robots) + len(self.drones)
        robot = GroundRobot(
            robot_id=robot_id,
            position=np.array([x, y, 0.0], dtype=float),  # 3D position with z=0
            orientation=theta,
            robot_type=robot_type,
            **kwargs
        )
        self.ground_robots.append(robot)
        return robot_id
    
    def add_drone(self, x: float, y: float, z: float = 1.0, **kwargs) -> int:
        """Add drone"""
        robot_id = len(self.ground_robots) + len(self.drones)
        drone = Drone(
            robot_id=robot_id,
            position=np.array([x, y, z], dtype=float),  # 3D position
            height=z,
            **kwargs
        )
        self.drones.append(drone)
        return robot_id
    
    def add_circle_obstacle(self, x: float, y: float, radius: float):
        """Add circular obstacle"""
        self.obstacles.append(CircleObstacle(x, y, radius))
    
    def add_rect_obstacle(self, x: float, y: float, width: float, height: float):
        """Add rectangular obstacle"""
        self.obstacles.append(RectObstacle(x, y, width, height))
    
    def reset(self) -> Dict:
        """Reset environment"""
        self.time = 0.0
        self.step_count = 0
        for robot in self.ground_robots + self.drones:
            robot.reset()
        return self._get_observations()
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict, Dict]:
        """
        Execute one simulation step
        
        Args:
            actions: Dict[robot_id, action]
                action can be:
                - np.ndarray: motion command only (backward compatible)
                - dict: {'motion': np.ndarray, 'broadcast': np.ndarray or None}
        """
        info = {'collisions': [], 'out_of_bounds': [], 'time': self.time}
        
        # Process actions
        for robot in self.ground_robots + self.drones:
            if robot.robot_id in actions:
                action = actions[robot.robot_id]
                
                # Parse action (support both array and dict formats)
                if isinstance(action, dict):
                    motion = action.get('motion', np.zeros(2))
                    broadcast = action.get('broadcast', None)
                else:
                    motion = action
                    broadcast = None
                
                # Execute motion
                robot.step(motion, self.config.dt)
                
                # Set broadcast message
                robot.set_broadcast(broadcast)
        
        if self.config.enable_collision:
            self._handle_collisions(info)
        
        self._check_boundaries(info)
        self.time += self.config.dt
        self.step_count += 1
        observations = self._get_observations()
        
        return observations, info
    
    def _get_observations(self) -> Dict:
        """Get observations for all robots"""
        observations = {}
        all_robots = self.ground_robots + self.drones
        
        for robot in all_robots:
            self_obs = robot.get_self_observation()
            self_obs['id'] = robot.robot_id
            
            # Detect neighbors within communication radius
            neighbors = []
            for other in all_robots:
                if other.robot_id == robot.robot_id:
                    continue
                distance = np.linalg.norm(robot.position - other.position)
                if distance <= self.config.communication_radius:
                    rel_pos = other.position - robot.position
                    rel_vel = other.velocity - robot.velocity
                    
                    # Map ground robot type to specific type for agent compatibility
                    if isinstance(other, GroundRobot):
                        neighbor_type = other.robot_type  # 'diff_drive' or 'holonomic'
                    else:
                        neighbor_type = 'drone'
                    
                    neighbors.append({
                        'id': other.robot_id,
                        'type': neighbor_type,
                        'relative_position': rel_pos,
                        'relative_velocity': rel_vel,
                        'distance': distance,
                        'broadcast': other.get_broadcast()  # Include neighbor's broadcast message
                    })
            
            # Detect obstacles within perception radius
            visible_obstacles = []
            for obs in self.obstacles:
                # Use only x,y for distance (obstacles are 2D, robots are 3D)
                robot_xy = robot.position[:2]
                distance = np.linalg.norm(robot_xy - obs.position)
                if distance <= self.config.perception_radius:
                    visible_obstacles.append({
                        'type': obs.type,
                        'position': obs.position - robot_xy,
                        'size': obs.radius if hasattr(obs, 'radius') else obs.size,
                        'distance': distance
                    })
            
            full_state = {
                'position': robot.position.copy(),
                'velocity': robot.velocity.copy(),
                'all_robots_positions': np.array([r.position for r in all_robots]),
                'all_robots_velocities': np.array([r.velocity for r in all_robots]),
                'time': self.time
            }
            
            if isinstance(robot, GroundRobot):
                full_state['orientation'] = robot.orientation
                full_state['angular_velocity'] = robot.angular_velocity
            
            observations[robot.robot_id] = {
                'self': self_obs,
                'neighbors': neighbors,
                'obstacles': visible_obstacles,
                'full_state': full_state
            }
            
            # Add occupancy map for all robots (drones and ground robots)
            if isinstance(robot, Drone):
                occupancy_map = self._generate_occupancy_map(robot.position, robot_type='drone')
            elif isinstance(robot, GroundRobot):
                occupancy_map = self._generate_occupancy_map(robot.position, robot_type='ground')
            else:
                occupancy_map = None
            
            if occupancy_map is not None:
                observations[robot.robot_id]['occupancy_map'] = occupancy_map
                observations[robot.robot_id]['map_resolution'] = self.config.occupancy_map_resolution
                observations[robot.robot_id]['map_axes'] = 'yx'
                observations[robot.robot_id]['time_step'] = self.config.dt
        
        return observations
    
    def _generate_occupancy_map(self, robot_position: np.ndarray, robot_type: str = 'drone') -> np.ndarray:
        """
        Generate binary occupancy map centered on robot position.
        
        Covers square perception area based on robot type:
        - Drones: perception_radius × perception_radius (default 60m × 60m)
        - Ground robots: ground_perception_radius × ground_perception_radius (default 8m × 8m)
        
        Args:
            robot_position: (x, y) position of robot in world frame
            robot_type: 'drone' or 'ground' to determine perception range
            
        Returns:
            occupancy_map: Binary numpy array (0=free, 1=occupied)
                          Shape: (map_size, map_size) based on robot type
        """
        # Choose perception parameters based on robot type
        if robot_type == 'ground':
            perception_range = self.config.ground_perception_radius
            map_size = self.config.ground_occupancy_map_size
        else:  # drone
            perception_range = self.config.perception_radius
            map_size = self.config.occupancy_map_size
        
        resolution = self.config.occupancy_map_resolution
        
        # Initialize as free space
        occupancy_map = np.zeros((map_size, map_size), dtype=np.uint8)
        
        # Calculate world bounds (square perception area)
        # Map covers: perception_range × perception_range centered on robot
        half_width = perception_range / 2.0
        min_x = robot_position[0] - half_width
        max_x = robot_position[0] + half_width
        min_y = robot_position[1] - half_width
        max_y = robot_position[1] + half_width

        obstacle_count = 0

        # Rasterize each obstacle
        for obs in self.obstacles:
            # Calculate distance to obstacle (use only x,y, obstacles are 2D)
            robot_xy = robot_position[:2]
            distance = np.linalg.norm(robot_xy - obs.position)
            
            # Skip if obstacle is too far (add safety margin for obstacle size)
            max_obstacle_size = obs.radius if hasattr(obs, 'radius') else max(obs.size) / 2.0
            if distance > perception_range + max_obstacle_size:
                continue
            
            obstacle_count += 1

            if obs.type == 'circle':
                # Rasterize circular obstacle
                obs_x, obs_y = obs.position
                radius = obs.radius
                
                # Convert to pixel coordinates
                pixel_cx = int((obs_x - min_x) / resolution)
                pixel_cy = int((obs_y - min_y) / resolution)
                pixel_radius = int(radius / resolution)

                # Fill circular area
                for dy in range(-pixel_radius - 2, pixel_radius + 3):
                    for dx in range(-pixel_radius - 2, pixel_radius + 3):
                        if dx*dx + dy*dy <= (pixel_radius + 1)**2:
                            py = pixel_cy + dy
                            px = pixel_cx + dx
                            if 0 <= py < map_size and 0 <= px < map_size:
                                occupancy_map[py, px] = 1
                                
            elif obs.type == 'rect':
                # Rasterize rectangular obstacle
                obs_x, obs_y = obs.position
                width, height = obs.size
                half_w = width / 2.0
                half_h = height / 2.0
                
                 # Calculate obstacle bounds in world coordinates
                obs_min_x = obs_x - half_w
                obs_max_x = obs_x + half_w
                obs_min_y = obs_y - half_h
                obs_max_y = obs_y + half_h
                
                # Clip to perception range (only rasterize the VISIBLE portion)
                visible_min_x = max(obs_min_x, min_x)
                visible_max_x = min(obs_max_x, max_x)
                visible_min_y = max(obs_min_y, min_y)
                visible_max_y = min(obs_max_y, max_y)

                if visible_min_x >= visible_max_x or visible_min_y >= visible_max_y:
                    continue

                # Convert visible portion to pixel coordinates
                pixel_x1 = int((visible_min_x - min_x) / resolution)
                pixel_x2 = int((visible_max_x - min_x) / resolution)
                pixel_y1 = int((visible_min_y - min_y) / resolution)
                pixel_y2 = int((visible_max_y - min_y) / resolution)
                
                # Clamp to map bounds (should be redundant now, but safe)
                pixel_x1 = max(0, min(map_size - 1, pixel_x1))
                pixel_x2 = max(0, min(map_size - 1, pixel_x2))
                pixel_y1 = max(0, min(map_size - 1, pixel_y1))
                pixel_y2 = max(0, min(map_size - 1, pixel_y2))
                
                # check errors
                if pixel_x2 - pixel_x1 > map_size * 0.8 or pixel_y2 - pixel_y1 > map_size * 0.8:
                    print(f"⚠️ WARNING: Large rect fill detected: "
                    f"[{pixel_y1}:{pixel_y2+1}, {pixel_x1}:{pixel_x2+1}] "
                    f"for obstacle at ({obs_x:.1f}, {obs_y:.1f})")
                
                # Fill rectangular area (only the visible portion)
                if pixel_x2 >= pixel_x1 and pixel_y2 >= pixel_y1:
                    occupancy_map[pixel_y1:pixel_y2+1, pixel_x1:pixel_x2+1] = 1

        return occupancy_map
    
    def _handle_collisions(self, info: Dict):
        """Handle ground robot collisions"""
        for robot in self.ground_robots:
            for obs in self.obstacles:
                # Use only x,y for collision (obstacles are 2D)
                robot_xy = robot.position[:2]
                
                if obs.type == 'circle':
                    if check_collision_circle_circle(
                        robot_xy, robot.radius,
                        obs.position, obs.radius
                    ):
                        robot.position = robot.prev_position.copy()
                        robot.velocity = np.zeros(3)  # 3D velocity
                        info['collisions'].append({
                            'robot_id': robot.robot_id,
                            'object': 'obstacle',
                            'obstacle_type': 'circle'
                        })
                elif obs.type == 'rect':
                    if check_collision_circle_rect(
                        robot_xy, robot.radius,
                        obs.position, obs.width, obs.height
                    ):
                        robot.position = robot.prev_position.copy()
                        robot.velocity = np.zeros(3)  # 3D velocity
                        info['collisions'].append({
                            'robot_id': robot.robot_id,
                            'object': 'obstacle',
                            'obstacle_type': 'rect'
                        })
        
        for i, robot1 in enumerate(self.ground_robots):
            for robot2 in self.ground_robots[i+1:]:
                if check_collision_circle_circle(
                    robot1.position, robot1.radius,
                    robot2.position, robot2.radius
                ):
                    robot1.position = robot1.prev_position.copy()
                    robot2.position = robot2.prev_position.copy()
                    robot1.velocity = np.zeros(2)
                    robot2.velocity = np.zeros(2)
                    info['collisions'].append({
                        'robot_id': robot1.robot_id,
                        'object': 'robot',
                        'other_robot_id': robot2.robot_id
                    })
    
    def _check_boundaries(self, info: Dict):
        """Check if out of bounds"""
        world_w, world_h = self.config.world_size
        for robot in self.ground_robots + self.drones:
            # Use x, y from 3D position
            x, y = robot.position[0], robot.position[1]
            if x < 0 or x > world_w or y < 0 or y > world_h:
                robot.position[0] = np.clip(x, 0, world_w)
                robot.position[1] = np.clip(y, 0, world_h)
                robot.velocity = np.zeros(3)  # 3D velocity
                info['out_of_bounds'].append(robot.robot_id)
    
    def render(self, mode='human', auto_flip=True):
        """
        Render environment
        
        Args:
            mode: 'human' or 'rgb_array'
            auto_flip: If True, automatically flip display (default: True)
        """
        if self.renderer is None:
            try:
                from .renderer_pygame import PygameRenderer
                self.renderer = PygameRenderer(self.config.world_size)
            except:
                print("Warning: Could not initialize renderer")
                return None
        return self.renderer.render(
            ground_robots=self.ground_robots,
            drones=self.drones,
            obstacles=self.obstacles,
            mode=mode,
            auto_flip=auto_flip
        )
    
    def close(self):
        """Close environment"""
        if self.renderer is not None:
            self.renderer.close()
    
    def get_all_robot_positions(self) -> np.ndarray:
        """Get all robot positions"""
        all_robots = self.ground_robots + self.drones
        return np.array([robot.position for robot in all_robots])
    
    def get_robot_by_id(self, robot_id: int):
        """Get robot by ID"""
        all_robots = self.ground_robots + self.drones
        for robot in all_robots:
            if robot.robot_id == robot_id:
                return robot
        return None
