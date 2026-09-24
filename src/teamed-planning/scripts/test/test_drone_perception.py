"""
Test 2: Drone Perception and Topological Graph Building
Tests semantic perception, skeleton extraction, and graph construction with visualization
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.env import MultiRobotEnv, EnvConfig
from simulator.obstacles import CircleObstacle, RectObstacle
from drone.drone_main import DroneController
from drone.semantic_segmentation import TerrainType

# Try to import pygame for visualization overlay
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available, skipping graph visualization overlay")


def generate_semantic_map_from_observation(drone_obs, drone_pos, map_size=200, resolution=0.5):
    """
    Generate semantic map from drone's observation (visible obstacles only).
    
    This simulates the drone's perception system which only sees obstacles within
    its observation radius, not the entire environment.
    
    Args:
        drone_obs: Observation dict from simulator for the drone
        drone_pos: (x, y, z) drone position
        map_size: Size of the map in pixels (will be map_size x map_size)
        resolution: Resolution in meters per pixel
        
    Returns:
        semantic_map: np.ndarray of shape (map_size, map_size) with TerrainType values
    """
    # Initialize map as traversable
    semantic_map = np.ones((map_size, map_size), dtype=int) * TerrainType.TRAVERSABLE
    
    # Compute map bounds (centered on drone)
    half_width = (map_size * resolution) / 2.0
    min_x = drone_pos[0] - half_width
    max_x = drone_pos[0] + half_width
    min_y = drone_pos[1] - half_width
    max_y = drone_pos[1] + half_width
    
    # Get visible obstacles from observation (only what the drone can see!)
    visible_obstacles = drone_obs.get('obstacles', [])
    
    # Rasterize ONLY visible obstacles onto the map
    for obs_info in visible_obstacles:
        # Obstacle position is relative to drone, convert to world coordinates
        rel_pos = obs_info['position']
        world_x = drone_pos[0] + rel_pos[0]
        world_y = drone_pos[1] + rel_pos[1]
        
        obs_type = obs_info['type']
        
        if obs_type == 'circle':
            # Rasterize circle
            radius = obs_info['size']
            
            # Convert to pixel coordinates
            pixel_cx = int((world_x - min_x) / resolution)
            pixel_cy = int((world_y - min_y) / resolution)
            pixel_radius = int(radius / resolution)
            
            # Fill circle
            for dy in range(-pixel_radius - 2, pixel_radius + 3):
                for dx in range(-pixel_radius - 2, pixel_radius + 3):
                    if dx*dx + dy*dy <= (pixel_radius + 1)**2:
                        py = pixel_cy + dy
                        px = pixel_cx + dx
                        if 0 <= py < map_size and 0 <= px < map_size:
                            semantic_map[py, px] = TerrainType.OBSTACLE
                            
        elif obs_type == 'rect':
            # Rasterize rectangle
            width, height = obs_info['size']
            half_w = width / 2.0
            half_h = height / 2.0
            
            # Convert to pixel coordinates
            pixel_x1 = int((world_x - half_w - min_x) / resolution)
            pixel_x2 = int((world_x + half_w - min_x) / resolution)
            pixel_y1 = int((world_y - half_h - min_y) / resolution)
            pixel_y2 = int((world_y + half_h - min_y) / resolution)
            
            # Clip to map bounds
            pixel_x1 = max(0, pixel_x1)
            pixel_x2 = min(map_size - 1, pixel_x2)
            pixel_y1 = max(0, pixel_y1)
            pixel_y2 = min(map_size - 1, pixel_y2)
            
            # Fill rectangle
            semantic_map[pixel_y1:pixel_y2+1, pixel_x1:pixel_x2+1] = TerrainType.OBSTACLE
    
    return semantic_map


def visualize_graph_on_pygame(env, graph, drone_pos, map_size, resolution):
    """
    Overlay topological graph on pygame renderer.
    
    Args:
        env: MultiRobotEnv instance
        graph: TopologicalGraph instance
        drone_pos: (x, y, z) drone position
        map_size: Map size in pixels
        resolution: Resolution in meters per pixel
    """
    if not PYGAME_AVAILABLE or env.renderer is None:
        return
    
    renderer = env.renderer
    screen = renderer.screen
    
    # Graph visualization colors
    NODE_COLOR = (255, 50, 50)      # Red nodes
    EDGE_COLOR = (50, 100, 255)     # Blue edges
    NODE_RADIUS = 4
    EDGE_WIDTH = 2
    
    # Draw edges first (so nodes appear on top)
    for edge in graph.graph.edges():
        node1_id, node2_id = edge
        pos1 = graph.get_node_position(node1_id)
        pos2 = graph.get_node_position(node2_id)
        
        if pos1 and pos2:
            # Convert world coordinates to screen coordinates
            screen_pos1 = renderer.world_to_screen(pos1)
            screen_pos2 = renderer.world_to_screen(pos2)
            
            pygame.draw.line(screen, EDGE_COLOR, screen_pos1, screen_pos2, EDGE_WIDTH)
    
    # Draw nodes
    for node_id in graph.graph.nodes():
        pos = graph.get_node_position(node_id)
        if pos:
            screen_pos = renderer.world_to_screen(pos)
            pygame.draw.circle(screen, NODE_COLOR, screen_pos, NODE_RADIUS)
    
    # Draw coverage area (semi-transparent circle around drone)
    half_coverage = (map_size * resolution) / 2.0
    drone_screen_pos = renderer.world_to_screen((drone_pos[0], drone_pos[1]))
    coverage_radius_screen = renderer.world_length_to_screen(half_coverage)
    
    # Create transparent surface for coverage area
    coverage_surface = pygame.Surface(renderer.window_size, pygame.SRCALPHA)
    pygame.draw.circle(coverage_surface, (255, 255, 0, 30), drone_screen_pos, coverage_radius_screen, 2)
    screen.blit(coverage_surface, (0, 0))


def main():
    print("=" * 60)
    print("Test 2: Drone Perception and Topological Graph Building")
    print("=" * 60)
    
    # Create environment (500m x 500m world)
    world_size = (500, 500)
    config = EnvConfig(
        world_size=world_size,
        dt=0.05,  # 20Hz simulation
        obs_radius=200.0,
        enable_collision=False
    )
    env = MultiRobotEnv(config)
    
    # Add obstacles manually for controlled testing
    print("\n[Setup] Adding obstacles...")
    
    # Add circular obstacles
    env.add_circle_obstacle(100, 100, 15)
    env.add_circle_obstacle(250, 150, 20)
    env.add_circle_obstacle(350, 300, 18)
    env.add_circle_obstacle(150, 350, 12)
    env.add_circle_obstacle(400, 400, 16)
    env.add_circle_obstacle(200, 250, 14)
    
    # Add rectangular obstacles
    env.add_rect_obstacle(200, 200, 30, 40)
    env.add_rect_obstacle(400, 100, 25, 25)
    env.add_rect_obstacle(300, 400, 35, 20)
    env.add_rect_obstacle(100, 300, 20, 30)
    
    print(f"[Setup] Added {len(env.obstacles)} obstacles")
    
    # Add drone
    drone_start_x, drone_start_y = 250, 250
    drone_height = 50.0
    drone_id = env.add_drone(
        x=drone_start_x,
        y=drone_start_y,
        z=drone_height,
        max_speed=30.0
    )
    print(f"[Setup] Added drone {drone_id} at ({drone_start_x}, {drone_start_y}, {drone_height}m)")
    
    # Initialize DroneController
    print("\n[Setup] Initializing DroneController...")
    map_resolution = 0.5  # 0.5 meters per pixel
    map_size_pixels = 200  # 200x200 pixels = 100m x 100m coverage
    
    drone_controller = DroneController(
        drone_id=f"drone_{drone_id}",
        resolution=map_resolution,
        broadcast_interval=5.0
    )
    print(f"[Setup] DroneController initialized (resolution={map_resolution}m/px, map_size={map_size_pixels}px)")
    print(f"[Setup] Coverage area: {map_size_pixels * map_resolution}m x {map_size_pixels * map_resolution}m")
    
    # Reset environment
    obs = env.reset()
    
    print("\n[Simulation] Starting perception and mapping test...")
    print(f"Observation radius: {config.obs_radius}m")
    
    # Patrol waypoints
    patrol_waypoints = [
        (150, 150),
        (350, 150),
        (350, 350),
        (150, 350)
    ]
    current_waypoint_idx = 0
    waypoint_threshold = 10.0
    
    # Simulation loop
    max_steps = 1000
    graph_update_interval = 50  # Update graph every 50 steps
    
    for step in range(max_steps):
        # Get drone observation
        drone_obs = obs[drone_id]
        drone_pos_2d = drone_obs['self']['position']
        drone_pos_3d = (drone_pos_2d[0], drone_pos_2d[1], drone_height)
        
        # Generate semantic map from drone's observation (only visible obstacles)
        if step % graph_update_interval == 0:
            semantic_map = generate_semantic_map_from_observation(
                drone_obs,  # Use drone's observation, not full env
                drone_pos_3d, 
                map_size=map_size_pixels,
                resolution=map_resolution
            )
            
            # Create observation for DroneController
            controller_obs = {
                'position': drone_pos_3d,
                'semantic_map': semantic_map,
                'nearby_robots': []  # No ground robots yet
            }
            
            # Process observation and build graph
            result = drone_controller.process_observation(controller_obs)
            
            # Print graph statistics
            if result['graph_built']:
                graph = drone_controller.current_graph
                num_visible_obstacles = len(drone_obs.get('obstacles', []))
                print(f"Step {step:4d}: Visible obstacles: {num_visible_obstacles}, "
                      f"Built graph with {result['graph_nodes']} nodes, "
                      f"{graph.graph.number_of_edges()} edges")
        
        # Simple patrol control
        target = patrol_waypoints[current_waypoint_idx]
        dx = target[0] - drone_pos_2d[0]
        dy = target[1] - drone_pos_2d[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < waypoint_threshold:
            current_waypoint_idx = (current_waypoint_idx + 1) % len(patrol_waypoints)
            vx, vy = 0.0, 0.0
        else:
            max_speed = 15.0
            vx = (dx / distance) * max_speed
            vy = (dy / distance) * max_speed
        
        # Step simulation
        actions = {drone_id: np.array([vx, vy])}
        obs, info = env.step(actions)
        
        # Render with graph overlay (fixed rendering flow to avoid flicker)
        env.render(auto_flip=False)  # Render but don't flip yet
        
        # Always draw graph overlay (even if empty) for consistent rendering
        if drone_controller.current_graph:
            visualize_graph_on_pygame(
                env, 
                drone_controller.current_graph, 
                drone_pos_3d,
                map_size_pixels,
                map_resolution
            )
        
        # Now flip once after all drawing is complete
        if PYGAME_AVAILABLE and env.renderer:
            import pygame
            pygame.display.flip()
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Final Graph Statistics:")
    if drone_controller.current_graph:
        graph = drone_controller.current_graph
        print(f"  Total nodes: {graph.graph.number_of_nodes()}")
        print(f"  Total edges: {graph.graph.number_of_edges()}")
        print(f"  Coverage bounds: {graph.metadata.get('bounds')}")
        print(f"  Drone ID: {graph.metadata.get('drone_id')}")
    else:
        print("  No graph built")
    
    print("=" * 60)
    print("Test 2 Finished")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()

