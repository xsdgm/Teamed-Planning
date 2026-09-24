"""
Test 3: Drone Graph Broadcasting to Ground Robots
Tests the complete communication pipeline: drone builds graph and broadcasts to ground robots
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
from common.message_definitions import SimpleMessageBroker, MessageType

# Try to import pygame for visualization
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


def generate_semantic_map_from_observation(drone_obs, drone_pos, map_size=200, resolution=0.5):
    """
    Generate semantic map from drone's observation (visible obstacles only).
    
    Args:
        drone_obs: Observation dict from simulator for the drone
        drone_pos: (x, y, z) drone position
        map_size: Size of the map in pixels
        resolution: Resolution in meters per pixel
        
    Returns:
        semantic_map: np.ndarray with TerrainType values
    """
    semantic_map = np.ones((map_size, map_size), dtype=int) * TerrainType.TRAVERSABLE
    
    half_width = (map_size * resolution) / 2.0
    min_x = drone_pos[0] - half_width
    max_x = drone_pos[0] + half_width
    min_y = drone_pos[1] - half_width
    max_y = drone_pos[1] + half_width
    
    # Get visible obstacles from observation
    visible_obstacles = drone_obs.get('obstacles', [])
    
    for obs_info in visible_obstacles:
        # Obstacle position is relative to drone
        rel_pos = obs_info['position']
        world_x = drone_pos[0] + rel_pos[0]
        world_y = drone_pos[1] + rel_pos[1]
        
        obs_type = obs_info['type']
        
        if obs_type == 'circle':
            radius = obs_info['size']
            
            pixel_cx = int((world_x - min_x) / resolution)
            pixel_cy = int((world_y - min_y) / resolution)
            pixel_radius = int(radius / resolution)
            
            for dy in range(-pixel_radius - 2, pixel_radius + 3):
                for dx in range(-pixel_radius - 2, pixel_radius + 3):
                    if dx*dx + dy*dy <= (pixel_radius + 1)**2:
                        py = pixel_cy + dy
                        px = pixel_cx + dx
                        if 0 <= py < map_size and 0 <= px < map_size:
                            semantic_map[py, px] = TerrainType.OBSTACLE
                            
        elif obs_type == 'rect':
            width, height = obs_info['size']
            half_w = width / 2.0
            half_h = height / 2.0
            
            pixel_x1 = int((world_x - half_w - min_x) / resolution)
            pixel_x2 = int((world_x + half_w - min_x) / resolution)
            pixel_y1 = int((world_y - half_h - min_y) / resolution)
            pixel_y2 = int((world_y + half_h - min_y) / resolution)
            
            pixel_x1 = max(0, pixel_x1)
            pixel_x2 = min(map_size - 1, pixel_x2)
            pixel_y1 = max(0, pixel_y1)
            pixel_y2 = min(map_size - 1, pixel_y2)
            
            semantic_map[pixel_y1:pixel_y2+1, pixel_x1:pixel_x2+1] = TerrainType.OBSTACLE
    
    return semantic_map


def visualize_graph_on_pygame(env, graph, drone_pos, map_size, resolution):
    """Overlay topological graph on pygame renderer."""
    if not PYGAME_AVAILABLE or env.renderer is None:
        return
    
    renderer = env.renderer
    screen = renderer.screen
    
    NODE_COLOR = (255, 50, 50)      # Red nodes
    EDGE_COLOR = (50, 100, 255)     # Blue edges
    NODE_RADIUS = 4
    EDGE_WIDTH = 2
    
    # Draw edges
    for edge in graph.graph.edges():
        node1_id, node2_id = edge
        pos1 = graph.get_node_position(node1_id)
        pos2 = graph.get_node_position(node2_id)
        
        if pos1 and pos2:
            screen_pos1 = renderer.world_to_screen(pos1)
            screen_pos2 = renderer.world_to_screen(pos2)
            pygame.draw.line(screen, EDGE_COLOR, screen_pos1, screen_pos2, EDGE_WIDTH)
    
    # Draw nodes
    for node_id in graph.graph.nodes():
        pos = graph.get_node_position(node_id)
        if pos:
            screen_pos = renderer.world_to_screen(pos)
            pygame.draw.circle(screen, NODE_COLOR, screen_pos, NODE_RADIUS)


def simple_ground_robot_controller(robot_pos, target, max_speed=5.0):
    """
    Simple controller for ground robot to move toward target.
    
    Returns:
        (v, omega) for differential drive
    """
    dx = target[0] - robot_pos[0]
    dy = target[1] - robot_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    if distance < 5.0:
        return (0.0, 0.0)
    
    # Simple proportional control
    target_angle = np.arctan2(dy, dx)
    v = min(max_speed, distance * 0.5)
    omega = target_angle * 0.5
    
    return (v, omega)


def main():
    print("=" * 60)
    print("Test 3: Drone Graph Broadcasting to Ground Robots")
    print("=" * 60)
    
    # Create environment
    world_size = (500, 500)
    config = EnvConfig(
        world_size=world_size,
        dt=0.05,
        obs_radius=200.0,
        enable_collision=True
    )
    env = MultiRobotEnv(config)
    
    # Add obstacles
    print("\n[Setup] Adding obstacles...")
    env.add_circle_obstacle(100, 100, 15)
    env.add_circle_obstacle(250, 150, 20)
    env.add_circle_obstacle(350, 300, 18)
    env.add_circle_obstacle(150, 350, 12)
    env.add_circle_obstacle(400, 400, 16)
    env.add_rect_obstacle(200, 200, 30, 40)
    env.add_rect_obstacle(400, 100, 25, 25)
    env.add_rect_obstacle(300, 400, 35, 20)
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
    
    # Add ground robots
    robot1_id = env.add_ground_robot(x=100, y=100, theta=0.0, robot_type='diff_drive', max_speed=10.0)
    robot2_id = env.add_ground_robot(x=400, y=400, theta=np.pi, robot_type='diff_drive', max_speed=10.0)
    print(f"[Setup] Added ground robot {robot1_id} at (100, 100)")
    print(f"[Setup] Added ground robot {robot2_id} at (400, 400)")
    
    # Initialize DroneController
    print("\n[Setup] Initializing DroneController...")
    map_resolution = 0.5
    map_size_pixels = 200
    
    drone_controller = DroneController(
        drone_id=f"drone_{drone_id}",
        resolution=map_resolution,
        broadcast_interval=3.0  # Broadcast every 3 seconds
    )
    
    # Initialize message broker for communication
    message_broker = SimpleMessageBroker()
    
    print(f"[Setup] Broadcast interval: {drone_controller.broadcaster.broadcast_interval}s")
    print(f"[Setup] Broadcast range: {drone_controller.broadcaster.broadcast_range}m")
    
    # Reset environment
    obs = env.reset()
    
    print("\n[Simulation] Starting broadcast test...")
    print("Drone will build graph based on observations and broadcast to nearby ground robots")
    print(f"Observation radius: {config.obs_radius}m")
    print(f"Broadcast range: {drone_controller.broadcaster.broadcast_range}m")
    print("Close the pygame window to stop simulation\n")
    
    # Tracking statistics
    total_broadcasts = 0
    robot_received_counts = {robot1_id: 0, robot2_id: 0}
    
    # Simulation loop
    max_steps = 1000
    graph_update_interval = 20
    
    for step in range(max_steps):
        # Get observations
        drone_obs = obs[drone_id]
        drone_pos_2d = drone_obs['self']['position']
        drone_pos_3d = (drone_pos_2d[0], drone_pos_2d[1], drone_height)
        
        robot1_obs = obs[robot1_id]
        robot2_obs = obs[robot2_id]
        robot1_pos = robot1_obs['self']['position']
        robot2_pos = robot2_obs['self']['position']
        
        # Update drone graph periodically
        if step % graph_update_interval == 0:
            semantic_map = generate_semantic_map_from_observation(
                drone_obs, drone_pos_3d, map_size=map_size_pixels, resolution=map_resolution
            )
            
            # Find nearby robots for broadcasting
            nearby_robots = []
            for rid, rpos in [(robot1_id, robot1_pos), (robot2_id, robot2_pos)]:
                distance = np.sqrt((drone_pos_2d[0] - rpos[0])**2 + (drone_pos_2d[1] - rpos[1])**2)
                if distance <= drone_controller.broadcaster.broadcast_range:
                    nearby_robots.append((f"robot_{rid}", tuple(rpos)))
            
            # Create observation for DroneController
            controller_obs = {
                'position': drone_pos_3d,
                'semantic_map': semantic_map,
                'nearby_robots': nearby_robots
            }
            
            # Get action (includes broadcast)
            action = drone_controller.get_action(controller_obs)
            
            # Handle broadcast message
            if action['broadcast'] is not None:
                total_broadcasts += 1
                broadcast_msg = action['broadcast']
                
                # Publish to message broker
                recipient_ids = [r[0] for r in nearby_robots]
                message_broker.publish(broadcast_msg, recipients=recipient_ids)
                
                print(f"Step {step:4d}: Drone broadcast graph ({broadcast_msg.node_count} nodes, "
                      f"{broadcast_msg.edge_count} edges) to {len(recipient_ids)} robots")
        
        # Ground robots check for messages
        for rid in [robot1_id, robot2_id]:
            messages = message_broker.receive(f"robot_{rid}")
            if messages:
                for msg in messages:
                    if msg.message_type == MessageType.GRAPH_BROADCAST:
                        robot_received_counts[rid] += 1
                        print(f"  -> Robot {rid} received graph with {msg.node_count} nodes")
        
        # Clear broadcast queue
        message_broker.clear_broadcasts()
        
        # Simple drone hovering (can add patrol if desired)
        drone_action = np.array([0.0, 0.0])
        
        # Simple ground robot control (move toward center)
        robot1_action = simple_ground_robot_controller(robot1_pos, (250, 250))
        robot2_action = simple_ground_robot_controller(robot2_pos, (250, 250))
        
        # Step simulation
        actions = {
            drone_id: drone_action,
            robot1_id: np.array(robot1_action),
            robot2_id: np.array(robot2_action)
        }
        obs, info = env.step(actions)
        
        # Render with graph overlay (fixed rendering flow to avoid flicker)
        env.render(auto_flip=False)  # Render but don't flip yet
        
        # Always draw graph overlay for consistent rendering
        if drone_controller.current_graph:
            visualize_graph_on_pygame(
                env, drone_controller.current_graph, drone_pos_3d,
                map_size_pixels, map_resolution
            )
        
        # Now flip once after all drawing is complete
        if PYGAME_AVAILABLE and env.renderer:
            pygame.display.flip()
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Broadcast Statistics:")
    print(f"  Total broadcasts sent: {total_broadcasts}")
    print(f"  Robot {robot1_id} received: {robot_received_counts[robot1_id]} graphs")
    print(f"  Robot {robot2_id} received: {robot_received_counts[robot2_id]} graphs")
    
    if drone_controller.current_graph:
        graph = drone_controller.current_graph
        print(f"\nFinal Graph:")
        print(f"  Nodes: {graph.graph.number_of_nodes()}")
        print(f"  Edges: {graph.graph.number_of_edges()}")
    
    print("=" * 60)
    print("Test 3 Finished")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()

