"""
Multi-Drone Perception, Mapping, and Fusion Test

Tests 3 drones with:
- Rectangular perception (30m × 30m square) with semantic segmentation
- Topological graph building with skeleton-based strategy
- Map fusion when drones meet (100m communication range)
- Broadcasting graphs between drones
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.env import MultiRobotEnv, EnvConfig
from simulator.obstacles import CircleObstacle, RectObstacle
from drone.drone_main import DroneController
from ground.ground_main import GroundRobotController

# Try to import pygame for visualization overlay
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available, skipping graph visualization overlay")


# Color schemes for each drone (distinct and different from background/obstacles)
DRONE_COLORS = [
    {'node': (255, 50, 50), 'edge': (200, 0, 0), 'name': 'Red'},
    {'node': (50, 150, 255), 'edge': (0, 80, 200), 'name': 'Blue'},
    {'node': (255, 50, 255), 'edge': (180, 0, 180), 'name': 'Magenta'}
]


def visualize_drone_perceptions(env, drone_controllers, obs, drone_robot_ids, perception_range):
    """
    Visualize what drones perceive: rectangular perception area and semantic maps.
    Uses red color for perception visualization.
    """
    if not PYGAME_AVAILABLE or env.renderer is None:
        return
    
    renderer = env.renderer
    screen = renderer.screen
    
    PERCEPTION_COLOR = (255, 0, 0)  # Red - Better contrast against background
    OBSTACLE_COLOR = (255, 50, 50, 120)  # Red, semi-transparent
    
    for drone_id, controller in drone_controllers.items():
        robot_id = drone_robot_ids[drone_id]
        if robot_id not in obs or controller.position is None:
            continue
        
        drone_pos = controller.position
        
        # Draw rectangular perception area (dotted outline)
        # Perception is square: perception_range × perception_range centered on drone
        half_width = perception_range / 2.0
        
        # Calculate corners in world coordinates
        top_left = (drone_pos[0] - half_width, drone_pos[1] - half_width)
        top_right = (drone_pos[0] + half_width, drone_pos[1] - half_width)
        bottom_right = (drone_pos[0] + half_width, drone_pos[1] + half_width)
        bottom_left = (drone_pos[0] - half_width, drone_pos[1] + half_width)
        
        # Convert to screen coordinates
        corners_screen = [
            renderer.world_to_screen(top_left),
            renderer.world_to_screen(top_right),
            renderer.world_to_screen(bottom_right),
            renderer.world_to_screen(bottom_left)
        ]
        
        # Draw dotted rectangle (draw segments with gaps)
        for i in range(4):
            start = corners_screen[i]
            end = corners_screen[(i + 1) % 4]
            
            # Draw dotted line between corners
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = np.sqrt(dx**2 + dy**2)
            num_segments = int(length / 10)  # ~10 pixel segments
            
            for j in range(num_segments):
                if j % 2 == 0:  # Every other segment for dotted effect
                    t1 = j / num_segments
                    t2 = (j + 1) / num_segments
                    x1 = int(start[0] + t1 * dx)
                    y1 = int(start[1] + t1 * dy)
                    x2 = int(start[0] + t2 * dx)
                    y2 = int(start[1] + t2 * dy)
                    pygame.draw.line(screen, PERCEPTION_COLOR, (x1, y1), (x2, y2), 2)
        
        # Visualize semantic map (what graph builder sees)
        # Get the semantic map from last processing
        if hasattr(controller, '_last_semantic_map') and controller._last_semantic_map is not None and hasattr(controller, '_last_semantic_map_position') and controller._last_semantic_map_position is not None:
            semantic_map = controller._last_semantic_map
            map_drone_pos = controller._last_semantic_map_position
            height, width = semantic_map.shape
            resolution = controller.resolution
            
            obstacle_pixels = np.sum(semantic_map == 2)
            
            # Calculate map bounds in world coordinates (matching SemanticProcessor)
            half_width = (width * resolution) / 2.0
            half_height = (height * resolution) / 2.0
            min_x = map_drone_pos[0] - half_width
            min_y = map_drone_pos[1] - half_height
            
            # Create a surface for the semantic map overlay
            overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
            
            # Draw obstacles from semantic map
            for py in range(0, height, 2):  # Skip pixels for performance
                for px in range(0, width, 2):
                    if semantic_map[py, px] == 2:  # TerrainType.OBSTACLE = 2
                        # Convert pixel to world
                        world_x = min_x + (px + 0.5) * resolution
                        world_y = min_y + (py + 0.5) * resolution
                        
                        # Convert world to screen
                        screen_pos = renderer.world_to_screen((world_x, world_y))
                        
                        # Draw small rectangle (2x2 pixels to show rasterization)
                        rect_size = max(2, int(resolution / renderer.scale))
                        pygame.draw.rect(overlay, OBSTACLE_COLOR, 
                                       (screen_pos[0] - rect_size//2, 
                                        screen_pos[1] - rect_size//2, 
                                        rect_size, rect_size))
            
            screen.blit(overlay, (0, 0))


def visualize_multi_drone_graphs(env, drone_controllers, drone_colors):
    """
    Overlay multiple drone graphs on pygame renderer with different colors.
    Uses translucent colors with bold lines for background context.
    """
    if not PYGAME_AVAILABLE or env.renderer is None:
        return
    
    renderer = env.renderer
    screen = renderer.screen
    
    NODE_RADIUS = 2
    EDGE_WIDTH = 3  # Bolder lines
    ALPHA = 120  # Transparency (~47%)
    
    # Create transparent surface for drone graphs
    overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
    
    # Draw each drone's graph with its color
    for drone_id, controller in drone_controllers.items():
        if controller.current_graph is None:
            continue
        
        graph = controller.current_graph
        colors = drone_colors[drone_id]
        
        # Create translucent colors
        edge_color = (*colors['edge'], ALPHA)
        node_color = (*colors['node'], ALPHA)
        
        # Draw edges first (so nodes appear on top)
        for edge in graph.graph.edges():
            node1_id, node2_id = edge
            pos1 = graph.get_node_position(node1_id)
            pos2 = graph.get_node_position(node2_id)
            
            if pos1 and pos2:
                screen_pos1 = renderer.world_to_screen(pos1)
                screen_pos2 = renderer.world_to_screen(pos2)
                pygame.draw.line(overlay, edge_color, screen_pos1, screen_pos2, EDGE_WIDTH)
        
        # Draw nodes on top
        for node_id, node_data in graph.graph.nodes(data=True):
            pos = node_data['position']
            screen_pos = renderer.world_to_screen(pos)
            pygame.draw.circle(overlay, node_color, screen_pos, NODE_RADIUS)
    
    # Blit overlay onto main screen
    screen.blit(overlay, (0, 0))


def visualize_ground_robot_unified_graphs(env, ground_robot_controllers):
    """
    Visualize unified graphs from ground robots with translucent orange dashed lines and square nodes.
    """
    if not PYGAME_AVAILABLE or env.renderer is None:
        return
    
    renderer = env.renderer
    screen = renderer.screen
    
    UNIFIED_COLOR = (255, 140, 0)  # Orange
    ALPHA = 140  # Transparency (~55%)
    NODE_SIZE = 4  # Square size (larger than drone nodes)
    EDGE_WIDTH = 4  # Bolder dashed lines
    DASH_LENGTH = 5  # Length of dashes
    
    # Create transparent surface for unified graph
    overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
    
    for ground_id, controller in ground_robot_controllers.items():
        if controller.unified_graph is None or len(controller.unified_graph) == 0:
            continue
        
        graph = controller.unified_graph
        
        # Translucent color
        color_with_alpha = (*UNIFIED_COLOR, ALPHA)
        
        # Draw edges first (dashed lines)
        for edge in graph.graph.edges():
            node1_id, node2_id = edge
            pos1 = graph.get_node_position(node1_id)
            pos2 = graph.get_node_position(node2_id)
            
            if pos1 and pos2:
                screen_pos1 = renderer.world_to_screen(pos1)
                screen_pos2 = renderer.world_to_screen(pos2)
                
                # Draw dashed line
                dx = screen_pos2[0] - screen_pos1[0]
                dy = screen_pos2[1] - screen_pos1[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 0:
                    num_dashes = int(distance / (DASH_LENGTH * 2))
                    for i in range(num_dashes):
                        t1 = (i * 2) / (num_dashes * 2)
                        t2 = (i * 2 + 1) / (num_dashes * 2)
                        x1 = int(screen_pos1[0] + t1 * dx)
                        y1 = int(screen_pos1[1] + t1 * dy)
                        x2 = int(screen_pos1[0] + t2 * dx)
                        y2 = int(screen_pos1[1] + t2 * dy)
                        pygame.draw.line(overlay, color_with_alpha, (x1, y1), (x2, y2), EDGE_WIDTH)
        
        # Draw nodes on top (squares)
        for node_id, node_data in graph.graph.nodes(data=True):
            pos = node_data['position']
            screen_pos = renderer.world_to_screen(pos)
            
            # Draw square (rect centered on position)
            rect = pygame.Rect(screen_pos[0] - NODE_SIZE // 2, 
                             screen_pos[1] - NODE_SIZE // 2,
                             NODE_SIZE, NODE_SIZE)
            pygame.draw.rect(overlay, color_with_alpha, rect)
    
    # Blit overlay onto main screen
    screen.blit(overlay, (0, 0))


def visualize_ground_drone_communication(env, ground_robot_controllers, drone_controllers, 
                                        ground_robot_ids, drone_robot_ids):
    """
    Visualize communication between ground robots and drones.
    Green line: Just fused
    Gray line: In communication range but in cooldown
    """
    if not PYGAME_AVAILABLE or env.renderer is None:
        return
    
    renderer = env.renderer
    screen = renderer.screen
    
    COLOR_FUSED = (0, 255, 0)  # Bright green - just fused
    COLOR_COOLDOWN = (180, 180, 180)  # Light gray - in cooldown
    LINE_WIDTH = 2  # Slightly thicker for visibility
    
    for ground_id, ground_controller in ground_robot_controllers.items():
        if ground_controller.position is None:
            continue
        
        ground_pos = ground_controller.position
        ground_screen_pos = renderer.world_to_screen(ground_pos)
        
        # Draw lines to drones based on communication state
        for drone_id, comm_state in ground_controller.current_drone_communications.items():
            # Find corresponding drone controller
            drone_controller = drone_controllers.get(drone_id)
            if drone_controller is None or drone_controller.position is None:
                continue
            
            drone_pos = drone_controller.position
            drone_screen_pos = renderer.world_to_screen(drone_pos)
            
            # Choose color based on fusion state
            if comm_state['fused']:
                color = COLOR_FUSED
            else:
                color = COLOR_COOLDOWN
            
            # Draw line (solid, opaque)
            pygame.draw.line(screen, color, ground_screen_pos, drone_screen_pos, LINE_WIDTH)


def visualize_ground_robot_paths(env, ground_robot_controllers):
    """
    Visualize planned paths for ground robots.
    Always shows in blue for consistency regardless of planning strategy.
    """
    if not PYGAME_AVAILABLE or env.renderer is None:
        return
    
    renderer = env.renderer
    screen = renderer.screen
    
    PATH_COLOR = (0, 150, 255)  # Blue - consistent for all paths
    WAYPOINT_RADIUS = 4
    PATH_WIDTH = 3
    
    for ground_id, controller in ground_robot_controllers.items():
        if controller.global_planner.current_path is None:
            continue
        
        remaining_waypoints = controller.global_planner.get_remaining_path()
        if not remaining_waypoints or len(remaining_waypoints) == 0:
            continue
        
        # Draw path lines (solid blue)
        for i in range(len(remaining_waypoints) - 1):
            pos1 = remaining_waypoints[i]
            pos2 = remaining_waypoints[i + 1]
            
            screen_pos1 = renderer.world_to_screen(pos1)
            screen_pos2 = renderer.world_to_screen(pos2)
            
            pygame.draw.line(screen, PATH_COLOR, screen_pos1, screen_pos2, PATH_WIDTH)
        
        # Draw waypoint markers (blue circles)
        for waypoint in remaining_waypoints:
            screen_pos = renderer.world_to_screen(waypoint)
            pygame.draw.circle(screen, PATH_COLOR, screen_pos, WAYPOINT_RADIUS)
        
        # Draw arrow from robot to first waypoint for clarity
        if len(remaining_waypoints) > 0 and controller.position:
            robot_screen_pos = renderer.world_to_screen(controller.position)
            first_wp_screen_pos = renderer.world_to_screen(remaining_waypoints[0])
            
            # Draw thin line from robot to first waypoint
            pygame.draw.line(screen, PATH_COLOR, robot_screen_pos, first_wp_screen_pos, 1)


def generate_forest_scenario(world_size, obstacle_density):
    """
    Generate a forest scenario with obstacles.
    
    Args:
        world_size: World size in meters
        obstacle_density: Density of obstacles (obstacles per 1000 sq meters)
        
    Returns:
        List of obstacle objects
    """
    obstacles = []
    
    total_area = world_size * world_size
    num_obstacles = int((total_area / 1000.0) * obstacle_density)
    
    # Mix of circular and rectangular obstacles
    num_circles = int(num_obstacles * 0.6)
    num_rects = num_obstacles - num_circles
    
    print(f"Generating {num_obstacles} obstacles ({num_circles} circles, {num_rects} rectangles)")
    
    # Generate circular obstacles (trees)
    for i in range(num_circles):
        x = np.random.uniform(0, world_size)
        y = np.random.uniform(0, world_size)
        radius = np.random.uniform(8.0, 20.0)  # 8-20m radius
        obstacles.append(CircleObstacle(x, y, radius))
    
    # Generate rectangular obstacles (bushes, rocks, logs)
    for i in range(num_rects):
        x = np.random.uniform(0, world_size)
        y = np.random.uniform(0, world_size)
        width = np.random.uniform(10.0, 30.0)
        height = np.random.uniform(10.0, 30.0)
        obstacles.append(RectObstacle(x, y, width, height))
    
    return obstacles


def create_patrol_waypoints(start_pos, world_size, num_waypoints=8, patrol_range=400.0):
    """
    Create patrol waypoints for a drone starting from a position.
    """
    waypoints = []
    center_x = start_pos[0]
    center_y = start_pos[1]
    altitude = start_pos[2]
    
    for i in range(num_waypoints):
        angle = 2 * np.pi * i / num_waypoints
        dx = patrol_range * np.cos(angle)
        dy = patrol_range * np.sin(angle)
        
        x = center_x + dx
        y = center_y + dy
        
        # Keep within world bounds
        x = max(100, min(world_size - 100, x))
        y = max(100, min(world_size - 100, y))
        
        waypoints.append((x, y, altitude))
    
    return waypoints


def run_test():
    """Run multi-drone perception and map fusion test."""
    
    print("="*70)
    print("Multi-Drone Perception and Map Fusion Test")
    print("="*70)
    
    # Configuration for 1km world (smaller for testing)
    WORLD_SIZE = 1000.0
    PERCEPTION_RANGE = 60.0  # Square perception area: 60m × 60m (rectangular)
    COMMUNICATION_RANGE = 100.0  # Communication range for fusion (100m radius)
    FUSION_INTERVAL = 20.0  # Minimum seconds between fusions with same drone
    
    print(f"World size: {WORLD_SIZE}m x {WORLD_SIZE}m")
    print(f"Perception range: {PERCEPTION_RANGE}m × {PERCEPTION_RANGE}m (rectangular, for semantic map)")
    print(f"Communication range: {COMMUNICATION_RANGE}m radius (for fusion)")
    print(f"Fusion interval: {FUSION_INTERVAL}s")
    print("="*70)
    
    # Generate obstacles with high density to create challenging navigation scenarios
    obstacles = generate_forest_scenario(world_size=WORLD_SIZE, obstacle_density=0.3)
    
    # Environment configuration
    # perception_radius: square perception area (30m × 30m)
    # communication_radius: circular communication range (100m radius)
    config = EnvConfig(
        world_size=(WORLD_SIZE, WORLD_SIZE),
        dt=0.1,
        perception_radius=PERCEPTION_RANGE,  # 30m × 30m square for obstacle detection
        communication_radius=COMMUNICATION_RANGE  # 100m radius for drone-to-drone communication
    )
    
    env = MultiRobotEnv(config)
    
    # Add obstacles to environment
    env.obstacles = obstacles
    print(f"Added {len(obstacles)} obstacles to environment")
    
    # Initialize 3 drones
    num_drones = 3
    drone_controllers = {}
    drone_colors = {}
    drone_robot_ids = {}  # Map drone_id (string) to robot_id (int)
    
    # All drones start near ground robot (realistic - all start together)
    drone_start_positions = [
        (520, 520, 50.0),   # Drone 0: at ground robot start
        (525, 525, 50.0),   # Drone 1: slightly offset
        (515, 525, 50.0),   # Drone 2: slightly offset
    ]
    
    # NOTE: No manual waypoints needed!
    # Drones will automatically:
    # 1. Discover ground robot's mission (start → target)
    # 2. Assign themselves to segments (3 drones = 3 segments)
    # 3. Generate vortex exploration patterns
    # 4. Each drone explores a different segment of the path
    
    for i in range(num_drones):
        drone_id = f"drone_{i}"
        start_pos = drone_start_positions[i]
        
        # Add drone to environment and get robot_id
        robot_id = env.add_drone(x=start_pos[0], y=start_pos[1], z=start_pos[2], max_speed=25.0)
        drone_robot_ids[drone_id] = robot_id
        
        controller = DroneController(
            drone_id=drone_id,
            resolution=0.5,
            perception_range=PERCEPTION_RANGE,  # 30m × 30m square
            communication_range=COMMUNICATION_RANGE,
            fusion_interval=FUSION_INTERVAL,
            max_speed=15.0,
            waypoint_threshold=10.0
            # map_size auto-calculated: 30m / 0.5m = 60 pixels
        )
        
        # Waypoints are automatically generated based on ground robot mission
        # No manual setting needed!
        
        drone_controllers[drone_id] = controller
        drone_colors[drone_id] = DRONE_COLORS[i]
        
        print(f"\n{drone_id} ({DRONE_COLORS[i]['name']} graph): Start at {start_pos}, robot_id={robot_id}")
    
    # Initialize 1 ground robot
    num_ground_robots = 1
    ground_robot_types = ['diff_drive']
    ground_robot_controllers = {}
    ground_robot_ids = {}
    ground_robot_start_positions = [(520, 520)]
    ground_robot_target_positions = [(900, 900)]

    for i in range(num_ground_robots):
        ground_robot_id = f"ground_{i}"
        robot_id = env.add_ground_robot(x=ground_robot_start_positions[i][0], y=ground_robot_start_positions[i][1], theta=0)
        ground_robot_ids[ground_robot_id] = robot_id
        ground_controller = GroundRobotController(
            robot_id=ground_robot_id,
            max_speed=2.0,  # Slower than drones
            fusion_interval=FUSION_INTERVAL
        )
        ground_controller.set_target(ground_robot_target_positions[i])
        ground_robot_controllers[ground_robot_id] = ground_controller

        print(f"\nground_{i}: Start at {ground_robot_start_positions[i]}, target at {ground_robot_target_positions[i]}, robot_id={robot_id}")

    print("\nDrone mission-based exploration:")
    print("  Ground robot path: (520, 520) → (900, 900)")
    print("  Path divided into 3 segments (one per drone)")
    print("  Each drone autonomously:")
    print("    1. Discovers ground robot mission")
    print("    2. Assigns itself to a segment")
    print("    3. Flies to segment start")
    print("    4. Executes vortex exploration pattern")
    print("  This tests: autonomous coordination + path-oriented exploration")
    
    # Main simulation loop
    print("\n" + "="*70)
    print("Starting simulation...")
    print("="*70)
    
    running = True
    max_steps = 2000
    step = 0
    last_print_time = time.time()
    print_interval = 2.0
    
    # Track previous graph sizes for change detection
    prev_graph_sizes = {}
    
    while running and step < max_steps:
        step += 1
        
        # Step 1: Get current observations
        if step == 1:
            # First step: reset to get initial observations
            obs = env.reset()
        else:
            # Subsequent steps: execute actions and get new observations
            obs, info = env.step(actions)
        
        # Step 2: Get actions from each drone controller
        actions = {}
        for drone_id, controller in drone_controllers.items():
            robot_id = drone_robot_ids[drone_id]
            if robot_id not in obs:
                continue
            
            # Get complete action from controller (handles everything internally)
            action = controller.get_action(obs[robot_id])
            actions[robot_id] = action

        # Step 3: Get actions from each ground robot controller
        for ground_robot_id, controller in ground_robot_controllers.items():
            robot_id = ground_robot_ids[ground_robot_id]
            if robot_id not in obs:
                continue
            
            # Get complete action from controller (handles everything internally)
            action = controller.get_action(obs[robot_id])
            actions[robot_id] = action
        
        # Step 4: Render current state
        env.render(auto_flip=False)
        
        if PYGAME_AVAILABLE:
            # Layer 1: Drone perception areas (semi-transparent)
            visualize_drone_perceptions(env, drone_controllers, obs, drone_robot_ids, PERCEPTION_RANGE)
            # Layer 2: Drone graphs (colored nodes/edges)
            visualize_multi_drone_graphs(env, drone_controllers, drone_colors)
            # Layer 3: Ground robot unified graphs (orange squares, dashed edges)
            visualize_ground_robot_unified_graphs(env, ground_robot_controllers)
            # Layer 4: Ground robot planned paths (yellow/cyan lines)
            visualize_ground_robot_paths(env, ground_robot_controllers)
            # Layer 5: Communication lines (on top)
            visualize_ground_drone_communication(env, ground_robot_controllers, drone_controllers,
                                                ground_robot_ids, drone_robot_ids)
            pygame.display.flip()
        
        # Handle events
        if PYGAME_AVAILABLE:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
        
        # Print periodic statistics
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            print(f"\n{'='*70}")
            print(f"Step {step}")
            print(f"{'='*70}")
            
            for drone_id, controller in drone_controllers.items():
                if controller.current_graph:
                    graph = controller.current_graph
                    pos = controller.position
                    
                    # Get current sizes
                    curr_nodes = graph.graph.number_of_nodes()
                    curr_edges = graph.graph.number_of_edges()
                    
                    # Get previous sizes
                    prev_nodes, prev_edges = prev_graph_sizes.get(drone_id, (0, 0))
                    
                    # Calculate changes
                    nodes_added = curr_nodes - prev_nodes
                    edges_added = curr_edges - prev_edges
                    
                    # Update tracking
                    prev_graph_sizes[drone_id] = (curr_nodes, curr_edges)
                    
                    print(f"\n{drone_id} ({drone_colors[drone_id]['name']}):")
                    if pos:
                        print(f"  Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
                    
                    # Show graph size with changes
                    change_str = ""
                    if nodes_added > 0 or edges_added > 0:
                        change_str = f" (+{nodes_added}n, +{edges_added}e)"
                    print(f"  Graph: {curr_nodes} nodes, {curr_edges} edges{change_str}")
                    
                    # Show edge/node ratio
                    ratio = curr_edges / curr_nodes if curr_nodes > 0 else 0
                    print(f"  Edge/Node ratio: {ratio:.2f}:1")
                    
                    print(f"  Fusions: {len(graph.fusion_history)}")
                    if len(graph.fusion_history) > 0:
                        for other_id, last_time in graph.fusion_history.items():
                            time_since = current_time - last_time
                            print(f"    Last fusion with {other_id}: {time_since:.1f}s ago")
                    
                    # Print navigation stats
                    nav_stats = controller.navigator.get_stats()
                    print(f"  Navigation: waypoint {nav_stats['current_waypoint']}/{nav_stats['total_waypoints']}, "
                          f"reached {nav_stats['waypoints_reached']}")
                    
                    controller.print_timing_stats(summary=True)
            
            # Print ground robot states
            for ground_id, controller in ground_robot_controllers.items():
                state = controller.get_state()
                print(f"\n{ground_id}:")
                if state['position']:
                    print(f"  Position: ({state['position'][0]:.1f}, {state['position'][1]:.1f})")
                if state['goal_position']:
                    goal_dist = np.sqrt((state['position'][0] - state['goal_position'][0])**2 + 
                                       (state['position'][1] - state['goal_position'][1])**2)
                    print(f"  Goal: ({state['goal_position'][0]:.1f}, {state['goal_position'][1]:.1f}) - {goal_dist:.1f}m away")
                print(f"  Unified graph: {state['unified_graph_nodes']} nodes, {state['unified_graph_edges']} edges" 
                      if state['has_graph'] else "  Unified graph: None")
                print(f"  Graphs received: {state['graphs_received']}")
                print(f"  Planning: {'GRAPH-based' if state['using_graph'] else 'DIRECT waypoints'}")
                print(f"  Waypoints remaining: {state['waypoints_remaining']}")
                controller.print_timing_stats(summary=True)
            
            last_print_time = current_time
    
    print("\n" + "="*70)
    print("Simulation complete!")
    print("="*70)
    
    for drone_id, controller in drone_controllers.items():
        print(f"\n{drone_id} final stats:")
        if controller.current_graph:
            graph = controller.current_graph
            print(f"  Total nodes: {graph.graph.number_of_nodes()}")
            print(f"  Total edges: {graph.graph.number_of_edges()}")
            print(f"  Fusions performed: {len(graph.fusion_history)}")
        controller.print_timing_stats(summary=False)
    
    for ground_id, controller in ground_robot_controllers.items():
        print(f"\n{ground_id} final stats:")
        state = controller.get_state()
        if state['has_graph']:
            print(f"  Unified graph: {state['unified_graph_nodes']} nodes, {state['unified_graph_edges']} edges")
        print(f"  Total graphs received: {state['graphs_received']}")
        if state['goal_position']:
            goal_dist = np.sqrt((state['position'][0] - state['goal_position'][0])**2 + 
                               (state['position'][1] - state['goal_position'][1])**2)
            print(f"  Distance to goal: {goal_dist:.1f}m")
        controller.print_timing_stats(summary=False)
    
    env.close()


def main():
    """Main entry point - runs unified perception and fusion test."""
    run_test()


if __name__ == "__main__":
    main()

