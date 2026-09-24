"""
Test 1: Basic Drone Movement in Simulator
Tests basic drone control and visualization without perception/mapping
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.env import MultiRobotEnv, EnvConfig
from simulator.forest_generator import create_forest_scenario


def simple_patrol_controller(drone_pos, target_waypoints, current_waypoint_idx, waypoint_threshold=10.0):
    """
    Simple patrol controller that moves drone through waypoints.
    
    Args:
        drone_pos: Current drone position (x, y)
        target_waypoints: List of (x, y) waypoints
        current_waypoint_idx: Current target waypoint index
        waypoint_threshold: Distance threshold to consider waypoint reached
        
    Returns:
        (vx, vy, next_waypoint_idx)
    """
    if not target_waypoints:
        return (0.0, 0.0, 0)
    
    # Get current target waypoint
    target = target_waypoints[current_waypoint_idx]
    
    # Compute direction to target
    dx = target[0] - drone_pos[0]
    dy = target[1] - drone_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    # Check if reached waypoint
    if distance < waypoint_threshold:
        # Move to next waypoint (circular)
        next_idx = (current_waypoint_idx + 1) % len(target_waypoints)
        return (0.0, 0.0, next_idx)
    
    # Move toward target
    max_speed = 20.0  # m/s
    vx = (dx / distance) * max_speed
    vy = (dy / distance) * max_speed
    
    return (vx, vy, current_waypoint_idx)


def main():
    print("=" * 60)
    print("Test 1: Basic Drone Movement")
    print("=" * 60)
    
    # Create environment (500m x 500m world)
    world_size = (500, 500)
    config = EnvConfig(
        world_size=world_size,
        dt=0.05,  # 20Hz simulation
        obs_radius=200.0,
        enable_collision=False  # Drones don't collide
    )
    env = MultiRobotEnv(config)
    
    # Add obstacles (forest scene)
    print("\n[Setup] Generating forest scene...")
    obstacles = create_forest_scenario(world_size, scenario_type='medium', seed=42)
    env.obstacles = obstacles
    print(f"[Setup] Added {len(obstacles)} obstacles")
    
    # Add drone at center
    drone_start_x, drone_start_y = 250, 250
    drone_height = 50.0  # 50m altitude
    drone_id = env.add_drone(
        x=drone_start_x,
        y=drone_start_y,
        z=drone_height,
        max_speed=30.0
    )
    print(f"[Setup] Added drone {drone_id} at ({drone_start_x}, {drone_start_y}, {drone_height}m)")
    
    # Define patrol waypoints (square pattern)
    patrol_waypoints = [
        (150, 150),
        (350, 150),
        (350, 350),
        (150, 350)
    ]
    current_waypoint_idx = 0
    
    print(f"[Setup] Patrol waypoints: {patrol_waypoints}")
    print("\n[Simulation] Starting...")
    print("Close the pygame window to stop simulation\n")
    
    # Reset environment
    obs = env.reset()
    
    # Simulation loop
    max_steps = 2000
    for step in range(max_steps):
        # Get drone observation
        drone_obs = obs[drone_id]
        drone_pos = drone_obs['self']['position']
        
        # Compute control action
        vx, vy, current_waypoint_idx = simple_patrol_controller(
            drone_pos, 
            patrol_waypoints, 
            current_waypoint_idx
        )
        
        # Create action
        actions = {
            drone_id: np.array([vx, vy])
        }
        
        # Step simulation
        obs, info = env.step(actions)
        
        # Render
        env.render()
        
        # Print status every 100 steps
        if step % 100 == 0:
            print(f"Step {step:4d}: Drone at ({drone_pos[0]:.1f}, {drone_pos[1]:.1f}), "
                  f"Target waypoint {current_waypoint_idx}: {patrol_waypoints[current_waypoint_idx]}")
    
    print("\n[Simulation] Complete!")
    env.close()
    
    print("\n" + "=" * 60)
    print("Test 1 Finished")
    print("=" * 60)


if __name__ == "__main__":
    main()

