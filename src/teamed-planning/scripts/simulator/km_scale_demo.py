"""
km_scale_demo.py - Kilometer-scale multi-robot simulation demo
Demonstrates navigation with broadcast communication
"""

import numpy as np
from .env import MultiRobotEnv, EnvConfig
from .forest_generator import create_forest_scenario

# Simple navigation policy with broadcast
def navigate_to_target(obs, target):
    """Simple navigation - returns action with broadcast"""
    robot_type = obs['self']['type']
    pos = obs['self']['position']
    to_target = target - pos
    distance = np.linalg.norm(to_target)
    
    if distance < 50:
        if robot_type == 'diff_drive':
            motion = np.array([0.0, 0.0])
        elif robot_type == 'holonomic':
            motion = np.array([0.0, 0.0, 0.0])
        else:
            motion = np.array([0.0, 0.0])
    else:
        direction = to_target / distance
        
        # Simple obstacle avoidance
        for neighbor in obs['neighbors']:
            if neighbor['distance'] < 100:
                repulsion = -neighbor['relative_position'] / (neighbor['distance'] + 1)
                direction += repulsion * 0.3
        
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
        if robot_type == 'diff_drive':
            current_orient = obs['self']['orientation']
            desired_orient = np.arctan2(direction[1], direction[0])
            angle_diff = desired_orient - current_orient
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            motion = np.array([25.0, 2.0 * angle_diff])
        elif robot_type == 'holonomic':
            motion = np.array([direction[0] * 22.0, direction[1] * 22.0, 0.0])
        else:  # drone
            motion = np.array([direction[0] * 40.0, direction[1] * 40.0])
    
    # Broadcast target position
    return {'motion': motion, 'broadcast': target}

def main():
    """Kilometer-scale scene demo with broadcast communication"""
    print("Starting 10km x 10km multi-robot simulation...")
    
    # Create large scene (10km x 10km)
    world_size = (10000, 10000)
    config = EnvConfig(
        world_size=world_size,
        dt=0.1,
        obs_radius=300.0,
        enable_collision=True
    )
    env = MultiRobotEnv(config)
    
    # Generate sparse forest
    obstacles = create_forest_scenario(world_size, 'sparse', seed=123)
    env.obstacles = obstacles
    
    # Add robots and assign targets
    robots_config = [
        # (x, y, theta, robot_type, target_x, target_y)
        (500, 500, np.pi/4, 'diff_drive', 9000, 9000),
        (800, 600, 0, 'holonomic', 9200, 8800),
        (650, 700, 0, 'drone', 9500, 9200),
        (5000, 5000, np.pi/2, 'diff_drive', 5000, 1000),
        (5200, 5200, 0, 'drone', 5500, 1200),
        (9200, 9300, -np.pi/4, 'holonomic', 1000, 1000),
        (9400, 9500, 0, 'drone', 1200, 1500),
    ]
    
    targets = {}
    for i, (x, y, theta, rtype, tx, ty) in enumerate(robots_config):
        if rtype == 'drone':
            env.add_drone(x=x, y=y, z=100, max_speed=50.0)
        else:
            env.add_ground_robot(x=x, y=y, theta=theta, 
                               robot_type=rtype, max_speed=30.0)
        targets[i] = np.array([tx, ty])
    
    print(f"Scene: {len(env.ground_robots)} ground robots, {len(env.drones)} drones, "
          f"{len(obstacles)} obstacles")
    
    observations = env.reset()
    
    arrived = set()
    
    # Simulation loop
    for step in range(10000):
        # Compute actions with broadcast
        actions = {}
        for robot_id, obs in observations.items():
            if robot_id not in arrived:
                actions[robot_id] = navigate_to_target(obs, targets[robot_id])
            else:
                # Stopped robots: no motion, no broadcast
                robot_type = obs['self']['type']
                if robot_type == 'diff_drive':
                    motion = np.array([0.0, 0.0])
                elif robot_type == 'holonomic':
                    motion = np.array([0.0, 0.0, 0.0])
                else:
                    motion = np.array([0.0, 0.0])
                actions[robot_id] = {'motion': motion, 'broadcast': None}
        
        # Execute step
        observations, info = env.step(actions)
        
        # Check arrival
        for robot_id, obs in observations.items():
            if robot_id not in arrived:
                pos = obs['full_state']['position']
                target = targets[robot_id]
                if np.linalg.norm(pos - target) < 100:
                    arrived.add(robot_id)
                    print(f"Robot {robot_id} arrived at target (step {step})")
        
        # Render
        if step % 3 == 0:
            env.render()
        
        # Progress update
        if step % 1000 == 0 and step > 0:
            print(f"Step {step}: {len(arrived)}/{len(targets)} arrived")
        
        # All arrived
        if len(arrived) == len(targets):
            print(f"All robots reached targets in {step} steps")
            for _ in range(100):
                env.render()
            break
    
    env.close()


if __name__ == "__main__":
    main()
