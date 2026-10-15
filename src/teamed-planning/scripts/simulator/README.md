# Simple 2D Heterogeneous Multi-Robot Simulator

MAPPO training, controlled baselines and offline 2D replay: [implementation and experiment instructions](MAPPO.md).

A kilometer-scale multi-robot forest simulation environment designed for reinforcement learning.

## ✨ Core Features

🎨 **Enhanced Visualization (Pygame)**
- Ground robots: Red squares (15px) + direction arrow + ID label
- Drones: Realistic quadcopter (X-frame + 4 propellers) + ID label
- Forest scene: Multi-layer tree rendering

🗺️ **Kilometer-Scale Support**
- Supports 100m - 10km+ maps
- Robots use fixed pixel sizes, always visible
- Random forest generator with controllable density

🔍 **Interactive View Controls**
- **Mouse Wheel**: Zoom in/out (0.1x - 20x)
- **Mouse Drag**: Pan view
- **R Key**: Reset view
- **G/L/V Keys**: Toggle grid/labels/velocity display

## 📁 Project Files

```
simple_multi_robot_sim/
├── env.py                  # Core environment class (Gym-like interface)
├── robots.py               # Robot definitions
├── obstacles.py            # Obstacles
├── utils.py                # Utility functions
├── renderer_pygame.py      # Pygame renderer (with zoom/pan)
├── forest_generator.py     # Forest scene generator
├── mission_demo.py         # Cooperative forest navigation
├── km_scale_demo.py        # 10km scene demo
└── requirements.txt
```

## 🚀 Quick Start

### Installation
```bash
pip install numpy pygame
```

### Cooperative Forest Navigation (1km x 1km)
```bash
python mission_demo.py --steps 20000
```

### Large Scene Demo (10km x 10km, 7 robots)
```bash
python km_scale_demo.py
```

## ⌨️ Interactive Controls

### Mouse Controls
- **Scroll Up/Down**: Zoom in/out
- **Left Click + Drag**: Pan view

### Keyboard Shortcuts
- **R**: Reset view to initial state
- **G**: Toggle grid display
- **L**: Toggle robot label display
- **V**: Toggle velocity arrow display
- **F**: Toggle FPS counter
- **1/2/3/0**: Set rendering FPS to 30/60/120/Unlimited

### View Information
- Top-left panel shows real-time:
  - Robot counts
  - Obstacle count
  - Current zoom level
  - Current FPS limit
- Top-right: FPS counter (when enabled)

## ⚙️ Rendering vs Simulation Frequency

### Rendering Frequency (Default: 30 FPS)
- **Visual display refresh rate**
- Default: 30 FPS (smooth and efficient)
- Adjustable: Press 1/2/3/0 for 30/60/120/Unlimited FPS
- Does NOT affect simulation accuracy

### Simulation Frequency (Configurable via `dt`)
- **Physics update rate**
- Configurable through `EnvConfig(dt=...)`
- Affects simulation accuracy and speed
- Independent of rendering

```python
# Example: Fast simulation (large dt), smooth rendering (30 FPS)
config = EnvConfig(
    world_size=(5000, 5000),
    dt=0.1  # 10Hz simulation (fast)
)
env = MultiRobotEnv(config)
# Renders at 30 FPS by default (smooth visuals)

# Example: Precise simulation (small dt), same rendering
config = EnvConfig(
    world_size=(100, 100),
    dt=0.01  # 100Hz simulation (precise)
)
# Still renders at 30 FPS
```

## ⚡ Performance Optimizations

The renderer includes several optimizations for smooth interaction:

### 1. **Background Caching**
- Static elements (obstacles, grid) are cached
- Only redrawn when view changes (zoom/pan)
- Reduces rendering overhead by 50-80%

### 2. **View Frustum Culling**
- Only objects within visible area are drawn
- Dramatically improves performance with many obstacles
- Automatic culling based on current view

### 3. **Fixed 30 FPS Rendering**
- Default 30 FPS provides smooth visuals
- Can adjust: Press 1/2/3/0 for 30/60/120/Max FPS
- Rendering is independent of simulation frequency

### 4. **Hardware Acceleration**
- Uses pygame's HWSURFACE and DOUBLEBUF flags
- Leverages GPU when available

## 🎨 Visualization Details

### Robots
- **Ground Robots** (Red squares - 15px)
  - Square body + direction arrow
  - White label box with ID (G0, G1, ...)
  
- **Drones** (Quadcopter design)
  - 4 gray arms in X-pattern
  - 4 blue propeller circles at arm ends
  - Dark blue cross frame in center
  - White label box with ID (D0, D1, ...)
  - Height display below (H:50m)

### Forest
- **Trees**: 3-layer circles (outer green + inner dark green + brown trunk)
- **Groves**: Rectangular areas

### Size Notes
- Robots use **fixed pixel sizes**, independent of map scale
- Ground robots: 15-pixel squares
- Drones: 12-pixel center + 20-pixel arm length
- Better conveys kilometer-scale scene magnitude

## 🌲 Forest Scene Generation

```python
from forest_generator import create_forest_scenario

# Generate forest
obstacles = create_forest_scenario(
    world_size=(5000, 5000),  # 5km x 5km
    scenario_type='medium',    # 'sparse', 'medium', 'dense'
    seed=42                    # Reproducible
)

# Add to environment
env.obstacles = obstacles
```

### Density Description
- **sparse**: ~1 tree/5000m², suitable for fast movement
- **medium**: ~1 tree/2000m², balanced
- **dense**: ~1 tree/1000m², requires obstacle avoidance

## 📊 Usage Examples

### Example 1: Basic Usage
```python
from env import MultiRobotEnv, EnvConfig
from forest_generator import create_forest_scenario
import numpy as np

# Create environment
config = EnvConfig(
    world_size=(2000, 2000),  # 2km x 2km
    dt=0.05,                  # 20Hz simulation (0.05s per step)
    obs_radius=200.0
)
env = MultiRobotEnv(config)

# Generate forest
obstacles = create_forest_scenario((2000, 2000), 'medium', seed=42)
env.obstacles = obstacles

# Add robots
env.add_ground_robot(x=100, y=100, robot_type='diff_drive', max_speed=20.0)
env.add_drone(x=200, y=200, z=50, max_speed=30.0)

# Simulation loop
obs = env.reset()
for step in range(1000):
    actions = {
        0: np.array([15.0, 0.1]),  # [v, omega]
        1: np.array([20.0, 15.0])  # [vx, vy]
    }
    obs, info = env.step(actions)
    env.render()  # Can zoom with scroll, pan with drag!

env.close()
```

### Example 2: Different Scale Scenes

**Small scene** (100m x 100m, high precision)
```python
config = EnvConfig(
    world_size=(100, 100), 
    dt=0.01  # 100Hz - precise simulation
)
env.add_ground_robot(x=10, y=10, max_speed=2.0)
```

**Medium scene** (1km x 1km, balanced)
```python
config = EnvConfig(
    world_size=(1000, 1000), 
    dt=0.05  # 20Hz - balanced
)
env.add_ground_robot(x=100, y=100, max_speed=15.0)
```

**Large scene** (5km x 5km, fast)
```python
config = EnvConfig(
    world_size=(5000, 5000), 
    dt=0.1  # 10Hz - fast simulation
)
env.add_ground_robot(x=500, y=500, max_speed=30.0)
```

**Note**: All render at 30 FPS regardless of `dt`

## 🎯 Observation Structure

```python
observations[robot_id] = {
    'self': {                    # For agent use
        'position': [x, y],
        'velocity': [vx, vy],
        'orientation': theta,
        'broadcast': np.ndarray or None,  # Current broadcast message
        ...
    },
    'neighbors': [               # For agent use
        {
            'id': neighbor_id,
            'type': 'ground' or 'drone',
            'relative_position': [dx, dy],
            'relative_velocity': [dvx, dvy],
            'distance': dist,
            'broadcast': np.ndarray or None  # Neighbor's broadcast message
        },
        ...
    ],
    'obstacles': [...],          # For agent use
    'full_state': {              # For reward calculation
        'position': [x, y],
        'all_robots_positions': [[x1,y1], [x2,y2], ...],
        ...
    }
}
```

## 📡 Broadcast Communication

Robots can broadcast information to neighbors within observation radius:

### Basic Usage
```python
# Action with broadcast
action = {
    'motion': np.array([v, omega]),      # Motion command
    'broadcast': np.array([1.0, 2.0, 3.0])  # Broadcast message (any vector)
}
obs, info = env.step({robot_id: action})

# Backward compatible: motion only (no broadcast)
action = np.array([v, omega])
obs, info = env.step({robot_id: action})
```

### Reading Broadcasts
```python
# Get own broadcast
my_broadcast = obs[robot_id]['self']['broadcast']  # None if not set

# Get neighbors' broadcasts
for neighbor in obs[robot_id]['neighbors']:
    neighbor_id = neighbor['id']
    neighbor_broadcast = neighbor['broadcast']  # None if neighbor didn't broadcast
    if neighbor_broadcast is not None:
        print(f"Robot {neighbor_id} broadcasts: {neighbor_broadcast}")
```

### Example Use Cases
- **Coordination**: Broadcast target position or task assignment
- **Information sharing**: Share detected obstacles or goal locations
- **Communication protocol**: Custom message formats for team coordination
- **Distributed consensus**: Exchange state information for coordination

## 🔧 Parameter Recommendations

### Simulation Time Step (`dt`)
Controls the simulation frequency and accuracy:

| dt | Frequency | Use Case | Performance |
|----|-----------|----------|-------------|
| 0.1 | 10Hz | Fast prototyping, large scenes | Very fast |
| 0.05 | 20Hz | General use, balanced | Fast |
| 0.033 | 30Hz | Standard robotics | Balanced |
| 0.01 | 100Hz | High precision, small scenes | Slower |

### Large Scenes (>1km)
```python
config = EnvConfig(
    world_size=(5000, 5000),
    dt=0.1,              # 10Hz sim (fast)
    obs_radius=300.0
)
env.add_ground_robot(..., max_speed=30.0)
env.add_drone(..., max_speed=50.0)
# Renders at 30 FPS regardless of dt
```

### Small Scenes (<100m)
```python
config = EnvConfig(
    world_size=(100, 100),
    dt=0.01,             # 100Hz sim (precise)
    obs_radius=20.0
)
env.add_ground_robot(..., max_speed=2.0)
env.add_drone(..., max_speed=5.0)
# Still renders at 30 FPS
```

### Training Speedup
```python
# Render every N steps for faster training
for step in range(1000):
    obs, info = env.step(actions)
    if step % 10 == 0:  # Render every 10 steps
        env.render()
```

## Example Programs

- **mission_demo.py**: Three drones map a 1km forest and support ground navigation.
- **km_scale_demo.py**: Larger forest visualization and navigation example.

## ⚡ Performance Tips

- **Large scenes**: Use larger `dt` (0.1) for faster simulation
- **Render selectively**: Call `env.render()` every 5-10 steps during training
- **Adjust rendering FPS**: Press 1 for 30 FPS (smooth), 0 for unlimited (fast)
- **Use zoom**: Zoom in to reduce visible objects (frustum culling)
- **Toggle displays**: Press G/L/V to disable grid/labels/velocity
- **Monitor FPS**: Press F to see actual rendering FPS

## 🐛 Common Questions

**Q: How to zoom in on a region?**  
A: Move mouse to the region, scroll up to zoom in, drag to adjust position

**Q: Robots too small to see?**  
A: Robots use fixed pixel sizes (30-40px), always clear when zoomed. Press L to show labels

**Q: How to return to global view?**  
A: Press R key to reset view

**Q: Forest too dense/sparse?**  
A: Use `scenario_type`: 'sparse', 'medium', 'dense'

**Q: How to save video?**  
A: Use `mode='rgb_array'` to get image arrays, save as video

## 📚 API Quick Reference

```python
# Environment configuration
config = EnvConfig(
    world_size=(w, h),   # meters
    dt=0.05,             # simulation time step (smaller = more accurate)
    obs_radius=200.0,    # meters
    enable_collision=True
)

# Add robots
env.add_ground_robot(x, y, theta=0, robot_type='diff_drive', max_speed=20)
env.add_drone(x, y, z=50, max_speed=30)

# Generate forest
obstacles = create_forest_scenario(world_size, 'medium', seed=42)

# Simulation loop
obs = env.reset()
obs, info = env.step(actions)  # Advances simulation by dt seconds
env.render()  # Renders at 30 FPS (independent of dt)
env.close()
```

### Key Concepts
- **`dt`**: Simulation time step (affects accuracy, not rendering)
- **Rendering FPS**: Visual refresh rate (default 30, adjustable with keys)
- **Decoupled**: Simulation and rendering are independent

## 📄 License

MIT

---

**Tip**: When running examples, try scrolling to zoom and dragging to pan - enjoy the smooth interaction! 🚀
