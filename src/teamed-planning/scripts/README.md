# Multi-Robot Navigation System

A hierarchical navigation system for coordinating drones and ground robots in forest environments. Drones build topological graphs from semantic segmentation and broadcast them to ground robots, which fuse multiple graphs for global path planning combined with local obstacle avoidance.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         DRONES                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  Perception  │→ │   Mapping    │→ │ Communication    │ │
│  │  (Semantic)  │  │ (Topological │  │ (Broadcasting)   │ │
│  │              │  │  Graph)      │  │                  │ │
│  └──────────────┘  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ↓
                       Graph Broadcast
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     GROUND ROBOTS                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │Communication │→ │Graph Fusion  │→ │   Planning       │ │
│  │(Receiving)   │  │(Multi-drone) │  │  Global + Local  │ │
│  └──────────────┘  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
multi_robot_navigation/
├── common/                          # Shared utilities
│   ├── graph_structure.py          # TopologicalGraph (NetworkX + KD-tree)
│   ├── coordinate_transforms.py    # Coordinate frame utilities
│   └── message_definitions.py      # Communication protocol
│
├── drone/                           # Drone-side code
│   ├── perception/
│   │   └── semantic_segmentation.py  # Semantic map processing
│   ├── mapping/
│   │   ├── skeleton_extractor.py     # Extract topology from maps
│   │   └── topological_graph_builder.py  # Build graphs
│   ├── communication/
│   │   └── graph_broadcaster.py      # Broadcast graphs to robots
│   └── drone_main.py                 # Main drone controller
│
├── ground_robot/                    # Ground robot code
│   ├── communication/
│   │   └── graph_receiver.py        # Receive graphs from drones
│   ├── mapping/
│   │   ├── graph_fusion.py          # Merge multi-drone graphs
│   │   └── uncertainty_handler.py   # Placeholder for future uncertainty
│   ├── planning/
│   │   ├── global_planner.py        # Topological path planning
│   │   ├── local_planner.py         # Obstacle avoidance
│   │   └── hierarchical_planner.py  # Coordinate global + local
│   └── robot_main.py                # Main robot controller
│
└── example_integration.py           # Integration example

```

## Key Components

### 1. Topological Graph (`common/graph_structure.py`)
- Uses NetworkX for graph operations (pathfinding, connectivity)
- Integrates KD-tree for fast spatial queries (O(log n) nearest neighbor)
- Supports serialization for network transmission
- Stores node attributes (position, terrain type, traversability)

### 2. Drone Perception & Mapping
- **Semantic Segmentation**: Processes ground truth terrain types (simulator provides this)
- **Skeleton Extraction**: Extracts topological skeleton from traversable regions
- **Node Placement**: Places nodes at junctions, endpoints, and regular intervals
- **Edge Creation**: Connects nearby nodes with weighted edges (distance × terrain cost)

### 3. Graph Broadcasting
- Periodic broadcasting when robots are in range (default: 5 second intervals)
- Efficient serialization using NetworkX node-link format
- Simple message passing in V1 (will be replaced with ROS topics)

### 4. Graph Fusion
- **V1 (Current)**: Simple spatial merging
  - Nodes within 2m threshold are merged
  - Positions averaged when merging
  - Edges from all source graphs included
- **Future**: Probabilistic merging with covariance tracking

### 5. Hierarchical Planning
- **Global Planner**: Uses fused topological graph for waypoint generation
  - A* pathfinding on graph
  - Replanning when robot deviates from path
- **Local Planner**: Reactive obstacle avoidance
  - Uses local semantic map from robot sensors
  - Simple proportional controller for velocity commands

## Installation

### Dependencies

```bash
pip install numpy scipy scikit-image networkx opencv-python --break-system-packages
```

### Optional (for future ROS integration)
```bash
# ROS Noetic
sudo apt install ros-noetic-desktop-full
```

## Usage

### Quick Start

Run the example integration:

```bash
cd multi_robot_navigation
python example_integration.py
```

This will simulate:
- 1 drone building and broadcasting graphs
- 2 ground robots receiving graphs and navigating to goals

### Integration with Your Simulator

#### Drone Integration

```python
from drone.drone_main import DroneController

# Initialize drone
drone = DroneController(drone_id="drone_0", resolution=0.5)

# In your simulation loop:
observation = {
    'position': (x, y, z),  # Drone position
    'semantic_map': semantic_map,  # np.ndarray of terrain types
    'nearby_robots': [(robot_id, (rx, ry)), ...]  # Detected robots
}

action = drone.get_action(observation, target_position=(tx, ty, tz))
# action contains:
# - 'movement': (vx, vy, vz) velocity command
# - 'broadcast': GraphBroadcastMessage or None
```

#### Ground Robot Integration

```python
from ground_robot.robot_main import RobotController

# Initialize robot
robot = RobotController(robot_id="robot_0")

# Set goal
robot.set_goal((goal_x, goal_y))

# In your simulation loop:
observation = {
    'position': (x, y),  # Robot position
    'heading': theta,  # Heading in radians
    'semantic_map': local_map,  # Local semantic map (optional)
    'resolution': 0.5,  # Map resolution
    'received_messages': messages  # List of GraphBroadcastMessage
}

action = robot.get_action(observation)
# action contains:
# - 'movement': (linear_vel, angular_vel)
```

### Semantic Map Format

Terrain types (as integers):
- `0`: Unknown
- `1`: Traversable
- `2`: Obstacle
- `3`: Difficult terrain
- `4`: Water
- `5`: Clearing (easy)

## Development Roadmap

### Phase 1: MVP (Current)
- [x] Basic graph construction from semantic maps
- [x] Graph serialization and broadcasting
- [x] Simple spatial graph fusion
- [x] Hierarchical planning (global + local)
- [x] Integration example
- [X] Integrate alg to sim
- [] Move action decision to dronemain and groundmain
- [] Graph generation has problem, need fix


### Phase 2: SLAM Integration
- [ ] Integrate FAST-LIO2 for localization
- [ ] Handle SLAM pose uncertainty
- [ ] Test with realistic localization drift

### Phase 3: Uncertainty Handling
- [ ] Covariance tracking for graph nodes
- [ ] Probabilistic graph fusion
- [ ] Mahalanobis distance for node matching
- [ ] Loop closure detection

### Phase 4: Advanced Features
- [ ] Multi-resolution graphs
- [ ] Dynamic obstacle handling
- [ ] Collaborative SLAM
- [ ] ROS/ROS2 integration

## Configuration

Key parameters can be adjusted in the controllers:

```python
# Drone
DroneController(
    drone_id="drone_0",
    resolution=0.5,  # meters per pixel
    broadcast_interval=5.0  # seconds
)

# Robot
RobotController(
    robot_id="robot_0",
    merge_threshold=2.0,  # meters (for graph fusion)
    max_linear_vel=1.0,  # m/s
    max_angular_vel=1.0  # rad/s
)
```

## Testing

Basic tests can be found in `tests/` directory:

```bash
python tests/test_graph_building.py
python tests/test_graph_fusion.py
```

## Communication Protocol

Messages are defined in `common/message_definitions.py`:

- **GraphBroadcastMessage**: Contains serialized graph + metadata
- **SimpleMessageBroker**: Basic pub-sub for simulation (will be replaced with ROS)

For ROS integration, replace `SimpleMessageBroker` with ROS topics:
- `/drone_*/graph`: Graph broadcasts
- `/robot_*/status`: Robot status updates

## Troubleshooting

### Graphs not being received
- Check broadcast range (default 100m)
- Verify robots are within drone's detection range
- Check message broker is properly routing messages

### Planning failures
- Ensure graph has sufficient nodes (check skeleton extraction)
- Verify start/goal positions are near graph nodes
- Check if path exists between start and goal nodes

### Poor path quality
- Adjust `max_edge_distance` in graph builder
- Modify terrain costs in semantic processor
- Tune local planner parameters

## Future Extensions

### Uncertainty Handling
See `ground_robot/mapping/uncertainty_handler.py` for placeholder implementation.

When adding uncertainty:
1. Modify `TopologicalGraph` to store node covariances
2. Update `GraphFusion` with probabilistic merging
3. Integrate FAST-LIO2 pose covariances
4. Implement loop closure and graph optimization

### ROS Integration
Replace communication layer:
```python
# Instead of SimpleMessageBroker
import rospy
from your_msgs.msg import TopologicalGraph

pub = rospy.Publisher('/drone_0/graph', TopologicalGraph, queue_size=10)
```

## Contributing

When adding features:
1. Follow the existing module structure
2. Add docstrings to all public methods
3. Update README with new capabilities
4. Add tests for new functionality

## License

[Your License Here]

## Citation

If you use this code in your research, please cite:

```
[Your citation here]
```

## Contact

[Your contact information]
