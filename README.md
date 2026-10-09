# Teamed Planning

Windows + Docker Desktop ROS 1 simulation: see [setup and commands](docker/ros1/README.md).
Run `./run-ros1.ps1 start` from PowerShell in this directory.

A multi-robot cooperative navigation system where UAV drones build topological maps
of the environment and broadcast them to a ground robot, which fuses the maps and
navigates to a goal using hierarchical planning.

> **Branch guide**
> | Branch | ROS version | Status |
> |--------|------------|--------|
> | `noetic` | ROS 1 Noetic | **Active development** |
> | `main`   | ROS 2 Humble | Archived |

---

## System Overview

```
Operator (main.sh)
      │  navigate <x> <y>
      ▼
  GCS Node  ──────────────────────────────────────────────────────────┐
      │  /operator/goal → /ground_0/goal                              │
      │  /operator/command → drones / ground                          │  RViz
      │                                                               │  /visualization_markers
      ├──── /drone_0/command  ──▶  Drone Node 0                      │
      ├──── /drone_1/command  ──▶  Drone Node 1                      │
      └──── /drone_2/command  ──▶  Drone Node 2                      │
                                        │                             │
                   /drone_N/graph_broadcast (TopologicalGraphMsg)     │
                                        ▼                             │
                                  Ground Node  ──── /ground_0/cmd_vel ──▶ robot
                                        │
                                  /ground_0/mission_broadcast
```

### What each component does

| Component | Role |
|-----------|------|
| **Drone Node** | MAVROS flight control, topological graph building via skeleton extraction, graph broadcast |
| **Ground Node** | Receives drone graphs, fuses them, runs hierarchical path planning, publishes cmd_vel |
| **GCS Node** | Operator interface, relays goals and commands, publishes RViz markers |
| **main.sh** | Single CLI entry point for the whole system (`start / takeoff / navigate / land / stop`) |

---

## Repository Structure

```
src/
├── teamed-planning/          # ◀ Main ROS package (this project)
│   ├── config/
│   │   ├── algorithm/        # Shared planning params (sim + real)
│   │   │   ├── drone.yaml
│   │   │   ├── ground.yaml
│   │   │   └── operator.yaml
│   │   ├── sim/              # XTDrone/Gazebo only
│   │   │   ├── world.yaml    # Spawn poses (single source of truth)
│   │   │   └── xtdrone.yaml
│   │   ├── real/             # Field deployment only
│   │   │   ├── fleet.yaml
│   │   │   └── sensors.yaml
│   │   └── gcs.rviz
│   ├── deploy/
│   │   └── main.sh           # Operator CLI — start here
│   ├── launch/
│   │   ├── multi_robot_xtdrone.launch   # Sim: all planning nodes
│   │   ├── drone_xtdrone.launch         # Sim: single drone node
│   │   ├── ground_robot_xtdrone.launch  # Sim: ground node + TF
│   │   ├── gcs.launch
│   │   ├── multi_vehicle_with_ground.launch  # Runs inside xtdrone-ad container
│   │   ├── drone_real.launch            # Real hardware
│   │   └── ground_robot_real.launch
│   ├── msg/
│   │   ├── TopologicalGraphMsg.msg
│   │   ├── GraphNode.msg / GraphEdge.msg
│   │   ├── MissionBroadcast.msg
│   │   ├── RobotCommand.msg
│   │   └── RobotStatus.msg
│   ├── scripts/
│   │   ├── nodes/
│   │   │   ├── drone_node.py    # ROS wrapper for drone algorithm
│   │   │   ├── ground_node.py   # ROS wrapper for ground algorithm
│   │   │   └── gcs_node.py      # Ground control station
│   │   ├── drone/
│   │   │   ├── drone_main.py          # DroneController
│   │   │   ├── drone_navigator.py     # Mission phases
│   │   │   ├── drone_local_planner.py # Step-wise path following
│   │   │   ├── skeleton_graph_builder.py
│   │   │   ├── graph_broadcaster.py
│   │   │   └── semantic_segmentation.py  # Stub (replace with real model)
│   │   ├── ground/
│   │   │   ├── ground_main.py    # GroundRobotController
│   │   │   ├── global_planner.py # A* on fused topological graph
│   │   │   ├── local_planner.py  # Reactive obstacle avoidance
│   │   │   ├── graph_fusion.py   # Merge graphs from multiple drones
│   │   │   └── graph_receiver.py
│   │   ├── common/
│   │   │   ├── graph_structure.py   # TopologicalGraph (NetworkX + KD-tree)
│   │   │   └── message_definitions.py
│   │   └── simulator/           # Pure-Python sim (no ROS needed)
│   └── urdf/
│       └── ground_robot_sim.urdf  # Diff-drive + 32-line LiDAR for Gazebo
│
├── aerial/                   # Legacy aerial code (scripts only)
├── ground/ground-planner/    # C++ ground planner (separate package)
├── dependencies/             # Third-party ROS packages
└── ...
```

---

## Simulation Setup

The simulation uses two Docker containers that share the host network:

| Container | Image | Role |
|-----------|-------|------|
| `xtdrone-ad` | XTDrone (PX4 + Gazebo + MAVROS) | Physics simulation |
| `teamed-planning-ad` | Ubuntu 20.04 + ROS Noetic | Planning nodes |

Both containers mount the workspace at `/home/dcist/Teamed-Planning` (inside) →
`/home/ad/Documents/p5/codegit/Teamed-Planning` (host).

### Prerequisites

- Host machine with Docker and Nvidia GPU
- `xtdrone-ad` image with `multi_vehicle_with_ground.launch` copied to
  `/root/PX4_Firmware/launch/` and `ground_robot_sim.urdf` placed alongside it

---

## Quick Start (Simulation)

All commands run from the **host machine** in `src/teamed-planning/deploy/`:

```bash
# 1. Start simulation (Gazebo + PX4 + MAVROS + planning nodes + RViz)
./main.sh --sim start

# 2. Arm and take off (all 3 drones)
./main.sh --sim takeoff

# 3. Send navigation goal to ground robot and start drone mission
./main.sh --sim navigate 10.0 5.0

# 4. Land all drones and stop ground robot
./main.sh --sim land

# 5. Shut down everything (Gazebo, MAVROS, planning nodes)
./main.sh --sim stop
```

Monitor in RViz:

```bash
./main.sh --sim rviz
```

---

## Coordinate Frames

All planning, status reporting, and RViz visualization use the **map frame**,
which is approximately aligned with the Gazebo world frame.

```
Gazebo world (0,0,0) ≈ map (0,0,0)
          │
          ├── map → ground_0/odom  (identity static TF)
          │         → ground_0/base_link  (published by diff-drive plugin)
          │
          └── map ≈ iris_N MAVROS local  (each drone: local + spawn_offset = map)
```

### Spawn positions (Gazebo world frame)

Defined in `config/sim/world.yaml` (must match `multi_vehicle_with_ground.launch`):

| Robot | x | y | z |
|-------|---|---|---|
| iris_0 / drone_0 | -2 | 0 | 0 |
| iris_1 / drone_1 | 2 | 0 | 0 |
| iris_2 / drone_2 | 0 | 2 | 0 |
| ground_0 | 0 | 0 | 0.15 |

Each drone node receives its `spawn_x / spawn_y` via ROS params and adds them to
MAVROS-local positions before reporting status or running the planning algorithm.
The ground node does the same with its odom values. Flight control setpoints remain
in MAVROS-local frame.

---

## Custom Messages

| Message | Fields | Purpose |
|---------|--------|---------|
| `TopologicalGraphMsg` | nodes[], edges[] | Drone → ground graph broadcast |
| `GraphNode` | id, x, y, confidence | Single graph node |
| `GraphEdge` | from_id, to_id, weight | Single graph edge |
| `MissionBroadcast` | position, target, has_target | Ground robot → drones |
| `RobotCommand` | command, target_robot_id, param1 | GCS → robots |
| `RobotStatus` | robot_id, position, flight_state, graph_nodes | All → GCS |

---

## ROS Topics (Simulation)

### Per-drone (`/drone_N/`)

| Topic | Type | Direction |
|-------|------|-----------|
| `mavros/local_position/odom` | `nav_msgs/Odometry` | MAVROS → node |
| `mavros/setpoint_position/local` | `geometry_msgs/PoseStamped` | node → MAVROS |
| `velodyne_points` | `sensor_msgs/PointCloud2` | Gazebo → node |
| `graph_broadcast` | `TopologicalGraphMsg` | node → all |
| `status` | `RobotStatus` | node → GCS |
| `command` | `RobotCommand` | GCS → node |

### Ground robot (`/ground_0/`)

| Topic | Type | Direction |
|-------|------|-----------|
| `odom` | `nav_msgs/Odometry` | Gazebo plugin → node |
| `cmd_vel` | `geometry_msgs/Twist` | node → Gazebo plugin |
| `velodyne_points` | `sensor_msgs/PointCloud2` | Gazebo → node |
| `mission_broadcast` | `MissionBroadcast` | node → drones |
| `goal` | `geometry_msgs/PoseStamped` | GCS → node |

---

## Current Status

### Working ✅
- Full simulation lifecycle: start → takeoff → navigate → land → stop
- Multi-drone OFFBOARD takeoff with PX4 EKF2 convergence check
- Hover altitude hold (locked setpoint, immune to EKF drift)
- Ground robot navigation toward operator goal
- Topological graph building and inter-drone broadcasting
- RViz visualization: robot poses, paths, goal marker, graph
- Coordinate frame alignment: all positions in shared map frame
- Robust startup / shutdown (no zombie processes between cycles)

### In Progress 🔧
- Graph visualization in RViz (node/edge markers from drone graph topics)
- Drone mission navigation using fused graph (MISSION state)

### Planned 📋
- FAST-LIO2 integration for SLAM-based localization
- Real semantic segmentation (replace stub)
- Real hardware deployment via `config/real/fleet.yaml`
- Probabilistic graph fusion with covariance tracking

---

## Configuration

Config is split into three layers so real-world tuning does not break simulation:

| Layer | Path | Edit when |
|-------|------|-----------|
| **Algorithm** (shared) | `config/algorithm/` | Planning behaviour changes |
| **Simulation** | `config/sim/` | Gazebo spawn, XTDrone quirks |
| **Real hardware** | `config/real/` | Field IPs, sensors, SLAM |

### `config/algorithm/drone.yaml` (shared)
```yaml
takeoff_height: 15.0
takeoff_height_stagger: 3.0
max_speed: 5.0
communication_range: 30.0
```

### `config/algorithm/ground.yaml` (shared)
```yaml
max_linear_vel: 1.0      # m/s
max_angular_vel: 1.0     # rad/s
perception_range: 8.0    # metres
```

### `config/sim/world.yaml` (sim only)
Spawn poses for all robots — must match Gazebo `multi_vehicle_with_ground.launch`.

### `config/real/fleet.yaml` (real only)
IPs and IDs for field deployment (3 drones + 1 ground robot).

---

## Dependencies

### Planning container (`teamed-planning-ad`)
```bash
pip install numpy scipy scikit-image networkx opencv-python
# ROS packages: rospy, mavros_msgs, sensor_msgs, nav_msgs, geometry_msgs, tf
```

### Simulation container (`xtdrone-ad`)
- PX4 SITL + Gazebo 11
- MAVROS
- `libgazebo_ros_diff_drive.so`, `libgazebo_ros_gpu_laser.so`, `libgazebo_ros_p3d.so`

---

## Acknowledgments

Architecture and coding style inspired by
[SPOMP](https://github.com/KumarRobotics/spomp-system),
[ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3), and
[FUEL](https://github.com/HKUST-Aerial-Robotics/FUEL).
