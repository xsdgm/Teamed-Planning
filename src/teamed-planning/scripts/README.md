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
│   └── timing.py                   # Controller performance measurements
│
├── drone/                           # Drone-side code
│   ├── __init__.py                  # Public controller/perception exports
│   ├── drone_main.py                # Controller and graph broadcasts
│   ├── drone_navigator.py           # CADMM waypoint navigation
│   ├── drone_local_planner.py       # ROS/PX4 waypoint tracking
│   ├── semantic_segmentation.py     # Semantic map processing and inference
│   ├── local_mapping.py             # Mapping views and temporal EvidenceGrid
│   ├── skeleton_graph_builder.py    # Build topology from semantic maps
│   ├── communication.py             # Chain state, local backbone, relay cache
│   ├── cbf_safety.py                # Independent and synchronous joint CBF
│   ├── mappo.py                     # Checkpoint loading and candidate baselines
│   ├── velocity_mappo.py            # Shared velocity encoding and control
│   ├── graph_attention_mappo.py     # Shared map graph attention
│   ├── cooperative_mappo.py         # Shared cooperative encoders and navigation
│   ├── relay_attention_mappo.py     # Relay attention used by v9 and later policies
│   ├── survey_mappo.py              # V10 survey and V11 fixed-route policies
│   ├── ros_drone_node.py            # ROS flight/control entry point
│   ├── ros_seg_node.py              # ROS segmentation entry point
│   └── models/download_models.sh    # Segmentation model preparation
│
├── ground/                          # Ground robot code
│   ├── __init__.py                  # Public controller/planning exports
│   ├── ground_main.py               # Controller, graph reception and navigation
│   ├── graph_fusion.py              # Fuse received drone graphs
│   ├── global_planner.py            # Topological path and graph-entry planning
│   ├── local_planner.py             # Local perception, avoidance and velocity control
│   ├── observed_grid.py             # Observed route memory and continuous escape checks
│   ├── subgoal_manager.py           # vtemp1 directional lookahead guidance
│   ├── vtemp.py                     # Original vtemp rolling200 waypoint selector
│   ├── vtemp4_*.py                  # vtemp4 ground observations and policies
│   ├── ros_ground_node.py           # ROS ground-controller entry point
│   └── ros_gcs_node.py              # ROS ground-control station
│
├── simulator/                       # Simulation, map backends and experiment runners
│   └── *vtemp4*.py                  # vtemp4 runtime, curriculum, training and evaluation
├── world_model/                     # Shared navigation and model implementations
│   ├── train_slow_policy.py         # VTemp2 training with frozen V9 UAV MAPPO
│   ├── train_vtemp3.py              # VTemp3 training with a graph world model
│   ├── navigation/                  # Execution, PPO policies, safety and graph dynamics
│   ├── rssm/                        # Single-target recurrent model, data and planning
│   ├── terrain/                     # Terrain prediction and decoupled planning models
│   └── survey/                      # A/B survey environment, models and planning
├── tests/                          # Tests grouped by their implementation package
│   └── world_model/                # Includes test_hierarchical_*.py
└── example_integration.py           # Integration example

```

### Drone module consolidation

Shared drone implementations are grouped by
responsibility; policy variants remain separate where their architectures differ.

| Former module | Current implementation |
| --- | --- |
| `chain_state.py`, `local_backbone.py`, `relay_cache.py` | [communication.py](drone/communication.py) |
| `coordinated_safety.py` | [cbf_safety.py](drone/cbf_safety.py), alongside the independent filter |
| `evidence_grid.py` | [local_mapping.py](drone/local_mapping.py) |
| `route_ab_mappo.py` | [survey_mappo.py](drone/survey_mappo.py), alongside its survey-policy base |
| `graph_broadcaster.py` | Removed: the controller already sends dictionary broadcasts directly |

`GraphBroadcaster` and `RobotDetector` were unused and are no longer package
exports. [DroneController](drone/drone_main.py) owns graph transmission and reports
the actual last graph-broadcast time in `get_state()`. State-dictionary checkpoint
schemas, policy parameters and ROS launch entry points are unchanged.
Archived experiment runtimes retain their original source layout; workers that
explicitly load those snapshots continue to use their historical imports.

### Ground module consolidation

Ground routing, lookahead guidance and the two ROS entry points retain their existing boundaries.

| Former module | Current implementation |
| --- | --- |
| `local_perception.py` | [local_planner.py](ground/local_planner.py), alongside local avoidance |
| `observed_escape.py` | [observed_grid.py](ground/observed_grid.py), alongside observed route memory |
| `graph_receiver.py` | Removed: [GroundRobotController](ground/ground_main.py) already receives and deserializes dictionary broadcasts directly |

`LocalPerception` remains available from `ground` and `ground.local_planner`.
The unused `GraphReceiver` export and controller instance are removed; actual
reception statistics, fusion throttling, reset behavior and navigation are
unchanged. The current candidate evaluator copies the merged modules, while
archived evaluation modes keep their original dependency layout.

### VTemp4 navigation

VTemp4 is a continuous navigation policy using frozen V9 drone control and the
ground local planner. Ground observations and policies live in `ground/`; its
simulation runtime, curricula, training and evaluation live in `simulator/`.
Corresponding tests are under `tests/ground/` and `tests/simulator/`. The
[experiment guide](../../../experiments/world_model/vtemp4/README.md)
describes the training and evaluation workflow. Historical `world_model.vtemp4_*`
module imports remain available for loading older checkpoints.

### VTemp2 and VTemp3 hierarchical training

`world_model/train_slow_policy.py` uses the archived V9 UAV MAPPO checkpoint for the fast layer.
Only the UGV's slow policy is updated, using duration-aware PPO. The slow
layer chooses subgoals from received map evidence. The V9 policy, local
planners and safety filters execute those subgoals. VTemp2 uses no world model.

From the repository root, run a small CPU training and validation pass with the
project's Python environment.

```bash
python -I -B src/teamed-planning/scripts/world_model/train_slow_policy.py \
  --output artifacts/vtemp2-slow-ppo-new --updates 1 \
  --episodes-per-update 2 --workers 2 --eval-episodes 1 \
  --rollout-steps 30 --option-steps 10
```

Use `--resume` with the original output directory to continue a run. Pass
`--evaluate artifacts/vtemp2-slow-ppo-new/best.pt` with a new output directory
to evaluate on independent synthetic tasks.

`world_model/train_vtemp3.py` trains the graph world model and its
prediction-conditioned policies. See the [world model directory guide](world_model/README.md).

| Responsibility | Directory under `world_model/` | Modules |
| --- | --- | --- |
| Dynamics and received telemetry | `navigation/` | `graph_dynamics.py`, `residual_world.py`, `telemetry.py` |
| Prediction-conditioned policies | `navigation/` | `graph_fast_policy.py`, `hierarchical_policy.py`, `belief_actor.py` |
| Observed candidate PPO | `navigation/` | `slow_policy.py` |
| Hierarchical execution and planning inputs | `navigation/` | `hierarchical_runtime.py`, `planning_graph.py`, `navigation_evidence.py` |
| Graph rollout collection | `navigation/` | `graph_rollouts.py`, `belief_rollouts.py`, `safety_context.py` |
| Local safety | `navigation/` | `hierarchical_cbf.py`, `hierarchical_safety.py` |
| Navigation scenes and training schedule | `navigation/` | `scene_manifest.py`, `training_schedule.py` |
| Single-target RSSM, mission collection and planning | `rssm/` | `model.py`, `data.py`, `mission.py`, `planner.py`, `runtime.py` |
| Scene manifests and WildScenes validation | `navigation/` | `scene_manifest.py`, `wildscenes_runtime.py`, `wildscenes_tasks.py` |
| Legacy training commands | Package root | `train_hierarchical.py` (forwards to `train_vtemp3.py`), `train_joint_policy.py` |
| Evaluation and archived runtime loading | Package root | `evaluate_safety_repairs.py`, `evaluate_slow_policy.py`, `evaluate_fixed_target.py`, `frozen_navigation.py` |

VTemp3 runs copy their selected modules directly into
`runtime_sources/world_model/`. For example, the frozen copy of
`navigation/graph_rollouts.py` is `runtime_sources/world_model/graph_rollouts.py`.
Older runs can also use `runtime_sources/vtemp2/`. Checkpoints and evidence
retain their historical `vtemp2` schema identifiers. Hierarchical regression tests live in
`tests/world_model/test_hierarchical_*.py`.

For VTemp3, `--fresh-all` initializes the fast policy,
world model and slow policy from fresh constructors, with new optimizer state,
zero training counters and empty replay. It loads no pretrained policy weights
and uses no learned teacher. The archived simulator architecture remains a
runtime dependency. The manifest records `training_mode=fresh_all_staged`,
initial model hashes, the random seed, source hashes and each stage's optimizer
targets.

The three-stage schedule trains the world model throughout:

| Stage | Tasks | Updates |
| --- | --- | --- |
| Short-distance `fast` | Rule-based UGV subgoals | World-model supervision and fast-policy PPO; slow policy fixed |
| Long-distance `slow` | Learned UGV subgoals | World-model supervision and slow-policy PPO; fast policy fixed |
| Long-distance `joint` | Learned UGV subgoals | World-model supervision and both policy PPO updates |

There is no separate imitation warmup in this mode. Four, sixteen and eight
16-episode batches give 64 short-distance tasks, 256 slow-stage tasks and 128
joint-stage tasks. `--update-plan` embeds the prepared schedule in the run's
configuration; the scene manifest fixes task identities and curriculum before
launch. Checkpoints retain model, optimizer and random-generator state for
later recovery, while per-update metrics and held-out evaluation remain separate.

Example invocation from the repository root, using prepared input files:

```bash
python src/teamed-planning/scripts/world_model/train_vtemp3.py \
  --repo . --output artifacts/hierarchical-fresh-all-new --fresh-all \
  --scene-manifest /path/to/fresh_all_scenes.json \
  --update-plan /path/to/fresh_all_schedule.json \
  --warmup-updates 0 --fast-updates 4 --slow-updates 16 --joint-updates 8 \
  --episodes-per-update 16 --workers 16 --device cpu \
  --warmup-distances 100 200 --mission-distances 1000 1000 --eval-scenes 9
```

For a separate slow-only experiment, `--fresh-slow --graph-init CHECKPOINT
--freeze-fast-world` transfers only fast/world weights and randomizes the slow
policy with fresh optimizer state. That optional mode rejects inherited slow
weights and fast/joint optimization stages; it is distinct from `--fresh-all`.

For independent fast/world training, use `--fresh-fast-world` with zero warmup,
slow and joint updates. It creates all models and optimizer state from scratch,
trains only world supervision and fast PPO, and freezes the unused slow model.
Both training and held-out validation use the geometry UGV subgoal selector;
validation uses deterministic fast actions and never calls the slow policy.
The manifest records `training_mode=fresh_fast_world` and the controller choices.
Geometry/evaluation decisions are marked ineligible for slow PPO. This mode
rejects weight initialization, resume/continuation, other fresh modes and random
UGV selection. The existing three-stage mode still defaults to learned-slow
validation; `--validation-slow-controller geometry` requests an explicit rule
control comparison for other modes.

An 800-task mixed-distance run uses 50 batches of 16, without claiming that this
fixed budget guarantees convergence:

```bash
python src/teamed-planning/scripts/world_model/train_vtemp3.py \
  --repo . --output artifacts/hierarchical-fresh-fast-world-new --fresh-fast-world \
  --scene-manifest /path/to/fast_world_scenes.json \
  --warmup-updates 0 --fast-updates 50 --slow-updates 0 --joint-updates 0 \
  --episodes-per-update 16 --workers 16 --device cpu \
  --warmup-distances 100 400 --mission-distances 100 400 \
  --validation-slow-controller geometry --eval-scenes 9
```

Training task distance is independent of the execution subgoal scale: the
existing approximately 67/133/200 m candidate rings and adaptive option duration
remain in use. Longer missions exercise repeated local commitments.

Run the relocated hierarchical regression suite with the project's Python
environment:

```bash
python -B -m unittest discover \
  -s src/teamed-planning/scripts/tests/world_model \
  -p 'test_hierarchical_*.py'
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
- **Current**: Preserve observed path geometry
  - Only nodes at identical coordinates are coalesced; observation evidence is deduplicated
  - Distinct source endpoints and their edge geometry are retained
  - Nearby components connect only through separately checked, received free-space evidence
  - Proximity alone never authorizes an edge or a change to a certified segment
- **Future**: Probabilistic merging with covariance tracking

### 5. Hierarchical Planning
- **Global Planner**: Uses fused topological graph for waypoint generation
  - A* pathfinding on graph
  - Replanning when robot deviates from path
- **Local Planner**: Reactive obstacle avoidance
  - Uses local semantic map from robot sensors
  - Simple proportional controller for velocity commands

The original **vtemp** controller selects rolling waypoints and switches on approach
or failure. Archived WildScenes runs use this policy. **vtemp1** provides directional
lookahead guidance. A point up to 200 m ahead communicates the vehicle's preferred
heading. The controller reviews it every 15 simulation seconds using the received
road graph and applies a turn penalty when changing direction. [Guidance details](world_model/README.md)

All retained 2D MAPPO versions and vtemp evaluators default to
`ground_safety_distance=1.0` metres for drone navigation graphs, vehicle local
planning and checked fusion bridges. The shared default is defined in
`common/navigation_config.py`; the enclosed-vehicle escape margin remains `0.5`.
WildScenes evaluators accept `--ground-safety-distance` (also `--safety-clearance`)
and record the actual planner, graph and bridge values in their signed results,
including when loading archived runtimes. New reference tasks must have a
continuously verified route at the normal 1 metre clearance; the escape margin
does not qualify an otherwise infeasible task.

ROS nodes and configuration are unchanged. Their shared controller constructor
defaults remain 1.5 metres for the ground vehicle and 3 metres for graph clearance;
2D entry points explicitly pass their own 1 metre setting. Existing frozen runs
retain their original source and configuration. An old training checkpoint without
the field is interpreted as historical 1.5 metres for resume compatibility, so
resuming it silently under 1 metre is rejected; use a new run to change the setting.

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
# - 'broadcast': dictionary or None
```

#### Ground Robot Integration

```python
from ground.ground_main import GroundRobotController

# Initialize robot
robot = GroundRobotController(robot_id="robot_0")

# Set goal
robot.set_goal((goal_x, goal_y))

# In your simulation loop:
observation = {
    'self': {'position': (x, y), 'orientation': theta},
    'occupancy_map': occupancy_map,  # Binary local sensor grid: 0=free, 1=occupied
    'map_resolution': 0.5,  # metres per pixel
    'neighbors': neighbors,  # Received neighbours, including drone broadcast dictionaries
    'full_state': {'time': simulation_time},
    'time_step': dt,
}

action = robot.get_action(observation)
# action contains:
# - 'motion': (linear_vel, angular_vel)
# - 'broadcast': robot position, current target and remaining path
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
GroundRobotController(
    robot_id="robot_0",
    merge_threshold=2.0,  # meters (for graph fusion)
    max_linear_vel=1.0,  # m/s
    max_angular_vel=1.0  # rad/s
)
```

## Testing

Simulator regressions live in [tests/simulator](tests/simulator), alongside
[ground tests](tests/ground), [drone tests](tests/drone) and
[world-model tests](tests/world_model). See the
[simulator test commands](simulator/README.md#regression-tests) for the regular
suite and the separately executed frozen-runtime checks.

## Communication Protocol

Controllers exchange dictionaries through the simulator's received neighbour
broadcasts. Drone packets contain `drone_id`, `position`, `graph` and `timestamp`;
ground packets contain `robot_id`, `position`, `target`, `planned_path` and
`timestamp`. The ground controller deserializes received graphs and applies its
per-drone fusion interval directly.

[ros_drone_node.py](drone/ros_drone_node.py) and
[ros_ground_node.py](ground/ros_ground_node.py) adapt these interfaces to ROS
topics.

## Troubleshooting

### Graphs not being received
- Check broadcast range (default 100m)
- Verify robots are within drone's detection range
- Check that received drone neighbours carry a broadcast dictionary with graph data

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
Covariance-based fusion is a future extension of [graph_fusion.py](ground/graph_fusion.py).

When adding uncertainty:
1. Modify `TopologicalGraph` to store node covariances
2. Update `GraphFusion` with probabilistic merging
3. Integrate FAST-LIO2 pose covariances
4. Implement loop closure and graph optimization

### ROS Integration
Replace communication layer:
```python
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
