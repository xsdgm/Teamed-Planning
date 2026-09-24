# Drone Testing Suite

This directory contains test scripts for the drone control, perception, and communication system.

## Test Files

### 1. `test_drone_basic.py` - Basic Movement Test
Tests basic drone movement and control in the simulator without perception/mapping.

**Features:**
- Simple patrol pattern (square waypoints)
- Basic velocity control
- Pygame visualization
- Forest obstacle scene

**Run:**
```bash
cd /path/to/teamed-planning/scripts
python test/test_drone_basic.py
```

**Expected Output:**
- Pygame window showing drone (quadcopter) moving in square pattern
- Console output showing drone position every 100 steps
- Green forest obstacles on yellow background

---

### 2. `test_drone_perception.py` - Perception and Mapping Test
Tests the complete perception pipeline: semantic segmentation → skeleton extraction → topological graph building.

**Features:**
- Generates semantic maps from simulator obstacles
- Builds topological graphs using `DroneController`
- Visualizes graph on pygame window:
  - **Red nodes** (graph vertices)
  - **Blue edges** (graph connections)
  - Semi-transparent yellow circle (coverage area)
- Prints graph statistics (nodes, edges, coverage bounds)

**Run:**
```bash
cd /path/to/teamed-planning/scripts
python test/test_drone_perception.py
```

**Expected Output:**
- Pygame window with drone and obstacles
- Red nodes and blue edges overlaid on the map
- Console output showing graph building progress
- Final statistics: node count, edge count, coverage bounds

**Key Parameters:**
- `map_resolution = 0.5` (0.5 meters per pixel)
- `map_size_pixels = 200` (200x200 pixels = 100m x 100m coverage)
- `graph_update_interval = 50` (update graph every 50 steps)

---

### 3. `test_drone_broadcast.py` - Communication Test

Tests graph broadcasting from drone to ground robots.

**Features:**
- Drone builds topological graph
- Broadcasts graph to nearby ground robots (within 100m range)
- Ground robots receive and count messages
- Message broker handles communication
- Visualizes graph on pygame

**Run:**
```bash
cd /path/to/teamed-planning/scripts
python test/test_multi_drone_perception.py
```

**Expected Output:**
- Pygame window showing 3 drones with color-coded graphs
- Console output every 100 steps showing per-drone statistics:
  - Position
  - Visible obstacles
  - Graph nodes and edges
- Graphs overlap visually when drones explore common areas

**Key Parameters:**
- World: 1000m x 1000m
- Obstacles: 100 circles + 60 rectangles (160 total)
- Perception range: 100m diameter (50m radius)
- Patrol range: 400m radius per drone

---

### 4. `test_multi_drone.py` - Multi-Drone Perception & Fusion ⭐ NEW
Unified test supporting two modes: fast perception test and comprehensive fusion test.

#### Mode 1: Perception (Fast Test)
**Features:**
- 3 drones patrolling 1km x 1km world independently
- No map fusion (faster, simpler test)
- Color-coded graphs (Red, Blue, Magenta)
- ~160 obstacles
- Perception range: 100m diameter

**Run:**
```bash
cd /path/to/teamed-planning/scripts
python test/test_multi_drone.py perception
```

**Key Parameters:**
- World: 1000m x 1000m
- Perception range: 100m diameter (50m radius)
- Patrol range: 400m radius per drone
- Simulation steps: 1000
- Print interval: Every 100 steps

---

#### Mode 2: Fusion (Comprehensive Test)
**Features:**
- **Large-scale world**: 5000m x 5000m (5km x 5km)
- **Small perception range**: 20m diameter (10m radius)
- **Map fusion**: Drones automatically fuse maps when within 30m
- **Fusion cooldown**: 30-second interval between fusions
- **Hybrid mapping**: 15m grid + skeleton critical points
- **Performance timing**: Real-time monitoring of all modules
- ~3750 obstacles

**Run:**
```bash
cd /path/to/teamed-planning/scripts
python test/test_multi_drone.py fusion
```

**Expected Output:**
- Pygame window with color-coded graphs
- Console output every 2 seconds:
  - Position, graph size (nodes, edges)
  - Fusion history (which drones, when)
  - Performance timing (semantic processing, fusion, broadcasting)
- Final detailed statistics

**Key Parameters:**
- World: 5000m x 5000m (5km)
- Perception range: 20m diameter
- Communication range: 30m (for fusion)
- Fusion interval: 30 seconds
- Grid spacing: 15m
- Max edge distance: 30m
- Patrol range: 800m radius per drone
- Simulation steps: 2000

**Fusion Mechanism:**
1. Drones detect each other within 30m range
2. If 30+ seconds passed since last fusion, trigger fusion
3. Node merging within 3m:
   - Update observation count and confidence
   - Position = weighted average based on confidence
4. New nodes from other drone added if >3m away
5. Edges merged without duplication

**Hybrid Mapping Strategy:**
1. Generate 15m grid points in traversable areas
2. Extract skeleton critical points (junctions, endpoints)
3. Merge points: keep skeleton when within 7.5m of grid
4. Connect each node to 4 nearest neighbors (max 30m)
5. Preserve skeleton connectivity

---

## Dependencies

All tests require:
- `numpy`
- `pygame`
- `scipy`
- `scikit-image`
- `networkx`

Install with:
```bash
pip install numpy pygame scipy scikit-image networkx
```

---

## Visualization

### Graph Visualization Colors
- **Nodes**: Red circles (radius 4px)
- **Edges**: Blue lines (width 2px)
- **Coverage area**: Semi-transparent yellow circle

### Interactive Controls
- **Mouse Wheel**: Zoom in/out
- **Left Click + Drag**: Pan view
- **R**: Reset view
- **G**: Toggle grid
- **L**: Toggle labels
- **V**: Toggle velocity arrows
- **F**: Toggle FPS counter

---

## Troubleshooting

### Import Errors
If you see import errors, make sure you're running from the `scripts/` directory:
```bash
cd /path/to/teamed-planning/scripts
python test/test_drone_perception.py
```

### No Graph Displayed
- Check that obstacles are within drone's coverage area (100m x 100m by default)
- Verify `graph_update_interval` is being reached
- Check console output for graph building messages

### Pygame Window Not Appearing
- Ensure pygame is installed: `pip install pygame`
- Check for display/graphics driver issues

---

## Next Steps

After successful drone tests, proceed to:
1. Ground robot control and perception tests
2. Multi-agent coordination tests
3. Full system integration tests

---

## File Structure

```
test/
├── README.md                    # This file
├── test_drone_basic.py          # Basic movement test
├── test_drone_perception.py     # Perception and mapping test
└── test_drone_broadcast.py      # Communication test
```

---

## Notes

- All tests use a 500m x 500m world
- Semantic maps are generated from simulator obstacles on-the-fly
- Graph visualization is overlaid on the pygame renderer
- Tests are designed to be run independently

