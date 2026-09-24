# C-ADMM Drone Dispatcher

Decentralised waypoint assignment for aerial relay drones supporting a
ground robot's navigation through unknown terrain.

## Problem Statement

A team of *N* drones must choose observation waypoints that maximise the
quality of the topological graph used by a ground robot for path planning.
Each drone runs a **local** optimisation — no central coordinator.

**Objective:** minimise the sum of local costs, where each drone's cost is
the *negative* marginal path-cost reduction its observation would produce.

**Constraints:**

- **Communication connectivity:** every pair of neighbouring drones must
  remain within wireless range `D_max` of each other to relay data.
- **Decentralised:** drones only exchange waypoints with direct neighbours.

## Decision Variables

Each drone *i* selects a 2-D waypoint `w_i = (x, y)` (height is fixed).

## Local Cost Function

```
f_i(w_i) = -(PathCost(G, s, g) - PathCost(G'(w_i), s, g))
         = -DeltaJ(w_i)
```

- `G` — current topological graph.
- `G'(w_i)` — predicted graph after drone *i* observes at `w_i`
  (observation counts of nodes within the FOV radius are incremented).
- `PathCost(G, s, g)` — A* shortest-path cost from ground robot start `s`
  to goal `g`, with edge cost = `distance / max(confidence, epsilon)`.
- Lower `f_i` means the waypoint produces a larger path-cost improvement.

### Frontier Proxy Cost

When `PathCost(G, s, g) = inf` (graph does not connect to the goal), the
cost function switches to a **frontier proxy**:

```
proxy(G) = PathCost(G, s, f*) + (1/eps) * ||f* - g||
```

where `f* = argmin_n [ PathCost(s, n) + (1/eps) * ||n - g|| ]` is the
**best frontier node** — the graph node that minimises the combined cost of
reaching it through explored terrain plus the optimistic remaining distance
to the goal.

The same proxy is computed for the predicted graph `G'(w_i)`.  If an
observation at `w_i` extends the graph frontier closer to the goal, the
proxy decreases, producing a positive `DeltaJ`.

## Coupling Constraints — Relay Chain Topology

Drones form a **relay chain** ordered by distance to the ground robot.
Communication constraints are enforced only between **adjacent** chain
members, not between all pairs.  This allows the fleet to extend
exploration range to `N * comm_range` instead of clustering within a
single `comm_range` bubble.

### Chain ordering

All known drones are sorted by Euclidean distance to the ground robot
(ties broken by drone ID string for determinism).  A **cooldown timer**
(default 3 s) prevents rank flapping when drones are equidistant.

```
Robot <-- Drone rank 0 <-- Drone rank 1 <-- ... <-- Drone rank N-1
```

### Constraints per rank

| Rank | ADMM constraint with |
|---|---|
| 0 | Ground robot (hard: candidates outside `D_max` filtered) + rank 1 |
| k | rank k-1 + rank k+1 |
| N-1 | rank N-2 only |

For every **adjacent** pair *(i, j)* in the chain:

```
D_min <= ||w_i - w_j|| <= D_max
```

- `D_max` (comm_range, default 50 m) — maximum distance to maintain relay.
- `D_min` (min_separation, default 15 m) — minimum distance for safety.

The max-distance bound is enforced via the ADMM consensus projection.
The min-distance bound is enforced via a quadratic repulsion penalty.
Repulsion (safety) is applied to **all** drones, not just chain neighbors.

### Decentralised properties

- Any drone can join or leave; the chain re-forms automatically.
- Each drone computes the same sort order from shared position data.
- An isolated drone (lost relay) falls back to rank-0 behaviour and
  returns toward the ground robot.

## C-ADMM Algorithm

Each drone repeats the following for *K* iterations per planning cycle:

### 1. Local optimisation (x-update)

Pick the candidate waypoint that minimises the augmented Lagrangian:

```
w_i* = argmin_{w in C_i}  f_i(w)
          + sum_j [ u_ij^T (w - z_ij)  +  (rho/2) ||w - z_ij||^2 ]
```

where `C_i` is the set of candidate waypoints generated from graph
topology (low-confidence edges, frontiers, gap midpoints, fallback line).

### 2. Consensus variable (z-update)

For each neighbour *j*, the auxiliary variable `z_ij` is the average of the
two waypoints, projected to satisfy the communication constraint:

```
z_ij = (w_i + w_j) / 2
```

(with a projection step that clips `||w_i - w_j||` to `D_max`).

### 3. Dual variable update (u-update)

```
u_ij <- u_ij + (w_i - z_ij)
```

### 4. Waypoint exchange

Drones publish their chosen `w_i` on the `/proposed_waypoint` ROS topic
and read neighbours' waypoints for the next iteration.

## Candidate Generation

Candidates are drawn from five sources (ranked by priority):

1. **Path-edge midpoints** — midpoints of planned-path segments whose graph
   edges have low confidence.
2. **Frontier nodes** — graph nodes with low observation counts, scored by
   progress toward the goal.
3. **Gap midpoints** — centroids between disconnected graph components.
4. **Straight-line fallback** — evenly spaced points along the
   robot-to-goal line (used when no graph exists yet).
5. **Frontier extension points** — waypoints placed beyond the current best
   frontier node toward the goal, spaced by `0.7 * FOV radius` so each
   observation overlaps the existing frontier.  These are the candidates
   that grow the graph toward the goal when it is disconnected.

## Convergence

With a fixed candidate set and convex coupling constraints, C-ADMM
converges to a consensus-feasible solution.  In practice, *K = 5*
iterations per planning cycle (at 10 Hz) provide sufficient convergence
for real-time operation.

## Ground Robot Frontier Behaviour

When the topological graph does not reach the goal, the ground robot
**stops at the best frontier node** instead of driving blind through
unexplored terrain.

- The ground planner (`global_planner.py`) only appends `goal_pos` to the
  path when the nearest graph node to the goal is within
  `frontier_threshold` (default 30 m).
- Otherwise the path ends at the last graph node and the planner sets
  `waiting_at_frontier = True`.
- Each replan cycle, if drones have extended the graph, the new path will
  include the newly explored portion and the robot resumes.
- For disconnected graphs, the same rule applies to the exit of the last
  component in the stitched path.

The best frontier node `f*` is selected by `find_best_frontier()` in
`graph_structure.py`, minimising `path_cost(s, f*) + (1/eps) * dist(f*, g)`.

## Implementation Files

| File | Role |
|---|---|
| `graph_structure.py` | `path_cost()`, `predict_observation()`, `find_best_frontier()` |
| `drone_navigator.py` | `DroneNavigator` with C-ADMM logic + frontier proxy cost |
| `ros_drone_node.py`  | Waypoint exchange topics, real positions |
| `drone_main.py`      | Passes `neighbor_waypoints` to navigator |
| `global_planner.py`  | Frontier-aware path planning, `waiting_at_frontier` flag |
