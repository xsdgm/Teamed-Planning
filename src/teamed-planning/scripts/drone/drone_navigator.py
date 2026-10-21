"""
Drone Navigator — C-ADMM Distributed Dispatcher

Each drone independently solves a local optimisation problem to choose the
observation waypoint that maximises marginal path-cost reduction for the
ground robot, while respecting communication connectivity constraints.

Drones negotiate via Consensus-ADMM: they exchange proposed waypoints with
neighbours and iteratively converge to a jointly feasible, high-quality
assignment.  See DISPATCHER.md for the full theoretical derivation.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
from enum import Enum


class NavigationPhase(Enum):
    DISCOVERING  = "discovering"    # waiting to find ground robot mission
    DISPATCHING  = "dispatching"    # C-ADMM optimisation loop


class DroneNavigator:
    """
    Decentralised C-ADMM dispatcher for a single drone.

    Public API
    ----------
    compute_next_target(current_position, observation_data) -> (x,y,z) | None
    get_stats()  -> dict
    reset()
    """

    def __init__(self,
                 drone_id: str,
                 total_drones: int = 1,
                 waypoint_threshold: float = 3.0,
                 comm_range: float = 50.0,
                 min_separation: float = 15.0,
                 fov_radius: float = 30.0,
                 admm_iterations: int = 5,
                 rho: float = 1.0,
                 separation_weight: float = 5.0):
        self.drone_id           = drone_id
        self.waypoint_threshold = waypoint_threshold
        self.comm_range         = comm_range
        self.min_separation     = min_separation
        self.fov_radius         = fov_radius
        self._fleet_total       = max(1, int(total_drones))
        self._admm_K            = admm_iterations
        self._rho               = rho
        self._sep_weight        = separation_weight

        self.phase = NavigationPhase.DISCOVERING

        # Ground robot state
        self.ground_robot_position: Optional[Tuple[float, float]] = None
        self.ground_robot_start:    Optional[Tuple[float, float]] = None
        self.ground_robot_target:   Optional[Tuple[float, float]] = None
        self._planned_path: List[Tuple[float, float]] = []

        # ADMM dual variables  {neighbor_drone_id: np.array([ux, uy])}
        self._dual_vars: Dict[str, np.ndarray] = {}

        # Chain rank state (cooldown prevents rank flapping)
        self._chain_rank: int = 0
        self._chain_order: List[str] = []          # drone IDs sorted by dist to robot
        self._chain_neighbors: Dict[str, tuple] = {}  # adjacent chain neighbors
        self._rank_cooldown: float = 3.0            # seconds before re-sorting
        self._last_rank_time: float = 0.0

        # Cached output
        self._current_target: Optional[Tuple[float, float, float]] = None

        # Stats
        self.waypoints_reached: int = 0
        self.assigned_segment:  Optional[int] = None
        self.total_drones_discovered: int = 0
        self.segment_start_position: Optional[Tuple] = None

    # ── Mission discovery ──────────────────────────────────────────────────

    def discover_mission(self, observation_data: Dict,
                         flight_height: float = 10.0) -> bool:
        ground_robots = observation_data.get('ground_robots', {})
        for robot_id, info in ground_robots.items():
            target   = info.get('target')
            position = info.get('position')
            if target is not None and position is not None:
                self.ground_robot_start    = tuple(position[:2])
                self.ground_robot_target   = tuple(target[:2])
                self.ground_robot_position = tuple(position[:2])
                self.total_drones_discovered = self._fleet_total
                print(f"[{self.drone_id}] Mission: "
                      f"start={self.ground_robot_start} "
                      f"target={self.ground_robot_target}")
                self.phase = NavigationPhase.DISPATCHING
                return True
        return False

    # ── Candidate generation ───────────────────────────────────────────────

    def _generate_candidates(self,
                             graph,
                             planned_path: List[Tuple[float, float]],
                             goal: Tuple[float, float],
                             current_pos: Tuple[float, float],
                             flight_height: float,
                             max_candidates: int = 20,
                             frontier=None,
                             ) -> List[Tuple[float, float, float]]:
        """Produce candidate waypoints from the graph topology.

        Sources:
          1. Midpoints of low-confidence edges along the planned path
          2. Frontier graph nodes toward the goal
          3. Midpoints of gaps between disconnected components
          4. Points along the straight line to the goal (fallback)
        """
        candidates: List[Tuple[float, float, float]] = []
        goal_arr = np.array(goal)
        cur_arr  = np.array(current_pos[:2])

        # --- 1. Low-confidence edge midpoints on the planned path ----------
        if len(planned_path) >= 2 and graph is not None:
            for i in range(len(planned_path) - 1):
                p1 = np.array(planned_path[i])
                p2 = np.array(planned_path[i + 1])
                mid = (p1 + p2) / 2.0
                candidates.append((float(mid[0]), float(mid[1]), flight_height))

        # --- 2. Frontier nodes (nodes with low observation count) ----------
        if graph is not None and graph.graph.number_of_nodes() > 0:
            scored = []
            for nid, data in graph.graph.nodes(data=True):
                obs = data.get('observation_count', 1)
                pos = np.array(data['position'])
                goal_progress = float(np.dot(pos - cur_arr, goal_arr - cur_arr))
                score = goal_progress / max(obs, 1)
                scored.append((nid, score, pos))
            scored.sort(key=lambda x: -x[1])
            for _, _, pos in scored[:5]:
                candidates.append((float(pos[0]), float(pos[1]), flight_height))

        # --- 3. Gap midpoints between disconnected components --------------
        if graph is not None and graph.graph.number_of_nodes() > 0:
            # Sorted members: set order follows string hash randomization, and
            # a different summation order changes centroids in the last bits.
            components = [sorted(c, key=str) for c in nx.connected_components(graph.graph)]
            if len(components) > 1:
                for ci in range(len(components)):
                    for cj in range(ci + 1, len(components)):
                        nodes_i = [graph.graph.nodes[n]['position']
                                   for n in components[ci]]
                        nodes_j = [graph.graph.nodes[n]['position']
                                   for n in components[cj]]
                        centroid_i = np.mean(nodes_i, axis=0)
                        centroid_j = np.mean(nodes_j, axis=0)
                        mid = (centroid_i + centroid_j) / 2.0
                        candidates.append(
                            (float(mid[0]), float(mid[1]), flight_height))

        # --- 4. Straight-line fallback points toward goal ------------------
        if self.ground_robot_position is not None:
            robot_pos = np.array(self.ground_robot_position)
            direction = goal_arr - robot_pos
            dist = float(np.linalg.norm(direction))
            if dist > 1.0:
                direction = direction / dist
                for frac in [0.15, 0.35, 0.55, 0.75, 0.95]:
                    pt = robot_pos + direction * dist * frac
                    candidates.append(
                        (float(pt[0]), float(pt[1]), flight_height))

        # --- 5. Frontier extension points toward goal ---------------------
        # When the graph doesn't reach the goal, place candidates just
        # beyond the current frontier so observing there extends the graph.
        if graph is not None and graph.graph.number_of_nodes() > 0:
            if frontier is None:
                frontier = graph.find_best_frontier(current_pos[:2], goal)
            if frontier is not None:
                _fid, fpos = frontier
                fpos_arr = np.array(fpos)
                extend_dir = goal_arr - fpos_arr
                extend_dist = float(np.linalg.norm(extend_dir))
                if extend_dist > 1.0:
                    extend_dir = extend_dir / extend_dist
                    step = self.fov_radius * 0.7
                    for k in range(1, 4):
                        pt = fpos_arr + extend_dir * step * k
                        if float(np.linalg.norm(pt - goal_arr)) < extend_dist:
                            candidates.insert(0,
                                (float(pt[0]), float(pt[1]), flight_height))

        # Deduplicate (keep candidates further than 2m apart)
        unique: List[Tuple[float, float, float]] = []
        for c in candidates:
            c_arr = np.array(c[:2])
            if all(np.linalg.norm(c_arr - np.array(u[:2])) > 2.0
                   for u in unique):
                unique.append(c)
            if len(unique) >= max_candidates:
                break
        return unique

    # ── Local cost function ────────────────────────────────────────────────

    def _local_cost(self,
                    w: Tuple[float, float, float],
                    graph,
                    start: Tuple[float, float],
                    goal: Tuple[float, float],
                    cost_before: Optional[float] = None,
                    frontier=None, predicted_cost=None) -> float:
        """f_i(w) = -DeltaJ(w) = -(cost_before - cost_after).

        Lower is better (we minimise f_i).

        When ``path_cost`` returns *inf* (graph doesn't connect to the
        goal), a **frontier proxy cost** is used instead::

            proxy = path_cost(start, frontier) + (1/eps) * dist(frontier, goal)

        This gives drones a meaningful gradient toward extending the
        graph frontier.
        """
        if graph is None or graph.graph.number_of_nodes() == 0:
            return 0.0

        epsilon = 0.05
        off_graph_penalty = 1.0 / epsilon

        if cost_before is None:
            cost_before = self._baseline_cost(graph, start, goal, epsilon)

        if frontier is None:
            frontier = graph.find_best_frontier(start, goal, epsilon=epsilon)
        # For a distant, unseen goal the topology outside the frontier does
        # not exist yet. Evaluate its optimistic footprint extension directly
        # instead of cloning/searching the identical graph per candidate.
        if frontier is not None:
            fpos = np.asarray(frontier[1], dtype=float)
            ray = np.asarray(goal, dtype=float) - fpos
            remaining = float(np.linalg.norm(ray))
            if remaining > self.fov_radius:
                ray /= remaining
                relative = np.asarray(w[:2]) - fpos
                along = float(np.dot(relative, ray))
                perpendicular = float(np.linalg.norm(relative - along * ray))
                gain = 0.
                if perpendicular < self.fov_radius:
                    span = float(np.sqrt(self.fov_radius ** 2 - perpendicular ** 2))
                    if along - span <= 0 <= along + span:
                        gain = (off_graph_penalty - 1.) * min(remaining, max(0., along + span))
                return -gain

        if predicted_cost is None:
            predicted = graph.predict_observation(
                (w[0], w[1]), self.fov_radius)
            cost_after = predicted.path_cost(start, goal, epsilon=epsilon)
            if cost_after == float('inf'):
                cost_after = self._frontier_proxy(predicted, start, goal,
                                                  epsilon, off_graph_penalty)
        else:
            cost_after = predicted_cost

        # Predict the extension of an *observed* frontier. Count-only copies
        # cannot add topology in unseen space, so they otherwise assign zero
        # gain to every new observation. This optimistic extension is used
        # only for choosing aerial observations, never as a ground map edge.
        if frontier is not None:
            fpos = np.asarray(frontier[1], dtype=float)
            direction = np.asarray(goal, dtype=float) - fpos
            remaining = float(np.linalg.norm(direction))
            if remaining > 0:
                direction /= remaining
                relative = np.asarray(w[:2]) - fpos
                along = float(np.dot(relative, direction))
                perpendicular = float(np.linalg.norm(relative - along * direction))
                if perpendicular < self.fov_radius:
                    half_span = float(np.sqrt(self.fov_radius ** 2 - perpendicular ** 2))
                    if along - half_span <= 0 <= along + half_span:
                        extension = max(0., min(remaining, along + half_span))
                        cost_after = min(cost_after,
                                         cost_before - (off_graph_penalty - 1.) * extension)

        delta_j = cost_before - cost_after
        return -delta_j

    def _baseline_cost(self, graph, start, goal, epsilon=0.05):
        """The unobserved graph's cost is identical for every candidate."""
        if graph is None or graph.graph.number_of_nodes() == 0:
            return 0.0
        cost = graph.path_cost(start, goal, epsilon=epsilon)
        if cost == float('inf'):
            cost = self._frontier_proxy(graph, start, goal, epsilon, 1.0 / epsilon)
        return cost

    @staticmethod
    def _frontier_proxy(graph,
                        start: Tuple[float, float],
                        goal: Tuple[float, float],
                        epsilon: float,
                        off_graph_penalty: float) -> float:
        """Compute a finite proxy cost when the graph is disconnected from
        the goal: path_cost(start, frontier) + penalty * dist(frontier, goal).
        """
        result = graph.find_best_frontier(start, goal, epsilon=epsilon)
        if result is None:
            return off_graph_penalty * float(
                np.linalg.norm(np.array(goal) - np.array(start)))
        _fid, fpos = result
        graph_cost = graph.path_cost(start, fpos, epsilon=epsilon)
        if graph_cost == float('inf'):
            graph_cost = 0.0
        remaining = float(np.linalg.norm(
            np.array(goal) - np.array(fpos)))
        return graph_cost + off_graph_penalty * remaining

    # ── C-ADMM step ───────────────────────────────────────────────────────

    def _admm_select(self,
                     candidates: List[Tuple[float, float, float]],
                     graph,
                     start: Tuple[float, float],
                     goal: Tuple[float, float],
                     neighbor_waypoints: Dict[str, Tuple[float, float, float]],
                     drone_positions: Dict[str, tuple],
                     current_pos: Tuple[float, float, float],
                     frontier=None,
                     fixed_candidates=False,
                     ) -> Optional[Tuple[float, float, float]]:
        """Run K C-ADMM iterations over the candidate set.

        Communication constraints are enforced only between **adjacent
        chain neighbors** (relay chain ordered by distance to the ground
        robot).  Repulsion is still applied to ALL drones for safety.

        For the rank-0 drone, candidates outside ``comm_range`` of the
        ground robot are filtered out (hard constraint).

        Returns the best waypoint or None.
        """
        if not candidates:
            return None

        # ── Compute relay chain and identify chain neighbors ──────────
        chain_nbrs = self._compute_chain_neighbors(drone_positions,
                                                   current_pos)

        # Build chain-neighbor waypoints: use proposed waypoint if
        # available, otherwise fall back to the neighbor's current
        # position.  The ground robot has no "waypoint" — use its
        # position directly.
        chain_wps: Dict[str, np.ndarray] = {}
        for did, pos in chain_nbrs.items():
            if did == '__ground_robot__':
                chain_wps[did] = np.array(pos[:2])
            elif did in neighbor_waypoints:
                chain_wps[did] = np.array(neighbor_waypoints[did][:2])
            else:
                chain_wps[did] = np.array(pos[:2])

        # ── Rank 0 hard constraint: filter candidates ─────────────────
        if not fixed_candidates and self._chain_rank == 0 and self.ground_robot_position is not None:
            robot_arr = np.array(self.ground_robot_position)
            # The radio limit is spatial; the ground endpoint is at z=0.
            horizontal_range = np.sqrt(max(0., self.comm_range ** 2 - current_pos[2] ** 2))
            filtered = [c for c in candidates
                        if np.linalg.norm(
                            np.array(c[:2]) - robot_arr) <= horizontal_range]
            if not filtered:
                closest = min(candidates, key=lambda c: np.linalg.norm(np.array(c[:2]) - robot_arr))
                delta = np.asarray(closest[:2]) - robot_arr
                point = robot_arr + delta * min(1., horizontal_range / max(np.linalg.norm(delta), 1e-9))
                filtered = [(float(point[0]), float(point[1]), closest[2])]
            candidates = filtered

        # ── Dual variable housekeeping (chain neighbors only) ─────────
        for did in chain_wps:
            if did not in self._dual_vars:
                self._dual_vars[did] = np.zeros(2)
        self._dual_vars = {k: v for k, v in self._dual_vars.items()
                           if k in chain_wps}

        # Repulsion refs: ALL drones (safety), not just chain neighbors.
        repulsion_refs: Dict[str, np.ndarray] = {}
        for did, pos in drone_positions.items():
            if did != self.drone_id:
                repulsion_refs[did] = np.array(pos[:2])
        for did, wp in neighbor_waypoints.items():
            if did != self.drone_id:
                repulsion_refs[did] = np.array(wp[:2])

        # Reserve lower-ID proposals first, so symmetric agents cannot keep
        # selecting the same point. Add nearby alternatives before filtering.
        reserved = [np.asarray(wp[:2]) for did, wp in neighbor_waypoints.items()
                    if did < self.drone_id]
        if reserved and not fixed_candidates:
            alternatives = list(candidates)
            for w in candidates[:4]:
                for angle in np.linspace(0., 2. * np.pi, 8, endpoint=False):
                    offset = (self.min_separation + 1.) * np.array([np.cos(angle), np.sin(angle)])
                    point = np.asarray(w[:2]) + offset
                    alternatives.append((float(point[0]), float(point[1]), w[2]))
            candidates = [w for w in alternatives
                          if all(np.linalg.norm(np.asarray(w[:2]) - ref) >= self.min_separation
                                 for ref in reserved)
                          and (self._chain_rank != 0 or
                               np.linalg.norm(np.asarray(w[:2]) - robot_arr) <= horizontal_range)]
            if not candidates:
                return None

        best_w = candidates[0]

        # Graph, start, goal and candidates stay fixed during this solve.
        # Only the dual penalties change between ADMM iterations.
        if frontier is None and graph is not None:
            frontier = graph.find_best_frontier(start, goal)
        distant_frontier = (frontier is not None and
                            np.linalg.norm(np.asarray(goal) - frontier[1]) > self.fov_radius)
        cost_before = 0. if distant_frontier else self._baseline_cost(graph, start, goal)
        predictions = ([None] * len(candidates) if distant_frontier or graph is None else
                       graph.predicted_path_costs([w[:2] for w in candidates], self.fov_radius, start, goal))
        local_costs = [self._local_cost(w, graph, start, goal, cost_before, frontier, prediction)
                       for w, prediction in zip(candidates, predictions)]

        for _k in range(self._admm_K):
            best_score = float('inf')
            for w, local_cost in zip(candidates, local_costs):
                w_arr = np.array(w[:2])

                score = local_cost

                # ADMM penalty: chain neighbors only
                for did, wp_arr in chain_wps.items():
                    u = self._dual_vars.get(did, np.zeros(2))
                    z = self._project_comm(w_arr, wp_arr)
                    score += float(np.dot(u, w_arr - z))
                    score += (self._rho / 2.0) * float(
                        np.linalg.norm(w_arr - z) ** 2)

                # Repulsion: all drones (safety)
                for did, ref in repulsion_refs.items():
                    dist = float(np.linalg.norm(w_arr - ref))
                    if dist < self.min_separation:
                        violation = self.min_separation - dist
                        score += self._sep_weight * (violation ** 2)

                if score < best_score:
                    best_score = score
                    best_w = w

            # Dual update: chain neighbors only
            w_arr = np.array(best_w[:2])
            for did, wp_arr in chain_wps.items():
                z = self._project_comm(w_arr, wp_arr)
                u = self._dual_vars.get(did, np.zeros(2))
                self._dual_vars[did] = u + w_arr - z

        return best_w

    def _project_comm(self,
                      w_i: np.ndarray,
                      w_j: np.ndarray) -> np.ndarray:
        """Auxiliary variable z_ij projected onto the feasible set
        [min_separation, comm_range] for inter-drone distance.

        Project the proposed waypoint onto the feasible distance annulus
        around the neighbor. Feasible proposals are unchanged.
        """
        displacement = w_i - w_j
        distance = float(np.linalg.norm(displacement))
        if self.min_separation <= distance <= self.comm_range:
            return w_i.copy()
        if distance < 1e-6:
            return w_j + np.array([self.min_separation, 0.])
        clamped = np.clip(distance, self.min_separation, self.comm_range)
        return w_j + displacement * (clamped / distance)

    # ── Relay chain ordering ─────────────────────────────────────────────

    def _compute_chain_neighbors(
            self,
            drone_positions: Dict[str, tuple],
            current_pos: Tuple[float, float, float],
    ) -> Dict[str, tuple]:
        """Determine this drone's adjacent neighbors in the relay chain.

        All known drones (including self) are sorted by stable drone ID.
        The chain is:
            Robot <-- rank 0 <-- rank 1 <-- ... <-- rank N-1

        Returns a dict of at most 2 neighbors (predecessor + successor).
        For rank 0, the ground robot is included as a fixed "neighbor"
        under the key ``'__ground_robot__'``.

        Position delays do not change ranks when the known fleet agrees.
        """
        import time as _time

        if self.ground_robot_position is None:
            return {}

        robot_xy = np.array(self.ground_robot_position)

        # Build the full list of drones with positions
        all_drones: Dict[str, np.ndarray] = {
            self.drone_id: np.array(current_pos[:2])
        }
        for did, pos in drone_positions.items():
            if did != self.drone_id:
                all_drones[did] = np.array(pos[:2])

        # IDs define a stable chain. Sorting independent, delayed position
        # snapshots made two nearly equidistant agents both claim rank 1.
        sorted_ids = sorted(all_drones)

        now = getattr(self, '_observation_time', None)
        if now is None:
            now = _time.time()
        self._chain_order = sorted_ids
        self._last_rank_time = now

        # Find my rank
        my_rank = 0
        for i, did in enumerate(sorted_ids):
            if did == self.drone_id:
                my_rank = i
                break
        self._chain_rank = my_rank

        # Build neighbor dict
        neighbors: Dict[str, tuple] = {}

        if my_rank == 0:
            neighbors['__ground_robot__'] = tuple(self.ground_robot_position)
        else:
            pred_id = sorted_ids[my_rank - 1]
            if pred_id in all_drones:
                neighbors[pred_id] = tuple(all_drones[pred_id])

        if my_rank < len(sorted_ids) - 1:
            succ_id = sorted_ids[my_rank + 1]
            if succ_id in all_drones:
                neighbors[succ_id] = tuple(all_drones[succ_id])

        self._chain_neighbors = neighbors
        return neighbors

    # ── Straight-line fallback ─────────────────────────────────────────────

    def _straight_line_fallback(self,
                                current_pos: Tuple[float, float, float],
                                drone_positions: Dict[str, tuple],
                                flight_height: float,
                                ) -> Optional[Tuple[float, float, float]]:
        """Deploy drones evenly along robot→goal line when no graph exists."""
        if self.ground_robot_position is None or self.ground_robot_target is None:
            return None

        robot_pos  = np.array(self.ground_robot_position)
        target_pos = np.array(self.ground_robot_target)
        direction  = target_pos - robot_pos
        dist       = float(np.linalg.norm(direction))
        if dist < 1.0:
            return None
        direction = direction / dist

        # Use the same stable rank as the dispatcher.
        all_pos = {self.drone_id: current_pos}
        all_pos.update(drone_positions)
        robot_xy = robot_pos

        sorted_drones = sorted(all_pos.items())
        n_drones = len(sorted_drones)
        rank = 0
        for r, (did, _) in enumerate(sorted_drones):
            if did == self.drone_id:
                rank = r
                break

        segment = min(self.fov_radius, self.comm_range * 0.6)
        deploy_dist = segment * (rank + 1)
        deploy_dist = min(deploy_dist, dist * 0.95)
        pt = robot_pos + direction * deploy_dist
        return (float(pt[0]), float(pt[1]), flight_height)

    # ── Main entry point ──────────────────────────────────────────────────

    def compute_next_target(
            self,
            current_position: Tuple[float, float, float],
            observation_data: Dict) -> Optional[Tuple[float, float, float]]:
        """Compute next (x, y, z) waypoint.

        observation_data keys:
            'graph'              — drone's own TopologicalGraph (or None)
            'ground_robots'      — {id: {'position', 'target', 'planned_path'}}
            'nearby_drones'      — {id: {'position'}}
            'neighbor_waypoints' — {id: (x, y, z)}  (from waypoint exchange)
        """

        # ── DISCOVERING ───────────────────────────────────────────────────
        self._observation_time = observation_data.get('time')
        if self.phase == NavigationPhase.DISCOVERING:
            self.discover_mission(observation_data,
                                  flight_height=current_position[2])
            return None

        # ── DISPATCHING ───────────────────────────────────────────────────
        graph          = observation_data.get('graph')
        ground_robots  = observation_data.get('ground_robots', {})
        nearby_drones  = observation_data.get('nearby_drones', {})
        neighbor_wps   = observation_data.get('neighbor_waypoints', {})

        # Refresh ground robot state
        for rid, info in ground_robots.items():
            pos = info.get('position')
            tgt = info.get('target')
            if pos:
                self.ground_robot_position = tuple(pos[:2])
            if tgt:
                self.ground_robot_target = tuple(tgt[:2])
            path_wps = info.get('planned_path')
            if path_wps:
                self._planned_path = [tuple(wp[:2]) for wp in path_wps]

        if self.ground_robot_position is None or self.ground_robot_target is None:
            return None

        start = self.ground_robot_position
        goal  = self.ground_robot_target

        # Build drone position map
        drone_positions: Dict[str, tuple] = {}
        for did, info in nearby_drones.items():
            dpos = info.get('position')
            if dpos:
                drone_positions[did] = tuple(dpos)

        # Generate candidates from graph topology
        frontier = graph.find_best_frontier(start, goal) if graph is not None else None
        candidates = self._generate_candidates(
            graph, self._planned_path, goal,
            current_position, current_position[2], frontier=frontier)

        # Choose waypoint via C-ADMM (or fallback)
        target: Optional[Tuple[float, float, float]] = None
        if candidates and graph is not None and graph.graph.number_of_nodes() > 0:
            target = self._admm_select(
                candidates, graph, start, goal, neighbor_wps,
                drone_positions, current_position, frontier=frontier)

        if target is None:
            target = self._straight_line_fallback(
                current_position, drone_positions, current_position[2])

        # Waypoint-reached detection
        if target is not None:
            dx = target[0] - current_position[0]
            dy = target[1] - current_position[1]
            if np.sqrt(dx**2 + dy**2) < self.waypoint_threshold:
                self.waypoints_reached += 1

        self._current_target = target
        return target

    # ── Utility ───────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        return {
            'phase':              self.phase.value,
            'planned_path_len':   len(self._planned_path),
            'waypoints_reached':  self.waypoints_reached,
            'current_target':     self._current_target,
            'num_dual_vars':      len(self._dual_vars),
            'assigned_segment':   self.assigned_segment,
            'total_drones':       self.total_drones_discovered,
            'segment_start':      self.segment_start_position,
        }

    def get_current_waypoint_index(self) -> int:
        return self.waypoints_reached

    def get_total_waypoints(self) -> int:
        return len(self._planned_path)

    def reset(self):
        self.phase                  = NavigationPhase.DISCOVERING
        self.ground_robot_position  = None
        self.ground_robot_start     = None
        self.ground_robot_target    = None
        self._planned_path          = []
        self._dual_vars             = {}
        self._current_target        = None
        self.waypoints_reached      = 0
        self.assigned_segment       = None
        self._chain_rank            = 0
        self._chain_order           = []
        self._chain_neighbors       = {}
        self._last_rank_time        = 0.0
        self.total_drones_discovered = 0
        self.segment_start_position  = None
