"""Select grounded road entries for the frozen vehicle controller using its received graph."""
from functools import partial
from types import FunctionType, MethodType

import networkx as nx
import numpy as np

# The extension subclasses the planner instance installed by frozen_navigation.runtime.
from ground.global_planner import GlobalPlanner


def _distance(a, b):
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))


def _goal_in_local_free_space(controller, goal):
    # Permit the native final connector when the actual goal cell, including
    # its clearance margin, is observable from this pose.
    tile = getattr(controller, '_last_semantic_map', None)
    if tile is None:
        return False
    height, width = tile.shape
    ix, iy = np.floor((np.asarray(goal)-controller.position)/controller.resolution
                      + np.array([width,height])/2.).astype(int)
    if not (0 <= ix < width and 0 <= iy < height):
        return False
    blocked, _ = controller.local_planner.observed_blocked_cells(tile, controller.resolution)
    return not blocked[iy,ix]


class ConnectedGroundPlanner(GlobalPlanner):
    """Keep internal waypoints on one reachable road component until more evidence arrives."""

    def __init__(self, *args, confidence_threshold=0., local_goal=None, arrival_threshold=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_threshold = confidence_threshold
        self.local_goal = local_goal
        self.arrival_threshold = arrival_threshold

    def _partial(self):
        return bool(self.current_path and self.goal_position is not None
                    and _distance(self.current_path[-1], self.goal_position) > 1.e-6)

    @property
    def waiting_at_frontier(self):
        return self._native_wait or (self._partial() and self.current_waypoint_idx >= len(self.current_path))

    @waiting_at_frontier.setter
    def waiting_at_frontier(self, value):
        self._native_wait = value

    def should_replan(self, current_pos):
        # Reassess locally accessible entries once on arriving at a frontier. A
        # parked car then waits for fusion instead of clearing its path each tick.
        if (self._partial() and self.current_waypoint_idx >= len(self.current_path)
                and _distance(current_pos, self.current_path[0]) > self.replan_threshold):
            return True
        return super().should_replan(current_pos)

    def advance_waypoint(self, current_pos, arrival_threshold=3.):
        advanced = super().advance_waypoint(current_pos, arrival_threshold)
        if advanced and self.current_waypoint_idx == self.start_connection_idx:
            super().advance_waypoint(current_pos, arrival_threshold)
        return advanced

    def _compute_path_length(self, waypoints):
        cost = super()._compute_path_length(waypoints)
        if waypoints and self.goal_position is not None:
            # Native periodic reviews compare old/new costs. Omitted unknown
            # distance must not make a short dead-end look like arrival.
            cost += _distance(waypoints[-1], self.goal_position)
        return cost

    def _road_waypoints(self, start, points, final):
        # Keep the true entry even when <1 m away. The next internal edge
        # must be checked from that node, never from an approximate car pose.
        self.start_connection_idx = 1
        result = [start, *points]
        if final is not None and _distance(result[-1], final) > 1.e-6:
            result.append(final)
        return result

    def _plan_with_graph(self, start_pos, goal_pos, graph):
        goal_node, _ = graph.find_nearest_node(goal_pos)
        roads = graph.graph
        if self.entry_validator is None:
            self.start_connection_idx = None
            return [start_pos]
        if self.local_goal is not None and self.local_goal(goal_pos) and self.entry_validator(goal_pos):
            return self._road_waypoints(start_pos, [goal_pos], None)
        # edge_confidence_from_positions calls this same function with the
        # graph-wide median. The observed planning graph is fixed for this search.
        confidence = partial(graph._edge_confidence, d_bar=graph.mean_edge_length())
        # Preserve the native entry and shortest path when its observed edges
        # already connect to the goal node. Only a broken route needs a search
        # over alternative components.
        nearest = self._entry_node(start_pos, graph)
        if (nearest is not None and nearest[1] <= self.direct_waypoint_distance
                and roads.degree(nearest[0])):
            nodes = graph.get_path(nearest[0], goal_node)
            if nodes is not None and len(nodes) > 1:
                points = [graph.get_node_position(n) for n in nodes]
                if all(confidence(a,b) > self.confidence_threshold
                       for a,b in zip(points,points[1:])):
                    return self._road_waypoints(start_pos, points, goal_pos)
        nearby = sorted(graph.find_nearby_nodes(start_pos, self.direct_waypoint_distance),
                        key=lambda item: (item[1], str(item[0])))
        entries = [node for node, _ in nearby if roads.degree(node)
                   and self.entry_validator(graph.get_node_position(node))]
        # Verify the actual proposed edges with the same confidence function
        # used at execution. Only copy the graph if an unusable edge needs removing.
        while entries:
            choices = []
            visited = set()
            for entry in entries:
                if entry in visited or not roads.degree(entry):
                    continue
                component = nx.node_connected_component(roads, entry)
                visited.update(component)
                exit_node = goal_node if goal_node in component else min(
                    component, key=lambda n: (_distance(graph.get_node_position(n), goal_pos), str(n)))
                remaining = _distance(graph.get_node_position(exit_node), goal_pos)
                complete = exit_node == goal_node
                if not complete and (remaining >= _distance(start_pos, goal_pos) or (
                        exit_node == entry and _distance(start_pos,graph.get_node_position(entry)) <= self.arrival_threshold)):
                    continue
                choices.append((not complete, remaining, _distance(start_pos, graph.get_node_position(entry)),
                                str(entry), entry, exit_node))
            if not choices:
                break
            partial_route, _, _, _, entry, exit_node = min(choices)
            nodes = nx.shortest_path(roads, entry, exit_node, weight='weight')
            points = [graph.get_node_position(n) for n in nodes]
            weak = [(u,v) for u,v,a,b in zip(nodes,nodes[1:],points,points[1:])
                    if confidence(a,b) <= self.confidence_threshold]
            if len(nodes) == 1:
                neighbors = list(roads.neighbors(entry))
                if not any(confidence(points[0],graph.get_node_position(n))
                           > self.confidence_threshold for n in neighbors):
                    weak = [(entry,n) for n in neighbors]
            if weak:
                if roads is graph.graph:
                    roads = roads.copy()
                roads.remove_edges_from(weak)
                continue
            return self._road_waypoints(start_pos, points, None if partial_route else goal_pos)
        self.start_connection_idx = None
        return [start_pos]


def _invalid_remaining_edge(planner, graph, threshold):
    if graph is None or not planner.using_graph or not planner.current_path:
        return False
    path = planner.current_path
    for i in range(max(1, planner.current_waypoint_idx), len(path)):
        if i == planner.start_connection_idx or (
                i == len(path)-1 and planner.goal_position is not None
                and _distance(path[i],planner.goal_position) < 1.e-6):
            continue
        a, b = graph.find_nearest_node(path[i-1]), graph.find_nearest_node(path[i])
        if (a is None or b is None or max(a[1],b[1]) > 1.e-6 or not graph.graph.has_edge(a[0],b[0])
                or graph.edge_confidence_from_positions(path[i-1],path[i]) <= threshold):
            return True
    return False


def install_connected_ground(controller):
    """Install the planner on this live vehicle without resetting perception, mission, or RNG."""
    planner = controller.global_planner
    if isinstance(planner, ConnectedGroundPlanner):
        return
    if type(planner) is not GlobalPlanner:
        raise TypeError('Connected ground control requires the native GlobalPlanner')
    threshold = controller.edge_confidence_threshold
    invalidate = _invalid_remaining_edge(planner, controller.unified_graph, threshold)
    waiting = planner.waiting_at_frontier
    # The archived get_action constructs an additional GlobalPlanner by name
    # for 5 s route reviews. Bind its unchanged code to an instance-local factory:
    # concurrent legacy episodes continue to instantiate their original planner.
    original = controller.get_action.__func__
    local_goal = partial(_goal_in_local_free_space, controller)
    factory = partial(ConnectedGroundPlanner, confidence_threshold=threshold, local_goal=local_goal,
                      arrival_threshold=controller.waypoint_threshold)
    if original.__globals__.get('GlobalPlanner') is not GlobalPlanner:
        raise TypeError('Unrecognized native vehicle route-review factory')
    action = FunctionType(original.__code__, dict(original.__globals__, GlobalPlanner=factory),
                          original.__name__, original.__defaults__, original.__closure__)
    action.__kwdefaults__ = original.__kwdefaults__
    planner.__class__ = ConnectedGroundPlanner
    planner.confidence_threshold = threshold
    planner.local_goal = local_goal
    planner.arrival_threshold = controller.waypoint_threshold
    planner.waiting_at_frontier = waiting
    if invalidate:
        planner.clear_path()
    controller.get_action = MethodType(action, controller)
