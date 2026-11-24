"""Prefer an observed vehicle-to-road connection when its road component can advance."""
from functools import partial

import numpy as np
from scipy.ndimage import label

from vtemp4.vtemp4_connected_ground import ConnectedGroundPlanner, install_connected_ground


class AnchoredGroundPlanner(ConnectedGroundPlanner):
    def __init__(self, *args, controller, **kwargs):
        super().__init__(*args, **kwargs)
        self.controller = controller

    def _observed_entries(self, graph):
        car = self.controller
        tile = getattr(car, '_last_semantic_map', None)
        if tile is None:
            return set()
        height, width = tile.shape
        half = np.array([width, height]) / 2.
        reach = float(np.linalg.norm(half)) * car.resolution
        nearby = graph.find_nearby_nodes(car.position, reach)
        if not nearby:
            return set()
        blocked, margin = car.local_planner.observed_blocked_cells(tile, car.resolution)
        if blocked[height // 2, width // 2]:
            return set()
        # The native eight-neighbor local router may not cut blocked corners.
        # Four-neighbor labeling gives the same set of reachable cells.
        regions, _ = label(~blocked)
        current = regions[height // 2, width // 2]
        entries = set()
        for node, _ in nearby:
            xy = tuple(graph.get_node_position(node))
            x, y = np.floor((np.asarray(xy) - car.position) / car.resolution + half).astype(int)
            if (margin <= x < width - margin and margin <= y < height - margin
                    and regions[y, x] == current and graph.graph.degree(node)):
                entries.add(xy)
        return entries

    def _plan_with_graph(self, start_pos, goal_pos, graph):
        entries = self._observed_entries(graph)
        native_check = self.entry_validator
        if entries and native_check is not None:
            self.entry_validator = lambda xy: tuple(xy) in entries and native_check(xy)
            try:
                route = super()._plan_with_graph(start_pos, goal_pos, graph)
            finally:
                self.entry_validator = native_check
            # Count progress on actual road edges after the entry. An appended
            # final connector has no road evidence past the last observed node.
            road = route[2:]
            goal = np.asarray(goal_pos)
            if road and np.linalg.norm(np.asarray(road[-1]) - goal) < 1.e-6:
                nearest = graph.find_nearest_node(road[-1])
                if nearest is None or nearest[1] > 1.e-6:
                    road = road[:-1]
            if road and (np.linalg.norm(np.asarray(start_pos) - goal)
                         - min(np.linalg.norm(np.asarray(xy) - goal) for xy in road) > 2.):
                return route
        return super()._plan_with_graph(start_pos, goal_pos, graph)


def install_anchored_ground(controller):
    """Upgrade this vehicle's live planner and its native periodic route-review factory."""
    install_connected_ground(controller)
    planner = controller.global_planner
    if isinstance(planner, AnchoredGroundPlanner):
        return
    # install_connected_ground binds a separate get_action globals dictionary to each vehicle.
    factories = controller.get_action.__func__.__globals__
    factory = factories['GlobalPlanner']
    if (type(planner) is not ConnectedGroundPlanner
            or not isinstance(factory, partial) or factory.func is not ConnectedGroundPlanner):
        raise TypeError('Expected an instance-local connected ground planner factory')
    planner.__class__ = AnchoredGroundPlanner
    planner.controller = controller
    factories['GlobalPlanner'] = partial(AnchoredGroundPlanner, controller=controller, **factory.keywords)
