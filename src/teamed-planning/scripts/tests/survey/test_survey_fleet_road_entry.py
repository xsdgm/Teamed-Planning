from copy import copy
from types import SimpleNamespace
import unittest

import numpy as np

from shared.common.graph_structure import TopologicalGraph
from survey.survey_fleet_runtime import SurveyFleetEpisode


class RoadEntryTests(unittest.TestCase):
    def setUp(self):
        self.start = (10., 10.)
        self.entry = (10.75, 10.)
        self.next = (23., 12.)
        self.goal = (45., 25.)
        self.graph = TopologicalGraph()
        for name, point in [('entry', self.entry), ('next', self.next), ('a', (40., 45.)), ('b', (40.5, 45.))]:
            self.graph.add_node(name, point)
        self.graph.add_edge('entry', 'next')
        self.graph.add_edge('a', 'b')
        self.planner = SimpleNamespace(current_path=[self.start, self.next, self.goal], using_graph=True,
                                       current_waypoint_idx=1, start_connection_idx=None)
        driver = SimpleNamespace(position=self.start, global_planner=self.planner,
                                 waypoint_threshold=1., edge_confidence_threshold=0.)
        self.tile = np.zeros((16, 16), np.int8)
        ground = dict(occupancy_map=self.tile, map_resolution=.5, map_axes='yx')
        self.ep = SurveyFleetEpisode.__new__(SurveyFleetEpisode)
        self.ep.env = SimpleNamespace(ground=0, obs={0:ground}, ground_controller=driver)
        self.ep._original_filter = lambda graph, semantic: graph
        self.view = self.ep._native_filter(self.graph, None)

    def test_rejected_synthetic_edge_uses_reached_entry_without_new_control_step(self):
        self.assertEqual(self.graph.edge_confidence_from_positions(self.start, self.next), 0.)
        actual = self.graph.edge_confidence_from_positions(self.entry, self.next)
        self.assertGreater(actual, 0.)
        self.assertEqual(self.view.edge_confidence_from_positions(self.start, self.next), actual)
        self.assertEqual(self.planner.current_path, [self.entry, self.next, self.goal])
        self.assertEqual(self.planner.current_waypoint_idx, 1)
        self.assertIsNone(self.planner.start_connection_idx)

    def test_healthy_edge_remains_untouched(self):
        self.graph.graph.remove_edge('a','b')
        original = self.graph.edge_confidence_from_positions(self.start, self.next)
        self.assertGreater(original, 0.)
        self.assertEqual(self.view.edge_confidence_from_positions(self.start, self.next), original)
        self.assertEqual(self.planner.current_path[0], self.start)

    def test_route_accepted_from_candidate_uses_actual_pruned_copy(self):
        # Mirror the frozen driver's shallow copy and local node/edge filtering,
        # followed by a candidate winning and replacing the live path list.
        local = copy(self.view)
        local.graph = local.graph.copy()
        local.graph.remove_edge('entry', 'next')
        local._spatial_index_valid = False
        self.assertTrue(self.graph.graph.has_edge('entry', 'next'))
        self.planner.current_path = [self.start, self.next, self.goal]
        self.assertEqual(local.edge_confidence_from_positions(self.start, self.next), 0.)
        self.assertEqual(self.planner.current_path[0], self.start)
        self.assertGreater(self.view.edge_confidence_from_positions(self.start, self.next), 0.)
        self.assertEqual(self.planner.current_path[0], self.entry)

    def test_blocked_short_connection_or_genuine_unknown_edge_stays_blocked(self):
        self.tile[8, 11] = 1
        self.assertEqual(self.view.edge_confidence_from_positions(self.start, self.next), 0.)
        self.assertEqual(self.planner.current_path[0], self.start)
        self.tile[:] = 0
        for node in self.graph.graph.nodes:
            self.graph.graph.nodes[node]['observation_count'] = 0
        self.assertEqual(self.graph.edge_confidence_from_positions(self.entry, self.next), 0.)
        self.assertEqual(self.view.edge_confidence_from_positions(self.start, self.next), 0.)
        self.assertEqual(self.planner.current_path[0], self.start)


if __name__ == '__main__':
    unittest.main()
