"""Fusion must retain observed edge geometry and deduplicate relay evidence."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import networkx as nx
import numpy as np

from shared.common.graph_structure import TopologicalGraph


class DroneFusionTests(unittest.TestCase):
    def graph(self, source, points, edges=()):
        graph = TopologicalGraph(source)
        for node, point in points.items():
            graph.add_node(node, point, dict(terrain_type='TRAVERSABLE',
                observation_count=0, legacy_observation_count=0))
            graph.observe_node(node, source, 0.)
        for a, b in edges:
            graph.add_edge(a, b)
        return graph

    @staticmethod
    def segments(graph):
        return {frozenset((tuple(graph.graph.nodes[a]['position']),
                          tuple(graph.graph.nodes[b]['position'])))
                for a, b in graph.graph.edges}

    @staticmethod
    def distance_to_origin(segment):
        a, b = (np.asarray(point) for point in segment)
        delta = b-a
        t = np.clip(-np.dot(a, delta)/np.dot(delta, delta), 0., 1.)
        return float(np.linalg.norm(a+t*delta))

    def test_nearby_routes_across_obstacle_keep_original_segments(self):
        # Both observed routes clear a radius-0.5 obstacle at the origin.
        # Averaging their nearby endpoints would move the route into it.
        west = self.graph('west', {'a': (-1., -2.), 'b': (-1., 2.)}, [('a', 'b')])
        east = self.graph('east', {'a': (1., -2.), 'b': (1., 2.)}, [('a', 'b')])
        expected = self.segments(west) | self.segments(east)
        self.assertTrue(all(self.distance_to_origin(edge) > .5 for edge in expected))

        west.fuse_with_graph(east, merge_distance_threshold=3.)

        self.assertEqual(self.segments(west), expected)
        self.assertEqual(west.graph.number_of_nodes(), 4)
        self.assertEqual(nx.number_connected_components(west.graph), 2)
        self.assertTrue(all(self.distance_to_origin(edge) > .5
                            for edge in self.segments(west)))

    def test_tiny_coordinate_difference_is_not_an_observed_connection(self):
        first = self.graph('first', {'a': (1., 2.)})
        second = self.graph('second', {'b': (1.+1e-12, 2.)})
        first.fuse_with_graph(second, merge_distance_threshold=100.)
        self.assertEqual(first.graph.number_of_nodes(), 2)
        self.assertEqual(first.graph.number_of_edges(), 0)
        self.assertEqual(first.graph.nodes['a']['position'], (1., 2.))

    def test_exact_position_merges_evidence_once(self):
        first = self.graph('first', {'a': (1., 2.)})
        second = self.graph('second', {'b': (1., 2.)})
        first.fuse_with_graph(second)
        first.fuse_with_graph(second)
        self.assertEqual(first.graph.number_of_nodes(), 1)
        self.assertEqual(first.graph.nodes['a']['observation_count'], 2)
        self.assertEqual(first.graph.nodes['a']['position'], (1., 2.))

    def test_bidirectional_relay_keeps_geometry_and_counts(self):
        first = self.graph('first', {'a': (-1., -2.), 'b': (-1., 2.)}, [('a', 'b')])
        second = self.graph('second', {'a': (1., -2.), 'b': (1., 2.)}, [('a', 'b')])
        expected = self.segments(first) | self.segments(second)
        for _ in range(5):
            first.fuse_with_graph(TopologicalGraph.from_message(second.to_message()))
            second.fuse_with_graph(TopologicalGraph.from_message(first.to_message()))
        for graph in (first, second):
            self.assertEqual(graph.graph.number_of_nodes(), 4)
            self.assertEqual(self.segments(graph), expected)
            self.assertTrue(all(data['observation_count'] == 1
                                for _, data in graph.graph.nodes(data=True)))

    def test_generated_id_collision_does_not_replace_existing_node(self):
        first = self.graph('first', {'first_fused_2': (0., 0.), 'other': (0., 4.)},
                           [('first_fused_2', 'other')])
        second = self.graph('second', {'first_fused_2': (10., 0.)})
        expected = self.segments(first)
        first.fuse_with_graph(second)
        self.assertEqual(first.graph.nodes['first_fused_2']['position'], (0., 0.))
        self.assertEqual(first.graph.number_of_nodes(), 3)
        self.assertEqual(self.segments(first), expected)
        self.assertIn((10., 0.), [tuple(data['position'])
                                for _, data in first.graph.nodes(data=True)])

    def test_same_position_edge_does_not_create_self_loop(self):
        first = self.graph('first', {})
        second = self.graph('second', {'a': (0., 0.), 'b': (0., 0.)}, [('a', 'b')])
        first.fuse_with_graph(second)
        self.assertEqual(first.graph.number_of_nodes(), 1)
        self.assertEqual(first.graph.number_of_edges(), 0)


if __name__ == '__main__':
    unittest.main()
