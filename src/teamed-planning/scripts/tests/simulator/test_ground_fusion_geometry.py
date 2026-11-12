"""Fusion must preserve certified source segments and require evidence for joins."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.graph_structure import TopologicalGraph
from ground.graph_fusion import GraphFusion


def graph(name, points):
    result = TopologicalGraph(name)
    for i, point in enumerate(points):
        result.add_node(str(i), point, {'terrain_type': 'TRAVERSABLE'})
    for i in range(len(points) - 1):
        result.add_edge(str(i), str(i + 1))
    return result


def segments(result):
    return {frozenset((tuple(result.get_node_position(a)), tuple(result.get_node_position(b))))
            for a, b in result.graph.edges}


def circle_clearance(a, b, centre=(0., 2.), radius=.25):
    a, b, centre = np.asarray(a), np.asarray(b), np.asarray(centre)
    delta = b - a
    t = np.clip(np.dot(centre - a, delta) / np.dot(delta, delta), 0., 1.)
    return float(np.linalg.norm(centre - (a + t * delta)) - radius)


class GroundFusionGeometryTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)

    def test_fusion_preserves_source_clearance_and_all_edge_geometry(self):
        # Both received paths clear this obstacle by the configured 1.5 m.
        # Averaging the nearby left endpoints used to produce (-4, .9)->(4, 0),
        # reducing clearance to 1.29 m and causing the vehicle to reject it.
        first = graph('a', [(-4., 0.), (4., 0.)])
        second = graph('b', [(-4., 1.8), (-4., 5.)])
        originals = segments(first) | segments(second)
        self.assertTrue(all(circle_clearance(*tuple(edge)) >= 1.5 for edge in originals))
        self.assertLess(circle_clearance((-4., .9), (4., 0.)), 1.5)
        fusion = GraphFusion('car')
        fusion.add_drone_graph('a', first)
        fusion.add_drone_graph('b', second)
        result = fusion.get_unified_graph()
        self.assertEqual(segments(result), originals)
        self.assertTrue(all(circle_clearance(*tuple(edge)) >= 1.5 for edge in segments(result)))
        self.assertEqual(nx.number_connected_components(result.graph), 2)

    def test_proximity_cannot_bypass_missing_or_blocked_bridge_evidence(self):
        for checker in (None, lambda a, b: False):
            with self.subTest(checker=checker):
                fusion = GraphFusion('car')
                fusion.bridge_distance = 15.
                fusion.bridge_check = checker
                fusion.add_drone_graph('a', graph('a', [(0., 0.), (10., 0.)]))
                fusion.add_drone_graph('b', graph('b', [(11., 0.), (21., 0.)]))
                self.assertEqual(nx.number_connected_components(fusion.get_unified_graph().graph), 2)

    def test_observed_short_connection_is_added_without_snapping_endpoints(self):
        fusion = GraphFusion('car')
        fusion.bridge_distance = 2.
        checked = []
        def clear(a, b):
            checked.append((tuple(a), tuple(b)))
            return True
        fusion.bridge_check = clear
        fusion.add_drone_graph('a', graph('a', [(0., 0.), (10., 0.)]))
        fusion.add_drone_graph('b', graph('b', [(11., 0.), (21., 0.)]))
        result = fusion.get_unified_graph()
        self.assertEqual(set(result.get_node_position(n) for n in result.graph),
                         {(0., 0.), (10., 0.), (11., 0.), (21., 0.)})
        self.assertEqual(checked, [((10., 0.), (11., 0.))])
        self.assertTrue(nx.is_connected(result.graph))
        bridges = [d for _, _, d in result.graph.edges(data=True) if d.get('bridged')]
        self.assertEqual(len(bridges), 1)
        self.assertEqual(bridges[0]['weight'], 1.)

    def test_identical_copies_and_repeated_packets_are_idempotent(self):
        source = graph('a', [(0., 0.), (10., 0.)])
        source.observe_node('0', 'a', 1.)
        fusion = GraphFusion('car')
        for _ in range(4):
            fusion.add_drone_graph('a', TopologicalGraph.from_message(source.to_message()))
            fusion.add_drone_graph('relay', TopologicalGraph.from_message(source.to_message()))
        result = fusion.get_unified_graph()
        self.assertEqual(len(result), 2)
        self.assertEqual(result.graph.number_of_edges(), 1)
        self.assertEqual(segments(result), segments(source))
        first = next(d for _, d in result.graph.nodes(data=True) if tuple(d['position']) == (0., 0.))
        self.assertEqual(first['observation_count'], source.graph.nodes['0']['observation_count'])

    def test_new_source_snapshot_replaces_removed_source_edges(self):
        source = graph('a', [(0., 0.), (10., 0.)])
        fusion = GraphFusion('car')
        fusion.add_drone_graph('a', source)
        replacement = TopologicalGraph.from_message(source.to_message())
        replacement.graph.remove_edge('0', '1')
        fusion.add_drone_graph('a', replacement)
        self.assertEqual(fusion.get_unified_graph().graph.number_of_edges(), 0)
        self.assertEqual(source.graph.number_of_edges(), 1)


if __name__ == '__main__':
    unittest.main()
