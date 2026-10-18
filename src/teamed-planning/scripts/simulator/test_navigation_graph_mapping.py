"""Task-oriented mapper regressions using only synthetic sensor observations."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import networkx as nx
import numpy as np
from common.graph_structure import TopologicalGraph
from drone.skeleton_graph_builder import SkeletonGraphBuilder
from simulator.navigation_metrics import task_graph_metrics


class MappingTests(unittest.TestCase):
    def builder(self):
        return SkeletonGraphBuilder('test', resolution=2., node_spacing=10.,
            min_clearance=3., max_edge_distance=15., k_neighbors=4)

    def test_open_sensor_footprint_has_connected_multiple_nodes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            graph = self.builder().build_graph(np.ones((12,12), np.uint8), (60.,60.,15.))
        self.assertGreater(len(graph), 1)
        self.assertTrue(nx.is_connected(graph.graph))

    def test_old_nodes_are_reconnected_without_new_nodes(self):
        builder = self.builder()
        semantic = np.ones((12,12), np.uint8)
        graph = TopologicalGraph('test')
        graph.add_node('a', (57.,59.), dict(observation_count=0, legacy_observation_count=0))
        graph.add_node('b', (63.,59.), dict(observation_count=0, legacy_observation_count=0))
        # Isolate edge refresh from new-node sampling.
        builder._sample_nodes_on_skeleton = lambda *args, **kw: []
        builder.update_graph(graph, semantic, (60.,60.,15.), observation_id=1)
        self.assertEqual(len(graph), 2)
        self.assertTrue(graph.graph.has_edge('a','b'))
        builder.update_graph(graph, semantic, (60.,60.,15.), observation_id=1)
        self.assertEqual(graph.graph.nodes['a']['observation_count'], 1)

    def test_wall_unknown_and_clearance_do_not_create_bridges(self):
        for blocked in (0, 2):
            builder = self.builder()
            semantic = np.ones((12,12), np.uint8)
            semantic[5:7, :] = blocked
            graph = TopologicalGraph('test')
            graph.add_node('a', (53.,59.))
            graph.add_node('b', (67.,59.))
            builder._sample_nodes_on_skeleton = lambda *args, **kw: []
            builder.update_graph(graph, semantic, (60.,60.,15.))
            self.assertFalse(graph.graph.has_edge('a','b'))
        builder = self.builder()
        semantic = np.ones((12,12), np.uint8)
        semantic[6,6] = 2
        # Adjacent to the obstacle: free centreline alone is insufficient.
        self.assertFalse(builder._is_path_traversable(semantic, (5,3), (5,9)))

    def test_valid_long_corridor_is_not_erased_by_pruning(self):
        builder = self.builder()
        semantic = np.full((12,12), 2, np.uint8)
        semantic[3:9,:] = 1
        with contextlib.redirect_stdout(io.StringIO()):
            graph = builder.build_graph(semantic, (60.,60.,15.))
        self.assertGreater(len(graph), 1)
        self.assertTrue(nx.is_connected(graph.graph))

    def test_known_world_bounds_exclude_outside_samples(self):
        builder=self.builder(); builder.world_bounds=(0.,0.,120.,120.)
        with contextlib.redirect_stdout(io.StringIO()):
            graph=builder.build_graph(np.ones((12,12),np.uint8),(3.,3.,15.))
        for _,data in graph.graph.nodes(data=True):
            self.assertTrue(all(3.<=v<=117. for v in data['position']))

    def test_task_connectivity_ignores_unrelated_components_and_requires_access(self):
        graph=TopologicalGraph('task')
        graph.add_node('a',(5.,5.)); graph.add_node('b',(15.,5.))
        graph.add_node('unrelated',(90.,90.)); graph.add_edge('a','b')
        result=task_graph_metrics(graph,np.array([0.,5.]),np.array([20.,5.]),lambda a,b:True,6.)
        self.assertEqual(result['graph_task_connected'],1.)
        self.assertEqual(result['graph_task_path_length'],20.)
        self.assertEqual(result['graph_components'],2.)
        blocked=task_graph_metrics(graph,np.array([0.,5.]),np.array([20.,5.]),lambda a,b:False,6.)
        self.assertEqual(blocked['graph_task_connected'],0.)

    def test_old_endpoint_pixel_quantization_cannot_hide_obstacle_clearance(self):
        builder=self.builder()
        semantic=np.ones((12,12),np.uint8); semantic[6,6]=2
        # Continuous endpoint locations matter, even inside the same free pixel.
        self.assertFalse(builder._observed_segment_clear(semantic,np.array([9.1,9.1]),np.array([9.1,16.])))


if __name__ == '__main__':
    unittest.main()
