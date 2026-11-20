"""VTemp2 slow-policy map evidence stays within the ground receiver boundary."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator.ground_evidence import GroundPolicyEpisode, observed_fused_graph, remember_local_tile
from world_model.hierarchical_runtime import HierarchyConfig
from world_model.navigation_evidence import AttemptMemory


def local_tile(position, pixels=None, **metadata):
    return dict(
        occupancy_map=np.zeros((2, 2)) if pixels is None else np.asarray(pixels),
        map_resolution=1.,
        self=dict(position=position, velocity=[0., 0., 0.]),
        **metadata,
    )


class GroundEvidenceTests(unittest.TestCase):
    def test_only_fresh_received_road_evidence_near_the_survey_counts(self):
        from simulator.ground_evidence import survey_has_road_evidence

        target = [100., 40.]
        def received(nodes, **edge_attributes):
            graph = nx.Graph()
            for name, (position, source_time) in enumerate(nodes):
                frames = {} if source_time is None else {f'uav:{name}': source_time}
                graph.add_node(name, position=position, observation_frames=frames)
            if len(nodes) == 2:
                graph.add_edge(0, 1, **edge_attributes)
            return observed_fused_graph(SimpleNamespace(graph=graph, node_confidence=lambda _: .8), 30.)

        def answered(nodes, **attributes):
            return survey_has_road_evidence(received(nodes, **attributes), target, 10., 30., 25.)

        fresh = [([101., 40.], 12.), ([104., 41.], 14.)]
        self.assertTrue(answered(fresh))
        self.assertFalse(answered(fresh, bridged=True))
        self.assertFalse(answered(fresh[:1]))
        self.assertFalse(answered([([101., 40.], 10.), fresh[1]]))
        self.assertFalse(answered([([101., 40.], None), fresh[1]]))
        self.assertFalse(answered([([101., 40.], 35.), fresh[1]]))
        self.assertFalse(answered([([201., 40.], 12.), ([204., 41.], 14.)]))

    def test_only_locally_measured_cells_enter_memory(self):
        cells = np.zeros((6, 6), np.int8)
        remember_local_tile(cells, 2., local_tile([5., 5.]))
        self.assertEqual(cells[2, 2], -1)
        self.assertEqual(np.count_nonzero(cells), 1)

        remember_local_tile(cells, 2., local_tile([9., 5.], [[0, 0], [0, 1]],
            occupancy_known_mask=[[False, False], [False, True]]))
        self.assertEqual(cells[2, 4], 1)
        self.assertEqual(cells[2, 2], -1)
        self.assertEqual(np.count_nonzero(cells), 2)

        remember_local_tile(cells, 2., local_tile([9., 5.], [[0, 0], [0, -1]]))
        remember_local_tile(cells, 2., local_tile([9., 5.], sensor_valid=False))
        self.assertEqual(cells[2, 4], 1)
        remember_local_tile(cells, 2., local_tile([9., 5.]))
        self.assertEqual(cells[2, 4], -1)

    def test_fused_graph_keeps_original_stamps_and_excludes_audit_bridges(self):
        graph = nx.Graph()
        graph.add_node('a', position=[20., 20.], observation_frames={'drone_1:1': 2., 'drone_2:4': 4.},
                       observation_count=9, last_updated=100.)
        graph.add_node('b', position=[21., 20.], last_updated=5.)
        graph.add_node('c', position=[22., 20.], observation_frames={'drone_1:2': 12.})
        graph.add_edge('a', 'b', weight=1.)
        graph.add_edge('b', 'c', weight=1., bridged=True)
        received = observed_fused_graph(SimpleNamespace(graph=graph, node_confidence=lambda node: .7), 7.)

        self.assertEqual(received['graph_edges'], 1)
        np.testing.assert_array_equal(received['graph_positions'], [[20., 20.], [21., 20.], [22., 20.]])
        np.testing.assert_array_equal(received['graph_edge_index'], [[0, 1], [1, 0]])
        np.testing.assert_array_equal(received['graph_observation_counts'], [9, 1, 1])
        np.testing.assert_array_equal(received['graph_source_stamps'], [4., np.nan, np.nan])
        self.assertTrue(graph['b']['c']['bridged'])

    def test_candidates_ignore_audit_ledger_and_unreceived_drone_maps(self):
        class ReceiverOnly(SimpleNamespace):
            @property
            def ledger(self):
                raise AssertionError('Slow policy consulted the simulator audit ledger')

        episode = GroundPolicyEpisode.__new__(GroundPolicyEpisode)
        local = local_tile([51., 51.])
        local['neighbors'] = [dict(id='drone_0', broadcast={'evidence_audit': object()})]
        episode.env = ReceiverOnly(
            ground='ground_0', ids=['drone_0'],
            env=SimpleNamespace(time=1.),
            obs={'ground_0': local, 'drone_0': {'current_map': object()}},
            ground_controller=SimpleNamespace(last_fusion_per_drone={}, unified_graph=None),
        )
        episode.goal = np.array([175., 51.])
        episode.memory = AttemptMemory()
        episode.config = HierarchyConfig()
        episode._local_resolution = 2.
        episode._world_size = 400.
        episode._local_cells = np.zeros((200, 200), np.int8)
        episode._local_time = episode._belief_time = episode._belief_cache = None
        episode.attempts, episode.mission_budget, episode.previous = 0, 600, None
        earlier = dict(one_hop_neighbors=[], relay_tokens=[])
        arrival = dict(one_hop_neighbors=[], relay_tokens=[{'origin_id': 'drone_0', 'age': 3.}])
        episode._relay_observation = Mock()
        episode._relay_observation.observe.side_effect = [earlier, arrival]

        def capture(_, candidates, peers, relay_rows):
            return candidates, peers, relay_rows

        with patch('simulator.ground_evidence.ground_observation', side_effect=capture):
            before, _, before_relay = episode.observation()
            self.assertEqual(before_relay, [])
            self.assertEqual(episode.belief()['graph_edges'], 0)
            self.assertEqual(before['statuses'][before['kinds'].index('final')], 'unknown')
            self.assertTrue(before['mask'][before['kinds'].index('final')])

            episode.env.env.time = 4.
            graph = nx.Graph()
            graph.add_node('air_evidence', position=[122., 51.], observation_frames={'drone_0:1': 1.})
            episode.env.ground_controller.unified_graph = SimpleNamespace(
                graph=graph, node_confidence=lambda node: .8)
            episode.env.ground_controller.last_fusion_per_drone = {'drone_0': 4.}
            after, _, after_relay = episode.observation()

        self.assertEqual(after_relay, arrival['relay_tokens'])
        self.assertIn('mapped', after['kinds'])
        self.assertEqual(episode.belief()['graph_source_stamps'][0], 1.)
        self.assertEqual(episode.belief()['grid'].cells[25, 61], 0)


if __name__ == '__main__':
    unittest.main()
