"""The heading actor can distinguish a reported barrier from unsurveyed terrain."""
from types import SimpleNamespace
import unittest

import numpy as np

from tests.simulator.test_ground_relay import ARCHIVE, archived_module
from vtemp2_3.navigation_evidence import EvidenceGrid, corridor_fractions
from survey.survey_fleet_observation import FleetSurveyAdapter
from vtemp4.vtemp4_observation import (
    CONTEXT_NAMES, DIRECTION_COUNT, FEATURE_NAMES, _connected_roads, build_observation,
)


class HeadingObservationTests(unittest.TestCase):
    @unittest.skipUnless(ARCHIVE.is_dir(), 'Frozen V9 sources unavailable')
    def test_barrier_reaches_heading_actor_after_receiver_delivery_with_no_road_edges(self):
        channel = archived_module('simulator/relay_channel.py').RelayChannel(seed=4, loss=0., delay_jitter=0)
        clock = SimpleNamespace(time=0., step_count=0)
        own = dict(position=[65., 65.], velocity=[1., 0.], orientation=0., angular_velocity=0.)
        ground_obs = dict(self=own, occupancy_map=np.zeros((12, 12)), map_resolution=1.)
        air_obs = dict(self=dict(position=(115., 65., 15.)), neighbors=[dict(id=0)],
                       occupancy_map=np.ones((34, 34)), map_resolution=1.)
        exp = SimpleNamespace(env=clock, ground=0, ids=[1], obs={0: ground_obs, 1: air_obs},
                              ground_controller=SimpleNamespace(unified_graph=None),
                              controllers=[SimpleNamespace(drone_id='uav')], relay_channel=channel)
        receiver = SimpleNamespace(observe=lambda: dict(one_hop_neighbors=[], relay_tokens=[]))
        ep = SimpleNamespace(env=exp, goal=np.array([225., 65.]), rollout_limit=260,
                             case=dict(world_size=250, seed=4), _relay_observation=receiver)
        adapter = FleetSurveyAdapter(ep, None)
        self.addCleanup(adapter.close)
        channel.transmit([dict(sender_id='uav', state={})], [], 0)
        before = build_observation(adapter, None, 0., None, lookahead_m=80.)
        blocked = FEATURE_NAMES.index('section_4_blocked')
        unknown = FEATURE_NAMES.index('section_4_unknown')
        self.assertEqual(before['features'][0, unknown], 1.)
        self.assertEqual(before['context'][CONTEXT_NAMES.index('terrain_receipt_valid')], 0.)

        clock.time, clock.step_count = 1., 1
        after = build_observation(adapter, 0., 1., [62., 65.], lookahead_m=80.)
        self.assertEqual(after['features'][0, blocked], 1.)
        self.assertEqual(after['features'][0, unknown], 0.)
        self.assertEqual(after['features'][DIRECTION_COUNT // 2, unknown], 1.)
        self.assertEqual(after['context'][CONTEXT_NAMES.index('terrain_receipt_valid')], 1.)
        self.assertAlmostEqual(after['context'][CONTEXT_NAMES.index('last_goal_progress_200m')], 3./200.)
        self.assertEqual(adapter.stamps[65, 115], 0.)  # Keep the UAV's source time.
        self.assertEqual(adapter.last_graph_edges, 0)
        rotated = build_observation(adapter, np.pi / 2, 0., None, lookahead_m=80.)
        self.assertEqual(rotated['features'][3 * DIRECTION_COUNT // 4, blocked], 1.)
        self.assertTrue(np.all(rotated['features'][DIRECTION_COUNT:] == 0.))

    def test_disconnected_road_beyond_gap_has_no_support(self):
        graph = dict(graph_positions=np.array([[0., 0.], [20., 0.], [40., 0.], [100., 0.]]),
                     graph_edge_index=np.array([[0, 1, 2, 3], [1, 0, 3, 2]]))
        car = np.zeros(2)
        rays = np.array([[1., 0.], [-1., 0.]])
        extent = np.full(2, 100.)
        free = np.zeros((20, 20), dtype=np.int8)
        np.testing.assert_array_equal(_connected_roads(graph, car, rays, extent, free, 1.), [20., 0.])
        np.testing.assert_array_equal(_connected_roads(graph, car, rays, extent, -np.ones_like(free), 1.),
                                      [0., 0.])

    def test_known_route_near_edge_has_no_false_unknown_and_uses_actual_world_extent(self):
        car = np.array([50., 2.])
        own = dict(position=car, orientation=0., velocity=[0., 0.], angular_velocity=0.)
        exp = SimpleNamespace(ground=0, obs={0: dict(self=own, map_resolution=1.)},
                              env=SimpleNamespace(time=0.), ground_controller=SimpleNamespace(unified_graph=None))
        receiver = SimpleNamespace(observe=lambda: dict(one_hop_neighbors=[], relay_tokens=[]))
        ep = SimpleNamespace(env=exp, goal=np.array([1., 40.]), rollout_limit=240,
                             case=dict(world_size=70.2), _relay_observation=receiver)
        adapter = SimpleNamespace(ep=ep, exp=exp, own=own, now=0., resolution=1.,
                                  cells=-np.ones((71, 71), np.int8), observe_receipts=lambda: None,
                                  last_received=None, ground_target=None)
        obs = build_observation(adapter, 0., 0., None, lookahead_m=30.)
        east = obs['features'][0]
        self.assertAlmostEqual(east[FEATURE_NAMES.index('range_fraction')], 20.2/30.)
        np.testing.assert_array_equal(east[4:].reshape(8, 3), np.tile([1., 0., 0.], (8, 1)))

        grid = EvidenceGrid(adapter.cells, 1., clearance=0.)
        endpoint = np.array([[65., 2.]])
        # VTemp2's original sampling convention remains available to its checkpoints.
        self.assertTrue(np.any(corridor_fractions(grid, car, endpoint).reshape(8, 3)[:, 2] > 0.))
        empty = corridor_fractions(grid, car, endpoint, sample_bounds=((0., 20.), (70.2, 70.2)))
        np.testing.assert_array_equal(empty.reshape(8, 3), np.tile([0., 0., 1.], (8, 1)))
