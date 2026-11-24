"""Local mapping keeps narrow routes while preserving the checkpoint sensor."""
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.drone.drone_main import DroneController
from shared.drone.local_mapping import (validate_local_mapping, prepare_local_mapping,
                                 mapping_semantic_map, with_local_mapping)
from shared.drone.semantic_segmentation import TerrainType, SemanticProcessor
from shared.drone.skeleton_graph_builder import SkeletonGraphBuilder


def observation():
    return dict(occupancy_map=np.zeros((12, 12), np.uint8), map_resolution=2.,
                map_axes="yx", mapping_occupancy_map=np.zeros((48, 48), np.uint8),
                mapping_resolution=.5, mapping_known_mask=np.ones((48, 48), bool),
                mapping_axes="yx", self=dict(position=(12., 12.), height=15.),
                neighbors=[], full_state=dict(time=0.), time_step=1.)


def controller():
    with contextlib.redirect_stdout(io.StringIO()):
        return DroneController("mapping_test", resolution=2., graph_min_clearance=1.5)


class LocalMappingTests(unittest.TestCase):
    def test_four_metre_corridor_keeps_continuously_safe_graph(self):
        obs = observation()
        obs["occupancy_map"][:] = 1
        obs["occupancy_map"][5:7, :] = 0
        obs["mapping_occupancy_map"][:] = 1
        obs["mapping_occupancy_map"][20:28, :] = 0
        c = controller()
        with contextlib.redirect_stdout(io.StringIO()):
            coarse = c.graph_builder.build_graph(
                c._generate_semantic_map(obs["occupancy_map"].T), (12., 12., 15.))
        self.assertEqual(len(coarse), 0)
        mapping = prepare_local_mapping(c, obs)
        semantic = mapping_semantic_map(mapping, c._generate_semantic_map)
        with contextlib.redirect_stdout(io.StringIO()):
            fine = c.graph_builder.build_graph(semantic, (12., 12., 15.))
        self.assertGreater(len(fine), 1)
        self.assertGreater(fine.graph.number_of_edges(), 0)
        points = [np.asarray(data["position"][:2]) for _, data in fine.graph.nodes(data=True)]
        # The real free strip is y in [10,14]; graph edges between these points
        # stay inside its convex 1.5m inset, including the observed x boundaries.
        for x, y in points:
            self.assertGreater(x, 1.5)
            self.assertLess(x, 22.5)
            self.assertGreater(y, 11.5)
            self.assertLess(y, 12.5)
        self.assertGreater(max(p[0] for p in points) - min(p[0] for p in points), 12.)

    def test_unknown_is_not_free_and_breaks_the_corridor(self):
        obs = observation()
        obs["mapping_occupancy_map"][:] = 1
        obs["mapping_occupancy_map"][20:28, :] = 0
        obs["mapping_known_mask"][:, 23:25] = False
        c = controller()
        mapping = prepare_local_mapping(c, obs)
        semantic = mapping_semantic_map(mapping, c._generate_semantic_map)
        self.assertTrue(np.all(semantic[23:25, :] == TerrainType.UNKNOWN.value))
        self.assertTrue(np.all(mapping.occupancy[23:25, :] == 1))
        self.assertFalse(c.graph_builder._observed_segment_clear(
            semantic, np.array([5., 12.]), np.array([19., 12.])))
        # Normalization must not write a fabricated obstacle into the source.
        self.assertEqual(obs["mapping_occupancy_map"][22, 24], 0)

    def test_axis_normalization_for_non_square_views(self):
        obs = observation()
        obs.update(occupancy_map=np.zeros((6, 12), np.uint8),
                   mapping_occupancy_map=np.zeros((24, 48), np.uint8),
                   mapping_known_mask=np.ones((24, 48), bool))
        obs["mapping_occupancy_map"][3, 17] = 1
        result = validate_local_mapping(obs)
        self.assertEqual(result.footprint_xy, (24., 12.))
        self.assertEqual(result.occupancy.shape, (48, 24))
        self.assertEqual(result.occupancy[17, 3], 1)
        obs["mapping_occupancy_map"] = obs["mapping_occupancy_map"].T
        obs["mapping_known_mask"] = obs["mapping_known_mask"].T
        obs["mapping_axes"] = "xy"
        np.testing.assert_array_equal(validate_local_mapping(obs).occupancy, result.occupancy)

    def test_bad_mapping_contracts_are_rejected(self):
        mutations = [
            ("mapping_resolution", 2.), ("mapping_resolution", float("nan")),
            ("mapping_resolution", 0.), ("mapping_axes", "camera"),
            ("mapping_occupancy_map", np.full((48, 48), np.nan)),
            ("mapping_occupancy_map", np.full((48, 48), 2)),
            ("mapping_known_mask", np.ones((12, 12))),
            ("mapping_known_mask", np.full((48, 48), .5)),
        ]
        for key, value in mutations:
            with self.subTest(key=key, shape=np.shape(value)):
                obs = observation()
                obs[key] = value
                with self.assertRaises(ValueError):
                    validate_local_mapping(obs)
        obs = observation()
        del obs["mapping_known_mask"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_local_mapping(obs)

    def test_mapper_cache_parameters_and_history_are_consistent(self):
        c = controller()
        c.graph_builder.world_bounds = (0., 0., 100., 100.)
        old = c.graph_builder
        obs = observation()
        prepare_local_mapping(c, obs)
        self.assertIsNot(c.graph_builder, old)
        self.assertEqual(c.resolution, 2.)
        self.assertEqual(c.local_mapping_resolution, .5)
        self.assertEqual(c.semantic_processor.resolution, .5)
        self.assertEqual(c.graph_builder.semantic_processor.resolution, .5)
        self.assertEqual(c.graph_builder.min_clearance_pixels, 3.)
        self.assertEqual(c.graph_builder.max_edge_pixels, 30.)
        self.assertEqual(c.graph_builder.node_spacing_pixels, 15.)
        self.assertEqual(c.graph_builder.world_bounds, old.world_bounds)
        same = c.graph_builder
        prepare_local_mapping(c, obs)
        self.assertIs(c.graph_builder, same)
        legacy = {k: v for k, v in obs.items() if not k.startswith("mapping_")}
        with self.assertRaisesRegex(ValueError, "missing"):
            prepare_local_mapping(c, legacy)
        obs.update(mapping_occupancy_map=np.zeros((96, 96)),
                   mapping_known_mask=np.ones((96, 96)), mapping_resolution=.25)
        with self.assertRaisesRegex(ValueError, "cannot change"):
            prepare_local_mapping(c, obs)
        late = controller()
        late.current_graph = object()
        with self.assertRaisesRegex(ValueError, "before constructing"):
            prepare_local_mapping(late, observation())

    def test_live_controller_preserves_policy_pixels(self):
        c = controller()
        captured = []
        c.navigator = SimpleNamespace(uses_velocity=True, nominal_velocity=np.zeros(3),
                                      _current_target=None,
                                      compute_velocity=lambda p, obs: captured.append(obs))
        obs = observation()
        obs["occupancy_map"][1, 8] = 1
        obs["mapping_occupancy_map"][12, 20] = 1
        obs["mapping_known_mask"][13, 20] = False
        original = obs["occupancy_map"].copy()
        with contextlib.redirect_stdout(io.StringIO()):
            c.get_action(obs)
        np.testing.assert_array_equal(captured[0]["occupancy_map"], original.T)
        np.testing.assert_array_equal(obs["occupancy_map"], original)
        self.assertEqual(c._last_semantic_map.shape, (48, 48))
        self.assertEqual(c._last_semantic_map[20, 12], TerrainType.OBSTACLE.value)
        self.assertEqual(c._last_semantic_map[20, 13], TerrainType.UNKNOWN.value)

    def test_graph_broadcast_state_follows_the_transmission_interval(self):
        c = controller()
        self.assertEqual(c.get_state()['last_broadcast_time'], 0.)
        obs = observation()
        packets = []
        for now, expected_time in ((10., 10.), (11., 10.), (15., 15.)):
            obs['full_state']['time'] = now
            with contextlib.redirect_stdout(io.StringIO()):
                action = c.get_action(obs)
            self.assertEqual(action['broadcast']['timestamp'], now)
            self.assertEqual(c.get_state()['last_broadcast_time'], expected_time)
            packets.append(action['broadcast']['graph'])
        self.assertIs(packets[0], packets[1])
        self.assertIsNot(packets[1], packets[2])

    def test_semantic_observation_broadcast_reports_its_timestamp(self):
        c = controller()
        semantic = np.full((12, 12), TerrainType.TRAVERSABLE.value, np.uint8)
        with patch('drone.drone_main.time.time', return_value=42.), \
                contextlib.redirect_stdout(io.StringIO()):
            result = c.process_observation(dict(position=(12., 12., 15.), semantic_map=semantic))
        self.assertEqual(result['broadcast_data']['timestamp'], 42.)
        self.assertEqual(c.get_state()['last_broadcast_time'], 42.)

    def test_wrapper_separates_mapping_from_legacy_parse_and_keeps_fallback(self):
        class LegacyController:
            def __init__(self):
                self.resolution = 2.
                self.current_graph = None
                self.semantic_processor = SemanticProcessor(2.)
                self.graph_builder = SkeletonGraphBuilder("legacy", resolution=2., min_clearance=1.5)

            def _parse_simulator_observation(self, obs):
                return {"occupancy_map": obs["occupancy_map"].T}

            def _generate_semantic_map(self, raw):
                return self.semantic_processor.process_occupancy_map(raw)

        wrapped = with_local_mapping(LegacyController)()
        obs = observation()
        obs["mapping_known_mask"][3, 8] = False
        parsed = wrapped._parse_simulator_observation(obs)
        self.assertEqual(parsed["occupancy_map"].shape, (12, 12))
        semantic = wrapped._generate_semantic_map(parsed["occupancy_map"])
        self.assertEqual(semantic.shape, (48, 48))
        self.assertEqual(semantic[8, 3], TerrainType.UNKNOWN.value)
        legacy = {k: v for k, v in obs.items() if not k.startswith("mapping_")}
        fresh = with_local_mapping(LegacyController)()
        parsed = fresh._parse_simulator_observation(legacy)
        self.assertEqual(fresh._generate_semantic_map(parsed["occupancy_map"]).shape, (12, 12))
        self.assertIs(with_local_mapping(DroneController), DroneController)


if __name__ == "__main__":
    unittest.main()
