"""Kilometre task scale must refer to endpoints and whole-mission outcomes."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests.world_model.vtmp3.test_wildscenes import fixture
from world_model.ab_environment import MappingConfig
from world_model.ab_wildscenes import WildScenesSource, load_manifest
from world_model.decoupled_observation import ReceivedObservation
from world_model.decoupled_pipeline import navigation_summary, task_step_budget
from world_model.decoupled_scenes import generate_scene, load_scene
from world_model.decoupled_planner import IndependentGroundManager, TerrainPlanner
from world_model.decoupled_observation import SIDE, crop_origin


class KilometerTests(unittest.TestCase):
    def test_metric_task_over_one_kilometer_with_different_split_world_sizes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); _, manifest = fixture(root)
            spec = json.loads(manifest.read_text())
            spec['world_size_m'] = 1000.
            spec['train'][0].update(origin_xy=[0., 0.], world_size_m=1000.)
            spec['val'][0].update(origin_xy=[1000., 0.], world_size_m=200.)
            spec['test'][0].update(origin_xy=[0., 1000.], world_size_m=200.)
            manifest.write_text(json.dumps(spec))
            provider = WildScenesSource(manifest, MappingConfig(world=1000., clearance=1.5))
            record = generate_scene(provider, 3, 'train', root, distance_range=(1000., 1200.))
            scene, certificate = load_scene(record['scene_path'], record['scene_sha256'])
            self.assertGreaterEqual(np.linalg.norm(np.asarray(scene['goal'])-scene['start']), 1000.)
            self.assertLessEqual(scene['distance'], 1200.)
            self.assertGreaterEqual(certificate['path_length_m'], scene['distance'])
            self.assertEqual(scene['coordinate_scale'], 1.)
            self.assertEqual(provider.spec['val'][0]['world_size_m'], 200.)

    def test_region_overlap_uses_each_regions_actual_size(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); _, manifest = fixture(root)
            spec = json.loads(manifest.read_text())
            spec['train'][0]['world_size_m'] = 1000.
            # Original validation at (600, 0) now lies inside training.
            manifest.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                load_manifest(manifest)

    def test_small_region_rejects_impossible_kilometer_request(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); _, manifest = fixture(root)
            provider = WildScenesSource(manifest, MappingConfig(world=600., clearance=1.5))
            with self.assertRaisesRegex(ValueError, 'cannot fit'):
                provider.scene(3, 'train', distance_range=(1000., 1200.))

    def test_received_memory_survives_multiple_local_windows(self):
        memory = ReceivedObservation((0., 0., 2000., 2000.))
        memory.receive([], 0.)
        memory.add_sensor((10., 10.), np.ones((16, 16)), .5)
        initial, _, _ = memory.encode((10., 10.), 0., (0., 0.), (1500., 10.))
        far, states, _ = memory.encode((1250., 10.), 0., (0., 0.), (1500., 10.))
        self.assertGreater(initial[1].sum(), 0)
        self.assertEqual(far[1].sum(), 0)
        returned, _, _ = memory.encode((10., 10.), 0., (0., 0.), (1500., 10.))
        np.testing.assert_array_equal(returned, initial)
        np.testing.assert_allclose(states[6:8], (1.5, .01))

    def test_budget_scales_with_physical_path_and_explicit_smoke_cap(self):
        self.assertEqual(task_step_budget(1200., 12000), 5040)
        self.assertEqual(task_step_budget(1200., 60), 60)
        self.assertEqual(task_step_budget(50., 12000), 600)
        with self.assertRaises(ValueError):
            task_step_budget(np.nan, 12000)

    def test_accumulated_short_task_distance_does_not_count_as_kilometer_navigation(self):
        rows = [dict(initial_distance_m=200., ground_path_length_m=1500., goal_reached=True,
                     kilometer_success=False, elapsed_seconds=2000., selector_mode='world_model_independent'),
                dict(initial_distance_m=1100., ground_path_length_m=50., goal_reached=False,
                     kilometer_success=False, elapsed_seconds=80., selector_mode='world_model_independent')]
        summary = navigation_summary(rows)
        self.assertEqual(summary['kilometer_tasks'], 1)
        self.assertEqual(summary['kilometer_successes'], 0)
        self.assertEqual(summary['kilometer_success_rate'], 0.)

    def test_subgoal_arrivals_do_not_erase_mission_loop_history(self):
        planner = TerrainPlanner()
        manager = IndependentGroundManager((1500., 0.), planner=planner)
        manager.set_observation((0., 0.), np.ones((16, 16)), .5)
        for now in range(0, 101, 10):
            manager.update((0., 0.), None, float(now))
        self.assertTrue(manager._loop_recovery)
        near = SimpleNamespace(point=np.array([1., 0.]), cost=0.)
        farther = SimpleNamespace(point=np.array([0., 10.]), cost=0.)
        with patch.object(planner, 'strategic_cost', return_value=100.):
            self.assertGreater(manager._score(near), manager._score(farther))
        manager.set_observation((20., 0.), np.ones((16, 16)), .5)
        manager.update((20., 0.), None, 110.)
        self.assertFalse(manager._loop_recovery)

    def test_coarse_occupied_samples_do_not_invent_an_impassable_wall(self):
        planner = TerrainPlanner()
        planner.model = object()  # Exercise the learned continuation scorer.
        maps = np.zeros((3, SIDE, SIDE), np.float32)
        maps[:2, :, SIDE//2] = 1.
        planner.observe(maps, np.zeros(50), crop_origin((0., 0.)), 0.)
        prediction = {name: np.zeros((1, SIDE, SIDE)) for name in ('occupancy', 'entropy', 'disagreement')}
        with patch.object(planner, 'predict', return_value=prediction):
            cost = planner.ground_cost(SimpleNamespace(point=np.array([-10., 0.]), cost=0.), (100., 0.))
        self.assertLess(cost, 300.)

    def test_bootstrap_information_scale_allows_exploration_away_from_goal(self):
        memory = ReceivedObservation((-300., -300., 300., 300.))
        memory.receive([], 0.)
        maps, states, origin = memory.encode((0., 0.), 0., (0., 0.), (1500., 0.))
        planner = TerrainPlanner(seed=0)
        planner.observe(maps, states, origin, 0.)
        first = planner.choose_survey((0., 0.), (1500., 0.), None, 200., memory.bounds, memory, 0.)
        later = planner.choose_survey((0., 0.), (1500., 0.), None, 200., memory.bounds, memory, 160.)
        self.assertGreater(first[0], 0.)
        self.assertLess(later[0], 0.)

    def test_vtmp_recovery_keeps_certified_clearance(self):
        from tests.world_model.vtmp2.test_decoupled import RuntimeTests
        controller = RuntimeTests().controller()
        controller.local_planner.escape_distance = .5
        controller.set_goal((200., 0.))
        self.assertEqual(controller.local_planner.escape_distance, controller.local_planner.safety_distance)


if __name__ == '__main__':
    unittest.main()
