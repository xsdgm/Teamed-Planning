"""Behavioral boundaries, physical navigation and genuine optimization checks."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from world_model.ab_environment import SurveyNavigationEnv
from world_model.ab_mapping import MappingConfig, ProxyMapper, make_task, task_valid
from world_model.ab_model import WorldModel
from world_model.ab_navigation import ReceivedMap
from world_model.ab_pipeline import collect_episode
from world_model.ab_planner import ModelPlanner, candidates
from world_model.ab_training import Windows, load_episodes, load_model, train


def config(**options):
    return replace(MappingConfig(world=200., delay_jitter=0., failure_probability=0.,
                                 distance_failure_probability=0., communication_failure_probability=0.), **options)


def scene(circles=(), seed=4):
    return dict(world=200., seed=seed, start=[30., 100.], goal=[170., 100.], circles=list(circles))


class MappingTests(unittest.TestCase):
    def test_only_requested_cells_are_delivered(self):
        cfg = config()
        mapper = ProxyMapper([([65., 100.], 3.)], cfg, 1)
        task = make_task([30., 100.], [110., 100.], cfg)
        packet = mapper.survey(0, task, np.array([30., 100.]))
        self.assertTrue(np.array_equal(packet.cell_ids, task.cell_ids))
        self.assertTrue(packet.occupied.any())
        received = ReceivedMap(cfg); received.ingest(packet)
        outside = np.ones(received.n**2, bool); outside[task.cell_ids] = False
        self.assertTrue((received.cells.ravel()[outside] == -1).all())

    def test_failure_delivers_only_partial_map_and_takes_time(self):
        cfg = config(failure_probability=1.)
        mapper = ProxyMapper([], cfg, 1)
        task = make_task([30., 100.], [110., 100.], cfg)
        packet = mapper.survey(0, task, np.array([30., 100.]))
        self.assertEqual(packet.status, 'partial_failure')
        self.assertGreater(len(packet.cell_ids), 0)
        self.assertLess(len(packet.cell_ids), task.cell_count)
        self.assertGreater(packet.elapsed_seconds, 0)

    def test_communication_loss_exposes_no_unreceived_cells(self):
        cfg = config(communication_failure_probability=1.)
        mapper = ProxyMapper([], cfg, 1)
        packet = mapper.survey(0, make_task([30., 100.], [110., 100.], cfg), np.array([30., 100.]))
        self.assertEqual(packet.status, 'communication_failure')
        self.assertEqual(len(packet.cell_ids), 0)

    def test_timeout_is_partial_and_capped(self):
        cfg = config(timeout_seconds=1.)
        mapper = ProxyMapper([], cfg, 1)
        task = make_task([30., 100.], [110., 100.], cfg)
        packet = mapper.survey(0, task, np.array([30., 100.]))
        self.assertEqual(packet.status, 'timeout')
        self.assertEqual(packet.elapsed_seconds, 1.)
        self.assertLess(len(packet.cell_ids), task.cell_count)

    def test_repeat_delivery_is_not_new_information(self):
        cfg = config(); mapper = ProxyMapper([], cfg, 1); received = ReceivedMap(cfg)
        task = make_task([30., 100.], [110., 100.], cfg)
        first = mapper.survey(0, task, np.array([30., 100.]))
        self.assertGreater(received.ingest(first), 0)
        self.assertEqual(received.ingest(first), 0)
        self.assertEqual(received.ingest(mapper.survey(1, task, np.array([30., 100.]))), 0)

    def test_two_independent_endpoints_and_200m_band(self):
        cfg = replace(config(), world=1000.)
        self.assertTrue(task_valid(make_task([500., 500.], [700., 500.], cfg), [500., 500.], cfg))
        self.assertFalse(task_valid(make_task([500., 500.], [720., 500.], cfg), [500., 500.], cfg))
        choices = candidates([500., 500.], [900., 500.], cfg)
        self.assertTrue(any(np.linalg.norm(a-[500., 500.]) > 1. for a, b in choices))
        self.assertTrue(any(np.linalg.norm(b-a) > 120. for a, b in choices))


class NavigationTests(unittest.TestCase):
    def test_unknown_space_is_not_navigable(self):
        received = ReceivedMap(config())
        path, connected = received.search([30., 100.], [170., 100.])
        self.assertEqual(len(path), 1)
        self.assertFalse(connected.any())

    def test_complete_survey_does_not_manufacture_path_through_wall(self):
        circles = [([80., y], 3.) for y in np.arange(1., 200., 4.)]
        env = SurveyNavigationEnv(scene(circles), config())
        _, _, info = env.step([[30., 100.], [170., 100.]])
        self.assertEqual(info['status'], 'complete')
        for _ in range(3):
            env.step([[30., 100.], [170., 100.]])
        self.assertLess(env.position[0], 80.)
        self.assertFalse(env.done)

    def test_open_route_moves_with_bounded_speed_after_wait(self):
        cfg = config(); env = SurveyNavigationEnv(scene(), cfg)
        _, _, info = env.step([[30., 100.], [170., 100.]])
        self.assertEqual(info['survey_ground_displacement_m'], 0.)
        self.assertLessEqual(info['vehicle_distance_m'], cfg.ground_speed*cfg.navigation_seconds)
        self.assertAlmostEqual(info['elapsed_seconds'], info['survey_seconds']+info['vehicle_distance_m']/cfg.ground_speed)
        self.assertGreater(info['goal_progress_m'], 20.)

    def test_hidden_terrain_cannot_change_initial_observation(self):
        first = SurveyNavigationEnv(scene(), config())
        second = SurveyNavigationEnv(scene([([150., 150.], 10.)]), config())
        for a, b in zip(first.observe(), second.observe()):
            self.assertTrue(np.array_equal(a, b))
        torch.manual_seed(8); model = WorldModel(hidden=8, latent=4)
        a, _ = ModelPlanner(model, config()).choose(first.observe())
        b, _ = ModelPlanner(model, config()).choose(second.observe())
        self.assertTrue(np.array_equal(a, b))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def dataset(self, root):
        for split, seed in [('train', 1), ('val', 2)]:
            item = scene(seed=seed)
            item['goal'] = [170., 90.+seed]
            arrays, _, _ = collect_episode(item, config(), 3, 1000.)
            arrays['split'] = np.asarray(split)
            np.savez_compressed(root/f'{split}_{seed}.npz', **arrays)

    def test_padding_does_not_create_training_transitions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.dataset(root)
            episodes, _ = load_episodes(root)
            window = Windows(episodes['train'], length=8)[0]
            self.assertEqual(window['mask'].sum(), 3)
            for key in ('maps', 'states'):
                self.assertTrue(np.array_equal(window[key][-1], window[key][3]))

    def test_split_geometry_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.dataset(root)
            with np.load(root/'train_1.npz') as src:
                data = {k: src[k] for k in src.files}
            data['split'] = np.asarray('val'); data['seed'] = np.asarray(2)
            np.savez_compressed(root/'val_2.npz', **data)
            with self.assertRaisesRegex(ValueError, 'geometry'):
                load_episodes(root)

    def test_real_update_checkpoint_reload_and_learned_decision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.dataset(root)
            trained, result = train(root, root/'training', epochs=2, model=WorldModel(hidden=8, latent=4))
            self.assertGreater(result['optimizer_steps'], 0)
            self.assertGreater(result['parameter_delta_l2'], 0.)
            loaded, metadata = load_model(root/'training/best.pt')
            for a, b in zip(trained.parameters(), loaded.parameters()):
                self.assertTrue(torch.equal(a, b))
            env = SurveyNavigationEnv(scene(seed=5), config())
            action, diagnostic = ModelPlanner(loaded, config()).choose(env.observe())
            self.assertEqual(action.shape, (2, 2))
            self.assertEqual(diagnostic['selector'], 'world_model_mpc')
            env.step(action)


if __name__ == '__main__':
    unittest.main()
