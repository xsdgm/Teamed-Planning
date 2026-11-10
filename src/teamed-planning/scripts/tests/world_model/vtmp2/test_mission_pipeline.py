"""Mission collection preserves executed drive/survey semantics and evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from world_model import data
from world_model import mission_pipeline


class FakeMissionEnvironment:
    def __init__(self, *, purpose='drive', changes=None, graph_growth=True,
                 survey_observed=False, target=(100., 0.)):
        self.env = SimpleNamespace(time=0.)
        self.changes = dict(changes or {})
        self.graph_growth = graph_growth
        self.survey_evidence = survey_observed
        self.subgoals = SimpleNamespace(target=np.asarray(target, dtype=float), purpose=purpose,
            target_status='unverified', last_reason='initial', last_decision={},
            selector_mode='bootstrap_drive_survey', survey_observed=self.observed)
        self.ground_controller = SimpleNamespace(position=np.zeros(2),
                                                 unified_graph=SimpleNamespace(graph=nx.Graph()))
        self.goal = np.array([500., 0.])
        self.initial_goal_distance = 500.
        self.scene_geometry_sha256 = 'b'*64
        self.goal_reached = False
        self.executed = []

    def observed(self, target, graph):
        return self.survey_evidence

    def step(self, frame):
        purpose = self.subgoals.purpose
        self.executed.append((purpose, self.subgoals.target.copy()))
        self.env.time += 1.
        if purpose == 'drive':
            self.ground_controller.position[0] += .1
        elif self.graph_growth:
            self.ground_controller.unified_graph.graph.add_node(frame)
        if frame in self.changes:
            # Deliberately keep coordinates unchanged: purpose alone changes
            # what the cached command did and must split the labelled sample.
            self.subgoals.purpose = self.changes[frame]
            self.subgoals.last_reason = 'mission_changed'


def encode_observation(controller, goal, now, velocity):
    maps = np.zeros((data.CHANNELS, data.SIDE, data.SIDE), np.float32)
    maps[0, 0, 0] = len(controller.unified_graph.graph)/100.
    states = np.zeros(data.FEATURES, np.float32)
    states[:2] = controller.position/1000.
    states[4:6] = velocity
    states[6:8] = goal/1000.
    return maps, states


class MissionCollectionTests(unittest.TestCase):
    def collect(self, env, directory, steps):
        job = dict(seed=11, split='train', steps=steps, output=directory,
                   scene_path='unused-certified-scene.json', scene_sha256='a'*64)
        with patch.object(mission_pipeline, 'provenance', return_value={}), \
             patch.object(mission_pipeline, 'build_environment', return_value=(env, RuntimeError)), \
             patch.object(mission_pipeline, 'encode', side_effect=encode_observation):
            result = mission_pipeline.collect_mission_episode(job)
        with np.load(Path(directory)/'train_11.npz', allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        return result, arrays

    def test_purpose_switch_at_same_coordinates_splits_executed_actions(self):
        env = FakeMissionEnvironment(changes={2: 'survey', 12: 'drive'})
        with TemporaryDirectory() as directory:
            summary, saved = self.collect(env, directory, 20)
        np.testing.assert_array_equal(saved['times'], [0., 3., 13., 20.])
        np.testing.assert_array_equal(saved['mission_purpose'], ['drive', 'survey', 'drive'])
        np.testing.assert_array_equal(saved['actions'][:, 2], [1., -1., 1.])
        np.testing.assert_allclose(saved['elapsed'], [.3, 1., .7])
        np.testing.assert_array_equal(saved['target_switched'], [True, True, False])
        np.testing.assert_array_equal(saved['switch_reason'], ['mission_changed', 'mission_changed', ''])
        np.testing.assert_array_equal(saved['targets'], np.tile([100., 0.], (3, 1)))
        for i, (start, end) in enumerate(((0, 3), (3, 13), (13, 20))):
            for purpose, target in env.executed[start:end]:
                self.assertEqual(saved['mission_purpose'][i], purpose)
                np.testing.assert_array_equal(saved['targets'][i], target)
                self.assertEqual(saved['actions'][i, 2], 1. if purpose == 'drive' else -1.)
        self.assertEqual(saved['schema_version'].item(), 3)
        self.assertEqual(saved['action_contract'].item(), 'drive_survey_v1')
        self.assertEqual(len(saved['maps']), len(saved['actions'])+1)
        # The vehicle stayed still for ten survey steps, but real graph evidence
        # arrived. This is productive surveying, not a vehicle-stall label.
        np.testing.assert_array_equal(saved['states'][1, :2], saved['states'][2, :2])
        self.assertGreater(saved['maps'][2, 0, 0, 0], saved['maps'][1, 0, 0, 0])
        np.testing.assert_array_equal(saved['labels'][1], [0., 0., 0.])
        self.assertEqual(summary['survey_steps'], 10)
        self.assertEqual(summary['drive_steps'], 10)
        self.assertEqual(summary['max_survey_ground_displacement_m'], 0.)

    def test_survey_completion_requires_received_evidence_and_is_not_vehicle_arrival(self):
        cases = ((False, False, (0., 0.), 0., 1.),
                 (True, False, (100., 0.), 0., 0.),
                 (False, True, (100., 0.), 1., 0.))
        for growth, observed, target, completed, stalled in cases:
            with self.subTest(growth=growth, observed=observed), TemporaryDirectory() as directory:
                env = FakeMissionEnvironment(purpose='survey', graph_growth=growth,
                                            survey_observed=observed, target=target)
                summary, saved = self.collect(env, directory, 10)
                self.assertEqual(saved['labels'][0, 0], completed)
                self.assertEqual(saved['labels'][0, 1], stalled)
                self.assertFalse(saved['terminated'][0])
                self.assertTrue(saved['truncated'][0])
                self.assertEqual(summary['reason'], 'budget')
                np.testing.assert_array_equal(saved['states'][:, :2], np.zeros((2, 2)))
                self.assertFalse(env.goal_reached)


if __name__ == '__main__':
    unittest.main()
