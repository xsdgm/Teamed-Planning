"""CBF provenance and causality across the graph/world/actor boundary."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from world_model.graph_fast_policy import graph_features, PREDICTION_WIDTH, TOKEN_WIDTH
from world_model.safety_context import (CBF_FEATURE_INDEX, normalized_margin,
    observation_graph, received_cbf_context, validate_cbf_context)
from world_model.residual_world import infer_graph
from world_model.telemetry import HISTORY, HISTORY_FEATURES, CONTEXT_FEATURES, packet


def received_fixture(margin=1.25):
    chain = SimpleNamespace(required=('drone_1',),
        task=lambda: dict(position=[20., 5.], source_time=2.))
    nav = SimpleNamespace(min_separation=3., world_bounds=[0., 100., 0., 100.],
        required_links=('drone_1',), chain=chain, ground_robot_position=[12., 4.])
    controller = SimpleNamespace(drone_id='drone_0', position=[10., 5., 3.],
        navigator=nav, max_speed=6., communication_range=95., ground_max_speed=2.,
        communication_mode='chain', multi_vehicle=False)
    observation = dict(time_step=1., full_state=dict(time=8.,
        robots={'hidden_unreceived_peer': {'position': [80., 80.]}}),
        neighbors=[dict(id=1, type='drone', relative_position=[4., 0., 0.],
            broadcast=dict(drone_id='drone_1', timestamp=3.))])
    context = received_cbf_context(controller, observation,
        dict(feasibility_margin=margin, missing_links=[]))
    return controller, observation, context


def graph_fixture(peers=True):
    controller, _, cbf = received_fixture()
    features = np.linspace(-.1, .1, 1826, dtype=np.float32)
    features[CBF_FEATURE_INDEX] = cbf['cbf_margin_norm']
    prepared = []
    if peers:
        last = packet('drone_1', 3., [14., 5., 3.], [1., 0., 0.], [1., 0., 0.],
                      goal=[20., 5.], goal_stamp=2.)
        last['hop'] = 2
        entry = dict(observer='drone_0', target='drone_1', frame=8., source_stamp=3.,
            origin=np.asarray([14., 5.]),
            history=np.zeros((HISTORY, HISTORY_FEATURES), np.float32),
            history_mask=np.ones(HISTORY, bool),
            context=np.zeros(CONTEXT_FEATURES, np.float32),
            baseline_states=np.zeros((5, 4), np.float32),
            baseline_actions=np.zeros((4, 2), np.float32))
        prepared.append(dict(entry=entry, packet=last))
    return observation_graph('drone_0', 8., controller.position, features, cbf, prepared)


class StrictWorld(torch.nn.Module):
    """Only declared inference fields are accepted; labels cannot enter forward."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, history, history_mask, context, baseline_states, baseline_actions):
        self.calls += 1
        return dict(states=baseline_states + .1, state_std=torch.full_like(baseline_states, .2),
                    actions=baseline_actions + .3, action_std=torch.full_like(baseline_actions, .4))


class CBFContextTests(unittest.TestCase):
    def test_constraint_free_and_infeasible_are_distinct_and_json_safe(self):
        for margin, normalized, status in [(None, 1., 'unconstrained'),
                                          (-float('inf'), -1., 'infeasible'),
                                          (-3., -.5, 'infeasible'), (12., 1., 'ok')]:
            with self.subTest(margin=margin):
                _, _, context = received_fixture(margin)
                restored = json.loads(json.dumps(context, allow_nan=False))
                self.assertEqual(validate_cbf_context(restored, 'drone_0', 8.), normalized)
                self.assertEqual(restored['calculation_status'], status)
                self.assertEqual(restored['has_constraints'], margin is not None)
        for invalid in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                normalized_margin(invalid, 6.)
        for invalid in (0., -1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                normalized_margin(1., invalid)

    def test_geometry_version_uses_received_geometry_and_preserves_source_time(self):
        controller, observation, context = received_fixture()
        self.assertEqual(context['computed_at'], 8.)
        self.assertEqual(context['evidence']['task_source_stamp'], 2.)
        self.assertEqual(context['evidence']['neighbor_broadcasts'][0]['source_stamp'], 3.)
        observation['full_state']['robots']['hidden_unreceived_peer']['position'] = [-1000., -1000.]
        unseen = received_cbf_context(controller, observation, dict(feasibility_margin=1.25))
        self.assertEqual(context['geometry_version'], unseen['geometry_version'])
        observation['neighbors'][0]['relative_position'][0] += 1.
        moved = received_cbf_context(controller, observation, dict(feasibility_margin=1.25))
        self.assertNotEqual(context['geometry_version'], moved['geometry_version'])
        controller.max_speed = 5.
        changed_config = received_cbf_context(controller, observation, dict(feasibility_margin=1.25))
        self.assertNotEqual(moved['geometry_version'], changed_config['geometry_version'])

    def test_unknown_failed_stale_and_estimated_context_cannot_be_actor_margin(self):
        _, _, context = received_fixture()
        for field, value in [('owner', 'drone_1'), ('computed_at', 7.),
                             ('geometry_source', 'world_model_prediction'),
                             ('calculation_status', 'failed'), ('calculation_status', 'unknown'),
                             ('has_constraints', False), ('cbf_margin_norm', 1.),
                             ('geometry_version', '')]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_cbf_context(dict(context, **{field: value}), 'drone_0', 8.)


class GraphBoundaryTests(unittest.TestCase):
    def test_world_preserves_received_cbf_and_peer_evidence_without_label_leakage(self):
        graph = graph_fixture()
        world = StrictWorld()
        belief = infer_graph(world, graph)
        self.assertEqual(world.calls, 1)
        self.assertEqual(belief['phase'], 'belief')
        self.assertEqual(belief['estimate_stamp'], 8.)
        self.assertEqual(belief['nodes']['drone_0']['cbf'], graph['nodes']['drone_0']['cbf'])
        self.assertIsNone(belief['nodes']['drone_1']['cbf'])
        self.assertEqual(belief['nodes']['drone_1']['source_stamp'], 3.)
        self.assertEqual(belief['nodes']['drone_1']['packet']['stamp'], 3.)
        self.assertEqual(belief['nodes']['drone_1']['packet']['hop'], 2)
        self.assertNotIn('prediction', graph['nodes']['drone_1'])
        graph['nodes']['drone_1']['world_input']['target_states'] = np.full((5, 4), np.nan)
        graph['nodes']['drone_1']['world_input']['target_actions'] = np.full((4, 2), 1e20)
        poisoned_labels = infer_graph(world, graph)
        for key, prediction in belief['nodes']['drone_1']['prediction'].items():
            np.testing.assert_array_equal(prediction, poisoned_labels['nodes']['drone_1']['prediction'][key])
        graph['nodes']['drone_0']['cbf']['computed_at'] = 99.
        self.assertEqual(belief['nodes']['drone_0']['cbf']['computed_at'], 8.)

    def test_actor_receives_original_margin_and_received_packet_ages(self):
        graph = graph_fixture()
        belief = infer_graph(StrictWorld(), graph)
        features = graph_features(belief)
        np.testing.assert_array_equal(features[:1826], graph['nodes']['drone_0']['actor_features'])
        self.assertEqual(features.shape, (1826 + PREDICTION_WIDTH,))
        np.testing.assert_allclose(features[1826 + TOKEN_WIDTH - 5:1826 + TOKEN_WIDTH],
                                   [5. / 30., 2. / 8., 6. / 30., 1., 1.])
        self.assertTrue((features[1826 + TOKEN_WIDTH:] == 0).all())
        belief['nodes']['drone_0']['actor_features'][CBF_FEATURE_INDEX] = -.9
        with self.assertRaises(ValueError):
            graph_features(belief)

    def test_first_frame_without_peers_still_validates_and_passes_ego_cbf(self):
        graph = graph_fixture(peers=False)
        world = StrictWorld()
        belief = infer_graph(world, graph)
        self.assertEqual(world.calls, 0)
        features = graph_features(belief)
        np.testing.assert_array_equal(features[:1826], graph['nodes']['drone_0']['actor_features'])
        self.assertTrue((features[1826:] == 0).all())
        stale = copy.deepcopy(graph)
        stale['estimate_stamp'] = 9.
        with self.assertRaises(ValueError):
            infer_graph(world, stale)

    def test_actor_rejects_observation_or_wrong_schema(self):
        graph = graph_fixture(peers=False)
        with self.assertRaises(ValueError):
            graph_features(graph)
        belief = infer_graph(StrictWorld(), graph)
        belief['schema'] = 'unrecognized'
        with self.assertRaises(ValueError):
            graph_features(belief)


if __name__ == '__main__':
    unittest.main(verbosity=2)
