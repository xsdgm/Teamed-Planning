"""Survey task reception, vehicle semantics and v9 actor migration contracts."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from drone.communication import ChainState
from drone.cooperative_mappo import CooperativeNavigator
from drone.drone_main import DroneController
from drone.mappo import ActorCritic, legal_observation
from drone.relay_attention_mappo import RelayActorCritic
from drone.survey_mappo import (
    SurveyActorCritic, survey_features, SURVEY_FEATURES, RAW_WIDTH,
    RELAY_RAW_WIDTH, RELAY_ENCODED_WIDTH,
)


def observation(a=(80., 50.), b=(180., 50.), c=(30., 50.), now=5.):
    return dict(graph=None, time=now, occupancy_map=np.zeros((12, 12)),
        ground_robots={'ground_0': dict(position=[*c, 0.], target=list(b),
            planned_path=[], source_time=5., direct=True,
            survey=dict(a=list(a), b=list(b), half_width=12.))},
        nearby_drones={}, neighbor_waypoints={}, relay=[],
        cooperation=dict(peers={}, velocities={}, ground=np.array([*c, 0.]),
                         required=('__ground__',)))


def features(model, obs=None):
    navigator = CooperativeNavigator('drone_0', total_drones=3, comm_range=95.,
        min_separation=15., model=model, max_speed=6., world_bounds=(0., 0., 1000., 1000.))
    navigator.cbf_margin_source = lambda: None
    return navigator.encode(np.array([50., 50., 15.]), observation() if obs is None else obs)[0]


class SurveyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_independent_a_changes_only_task_suffix_and_keeps_real_vehicle_c(self):
        model = SurveyActorCritic()
        first = features(model, observation(a=(80., 50.)))
        second = features(model, observation(a=(100., 90.)))
        self.assertEqual(first.shape, (RAW_WIDTH,))
        np.testing.assert_array_equal(first[:RELAY_RAW_WIDTH], second[:RELAY_RAW_WIDTH])
        self.assertFalse(np.array_equal(first[-SURVEY_FEATURES:], second[-SURVEY_FEATURES:]))
        np.testing.assert_allclose(first[6:8], [(30.-50.)/95., 0.])
        original = features(RelayActorCritic(), observation(a=(80., 50.)))
        np.testing.assert_array_equal(first[:RELAY_RAW_WIDTH], original)

    def test_a_equal_c_is_a_valid_task_without_changing_vehicle_features(self):
        model = SurveyActorCritic()
        encoded = features(model, observation(a=(30., 50.)))
        np.testing.assert_allclose(encoded[-8:-6], [(30.-50.)/350., 0.])
        np.testing.assert_allclose(encoded[6:8], [(30.-50.)/95., 0.])
        self.assertEqual(encoded[-1], 1.)

    def test_missing_expired_and_future_commands_are_masked(self):
        p = (50., 50., 15.)
        obs = observation()
        obs['ground_robots']['ground_0'].pop('survey')
        np.testing.assert_array_equal(survey_features(p, obs), np.zeros(SURVEY_FEATURES))
        for now in (16., 3.):
            np.testing.assert_array_equal(survey_features(p, observation(now=now)),
                                          np.zeros(SURVEY_FEATURES))
        self.assertEqual(survey_features(p, observation(now=15.))[-1], 1.)

    def test_malformed_corridor_does_not_enter_actor(self):
        for change in ({'a': [1.]}, {'b': [1., float('nan')]}, {'half_width': 0.},
                       {'half_width': True}, {'half_width': float('inf')}):
            with self.subTest(change=change):
                obs = observation()
                obs['ground_robots']['ground_0']['survey'].update(change)
                with self.assertRaises(ValueError):
                    survey_features((50., 50., 15.), obs)

    def test_v9_warm_start_preserves_actions_and_leaves_critic_popart_fresh(self):
        torch.manual_seed(19)
        source = RelayActorCritic()
        with torch.no_grad():
            source.value_mean.fill_(17.)
            source.value_std.fill_(9.)
            for parameter in source.critic.parameters():
                parameter.fill_(.125)
        target = SurveyActorCritic()
        fresh = {key: value.clone() for key, value in target.state_dict().items()
                 if key.startswith(('critic.', 'value_'))}
        report = target.warm_start_actor(source.state_dict())
        self.assertIn('gat_actor.1.weight', report['partial'])
        for key, value in fresh.items():
            torch.testing.assert_close(target.state_dict()[key], value, rtol=0., atol=0.)
        torch.testing.assert_close(target.gat_actor[1].weight[:, RELAY_ENCODED_WIDTH:],
                                   torch.zeros_like(target.gat_actor[1].weight[:, RELAY_ENCODED_WIDTH:]))
        x = torch.tensor(features(target)).unsqueeze(0)
        before, after = source.distribution(x[:, :RELAY_RAW_WIDTH]), target.distribution(x)
        torch.testing.assert_close(before.mean, after.mean, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(before.stddev, after.stddev, rtol=0., atol=0.)
        target.gat_actor(x).sum().backward()
        self.assertGreater(float(target.gat_actor[1].weight.grad[:, RELAY_ENCODED_WIDTH:].abs().sum()), 0.)

    def test_migration_rejects_incompatible_actor_and_relay_mode(self):
        model = SurveyActorCritic()
        bad = RelayActorCritic().state_dict()
        bad['gat_actor.1.weight'] = torch.zeros(128, 1)
        with self.assertRaisesRegex(ValueError, 'first-layer shape'):
            model.warm_start_actor(bad)
        with self.assertRaisesRegex(ValueError, 'relay_attention mode'):
            model.warm_start_actor(RelayActorCritic(relay_attention='age').state_dict())

    def test_generic_worker_loader_roundtrips_survey_model_and_critic(self):
        model = SurveyActorCritic(global_features=10)
        model.warm_start_actor(RelayActorCritic().state_dict())
        restored = ActorCritic.from_state_dict(copy.deepcopy(model.state_dict()))
        self.assertIsInstance(restored, SurveyActorCritic)
        self.assertEqual(restored.feature_version, 11)
        self.assertEqual(int(restored.survey_schema), 1)
        x = torch.tensor(np.stack([features(model)]*3))
        state = torch.zeros(10)
        torch.testing.assert_close(model.distribution(x).mean, restored.distribution(x).mean)
        torch.testing.assert_close(model.value(x, state=state), restored.value(x, state=state))
        old = ActorCritic.from_state_dict(RelayActorCritic().state_dict())
        self.assertIsInstance(old, RelayActorCritic)
        self.assertNotIsInstance(old, SurveyActorCritic)

    def test_survey_command_travels_with_received_missions_over_multiple_hops(self):
        controller = object.__new__(DroneController)
        controller.drone_id, controller.multi_vehicle = 'drone_0', False
        mission = dict(robot_id='ground_0', position=[30., 50., 0.], target=[180., 50.],
                       timestamp=5., survey=dict(a=[80., 50.], b=[180., 50.], half_width=12.))
        raw = dict(self=dict(position=[50., 50.], height=15.),
                   occupancy_map=np.zeros((12, 12)), full_state=dict(time=5.),
                   neighbors=[dict(type='diff_drive', broadcast=mission)])
        received = controller._parse_simulator_observation(raw)
        for step in range(3):
            legal = legal_observation(dict(received, time=5.+step))
            chain = ChainState('drone_0', 1)
            chain.update(legal)
            self.assertEqual(chain.task()['survey'], mission['survey'])
            self.assertEqual(chain.task()['position'], tuple(mission['position']))
            np.testing.assert_allclose(survey_features((50., 50., 15.), legal)[-1], 1.)
            raw['neighbors'] = [dict(type='drone', broadcast=dict(drone_id='relay',
                position=[70., 50., 15.], ground_missions=received['ground_robots']))]
            received = controller._parse_simulator_observation(raw)
        self.assertFalse(received['ground_robots']['ground_0']['direct'])
        self.assertEqual(received['ground_robots']['ground_0']['source_time'], 5.)


if __name__ == '__main__':
    unittest.main()
