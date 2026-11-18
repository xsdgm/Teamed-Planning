"""Controller observations, actor features and policy drift."""
import contextlib
import io
import json
import sys
import unittest
import unittest.mock
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from drone.mappo import ActorCritic, GLOBAL_FEATURES
from drone.velocity_mappo import VelocityNavigator, STATE_FEATURES, SENSOR_SIZE
from drone.graph_attention_mappo import GraphAttentionActorCritic, prefix_width
from drone.relay_attention_mappo import RAW_WIDTH, RelayActorCritic
from simulator.mappo_experiment import Experiment
from simulator.ppo_batch import policy_drift
from simulator.train_mappo import validate_config, run_episode, fixed_schedule, make_model

CONFIGS = Path(__file__).resolve().parents[3]/'config/algorithm'


def config():
    return validate_config(json.loads((CONFIGS/'mappo_training_v9.json').read_text()))


def stage_one(c):
    return dict(c, task_distance_range=[100., 283.8083151964686], max_task_distance=750.,
                distance_sampling='annulus_area')


class ActorInputTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(62)
        torch.set_num_threads(1)

    def test_config_validation(self):
        c = config()
        for delta in [dict(episodes_per_batch=0), dict(episodes_per_batch=True),
                      dict(snapshot_interval=24), dict(validation_interval=40),
                      dict(actor_input_version=3), dict(actor_input_version=True),
                      dict(safety_filter='legacy'), dict(rollout_mode='asynchronous')]:
            with self.assertRaises(ValueError, msg=str(delta)):
                validate_config(dict(c, **delta))
        self.assertIsNone(validate_config(dict(c, episodes_per_batch=None))['episodes_per_batch'])

    def test_checkpoint_widths_round_trip(self):
        for version in (1, 2):
            gat = GraphAttentionActorCritic(GLOBAL_FEATURES, version)
            rebuilt = ActorCritic.from_state_dict(gat.state_dict())
            self.assertIsInstance(rebuilt, GraphAttentionActorCritic)
            self.assertEqual((rebuilt.input_version, rebuilt.prefix, rebuilt.feature_version),
                             (version, prefix_width(version), 4 if version == 1 else 6))
        self.assertEqual((prefix_width(1), prefix_width(2)), (320, 322))
        self.assertEqual(make_model(config()).prefix, 322)

    def test_version_two_requires_the_controller_margin(self):
        nav = VelocityNavigator('drone_1', model=GraphAttentionActorCritic(GLOBAL_FEATURES, 2), total_drones=3)
        with self.assertRaises(ValueError):
            nav.encode([0., 0., 15.], dict(time=0.))

    def test_chain_bend_and_margin_features(self):
        nav = VelocityNavigator('drone_1', model=GraphAttentionActorCritic(GLOBAL_FEATURES, 2), total_drones=3)
        nav.cbf_margin_source = lambda: 3.
        straight, _ = nav.encode([0., 0., 15.], dict(time=0., nearby_drones={
            'drone_0': dict(position=[-90., 0., 15.]), 'drone_2': dict(position=[90., 0., 15.])}))
        self.assertAlmostEqual(straight[33], -1., places=6)
        self.assertAlmostEqual(straight[32], .5, places=6)
        bent, _ = nav.encode([0., 0., 15.], dict(time=0., nearby_drones={
            'drone_0': dict(position=[-90., 0., 15.]), 'drone_2': dict(position=[0., 90., 15.])}))
        self.assertAlmostEqual(bent[33], 0., places=6)
        alone, _ = nav.encode([0., 0., 15.], dict(time=0., nearby_drones={}))
        self.assertEqual(alone[33], 0.)
        for margin, expected in ((None, 1.), (float('-inf'), -1.), (12., 1.), (-2., -1/3)):
            nav.cbf_margin_source = lambda margin=margin: margin
            x, _ = nav.encode([0., 0., 15.], dict(time=0.))
            self.assertAlmostEqual(x[32], expected, places=6)
        # No sensor data: every pixel is flagged invalid.
        x, _ = nav.encode([0., 0., 15.], dict(time=0.))
        offset = STATE_FEATURES[2]+SENSOR_SIZE**2
        self.assertTrue(np.all(x[offset:offset+SENSOR_SIZE**2] == 1.))

    def test_v9_rollout_encodes_sensor_validity_and_applied_cbf_margin(self):
        c = dict(stage_one(config()), steps=25, surprise_weight=0.)
        original = Experiment.step
        margins = []

        def step(env, frame):
            samples, reward, metrics = original(env, frame)
            for sample, controller in zip(samples, env.controllers):
                margins.append((float(sample[0][32]), controller.safety_stats['feasibility_margin']))
            return samples, reward, metrics

        model = make_model(c)
        with unittest.mock.patch.object(Experiment, 'step', step), contextlib.redirect_stdout(io.StringIO()):
            rollout, _ = run_episode(model, c, 50000, fixed_schedule(c, 0), True)
        features = np.concatenate([row[0] for row in rollout])
        for feature, margin in margins:
            expected = 1. if margin is None else float(np.clip(margin/6., -1., 1.))
            self.assertAlmostEqual(feature, expected, places=6)
        self.assertTrue(any(margin is not None and np.isfinite(margin) for _, margin in margins))
        offset = STATE_FEATURES[2] + SENSOR_SIZE**2
        self.assertTrue(np.all(features[:, offset:offset+SENSOR_SIZE**2] == 0.))
        self.assertTrue(np.all(features[:, 27] == 0.))

    def test_policy_drift_detects_state_independent_velocity(self):
        model = RelayActorCritic()
        x = torch.zeros(64, 3, RAW_WIDTH)
        x[..., :32] = torch.randn(64, 3, 32)
        with torch.no_grad():
            baseline = policy_drift(model, [model.distribution(x)])
            model.gat_actor[-1].bias.copy_(torch.tensor([.5, -.5]))
            drifting = policy_drift(model, [model.distribution(x)])
        self.assertLess(baseline['policy_drift_speed'], .1)
        self.assertGreater(drifting['policy_drift_coherence'], .99)
        self.assertGreater(drifting['policy_drift_vx'], 2.)
        self.assertLess(drifting['policy_drift_vy'], -2.)


if __name__ == '__main__':
    unittest.main()
