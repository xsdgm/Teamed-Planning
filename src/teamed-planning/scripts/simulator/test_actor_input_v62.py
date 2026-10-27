"""v6.2 actor inputs (no constant bias lever, own CBF margin, chain bend),
policy-drift diagnostics and synchronous episodes_per_batch."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from drone.mappo import ActorCritic, GLOBAL_FEATURES
from drone.velocity_mappo import (VelocityActorCritic, VelocityNavigator, velocity_features,
                                  STATE_FEATURES, SENSOR_SIZE)
from drone.graph_attention_mappo import GraphAttentionActorCritic, prefix_width, GRAPH_BLOCK
from simulator.mappo_experiment import Experiment
from simulator.ppo_batch import policy_drift
from simulator.train_mappo import (train, validate_config, run_episode, fixed_schedule,
                                   feature_version, make_model)

CONFIGS = Path(__file__).resolve().parents[2]/'config/algorithm'


def config(name='mappo_training_v6_2_1_mlp.json'):
    return validate_config(json.loads((CONFIGS/name).read_text()))


def stage_one(c):
    return dict(c, task_distance_range=[100., 283.8083151964686], max_task_distance=750.,
                distance_sampling='annulus_area')


def constant_nonzero(x):
    return np.flatnonzero((x.std(0) < 1e-7) & (np.abs(x.mean(0)) > 1e-6))


class ActorInputV62Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(62)
        torch.set_num_threads(1)

    def test_configs_change_only_inputs_and_batching(self):
        for name, base, gat in (('mappo_training_v6_2_1_mlp.json', 'mappo_training_v6_surprise.json', False),
                                ('mappo_training_v6_2_2_gat.json', 'mappo_training_v6_gat_surprise.json', True)):
            new, old = config(name), config(base)
            changed = {k for k in new if new[k] != old[k]}
            self.assertEqual(changed, {'actor_input_version', 'episodes_per_batch',
                                       'snapshot_interval', 'validation_interval'})
            self.assertEqual((new['actor_input_version'], new['episodes_per_batch']), (2, 16))
            self.assertEqual(feature_version(new), 6 if gat else 5)
            self.assertEqual(feature_version(old), 4 if gat else 2)

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
            model = VelocityActorCritic(GLOBAL_FEATURES, version)
            self.assertEqual(model.velocity_actor[0].in_features, velocity_features(version))
            rebuilt = ActorCritic.from_state_dict(model.state_dict())
            self.assertIsInstance(rebuilt, VelocityActorCritic)
            self.assertEqual((rebuilt.input_version, rebuilt.global_features), (version, GLOBAL_FEATURES))
            gat = GraphAttentionActorCritic(GLOBAL_FEATURES, version)
            rebuilt = ActorCritic.from_state_dict(gat.state_dict())
            self.assertIsInstance(rebuilt, GraphAttentionActorCritic)
            self.assertEqual((rebuilt.input_version, rebuilt.prefix, rebuilt.feature_version),
                             (version, prefix_width(version), 4 if version == 1 else 6))
        self.assertEqual((velocity_features(1), velocity_features(2)), (832, 834))
        self.assertEqual((prefix_width(1), prefix_width(2)), (320, 322))
        self.assertEqual(make_model(config('mappo_training_v6_2_2_gat.json')).prefix, 322)

    def test_version_two_requires_the_controller_margin(self):
        nav = VelocityNavigator('drone_1', model=VelocityActorCritic(GLOBAL_FEATURES, 2), total_drones=3)
        with self.assertRaises(ValueError):
            nav.encode([0., 0., 15.], dict(time=0.))

    def test_chain_bend_and_margin_features(self):
        nav = VelocityNavigator('drone_1', model=VelocityActorCritic(GLOBAL_FEATURES, 2), total_drones=3)
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

    def test_real_rollout_has_no_constant_bias_lever_and_true_cbf_margin(self):
        c = dict(stage_one(config()), steps=25, surprise_weight=0.)
        original = Experiment.step
        margins = []

        def step(env, frame):
            samples, reward, metrics = original(env, frame)
            for sample, controller in zip(samples, env.controllers):
                margins.append((float(sample[0][32]), controller.safety_stats['feasibility_margin']))
            return samples, reward, metrics

        for version in (1, 2):
            model = VelocityActorCritic(GLOBAL_FEATURES, version)
            margins.clear()
            with unittest.mock.patch.object(Experiment, 'step', step), contextlib.redirect_stdout(io.StringIO()):
                rollout, _ = run_episode(model, c, 50000, fixed_schedule(c, 0), True)
            x = np.concatenate([row[0] for row in rollout])
            if version == 2:
                for feature, margin in margins:
                    # The policy saw the margin of the very filter its command then passed.
                    expected = 1. if margin is None else float(np.clip(margin/6., -1., 1.))
                    self.assertAlmostEqual(feature, expected, places=6)
                self.assertTrue(any(m is not None and np.isfinite(m) and m < 6. for _, m in margins))
            lever = constant_nonzero(x)
            if version == 1:
                # The v6 validity plane: 144 identical ones on every step.
                self.assertTrue(set(range(176, 320)) <= set(lever))
            else:
                sensor = np.arange(STATE_FEATURES[2]+SENSOR_SIZE**2, STATE_FEATURES[2]+2*SENSOR_SIZE**2)
                self.assertTrue(np.all(x[:, sensor] == 0.))
                self.assertTrue(np.all(x[:, 27] == 0.))
                self.assertLessEqual(len(lever), 4, lever)

    def test_policy_drift_detects_state_independent_velocity(self):
        model = VelocityActorCritic(GLOBAL_FEATURES, 2)
        x = torch.randn(64, 3, velocity_features(2))
        with torch.no_grad():
            baseline = policy_drift(model, [model.distribution(x)])
            model.velocity_actor[-1].bias.copy_(torch.tensor([.5, -.5]))
            drifting = policy_drift(model, [model.distribution(x)])
        self.assertLess(baseline['policy_drift_speed'], .1)
        self.assertGreater(drifting['policy_drift_coherence'], .99)
        self.assertGreater(drifting['policy_drift_vx'], 2.)
        self.assertLess(drifting['policy_drift_vy'], -2.)

    def test_synchronous_batches_follow_episodes_per_batch(self):
        for name in ('mappo_training_v6_2_1_mlp.json', 'mappo_training_v6_2_2_gat.json'):
            c = config(name)
            c.update(episodes=6, steps=5, episodes_per_batch=3, snapshot_interval=3, validation_interval=6,
                     validation_seeds=[60000], validation_samples=1, ppo_epochs=1, critic_epochs=1)
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                train(c, directory, workers=2)
                p = Path(directory)
                rows = [json.loads(x) for x in (p/'training.jsonl').read_text().splitlines()]
                self.assertEqual([r['batch_size'] for r in rows], [3]*6)
                self.assertEqual(sorted({r['optimizer_batch'] for r in rows}), [1, 2])
                self.assertTrue(all('policy_drift_coherence' in r['ppo'] for r in rows if r.get('ppo')))
                self.assertTrue(any(r.get('ppo') for r in rows))
                saved = torch.load(p/'latest.pt', weights_only=True)
                self.assertEqual((saved['feature_version'], saved['actor_input_version']),
                                 (feature_version(c), 2))
                model = ActorCritic.from_state_dict(saved['model'])
                self.assertEqual(model.input_version, 2)
                episode, _ = run_episode(model, c, 50000, fixed_schedule(c, 0), True)
                width = (prefix_width(2)+GRAPH_BLOCK if c['policy_architecture'] == 'gat_map_v1'
                         else velocity_features(2))
                self.assertEqual(episode[0][0].shape, (3, width))


if __name__ == '__main__':
    unittest.main()
