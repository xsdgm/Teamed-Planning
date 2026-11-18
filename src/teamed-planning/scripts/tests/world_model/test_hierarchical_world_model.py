"""Information boundaries and physical-label alignment for the local model."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from world_model.telemetry import packet, ReceivedHistory, TelemetryNetwork, context_features
from world_model.belief_actor import recentered_sensor
from world_model.residual_world import ResidualWorldModel, objective, infer


def source(name='drone_0', stamp=0., **kwargs):
    return packet(name, stamp, [0., 0., 15.], [1., 0., 0.], [2., 0., 0.], **kwargs)


class TelemetryTests(unittest.TestCase):
    def test_forwarding_keeps_original_timestamp_and_no_same_step_chain(self):
        network = TelemetryNetwork(['drone_0', 'drone_1', 'ground_0'], 3, loss=0, jitter=0, period=1)
        links = {'drone_0': ['drone_1'], 'drone_1': ['ground_0']}
        for frame in range(3):
            own = {name: source(name, frame) for name in network.receivers}
            network.advance(own, links, frame)
            if frame == 1:
                self.assertNotIn('drone_0', network.receivers['ground_0'].rows(frame))
        row = network.receivers['ground_0'].rows(2)['drone_0']
        self.assertEqual(row['stamp'], 0.); self.assertEqual(row['hop'], 2)
        self.assertEqual(row['received_at'], 2.)

    def test_duplicate_future_and_expired_do_not_refresh_history(self):
        history = ReceivedHistory('ground_0')
        history.receive([source(stamp=2)], 4)
        history.receive([source(stamp=2), source(stamp=1), source(stamp=10)], 5)
        self.assertEqual(len(history.histories['drone_0']), 1)
        self.assertEqual(history.rows(5)['drone_0']['received_at'], 4)
        self.assertEqual(history.rows(33), {})
        with self.assertRaises(KeyError): history.encode('drone_0', 33)

    def test_task_sent_and_task_ack_are_separate_features(self):
        own = source('ground_0', 12, goal=[100., 0.], goal_stamp=12, is_ground=True)
        last = source(stamp=8, goal=[0., 100.], goal_stamp=3)
        rows = {'drone_0': last}
        features = context_features(own, last, rows, 12, 0., 0, np.ones(2))
        self.assertEqual(features.shape, (20,))
        self.assertAlmostEqual(float(features[12]), 0.)
        self.assertAlmostEqual(float(features[16]), 9 / 30.)
        self.assertGreater(features[17], 0.)

    def test_lost_packets_cannot_yield_true_remote_state(self):
        network = TelemetryNetwork(['drone_0', 'ground_0'], 8, loss=0, jitter=0)
        for frame in range(5):
            network.advance({n: source(n, frame) for n in network.receivers}, {}, frame)
        self.assertEqual(network.receivers['ground_0'].rows(4), {})

    def test_unknown_remote_pixels_stay_unknown(self):
        sensor = dict(cells=np.zeros((12, 12)), resolution=2., center=[10., 10., 0.])
        self.assertTrue((recentered_sensor(sensor, [300., 300., 15.]) == -1).all())
        self.assertTrue((recentered_sensor(sensor, [10., 10., 15.]) == 0).all())


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(11)
        self.model = ResidualWorldModel(32, 4)
        self.batch = dict(history=torch.randn(3, 8, 14), history_mask=torch.ones(3, 8, dtype=torch.bool),
            context=torch.randn(3, 20), baseline_states=torch.zeros(3, 5, 4), baseline_actions=torch.zeros(3, 4, 2),
            target_states=torch.ones(3, 5, 4) * .2, target_actions=torch.ones(3, 4, 2) * .1,
            state_mask=torch.ones(3, 5, dtype=torch.bool), action_mask=torch.ones(3, 4, dtype=torch.bool))

    def test_labels_never_enter_inference_and_padding_is_ignored(self):
        self.batch['history_mask'][:, :4] = False
        before = infer(self.model, self.batch)
        self.batch['target_states'].add_(100.)
        self.batch['target_actions'].mul_(50.)
        self.batch['history'][:, :4].fill_(1e4)
        after = infer(self.model, self.batch)
        for key in before: self.assertTrue(torch.equal(before[key], after[key]))

    def test_censored_labels_do_not_change_loss(self):
        self.batch['state_mask'][:, 3:] = False; self.batch['action_mask'][:, 2:] = False
        pred = infer(self.model, self.batch); before = objective(pred, self.batch)
        self.batch['target_states'][:, 3:] = 1e5; self.batch['target_actions'][:, 2:] = -1e5
        self.assertTrue(torch.equal(before, objective(pred, self.batch)))

    def test_autoregressive_uncertainty_and_speed_bound(self):
        pred = infer(self.model, self.batch)
        self.assertTrue((pred['state_std'][:, 1:, :2] >= pred['state_std'][:, :-1, :2]).all())
        self.assertTrue((pred['actions'].norm(dim=-1) <= 1.000001).all())
        # A future actor command must affect its own future, not the present.
        self.batch['baseline_actions'][:, 3, 0] = .8
        after = infer(self.model, self.batch)
        self.assertTrue(torch.equal(pred['states'][:, :4], after['states'][:, :4]))
        self.assertFalse(torch.equal(pred['states'][:, 4], after['states'][:, 4]))

    def test_real_gradient_updates_reduce_supervised_objective(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.005)
        before = float(objective(infer(self.model, self.batch), self.batch).detach())
        for _ in range(12):
            optimizer.zero_grad(); loss = objective(infer(self.model, self.batch), self.batch)
            loss.backward(); optimizer.step()
        after = float(objective(infer(self.model, self.batch), self.batch).detach())
        self.assertTrue(np.isfinite(after)); self.assertLess(after, before)


class RuntimeLabels(unittest.TestCase):
    def test_true_frozen_rollout_labels_and_censored_tail(self):
        from world_model.belief_rollouts import collect_scene
        from world_model.navigation_scenes import make_scene
        repo = Path(__file__).resolve().parents[5]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            for name in ('data', 'episodes', 'logs'): (output / name).mkdir()
            result = collect_scene((str(repo), make_scene(6161, (150, 150), 1000, 'open'),
                dict(option_steps=8, steps=10, horizon=4, stride=4, loss=.1, jitter=3), temp, 'unit'))
            with np.load(result['path']) as data:
                self.assertGreater(result['ground_samples'], 0)
                self.assertTrue((data['source_stamp'] < data['frame']).all())
                self.assertTrue((~data['state_mask']).any())
                self.assertTrue((data['state_mask'][:, 0]).all())
                # One-second displacement must match the executed action label.
                displacement = np.diff(data['target_states'][:, :, :2], axis=1) * 20.
                valid = data['action_mask'] & data['state_mask'][:, 1:]
                np.testing.assert_allclose(displacement[valid], data['target_actions'][valid] * 6., atol=2e-4)


if __name__ == '__main__': unittest.main(verbosity=2)
