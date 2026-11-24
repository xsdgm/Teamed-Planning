"""Behavior likelihoods, causal prediction inputs and paired gradient updates."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vtemp2_3.graph_fast_policy import (make_fast, numpy_state, state_hash, advantages, ppo_update,
                         PeerEncoder, PEERS, TOKEN_WIDTH, PREDICTION_WIDTH)
from vtemp2_3.joint_rollouts import JointRecorder, collect_joint
from vtemp2_3.safety_context import normalized_margin, validate_cbf_context
from vtemp2_3.navigation_scenes import make_scene
from vtemp2_3.residual_world import ResidualWorldModel
from vtemp2_3.train_joint_policy import train_world_batch
from vtemp2_3.frozen_navigation import runtime


class AdvantageTests(unittest.TestCase):
    def test_real_terminal_and_time_limit_are_distinct(self):
        a, r = advantages([1., 1.], [0., 0.], 10., [False, False], gamma=1, lam=1)
        np.testing.assert_equal(r, [12., 11.])
        a, r = advantages([1., 1.], [0., 0.], 10., [False, True], gamma=1, lam=1)
        np.testing.assert_equal(r, [2., 1.])

    def test_mask_and_peer_order(self):
        encoder = PeerEncoder()
        tokens = torch.randn(2, PEERS, TOKEN_WIDTH); tokens[..., -1] = 1.
        self.assertTrue(torch.allclose(encoder(tokens), encoder(tokens.flip(-2))))
        tokens[..., -1] = 0.
        before = encoder(tokens); tokens[..., :-1] = 123456.
        self.assertTrue(torch.equal(before, encoder(tokens)))
        self.assertTrue(torch.equal(before, torch.zeros_like(before)))


class JointRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); torch.manual_seed(3)
        cls.repo = Path(__file__).resolve().parents[5]
        cls.original_sha = state_hash(numpy_state(runtime(cls.repo)[2]))
        cls.fast = make_fast(cls.repo)
        cls.world = ResidualWorldModel(32, 4)
        cls.temp = tempfile.TemporaryDirectory(); output = Path(cls.temp.name)
        for name in ('data', 'episodes', 'logs'): (output / name).mkdir()
        cls.executed_cbf = []
        original = JointRecorder.append_transition
        def capture(recorder, samples, state, reward, terminal):
            for controller in recorder.env.controllers:
                cls.executed_cbf.append(dict(owner=controller.drone_id, frame=recorder.decision_frame,
                    normalized=normalized_margin(controller.safety_stats['feasibility_margin'],
                                                 controller.max_speed)))
            return original(recorder, samples, state, reward, terminal)
        with patch.object(JointRecorder, 'append_transition', capture):
            cls.result = collect_joint((str(cls.repo), make_scene(7171, (150, 150), 1000, 'open'),
                dict(hidden=32, horizon=4, option_steps=8, steps=10, stride=4, loss=0., jitter=0),
                str(output), 'unit', numpy_state(cls.fast), numpy_state(cls.world), 0))
        with np.load(cls.result['ppo_path']) as data: cls.ppo = {k: data[k] for k in data.files}
        with np.load(cls.result['wm_path']) as data: cls.wm = {k: data[k] for k in data.files}
        cls.cbf = json.loads(Path(cls.result['cbf_context_path']).read_text())['decisions']

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_predictions_reach_actual_shared_actor_and_likelihoods_match(self):
        self.assertGreater(self.result['predicted_agent_steps'], 0)
        x = torch.from_numpy(self.ppo['features'])
        self.assertEqual(x.shape[-1], self.fast.raw_width + PREDICTION_WIDTH)
        self.assertTrue((x[0, :, self.fast.raw_width:] == 0).all())
        self.assertTrue((x[-1, :, self.fast.raw_width:] != 0).any())
        with torch.no_grad():
            dist = self.fast.distribution(x)
            base = runtime(self.repo)[2].distribution(x[..., :self.fast.raw_width])
            self.assertTrue(torch.equal(dist.mean, base.mean))
            np.testing.assert_allclose(dist.log_prob(torch.from_numpy(self.ppo['actions'])).numpy(),
                                       self.ppo['logp'], atol=5e-5)
        self.assertEqual(state_hash(numpy_state(runtime(self.repo)[2])), self.original_sha)

    def test_real_labels_are_timestamped_and_not_speculative(self):
        self.assertTrue((self.wm['source_stamp'] < self.wm['frame']).all())
        displacement = np.diff(self.wm['target_states'][:, :, :2], axis=1) * 20.
        valid = self.wm['action_mask'] & self.wm['state_mask'][:, 1:]
        np.testing.assert_allclose(displacement[valid], self.wm['target_actions'][valid] * 6., atol=2e-4)
        self.assertTrue((~self.wm['state_mask']).any())
        self.assertEqual(len(self.ppo['rewards']), 10)
        self.assertFalse(self.ppo['terminals'][-1])
        self.assertTrue(np.isfinite(self.ppo['bootstrap']))

    def test_graph_margin_matches_same_frame_executed_cbf_and_ppo_input(self):
        self.assertEqual(len(self.cbf), len(self.executed_cbf))
        self.assertEqual(self.result['cbf_decisions'], 33)  # Includes the bootstrap step.
        for context, executed in zip(self.cbf, self.executed_cbf):
            value = validate_cbf_context(context, executed['owner'], executed['frame'])
            self.assertEqual(value, executed['normalized'])
            self.assertEqual(context['geometry_source'], 'received_local_geometry')
        norms = np.asarray([row['cbf_margin_norm'] for row in self.cbf], np.float32).reshape(-1, 3)
        np.testing.assert_array_equal(self.ppo['features'][..., 32], norms[:10])

    def test_actor_and_world_both_update_and_prediction_changes_action(self):
        model = make_fast(self.repo)
        actor = torch.optim.Adam(model.actor_parameters(), lr=1e-5)
        critic = torch.optim.Adam(model.critic_parameters(), lr=1e-4)
        metrics = ppo_update(model, [self.ppo], actor, critic, 'cpu', np.random.default_rng(1),
                             epochs=2, minibatch=5)
        self.assertGreater(metrics['actor_optimizer_steps'], 0)
        self.assertGreater(metrics['base_actor_parameter_delta'], 0.)
        self.assertGreater(metrics['prediction_adapter_parameter_delta'], 0.)
        x = torch.from_numpy(self.ppo['features'][-1:])
        with torch.no_grad():
            self.assertGreater(float((model.distribution(x).mean -
                                      model.distribution(x[..., :model.raw_width]).mean).abs().max()), 0.)
        world = ResidualWorldModel(32, 4)
        optimizer = torch.optim.Adam(world.parameters(), lr=1e-4)
        result = train_world_batch(world, optimizer, self.wm, 'cpu', np.random.default_rng(2), epochs=1)
        self.assertGreater(result['parameter_delta'], 0.)
        self.assertGreater(result['optimizer_steps'], 0)

    def test_stale_behavior_rejected_and_kl_rolls_back_adam(self):
        model = make_fast(self.repo)
        actor = torch.optim.Adam(model.actor_parameters(), lr=1e-4)
        critic = torch.optim.Adam(model.critic_parameters(), lr=1e-4)
        stale = dict(self.ppo, logp=self.ppo['logp'] + .1)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            ppo_update(model, [stale], actor, critic, 'cpu', np.random.default_rng(1), epochs=1)
        before = [p.detach().clone() for p in model.actor_parameters()]
        result = ppo_update(model, [self.ppo], actor, critic, 'cpu', np.random.default_rng(1),
                            epochs=1, minibatch=5, max_kl=-1.)
        self.assertEqual(result['rejected_actor_epochs'], 1)
        self.assertEqual(result['actor_optimizer_steps'], 0)
        self.assertFalse(actor.state)
        for a, b in zip(before, model.actor_parameters()): self.assertTrue(torch.equal(a, b))


if __name__ == '__main__': unittest.main(verbosity=2)
