"""Behavioural checks for training on partial, action-dependent terrain evidence."""
import math
from unittest import TestCase

import numpy as np

from vtemp5.vtemp5_env import VTemp5Env, rule_offset
from vtemp5.vtemp5_large import VTemp5LargeObstacleEnv


class LocalObstacleTests(TestCase):
    def test_changing_the_hidden_exit_does_not_change_initial_evidence(self):
        left = VTemp5Env(17, blocked_side=1)
        right = VTemp5Env(17, blocked_side=-1)
        for env in (left, right):
            view = env.observe()
            self.assertFalse(view['detected'])
            self.assertFalse(any(view['routes'].values()))
        np.testing.assert_equal(left.observe(), right.observe())

    def test_zero_motion_never_earns_route_progress(self):
        env = VTemp5Env(31, step_m=0., horizon=8)
        start = env.observe()['position']
        for _ in range(8):
            view, _, done, info = env.step(math.pi / 4)
            np.testing.assert_equal(view['position'], start)
            self.assertAlmostEqual(info['progress_m'], 0.)
            if done:
                break

    def test_steering_decides_which_bank_is_explored(self):
        left, right = (VTemp5Env(11, blocked_side=0, wait_probability=0.) for _ in range(2))
        while not left.observe()['detected']:
            left.step(0.)
            right.step(0.)
        for _ in range(3):
            a, _, _, _ = left.step(math.pi / 3)
            b, _, _, _ = right.step(-math.pi / 3)
        self.assertGreater(a['position'][1], b['position'][1])
        self.assertTrue(a['route_observed'][1][0])
        self.assertFalse(a['route_observed'][-1][0])
        self.assertTrue(b['route_observed'][-1][0])
        self.assertFalse(b['route_observed'][1][0])

    def test_same_evidence_until_a_return_confirms_or_refutes_left(self):
        environments = [VTemp5Env(6, blocked_side=s, wait_probability=0.) for s in (1, -1)]
        for _ in range(40):
            left, right = (env.observe() for env in environments)
            if left['side_status'][1] == 'blocked':
                self.assertEqual(right['side_status'][1], 'open')
                after, _, _, info = environments[0].step(rule_offset(left))
                self.assertTrue(info['side_switch'])
                self.assertEqual(after['locked_side'], -1)
                return
            self.assertEqual(left['routes'], right['routes'])
            self.assertEqual(left['route_observed'], right['route_observed'])
            self.assertEqual(left['side_status'], right['side_status'])
            offset = rule_offset(left)
            for env in environments:
                env.step(offset)
        self.fail('The selected side was never surveyed')

    def test_sides_stay_locked_without_an_adverse_report(self):
        env = VTemp5Env(31, blocked_side=1, horizon=200)
        locked_steps = 0
        for _ in range(200):
            before = env.observe()
            side = before['locked_side']
            if side and before['side_status'][side] != 'blocked':
                offset = -side * math.pi / 2
                locked_steps += 1
            else:
                offset = rule_offset(before)
            after, _, done, info = env.step(offset)
            if side and after['locked_side'] == -side:
                self.assertEqual(before['side_status'][side], 'blocked')
            self.assertFalse(info['unauthorized_switch'])
            if done:
                break
        self.assertGreater(locked_steps, 0)


class LargeTerrainTests(TestCase):
    def test_one_joint_prediction_per_detected_review_even_without_new_reports(self):
        from vtemp5.vtemp5_policy import policy_observation

        class RecordingPredictor:
            def __init__(self):
                self.inputs = []

            def predict_sides(self, features):
                self.inputs.append(features.copy())
                # Vary the output per call to expose stale reads in the vehicle graph.
                return np.tile([110. + len(self.inputs), 60., .5], (2, 1))

        predictor = RecordingPredictor()
        env = VTemp5LargeObstacleEnv(11, terrain='bridge', predictor=predictor, step_m=0.)
        while not env.observe()['detected']:
            self.assertEqual(len(predictor.inputs), 0)
            env.step(0.)
        self.assertEqual(len(predictor.inputs), 1)
        np.testing.assert_equal(predictor.inputs[0][:, 1], [1., -1.])
        before = policy_observation(env.observe())['road_nodes']
        # Look away from the obstacle so the next review has identical evidence.
        new, _, _, info = env.step(math.pi)
        self.assertFalse(info['delivered'])
        self.assertEqual(len(predictor.inputs), 2)
        np.testing.assert_equal(*predictor.inputs)
        after = policy_observation(new)['road_nodes']
        self.assertFalse(np.array_equal(before, after))
        np.testing.assert_equal(after, policy_observation(env.observe())['road_nodes'])
        self.assertEqual(len(predictor.inputs), 2)

    def test_predictions_only_follow_a_report_and_reach_the_vehicle_graph(self):
        from vtemp5.vtemp5_policy import policy_observation

        calls = [0, 0]

        def predictor(index):
            def predict(features):
                calls[index] += 1
                return [115. + 30. * index, 60., .5]
            return predict

        envs = [VTemp5LargeObstacleEnv(11, terrain='bridge', predictor=predictor(i)) for i in (0, 1)]
        for env in envs:
            obs = env.observe()
            self.assertFalse(obs['detected'])
            self.assertFalse(policy_observation(obs)['road_valid'].any())
        self.assertEqual(calls, [0, 0])
        for env in envs:
            while not env.observe()['detected']:
                env.step(0.)
        observations = [env.observe() for env in envs]
        self.assertEqual(observations[0]['side_status'], observations[1]['side_status'])
        self.assertGreater(min(calls), 0)
        left, right = [policy_observation(o)['road_nodes'] for o in observations]
        self.assertFalse(np.array_equal(left, right))

    def test_single_bridge_identity_is_hidden_on_the_first_report(self):
        envs = [VTemp5LargeObstacleEnv(52, terrain='bridge', blocked_side=s) for s in (1, -1)]
        for env in envs:
            while not env.observe()['detected']:
                env.step(0.)
        np.testing.assert_equal(envs[0].observe(), envs[1].observe())
        for s in (-1, 1):
            for env in envs:
                features, labels = env.training_example(s)
                self.assertTrue(np.isnan(labels).all())
                self.assertEqual(features[-1], 0.)

    def test_revised_reference_requires_delivered_closure_and_a_new_actor_command(self):
        from vtemp5.vtemp5_policy import policy_observation

        env = VTemp5LargeObstacleEnv(35, terrain='bridge', blocked_side=1, wait_probability=0.)
        while not env.done:
            public = env.observe()
            original = policy_observation(public)
            revised = policy_observation(public, evidence_reference=True)
            if public['side_status'][1] == 'blocked':
                self.assertEqual(public['locked_side'], 1)
                self.assertFalse(np.allclose(original['local_target'], revised['local_target']))
                for key in ('angle_arcs', 'drive_segments', 'command_nodes', 'road_nodes'):
                    np.testing.assert_equal(original[key], revised[key])
                self.assertEqual(env.observe()['locked_side'], 1)
                self.assertTrue(env.step(rule_offset(public))[3]['side_switch'])
                return
            np.testing.assert_equal(original['local_target'], revised['local_target'])
            env.step(rule_offset(public))
        self.fail('The chosen side was never refuted')
