"""Real short rollouts for shared graph dynamics and both control time scales."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vtemp2_3.graph_fast_policy import numpy_state, state_hash
from vtemp2_3.graph_rollouts import GraphRecorder, collect_graph
from vtemp2_3.hierarchical_policy import GraphSlowActorCritic, make_graph_fast
from vtemp2_3.graph_dynamics import GraphWorldModel, NODE_COUNT, NODE_IDS
from vtemp2_3.navigation_scenes import make_scene


class GraphRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(48)
        cls.repo = Path(__file__).resolve().parents[5]
        cls.fast = make_graph_fast(cls.repo)
        cls.world = GraphWorldModel(32, 4)
        cls.slow = GraphSlowActorCritic(hidden=32)
        cls.states = [numpy_state(model) for model in (cls.fast, cls.world, cls.slow)]
        cls.temp = tempfile.TemporaryDirectory()
        output = Path(cls.temp.name)
        for name in ('data', 'episodes', 'logs'):
            (output / name).mkdir()
        cls.results, cls.ppo, cls.wm, cls.recorders = {}, {}, {}, {}
        original_init = GraphRecorder.__init__
        original_graph = GraphRecorder.graph
        for phase in ('fast', 'slow'):
            def capture(recorder, *args, phase=phase, **kwargs):
                original_init(recorder, *args, **kwargs)
                cls.recorders[phase] = recorder

            def graph_with_missing_probe(recorder, owner, local, raw=None):
                if recorder.frame == 0 and owner == 'ground_0' and not hasattr(recorder, 'ground_initial_probe'):
                    recorder.ground_initial_probe = original_graph(recorder, owner, local, raw)
                    recorder.ground_initial_raw_neighbors = len(recorder.env.obs[recorder.env.ground]['neighbors'])
                if recorder.frame == 0 and owner == 'drone_0' and not hasattr(recorder, 'missing_probe'):
                    # The predictor is given no local neighbors and no packets
                    # have arrived. True remote robots still exist in the sim.
                    without_neighbors = dict(local, neighbors=[])
                    recorder.missing_probe = original_graph(recorder, owner, without_neighbors, raw)
                return original_graph(recorder, owner, local, raw)

            config = dict(hidden=32, horizon=4, option_steps=4, steps=8, stride=1, loss=0., jitter=1,
                          complete_missions=False)
            case = make_scene(8484, (150., 150.), 1000., 'open')
            with patch.object(GraphRecorder, '__init__', capture), \
                    patch.object(GraphRecorder, 'graph', graph_with_missing_probe):
                result = collect_graph((str(cls.repo), case, config, str(output), phase,
                    *cls.states, 0, phase, False))
            cls.results[phase] = result
            with np.load(result['ppo_path'], allow_pickle=False) as data:
                cls.ppo[phase] = {key: data[key] for key in data.files}
            with np.load(result['wm_path'], allow_pickle=False) as data:
                cls.wm[phase] = {key: data[key] for key in data.files}
        with np.load(cls.results['slow']['slow_path'], allow_pickle=False) as data:
            cls.slow_data = {key: data[key] for key in data.files}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_both_platforms_use_graph_world_and_rollout_versions_are_frozen(self):
        for phase, result in self.results.items():
            self.assertGreaterEqual(result['world_calls']['uav'], 3 * result['transitions'])
            self.assertGreaterEqual(result['world_calls']['ugv'], result['transitions'])
            self.assertEqual(set(self.wm[phase]['observer_id']), set(range(NODE_COUNT)))
            for key, state in zip(('fast_sha256', 'world_sha256', 'slow_sha256'), self.states):
                self.assertEqual(result[key], state_hash(state))
        for model, state in zip((self.fast, self.world, self.slow), self.states):
            self.assertEqual(state_hash(numpy_state(model)), state_hash(state))

    def test_current_observed_states_equal_actual_same_clock_labels(self):
        for phase, data in self.wm.items():
            observed = data['state_observed'] & data['state_mask'][..., 0]
            self.assertTrue(observed.any())
            np.testing.assert_allclose(data['baseline_states'][..., 0, :][observed],
                                       data['target_states'][..., 0, :][observed], atol=1e-5, rtol=1e-5)
            self.assertTrue((data['state_observed'] | ~data['ego_mask']).all())
            displacement = np.diff(data['target_states'][..., :2], axis=2) * 20.
            valid = data['action_mask'] & data['state_mask'][..., 1:] & data['state_mask'][..., :-1]
            np.testing.assert_allclose(displacement[valid], data['target_actions'][valid] * 6., atol=2e-4)

    def test_direct_measurement_does_not_refresh_forwarded_source_age_or_hops(self):
        for data in self.wm.values():
            valid = data['node_mask']
            age = data['frame'][:, None] - data['source_stamp']
            self.assertTrue((age[valid] >= 0.).all())
            np.testing.assert_allclose(data['history'][..., -1, 9][valid], age[valid] / 30., atol=1e-6)
            direct_with_old_packet = valid & data['state_observed'] & ~data['ego_mask'] & (age > 0.)
            self.assertTrue(direct_with_old_packet.any())
            self.assertTrue((data['history'][..., -1, 11][direct_with_old_packet] > 0.).all())

    def test_same_frame_own_task_in_history_matches_current_local_context(self):
        for data in self.wm.values():
            own = data['ego_mask']
            np.testing.assert_allclose(data['history'][..., -1, 6:8][own],
                                       data['context'][..., 9:11][own], atol=1e-6)
            np.testing.assert_array_equal(data['history'][..., -1, 8][own],
                                          data['context'][..., 11][own])

    def test_unknown_nodes_are_absent_even_when_truth_contains_them(self):
        recorder = self.recorders['fast']
        graph = recorder.missing_probe
        expected = np.asarray([name == 'drone_0' for name in NODE_IDS])
        np.testing.assert_array_equal(graph['node_mask'], expected)
        self.assertTrue((graph['history'][~expected] == 0.).all())
        self.assertTrue((graph['baseline_states'][~expected] == 0.).all())
        proxy = SimpleNamespace(inputs=[copy.deepcopy(graph)], horizon=recorder.horizon,
                                truth_states=recorder.truth_states, truth_actions=recorder.truth_actions)
        labels = GraphRecorder.world_dataset(proxy)
        self.assertEqual(set(recorder.truth_states[0]), set(NODE_IDS))
        self.assertFalse(labels['state_mask'][0, ~expected].any())
        self.assertFalse(labels['action_mask'][0, ~expected].any())

    def test_ground_receipt_excludes_initial_or_dropped_neighbor_packets(self):
        recorder = self.recorders['fast']
        self.assertGreater(recorder.ground_initial_raw_neighbors, 0)
        only_ground = np.asarray([name == 'ground_0' for name in NODE_IDS])
        np.testing.assert_array_equal(recorder.ground_initial_probe['node_mask'], only_ground)
        # Simulate the controller having received the current own sensor tile
        # but no neighbors. Raw simulator neighbors must not restore anchoring.
        raw = recorder.env.obs[recorder.env.ground]
        self.assertGreater(len(raw['neighbors']), 0)
        saved = recorder.ground_received
        try:
            recorder.ground_received = dict(raw, neighbors=[])
            local = recorder.ground_observation()
            self.assertEqual(local['neighbors'], [])
            graph = recorder.graph('ground_0', local)
            np.testing.assert_array_equal(graph['state_observed'], only_ground)
        finally:
            recorder.ground_received = saved

    def test_fast_behavior_likelihoods_match_recorded_graph_features(self):
        data = self.ppo['fast']
        with torch.inference_mode():
            distribution = self.fast.distribution(torch.from_numpy(data['features']))
            logp = distribution.log_prob(torch.from_numpy(data['actions'])).numpy()
        np.testing.assert_allclose(logp, data['logp'], atol=5e-5, rtol=0.)
        self.assertEqual(len(data['rewards']), 8)
        self.assertTrue(np.isfinite(data['bootstrap']))

    def test_slow_behavior_likelihoods_and_actual_option_durations(self):
        data = self.slow_data
        from vtemp2_3.train_hierarchical import slow_episode
        graphs = [row['environment_graph'] for row in slow_episode(self.results['slow']['slow_path'])]
        with torch.inference_mode():
            distribution, values = self.slow(torch.from_numpy(data['features']),
                torch.from_numpy(data['mask']), torch.from_numpy(data['present']), graphs)
            logp = distribution.log_prob(torch.from_numpy(data['action'])).numpy()
        np.testing.assert_allclose(logp, data['logp'], atol=5e-5, rtol=0.)
        np.testing.assert_allclose(values.numpy(), data['value'], atol=1e-6, rtol=0.)
        self.assertTrue((data['duration'] > 0).all())
        self.assertEqual(int(data['duration'].sum()), len(self.recorders['slow'].transitions))
        self.assertTrue(bool(data['terminated'][-1]) or bool(data['truncated'][-1]))
        self.assertEqual(self.results['slow']['slow_decisions'], len(data['action']))

    def test_complete_mission_outlives_fast_chunk_and_records_success(self):
        from vtemp2_3.train_hierarchical import slow_episode
        config = dict(hidden=32, horizon=4, option_steps=4, steps=4, stride=4,
                      loss=0., jitter=1, complete_missions=True)
        case = make_scene(8485, (5., 5.), 1000., 'open')
        result = collect_graph((str(self.repo), case, config, self.temp.name, 'complete',
                                *self.states, 0, 'slow', True))
        self.assertTrue(result['goal_reached'])
        self.assertEqual(result['termination'], 'mission_arrival')
        self.assertGreater(result['executed_steps'], config['steps'] + 1)
        rows = slow_episode(result['slow_path'])
        self.assertTrue(rows[-1]['terminated'])
        self.assertFalse(rows[-1]['truncated'])
        self.assertEqual(rows[-1]['next_value'], 0.)
        self.assertIn('environment_graph', rows[-1])
        self.assertEqual(sum(row['duration'] for row in rows), result['executed_steps'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
