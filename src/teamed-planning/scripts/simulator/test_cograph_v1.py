"""CoGraph v1: multi-target scenes and metrics, multi-vehicle backbone/CBF, target tokens, warm start, training."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.graph_structure import TopologicalGraph
from drone.cbf_safety import CBFSafetyFilter
from drone.cograph_mappo import (CoGraphActorCritic, MultiTargetTaskState, target_tokens, RAW_WIDTH,
    ENCODED_WIDTH, MAX_TARGETS, GLOBAL_FEATURES)
from drone.local_backbone import backbone_metrics, backbone_metrics_multi, received_geometry_multi
from drone.mappo import ActorCritic
from drone.relay_attention_mappo import RelayActorCritic, RAW_WIDTH as V8_WIDTH
from simulator.cograph_scenarios import generate_cograph_scene, TARGET_MARGIN, TARGET_SEPARATION
from simulator.navigation_metrics import task_graph_metrics, multi_target_graph_metrics
from simulator.ppo_batch import update_batch
from simulator.train_mappo import validate_config, run_episode, fixed_schedule, make_model, train

CONFIG = Path(__file__).resolve().parents[2]/'config/algorithm/cograph_v1.json'


def config(**changes):
    value = dict(json.loads(CONFIG.read_text()), init_checkpoint=None)
    value.update(changes)
    return validate_config(value)


def lattice(points, spacing=10.):
    graph = TopologicalGraph('test')
    names = []
    for i, p in enumerate(points):
        names.append(f'n{i}')
        graph.add_node(names[-1], tuple(p))
    for i, p in enumerate(points):
        for j in range(i):
            if np.linalg.norm(np.asarray(p)-points[j]) <= spacing*1.01:
                graph.add_edge(names[i], names[j], weight=1.)
    return graph


class SceneTests(unittest.TestCase):
    def test_scene_is_deterministic_and_respects_ranges(self):
        a, b = generate_cograph_scene(50003), generate_cograph_scene(50003)
        self.assertEqual(a['goals'], b['goals'])
        counts = set()
        for seed in range(50000, 50012):
            scene = generate_cograph_scene(seed)
            goals, start = np.asarray(scene['goals']), np.asarray(scene['start'])
            counts.add(scene['target_count'])
            self.assertTrue(2 <= len(goals) <= 8)
            distance = np.linalg.norm(goals-start, axis=1)
            self.assertTrue(((distance >= 100.-1e-6) & (distance <= 400.+1e-6)).all())
            self.assertTrue(((goals >= TARGET_MARGIN) & (goals <= 1000.-TARGET_MARGIN)).all())
            gaps = [np.linalg.norm(goals[i]-goals[j]) for i in range(len(goals)) for j in range(i)]
            self.assertGreaterEqual(min(gaps), TARGET_SEPARATION)
            self.assertEqual(len(scene['vehicle_starts']), 3)
            for p, r in scene['circles']:
                self.assertTrue(all(np.linalg.norm(np.asarray(p)-v) > r+10. for v in scene['vehicle_starts']))
            self.assertEqual(len(scene['reference_lengths']), len(goals))
        self.assertGreater(len(counts), 3)


class MetricTests(unittest.TestCase):
    def test_one_search_matches_single_goal_metrics(self):
        rng = np.random.default_rng(3)
        points = [(x, y) for x in np.arange(0., 120., 10.) for y in np.arange(0., 60., 10.)
                  if rng.random() > .2]
        points += [(200., 200.), (210., 200.)]  # a separate component
        graph = lattice(points)
        start, goals = np.array([0., 0.]), np.array([[110., 50.], [205., 200.], [55., 30.], [60., 300.]])
        clear = lambda a, b: True
        multi = multi_target_graph_metrics(graph, start, goals, clear)
        for m, goal in enumerate(goals):
            single = task_graph_metrics(graph, start, goal, clear)
            self.assertEqual(bool(multi['connected'][m]), bool(single['graph_task_connected']))
            if single['graph_task_connected']:
                self.assertAlmostEqual(multi['length'][m], single['graph_task_path_length'], places=6)
            self.assertAlmostEqual(multi['frontier'][m], single['graph_frontier_goal_distance_m'], places=6)

    def test_empty_graph_and_no_start_access(self):
        goals = np.array([[50., 0.], [0., 80.]])
        empty = multi_target_graph_metrics(None, (0., 0.), goals, lambda a, b: True)
        self.assertFalse(empty['connected'].any())
        np.testing.assert_allclose(empty['frontier'], [50., 80.])
        blocked = multi_target_graph_metrics(lattice([(0., 0.), (10., 0.)]), (0., 0.), goals, lambda a, b: False)
        self.assertFalse(blocked['connected'].any())


class VehicleBackboneTests(unittest.TestCase):
    def test_grounds_link_equals_single_ground_link(self):
        cbf = CBFSafetyFilter()
        p, ground = np.array([0., 0., 15.]), np.array([90., 10.])
        peers = {'drone_1': np.array([-20., 5., 15.])}
        single = cbf.filter(np.array([6., 0., 0.]), p, peers, 1., ('__ground__', 'drone_1'), ground=ground,
                            ground_max_speed=1.)
        multi = cbf.filter(np.array([6., 0., 0.]), p, peers, 1., ('ground_2', 'drone_1'),
                           grounds={'ground_2': ground, 'ground_0': np.array([0., 400.])}, ground_max_speed=1.)
        np.testing.assert_allclose(single[0], multi[0])
        self.assertAlmostEqual(single[1]['feasibility_margin'], multi[1]['feasibility_margin'])
        self.assertEqual(multi[1]['required_link_distances_m']['ground_2'],
                         single[1]['required_link_distances_m']['__ground__'])
        self.assertEqual(multi[1]['missing_links'], [])
        lost = cbf.filter(np.zeros(3), p, peers, 1., ('ground_1',), grounds={'ground_2': ground})
        self.assertEqual(lost[1]['missing_links'], ['ground_1'])

    def test_received_vehicles_are_separate_nodes(self):
        observation = dict(neighbors=[
            dict(id=0, type='diff_drive', relative_position=np.array([5., 0., -15.]), broadcast=dict(robot_id='ground_0')),
            dict(id=1, type='diff_drive', relative_position=np.array([5., 14., -15.]), broadcast=None),
            dict(id=4, type='drone', relative_position=np.array([20., 0., 0.]), broadcast=dict(drone_id='drone_1'))])
        peers, _, grounds = received_geometry_multi((0., 0., 15.), observation)
        self.assertEqual(set(grounds), {'ground_0', 'robot_1'})
        self.assertEqual(set(peers), {'drone_1'})
        drones = {'a': np.array([0., 0., 15.]), 'b': np.array([80., 0., 15.])}
        self.assertEqual(backbone_metrics_multi(drones, {'g': np.zeros(3)})[0],
                         backbone_metrics(drones, np.zeros(3))[0])
        self.assertFalse(backbone_metrics_multi(drones, {'g': np.zeros(3), 'far': np.array([500., 0., 0.])})[0])


class TaskStateTests(unittest.TestCase):
    def mission(self, now, source, statuses, position=(0., 0.)):
        targets = [[100., 0., statuses[0]], [0., 300., statuses[1]], [-50., 0., statuses[2]]]
        return dict(time=now, ground_robots={'ground_0': dict(position=position, targets=targets, source_time=source)},
                    nearby_drones={})

    def test_focus_is_nearest_pending_and_any_vehicle_connects(self):
        state = MultiTargetTaskState('drone_0')
        state.update(self.mission(5., 4., (0., 0., 1.)))
        view = state.view((-40., 0.))
        self.assertEqual(view['focus'], 0)  # (-50, 0) is nearer but already connected
        other = self.mission(6., 6., (1., 0., 0.), position=(0., 20.))
        other['ground_robots']['ground_1'] = other['ground_robots'].pop('ground_0')
        state.update(other)
        view = state.view((-40., 0.))
        np.testing.assert_array_equal(view['connected'], [True, False, True])
        self.assertEqual(view['focus'], 1)
        state.self_position = np.array([-40., 0.])
        self.assertEqual(state.task()['target'], (0., 300.))

    def test_stale_missions_expire_and_older_never_replace_newer(self):
        state = MultiTargetTaskState('drone_0', timeout=10.)
        state.update(self.mission(5., 5., (0., 0., 0.)))
        state.update(self.mission(6., 2., (1., 1., 1.)))
        self.assertFalse(state.view((0., 0.))['connected'].any())
        state.update(dict(time=16., ground_robots={}, nearby_drones={}))
        self.assertIsNone(state.view((0., 0.)))

    def test_tokens(self):
        tokens, valid = target_tokens((0., 0.), None)
        self.assertFalse(valid.any())
        view = dict(targets=np.array([[35., 0.], [0., 70.]]), connected=np.array([False, True]), focus=0, age=2.,
                    vehicle=np.array([0., 0.]))
        tokens, valid = target_tokens((0., 0.), view)
        np.testing.assert_array_equal(valid, [1, 1]+[0]*(MAX_TARGETS-2))
        np.testing.assert_allclose(tokens[0], [.1, 0., .1, 0., 1., .2, .1], atol=1e-6)
        self.assertEqual(tokens[1, 3], 1.)


class ModelTests(unittest.TestCase):
    def test_warm_start_reproduces_v8_actor_and_round_trips(self):
        torch.manual_seed(0)
        v8 = RelayActorCritic(10, 2, 'full')
        model = CoGraphActorCritic()
        report = model.warm_start(v8.state_dict())
        self.assertEqual(report['partial'], ['gat_actor.1.weight', 'critic.1.weight'])
        self.assertTrue(all('targets' in key for key in report['fresh']))
        x = torch.randn(5, RAW_WIDTH)
        x[:, V8_WIDTH-16:V8_WIDTH] = (torch.rand(5, 16) > .5).float()
        x[:, -MAX_TARGETS:] = 1.
        with torch.no_grad():
            np.testing.assert_allclose(model.distribution(x).mean, v8.distribution(x[:, :V8_WIDTH]).mean, atol=1e-6)
        clone = ActorCritic.from_state_dict(model.state_dict())
        self.assertIsInstance(clone, CoGraphActorCritic)
        states = torch.randn(5, 9, GLOBAL_FEATURES)
        self.assertEqual(tuple(clone.value(x[:, None].expand(5, 9, RAW_WIDTH), None, states[:, 0]).shape), (5,))
        self.assertEqual(model.gat_actor[0](x).shape[-1], ENCODED_WIDTH)


class TrainingTests(unittest.TestCase):
    def test_config_contract(self):
        value = config()
        self.assertEqual(value['policy_architecture'], 'cograph_v1')
        with self.assertRaises(ValueError):
            validate_config(dict(json.loads(CONFIG.read_text()), target_count_range=[0, 8]))
        v8 = json.loads((CONFIG.parent/'mappo_training_v8.json').read_text())
        with self.assertRaises(ValueError):
            validate_config(dict(v8, target_count_range=[2, 4]))

    def test_short_episode_feeds_a_ppo_batch(self):
        value = config(steps=6, min_episode_steps=1, budget_extra=1)
        torch.manual_seed(1)
        model = make_model(value)
        rollout, summary = run_episode(model, value, 50001, fixed_schedule(value, 0), True)
        self.assertEqual(len(rollout), 6)
        self.assertEqual(rollout[0][0].shape, (9, RAW_WIDTH))
        self.assertEqual(rollout[0][5].shape, (GLOBAL_FEATURES,))
        for key in ('map_completed', 'safety_terminated', 'graph_task_path_ratio', 'targets_connected',
                    'route_efficiency', 'map_completion_step'):
            self.assertIn(key, summary)
        self.assertTrue(all(isinstance(v, (int, float)) for v in summary.values()))
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        _, ppo = update_batch(model, optimizer, [rollout], critic_epochs=1, value_normalization=True)
        self.assertTrue(np.isfinite(ppo['exact_kl']))

    def test_train_smoke_with_v8_warm_start(self):
        with tempfile.TemporaryDirectory() as folder:
            v8 = RelayActorCritic(10, 2, 'full')
            checkpoint = Path(folder)/'v8.pt'
            torch.save(dict(model=v8.state_dict(), policy_architecture='gat_relay_attention_v8',
                            config=dict(relay_attention='full'), training_episodes=7), checkpoint)
            value = config(steps=4, min_episode_steps=1, budget_extra=1, episodes=2, episodes_per_batch=1,
                           snapshot_interval=1, validation_interval=2, validation_seeds=[62000],
                           validation_samples=1, init_checkpoint=str(checkpoint))
            with contextlib.redirect_stdout(io.StringIO()):
                train(value, Path(folder)/'run', workers=1)
            progress = json.loads((Path(folder)/'run/progress.json').read_text())
            self.assertEqual(progress['status'], 'completed')
            provenance = json.loads((Path(folder)/'run/provenance.json').read_text())
            self.assertEqual(provenance['init_checkpoint']['source_architecture'], 'gat_relay_attention_v8')
            self.assertEqual(provenance['reward_contract'], 'cograph_v1_multi_target_option_space_bridged')
            saved = torch.load(Path(folder)/'run/latest.pt', weights_only=True)
            self.assertEqual(saved['critic_global_features'], GLOBAL_FEATURES)
            self.assertEqual(saved['feature_version'], 9)


if __name__ == '__main__':
    unittest.main()
