"""Independent command ownership, partial observations and learned-model contract."""
import json
import inspect
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import torch

from simulator.train_world_model import build_command
from world_model.data import CHECKPOINT_SHA
from world_model.decoupled_model import DecoupledWorldModel, prediction_loss
from world_model.decoupled_observation import (ACTION_CONTRACT, ARCHITECTURE, DRONE_FEATURES,
    GROUND_FEATURES, PACKET_KEY, RESOLUTION, SIDE, ReceivedObservation, action_vector, crop_origin)
from world_model.decoupled_planner import (CHECKPOINT_FORMAT, IndependentGroundManager,
    IndependentSurveyManager, TerrainPlanner, distance_field)
from world_model.decoupled_runtime import (ground_navigation_controller,
    make_decoupled_controller, make_information_drone_controller)
from world_model.decoupled_training import load_dataset, train_model
from world_model import decoupled_pipeline as pipeline
from world_model.models import make_model
from world_model.runtime import ReconnaissanceEnvelope


def packet(now=0., source='drone_0', position=(15., 0., 15.), occupied=False):
    return dict(source=source, time=now, position=position, velocity=(1., 2., 0.),
                resolution=2., occupancy=np.full((12, 12), occupied, np.uint8), valid=True)


class ObservationTests(unittest.TestCase):
    def test_unknown_free_and_occupied_are_distinct(self):
        memory = ReceivedObservation((-300., -300., 300., 300.))
        memory.receive([packet()], 0.)
        maps, states, origin = memory.encode((0., 0.), 0., (0., 0.), (200., 0.))
        self.assertGreater(maps[1].sum(), 0)
        self.assertEqual(maps[0].sum(), 0)
        self.assertLess(maps[1].sum(), SIDE*SIDE)
        memory.receive([packet(now=1., occupied=True)], 1.)
        self.assertGreater(memory.raster(origin)[0].sum(), 0)

    def test_missing_and_stale_drone_states_have_explicit_masks(self):
        memory = ReceivedObservation((-300., -300., 300., 300.))
        memory.receive([packet()], 0.)
        _, first, _ = memory.encode((0., 0.), 0., (0., 0.), (200., 0.))
        row = first[GROUND_FEATURES:GROUND_FEATURES+DRONE_FEATURES]
        np.testing.assert_allclose(row[:3], np.array([15., 0., 15.])/200.)
        np.testing.assert_array_equal(row[7:10], (1., 1., 1.))
        self.assertEqual(first[GROUND_FEATURES+DRONE_FEATURES+7], 0)
        memory.receive([packet(now=-100.), packet(now=100.)], 40.)
        _, last, _ = memory.encode((0., 0.), 0., (0., 0.), (200., 0.))
        self.assertEqual(last[GROUND_FEATURES+8], 0)
        self.assertEqual(last[GROUND_FEATURES+11], 0)
        self.assertGreater(last[GROUND_FEATURES+6], 0)

    def test_replayed_and_out_of_order_packets_do_not_refresh_age(self):
        memory = ReceivedObservation((-300., -300., 300., 300.))
        memory.receive([packet(now=5., occupied=True)], 5.)
        before = dict(memory.cells)
        memory.receive([packet(now=4.), packet(now=5.)], 6.)
        self.assertEqual(memory.telemetry['drone_0']['time'], 5.)
        self.assertEqual(memory.received, set())
        self.assertEqual(memory.cells, before)

    def test_invalid_sensor_can_update_telemetry_without_marking_terrain_seen(self):
        memory = ReceivedObservation((-300., -300., 300., 300.))
        p = packet(); p['valid'] = False
        memory.receive([p], 0.)
        self.assertEqual(memory.cells, {})
        self.assertIn('drone_0', memory.telemetry)

    def test_commands_are_independent_and_masked(self):
        np.testing.assert_allclose(action_vector((100., 0.), (0., 100.)), (.1, 0, 1, 0, .1, 1))
        np.testing.assert_array_equal(action_vector(None, None), np.zeros(6))
        with self.assertRaises(ValueError):
            action_vector((np.nan, 1.), None)


class RuntimeTests(unittest.TestCase):
    def test_current_vtmp_vehicle_import_does_not_replace_existing_runtime_modules(self):
        before = {name: module for name, module in sys.modules.items()
                  if name == 'ground' or name.startswith('ground.')}
        controller = ground_navigation_controller()
        expected = Path(__file__).resolve().parents[3]/'ground/ground_main.py'
        self.assertEqual(Path(inspect.getfile(controller)).resolve(), expected)
        after = {name: module for name, module in sys.modules.items()
                 if name == 'ground' or name.startswith('ground.')}
        self.assertEqual(before, after)

    def controller(self):
        graph = nx.Graph()
        graph.add_node('near', position=np.array([1.5, 0.]), confidence=1.)
        graph.add_node('far', position=np.array([20., 0.]), confidence=1.)
        graph.add_edge('near', 'far')
        network = SimpleNamespace(graph=graph)

        class Base:
            world_bounds = (-300., -300., 300., 300.)
            def __init__(self):
                self.position, self.heading, self.resolution = (0., 0.), 0., .5
                self.unified_graph = network
                self.graph_fusion = SimpleNamespace(get_unified_graph=lambda: network)
                self.global_planner = SimpleNamespace(clear_path=lambda: None)
                self.local_planner = SimpleNamespace(safety_distance=1.,
                    observed_blocked_cells=lambda grid, resolution: (grid == 2, 0))
                self.driven = []
                self.semantic_map = np.ones((16, 16), np.uint8)

            def _receive_and_fuse_graphs(self, data, current_time=None):
                return False

            def _filter_observed_obstacles(self, graph, semantic):
                return graph

            def _drive_to(self, target):
                self.goal_position = target
                self.driven.append(target)

            def get_action(self, observation):
                if self.subgoals is not None:
                    raise AssertionError('Suppress the stale pre-fusion update')
                self._last_semantic_map = self.semantic_map
                self._receive_and_fuse_graphs({}, observation['full_state']['time'])
                return dict(motion=(1., .2), broadcast=dict(target=self.goal_position, planned_path=[(1., 0.)]))

        controller = make_decoupled_controller(Base, planner_factory=TerrainPlanner)()
        controller.set_goal((200., 0.))
        return controller

    def observation(self, now=0., neighbors=()):
        return dict(self=dict(position=np.zeros(3), velocity=np.zeros(3)),
                    occupancy_map=np.zeros((16, 16), np.uint8), neighbors=neighbors,
                    full_state=dict(time=now, hidden_obstacles='must never be read'))

    def test_simultaneous_survey_preserves_vehicle_motion_and_own_target(self):
        c = self.controller()
        action = c.get_action(self.observation())
        self.assertEqual(action['motion'], (1., .2))
        self.assertIsNotNone(c.survey.target)
        self.assertEqual(c.subgoals.target_status, 'unverified')
        self.assertIsNotNone(c.drive_target)
        self.assertIsNotNone(c.execution_target)
        self.assertFalse(np.allclose(c.drive_target, c.execution_target))
        np.testing.assert_allclose(c.driven[-1], c.execution_target)
        self.assertFalse(np.allclose(c.drive_target, c.survey.target))
        np.testing.assert_allclose(action['broadcast']['target'], c.survey.target)
        np.testing.assert_allclose(action['broadcast']['vehicle_target'], c.drive_target)
        np.testing.assert_allclose(action['broadcast']['vehicle_waypoint'], c.execution_target)
        self.assertEqual(action['broadcast']['planned_path'], [(1., 0.)])

    def test_survey_switch_does_not_change_vehicle_target_or_replan(self):
        c = self.controller(); c.get_action(self.observation())
        drive = c.drive_target.copy(); driven = len(c.driven)
        c.survey.target = np.array([50., 60.]); c.survey.chosen_at = -200.
        action = c.get_action(self.observation(1.))
        self.assertFalse(np.allclose(c.survey.target, (50., 60.)))
        np.testing.assert_array_equal(c.drive_target, drive)
        self.assertEqual(len(c.driven), driven)
        self.assertEqual(action['motion'], (1., .2))

    def test_loss_of_reconnaissance_does_not_stop_proven_ground_route(self):
        c = self.controller(); c.get_action(self.observation())
        drive = c.drive_target.copy()
        c.reconnaissance = ReconnaissanceEnvelope(drone_count=0)
        action = c.get_action(self.observation(1.))
        self.assertIsNone(c.survey.target)
        self.assertEqual(action['motion'], (1., .2))
        np.testing.assert_array_equal(c.drive_target, drive)

    def test_only_explicit_received_sensor_packets_enter_memory(self):
        c = self.controller()
        audit = dict(type='drone', broadcast=dict(evidence_audits=[packet()]))
        c.get_action(self.observation(neighbors=[audit]))
        self.assertEqual(c.observation_memory.telemetry, {})
        delivered = dict(type='drone', broadcast={PACKET_KEY: [packet(now=1.)]})
        c.get_action(self.observation(1., [delivered]))
        self.assertIn('drone_0', c.observation_memory.telemetry)

    def test_unknown_intention_is_allowed_without_evidence_but_execution_holds(self):
        planner = TerrainPlanner()
        planner.observe(np.zeros((3, SIDE, SIDE)), np.zeros(50), crop_origin((0., 0.)), 0.)
        manager = IndependentGroundManager((200., 0.), planner=planner)
        manager.set_observation((0., 0.), np.full((16, 16), 2), .5)
        target, _ = manager.update((0., 0.), None, 0.)
        self.assertEqual(manager.target_status, 'unverified')
        np.testing.assert_allclose(target, (200., 0.))
        self.assertIsNone(manager.execution_target)
        manager.request_target((150., 0.))
        manager.update((0., 0.), None, 1.)
        self.assertEqual(manager.rejected_requests, 0)
        np.testing.assert_allclose(manager.target, (150., 0.))
        self.assertIsNone(manager.execution_target)

    def test_motion_waits_for_safe_prefix_even_when_intention_is_present(self):
        c = self.controller()
        c.semantic_map[:] = 0
        c.unified_graph.graph.clear()
        first = c.get_action(self.observation())
        self.assertIsNotNone(c.drive_target)
        self.assertIsNone(c.execution_target)
        self.assertEqual(first['motion'], (0., 0.))
        self.assertEqual(c.driven, [])
        intention = c.drive_target.copy()
        c.semantic_map[:] = 1
        second = c.get_action(self.observation(1.))
        np.testing.assert_array_equal(c.drive_target, intention)
        self.assertIsNotNone(c.execution_target)
        np.testing.assert_allclose(c.driven[-1], c.execution_target)
        self.assertEqual(second['motion'], (1., .2))

    def test_withdrawn_graph_revalidates_execution_without_cancelling_intention(self):
        c = self.controller()
        c.subgoals.request_target((20., 0.))
        c.get_action(self.observation())
        self.assertEqual(c.subgoals.target_status, 'verified')
        np.testing.assert_allclose(c.execution_target, (20., 0.))
        c.unified_graph.graph.remove_edge('near', 'far')
        c.get_action(self.observation(1.))
        np.testing.assert_allclose(c.drive_target, (20., 0.))
        self.assertEqual(c.subgoals.target_status, 'unverified')
        self.assertFalse(np.allclose(c.execution_target, (20., 0.)))
        self.assertEqual(c.subgoals.failed, 0)

    def test_obstacle_between_vehicle_and_unknown_intention_does_not_forbid_target(self):
        c = self.controller()
        c.unified_graph.graph.clear()
        c.semantic_map[:, 12] = 2
        c.subgoals.request_target((150., 0.))
        c.get_action(self.observation())
        np.testing.assert_allclose(c.drive_target, (150., 0.))
        self.assertEqual(c.subgoals.target_status, 'unverified')
        self.assertIsNotNone(c.execution_target)
        self.assertLess(c.execution_target[0], 1.)

    def test_manual_target_can_change_before_arrival_and_rejects_out_of_range(self):
        c = self.controller()
        for now, target in enumerate(((150., 0.), (0., 150.))):
            c.subgoals.request_target(target)
            c.get_action(self.observation(float(now)))
            np.testing.assert_allclose(c.drive_target, target)
        self.assertEqual(c.subgoals.arrivals, 0)
        c.subgoals.request_target((250., 0.))
        c.get_action(self.observation(2.))
        self.assertEqual(c.subgoals.rejected_requests, 1)
        np.testing.assert_allclose(c.drive_target, (0., 150.))

    def test_better_prediction_changes_intention_before_arrival(self):
        c = self.controller()
        with patch.object(c.world_planner, 'strategic_cost',
                          side_effect=lambda target, position, goal: float(np.linalg.norm(target-(200., 0.)))):
            c.get_action(self.observation())
        np.testing.assert_allclose(c.drive_target, (200., 0.))
        with patch.object(c.world_planner, 'strategic_cost',
                          side_effect=lambda target, position, goal: float(np.linalg.norm(target-(0., 200.)))):
            c.get_action(self.observation(5.))
        np.testing.assert_allclose(c.drive_target, (0., 200.), atol=1e-6)
        self.assertEqual(c.subgoals.last_reason, 'better_candidate')
        self.assertEqual(c.subgoals.arrivals, 0)

    def test_new_obstacle_at_target_overrides_hold_timer(self):
        c = self.controller()
        c.subgoals.min_hold = 100.
        c.subgoals.request_target((2.25, .25))
        c.get_action(self.observation())
        np.testing.assert_allclose(c.drive_target, (2.25, .25))
        c.semantic_map[8, 12] = 2
        c.get_action(self.observation(1.))
        self.assertFalse(np.allclose(c.drive_target, (2.25, .25)))
        self.assertEqual(c.subgoals.last_reason, 'target_invalid')

    def test_arrived_ground_goal_does_not_turn_into_an_exploration_task(self):
        planner = TerrainPlanner()
        planner.observe(np.zeros((3, SIDE, SIDE)), np.zeros(50), crop_origin((0., 0.)), 0.)
        manager = IndependentGroundManager((.5, 0.), planner=planner)
        manager.set_observation((0., 0.), np.ones((16, 16)), .5)
        target, _ = manager.update((0., 0.), None, 0.)
        np.testing.assert_allclose(target, (.5, 0.))
        self.assertTrue(manager.final)
        self.assertIsNone(manager.execution_target)

    def test_aircraft_adapter_keeps_v9_motion_and_adds_only_real_received_packets(self):
        class Base:
            drone_id = 'drone_0'
            navigator = SimpleNamespace(total_drones=3)
            def get_action(self, observation):
                return dict(motion=np.array([1., 2., 0.]), broadcast=dict(drone_id=self.drone_id))
        adapted = make_information_drone_controller(Base)()
        observation = self.observation()
        observation['map_resolution'] = 2.
        observation['neighbors'] = [dict(broadcast={PACKET_KEY: [packet(source='drone_1')]})]
        action = adapted.get_action(observation)
        np.testing.assert_array_equal(action['motion'], (1., 2., 0.))
        self.assertEqual({p['source'] for p in action['broadcast'][PACKET_KEY]}, {'drone_0', 'drone_1'})


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_completion_preserves_received_evidence_without_mutating_inputs(self):
        model = DecoupledWorldModel(hidden=4, members=2)
        maps, states = torch.zeros(1, 3, SIDE, SIDE), torch.zeros(1, 50)
        maps[:, 1, 0, :2] = 1.; maps[:, 0, 0, 0] = 1.
        original = maps.clone()
        result = model.predict(maps, states, torch.zeros(1, 6), torch.ones(1))
        self.assertEqual(float(result['occupancy'][0, 0, 0]), 1.)
        self.assertEqual(float(result['occupancy'][0, 0, 1]), 0.)
        self.assertTrue(torch.equal(maps, original))
        self.assertEqual(float(result['information'][0, 0, :2].sum()), 0.)

    def test_both_commands_condition_future_information_but_not_static_terrain(self):
        model = DecoupledWorldModel(hidden=4, members=2)
        maps, states = torch.zeros(3, 3, SIDE, SIDE), torch.zeros(3, 50)
        actions = torch.tensor(np.stack([action_vector((30., 0.), (0., 40.)),
            action_vector((50., 0.), (0., 40.)), action_vector((30., 0.), (0., 80.))]))
        result = model(maps, states, actions, torch.ones(3)*5)
        self.assertTrue(torch.equal(result['terrain_logits'][:, 0], result['terrain_logits'][:, 1]))
        self.assertFalse(torch.equal(result['information_logits'][:, 0], result['information_logits'][:, 1]))
        self.assertFalse(torch.equal(result['information_logits'][:, 0], result['information_logits'][:, 2]))

    def test_fleet_telemetry_influences_prediction(self):
        model = DecoupledWorldModel(hidden=4, members=1)
        maps, states = torch.zeros(2, 3, SIDE, SIDE), torch.zeros(2, 50)
        states[1, GROUND_FEATURES:GROUND_FEATURES+3] = 1.
        result = model(maps, states, torch.zeros(2, 6), torch.ones(2))
        self.assertFalse(torch.equal(result['information_logits'][:, 0], result['information_logits'][:, 1]))

    def test_prediction_changes_ground_ranking_with_same_observed_candidates(self):
        model = DecoupledWorldModel(hidden=4, members=1)
        planner = TerrainPlanner(model)
        maps = np.zeros((3, SIDE, SIDE), np.float32)
        planner.observe(maps, np.zeros(50), crop_origin((0., 0.)), 0.)
        a = SimpleNamespace(point=np.array([100., 60.]), cost=120.)
        b = SimpleNamespace(point=np.array([100., -60.]), cost=120.)
        low = dict(occupancy=np.zeros((1, SIDE, SIDE)), entropy=np.zeros((1, SIDE, SIDE)),
                   disagreement=np.zeros((1, SIDE, SIDE)))
        high = {key: value.copy() for key, value in low.items()}
        low['occupancy'][:, :SIDE//2] = 1.
        high['occupancy'][:, SIDE//2:] = 1.
        with patch.object(planner, 'predict', return_value=low):
            first = planner.ground_cost(a, (180., 0.))-planner.ground_cost(b, (180., 0.))
            first_intention = (planner.strategic_cost(a.point, (0., 0.), (180., 0.))
                               - planner.strategic_cost(b.point, (0., 0.), (180., 0.)))
        planner.observe(maps, np.zeros(50), crop_origin((0., 0.)), 1.)
        with patch.object(planner, 'predict', return_value=high):
            second = planner.ground_cost(a, (180., 0.))-planner.ground_cost(b, (180., 0.))
            second_intention = (planner.strategic_cost(a.point, (0., 0.), (180., 0.))
                                - planner.strategic_cost(b.point, (0., 0.), (180., 0.)))
        self.assertLess(first, 0.)
        self.assertGreater(second, 0.)
        self.assertLess(first_intention, 0.)
        self.assertGreater(second_intention, 0.)

    def test_dijkstra_cannot_cross_blocked_corners(self):
        costs = np.array([[1., np.inf], [np.inf, 1.]])
        result = distance_field(costs, [(0, 0, 0.)])
        self.assertTrue(np.isinf(result[1, 1]))

    def test_version_dispatch_and_legacy_checkpoint_rejection(self):
        model = make_model(dict(algorithm='vtmp2', model_architecture=ARCHITECTURE, hidden=4, members=1))
        self.assertIsInstance(model, DecoupledWorldModel)
        with TemporaryDirectory() as directory:
            path = Path(directory)/'legacy.pt'
            torch.save(dict(format='vtmp2-world-model-training-v1'), path)
            with self.assertRaisesRegex(ValueError, 'schema-4'):
                TerrainPlanner.from_checkpoint(path)
            config = Path(directory)/'config.json'
            config.write_text(json.dumps(dict(algorithm='vtmp2', model_architecture=ARCHITECTURE)))
            self.assertEqual(Path(build_command(config)[3]).name, 'decoupled_pipeline.py')


class FakeEnvironment:
    def __init__(self, changes=None, geometry='a'*64):
        self.env = SimpleNamespace(time=0.)
        self.goal = np.array([200., 0.]); self.goal_reached = False
        self.scene_geometry_sha256 = geometry
        self.changes = changes or {}
        memory = ReceivedObservation((-300., -300., 300., 300.))
        memory.receive([], 0.)
        memory.add_sensor((0., 0.), np.zeros((16, 16)), .5)
        self.ground_controller = SimpleNamespace(position=np.array([0., 0.]), drive_target=np.array([100., 0.]),
            survey=SimpleNamespace(target=np.array([0., 100.]), completed=0, selected=1),
            observation_memory=memory, subgoals=SimpleNamespace(target_status='verified', last_reason='retained'))
        self.refresh()

    def refresh(self):
        c = self.ground_controller
        c.world_observation = c.observation_memory.encode(c.position, 0., (1., 0.), self.goal,
                                                         c.drive_target, c.survey.target)

    def step(self, frame):
        self.env.time += 1.
        c = self.ground_controller
        c.position[0] += 1.
        if frame in self.changes:
            field, target = self.changes[frame]
            if field == 'drive':
                c.drive_target = np.asarray(target)
            else:
                c.survey.target = np.asarray(target)
        c.observation_memory.receive([], self.env.time)
        self.refresh()


class CollectionTrainingTests(unittest.TestCase):
    def collect(self, directory, split, seed, env, steps=8):
        scene = Path(directory)/f'scene_{seed}.json'
        scene.write_text(json.dumps(dict(split=split, scene=dict(circles=[[[50., 50.], 15.]], world=1000.))))
        job = dict(seed=seed, split=split, steps=steps, output=directory,
                   scene_path=str(scene), scene_sha256='a'*64)
        with patch.object(pipeline, 'provenance', return_value={}), \
             patch.object(pipeline, 'build_environment', return_value=(env, RuntimeError)):
            summary = pipeline.collect_episode(job)
        with np.load(Path(summary['file']), allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
        return summary, data

    def test_either_command_change_splits_samples_and_crop_motion_is_not_information(self):
        env = FakeEnvironment(changes={2: ('survey', (50., 50.)), 5: ('drive', (80., 0.))})
        with TemporaryDirectory() as directory:
            summary, data = self.collect(directory, 'train', 11, env)
        np.testing.assert_array_equal(data['times'], (0., 3., 6., 8.))
        np.testing.assert_allclose(data['actions'][:, :2], ((.1, 0.), (.1, 0.), (.08, 0.)))
        np.testing.assert_allclose(data['actions'][:, 3:5], ((0., .1), (.05, .05), (.05, .05)))
        self.assertEqual(data['newly_observed'].sum(), 0)
        self.assertEqual(summary['ground_distance_during_survey_m'], 8.)
        self.assertFalse(summary['v9_weights_trained'])

    def test_real_training_checkpoint_and_split_isolation(self):
        with TemporaryDirectory() as directory:
            self.collect(directory, 'train', 11, FakeEnvironment())
            self.collect(directory, 'val', 12, FakeEnvironment(geometry='b'*64))
            episodes, manifest = load_dataset(directory)
            self.assertEqual(len(manifest), 2)
            result = train_model(directory, Path(directory)/'training', epochs=2,
                                 hidden=4, members=2, batch_size=2)
            loaded = TerrainPlanner.from_checkpoint(result['checkpoint'])
            self.assertEqual(loaded.selector_mode, 'world_model_independent')
            self.assertEqual(result['epochs'], 2)
            saved = torch.load(result['checkpoint'], weights_only=True)
            self.assertEqual(saved['v9_checkpoint_sha256'], CHECKPOINT_SHA)
            self.assertTrue((Path(directory)/'training/cache/train_11/maps.npy').exists())
            torch.manual_seed(20261112)
            initial = DecoupledWorldModel(hidden=4, members=2)
            self.assertTrue(any(not torch.equal(initial.state_dict()[name], value)
                                for name, value in saved['model'].items()))
            saved['v9_checkpoint_sha256'] = '0'*64
            wrong_actor = Path(directory)/'wrong_actor.pt'
            torch.save(saved, wrong_actor)
            with self.assertRaisesRegex(ValueError, 'v9 aircraft'):
                TerrainPlanner.from_checkpoint(wrong_actor)
            val = episodes['val'][0]
            val['scene_geometry_sha256'] = np.asarray('a'*64)
            np.savez_compressed(Path(directory)/'val_12.npz', **val)
            with self.assertRaisesRegex(ValueError, 'leakage'):
                load_dataset(directory)


if __name__ == '__main__':
    unittest.main()
