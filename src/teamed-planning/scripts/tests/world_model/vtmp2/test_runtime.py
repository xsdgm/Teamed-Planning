"""The online selector must see current evidence before planning/broadcast."""
from types import SimpleNamespace
import unittest

import networkx as nx
import numpy as np

from world_model.runtime import make_vtmp2_controller


class RuntimeTests(unittest.TestCase):
    def test_selects_once_after_current_sensing_and_fusion(self):
        events = []
        current_graph = object()

        class Manager:
            def __init__(self, goal, **kwargs):
                self.goal = goal

            def set_reconnaissance_range(self, radius):
                self.radius = radius

            def set_observation(self, position, semantic, resolution, blocked=None):
                self.position, self.semantic = position, semantic

            def update(self, position, graph, now):
                events.append('select')
                self.graph, self.now = graph, now
                return np.array([20., 30.]), True

        class Controller:
            subgoal_arrival = 1.
            world_bounds = (0., 0., 1000., 1000.)

            def __init__(self):
                self.position = (0., 0.)
                self.resolution = .5
                self.unified_graph = None
                self.graph_fusion = SimpleNamespace(get_unified_graph=lambda: current_graph)
                self.global_planner = SimpleNamespace(clear_path=lambda: None)
                self.local_planner = SimpleNamespace(safety_distance=1.5,
                    observed_blocked_cells=lambda grid, resolution: (grid == 2, 3))

            def _filter_observed_obstacles(self, graph, semantic):
                return graph

            def _receive_and_fuse_graphs(self, drones, current_time=None):
                events.append('fusion')
                return True

            def _drive_to(self, target):
                self.goal_position = target

            def get_action(self, observation):
                if self.subgoals is not None:
                    raise AssertionError('Stale pre-observation selector was not suppressed')
                events.append('sense')
                self.position = observation['position']
                self._last_semantic_map = observation['semantic']
                self._receive_and_fuse_graphs({}, 7.)
                events.append('plan')
                return {'broadcast': {'target': self.goal_position}}

        adapted = make_vtmp2_controller(Controller, manager_factory=Manager)()
        adapted.set_goal((200., 200.))
        manager = adapted.subgoals
        semantic = np.ones((16, 16), dtype=np.uint8)
        action = adapted.get_action({'position': (5., 6.), 'semantic': semantic,
                                     'full_state': {'time': 7.}})
        self.assertEqual(events, ['sense', 'fusion', 'select', 'plan'])
        self.assertIs(adapted.subgoals, manager)
        self.assertIs(manager.graph, current_graph)
        self.assertIs(manager.semantic, semantic)
        self.assertEqual(manager.position, (5., 6.))
        self.assertEqual(manager.now, 7.)
        self.assertEqual(action['broadcast']['target'], (20., 30.))
        self.assertEqual(adapted.final_goal, (200., 200.))


class MissionRuntimeTests(unittest.TestCase):
    def make_controller(self):
        events, observed, driven = [], [], []
        current = nx.Graph()
        current.add_node('new', position=np.array([25., 30.]), confidence=.8)
        current_graph = SimpleNamespace(graph=current)

        class Planner:
            def reset(self):
                events.append('reset')

            def observe(self, maps, states, now, executed_target=None, executed_purpose='drive'):
                events.append('observe')
                observed.append(dict(maps=maps.copy(), states=states.copy(), now=now,
                                     executed_target=np.asarray(executed_target).copy(),
                                     executed_purpose=executed_purpose))
                # Predictions belong to the model, never to received evidence.
                self.predicted_maps = np.full_like(maps, .99)
                self.predicted_states = np.full_like(states, .99)

        class Manager:
            def __init__(self, goal, planner, **kwargs):
                self.planner = planner
                self.goal = goal
                self.target = np.array([40., 0.])
                self.purpose = 'drive'
                self.drive_target = self.target.copy()
                self.calls = 0

            def set_reconnaissance_range(self, radius):
                self.radius = radius

            def set_observation(self, position, semantic, resolution, blocked=None):
                self.semantic = semantic

            def update(self, position, graph, now):
                events.append('select')
                self.calls += 1
                self.purpose = 'survey' if self.calls == 1 else 'drive'
                self.target = np.array([80., 60.]) if self.calls == 1 else np.array([90., 40.])
                self.drive_target = None if self.purpose == 'survey' else self.target.copy()
                return self.target.copy(), True

        class Controller:
            subgoal_arrival = 1.
            world_bounds = (0., 0., 1000., 1000.)

            def __init__(self):
                self.position = (0., 0.)
                self.heading = 0.
                self.resolution = .5
                self.unified_graph = SimpleNamespace(graph=nx.Graph())
                self.graph_fusion = SimpleNamespace(get_unified_graph=lambda: current_graph)
                self.global_planner = SimpleNamespace(clear_path=lambda: events.append('clear'))
                self.local_planner = SimpleNamespace(safety_distance=1.5,
                    observed_blocked_cells=lambda grid, resolution: (grid == 2, 0))
                self.last_fusion_per_drone = {}
                self.current_drone_communications = {}

            def _filter_observed_obstacles(self, graph, semantic):
                return graph

            def _receive_and_fuse_graphs(self, drones, current_time=None):
                events.append('fusion')
                return True

            def _drive_to(self, target):
                driven.append(target)
                self.goal_position = target

            def get_action(self, observation):
                self.position = observation['position']
                self._last_semantic_map = observation['semantic']
                self.last_fusion_per_drone = {'drone_0': observation['full_state']['time']}
                self.current_drone_communications = {'drone_0': {'received': True}}
                events.append('sense')
                self._receive_and_fuse_graphs({}, observation['full_state']['time'])
                events.append('plan')
                return dict(motion=(1., .25), broadcast=dict(target=self.goal_position,
                                                           planned_path=[(1., 2.)]))

        planner = Planner()
        controller = make_vtmp2_controller(Controller, manager_factory=Manager,
                                           planner_factory=lambda: planner)()
        controller.set_goal((200., 200.))
        return controller, planner, current_graph, events, observed, driven

    def test_survey_broadcast_is_independent_of_ground_target_and_drive_resumes(self):
        controller, planner, graph, events, observed, driven = self.make_controller()
        observation = dict(position=(5., 6.), semantic=np.ones((16, 16), dtype=np.uint8),
                           full_state={'time': 7.})
        survey = controller.get_action(observation)
        self.assertEqual(survey['motion'], (0., 0.))
        self.assertEqual(survey['broadcast']['target'], [80., 60.])
        self.assertEqual(survey['broadcast']['task_purpose'], 'survey')
        self.assertIsNone(survey['broadcast']['vehicle_target'])
        self.assertEqual(survey['broadcast']['planned_path'], [])
        self.assertIsNone(controller.goal_position)
        self.assertIsNone(controller.subgoals.drive_target)
        self.assertEqual(driven, [])
        self.assertEqual(controller.final_goal, (200., 200.))
        self.assertEqual(events.count('select'), 1)
        self.assertEqual(len(observed), 2)
        observation['full_state']['time'] = 8.
        drive = controller.get_action(observation)
        self.assertEqual(drive['motion'], (1., .25))
        self.assertEqual(drive['broadcast']['task_purpose'], 'drive')
        self.assertEqual(drive['broadcast']['target'], [90., 40.])
        self.assertEqual(drive['broadcast']['vehicle_target'], [90., 40.])
        self.assertEqual(controller.goal_position, (90., 40.))
        self.assertEqual(driven, [(90., 40.)])
        self.assertEqual(controller.final_goal, (200., 200.))
        self.assertEqual(events.count('select'), 2)
        self.assertEqual(len(observed), 4)
        self.assertEqual([item['executed_purpose'] for item in observed],
                         ['drive', 'survey', 'survey', 'drive'])
        np.testing.assert_array_equal(observed[0]['executed_target'], [40., 0.])
        np.testing.assert_array_equal(observed[1]['executed_target'], [80., 60.])
        np.testing.assert_array_equal(observed[2]['executed_target'], [80., 60.])
        np.testing.assert_array_equal(observed[3]['executed_target'], [90., 40.])
        for before, after, expected_time in ((observed[0], observed[1], 7.),
                                             (observed[2], observed[3], 8.)):
            self.assertEqual(before['now'], expected_time)
            self.assertEqual(after['now'], expected_time)
            np.testing.assert_array_equal(before['maps'], after['maps'])
            np.testing.assert_array_equal(before['states'], after['states'])

    def test_model_observes_post_fusion_evidence_without_writing_predictions_back(self):
        controller, planner, graph, events, observed, driven = self.make_controller()
        semantic = np.ones((16, 16), dtype=np.uint8)
        semantic[8, 8] = 2
        controller.get_action(dict(position=(5., 6.), semantic=semantic, full_state={'time': 7.}))
        self.assertLess(events.index('sense'), events.index('fusion'))
        self.assertLess(events.index('fusion'), events.index('observe'))
        self.assertLess(events.index('observe'), events.index('select'))
        self.assertLess(events.index('select'), events.index('plan'))
        observation_events = [i for i, value in enumerate(events) if value == 'observe']
        self.assertEqual(len(observation_events), 2)
        self.assertLess(events.index('select'), observation_events[1])
        self.assertLess(observation_events[1], events.index('plan'))
        self.assertEqual(float(observed[0]['maps'][0].sum()), float(np.float32(.2)))
        self.assertEqual(float(observed[0]['maps'][3].sum()), 4.)
        np.testing.assert_allclose(observed[0]['states'][:2], [.005, .006])
        np.testing.assert_allclose(observed[0]['states'][6:8], [.2, .2])
        self.assertEqual(observed[0]['states'][10], 0.)
        self.assertEqual(observed[0]['states'][13], 1.)
        self.assertIs(controller.unified_graph, graph)
        np.testing.assert_array_equal(graph.graph.nodes['new']['position'], [25., 30.])
        self.assertEqual(graph.graph.nodes['new']['confidence'], .8)
        np.testing.assert_array_equal(controller._last_semantic_map, semantic)
        controller.get_action(dict(position=(6., 6.), semantic=semantic, full_state={'time': 9.}))
        np.testing.assert_allclose(observed[2]['states'][4:6], [.5, 0.])
        self.assertNotAlmostEqual(float(observed[2]['maps'].mean()), .99)


if __name__ == '__main__':
    unittest.main()
