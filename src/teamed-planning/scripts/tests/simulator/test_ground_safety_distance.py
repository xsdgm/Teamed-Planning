"""One ground clearance contract across aerial maps, driving, and bridging."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.ndimage import label
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from drone.cograph_v2_mappo import CoGraphV2ActorCritic
from drone.relay_attention_mappo import RelayActorCritic
from drone.semantic_segmentation import TerrainType
from drone.drone_main import DroneController
from drone.skeleton_graph_builder import SkeletonGraphBuilder
from ground.ground_main import GroundRobotController
from common.navigation_config import DEFAULT_GROUND_CLEARANCE_M
from ground.local_planner import LocalPlanner
from simulator.cograph_experiment import CoGraphExperiment
from simulator.mappo_experiment import Experiment, BRIDGE_CLEARANCE_M
from simulator.large_world_scenarios import CircleGeometry, generate_large_scene
from simulator.train_mappo import optimizer_config_changes, validate_config


CONFIGS = Path(__file__).resolve().parents[3] / 'config' / 'algorithm'


def config(version):
    return json.loads((CONFIGS / f'mappo_training_{version}.json').read_text())


class GroundSafetyDistanceConfigTests(unittest.TestCase):
    def test_every_training_version_defaults_to_one_metre(self):
        paths = sorted(CONFIGS.glob('mappo_training*.json')) + sorted(CONFIGS.glob('cograph_v*.json'))
        for path in paths:
            with self.subTest(config=path.name):
                settings = json.loads(path.read_text())
                self.assertEqual(validate_config(settings)['ground_safety_distance'], 1.)
                settings.pop('ground_safety_distance', None)
                self.assertEqual(validate_config(settings)['ground_safety_distance'], 1.)

    def test_explicit_historical_margin_is_preserved(self):
        settings = dict(config('v9'), ground_safety_distance=1.5)
        self.assertEqual(validate_config(settings)['ground_safety_distance'], 1.5)

    def test_reference_geometry_uses_selected_clearance(self):
        circles = [([15., 15.], 1.)]
        start, goal = [10., 17.25], [20., 17.25]
        self.assertTrue(CircleGeometry(circles, world=30.).clear(start, goal))
        self.assertFalse(CircleGeometry(circles, world=30., clearance=1.5).clear(start, goal))
        for clearance in (1., 1.5):
            scene = generate_large_scene(5, distances=(20.,), world=100., density=.001,
                                         ground_safety_distance=clearance)
            self.assertEqual(scene['reference_geometric_clearance_m'], clearance)
            self.assertAlmostEqual(scene['reference_clearance_m'], clearance+np.sqrt(.5))

    def test_invalid_margins_are_rejected(self):
        for distance in (True, False, '1.0', None, 0., -1., np.nan, np.inf, -np.inf):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                validate_config(dict(config('v9'), ground_safety_distance=distance))

    def test_implicit_historical_margin_cannot_silently_resume_at_one_metre(self):
        previous = config('v9')
        previous.pop('ground_safety_distance', None)
        for allow_upgrade in (False, True):
            with self.subTest(allow_upgrade=allow_upgrade), self.assertRaises(ValueError):
                optimizer_config_changes(previous, config('v9'), allow_upgrade)
        replay = dict(previous, ground_safety_distance=1.5)
        self.assertEqual(optimizer_config_changes(previous, replay), {})

    def test_margin_change_cannot_be_hidden_in_optimizer_resume_upgrade(self):
        original = validate_config(dict(config('v9'), ground_safety_distance=1.5))
        changed = validate_config(dict(original, ground_safety_distance=1.))
        self.assertEqual(optimizer_config_changes(original, dict(original)), {})
        for allow_upgrade in (False, True):
            with self.subTest(allow_upgrade=allow_upgrade), self.assertRaises(ValueError):
                optimizer_config_changes(original, changed, allow_upgrade=allow_upgrade)


class GroundSafetyDistanceComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.single_model = RelayActorCritic().eval()
        cls.team_model = CoGraphV2ActorCritic().eval()

    def setUp(self):
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def experiment(self, team=False, **options):
        if team:
            scene = dict(start=[100., 100.], goals=[[250., 100.], [100., 250.]],
                         attempts=1, vehicle_starts=[[100., 100.], [100., 114.], [100., 86.]],
                         circles=[], reference_polyline_lengths=[150., 150.])
            env = CoGraphExperiment(23, self.team_model, scene, num_drones=3,
                                    fusion_bridge_distance=15., **options)
        else:
            env = Experiment(23, self.single_model, communication_mode='backbone',
                             safety_filter='cbf', moving_ground=True, start=(30., 60.),
                             scene=([95., 60.], [], 1), fusion_bridge_distance=15., **options)
        self.addCleanup(env.env.close)
        return env

    def check_components_and_bridge(self, env, expected):
        self.assertEqual(env.ground_safety_distance, expected)
        for drone in env.controllers:
            self.assertEqual(drone.graph_builder.min_clearance, expected)
        vehicles = env.vehicles if hasattr(env, 'vehicles') else [env.ground_controller]
        for vehicle in vehicles:
            self.assertEqual(vehicle.local_planner.safety_distance, expected)
            self.assertEqual(vehicle.local_planner.escape_distance, .5)

        # Delivered 2 m cells: a horizontal segment at y=14.75 has free
        # evidence through y<16. The 1 m strip stays free, while 1.5 m reaches
        # the occupied cell at x=[22,24), y=[16,18).
        env.ledger.seen[:] = True
        env.ledger.log_odds[:] = -3.
        env.ledger.log_odds[8, 11] = 3.
        start, end = (20., 14.75), (23.7, 14.75)
        for vehicle in vehicles:
            check = vehicle.graph_fusion.bridge_check
            self.assertEqual(check(start, end), expected == 1.)
        # Reducing clearance never authorizes a wholly unobserved segment.
        env.ledger.seen[:] = False
        for vehicle in vehicles:
            self.assertFalse(vehicle.graph_fusion.bridge_check(start, end))

    def test_2d_defaults_preserve_legacy_ros_controller_interfaces(self):
        self.assertEqual(DEFAULT_GROUND_CLEARANCE_M, 1.)
        self.assertEqual(BRIDGE_CLEARANCE_M, DEFAULT_GROUND_CLEARANCE_M)
        self.assertEqual(LocalPlanner('default').safety_distance, 1.)
        # Unchanged ROS nodes construct these interfaces without overrides.
        self.assertEqual(GroundRobotController('default').local_planner.safety_distance, 1.5)
        self.assertEqual(DroneController('default').graph_builder.min_clearance, 3.)
        self.assertEqual(SkeletonGraphBuilder('default').min_clearance, 1.)

    def test_single_vehicle_components_use_default_and_configured_clearance(self):
        for options, expected in (({}, 1.), ({'ground_safety_distance': 1.5}, 1.5)):
            with self.subTest(options=options):
                self.check_components_and_bridge(self.experiment(**options), expected)

    def test_cograph_components_use_default_and_configured_clearance(self):
        for options, expected in (({}, 1.), ({'ground_safety_distance': 1.5}, 1.5)):
            with self.subTest(options=options):
                self.check_components_and_bridge(self.experiment(team=True, **options), expected)

    def test_one_metre_clearance_opens_a_corridor_closed_by_one_point_five(self):
        # A 2.5 m observed corridor at 0.5 m resolution. Inflation by 1 m
        # leaves its centre row connected; inflation by 1.5 m closes it.
        semantic = np.full((27, 27), TerrainType.OBSTACLE.value, np.uint8)
        semantic[11:16, :] = TerrainType.TRAVERSABLE.value
        start, end = (13, 4), (13, 22)
        for distance, connected in ((1., True), (1.5, False)):
            with self.subTest(distance=distance):
                blocked, _ = LocalPlanner('corridor', safety_distance=distance).observed_blocked_cells(
                    semantic, .5)
                components, _ = label(~blocked)
                reachable = bool(components[start] and components[start] == components[end])
                self.assertEqual(reachable, connected)
                self.assertTrue(np.all(blocked[semantic == TerrainType.OBSTACLE.value]))


if __name__ == '__main__':
    unittest.main()
