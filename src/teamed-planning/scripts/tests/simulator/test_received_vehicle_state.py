"""Reception freshness and goal-independent radio-anchor regressions (no training)."""
import copy
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from drone.communication import ChainState
from drone.cooperative_mappo import CooperativeNavigator
from drone.drone_main import DroneController
from drone.survey_mappo import RouteABActorCritic, survey_features
from drone.graph_attention_mappo import GraphAttentionActorCritic
from drone.velocity_mappo import VelocityNavigator


def observation(now=5., stamp=5., target=(180., 50.), x=30., direct=True):
    return dict(time=now, graph=None, nearby_drones={}, occupancy_map=np.zeros((12, 12)),
        ground_robots={'ground_0': dict(position=[x, 50., 0.], target=target,
            source_time=stamp, direct=direct,
            survey=dict(a=[80., 50.], b=[180., 50.], half_width=12.))},
        cooperation=dict(peers={}, velocities={}, ground=np.array([x, 50., 0.]),
            required=('__ground__',)), relay=[])


def parse(neighbors, now=5., multi=False):
    device = object.__new__(DroneController)
    device.drone_id, device.multi_vehicle = 'drone_0', multi
    return device._parse_simulator_observation(dict(
        self=dict(position=[50., 50.], height=15.), occupancy_map=np.zeros((12, 12)),
        full_state=dict(time=now), neighbors=neighbors))['ground_robots']


def forwarded(stamp, name='drone_1', missions=None):
    missions = observation(stamp=stamp)['ground_robots'] if missions is None else missions
    return dict(type='drone', broadcast=dict(drone_id=name, position=[70., 50., 15.],
        ground_missions=copy.deepcopy(missions)))


class ReceivedMissionTests(unittest.TestCase):
    def test_forwarded_selection_is_order_independent_for_single_and_multi_vehicle(self):
        for multi in (False, True):
            for stamps in itertools.permutations((1., 4., 3.)):
                with self.subTest(multi=multi, stamps=stamps):
                    report = parse([forwarded(s, str(i)) for i, s in enumerate(stamps)], multi=multi)
                    self.assertEqual(report['ground_0']['source_time'], 4.)
                    self.assertFalse(report['ground_0']['direct'])

    def test_fresh_command_survives_cyclic_forwarding(self):
        history = [parse([forwarded(0.)]), parse([forwarded(0.)])]
        for now in range(1, 16):
            previous = history
            history = [parse([forwarded(0., 'echo', previous[1-i]), forwarded(now-1., 'gateway')], now)
                       for i in (0, 1)]
            for reports in history:
                self.assertEqual(reports['ground_0']['source_time'], now-1.)
                self.assertEqual(survey_features([50., 50., 15.],
                    dict(time=now, ground_robots=reports))[-1], 1.)

    def test_direct_report_has_priority_and_forwarding_keeps_source_stamp(self):
        direct = dict(type='diff_drive', broadcast=dict(robot_id='ground_0',
            position=[30., 50., 0.], target=None, timestamp=4.))
        for packets in ([direct, forwarded(5.)], [forwarded(5.), direct]):
            report = parse(packets)['ground_0']
            self.assertTrue(report['direct'])
            self.assertEqual(report['source_time'], 4.)
        missions = parse([forwarded(1.)])
        for now in (6., 9., 12.):
            missions = parse([forwarded(1., missions=missions)], now)
            self.assertEqual(missions['ground_0']['source_time'], 1.)
        self.assertEqual(survey_features([50., 50., 15.],
            dict(time=12., ground_robots=missions))[-1], 0.)


class VehicleCacheTests(unittest.TestCase):
    def test_fresh_empty_goal_clears_navigation_but_keeps_vehicle_position(self):
        chain = ChainState('drone_0', 1)
        chain.update(observation(now=0., stamp=0.))
        self.assertIsNotNone(chain.task())
        chain.update(observation(now=1., stamp=1., target=None, x=31.))
        self.assertIsNone(chain.task())
        self.assertEqual(chain.vehicle_state()['position'][0], 31.)
        np.testing.assert_array_equal(chain.visible['__ground__'], [31., 50., 0.])
        # A delayed prior mission must not resurrect the cleared goal.
        chain.update(observation(now=2., stamp=0., direct=False))
        self.assertIsNone(chain.task())
        self.assertEqual(chain.vehicle_state()['position'][0], 31.)

    def test_position_expires_from_source_time_without_refresh_on_old_copies(self):
        chain = ChainState('drone_0', 1)
        chain.update(observation(now=5., stamp=5., target=None, direct=False))
        self.assertNotIn('__ground__', chain.visible)
        chain.update(observation(now=15., stamp=5., target=None, direct=False))
        self.assertIsNotNone(chain.vehicle_state())
        chain.update(observation(now=15.001, stamp=5., target=None, direct=False))
        self.assertIsNone(chain.vehicle_state())
        chain.update(observation(now=16., stamp=20., target=None))
        self.assertIsNone(chain.vehicle_state())

    def test_goal_less_direct_vehicle_still_enters_chain_safety_filter(self):
        device = object.__new__(DroneController)
        device.drone_id, device.communication_mode = 'drone_0', 'chain'
        device.position, device.communication_range, device.max_speed = [50., 50., 15.], 95., 6.
        chain = ChainState('drone_0', 1)
        chain.update(observation(target=None))
        device._chain_state = chain
        device.navigator = SimpleNamespace(chain=chain, min_separation=15.)
        with patch('drone.cbf_safety.CBFSafetyFilter.filter', return_value=(np.zeros(3), {})) as filt:
            device._cbf_filter(np.zeros(3), dict(neighbors=[]), 1.)
        np.testing.assert_array_equal(filt.call_args.kwargs['ground'], [30., 50., 0.])


class VehicleFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_v11_vehicle_features_follow_fresh_positions_after_goal_clears(self):
        nav = CooperativeNavigator('drone_0', total_drones=3, model=RouteABActorCritic())
        nav.cbf_margin_source = lambda: None
        nav.encode([50., 50., 15.], observation(now=0., stamp=0.))
        for now in range(1, 16):
            x = 30.+now
            features, _ = nav.encode([50., 50., 15.],
                observation(now=now, stamp=now, target=None, x=x))
            np.testing.assert_allclose(features[6:8], [(x-50.)/95., 0.], atol=1e-7)
            np.testing.assert_array_equal(features[[4, 5, 8, 9, 10]], np.zeros(5))
            self.assertEqual(features[-1], 1.)
        expired, _ = nav.encode([50., 50., 15.], dict(time=26., ground_robots={},
            cooperation=dict(peers={}, velocities={}, ground=None, required=()), relay=[]))
        np.testing.assert_array_equal(expired[4:12], np.zeros(8))

    def test_legacy_target_present_prefix_retains_values_for_both_input_versions(self):
        for version in (1, 2):
            nav = VelocityNavigator('drone_0', total_drones=3,
                model=GraphAttentionActorCritic(input_version=version))
            nav.cbf_margin_source = lambda: None
            features, _ = nav.encode([50., 50., 15.], observation())
            expected = [130./350., 0., -20./95., 0., 150./350., 0., 1., 25./95.]
            np.testing.assert_allclose(features[4:12], expected, rtol=1e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
