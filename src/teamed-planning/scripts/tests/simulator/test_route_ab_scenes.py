"""Setup-only v11 scene certificates; no training or checkpoint loading."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from drone.cbf_safety import CBFSafetyFilter
from drone.communication import relative_neighborhood, backbone_metrics
from simulator.large_world_scenarios import CircleGeometry
from simulator.route_ab_experiment import make_survey_scene, route_radio_feasible
from simulator.route_ab_scenes import (certify_scene, observation_sweep, validate_sweep,
    sample_certified_scene, selected_distance_band, validate_distance_bands)
from simulator.survey_reward_v10 import SurveyTask

CONFIG = Path(__file__).resolve().parents[3]/'config/algorithm/mappo_training_v11.json'
BANDS = [[20., 60.], [60., 120.], [120., 200.]]


def config(**changes):
    return dict(json.loads(CONFIG.read_text()), **changes)


def fixture(length=200., a=(500., 500.), c=(500., 500.), circles=()):
    b = (a[0]+length, a[1])
    return dict(start=list(c), goal=list(b), circles=list(circles), attempts=1,
        survey_task=SurveyTask(a, b, 12.), survey_stationary=True,
        survey_anchored=bool(np.array_equal(a, c)), survey_envelope_m=224.)


class RouteABSceneTests(unittest.TestCase):
    def test_invalid_bands_are_rejected(self):
        for bands in ([], [[0., 10.]], [[10., 10.]], [[10., 201.]],
                      [[20., 80.], [60., 100.]], [[50., 100.], [10., 20.]],
                      [[True, 10.]], [[20., float('nan')]], '20:200'):
            with self.subTest(bands=bands), self.assertRaises(ValueError):
                validate_distance_bands(dict(route_ab_distance_bands=bands))
        validate_distance_bands(dict(route_ab_distance_bands=BANDS))
        validate_distance_bands(dict(route_ab_distance_bands=None))

    def test_rejection_does_not_change_selected_band(self):
        settings = config(route_ab_distance_bands=BANDS)
        seed = next(s for s in range(100) if selected_distance_band(settings, s)[0] == 2)
        candidates, proposals = iter([fixture(40.), fixture(80.), fixture(180.)]), []

        def sampler(proposal, candidate_seed):
            proposals.append(dict(seed=candidate_seed, bounds=proposal['task_distance_range']))
            return next(candidates)

        scene = sample_certified_scene(settings, seed, sampler, lambda *args: False)
        self.assertEqual(scene['route_ab_sampling_attempts'], 3)
        self.assertEqual(scene['route_ab_distance_band_index'], 2)
        self.assertEqual(scene['route_ab_distance_band'], [120., 200.])
        self.assertEqual(scene['route_ab_distance_m'], 180.)
        self.assertTrue(all(row['bounds'] == proposals[0]['bounds'] for row in proposals))
        self.assertEqual(proposals[0]['bounds'], [95., 200.])

    def test_200m_route_has_a_sequential_connected_mapping_witness(self):
        settings = config(route_ab_distance_bands=BANDS, survey_a=[500., 500.],
            survey_b=[700., 500.], survey_vehicle_start=[500., 500.])
        self.assertFalse(route_radio_feasible((500., 500.), (700., 500.), (500., 500.), 3))
        scene = sample_certified_scene(settings, 17, lambda *args: fixture(), route_radio_feasible)
        certificate = scene['route_ab_feasibility']
        self.assertEqual(scene['route_ab_distance_m'], 200.)
        self.assertLess(certificate['max_required_link_m'], 80.)
        self.assertGreater(certificate['min_continuous_uav_separation_m'], 17.9)
        self.assertLessEqual(certificate['required_steps'], certificate['episode_budget_steps'])
        self.assertTrue(certificate['assumes_stationary_vehicle'])
        self.assertFalse(certificate['moving_vehicle_guaranteed'])
        json.dumps(certificate, allow_nan=False)

    def test_explicit_task_outside_band_support_is_rejected(self):
        settings = config(route_ab_distance_bands=BANDS, survey_a=[500., 500.],
            survey_b=[701., 500.], survey_vehicle_start=[500., 500.])
        with self.assertRaisesRegex(ValueError, 'distance'):
            sample_certified_scene(settings, 17, lambda *args: fixture(201.), route_radio_feasible)

    def test_known_generation_exhaustion_retries_but_programming_errors_propagate(self):
        settings = config(route_ab_distance_bands=BANDS)
        seed = next(s for s in range(100) if selected_distance_band(settings, s)[0] == 2)
        with patch('simulator.route_ab_scenes.certify_scene', return_value={}), \
                patch('simulator.survey_experiment.make_survey_scene',
                    side_effect=[RuntimeError('No feasible large scene for seed 7'), fixture(160.)]) as sampler:
            scene = sample_certified_scene(settings, seed, sampler, route_radio_feasible)
        self.assertEqual(scene['route_ab_sampling_attempts'], 2)
        with self.assertRaisesRegex(RuntimeError, 'unrelated'):
            sample_certified_scene(settings, seed,
                lambda *args: (_ for _ in ()).throw(RuntimeError('unrelated implementation error')),
                route_radio_feasible)

    def test_too_short_episode_rejects_geometrically_feasible_task(self):
        settings = config(steps=20, min_episode_steps=20)
        with self.assertRaisesRegex(ValueError, 'budget'):
            certify_scene(settings, fixture(150.))

    def test_witness_controls_satisfy_real_cbf_with_changing_topology(self):
        c = np.array([500., 500.])
        ground = np.r_[c, 0.]
        frames, observed_at = observation_sweep(c,
            [(480., 500.), (500., 500.), (650., 500.), (650., 540.), (700., 500.)])
        facts = validate_sweep(frames, c, 1000., 1.)
        self.assertLessEqual(facts['max_cbf_constraint_residual'], 1e-7)
        self.assertEqual(len(observed_at), 5)
        # Include entry, angular reversal near C, long-chain states and arrival.
        sample = set(range(0, len(frames)-1, 25)) | {k-1 for k in observed_at} | {len(frames)-2}
        for index in sorted(sample):
            points, following = frames[index:index+2]
            fleet = {f'drone_{i}': point for i, point in enumerate(points)}
            self.assertTrue(backbone_metrics(fleet, ground)[0])
            for i, point in enumerate(points):
                peers = {key: other for key, other in fleet.items()
                         if key != f'drone_{i}' and np.linalg.norm(other-point) <= 95.}
                visible = dict(peers)
                heard_ground = ground if np.linalg.norm(point-ground) <= 95. else None
                if heard_ground is not None:
                    visible['__ground__'] = ground
                desired = following[i]-point
                actual, stats = CBFSafetyFilter().filter(desired, point, peers, 1.,
                    required_links=relative_neighborhood(point, visible, 95.), ground=heard_ground,
                    ground_max_speed=1., world_bounds=(0., 0., 1000., 1000.), drone_id=f'drone_{i}')
                self.assertFalse(stats['infeasible'])
                np.testing.assert_allclose(actual, desired, atol=1e-8)

    def test_actual_120011_is_rejected_or_resampled_to_certified_endpoints(self):
        settings = config(route_ab_distance_bands=BANDS)
        scene = make_survey_scene(settings, 120011)
        task = scene['survey_task']
        certificate = scene['route_ab_feasibility']
        geometry = CircleGeometry(scene['circles'], clearance=certificate['ground_clearance_m'])
        points = certificate['ground_path']
        self.assertTrue(all(geometry.clear(a, b) for a, b in zip(points, points[1:])))
        np.testing.assert_allclose(points[0], task.a)
        np.testing.assert_allclose(points[-1], task.b)
        index, band = selected_distance_band(settings, 120011)
        self.assertEqual(scene['route_ab_distance_band_index'], index)
        self.assertGreaterEqual(scene['route_ab_distance_m'], band[0])
        self.assertLessEqual(scene['route_ab_distance_m'], band[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
