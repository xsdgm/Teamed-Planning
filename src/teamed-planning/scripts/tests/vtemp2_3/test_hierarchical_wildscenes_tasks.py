"""True metric endpoint tasks from independently audited offline polylines."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vtemp2_3.wildscenes_tasks import load_verified_routes, sample_task_manifest


def route(points=None):
    return dict(route_id='synthetic-audited-route', all_segments_clear=True, required_clearance_m=1.5,
        reference_points=points or [[20., 20.], [1420., 20.], [1420., 1420.]],
        map_contract=dict(terrain='wildscenes_raster', map_path='/tmp/test-audited-map.npz',
            map_sha256='a' * 64, world_size=3000., kind='wildscenes', circles=[], resolution_m=.25,
            required_clearance_m=1.5, source_unknown_space='blocked', map_group='Karawatha'))


class WildScenesTaskTests(unittest.TestCase):
    def test_bent_route_keeps_endpoint_distance_and_reference_arc_distinct(self):
        rows, audit = sample_task_manifest([route()], [[100., 200.], [1000., 1000.]], 4,
            [1000., 1000.], 9, 100, return_audit=True)
        self.assertEqual((len(rows['train']), len(rows['validation'])), (8, 9))
        self.assertEqual(audit['update_ranges_m'], [[100., 200.], [1000., 1000.]])
        self.assertEqual(audit['validation_scope'], 'heldout_tasks')
        self.assertEqual(len(audit['segments']), 17)
        self.assertTrue(any(c['reference_distance_m'] > c['direct_distance_m'] + 1.
                            for c in rows['validation']))
        for split in ('train', 'validation'):
            for i, case in enumerate(rows[split]):
                actual = float(np.linalg.norm(np.array(case['goal']) - case['start']))
                self.assertAlmostEqual(actual, case['direct_distance_m'], places=9)
                self.assertGreaterEqual(case['reference_distance_m'] + 1e-8, actual)
                self.assertNotIn('reference_points', case)
                if split == 'validation' or i >= 4:
                    self.assertGreaterEqual(actual, 1000.)
                    self.assertLessEqual(actual, 1000. + 1e-6)
        self.assertEqual(len({c['seed'] for split in rows for c in rows[split]}), 17)

    def test_sampling_reproducible_and_same_reverse_endpoints_excluded(self):
        args = ([route()], [[30., 60.], [400., 600.]], 6, [1000., 1000.], 9, 8)
        first = sample_task_manifest(*args)
        self.assertEqual(first, sample_task_manifest(*args))
        cases = first['train'] + first['validation']
        for i, left in enumerate(cases):
            for right in cases[i + 1:]:
                a, b = np.array([left['start'], left['goal']]), np.array([right['start'], right['goal']])
                self.assertFalse((np.linalg.norm(a - b, axis=1) <= 2.).all())
                self.assertFalse((np.linalg.norm(a - b[::-1], axis=1) <= 2.).all())

    def test_long_winding_arc_cannot_replace_kilometre_endpoint_separation(self):
        points = [[20. + 100. * (i % 2), 20. + i] for i in range(60)]
        self.assertGreater(np.linalg.norm(np.diff(points, axis=0), axis=1).sum(), 5000.)
        with self.assertRaisesRegex(ValueError, 'Euclidean diameter'):
            sample_task_manifest([route(points)], [[30., 60.]], 1, [1000., 1000.], 1, 8)

    def test_insufficient_span_or_unique_endpoint_capacity_fails_without_relaxation(self):
        with self.assertRaisesRegex(ValueError, 'cannot cover requested'):
            sample_task_manifest([route([[20., 20.], [1120., 20.]])], [[30., 60.]], 1,
                                 [1000., 1200.], 1, 8)
        with self.assertRaisesRegex(ValueError, 'no range or clearance was relaxed'):
            sample_task_manifest([route([[20., 20.], [1020., 20.]])], [[30., 60.]], 1,
                                 [1000., 1000.], 2, 8, max_attempts_per_task=30)

    def test_exact_distance_can_be_sampled_and_unverified_routes_are_rejected(self):
        manifest = sample_task_manifest([route([[20., 20.], [2520., 20.]])], [[1000., 1000.]],
            4, [1000., 1000.], 9, 17)
        self.assertTrue(all(1000. <= c['direct_distance_m'] <= 1000. + 1e-6
                            for split in manifest for c in manifest[split]))
        for field, value in [('all_segments_clear', False), ('required_clearance_m', .3)]:
            invalid = route(); invalid[field] = value
            with self.assertRaisesRegex(ValueError, 'previously audited'):
                sample_task_manifest([invalid], [[30., 60.]], 1, [1000., 1000.], 1, 1)

    def test_nonfinite_ranges_bool_counts_and_unknown_free_are_rejected(self):
        for bounds in ([0., 1.], [2., 1.], [1., float('inf')]):
            with self.assertRaises(ValueError):
                sample_task_manifest([route()], [bounds], 1, [1000., 1000.], 1, 1)
        with self.assertRaises(ValueError):
            sample_task_manifest([route()], [[30., 60.]], True, [1000., 1000.], 1, 1)
        invalid = route(); invalid['map_contract']['source_unknown_space'] = 'free_by_simulation_assumption'
        with self.assertRaisesRegex(ValueError, 'remain blocked'):
            sample_task_manifest([invalid], [[30., 60.]], 1, [1000., 1000.], 1, 1)

    def test_geometry_acceptance_rejects_before_consuming_pair_and_preserves_distance(self):
        checked = []

        def geometry(case):
            accepted = case['start'][0] >= 1000.
            checked.append(accepted)
            # A callback cannot mutate endpoints or distance in the manifest.
            case['start'][0] = -100.
            case['direct_distance_m'] = 0.
            return accepted

        manifest, audit = sample_task_manifest([route([[20., 20.], [2520., 20.]])],
            [[1000., 1000.]], 2, [1000., 1000.], 2, 17,
            case_acceptance=geometry, return_audit=True)
        self.assertEqual((len(manifest['train']), len(manifest['validation'])), (2, 2))
        self.assertTrue(audit['case_acceptance_enabled'])
        self.assertEqual(sum(audit['geometry_acceptance_checks'].values()), len(checked))
        self.assertEqual(sum(audit['geometry_acceptance_rejections'].values()), checked.count(False))
        self.assertGreater(checked.count(False), 0)
        self.assertEqual(sum(row['geometry_acceptance_rejections'] for row in audit['segments']),
                         checked.count(False))
        for split in manifest:
            for case in manifest[split]:
                self.assertGreaterEqual(case['start'][0], 1000.)
                self.assertGreaterEqual(case['direct_distance_m'], 1000.)
                self.assertLessEqual(case['direct_distance_m'], 1000. + 1e-6)

    def test_all_geometry_rejections_are_bounded_and_report_acceptance_capacity(self):
        checked = []

        def reject(case):
            checked.append(case)
            return False

        with self.assertRaisesRegex(ValueError, 'capacity/acceptance insufficient.*geometry rejections'):
            sample_task_manifest([route([[20., 20.], [2520., 20.]])], [[30., 60.]], 1,
                [1000., 1000.], 1, 17, max_attempts_per_task=20, case_acceptance=reject)
        self.assertGreater(len(checked), 0)
        self.assertLessEqual(len(checked), 20)
        self.assertTrue(all(1000. <= row['direct_distance_m'] <= 1000. + 1e-6 for row in checked))
        with self.assertRaisesRegex(TypeError, 'return a boolean'):
            sample_task_manifest([route()], [[30., 60.]], 1, [1000., 1000.], 1, 17,
                                 case_acceptance=lambda case: {'accepted': False})

    def test_explicit_simplified_protocol_and_separate_validation_routes_are_preserved(self):
        train, validation = route(), route()
        for r in (train, validation):
            r['map_contract'].update(terrain='wildscenes_simplified_raster',
                source_unknown_space='free_by_simulation_assumption', coordinate_scale=1.,
                source_scene_path='/tmp/explicit-simplified-scene.json', source_scene_sha256='c' * 64)
        validation['map_contract'].update(map_group='Venman', map_sha256='b' * 64,
                                          map_path='/tmp/validation-simplified-map.npz')
        manifest, audit = sample_task_manifest([train], [[100., 200.], [1000., 1000.]], 4,
            [1000., 1000.], 9, 19, validation_routes=[validation], return_audit=True)
        self.assertEqual((len(manifest['train']), len(manifest['validation'])), (8, 9))
        self.assertEqual({c['map_group'] for c in manifest['train']}, {'Karawatha'})
        self.assertEqual({c['map_group'] for c in manifest['validation']}, {'Venman'})
        self.assertEqual(audit['validation_scope'], 'heldout_map_groups')
        self.assertEqual(audit['terrain'], 'wildscenes_simplified_raster')
        self.assertEqual(audit['source_unknown_space'], 'free_by_simulation_assumption')
        self.assertTrue(all(c['source_scene_sha256'] == 'c' * 64 and c['coordinate_scale'] == 1.
                            for split in manifest for c in manifest[split]))
        with self.assertRaisesRegex(ValueError, 'Do not mix'):
            sample_task_manifest([route()], [[100., 200.]], 1, [1000., 1000.], 1, 19,
                                 validation_routes=[validation])
        del train['map_contract']['source_scene_sha256']
        with self.assertRaisesRegex(ValueError, 'original scene SHA-256'):
            sample_task_manifest([train], [[100., 200.]], 1, [1000., 1000.], 1, 19)

    def test_supervisor_audit_loader_binds_map_metadata_endpoints_and_route_length(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            map_path = directory / 'map.npz'
            np.savez_compressed(map_path, known=np.ones((64, 64), bool),
                                occupied=np.zeros((64, 64), bool), resolution_m=np.array(40.))
            sha = hashlib.sha256(map_path.read_bytes()).hexdigest()
            metadata_path = directory / 'map_metadata.json'
            metadata_path.write_text(json.dumps(dict(map_sha256=sha, coordinate_scale=1., resolution_m=40.)))
            source = route([[20., 20.], [1420., 20.], [1420., 1420.]])
            contract = source['map_contract']
            contract.update(map_path=str(map_path), map_sha256=sha,
                            metadata_path=str(metadata_path), resolution_m=40.)
            points = source['reference_points']
            case = dict(contract, case_id='case', start=points[0], goal=points[-1],
                direct_distance_m=float(np.linalg.norm(np.array(points[-1]) - points[0])),
                reference_distance_m=2800.)
            audit = dict(status='verified_kilometer_tasks_available', map_path=str(map_path), map_sha256=sha,
                required_clearance_m=1.5, source_unknown_space='blocked', kilometer_cases=[case],
                route_audits=[dict(case_id='case', reference_points=points, all_segments_clear=True,
                                   required_clearance_m=1.5)])
            path = directory / 'audit.json'; path.write_text(json.dumps(audit))
            loaded = load_verified_routes(path)
            self.assertEqual(loaded[0]['reference_points'], points)
            self.assertEqual(loaded[0]['map_contract']['metadata_sha256'],
                             hashlib.sha256(metadata_path.read_bytes()).hexdigest())
            for key, value in [('reference_distance_m', 2000.), ('start', [21., 20.])]:
                invalid = deepcopy(audit); invalid['kilometer_cases'][0][key] = value
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):load_verified_routes(path)
            invalid = deepcopy(audit); invalid['route_audits'][0]['all_segments_clear'] = False
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, 'exact-segment'):load_verified_routes(path)
            path.write_text(json.dumps(audit)); map_path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA-256 changed'):load_verified_routes(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
