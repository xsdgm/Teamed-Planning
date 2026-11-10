"""Independent continuous-geometry and manifest-integrity regressions."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from world_model.feasibility import certify_scene, generate_certified_scene, load_certified_scene


def scene_fixture(start=(5.01, 5.01), goal=(24.01, 5.01), circles=()):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    distance = float(np.linalg.norm(goal-start))
    key = str(int(distance)) if distance.is_integer() else repr(distance)
    points = [(np.floor(start)+.5).tolist(), (np.floor(goal)+.5).tolist()]
    return dict(seed=123, start=start.tolist(), goal=goal.tolist(), world=30., distance=distance,
        goals={key: goal.tolist()}, references={key: {'points': points}},
        reference_resolution_m=1., circles=list(circles))


def document(scene):
    return dict(algorithm='vtmp2', schema_version=1, seed=scene['seed'], split='train',
                scene=scene, certificate=certify_scene(scene))


class FeasibilityTests(unittest.TestCase):
    def test_full_route_includes_real_endpoints(self):
        scene = scene_fixture()
        certificate = certify_scene(scene)
        self.assertTrue(certificate['valid'])
        self.assertGreater(certificate['path_length_m'], scene['distance'])
        self.assertEqual(certificate['segment_count'], 3)
        self.assertAlmostEqual(certificate['boundary_clearance_m'], 5.01)
        self.assertIsNone(certificate['min_obstacle_clearance_m'])

    def test_continuous_segment_crosses_obstacle_despite_clear_waypoints(self):
        scene = scene_fixture(circles=[([15., 5.5], 1.)])
        with self.assertRaisesRegex(ValueError, 'continuous reference segment'):
            certify_scene(scene)

    def test_real_start_to_grid_connector_is_checked(self):
        scene = scene_fixture()
        first, second = np.asarray(scene['start']), np.array([5.5, 5.5])
        center = (first+second)/2 + np.array([-1., 1.])/np.sqrt(2)*1.6
        scene['circles'] = [(center.tolist(), .11)]
        self.assertGreater(np.linalg.norm(center-first)-.11, 1.5)
        self.assertGreater(np.linalg.norm(center-second)-.11, 1.5)
        with self.assertRaisesRegex(ValueError, 'continuous reference segment'):
            certify_scene(scene)

    def test_reference_must_end_in_actual_goal_cell(self):
        scene = scene_fixture()
        next(iter(scene['references'].values()))['points'][-1] = [19.5, 5.5]
        with self.assertRaisesRegex(ValueError, 'actual goal'):
            certify_scene(scene)

    def test_goal_metadata_cannot_describe_a_different_task(self):
        scene = scene_fixture()
        next(iter(scene['goals'].values()))[0] -= 1.
        with self.assertRaisesRegex(ValueError, 'reference goal differs'):
            certify_scene(scene)

    def test_wrong_distance_reference_is_rejected(self):
        scene = scene_fixture()
        scene['references'] = {'20': next(iter(scene['references'].values()))}
        with self.assertRaisesRegex(ValueError, 'actual task distance'):
            certify_scene(scene)

    def test_world_boundary_clearance_checks_internal_waypoints(self):
        scene = scene_fixture()
        next(iter(scene['references'].values()))['points'].insert(1, [.1, 10.])
        with self.assertRaisesRegex(ValueError, 'world-boundary clearance'):
            certify_scene(scene)

    def test_occupied_actual_goal_is_rejected(self):
        scene = scene_fixture()
        scene['circles'] = [(scene['goal'], 1.)]
        with self.assertRaisesRegex(ValueError, 'obstacle clearance'):
            certify_scene(scene)

    def test_nonfinite_geometry_is_rejected(self):
        changes = [lambda s: s['start'].__setitem__(0, float('nan')),
                   lambda s: s.__setitem__('world', float('inf')),
                   lambda s: next(iter(s['references'].values()))['points'][0].__setitem__(0, float('nan')),
                   lambda s: s.__setitem__('circles', [([10., 10.], float('nan'))])]
        for change in changes:
            scene = scene_fixture()
            change(scene)
            with self.subTest(scene=scene), self.assertRaises(ValueError):
                certify_scene(scene)

    def test_missing_reference_fields_are_rejected(self):
        for key in ('start', 'goal', 'references', 'reference_resolution_m', 'circles'):
            scene = scene_fixture()
            del scene[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                certify_scene(scene)

    def test_frozen_clearance_cannot_be_lowered(self):
        with self.assertRaisesRegex(ValueError, 'cannot be below'):
            certify_scene(scene_fixture(), clearance=1.)

    def test_load_rechecks_geometry_and_expected_hash(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'scene.json'
            original = document(scene_fixture())
            path.write_text(json.dumps(original))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            scene, certificate = load_certified_scene(path, expected_sha256=digest)
            self.assertEqual(scene['seed'], 123)
            self.assertTrue(certificate['valid'])
            changed = copy.deepcopy(original)
            changed['scene']['circles'] = [([15., 5.5], 1.)]
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                load_certified_scene(path, expected_sha256=digest)
            with self.assertRaisesRegex(ValueError, 'continuous reference segment'):
                load_certified_scene(path)

    def test_false_stored_certificate_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'scene.json'
            value = document(scene_fixture())
            value['certificate']['path_length_m'] += 1.
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'stored certificate disagrees'):
                load_certified_scene(path)

    def test_generator_uses_frozen_file_and_does_not_overwrite(self):
        # A tiny deterministic stand-in isolates manifest plumbing; exact
        # geometry is covered independently above.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root/'frozen'
            source = frozen/'runtime_sources/simulator/large_world_scenarios.py'
            source.parent.mkdir(parents=True)
            scene = scene_fixture()
            source.write_text('def generate_training_scene(seed, distances, **kwargs):\n'
                              f'    scene = {scene!r}\n'
                              '    scene["seed"] = seed\n'
                              '    return scene\n')
            (frozen/'config.json').write_text(json.dumps({'task_distances': [19.], 'world_size': 30.}))
            record = generate_certified_scene(123, 'train', root/'run', frozen=frozen)
            self.assertTrue(record['certificate']['valid'])
            loaded, _ = load_certified_scene(record['scene_path'], record['scene_sha256'])
            self.assertEqual(loaded['goal'], scene['goal'])
            with self.assertRaises(FileExistsError):
                generate_certified_scene(123, 'train', root/'run', frozen=frozen)
            self.assertEqual(list((root/'run/scenes').glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
