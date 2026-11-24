from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from shared.simulator import large_world_scenarios
from vtemp4 import vtemp4_local_curriculum as local
from vtemp2_3.scene_manifest import load_v9_circles
from vtemp4.vtemp4_curriculum import runtime_case


class LocalCurriculumTests(TestCase):
    start = np.array([120., 300.])
    goal = np.array([1320., 300.])

    def test_centreline_must_be_blocked_and_passable_nearby(self):
        open_route = np.array([[350., 330., 4.]])
        single_tree = np.array([[350., 300., 4.]])
        wide_wall = np.array([[350., y, 4.] for y in range(252, 349, 5)])
        for circles, expected in ((open_route, 0), (single_tree, 1), (wide_wall, 0)):
            self.assertEqual(local._local_route(large_world_scenarios, circles, self.start, self.goal), expected)

    def test_native_cache_split_hash_and_policy_fields(self):
        def generated(seed, **settings):
            return dict(start=self.start.tolist(), goal=self.goal.tolist(), distance=1200.,
                        circles=[([320. + seed % 100, 300.], 4.)], reference_polyline_length=1202.,
                        reference_geometry_validated=True, reference_geometric_clearance_m=1.5)

        with (TemporaryDirectory() as folder, patch.object(local, '_generator', return_value=large_world_scenarios),
              patch.object(large_world_scenarios, 'generate_training_scene', side_effect=generated) as generate):
            train = local.local_case('.', 7, geometry_dir=folder)
            dev = local.local_case('.', 7, geometry_dir=folder, split='development')
            self.assertEqual(train, local.local_case('.', 7, geometry_dir=folder))
            self.assertEqual(generate.call_count, 2)
            self.assertNotEqual(train['circles_sha256'], dev['circles_sha256'])
            self.assertEqual(load_v9_circles(train).shape, (1, 3))
            self.assertNotIn('certified_corridor_half_width_m', runtime_case(train))
            self.assertNotIn('reference_distance_m', runtime_case(train))
            Path(train['circles_path']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Cached online geometry differs'):
                local.local_case('.', 7, geometry_dir=folder)
