import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from vtemp5.run_vtemp5_circle_pretrain import freeze_example, score
from vtemp4.vtemp4_body_angle import BodyAngleDistribution


class FixedBearingPolicy(torch.nn.Module):
    def forward(self, observations):
        params = torch.zeros((len(observations), 2))
        reference = torch.full((len(observations),), math.radians(-179))
        return BodyAngleDistribution(params, [o['angle_arcs'] for o in observations], reference), None


class CirclePretrainCacheTests(unittest.TestCase):
    def example(self, angle, arcs):
        return dict(observation={'angle_arcs': np.asarray(arcs)}, angle_rad=angle,
                    time_s=120, actor_angle_rad=-1., teacher_executed=False, label='return_to_goal')

    def test_cache_preserves_teacher_provenance_and_wraps_heading_error(self):
        row = freeze_example(self.example(math.radians(179), [[-math.pi, math.pi]]))
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory)/'samples.pt'
            torch.save([row], filename)
            examples = torch.load(filename, weights_only=True)
        self.assertEqual(examples[0]['time_s'], 120)
        self.assertFalse(examples[0]['teacher_executed'])
        metrics = score(FixedBearingPolicy(), examples, 1)['return_to_goal']
        self.assertAlmostEqual(metrics['mean_error_deg'], 2., places=4)
        self.assertEqual(metrics['within_30_deg'], 1)

    def test_teacher_outside_native_legal_angles_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside the legal'):
            freeze_example(self.example(-.5, [[0., 1.]]))
