"""Information acquisition, local fusion and fixed survey-domain accounting."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator.survey_reward_v10 import LocalSurveyEvidence, SurveyTask


def packet(frame=0, source='drone_0', position=(5., 5., 15.), occupancy=None, resolution=2., valid=True):
    return dict(source=source, frame=frame, position=position,
                occupancy=np.zeros((1, 1)) if occupancy is None else occupancy,
                resolution=resolution, valid=valid)


class SurveyTaskTests(unittest.TestCase):
    def test_capsule_cells_have_fixed_area_and_independent_endpoints(self):
        task = SurveyTask((10., 10.), (30., 10.), 4., world_size=100.)
        xy = task.cells_xy
        nearest_x = np.clip(xy[:, 0], 10., 30.)
        self.assertTrue((np.hypot(xy[:, 0]-nearest_x, xy[:, 1]-10.) <= 4.+1e-9).all())
        self.assertEqual(task.area_m2, task.cell_count*4.)
        self.assertLess(task.cell_count, task.grid_size**2)
        self.assertEqual(task.to_message(), dict(a=[10., 10.], b=[30., 10.], half_width=4.))
        self.assertTrue(np.all(np.diff(task.cell_ids) > 0))
        with self.assertRaises(FrozenInstanceError):
            task.a = (20., 20.)
        with self.assertRaises(ValueError):
            task.cells_xy[0] = 0.

    def test_invalid_geometry_is_rejected(self):
        invalid = [dict(a=(float('nan'), 2.)), dict(b=(101., 2.)), dict(a=(1.,)),
                   dict(half_width=0.), dict(half_width=True), dict(resolution=0.),
                   dict(world_size=99., resolution=2.), dict(a=(0., 0.), b=(0., 0.), half_width=.01)]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                SurveyTask(**dict(dict(a=(10., 10.), b=(30., 10.), half_width=4., world_size=100.), **change))

    def test_reversing_endpoints_preserves_area(self):
        first = SurveyTask((10., 10.), (30., 34.), 4., world_size=100.)
        second = SurveyTask(first.b, first.a, 4., world_size=100.)
        np.testing.assert_array_equal(first.cell_ids, second.cell_ids)


class LocalSurveyEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.task = SurveyTask((5., 5.), (5., 5.), .6, world_size=10.)
        self.assertEqual(self.task.cell_count, 1)
        self.evidence = LocalSurveyEvidence(self.task, 2)

    def test_own_information_gains_without_any_ground_delivery(self):
        self.assertTrue(self.evidence.ingest(0, packet()))
        terms = self.evidence.step_metrics()
        self.assertGreater(terms['survey_confidence_gain'], 0.)
        self.assertEqual(terms['survey_observed_fraction'], 1.)
        self.assertEqual(terms['survey_coverage'], 0.)
        self.assertEqual(len(self.evidence.received[1]), 0)
        self.evidence.ingest(0, packet(frame=1))
        self.assertEqual(self.evidence.metrics()['survey_coverage'], 1.)

    def test_forwarded_copy_is_deduplicated_at_recipient_and_team_level(self):
        self.evidence.ingest(0, packet())
        self.evidence.step_metrics()
        self.assertFalse(self.evidence.ingest(0, packet()))
        self.assertTrue(self.evidence.ingest(1, packet()))
        self.assertEqual(self.evidence.step_metrics()['survey_confidence_gain'], 0.)
        self.assertFalse(self.evidence.ingest(1, packet()))
        self.assertEqual(self.evidence.step_metrics()['survey_confidence_gain'], 0.)

    def test_distinct_received_evidence_can_improve_local_fusion(self):
        self.evidence.ingest(0, packet())
        self.evidence.ingest(1, packet(source='drone_1'))
        self.evidence.step_metrics()
        self.evidence.ingest(0, packet(source='drone_1'))
        self.assertGreater(self.evidence.step_metrics()['survey_confidence_gain'], 0.)

    def test_contradictory_measurement_has_negative_signed_gain(self):
        self.evidence.ingest(0, packet())
        first = self.evidence.step_metrics()['survey_confidence_gain']
        self.evidence.ingest(0, packet(frame=1, occupancy=np.ones((1, 1))))
        terms = self.evidence.step_metrics()
        self.assertAlmostEqual(terms['survey_confidence_gain'], -first)
        self.assertAlmostEqual(terms['survey_confidence'], 0.)
        self.assertEqual(terms['survey_observed_fraction'], 1.)

    def test_obstacles_are_information_and_confidence_saturates(self):
        for frame in range(10):
            self.evidence.ingest(0, packet(frame=frame, occupancy=np.ones((1, 1))))
        terms = self.evidence.step_metrics()
        self.assertEqual(terms['survey_confidence'], 1.)
        self.assertEqual(terms['survey_coverage'], 1.)
        self.assertFalse(self.evidence.ingest(0, packet(frame=10, occupancy=np.ones((1, 1)))))
        self.assertEqual(self.evidence.step_metrics()['survey_confidence_gain'], 0.)

    def test_outside_unknown_or_invalid_pixels_do_not_create_information(self):
        observations = [packet(position=(9., 9.)), packet(frame=1, valid=False),
                        packet(frame=2, occupancy=np.array([[-1.]])),
                        packet(frame=3, occupancy=np.array([[np.nan]])),
                        packet(frame=4, occupancy=np.array([[2.]]))]
        for observation in observations:
            self.assertFalse(self.evidence.ingest(0, observation))
        self.assertEqual(self.evidence.step_metrics()['survey_confidence_gain'], 0.)
        self.assertEqual(self.evidence.metrics()['survey_observed_fraction'], 0.)

    def test_finer_sensor_pixels_are_one_vote_per_world_cell(self):
        self.evidence.ingest(0, packet())
        self.evidence.ingest(1, packet(occupancy=np.zeros((4, 4)), resolution=.5))
        np.testing.assert_array_equal(self.evidence.log_odds[0], self.evidence.log_odds[1])

    def test_metric_reads_and_motion_without_observation_cannot_earn_reward(self):
        self.evidence.ingest(0, packet())
        first = self.evidence.metrics()['survey_confidence']
        self.assertEqual(self.evidence.metrics()['survey_confidence'], first)
        self.assertEqual(self.evidence.step_metrics()['survey_confidence_gain'], first)
        # No UAV or car position enters metrics, so moving either without a
        # new picture cannot change the fixed task denominator or its score.
        self.assertEqual(self.evidence.step_metrics()['survey_confidence_gain'], 0.)

    def test_unknown_cells_remain_in_denominator(self):
        task = SurveyTask((5., 5.), (9., 5.), .6, world_size=10.)
        evidence = LocalSurveyEvidence(task, 2)
        evidence.ingest(0, packet())
        self.evidence.ingest(0, packet())
        self.assertAlmostEqual(evidence.metrics()['survey_confidence'],
                               self.evidence.metrics()['survey_confidence']/task.cell_count)
        self.assertAlmostEqual(evidence.metrics()['survey_observed_fraction'], 1./task.cell_count)

    def test_validation_prevents_nonfinite_updates(self):
        for change in (dict(frame=-1), dict(frame=float('nan')), dict(resolution=0.),
                       dict(position=(float('inf'), 5.)), dict(occupancy=np.zeros(3))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.evidence.ingest(0, packet(**change))
        for count in (0, True, 1.5):
            with self.assertRaises(ValueError):
                LocalSurveyEvidence(self.task, count)
        with self.assertRaises(ValueError):
            LocalSurveyEvidence(self.task, 2, confidence_threshold=1.1)


if __name__ == '__main__':
    unittest.main()
