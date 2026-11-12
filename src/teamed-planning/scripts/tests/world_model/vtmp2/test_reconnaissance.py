"""Check vtmp2's communication envelope without simulator state or GPU work."""

import json
import math
import unittest

from world_model.runtime import ReconnaissanceEnvelope


class ReconnaissanceEnvelopeTests(unittest.TestCase):
    def test_three_drone_default_uses_height_and_per_hop_margin(self):
        envelope = ReconnaissanceEnvelope()
        self.assertAlmostEqual(envelope.theoretical_radius_m, 295.05513080531716)
        self.assertAlmostEqual(envelope.radius_m, 252.2033066054746)
        self.assertLess(envelope.radius_m, envelope.theoretical_radius_m)

    def test_no_drone_has_no_footprint(self):
        envelope = ReconnaissanceEnvelope(drone_count=0)
        self.assertEqual(envelope.radius_m, 0.)
        self.assertEqual(envelope.theoretical_radius_m, 0.)

    def test_one_drone_projects_only_the_ground_link(self):
        envelope = ReconnaissanceEnvelope(drone_count=1)
        self.assertAlmostEqual(envelope.theoretical_radius_m, 105.55513080531713)
        self.assertAlmostEqual(envelope.radius_m, 91.12830660547462)

    def test_extra_drones_add_horizontal_air_links(self):
        one = ReconnaissanceEnvelope(drone_count=1)
        three = ReconnaissanceEnvelope(drone_count=3)
        self.assertAlmostEqual(three.radius_m - one.radius_m, 161.075)
        self.assertAlmostEqual(three.theoretical_radius_m - one.theoretical_radius_m, 189.5)

    def test_insufficient_ground_link_cannot_deploy_any_drone(self):
        envelope = ReconnaissanceEnvelope(communication_range=10., altitude=15.)
        self.assertEqual(envelope.radius_m, 0.)
        self.assertEqual(envelope.theoretical_radius_m, 0.)

    def test_discount_can_prevent_deployment_without_changing_geometric_limit(self):
        envelope = ReconnaissanceEnvelope(communication_range=20., altitude=18.)
        self.assertGreater(envelope.theoretical_radius_m, 0.)
        self.assertEqual(envelope.radius_m, 0.)

    def test_height_equal_to_ground_range_leaves_only_sensor_footprint(self):
        envelope = ReconnaissanceEnvelope(drone_count=1, communication_range=20.25,
                                           altitude=20., deployment_ratio=1.)
        self.assertEqual(envelope.radius_m, 12.)

    def test_ratio_one_preserves_geometric_limit(self):
        envelope = ReconnaissanceEnvelope(deployment_ratio=1.)
        self.assertEqual(envelope.radius_m, envelope.theoretical_radius_m)

    def test_invalid_fleet_count(self):
        for count in (-1, 1.5, True, '3', math.inf, math.nan):
            with self.subTest(count=count), self.assertRaises(ValueError):
                ReconnaissanceEnvelope(drone_count=count)

    def test_invalid_positive_parameters(self):
        for field in ('communication_range', 'altitude', 'footprint_width',
                      'link_margin', 'deployment_ratio'):
            for value in (0., -1., math.nan, math.inf, -math.inf, True, '95'):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    ReconnaissanceEnvelope(**{field: value})

    def test_invalid_ratio_and_link_margin(self):
        for parameters in ({'deployment_ratio': 1.001}, {'link_margin': 95.},
                           {'link_margin': 96.}, {'communication_range': .2}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                ReconnaissanceEnvelope(**parameters)

    def test_description_distinguishes_deployment_from_observation(self):
        envelope = ReconnaissanceEnvelope()
        description = json.loads(json.dumps(envelope.describe(), allow_nan=False))
        self.assertEqual(description['drone_count'], 3)
        self.assertEqual(description['planning_radius_m'], envelope.radius_m)
        self.assertEqual(description['theoretical_radius_m'], envelope.theoretical_radius_m)
        self.assertIn('not_observed_or_traversable', description['interpretation'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
