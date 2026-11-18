"""Kilometer acceptance requires a safe full mission with separated endpoints."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from world_model.graph_rollouts import navigation_outcome


class NavigationOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.case = dict(start=[100., 100.], goal=[1100., 100.], direct_distance_m=1000.,
                         world_size=2000., reference_distance_m=1200., terrain='wildscenes_raster')
        self.summary = dict(goal_reached=True, remaining_m=1.5, termination='mission_arrival',
                            terminated=True, truncated=False, collisions=0, mission_budget=5040)

    def test_safe_arrival_counts_and_short_endpoint_tasks_never_count_as_kilometer(self):
        result = navigation_outcome(self.case, self.summary)
        self.assertTrue(result['kilometer_success'])
        # A large map, long reference route, or a stale distance field is insufficient.
        self.case.update(goal=[200., 100.], reference_distance_m=1500.)
        result = navigation_outcome(self.case, self.summary)
        self.assertEqual(result['direct_distance_m'], 100.)
        self.assertTrue(result['navigation_success'])
        self.assertFalse(result['kilometer_task'])
        self.assertFalse(result['kilometer_success'])

    def test_rollbacks_do_not_hide_safety_or_unknown_space_violations(self):
        for key in ('collisions', 'ground_safety_violations', 'raster_collision_attempts',
                    'raster_unknown_attempts'):
            with self.subTest(event=key):
                result = navigation_outcome(self.case, dict(self.summary, **{key: 1}))
                self.assertTrue(result['kilometer_task'])
                self.assertFalse(result['navigation_success'])
                self.assertFalse(result['kilometer_success'])

    def test_timeout_or_subgoal_arrival_cannot_pass_final_goal_acceptance(self):
        for changes in (dict(goal_reached=False), dict(remaining_m=25.),
                        dict(termination='mission_budget'), dict(termination='arrival'),
                        dict(terminated=False), dict(truncated=True)):
            with self.subTest(changes=changes):
                self.assertFalse(navigation_outcome(self.case, dict(self.summary, **changes))['kilometer_success'])

    def test_simplified_success_retains_explicit_source_geometry_assumption(self):
        result = navigation_outcome(dict(self.case, terrain='wildscenes_simplified_raster',
            source_unknown_space='free_by_simulation_assumption'), self.summary)
        self.assertTrue(result['kilometer_success'])
        self.assertEqual(result['terrain'], 'wildscenes_simplified_raster')
        self.assertEqual(result['source_unknown_space'], 'free_by_simulation_assumption')


if __name__ == '__main__':
    unittest.main()
