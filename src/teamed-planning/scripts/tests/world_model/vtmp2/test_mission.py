"""Aerial tasks stay separate from vehicle waypoints and real observations."""
import unittest

import networkx as nx
import numpy as np

from world_model.mission import WorldModelMissionManager


class Ranker:
    selector_mode = 'controlled_test_ranker'

    def __init__(self, score=None):
        self.score = score or (lambda c: -c.point[0] if c.purpose == 'survey' else 1000.)
        self.calls = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def observe(self, *args, **kwargs):
        pass

    def rank(self, candidates, goal):
        self.calls.append(list(candidates))
        return [self.score(c) for c in candidates]


def graph(points, edges=()):
    result = nx.Graph()
    for name, point in points.items():
        result.add_node(name, position=point, confidence=1.)
    result.add_edges_from(edges)
    return result


class MissionTests(unittest.TestCase):
    def manager(self, planner=None, **options):
        manager = WorldModelMissionManager((300., 0.), planner=planner or Ranker(), **options)
        manager.set_observation((0., 0.), np.ones((16, 16), int), .5)
        return manager

    def test_unknown_aerial_task_does_not_command_vehicle(self):
        manager = self.manager()
        before = manager._observation[2].copy()
        target, changed = manager.update((0., 0.), None, 0.)
        self.assertTrue(changed)
        self.assertEqual(manager.purpose, 'survey')
        self.assertIsNone(manager.drive_target)
        self.assertEqual(manager.target_status, 'unverified')
        self.assertFalse(manager.final)
        self.assertGreater(np.linalg.norm(target), 100.)
        self.assertEqual(manager.selector_mode, 'controlled_test_ranker')
        np.testing.assert_array_equal(manager._observation[2], before)

    def test_ground_obstacle_does_not_block_aerial_survey_task(self):
        manager = self.manager()
        manager.set_observation((0., 0.), np.full((16, 16), 2), .5)
        target, _ = manager.update((0., 0.), None, 0.)
        self.assertEqual(manager.purpose, 'survey')
        self.assertIsNone(manager.drive_target)
        self.assertGreater(target[0], 100.)
        self.assertTrue(all(c.purpose == 'survey' for c in manager.planner.calls[-1]))

    def test_mixed_rank_can_prefer_survey_over_known_forward_route(self):
        manager = self.manager()
        known = graph({'near': (1.5, 0.), 'far': (20., 0.)}, [('near', 'far')])
        manager.update((0., 0.), known, 0.)
        batch = manager.planner.calls[-1]
        self.assertTrue(any(c.purpose == 'drive' and c.status == 'verified' for c in batch))
        self.assertTrue(any(c.purpose == 'survey' for c in batch))
        self.assertEqual(manager.purpose, 'survey')

    def test_model_prediction_reverses_choice_and_reselection_uses_same_batch(self):
        ranker = Ranker()
        manager = self.manager(ranker, switch_improvement=.1)
        manager.update((0., 0.), None, 0.)
        old_target = manager.target.copy()
        ranker.score = lambda c: -1000.-c.point[0] if c.purpose == 'drive' else 1000.
        _, changed = manager.update((0., 0.), None, 5.)
        self.assertTrue(changed)
        self.assertEqual(manager.purpose, 'drive')
        self.assertIsNotNone(manager.drive_target)
        batch = ranker.calls[-1]
        self.assertTrue(any(c.purpose == 'survey' and np.allclose(c.point, old_target) for c in batch))
        self.assertEqual(manager.last_reason, 'better_prediction')

    def test_survey_completion_uses_received_evidence_not_vehicle_arrival(self):
        manager = self.manager()
        first, _ = manager.update((0., 0.), None, 0.)
        received = graph({'remote': first.copy()})
        second, changed = manager.update((0., 0.), received, 1.)
        self.assertTrue(changed)
        self.assertEqual(manager.survey_completed, 1)
        self.assertEqual(manager.arrivals, 0)
        self.assertEqual(manager.last_reason, 'survey_observed')
        self.assertTrue(manager.survey_observed(first, received))
        self.assertGreater(np.linalg.norm(second-first), manager.survey_evidence_radius)
        self.assertFalse(manager.final)

    def test_existing_evidence_excludes_survey_candidate_and_bad_confidence_is_ignored(self):
        manager = self.manager()
        received = graph({'remote': (200., 0.)})
        manager.update((0., 0.), received, 0.)
        self.assertTrue(all(c.purpose != 'survey' or np.linalg.norm(c.point-[200., 0.]) > 6.
                            for c in manager.planner.calls[-1]))
        for confidence in (0., -1., np.nan, np.inf):
            received.nodes['remote']['confidence'] = confidence
            self.assertFalse(manager.survey_observed((200., 0.), received))

    def test_received_survey_evidence_is_independent_of_ground_route_filter(self):
        manager = self.manager()
        target, _ = manager.update((0., 0.), None, 0.)
        received = graph({'remote': target.copy()})
        self.assertTrue(manager.survey_observed(target, received))
        manager.set_received_survey_graph(received)
        # Route filtering can omit this node without erasing receipt of the
        # aerial evidence; completion does not claim a drivable route.
        manager.update((0., 0.), nx.Graph(), 1.)
        self.assertEqual(manager.survey_completed, 1)
        self.assertEqual(manager.last_reason, 'survey_observed')
        self.assertFalse(manager.final)

    def test_received_survey_graph_is_snapshotted_not_a_mutable_reference(self):
        manager = self.manager()
        target, _ = manager.update((0., 0.), None, 0.)
        received = graph({'remote': target.copy()})
        manager.set_received_survey_graph(received)
        received.nodes['remote']['position'][:] += 100.
        received.nodes['remote']['confidence'] = 0.
        manager.update((0., 0.), None, 1.)
        self.assertEqual(manager.survey_completed, 1)

    def test_survey_stall_is_no_nearby_information_not_vehicle_motion(self):
        manager = self.manager(stall=10., max_wait=30., survey_timeout=30.)
        first, _ = manager.update((0., 0.), None, 0.)
        # A real new node near the task extends the information timer, although
        # the holding vehicle has not moved; it is outside completion radius.
        received = graph({'approach': first+np.array([0., 10.])})
        _, changed = manager.update((0., 0.), received, 9.)
        self.assertFalse(changed)
        _, changed = manager.update((0., 0.), received, 11.)
        self.assertFalse(changed)
        second, changed = manager.update((0., 0.), received, 19.)
        self.assertTrue(changed)
        self.assertEqual(manager.last_reason, 'survey_timeout')
        self.assertGreater(np.linalg.norm(second-first), manager.failure_radius)

    def test_radius_shrink_revokes_task_before_minimum_hold(self):
        manager = self.manager(min_hold=100.)
        first, _ = manager.update((0., 0.), None, 0.)
        self.assertGreater(np.linalg.norm(first), 100.)
        manager.set_reconnaissance_range(20.)
        second, changed = manager.update((0., 0.), None, 1.)
        self.assertTrue(changed)
        self.assertLessEqual(np.linalg.norm(second), 20.+1e-8)
        self.assertEqual(manager.last_reason, 'target_invalid')

    def test_candidates_are_bounded_diverse_and_current_is_reserved(self):
        manager = self.manager()
        manager.update((0., 0.), None, 0.)
        manager.update((0., 0.), None, 5.)
        batch = manager.planner.calls[-1]
        self.assertLessEqual(len(batch), 24)
        survey = [c for c in batch if c.purpose == 'survey']
        self.assertLessEqual(len(survey), 13)
        self.assertTrue(any(c.point[0] < -50 for c in survey))
        self.assertTrue(any(c.point[1] > 50 for c in survey))
        self.assertTrue(any(c.point[1] < -50 for c in survey))
        self.assertTrue(all(np.linalg.norm(c.point) <= 200.+1e-8 for c in batch))

    def test_purpose_change_at_same_coordinate_is_an_action_change(self):
        manager = self.manager()
        manager.request_target((100., 0.))
        _, changed = manager.update((0., 0.), None, 0.)
        self.assertTrue(changed)
        self.assertEqual(manager.purpose, 'drive')
        manager.request_survey_target((100., 0.))
        target, changed = manager.update((0., 0.), None, 1.)
        self.assertTrue(changed)
        np.testing.assert_allclose(target, (100., 0.))
        self.assertEqual(manager.purpose, 'survey')
        self.assertIsNone(manager.drive_target)
        self.assertEqual(manager.selected, 2)

    def test_invalid_ranking_does_not_fall_back_to_geometry(self):
        manager = self.manager(Ranker(lambda c: np.nan))
        with self.assertRaisesRegex(ValueError, 'finite cost'):
            manager.update((0., 0.), None, 0.)
        self.assertIsNone(manager.target)

    def test_zero_radius_holds_and_stats_are_numeric(self):
        manager = self.manager(recon_radius=0.)
        target, _ = manager.update((0., 0.), None, 0.)
        np.testing.assert_array_equal(target, (0., 0.))
        self.assertIsNone(manager.drive_target)
        self.assertEqual(manager.target_status, 'hold')
        self.assertTrue(all(isinstance(value, (int, float)) for value in manager.stats().values()))


if __name__ == '__main__':
    unittest.main()
