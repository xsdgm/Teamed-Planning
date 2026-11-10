"""Safety, reconnaissance-range and online-reselection checks for vtmp2."""
import unittest

import networkx as nx
import numpy as np

from world_model.vtmp2 import Vtmp2SubgoalManager


def graph(points, edges=()):
    result = nx.Graph()
    for key, point in points.items():
        result.add_node(key, position=point)
    result.add_edges_from(edges)
    return result


class Vtmp2Tests(unittest.TestCase):
    def manager(self, **options):
        manager = Vtmp2SubgoalManager((100., 0.), **options)
        manager.set_observation((0., 0.), np.ones((16, 16), int), .5)
        return manager

    def test_recon_range_is_not_the_local_sensor_window(self):
        manager = self.manager(radius=30., recon_radius=20.)
        target, changed = manager.update((0., 0.), None, 0.)
        self.assertTrue(changed)
        np.testing.assert_allclose(target, (20., 0.), atol=1e-8)
        self.assertEqual((manager.last_source, manager.target_status), ('recon_frontier', 'unverified'))
        self.assertFalse(manager.final)

    def test_local_mode_keeps_targets_in_observed_free_component(self):
        manager = self.manager(visibility_mode='local')
        target, _ = manager.update((0., 0.), None, 0.)
        self.assertLessEqual(np.linalg.norm(target), 4.)
        self.assertEqual(manager.target_status, 'verified')

    def test_observed_mode_requires_safe_local_graph_entry(self):
        manager = self.manager(visibility_mode='observed', radius=25.)
        disconnected = graph({'near': (1.5, 0.), 'far': (20., 0.)})
        target, _ = manager.update((0., 0.), disconnected, 0.)
        self.assertLess(np.linalg.norm(target), 4.)
        connected = graph({'near': (1.5, 0.), 'far': (20., 0.)}, [('near', 'far')])
        manager = self.manager(visibility_mode='observed', radius=25.)
        target, _ = manager.update((0., 0.), connected, 0.)
        np.testing.assert_allclose(target, (20., 0.))
        self.assertEqual(manager.target_status, 'verified')

    def test_known_graph_route_is_preferred_to_unknown_ring(self):
        manager = self.manager(radius=60.)
        known = graph({'near': (1.5, 0.), 'far': (20., 0.)}, [('near', 'far')])
        target, _ = manager.update((0., 0.), known, 0.)
        np.testing.assert_allclose(target, (20., 0.))
        self.assertEqual(manager.last_source, 'graph')

    def test_blocked_or_unknown_map_does_not_authorize_exploration(self):
        for value in (0, 2, 4):
            with self.subTest(value=value):
                manager = self.manager()
                manager.set_observation((0., 0.), np.full((16, 16), value), .5)
                target, _ = manager.update((0., 0.), None, 0.)
                np.testing.assert_allclose(target, (0., 0.))
                self.assertEqual(manager.target_status, 'hold')
                self.assertFalse(manager.final)

    def test_new_obstacle_replaces_target_before_minimum_hold(self):
        manager = self.manager(radius=20., min_hold=100.)
        first, _ = manager.update((0., 0.), None, 0.)
        observed = np.ones((16, 16), int)
        observed[:, 13] = 2  # new wall east, leaving safe directions north/south
        manager.set_observation((0., 0.), observed, .5)
        second, changed = manager.update((0., 0.), None, 1.)
        self.assertTrue(changed)
        self.assertGreater(np.linalg.norm(second-first), 1.)
        self.assertEqual(manager.last_reason, 'target_invalid')

    def test_graph_edge_crossing_new_local_obstacle_is_not_verified(self):
        manager = self.manager(visibility_mode='observed')
        observed = np.ones((16, 16), int)
        observed[:, 13] = 2
        manager.set_observation((0., 0.), observed, .5)
        known = graph({'near': (-1., 0.), 'far': (20., 0.)}, [('near', 'far')])
        target, _ = manager.update((0., 0.), known, 0.)
        self.assertLess(np.linalg.norm(target), 4.)

    def test_shrinking_communication_range_revokes_old_target(self):
        manager = self.manager(radius=30., min_hold=100.)
        manager.update((0., 0.), None, 0.)
        manager.set_reconnaissance_range(10.)
        target, changed = manager.update((0., 0.), None, 1.)
        self.assertTrue(changed)
        self.assertLessEqual(np.linalg.norm(target), 10.+1e-8)
        self.assertEqual(manager.last_reason, 'target_invalid')

    def test_stall_changes_target_before_arrival_and_suppresses_old_point(self):
        manager = self.manager(radius=20., stall=10., max_wait=30.)
        first, _ = manager.update((0., 0.), None, 0.)
        second, changed = manager.update((0., 0.), None, 10.)
        self.assertTrue(changed)
        self.assertGreater(np.linalg.norm(second-first), manager.failure_radius)
        self.assertEqual(manager.last_reason, 'stalled')
        self.assertEqual(manager.failed, 1)

    def test_graph_growth_cannot_extend_wait_forever(self):
        manager = self.manager(radius=20., stall=10., max_wait=25.)
        first, _ = manager.update((0., 0.), None, 0.)
        for t in (5., 10., 15., 20., 25.):
            unseen = graph({str(i): (50.+i, 50.) for i in range(int(t))})
            target, changed = manager.update((0., 0.), unseen, t)
        self.assertTrue(changed)
        self.assertGreater(np.linalg.norm(target-first), manager.failure_radius)
        self.assertEqual(manager.last_reason, 'stalled')

    def test_healthy_target_is_stable(self):
        manager = self.manager(radius=20.)
        first, _ = manager.update((0., 0.), None, 0.)
        for t in (1., 5., 10., 15.):
            target, changed = manager.update((0., 0.), None, t)
            self.assertFalse(changed)
            np.testing.assert_allclose(target, first)
        self.assertEqual(manager.selected, 1)

    def test_manual_requests_are_checked_and_safety_can_override_them(self):
        manager = self.manager(radius=20.)
        manager.update((0., 0.), None, 0.)
        manager.request_target((0., 15.))
        target, changed = manager.update((0., 0.), None, 1.)
        self.assertTrue(changed)
        np.testing.assert_allclose(target, (0., 15.))
        self.assertEqual(manager.last_reason, 'manual_request')
        manager.request_target((50., 0.))
        target, changed = manager.update((0., 0.), None, 2.)
        self.assertFalse(changed)
        np.testing.assert_allclose(target, (0., 15.))
        self.assertEqual(manager.last_reason, 'manual_rejected')
        manager.set_reconnaissance_range(5.)
        target, changed = manager.update((0., 0.), None, 3.)
        self.assertTrue(changed)
        self.assertLessEqual(np.linalg.norm(target), 5.+1e-8)

    def test_final_goal_requires_verified_route_and_can_be_revoked(self):
        manager = Vtmp2SubgoalManager((2., 0.), radius=20.)
        clear = np.ones((16, 16), int)
        manager.set_observation((0., 0.), clear, .5)
        target, _ = manager.update((0., 0.), None, 0.)
        np.testing.assert_allclose(target, (2., 0.))
        self.assertTrue(manager.final)
        clear[8, 12] = 2
        manager.set_observation((0., 0.), clear, .5)
        target, changed = manager.update((0., 0.), None, 1.)
        self.assertTrue(changed)
        self.assertFalse(manager.final)

    def test_zero_survey_range_holds_position(self):
        manager = self.manager(recon_radius=0.)
        target, _ = manager.update((0., 0.), None, 0.)
        np.testing.assert_allclose(target, (0., 0.))
        self.assertFalse(manager.final)

    def test_already_reached_verified_final_does_not_trigger_exploration(self):
        manager = Vtmp2SubgoalManager((.5, 0.))
        manager.set_observation((0., 0.), np.ones((16, 16), int), .5)
        target, _ = manager.update((0., 0.), None, 0.)
        np.testing.assert_allclose(target, (.5, 0.))
        self.assertTrue(manager.final)
        _, changed = manager.update((0., 0.), None, 10.)
        self.assertFalse(changed)
        self.assertEqual(manager.last_reason, 'final_reached')

    def test_external_failure_overrides_hold_and_does_not_reselect_same_point(self):
        manager = self.manager(radius=20., min_hold=100.)
        first, _ = manager.update((0., 0.), None, 0.)
        manager.invalidate_target('planner_rejected')
        second, changed = manager.update((0., 0.), None, 1.)
        self.assertTrue(changed)
        self.assertGreater(np.linalg.norm(second-first), manager.failure_radius)
        self.assertEqual(manager.last_reason, 'planner_rejected')

    def test_hold_is_stable_and_does_not_count_selections(self):
        manager = self.manager(recon_radius=0.)
        manager.update((0., 0.), None, 0.)
        _, changed = manager.update((0., 0.), None, 1.)
        self.assertFalse(changed)
        self.assertEqual(manager.selected, 0)

    def test_recon_mode_does_not_hide_withdrawn_graph_route_as_unverified(self):
        manager = self.manager(radius=30., min_hold=100.)
        known = graph({'near': (1.5, 0.), 'far': (20., 0.)}, [('near', 'far')])
        first, _ = manager.update((0., 0.), known, 0.)
        np.testing.assert_allclose(first, (20., 0.))
        self.assertEqual(manager.target_status, 'verified')
        known.remove_edge('near', 'far')
        second, changed = manager.update((0., 0.), known, 1.)
        self.assertTrue(changed)
        self.assertEqual(manager.last_reason, 'target_invalid')
        self.assertGreater(np.linalg.norm(second-first), manager.failure_radius)
        self.assertEqual(manager.failed, 1)

    def test_explicit_nonpositive_or_nonfinite_confidence_is_not_a_route(self):
        for confidence in (0., -1., np.nan, np.inf, None):
            for field in ('node', 'edge'):
                with self.subTest(confidence=confidence, field=field):
                    manager = self.manager(visibility_mode='observed')
                    known = graph({'near': (1.5, 0.), 'far': (20., 0.)}, [('near', 'far')])
                    attributes = known.nodes['far'] if field == 'node' else known.edges['near', 'far']
                    attributes['confidence'] = confidence
                    target, _ = manager.update((0., 0.), known, 0.)
                    self.assertLess(np.linalg.norm(target), 4.)


if __name__ == '__main__':
    unittest.main()
