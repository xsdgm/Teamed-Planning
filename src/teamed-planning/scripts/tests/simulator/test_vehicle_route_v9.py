"""Regressions for map credit on a moving vehicle's remaining rolling route."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from common.graph_structure import TopologicalGraph
from simulator.navigation_metrics import task_graph_metrics, vehicle_route_update


class VehicleRouteUpdateTests(unittest.TestCase):
    @staticmethod
    def metrics(length=None, frontier=100.):
        return dict(graph_task_connected=float(length is not None),
                    graph_task_path_length=length or 0.,
                    graph_frontier_goal_distance_m=0. if length is not None else frontier)

    def test_shortening_behind_vehicle_earns_no_remaining_route_credit(self):
        graph = TopologicalGraph('remaining')
        for node, point in dict(initial=(0., 0.), a=(0., 20.), b=(20., 20.),
                                vehicle=(20., 0.), detour=(30., -20.), goal=(40., 0.)).items():
            graph.add_node(node, point)
        for first, second in [('initial', 'a'), ('a', 'b'), ('b', 'vehicle'),
                              ('vehicle', 'detour'), ('detour', 'goal')]:
            graph.add_edge(first, second)
        vehicle, goal = np.array([20., 0.]), np.array([40., 0.])
        def measured(start):
            return task_graph_metrics(graph, start, goal, lambda a, b: True, endpoint_radius=.1)
        before = measured(vehicle)
        initial_before = measured(np.zeros(2))
        graph.add_edge('initial', 'vehicle')
        after = measured(vehicle)
        self.assertLess(measured(np.zeros(2))['graph_task_path_length'],
                        initial_before['graph_task_path_length'])
        unchanged = vehicle_route_update(before, after, vehicle, goal)
        self.assertEqual(unchanged['route_quality_gain'], 0.)
        self.assertEqual(unchanged['route_length_delta_m'], 0.)
        graph.add_edge('vehicle', 'goal')
        shortened = vehicle_route_update(after, measured(vehicle), vehicle, goal)
        self.assertGreater(shortened['route_quality_gain'], 0.)
        self.assertAlmostEqual(shortened['route_quality'], 1.)
        self.assertAlmostEqual(shortened['route_length_after_m'], 20.)

    def test_unchanged_maps_earn_nothing_after_movement_or_target_switch(self):
        # Each call compares maps at one frozen endpoint pair. There is no
        # cross-frame reference against which driving or retargeting can pay.
        for position, target, length in [((0., 0.), (100., 0.), 150.),
                                         ((60., 0.), (100., 0.), 50.),
                                         ((60., 0.), (60., 200.), 250.)]:
            metrics = self.metrics(length)
            update = vehicle_route_update(metrics, metrics, position, target)
            for key in ('route_quality_gain', 'route_length_delta_m', 'frontier_progress_gain'):
                self.assertEqual(update[key], 0.)

    def test_disconnect_reconnect_cannot_farm_positive_only_credit(self):
        unknown, route = self.metrics(frontier=40.), self.metrics(125.)
        connected = vehicle_route_update(unknown, route, (0., 0.), (100., 0.))
        lost = vehicle_route_update(route, unknown, (0., 0.), (100., 0.))
        self.assertAlmostEqual(connected['route_first_quality_gain'], .8)
        self.assertAlmostEqual(lost['route_quality_gain'], -.8)
        self.assertAlmostEqual(connected['route_quality_gain']+lost['route_quality_gain'], 0.)
        self.assertAlmostEqual(connected['frontier_progress_gain']+lost['frontier_progress_gain'], 0.)
        self.assertEqual(connected['route_lengths_comparable'], 0.)
        self.assertEqual(connected['route_length_delta_m'], 0.)

    def test_physical_detour_regression_is_signed(self):
        update = vehicle_route_update(self.metrics(125.), self.metrics(200.), (0., 0.), (100., 0.))
        self.assertAlmostEqual(update['route_shortening_quality_gain'], -.3)
        self.assertEqual(update['route_length_delta_m'], -75.)
        self.assertEqual(update['route_first_quality_gain'], 0.)
        self.assertEqual(update['route_reference_length_m'], 100.)
        self.assertEqual(update['route_reference_is_lower_bound'], 1.)
        self.assertNotIn('route_best_quality', update)

    def test_disconnected_frontier_is_measured_from_current_vehicle(self):
        update = vehicle_route_update(self.metrics(frontier=80.), self.metrics(frontier=40.),
                                      (50., 0.), (150., 0.))
        self.assertAlmostEqual(update['frontier_progress_gain'], .4)
        self.assertEqual(update['route_quality_gain'], 0.)

    def test_zero_distance_and_nonfinite_routes_have_finite_outputs(self):
        for before, after in [(self.metrics(), self.metrics(0.)),
                              (self.metrics(20.), self.metrics(0.)),
                              (self.metrics(float('inf')), self.metrics(float('nan')))]:
            update = vehicle_route_update(before, after, (50., 50.), (50., 50.))
            self.assertTrue(all(np.isfinite(value) for value in update.values()))
        zero = vehicle_route_update(self.metrics(0.), self.metrics(0.), (50., 50.), (50., 50.))
        self.assertEqual(zero['route_quality'], 1.)
        self.assertEqual(zero['route_quality_gain'], 0.)


if __name__ == '__main__':
    unittest.main()
