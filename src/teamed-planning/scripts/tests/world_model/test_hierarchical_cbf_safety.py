"""Safety projection regressions, including the four v18 terminal geometries."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from world_model import hierarchical_cbf as stable
from world_model.hierarchical_cbf import CBFSafetyFilter, project_velocity


class ProjectionTests(unittest.TestCase):
    def assert_safe(self, result, matrix, bound, speed):
        self.assertIsNotNone(result)
        self.assertLessEqual(np.linalg.norm(result), speed + 1e-7)
        self.assertTrue(np.all(np.asarray(matrix) @ result <= np.asarray(bound) + 1e-7))

    def test_line_projection_is_nearest_point(self):
        result, info = project_velocity([2., 2.], [[1., 1.]], [1.], 6.)
        np.testing.assert_allclose(result, [.5, .5], atol=1e-12)
        self.assertEqual(info['solver_status'], 'projected')
        self.assert_safe(result, [[1., 1.]], [1.], 6.)

    def test_disk_and_line_intersection(self):
        result, _ = project_velocity([0., 1.], [[-1., 0.]], [-.9], 1.)
        np.testing.assert_allclose(result, [.9, np.sqrt(.19)], atol=1e-12)
        self.assert_safe(result, [[-1., 0.]], [-.9], 1.)

    def test_raw_constraint_tolerance_does_not_expand_speed_circle(self):
        # The line/line vertex is 5e-8 outside the disk and closer to the
        # nominal command. Accepting it under a speed+1e-7 tolerance causes a
        # later hard-filter certification to clip and change the command.
        limit = np.sqrt((6. + 5e-8)**2 - 16.)
        matrix, bound = [[1., 0.], [0., -100.]], [limit, -400.]
        result, _ = project_velocity([6., 0.], matrix, bound, 6.)
        self.assertLessEqual(np.linalg.norm(result), 6.)
        np.testing.assert_allclose(result, [np.sqrt(20.), 4.], atol=1e-12)
        self.assert_safe(result, matrix, bound, 6.)

    def test_polygon_negative_margin_does_not_reject_feasible_circle_tangent(self):
        matrix, bound = np.array([[-1., 0.]]), np.array([-6.])
        self.assertLess(stable.feasibility_margin(matrix, bound, 6.), 0.)
        result, _ = project_velocity([0., 1.], matrix, bound, 6.)
        np.testing.assert_allclose(result, [6., 0.], atol=1e-12)
        self.assert_safe(result, matrix, bound, 6.)

    def test_raw_tolerance_not_relaxed_for_scaled_thin_strip(self):
        matrix = np.array([[1e7, 0.], [-1e7, 0.], [0., 1.]])
        bound = np.array([20., -20., 1.])
        result, _ = project_velocity([3., 2.], matrix, bound, 6.)
        np.testing.assert_allclose(result, [2e-6, 1.], atol=1e-12)
        self.assert_safe(result, matrix, bound, 6.)
        # A 1e-6 raw violation is still rejected despite the enormous row norm.
        result, info = project_velocity([2e-6, 0.], matrix[:2], [20. - 1e-6, -20.], 6.)
        self.assertIsNone(result)
        self.assertEqual(info['solver_status'], 'infeasible')

    def test_parallel_halfspaces_and_speed_limit_really_infeasible(self):
        for matrix, bound in [([[1., 0.], [-1., 0.]], [1., -2.]),
                              ([[-1., 0.]], [-6.01])]:
            result, info = project_velocity([0., 0.], matrix, bound, 6.)
            self.assertIsNone(result)
            self.assertEqual(info['solver_status'], 'infeasible')

    def test_zero_rows_and_unconstrained_disk(self):
        result, _ = project_velocity([10., 0.], np.empty((0, 2)), [], 6.)
        np.testing.assert_allclose(result, [6., 0.], atol=1e-12)
        result, _ = project_velocity([0., 0.], [[0., 0.]], [-1.], 6.)
        self.assertIsNone(result)

    def test_random_projections_satisfy_convex_optimality_conditions(self):
        from scipy.optimize import nnls
        rng = np.random.default_rng(2117)
        for _ in range(24):
            angles = rng.uniform(0, 2 * np.pi, 7)
            matrix = np.c_[np.cos(angles), np.sin(angles)]
            # Independent KKT certificate: the outward objective gradient must
            # be a nonnegative combination of active constraint normals. For
            # this convex problem, feasibility plus this certificate proves
            # global optimality without depending on SLSQP convergence.
            bound = rng.uniform(.3, 3., 7)
            desired = rng.normal(0, 3., 2)
            result, _ = project_velocity(desired, matrix, bound, 2.)
            self.assert_safe(result, matrix, bound, 2.)
            active = list(matrix[np.abs(matrix @ result - bound) < 1e-7])
            if abs(np.linalg.norm(result) - 2.) < 1e-7:
                active.append(result)
            if active:
                _, residual = nnls(np.asarray(active).T, desired - result)
                self.assertLess(residual, 1e-7)
            else:
                np.testing.assert_allclose(result, desired, atol=1e-12)


# Reconstructed from validation-slow-v018 terminal PPO float32 XY features
# (x1000), its stored ground trajectory, and its received-neighbor evidence.
# These are lightweight geometry regressions, not full-precision replay of the
# original SLSQP numerical failure. The original artifacts are not required.
TERMINAL_GEOMETRIES = {
    0: dict(position=[97.93264419, 244.48780715, 15.],
            peers={'drone_1': [167.76651144, 307.39086866, 15.]},
            ground=[43.19147618, 170.17515852, 0.],
            nominal=[1.65838302, 2.57241341, 0.], expected_infeasible=False),
    1: dict(position=[483.80646110, 326.29883289, 15.],
            peers={'drone_1': [577.40700245, 316.90669060, 15.]},
            ground=[392.28251398, 342.68706634, 0.],
            nominal=[-2.55995403, -.16472819, 0.], expected_infeasible=True),
    2: dict(position=[500.69689751, 261.24200225, 15.],
            peers={'drone_1': [578.14687490, 314.48730826, 15.]},
            ground=[475.44646941, 350.01913684, 0.],
            nominal=[-1.18539711, -1.07791884, 0.], expected_infeasible=False),
    5: dict(position=[738.40343952, 631.58047199, 15.],
            peers={'drone_1': [790.60322046, 553.42566967, 15.]},
            ground=[687.11652957, 708.33305473, 0.],
            nominal=[.41756032, -.47869683, 0.], expected_infeasible=True),
}


class FilterTests(unittest.TestCase):
    def terminal_call(self, case):
        fixture = dict(TERMINAL_GEOMETRIES[case])
        fixture.pop('expected_infeasible')
        return CBFSafetyFilter().filter(**fixture, dt=1.,
            required_links=('drone_1', '__ground__'), world_bounds=(0., 0., 887., 887.),
            drone_id='drone_0' if case == 0 else 'drone_2', ground_max_speed=1.)

    def test_four_terminal_geometries_keep_true_failures_and_accept_feasible_cases(self):
        for case, fixture in TERMINAL_GEOMETRIES.items():
            with self.subTest(case=case):
                result, info = self.terminal_call(case)
                self.assertEqual(info['infeasible'], fixture['expected_infeasible'])
                self.assertEqual(info['missing_links'], [])
                self.assertEqual(info['projection_solver'], stable.PROJECTION_VERSION)
                if not info['infeasible']:
                    self.assertLessEqual(info['max_constraint_residual'], 1e-7)
                    self.assertLessEqual(np.linalg.norm(result), 6. + 1e-7)
                else:
                    self.assertEqual(info['solver_status'], 'infeasible')

    def test_true_terminal_failures_exceed_exact_disk_not_just_margin_polygon(self):
        for case, expected_minimum in [(1, 9.4404), (5, 52.9374)]:
            captured = []
            original = stable.feasibility_margin

            def capture(matrix, bound, speed):
                captured.append((matrix.copy(), bound.copy()))
                return original(matrix, bound, speed)

            with patch.object(stable, 'feasibility_margin', capture):
                self.terminal_call(case)
            matrix, bound = captured[0]
            # Independent analytic intersection of the two communication lines
            # is their minimum-norm admissible point in these geometries.
            vertex = np.linalg.solve(matrix[[1, 2]], bound[[1, 2]])
            self.assertTrue(np.all(matrix @ vertex <= bound + 1e-7))
            self.assertAlmostEqual(np.linalg.norm(vertex), expected_minimum, places=3)
            result, _ = project_velocity([0., 0.], matrix, bound, 80.)
            np.testing.assert_allclose(result, vertex, atol=1e-8)
            result, _ = project_velocity([0., 0.], matrix, bound, 6.)
            self.assertIsNone(result)

    def test_nominal_safe_command_is_unchanged_and_missing_links_remain_explicit(self):
        result, info = CBFSafetyFilter().filter([1., 2., 0.], [50., 50., 15.], {}, 1.,
            required_links=('absent',), world_bounds=(0., 0., 100., 100.))
        np.testing.assert_array_equal(result, [1., 2., 0.])
        self.assertEqual(info['missing_links'], ['absent'])
        self.assertFalse(info['infeasible'])
        self.assertEqual(info['solver_status'], 'nominal_feasible')

    def legacy(self):
        repo = Path(__file__).resolve().parents[5]
        path = repo / ('artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146/'
                       'runtime_sources/drone/cbf_safety.py')
        if not path.exists():
            self.skipTest('Archived v9 source is not installed')
        spec = importlib.util.spec_from_file_location('legacy_cbf_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_failed_iterative_solve_cannot_masquerade_as_geometric_infeasibility(self):
        legacy = self.legacy()
        arguments = dict(nominal=[1., 0., 0.], position=[50., 50., 15.],
            peers={'peer': [65., 50., 15.]}, dt=1., world_bounds=(0., 0., 100., 100.))
        # Explicitly inject an unsuccessful optimizer result. This reproduces
        # the erroneous fallback branch without relying on a SciPy version or
        # pretending the archived float32 geometry is a bit-exact replay.
        with patch.object(legacy, 'minimize', return_value=SimpleNamespace(x=np.zeros(2))):
            _, previous = legacy.CBFSafetyFilter().filter(**arguments)
        self.assertTrue(previous['infeasible'])
        self.assertGreater(previous['feasibility_margin'], 0.)
        result, repaired = CBFSafetyFilter().filter(**arguments)
        self.assertFalse(repaired['infeasible'])
        np.testing.assert_allclose(result, [-.125, 0., 0.], atol=1e-12)
        self.assertLessEqual(repaired['max_constraint_residual'], 1e-7)

    def test_constraints_and_margin_match_archived_filter_including_multivehicle(self):
        legacy = self.legacy()
        arguments = dict(nominal=[1., -2., 0.], position=[50., 50., 15.],
            peers={'one': [70., 55., 15.], 'two': [45., 90., 20.]}, dt=.5,
            required_links=('one', '__ground__', 'vehicle', 'missing'), ground=[40., 40.],
            grounds={'vehicle': [55., 55.]}, ground_max_speed=1.,
            world_bounds=(0., 0., 100., 100.), drone_id='self')
        captured = []
        margin = stable.feasibility_margin

        def capture(matrix, bound, speed):
            captured.append((matrix.copy(), bound.copy(), speed))
            return margin(matrix, bound, speed)

        with patch.object(legacy, 'feasibility_margin', capture):
            _, old = legacy.CBFSafetyFilter().filter(**arguments)
        with patch.object(stable, 'feasibility_margin', capture):
            _, new = CBFSafetyFilter().filter(**arguments)
        np.testing.assert_array_equal(captured[0][0], captured[1][0])
        np.testing.assert_array_equal(captured[0][1], captured[1][1])
        for key in ('feasibility_margin', 'missing_links', 'required_link_distances_m',
                    'required_link_nominal_outward_m_s', 'min_received_separation'):
            self.assertEqual(old[key], new[key])


if __name__ == '__main__':
    unittest.main(verbosity=2)
