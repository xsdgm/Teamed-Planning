"""Communication reserve and instance-local received-geometry adapter contracts."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vtemp2_3 import hierarchical_safety as control
from vtemp2_3.hierarchical_cbf import CBFSafetyFilter
from vtemp2_3.safety_context import received_cbf_context, validate_cbf_context


class CommunicationGuardTests(unittest.TestCase):
    def arguments(self, peers=None, nominal=None):
        return dict(velocity=[-1., 0., 0.] if nominal is None else nominal,
            position=[100., 100., 15.],
            peers={'peer': [189., 100., 15.]} if peers is None else peers,
            dt=1., min_separation=15., comm_range=95., max_speed=6.,
            required_links=('peer',) if peers is None else tuple(peers),
            world_bounds=(0., 0., 300., 300.), drone_id='self')

    def hard_call(self, args, nominal=None):
        args = dict(args)
        velocity = args.pop('velocity') if nominal is None else nominal
        args.pop('velocity', None)
        parameters = [args.pop(key) for key in ('min_separation', 'comm_range', 'max_speed')]
        return CBFSafetyFilter(*parameters).filter(velocity, **args)

    def test_feasible_guard_shortens_stretched_link_and_final_command_is_hard_safe(self):
        args = self.arguments()
        original, hard = self.hard_call(args)
        result, info = control.guarded_filter(**args)
        self.assertTrue(info['communication_guard_feasible'])
        self.assertTrue(info['communication_guard_applied'])
        self.assertFalse(info['infeasible'])
        self.assertLess(np.linalg.norm(np.asarray(args['peers']['peer']) - (args['position'] + result)),
                        np.linalg.norm(np.asarray(args['peers']['peer']) - args['position']))
        self.assertGreater(result[0], 0.)
        self.assertLess(original[0], 0.)
        certified, certificate = self.hard_call(args, result)
        np.testing.assert_allclose(result, certified, atol=1e-10, rtol=0.)
        self.assertFalse(certificate['infeasible'])
        self.assertLessEqual(certificate['max_constraint_residual'], 1e-7)
        # Geometry/nominal diagnostics retain their original hard-envelope
        # meanings even though the command and intervention have changed.
        for key in ('feasibility_margin', 'required_link_distances_m',
                    'required_link_nominal_outward_m_s', 'missing_links'):
            self.assertEqual(info[key], hard[key])
        self.assertAlmostEqual(info['intervention'], np.linalg.norm(result[:2] - [-1., 0.]))

    def test_inner_infeasible_falls_back_to_hard_feasible_command(self):
        args = self.arguments(peers={'left': [7., 100., 15.], 'right': [193., 100., 15.]},
                              nominal=[0., 1., 0.])
        expected, hard = self.hard_call(args)
        result, info = control.guarded_filter(**args)
        self.assertFalse(hard['infeasible'])
        self.assertFalse(info['infeasible'])
        self.assertFalse(info['communication_guard_feasible'])
        self.assertFalse(info['communication_guard_applied'])
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(info['solver_status'], hard['solver_status'])

    def test_hard_infeasible_is_still_rejected(self):
        args = self.arguments(peers={'left': [6., 100., 15.], 'right': [194., 100., 15.]},
                              nominal=[0., 0., 0.])
        _, info = control.guarded_filter(**args)
        self.assertTrue(info['infeasible'])
        self.assertEqual(info['solver_status'], 'infeasible')
        self.assertIsNone(info['communication_guard_feasible'])
        self.assertFalse(info['communication_guard_applied'])

    def test_disabled_reserve_or_no_required_links_preserves_hard_command(self):
        for extra in [dict(reserve_m=0.), dict(required_links=())]:
            args = dict(self.arguments(), **extra)
            hard_args = dict(args)
            hard_args.pop('reserve_m', None)
            expected, _ = self.hard_call(hard_args)
            result, info = control.guarded_filter(**args)
            self.assertFalse(info['communication_guard_enabled'])
            np.testing.assert_array_equal(result, expected)
        for reserve in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                control.guarded_filter(**self.arguments(), reserve_m=reserve)

    def certificate_fixture(self, displacement):
        base_stats = dict(infeasible=False, solver_status='projected',
            feasibility_margin=1., required_link_nominal_outward_m_s={'peer': 1.},
            candidate_count=12, max_constraint_residual=0., intervention=0.)
        preferred = np.array([6. + displacement, 0., 0.])
        certified = np.array([6., 0., 0.])
        hard = SimpleNamespace(margin=.25, filter=Mock(side_effect=[
            (np.zeros(3), dict(base_stats)), (certified, dict(base_stats))]))
        inner = SimpleNamespace(filter=Mock(return_value=(preferred, dict(base_stats))))
        return hard, inner, certified

    def test_ulp_normalization_returns_certified_command(self):
        hard, inner, certified = self.certificate_fixture(1e-12)
        with patch.object(control, 'CBFSafetyFilter', side_effect=[hard, inner]):
            result, info = control.guarded_filter(**self.arguments())
        np.testing.assert_array_equal(result, certified)
        self.assertEqual(np.linalg.norm(result), 6.)
        self.assertTrue(info['communication_guard_applied'])

    def test_material_change_during_certificate_falls_back_to_safe_hard_command(self):
        hard, inner, _ = self.certificate_fixture(1e-6)
        with patch.object(control, 'CBFSafetyFilter', side_effect=[hard, inner]):
            result, stats = control.guarded_filter(**self.arguments())
        np.testing.assert_array_equal(result, np.zeros(3))
        self.assertFalse(stats['infeasible'])
        self.assertFalse(stats['communication_guard_applied'])
        self.assertEqual(stats['communication_guard_certificate_status'], 'rejected')
        self.assertGreater(stats['communication_guard_certificate_delta'], 1e-10)

    def test_failed_inner_certificate_preserves_previously_verified_hard_command(self):
        hard, inner, _ = self.certificate_fixture(0.)
        hard.filter.side_effect = [
            (np.array([.25, .5, 0.]), dict(infeasible=False, solver_status='projected',
                candidate_count=3, max_constraint_residual=0.)),
            (np.zeros(3), dict(infeasible=True))]
        with patch.object(control, 'CBFSafetyFilter', side_effect=[hard, inner]):
            result, stats = control.guarded_filter(**self.arguments())
        np.testing.assert_array_equal(result, [.25, .5, 0.])
        self.assertFalse(stats['infeasible'])
        self.assertFalse(stats['communication_guard_applied'])
        self.assertTrue(stats['communication_guard_certificate_infeasible'])
        self.assertEqual(stats['communication_guard_certificate_status'], 'rejected')


class ReceivedAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the real archived module through its normal isolation guard.
        # Controller instances below are minimal objects; no rollout or new
        # training process is started.
        from vtemp2_3.frozen_navigation import runtime
        cls.repo = Path(__file__).resolve().parents[5]
        runtime(cls.repo)
        from shared.drone.drone_main import DroneController
        from shared.drone import cbf_safety as archived
        cls.controller_type = DroneController
        cls.archived_filter = archived.CBFSafetyFilter
        cls.original_method = DroneController._cbf_filter

    def fixture(self, mode='backbone', multi=False):
        controller = self.controller_type.__new__(self.controller_type)
        chain = SimpleNamespace(required=('peer', '__ground__'), visible={'__ground__'},
            task=lambda: dict(position=[5., 5., 0.], source_time=3.))
        controller.__dict__.update(position=np.array([100., 100., 15.]),
            drone_id='self', communication_mode=mode, multi_vehicle=multi,
            communication_range=95., max_speed=6., ground_max_speed=1., _chain_state=chain,
            navigator=SimpleNamespace(chain=chain, min_separation=15.,
                world_bounds=(0., 0., 300., 300.), required_links=('peer', '__ground__'),
                ground_robot_position=np.array([5., 5., 0.])))
        observation = dict(time=10., time_step=1., full_state=dict(time=10.,
            robots={'hidden': dict(position=[101., 100., 15.])}), neighbors=[
                dict(id=1, type='drone', relative_position=[20., 5., 0.],
                     broadcast=dict(drone_id='peer', position=[999., 999., 15.], timestamp=7.)),
                dict(id=0, type='diff_drive', relative_position=[-20., 0., -15.],
                     broadcast=dict(robot_id='car', position=[999., 999., 0.], timestamp=8.))])
        return controller, observation

    def test_instance_binding_does_not_modify_frozen_class_module_or_other_instances(self):
        first, _ = self.fixture()
        untouched, _ = self.fixture()
        control.install_control_safety(SimpleNamespace(controllers=[first]))
        self.assertIs(first._cbf_filter.__func__, control._received_filter)
        self.assertIs(untouched._cbf_filter.__func__, type(self).original_method)
        self.assertIs(self.controller_type._cbf_filter, type(self).original_method)
        from shared.drone import cbf_safety as archived
        self.assertIs(archived.CBFSafetyFilter, self.archived_filter)
        self.assertIn('/artifacts/mappo-v9-', archived.__file__)

    def test_geometry_matches_frozen_adapter_in_all_communication_modes(self):
        stats = dict(infeasible=False, feasibility_margin=1., missing_links=[])
        for mode, multi in [('backbone', False), ('backbone', True), ('chain', False), ('direct', False)]:
            with self.subTest(mode=mode, multi=multi):
                controller, observation = self.fixture(mode, multi)
                received = []

                def original_capture(_filter, velocity, position, peers, dt, **geometry):
                    received.append(dict(position=position, peers=peers, dt=dt, **geometry))
                    return np.zeros(3), dict(stats)

                with patch.object(self.archived_filter, 'filter', original_capture):
                    controller._cbf_filter((1., 0., 0.), observation, 1.)
                control.install_control_safety(SimpleNamespace(controllers=[controller]))

                def repaired_capture(velocity, position, peers, dt, **geometry):
                    for key in ('min_separation', 'comm_range', 'max_speed'):
                        geometry.pop(key)
                    received.append(dict(position=position, peers=peers, dt=dt, **geometry))
                    return np.zeros(3), dict(stats)

                with patch.object(control, 'guarded_filter', repaired_capture):
                    controller._cbf_filter((1., 0., 0.), observation, 1.)
                before, after = map(control._json_value, received)
                before['required_links'] = sorted(before['required_links'])
                self.assertEqual(before, after)
                # Fresh sensor offsets override stale broadcasts/task cache.
                np.testing.assert_allclose(after['peers']['peer'], [120., 105., 15.])
                if multi:
                    self.assertEqual(after['grounds']['car'], [80., 100., 0.])
                else:
                    self.assertEqual(after['ground'], [80., 100., 0.])

    def test_context_and_json_diagnostics_preserve_received_evidence_and_final_action(self):
        controller, observation = self.fixture()
        experiment = SimpleNamespace(controllers=[controller])
        control.install_control_safety(experiment)
        result, stats = controller._cbf_filter((1., 0., 0.), observation, 1.)
        context = received_cbf_context(controller, observation, stats)
        validate_cbf_context(context, 'self', 10.)
        self.assertEqual(context['constraint_config']['communication_range'], 95.)
        self.assertEqual(context['cbf_margin_raw'], stats['feasibility_margin'])
        self.assertEqual([x['source_stamp'] for x in context['evidence']['neighbor_broadcasts']], [8., 7.])
        diagnostics = control.failure_diagnostics(experiment)
        restored = json.loads(json.dumps(diagnostics, allow_nan=False))
        self.assertEqual(restored[0]['filtered_velocity'], result.tolist())
        self.assertEqual(restored[0]['geometry']['ground'], [80., 100., 0.])
        self.assertEqual(restored[0]['communication_range'], 95.)
        self.assertEqual(restored[0]['control_version'], control.CONTROL_VERSION)
        # Full simulator state is never a geometry source.
        old = copy.deepcopy(controller._last_cbf_diagnostic)
        observation['full_state']['robots']['hidden']['position'] = [-10000., 99999., 0.]
        controller._cbf_filter((1., 0., 0.), observation, 1.)
        self.assertEqual(old, controller._last_cbf_diagnostic)
        result[:] = 999.
        self.assertEqual(old, controller._last_cbf_diagnostic)


if __name__ == '__main__':
    unittest.main(verbosity=2)
