from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from vtemp4.evaluate_vtemp4_waypoints import (
    ARMS, FORMAT, RING_ACTION_CONTRACT, RING_HEADING_FORMAT, RING_WAYPOINT_FORMAT,
    aggregate, audit_ring_markers, checkpoint_contract, excursion_returns, paired_results, validate_arms,
)
from vtemp4.vtemp4_ring_contract import (
    PERIODIC_RING_ACTION_CONTRACT, RING_CONTINUOUS_HISTORY_FORMAT, RING_PERIODIC_HISTORY_FORMAT,
    RING_UNIFIED_ATTENTION_FORMAT, RING_UNIFIED_GOAL_FORMAT, PERIODIC_GOAL_RING_ACTION_CONTRACT,
    PERIODIC_GOAL_RING_CONTRACTS, BODY_ANGLE_GOAL_RING_CONTRACTS, GOAL_RING_REVIEW_INTERVALS,
    RING_VEHICLE_GRAPH_BODY_LSTM_FORMAT, RING_VEHICLE_GRAPH_LSTM_FORMAT, VEHICLE_GRAPH_BODY_LSTM_VARIANT,
    GOAL_RESIDUAL_RING_CONTRACTS, RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT,
    VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_VARIANT,
)


class WaypointEvaluationTest(unittest.TestCase):
    def test_traversable_reward_must_match_the_checkpoint_and_manifest(self):
        arm = VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_VARIANT
        fmt = RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT
        contract = GOAL_RESIDUAL_RING_CONTRACTS[30]
        common = dict(clock=dict(lookahead_m=200., review_interval_s=30), action_contract=contract,
                      policy_variant=arm)
        state, manifest = dict(common, format=fmt), dict(common, policy_format=fmt)
        with self.assertRaisesRegex(ValueError, 'disagree on progress rewards'):
            checkpoint_contract(state, dict(manifest, progress_distance='traversable'), arm)
        self.assertEqual(checkpoint_contract(dict(state, progress_distance='traversable'),
                                             dict(manifest, progress_distance='traversable'), arm), contract)

    def test_all_new_arms_bind_format_variant_and_geometry_to_manifest(self):
        for arm in ARMS:
            fmt = RING_HEADING_FORMAT if arm == 'heading' else RING_WAYPOINT_FORMAT
            state = dict(format=fmt, clock=dict(lookahead_m=200.), action_contract=RING_ACTION_CONTRACT,
                         policy_variant=arm)
            manifest = dict(policy_format=fmt, clock=state['clock'], action_contract=RING_ACTION_CONTRACT,
                            policy_variant=arm)
            self.assertEqual(checkpoint_contract(state, manifest, arm), RING_ACTION_CONTRACT)
            for incompatible in (dict(action_contract=None), dict(policy_format=FORMAT), dict(policy_variant='other')):
                with self.subTest(arm=arm, incompatible=incompatible), self.assertRaises(ValueError):
                    checkpoint_contract(state, dict(manifest, **incompatible), arm)
            with self.assertRaises(ValueError):
                short = dict(lookahead_m=150.)
                checkpoint_contract(dict(state, clock=short), dict(manifest, clock=short), arm)
        self.assertIsNone(checkpoint_contract(dict(format=FORMAT, clock={}), dict(policy_format=FORMAT, clock={}),
                                              'heading'))

    def test_mixed_contracts_rejected_even_for_pilot(self):
        ring = (dict(action_contract=RING_ACTION_CONTRACT), {}, {})
        legacy = (dict(action_contract=None), {}, {})
        with patch('simulator.evaluate_vtemp4_waypoints.checkpoint_info', side_effect=[legacy, ring, ring]):
            with self.assertRaisesRegex(ValueError, 'three fixed-ring'):
                validate_arms(ARMS, [], None, True)

    def test_continuous_history_has_its_own_checkpoint_format(self):
        arm = 'continuous_heading_history'
        common = dict(clock={'lookahead_m': 200.}, policy_variant=arm, action_contract=RING_ACTION_CONTRACT)
        def verify(fmt):
            return checkpoint_contract(dict(common, format=fmt), dict(common, policy_format=fmt), arm)
        self.assertEqual(verify(RING_CONTINUOUS_HISTORY_FORMAT), RING_ACTION_CONTRACT)
        for fmt in (RING_HEADING_FORMAT, RING_WAYPOINT_FORMAT, FORMAT):
            with self.subTest(fmt=fmt), self.assertRaises(ValueError):
                verify(fmt)

    def test_ring_audit_checks_at_issue_and_through_ground_replans(self):
        track = np.array([[0., 0.], [4., 1.], [4., 2.]])
        target = [4., 201.]
        record = dict(shared_markers=[dict(id=0, issued_at_s=1, origin_xy=track[1].tolist(), target=target,
                                           vehicle_yaw_rad=np.pi, relative_angle_rad=-np.pi/2, radius_m=200.)],
                      waypoint_reviews=[dict(time_s=1, changed=True, waypoint_id=0, target=target)],
                      ground_replan_events=[dict(time_s=0, waypoint_id=None, shared_marker=None),
                                            dict(time_s=2, waypoint_id=0, shared_marker=target)])
        self.assertEqual(audit_ring_markers(record, track), 1)
        record['ground_replan_events'][1]['shared_marker'] = [4., 202.]
        with self.assertRaisesRegex(ValueError, 'moved'):
            audit_ring_markers(record, track)
        record['shared_markers'][0]['relative_angle_rad'] = 0.
        with self.assertRaisesRegex(ValueError, 'bearing'):
            audit_ring_markers(record, track)

    def test_periodic_contract_is_distinct_and_requires_15_second_clock(self):
        arm = 'periodic_heading_history'
        def check(fmt=RING_PERIODIC_HISTORY_FORMAT, contract=PERIODIC_RING_ACTION_CONTRACT, interval=15):
            values = dict(clock=dict(review_interval_s=interval, lookahead_m=200.),
                          policy_variant=arm, action_contract=contract)
            return checkpoint_contract(dict(values, format=fmt), dict(values, policy_format=fmt), arm)
        self.assertEqual(check(), PERIODIC_RING_ACTION_CONTRACT)
        for options in (dict(fmt=RING_CONTINUOUS_HISTORY_FORMAT), dict(contract=RING_ACTION_CONTRACT),
                        dict(interval=10), dict(interval=30)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                check(**options)

    def test_unified_and_history_only_periodic_weights_are_incompatible(self):
        for arm, expected, other in (
            ('periodic_unified_attention', RING_UNIFIED_ATTENTION_FORMAT, RING_PERIODIC_HISTORY_FORMAT),
            ('periodic_heading_history', RING_PERIODIC_HISTORY_FORMAT, RING_UNIFIED_ATTENTION_FORMAT),
        ):
            common = dict(clock=dict(review_interval_s=15, lookahead_m=200.),
                          policy_variant=arm, action_contract=PERIODIC_RING_ACTION_CONTRACT)
            for fmt in (expected, other):
                state, manifest = dict(common, format=fmt), dict(common, policy_format=fmt)
                with self.subTest(arm=arm, fmt=fmt):
                    if fmt == expected:
                        self.assertEqual(checkpoint_contract(state, manifest, arm), PERIODIC_RING_ACTION_CONTRACT)
                    else:
                        with self.assertRaises(ValueError):
                            checkpoint_contract(state, manifest, arm)

    def test_periodic_audit_covers_all_reviews_and_successful_physics_ticks(self):
        # A stopped vehicle with unchanged angle still gets a new ID at t=15.
        track = np.zeros((18, 2))
        target = [200., 0.]
        markers = [dict(id=i, issued_at_s=t, origin_xy=[0., 0.], target=target, vehicle_yaw_rad=0.,
                        relative_angle_rad=0., radius_m=200., first_physics_broadcast_s=t, physics_broadcasts=n)
                   for i, (t,n) in enumerate(((0,15), (15,2)))]
        reviews = [dict(time_s=m['issued_at_s'], turn=True, changed=True, accepted=True, angle_rad=0.,
                        vehicle_yaw_rad=0., waypoint_id=m['id'], target=target) for m in markers]
        record = dict(action_contract=PERIODIC_RING_ACTION_CONTRACT, shared_markers=markers,
                      heading_reviews=reviews, ground_replan_events=[], end_reason='collector_limit')
        self.assertEqual(audit_ring_markers(record, track), 2)
        with self.assertRaisesRegex(ValueError, '15-second'):
            audit_ring_markers(dict(record, shared_markers=markers[:1], heading_reviews=reviews[:1]), track)
        with self.assertRaisesRegex(ValueError, 'broadcast'):
            audit_ring_markers(dict(record, shared_markers=[markers[0],dict(markers[1], physics_broadcasts=1)]), track)
        with self.assertRaisesRegex(ValueError, 'angle'):
            audit_ring_markers(dict(record, heading_reviews=[reviews[0],dict(reviews[1],angle_rad=1.)]), track)
        # At 15s a time-budget ending never prepares another decision; an attempted
        # safety-failed physics tick did prepare one, without a completed broadcast.
        self.assertEqual(audit_ring_markers(dict(record, shared_markers=markers[:1], heading_reviews=reviews[:1]),
                                           track[:16]), 1)
        last = dict(markers[1], physics_broadcasts=0, first_physics_broadcast_s=None)
        self.assertEqual(audit_ring_markers(dict(record, end_reason='safety_failure',
                                                 shared_markers=[markers[0], last]), track[:16]), 2)
        first = dict(markers[0], physics_broadcasts=0, first_physics_broadcast_s=None)
        self.assertEqual(audit_ring_markers(dict(record, end_reason='safety_failure',
                                                 shared_markers=[first], heading_reviews=reviews[:1]), track[:1]), 1)
        with self.assertRaisesRegex(ValueError, '15-second'):
            audit_ring_markers(dict(record, end_reason='safety_failure', shared_markers=[], heading_reviews=[]), track[:1])

    def test_goal_override_is_versioned_and_requires_periodic_clock(self):
        arm = 'periodic_unified_attention_goal'
        for fmt, contract, interval, valid in (
            (RING_UNIFIED_GOAL_FORMAT, PERIODIC_GOAL_RING_ACTION_CONTRACT, 15, True),
            (RING_UNIFIED_GOAL_FORMAT, PERIODIC_GOAL_RING_CONTRACTS[30], 30, True),
            (RING_UNIFIED_GOAL_FORMAT, PERIODIC_GOAL_RING_ACTION_CONTRACT, 30, False),
            (RING_UNIFIED_GOAL_FORMAT, PERIODIC_GOAL_RING_CONTRACTS[30], 15, False),
            (RING_UNIFIED_ATTENTION_FORMAT, PERIODIC_RING_ACTION_CONTRACT, 15, False),
            (RING_UNIFIED_GOAL_FORMAT, PERIODIC_RING_ACTION_CONTRACT, 15, False),
            (RING_UNIFIED_GOAL_FORMAT, PERIODIC_GOAL_RING_ACTION_CONTRACT, 10, False),
        ):
            common = dict(clock=dict(review_interval_s=interval, lookahead_m=200.), policy_variant=arm,
                          action_contract=contract)
            args = dict(common, format=fmt), dict(common, policy_format=fmt), arm
            with self.subTest(fmt=fmt, contract=contract, interval=interval):
                if valid:
                    self.assertEqual(checkpoint_contract(*args), contract)
                else:
                    with self.assertRaises(ValueError):
                        checkpoint_contract(*args)

    def test_body_frame_checkpoint_rejects_legacy_formats_and_latent_actions(self):
        arm = VEHICLE_GRAPH_BODY_LSTM_VARIANT
        for interval, contract in BODY_ANGLE_GOAL_RING_CONTRACTS.items():
            common = dict(clock=dict(review_interval_s=interval, lookahead_m=200.), policy_variant=arm,
                          action_contract=contract)
            state = dict(common, format=RING_VEHICLE_GRAPH_BODY_LSTM_FORMAT)
            manifest = dict(common, policy_format=RING_VEHICLE_GRAPH_BODY_LSTM_FORMAT)
            self.assertEqual(checkpoint_contract(state, manifest, arm), contract)
            with self.assertRaises(ValueError):
                checkpoint_contract(dict(state, format=RING_VEHICLE_GRAPH_LSTM_FORMAT),
                                    dict(manifest, policy_format=RING_VEHICLE_GRAPH_LSTM_FORMAT), arm)
            legacy = PERIODIC_GOAL_RING_CONTRACTS[interval]
            with self.assertRaises(ValueError):
                checkpoint_contract(dict(state, action_contract=legacy), dict(manifest, action_contract=legacy), arm)

    def test_goal_override_is_evaluated_at_each_issue_and_holds_between_decisions(self):
        for contract, interval in GOAL_RING_REVIEW_INTERVALS.items():
            with self.subTest(contract=contract):
                self.check_periodic_goal_markers(interval, contract)

    def test_body_and_goal_residual_weights_require_their_own_action_meanings(self):
        variants = [(VEHICLE_GRAPH_BODY_LSTM_VARIANT, RING_VEHICLE_GRAPH_BODY_LSTM_FORMAT,
                     BODY_ANGLE_GOAL_RING_CONTRACTS),
                    (VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_VARIANT, RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT,
                     GOAL_RESIDUAL_RING_CONTRACTS)]
        for arm, fmt, contracts in variants:
            for interval, contract in contracts.items():
                common = dict(clock=dict(review_interval_s=interval, lookahead_m=200.), policy_variant=arm,
                              action_contract=contract)
                state, manifest = dict(common, format=fmt), dict(common, policy_format=fmt)
                self.assertEqual(checkpoint_contract(state, manifest, arm), contract)
                for other, other_fmt, other_contracts in variants:
                    if arm == other:
                        continue
                    with self.assertRaises(ValueError):
                        checkpoint_contract(dict(state, format=other_fmt), dict(manifest, policy_format=other_fmt), arm)
                    with self.assertRaises(ValueError):
                        mismatched = other_contracts[interval]
                        checkpoint_contract(dict(state, action_contract=mismatched),
                                            dict(manifest, action_contract=mismatched), arm)

    def check_periodic_goal_markers(self, interval, contract):
        goal = [400., 20.]
        # Cross inside the threshold between decisions, hit it exactly at the next issue,
        # then move closer and finally leave the radius on a later review.
        ticks = np.arange(3*interval + 3)
        track = np.column_stack((np.interp(ticks, [0, 5, interval, 2*interval, 3*interval],
                                          [199.999, 202., 200., 235., 198.]), np.full(len(ticks), 20.)))
        cases = [(0, 'policy_angle', [199.999, 220.], np.pi/2, 200.),
                 (interval, 'final_goal', goal, 0., 200.), (2*interval, 'final_goal', goal, 0., 165.),
                 (3*interval, 'policy_angle', [198., 220.], np.pi/2, 200.)]
        markers = [dict(id=i, issued_at_s=t, origin_xy=track[t].tolist(), target=target, vehicle_yaw_rad=np.pi/2,
                        relative_angle_rad=bearing-np.pi/2, radius_m=radius, designation_source=source,
                        first_physics_broadcast_s=t, physics_broadcasts=min(interval, len(track)-1-t))
                   for i, (t, source, target, bearing, radius) in enumerate(cases)]
        reviews = [dict(time_s=m['issued_at_s'], turn=True, changed=True, accepted=True,
                        angle_rad=m['relative_angle_rad'], vehicle_yaw_rad=m['vehicle_yaw_rad'], waypoint_id=m['id'],
                        target=m['target'], designation_source=m['designation_source']) for m in markers]
        record = dict(action_contract=contract, clock=dict(review_interval_s=interval), final_goal_xy=goal,
                      shared_markers=markers, heading_reviews=reviews,
                      ground_replan_events=[dict(time_s=5, waypoint_id=0, shared_marker=markers[0]['target'])],
                      end_reason='collector_limit')
        self.assertEqual(audit_ring_markers(record, track, goal), 4)
        with self.assertRaisesRegex(ValueError, 'clock'):
            audit_ring_markers(dict(record, clock=dict(review_interval_s=5)), track, goal)
        for index, marker in enumerate(markers):
            broken = deepcopy(record)
            broken['shared_markers'][index]['physics_broadcasts'] = marker['physics_broadcasts'] - 1
            with self.assertRaisesRegex(ValueError, 'broadcast'):
                audit_ring_markers(broken, track, goal)
        for index, bad in ((0, {'designation_source': 'final_goal'}), (1, {'designation_source': 'policy_angle'}),
                           (1, {'target': [399.999, 20.]}), (2, {'radius_m': 200.}),
                           (2, {'relative_angle_rad': 0.}), (3, {'designation_source': 'final_goal'})):
            broken = deepcopy(record)
            broken['shared_markers'][index].update(bad)
            with self.subTest(index=index, bad=bad), self.assertRaises(ValueError):
                audit_ring_markers(broken, track, goal)
        with self.assertRaisesRegex(ValueError, 'mission geometry'):
            audit_ring_markers(dict(record, final_goal_xy=[401., 20.]), track, goal)
        broken = deepcopy(record)
        broken['heading_reviews'][2]['designation_source'] = 'policy_angle'
        with self.assertRaisesRegex(ValueError, 'designation source'):
            audit_ring_markers(broken, track, goal)
        broken = deepcopy(record)
        broken['ground_replan_events'][0]['shared_marker'] = goal
        with self.assertRaisesRegex(ValueError, 'moved'):
            audit_ring_markers(broken, track, goal)

    def test_backtracking_requires_an_excursion_and_debounces_each_return(self):
        outbound = np.arange(21)
        loop = np.r_[outbound, outbound[-2::-1]]
        for path, count in ((np.zeros(120), 0), (outbound, 0), (loop, 1),
                            (np.r_[loop, np.zeros(60), loop[1:]], 2)):
            track = np.column_stack((path, np.zeros_like(path)))
            self.assertEqual(len(excursion_returns(track)), count)

    def test_collector_cutoff_keeps_distance_but_does_not_score_as_failure(self):
        row = dict(case_id='lake-dev', stratum='detour_left', fast_seed=7, arrived=True, truncated=False,
                   net_progress_m=100., initial_distance_m=100., terminal_distance_m=0., travelled_m=120.,
                   backtracking_count=1)
        partial = dict(row, arrived=False, truncated=True, net_progress_m=10., terminal_distance_m=90.)
        self.assertIsNone(aggregate([row, partial])['arrival_rate'])
        self.assertEqual(aggregate([row])['arrival_rate'], 1.)
        results = {arm: [row] for arm in ARMS}
        results['waypoint_history'] = [partial]
        paired = paired_results(results)[0]['differences']
        self.assertIsNone(paired['waypoint_history_minus_heading']['arrival'])
        self.assertEqual(paired['waypoint_history_minus_waypoint']['net_progress_m'], -90.)


if __name__ == '__main__':
    unittest.main()
