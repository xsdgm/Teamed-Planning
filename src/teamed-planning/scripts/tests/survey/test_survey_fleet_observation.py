"""Survey evidence must cross physical radio links and answer the dispatched question."""
from types import SimpleNamespace
import unittest

import numpy as np

from tests.simulator.test_ground_relay import ARCHIVE, archived_module
from survey.survey_fleet_observation import (
    FleetSurveyAdapter, _TerrainRadio, pack_tile, unpack_tile,
)


class SurveyTerrainTests(unittest.TestCase):
    def test_three_states_pack_in_two_bits(self):
        tile = np.array([[0, -1, 1], [1, -1, 0], [-1, 1, 0]], dtype=np.int8)
        payload = pack_tile(tile)
        self.assertEqual(len(payload), 3)
        np.testing.assert_array_equal(unpack_tile(dict(shape=(3, 3), pixels=payload)), tile)

    @unittest.skipUnless(ARCHIVE.is_dir(), 'Frozen V9 sources unavailable')
    def test_delayed_multihop_map_retains_source_pose_time_and_yx_axes(self):
        channel = archived_module('simulator/relay_channel.py').RelayChannel(seed=12)
        simulation = SimpleNamespace(time=0.)
        raw = np.zeros((4, 4))
        raw[1, 3] = 1.
        obs = {rid: dict(self=dict(position=(8.*rid, 16., 15.)), occupancy_map=raw.copy(),
                         map_axes='xy', map_resolution=2., neighbors=[]) for rid in (1, 2)}
        exp = SimpleNamespace(ground=0, ids=[1, 2], env=simulation, obs=obs, relay_channel=channel,
                              controllers=[SimpleNamespace(drone_id='a'), SimpleNamespace(drone_id='b')])
        radio = _TerrainRadio(exp, 12)
        self.addCleanup(radio.close)
        send = lambda sender: [dict(sender_id=sender, state={})]
        channel.transmit(send('b'), [1], 0)
        self.assertEqual(radio.link.deliver(0, 0), [])
        simulation.time = 1.
        obs[1]['neighbors'] = [dict(id=0)]
        channel.transmit(send('a'), [], 1)
        self.assertEqual(radio.link.deliver(0, 1), [])
        [arrivals] = radio.link.deliver(0, 2)
        packets = {entry['state']['source']:entry['state'] for entry in arrivals}
        self.assertEqual((packets[1]['hops'], packets[1]['observed_at']), (1, 1.))
        self.assertEqual((packets[2]['hops'], packets[2]['observed_at']), (2, 0.))
        self.assertEqual(packets[2]['position'], (16., 16.))
        np.testing.assert_array_equal(unpack_tile(packets[2]), raw.T)
        self.assertEqual(channel.stats['sent'], 1)
        radio.close()
        self.assertNotIn('transmit', vars(channel))

    def adapter(self, time=5.):
        a = FleetSurveyAdapter.__new__(FleetSurveyAdapter)
        a.exp = SimpleNamespace(env=SimpleNamespace(time=time))
        a.cells = np.zeros((40, 40), np.int8)
        a.stamps = np.full((40, 40), -1., np.float32)
        a.bounds = None
        a.air_new_cells = a.credit_cells = 0
        a.survey_answered = a.dispatch_answered = False
        a.last_survey_evidence = None
        a.last_choice = dict(target=np.array([18., 10.]), issued_at=2., route_relevance=.7)
        a.active_gap = np.array([40*10+10, 40*10+18])
        a.credited = np.zeros(2, bool)
        return a

    def test_paid_only_for_fresh_new_air_evidence_in_gap(self):
        free = np.zeros((8, 8), np.int8)
        a = self.adapter()
        a._ingest(free, (10, 10), 1., 4., 'd0')
        self.assertEqual(a.last_survey_evidence['kind'], 'partial')
        self.assertAlmostEqual(a.survey_credit()['reward_evidence'], .35)
        a._ingest(free, (10, 10), 1., 4., 'd0')
        a._ingest(free, (30, 30), 1., 4., 'd0')
        self.assertEqual(a.credit_cells, 1)
        a._ingest(free, (18, 10), 1., 4., 'd1')
        result = a.survey_credit()
        self.assertEqual(result['answer'], 'passable')
        self.assertEqual(result['gap_fraction'], 1.)
        self.assertAlmostEqual(result['reward_evidence'], 1.4)
        self.assertEqual(a.last_survey_evidence['observed_at'], 4.)
        self.assertEqual(a.last_survey_evidence['received_at'], 5.)
        self.assertEqual(a.last_survey_evidence['source'], 'd1')

    def test_ground_and_expired_or_predispatch_observations_do_not_earn_credit(self):
        blocked = np.ones((8, 8), np.int8)
        for time, stamp, source in [(5., 4., None), (5., 2., 'd0'), (45., 4., 'd0')]:
            with self.subTest(time=time, source=source):
                a = self.adapter(time)
                a._ingest(blocked, (10, 10), 1., stamp, source)
                self.assertFalse(a.survey_answered)
                self.assertEqual(a.survey_credit()['reward_evidence'], 0.)
                self.assertIsNone(a.last_survey_evidence)
        a = self.adapter()
        a._ingest(blocked, (10, 10), 1., 4., 'd0')
        self.assertEqual(a.survey_credit()['answer'], 'blocked')
        self.assertAlmostEqual(a.survey_credit()['reward_evidence'], 1.05)
