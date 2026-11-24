import math
import unittest
from types import SimpleNamespace

import numpy as np

from vtemp5.vtemp5_received_obstacle import ReceivedObstacleTracker
from vtemp5.vtemp5_world_model import fuse_measurements


class Predict:
    def __init__(self):
        self.calls = []

    def predict_sides(self, values):
        self.calls.append(values.copy())
        return fuse_measurements(np.tile([70.,25.,.6],(len(values),1)),values)


class ReceivedObstacleTests(unittest.TestCase):
    def setUp(self):
        self.cells = np.zeros((180,420),np.int8)
        self.model = Predict()
        self.tracker = ReceivedObstacleTracker(self.model,self.cells.shape,1.)
        self.car = np.array([65.5,90.5])
        self.goal = np.array([400.5,90.5])

    def report(self,mask,t=3.):
        ids = np.flatnonzero(mask)
        self.tracker.ingest_air(ids,self.cells.flat[ids],t-1.,t)

    def review(self,t=3.):
        return self.tracker.review(self.car,self.goal,self.cells,t)

    def test_waits_for_received_large_obstacle_and_leaves_unknown_measured_flags_false(self):
        self.cells[68:118,105:125] = 1
        self.assertFalse(self.review()['active'])
        subset = np.zeros_like(self.cells,bool)
        subset[89:92,105:125] = True
        self.report(subset)
        self.assertFalse(self.review()['active'])
        self.report(self.cells==1)
        state=self.review()
        self.assertTrue(state['active'])
        self.assertEqual(self.model.calls[0][:,:2].tolist(),[[0.,1.],[0.,-1.],[1.,1.],[1.,-1.]])
        self.assertEqual(self.tracker.prediction_calls,1)
        for side in (1,-1):
            self.assertEqual(state['side_status'][side],'unknown')
            self.assertEqual(state['route_observed'][side],[False,False,False])
        self.assertEqual(state['evidence_time'],3.)

    def test_only_complete_observed_cut_can_unlock_and_denial_is_remembered(self):
        self.cells[65:120,105:126] = 1
        self.report(self.cells==1)
        self.review()
        self.assertEqual(self.tracker.commit(math.pi/3,self.car),1)
        # The first confirmed solid mouth and a subsequent report leave a gap.
        self.cells[121:,105:126] = 1
        self.report(self.cells==1,t=4.)
        state=self.review(4.)
        self.assertEqual(state['side_status'][1],'unknown')
        self.assertEqual(self.tracker.commit(-math.pi/3,self.car),1)
        allowed=self.tracker.constrain_arcs([[-math.pi,math.pi]],self.car,0.)
        self.assertTrue(all(not lo<=-math.pi/3<=hi for lo,hi in allowed))
        # A fused-map-only cell cannot close the unmeasured air gap.
        self.cells[120,105:126] = 1
        self.assertEqual(self.review(5.)['side_status'][1],'unknown')
        self.report(self.cells==1,t=6.)
        state=self.review(6.)
        self.assertEqual(state['side_status'],{1:'blocked',-1:'unknown'})
        self.assertEqual(state['closure_evidence'][1]['received_at'],6.)
        self.assertEqual(self.tracker.commit(-math.pi/3,self.car),-1)
        self.assertEqual(self.tracker.switch_count,1)
        self.assertEqual(self.tracker.commit(math.pi/3,self.car),-1)

    def test_out_of_order_packets_and_actual_passage_release_lock_for_new_obstacle(self):
        self.cells[65:120,105:126] = 1
        self.report(self.cells==1)
        old = np.flatnonzero(self.cells==1)
        self.tracker.ingest_air(old,-np.ones(len(old)),1.,4.)
        self.cells[65:120,126:135] = -1
        state=self.review(4.)
        self.assertEqual(state['obstacle_id'],1)
        self.tracker.commit(-1.,self.car)
        self.car=np.array([155.5,90.5])
        self.assertFalse(self.review(5.)['active'])
        self.cells[66:121,205:226] = 1
        self.report(self.cells==1,t=6.)
        state=self.review(6.)
        self.assertEqual(state['obstacle_id'],2)
        self.assertEqual(self.tracker.locked_side,0)
        self.assertEqual(state['closure_evidence'],{})

    def test_known_open_route_needs_received_free_corridor_and_reports_provenance(self):
        self.cells[65:121,105:126] = 1
        self.cells[88:93,126:132] = -1
        self.report(self.cells==1)
        initial=self.review()
        self.assertEqual(initial['side_status'][1],'unknown')
        self.cells[128:140,90:140] = -1
        self.assertEqual(self.review(4.)['side_status'][1], 'unknown')
        self.cells[89:135, 65:91] = -1
        state=self.review(5.)
        self.assertEqual(state['side_status'][1],'open')
        self.assertEqual(state['route_open_probability'][1],1.)
        self.assertEqual(state['route_observed'][1],[True,True,False])
        self.assertEqual(state['route_observed'][-1],[False,False,False])

    def test_goalward_free_corridor_releases_off_axis_and_same_obstacle_recalls_side(self):
        self.cells[65:120,105:126] = 1
        self.report(self.cells == 1)
        self.review()
        self.assertEqual(self.tracker.commit(1., self.car), 1)
        self.car = np.array([154.5, 145.5])
        self.assertTrue(self.review(30.)['active'])  # downstream, but still unknown
        self.cells[130:155,150:185] = -1
        self.cells[130:155,165] = 0
        self.assertTrue(self.review(60.)['active'])  # an unknown strip cuts the corridor
        self.cells[130:155,165] = 1
        self.assertTrue(self.review(90.)['active'])
        self.cells[130:155,165] = -1
        state = self.review(120.)
        self.assertFalse(state['active'])
        self.assertEqual(state['release_evidence']['reason'], 'observed_downstream_goal_corridor')
        self.car = np.array([68.5, 102.5])
        resumed = self.review(150.)
        self.assertEqual(resumed['obstacle_id'], 1)
        self.assertEqual(self.tracker.obstacle_count, 1)
        self.assertEqual(resumed['locked_side'], 1)
        self.assertEqual(self.tracker.commit(-1., self.car), 1)

    def test_downstream_subcell_hypotheses_cannot_create_heading_split(self):
        self.cells[65:120,105:126] = 1
        self.report(self.cells == 1)
        self.review()
        self.tracker.commit(1., self.car)
        self.car = np.array([175.5, 131.5])
        state = self.review(30.)
        self.assertTrue(state['active'])
        state['routes'][-1][2,0] = state['routes'][1][2,0] + .6
        arcs = np.array([[-math.pi, math.pi]])
        np.testing.assert_array_equal(self.tracker.constrain_arcs(arcs,self.car,0.),arcs)
        self.assertEqual(self.tracker.commit(-1., self.car), 1)
        upstream = self.tracker.constrain_arcs(arcs,np.array([65.5,90.5]),0.)
        self.assertTrue(all(not lo <= -1. <= hi for lo, hi in upstream))

    def test_measured_rear_releases_off_axis_without_an_extra_corridor_report(self):
        self.cells[65:120,105:126] = 1
        self.cells[89:92,126:136] = -1
        self.report(self.cells == 1)
        self.review()
        self.tracker.commit(1., self.car)
        self.car = np.array([160.5, 146.5])
        state = self.review(30.)
        self.assertFalse(state['active'])
        self.assertEqual(state['release_evidence']['reason'], 'passed_measured_rear')

    def test_clear_goal_strip_before_downstream_exit_keeps_detour_active(self):
        self.cells[65:120,105:126] = 1
        self.report(self.cells == 1)
        self.review()
        self.tracker.commit(1., self.car)
        self.cells[125:165,90:155] = -1
        self.car = np.array([116.5, 147.5])
        self.assertTrue(self.review(30.)['active'])
        self.assertEqual(self.tracker.locked_side, 1)

    def test_attach_ignores_own_sensor_until_a_delivered_air_packet(self):
        count = []
        adapter = SimpleNamespace(cells=self.cells, resolution=1., now=10., _ingest=lambda *args: count.append(args[-1]))
        self.tracker.attach(adapter)
        self.tracker.attach(adapter)
        self.cells[75:106, 105:112] = 1
        occupied = np.ones((31,7),np.int8)
        adapter._ingest(occupied, [108.5,90.5], 1., 9., None)
        self.assertFalse(self.review(10.)['active'])
        adapter._ingest(occupied, [108.5,90.5], 1., 9., 'drone1')
        state = self.review(10.)
        self.assertTrue(state['active'])
        self.assertEqual(state['evidence_time'], 10.)
        self.assertEqual(count, [None, 'drone1'])
