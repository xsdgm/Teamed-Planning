"""Behavioural contracts: uncertainty, obstacle evidence and value selection."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from vtemp2_3.navigation_evidence import EvidenceGrid,AttemptMemory,candidates,features
from vtemp2_3.subgoal_value import ReturnModel,select


class EvidenceContracts(unittest.TestCase):
    def test_no_path_in_partial_map_stays_unknown(self):
        cells=np.zeros((30,40),np.int8)
        cells[8:13,3:8]=-1;cells[8:13,28:33]=-1
        grid=EvidenceGrid(cells,1.,clearance=0)
        self.assertEqual(grid.classify((5.5,10.5),(30.5,10.5),1.),'unknown')

    def test_observed_wall_proves_only_bounded_raster_blockage(self):
        cells=np.zeros((30,40),np.int8);cells[:,20]=1
        grid=EvidenceGrid(cells,1.,clearance=0)
        self.assertEqual(grid.classify((5.5,10.5),(30.5,10.5),1.),'blocked_within_bounds')

    def test_unobserved_gap_in_wall_keeps_possibility_open(self):
        cells=np.zeros((30,40),np.int8);cells[:,20]=1;cells[20,20]=0
        grid=EvidenceGrid(cells,1.,clearance=0)
        self.assertEqual(grid.classify((5.5,10.5),(30.5,10.5),1.),'unknown')

    def test_known_detour_is_reachable_even_without_straight_visibility(self):
        cells=-np.ones((30,40),np.int8);cells[:20,20]=1
        grid=EvidenceGrid(cells,1.,clearance=0)
        self.assertEqual(grid.classify((5.5,10.5),(30.5,10.5),1.),'known_reachable')

    def test_tree_at_target_does_not_block_whole_arrival_region(self):
        cells=-np.ones((40,40),np.int8);cells[20,25]=1
        grid=EvidenceGrid(cells,1.,clearance=1.)
        self.assertEqual(grid.classify((5.5,20.5),(25.5,20.5),0.),'blocked_within_bounds')
        self.assertEqual(grid.classify((5.5,20.5),(25.5,20.5),4.),'known_reachable')
        self.assertFalse(grid.blocked[20,30])

    def test_vehicle_clearance_can_close_too_narrow_gap(self):
        cells=-np.ones((30,40),np.int8);cells[:,20]=1;cells[20,20]=-1
        self.assertEqual(EvidenceGrid(cells,1.,clearance=0).classify((5.5,10.5),(30.5,10.5),1.),'known_reachable')
        self.assertEqual(EvidenceGrid(cells,1.,clearance=1).classify((5.5,10.5),(30.5,10.5),1.),'blocked_within_bounds')

    def test_timeout_never_changes_occupancy_or_unknown_label(self):
        grid=EvidenceGrid(np.zeros((40,40),np.int8),1.,clearance=0)
        before=grid.cells.copy();memory=AttemptMemory()
        for now in (180.,360.,540.):memory.record((5,20),(25,20),now,'timeout')
        np.testing.assert_array_equal(grid.cells,before)
        self.assertEqual(grid.classify((5,20),(25,20),1.),'unknown')
        self.assertGreater(memory.cost((5,20),(25,20),540.),0.)

    def test_failed_approach_does_not_ban_another_entrance(self):
        memory=AttemptMemory();memory.record((5,20),(25,20),0.,'control_stall')
        self.assertEqual(memory.cost((5,30),(25,20),1.),0.)
        self.assertEqual(memory.cost((5,20),(25,30),1.),0.)
        self.assertGreater(memory.cost((5,20),(25,20),1.),memory.cost((5,20),(25,20),600.))

    def test_execution_memory_cannot_claim_observed_obstacle(self):
        with self.assertRaises(ValueError):AttemptMemory().record((0,0),(10,0),0.,'physically_impossible')

    def test_pose_conflict_does_not_make_every_goal_impossible(self):
        cells=np.zeros((40,40),np.int8);cells[20,5]=1
        self.assertEqual(EvidenceGrid(cells,1.,clearance=0).classify((5.5,20.5),(25.5,20.5),1.),'source_conflict')

    def test_diagonal_contact_does_not_create_a_path_through_corners(self):
        cells=np.ones((3,3),np.int8);np.fill_diagonal(cells,-1)
        grid=EvidenceGrid(cells,1.,clearance=0)
        self.assertEqual(grid.classify((.5,.5),(2.5,2.5),0.),'blocked_within_bounds')

    def test_hidden_fields_do_not_enter_features(self):
        belief=dict(position=[20,20],goal=[80,20],grid=EvidenceGrid(np.zeros((100,100),np.int8),1.,clearance=0))
        first=features(belief,[50,20])[0]
        belief.update(hidden_map=np.ones((100,100)),future_reward=999.,all_drone_true_positions=[[80,80]])
        np.testing.assert_array_equal(first,features(belief,[50,20])[0])

    def test_candidates_include_original_unpenalized_winner(self):
        points=candidates([100,500],[800,500])
        np.testing.assert_allclose(points[0],[300,500])
        self.assertEqual(len(points),5)
        self.assertTrue((np.linalg.norm(points-[100,500],axis=1)<=200.00001).all())


class ValueContracts(unittest.TestCase):
    def test_unknown_high_value_candidate_remains_selectable(self):
        self.assertEqual(select([1.,10.,100.],['known_reachable','unknown','blocked_within_bounds']),1)

    def test_no_safe_candidate_returns_hold(self):
        self.assertIsNone(select([1.,2.],['outside_bounds','blocked_within_bounds']))

    def test_regressor_uses_execution_labels_and_roundtrips(self):
        x=np.array([[-2.],[-1.],[0.],[1.],[2.]])
        model=ReturnModel(.1).fit(x,-3*x[:,0]+2)
        self.assertGreater(model.predict([[-1.]])[0],model.predict([[1.]])[0])
        original_mean=model.mean.copy()
        model.predict([[100.]])
        np.testing.assert_array_equal(model.mean,original_mean)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'q.npz';model.save(path)
            np.testing.assert_allclose(ReturnModel.load(path).predict(x),model.predict(x))


if __name__=='__main__':unittest.main(verbosity=2)
