"""Large-world sensing, graph evidence and reference route geometry."""
import contextlib, io, sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from simulator.mappo_experiment import Experiment,DeliveredEvidence
from simulator.large_world_scenarios import CircleGeometry,raster_free,reference_route


class TransferTests(unittest.TestCase):
    def test_ledger_and_boundary_expand_without_changing_sensor(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env=Experiment(1,None,world_size=1000.,start=(30.,500.),
                scene=(np.array([780.,500.]),[],1),moving_ground=True,stop_on_ground_arrival=False)
        try:
            self.assertEqual(env.ledger.seen.shape,(500,500))
            self.assertEqual(env.env.config.communication_radius,95.)
            self.assertEqual(env.obs[env.ids[0]]['occupancy_map'].shape,(12,12))
            self.assertEqual(env.controllers[0].navigator.world_bounds,(0.,0.,1000.,1000.))
            env.goal_reached=True
            self.assertFalse(env.done)
            np.testing.assert_array_equal(env.start,[30.,500.])
            ledger=DeliveredEvidence(world=1000.)
            self.assertTrue(ledger.receive(1,0,[950,950,15],np.zeros((12,12)),2.))
            self.assertTrue(ledger.seen[475,475])
        finally:env.env.close()

    def test_reference_and_geometry_never_cut_through_obstacle(self):
        circles=[([10.,10.],3.)]
        grid=raster_free(circles,world=24.,clearance=1.5+np.sqrt(.5))
        route=reference_route(grid,[3.,10.],[21.,10.])
        self.assertGreater(route['length'],18.)
        geometry=CircleGeometry(circles,world=24.)
        self.assertFalse(geometry.clear([3,10],[21,10]))
        for a,b in zip(route['points'],route['points'][1:]):self.assertTrue(geometry.clear(a,b))


if __name__=='__main__':unittest.main()
