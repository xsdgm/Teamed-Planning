"""Large-world plumbing must not alter old-model features or small-world physics."""
import contextlib,io,runpy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from drone.mappo import ActorCritic
from simulator.mappo_experiment import Experiment,DeliveredEvidence
from simulator.large_world_scenarios import CircleGeometry,raster_free,reference_route


class TransferTests(unittest.TestCase):
    def test_ledger_and_boundary_expand_without_changing_sensor(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env=Experiment(1,ActorCritic(),world_size=1000.,start=(30.,500.),
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

    def test_default_experiment_matches_frozen_training_implementation(self):
        root=Path(__file__).resolve().parents[4]
        source=root/'artifacts/mappo-navigation-v4-20261017/sources/simulator/mappo_experiment.py'
        if not source.exists():self.skipTest('Frozen local run not available')
        old=runpy.run_path(str(source))['Experiment']
        # Use exactly the same weights and RNG for both implementations.
        torch.manual_seed(42);model=ActorCritic();model.eval();summaries=[];positions=[]
        torch.set_num_threads(1)
        for cls in (old,Experiment):
            torch.manual_seed(9);np.random.seed(9)
            with contextlib.redirect_stdout(io.StringIO()):
                env=cls(31000,model,stochastic=False,exploration_candidates=5,
                    communication_mode='chain',safety_filter='cbf',moving_ground=True,reward_mode='cadmm_exploration')
                try:
                    with torch.inference_mode():
                        for frame in range(10):env.step(frame)
                    summaries.append(env.summary());positions.append(env.trace[-1]['positions'])
                finally:env.env.close()
        for key in ('reward','cadmm_reward','coverage_area','ground_goal_distance','graph_nodes','graph_edges','brier_error','calibration_error'):
            self.assertAlmostEqual(summaries[0][key],summaries[1][key],places=8,msg=key)
        np.testing.assert_allclose(positions[0],positions[1],rtol=0,atol=1e-10)


if __name__=='__main__':unittest.main()
