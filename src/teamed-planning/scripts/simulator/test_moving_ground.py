"""Moving receiver integration and the two-term observation reward."""
import contextlib
import io
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from drone.mappo import ActorCritic
from drone.cbf_safety import CBFSafetyFilter
from simulator.mappo_experiment import Experiment,navigation_cost,cadmm_exploration_reward
from ground.ground_main import GroundRobotController


class MovingGroundTests(unittest.TestCase):
    def test_vehicle_motion_without_delivered_maps_has_no_map_reward(self):
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            env=Experiment(20000,ActorCritic(),drop=1.,moving_ground=True,
                reward_mode='cadmm_exploration',communication_mode='chain',safety_filter='cbf')
            for frame in range(8):
                _,reward,metrics=env.step(frame)
                self.assertEqual(reward,0.)
                self.assertEqual(metrics['cadmm_gain'],0.)
                self.assertEqual(metrics['exploration_area'],0.)
            self.assertEqual(env.ground_controller.graphs_received,0)
            self.assertGreater(env.summary()['ground_distance'],0.)
            env.env.close()

    def test_one_second_turn_does_not_oscillate_in_place(self):
        with contextlib.redirect_stdout(io.StringIO()):
            car=GroundRobotController('test',resolution=.5)
            car.set_goal((60.,75.))
            obs=dict(self=dict(position=np.array([15.,60.,0.]),orientation=0.),
                neighbors=[],occupancy_map=np.zeros((16,16),np.uint8),
                map_resolution=.5,time_step=1.,full_state=dict(time=0.))
            first=car.get_action(obs)
            self.assertGreater(abs(first['motion'][1]),.25)
            self.assertEqual(first['motion'][0],0.)
            obs['self']['orientation']=first['motion'][1]
            obs['full_state']['time']=1.
            second=car.get_action(obs)
            self.assertGreater(second['motion'][0],0.)
            self.assertLess(abs(second['motion'][1]),1e-6)

    def test_reward_terms_and_duplicate_observation(self):
        before=np.zeros((2,2),bool)
        after=before.copy();after[0,0]=True
        reward,terms=cadmm_exploration_reward(100.,80.,before,after,.5)
        self.assertAlmostEqual(reward,.205)
        self.assertEqual(terms['cadmm_gain'],20.)
        self.assertEqual(cadmm_exploration_reward(80.,80.,after,after,.5)[0],0.)
        self.assertLess(cadmm_exploration_reward(80.,100.,after,after,.5)[0],0.)
        self.assertEqual(navigation_cost(None,np.array([0.,0.]),[3.,4.]),100.)

    def test_ground_link_accounts_for_arbitrary_ground_motion(self):
        filt=CBFSafetyFilter()
        ground=np.array([0.,0.,0.]);position=np.array([93.,0.,15.])
        command,stats=filt.filter([6.,0.,0.],position,{},1.,
            required_links=['__ground__'],ground=ground,ground_max_speed=1.)
        self.assertFalse(stats['infeasible'])
        for angle in np.linspace(0,2*np.pi,50):
            ground_after=ground+np.array([np.cos(angle),np.sin(angle),0.])
            self.assertLessEqual(np.linalg.norm(position+command-ground_after),94.75+1e-6)

    def test_ground_moves_receives_maps_and_updates_mission(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        with contextlib.redirect_stdout(io.StringIO()):
            env=Experiment(20000,ActorCritic(),stochastic=True,exploration_candidates=5,
                communication_mode='chain',safety_filter='cbf',moving_ground=True,
                reward_mode='cadmm_exploration')
            start=env.obs[env.ground]['self']['position'].copy()
            for frame in range(36):
                _,reward,metrics=env.step(frame)
                self.assertTrue(np.isfinite(reward))
                self.assertEqual(metrics['chain_link_violation'],0.)
            summary=env.summary()
            self.assertGreater(summary['ground_distance'],1.)
            self.assertGreater(np.linalg.norm(env.obs[env.ground]['self']['position']-start),1.)
            self.assertGreater(env.ground_controller.graphs_received,0)
            self.assertGreater(len(env.ground_controller.unified_graph),0)
            self.assertGreater(summary['exploration_area'],0.)
            self.assertEqual(summary['ground_collisions'],0.)
            self.assertIn('planned_path',env.ground_action['broadcast'])
            self.assertAlmostEqual(summary['reward'],summary['cadmm_reward']+summary['exploration_reward'])
            env.env.close()


if __name__=='__main__':
    unittest.main()
