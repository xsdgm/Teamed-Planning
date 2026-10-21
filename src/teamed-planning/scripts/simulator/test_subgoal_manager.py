"""Relay subgoals: selection, arrival, stall fallback and vehicle integration."""
import contextlib
import io
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from common.graph_structure import TopologicalGraph
from drone.mappo import ActorCritic
from ground.ground_main import GroundRobotController
from ground.subgoal_manager import SubgoalManager
from simulator.mappo_experiment import Experiment


def graph_with(*positions):
    graph=TopologicalGraph()
    for i,position in enumerate(positions):
        graph.add_node(f'n{i}',tuple(position))
    return graph


class SubgoalManagerTests(unittest.TestCase):
    def test_far_goal_gets_a_subgoal_on_the_straight_line(self):
        manager=SubgoalManager((800.,500.),radius=200.)
        target,changed=manager.update((100.,500.),None,0.)
        self.assertTrue(changed)
        np.testing.assert_allclose(target,(300.,500.),atol=1e-6)
        self.assertFalse(manager.final)

    def test_larger_arrival_radius_switches_earlier_but_not_at_the_final_goal(self):
        manager=SubgoalManager((800.,500.),radius=200.,arrival=60.)
        manager.update((100.,500.),None,0.)
        target,changed=manager.update((245.,500.),None,145.)
        self.assertTrue(changed)
        self.assertEqual(manager.arrivals,1)
        np.testing.assert_allclose(target,(445.,500.),atol=1e-6)
        final=SubgoalManager((250.,500.),radius=200.,arrival=60.)
        final.update((100.,500.),None,0.)
        _,changed=final.update((200.,500.),None,100.)
        self.assertFalse(changed)
        self.assertEqual(final.arrivals,0)

    def test_goal_within_radius_is_the_final_target(self):
        manager=SubgoalManager((250.,500.),radius=200.)
        target,_=manager.update((100.,500.),None,0.)
        np.testing.assert_allclose(target,(250.,500.))
        self.assertTrue(manager.final)

    def test_arrival_selects_the_next_subgoal(self):
        manager=SubgoalManager((800.,500.),radius=200.)
        manager.update((100.,500.),None,0.)
        target,changed=manager.update((280.,500.),None,180.)
        self.assertTrue(changed)
        np.testing.assert_allclose(target,(480.,500.),atol=1e-6)
        self.assertEqual(manager.arrivals,1)

    def test_stall_fails_and_detours_away_from_the_failed_target(self):
        manager=SubgoalManager((800.,500.),radius=200.,stall=100.)
        first,_=manager.update((100.,500.),None,0.)
        _,changed=manager.update((100.,500.),None,99.)
        self.assertFalse(changed)
        second,changed=manager.update((100.,500.),None,100.)
        self.assertTrue(changed)
        self.assertEqual(manager.failed,1)
        self.assertGreater(np.linalg.norm(second-first),manager.penalty_radius)
        # Still the best remaining progress: an offset ahead, not a retreat.
        self.assertGreater(second[0],100.)

    def test_failure_penalty_decays_so_an_area_can_be_retried(self):
        manager=SubgoalManager((800.,500.),radius=200.,stall=100.,penalty_decay=50.)
        first,_=manager.update((100.,500.),None,0.)
        manager.update((100.,500.),None,100.)
        manager.target=None
        retried,_=manager.update((100.,500.),None,2000.)
        np.testing.assert_allclose(retried,first,atol=1e-6)

    def test_repeated_dead_ends_back_off_toward_the_rear(self):
        manager=SubgoalManager((800.,500.),radius=200.,stall=10.,penalty_radius=120.)
        targets=[manager.update((100.,500.),None,0.)[0]]
        for i in range(1,40):
            target,changed=manager.update((100.,500.),None,10.*i)
            if changed:
                targets.append(target)
        self.assertTrue(any(t[0]<100. for t in targets))

    def test_target_snaps_onto_a_mapped_free_node_only_when_close(self):
        manager=SubgoalManager((800.,500.),radius=200.)
        manager.update((100.,500.),None,0.)
        _,changed=manager.update((100.,500.),graph_with((305.,510.)),1.)
        self.assertFalse(changed)  # Far away: no needless replanning.
        target,changed=manager.update((260.,500.),graph_with((305.,510.)),2.)
        self.assertTrue(changed)
        np.testing.assert_allclose(target,(305.,510.))
        _,changed=manager.update((260.,500.),graph_with((305.,510.),(301.,501.)),3.)
        self.assertFalse(changed)

    def test_graph_growth_in_the_corridor_postpones_failure(self):
        manager=SubgoalManager((800.,500.),radius=200.,stall=100.)
        manager.update((100.,500.),None,0.)
        nodes=[]
        for t in range(1,30):
            nodes+=[(200.+t,480.+i) for i in range(11)]
            manager.update((100.,500.),graph_with(*nodes),10.*t)
        self.assertEqual(manager.failed,0)

    def test_graph_growth_postpones_failure_only_up_to_max_wait(self):
        manager=SubgoalManager((800.,500.),radius=200.,stall=100.,max_wait=300.)
        manager.update((100.,500.),None,0.)
        nodes=[]
        for t in range(1,31):
            nodes+=[(200.+t,480.+i) for i in range(11)]
            manager.update((100.,500.),graph_with(*nodes),10.*t)
            if manager.failed:
                break
        self.assertEqual((manager.failed,t),(1,30))

    def test_graph_growth_outside_the_corridor_is_not_progress(self):
        manager=SubgoalManager((800.,500.),radius=200.,stall=100.)
        manager.update((100.,500.),None,0.)
        nodes=[]
        for t in range(1,11):
            nodes+=[(50.+t,300.+i) for i in range(11)]
            manager.update((100.,500.),graph_with(*nodes),10.*t)
        self.assertEqual(manager.failed,1)

    def test_driving_away_from_the_subgoal_fails_it(self):
        manager=SubgoalManager((800.,500.),radius=200.)
        first,_=manager.update((100.,500.),None,0.)
        manager.update((150.,500.),None,50.)
        _,changed=manager.update((110.,500.),None,90.)
        self.assertFalse(changed)  # 40 m back from the closest approach
        second,changed=manager.update((105.,500.),None,95.)
        self.assertTrue(changed)
        self.assertEqual(manager.failed,1)
        self.assertGreater(np.linalg.norm(second-first),manager.penalty_radius)

    def test_subgoals_stay_inside_the_surveyed_bounds(self):
        manager=SubgoalManager((990.,990.),radius=200.,stall=1.,world_bounds=(0.,0.,1000.,1000.))
        for i in range(30):
            target,_=manager.update((900.,500.),None,float(i))
            self.assertTrue(((target>=10.)&(target<=990.)).all())


class SubgoalVehicleTests(unittest.TestCase):
    def test_controller_relay_mode_drives_to_and_broadcasts_subgoals(self):
        # The ROS ground node uses this controller path with ~subgoal_radius.
        obs=dict(self=dict(position=np.array([100.,500.,0.]),orientation=0.),neighbors=[],
                 occupancy_map=np.zeros((16,16),np.uint8),map_resolution=.5,time_step=1.,
                 full_state=dict(time=0.))
        with contextlib.redirect_stdout(io.StringIO()):
            car=GroundRobotController('test',resolution=.5,subgoal_radius=200.,world_bounds=(0.,0.,1000.,1000.))
            car.set_goal((800.,500.))
            self.assertIsNone(car.goal_position)
            action=car.get_action(obs)
            np.testing.assert_allclose(car.goal_position,(300.,500.),atol=1e-6)
            np.testing.assert_allclose(action['broadcast']['target'],(300.,500.),atol=1e-6)
            self.assertEqual(car.final_goal,(800.,500.))
            car.stop()
            self.assertIsNone(car.goal_position or car.subgoals)
            self.assertIsNone(car.get_action(obs)['broadcast']['target'])
            car.set_goal((800.,500.));car.reset()
            self.assertIsNone(car.subgoals)
            plain=GroundRobotController('test',resolution=.5)
            plain.set_goal((800.,500.))
            self.assertEqual(plain.goal_position,(800.,500.))
            self.assertIsNone(plain.subgoals)

    def test_vehicle_drives_to_and_broadcasts_the_subgoal(self):
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            env=Experiment(20000,ActorCritic(),subgoal_radius=30.)
            self.assertGreater(np.linalg.norm(env.goal-env.start),30.)
            targets=[env.subgoals.target.copy()]
            for frame in range(6):
                env.step(frame)
                targets.append(env.subgoals.target.copy())
            self.assertFalse(env.subgoals.final)
            np.testing.assert_allclose(env.ground_controller.goal_position,targets[-1])
            np.testing.assert_allclose(env.ground_action['broadcast']['target'],targets[-1])
            # Rank 0 hears the vehicle directly (a step or two late) and plans
            # toward a subgoal, never toward the final goal.
            heard=np.asarray(env.controllers[0].navigator.ground_robot_target)
            self.assertTrue(any(np.allclose(heard,t) for t in targets))
            self.assertIn('subgoal_failures',env.summary())
            env.env.close()


if __name__=='__main__':
    unittest.main()
