"""Behavioral checks for UAV-only map reward and reception-only chain shaping."""
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from drone.cbf_safety import CBFSafetyFilter
from drone.velocity_mappo import VelocityActorCritic
from simulator.mappo_experiment import (Experiment,communication_chain_terms,cbf_margin_terms,
    velocity_mapping_reward,SafetyTermination)
from simulator.navigation_metrics import RouteQualityTracker,task_graph_metrics
from simulator.large_world_scenarios import reference_polyline_length
from common.graph_structure import TopologicalGraph
from simulator.train_mappo import validate_config,run_episode,fixed_schedule,score_validation,reward_contract
from simulator.train_distance_curriculum import distance_band_summary


class MappingCommunicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self,**kw):
        base=dict(action_mode='velocity',exploration_candidates=0,world_size=1000.,
            reward_mode='velocity_mapping',termination='map',arrival_reward=0.,
            map_success_reward=100.,steps=6,map_stability_steps=2,initial_validation=False,
            path_quality_weight=20.,map_quality_reward=20.,safety_failure_penalty=100.,
            task_distances=[250],exploration_weight=.2,intervention_weight=.05,
            time_penalty=.01,chain_risk_weight=.1,chain_outward_weight=.05,chain_missing_weight=1.)
        return validate_config(dict(base,**kw))

    def test_mapping_potential_telescopes_and_ignores_revisits(self):
        empty=np.zeros((2,2),bool);mask=np.array([[True,False],[False,False]])
        costs=[4000.,3000.,3500.,1000.]
        reward=sum(velocity_mapping_reward(a,b,4000.,empty,empty,mask)[0]
                   for a,b in zip(costs,costs[1:]))
        self.assertAlmostEqual(reward,10.*(1.-1000./4000.)-.03)
        seen=np.ones((2,2),bool)
        fresh,terms=velocity_mapping_reward(1000.,1000.,4000.,empty,seen,mask)
        self.assertEqual(terms['task_exploration_area'],4.)
        repeated,_=velocity_mapping_reward(1000.,1000.,4000.,seen,seen,mask)
        self.assertAlmostEqual(fresh-repeated,.2*4./400.)

    def link_stats(self,x,nominal=6.,ground=(0.,0.,0.)):
        _,stats=CBFSafetyFilter().filter([nominal,0.,0.],[x,0.,15.],{},1.,
            required_links=('__ground__',),ground=ground,ground_max_speed=1.)
        return stats

    def test_margin_penalty_is_early_and_outward_action_is_distinguished(self):
        safe=communication_chain_terms(self.link_stats(70.))
        near=communication_chain_terms(self.link_stats(90.))
        nearer=communication_chain_terms(self.link_stats(92.))
        inward=communication_chain_terms(self.link_stats(90.,nominal=-6.))
        self.assertEqual(safe['chain_risk'],0.)
        self.assertGreater(near['chain_risk'],0.)
        self.assertGreater(nearer['chain_risk'],near['chain_risk'])
        self.assertEqual(near['chain_risk'],inward['chain_risk'])
        self.assertGreater(near['chain_outward_risk'],0.)
        self.assertEqual(inward['chain_outward_risk'],0.)
        self.assertAlmostEqual(self.link_stats(90.)['required_link_distances_m']['__ground__'],np.hypot(90.,15.))

    def test_missing_links_are_penalized_without_invented_positions(self):
        terms=communication_chain_terms(self.link_stats(90.,ground=None))
        self.assertEqual(terms['chain_missing_fraction'],1.)
        self.assertEqual(terms['chain_links_available'],0.)

    def test_local_mean_does_not_grow_with_number_of_identical_links(self):
        one=dict(required_link_distances_m={'a':90.},required_link_nominal_outward_m_s={'a':3.},missing_links=[])
        two=dict(required_link_distances_m={'a':90.,'b':90.},required_link_nominal_outward_m_s={'a':3.,'b':3.},missing_links=[])
        self.assertEqual(communication_chain_terms(one),communication_chain_terms(two))

    def test_experiment_reward_ignores_vehicle_relative_path_cost(self):
        torch.manual_seed(19);model=VelocityActorCritic();results=[]
        for vehicle_costs in ([10000.,-10000.],[-10000.,10000.]):
            with contextlib.redirect_stdout(io.StringIO()):
                env=Experiment(10021,model,stochastic=False,reward_mode='velocity_mapping',
                    exploration_weight=.2,time_penalty=.01,chain_risk_weight=.1)
                try:
                    # Each step evaluates vehicle-before, vehicle-after, then
                    # the fixed start->goal map. Only the third may pay UAVs.
                    with patch('simulator.mappo_experiment.navigation_cost',side_effect=[*vehicle_costs,1000.]) as cost:
                        _,reward,metrics=env.step(0)
                    np.testing.assert_array_equal(cost.call_args_list[-1].args[1],env.start)
                    results.append((reward,metrics['cadmm_gain']))
                finally:env.env.close()
        self.assertAlmostEqual(results[0][0],results[1][0])
        self.assertNotEqual(results[0][1],results[1][1])

    def handover_steps(self,until_connected):
        # Frame 0 disconnected, frame 1 first route (170 m), frame 2 shorter (100 m).
        torch.manual_seed(19);model=VelocityActorCritic();steps=[]
        with contextlib.redirect_stdout(io.StringIO()):
            env=Experiment(10021,model,stochastic=False,reward_mode='velocity_mapping',
                exploration_weight=.2,time_penalty=.01,chain_risk_weight=.1,path_quality_weight=20.,
                route_reference_length=85.,mapping_until_connected=until_connected)
            try:
                initial=env._initial_route_cost
                for frame,(route,length) in enumerate(((.8*initial,0.),(.5*initial,170.),(.2*initial,100.))):
                    metrics=dict(graph_task_connected=float(length>0),graph_task_path_length=length,
                        graph_task_path_ratio=length/85.,graph_nodes=10.,graph_edges=9.,graph_components=1.)
                    with patch('simulator.mappo_experiment.task_graph_metrics',return_value=metrics), \
                         patch('simulator.mappo_experiment.navigation_cost',side_effect=[0.,0.,route]):
                        _,reward,step=env.step(frame)
                    steps.append((reward,step['mapping_reward'],step['path_quality_reward']))
            finally:env.env.close()
        return steps

    def test_map_term_stops_at_first_connection_and_route_quality_takes_over(self):
        both=self.handover_steps(False);handed=self.handover_steps(True)
        np.testing.assert_allclose([s[1] for s in both],[2.,3.,3.])
        np.testing.assert_allclose([s[1] for s in handed],[2.,0.,0.])
        # Route quality is unchanged: first route 20*85/170, then 20*(85/100-.5).
        np.testing.assert_allclose([s[2] for s in handed],[0.,10.,7.])
        np.testing.assert_allclose([s[2] for s in both],[s[2] for s in handed])
        np.testing.assert_allclose([b[0]-h[0] for b,h in zip(both,handed)],[0.,3.,3.],atol=1e-10)

    def test_map_term_handover_requires_route_quality_and_marks_contract(self):
        self.assertEqual(reward_contract(self.config()),'uav_mapping_route_quality_and_communication_v2')
        self.assertEqual(reward_contract(self.config(mapping_reward_until_connected=True)),
                         'uav_mapping_route_quality_and_communication_v3')
        for fields in (dict(mapping_reward_until_connected=1),
                       dict(mapping_reward_until_connected=True,path_quality_weight=0.)):
            with self.assertRaises(ValueError):self.config(**fields)
        with self.assertRaises(ValueError):
            self.config(mapping_reward_until_connected=True,reward_mode='velocity_task',
                        path_quality_weight=0.,map_quality_reward=0.,map_success_reward=0.)
        with self.assertRaises(ValueError),contextlib.redirect_stdout(io.StringIO()):
            Experiment(10021,VelocityActorCritic(),reward_mode='velocity_mapping',
                       path_quality_weight=20.,mapping_until_connected=True)

    def execute(self,arrived=False,missing_frame=None,length=1000.,fail_frame=None,**options):
        config=self.config(**options);original=Experiment.step;observed=[]
        def mark(env,frame):
            if frame==fail_frame:
                raise SafetyTermination(1,[c.navigator.last_sample for c in env.controllers])
            route=length(frame) if callable(length) else length
            metrics=dict(graph_task_connected=float(frame>=1),graph_task_path_length=route if frame>=1 else 0.,
                graph_task_path_ratio=route/85. if frame>=1 else 0.,graph_nodes=10.,graph_edges=9.,graph_components=1.)
            with patch('simulator.mappo_experiment.task_graph_metrics',return_value=metrics):
                result=original(env,frame)
            observed.append(result[1])
            env.goal_reached=arrived
            env.trace[-1].update(chain_links_available=float(frame!=missing_frame),
                chain_max_distance_ratio=.5)
            return result
        scene=dict(start=np.array([15.,60.]),goal=np.array([100.,60.]),circles=[],attempts=1,reference_length=85.)
        torch.manual_seed(17);model=VelocityActorCritic()
        with patch('simulator.train_mappo.generate_training_scene',return_value=scene), \
             patch.object(Experiment,'step',mark),contextlib.redirect_stdout(io.StringIO()):
            rollout,summary=run_episode(model,config,50001,fixed_schedule(config,0),True)
        return rollout,summary,observed

    def test_margin_risk_uses_the_least_agent_and_missing_links(self):
        self.assertAlmostEqual(cbf_margin_terms([7.,2.,None])['cbf_margin_risk'],(6./8.)**2)
        self.assertEqual(cbf_margin_terms([7.,float('-inf')]),dict(cbf_margin_risk=1.,cbf_margin_min_m_s=-100.))
        self.assertEqual(cbf_margin_terms([None,9.]),dict(cbf_margin_risk=0.,cbf_margin_min_m_s=9.))
        self.assertEqual(cbf_margin_terms([None])['cbf_margin_risk'],0.)

    def test_margin_penalty_enters_step_reward_and_first_packets_are_not_losses(self):
        torch.manual_seed(19);model=VelocityActorCritic();steps=[]
        for weight in (0.,.25):
            with contextlib.redirect_stdout(io.StringIO()):
                env=Experiment(10021,model,stochastic=False,reward_mode='velocity_mapping',
                    exploration_weight=0.,time_penalty=.01,cbf_margin_weight=weight)
                try:
                    _,reward,metrics=env.step(0)
                    missing=[c.safety_stats['missing_links'] for c in env.controllers]
                finally:env.env.close()
            steps.append((reward,metrics))
        self.assertTrue(any(missing))  # nothing received before the first step
        self.assertLess(steps[1][1]['cbf_margin_risk'],1.)
        self.assertAlmostEqual(steps[0][0]-steps[1][0],.25*steps[1][1]['cbf_margin_risk'])
        self.assertEqual(steps[0][1]['cbf_margin_cost'],0.)

    def improvement(self,**options):
        return self.execute(**dict(dict(map_quality_reward=0.,steps=12),**options))

    def test_stable_map_pays_100_then_ends_after_an_improvement_plateau(self):
        rollout,summary,_=self.improvement(length=170.,map_improvement_steps=3)
        # Frame 1 first route, frame 2 stable map (+100), frames 3-5 without a better route.
        self.assertEqual(len(rollout),6)
        self.assertEqual((summary['map_completed'],summary['map_plateau'],summary['map_optimal']),(1.,1.,0.))
        self.assertEqual((summary['map_completion_step'],summary['improvement_steps']),(3.,3.))
        self.assertGreater(rollout[2][4],90.);self.assertTrue(all(r[4]<90. for i,r in enumerate(rollout) if i!=2))
        self.assertEqual((summary['map_completion_bonus'],summary['map_quality_bonus']),(100.,0.))
        self.assertEqual(summary['budget_exhausted'],0.)
        self.assertAlmostEqual(summary['reward'],sum(r[4] for r in rollout))
        # Critic phase: stability before the map, then 1 + plateau clock.
        np.testing.assert_allclose([r[5][4] for r in rollout],[0.,0.,.5,1.,1.+1/3,1.+2/3])

    def test_improvement_ends_at_target_quality_and_pays_only_quality_gains(self):
        rollout,summary,_=self.improvement(length=lambda f:170. if f<4 else 85.,map_improvement_steps=50,
                                           map_target_quality=.95)
        self.assertEqual(len(rollout),5)
        self.assertEqual((summary['map_optimal'],summary['map_completion_step']),(1.,3.))
        self.assertAlmostEqual(summary['path_quality_reward'],20.)  # 20*.5 first route, 20*.5 shortcut
        self.assertAlmostEqual(summary['route_efficiency'],1.)

    def test_cbf_failure_after_the_map_keeps_the_map_and_costs_100(self):
        rollout,summary,_=self.improvement(length=170.,map_improvement_steps=10,fail_frame=4)
        self.assertEqual(len(rollout),5)
        self.assertEqual(rollout[-1][4],-100.);self.assertGreater(rollout[2][4],90.)
        self.assertEqual((summary['map_completed'],summary['safety_terminated'],summary['budget_exhausted']),(1.,1.,0.))
        self.assertAlmostEqual(summary['reward'],sum(r[4] for r in rollout))

    def test_budget_after_the_map_is_not_a_timeout(self):
        rollout,summary,_=self.improvement(length=170.,map_improvement_steps=100,steps=8)
        self.assertEqual(len(rollout),8)
        self.assertEqual((summary['map_completed'],summary['budget_exhausted'],summary['map_plateau']),(1.,0.,0.))
        self.assertEqual(summary['improvement_steps'],5.)

    def test_v4_config_rules_and_contract(self):
        v4=dict(map_quality_reward=0.,map_improvement_steps=150,map_target_quality=.95,cbf_margin_weight=.25,
                mapping_reward_until_connected=True)
        self.assertEqual(reward_contract(self.config(**v4)),'uav_mapping_route_quality_and_communication_v4')
        for fields in (dict(map_quality_reward=20.),dict(path_quality_weight=0.,mapping_reward_until_connected=False),
                       dict(map_target_quality=0.),dict(map_target_quality=1.5),dict(map_improvement_steps=-1),
                       dict(cbf_margin_scale=0.),dict(cbf_margin_weight=-1.),
                       dict(cbf_margin_weight=.5,discount=.995)):  # 0.71/step x 200 >= 100
            with self.assertRaises(ValueError):self.config(**dict(v4,**fields))
        row=dict(seed=1,sample=0,episode_steps=100,map_completed=1.,map_completion_step=70.,graph_task_path_ratio=1.,
                 reward=0.,collisions=0.,chain_link_violation=0.,safety_terminated=0.,route_efficiency=1.)
        late=dict(row,map_completion_step=90.)
        self.assertGreater(score_validation([row],self.config(**v4))['score'],score_validation([late],self.config(**v4))['score'])

    def frontier_steps(self,weight,remaining):
        # Frontier goal distance per frame as a fraction of the task distance; None: connected.
        torch.manual_seed(19);model=VelocityActorCritic();steps=[]
        with contextlib.redirect_stdout(io.StringIO()):
            env=Experiment(10021,model,stochastic=False,reward_mode='velocity_mapping',
                exploration_weight=0.,time_penalty=.01,frontier_progress_weight=weight)
            try:
                distance=env.initial_goal_distance
                for frame,fraction in enumerate(remaining):
                    metrics=dict(graph_task_connected=float(fraction is None),graph_task_path_length=0.,
                        graph_task_path_ratio=0.,graph_nodes=10.,graph_edges=9.,graph_components=1.,
                        graph_frontier_goal_distance_m=0. if fraction is None else fraction*distance)
                    with patch('simulator.mappo_experiment.task_graph_metrics',return_value=metrics):
                        _,reward,step=env.step(frame)
                    steps.append((reward,step))
                summary=env.summary()
            finally:env.env.close()
        return steps,summary

    def test_frontier_progress_pays_only_best_so_far_gains(self):
        # Progress 0 (start node behind the start), .4, a rebuild back to .2, .5, then connected.
        remaining=[1.2,.6,.8,.5,None]
        paid,summary=self.frontier_steps(50.,remaining)
        unpaid,plain=self.frontier_steps(0.,remaining)
        np.testing.assert_allclose([s[1]['frontier_progress'] for s in paid],[0.,.4,.2,.5,1.],atol=1e-12)
        np.testing.assert_allclose([s[1]['frontier_best_progress'] for s in paid],[0.,.4,.4,.5,1.],atol=1e-12)
        np.testing.assert_allclose([p[0]-u[0] for p,u in zip(paid,unpaid)],[0.,20.,0.,5.,25.],atol=1e-9)
        self.assertAlmostEqual(summary['frontier_progress_reward'],50.)
        self.assertEqual(summary['frontier_best_progress'],1.)
        self.assertEqual(plain['frontier_progress_reward'],0.)
        self.assertEqual(plain['frontier_best_progress'],1.)  # logged even when unpaid
        for options in (dict(frontier_progress_weight=-1.),dict(reward_mode='velocity_task',frontier_progress_weight=1.)):
            with self.assertRaises(ValueError),contextlib.redirect_stdout(io.StringIO()):
                Experiment(10021,VelocityActorCritic(),**dict(dict(reward_mode='velocity_mapping'),**options))

    def test_v5_frontier_entropy_decay_config_and_contract(self):
        v5=dict(map_quality_reward=0.,map_improvement_steps=150,map_target_quality=.95,cbf_margin_weight=.1,
                mapping_reward_until_connected=True,frontier_progress_weight=50.,
                entropy_coefficient=.003,entropy_coefficient_final=.001,episodes=101)
        config=self.config(**v5)
        self.assertEqual(reward_contract(config),'uav_mapping_route_quality_and_communication_v5')
        self.assertEqual(reward_contract(self.config(**dict(v5,frontier_progress_weight=0.))),
                         'uav_mapping_route_quality_and_communication_v4')
        np.testing.assert_allclose([fixed_schedule(config,k)['entropy'] for k in (0,50,100,500)],
                                   [.003,.002,.001,.001])
        self.assertEqual(fixed_schedule(self.config(episodes=101),100)['entropy'],.01)  # no final value: constant
        for fields in (dict(frontier_progress_weight=-1.),dict(frontier_progress_weight=float('nan')),
                       dict(frontier_progress_weight=100.),dict(entropy_coefficient_final=-.001),
                       dict(entropy_coefficient_final=True),dict(entropy_coefficient_final=float('inf'))):
            with self.assertRaises(ValueError):self.config(**dict(v5,**fields))
        route=dict(reward_mode='route_graph',path_quality_weight=0.,map_quality_reward=0.)
        self.config(**route)
        with self.assertRaises(ValueError):self.config(**dict(route,frontier_progress_weight=50.))

    def test_vehicle_arrival_changes_neither_uav_reward_nor_map_termination(self):
        first,a,rewards=self.execute(False)
        second,b,_=self.execute(True)
        self.assertEqual(len(first),3)
        np.testing.assert_allclose([r[4] for r in first],[r[4] for r in second],atol=1e-10)
        self.assertEqual((a['map_completed'],b['map_completed']),(1.,1.))
        self.assertEqual((a['goal_reached'],b['goal_reached']),(0.,1.))
        self.assertAlmostEqual(first[-1][4],rewards[-1]+100.+20.*85./1000.)
        self.assertEqual(a['map_completion_bonus'],100.)
        self.assertAlmostEqual(a['map_quality_bonus'],20.*85./1000.)
        self.assertAlmostEqual(a['path_quality_reward'],20.*85./1000.)
        self.assertEqual(a['reward'],a['uav_reward'])
        self.assertAlmostEqual(a['reward'],sum(r[4] for r in first))

    def test_real_thirty_step_window_and_distinct_quality_bonus(self):
        rollout,summary,_=self.execute(length=170.,map_stability_steps=30,steps=33)
        self.assertEqual(len(rollout),31)  # frame 0 is disconnected
        self.assertEqual(summary['map_completion_bonus'],100.)
        self.assertEqual(summary['map_quality_bonus'],10.)
        self.assertEqual(summary['route_efficiency'],.5)

    def test_cbf_failure_still_costs_100_and_no_completion_bonus(self):
        rollout,summary,_=self.execute(fail_frame=2)
        self.assertEqual(rollout[-1][4],-100.)
        self.assertEqual(summary['safety_failure_cost'],100.)
        self.assertEqual(summary['map_completion_bonus'],0.)
        self.assertEqual(summary['map_quality_bonus'],0.)
        self.assertEqual(summary['map_completed'],0.)
        self.assertAlmostEqual(summary['reward'],sum(r[4] for r in rollout))

    def test_shorter_final_route_gets_higher_quality_reward_at_same_completion(self):
        _,long,_=self.execute(length=340.)
        _,short,_=self.execute(length=170.)
        self.assertEqual(long['map_completion_bonus'],short['map_completion_bonus'])
        self.assertAlmostEqual(short['reward']-long['reward'],2.*20.*(.5-.25))

    def test_missing_chain_resets_map_stability_window(self):
        rollout,summary,_=self.execute(missing_frame=2)
        self.assertEqual(len(rollout),5)
        self.assertEqual(summary['map_completed'],1.)

    def test_selection_ignores_ground_arrival_collisions_and_progress(self):
        row=dict(seed=1,sample=0,episode_steps=100,map_completed=1.,graph_task_connected=1.,
            graph_first_connection_step=70.,graph_task_path_ratio=1.2,reward=100.,ground_collisions=0.,
            goal_reached=0.,ground_goal_distance=100.,collisions=0.,chain_link_violation=0.,safety_terminated=0.)
        other=dict(row,ground_collisions=100.,goal_reached=1.,ground_goal_distance=0.)
        self.assertEqual(score_validation([row],self.config())['score'],score_validation([other],self.config())['score'])
        incomplete=dict(row,map_completed=0.)
        self.assertGreater(score_validation([row],self.config())['score'],score_validation([incomplete],self.config())['score'])

    def test_config_rejects_accidental_vehicle_reward_and_bad_chain_parameters(self):
        for fields in (dict(termination='arrival'),dict(arrival_reward=100.),
                       dict(chain_soft_ratio=1.),dict(chain_soft_ratio=0.),dict(chain_risk_weight=-1.),
                       dict(chain_missing_weight=float('nan'))):
            with self.assertRaises(ValueError):self.config(**fields)
        for fields in (dict(path_quality_weight=-1.),dict(map_quality_reward=float('inf'))):
            with self.assertRaises(ValueError):self.config(**fields)

    def test_distance_reports_measure_mapping_not_vehicle_arrival(self):
        row=dict(map_completed=1.,goal_reached=0.,safety_terminated=0.,budget_exhausted=0.,
            ground_goal_distance=200.,episode_steps=100,out_of_bounds=0.,boundary_roundoff_count=0.)
        result=distance_band_summary(dict(stage=1),[row],self.config())
        self.assertEqual(result['success_metric'],'map_completed')
        self.assertEqual(result['success_rate'],1.)
        self.assertEqual(result['vehicle_arrival_rate'],0.)

    def test_distance_reports_include_physical_route_quality(self):
        row=dict(map_completed=1.,goal_reached=0.,safety_terminated=0.,budget_exhausted=0.,
            ground_goal_distance=200.,episode_steps=100,out_of_bounds=0.,boundary_roundoff_count=0.,
            graph_task_connected=1.,route_efficiency=.75,route_excess_m=50.,route_new_best_saving_m=40.)
        result=distance_band_summary(dict(stage=1),[row],self.config())
        self.assertEqual(result['mean_route_efficiency_connected'],.75)
        self.assertEqual(result['mean_route_new_best_saving_m'],40.)
        row['graph_task_connected']=0.
        self.assertIsNone(distance_band_summary(dict(stage=1),[row],self.config())['mean_route_efficiency_connected'])


class PhysicalRouteQualityTests(unittest.TestCase):
    def metrics(self,length):
        return dict(graph_task_connected=float(length is not None),graph_task_path_length=length or 0.)

    def test_first_route_is_quality_not_fictitious_meters_and_savings_are_monotone(self):
        tracker=RouteQualityTracker(120.)
        empty=tracker.update(self.metrics(None))
        first=tracker.update(self.metrics(200.))
        improved=tracker.update(self.metrics(150.))
        self.assertEqual(empty['route_quality_gain'],0.)
        self.assertEqual(first['route_new_best_saving_m'],0.)
        self.assertEqual(first['route_lengths_comparable'],0.)
        self.assertEqual(first['route_first_quality_gain'],.6)
        self.assertEqual(improved['route_length_delta_m'],50.)
        self.assertEqual(improved['route_new_best_saving_m'],50.)
        self.assertAlmostEqual(improved['route_shortening_quality_gain'],.2)
        self.assertEqual(improved['route_excess_m'],30.)

    def test_detour_reconnection_and_arbitrarily_bad_first_route_cannot_farm_credit(self):
        tracker=RouteQualityTracker(100.)
        rows=[tracker.update(self.metrics(length)) for length in (10000.,200.,300.,None,200.,150.,150.)]
        self.assertEqual(rows[2]['route_length_delta_m'],-100.)
        for i in (2,3,4,6):self.assertEqual(rows[i]['route_quality_gain'],0.)
        self.assertEqual(rows[4]['route_lengths_comparable'],0.)
        self.assertEqual(rows[4]['route_new_best_saving_m'],0.)
        direct=RouteQualityTracker(100.).update(self.metrics(150.))
        self.assertAlmostEqual(sum(r['route_quality_gain'] for r in rows),direct['route_quality_gain'])

    def test_approximate_reference_caps_credit_and_scaling_is_distance_independent(self):
        a=RouteQualityTracker(100.);b=RouteQualityTracker(500.)
        for length in (200.,150.,100.,90.):
            self.assertAlmostEqual(a.update(self.metrics(length))['route_quality_gain'],
                b.update(self.metrics(length*5.))['route_quality_gain'])
        self.assertEqual(a.update(self.metrics(80.))['route_quality_gain'],0.)
        for invalid in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):RouteQualityTracker(invalid)

    def test_physical_graph_shortcut_pays_but_confidence_and_unrelated_edges_do_not(self):
        graph=TopologicalGraph('quality')
        for node,position in [('a',(5.,5.)),('b',(5.,15.)),('c',(25.,15.)),('d',(25.,5.))]:
            graph.add_node(node,position)
        for a,b in [('a','b'),('b','c'),('c','d')]:graph.add_edge(a,b)
        tracker=RouteQualityTracker(30.)
        def update():
            return tracker.update(task_graph_metrics(graph,np.array([0.,5.]),np.array([30.,5.]),lambda a,b:True,6.))
        first=update()
        self.assertEqual(first['route_length_after_m'],50.)
        graph.graph['a']['b']['weight']=100000.
        graph.add_node('elsewhere',(100.,100.))
        self.assertEqual(update()['route_quality_gain'],0.)
        graph.add_edge('a','d')
        shortcut=update()
        self.assertEqual(shortcut['route_length_after_m'],30.)
        self.assertEqual(shortcut['route_new_best_saving_m'],20.)
        self.assertAlmostEqual(shortcut['route_quality_gain'],.4)

    def test_frontier_distance_follows_only_the_start_connected_graph(self):
        graph=TopologicalGraph('frontier')
        start,goal=np.array([0.,0.]),np.array([100.,0.])
        def metrics():
            return task_graph_metrics(graph,start,goal,lambda a,b:True,6.)
        self.assertEqual(metrics()['graph_frontier_goal_distance_m'],100.)  # empty graph
        graph.add_node('far',(95.,0.))  # a goal node, but nothing reaches it from the start
        self.assertEqual(metrics()['graph_frontier_goal_distance_m'],100.)
        graph.add_node('a',(5.,0.))
        self.assertEqual(metrics()['graph_frontier_goal_distance_m'],95.)
        graph.add_node('b',(40.,30.));graph.add_edge('a','b')
        self.assertAlmostEqual(metrics()['graph_frontier_goal_distance_m'],np.hypot(60.,30.))
        graph.add_edge('b','far')
        connected=metrics()
        self.assertEqual((connected['graph_task_connected'],connected['graph_frontier_goal_distance_m']),(1.,0.))

    def test_reference_length_includes_both_endpoint_connectors(self):
        length=reference_polyline_length([0.,0.],[4.,0.],[[.5,.5],[3.5,.5]])
        self.assertAlmostEqual(length,3.+2.*np.sqrt(.5))


if __name__=='__main__':
    unittest.main()
