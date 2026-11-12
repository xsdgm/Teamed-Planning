"""V11 route task integration and reward regressions."""
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulator import train_mappo as training
from simulator.route_ab_experiment import RouteABExperiment
from tests.simulator.test_survey_training_v10 import scene
from drone.mappo import ActorCritic
from drone.survey_mappo import RouteABActorCritic

CONFIG = Path(__file__).resolve().parents[3]/'config/algorithm/mappo_training_v11.json'

class RouteABTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def settings(self):
        return training.validate_config(dict(json.loads(CONFIG.read_text()), steps=5, min_episode_steps=5, map_stability_steps=2))

    def model(self):
        model = training.make_model(self.settings())
        with torch.no_grad():
            model.gat_actor[-1].weight.zero_()
            model.gat_actor[-1].bias.zero_()
        return model

    def run_short(self):
        config = self.settings()
        with patch('simulator.route_ab_experiment.make_survey_scene', return_value=scene()):
            return training.run_episode(self.model(), config, 120000, training.fixed_schedule(config,0), False)

    def test_version_roundtrip_and_v9_warm_start(self):
        model = self.model()
        self.assertIsInstance(ActorCritic.from_state_dict(model.state_dict()), RouteABActorCritic)
        self.assertEqual(training.feature_version(self.settings()), 12)
        from drone.relay_attention_mappo import RelayActorCritic
        model.warm_start_actor(RelayActorCritic().state_dict())
        self.assertEqual(int(model.route_ab_schema), 1)

    def test_real_vehicle_and_task_origin_are_separate(self):
        fixture=scene()
        with contextlib.redirect_stdout(io.StringIO()):
            env=RouteABExperiment(120000,self.model(),survey_task=fixture['survey_task'],
                stochastic=False,world_size=1000.,start=fixture['start'],scene=(fixture['goal'],[],1),
                stop_on_ground_arrival=False,moving_ground=True,ground_stationary=True,
                communication_mode='backbone',safety_filter='cbf',reward_mode='velocity_mapping',
                mapping_route_origin='initial',exploration_weight=0.)
            self.addCleanup(env.env.close)
            _,reward,metrics=env.step(0)
        np.testing.assert_allclose(env.ground_action['broadcast']['position'][:2],(500.,500.))
        self.assertEqual(metrics['route_origin_x'],506.)
        self.assertEqual(metrics['route_target_x'],516.)
        self.assertNotIn('survey_confidence_reward',metrics)

    def test_rollout_reward_balances_and_selection_uses_route(self):
        rollout,summary=self.run_short()
        self.assertAlmostEqual(sum(r[-2] for r in rollout),summary['reward'])
        rewards=sum(summary[k] for k in ('mapping_reward','path_quality_reward','frontier_progress_reward','map_completion_bonus'))
        costs=sum(summary[k] for k in ('time_cost','intervention_cost','cbf_margin_cost','spacing_cost','chain_risk_cost','chain_outward_cost','chain_missing_cost','safety_failure_cost'))
        self.assertAlmostEqual(summary['reward'],rewards-costs)
        self.assertNotIn('survey_confidence_reward',summary)
        self.assertEqual(training.score_validation([summary],self.settings())['success_metric'],'route_completed')
        json.dumps(summary,allow_nan=False)

    def test_stable_route_completes_without_confidence_threshold(self):
        original=RouteABExperiment.step
        def connected(env,frame):
            samples,reward,metrics=original(env,frame)
            metrics['graph_task_connected']=1.
            metrics['survey_connected']=1.
            env.trace[-1].update(metrics)
            return samples,reward,metrics
        with patch.object(RouteABExperiment,'step',connected):
            rollout,summary=self.run_short()
        self.assertEqual(len(rollout),2)
        self.assertEqual(summary['route_completed'],1.)
        self.assertEqual(summary['map_completion_bonus'],100.)
        self.assertEqual(summary['budget_exhausted'],0.)

    def test_stationary_map_has_no_route_gain_when_vehicle_moves(self):
        from simulator.navigation_metrics import vehicle_route_update
        metrics=dict(graph_task_connected=0.,graph_task_path_length=0.,graph_frontier_goal_distance_m=5.)
        terms=vehicle_route_update(metrics,metrics,np.array([506.,496.]),np.array([516.,504.]))
        self.assertEqual(terms['route_quality_gain'],0.)
        self.assertEqual(terms['frontier_progress_gain'],0.)

    def test_cbf_rejection_terminates_without_completion(self):
        from drone.cbf_safety import CBFSafetyFilter
        original=CBFSafetyFilter.filter
        def reject(safety_filter,*args,**kwargs):
            velocity,stats=original(safety_filter,*args,**kwargs)
            stats['infeasible']=True
            return velocity,stats
        with patch.object(CBFSafetyFilter,'filter',reject):
            rollout,summary=self.run_short()
        self.assertEqual(len(rollout),1)
        self.assertEqual(summary['reward'],-100.)
        self.assertEqual(summary['route_completed'],0.)
        self.assertEqual(summary['safety_terminated'],1.)

    def test_relay_envelope_accepts_chain_and_rejects_remote_vehicle(self):
        from simulator.route_ab_experiment import route_radio_feasible
        self.assertTrue(route_radio_feasible((0.,0.),(140.,0.),(0.,0.),3))
        self.assertFalse(route_radio_feasible((0.,0.),(140.,0.),(500.,0.),3))

    def test_vehicle_waits_when_radio_neighbor_is_too_far(self):
        from types import SimpleNamespace
        from simulator.survey_experiment import SurveyExperiment
        from simulator.survey_reward_v10 import SurveyTask
        env=object.__new__(RouteABExperiment)
        env.vehicle_blocked_steps=0
        env.ground='car'
        env.ids=['d1','d2','d3']
        env.survey_task=SurveyTask((500.,500.),(640.,500.),12.)
        env.env=SimpleNamespace(config=SimpleNamespace(dt=1.))
        env.ground_controller=SimpleNamespace(heading=0.)
        env.obs={'car':{'self':{'position':[500.,500.,0.]},
                       'neighbors':[{'relative_position':[-95.,0.,15.]}]}}
        with patch.object(RouteABExperiment,'_navigation_action',return_value={'motion':(1.,.2)}):
            action=env._ground_action()
        self.assertEqual(action['motion'],(0.,.2))
        env.obs['car']['neighbors']=[{'relative_position':[10.,0.,15.]}]
        with patch.object(RouteABExperiment,'_navigation_action',return_value={'motion':(1.,.2)}):
            self.assertEqual(env._ground_action()['motion'],(1.,.2))

    def test_vehicle_selects_reachable_goal_other_than_b(self):
        import networkx as nx
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from simulator.survey_reward_v10 import SurveyTask
        graph=MagicMock()
        graph.__len__.return_value=2
        graph.graph=nx.Graph()
        graph.graph.add_node(0,position=np.array([500.,500.]))
        graph.graph.add_node(1,position=np.array([530.,510.]))
        graph.graph.add_edge(0,1)
        env=object.__new__(RouteABExperiment)
        env.ground_controller=SimpleNamespace(unified_graph=graph)
        env.ground_stationary=False
        env.ledger=SimpleNamespace(segment_clear=lambda a,b,clearance: True)
        env.ground_safety_distance=1.
        env.vehicle_target=None
        env.vehicle_blocked_steps=0
        env.vehicle_target_selected_at=-100.
        env.vehicle_target_switches=0
        env.env=SimpleNamespace(time=0.)
        env.survey_task=SurveyTask((500.,500.),(640.,500.),12.)
        env.ids=['d1','d2','d3']
        env.rng=np.random.default_rng(1)
        target=env._select_vehicle_target(np.array([500.,500.]))
        np.testing.assert_allclose(target,[530.,510.])
        self.assertFalse(np.array_equal(target,env.survey_task.b))
        self.assertEqual(env.vehicle_target_switches,1)

    def test_disconnected_route_times_out(self):
        original=RouteABExperiment.step
        def disconnected(env,frame):
            samples,reward,metrics=original(env,frame)
            metrics['graph_task_connected']=0.
            env.trace[-1].update(metrics)
            return samples,reward,metrics
        with patch.object(RouteABExperiment,'step',disconnected):
            rollout,summary=self.run_short()
        self.assertEqual(len(rollout),5)
        self.assertEqual(summary['route_completed'],0.)
        self.assertEqual(summary['map_completion_bonus'],0.)

if __name__=='__main__':
    unittest.main()
