"""Regression cases found during the fixed-reward convergence diagnosis."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from common.graph_structure import TopologicalGraph
from drone.skeleton_graph_builder import SkeletonGraphBuilder
from drone.semantic_segmentation import TerrainType
from drone.mappo import ActorCritic
from ground.graph_fusion import GraphFusion
from simulator.scenarios import generate_scene,scene_feasible
from simulator.mappo_experiment import navigation_cost,update,Experiment
from simulator.validate_ground_navigation import validate_scene


class ConvergenceTests(unittest.TestCase):
    def graph(self,source='a'):
        graph=TopologicalGraph(source)
        graph.add_node('a',(0.,0.),dict(terrain_type='TRAVERSABLE',observation_count=0,
            legacy_observation_count=0))
        graph.observe_node('a',source,0.)
        return graph

    def test_partial_map_never_receives_same_node_discount(self):
        graph=TopologicalGraph('g')
        costs=[navigation_cost(graph,(0,0),(100,0))]
        for index,x in enumerate((0.,50.,100.)):
            node=str(index)
            graph.add_node(node,(x,0.),dict(terrain_type='TRAVERSABLE',observation_count=100))
            if index:
                graph.add_edge(str(index-1),node)
            costs.append(navigation_cost(graph,(0,0),(100,0)))
        np.testing.assert_allclose(costs,[2000.,2000.,1050.,100.])
        self.assertTrue(all(a>=b for a,b in zip(costs,costs[1:])))

    def test_batched_prediction_matches_scalar_same_node_and_disconnected(self):
        for disconnected in (False,True):
            graph=self.graph()
            if disconnected:
                graph.add_node('other',(80.,0.))
            centres=[(0.,0.),(40.,0.)]
            expected=[]
            for centre in centres:
                predicted=graph.predict_observation(centre,12.)
                expected.append(navigation_cost(predicted,(-5.,0.),(100.,0.)))
            actual=graph.predicted_path_costs(centres,12.,(-5.,0.),(100.,0.))
            np.testing.assert_allclose(actual,expected)

    def test_relay_loop_and_repeated_packet_do_not_inflate_counts(self):
        first=self.graph('a')
        second=TopologicalGraph.from_json(first.to_json())
        for i in range(8):
            first.fuse_with_graph(TopologicalGraph.from_message(second.to_message()),current_time=i)
            second.fuse_with_graph(TopologicalGraph.from_message(first.to_message()),current_time=i)
        self.assertEqual(first.graph.nodes['a']['observation_count'],1)
        self.assertAlmostEqual(first.graph.nodes['a']['confidence'],first.node_confidence('a'))
        second.observe_node('a','b',10.)
        first.fuse_with_graph(second)
        first.fuse_with_graph(second)
        self.assertEqual(first.graph.nodes['a']['observation_count'],2)

    def test_cumulative_source_updates_are_merged_by_max(self):
        first=self.graph()
        packet=first.to_message()
        first.observe_node('a','a',1.)
        first.fuse_with_graph(TopologicalGraph.from_message(packet))
        self.assertEqual(first.graph.nodes['a']['observation_count'],2)

    def test_local_observation_updates_once_per_visible_frame(self):
        builder=SkeletonGraphBuilder('local',resolution=2.,node_spacing=10.,min_clearance=3.,max_edge_distance=15.)
        semantic=np.full((12,12),TerrainType.TRAVERSABLE.value,np.uint8)
        with contextlib.redirect_stdout(io.StringIO()):
            graph=builder.build_graph(semantic,(24.,42.,15.),observation_id=0)
            node=next(iter(graph.graph))
            builder.update_graph(graph,semantic,(24.,42.,15.),observation_id=1)
            builder.update_graph(graph,semantic,(24.,42.,15.),observation_id=1)
            self.assertEqual(graph.graph.nodes[node]['observation_count'],2)
            builder.update_graph(graph,semantic,(100.,100.,15.),observation_id=2)
            self.assertEqual(graph.graph.nodes[node]['observation_count'],2)

    def test_packet_snapshot_and_receiver_do_not_alias(self):
        graph=self.graph()
        packet=graph.to_message()
        receiver=TopologicalGraph.from_message(packet)
        graph.observe_node('a','a',1.)
        receiver.observe_node('a','b',2.)
        self.assertEqual(TopologicalGraph.from_message(packet).graph.nodes['a']['observation_count'],1)

    def test_ground_merge_deduplicates_forwarded_sources(self):
        original=self.graph()
        relay=TopologicalGraph.from_message(original.to_message())
        fusion=GraphFusion('car')
        with contextlib.redirect_stdout(io.StringIO()):
            fusion.add_drone_graph('a',original)
            fusion.add_drone_graph('relay',relay)
        node=next(iter(fusion.unified_graph.graph))
        self.assertEqual(fusion.unified_graph.graph.nodes[node]['observation_count'],1)

    def test_legacy_packets_are_conservative(self):
        graph=TopologicalGraph('a');graph.add_node('x',(0,0),dict(observation_count=3))
        packet=graph.to_message()
        for _ in range(5):
            graph.fuse_with_graph(TopologicalGraph.from_message(packet))
        self.assertEqual(graph.graph.nodes['x']['observation_count'],3)

    def test_generated_validation_and_training_scenes_are_feasible(self):
        for seed in list(range(10000,10020))+list(range(20000,20005)):
            goal,circles,_=generate_scene(seed)
            self.assertTrue(scene_feasible(goal,circles),seed)
            again=generate_scene(seed)
            np.testing.assert_array_equal(goal,again[0])
            self.assertTrue(all(np.linalg.norm(goal-p)>r+3.5 for p,r in circles))

    def test_ppo_metrics_are_finite_and_update_parameters(self):
        torch.manual_seed(42)
        model=ActorCritic();rollout=[]
        for i in range(4):
            x=torch.randn(3,21,12);mask=torch.ones(3,21,dtype=torch.bool)
            with torch.no_grad():
                dist=model.distribution(x,mask);a=dist.sample();lp=dist.log_prob(a)
            rollout.append((x.numpy(),mask.numpy(),a.numpy(),lp.numpy(),float(i)))
        before=model.actor[0].weight.detach().clone()
        loss,metrics=update(model,torch.optim.Adam(model.parameters(),lr=3e-4),rollout,return_metrics=True)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertFalse(torch.equal(before,model.actor[0].weight))
        self.assertGreaterEqual(metrics['approx_kl'],0.)

    def test_packet_outage_is_distinct_from_geometric_connectivity(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env=Experiment(20000,ActorCritic(),drop=1.,moving_ground=True,
                communication_mode='chain',safety_filter='cbf',reward_mode='cadmm_exploration')
            try:
                for frame in range(8):
                    _,_,metrics=env.step(frame)
                self.assertEqual(metrics['chain_link_violation'],0.)
                self.assertEqual(metrics['communication_outage'],1.)
                self.assertEqual(metrics['evidence_source_coverage'],0.)
            finally:
                env.env.close()

    def test_ground_controller_reaches_goal_with_repeated_full_map_broadcast(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result=validate_scene(20000,steps=300,obstacle_count=0)
        self.assertTrue(result['goal_reached'],result['remaining'])
        self.assertEqual(result['collisions'],0)


if __name__=='__main__':
    unittest.main()
