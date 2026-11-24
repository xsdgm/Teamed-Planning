"""Do not route via a nearby dead-end when a better local graph entry exists."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from shared.common.graph_structure import TopologicalGraph
from shared.ground.global_planner import GlobalPlanner


def graph(points, edges):
    result=TopologicalGraph('test')
    for key,position in points.items():
        result.add_node(key,position,dict(terrain_type='TRAVERSABLE',observation_count=20))
    for a,b in edges:
        result.add_edge(a,b)
    return result


class RouteEntryTests(unittest.TestCase):
    def setUp(self):
        output=contextlib.redirect_stdout(io.StringIO())
        output.__enter__();self.addCleanup(output.__exit__,None,None,None)
        self.points=dict(near=(-1.,0.),back=(-10.,0.),side=(0.,3.),front=(10.,0.),goal=(20.,0.))
        self.edges=[('near','back'),('back','side'),('side','front'),('front','goal')]
        self.graph=graph(self.points,self.edges)
        self.planner=GlobalPlanner('test')

    def test_entry_minimizes_approach_plus_graph_route(self):
        self.planner.entry_validator=lambda p:tuple(p) in (self.points['near'],self.points['side'])
        route=self.planner.plan_path((0.,0.),self.points['goal'],self.graph)
        self.assertEqual(tuple(route[1]),self.points['side'])
        self.assertNotIn(self.points['back'],route)
        self.assertEqual(self.planner.start_connection_idx,1)
        self.assertIn(1,self.planner.observed_connection_indices)

    def test_blocked_and_previously_failed_approaches_remain_rejected(self):
        self.planner.entry_validator=lambda p:tuple(p) in (self.points['near'],self.points['side'])
        self.planner.connection_validator=lambda a,b:tuple(b)!=self.points['side']
        route=self.planner.plan_path((0.,0.),self.points['goal'],self.graph)
        self.assertEqual(tuple(route[1]),self.points['near'])
        self.assertIn(self.points['back'],route)

    def test_distant_goal_vertex_cannot_skip_the_observed_route(self):
        g=graph(dict(near=(1.,0.),back=(-20.,0.),front=(16.,0.),goal=(30.,0.)),
                [('near','back'),('back','front'),('front','goal')])
        self.planner.entry_validator=lambda p:True
        route=self.planner.plan_path((0.,0.),(30.,0.),g)
        self.assertEqual(tuple(route[1]),(1.,0.))
        self.assertIn((-20.,0.),route)

    def test_disconnected_nearest_island_does_not_hide_a_reachable_goal_component(self):
        g=graph(dict(island=(0.,.5),side=(0.,3.),goal=(20.,0.)),[('side','goal')])
        self.planner.entry_validator=lambda p:True
        route=self.planner.plan_path((0.,0.),(20.,0.),g)
        self.assertEqual(tuple(route[1]),(0.,3.))

    def test_healthy_forward_entry_is_not_replaced_by_a_connector_estimate(self):
        g=graph(dict(near=(1.,0.),side=(5.,3.),goal=(20.,0.)),
                [('near','side'),('side','goal')])
        self.planner.entry_validator=lambda p:True
        route=self.planner.plan_path((0.,0.),(20.,0.),g)
        self.assertEqual(tuple(route[1]),(1.,0.))


if __name__=='__main__':
    unittest.main()
