"""Vehicle-side bridging of graph components across delivered free corridors."""
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
import networkx as nx
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.graph_structure import TopologicalGraph
from ground.graph_fusion import GraphFusion
from simulator.mappo_experiment import DeliveredEvidence, BRIDGE_CLEARANCE_M
from simulator.train_mappo import validate_config, reward_contract, run_episode, fixed_schedule, make_model


def config():
    return json.loads((Path(__file__).resolve().parents[2]/'config/algorithm/mappo_training_v8.json').read_text())


def chain(name, points):
    graph = TopologicalGraph(drone_id=name)
    for i, p in enumerate(points):
        graph.add_node(f'{name}{i}', p, {})
    for i in range(len(points)-1):
        graph.add_edge(f'{name}{i}', f'{name}{i+1}')
    return graph


def free_ledger(lo=(0., 0.), hi=(60., 30.)):
    """Delivered, free cells over [lo, hi); everything else unseen."""
    ledger = DeliveredEvidence(world=120.)
    xs = np.floor(np.asarray([lo[0], hi[0]])/ledger.resolution).astype(int)
    ys = np.floor(np.asarray([lo[1], hi[1]])/ledger.resolution).astype(int)
    ledger.seen[ys[0]:ys[1], xs[0]:xs[1]] = True
    ledger.log_odds[ys[0]:ys[1], xs[0]:xs[1]] = -np.log(4.)
    return ledger


def fuse(ledger, gap, bridge=15.):
    fusion = GraphFusion('ground_0')
    if ledger is not None:
        fusion.bridge_distance = bridge
        fusion.bridge_check = lambda a, b: ledger.segment_clear(a, b, BRIDGE_CLEARANCE_M)
    with contextlib.redirect_stdout(io.StringIO()):
        fusion.add_drone_graph('a', chain('a', [(10., 15.), (20., 15.)]))
        fusion.add_drone_graph('b', chain('b', [(20.+gap, 15.), (30.+gap, 15.)]))
    return fusion.get_unified_graph().graph


class BridgeTests(unittest.TestCase):
    def test_off_by_default_keeps_components_apart(self):
        graph = fuse(None, 3.7)
        self.assertEqual(nx.number_connected_components(graph), 2)

    def test_short_gap_across_delivered_free_corridor_gets_one_bridge(self):
        graph = fuse(free_ledger(), 3.7)
        self.assertEqual(nx.number_connected_components(graph), 1)
        bridged = [(a, b, d) for a, b, d in graph.edges(data=True) if d.get('bridged')]
        self.assertEqual(len(bridged), 1)
        a, b, data = bridged[0]
        self.assertAlmostEqual(data['weight'], 3.7)
        self.assertEqual({graph.nodes[a]['position'], graph.nodes[b]['position']}, {(20., 15.), (23.7, 15.)})

    def test_no_bridge_without_delivered_free_evidence(self):
        self.assertEqual(nx.number_connected_components(fuse(DeliveredEvidence(world=120.), 3.7)), 2)
        unseen_gap = free_ledger()
        unseen_gap.seen[7, 10:12] = False  # cells around x 20-24 m, y 14-16 m
        self.assertEqual(nx.number_connected_components(fuse(unseen_gap, 3.7)), 2)

    def test_occupied_cell_within_clearance_beside_the_line_blocks_it(self):
        ledger = free_ledger()
        ledger.log_odds[8, 11] = np.log(4.)  # y 16-18 m: off the y=15 line, within 1.5 m of it
        self.assertTrue(ledger.segment_clear((20., 15.), (23.7, 15.), 0.))
        self.assertFalse(ledger.segment_clear((20., 15.), (23.7, 15.), BRIDGE_CLEARANCE_M))
        self.assertEqual(nx.number_connected_components(fuse(ledger, 3.7)), 2)

    def test_gap_beyond_bridge_distance_stays_open(self):
        self.assertEqual(nx.number_connected_components(fuse(free_ledger(), 16.)), 2)
        self.assertEqual(nx.number_connected_components(fuse(free_ledger(), 7.2, bridge=5.)), 2)

    def test_outside_the_world_is_never_clear(self):
        self.assertFalse(free_ledger().segment_clear((-1., 15.), (5., 15.), 0.))


class BridgeConfigTests(unittest.TestCase):
    def test_default_off_bounds_and_contract_label(self):
        c = validate_config(config())
        self.assertEqual(c['fusion_bridge_distance'], 0.)
        self.assertFalse(reward_contract(c).endswith('_bridged'))
        self.assertTrue(reward_contract(validate_config(dict(c, fusion_bridge_distance=15.))).endswith('_bridged'))
        for bad in (-1., 15.5, True, 'x'):
            with self.assertRaises(ValueError):
                validate_config(dict(c, fusion_bridge_distance=bad))

    def test_short_episode_with_bridging(self):
        c = validate_config(dict(config(), steps=5, fusion_bridge_distance=15.))
        with contextlib.redirect_stdout(io.StringIO()):
            episode, summary = run_episode(make_model(c), c, 50000, fixed_schedule(c, 0), True)
        self.assertEqual(summary['episode_steps'], 5)
        self.assertTrue(all(np.isfinite(row[0]).all() for row in episode))


if __name__ == '__main__':
    unittest.main()
