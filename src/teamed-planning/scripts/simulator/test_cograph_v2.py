"""CoGraph v2: single-task scenes, tree budget, decentralized vehicle claims, task geometry,
and the learned per-drone target choice."""
import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.graph_structure import TopologicalGraph
from drone.cograph_mappo import CoGraphActorCritic
from ground.task_planner import (TaskDecomposer, coverage_tree, drones_to_cover, GraphTravel, snap_to_graph,
    GROUND_LINK_M, LINK_M)
from ground.vehicle_claims import VehicleClaimAgent
from simulator.cograph_experiment import CoGraphExperiment, episode_budget
from simulator.cograph_scenarios import generate_cograph_scene
from simulator.train_mappo import validate_config, reward_contract, make_model, run_episode, fixed_schedule, train
from simulator.ppo_batch import update_batch
from torch.distributions import Independent, Normal, kl_divergence
from drone.cograph_v2_mappo import (CoGraphV2ActorCritic, SelectVelocity, RAW_WIDTH, VALID_START, TOKENS_START,
    TARGET_FEATURES, MAX_TARGETS, FOCUS_COLUMN, COMMIT_S)
from drone.relay_attention_mappo import RelayActorCritic, RAW_WIDTH as V8_WIDTH
from drone.mappo import ActorCritic

CONFIGS = Path(__file__).resolve().parents[2]/'config/algorithm'
NEAR8 = np.array([(100, 0), (0, 100), (-100, 0), (0, -100), (70, 70), (-70, 70), (70, -70), (300, 0)], float)


def config(name, **changes):
    value = dict(json.loads((CONFIGS/name).read_text()), init_checkpoint=None)
    value.update(changes)
    return validate_config(value)


def line_graph(points):
    graph = TopologicalGraph('test')
    for i, p in enumerate(points):
        graph.add_node(f'n{i}', tuple(map(float, p)))
    for i in range(1, len(points)):
        graph.add_edge(f'n{i-1}', f'n{i}')
    return graph


class CoverageTests(unittest.TestCase):
    def test_drone_count_per_distance(self):
        # The first hop climbs to 15 m (93.56 m horizontal); relays every 94.75 m.
        for distance, drones in ((50, 1), (93, 1), (94, 2), (188, 2), (283, 3), (284, 4), (400, 5)):
            self.assertEqual(drones_to_cover((0, 0), [(distance, 0)]), drones, distance)
        self.assertAlmostEqual(GROUND_LINK_M, (LINK_M**2-15.**2)**.5)

    def test_tree_shares_branches_and_counts_stars(self):
        self.assertEqual(drones_to_cover((0, 0), [(150, 0), (250, 0), (350, 0)]), 6)
        self.assertEqual(drones_to_cover((0, 0), [(200, 0), (-200, 0), (0, 200)]), 9)
        # 150 m from the vehicles (2 drones), then 90 m on (1 drone); 100 m on would need 2.
        drones, length, parent = coverage_tree((0, 0), [(150, 0), (240, 0)])
        self.assertEqual((drones, length, parent.tolist()), (3, 240., [-1, 0]))
        self.assertEqual(drones_to_cover((0, 0), [(150, 0), (250, 0)]), 4)
        self.assertEqual(drones_to_cover((0, 0), []), 0)

    def test_decomposer_builds_covered_targets_together(self):
        decomposer = TaskDecomposer(9)
        self.assertEqual([t.targets for t in decomposer.decompose((0, 0), NEAR8[:4], range(4), None)], [(0, 1, 2, 3)])
        # Seven near targets and one far one need 11 drones: the coverable cluster stays together.
        plan = decomposer.decompose((0, 0), NEAR8, range(8), None)
        self.assertEqual([t.targets for t in plan], [(0, 1, 2, 3, 4, 5, 6), (7,)])
        far = np.array([(760, 0), (760, 50), (760, -50)], float)
        task, = decomposer.decompose((0, 0), far, range(3), None)
        self.assertTrue(decomposer.covers(task.anchor, far))
        self.assertFalse(decomposer.covers(task.anchor-np.array([decomposer.anchor_step, 0.]), far))

    def test_graph_travel_and_snap(self):
        graph = line_graph([(0, 0), (50, 0), (50, 100)])
        travel = GraphTravel(graph, (0, 0), detour=1.4)
        self.assertAlmostEqual(travel.to((50, 100)), 150.)
        self.assertAlmostEqual(travel.to((50, 120)), 150.+1.4*20.)
        self.assertAlmostEqual(GraphTravel(None, (0, 0), detour=1.4).to((30, 40)), 70.)
        self.assertAlmostEqual(GraphTravel(graph, (500, 500), detour=1.4).to((500, 510)), 14.)
        np.testing.assert_allclose(snap_to_graph(graph, (55, 95)), (50, 100))
        np.testing.assert_allclose(snap_to_graph(graph, (300, 300)), (300, 300))


class ClaimTests(unittest.TestCase):
    """Each agent sees only its own position/graph and the messages handed to it."""
    @staticmethod
    def decide(agent, now, position, targets, progress=None, connected=None):
        n = len(targets)
        return agent.decide(now, position, targets, None, np.zeros(n) if progress is None else np.asarray(progress, float),
                            np.zeros(n, bool) if connected is None else np.asarray(connected, bool))

    @staticmethod
    def exchange(agents, now):
        messages = [dict(robot_id=a.robot_id, timestamp=now, **a.message()) for a in agents]
        for a in agents:
            a.receive(messages)

    def test_vehicles_split_targets_without_a_coordinator(self):
        targets = np.array([(100, 0), (0, 100), (-100, 0)], float)
        agents = [VehicleClaimAgent(f'ground_{k}', k) for k in range(3)]
        positions = [(10, 0), (0, 10), (-10, 0)]
        for now in (0., 1., 2.):
            for a, p in zip(agents, positions):
                self.decide(a, now, p, targets)
            self.exchange(agents, now)
        self.assertEqual([a.claim for a in agents], [0, 1, 2])

    def test_simultaneous_claim_goes_to_lower_cost_then_rank(self):
        targets = np.array([(100, 0), (-150, 0)], float)
        a, b = VehicleClaimAgent('ground_0', 0), VehicleClaimAgent('ground_1', 1)
        for now in (0., 1., 2.):
            self.decide(a, now, (0, 0), targets)
            self.decide(b, now, (0, 0), targets)
            self.exchange([a, b], now)
        self.assertEqual((a.claim, b.claim), (0, 1))
        self.assertEqual((b.conflicts, b.transfers_out), (1, 0))

    def test_mid_course_handover_to_a_clearly_better_vehicle(self):
        targets = np.array([(0, 0)], float)
        a = VehicleClaimAgent('ground_0', 0)
        for now in range(10):
            self.decide(a, float(now), (100, 0), targets)
        near, close = VehicleClaimAgent('ground_1', 1), VehicleClaimAgent('ground_2', 2)
        self.exchange([a, near, close], 10.)
        # 1.4*90 = 126 is not 15 % below A's 140: no hand-over.
        self.decide(near, 10., (90, 0), targets)
        self.assertIsNone(near.claim)
        # 1.4*50 = 70 is: the vehicle claims it and A releases on hearing it.
        self.decide(close, 10., (50, 0), targets)
        self.assertEqual((close.claim, close.transfers_in), (0, 1))
        self.exchange([close], 11.)
        a.receive([dict(robot_id='ground_2', timestamp=11., **close.message())])
        self.decide(a, 11., (100, 0), targets)
        self.assertIsNone(a.claim)
        self.assertEqual((a.transfers_out, a.conflicts), (1, 0))

    def test_busy_vehicles_swap_when_the_graph_changes_costs(self):
        targets = np.array([(100, 0), (-100, 0)], float)
        a, b = VehicleClaimAgent('ground_0', 0), VehicleClaimAgent('ground_1', 1)
        for now in (0., 1., 2.):
            self.decide(a, now, (10, 0), targets)
            self.decide(b, now, (-10, 0), targets)
            self.exchange([a, b], now)
        self.assertEqual((a.claim, b.claim), (0, 1))
        for now in (10., 11., 12.):
            self.decide(a, now, (-80, 0), targets)
            self.exchange([a, b], now)
            self.decide(b, now, (80, 0), targets)
            self.exchange([a, b], now)
        self.assertEqual((a.claim, b.claim), (1, 0))
        self.assertEqual((a.switches, b.transfers_out), (1, 1))

    def test_departure_waits_for_part_of_the_graph(self):
        targets = np.array([(200, 0), (-300, 0)], float)
        agent = VehicleClaimAgent('ground_0', 0, depart_progress=.5)
        self.assertIsNone(self.decide(agent, 0., (0, 0), targets, progress=[0., 0.])['goal'])
        self.assertIsNone(self.decide(agent, 1., (0, 0), targets, progress=[.3, 0.])['goal'])
        decision = self.decide(agent, 2., (0, 0), targets, progress=[.5, 0.])
        self.assertTrue(decision['departed'])
        np.testing.assert_allclose(decision['goal'], (200, 0))
        other = VehicleClaimAgent('ground_1', 1, depart_progress=.5)
        self.assertTrue(self.decide(other, 0., (-200, 0), targets, connected=[False, True])['departed'])

    def test_more_targets_than_vehicles_are_visited_in_turn(self):
        targets = np.array([(50, 0), (120, 0), (300, 0)], float)
        agent, peer = VehicleClaimAgent('ground_0', 0), VehicleClaimAgent('ground_1', 1)
        self.assertEqual(self.decide(agent, 0., (0, 0), targets)['claim'], 0)
        decision = self.decide(agent, 1., (48, 0), targets)
        self.assertEqual((agent.served, decision['claim'], agent.arrivals), ({0}, 1, 1))
        peer.receive([dict(robot_id='ground_0', timestamp=1., **agent.message())])
        self.assertEqual(self.decide(peer, 1., (40, 0), targets)['claim'], 2)

    def test_stale_claims_expire_and_newer_messages_win(self):
        targets = np.array([(100, 0)], float)
        agent = VehicleClaimAgent('ground_1', 1, claim_timeout=10.)
        claim = dict(target=0, cost=999., since=0., rank=0)
        agent.receive([dict(robot_id='ground_0', timestamp=5., claim=claim, served=[])])
        agent.receive([dict(robot_id='ground_0', timestamp=3., claim=None, served=[])])
        self.assertEqual(agent.heard['ground_0'][1], claim)
        self.assertEqual(len(agent.claims_on(0, 15.)), 1)
        self.assertEqual(agent.claims_on(0, 15.1), [])


class SceneAndConfigTests(unittest.TestCase):
    def test_v1_scenes_unchanged(self):
        scene = json.dumps(generate_cograph_scene(62011), sort_keys=True).encode()
        self.assertEqual(hashlib.sha256(scene).hexdigest(),
                         '5cd6078244997556dda5957c8fb6ee64e6e2d5c3ed616a3ddb5e6e4d6d4b28ac')

    def test_single_task_scenes_are_covered_from_the_start(self):
        counts = set()
        for seed in range(50000, 50012):
            scene = generate_cograph_scene(seed, single_task=True)
            self.assertTrue(scene['single_task'])
            self.assertLessEqual(drones_to_cover(scene['start'], scene['goals']), 9)
            self.assertEqual(scene['task_drones'], drones_to_cover(scene['start'], scene['goals']))
            self.assertTrue(all(100. <= d <= 400. for d in scene['distances']))
            counts.add(scene['target_count'])
        self.assertGreater(len(counts), 3)

    def test_v2_config_contract_and_budget(self):
        v1, v2 = config('cograph_v1.json'), config('cograph_v2.json')
        self.assertEqual(reward_contract(v1), 'cograph_v1_multi_target_option_space_bridged')
        self.assertEqual(reward_contract(v2),
                         'cograph_v2_task_option_space_single_task_decentralized_vehicles_tree_budget_bridged')
        self.assertEqual((v2['ground_stationary'], v2['vehicle_planner'], v2['policy_architecture']),
                         (False, 'decentralized', 'cograph_v2'))
        for bad in (dict(ground_stationary=True), dict(scene_regime='far'), dict(vehicle_planner='tasks'),
                    dict(vehicle_depart_progress=0.), dict(vehicle_transfer_margin=1.)):
            with self.assertRaises(ValueError, msg=str(bad)):
                config('cograph_v2.json', **bad)
        for bad in (dict(ground_stationary=False), dict(vehicle_depart_progress=.3)):
            with self.assertRaises(ValueError, msg=str(bad)):
                config('cograph_v1.json', **bad)
        scene = generate_cograph_scene(50003, single_task=True)
        tree, total = episode_budget(v2, scene), episode_budget(v1, scene)
        self.assertLessEqual(tree, total)
        self.assertGreaterEqual(tree, v2['min_episode_steps'])


class TeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = CoGraphActorCritic().eval()
        cls.scene = generate_cograph_scene(50000, single_task=True)   # 8 targets

    def make_env(self, planner, **claims):
        return CoGraphExperiment(1, self.model, self.scene, vehicle_planner=planner, fusion_bridge_distance=15.,
                                 claim_options=claims or None)

    def run_env(self, planner, steps, **claims):
        rows = []
        with contextlib.redirect_stdout(io.StringIO()):
            env = self.make_env(planner, **claims)
            try:
                for frame in range(steps):
                    _, reward, _ = env.step(frame)
                    rows.append((np.array([env.obs[i]['self']['position'] for i in env.ids]), reward))
            finally:
                env.env.close()
        return env, rows

    def test_before_departure_drones_match_the_parked_team_and_claims_split(self):
        _, parked = self.run_env('parked', 5)
        env, free = self.run_env('decentralized', 5)
        for (a, r), (b, s) in zip(parked, free):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(r, s)
        claims = [agent.claim for agent in env.agents]
        self.assertNotIn(None, claims)
        self.assertEqual(len(set(claims)), 3)
        self.assertTrue(all(len(agent.heard) == 2 for agent in env.agents))
        self.assertFalse(any(agent.departed for agent in env.agents))
        self.assertEqual(env.controllers[0].ground_max_speed, 0.)

    def test_departed_vehicles_drive_and_the_cbf_assumes_motion(self):
        env, _ = self.run_env('decentralized', 20, depart_progress=.05)
        summary = env.summary()
        self.assertGreater(summary['vehicle_departures'], 0)
        self.assertGreater(summary['ground_distance'], 0.)
        self.assertEqual(env.controllers[0].ground_max_speed, 1.)

    def test_claims_arrive_through_drone_forwarding(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env = self.make_env('decentralized')
            env.env.close()
        message = dict(robot_id='ground_2', timestamp=5., claim=dict(target=1, cost=10., since=4., rank=2), served=[3])
        env.obs[env.vehicle_ids[0]] = dict(env.obs[env.vehicle_ids[0]], neighbors=[
            dict(type='drone', broadcast=dict(ground_missions={'ground_2': dict(broadcast=message)}))])
        received = env._received_vehicle_messages(0)
        self.assertEqual(received, [message])
        env.agents[0].receive(received)
        self.assertEqual((env.agents[0].heard['ground_2'][1]['target'], env.agents[0].served), (1, {3}))

    def test_graph_pool_splits_only_out_of_radio_range(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env = self.make_env('decentralized')
            env.env.close()
        positions = [np.array([0., 0., 0.]), np.array([0., 60., 0.]), np.array([0., 120., 0.])]
        env._split_pools(positions)   # a chain in range: still one pool
        self.assertEqual(len({id(v.graph_fusion) for v in env.vehicles}), 1)
        positions[2] = np.array([0., 300., 0.])
        env._split_pools(positions)
        self.assertIs(env.vehicles[0].graph_fusion, env.vehicles[1].graph_fusion)
        self.assertIsNot(env.vehicles[2].graph_fusion, env.vehicles[0].graph_fusion)
        env._split_pools(positions)
        self.assertEqual(env.pool_splits, 1)


class PolicyTests(unittest.TestCase):
    """The learned per-drone target choice (cograph_v2)."""
    @staticmethod
    def raw(tokens, decide=0.):
        x = torch.zeros(RAW_WIDTH)
        x[TOKENS_START:VALID_START] = tokens.ravel()
        x[VALID_START:VALID_START+MAX_TARGETS] = (tokens.abs().sum(-1) > 0).float()
        x[-1] = decide
        return x

    def test_warm_start_keeps_the_v8_velocity_policy_and_round_trips(self):
        torch.manual_seed(0)
        v8, model = RelayActorCritic(10, 2, 'full'), CoGraphV2ActorCritic()
        report = model.warm_start(v8.state_dict())
        self.assertEqual(report['partial'], ['gat_actor.1.weight', 'critic.1.weight'])
        self.assertTrue(all('targets' in key or key.startswith('selector') for key in report['fresh']))
        x = torch.randn(5, RAW_WIDTH)
        x[:, V8_WIDTH-16:V8_WIDTH] = (torch.rand(5, 16) > .5).float()
        x[:, VALID_START:VALID_START+MAX_TARGETS] = 1.
        with torch.no_grad():
            np.testing.assert_allclose(model.distribution(x).mean, v8.distribution(x[:, :V8_WIDTH]).mean, atol=1e-6)
        clone = ActorCritic.from_state_dict(model.state_dict())
        self.assertIsInstance(clone, CoGraphV2ActorCritic)

    def test_initial_choice_is_nearest_unconnected_and_ignores_the_choice_inputs(self):
        model = CoGraphV2ActorCritic()
        tokens = torch.zeros(MAX_TARGETS, TARGET_FEATURES)
        tokens[:4, 2] = torch.tensor([.3, .2, .5, .25])
        tokens[:4, 0] = 1e-3
        tokens[1, 3] = 1.                     # the nearest one is already connected
        plain = self.raw(tokens)
        chosen = tokens.clone()
        chosen[2, FOCUS_COLUMN] = 1.
        with torch.no_grad():
            logits = model.selection_logits(plain)
            self.assertEqual(int(logits.argmax()), 3)
            self.assertTrue(torch.equal(logits, model.selection_logits(self.raw(chosen, decide=1.))))
            self.assertTrue((logits[4:] < -1e8).all())
            probs = torch.softmax(logits, -1)
            self.assertGreater(float(probs[0]), .1)   # a nearby alternative is explored

    def test_choice_counts_only_at_decision_points(self):
        velocity = Independent(Normal(torch.zeros(2, 2), torch.ones(2, 2)), 1)
        logits = torch.tensor([[0., 1., -1e9], [0., 1., -1e9]])
        dist = SelectVelocity(velocity, logits, torch.tensor([0., 1.]))
        action = torch.tensor([[.1, .2, 1.], [.1, .2, 1.]])
        choice = torch.log_softmax(logits, -1)[:, 1]
        np.testing.assert_allclose(dist.log_prob(action), velocity.log_prob(action[:, :2])+torch.tensor([0., 1.])*choice)
        self.assertGreater(float(dist.entropy()[1]), float(dist.entropy()[0]))
        np.testing.assert_allclose(kl_divergence(dist, copy.deepcopy(dist)), 0., atol=1e-7)
        self.assertEqual(tuple(dist.sample().shape), (2, 3))

    def test_episode_feeds_ppo_with_exact_choice_probabilities(self):
        value = config('cograph_v2.json', steps=8, min_episode_steps=1, budget_extra=1)
        torch.manual_seed(1)
        model = make_model(value)
        rollout, summary = run_episode(model, value, 50001, fixed_schedule(value, 0), True)
        self.assertEqual(rollout[0][0].shape, (9, RAW_WIDTH))
        self.assertEqual(rollout[0][2].shape, (9, 3))
        decided = np.array([step[0][:, -1] for step in rollout])
        self.assertTrue(set(np.unique(decided)) <= {0., 1.})
        self.assertTrue((decided.sum(0) == 1).all())   # one choice per drone in 8 s
        self.assertIn('drone_distinct_targets', summary)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        _, ppo = update_batch(model, optimizer, [rollout], critic_epochs=1, value_normalization=True)
        self.assertLess(ppo['on_policy_logp_max_error'], 1e-5)
        self.assertTrue(np.isfinite(ppo['exact_kl']))

    def test_drones_choose_again_after_the_commitment(self):
        torch.manual_seed(0)
        scene = generate_cograph_scene(50000, single_task=True)
        with contextlib.redirect_stdout(io.StringIO()):
            env = CoGraphExperiment(1, CoGraphV2ActorCritic().eval(), scene, stochastic=True)
            try:
                for frame in range(4):
                    env.step(frame)
                self.assertEqual([c.navigator.decisions for c in env.controllers], [1]*9)
                env.controllers[0].navigator.choice_time -= COMMIT_S
                env.step(4)
            finally:
                env.env.close()
        self.assertEqual([c.navigator.decisions for c in env.controllers], [2]+[1]*8)
        self.assertTrue(all(c.navigator.chain.choice is not None for c in env.controllers))

    def test_train_smoke_with_v8_warm_start(self):
        with tempfile.TemporaryDirectory() as folder:
            v8 = RelayActorCritic(16, 2, 'full')
            checkpoint = Path(folder)/'v8.pt'
            torch.save(dict(model=v8.state_dict(), policy_architecture='gat_relay_attention_v8',
                            config=dict(relay_attention='full'), training_episodes=7), checkpoint)
            value = config('cograph_v2.json', steps=4, min_episode_steps=1, budget_extra=1, episodes=2,
                           episodes_per_batch=1, snapshot_interval=1, validation_interval=2, validation_seeds=[62000],
                           validation_samples=1, init_checkpoint=str(checkpoint))
            with contextlib.redirect_stdout(io.StringIO()):
                train(value, Path(folder)/'run', workers=1)
            progress = json.loads((Path(folder)/'run/progress.json').read_text())
            self.assertEqual(progress['status'], 'completed')
            provenance = json.loads((Path(folder)/'run/provenance.json').read_text())
            self.assertEqual(provenance['init_checkpoint']['source_architecture'], 'gat_relay_attention_v8')
            saved = torch.load(Path(folder)/'run/latest.pt', weights_only=True)
            self.assertEqual((saved['feature_version'], saved['policy_architecture']), (10, 'cograph_v2'))


if __name__ == '__main__':
    unittest.main()
