"""v8 relay protocol, age/hop attention with absolute gate, transport causality and training."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drone.relay_cache import RelayManager, make_entry
from drone.relay_attention_mappo import (RelayActorCritic, RelayAttention, RAW_WIDTH, RELAY_TOKENS,
    RELAY_FEATURES, AGE, HOP, VEHICLE, SIGMA, MODES)
from drone.cooperative_mappo import CooperativeActorCritic, CooperativeNavigator, RAW_WIDTH as V7_WIDTH
from drone.mappo import ActorCritic
from simulator.relay_channel import RelayChannel
from simulator.mappo_experiment import Experiment
from simulator.train_mappo import validate_config, train, run_episode, fixed_schedule, make_model, feature_version


def config():
    return json.loads((Path(__file__).resolve().parents[2]/'config/algorithm/mappo_training_v8.json').read_text())


def state(device, now, position=(0., 0., 15.), boot=0):
    manager = RelayManager(device, boot_id=boot)
    manager.originate(now, 'drone_state', position, (1., 0., 0.))
    return manager


class RelayProtocolTests(unittest.TestCase):
    def test_source_time_kept_and_hops_increment_across_three_relays(self):
        a, b, c, d = (RelayManager(k) for k in 'abcd')
        a.originate(10., 'drone_state', (1., 2., 15.), (0., 1., 0.))
        b.receive(a.outgoing(10.), 11.)
        c.receive(b.outgoing(11.), 12.)
        d.receive(c.outgoing(12.), 14.)
        for manager, hops in ((b, 1), (c, 2), (d, 3)):
            entry = manager.cache['a']
            self.assertEqual((entry['source_stamp'], entry['hop_count'], entry['expires_at']), (10., hops, 40.))
        self.assertEqual(d.cache['a']['sender_id'], 'c')
        row = d.tokens(14.)[0]
        self.assertEqual((row['age'], row['hop']), (4., 3))
        # The origin's own record never appears among its own tokens.
        a.receive(d.outgoing(14.), 15.)
        self.assertNotIn('a', a.cache)

    def test_old_sequence_never_overwrites_newer_and_duplicates_count_once(self):
        a, b = RelayManager('a'), RelayManager('b')
        a.originate(1., 'drone_state', (1., 0., 15.))
        first = a.outgoing(1.)
        a.originate(2., 'drone_state', (2., 0., 15.))
        second = a.outgoing(2.)
        b.receive(second, 3.)
        b.receive(first, 3.)  # reordered arrival
        self.assertEqual(b.cache['a']['seq'], 1)
        self.assertEqual(b.stats['stale'], 1)
        b.receive(second, 4.)
        self.assertEqual((b.stats['accepted'], b.stats['duplicate']), (1, 1))
        self.assertEqual(b.cache['a']['receive_stamp'], 3.)

    def test_duplicate_over_shorter_path_keeps_measurement_time(self):
        a, relay, b = RelayManager('a'), RelayManager('r'), RelayManager('b')
        a.originate(5., 'drone_state', (0., 0., 15.))
        packet = a.outgoing(5.)
        relay.receive(packet, 6.)
        b.receive(relay.outgoing(6.), 7.)
        self.assertEqual(b.cache['a']['hop_count'], 2)
        b.receive(packet, 8.)
        self.assertEqual((b.cache['a']['hop_count'], b.cache['a']['source_stamp']), (1, 5.))
        self.assertEqual(b.stats['better_path'], 1)

    def test_restart_and_clock_errors(self):
        b = RelayManager('b')
        b.receive(state('a', 50.).outgoing(50.), 51.)
        b.receive(state('a', 40., boot=7).outgoing(40.), 52.)  # new boot, older time
        self.assertEqual(b.cache['a']['boot_id'], 0)
        b.receive(state('a', 53., boot=7).outgoing(53.), 53.)  # restarted, seq reset to 0
        self.assertEqual((b.cache['a']['boot_id'], b.cache['a']['seq']), (7, 0))
        b.receive(state('a', 90., boot=7).outgoing(90.), 60.)  # clock ahead beyond tolerance
        self.assertEqual(b.stats['future'], 1)
        self.assertEqual(b.cache['a']['source_stamp'], 53.)
        # Unknown source time: conservative receive age plus a per-hop delay bound.
        entry = make_entry('x', 0, 0, None, 'drone_state', (0., 0., 15.))
        b.receive([entry], 60.)
        age, unknown = b.age(b.cache['x'], 63.)
        self.assertTrue(unknown)
        self.assertEqual(age, 3.+1*b.per_hop_delay_bound)

    def test_expiry_is_not_refreshed_by_forwarding_and_goal_epoch_is_monotone(self):
        a, relay, b = RelayManager('a'), RelayManager('r'), RelayManager('b')
        a.originate(0., 'drone_state', (0., 0., 15.))
        relay.receive(a.outgoing(0.), 29.)
        self.assertIn('a', relay.cache)
        b.receive(relay.outgoing(29.5), 31.)  # forwarded copy already expired at 30 s
        self.assertNotIn('a', b.cache)
        self.assertEqual(relay.tokens(30.5), [])
        vehicle = RelayManager('v')
        vehicle.originate(1., 'vehicle_state', (0., 0., 0.), goal=(9., 9.), goal_epoch=2)
        b.receive(vehicle.outgoing(1.), 2.)
        vehicle.originate(3., 'vehicle_state', (0., 0., 0.), goal=(1., 1.), goal_epoch=1)
        b.receive(vehicle.outgoing(3.), 4.)
        self.assertEqual(b.cache['v']['state']['goal'], (9., 9.))
        self.assertEqual(b.stats['rejected_epoch'], 1)

    def test_channel_is_causal_lossy_and_reorders_only_with_jitter(self):
        channel = RelayChannel(3, loss=0., delay_jitter=0)
        channel.transmit([make_entry('a', 0, 0, 0., 'drone_state', (0, 0, 15))], ['b'], 5)
        self.assertEqual(channel.deliver('b', 5), [])
        self.assertEqual(len(channel.deliver('b', 6)), 1)
        lossy = RelayChannel(3, loss=.5, delay_jitter=3)
        for frame in range(200):
            lossy.transmit([make_entry('a', 0, frame, frame, 'drone_state', (0, 0, 15))], ['b'], frame)
        arrivals = [p[0]['seq'] for f in range(205) for p in lossy.deliver('b', f)]
        self.assertTrue(.35 < lossy.stats['lost']/200 < .65)
        self.assertNotEqual(arrivals, sorted(arrivals))
        periodic = RelayChannel(3, period=3)
        self.assertEqual(sum(periodic.due('x', f) for f in range(30)), 10)


class RelayAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)

    def tokens(self, ages, hops=None, vehicle=()):
        x = torch.zeros(RELAY_TOKENS, RELAY_FEATURES)
        valid = torch.zeros(RELAY_TOKENS, dtype=torch.bool)
        for i, age in enumerate(ages):
            x[i, :3] = torch.tensor([.1*i, -.05*i, 0.])
            x[i, AGE], x[i, HOP] = age, (hops or [1]*len(ages))[i]
            x[i, VEHICLE if i in vehicle else VEHICLE-1] = 1.
            x[i, SIGMA] = .5/95
            valid[i] = True
        return x, valid

    def test_empty_cache_is_zero_and_finite(self):
        attention = RelayAttention(386)
        x, valid = self.tokens([])
        x[:] = 1e3  # padding content must not matter
        context, stats = attention(torch.randn(386), x, valid)
        self.assertTrue(torch.equal(context, torch.zeros_like(context)))
        self.assertEqual(stats.tolist(), [0., 0., 0., 1.])

    def test_gate_is_absolute_and_not_renormalized(self):
        attention = RelayAttention(386)
        softmax_only = RelayAttention(386, 'softmax')
        softmax_only.load_state_dict({k: v for k, v in attention.state_dict().items() if k != 'mode'}, strict=False)
        query = torch.randn(386)
        reliability = []
        for ages in ([1.], [25.], [25., 28.]):
            x, valid = self.tokens(ages)
            gated, stats = attention(query, x, valid)
            relative, plain = softmax_only(query, x, valid)
            self.assertEqual(float(plain[1]), 1.)
            if len(ages) == 1:
                # Softmax weight is 1 whatever the age; only the gate shrinks it.
                torch.testing.assert_close(gated, relative*stats[1])
            reliability.append(float(stats[1]))
        self.assertLess(reliability[1], .1*reliability[0])
        self.assertLess(reliability[2], .1*reliability[0])  # all stale stays small
        # More hops: lower reliability at equal age.
        x, valid = self.tokens([2., 2.], hops=[1, 6])
        self.assertLess(float(attention(query, x[1:2].expand(RELAY_TOKENS, -1), valid)[1][2]),
                        float(attention(query, x[0:1].expand(RELAY_TOKENS, -1), valid)[1][2]))

    def test_token_order_and_padding_do_not_matter(self):
        attention = RelayAttention(386)
        query = torch.randn(386)
        x, valid = self.tokens([1., 3., 7.], hops=[1, 2, 3], vehicle=(1,))
        reordered = x.clone()
        reordered[[0, 1, 2]] = x[[2, 0, 1]]
        reordered[3:] = 77.
        a = attention(query, x, valid)
        b = attention(query, reordered, valid)
        torch.testing.assert_close(a[0], b[0])
        torch.testing.assert_close(a[1], b[1])

    def test_modes_survive_checkpoints_and_v7_still_loads(self):
        for mode in MODES:
            model = ActorCritic.from_state_dict(RelayActorCritic(10, 2, mode).state_dict())
            self.assertIsInstance(model, RelayActorCritic)
            self.assertEqual(model.relay_attention, mode)
        v7 = ActorCritic.from_state_dict(CooperativeActorCritic().state_dict())
        self.assertNotIsInstance(v7, RelayActorCritic)

    def test_features_ignore_device_names_and_require_the_cache(self):
        model = RelayActorCritic().eval()
        def features(name, origin):
            nav = CooperativeNavigator(name, total_drones=3, model=model)
            nav.cbf_margin_source = lambda: 2.
            peers = {origin: [20., 0., 15.]}
            row = dict(origin_id=origin, payload_type='drone_state', position=np.array([20., 0., 15.]),
                       velocity=np.zeros(3), goal=None, age=2., hop=1, time_unknown=False, sigma=.5)
            remote = dict(row, origin_id='far_'+origin, position=np.array([150., 10., 15.]), age=4., hop=2)
            obs = dict(time=0., graph=None, nearby_drones={}, ground_robots={}, occupancy_map=np.zeros((12, 12)),
                       cooperation=dict(peers=peers, velocities={origin: np.zeros(3)}, ground=None,
                                        required=(origin,)), relay=[row, remote])
            return nav.encode([0., 0., 15.], obs)[0]
        x = features('device', 'peer')
        np.testing.assert_array_equal(x, features('other_name', 'renamed_peer'))
        self.assertEqual(x.shape, (RAW_WIDTH,))
        tokens = x[V7_WIDTH:V7_WIDTH+RELAY_TOKENS*RELAY_FEATURES].reshape(RELAY_TOKENS, RELAY_FEATURES)
        self.assertEqual(tokens[:2, 12].tolist(), [1., 0.])  # sensed one hop vs remote
        self.assertEqual(x[-RELAY_TOKENS:].sum(), 2.)
        nav = CooperativeNavigator('d', total_drones=3, model=model)
        nav.cbf_margin_source = lambda: 2.
        with self.assertRaises(KeyError):
            nav.encode([0., 0., 15.], dict(time=0., graph=None, nearby_drones={}, ground_robots={},
                occupancy_map=np.zeros((12, 12)), cooperation=dict(peers={}, velocities={}, ground=None, required=())))


class RelayIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)

    def test_remote_state_reaches_far_drone_over_two_hops(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env = Experiment(20000, RelayActorCritic(), communication_mode='backbone', safety_filter='cbf')
            # drone_1 beside the vehicle, drone_2 in the middle, drone_0 out of
            # radio range of both drone_1 and the vehicle.
            for drone, p in zip(env.env.drones, ([115, 115, 15], [5, 5, 15], [60, 60, 15])):
                drone.position = np.array(p, float)
            env.obs = env.env._get_observations()
            far = env.ids[0]
            self.assertEqual({n['id'] for n in env.obs[far]['neighbors']}, {env.ids[2]})
            for frame in range(6):
                _, _, metrics = env.step(frame)
                self.assertEqual(metrics['backbone_connected'], 1.)
        tokens = {row['origin_id']: row for row in env.controllers[0].relay.tokens(5.)}
        self.assertEqual(tokens['drone_1']['hop'], 2)
        self.assertEqual(tokens['ground_0']['hop'], 2)
        self.assertGreaterEqual(tokens['drone_1']['age'], 2.)
        self.assertGreater(metrics['relay_remote_fraction'], 0.)
        self.assertEqual(metrics['relay_max_hop'], 2.)

    def test_ideal_relay_degenerates_to_one_step_per_hop(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env = Experiment(20000, RelayActorCritic(), communication_mode='backbone', safety_filter='cbf')
            for frame in range(3):
                env.step(frame)
        for controller in env.controllers:
            for row in controller.relay.tokens(2.):  # last decision time
                self.assertEqual(row['age'], float(row['hop']))
        self.assertEqual(env.relay_channel.stats['lost'], 0)


class RelayConfigTests(unittest.TestCase):
    def test_version_contract_and_relay_fields_are_explicit(self):
        c = validate_config(config())
        self.assertEqual(feature_version(c), 8)
        self.assertIsInstance(make_model(c), RelayActorCritic)
        for changes in (dict(drop=.1), dict(communication_mode='chain'), dict(relay_loss=1.),
                        dict(relay_delay_jitter=-1), dict(relay_period=0), dict(relay_capacity=1.5),
                        dict(relay_attention='mean')):
            with self.assertRaises(ValueError):
                validate_config(dict(c, **changes))
        v7 = json.loads((Path(__file__).resolve().parents[2]/'config/algorithm/mappo_training_v7.json').read_text())
        self.assertEqual(feature_version(validate_config(v7)), 7)
        with self.assertRaises(ValueError):
            validate_config(dict(v7, relay_loss=.1))
        with self.assertRaises(ValueError):
            Experiment(1, CooperativeActorCritic(), communication_mode='backbone', relay_loss=.1)

    def test_real_training_checkpoint_validation_and_resume(self):
        c = config()
        c.update(episodes=2, steps=5, episodes_per_batch=2, validation_interval=2, snapshot_interval=2,
                 validation_seeds=[60000], validation_samples=1, ppo_epochs=1, critic_epochs=1,
                 relay_attention='hop')
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            train(c, directory, workers=1)
            path = Path(directory)
            saved = torch.load(path/'latest.pt', weights_only=True)
            self.assertEqual(saved['feature_version'], 8)
            model = ActorCritic.from_state_dict(saved['model'])
            self.assertIsInstance(model, RelayActorCritic)
            self.assertEqual(model.relay_attention, 'hop')
            episode, summary = run_episode(model, validate_config(c), 50000, fixed_schedule(c, 0), True)
            self.assertEqual(episode[0][0].shape, (3, RAW_WIDTH))
            self.assertTrue(all(np.isfinite(row[0]).all() for row in episode))
            self.assertIn('relay_packets_delivered', summary)
            validation = json.loads((path/'validation.jsonl').read_text().splitlines()[-1])
            self.assertIn('relay_age_s', validation['means'])
            train(c, directory, resume=True, workers=1)


if __name__ == '__main__':
    unittest.main()
