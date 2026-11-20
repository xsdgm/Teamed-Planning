"""Exercise the UGV relay tap against the archived V9 transport and cache."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from world_model.frozen_navigation import V9_RUN
from simulator.ground_relay import GroundRelayObservation


ARCHIVE = Path(__file__).resolve().parents[5] / V9_RUN / "runtime_sources"


def archived_module(relative_path):
    spec = spec_from_file_location(Path(relative_path).stem, ARCHIVE / relative_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def neighbor(rid):
    return dict(id=rid, type="drone", relative_position=np.array([8., 0., 15.]),
                relative_velocity=np.array([1., 0., 0.]))


class GroundRelayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ARCHIVE.is_dir():
            raise unittest.SkipTest("Archived V9 runtime is unavailable")
        cls.Channel = archived_module("simulator/relay_channel.py").RelayChannel
        cls.cache = archived_module("drone/relay_cache.py")

    def make_observer(self, *, loss=0., jitter=0):
        simulation = SimpleNamespace(step_count=0, time=0., config=SimpleNamespace(dt=1.))
        experiment = SimpleNamespace(
            env=simulation, ground=0, ids=[1, 2], obs={0: {"neighbors": [neighbor(1)]}},
            controllers=[SimpleNamespace(drone_id="a"), SimpleNamespace(drone_id="b")],
            relay_channel=self.Channel(42, loss, jitter),
            ground_relay=self.cache.RelayManager("ground_0"),
            ground_controller=SimpleNamespace(robot_id="ground_0"),
        )
        relay = GroundRelayObservation(experiment, seed=42)
        self.addCleanup(relay.close)
        return experiment, relay

    def set_time(self, experiment, frame):
        experiment.env.step_count = frame
        experiment.env.time = float(frame)

    def test_multihop_state_enters_only_through_arrived_envelope(self):
        experiment, observer = self.make_observer()
        channel = experiment.relay_channel
        drone_a = self.cache.RelayManager("a")
        drone_b = self.cache.RelayManager("b")
        drone_b.originate(0., "drone_state", [110., 8., 15.], velocity=[3., 0., 0.])
        channel.transmit(drone_b.outgoing(0.), [1], 0)
        initial = observer.observe()
        self.assertEqual(initial["relay_tokens"], [])
        self.assertEqual([row["origin_id"] for row in initial["one_hop_neighbors"]], ["a"])
        initial["one_hop_neighbors"][0]["relative_position"][0] = 99.
        self.assertEqual(experiment.obs[0]["neighbors"][0]["relative_position"][0], 8.)
        self.assertTrue(initial["one_hop_neighbors"][0]["velocity_known"])

        self.set_time(experiment, 1)
        for envelope in channel.deliver(1, 1):
            drone_a.receive(envelope, 1.)
        self.assertEqual(observer.observe()["relay_tokens"], [])
        drone_a.originate(1., "drone_state", [5., 10., 15.])
        channel.transmit(drone_a.outgoing(1.), [], 1)
        self.assertEqual(observer.observe()["relay_tokens"], [])

        experiment.obs[0]["neighbors"] = []
        self.set_time(experiment, 4)
        result = observer.observe()
        self.assertEqual(result["one_hop_neighbors"], [])
        remote = {token["origin_id"]: token for token in result["relay_tokens"]}
        self.assertEqual({key: row["hop"] for key, row in remote.items()}, {"a": 1, "b": 2})
        self.assertEqual({key: row["age"] for key, row in remote.items()}, {"a": 3., "b": 4.})
        np.testing.assert_array_equal(remote["b"]["position"], [110., 8., 15.])
        np.testing.assert_array_equal(remote["b"]["velocity"], [3., 0., 0.])
        self.assertEqual(observer._receiver.cache["b"]["receive_stamp"], 2.)

        experiment.obs[0]["neighbors"] = [neighbor(1)]
        channel.transmit(drone_a.outgoing(4.), [], 4)
        self.set_time(experiment, 5)
        self.assertEqual(len(observer.observe()["relay_tokens"]), 2)
        self.assertEqual(observer._receiver.stats["duplicate"], 2)
        self.assertEqual(observer._receiver.cache["b"]["receive_stamp"], 2.)
        self.set_time(experiment, 31)
        self.assertEqual([row["origin_id"] for row in observer.observe()["relay_tokens"]], ["a"])
        self.set_time(experiment, 32)
        self.assertEqual(observer.observe()["relay_tokens"], [])

    def test_observer_keeps_native_delay_loss_rng_and_queue_unchanged(self):
        experiment, observer = self.make_observer(loss=.25, jitter=2)
        channel = experiment.relay_channel
        baseline = self.Channel(42, .25, 2)
        drone = self.cache.RelayManager("a")
        arrivals = set()
        for frame in range(50):
            self.set_time(experiment, frame)
            drone.originate(float(frame), "drone_state", [8., 0., 15.])
            envelope = drone.outgoing(float(frame))
            channel.transmit(envelope, [2], frame)
            baseline.transmit(envelope, [2], frame)
            self.assertEqual(channel.deliver(2, frame), baseline.deliver(2, frame))
            tokens = observer.observe()["relay_tokens"]
            if tokens:
                self.assertGreaterEqual(tokens[0]["age"], 1.)
                arrivals.add(observer._receiver.cache["a"]["seq"])

        self.assertGreater(len(arrivals), 0)
        self.assertLess(len(arrivals), 49)
        self.assertEqual(channel.stats, baseline.stats)
        self.assertEqual(channel.queues, baseline.queues)
        self.assertEqual(channel.rng.bit_generator.state, baseline.rng.bit_generator.state)
        observer.close()
        self.assertNotIn("transmit", vars(channel))
