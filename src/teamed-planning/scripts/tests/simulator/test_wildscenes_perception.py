"""Exercise local mapping with best576 in the frozen v9 inference runtime.

Run this file in its own Python process.  The only mutable code loaded is the
explicit WildScenes sensor adapter, runner, and local-mapping wrapper.  No ROS
nodes, training, or long evaluation runs are started by these tests.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

REPOSITORY = Path(__file__).resolve().parents[5]
RUN = REPOSITORY / "artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146"
SOURCES = RUN / "runtime_sources"
sys.path.insert(0, str(SOURCES))

import numpy as np
import torch
from drone.velocity_mappo import SENSOR_SIZE, STATE_FEATURES
from simulator import mappo_experiment as me
from simulator import train_mappo as tm


def load_runner():
    path = Path(__file__).resolve().parents[2] / "simulator" / ("validate_wildscenes_vtemp.py")
    spec = importlib.util.spec_from_file_location("_perception_test_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class WildScenesPerceptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.runner = load_runner()
        cls.adapter = cls.runner.load_adapter()
        cls.config = json.loads((RUN / "config.json").read_text())
        cls.schedule = tm.fixed_schedule(cls.config, cls.config["episodes"] - 1)
        cls.checkpoint = torch.load(RUN / "best.pt", map_location="cpu", weights_only=True)
        cls.asset_hashes = {
            path: file_digest(path)
            for path in [RUN / "best.pt", RUN / "config.json", *sorted(SOURCES.rglob("*.py"))]
        }
        # A 4m-wide, horizontal corridor.  The central drone starts above it;
        # the other two retain their original 18m initial formation offsets.
        cls.world = 320.
        cls.case = dict(scene_seed=92000, start=[40., 160.], goal=[280., 160.])
        size = int(cls.world / .25)
        occupied = np.ones((size, size), dtype=np.uint8)
        occupied[int(158. / .25):int(162. / .25), :] = 0
        cls.forest = cls.adapter.RasterForestMap(
            occupied, np.ones_like(occupied, dtype=bool), .25)

    def make_experiment(self, resolution):
        torch.manual_seed(123)
        np.random.seed(123)
        model = tm.ActorCritic.from_state_dict(self.checkpoint["model"])
        model.eval()
        options = self.runner.experiment_options(
            self.config, self.schedule, self.world, self.case)
        with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
            with self.runner.raster_runtime(me, self.forest, resolution):
                env = me.Experiment(self.case["scene_seed"], model, stochastic=False, **options)
        self.addCleanup(env.env.close)
        return env

    def step(self, env, frame):
        with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
            env.step(frame)

    def sensor_features(self, controller):
        features = controller.navigator.last_sample[0]
        start = STATE_FEATURES[controller.navigator.model.input_version]
        return features[start:start + 2 * SENSOR_SIZE ** 2].copy()

    def test_legacy_keeps_original_controller_and_all_policy_observations(self):
        original_env, original_controller = me.MultiRobotEnv, me.DroneController
        coarse = self.make_experiment(2.)
        fine = self.make_experiment(.5)
        self.assertIs(me.MultiRobotEnv, original_env)
        self.assertIs(me.DroneController, original_controller)
        for old, new in zip(coarse.controllers, fine.controllers):
            self.assertIs(type(old), original_controller)
            self.assertIsNot(type(new), original_controller)
        for rid in coarse.ids:
            old, new = coarse.obs[rid], fine.obs[rid]
            self.assertEqual(old["occupancy_map"].shape, (12, 12))
            self.assertEqual(new["map_resolution"], 2.)
            np.testing.assert_array_equal(old["occupancy_map"], new["occupancy_map"])
            self.assertNotIn("mapping_occupancy_map", old)
            self.assertEqual(new["mapping_occupancy_map"].shape, (48, 48))
            self.assertEqual(new["mapping_resolution"], .5)
            self.assertEqual(new["mapping_axes"], "yx")
            self.assertEqual(new["mapping_known_mask"].shape, (48, 48))
            self.assertEqual(new["mapping_known_mask"].dtype, np.bool_)
        old, new = coarse.obs[coarse.ground], fine.obs[fine.ground]
        self.assertEqual(new["occupancy_map"].shape, (16, 16))
        self.assertEqual(new["map_resolution"], .5)
        np.testing.assert_array_equal(old["occupancy_map"], new["occupancy_map"])
        self.assertNotIn("mapping_occupancy_map", new)

    def test_fine_mapping_builds_safe_corridor_graph_without_changing_actor_sensor(self):
        coarse, fine = self.make_experiment(2.), self.make_experiment(.5)
        self.step(coarse, 0)
        self.step(fine, 0)
        for old, new in zip(coarse.controllers, fine.controllers):
            np.testing.assert_array_equal(self.sensor_features(old), self.sensor_features(new))
            self.assertTrue(np.isfinite(new.navigator.last_sample[0]).all())
            self.assertEqual(old.navigator.last_sample[0].shape, new.navigator.last_sample[0].shape)
            self.assertEqual(new.resolution, 2., "Policy resolution must stay frozen")
            self.assertEqual(new.graph_builder.resolution, .5)
            self.assertEqual(new.semantic_processor.resolution, .5)
            self.assertEqual(new.graph_builder.min_clearance, 1.5)
            self.assertEqual(new._last_semantic_map.shape, (48, 48))
        old_graph = coarse.controllers[1].current_graph.graph
        graph = fine.controllers[1].current_graph.graph
        self.assertGreater(graph.number_of_nodes(), old_graph.number_of_nodes())
        self.assertGreater(graph.number_of_nodes(), 0)
        self.assertGreater(graph.number_of_edges(), 0)
        for _, data in graph.nodes(data=True):
            self.assertTrue(self.forest.point_clear(data["position"][:2], 1.5))
        for a, b in graph.edges:
            self.assertTrue(self.forest.segment_clear(
                graph.nodes[a]["position"][:2], graph.nodes[b]["position"][:2], 1.5))

    def test_real_delivered_audit_stays_coarse_and_vtemp_stays_active(self):
        env = self.make_experiment(.5)
        delivered = []
        original_receive = env.ledger.receive

        def capture(source, frame, position, occupancy, resolution, valid=True):
            delivered.append((source, frame, np.asarray(position).copy(),
                              np.asarray(occupancy).copy(), resolution, valid))
            return original_receive(source, frame, position, occupancy, resolution, valid)

        env.ledger.receive = capture
        for frame in range(4):
            self.step(env, frame)
            self.assertEqual(env.ledger.resolution, 2.)
            np.testing.assert_array_equal(env.goal, self.case["goal"])
            np.testing.assert_array_equal(env.ground_action["broadcast"]["target"], env.subgoals.target)
            self.assertEqual(env.subgoals.radius, 200.)
            self.assertEqual(env.subgoals.arrival, 25.)
            self.assertFalse(env.subgoals.final)
            self.assertFalse(env.goal_reached)
            for rid in env.ids:
                self.assertEqual(env.obs[rid]["occupancy_map"].shape, (12, 12))
                self.assertEqual(env.obs[rid]["mapping_occupancy_map"].shape, (48, 48))
            self.assertEqual(env.obs[env.ground]["occupancy_map"].shape, (16, 16))
        self.assertTrue(delivered, "No actual graph packet reached the reward ledger")
        for _, _, position, occupancy, resolution, valid in delivered:
            self.assertTrue(valid)
            self.assertEqual(resolution, 2.)
            self.assertEqual(occupancy.shape, (12, 12))
            np.testing.assert_array_equal(
                occupancy, self.forest.local_occupancy(position, 12, 2., 24.))
        self.assertAlmostEqual(env.summary()["coverage_area"], float(env.ledger.seen.sum() * 4))

    def test_runner_restores_frozen_classes_after_failure_and_distinguishes_contracts(self):
        original_env, original_controller = me.MultiRobotEnv, me.DroneController
        with self.assertRaisesRegex(RuntimeError, "intentional test interruption"):
            with self.runner.raster_runtime(me, self.forest, .5):
                self.assertIsNot(me.DroneController, original_controller)
                raise RuntimeError("intentional test interruption")
        self.assertIs(me.MultiRobotEnv, original_env)
        self.assertIs(me.DroneController, original_controller)
        fine, coarse = self.runner.perception_contract(.5), self.runner.perception_contract(2.)
        self.assertNotEqual(self.runner.object_digest(fine), self.runner.object_digest(coarse))
        self.assertEqual(fine["drone_footprint_m"], [24., 24.])
        self.assertEqual(fine["actor_shape"], [12, 12])
        self.assertEqual(fine["evidence_resolution_m"], 2.)
        for bad in (0., -.5, float("nan"), .3):
            with self.assertRaises(ValueError):
                self.runner.perception_contract(bad)

    def test_z_checkpoint_and_runtime_sources_are_unchanged_and_frozen(self):
        self.assertEqual(self.checkpoint["best_episode"], 576)
        self.assertEqual(self.checkpoint["reward_contract"], "uav_remaining_route_mapping_v9")
        self.assertEqual(self.asset_hashes[RUN / "best.pt"],
                         "7ac3aedcc3876902773b2241f60e839eb3e3b5630ddeb3c8be660421f2c24198")
        for path, expected in self.asset_hashes.items():
            self.assertEqual(file_digest(path), expected, f"Frozen asset changed: {path}")
        for name, module in tuple(sys.modules.items()):
            if name.split(".")[0] not in {"common", "drone", "ground", "simulator"}:
                continue
            path = getattr(module, "__file__", None)
            if path:
                self.assertTrue(Path(path).resolve().is_relative_to(SOURCES.resolve()), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
