"""Check the real frozen v9 runtime's VTemp temporary-goal contract.

Run this file in its own process: it deliberately refuses to mix a mutable
working-tree simulator/drone/ground package with the frozen inference runtime.
The fixtures use an empty 2D sensor map only to isolate goal handover semantics.
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
OLD_VTEMP = REPOSITORY / "artifacts/vtemp-5km-400best-20261102/sources"
sys.path.insert(0, str(SOURCES))

import numpy as np
import torch
from ground import subgoal_manager as subgoal_module
from ground.ground_main import GroundRobotController
from ground.subgoal_manager import SubgoalManager
from simulator import mappo_experiment as me
from simulator import train_mappo as tm


def load_runner(name):
    path = Path(__file__).resolve().parents[2] / "simulator" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WildScenesVTempTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.runner = load_runner("validate_wildscenes_vtemp")
        cls.baseline = load_runner("validate_wildscenes_v9")
        cls.config = json.loads((RUN / "config.json").read_text())
        cls.schedule = tm.fixed_schedule(cls.config, cls.config["episodes"] - 1)

    def test_frozen_imports_and_identical_original_vtemp_manager(self):
        for name, module in tuple(sys.modules.items()):
            if name.split(".")[0] not in {"common", "ground", "drone", "simulator"}:
                continue
            path = getattr(module, "__file__", None)
            if path:
                self.assertTrue(Path(path).resolve().is_relative_to(SOURCES.resolve()),
                                f"Mutable runtime imported: {name}: {path}")
        frozen = Path(subgoal_module.__file__).read_bytes()
        original = (OLD_VTEMP / "ground/subgoal_manager.py").read_bytes()
        expected = "e36971e1a194210a972c38c9d393e7f4c1d6ee970fc077d885aa141fef4e2702"
        self.assertEqual(hashlib.sha256(frozen).hexdigest(), expected)
        self.assertEqual(frozen, original)

    def test_runner_changes_only_temporary_goal_options(self):
        case = dict(scene_seed=92000, start=[100., 500.], goal=[800., 500.])
        baseline = self.baseline.experiment_options(self.config, self.schedule, 1000., case)
        rolling = self.runner.experiment_options(self.config, self.schedule, 1000., case)
        self.assertEqual(rolling["subgoal_radius"], 200.)
        self.assertEqual(rolling["subgoal_arrival"], 25.)
        expected = dict(baseline, subgoal_radius=200., subgoal_arrival=25.)
        self.assertEqual(rolling, expected)
        self.assertIsNone(self.config["subgoal_radius"], "Do not mutate training config")

    def test_initial_target_arrival_switch_and_final_goal_are_distinct(self):
        manager = SubgoalManager((800., 500.), radius=200., arrival=25.)
        first, changed = manager.update((100., 500.), None, 0.)
        self.assertTrue(changed)
        np.testing.assert_allclose(first, [300., 500.])
        _, changed = manager.update((274.9, 500.), None, 100.)
        self.assertFalse(changed)
        second, changed = manager.update((275., 500.), None, 101.)
        self.assertTrue(changed)
        np.testing.assert_allclose(second, [475., 500.])
        np.testing.assert_array_equal(manager.goal, [800., 500.])
        self.assertEqual((manager.arrivals, manager.selected), (1, 2))
        # Near the final target, the 25 m intermediate-arrival rule must not
        # count a temporary success or start a fresh mission.
        last = SubgoalManager((800., 500.), radius=200., arrival=25.)
        last.update((650., 500.), None, 0.)
        _, changed = last.update((775., 500.), None, 125.)
        self.assertTrue(last.final)
        self.assertFalse(changed)
        self.assertEqual(last.arrivals, 0)

    def test_stall_reselects_without_any_reference_or_truth_map(self):
        manager = SubgoalManager((800., 500.), radius=200., arrival=25.)
        first, _ = manager.update((100., 500.), None, 0.)
        _, changed = manager.update((100., 500.), None, 179.)
        self.assertFalse(changed)
        second, changed = manager.update((100., 500.), None, 180.)
        self.assertTrue(changed)
        self.assertEqual((manager.failed, manager.selected), (1, 2))
        self.assertGreater(np.linalg.norm(second-first), manager.penalty_radius)
        self.assertLessEqual(np.linalg.norm(second-[100., 500.]), 200.+1e-8)
        np.testing.assert_array_equal(manager.goal, [800., 500.])

    def test_controller_broadcast_switches_with_temporary_goal(self):
        observation = dict(self=dict(position=np.array([100., 500., 0.]), orientation=0.),
                           neighbors=[], occupancy_map=np.zeros((16, 16), np.uint8),
                           map_resolution=.5, time_step=1., full_state=dict(time=0.))
        with contextlib.redirect_stdout(io.StringIO()):
            controller = GroundRobotController("ground_0", resolution=.5,
                subgoal_radius=200., subgoal_arrival=25., world_bounds=(0., 0., 1000., 1000.))
            controller.set_goal((800., 500.))
            first = controller.get_action(observation)
            np.testing.assert_allclose(first["broadcast"]["target"], [300., 500.])
            observation["self"]["position"] = np.array([275., 500., 0.])
            observation["full_state"]["time"] = 175.
            second = controller.get_action(observation)
        np.testing.assert_allclose(second["broadcast"]["target"], [475., 500.])
        np.testing.assert_array_equal(second["broadcast"]["target"], controller.subgoals.target)
        np.testing.assert_array_equal(controller.goal_position, controller.subgoals.target)
        self.assertEqual(controller.final_goal, (800., 500.))

    def test_v9_actor_receives_temporary_target_through_real_packets(self):
        # This exercises the requested checkpoint and actual controller packet
        # delivery. No target is injected into the navigator or its features.
        torch.manual_seed(123)
        checkpoint = torch.load(RUN / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["best_episode"], 576)
        self.assertEqual(checkpoint["reward_contract"], "uav_remaining_route_mapping_v9")
        model = tm.ActorCritic.from_state_dict(checkpoint["model"])
        model.eval()
        case = dict(scene_seed=92000, start=[100., 500.], goal=[800., 500.])
        options = self.runner.experiment_options(self.config, self.schedule, 1000., case)
        with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
            env = me.Experiment(case["scene_seed"], model, stochastic=False, **options)
            self.addCleanup(env.env.close)
            np.testing.assert_allclose(env.subgoals.target, [300., 500.])
            heard = 0
            for frame in range(8):
                before = [np.asarray(env.obs[r]["self"]["position"]).copy() for r in env.ids]
                env.step(frame)
                np.testing.assert_array_equal(env.ground_action["broadcast"]["target"], env.subgoals.target)
                for controller, position in zip(env.controllers, before):
                    task = controller.navigator.chain.task()
                    if task is None:
                        continue
                    heard += 1
                    target = np.asarray(task["target"][:2])
                    np.testing.assert_allclose(target, [300., 500.])
                    self.assertFalse(np.allclose(target, env.goal))
                    features = controller.navigator.last_sample[0]
                    np.testing.assert_allclose(features[4:6], (target-position[:2])/350., atol=1e-6)
                    ground = np.asarray(task["position"][:2])
                    np.testing.assert_allclose(features[8:10], (target-ground)/350., atol=1e-6)
                np.testing.assert_allclose(
                    [env.trace[-1]["route_target_x"], env.trace[-1]["route_target_y"]],
                    env.subgoals.target)
            self.assertGreater(heard, 0, "The learned actor never received a temporary-goal mission")
            np.testing.assert_array_equal(env.goal, [800., 500.])
            self.assertFalse(env.goal_reached)
            record = self.runner.position_record(env, len(env.trace))
            np.testing.assert_array_equal(record["temporary_goal"], env.subgoals.target)


if __name__ == "__main__":
    unittest.main(verbosity=2)
