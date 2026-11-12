"""Run directly in a fresh Python process; exercises the real frozen best576."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_var] = "1"

REPOSITORY = Path(__file__).resolve().parents[5]
RUN = REPOSITORY / "artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146"
sys.path.insert(0, str(RUN / "runtime_sources"))

import numpy as np
import torch
from simulator import mappo_experiment as me
from simulator import train_mappo as tm


def load_runner():
    path = Path(__file__).resolve().parents[2] / "simulator" / ("validate_wildscenes_legacy.py")
    spec = importlib.util.spec_from_file_location("_legacy_runtime_test_runner", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class LegacyRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.runner = load_runner()
        cls.adapter = cls.runner.load_scene_adapter()
        cls.directory = tempfile.TemporaryDirectory()
        cls.scene_path = Path(cls.directory.name) / "scene.json"
        cls.scene_path.write_text(json.dumps(dict(
            protocol="wildscenes-derived-legacy-2d-v1", simplified_scene=True,
            unknown_space="free_by_simulation_assumption", world_size_m=320., origin_xy=[0., 0.],
            geometry=dict(type="axis_aligned_rectangles",
                          rectangles=[[56., 160., 4., 6.], [110., 185., 10., 4.]]),
            metadata=dict(source=dict(kind="synthetic rectangles for native runtime testing")))))
        cls.scene = cls.adapter.LegacyObstacleScene.from_json(cls.scene_path)
        cls.hashes = cls.runner.frozen_hashes(RUN)
        cls.config = json.loads((RUN / "config.json").read_text())
        cls.checkpoint = torch.load(RUN / "best.pt", map_location="cpu", weights_only=True)
        cls.case = dict(id="synthetic_native_rectangles", scene_seed=92000,
                        start=[40., 160.], goal=[280., 160.], reference_length_m=240.)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.runner.verify_frozen_inputs(RUN, cls.hashes)
        finally:
            cls.directory.cleanup()

    def make_experiment(self, safety_distance=1.):
        torch.manual_seed(123)
        np.random.seed(123)
        model = tm.ActorCritic.from_state_dict(self.checkpoint["model"])
        model.eval()
        schedule = tm.fixed_schedule(self.config, self.config["episodes"] - 1)
        options = self.runner.clearance.experiment_options(me,
            self.runner.common.experiment_options(self.config, schedule, 320., self.case), safety_distance, .5)
        original_env, original_controller = me.MultiRobotEnv, me.DroneController
        with self.runner.native_runtime(me, self.scene, self.adapter), \
                self.runner.clearance.planning_runtime(me, safety_distance), \
                contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
            experiment = me.Experiment(92000, model, stochastic=False, **options)
            self.runner.clearance.verify_experiment(experiment, safety_distance)
        self.assertIs(me.MultiRobotEnv, original_env)
        self.assertIs(me.DroneController, original_controller)
        self.addCleanup(experiment.env.close)
        return experiment

    def test_frozen_components_default_to_one_metre_and_allow_explicit_replay(self):
        original_bridge = me.BRIDGE_CLEARANCE_M
        for distance in (1., 1.5):
            with self.subTest(distance=distance):
                experiment = self.make_experiment(distance)
                actual = self.runner.clearance.verify_experiment(experiment, distance)
                self.assertEqual(actual["ground_safety_distance"], distance)
                self.assertEqual(actual["ground_escape_distance"], .5)
                self.assertEqual(actual["graph_clearance_m"], [distance] * 3)
                self.assertEqual(actual["bridge_clearance_m"], distance)
                self.assertEqual(me.BRIDGE_CLEARANCE_M, original_bridge)
                self.step(experiment, 1)
                for controller in experiment.controllers:
                    self.assertEqual(controller.navigator.last_sample[0].shape, (1826,))

    def test_cli_default_and_resume_reject_different_planning_distance(self):
        output = Path(self.directory.name) / "cli-clearance"
        cases = Path(self.directory.name) / "cli-cases.json"
        cases.write_text(json.dumps(dict(world_size_m=320., cases=[self.case])))
        arguments = ["validate_wildscenes_legacy.py", "--scene", str(self.scene_path),
                     "--cases", str(cases), "--run", str(RUN), "--max-steps", "1",
                     "--output", str(output), "--deterministic"]
        with mock.patch.object(sys, "argv", arguments + ["--prepare-only"]):
            self.runner.main()
        prepared = json.loads((output / "manifest.json").read_text())
        self.assertEqual(prepared["status"], "prepared")
        self.assertFalse((output / "cases").exists())
        with mock.patch.object(sys, "argv", arguments + ["--resume"]), contextlib.redirect_stdout(io.StringIO()):
            self.runner.main()
        report = json.loads((output / "results.json").read_text())
        self.assertEqual(prepared["signature"], report["signature"])
        self.assertEqual(report["provenance"]["settings"]["ground_safety_distance"], 1.)
        self.assertEqual(report["records"][0]["summary"]["planning_clearances"]["bridge_clearance_m"], 1.)
        with mock.patch.object(sys, "argv", arguments + ["--resume", "--ground-safety-distance", "1.5"]):
            with self.assertRaisesRegex(ValueError, "identical inputs"):
                self.runner.main()

    def step(self, experiment, count=3):
        with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
            for frame in range(count):
                experiment.step(frame)

    def test_observations_match_native_rectangles_and_keep_sensor_shapes(self):
        experiment = self.make_experiment()
        native = me.MultiRobotEnv(experiment.env.config)
        for rectangle in self.scene.rectangles:
            native.add_rect_obstacle(*rectangle)
        for rid in experiment.ids + [experiment.ground]:
            obs = experiment.obs[rid]
            kind = "ground" if rid == experiment.ground else "drone"
            reference = native._generate_occupancy_map(obs["self"]["position"], kind)
            np.testing.assert_array_equal(obs["occupancy_map"], reference)
            self.assertEqual(obs["occupancy_map"].shape, (16, 16) if kind == "ground" else (12, 12))
            self.assertEqual(obs["map_resolution"], .5 if kind == "ground" else 2.)
            self.assertFalse(any(key.startswith("mapping_") for key in obs))
        self.assertEqual(len(experiment.env.obstacles), 2)
        self.assertTrue(all(obstacle.type == "rect" for obstacle in experiment.env.obstacles))
        native.close()

    def test_best576_runs_with_native_actor_inputs_and_vtemp(self):
        experiment = self.make_experiment()
        self.step(experiment)
        self.assertEqual(experiment.subgoals.radius, 200.)
        self.assertEqual(experiment.subgoals.arrival, 25.)
        self.assertEqual(experiment.subgoals.selected, 1)
        for controller in experiment.controllers:
            self.assertEqual(controller.navigator.last_sample[0].shape, (1826,))
            self.assertTrue(np.isfinite(controller.navigator.last_sample[0]).all())
            self.assertEqual(controller.graph_builder.resolution, 2.)
            self.assertEqual(controller._last_semantic_map.shape, (12, 12))
        np.testing.assert_array_equal(experiment.goal, self.case["goal"])

    def test_summary_restores_rectangles_and_uses_scene_truth(self):
        experiment = self.make_experiment()
        self.step(experiment)
        obstacles = experiment.env.obstacles
        summary = self.runner.legacy_summary(experiment, self.scene)
        self.assertIs(experiment.env.obstacles, obstacles)
        self.assertGreater(summary["geometry_evaluation_points"], 0)
        self.assertGreaterEqual(summary["brier_error"], 0.)
        self.assertLessEqual(summary["brier_error"], 1.)
        self.assertNotIn("raster_known_evaluation_points", summary)
        seen = experiment.ledger.seen
        probability = 1 / (1 + np.exp(-experiment.ledger.log_odds[seen]))
        truth = self.scene.truth(experiment.ledger.xy[seen])
        self.assertAlmostEqual(summary["brier_error"], float(np.mean((probability - truth) ** 2)))

    def test_summary_and_runtime_restore_on_errors(self):
        obstacles = [object()]

        def fail():
            raise RuntimeError("summary failure")

        fake = SimpleNamespace(env=SimpleNamespace(obstacles=obstacles), summary=fail)
        with self.assertRaisesRegex(RuntimeError, "summary failure"):
            self.runner.legacy_summary(fake, self.scene)
        self.assertIs(fake.env.obstacles, obstacles)
        original = me.MultiRobotEnv
        with self.assertRaisesRegex(RuntimeError, "body failure"):
            with self.runner.native_runtime(me, self.scene, self.adapter):
                raise RuntimeError("body failure")
        self.assertIs(me.MultiRobotEnv, original)

    def test_evaluator_records_simplified_scope_and_checks_frozen_inputs(self):
        output = Path(self.directory.name) / "evaluation"
        job = dict(id="synthetic_native_rectangles_r0", case=self.case, sample=0, sample_seed=123,
                   run=str(RUN), scene_path=str(self.scene_path), output=str(output),
                   frozen_hashes=self.hashes, signature="unit-test", stochastic=False, max_steps=3)
        record = self.runner.evaluate_job(job)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["summary"]["episode_steps"], 3)
        saved = json.loads((output / "cases" / (job["id"] + ".json")).read_text())
        self.assertTrue(saved["simplified_scene"])
        self.assertIn("free", saved["unknown_space_assumption"])
        self.assertEqual(saved["summary"]["inherited_reference_length_m"], 240.)
        self.assertNotIn("actual_route_over_reference", saved["summary"])
        self.assertFalse(saved["evaluation_overrides"]["fine_mapping"])
        self.assertTrue(saved["trace"])
        self.assertEqual(saved["summary"]["safety_clearance_m"], 1.)
        self.assertEqual(saved["summary"]["planning_clearances"]["graph_clearance_m"], [1.] * 3)
        self.runner.verify_frozen_inputs(RUN, self.hashes)

    def test_source_map_pairing_rejects_same_size_different_map(self):
        scene = SimpleNamespace(metadata={"source": {"map_sha256": "source-a"}})
        manifest = dict(metadata=dict(map_sha256="source-b"), world_size_m=320.)
        with self.assertRaisesRegex(ValueError, "source map hashes differ"):
            self.runner.paired_source_hash(manifest, scene, self.scene_path)
        manifest["metadata"]["map_sha256"] = "source-a"
        self.assertEqual(self.runner.paired_source_hash(manifest, scene, self.scene_path), "source-a")

    def test_scene_and_case_files_are_checked_in_addition_to_frozen_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            scene, cases = Path(directory) / "scene.json", Path(directory) / "cases.json"
            scene.write_text("original scene")
            cases.write_text("original cases")
            job = dict(run=str(RUN), frozen_hashes=self.hashes, scene_path=str(scene),
                       cases_path=str(cases), scene_sha256=self.runner.digest(scene),
                       cases_sha256=self.runner.digest(cases))
            self.runner.verify_job_inputs(job)
            scene.write_text("changed scene")
            with self.assertRaisesRegex(RuntimeError, "scene input changed"):
                self.runner.verify_job_inputs(job)
            scene.write_text("original scene")
            cases.write_text("changed cases")
            with self.assertRaisesRegex(RuntimeError, "cases input changed"):
                self.runner.verify_job_inputs(job)


if __name__ == "__main__":
    unittest.main()
