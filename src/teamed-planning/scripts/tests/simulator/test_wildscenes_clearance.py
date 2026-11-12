"""Archive compatibility checks that do not require a learned-policy runtime."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import shutil
import tempfile
import unittest


def load(name):
    path = Path(__file__).resolve().parents[2] / "simulator" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load("wildscenes_clearance")


class ClearanceCompatibilityTests(unittest.TestCase):
    def runtime(self):
        def ground(*args, safety_distance=1.5, **kwargs):
            return SimpleNamespace(local_planner=SimpleNamespace(safety_distance=safety_distance,
                                                                escape_distance=.5),
                                   graph_fusion=SimpleNamespace(bridge_check=None))

        def drone(*args, graph_min_clearance=1.5, **kwargs):
            return SimpleNamespace(graph_builder=SimpleNamespace(min_clearance=graph_min_clearance,
                min_clearance_pixels=graph_min_clearance / 2., resolution=2.))

        runtime = SimpleNamespace(GroundRobotController=ground, DroneController=drone, BRIDGE_CLEARANCE_M=1.5)

        def experiment():
            g, d = runtime.GroundRobotController(safety_distance=1.5), runtime.DroneController(graph_min_clearance=1.5)
            ledger = SimpleNamespace(segment_clear=lambda a, b, clearance: clearance)
            g.graph_fusion.bridge_check = lambda a, b: ledger.segment_clear(a, b, runtime.BRIDGE_CLEARANCE_M)
            return SimpleNamespace(ground_controller=g, controllers=[d], ledger=ledger,
                first_action_clearance=g.local_planner.safety_distance)

        runtime.Experiment = experiment
        return runtime

    def test_override_precedes_initial_action_and_legacy_bridge_stays_bound(self):
        for distance in (1., 1.5):
            with self.subTest(distance=distance):
                runtime = self.runtime()
                original_ground, original_drone = runtime.GroundRobotController, runtime.DroneController
                with adapter.planning_runtime(runtime, distance):
                    experiment = runtime.Experiment()
                    actual = adapter.verify_experiment(experiment, distance)
                    self.assertEqual(experiment.first_action_clearance, distance)
                    self.assertEqual(actual["graph_clearance_pixels"], [distance / 2.])
                self.assertIs(runtime.GroundRobotController, original_ground)
                self.assertIs(runtime.DroneController, original_drone)
                self.assertEqual(runtime.BRIDGE_CLEARANCE_M, 1.5)
                self.assertEqual(experiment.ground_controller.graph_fusion.bridge_check((0, 0), (1, 1)), distance)

    def test_actual_bridge_mismatch_is_rejected(self):
        runtime = self.runtime()
        with adapter.planning_runtime(runtime):
            experiment = runtime.Experiment()
            experiment.ground_controller.graph_fusion.bridge_check = lambda a, b: experiment.ledger.segment_clear(a, b, 1.5)
            with self.assertRaisesRegex(RuntimeError, "bridge"):
                adapter.verify_experiment(experiment)

    def test_exception_restores_runtime(self):
        runtime = self.runtime()
        original = runtime.GroundRobotController
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with adapter.planning_runtime(runtime):
                raise RuntimeError("stop")
        self.assertIs(runtime.GroundRobotController, original)
        self.assertEqual(runtime.BRIDGE_CLEARANCE_M, 1.5)

    def test_old_and_new_constructor_options(self):
        old = self.runtime()
        options = dict(ground_safety_distance=1.5, seed=1)
        self.assertEqual(adapter.experiment_options(old, options, 1., .5), dict(seed=1))

        def new(seed, ground_safety_distance=1.5, ground_escape_distance=.5):
            pass

        old.Experiment = new
        self.assertEqual(adapter.experiment_options(old, options, 1., .5),
            dict(seed=1, ground_safety_distance=1., ground_escape_distance=.5))
        self.assertEqual(options["ground_safety_distance"], 1.5)

    def test_defaults_explicit_replay_invalid_values_and_source_guard(self):
        self.assertEqual(adapter.job_distances({}), (1., .5))
        self.assertEqual(adapter.job_distances(dict(ground_safety_distance=1.5)), (1.5, .5))
        for value in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                adapter.distances(value)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            adapter.verify_sources({})
        adapter.verify_sources(adapter.source_hashes())

    def test_standalone_snapshot_uses_adjacent_config_and_hashes_both_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            helper = directory / "wildscenes_clearance.py"
            shutil.copyfile(adapter.__file__, helper)
            shutil.copyfile(adapter.navigation_config_path(), directory / "navigation_config.py")
            spec = importlib.util.spec_from_file_location("_clearance_snapshot_test", helper)
            snapshot = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(snapshot)
            self.assertEqual(snapshot.job_distances({}), (1., .5))
            hashes = snapshot.source_hashes()
            self.assertEqual(set(hashes), {str(helper), str(directory / "navigation_config.py")})
            (directory / "navigation_config.py").write_text("DEFAULT_GROUND_CLEARANCE_M = 1.5\n")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                snapshot.verify_sources(hashes)


if __name__ == "__main__":
    unittest.main()
