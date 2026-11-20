"""Keep legacy and fine-mapping result files separate when rendering reports."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def load_renderer(name="render_wildscenes_vtemp"):
    path = Path(__file__).resolve().parents[2] / "simulator" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_perception_report_test_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WildScenesPerceptionReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = load_renderer()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="wildscenes-perception-report-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.results_path = self.directory / "results.json"
        (self.directory / "cases").mkdir()
        self.detail_path = self.directory / "cases" / "corridor_r0.json"
        self.tasks = dict(cases=[dict(id="corridor", start=[20., 20.], goal=[100., 20.])])
        self.perception = dict(mapping_resolution_m=.5, mapping_shape=[48, 48],
                               actor_resolution_m=2., actor_shape=[12, 12])
        self.protocol = "wildscenes_vtemp_dual_resolution_v2"
        self.report = dict(
            signature="fine-mapping-signature",
            provenance=dict(protocol=self.protocol, perception=self.perception,
                            settings=dict(strategy="vtemp", subgoal_radius=200.,
                                          subgoal_arrival=25., mapping_resolution=.5)),
            scheduled=[dict(id="corridor_r0", case_id="corridor", sample_seed=123)],
            records=[dict(id="corridor_r0", case_id="corridor", status="completed",
                          summary=dict(termination="budget_exhausted"))])
        self.detail = dict(
            id="corridor_r0", case_id="corridor", status="completed", sample_seed=123,
            signature=self.report["signature"], protocol=self.protocol,
            summary=dict(termination="budget_exhausted", graph_nodes=3),
            trajectory=[dict(step=0, ground=[20., 20., 0.], uavs=[])],
            evaluation_overrides=dict(strategy="vtemp", subgoal_radius=200.,
                                      subgoal_arrival=25., perception=self.perception))

    def load(self):
        self.results_path.write_text(json.dumps(self.report))
        self.detail_path.write_text(json.dumps(self.detail))
        return self.renderer.load_runs(self.results_path, self.tasks)

    def test_matching_fine_mapping_case_is_read(self):
        report, runs = self.load()
        self.assertEqual(report["signature"], self.report["signature"])
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["summary"]["graph_nodes"], 3)
        self.assertEqual(runs[0]["trajectory"], self.detail["trajectory"])
        self.assertIn("0.5m", self.renderer.perception_description(report["provenance"]))

    def test_same_case_id_with_other_signature_is_rejected(self):
        self.detail["signature"] = "legacy-mapping-signature"
        with self.assertRaisesRegex(ValueError, "signature"):
            self.load()

    def test_different_protocol_is_rejected_even_with_matching_signature(self):
        self.detail["protocol"] = "wildscenes_vtemp_rolling200_frozen_v9_v1"
        with self.assertRaisesRegex(ValueError, "protocol"):
            self.load()

    def test_equivalent_strategy_names_in_one_run_are_supported(self):
        self.detail["evaluation_overrides"]["strategy"] = "vtemp_rolling200"
        _, runs = self.load()
        self.assertEqual(runs[0]["status"], "completed")

    def test_old_vtemp1_label_requires_an_archived_manager_and_protocol(self):
        provenance = self.report["provenance"]
        provenance["settings"]["strategy"] = "vtemp1"
        self.detail["evaluation_overrides"]["strategy"] = "vtemp1"
        with self.assertRaisesRegex(ValueError, "vtemp strategy"):
            self.load()
        provenance["source_hashes"] = {"ground/subgoal_manager.py": self.renderer.FROZEN_MANAGER_SHA256}
        _, runs = self.load()
        self.assertEqual(runs[0]["status"], "completed")
        provenance["protocol"] = self.detail["protocol"] = "vtemp1_directional"
        with self.assertRaisesRegex(ValueError, "protocol"):
            self.load()

    def test_baseline_result_is_rejected(self):
        for settings in (self.report["provenance"]["settings"], self.detail["evaluation_overrides"]):
            settings.update(strategy="v9_no_subgoal", subgoal_radius=None)
        with self.assertRaisesRegex(ValueError, "vtemp strategy"):
            self.load()

    def test_baseline_case_in_vtemp_report_is_rejected(self):
        self.detail["evaluation_overrides"].update(strategy="v9_no_subgoal", subgoal_radius=None)
        with self.assertRaisesRegex(ValueError, "no-subgoal baseline"):
            self.load()

    def test_different_perception_is_rejected_even_with_matching_signature(self):
        self.detail["evaluation_overrides"]["perception"] = dict(
            self.perception, mapping_resolution_m=2., mapping_shape=[12, 12])
        with self.assertRaisesRegex(ValueError, "perception"):
            self.load()

    def test_historical_two_metre_result_without_new_fields_is_supported(self):
        protocol = "wildscenes_vtemp_rolling200_frozen_v9_v1"
        self.report["provenance"]["protocol"] = protocol
        self.report["provenance"].pop("perception")
        self.report["provenance"]["settings"].pop("mapping_resolution")
        self.report["provenance"]["settings"]["strategy"] = "vtemp_rolling200"
        self.detail["protocol"] = protocol
        self.detail["evaluation_overrides"].pop("perception")
        self.detail["evaluation_overrides"]["strategy"] = "vtemp_rolling200"
        report, runs = self.load()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "completed")
        self.assertIn("建图与策略共用12×12", self.renderer.perception_description(report["provenance"]))

    def test_reporting_uses_vtemp_title(self):
        self.tasks["cases"][0]["reference_length_m"] = 80.
        report, runs = self.load()
        stress, note, assumptions = self.renderer.evaluation_scope(self.tasks, runs, {}, [])
        output = self.directory / "report.md"
        self.renderer.write_report(output, self.directory / "tasks.json", self.results_path,
                                   report, runs, False, stress, note, assumptions)
        self.assertIn("# WildScenes vtemp + v9 二维验证", output.read_text())
        self.assertIn("<title>WildScenes · vtemp + v9", self.renderer.HTML)

    def test_new_clearance_is_read_from_result_and_changes_stress_classification(self):
        self.report["provenance"]["settings"]["ground_safety_distance"] = 1.
        self.perception["graph_clearance_m"] = 1.
        self.detail["summary"].update(safety_clearance_m=1., planning_clearances=dict(ground_safety_distance=1.))
        self.detail["evaluation_overrides"]["ground_safety_distance"] = 1.
        self.tasks["cases"][0]["reference_clearance_m"] = 1.
        report, runs = self.load()
        stress, note, assumptions = self.renderer.evaluation_scope(self.tasks, runs, {}, [])
        self.assertFalse(stress)
        self.assertEqual(note, "")
        self.assertEqual(assumptions["planning_safety_clearances_m"], [1.])
        self.assertIn("要求1m净空", self.renderer.perception_description(report["provenance"]))

    def test_report_rejects_same_signature_with_other_actual_clearance(self):
        self.report["provenance"]["settings"]["ground_safety_distance"] = 1.
        self.detail["summary"].update(safety_clearance_m=1., planning_clearances=dict(ground_safety_distance=1.5))
        self.detail["evaluation_overrides"]["ground_safety_distance"] = 1.
        with self.assertRaisesRegex(ValueError, "planning clearance"):
            self.load()

    def test_old_one_point_five_metre_audit_is_not_relabelled_as_one_metre(self):
        runs = [dict(case_id="corridor", case=dict(reference_clearance_m=.3),
                     planning_safety_clearance_m=1., summary={})]
        stress, note, assumptions = self.renderer.evaluation_scope({}, runs, {},
            [dict(id="corridor", connected=False)])
        self.assertTrue(stress)
        self.assertIn("规划净空为 1m", note)
        self.assertNotIn("不连通", note)

    def test_no_subgoal_renderer_reads_same_clearance_contract(self):
        self.report["provenance"]["settings"]["ground_safety_distance"] = 1.
        self.detail["summary"].update(safety_clearance_m=1., planning_clearances=dict(ground_safety_distance=1.))
        self.detail["evaluation_overrides"]["ground_safety_distance"] = 1.
        self.tasks["cases"][0]["reference_clearance_m"] = 1.
        self.load()
        renderer = load_renderer("render_wildscenes_validation")
        _, runs = renderer.load_runs(self.results_path, self.tasks)
        stress, _, assumptions = renderer.evaluation_scope(self.tasks, runs, {}, [])
        self.assertFalse(stress)
        self.assertEqual(assumptions["planning_safety_clearance_m"], 1.)


class WildScenesFormalHistoricalReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = load_renderer("render_wildscenes_legacy_formal")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="wildscenes-formal-report-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def load(self, strategy, *, case_strategy=None, case_protocol=None, archived_manager=False):
        directory = self.directory / "evaluation"
        (directory / "cases").mkdir(parents=True, exist_ok=True)
        protocol = self.renderer.HISTORICAL_PROTOCOL
        provenance = dict(protocol=protocol, checkpoint_sha256=self.renderer.CHECKPOINT_SHA256)
        if archived_manager:
            provenance["source_hashes"] = {"ground/subgoal_manager.py": self.renderer.FROZEN_MANAGER_SHA256}
        if strategy is not None:
            provenance["settings"] = dict(strategy=strategy)
        signature = self.renderer.object_digest(provenance)
        records = []
        for sample in range(9):
            record = dict(id=f"case_r{sample}", summary=dict(termination="budget_exhausted"))
            detail = dict(record, signature=signature, status="completed", sample_seed=sample)
            if case_protocol is not None:
                detail["protocol"] = case_protocol
            if case_strategy is not None:
                detail["evaluation_overrides"] = dict(strategy=case_strategy)
            (directory / "cases" / f"case_r{sample}.json").write_text(json.dumps(detail))
            records.append(record)
        (directory / "results.json").write_text(json.dumps(
            dict(provenance=provenance, signature=signature, records=records, scheduled=records)))
        reader = self.renderer.Reader(self.directory)
        return self.renderer.read_historical_context(reader, self.directory)

    def test_both_names_preserve_the_historical_denominator(self):
        for strategy in ("vtemp", "vtemp_rolling200", None):
            with self.subTest(strategy=strategy):
                contexts = self.load(strategy, case_strategy="vtemp_rolling200")
                self.assertEqual(len(contexts), 1)
                self.assertEqual(contexts[0]["planned_runs"], 9)

    def test_historical_run_rejects_mixed_strategy_and_protocol(self):
        with self.assertRaisesRegex(ValueError, "detail uses a different vtemp strategy"):
            self.load("vtemp", case_strategy="v9_no_subgoal")
        with self.assertRaisesRegex(ValueError, "detail uses a different protocol"):
            self.load("vtemp", case_protocol=self.renderer.EXPECTED_PROTOCOL)

    def test_historical_report_rejects_baseline(self):
        with self.assertRaisesRegex(ValueError, "report uses a different vtemp strategy"):
            self.load("v9_no_subgoal")

    def test_old_vtemp1_label_is_read_only_with_archived_manager_identity(self):
        with self.assertRaisesRegex(ValueError, "report uses a different vtemp strategy"):
            self.load("vtemp1", case_strategy="vtemp1")
        contexts = self.load("vtemp1", case_strategy="vtemp1", archived_manager=True)
        self.assertEqual(contexts[0]["planned_runs"], 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
