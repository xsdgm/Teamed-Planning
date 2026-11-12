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
                            settings=dict(strategy="vtemp_rolling200", subgoal_radius=200.,
                                          subgoal_arrival=25., mapping_resolution=.5)),
            scheduled=[dict(id="corridor_r0", case_id="corridor", sample_seed=123)],
            records=[dict(id="corridor_r0", case_id="corridor", status="completed",
                          summary=dict(termination="budget_exhausted"))])
        self.detail = dict(
            id="corridor_r0", case_id="corridor", status="completed", sample_seed=123,
            signature=self.report["signature"], protocol=self.protocol,
            summary=dict(termination="budget_exhausted", graph_nodes=3),
            trajectory=[dict(step=0, ground=[20., 20., 0.], uavs=[])],
            evaluation_overrides=dict(strategy="vtemp_rolling200", subgoal_radius=200.,
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
        self.detail["protocol"] = protocol
        self.detail["evaluation_overrides"].pop("perception")
        report, runs = self.load()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "completed")
        self.assertIn("建图与策略共用12×12", self.renderer.perception_description(report["provenance"]))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
