"""Audit and report a fixed WildScenes-derived native-2D evaluation plan.

Read-only with respect to inputs and rollouts. Outputs are written only under
--output. No torch, ROS, controller imports, or log parsing are required.
Incomplete or inconsistent evaluations produce audit.json and exit 2, without
publishing a final report. Run again once all planned samples have finished.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path

import numpy as np


REPOSITORY = next(parent for parent in Path(__file__).resolve().parents
                  if (parent / "src/teamed-planning/scripts").is_dir())
EXPECTED_PROTOCOL = "wildscenes_obstacle_layout_native2d_unknown_free_vtemp_v9_v1"
CHECKPOINT_SHA256 = "7ac3aedcc3876902773b2241f60e839eb3e3b5630ddeb3c8be660421f2c24198"
FAILURE_DIAGNOSTIC_SHA256 = "392f160180d0d3b5ada4665c1cd880014c82023424d4d5e9fca0cf002db96311"
PRIMARY_ZERO_METRICS = ("native_collision_attempts", "physical_swept_violations", "chain_link_violation",
                        "spacing_violation", "out_of_bounds")
SCOPE = (
    "这是从WildScenes二维投影提取障碍布局的简化仿真。实测occupied单元保留为矩形，"
    "未知区域按仿真空地处理，不代表真实地面可通行。原生无人机24×24m/2m和地面车8×8m/0.5m"
    "感知、冻结v9 best576权重与VTemp 200m/25m机制保持。"
    "原unknown阻挡实验与本实验环境不同，对照仅供背景参考，不能解释为同环境性能提升。"
)
METRIC_NOTE = (
    "到达、无碰撞到达及原严格成功独立统计；原严格成功要求终点到达且碰撞尝试、物理扫掠、"
    "1.5m净空、通信保链/间距及边界均无违规。保链指标检查所需链路不超过95m，"
    "并非对任意丢包、延迟或真实无线传播的保证。碰撞尝试包含原生端点命中或独立扫掠命中，"
    "独立扫掠审计本身不回退运动。净空违规步数可能重复计入停在同一违规位置的多个时刻。"
    "预算参考路线仅用于固定预算和来源背景：K-01继承旧路线，V-01在简化场景中预选约214米路线；"
    "这些长度不作为新场景全局最短路保证，不据此计算路径最优性。"
    "覆盖面积属于简化仿真的观测面积，不是新确认的真实可通行面积。"
)
TERMINATIONS = {
    "vehicle_arrival": "到达终点", "cbf_infeasible": "CBF不可行",
    "budget_exhausted": "预算耗尽", "physical_geometry_violation": "物理扫掠违规",
    "evaluation_error": "运行错误", "user_stop": "停止", "pending": "待完成",
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve(path, root):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def outcome(summary):
    arrived = bool(summary.get("arrived", False))
    collision_free = (arrived and summary.get("native_collision_attempts", 0) == 0
                      and summary.get("physical_swept_violations", 0) == 0)
    strict = (summary.get("termination") == "vehicle_arrival" and collision_free
              and all(summary.get(key, 0) == 0 for key in (
                  "safety_clearance_violations", "chain_link_violation", "spacing_violation", "out_of_bounds")))
    return dict(arrived=arrived, collision_free_arrival=collision_free, strict_success=strict)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def physical_safety_success(summary, status):
    return (status == "completed" and summary.get("termination") == "vehicle_arrival" and summary.get("arrived") is True
            and all(key in summary and finite_number(summary[key]) and summary[key] == 0 for key in PRIMARY_ZERO_METRICS))


def aggregate(rows):
    result = dict(planned_runs=len(rows), completed_rollouts=0, execution_errors=0,
                  pending_runs=0, arrivals=0, collision_free_arrivals=0, strict_successes=0,
                  physical_safety_successes=0)
    for row in rows:
        status, summary = row["status"], row["summary"]
        result["completed_rollouts"] += status == "completed"
        result["execution_errors"] += status == "error"
        result["pending_runs"] += status not in ("completed", "error")
        result["arrivals"] += bool(summary.get("arrived", False))
        result["collision_free_arrivals"] += bool(summary.get("collision_free_arrival", False))
        result["strict_successes"] += bool(summary.get("strict_success", False))
        result["physical_safety_successes"] += physical_safety_success(summary, status)
    for metric in ("native_collision_attempts", "physical_swept_violations", "safety_clearance_violations"):
        values = [r["summary"].get(metric) for r in rows]
        result[metric + "_runs"] = sum(v is not None and v > 0 for v in values)
        result[metric + "_total"] = sum(v for v in values if v is not None)
    result["terminations"] = dict(Counter(r["summary"].get("termination", "pending") for r in rows))
    return result


class Reader:
    def __init__(self, root):
        self.root = root
        self.errors, self.warnings, self.inputs = [], [], {}
        self.rows, self.details, self.scenes = [], {}, {}
        self.pending_jobs = []
        self.pending_execution = False
        self.safety_clearance_m = 1.5  # Historical protocols did not expose an override.
        self.require_reference_geometry = False

    def check(self, condition, message):
        if not condition:
            self.errors.append(message)

    def hash_check(self, path, expected, label):
        path = Path(path)
        if not path.is_file():
            self.errors.append(f"{label}: missing {path}")
            return None
        actual = self.inputs.get(str(path))
        if actual is None:
            actual = digest(path)
            self.inputs[str(path)] = actual
        self.check(actual == expected, f"{label}: SHA256 mismatch: {path}")
        return actual

    def read_job(self, job):
        case_id, group = job["case_id"], job["group"]
        count, seeds = int(job["samples"]), job["sample_seeds"]
        self.check(count == len(seeds), f"{case_id}: sample count does not match seeds")
        directory = resolve(job["output"], self.root)
        manifest_path, scene_path = resolve(job["case_manifest"], self.root), resolve(job["scene_path"], self.root)
        manifest, scene = read_json(manifest_path), read_json(scene_path)
        cases = [case for case in manifest["cases"] if case["id"] == case_id]
        if len(cases) != 1:
            raise ValueError(f"{case_id}: expected exactly one source case")
        source_case = cases[0]
        if self.require_reference_geometry:
            self.check(source_case.get("reference_geometry_validated") is True
                       and source_case.get("reference_clearance_m", 0) >= self.safety_clearance_m,
                       f"{case_id}: no reference route validated at the normal planning clearance")
        self.scenes[str(scene_path)] = scene
        source = scene["source"]
        source_map = resolve(source["map_path"], scene_path.parent)
        self.hash_check(source_map, source["map_sha256"], "source map")
        self.check(source["map_sha256"] == manifest.get("metadata", {}).get("map_sha256"),
                   f"{case_id}: scene and task manifest come from different source maps")
        self.check(source["map_sha256"] == job["source_map_sha256"], f"{case_id}: source map differs from formal plan")
        self.check(source_case.get("scene_seed") == job["scene_seed"], f"{case_id}: scene seed differs from formal plan")
        self.check(scene.get("simplified_scene") is True
                   and scene.get("unknown_space") == "free_by_simulation_assumption",
                   f"{case_id}: scene does not declare unknown-as-free simplification")
        expected_ids = [f"{case_id}_r{i}" for i in range(count)]
        report_path = directory / "results.json"
        report = read_json(report_path) if report_path.is_file() else None
        record_lookup = {}
        if not report or report.get("status") not in ("completed", "completed_with_errors"):
            self.pending_jobs.append(case_id)
        if report:
            self.inputs[str(report_path)] = digest(report_path)
            provenance = report["provenance"]
            self.check(report.get("signature") == object_digest(provenance), f"{case_id}: report signature is invalid")
            self.check(provenance.get("protocol") == EXPECTED_PROTOCOL, f"{case_id}: unexpected protocol")
            self.check(provenance.get("simplified_scene") is True, f"{case_id}: missing simplified-scene declaration")
            self.hash_check(scene_path, provenance.get("scene_sha256"), case_id + " scene")
            self.hash_check(manifest_path, provenance.get("cases_sha256"), case_id + " cases")
            self.check(provenance.get("source_map_sha256") == source["map_sha256"], f"{case_id}: source hash differs")
            self.check(provenance.get("settings", {}).get("samples") == count, f"{case_id}: samples differ from plan")
            self.check(provenance.get("settings", {}).get("max_steps") is None, f"{case_id}: formal run has a step cap")
            self.check(provenance.get("selected_case_ids") == [case_id], f"{case_id}: case selection differs")
            self.check(provenance.get("settings", {}).get("stochastic") is True, f"{case_id}: stochastic protocol differs")
            settings = provenance.get("settings", {})
            if "ground_safety_distance" in settings or self.safety_clearance_m != 1.5:
                self.check(settings.get("ground_safety_distance") == self.safety_clearance_m
                           and settings.get("safety_clearance_m") == self.safety_clearance_m
                           and settings.get("ground_escape_distance") == .5,
                           f"{case_id}: planning/audit clearance differs from the formal plan")
                for path, expected in provenance.get("clearance_source_hashes", {}).items():
                    self.hash_check(resolve(path, self.root), expected, "clearance override source")
                self.check(len(provenance.get("clearance_source_hashes", {})) == 2,
                           f"{case_id}: missing clearance adapter/configuration hashes")
            for key, filename in dict(runner_sha256="validate_wildscenes_legacy.py",
                                      shared_helper_sha256="validate_wildscenes_vtemp.py",
                                      scene_adapter_sha256="wildscenes_legacy_scene.py").items():
                self.hash_check(self.root / "implementation" / filename, provenance.get(key), case_id + " execution source")
            perception = provenance.get("perception", {})
            for key, value in dict(drone_shape=[12, 12], drone_resolution_m=2., drone_footprint_m=[24., 24.],
                                   ground_shape=[16, 16], ground_resolution_m=.5,
                                   ground_footprint_m=[8., 8.], fine_mapping=False).items():
                self.check(perception.get(key) == value, f"{case_id}: sensor contract differs: {key}")
            self.check(provenance.get("strategy") == dict(subgoal_radius_m=200., subgoal_arrival_m=25., checkpoint_episode=576),
                       f"{case_id}: VTemp/model strategy differs")
            frozen = provenance.get("frozen_hashes", {})
            self.check(frozen.get("best.pt") == CHECKPOINT_SHA256, f"{case_id}: wrong model checkpoint")
            run = resolve(provenance["run"], self.root)
            required_frozen = {"best.pt", "config.json"} | {
                str(path.relative_to(run)) for path in (run / "runtime_sources").rglob("*.py")}
            self.check(len(required_frozen) > 2 and set(frozen) == required_frozen,
                       f"{case_id}: frozen runtime hash inventory is incomplete")
            for relative, sha in frozen.items():
                self.hash_check(run / relative, sha, "frozen runtime")
            scheduled = report.get("scheduled", [])
            self.check([r["id"] for r in scheduled] == expected_ids, f"{case_id}: scheduled IDs differ from plan")
            self.check([r["sample_seed"] for r in scheduled] == seeds, f"{case_id}: scheduled seeds differ from plan")
            records = report.get("records", [])
            record_lookup = {r["id"]: r for r in records}
            self.check(len(record_lookup) == len(records), f"{case_id}: duplicate records")
            self.check(set(record_lookup).issubset(expected_ids), f"{case_id}: unexpected recorded samples")
        rows = []
        for sample, identifier in enumerate(expected_ids):
            detail_path = directory / "cases" / f"{identifier}.json"
            row = dict(id=identifier, case_id=case_id, group=group, sample=sample,
                       sample_seed=seeds[sample], status="pending", summary=dict(termination="pending"),
                       detail_path=str(detail_path), scene_path=str(scene_path),
                       budget_reference_length_m=source_case["reference_length_m"],
                       straight_distance_m=float(np.linalg.norm(np.asarray(source_case["goal"]) - source_case["start"])))
            if detail_path.is_file() and report and identifier in record_lookup:
                detail = read_json(detail_path)
                self.inputs[str(detail_path)] = digest(detail_path)
                record = record_lookup[identifier]
                self.check(detail.get("signature") == report["signature"], f"{identifier}: case/report signature mismatch")
                self.check(detail.get("sample_seed") == seeds[sample], f"{identifier}: sample seed differs")
                self.check(detail.get("sample") == sample and detail.get("case_id") == case_id, f"{identifier}: case/sample identity differs")
                self.check(detail.get("summary") == record.get("summary"), f"{identifier}: stale aggregate record summary")
                self.check(detail.get("status") == record.get("status"), f"{identifier}: status mismatch")
                row.update(status=detail["status"], summary=detail["summary"])
                self.details[identifier] = detail
                if detail["status"] == "completed":
                    self.check(detail.get("protocol") == EXPECTED_PROTOCOL, f"{identifier}: mixed case protocol")
                    self.check(detail.get("checkpoint_sha256") == CHECKPOINT_SHA256, f"{identifier}: mixed model")
                    self.check(detail.get("simplified_scene") is True, f"{identifier}: missing scene declaration")
                    for key in ("start", "goal", "reference_length_m", "scene_seed"):
                        if key in source_case:
                            self.check(detail["case"].get(key) == source_case[key], f"{identifier}: inherited case changed: {key}")
                    summary = detail["summary"]
                    self.check(summary.get("safety_clearance_m") == self.safety_clearance_m,
                               f"{identifier}: summary clearance differs from the formal plan")
                    if "ground_safety_distance" in provenance.get("settings", {}):
                        actual = detail.get("evaluation_overrides", {}).get("planning_clearances", {})
                        self.check(actual.get("ground_safety_distance") == self.safety_clearance_m
                                   and actual.get("ground_escape_distance") == .5
                                   and actual.get("graph_clearance_m") == [self.safety_clearance_m] * 3
                                   and actual.get("graph_clearance_pixels") == [self.safety_clearance_m / 2.] * 3
                                   and actual.get("bridge_clearance_m") == self.safety_clearance_m,
                                   f"{identifier}: actual controller/graph/bridge clearance differs")
                    for key in ("arrived", "collision_free_arrival", "strict_success", "termination", "episode_steps",
                                "episode_budget_steps", "native_collision_attempts", "physical_swept_violations",
                                "safety_clearance_violations", "ground_distance", "ground_goal_distance",
                                "chain_link_violation", "spacing_violation", "out_of_bounds"):
                        self.check(key in summary, f"{identifier}: missing completed-run metric {key}")
                    budget = int(job.get("budget_steps", max(600, math.ceil(3 * source_case["reference_length_m"] + 240))))
                    self.check(summary.get("episode_budget_steps") == budget, f"{identifier}: budget differs from plan")
                    for key, value in outcome(summary).items():
                        self.check(summary.get(key) == value, f"{identifier}: inconsistent outcome flag {key}")
                    for key in PRIMARY_ZERO_METRICS:
                        self.check(finite_number(summary.get(key)) and summary[key] >= 0,
                                   f"{identifier}: invalid required physical-safety metric {key}")
                    if summary.get("native_collision_attempts") == 0:
                        self.check(summary.get("ground_collisions", 0) == 0,
                                   f"{identifier}: native collision attempts contradict ground collision count")
                    if summary.get("spacing_violation") == 0:
                        self.check(summary.get("collisions", 0) == 0,
                                   f"{identifier}: spacing violations contradict drone collision count")
                    row["max_graph_nodes"] = max((t.get("graph_nodes", 0) for t in detail.get("trace", [])), default=0)
                    row["final_graph_nodes"] = summary.get("graph_nodes")
                    row["subgoals_selected"] = summary.get("subgoals_selected")
                    row["subgoal_arrivals"] = summary.get("subgoal_arrivals")
                    frames = detail.get("trace", [])
                    row["network_diagnostics"] = dict(
                        communication_outage=summary.get("communication_outage"),
                        trace_frames=len(frames),
                        backbone_disconnected_frames=sum(t.get("backbone_connected") == 0 for t in frames),
                        backbone_missing_frames=sum(t.get("backbone_connected") not in (0, 1) for t in frames))
            row["physical_safety_success"] = physical_safety_success(row["summary"], row["status"])
            rows.append(row)
        if report and report.get("status") in ("completed", "completed_with_errors"):
            execution_id = report.get("dispatcher_execution_id")
            completion_paths = [(self.root / name / execution_id / "completion.json")
                                for name in ("execution_history", "background_executions")] if execution_id else []
            awaiting_batch = execution_id and not any(path.exists() for path in completion_paths)
            if awaiting_batch:
                self.pending_execution = True
            else:
                self.check(report.get("frozen_inputs_unchanged") is True, f"{case_id}: no final frozen-input verification")
            recomputed = aggregate(rows)
            reported = report.get("aggregate", {})
            comparisons = dict(scheduled_runs=count, recorded_runs=len(record_lookup),
                               completed_rollouts=recomputed["completed_rollouts"], execution_errors=recomputed["execution_errors"],
                               arrivals=recomputed["arrivals"], collision_free_arrivals=recomputed["collision_free_arrivals"],
                               strict_successes=recomputed["strict_successes"], terminations=recomputed["terminations"])
            for key, value in comparisons.items():
                self.check(reported.get(key) == value, f"{case_id}: aggregate differs from case files: {key}")
        self.rows.extend(rows)

    def audit(self):
        protocol_path = self.root / "protocol.json"
        protocol = read_json(protocol_path)
        self.safety_clearance_m = float(protocol.get("safety_clearance_m", 1.5))
        self.require_reference_geometry = protocol.get("require_reference_geometry", False)
        self.inputs[str(protocol_path)] = digest(protocol_path)
        for path, sha in protocol.get("file_hashes", {}).items():
            self.hash_check(resolve(path, self.root), sha, "formal protocol input")
        groups = {g["name"]: g for g in protocol["groups"]}
        self.check(len(groups) == len(protocol["groups"]), "Duplicate group names")
        self.check(len({j["case_id"] for j in protocol["jobs"]}) == len(protocol["jobs"]), "Duplicate case jobs")
        for job in protocol["jobs"]:
            self.check(job["group"] in groups, f"Unknown group: {job['group']}")
            try:
                self.read_job(job)
            except (KeyError, ValueError, TypeError, OSError) as exc:
                self.errors.append(f"{job.get('case_id', '?')}: {type(exc).__name__}: {exc}")
        grouped = []
        for name, group in groups.items():
            jobs = [j for j in protocol["jobs"] if j["group"] == name]
            self.check(set(j["case_id"] for j in jobs) == set(group["case_ids"]), f"{name}: case IDs differ from group plan")
            planned = sum(int(j["samples"]) for j in jobs)
            self.check(planned == group["expected_samples"], f"{name}: sample denominator differs from group plan")
            rows = [r for r in self.rows if r["group"] == name]
            self.check(len(rows) == planned, f"{name}: some planned rows could not be read")
            grouped.append(dict(name=name, label=group["label"], **aggregate(rows)))
        total = aggregate(self.rows)
        planned_total = sum(g["expected_samples"] for g in groups.values())
        self.check(len(self.rows) == planned_total, "Overall planned denominator is incomplete")
        self.check(protocol.get("total_samples") == planned_total, "Protocol total_samples differs from group denominators")
        try:
            geometry = read_geometry_context(self, protocol)
        except (KeyError, ValueError, TypeError, OSError, StopIteration) as exc:
            self.errors.append(f"Independent geometry context: {type(exc).__name__}: {exc}")
            geometry = dict(status="invalid", notes=["独立几何证据未通过完整性核验。"])
        try:
            acceptance = read_acceptance(reader=self)
        except (KeyError, ValueError, TypeError, OSError) as exc:
            self.errors.append(f"Acceptance amendment: {type(exc).__name__}: {exc}")
            acceptance = dict(status="invalid")
        try:
            failure_behavior = read_failure_behavior(self)
        except (KeyError, ValueError, TypeError, OSError) as exc:
            self.errors.append(f"Failure-behavior diagnostic: {type(exc).__name__}: {exc}")
            failure_behavior = dict(status="invalid", notes=[])
        try:
            execution = read_execution_context(self, protocol)
        except (KeyError, ValueError, TypeError, OSError) as exc:
            self.errors.append(f"Execution recovery: {type(exc).__name__}: {exc}")
            execution = dict(status="invalid", notes=[])
        network_rows = [r for r in self.rows if r["status"] == "completed"]
        network = dict(
            completed_trials=len(network_rows),
            communication_outage_trials=sum(finite_number(r.get("network_diagnostics", {}).get("communication_outage"))
                and r["network_diagnostics"]["communication_outage"] != 0 for r in network_rows),
            backbone_disconnected_trials=sum(r.get("network_diagnostics", {}).get("backbone_disconnected_frames", 0) > 0
                                            for r in network_rows),
            incomplete_diagnostic_trials=sum(not finite_number(r.get("network_diagnostics", {}).get("communication_outage"))
                or not r.get("network_diagnostics", {}).get("trace_frames")
                or r.get("network_diagnostics", {}).get("backbone_missing_frames", 0) > 0 for r in network_rows))
        network["notes"] = [
            f"补充检查{network['completed_trials']}份完整结果：通信中断统计非零的试验"
            f"{network['communication_outage_trials']}次，逐步骨干图出现断连的试验"
            f"{network['backbone_disconnected_trials']}次，诊断字段不完整的试验"
            f"{network['incomplete_diagnostic_trials']}次。这些诊断不修改已冻结的主验收公式。"]
        status = "invalid" if self.errors else "incomplete" if total["pending_runs"] or self.pending_jobs or self.pending_execution else "complete"
        audit = dict(status=status, created_utc=datetime.now(timezone.utc).isoformat(),
                     root=str(self.root), expected_samples=planned_total, scope=SCOPE,
                     safety_clearance_m=self.safety_clearance_m,
                     metric_contract=METRIC_NOTE.replace("1.5m", f"{self.safety_clearance_m:g}m"),
                     checkpoint_sha256=CHECKPOINT_SHA256, groups=grouped, aggregate=total,
                     rows=self.rows, errors=self.errors, warnings=self.warnings, pending_jobs=self.pending_jobs,
                     input_sha256=self.inputs, geometry_context=geometry, acceptance=acceptance,
                     failure_behavior=failure_behavior, execution_history=execution, network_diagnostics=network,
                     report_generator_sha256=digest(__file__))
        if self.require_reference_geometry:
            audit["metric_contract"] = audit["metric_contract"].replace(
                "预算参考路线仅用于固定预算和来源背景：K-01继承旧路线，V-01在简化场景中预选约214米路线；",
                "本批逐段验证的参考路线仅用于固定预算和几何可行性证明；")
        return protocol, audit


def read_acceptance(reader):
    path = reader.root / "evaluation_amendment.json"
    if not path.is_file():
        protocol = read_json(reader.root / "protocol.json")
        if protocol.get("primary_metric") == "physical_safety_success":
            reader.check(protocol.get("ground_escape_distance_m") == .5,
                         "Unexpected escape clearance in the physical-safety protocol")
            return dict(primary_metric="physical_safety_success", aggregate_key="physical_safety_successes",
                        label="验收通过（物理安全）",
                        note=f"正常规划、建图和桥接净空统一为{reader.safety_clearance_m:g}m，受困逃逸保留0.5m。"
                             "沿用用户确认的物理安全验收：到达且无碰撞尝试、0.3m实体扫掠、保链、间距及边界违规。"
                             f"含{reader.safety_clearance_m:g}m净空的严格成功另列诊断；允许的0.5m逃逸也可能产生该净空违规。")
        return dict(primary_metric="strict_success", aggregate_key="strict_successes", label="主验收（原严格）",
                    note=f"未提供验收标准修订；主指标包含{reader.safety_clearance_m:g}m净空要求。")
    amendment = read_json(path)
    reader.inputs[str(path)] = digest(path)
    definition = dict(required_termination="vehicle_arrival", required_arrival=True, zero_metrics=list(PRIMARY_ZERO_METRICS))
    reader.check(amendment.get("protocol") == "wildscenes_formal_user_requested_acceptance_amendment_v1"
                 and amendment.get("formal_protocol_sha256") == reader.inputs[str(reader.root / "protocol.json")],
                 "Acceptance amendment does not match the frozen formal protocol")
    reader.check(amendment.get("primary_metric") == "physical_safety_success"
                 and amendment.get("primary_definition") == definition
                 and amendment.get("physical_radius_m") == .3
                 and amendment.get("scope") == "evaluation acceptance only; no controller or runtime changes"
                 and amendment.get("post_start_change") is True,
                 "Acceptance amendment differs from the supported physical-safety definition")
    note = ("本次主验收按用户在运行中提出的要求修订：不要求1.5m净空；运行完整结束、到达终点且碰撞尝试、"
            "0.3m物理扫掠、通信、间距和边界均无违规，即计为“验收通过（物理安全）”。"
            "修订追溯适用于全部21次预定运行，包括修订前已完成的样本；原始结果不改写。"
            "规划器默认1.5m净空和0.5m逃逸配置保持，原严格成功及1.5m违规仅作附加诊断。")
    return dict(primary_metric="physical_safety_success", aggregate_key="physical_safety_successes",
                label="验收通过（物理安全）", note=note, source=str(path), source_sha256=digest(path), amendment=amendment)


def read_failure_behavior(reader):
    path = reader.root / "diagnostics/completed_short_budget_failures.json"
    if not path.is_file():
        return dict(status="unavailable", notes=[])
    reader.hash_check(path, FAILURE_DIAGNOSTIC_SHA256, "completed short-run failure diagnostic")
    data = read_json(path)
    reader.check(data.get("protocol") == "wildscenes_completed_short_budget_failure_diagnosis_v1"
                 and data.get("sources_unchanged_during_audit") is True, "Unexpected failure-diagnostic protocol")
    for source, expected in data["all_source_hashes"].items():
        reader.hash_check(resolve(source, reader.root), expected, "failure-diagnostic input")
    planned = {row["id"] for row in reader.rows}
    cases = {case["case_id"]: case for case in data["cases"]}
    reader.check(set(cases).issubset(planned), "Failure diagnostic includes samples outside the formal plan")
    for case in cases.values():
        reader.check(case["termination"] == "budget_exhausted", "Failure diagnostic includes a different termination")
    scores = sum(len(case["selection_score_evidence"]) for case in cases.values())
    stationary = cases["K_01_control_200m_reverse_r2"]["statistics"]["longest_stationary_intervals"][0]
    excursion = cases["K_01_control_200m_reverse_r1"]["statistics"]["max_displacement_from_start_m"]
    notes = [
        "该补充诊断仅覆盖5次已完成的短程预算耗尽运行。K-01正向r1/r2及V-01正向r0将最终终点标记失败后，"
        "其60m邻域中的候选点受到200×exp(−age/600)失败罚分（age以秒计），最终终点自身也受罚。"
        "这抬高终点的候选成本，使外围临时目标获得优先级，并解释部分后续选点远离终点的行为。"
        f"诊断文件合计保留{scores}条精确选点评分证据，覆盖这5次运行。",
        f"K-01反向r2在第{stationary['start_step']}–{stationary['end_step']}步连续{stationary['count']}步位移为0；"
        f"反向r1则在起点周围{excursion:.2f}m范围内来回移动。"
        "这些是特定运行的直接行为证据，不能将所有失败归因于罚分或1.5m要求；"
        "尚未进行修复后的反事实验证，不能断言移除罚分后必定成功。",
    ]
    return dict(status="verified_supplement", source=str(path), source_sha256=reader.inputs[str(path)],
                case_ids=list(cases), selection_score_records=scores, notes=notes)


def read_execution_context(reader, protocol):
    history = reader.root / "execution_history"
    recovery_paths = sorted(history.glob("*/recovery_plan.json"))
    if not recovery_paths:
        launch_path = reader.root / "execution_plan.json"
        if launch_path.is_file():
            launch = read_json(launch_path)
            reader.inputs[str(launch_path)] = digest(launch_path)
            reader.check(launch.get("protocol_sha256") == digest(reader.root / "protocol.json")
                         and launch.get("total_samples") == protocol["total_samples"]
                         and launch.get("maximum_live_evaluators") == protocol["maximum_concurrent_cases"],
                         "Background launcher plan differs from the formal protocol")
            reader.hash_check(resolve(launch["launcher_path"], reader.root), launch["launcher_sha256"], "background launcher")
            completion_path, batch_path = reader.root / "completion.json", reader.root / "batch_state.json"
            final = completion_path.is_file() and batch_path.is_file()
            if not final:
                reader.pending_execution = True
            else:
                completion, batch = read_json(completion_path), read_json(batch_path)
                reader.check(completion == batch, "Background completion differs from batch state")
                reader.check(batch.get("status") in ("completed", "completed_with_errors")
                             and batch.get("total_samples") == protocol["total_samples"]
                             and batch.get("frozen_inputs_unchanged") is True
                             and batch.get("peak_live_evaluators", 99) <= protocol["maximum_concurrent_cases"],
                             "Background batch did not finish under the declared contract")
                for job in protocol["jobs"]:
                    report = read_json(resolve(job["output"], reader.root) / "results.json")
                    reader.check(report.get("frozen_inputs_unchanged") is True
                                 and report.get("dispatcher_execution_id") == batch.get("execution_id")
                                 and batch.get("jobs", {}).get(job["case_id"], {}).get("aggregate") == report.get("aggregate"),
                                 f"{job['case_id']}: final aggregate/verification differs from batch state")
            return dict(status="complete" if final else "pending_background_completion", sources=[
                dict(label="独立后台执行计划", path=str(launch_path), sha256=digest(launch_path))],
                notes=["本批在独立后台会话中运行，所有预定试验从各自种子的初始状态开始；"
                       "仅在全部终态结果和源文件散列核验结束后生成正式报告。"])
        return dict(status="no_recovery_record", notes=[])
    latest = recovery_paths[-1]
    execution_id = latest.parent.name
    plan = read_json(latest)
    reader.inputs[str(latest)] = digest(latest)
    reader.hash_check(resolve(plan["dispatcher"], reader.root), plan["sha256"], "recovery dispatcher")
    for source, expected in plan["source_hashes"].items():
        reader.hash_check(resolve(source, reader.root), expected, "recovery frozen input")
    reader.check(plan.get("maximum_live_evaluators") == 5, "Recovery concurrency differs from declared limit")
    sources = [dict(label="最新恢复计划", path=str(latest), sha256=reader.inputs[str(latest)])]
    interrupted, retained, interrupted_evidence = [], set(), {}
    for path in sorted(history.glob("*/handoff_audit.json")):
        data = read_json(path)
        reader.inputs[str(path)] = digest(path)
        reader.check(data["source_evidence"]["protocol_sha256"] == reader.inputs[str(reader.root / "protocol.json")],
                     "Handoff audit belongs to a different formal protocol")
        handoff_path = resolve(data["source_evidence"]["handoff_path"], reader.root)
        reader.hash_check(handoff_path, data["source_evidence"]["handoff_sha256"], "handoff evidence")
        handoff = read_json(handoff_path)
        retained.update(handoff["plan"]["completed_trial_ids"])
        interrupted.extend(row["trial_id"] for row in data["attached_trials"] if row["canonical_case_exists"] is False)
        interrupted_evidence.update({row["trial_id"]: row for row in data["attached_trials"]
                                     if row["canonical_case_exists"] is False})
        reader.check(data["frozen_inputs"]["all_unchanged"] is True, "Handoff recorded changed formal inputs")
        sources.append(dict(label="接管中断审计", path=str(path), sha256=reader.inputs[str(path)]))
    preserved_paths = sorted(history.glob("*/preserved_completed_cases.json"))
    if not preserved_paths:
        raise ValueError("Missing hash inventory for the originally retained completed cases")
    preserved_path = preserved_paths[0]
    preserved = read_json(preserved_path)
    reader.inputs[str(preserved_path)] = digest(preserved_path)
    reader.check(preserved["completed_count"] == len(preserved["files"]) == len(retained)
                 and {Path(path).stem for path in preserved["files"]} == retained,
                 "Preserved-case hash inventory differs from the completed handoff set")
    reader.check(preserved.get("recorded_after_recovery_started") is True,
                 "Preserved-case snapshot timing is not explicitly recorded")
    for path, expected in preserved["files"].items():
        reader.hash_check(resolve(path, reader.root), expected, "retained completed case bytes")
    sources.append(dict(label="恢复启动后记录的14份完整case哈希快照", path=str(preserved_path), sha256=digest(preserved_path)))
    session_recoveries = []
    for path in sorted(history.glob("*/session_recovery_audit.json")):
        data = read_json(path)
        reader.inputs[str(path)] = digest(path)
        reader.check(data["protocol_sha256"] == reader.inputs[str(reader.root / "protocol.json")],
                     "Session recovery audit belongs to a different formal protocol")
        reader.hash_check(resolve(data["wrapper"], reader.root), data["wrapper_sha256"], "detached recovery wrapper")
        for saved, expected in data["retained_terminal_files"].items():
            reader.hash_check(resolve(saved, reader.root), expected, "terminal case retained across session recovery")
        reader.check({Path(p).stem for p in data["retained_terminal_files"]} == set(data["retained_terminal_trial_ids"]),
                     "Session recovery terminal inventory is inconsistent")
        for previous in data["previous_progress"]:
            reader.hash_check(resolve(previous["path"], reader.root), previous["sha256"], "previous supervisor heartbeat")
        lost_staged = []
        for prefix_path in sorted((path.parent / "interrupted_staging").glob("*/progress/*.json")):
            before = read_json(prefix_path)
            if before.get("status") != "running":
                continue
            trial_id = before["id"]
            lost_staged.append(trial_id)
            interrupted_evidence[trial_id] = dict(last_progress_sha256=digest(prefix_path),
                                                 archived_path=str(prefix_path))
        session_recoveries.append(dict(source=str(path), retained_trials=len(data["retained_terminal_trial_ids"]),
                                       interrupted_staged_trials=lost_staged, audit=data))
        sources.append(dict(label="会话中断后的再次恢复与完整结果保留记录", path=str(path), sha256=digest(path)))
    batch_path = reader.root / "batch_state.json"
    batch = read_json(batch_path)
    reader.inputs[str(batch_path)] = digest(batch_path)
    completion_path = latest.parent / "completion.json"
    final = batch.get("status") in ("completed", "completed_with_errors") and completion_path.is_file()
    prefix_reproduction = []
    if not final:
        reader.pending_execution = True
    else:
        reader.check(batch.get("dispatcher_execution_id") == execution_id,
                     "Final batch_state does not identify the latest recovery execution")
        completion = read_json(completion_path)
        reader.inputs[str(completion_path)] = digest(completion_path)
        reader.check(completion == batch, "Recovery completion record differs from final batch_state")
        reader.check(batch.get("total_samples") == protocol["total_samples"] and batch.get("frozen_inputs_unchanged") is True,
                     "Recovered batch lacks the full denominator or final frozen-input verification")
        reader.check(batch.get("peak_live_evaluators", 6) <= 5, "Recovered batch exceeded concurrency limit")
        reader.hash_check(reader.root / "parallel_trials.py", batch["dispatcher_source_sha256"], "final dispatcher source")
        reader.check(set(batch["jobs"]) == {job["case_id"] for job in protocol["jobs"]},
                     "Recovered batch jobs differ from the original formal plan")
        for job in protocol["jobs"]:
            report = read_json(resolve(job["output"], reader.root) / "results.json")
            reader.check(report.get("dispatcher_execution_id") == execution_id,
                         f"{job['case_id']}: final results belong to a different recovery execution")
            reader.check(batch["jobs"][job["case_id"]].get("aggregate") == report.get("aggregate"),
                         f"{job['case_id']}: final batch aggregate differs from canonical results")
        for trial_id, original in interrupted_evidence.items():
            if "archived_path" in original:
                prefix_path = Path(original["archived_path"])
            else:
                candidates = sorted(history.glob(f"*/interrupted_canonical/{trial_id}/progress/{trial_id}.json"))
                matches = [p for p in candidates if digest(p) == original["last_progress_sha256"]]
                if not matches:
                    raise ValueError(f"No archived prefix matches the recorded interruption: {trial_id}")
                prefix_path = matches[0]
            reader.hash_check(prefix_path, original["last_progress_sha256"], "archived interrupted prefix")
            before = read_json(prefix_path)
            detail = reader.details.get(trial_id, {})
            step = before["step"]
            row = dict(trial_id=trial_id, archived_progress=str(prefix_path), comparison_step=step)
            if len(detail.get("trace", [])) < step:
                row.update(status="not_comparable", reason="Final rollout has no trace at the archived step")
            else:
                trace = detail["trace"][:step]
                violations = [v for v in detail.get("swept_violation_events", []) if v["step"] <= step]
                after = dict(ground_remaining_m=float(trace[-1]["ground_goal_distance"]),
                             ground_distance_m=float(sum(frame["ground_distance"] for frame in trace)),
                             native_collision_attempts=sum(e["step"] <= step for e in detail.get("native_collision_events", [])),
                             physical_swept_violations=sum(v["physical_violation"] for v in violations),
                             safety_clearance_violations=sum(v["safety_clearance_violation"] for v in violations))
                comparisons = {key: dict(interrupted=before[key], restarted=value, exact_match=before[key] == value)
                               for key, value in after.items()}
                row.update(status="compared", metrics=comparisons,
                           all_compared_metrics_exact=all(v["exact_match"] for v in comparisons.values()))
                if not row["all_compared_metrics_exact"]:
                    reader.warnings.append(f"{trial_id}: restarted prefix differs from archived progress at step {step}")
            prefix_reproduction.append(row)
        sources.append(dict(label="最终恢复完成记录", path=str(completion_path), sha256=digest(completion_path)))
    notes = [
        f"执行过程中发生过基础设施异常：接管原CLI时，{len(interrupted)}个未完成试验进程退出且未生成终态case。"
        "现有证据不能确定退出原因，不能表述为全程零基础设施异常。",
        f"恢复计划保留此前{len(retained)}份已完成终态结果，不覆盖已有case；收养仍存活的独立工作进程，"
        "缺失试验按原随机种子、原完整预算恢复执行，冻结策略和正式输入保持不变。"
        "中断前缀不计为额外样本，固定分母仍为21；表中execution_errors只统计最终case的运行错误，"
        "不代表执行过程中没有中断。",
        f"保留case的SHA256快照记录于{preserved['recorded_utc']}（恢复启动后）；"
        f"本次读取逐一验证了其中{preserved['completed_count']}份完整case字节未变。",
    ]
    for session in session_recoveries:
        notes.append(f"后来一次会话中断后，恢复调度的心跳停止，退出原因未确定。再次恢复时核验并保留"
                     f"{session['retained_trials']}份完整试验结果（包括独立子进程完成但未汇总的结果）；"
                     f"另有{len(session['interrupted_staged_trials'])}个暂存试验没有终态结果，连同尚未启动的试验"
                     "按原种子和完整预算补跑。第二次恢复使用独立会话启动器，并保留了之前的进度和日志。")
    if prefix_reproduction:
        exact = sum(row.get("all_compared_metrics_exact", False) for row in prefix_reproduction)
        notes.append(f"重跑前缀核对：{exact}/{len(prefix_reproduction)}条在原中断步的剩余距离、累计车程及三类违规计数"
                     "与归档progress精确一致；逐项数值和不可比较情况见audit.json。该局部核对不替代完整样本审计。")
    return dict(status="complete" if final else "pending_recovery_completion", dispatcher_execution_id=execution_id,
                interrupted_trial_ids=interrupted, retained_completed_trial_ids=sorted(retained),
                preserved_cases_snapshot=preserved, session_recoveries=session_recoveries,
                prefix_reproduction=prefix_reproduction, notes=notes, sources=sources)


def read_geometry_context(reader, protocol):
    if protocol.get("require_reference_geometry"):
        evidence = protocol["route_validation"]
        path = resolve(evidence["path"], reader.root)
        reader.hash_check(path, evidence["sha256"], "normal-clearance reference route validation")
        data = read_json(path)
        reader.check(data.get("normal_clearance_m") == reader.safety_clearance_m
                     and data.get("all_routes_exact_continuous_clearance_verified") is True
                     and data.get("escape_not_used_in_task_feasibility") is True
                     and data.get("policy_results_read") is False
                     and {c["id"] for c in data["cases"]} == {j["case_id"] for j in protocol["jobs"]},
                     "Reference route proof does not match the declared clearance/task set")
        for source, expected in data.get("source_hashes", {}).items():
            reader.hash_check(resolve(source, reader.root), expected, "route proof source")
        return dict(status="validated_reference_routes", notes=[
            f"本批每个任务均预先验证了{reader.safety_clearance_m:g}m正常净空的连续参考路线。"
            "几何检查独立于策略运行；0.5m受困逃逸不作为任务可行性标准。",
            "参考折线仅用于任务可行性证明和预算，不输入策略或控制器；不保证连续空间最短。"
            "未通过净空验证的旧任务已排除，新的起终点及实际参考长度以本批任务清单为准。",
            "部分公里参考路线主要经过未采集区域，依照本场景的空地假设通行；"
            "这不是已观测真实森林路线的通行性证明。原已观测区域长度占比详见几何证据。"],
            sources=[dict(label="本批逐段连续净空验证", path=str(path), sha256=digest(path))],
            validation=data, subsets=[])
    preparation = reader.root / "task_preparation"
    path = preparation / "K-long-clearance-feasibility-refined.json"
    if not path.is_file():
        note = "未找到独立K长程1.5m净空审计；继承的物理参考路线本身不证明全程满足1.5m净空，不能把严格失败全部归因策略。"
        reader.warnings.append(note)
        return dict(status="unavailable", notes=[note])
    data = read_json(path)
    reader.inputs[str(path)] = digest(path)
    reader.hash_check(preparation / "audit_k_long_clearance_refined.py", data["script_sha256"],
                      "independent witness audit script")
    reader.check(data.get("formal_protocol_sha256") == reader.inputs[str(reader.root / "protocol.json")],
                 "Independent geometry diagnostic belongs to a different formal plan")
    for file_key, hash_key in (("scene", "scene_sha256"), ("source_npz", "source_npz_sha256"),
                               ("frozen_tasks", "frozen_tasks_sha256"),
                               ("geometry_implementation", "geometry_implementation_sha256")):
        reader.hash_check(resolve(data[file_key], reader.root), data[hash_key], "independent geometry diagnostic")
    reader.check(data.get("clearance_m") == 1.5 and data.get("simplified_scene") is True,
                 "Independent geometry diagnostic uses a different clearance or environment")
    planned = {job["case_id"]: job for job in protocol["jobs"]}
    notes, rows, sources = [], [], [dict(label="K长程连续净空路径见证", path=str(path), sha256=digest(path))]
    witnesses = set()
    for result in data["results"]:
        case_id = result["case_id"]
        reader.check(case_id in planned, f"Independent geometry diagnostic has unplanned case {case_id}")
        if case_id not in planned:
            continue
        manifest = read_json(resolve(planned[case_id]["case_manifest"], reader.root))
        case = next(c for c in manifest["cases"] if c["id"] == case_id)
        reader.check(all(result.get(key) == case[key] for key in ("start", "goal")),
                     f"{case_id}: independent geometry diagnostic endpoints differ")
        reader.check(result.get("original_budget_reference_length_m") == case["reference_length_m"],
                     f"{case_id}: independent geometry diagnostic budget reference differs")
        row = {key: result.get(key) for key in ("case_id", "status", "start_clear_1_5", "goal_clear_1_5",
                                               "witness_length_m", "exact_segments_verified")}
        rows.append(row)
        if result["status"] == "continuous_path_found":
            witnesses.add(case_id)
            notes.append(f"{case_id}：独立几何搜索找到{result['witness_length_m']:.2f}m路线见证，"
                         f"{result['exact_segments_verified']}段逐段通过连续1.5m净空检查。它不是最短路保证，"
                         "未输入策略，也未替换正式预算参考。")
    impossible = set()
    arrival_path = preparation / "K-long-arrival-disk-connectivity.json"
    mathematical_justification = []
    if arrival_path.is_file():
        arrival = read_json(arrival_path)
        reader.inputs[str(arrival_path)] = digest(arrival_path)
        proof_path = resolve(arrival["previous_mathematical_audit"], reader.root)
        reader.hash_check(proof_path, arrival["previous_audit_sha256"], "arrival-disk mathematical proof")
        proof = read_json(proof_path)
        reader.hash_check(preparation / "audit_k_arrival_disk_connectivity.py", arrival["script_sha256"],
                          "arrival-disk proof script")
        reader.hash_check(preparation / "audit_k_optimistic_connectivity.py", proof["script_sha256"],
                          "optimistic-connectivity proof script")
        reader.check(proof["formal_protocol_sha256"] == reader.inputs[str(reader.root / "protocol.json")],
                     "Optimistic-connectivity proof belongs to a different formal plan")
        reader.check(proof.get("protocol") == "occupied_only_optimistic_continuous_clearance_necessary_condition_v1"
                     and arrival.get("protocol") == "optimistic_connectivity_for_actual_goal_arrival_disk_v1",
                     "Unexpected mathematical proof protocol")
        for document in (proof, arrival):
            for key in ("scene", "source_npz"):
                reader.hash_check(resolve(document[key], reader.root), document[key + "_sha256"], "connectivity proof input")
                reader.check(document[key + "_sha256"] == data[key + "_sha256"],
                             "Connectivity proof and witness refer to different geometry")
            reader.check(document.get("clearance_m") == 1.5 and document.get("policy_results_read") is False,
                         "Connectivity proof must be independent of policy outcomes and use 1.5m clearance")
        reader.hash_check(resolve(proof["tasks"], reader.root), proof["tasks_sha256"], "connectivity proof tasks")
        reader.hash_check(resolve(proof["independent_witness_file"], reader.root),
                          proof["independent_witness_sha256"], "connectivity proof positive control")
        contract = arrival["runtime_arrival_contract"]
        reader.hash_check(resolve(contract["file"], reader.root), contract["sha256"], "actual frozen goal arrival condition")
        reader.check(protocol["file_hashes"].get(contract["file"]) == contract["sha256"],
                     "Arrival contract is not the frozen formal runtime")
        reader.check(arrival.get("arrival_radius_m") == 2. and contract.get("method") == "Experiment._ground_action"
                     and contract.get("condition") == "Euclidean distance to final goal <= 2.0m",
                     "Arrival-disk proof does not cover the actual goal threshold")
        proof_rows = {r["case_id"]: r for r in proof["results"]}
        mathematical_justification = proof["mathematical_justification"] + [arrival["justification"]]
        for result in arrival["results"]:
            case_id = result["case_id"]
            manifest = read_json(resolve(planned[case_id]["case_manifest"], reader.root))
            case = next(c for c in manifest["cases"] if c["id"] == case_id)
            endpoint_proof = proof_rows[case_id]
            reader.check(all(endpoint_proof[key]["position_xy"] == case[key] for key in ("start", "goal")),
                         f"{case_id}: disconnection proof uses different endpoints")
            reader.check(result["start_component"] == endpoint_proof["start"]["component"]
                         and result["arrival_radius_m"] == 2., f"{case_id}: inconsistent arrival-disk proof")
            if result["arrival_disk_touches_start_component"] is False:
                reader.check(result["start_component"] > 0
                             and result["start_component"] not in result["arrival_disk_optimistic_components"]
                             and result["start_component_cells_intersecting_arrival_disk"] == 0,
                             f"{case_id}: contradictory disconnection certificate")
                impossible.add(case_id)
                row = next(r for r in rows if r["case_id"] == case_id)
                row.update(status="proven_no_1_5m_path_to_2m_arrival_disk", arrival_disk_proof=result)
                notes.append(f"{case_id}：起点位于乐观分量{result['start_component']}，终点2m到达盘仅接触"
                             f"分量{result['arrival_disk_optimistic_components']}。已证明在本简化障碍几何内，"
                             "不存在保持1.5m净空并进入正式到达盘的连续路线，严格成功在几何上不可行。"
                             "这些正式样本仍保留，严格失败不能归因于模型。")
        sources.extend([dict(label="1.5m净空不连通的数学证明", path=str(proof_path), sha256=digest(proof_path)),
                        dict(label="正式2m终点到达盘证明", path=str(arrival_path), sha256=digest(arrival_path)),
                        dict(label="必要连通性审计脚本", path=str(preparation / "audit_k_optimistic_connectivity.py"),
                             sha256=proof["script_sha256"]),
                        dict(label="到达盘审计脚本", path=str(preparation / "audit_k_arrival_disk_connectivity.py"),
                             sha256=arrival["script_sha256"])])
    for row in rows:
        if row["status"] == "inconclusive":
            notes.append(f"{row['case_id']}：有界保守栅格搜索未找到连接，且尚无正式到达盘不连通证明；"
                         "连续空间可达性仍未确定，不能把严格失败全部归因策略。")
    for job in protocol["jobs"]:
        if job["group"] not in ("K_control", "V_control"):
            continue
        manifest_path = resolve(job["case_manifest"], reader.root)
        manifest = read_json(manifest_path)
        case = next(c for c in manifest["cases"] if c["id"] == job["case_id"])
        validated = case.get("reference_geometry_validated") is True and case.get("reference_clearance_m", 0) >= 1.5
        reader.check(validated, f"{job['case_id']}: short-task reference lacks a 1.5m geometry witness")
        if validated:
            witnesses.add(job["case_id"])
            rows.append(dict(case_id=job["case_id"], status="continuous_path_found",
                             witness_length_m=case["reference_length_m"], witness_source=str(manifest_path)))
        if str(manifest_path) not in {s["path"] for s in sources}:
            sources.append(dict(label=job["group"] + "冻结任务中的净空路线见证", path=str(manifest_path), sha256=digest(manifest_path)))
    reader.check(not impossible.intersection(witnesses), "Geometric witnesses contradict disconnection certificates")
    subsets = []
    for name, label, cases in (("proven_infeasible", "1.5m净空到达在几何上不可行", impossible),
                               ("path_witness", "具有1.5m连续净空路径见证", witnesses)):
        if cases:
            selected = [row for row in reader.rows if row["case_id"] in cases]
            subsets.append(dict(name=name, label=label, case_ids=sorted(cases), **aggregate(selected)))
    if impossible:
        notes.append("附加几何分层来自不读取策略结果的独立审计，不替换原21次计划、预算或分母。"
                     "路径见证仅证明车辆静态净空路线存在，不保证通信、队形、策略观测或预算等其他严格条件可满足。")
    return dict(status="independent_diagnostic", source=str(path), source_sha256=reader.inputs[str(path)],
                notes=notes, rows=rows, sources=sources, mathematical_justification=mathematical_justification,
                subsets=subsets, scope="Independent geometry audit; separate from policy rollouts and frozen budget references.")


def read_historical_context(reader, baseline_root):
    contexts = []
    planned = {row["id"]: row for row in reader.rows}
    for directory, label in (("evaluation", "原K-01长程：unknown阻挡"),
                             ("evaluation_control", "原K-01短程：unknown阻挡")):
        path = baseline_root / directory / "results.json"
        if not path.is_file():
            reader.warnings.append(f"Historical context missing: {path}")
            continue
        report = read_json(path)
        if object_digest(report["provenance"]) != report["signature"]:
            raise ValueError(f"Historical report signature is inconsistent: {path}")
        if (report["provenance"].get("protocol") != "wildscenes_vtemp_rolling200_frozen_v9_v1"
                or report["provenance"].get("checkpoint_sha256") != CHECKPOINT_SHA256):
            raise ValueError(f"Historical report uses a different protocol or model: {path}")
        expected_count = 9 if directory == "evaluation" else 6
        identifiers = [record["id"] for record in report["records"]]
        scheduled_ids = [record["id"] for record in report["scheduled"]]
        if (len(identifiers) != expected_count or len(set(identifiers)) != expected_count
                or set(identifiers) != set(scheduled_ids)):
            raise ValueError(f"Historical report denominator differs from the 9/6 plan: {path}")
        reader.inputs[str(path)] = digest(path)
        rows, paired = [], 0
        for record in report["records"]:
            detail_path = path.parent / "cases" / f"{record['id']}.json"
            detail = read_json(detail_path)
            reader.inputs[str(detail_path)] = digest(detail_path)
            if detail.get("signature") != report["signature"] or detail.get("summary") != record["summary"]:
                raise ValueError(f"Historical detail differs from report: {detail_path}")
            summary = dict(detail["summary"])
            summary["native_collision_attempts"] = summary.get("raster_collision_attempts", 0)
            rows.append(dict(status=detail["status"], summary=summary))
            current = planned.get(record["id"])
            if current and current["sample_seed"] == detail["sample_seed"]:
                now = reader.details[current["id"]]
                paired += now.get("case", {}).get("start") == detail["case"]["start"] and now.get("case", {}).get("goal") == detail["case"]["goal"]
        contexts.append(dict(label=label, results_path=str(path), results_sha256=reader.inputs[str(path)],
                             paired_endpoints_and_seeds=paired, **aggregate(rows)))
    return contexts


def plot_trajectories(reader, protocol, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    jobs = protocol["jobs"]
    columns, rows = 2, math.ceil(len(jobs) / 2)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4.4 * rows), squeeze=False)
    rasters = {}
    colors = ("#2878b5", "#dc6c38", "#8447a4")
    for ax, job in zip(axes.ravel(), jobs):
        scene_path = str(resolve(job["scene_path"], reader.root))
        source = reader.scenes[scene_path]["source"]
        map_path = resolve(source["map_path"], Path(scene_path).parent)
        if str(map_path) not in rasters:
            with np.load(map_path, allow_pickle=False) as data:
                rasters[str(map_path)] = (data["occupied"].astype(bool), float(data["resolution_m"]),
                                          np.asarray(data["origin_xy"], dtype=float))
        occupied, resolution, origin = rasters[str(map_path)]
        case_rows = [r for r in reader.rows if r["case_id"] == job["case_id"]]
        details = [reader.details[r["id"]] for r in case_rows if r["id"] in reader.details]
        case = next((d["case"] for d in details if "case" in d), None)
        if case is None:
            manifest = read_json(resolve(job["case_manifest"], reader.root))
            case = next(c for c in manifest["cases"] if c["id"] == job["case_id"])
        points = [case["start"], case["goal"]]
        for detail in details:
            points.extend(frame["ground"][:2] for frame in detail.get("trajectory", []))
        points = np.asarray(points)
        padding = max(15., float(np.ptp(points, axis=0).max()) * .06)
        world_span = np.asarray(occupied.shape[::-1]) * resolution
        span = np.ptp(points, axis=0) + 2 * padding
        span[0] = max(span[0], 1.6 * span[1])
        span[1] = max(span[1], span[0] / 1.6)
        span = np.minimum(span, world_span)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        lower = np.minimum(np.maximum(center - span / 2, origin), origin + world_span - span)
        upper = lower + span
        ix0, iy0 = np.maximum(0, np.floor((lower - origin) / resolution).astype(int))
        ix1, iy1 = np.minimum(occupied.shape[::-1], np.ceil((upper - origin) / resolution).astype(int))
        ax.imshow(occupied[iy0:iy1, ix0:ix1], origin="lower", interpolation="nearest",
                  extent=(origin[0]+ix0*resolution, origin[0]+ix1*resolution,
                          origin[1]+iy0*resolution, origin[1]+iy1*resolution),
                  cmap=ListedColormap(["#faf8f0", "#9aa58c"]), vmin=0, vmax=1, rasterized=True)
        for index, row in enumerate(case_rows):
            detail = reader.details.get(row["id"], {})
            track = np.asarray([frame["ground"][:2] for frame in detail.get("trajectory", [])])
            if len(track):
                ax.plot(track[:, 0], track[:, 1], color=colors[index % len(colors)], linewidth=1.6,
                        alpha=.85, label=f"sample {row['sample']}")
                ax.plot(track[-1, 0], track[-1, 1], marker="x", color=colors[index % len(colors)], markersize=7)
        ax.scatter(*case["start"], s=65, c="#16834b", marker="o", edgecolors="white", zorder=6)
        ax.scatter(*case["goal"], s=130, c="#ba3444", marker="*", edgecolors="white", zorder=6)
        ax.annotate("S", case["start"], xytext=(5, 5), textcoords="offset points", fontweight="bold")
        ax.annotate("G", case["goal"], xytext=(5, 5), textcoords="offset points", fontweight="bold")
        title = job["case_id"]
        if job["case_id"].startswith("V_01_"):
            direction = "reverse" if job["case_id"].endswith("reverse") else "forward"
            title = f"V-01 short {direction} (~214 m budget reference)"
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
               xlabel="x (m)", ylabel="y (m)", title=title)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=.12)
    for ax in axes.ravel()[len(jobs):]:
        ax.axis("off")
        ax.text(.05, .8, "Measured occupied cells: grey-green\nAssumed free simulation background: ivory\n"
                "S: start   G: final goal   x: last recorded position\nThree preset random samples per task\n"
                "Views have task-specific extents.\nInherited routes are not new-scene shortest paths.\n"
                "Paths join samples every 5 steps, subgoal changes,\nand run endpoints. Safety audits use every executed\n"
                "motion segment; plotted chords alone do not\nestablish a collision.\n"
                "This is a simplified 2D simulation, not a field trial.", transform=ax.transAxes, fontsize=11, va="top")
    fig.suptitle("WildScenes-derived obstacle layouts | native 2D sensing | unknown space assumed free", fontsize=14)
    fig.tight_layout(rect=(0, .01, 1, .98))
    fig.savefig(destination, dpi=155, facecolor="white")
    plt.close(fig)


def number(value, decimals=1):
    return "—" if value is None else f"{value:.{decimals}f}"


def group_columns(group, primary_key):
    n = group["planned_runs"]
    return [group["label"], str(n), f"{group[primary_key]}/{n}", f"{group['arrivals']}/{n}",
            f"{group['collision_free_arrivals']}/{n}", f"{group['strict_successes']}/{n}",
            str(group["execution_errors"]),
            f"{group['physical_swept_violations_runs']} / {group['physical_swept_violations_total']}",
            f"{group['safety_clearance_violations_runs']} / {group['safety_clearance_violations_total']}"]


def row_columns(row, primary_metric):
    s = row["summary"]
    yes = lambda key: "是" if s.get(key) else "否"
    return [row["id"], str(row["sample_seed"]),
            TERMINATIONS.get(s.get("termination"), str(s.get("termination", row["status"]))),
            "是" if (row["physical_safety_success"] if primary_metric == "physical_safety_success" else s.get("strict_success")) else "否",
            yes("arrived"), yes("collision_free_arrival"), yes("strict_success"),
            str(s.get("episode_steps", "—")), number(s.get("ground_distance")),
            number(s.get("ground_goal_distance")), str(s.get("native_collision_attempts", "—")),
            str(s.get("physical_swept_violations", "—")), str(s.get("safety_clearance_violations", "—")),
            number(row.get("final_graph_nodes"), 0), number(row.get("max_graph_nodes"), 0),
                      number(row["budget_reference_length_m"])]


def table(headers, rows):
    th = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def publish_reports(audit, output):
    acceptance = audit["acceptance"]
    clearance_label = f"{audit.get('safety_clearance_m', 1.5):g}m"
    metric_note = audit.get("metric_contract", METRIC_NOTE)
    validated_routes = audit["geometry_context"].get("status") == "validated_reference_routes"
    geometry_title = "正常净空参考路线验证" if validated_routes else "附加诊断：1.5m净空几何"
    geometry_intro = ("本批任务在运行前均通过正常净空的逐段连续几何验证；不依据策略成败选取任务。"
                      if validated_routes else "下列证据来自独立1.5m几何审计，不属于策略运行结果，不决定物理安全主验收。")
    primary_key, primary_metric = acceptance["aggregate_key"], acceptance["primary_metric"]
    group_headers = ["任务组", "预定次数", acceptance["label"], "到达", "无碰撞到达", "原严格成功（含1.5m）", "运行错误", "物理违规：运行/步", "1.5m违规：运行/步"]
    detail_headers = ["运行", "随机种子", "终止原因", acceptance["label"], "到达", "无碰撞到达", "原严格成功（含1.5m）", "已执行步", "车程m", "剩余直线m",
                      "碰撞尝试", "物理违规步", "1.5m违规步", "最终节点", "最大节点", "预算参考m"]
    group_headers = [value.replace("1.5m", clearance_label) for value in group_headers]
    detail_headers = [value.replace("1.5m", clearance_label) for value in detail_headers]
    historical_headers = [value.replace(clearance_label, "1.5m") for value in group_headers]
    groups = [group_columns(g, primary_key) for g in audit["groups"]]
    groups.append(group_columns(dict(label="全计划合计（固定分母）", **audit["aggregate"]), primary_key))
    details = [row_columns(r, primary_metric) for r in audit["rows"]]
    context = [group_columns(g, primary_key) for g in audit.get("historical_context", [])]
    subsets = [group_columns(g, primary_key) for g in audit["geometry_context"].get("subsets", [])]
    sources = audit["geometry_context"].get("sources", [])
    failure = audit.get("failure_behavior", {})
    execution = audit.get("execution_history", {})
    network = audit.get("network_diagnostics", {})
    text = ["WildScenes障碍布局简化二维仿真：正式验证", "", SCOPE, "", acceptance["note"], "", metric_note, "",
            f"审计：{audit['status']}；预定{audit['expected_samples']}次，逐例结果全部纳入。", "",
            "\t".join(group_headers), *("\t".join(row) for row in groups), "", "逐例结果", "\t".join(detail_headers),
            *("\t".join(row) for row in details), "", "历史背景：原unknown阻挡、原1.5m设置，环境及配置不同", "\t".join(historical_headers),
            *("\t".join(row) for row in context), "",
            "各组终止原因：", *(f"{g['label']}: {json.dumps(g['terminations'], ensure_ascii=False)}" for g in audit["groups"]), "",
            *network.get("notes", []), "",
            "执行与恢复：", *execution.get("notes", []),
            *(f"{s['label']}: {s['path']} (SHA256 {s['sha256']})" for s in execution.get("sources", [])), "",
            "失败行为（补充诊断）：", *failure.get("notes", []),
            *([f"证据：{failure['source']} (SHA256 {failure['source_sha256']})", ""] if failure.get("source") else []),
            geometry_title, *audit["geometry_context"]["notes"], "",
            "附加几何分层（不替换全计划分母）：", "\t".join(group_headers),
            *("\t".join(row) for row in subsets), "",
            "数学证明与路线见证来源：", *(f"{s['label']}: {s['path']} (SHA256 {s['sha256']})" for s in sources), "",
            *( [f"验收标准修订：{acceptance['source']} (SHA256 {acceptance['source_sha256']})", ""] if acceptance.get("source") else []),
            "轨迹图为每5步及临时目标变更时采样并保留首尾的折线；物理/净空判定使用每一步真实运动段，图上连接线不能单独作为碰撞证据。",
            f"权重SHA256：{CHECKPOINT_SHA256}", "输入及结果核验详见audit.json；轨迹见trajectories.png。"]
    (output / "report.txt").write_text("\n".join(text) + "\n")
    image = base64.b64encode((output / "trajectories.png").read_bytes()).decode()
    outcomes = "；".join(f"{html.escape(g['label'])}：" + "、".join(
        f"{html.escape(TERMINATIONS.get(k, k))} {v}" for k, v in g["terminations"].items()) for g in audit["groups"])
    geometry = "".join(f"<p>{html.escape(note)}</p>" for note in audit["geometry_context"]["notes"])
    proof_links = " · ".join(f'<a href="{html.escape(os.path.relpath(s["path"], output), quote=True)}">'
                             f'{html.escape(s["label"])}</a>' for s in sources)
    amendment_link = (f'<p><a href="{html.escape(os.path.relpath(acceptance["source"], output), quote=True)}">'
                      '用户运行中请求的验收标准修订及公式</a></p>') if acceptance.get("source") else ""
    failure_section = ""
    if failure.get("notes"):
        failure_section = '<h2>失败行为</h2>' + "".join(f'<p>{html.escape(note)}</p>' for note in failure["notes"])
        failure_section += (f'<p><a href="{html.escape(os.path.relpath(failure["source"], output), quote=True)}">'
                            '逐例轨迹、终点失败事件及精确候选评分证据</a></p>')
    execution_section = ""
    if execution.get("notes"):
        execution_section = '<h2>执行与恢复</h2>' + "".join(f'<p>{html.escape(note)}</p>' for note in execution["notes"])
        execution_section += '<p>' + ' · '.join(
            f'<a href="{html.escape(os.path.relpath(s["path"], output), quote=True)}">{html.escape(s["label"])}</a>'
            for s in execution.get("sources", [])) + '</p>'
    network_section = ''.join(f'<p class="note">{html.escape(note)}</p>' for note in network.get("notes", []))
    proof = "".join(f"<li>{html.escape(note)}</li>" for note in audit["geometry_context"].get("mathematical_justification", []))
    geometry_extra = (f'<h3>附加几何分层</h3><p class="note">全计划21次分母保持不变；'
                      f'下表按1.5m净空证据分类，不依据策略结果筛选样本，不改变任何样本的物理安全主验收资格。</p>{table(group_headers, subsets)}') if subsets else ""
    geometry_extra += (f'<p>{proof_links}</p><details><summary>数学必要条件证明</summary><ol>{proof}</ol>'
                       '<p>对每个0.25m源栅格，格内任一点到occupied集合的净空不超过'
                       'EDT(格中心)+0.25/√2。保留上界达到1.5m的格后，任何安全连续路线必然对应一个8连通格链。'
                       '起点分量与2m终点盘相交的所有保留格不连通，因此不存在符合净空要求的正式到达路线。'
                       '证明范围限于本简化投影障碍几何。</p></details>') if proof else f'<p>{proof_links}</p>'
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WildScenes简化二维场景正式验证</title><style>
body{{margin:0;background:#f6f5ef;color:#26342f;font:15px/1.7 system-ui,-apple-system,"Microsoft YaHei",sans-serif}}
main{{max-width:1500px;margin:30px auto;padding:0 24px 50px}}h1{{font-size:29px;line-height:1.35}}h2{{margin-top:32px;font-size:20px}}
.notice{{border-left:4px solid #aa6b32;background:#fff5e4;padding:16px 20px}}.note{{color:#52625a}}
.scroll{{overflow:auto;background:white;border:1px solid #dbe0d5;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}}
td,th{{padding:10px 12px;border-bottom:1px solid #e5e8df;text-align:left}}th{{background:#eaf0e5}}tr:last-child td{{border-bottom:0}}
img{{width:100%;height:auto;background:white;border:1px solid #dbe0d5}}code{{overflow-wrap:anywhere}}a{{color:#206841}}
</style><main><p class="note">冻结v9 best576 · VTemp rolling200 · 三个独立任务组 · {audit['expected_samples']}次预定运行</p>
<h1>WildScenes障碍布局简化二维仿真：正式验证</h1><p class="notice">{html.escape(SCOPE)}</p>
<h2>结果与分母</h2><p class="notice">{html.escape(acceptance['note'])}</p>{amendment_link}<p>计划样本全部读取，原严格成功标记先核验，再从逐例summary派生主验收指标；运行错误仍在预定分母中。</p>
{table(group_headers, groups)}<p class="note">{outcomes}</p><p class="note">{html.escape(metric_note)}</p>
{network_section}
{execution_section}
{failure_section}
<h2>七个几何任务的车辆轨迹</h2><p class="note">每图三条预设种子轨迹；S为起点，G为最终目标，×为最后记录位置。背景仅表示简化场景中的空地假设。各子图范围不同。轨迹为每5步及临时目标变更时采样并保留首尾的折线；物理/净空判定来自每一步真实运动段，图上连接线不能单独作为碰撞证据。</p>
<img src="data:image/png;base64,{image}" alt="Seven task maps with measured obstacle layouts and three ground-vehicle trajectories per task">
<h2>逐例结果</h2>{table(detail_headers, details)}
<h2>{geometry_title}</h2><p class="note">{geometry_intro}</p>{geometry}{geometry_extra}
<h2>历史背景：原unknown阻挡实验</h2><p class="notice">下表属于原unknown阻挡环境和原净空配置。新任务可能重新选择起终点；实际配对数量见audit.json。可行空间、观测和任务难度不同，不能将两表差值归因于同环境算法提升。V-01没有对应旧实验。</p>
<p class="note">历史结果采用原1.5m规划和净空诊断设置。</p>{table(historical_headers, context)}
<h2>可核验记录</h2><p>审计状态：{audit['status']}。完整分母、协议、源地图、场景、权重和冻结代码散列核对结果保存在audit.json。PNG与TXT可独立读取；本HTML不依赖网络资源。</p>
<p class="note">权重 SHA256：<code>{CHECKPOINT_SHA256}</code></p>
<p><a href="audit.json">audit.json</a> · <a href="report.txt">report.txt</a> · <a href="trajectories.png">trajectories.png</a></p></main></html>'''
    (output / "report.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path,
                        default=REPOSITORY / "artifacts/wildscenes-v9-20261106/vtemp_rolling200")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reader = Reader(root)
    protocol, audit = reader.audit()
    write_json(output / "audit.json", audit)
    if audit["status"] != "complete":
        print(json.dumps(dict(status=audit["status"], expected_samples=audit["expected_samples"],
                              pending=audit["aggregate"]["pending_runs"], errors=audit["errors"],
                              audit=str(output / "audit.json")), ensure_ascii=False))
        return 2
    try:
        audit["historical_context"] = read_historical_context(reader, args.baseline_root.resolve())
    except (KeyError, ValueError, TypeError, OSError) as exc:
        audit["status"] = "invalid"
        audit["errors"].append(f"Historical context: {type(exc).__name__}: {exc}")
        write_json(output / "audit.json", audit)
        print(json.dumps(dict(status="invalid", errors=audit["errors"], audit=str(output / "audit.json")), ensure_ascii=False))
        return 2
    audit["warnings"] = reader.warnings
    plot_trajectories(reader, protocol, output / "trajectories.png")
    publish_reports(audit, output)
    audit["output_sha256"] = {name: digest(output / name) for name in ("report.txt", "report.html", "trajectories.png")}
    write_json(output / "audit.json", audit)
    print(json.dumps(dict(status="complete", expected_samples=audit["expected_samples"],
                          report=str(output / "report.html"), audit=str(output / "audit.json")), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
