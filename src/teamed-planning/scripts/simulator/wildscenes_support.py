"""Shared WildScenes report I/O and frozen-raster evaluation helpers.

This module loads only standard-library dependencies at import time. Raster
adapters are loaded by file path so policy modules can still come exclusively
from the selected immutable runtime.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import traceback


def utc():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_clearance_adapter():
    path = Path(__file__).with_name("wildscenes_clearance.py")
    spec = importlib.util.spec_from_file_location("_wildscenes_clearance_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_adapter():
    # Avoid importing the mutable working-tree simulator package. The policy and
    # all its dependent simulator/drone/ground modules come from runtime_sources.
    path = Path(__file__).with_name("wildscenes_map.py")
    spec = importlib.util.spec_from_file_location("_wildscenes_v9_map_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def correct_raster_calibration(summary, env, forest):
    """Replace circle-only calibration with audit against known raster cells."""
    import numpy as np

    points = env.ledger.xy[env.ledger.seen]
    known = forest.known_at(points)
    probability = 1 / (1 + np.exp(-env.ledger.log_odds[env.ledger.seen]))
    probability = probability[known]
    truth = forest.truth(points[known])
    summary.pop("brier_error", None)
    summary.pop("calibration_error", None)
    summary["raster_known_evaluation_points"] = int(known.sum())
    summary["raster_unknown_evaluation_points"] = int((~known).sum())
    summary["brier_error"] = float(np.mean((probability - truth) ** 2)) if len(truth) else None
    certainty = np.maximum(probability, 1 - probability)
    correct = ((probability >= .5) == truth).astype(float)
    error = 0.
    for lower in np.arange(.5, 1., .05):
        selected = (certainty >= lower) & (certainty < lower + .05)
        if selected.any():
            error += float(selected.mean() * abs(certainty[selected].mean() - correct[selected].mean()))
    summary["calibration_error"] = error if len(truth) else None


def summarize(records, expected):
    completed = [r for r in records if r["status"] == "completed"]
    counts = {}
    for row in records:
        term = row["summary"]["termination"]
        counts[term] = counts.get(term, 0) + 1
    # Explicit denominators: execution errors are not silently dropped from the
    # scheduled set, while completed-rollout counts remain separately visible.
    return dict(scheduled_runs=expected, recorded_runs=len(records), completed_rollouts=len(completed),
        execution_errors=sum(r["status"] == "error" for r in records),
        arrivals=sum(bool(r["summary"].get("arrived")) for r in records),
        collision_free_arrivals=sum(bool(r["summary"].get("collision_free_arrival")) for r in records),
        strict_successes=sum(bool(r["summary"].get("strict_success")) for r in records),
        arrival_rate=sum(bool(r["summary"].get("arrived")) for r in records) / expected,
        strict_success_rate=sum(bool(r["summary"].get("strict_success")) for r in records) / expected,
        terminations=counts)


def safe_evaluate(job, evaluate_job):
    try:
        return evaluate_job(job)
    except Exception as exc:
        result = dict(id=job["id"], case_id=job["case"]["id"], sample=job["sample"],
            sample_seed=job["sample_seed"], status="error", signature=job["signature"],
            completed_utc=utc(), error=str(exc), traceback=traceback.format_exc(),
            summary=dict(arrived=False, collision_free_arrival=False, strict_success=False,
                         termination="evaluation_error"))
        write_json(Path(job["output"]) / "cases" / (job["id"] + ".json"), result)
        write_json(Path(job["output"]) / "progress" / (job["id"] + ".json"), result)
        return result
