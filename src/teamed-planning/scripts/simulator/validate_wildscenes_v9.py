"""Evaluate frozen v9 weights on a recorded WildScenes raster map.

The raster is visible only to the simulator's local sensor/collision adapter.
Reference paths are evaluation metadata and never enter controllers or policy
observations. No weights, training sources, or training configuration are edited.
"""
from __future__ import annotations

import os

for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import argparse
import concurrent.futures
import contextlib
import hashlib
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path
import re
import sys
import time
import traceback
from datetime import datetime, timezone


REPOSITORY = Path(__file__).resolve().parents[4]
DEFAULT_RUN = REPOSITORY / "artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146"
PROTOCOL = "wildscenes_frozen_v9_raster_navigation_v1"


def load_clearance_adapter():
    path = Path(__file__).with_name("wildscenes_clearance.py")
    spec = importlib.util.spec_from_file_location("_wildscenes_clearance_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


clearance = load_clearance_adapter()


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


def load_adapter():
    # Avoid importing the mutable working-tree simulator package. The policy and
    # all its dependent simulator/drone/ground modules come from runtime_sources.
    path = Path(__file__).with_name("wildscenes_map.py")
    spec = importlib.util.spec_from_file_location("_wildscenes_v9_map_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def experiment_options(config, schedule, world, case):
    """Match frozen train_mappo.run_episode's constructor parameters."""
    direct = (
        "drop", "exploration_candidates", "communication_mode", "safety_filter",
        "spacing_penalty_weight", "intervention_weight", "moving_ground", "reward_mode",
        "ground_speed", "path_reward_scale", "subgoal_radius", "subgoal_arrival",
        "route_reward_weight", "time_penalty", "ground_stationary", "chain_soft_ratio",
        "chain_risk_weight", "chain_outward_weight", "chain_missing_weight",
        "path_quality_weight", "cbf_margin_weight", "cbf_margin_scale",
        "frontier_progress_weight", "chain_ground_range", "chain_link_range",
        "relay_loss", "relay_delay_jitter", "relay_period", "relay_capacity",
        "fusion_bridge_distance",
    )
    options = {key: config[key] for key in direct}
    if "ground_safety_distance" in config:
        options["ground_safety_distance"] = config["ground_safety_distance"]
    options.update(
        exploration_weight=schedule["eta"],
        mapping_until_connected=config["mapping_reward_until_connected"],
        mapping_route_origin=config.get("mapping_route_origin", "initial"),
        num_drones=config["fleet_sizes"][case["scene_seed"] % len(config["fleet_sizes"])],
        world_size=world, start=case["start"], scene=(case["goal"], [], 1),
        stop_on_ground_arrival=False,
    )
    if options["subgoal_radius"] is not None or options["mapping_route_origin"] != "vehicle":
        raise ValueError("Expected original v9 vehicle-origin mapping without subgoals")
    return options


def position_record(env, step):
    import numpy as np

    return dict(
        step=step,
        ground=np.asarray(env.obs[env.ground]["self"]["position"], dtype=float).tolist(),
        uavs=[np.asarray(env.obs[rid]["self"]["position"], dtype=float).tolist() for rid in env.ids],
    )


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


def evaluate_job(job):
    import numpy as np
    import torch

    sources = Path(job["sources"])
    sys.path.insert(0, str(sources))
    from simulator import mappo_experiment as me
    from simulator import train_mappo as tm

    for module in (me, tm):
        if not Path(module.__file__).resolve().is_relative_to(sources.resolve()):
            raise RuntimeError("A mutable simulator module was imported instead of frozen v9 sources")
    torch.set_num_threads(1)
    adapter = load_adapter()
    forest = adapter.RasterForestMap.from_npz(job["map_path"])
    case = job["case"]
    config = json.loads(Path(job["config_path"]).read_text())
    checkpoint = torch.load(job["checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint.get("best_episode") != 576 or checkpoint.get("reward_contract") != "uav_remaining_route_mapping_v9":
        raise ValueError("Checkpoint is not the requested frozen v9 best576 model")
    if digest(job["checkpoint"]) != job["checkpoint_sha256"]:
        raise ValueError("Checkpoint bytes changed before evaluation")
    # Original validator seeds before model reconstruction. Keep that RNG order.
    torch.manual_seed(job["sample_seed"])
    np.random.seed(job["sample_seed"])
    model = tm.ActorCritic.from_state_dict(checkpoint["model"])
    model.eval()
    schedule = tm.fixed_schedule(config, config["episodes"] - 1)
    safety_distance, escape_distance = clearance.job_distances(job)
    clearance.verify_sources(job.get("clearance_source_hashes"))
    options = clearance.experiment_options(me,
        experiment_options(config, schedule, job["world_size_m"], case), safety_distance, escape_distance)
    start, goal = np.asarray(case["start"], dtype=float), np.asarray(case["goal"], dtype=float)
    budget = max(job["min_steps"], math.ceil(job["budget_multiplier"] * case["reference_length_m"] + job["budget_extra"]))
    if job["max_steps"] is not None:
        budget = min(budget, job["max_steps"])
    if not forest.point_clear(start, clearance=.3) or not forest.point_clear(goal, clearance=.3):
        raise ValueError("Task endpoint does not fit the physical 0.3m vehicle in known free space")
    output = Path(job["output"])
    original_env = me.MultiRobotEnv

    def raster_environment(config):
        result = original_env(config)
        adapter.install_raster_environment(result, forest)
        return result

    # Install before Experiment calls reset and generates its first observation.
    me.MultiRobotEnv = raster_environment
    env = None
    started = time.monotonic()
    trajectory = []
    physical_violations = 0
    clearance_violations = 0
    violation_events = []
    termination = "budget_exhausted"
    caught_error = None
    safety_terminated = False
    attempted_steps = 0
    logfile = output / "logs" / (job["id"] + ".log")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        with clearance.planning_runtime(me, safety_distance, escape_distance), \
                logfile.open("w") as log, contextlib.redirect_stdout(log), torch.inference_mode():
            env = me.Experiment(case["scene_seed"], model, stochastic=job["stochastic"], **options)
            planning_clearances = clearance.verify_experiment(env, safety_distance, escape_distance)
            physical_radius = float(env.env.ground_robots[0].radius)
            trajectory.append(position_record(env, 0))
            previous = np.asarray(trajectory[-1]["ground"][:2])
            for frame in range(budget):
                if (output / "STOP").exists():
                    termination = "user_stop"
                    break
                attempted_steps = frame + 1
                try:
                    env.step(frame)
                except me.SafetyTermination as exc:
                    termination = "cbf_infeasible"
                    safety_terminated = True
                    caught_error = str(exc)
                    break
                current = np.asarray(env.obs[env.ground]["self"]["position"][:2], dtype=float)
                physical_clear = forest.segment_clear(previous, current, clearance=physical_radius)
                safety_clear = forest.segment_clear(previous, current, clearance=safety_distance)
                physical_violations += int(not physical_clear)
                clearance_violations += int(not safety_clear)
                if not physical_clear or not safety_clear:
                    violation_events.append(dict(step=frame + 1, start=previous.tolist(), end=current.tolist(),
                        physical_violation=not physical_clear, safety_clearance_violation=not safety_clear))
                previous = current.copy()
                if (frame + 1) % job["trace_interval"] == 0:
                    trajectory.append(position_record(env, frame + 1))
                if frame % 25 == 0 or frame + 1 == budget:
                    write_json(output / "progress" / (job["id"] + ".json"), dict(
                        status="running", id=job["id"], updated_utc=utc(), step=frame + 1, budget=budget,
                        ground_remaining_m=float(np.linalg.norm(current - goal)),
                        ground_distance_m=float(sum(t["ground_distance"] for t in env.trace)),
                        raster_collision_attempts=int(env.env.raster_collision_attempts),
                        physical_swept_violations=physical_violations,
                        safety_clearance_violations=clearance_violations,
                        graph_connected=bool(env.trace[-1]["graph_task_connected"]),
                        wall_seconds=round(time.monotonic() - started, 2)))
                if not physical_clear:
                    termination = "physical_geometry_violation"
                    break
                if job["stop_on_attempted_collision"] and env.env.raster_collision_attempts:
                    termination = "attempted_ground_collision"
                    break
                if env.goal_reached:
                    termination = "vehicle_arrival"
                    break
            if trajectory[-1]["step"] != len(env.trace):
                trajectory.append(position_record(env, len(env.trace)))
            planning_clearances = clearance.verify_experiment(env, safety_distance, escape_distance)
            summary = env.summary() if env.trace else dict(
                episode_steps=0, goal_reached=0., ground_distance=0.,
                ground_goal_distance=float(np.linalg.norm(goal - start)), ground_collisions=0.,
                chain_link_violation=0., spacing_violation=0., out_of_bounds=0.)
            correct_raster_calibration(summary, env, forest)
            if env.trace and termination == "vehicle_arrival":
                summary["reward"] += config["arrival_reward"]
            elif env.trace and safety_terminated:
                summary["reward"] -= config["safety_failure_penalty"]
            collisions = int(env.env.raster_collision_attempts)
            arrived = bool(env.goal_reached)
            collision_free_arrival = arrived and collisions == 0 and physical_violations == 0
            strict_success = (collision_free_arrival and clearance_violations == 0
                              and summary.get("chain_link_violation", 0.) == 0
                              and summary.get("spacing_violation", 0.) == 0
                              and summary.get("out_of_bounds", 0.) == 0 and not safety_terminated)
            summary.update(
                arrived=arrived, collision_free_arrival=collision_free_arrival, strict_success=strict_success,
                physical_swept_violations=physical_violations,
                safety_clearance_violations=clearance_violations,
                raster_collision_attempts=collisions,
                raster_unknown_attempts=int(env.env.raster_unknown_attempts),
                safety_terminated=safety_terminated, budget_exhausted=termination == "budget_exhausted",
                user_stopped=termination == "user_stop", termination=termination,
                episode_budget_steps=budget, attempted_steps=attempted_steps,
                task_distance_m=float(np.linalg.norm(goal - start)),
                reference_length_m=case["reference_length_m"],
                actual_route_over_reference=(summary.get("ground_distance", 0.) / case["reference_length_m"]
                                             if arrived else None),
                max_link_distance_m=summary.get("chain_max_distance_ratio", 0.) * 95.,
                wall_seconds=round(time.monotonic() - started, 3),
                safety_clearance_m=safety_distance, planning_clearances=planning_clearances, physical_radius_m=physical_radius)
            result = dict(
                id=job["id"], case_id=case["id"], case=case, sample=job["sample"],
                sample_seed=job["sample_seed"], scene_seed=case["scene_seed"], stochastic=job["stochastic"],
                status="stopped" if termination == "user_stop" else "completed", protocol=PROTOCOL,
                signature=job["signature"], completed_utc=utc(),
                checkpoint=job["checkpoint"], checkpoint_sha256=job["checkpoint_sha256"],
                checkpoint_episode=int(checkpoint["training_episodes"]),
                summary=summary, trajectory=trajectory, trace=env.trace,
                raster_collision_events=env.env.raster_collision_events,
                swept_violation_events=violation_events, error=caught_error,
                map_path=job["map_path"], world_size_m=job["world_size_m"],
                evaluation_overrides=dict(world_size_m=job["world_size_m"], map="real_recorded_raster",
                    ground_safety_distance=safety_distance, ground_escape_distance=escape_distance,
                    planning_clearances=planning_clearances,
                    step_cap=job["max_steps"], episode_budget_steps=budget,
                    budget_multiplier=job["budget_multiplier"], budget_extra=job["budget_extra"],
                    subgoal_radius=None, reference_path_in_policy=False),
            )
            if digest(job["checkpoint"]) != job["checkpoint_sha256"]:
                raise RuntimeError("Checkpoint bytes changed during evaluation")
            write_json(output / "cases" / (job["id"] + ".json"), result)
            write_json(output / "progress" / (job["id"] + ".json"), dict(
                id=job["id"], status=result["status"], completed_utc=utc(), **summary))
            return {key: result[key] for key in ("id", "case_id", "sample", "sample_seed", "status", "summary")}
    finally:
        me.MultiRobotEnv = original_env
        if env is not None:
            env.env.close()
        clearance.verify_sources(job.get("clearance_source_hashes"))


def safe_job(job):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="Frozen task JSON manifest")
    parser.add_argument("--map", type=Path, help="NPZ; otherwise cases.map_npz relative to manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--case", action="append", dest="case_ids", help="Select a named case (repeatable)")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--sample-seeds", type=int, nargs="+", help="Explicit frozen torch sample seed list")
    parser.add_argument("--seed-base", type=int, default=92000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--budget-multiplier", type=float, default=3.)
    parser.add_argument("--budget-extra", type=int, default=240)
    parser.add_argument("--min-steps", type=int, default=600)
    parser.add_argument("--max-steps", type=int, help="Explicit cap for smoke checks; omitted for full validation")
    parser.add_argument("--ground-safety-distance", "--safety-clearance", dest="ground_safety_distance",
                        type=float, default=clearance.DEFAULT_GROUND_CLEARANCE_M,
                        help="Ground planner, graph, bridge and audit clearance in metres (default 1); escape remains 0.5m")
    parser.add_argument("--trace-interval", type=int, default=5)
    parser.add_argument("--stop-on-attempted-collision", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse only completed cases with matching provenance")
    args = parser.parse_args()
    if min(args.samples, args.workers, args.trace_interval, args.min_steps) < 1:
        parser.error("samples, workers, trace-interval, min-steps must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    if args.budget_multiplier <= 0 or args.budget_extra < 0 or args.ground_safety_distance < 0:
        parser.error("invalid budget or clearance")
    clearance.distances(args.ground_safety_distance)
    import numpy as np

    cases_path = args.cases.resolve()
    manifest = json.loads(cases_path.read_text())
    map_path = (args.map.resolve() if args.map is not None else
                (cases_path.parent / manifest["map_npz"]).resolve())
    run = args.run.resolve()
    sources = run / "runtime_sources"
    checkpoint = run / "best.pt"
    config_path = run / "config.json"
    config = json.loads(config_path.read_text())
    world = float(manifest["world_size_m"])
    if not math.isfinite(world) or world <= 0:
        raise ValueError("Invalid world_size_m")
    cases = []
    identifiers = set()
    for index, original in enumerate(manifest["cases"]):
        case = dict(original)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(case["id"])) or case["id"] in identifiers:
            raise ValueError("Case IDs must be unique and safe file names")
        identifiers.add(case["id"])
        case["scene_seed"] = int(case.get("scene_seed", case.get("seed", args.seed_base + index)))
        if args.case_ids and case["id"] not in args.case_ids:
            continue
        if not math.isfinite(case["reference_length_m"]) or case["reference_length_m"] <= 0:
            raise ValueError("A positive, finite reference_length_m is required")
        for name in ("start", "goal"):
            point = np.asarray(case[name], dtype=float)
            if point.shape != (2,) or not np.isfinite(point).all() or (point < 0).any() or (point > world).any():
                raise ValueError(f"Invalid {name} for {case['id']}")
        cases.append(case)
    if args.case_ids and set(args.case_ids) - identifiers:
        raise ValueError("Unknown case selection")
    if not cases:
        raise ValueError("No cases selected")
    source_hashes = {str(p.relative_to(sources)): digest(p) for p in sorted(sources.rglob("*.py"))}
    if not source_hashes:
        raise ValueError("Missing frozen v9 runtime_sources")
    settings = dict(stochastic=not args.deterministic, budget_multiplier=args.budget_multiplier,
        budget_extra=args.budget_extra, min_steps=args.min_steps, max_steps=args.max_steps,
        safety_clearance_m=args.ground_safety_distance, ground_safety_distance=args.ground_safety_distance,
        ground_escape_distance=.5, trace_interval=args.trace_interval,
        stop_on_attempted_collision=args.stop_on_attempted_collision,
        samples=args.samples if args.sample_seeds is None else len(args.sample_seeds),
        sample_seeds=args.sample_seeds, seed_base=args.seed_base)
    provenance = dict(protocol=PROTOCOL, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        config_path=str(config_path), config_sha256=digest(config_path), sources=str(sources),
        source_hashes=source_hashes, clearance_source_hashes=clearance.source_hashes(), cases_path=str(cases_path), cases_sha256=digest(cases_path),
        map_path=str(map_path), map_sha256=digest(map_path), runner_sha256=digest(__file__),
        adapter_sha256=digest(Path(__file__).with_name("wildscenes_map.py")), settings=settings,
        selected_case_ids=[c["id"] for c in cases], metadata=manifest.get("metadata", {}),
        policy_contract="Frozen v9 best576; local raster sensors; no reference path or full map in policy",
        success_contract="Arrival plus zero attempted collisions, physical/safety swept violations, link/spacing violations and boundary violations",
        geometry_scope="2D raster ground navigation; existing idealized aerial mapping/flight and communication models retained")
    signature = object_digest(provenance)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prior_path = output / "manifest.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        if not args.resume or prior.get("signature") != signature:
            raise ValueError("Output already exists; use a new directory or --resume with identical inputs")
    jobs = []
    for case in cases:
        seeds = (args.sample_seeds if args.sample_seeds is not None else [
            int(np.random.SeedSequence([config["seed"], case["scene_seed"], sample]).generate_state(1)[0])
            for sample in range(args.samples)])
        for sample, sample_seed in enumerate(seeds):
            if not 0 <= sample_seed < 2 ** 32:
                raise ValueError("Sample seeds must be uint32")
            jobs.append(dict(id=f"{case['id']}_r{sample}", case=case, sample=sample, sample_seed=int(sample_seed),
                checkpoint=str(checkpoint), checkpoint_sha256=provenance["checkpoint_sha256"],
                config_path=str(config_path), sources=str(sources), map_path=str(map_path),
                output=str(output), world_size_m=world, signature=signature,
                clearance_source_hashes=provenance["clearance_source_hashes"], **settings))
    report = dict(status="running", created_utc=utc(), signature=signature, provenance=provenance,
        scheduled=[dict(id=j["id"], case_id=j["case"]["id"], sample=j["sample"], sample_seed=j["sample_seed"],
                        scene_seed=j["case"]["scene_seed"]) for j in jobs], records=[])
    write_json(prior_path, report)
    remaining = []
    for job in jobs:
        previous_path = output / "cases" / (job["id"] + ".json")
        if args.resume and previous_path.exists():
            previous = json.loads(previous_path.read_text())
            if previous.get("signature") != signature:
                raise ValueError(f"Mismatched case provenance: {job['id']}")
            if previous["status"] == "completed":
                report["records"].append({key: previous[key] for key in
                    ("id", "case_id", "sample", "sample_seed", "status", "summary")})
                continue
        remaining.append(job)

    def collect(record):
        report["records"].append(record)
        report["records"].sort(key=lambda row: row["id"])
        report.update(updated_utc=utc(), aggregate=summarize(report["records"], len(jobs)))
        write_json(output / "results.json", report)
        fields = ("termination", "arrived", "strict_success", "episode_steps",
                  "ground_distance", "ground_goal_distance", "raster_collision_attempts",
                  "physical_swept_violations", "safety_clearance_violations", "wall_seconds")
        print(json.dumps(clean(dict(id=record["id"], status=record["status"],
            **{key: record["summary"].get(key) for key in fields})), ensure_ascii=False), flush=True)

    if args.workers == 1:
        for job in remaining:
            if (output / "STOP").exists():
                break
            collect(safe_job(job))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(safe_job, job): job for job in remaining}
            for future in concurrent.futures.as_completed(futures):
                collect(future.result())
    report.update(updated_utc=utc(), aggregate=summarize(report["records"], len(jobs)))
    report["status"] = ("stopped" if len(report["records"]) < len(jobs) or
                        any(r["status"] == "stopped" for r in report["records"]) else
                        "completed_with_errors" if any(r["status"] == "error" for r in report["records"]) else "completed")
    write_json(output / "results.json", report)
    write_json(prior_path, report)
    print(json.dumps(dict(status=report["status"], aggregate=report["aggregate"])), flush=True)
    return 1 if report["status"] == "completed_with_errors" else 0


if __name__ == "__main__":
    raise SystemExit(main())
