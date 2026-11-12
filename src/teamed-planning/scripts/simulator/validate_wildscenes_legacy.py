"""Run frozen VTemp + v9 in native 2D scenes derived from WildScenes obstacles.

Only the recorded obstacle layout is retained. Unrecorded space is assumed free
inside the simulation world, so these are simplified synthetic scenes, not
navigation tests on the original conservative real-data map.
"""
from __future__ import annotations

import os

for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import argparse
import contextlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback


REPOSITORY = Path(__file__).resolve().parents[4]
DEFAULT_RUN = REPOSITORY / "artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146"
PROTOCOL = "wildscenes_obstacle_layout_native2d_unknown_free_vtemp_v9_v1"
BEST576_SHA256 = "7ac3aedcc3876902773b2241f60e839eb3e3b5630ddeb3c8be660421f2c24198"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load_file("_wildscenes_legacy_shared", Path(__file__).with_name("validate_wildscenes_vtemp.py"))
clean, digest, object_digest = common.clean, common.digest, common.object_digest
write_json, utc = common.write_json, common.utc
clearance = common.clearance


def load_scene_adapter():
    return load_file("_wildscenes_legacy_geometry", Path(__file__).with_name("wildscenes_legacy_scene.py"))


def frozen_hashes(run):
    run = Path(run)
    files = [run / "best.pt", run / "config.json", *sorted((run / "runtime_sources").rglob("*.py"))]
    if len(files) == 2:
        raise ValueError("Missing frozen runtime_sources")
    return {str(path.relative_to(run)): digest(path) for path in files}


def verify_frozen_inputs(run, expected):
    current = frozen_hashes(run)
    if current != expected:
        changed = sorted(key for key in set(current) | set(expected) if current.get(key) != expected.get(key))
        raise RuntimeError("Frozen inputs changed: " + ", ".join(changed))


def verify_job_inputs(job):
    verify_frozen_inputs(job["run"], job["frozen_hashes"])
    clearance.verify_sources(job.get("clearance_source_hashes"))
    for key in ("scene", "cases"):
        if key + "_sha256" in job and digest(job[key + "_path"]) != job[key + "_sha256"]:
            raise RuntimeError(f"{key} input changed during evaluation")


def paired_source_hash(manifest, scene, cases_path):
    """Reject a task manifest from a different source map, even at equal size."""
    expected = manifest.get("metadata", {}).get("map_sha256")
    actual = scene.metadata.get("source", {}).get("map_sha256")
    if expected is not None and expected != actual:
        raise ValueError("Task and scene source map hashes differ")
    relative_map = manifest.get("map_npz")
    if relative_map:
        source_path = (Path(cases_path).parent / relative_map).resolve()
        if source_path.exists():
            measured = digest(source_path)
            if actual != measured or (expected is not None and expected != measured):
                raise ValueError("Task source map file does not match the exported scene")
    return actual


@contextlib.contextmanager
def native_runtime(me, scene, adapter=None):
    """Inject native obstacles before reset; retain native sensor/controller code."""
    adapter = load_scene_adapter() if adapter is None else adapter
    original_env = me.MultiRobotEnv

    def environment(config):
        env = original_env(config)
        adapter.install_legacy_environment(env, scene)
        return env

    try:
        me.MultiRobotEnv = environment
        yield
    finally:
        me.MultiRobotEnv = original_env


def legacy_summary(experiment, scene):
    """Reuse frozen aggregation, then replace its circle-only calibration.

    The frozen summary's only obstacle access is its evaluator calibration loop.
    Temporarily omit geometry solely during aggregation, restore the original
    list even on errors, and replace both calibration fields using native scene
    truth. No action, observation, reward or physics step runs in this interval.
    """
    obstacles = experiment.env.obstacles
    try:
        experiment.env.obstacles = []
        summary = experiment.summary()
    finally:
        experiment.env.obstacles = obstacles
    common.correct_raster_calibration(summary, experiment, scene)
    summary["geometry_evaluation_points"] = summary.pop("raster_known_evaluation_points")
    summary["geometry_outside_evaluation_points"] = summary.pop("raster_unknown_evaluation_points")
    return summary


def evaluate_job(job):
    import numpy as np
    import torch

    run = Path(job["run"])
    verify_job_inputs(job)
    sources = run / "runtime_sources"
    sys.path.insert(0, str(sources))
    from simulator import mappo_experiment as me
    from simulator import train_mappo as tm

    for module in (me, tm):
        if not Path(module.__file__).resolve().is_relative_to(sources.resolve()):
            raise RuntimeError("Mutable runtime imported; run this evaluator in a fresh process")
    adapter = load_scene_adapter()
    scene = adapter.LegacyObstacleScene.from_json(job["scene_path"])
    if not np.allclose(scene.origin_xy, (0., 0.), rtol=0., atol=1e-9):
        raise ValueError("Native scene must use the zero simulation origin")
    config = json.loads((run / "config.json").read_text())
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if (checkpoint.get("best_episode") != 576 or
            checkpoint.get("reward_contract") != "uav_remaining_route_mapping_v9" or
            job["frozen_hashes"]["best.pt"] != BEST576_SHA256):
        raise ValueError("This evaluator requires frozen v9 best576")
    torch.set_num_threads(1)
    torch.manual_seed(job["sample_seed"])
    np.random.seed(job["sample_seed"])
    model = tm.ActorCritic.from_state_dict(checkpoint["model"])
    model.eval()
    schedule = tm.fixed_schedule(config, config["episodes"] - 1)
    case = job["case"]
    safety_distance, escape_distance = clearance.job_distances(job)
    options = clearance.experiment_options(me,
        common.experiment_options(config, schedule, scene.world_size_m, case), safety_distance, escape_distance)
    start, goal = np.asarray(case["start"], float), np.asarray(case["goal"], float)
    for point in (start, goal):
        if not scene.point_clear(point, clearance=.3):
            raise ValueError("Task endpoint intersects simplified scene geometry at 0.3m clearance")
    budget = max(600, math.ceil(3. * case["reference_length_m"] + 240))
    if job["max_steps"] is not None:
        budget = min(budget, job["max_steps"])
    output = Path(job["output"])
    logfile = output / "logs" / (job["id"] + ".log")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = None
    trajectory, events, violations = [], [], []
    physical_count = safety_count = attempted_steps = 0
    termination, caught_error = "budget_exhausted", None
    try:
        with native_runtime(me, scene, adapter), \
                clearance.planning_runtime(me, safety_distance, escape_distance), logfile.open("w") as log, \
                contextlib.redirect_stdout(log), torch.inference_mode():
            env = me.Experiment(case["scene_seed"], model, stochastic=job["stochastic"], **options)
            planning_clearances = clearance.verify_experiment(env, safety_distance, escape_distance)
            if env.subgoals is None:
                raise RuntimeError("VTemp manager is missing")
            physical_radius = float(env.env.ground_robots[0].radius)
            trajectory.append(common.position_record(env, 0))
            events.append(common.subgoal_event(env, 0))
            previous = start.copy()
            for frame in range(budget):
                if (output / "STOP").exists():
                    termination = "user_stop"
                    break
                attempted_steps = frame + 1
                try:
                    env.step(frame)
                except me.SafetyTermination as exc:
                    termination, caught_error = "cbf_infeasible", str(exc)
                    break
                event = common.subgoal_event(env, frame + 1, events[-1])
                if event is not None:
                    events.append(event)
                current = np.asarray(env.obs[env.ground]["self"]["position"][:2], float)
                physical_clear = scene.segment_clear(previous, current, clearance=physical_radius)
                safety_clear = scene.segment_clear(previous, current, clearance=safety_distance)
                physical_count += int(not physical_clear)
                safety_count += int(not safety_clear)
                if not physical_clear or not safety_clear:
                    violations.append(dict(step=frame + 1, start=previous.tolist(), end=current.tolist(),
                        physical_violation=not physical_clear, safety_clearance_violation=not safety_clear))
                previous = current.copy()
                if (frame + 1) % 5 == 0 or event is not None:
                    trajectory.append(common.position_record(env, frame + 1))
                if frame % 25 == 0 or frame + 1 == budget:
                    write_json(output / "progress" / (job["id"] + ".json"), dict(
                        id=job["id"], status="running", step=frame + 1, budget=budget,
                        temporary_goal=env.subgoals.target.tolist(),
                        ground_remaining_m=float(np.linalg.norm(current - goal)),
                        graph_nodes=float(env.trace[-1].get("graph_nodes", 0.)),
                        ground_distance_m=float(sum(t["ground_distance"] for t in env.trace)),
                        native_collision_attempts=int(env.env.legacy_collision_attempts),
                        physical_swept_violations=physical_count, safety_clearance_violations=safety_count,
                        wall_seconds=round(time.monotonic() - started, 2)))
                # Native endpoint collision handling is unchanged. Independent
                # geometry stops a rollout that tunnels through a thin obstacle.
                if not physical_clear:
                    termination = "physical_geometry_violation"
                    break
                if env.goal_reached:
                    termination = "vehicle_arrival"
                    break
            if trajectory[-1]["step"] != len(env.trace):
                trajectory.append(common.position_record(env, len(env.trace)))
            planning_clearances = clearance.verify_experiment(env, safety_distance, escape_distance)
            summary = legacy_summary(env, scene) if env.trace else dict(
                episode_steps=0, goal_reached=0., ground_distance=0., ground_collisions=0.,
                ground_goal_distance=float(np.linalg.norm(goal - start)),
                chain_link_violation=0., spacing_violation=0., out_of_bounds=0.)
            if env.trace and termination == "vehicle_arrival":
                summary["reward"] += config["arrival_reward"]
            elif env.trace and termination == "cbf_infeasible":
                summary["reward"] -= config["safety_failure_penalty"]
            collisions = int(env.env.legacy_collision_attempts)
            arrived = bool(env.goal_reached)
            collision_free = arrived and collisions == 0 and physical_count == 0
            strict = (termination == "vehicle_arrival" and collision_free and safety_count == 0 and
                      summary.get("chain_link_violation", 0.) == 0 and
                      summary.get("spacing_violation", 0.) == 0 and summary.get("out_of_bounds", 0.) == 0)
            summary.update(arrived=arrived, collision_free_arrival=collision_free, strict_success=strict,
                termination=termination, safety_terminated=termination == "cbf_infeasible",
                budget_exhausted=termination == "budget_exhausted", user_stopped=termination == "user_stop",
                episode_budget_steps=budget, attempted_steps=attempted_steps,
                native_collision_attempts=collisions, physical_swept_violations=physical_count,
                safety_clearance_violations=safety_count, safety_clearance_m=safety_distance,
                planning_clearances=planning_clearances,
                physical_radius_m=physical_radius, task_distance_m=float(np.linalg.norm(goal - start)),
                inherited_reference_length_m=case["reference_length_m"],
                wall_seconds=round(time.monotonic() - started, 3))
            result = dict(id=job["id"], case_id=case["id"], case=case, sample=job["sample"],
                sample_seed=job["sample_seed"], scene_seed=case["scene_seed"], stochastic=job["stochastic"],
                status="stopped" if termination == "user_stop" else "completed", protocol=PROTOCOL,
                signature=job["signature"], completed_utc=utc(), simplified_scene=True,
                unknown_space_assumption="free inside world; does not establish real traversability",
                scene_path=job["scene_path"], world_size_m=scene.world_size_m,
                checkpoint=str(run / "best.pt"), checkpoint_sha256=job["frozen_hashes"]["best.pt"],
                summary=summary, trace=env.trace, trajectory=trajectory, subgoal_events=events,
                native_collision_events=env.env.legacy_collision_events,
                swept_violation_events=violations, error=caught_error,
                evaluation_overrides=dict(strategy="vtemp_rolling200", subgoal_radius=200.,
                    ground_safety_distance=safety_distance, ground_escape_distance=escape_distance,
                    planning_clearances=planning_clearances,
                    subgoal_arrival=25., native_sensor=True, fine_mapping=False,
                    reference_path_in_policy=False, budget_reference="inherited task manifest route length; not a new shortest path",
                    step_cap=job["max_steps"], episode_budget_steps=budget))
    finally:
        if env is not None:
            env.env.close()
        verify_job_inputs(job)
    write_json(output / "cases" / (job["id"] + ".json"), result)
    write_json(output / "progress" / (job["id"] + ".json"), dict(id=job["id"], status=result["status"], **summary))
    return {key: result[key] for key in ("id", "case_id", "sample", "sample_seed", "status", "summary")}


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
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--sample-seeds", nargs="+", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--ground-safety-distance", "--safety-clearance", dest="ground_safety_distance",
                        type=float, default=clearance.DEFAULT_GROUND_CLEARANCE_M,
                        help="Ground planner, graph, bridge and audit clearance in metres (default 1); escape remains 0.5m")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Validate inputs and write the signed schedule without loading or running the policy")
    args = parser.parse_args()
    if args.samples < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("samples and max-steps must be positive")
    if args.sample_seeds is not None and any(not 0 <= seed < 2 ** 32 for seed in args.sample_seeds):
        parser.error("sample-seeds must be uint32")
    clearance.distances(args.ground_safety_distance)
    import numpy as np

    run, cases_path, scene_path, output = (p.resolve() for p in (args.run, args.cases, args.scene, args.output))
    hashes = frozen_hashes(run)
    if hashes["best.pt"] != BEST576_SHA256:
        raise ValueError("Checkpoint does not match frozen v9 best576")
    config = json.loads((run / "config.json").read_text())
    manifest = json.loads(cases_path.read_text())
    scene = load_scene_adapter().LegacyObstacleScene.from_json(scene_path)
    world = float(scene.world_size_m)
    if not math.isfinite(world) or world <= 0 or not np.isclose(world, manifest["world_size_m"]):
        raise ValueError("Scene world size must match the inherited case coordinates")
    source_map_sha256 = paired_source_hash(manifest, scene, cases_path)
    cases, identifiers = [], set()
    for index, row in enumerate(manifest["cases"]):
        case = dict(row)
        identifier = str(case["id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in identifiers:
            raise ValueError("Case IDs must be unique safe filenames")
        identifiers.add(identifier)
        if args.case_ids and identifier not in args.case_ids:
            continue
        case["scene_seed"] = int(case.get("scene_seed", case.get("seed", 92000 + index)))
        if not math.isfinite(case["reference_length_m"]) or case["reference_length_m"] <= 0:
            raise ValueError("A positive inherited reference_length_m is required for the fixed budget")
        for key in ("start", "goal"):
            point = np.asarray(case[key], float)
            if point.shape != (2,) or not np.isfinite(point).all() or (point < 0).any() or (point > world).any():
                raise ValueError("Case endpoints must be finite coordinates inside the scene")
        cases.append(case)
    if not cases or (args.case_ids and set(args.case_ids) - identifiers):
        raise ValueError("No cases selected or an unknown case ID was requested")
    settings = dict(stochastic=not args.deterministic,
                    samples=len(args.sample_seeds) if args.sample_seeds is not None else args.samples,
                    sample_seeds=args.sample_seeds, max_steps=args.max_steps, workers=1,
                    ground_safety_distance=args.ground_safety_distance, ground_escape_distance=.5,
                    safety_clearance_m=args.ground_safety_distance)
    provenance = dict(protocol=PROTOCOL, simplified_scene=True,
        geometry_scope="WildScenes obstacle layout in native 2D simulation; not a real-data traversability benchmark",
        unknown_space_assumption="free inside the world", run=str(run), frozen_hashes=hashes,
        scene_path=str(scene_path), scene_sha256=digest(scene_path),
        source_map_sha256=source_map_sha256,
        cases_path=str(cases_path), cases_sha256=digest(cases_path),
        runner_sha256=digest(__file__), shared_helper_sha256=digest(common.__file__),
        clearance_source_hashes=clearance.source_hashes(),
        scene_adapter_sha256=digest(Path(__file__).with_name("wildscenes_legacy_scene.py")),
        rectangle_count=len(scene.rectangles), settings=settings, selected_case_ids=[c["id"] for c in cases],
        perception=dict(drone_shape=[12, 12], drone_resolution_m=2., drone_footprint_m=[24., 24.],
                        ground_shape=[16, 16], ground_resolution_m=.5, ground_footprint_m=[8., 8.],
                        source="original frozen native circle/rectangle rasterizer", fine_mapping=False),
        strategy=dict(subgoal_radius_m=200., subgoal_arrival_m=25., checkpoint_episode=576),
        budget_contract="max(600, ceil(3 * inherited_reference_length_m + 240)); optional explicit step cap",
        reference_contract="Original manifest route retained only for budget/context; not a shortest path in the simplified scene",
        collision_attempt_contract="native_collision_attempts counts proposed motion rejected by either independent swept geometry or native endpoint collision; it is not limited to native endpoint events",
        success_contract="Vehicle-arrival termination plus zero collision attempts, executed physical/safety swept violations, link/spacing and boundary violations")
    signature = object_digest(provenance)
    output.mkdir(parents=True, exist_ok=True)
    prior_path = output / "manifest.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        if not args.resume or prior.get("signature") != signature:
            raise ValueError("Existing output requires --resume and identical inputs; otherwise use a new directory")
        if args.prepare_only:
            return  # An existing matching manifest must retain its completed records.
    elif any(output.iterdir()):
        raise ValueError("Refusing to mix results into a nonempty directory without a matching manifest")
    jobs = []
    for case in cases:
        seeds = args.sample_seeds if args.sample_seeds is not None else [
            int(np.random.SeedSequence([config["seed"], case["scene_seed"], sample]).generate_state(1)[0])
            for sample in range(args.samples)]
        for sample, seed in enumerate(seeds):
            jobs.append(dict(id=f"{case['id']}_r{sample}", case=case, sample=sample,
                sample_seed=seed, scene_path=str(scene_path), run=str(run), output=str(output),
                scene_sha256=provenance["scene_sha256"], cases_path=str(cases_path),
                cases_sha256=provenance["cases_sha256"],
                frozen_hashes=hashes, signature=signature, max_steps=args.max_steps,
                stochastic=not args.deterministic, ground_safety_distance=args.ground_safety_distance,
                ground_escape_distance=.5, safety_clearance_m=args.ground_safety_distance,
                clearance_source_hashes=provenance["clearance_source_hashes"]))
    report = dict(status="prepared" if args.prepare_only else "running", created_utc=utc(), signature=signature, provenance=provenance,
                  scheduled=[dict(id=j["id"], sample_seed=j["sample_seed"]) for j in jobs], records=[])
    write_json(prior_path, report)
    if args.prepare_only:
        return
    for job in jobs:
        if (output / "STOP").exists():
            break
        previous_path = output / "cases" / (job["id"] + ".json")
        previous = json.loads(previous_path.read_text()) if args.resume and previous_path.exists() else None
        if previous is not None and previous.get("signature") != signature:
            raise ValueError("Case provenance does not match: " + job["id"])
        record = ({key: previous[key] for key in ("id", "case_id", "sample", "sample_seed", "status", "summary")}
                  if previous is not None and previous["status"] == "completed" else safe_job(job))
        report["records"].append(record)
        report.update(updated_utc=utc(), aggregate=common.summarize(report["records"], len(jobs)))
        write_json(output / "results.json", report)
        keys = ("termination", "episode_steps", "ground_distance", "ground_goal_distance",
                "native_collision_attempts", "strict_success")
        print(json.dumps(clean(dict(id=record["id"], status=record["status"],
            **{key: record["summary"].get(key) for key in keys})), ensure_ascii=False), flush=True)
    verify_frozen_inputs(run, hashes)
    if digest(scene_path) != provenance["scene_sha256"] or digest(cases_path) != provenance["cases_sha256"]:
        raise RuntimeError("Scene or cases input changed during evaluation")
    report.update(status="stopped" if (output / "STOP").exists() else
                  "completed_with_errors" if any(r["status"] == "error" for r in report["records"]) else "completed",
                  completed_utc=utc(), frozen_inputs_unchanged=True,
                  aggregate=common.summarize(report["records"], len(jobs)))
    write_json(prior_path, report)
    write_json(output / "results.json", report)
    if report["aggregate"]["execution_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
