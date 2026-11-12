"""Audit long-route feasibility, then repeat the frozen r14 navigation review.

Global geometry is used only by this offline auditor. Child evaluators receive
the original archived task IDs, never the offline certificate paths.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


CASES = (
    "K_01_1500m_forward_r0",
    "K_01_1500m_forward_r1",
    "K_01_1500m_forward_r2",
    "K_01_1000m_forward_r0",
    "K_01_1000m_reverse_r0",
)
SAFETY = (
    "collisions", "ground_collisions", "out_of_bounds", "spacing_violation",
    "chain_link_violation", "communication_outage",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def process_identity(pid):
    """PID plus Linux start time, excluding zombies and reused process IDs."""
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def live_status(output):
    state=json.loads((output/'batch_summary.json').read_text())
    processes=json.loads((output/'processes.json').read_text())
    def alive(record):
        actual=process_identity(record['pid'])
        return actual is not None and (record.get('started') is None or actual==record['started'])
    supervisor=dict(pid=processes['supervisor_pid'],started=processes.get('supervisor_started'))
    state['supervisor_alive']=alive(supervisor)
    state['checked_at']=utc()
    children={item['case']:item for item in processes['children']}
    for row in state['runs']:
        if row['status'] in ('starting','running'):
            row['process_alive']=alive(children[row['case']])
            if not row['process_alive']:
                result_path=output/row['case']/(row['case']+'.result.json')
                if result_path.exists():
                    result=json.loads(result_path.read_text())
                    row.update(status='complete',termination=result['termination'],
                        steps=result['summary']['episode_steps'],
                        remaining_m=result['summary']['ground_goal_distance'])
                else:
                    row['status']='interrupted'
    if not state['supervisor_alive'] and state['status'] in ('running','stopping'):
        if any(r.get('process_alive') for r in state['runs']):
            state['status']='orphaned'
        elif all(r['status']=='complete' and r.get('termination')!='external_signal' for r in state['runs']):
            state['status']='complete'
        else:
            state['status']='interrupted'
    state['completed']=sum(r['status']=='complete' for r in state['runs'])
    state['arrived']=sum(r.get('termination')=='vehicle_arrival' for r in state['runs'])
    return state


def feasible_route(scene, start, goal, margin, resolution=.5):
    """Conservative cell-footprint raster + four-neighbor A* + exact sweep.

    A free cell's entire square avoids every rectangle inflated by margin in
    both axes. Thus cardinal edges cannot cut corners. Exact geometry checks
    include the start/goal connectors; the returned certificate reaches the
    actual goal, not an arbitrary nearby frontier.
    """
    import numpy as np

    start, goal = np.asarray(start, float), np.asarray(goal, float)
    record = dict(clearance_m=margin, resolution_m=resolution,
                  start_clear=bool(scene.point_clear(start, margin)),
                  goal_clear=bool(scene.point_clear(goal, margin)),
                  feasible=False)
    if not record["start_clear"] or not record["goal_clear"]:
        record["reason"] = "endpoint_clearance"
        return record
    size = math.ceil(scene.world_size_m / resolution)
    blocked = np.zeros((size, size), dtype=bool)
    # Cells touching inflated rectangles are excluded, including boundary ties.
    lower = np.floor((scene.lower - margin - 1e-9) / resolution).astype(int)
    upper = np.floor((scene.upper + margin + 1e-9) / resolution).astype(int)
    lower = np.clip(lower, 0, size - 1)
    upper = np.clip(upper, 0, size - 1)
    for (x0, y0), (x1, y1) in zip(lower, upper):
        blocked[y0:y1 + 1, x0:x1 + 1] = True
    edges = np.arange(size) * resolution
    outside = (edges <= margin) | (edges + resolution >= scene.world_size_m - margin)
    blocked[outside, :] = True
    blocked[:, outside] = True
    sx, sy = np.floor(start / resolution).astype(int)
    gx, gy = np.floor(goal / resolution).astype(int)
    if blocked[sy, sx] or blocked[gy, gx]:
        record["reason"] = "conservative_endpoint_cell_blocked"
        return record
    source, target = int(sy * size + sx), int(gy * size + gx)
    distance = np.full(size * size, np.iinfo(np.int32).max, dtype=np.int32)
    parent = np.full(size * size, -1, dtype=np.int8)
    visited = np.zeros(size * size, dtype=bool)
    flat_blocked = blocked.ravel()
    shifts = (1, -1, size, -size)
    distance[source] = 0
    initial_h = abs(int(sx - gx)) + abs(int(sy - gy))
    queue = [(initial_h, initial_h, 0, source)]
    explored = 0
    while queue:
        _, _, cost, current = heapq.heappop(queue)
        if visited[current]:
            continue
        visited[current] = True
        explored += 1
        if current == target:
            break
        y, x = divmod(current, size)
        for direction, shift in enumerate(shifts):
            if ((direction == 0 and x == size - 1) or (direction == 1 and x == 0)
                    or (direction == 2 and y == size - 1) or (direction == 3 and y == 0)):
                continue
            nxt = current + shift
            new_cost = cost + 1
            if flat_blocked[nxt] or visited[nxt] or new_cost >= distance[nxt]:
                continue
            distance[nxt] = new_cost
            parent[nxt] = direction
            ny, nx = divmod(nxt, size)
            heuristic = abs(nx - gx) + abs(ny - gy)
            heapq.heappush(queue, (new_cost + heuristic, heuristic, new_cost, nxt))
    record["explored_cells"] = explored
    if not visited[target]:
        record["reason"] = "no_route_on_conservative_grid_not_proof_of_impossibility"
        return record
    indices = [target]
    while indices[-1] != source:
        indices.append(indices[-1] - shifts[int(parent[indices[-1]])])
    centers = [((i % size + .5) * resolution, (i // size + .5) * resolution)
               for i in reversed(indices)]
    path = np.asarray([start, *centers, goal])
    invalid = [i for i, (a, b) in enumerate(zip(path[:-1], path[1:]))
               if not scene.segment_clear(a, b, margin)]
    record.update(feasible=not invalid, invalid_segments=invalid,
                  exact_segment_count=len(path) - 1,
                  length_m=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                  path=path.tolist())
    return record


def check_auditor(scene_type, protocol):
    def scene(rectangles):
        return scene_type(dict(protocol=protocol, simplified_scene=True,
            unknown_space="free_by_simulation_assumption", world_size_m=20., origin_xy=[0., 0.],
            geometry=dict(type="axis_aligned_rectangles", rectangles=rectangles)))

    detour_scene = scene([[10., 10., 1., 12.]])
    detour = feasible_route(detour_scene, [3., 10.], [17., 10.], .5)
    assert detour["feasible"] and detour["length_m"] > 14.
    assert detour["path"][0] == [3., 10.] and detour["path"][-1] == [17., 10.]
    assert not feasible_route(scene([[10., 10., 1., 20.]]), [3., 10.], [17., 10.], .5)["feasible"]
    assert not feasible_route(detour_scene, [10., 10.], [17., 10.], .5)["feasible"]
    return ["finite-wall detour with exact endpoint connectors",
            "boundary-spanning wall rejected", "occupied start rejected"]


def prepare(repo, output, allow_runtime_change=False):
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "runtime_sources"
    baseline = repo / "artifacts/v9-navigation-repair-20261111-r14-full/manifest.json"
    expected = json.loads(baseline.read_text())["source_sha256"]
    current = repo / "src/teamed-planning/scripts"
    changed = [name for name, value in expected.items()
               if not (current / name).is_file() or digest(current / name) != value]
    runtime_changed = [name for name in changed
                       if not name.startswith("tests/") and not Path(name).name.startswith("test_")]
    if runtime_changed and not allow_runtime_change:
        raise RuntimeError("Runtime differs from the reviewed r14: " + ", ".join(runtime_changed))
    expected = {str(p.relative_to(current)): digest(p) for p in sorted(current.rglob("*.py"))}
    if not snapshot.exists():
        shutil.copytree(current, snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if any(digest(snapshot / name) != value for name, value in expected.items()):
        raise RuntimeError("Frozen snapshot differs from r14")
    sys.path.insert(0, str(snapshot))
    from simulator.wildscenes_legacy_scene import LegacyObstacleScene, PROTOCOL
    auditor_checks = check_auditor(LegacyObstacleScene, PROTOCOL)

    cache, scenes, jobs = {}, {}, []
    for name in CASES:
        original_path = (repo / "artifacts/wildscenes-legacy2d-formal-20261106/runs"
                         / name.rsplit("_r", 1)[0] / "cases" / (name + ".json"))
        archived = json.loads(original_path.read_text())
        task = archived["case"]
        scene_path = Path(archived["scene_path"])
        if str(scene_path) not in scenes:
            scenes[str(scene_path)] = LegacyObstacleScene.from_json(scene_path)
        scene = scenes[str(scene_path)]
        route_key = task["id"]
        if route_key not in cache:
            certificates = []
            for margin in (1., .5):
                audit = feasible_route(scene, task["start"], task["goal"], margin)
                if not audit["feasible"]:
                    audit = feasible_route(scene, task["start"], task["goal"], margin, .25)
                certificates.append(audit)
            certificate = dict(task_id=route_key, created_at=utc(),
                scope="Offline ground geometry in the simplified native 2D scene; not a joint UAV feasibility proof or real-forest validation",
                controller_receives_certificate=False, start=task["start"], goal=task["goal"],
                scene_sha256=digest(scene_path), routes=certificates)
            write_json(output / (route_key + ".feasibility.json"), certificate)
            cache[route_key] = certificate
            print(json.dumps(dict(task=route_key, feasibility=[
                {k: r.get(k) for k in ("clearance_m", "feasible", "length_m", "reason")}
                for r in certificates])), flush=True)
        certificate = cache[route_key]
        if not any(r["feasible"] for r in certificate["routes"]):
            raise RuntimeError("No verified feasible route: " + route_key)
        jobs.append(dict(case=name, sample_seed=archived["sample_seed"],
            scene_seed=task["scene_seed"], stochastic=archived["stochastic"],
            reference_length_m=task["reference_length_m"],
            original_budget=max(600, math.ceil(3. * task["reference_length_m"] + 240)),
            original_source=str(original_path), original_sha256=digest(original_path),
            scene_path=str(scene_path), scene_sha256=certificate["scene_sha256"],
            certificate=route_key + ".feasibility.json",
            replicate_of_previous_success=name == "K_01_1500m_forward_r0"))
    manifest = dict(created_at=utc(), status="preflight_passed", jobs=jobs,
        auditor_checks_passed=auditor_checks,
        runtime_matches_r14=not runtime_changed, source_sha256=expected,
        changes_from_r14=changed,
        ground_clearance_m=1., recovery_clearance_m=.5, physical_swept_clearance_m=.3,
        arrival_tolerance_m=2., reference_path_in_policy=False,
        notes="Three 1.5km sampling seeds plus 1km forward/reverse on one scene; not independent maps. No budget or success-threshold changes.")
    write_json(output / "batch_manifest.json", manifest)
    return snapshot, jobs


def run(repo, output, snapshot, jobs):
    active = []
    stop_signals=[]
    def stop(signum, frame):
        stop_signals.append(signum)
        for _, process in active:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, stop)
    environment = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    for job in jobs:
        directory = output / job["case"]
        directory.mkdir(exist_ok=False)
        log = (directory / "launcher.log").open("w")
        command = [sys.executable, "-I", "-B", str(snapshot / "tests/ground/validate_v9_recovery.py"),
                   "--repo", str(repo), "--output", str(directory), "--case", job["case"], "--frozen"]
        process = subprocess.Popen(command, cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        active.append((job, process))
        print(json.dumps(dict(started=job["case"], pid=process.pid)), flush=True)
    write_json(output / "processes.json", dict(supervisor_pid=os.getpid(),
        supervisor_started=process_identity(os.getpid()),
        children=[dict(case=job["case"], pid=p.pid,started=process_identity(p.pid)) for job, p in active]))
    while True:
        rows = []
        for job, process in active:
            directory = output / job["case"]
            result_path = directory / (job["case"] + ".result.json")
            row = dict(case=job["case"], sample_seed=job["sample_seed"], pid=process.pid,
                       returncode=process.poll())
            if process.returncode is not None and not (directory/'exit.json').exists():
                write_json(directory/'exit.json',dict(returncode=process.returncode,updated_at=utc()))
            if result_path.exists():
                result = json.loads(result_path.read_text())
                metrics = result["summary"]
                row.update(status="complete", termination=result["termination"],
                    steps=metrics["episode_steps"], remaining_m=metrics["ground_goal_distance"],
                    wall_seconds=result["wall_seconds"],
                    physical_swept_violations=result["physical_swept_violations"],
                    native_collision_attempts=result["native_collision_attempts"],
                    safety={key: metrics[key] for key in SAFETY},
                    result=str(result_path.relative_to(output)))
            elif process.returncode is not None:
                row.update(status="error", log=str((directory / "launcher.log").relative_to(output)))
            else:
                progress = directory / (job["case"] + ".progress.json")
                row.update(json.loads(progress.read_text()) if progress.exists() else {"status": "starting"})
            rows.append(row)
        finished = all(p.returncode is not None for _, p in active)
        errors=any(r['status']=='error' for r in rows)
        state = dict(updated_at=utc(), status=("interrupted" if finished and stop_signals else
                     "finished_with_errors" if finished and errors else
                     "complete" if finished else "stopping" if stop_signals else "running"),
            stop_signals=stop_signals, supervisor_pid=os.getpid(), planned=len(jobs),
            completed=sum(r["status"] == "complete" for r in rows),
            arrived=sum(r.get("termination") == "vehicle_arrival" for r in rows), runs=rows)
        write_json(output / "batch_summary.json", state)
        print(json.dumps({**state, "runs": [{k: r[k] for k in
            ("case", "status", "step", "steps", "remaining_m", "termination") if k in r} for r in rows]}), flush=True)
        if finished:
            return
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--status",action="store_true",help="Check process identities instead of trusting an old running file")
    parser.add_argument("--allow-runtime-change",action="store_true",
                        help="Freeze and record an explicitly repaired runtime instead of requiring r14")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if args.status:
        print(json.dumps(live_status(output),indent=2))
        return
    snapshot, jobs = prepare(repo, output, args.allow_runtime_change)
    if args.run:
        run(repo, output, snapshot, jobs)


if __name__ == "__main__":
    main()
