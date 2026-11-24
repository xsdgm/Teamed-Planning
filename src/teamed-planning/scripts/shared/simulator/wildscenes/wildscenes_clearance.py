"""Explicit ground-clearance override for archived simulation runtimes.

Archived modules and checkpoints remain unchanged. Load this file and the
navigation configuration by path so an archived ``common`` package cannot hide
the evaluator's configuration. Snapshots copy navigation_config.py adjacent.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import inspect
import math
from pathlib import Path


def navigation_config_path():
    adjacent = Path(__file__).with_name("navigation_config.py")
    return adjacent if adjacent.is_file() else Path(__file__).parents[3] / "shared/common/navigation_config.py"


_spec = importlib.util.spec_from_file_location("_wildscenes_navigation_config", navigation_config_path())
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)
DEFAULT_GROUND_CLEARANCE_M = _config.DEFAULT_GROUND_CLEARANCE_M


def distances(safety=DEFAULT_GROUND_CLEARANCE_M, escape=None):
    safety = float(safety)
    escape = .5 if escape is None else float(escape)
    if not all(math.isfinite(value) and value >= 0. for value in (safety, escape)):
        raise ValueError("Ground safety and escape distances must be finite and nonnegative")
    if escape > safety:
        raise ValueError("Ground escape distance cannot exceed the normal safety distance")
    return safety, escape


def job_distances(job):
    return distances(job.get("ground_safety_distance", job.get("safety_clearance_m", DEFAULT_GROUND_CLEARANCE_M)),
                     job.get("ground_escape_distance"))


def source_hashes():
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), navigation_config_path())}


def verify_sources(expected):
    if expected is not None and source_hashes() != expected:
        raise RuntimeError("Clearance adapter/configuration changed during evaluation")


def experiment_options(me, options, safety, escape):
    """Pass new options only to runtimes that support them; old ones use wrappers."""
    safety, escape = distances(safety, escape)
    result = dict(options)
    parameters = inspect.signature(me.Experiment).parameters
    for name, value in (("ground_safety_distance", safety), ("ground_escape_distance", escape)):
        result.pop(name, None)
        if name in parameters:
            result[name] = value
    return result


@contextlib.contextmanager
def planning_runtime(me, safety=DEFAULT_GROUND_CLEARANCE_M, escape=None):
    """Override before construction, including the initial ground action."""
    safety, escape = distances(safety, escape)
    original_ground, original_drone = me.GroundRobotController, me.DroneController
    sentinel = object()
    original_bridge = getattr(me, "BRIDGE_CLEARANCE_M", sentinel)

    def ground(*args, **kwargs):
        kwargs["safety_distance"] = safety
        controller = original_ground(*args, **kwargs)
        controller.local_planner.escape_distance = escape
        return controller

    def drone(*args, **kwargs):
        kwargs["graph_min_clearance"] = safety
        return original_drone(*args, **kwargs)

    try:
        me.GroundRobotController, me.DroneController = ground, drone
        me.BRIDGE_CLEARANCE_M = safety
        yield
    finally:
        me.GroundRobotController, me.DroneController = original_ground, original_drone
        if original_bridge is sentinel:
            delattr(me, "BRIDGE_CLEARANCE_M")
        else:
            me.BRIDGE_CLEARANCE_M = original_bridge


def verify_experiment(experiment, safety=DEFAULT_GROUND_CLEARANCE_M, escape=None):
    """Verify real components and bind legacy bridge closure to this run's value."""
    safety, escape = distances(safety, escape)
    planner = experiment.ground_controller.local_planner
    actual = dict(ground_safety_distance=float(planner.safety_distance),
                  ground_escape_distance=float(planner.escape_distance),
                  graph_clearance_m=[float(c.graph_builder.min_clearance) for c in experiment.controllers],
                  graph_clearance_pixels=[float(c.graph_builder.min_clearance_pixels) for c in experiment.controllers],
                  bridge_clearance_m=None)
    if actual["ground_safety_distance"] != safety or actual["ground_escape_distance"] != escape:
        raise RuntimeError("Ground planner did not apply the requested clearance")
    for controller in experiment.controllers:
        builder = controller.graph_builder
        if builder.min_clearance != safety or not math.isclose(builder.min_clearance_pixels, safety / builder.resolution):
            raise RuntimeError("Drone graph builder did not apply the requested clearance")
    fusion = experiment.ground_controller.graph_fusion
    if getattr(fusion, "bridge_check", None) is not None:
        ledger = experiment.ledger
        sentinel = object()
        original = ledger.__dict__.get("segment_clear", sentinel)
        observed = []

        def capture(a, b, clearance):
            observed.append(float(clearance))
            return True

        try:
            ledger.segment_clear = capture
            fusion.bridge_check((0., 0.), (0., 0.))
        finally:
            if original is sentinel:
                del ledger.segment_clear
            else:
                ledger.segment_clear = original
        if observed != [safety]:
            raise RuntimeError("Delivered-evidence bridge did not apply the requested clearance")
        # Legacy callback reads a mutable module global. Capture this run's
        # distance so it stays correct after the scoped override is restored.
        fusion.bridge_check = lambda a, b: ledger.segment_clear(a, b, safety)
        actual["bridge_clearance_m"] = observed[0]
    return actual
