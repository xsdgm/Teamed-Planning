"""Isolate ground recovery using an archived pose and locally sensed geometry.

Fresh controller state and no UAV graph: this is a counterfactual diagnosis,
not an exact continuation or a full mission evaluation. Geometry stays in the
sensor simulator and swept-clearance auditor, never the controller input.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--steps', type=int, default=180)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--heading', type=float, default=0.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.runtime.resolve()))
    import numpy as np
    from ground.ground_main import GroundRobotController
    from ground.observed_grid import ObservedGrid
    from simulator.env import EnvConfig, MultiRobotEnv
    from simulator.wildscenes_legacy_scene import LegacyObstacleScene, install_legacy_environment

    source = json.loads(args.progress.read_text())
    start, goal = source['position'], source['target']
    scene = LegacyObstacleScene.from_json(args.scene)
    records, local_calls, memory_calls = [], [], []
    original = ObservedGrid.waypoint

    def traced_waypoint(memory, position, clearance):
        captured = {}

        def trace(frame, event, arg):
            if frame.f_code is not original.__code__:
                return None
            if event == 'return':
                state = frame.f_locals
                for key in ('threshold', 'destination', 'sy', 'sx', 'start'):
                    if key in state:
                        captured[key] = float(state[key])
                for key in ('known_y', 'route'):
                    if key in state:
                        captured[key+'_count'] = len(state[key])
                for key in ('safe', 'candidates'):
                    if key in state:
                        captured[key+'_count'] = int(np.count_nonzero(state[key]))
                if 'distances' in state and 'sy' in state:
                    sy, sx = state['sy'], state['sx']
                    if 0 <= sy < len(state['distances']) and 0 <= sx < state['distances'].shape[1]:
                        captured['start_clearance'] = float(state['distances'][sy, sx])
                captured['return_line'] = frame.f_lineno
            return trace

        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            result = original(memory, position, clearance)
        finally:
            sys.settrace(previous)
        captured.update(waypoint=result, frontier=None if memory.frontier is None else memory.frontier.tolist(),
                        path_length=len(memory.path), clearance=clearance)
        memory_calls.append(captured)
        return result

    ObservedGrid.waypoint = traced_waypoint
    with (args.output/'controller.log').open('w') as log, contextlib.redirect_stdout(log):
        env = MultiRobotEnv(EnvConfig(world_size=(scene.world_size_m,)*2, dt=1.,
            ground_occupancy_map_size=16, ground_occupancy_map_resolution=.5))
        install_legacy_environment(env, scene)
        rid = env.add_ground_robot(*start)
        obs = env.reset()[rid]
        env.ground_robots[0].orientation = args.heading
        obs = env._get_observations()[rid]
        controller = GroundRobotController('diagnostic', safety_distance=1., subgoal_radius=200.,
                                           world_bounds=(0., 0., scene.world_size_m, scene.world_size_m))
        controller.set_goal(tuple(goal))
        local = controller._local_waypoint

        def record_local(position, target, semantic, resolution):
            result = local(position, target, semantic, resolution)
            local_calls.append(dict(target=list(target), waypoint=result))
            return result

        controller._local_waypoint = record_local
        violations = 0
        for step in range(args.steps):
            local_calls.clear(); memory_calls.clear()
            obs.pop('obstacles', None)
            obs['full_state'] = {'time': float(step)}
            position = np.asarray(obs['self']['position'][:2]).copy()
            action = controller.get_action(obs)
            observations, info = env.step({rid: action})
            obs = observations[rid]
            after = np.asarray(obs['self']['position'][:2])
            violations += int(not scene.segment_clear(position, after, .3))
            records.append(dict(step=step+1, position=position.tolist(), after=after.tolist(),
                heading=float(controller.heading), motion=list(action['motion']),
                remaining_m=float(np.linalg.norm(after-goal)), recovery=controller._terminal_recovery,
                escape=controller._allow_local_escape, local_calls=list(local_calls), memory_calls=list(memory_calls),
                route_best=float(controller._local_route_best), route_progress_at=controller._local_route_progress_time,
                local_progress_at=controller._local_progress_time))
            if violations or np.linalg.norm(after-goal) <= 2.:
                break
        memory = controller._terminal_map
        if memory is not None:
            np.savez_compressed(args.output/'observed_memory.npz', cells=memory.cells, visits=memory.visits,
                                origin=memory.origin, position=after, goal=goal, resolution=memory.resolution)
        env.close()
    result = dict(scope='Ground-only restart, fresh state, no received UAV graph; not a full mission replay',
        source=str(args.progress.resolve()), source_sha256=hashlib.sha256(args.progress.read_bytes()).hexdigest(),
        scene=str(args.scene.resolve()), scene_sha256=hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        runtime=str(args.runtime.resolve()), source_sha256_by_file={name:hashlib.sha256((args.runtime/name).read_bytes()).hexdigest()
            for name in ('ground/ground_main.py', 'ground/local_planner.py', 'ground/observed_grid.py')},
        swept_violations=violations, goal_reached=records[-1]['remaining_m'] <= 2., records=records)
    (args.output/'diagnosis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(steps=len(records), remaining_m=records[-1]['remaining_m'],
        swept_violations=violations, output=str(args.output))))


if __name__ == '__main__':
    main()
