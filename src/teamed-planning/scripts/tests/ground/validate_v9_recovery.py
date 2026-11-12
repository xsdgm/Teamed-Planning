"""Replay selected archived tasks with v9 best576 and repaired navigation.

This is a new evaluation, not a rewrite of the archived frozen experiments.
Usage: python -I -B <this file> --case V_01_control_200m_forward_r2 --output <dir>
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time


def write_json(path, data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',action='append',required=True)
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--clearance',type=float,default=1.)
    parser.add_argument('--disable-joint-recovery',action='store_true')
    parser.add_argument('--terminal-step',type=int,
                        help='Restart at an archived 1-based trace step with fresh maps; NOT a full-task replay')
    parser.add_argument('--frozen',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    stop_signals=[]
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda signum, frame: stop_signals.append(signum))
    args.repo=args.repo.resolve();args.output=args.output.resolve()
    scripts=Path(__file__).resolve().parents[2]
    if not args.frozen:
        args.output.mkdir(parents=True,exist_ok=True)
        snapshot=args.output/'runtime_sources'
        shutil.copytree(scripts,snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        entry=snapshot/'tests/ground'/Path(__file__).name
        os.execv(sys.executable,[sys.executable,'-I','-B',str(entry),*sys.argv[1:],'--frozen'])
    sys.path.insert(0,str(scripts))
    import numpy as np
    import torch
    from simulator import mappo_experiment as me
    from simulator import train_mappo as tm
    from simulator import validate_wildscenes_vtemp as common
    from simulator.validate_wildscenes_legacy import native_runtime, legacy_summary, BEST576_SHA256
    from simulator import wildscenes_legacy_scene as adapter

    root=args.repo/'artifacts/wildscenes-legacy2d-formal-20261106'
    training=args.repo/'artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146'
    weight=training/'best.pt'
    weight_hash=hashlib.sha256(weight.read_bytes()).hexdigest()
    if weight_hash!=BEST576_SHA256:
        raise ValueError('Expected the archived v9 best576 checkpoint')
    config=json.loads((training/'config.json').read_text())
    checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
    torch.set_num_threads(1)
    source_hashes={str(p.relative_to(scripts)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(scripts.rglob('*.py'))}
    write_json(args.output/'manifest.json',dict(protocol='v9_repaired_navigation_review_v1',
        policy='v9 best576, frozen weights, no training',checkpoint=str(weight),
        checkpoint_sha256=weight_hash,cases=args.case,clearance_m=args.clearance,
        ground_navigation_resolution_m=.1,ground_sensor_width_m=8.,
        terminal_restart_step=args.terminal_step,
        terminal_memory='80m square at 0.1m; local observations only; takeover after stall',
        coordinated_safety=not args.disable_joint_recovery,source_sha256=source_hashes,
        scope='Selected regression tasks. Different controller/sensor settings from original evaluation.',
        coordination_contract='Synchronous additional joint safety solve; not a delayed decentralized deployment.'))
    results=[]
    for name in args.case:
        original_path=root/'runs'/name.rsplit('_r',1)[0]/'cases'/f'{name}.json'
        original=json.loads(original_path.read_text());case=dict(original['case'])
        terminal_state=None
        if args.terminal_step is not None:
            if not 1<=args.terminal_step<=len(original['trace']):
                raise ValueError('terminal-step is outside the archived trace')
            terminal_state=original['trace'][args.terminal_step-1]
            case['start']=terminal_state['ground_position'][:2]
        scene_path=Path(original['scene_path'])
        scene=adapter.LegacyObstacleScene.from_json(scene_path)
        torch.manual_seed(original['sample_seed']);np.random.seed(original['sample_seed'])
        model=tm.ActorCritic.from_state_dict(checkpoint['model']).eval()
        schedule=tm.fixed_schedule(config,config['episodes']-1)
        options=common.experiment_options(config,schedule,scene.world_size_m,case)
        options.update(ground_safety_distance=args.clearance,
                       coordinated_safety=not args.disable_joint_recovery)
        budget=max(600,math.ceil(3.*case['reference_length_m']+240))
        if args.max_steps is not None:
            budget=min(budget,args.max_steps)
        started=time.monotonic();physical=0;events=[];trajectory=[];termination='budget_exhausted'
        previous=np.asarray(case['start'],float)
        with native_runtime(me,scene,adapter), (args.output/f'{name}.log').open('w') as log, \
                (args.output/f'{name}.steps.jsonl').open('w',buffering=1) as steps_log, \
                contextlib.redirect_stdout(log),torch.inference_mode():
            env=me.Experiment(case['scene_seed'],model,stochastic=original['stochastic'],**options)
            if terminal_state is not None:
                for robot,position in zip(env.env.drones,terminal_state['positions']):
                    robot.position[:]=position
                env.obs=env.env._get_observations()
                env.ground_action=env._ground_action()
            try:
                for frame in range(budget):
                    if stop_signals:
                        termination='external_signal';break
                    if (args.output/'STOP').exists():
                        termination='review_stopped';break
                    try:
                        env.step(frame)
                    except me.SafetyTermination:
                        termination='cbf_infeasible';break
                    current=np.asarray(env.obs[env.ground]['self']['position'][:2],float)
                    if not scene.segment_clear(previous,current,.3):
                        physical+=1
                    previous=current.copy()
                    controller=env.ground_controller
                    planner=controller.global_planner
                    step_record=dict(step=frame+1, position=current.tolist(),
                        heading=float(controller.heading), target=env.subgoals.target.tolist(),
                        remaining_m=float(np.linalg.norm(current-case['goal'])),
                        motion=list(env.ground_action['motion']), waiting=planner.waiting_at_frontier,
                        using_graph=planner.using_graph, waypoint=planner.get_current_waypoint(),
                        waypoint_index=planner.current_waypoint_idx, path=planner.current_path,
                        escape=controller._allow_local_escape, terminal_recovery=controller._terminal_recovery,
                        route_loop=controller._route_loop,
                        route_recovery=controller._route_recovery,
                        route_best=float(controller._local_route_best),
                        route_progress_time=controller._local_route_progress_time,
                        route_progress_position=controller._local_route_progress_position,
                        local_progress_time=controller._local_progress_time,
                        local_progress_position=controller._local_progress_position,
                        failed_connections=controller._failed_local_connections,
                        subgoal_selected=env.subgoals.selected, subgoal_failed=env.subgoals.failed,
                        subgoal_best=env.subgoals._best, subgoal_since=env.subgoals._since)
                    encode=lambda value: value.tolist() if hasattr(value,'tolist') else float(value)
                    steps_log.write(json.dumps(step_record,default=encode)+'\n')
                    if frame%100==0:
                        latest=dict(step_record, sensor=env.obs[env.ground].get('navigation_map'),
                            sensor_resolution=env.obs[env.ground].get('navigation_resolution'),
                            graph=(None if controller.unified_graph is None else
                                   json.loads(controller.unified_graph.to_json())))
                        write_json(args.output/f'{name}.diagnostic.json',json.loads(json.dumps(latest,default=encode)))
                    if not events or env.subgoals.selected!=events[-1]['selected']:
                        events.append(dict(step=frame+1,target=env.subgoals.target.tolist(),
                            selected=env.subgoals.selected,final=env.subgoals.final,
                            failures=env.subgoals.failed))
                    if frame%5==0:
                        trajectory.append(dict(step=frame+1,ground=current.tolist(),
                            remaining_m=float(np.linalg.norm(current-case['goal']))))
                    if frame%25==0:
                        write_json(args.output/f'{name}.progress.json',dict(status='running',step=frame+1,
                            budget=budget,remaining_m=float(np.linalg.norm(current-case['goal'])),
                            joint_recoveries=len(env.coordinated_safety_events),
                            physical_swept_violations=physical,wall_seconds=time.monotonic()-started,
                            position=current.tolist(),target=env.subgoals.target.tolist(),
                            next_motion=list(env.ground_action['motion']),
                            waypoint=env.ground_controller.global_planner.get_current_waypoint(),
                            waypoint_index=env.ground_controller.global_planner.current_waypoint_idx,
                            waiting=env.ground_controller.global_planner.waiting_at_frontier))
                    if physical:
                        termination='physical_geometry_violation';break
                    if env.goal_reached:
                        termination='vehicle_arrival';break
                summary=legacy_summary(env,scene)
                graph=env.ground_controller.unified_graph
                if graph is not None:
                    (args.output/f'{name}.final_graph.json').write_text(graph.to_json())
                result=dict(case=name,termination=termination,summary=summary,
                    stop_signals=stop_signals,
                    physical_swept_violations=physical,native_collision_attempts=env.env.legacy_collision_attempts,
                    wall_seconds=time.monotonic()-started,original_source=str(original_path),
                    original_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
                    original_termination=original['summary']['termination'],
                    scene_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
                    terminal_restart_step=args.terminal_step,
                    evaluation_scope=('counterfactual terminal recovery; fresh graphs, relay and policy state; '
                        'default initial heading; not a continuation or full-mission result'
                        if terminal_state is not None else 'full mission from archived initial conditions'),
                    trajectory=trajectory,subgoal_events=events,joint_recoveries=env.coordinated_safety_events,
                    final_position=previous.tolist(),
                    final_waypoint=env.ground_controller.global_planner.get_current_waypoint(),
                    final_path=env.ground_controller.global_planner.current_path)
                write_json(args.output/f'{name}.result.json',result)
                write_json(args.output/f'{name}.progress.json',dict(status='complete',termination=termination,
                    steps=summary['episode_steps'],remaining_m=summary['ground_goal_distance'],
                    wall_seconds=result['wall_seconds']))
                results.append({k:result[k] for k in ('case','termination','summary','physical_swept_violations',
                    'native_collision_attempts','wall_seconds','original_termination')})
            finally:
                env.env.close()
        write_json(args.output/'results.json',dict(status='complete' if len(results)==len(args.case) else 'running',
            results=results))
        print(json.dumps(dict(case=name,termination=termination,
            remaining_m=summary['ground_goal_distance'],steps=summary['episode_steps'],
            joint_recoveries=summary['coordinated_safety_steps'],wall_seconds=result['wall_seconds'])),flush=True)


if __name__=='__main__':
    main()
