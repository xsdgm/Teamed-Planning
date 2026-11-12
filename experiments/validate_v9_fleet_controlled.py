"""Controlled 3/4-aircraft evaluation with frozen policies and isolated runtimes.

Archived mode reproduces the original integration. Repaired mode overlays the
reviewed vehicle navigation and its same-footprint local sensor; the aircraft
policy, mapper, communication limits, reward and independent CBF stay frozen.
Every worker is a fresh process, avoiding cross-runtime Python module reuse.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fleet_controlled_support import (action_stream, coupled_relay_class,
    fleet_positions, loop_statistics, minimum_constraint_speed, swept_clearance)

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / 'artifacts/v9-fleet4-wait30-20261111'
REPAIR = REPO / 'artifacts/v9-navigation-repair-20261111-r22-multirun/runtime_sources'
PREVIOUS_REPAIR = REPO / 'artifacts/v9-fleet-controlled-final-20261111/runtime_repaired'
OVERLAY = ('ground/ground_main.py', 'ground/global_planner.py', 'ground/local_planner.py',
           'ground/graph_fusion.py', 'ground/observed_grid.py', 'ground/observed_escape.py',
           'common/navigation_config.py', 'simulator/env.py')
SAFETY = ('ground_collisions', 'collisions', 'spacing_violation',
          'chain_link_violation', 'out_of_bounds', 'communication_outage')
RUNTIME_OVERLAYS = {
    'archived': (),
    'repaired': OVERLAY,
    'navigation_only': tuple(p for p in OVERLAY if p != 'ground/graph_fusion.py'),
    'fusion_only': ('ground/graph_fusion.py',),
    'coarse_sensor': tuple(p for p in OVERLAY if p != 'simulator/env.py'),
    # Current local_planner imports escape geometry from observed_grid. Frozen
    # modes above keep their original, separate observed_escape dependency.
    'candidate': tuple(p for p in OVERLAY if p != 'ground/observed_escape.py'),
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    return value


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def hashes(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*.py'))}


def process_identity(pid):
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return fields[19] if fields[0]!='Z' else None
    except (FileNotFoundError, ProcessLookupError):
        return None


def live_status(root):
    """Reconcile saved progress with results and live process identities."""
    state=json.loads((root/'status.json').read_text())
    def alive(pid, expected=None):
        current=process_identity(pid)
        return current is not None and (expected is None or current==expected)
    state['checked_at']=utc()
    state['supervisor_alive']=alive(state['supervisor_pid'],state.get('supervisor_started'))
    manifest_path=root/'manifest.json'
    jobs=({j['key']:j for j in json.loads(manifest_path.read_text())['jobs']}
          if manifest_path.exists() else {})
    for row in state['runs']:
        for key,value in jobs.get(row['key'],{}).items():
            row.setdefault(key,value)
        out=root/'data'/row['key']
        result=out/'result.json'
        if result.exists():
            row.update(json.loads(result.read_text()))
        elif row['status'] in ('queued','running') and (out/'process.json').exists():
            process=json.loads((out/'process.json').read_text())
            row['process_alive']=alive(process['pid'],process.get('started'))
            row['status']='running' if row['process_alive'] else 'interrupted'
    if not state['supervisor_alive'] and state['status']=='running':
        if any(r.get('process_alive') for r in state['runs']):
            state['status']='orphaned'
        elif all(r['status'] in ('complete','error') for r in state['runs']):
            state['status']='error' if any(r['status']=='error' for r in state['runs']) else 'complete'
        else:
            state['status']='interrupted'
    state['completed']=sum(r['status'] in ('complete','error') for r in state['runs'])
    return state


def worker(root, key):
    import numpy as np
    import torch
    manifest = json.loads((root/'manifest.json').read_text())
    spec = next(j for j in manifest['jobs'] if j['key'] == key)
    out = root/'data'/key
    out.mkdir(exist_ok=True)
    started = time.monotonic()
    runtime = root/('runtime_' + spec['runtime'])
    sys.path.insert(0, str(runtime))
    from simulator import train_mappo as tm
    from simulator import mappo_experiment as me
    import drone.cbf_safety as cbf
    from drone.local_backbone import relative_neighborhood
    case = json.loads((root/'cases.json').read_text())[str(spec['case'])]
    config = json.loads((root/'config.json').read_text())
    config['fleet_sizes'] = [spec['fleet']]
    if spec['smoke_steps']:
        config['steps'] = spec['smoke_steps']
    torch.set_num_threads(1)
    torch.manual_seed(spec['sample_seed'])
    model = tm.ActorCritic.from_state_dict(torch.load(root/'best.pt', map_location='cpu',
                                                    weights_only=True)['model']).eval()
    trajectory, diagnostics, failures, replans, snapshots = [], [], [], [], []
    state = {'hold_displacement_m': 0., 'ground_min_swept_clearance_m': float('inf'),
             'ground_swept_violations': 0, 'uav_swept_violations': 0, 'radio_step_violations': 0,
             'parity_max_position_error_m': 0.}
    baseline = None
    if spec['noise'] == 'historical' and spec['layout'] == 'native' and spec['runtime'] == 'archived':
        baseline = json.loads((Path(manifest['archive_path'])/'data'/
            f"case{spec['case']}_r{spec['sample']}_n{spec['fleet']}_native.json").read_text())
    if spec['runtime']=='repaired' and spec['noise']=='coupled' and spec['layout']=='common':
        previous=Path(manifest['previous_repair_path']).parent/'data'/key
        if (previous/'result.json').exists():
            baseline={**json.loads((previous/'result.json').read_text()),
                      **json.loads((previous/'trace.json').read_text())}
    original_margin, original_filter = cbf.feasibility_margin, cbf.CBFSafetyFilter.filter
    constraint = {}

    def margin(matrix, bound, speed, sides=32):
        constraint.update(matrix=matrix.copy(), bound=bound.copy())
        return original_margin(matrix, bound, speed, sides)

    def filtered(self, nominal, position, peers, dt, **kwargs):
        command, stats = original_filter(self, nominal, position, peers, dt, **kwargs)
        if stats['infeasible']:
            minimum = minimum_constraint_speed(constraint['matrix'], constraint['bound'])
            failures.append(plain(dict(frame=state.get('frame'), aircraft=kwargs.get('drone_id'),
                position=position, nominal=nominal, stats=stats, minimum_constraint_speed_m_s=minimum,
                speed_limit_m_s=self.max_speed,
                classification=('independent_constraints_infeasible' if minimum is None or
                                minimum > self.max_speed + 1e-7 else 'solver_failure'),
                matrix=constraint['matrix'], bound=constraint['bound'])))
        return command, stats

    cbf.feasibility_margin, cbf.CBFSafetyFilter.filter = margin, filtered
    original_env, original_experiment = me.MultiRobotEnv, tm.Experiment
    original_ground_controller = me.GroundRobotController
    if spec.get('save_replans'):
        from ground.global_planner import GlobalPlanner
        original_plan = GlobalPlanner.plan_path

        def audited_plan(planner, start_pos, goal_pos, graph=None):
            caller=inspect.currentframe().f_back
            local=caller.f_locals
            controller=local.get('self')
            now=float(local.get('current_time',-1.))
            path=original_plan(planner,start_pos,goal_pos,graph)
            entry=next((p for p in (path or []) if np.linalg.norm(np.asarray(p)-start_pos)>1.),None)
            direction=np.asarray(goal_pos)-start_pos
            projection=(float((np.asarray(entry)-start_pos)@direction/max(np.linalg.norm(direction),1e-12))
                        if entry is not None else None)
            event=dict(time=now,start=start_pos,goal=goal_pos,entry=entry,entry_progress_m=projection,
                route_blocked=bool(local.get('route_blocked')),needs_replan=bool(local.get('needs_replan')),
                waiting=bool(getattr(getattr(controller,'global_planner',None),'waiting_at_frontier',False)),
                path=path,using_graph=planner.using_graph)
            replans.append(plain(event))
            if (now>=30. and projection is not None and projection < -3.
                    and len(snapshots)<4 and graph is not None and hasattr(controller,'_last_semantic_map')):
                snapshots.append(plain(dict(**event,graph=graph.to_message(),
                    semantic=controller._last_semantic_map,resolution=controller.resolution,
                    heading=controller.heading,world_bounds=controller.world_bounds,
                    failed_connections=getattr(controller,'_failed_local_connections',[]),
                    certified_entries=[dict(node=str(node),position=data['position'],
                        clear=bool(controller.graph_fusion.bridge_check(start_pos,data['position'])))
                        for node,data in graph.graph.nodes(data=True)
                        if controller.graph_fusion.bridge_check is not None
                        and np.linalg.norm(np.asarray(data['position'])-start_pos)<=15.])))
            del caller
            return path

        GlobalPlanner.plan_path=audited_plan

    class ScheduledGroundController(original_ground_controller):
        def get_action(self, observation):
            observation = dict(observation)
            observation['motion_held'] = observation['full_state']['time'] < 30.
            return super().get_action(observation)

    me.GroundRobotController = ScheduledGroundController
    if spec['noise'] == 'coupled':
        me.RelayChannel = coupled_relay_class(me.RelayChannel)

    class ControlledEnv(original_env):
        def add_drone(self, x, y, z=1., **kwargs):
            if spec['layout'] == 'common':
                x, y = fleet_positions(case['start'], spec['fleet'])[len(self.drones)]
            return super().add_drone(x, y, z, **kwargs)

    class RecordingExperiment(original_experiment):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            state['experiment'] = self
            np.testing.assert_allclose(self.start, case['start'], rtol=0, atol=1e-10)
            np.testing.assert_allclose(self.goal, case['goal'], rtol=0, atol=1e-10)
            actual = [[*o.position, o.radius] for o in self.env.obstacles]
            np.testing.assert_allclose(actual, case['obstacles'], rtol=0, atol=1e-10)
            assert self.ground_controller.local_planner.safety_distance == 1.5
            assert self.ground_controller.local_planner.escape_distance == .5
            assert all(c.graph_builder.min_clearance == 1.5 for c in self.controllers)
            if spec['noise'] == 'coupled':
                for c in self.controllers:
                    previous = c.get_action
                    name = c.drone_id
                    def controlled_action(obs, previous=previous, name=name):
                        frame = round(obs['full_state']['time'])
                        with action_stream(spec['sample_seed'], name, frame):
                            return previous(obs)
                    c.get_action = controlled_action
            self.record(0)

        def _ground_action(self, neighbors=None):
            action = super()._ground_action(neighbors)
            if self.env.time < 30:
                action = dict(action)
                action['motion'] = (0., 0.)
            return action

        def record(self, step):
            points = [np.asarray(self.obs[r]['self']['position'])[:2].tolist()
                      for r in self.ids + [self.ground]]
            trajectory.append(dict(step=step, positions=points))
            if step == 0:
                state['ground_min_swept_clearance_m'] = swept_clearance(
                    points[-1], points[-1], case['obstacles'])
            if step <= 30:
                state['hold_displacement_m'] = max(state['hold_displacement_m'],
                    float(np.linalg.norm(np.asarray(points[-1]) - self.start)))
                assert state['hold_displacement_m'] < 1e-9
                if spec['runtime'] not in ('archived','fusion_only'):
                    assert not self.ground_controller._route_recovery, 'Intentional hold triggered route recovery'
                    assert not self.ground_controller._allow_local_escape, 'Intentional hold reduced clearance'
            if baseline is not None:
                expected = baseline['trajectory'][step]['positions']
                err = float(np.max(np.abs(np.asarray(points)-expected)))
                state['parity_max_position_error_m'] = max(state['parity_max_position_error_m'], err)
                assert err < 1e-7, 'Archived replay diverged'
            if len(trajectory) > 1:
                before, after = (np.asarray(t['positions']) for t in trajectory[-2:])
                clearance = swept_clearance(before[-1], after[-1], case['obstacles'])
                state['ground_min_swept_clearance_m'] = min(state['ground_min_swept_clearance_m'], clearance)
                state['ground_swept_violations'] += int(clearance < .3-1e-7)
                p = np.column_stack((before, [15.]*spec['fleet']+[0.]))
                q = np.column_stack((after, [15.]*spec['fleet']+[0.]))
                for i in range(spec['fleet']):
                    for j in range(i+1, spec['fleet']):
                        separation = swept_clearance(before[i]-before[j], after[i]-after[j], [[0.,0.,0.]])
                        state['uav_swept_violations'] += int(separation < 15.-1e-7)
                    visible = {str(j): v for j,v in enumerate(p) if j != i and np.linalg.norm(v-p[i]) <= 95.+1e-9}
                    required = relative_neighborhood(p[i], visible, 95.)
                    for other in required:
                        state['radio_step_violations'] += int(np.linalg.norm(q[i]-q[int(other)]) > 95.+1e-7)
            planner = self.ground_controller.global_planner
            if step % 10 == 0 or self.goal_reached:
                diagnostics.append(dict(step=step, position=points[-1],
                    remaining_m=float(np.linalg.norm(np.asarray(points[-1])-self.goal)),
                    path=planner.get_remaining_path() or [], waiting=planner.waiting_at_frontier,
                    using_graph=planner.using_graph,
                    route_recovery=bool(getattr(self.ground_controller,'_route_recovery',False))))

        def step(self, frame):
            state['frame'] = frame
            result = super().step(frame)
            self.record(frame+1)
            return result

    tm.Experiment, me.MultiRobotEnv = RecordingExperiment, ControlledEnv

    def progress(value):
        write_json(out/'progress.json', dict(**spec, pid=os.getpid(), status='running',
            updated_at=utc(), wall_seconds=time.monotonic()-started, **value))

    try:
        with torch.inference_mode():
            rollout, summary = tm.run_episode(model, config, case['seed'],
                tm.fixed_schedule(config, config['episodes']-1), True, progress=progress)
        reason = ('cbf_infeasible' if summary['safety_terminated'] else
                  'vehicle_arrival' if summary['goal_reached'] else
                  'budget_exhausted' if summary['budget_exhausted'] else 'other')
        assert len(rollout) <= summary['episode_budget_steps']
        safety = {k: summary[k] for k in SAFETY}
        audits = {k:v for k,v in state.items() if k not in ('experiment','frame')}
        passed = not any(safety.values()) and not any(audits[k] for k in
                    ('ground_swept_violations','uav_swept_violations','radio_step_violations'))
        points = np.asarray([r['positions'][-1] for r in trajectory])
        travel = float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
        np.testing.assert_allclose(travel, summary['ground_distance'], rtol=0, atol=1e-7)
        if baseline is not None and not spec['smoke_steps']:
            assert reason == baseline['termination'] and len(rollout) == baseline['steps']
        result = dict(**spec, status='complete', termination=reason, steps=len(rollout),
            success=bool(summary['goal_reached']), strict_success=bool(reason=='vehicle_arrival' and passed),
            budget=summary['episode_budget_steps'], remaining_m=summary['ground_goal_distance'],
            travel_m=travel, safety=safety, audits=audits, loop=loop_statistics(trajectory),
            cbf_failures=failures, wall_seconds=time.monotonic()-started)
        write_json(out/'trace.json', dict(summary=summary, trajectory=trajectory, diagnostics=diagnostics,
                                        replans=replans,planning_snapshots=snapshots))
    except Exception:
        result = dict(**spec, status='error', traceback=traceback.format_exc(),
                      wall_seconds=time.monotonic()-started)
    finally:
        if state.get('experiment') is not None:
            state['experiment'].env.close()
    write_json(out/'result.json', result)
    write_json(out/'progress.json', result)
    return 0 if result['status']=='complete' else 1


def prepare(root, args):
    root.mkdir(parents=True, exist_ok=False)
    (root/'data').mkdir()
    for name in ('best.pt','config.json','cases.json'):
        shutil.copy2(ARCHIVE/name,root/name)
    for name in ('validate_v9_fleet_controlled.py','fleet_controlled_support.py'):
        shutil.copy2(Path(__file__).with_name(name),root/name)
    runtimes = {}
    for mode in args.runtimes:
        destination = root/('runtime_'+mode)
        shutil.copytree(ARCHIVE/'runtime_sources', destination,
                       ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in RUNTIME_OVERLAYS[mode]:
            source=(REPO/'src/teamed-planning/scripts'/name if mode=='candidate'
                    else PREVIOUS_REPAIR/name)
            shutil.copy2(source,destination/name)
        runtimes[mode] = hashes(destination)
        differences = [name for name,h in runtimes[mode].items()
            if not (ARCHIVE/'runtime_sources'/name).exists() or digest(ARCHIVE/'runtime_sources'/name)!=h]
        assert set(differences) <= set(OVERLAY)
    cases=json.loads((root/'cases.json').read_text())
    config=json.loads((root/'config.json').read_text())
    import numpy as np
    jobs=[]
    for mode in args.runtimes:
        for case in args.cases:
            for sample in range(args.samples):
                seed=int(np.random.SeedSequence([config['seed'],cases[str(case)]['seed'],sample]).generate_state(1)[0])
                for fleet in args.fleets:
                    fleet_positions(cases[str(case)]['start'],fleet)
                    key=f'case{case}_r{sample}_n{fleet}_{mode}'
                    jobs.append(dict(key=key,case=case,sample=sample,sample_seed=seed,fleet=fleet,
                        runtime=mode,noise=args.noise,layout=args.layout,smoke_steps=args.smoke_steps,
                        save_replans=args.save_replans))
    manifest=dict(created_at=utc(),jobs=jobs,runtimes=runtimes,overlay=list(OVERLAY),archive_path=str(ARCHIVE),
        previous_repair_path=str(PREVIOUS_REPAIR),runtime_overlays={m:list(RUNTIME_OVERLAYS[m]) for m in args.runtimes},
        checkpoint_sha256=digest(root/'best.pt'), config_sha256=digest(root/'config.json'),
        cases_sha256=digest(root/'cases.json'), runner_sha256=digest(root/'validate_v9_fleet_controlled.py'),
        support_sha256=digest(root/'fleet_controlled_support.py'),
        intentional_hold='motion_held is communicated to the controller; no stall clock is consumed before departure',
        policy='Frozen v9 best576; same actor, observations, mapping, reward, and independent CBF',
        ground='1.5m normal clearance, existing 0.5m escape, 0.3m physical swept audit',
        repaired_sensor='Additional 0.1m local ground raster in the same 8m footprint; UAV sensor unchanged',
        safety='15m separation; 95m 3D radio; 6m/s UAV; no coordinated/global safety fallback',
        success='2m arrival; original budget includes the 30s startup hold',
        geometry='Truth is used only by offline physical audit and scene construction',
        scope='Three selected scenes, paired sampling repeats; not independent map generalization')
    write_json(root/'manifest.json',manifest)
    return manifest


def execute(root, spec):
    out=root/'data'/spec['key'];out.mkdir(exist_ok=True)
    with (out/'worker.log').open('w') as stream:
        proc=subprocess.Popen([sys.executable,'-I','-B','-u',str(root/'validate_v9_fleet_controlled.py'),
                               '--worker',str(root),spec['key']],stdout=stream,stderr=subprocess.STDOUT)
        write_json(out/'process.json',dict(pid=proc.pid,started=process_identity(proc.pid),started_at=utc()))
        code=proc.wait()
    path=out/'result.json'
    result=(json.loads(path.read_text()) if path.exists() else
            dict(**spec,status='error',returncode=code,reason='worker_did_not_write_result'))
    result['returncode']=code
    return result


def run(root, manifest, workers):
    results=[]
    def update(status):
        rows=[]
        for spec in manifest['jobs']:
            p=root/'data'/spec['key']/'progress.json'
            rows.append(json.loads(p.read_text()) if p.exists() else dict(**spec,status='queued'))
        write_json(root/'status.json',dict(updated_at=utc(),status=status,supervisor_pid=os.getpid(),
            supervisor_started=process_identity(os.getpid()),
            completed=len(results),planned=len(manifest['jobs']),runs=rows))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(execute,root,spec) for spec in manifest['jobs']}
        while pending:
            done,pending=wait(pending,timeout=5.,return_when=FIRST_COMPLETED)
            for future in done:
                result=future.result();results.append(result)
                write_json(root/'results.json',results)
                print(json.dumps({k:result.get(k) for k in ('key','status','termination','steps','remaining_m')},ensure_ascii=False),flush=True)
            update('running')
    verified=dict(checkpoint_unchanged=digest(root/'best.pt')==manifest['checkpoint_sha256'],
        config_unchanged=digest(root/'config.json')==manifest['config_sha256'],
        runtimes_unchanged=all(hashes(root/('runtime_'+m))==h for m,h in manifest['runtimes'].items()),
        no_execution_errors=all(r['status']=='complete' and r['returncode']==0 for r in results),
        is_full_evaluation=not any(j['smoke_steps'] for j in manifest['jobs']))
    write_json(root/'verification.json',verified)
    update('complete' if verified['no_execution_errors'] else 'error')
    return 0 if all(v for k,v in verified.items() if k!='is_full_evaluation') else 1


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        return worker(Path(sys.argv[2]),sys.argv[3])
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--status',action='store_true',help='Read current results and verify process liveness')
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--samples',type=int,default=3)
    parser.add_argument('--cases',type=int,nargs='+',choices=(2,3,4),default=[2,3,4])
    parser.add_argument('--fleets',type=int,nargs='+',choices=(3,4),default=[3,4])
    parser.add_argument('--runtimes',nargs='+',choices=tuple(RUNTIME_OVERLAYS),default=['archived','candidate'])
    parser.add_argument('--noise',choices=('historical','coupled'),default='coupled')
    parser.add_argument('--layout',choices=('native','common'),default='common')
    parser.add_argument('--smoke-steps',type=int,default=0)
    parser.add_argument('--save-replans',action='store_true',help='Save received planning snapshots for diagnosis')
    args=parser.parse_args()
    if args.status:
        print(json.dumps(live_status(args.output.resolve()),ensure_ascii=False,indent=2))
        return 0
    if args.workers<1 or args.samples<1 or args.smoke_steps<0:
        parser.error('workers/samples must be positive and smoke steps nonnegative')
    root=args.output.resolve()
    manifest=prepare(root,args)
    return run(root,manifest,args.workers)


if __name__=='__main__':
    raise SystemExit(main())
