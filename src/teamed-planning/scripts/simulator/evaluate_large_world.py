"""Frozen 1505-episode policy transfer: 1000m maps, unchanged sensing/links."""
import argparse,contextlib,hashlib,json,math,os,shutil,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import networkx as nx
import numpy as np
import torch
from drone.mappo import ActorCritic
from simulator.mappo_experiment import Experiment
from simulator.large_world_scenarios import generate_large_scene,CircleGeometry
from simulator.train_mappo import atomic_json,fixed_schedule


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def graph_audit(graph,start,goal,geometry,ledger):
    if graph is None or not len(graph):return dict(connected=False,length=None,points=[],invalid_nodes=0,invalid_edges=0)
    g=graph.graph
    valid=g.subgraph([n for n,d in g.nodes(data=True) if geometry.clear(d['position'],d['position'])]).copy()
    bad=[(a,b) for a,b in valid.edges if not geometry.clear(valid.nodes[a]['position'],valid.nodes[b]['position'])]
    valid.remove_edges_from(bad)
    result=dict(connected=False,length=None,points=[],invalid_nodes=len(g)-len(valid),invalid_edges=len(bad))
    for a,b in valid.edges:
        valid.edges[a,b]['physical_length']=float(np.linalg.norm(np.asarray(valid.nodes[a]['position'])-valid.nodes[b]['position']))
    source,target=('evaluation_start',),('evaluation_goal',)
    for node,p in ((source,start),(target,goal)):
        for other,data in list(valid.nodes(data=True)):
            if 'position' not in data:continue
            q=data['position'];distance=float(np.linalg.norm(np.asarray(q)-p))
            if distance<=10. and geometry.clear(p,q) and ledger.connector_clear(p,q):
                valid.add_edge(node,other,physical_length=distance)
        valid.add_node(node)
    try:
        route=nx.shortest_path(valid,source,target,weight='physical_length')
        result.update(connected=True,length=float(nx.path_weight(valid,route,'physical_length')),
            points=[list(start)]+[list(g.nodes[n]['position']) for n in route[1:-1]]+[list(goal)])
    except nx.NetworkXNoPath:pass
    return result


def write_report(report,output):
    lines=['# 1505轮模型：1000×1000米迁移测试','',f"状态：{report['status']}；已完成{len(report['records'])}/{len(report['seeds'])*len(report['distances'])}次。",'',
        '冻结同一个1505轮模型，不训练。传感器24×24米、2米分辨率、通信95米、3机、高度15米、速度与原实验不变，原12维特征及/100归一化保持不变。',
        '障碍半径2–5米，按原障碍生成区域22/80²的空间密度扩展至约3438个；不是按整个120米地图的平均密度稀释。每个种子一张地图，四档距离共享障碍物，起终点及参考路径均经过离线可行性检查。','',
        '以原始观测支持的任务图连通为完成条件，首次连通后再运行30秒；车辆到达不结束回合。步数预算为max(400,ceil(2.5×参考路径长度+120))。参考为1米栅格、物理净空1.5米外加半格对角线裕量、禁止对角穿角的可行路径，非严格连续最优解。',
        '真值只用于生成、参考路线和独立几何审计，不输入控制器/Actor。通信检查包含初始状态与各步端点的三维距离。此为单个训练模型、每档2场景的探索性测试，不能代替大地图重新训练与多种子实验。','',
        '|距离(m)|种子|终止原因|图连通|几何审计连通|首次连通(s)|安全图路径/参考|最大链路(m)|超距步数|车辆剩余(m)|',
        '|---|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in report['records']:
        ratio='—' if r['path_over_reference'] is None else f"{r['path_over_reference']:.3f}"
        lines.append(f"|{r['distance']}|{r['seed']}|{r['termination']}|{bool(r.get('graph_task_connected',0))}|{r['audit']['connected']}|{r.get('graph_first_connection_step',-1):.0f}|{ratio}|{r['max_link_distance_m']:.3f}|{r['link_violation_steps']}|{r.get('ground_goal_distance',float('nan')):.2f}|")
    lines+=['','当前任务逐步进度见`episode_progress.json`；场景、参考路径、模型哈希和汇总见`results.json`；各案例JSON保留轨迹与最终图。训练保持停止，不会自动重训。','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def evaluate(run,output,seeds,distances,resume=False):
    checkpoint=run/'latest.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['training_episodes']==1505 and saved['environment_version']==4 and saved['feature_version']==1
    if json.loads((run/'progress.json').read_text())['status']!='stopped':raise ValueError('Training must remain stopped')
    config=saved['config'];scripts=Path(__file__).resolve().parents[1]
    training_seeds={json.loads(line)['scene_seed'] for line in (run/'training.jsonl').read_text().splitlines() if line.strip()}
    assert not(set(seeds)&(training_seeds|set(config['test_seeds'])|set(config['validation_seeds'])))
    hashes={str(p.relative_to(scripts)).replace('\\','/'):digest(p) for p in scripts.rglob('*.py')}
    if resume:
        report=json.loads((output/'results.json').read_text())
        assert report['checkpoint']['sha256']==digest(checkpoint)
        if report['source_sha256']!=hashes:
            changed=[n for n in set(report['source_sha256'])|set(hashes) if report['source_sha256'].get(n)!=hashes.get(n)]
            # A failed initialization has no measurements to mix; preserve its
            # report and allow an instrumentation-only repair before first case.
            if report['records'] or changed!=['simulator/evaluate_large_world.py']:
                raise ValueError(f'Resume sources differ: {changed}')
            atomic_json(output/'initialization_failure.json',report)
            report['source_sha256']=hashes
            shutil.copy2(scripts/'simulator/evaluate_large_world.py',output/'sources/simulator/evaluate_large_world.py')
        report.pop('error',None)
        assert report['seeds']==seeds and report['distances']==distances
        report['status']='running'
    else:
        if output.exists() and any(output.iterdir()):raise ValueError('Use a new directory or --resume')
        output.mkdir(parents=True)
        report=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
            protocol='large-world-transfer-v1',world_size_m=1000.,seeds=seeds,distances=distances,
            checkpoint=dict(path=str(checkpoint),episode=1505,sha256=digest(checkpoint)),
            unchanged=dict(sensor_width_m=24.,sensor_resolution_m=2.,communication_range_m=95.,
                drones=3,height_m=15.,drone_max_speed=6.,ground_max_speed=1.,feature_version=1,feature_distance_scale=100.),
            source_sha256=hashes,records=[],scenes={})
        for name in hashes:
            target=output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(scripts/name,target)
    atomic_json(output/'results.json',report);write_report(report,output)
    torch.set_num_threads(1);model=ActorCritic();model.load_state_dict(saved['model']);model.eval()
    completed={(r['seed'],r['distance']) for r in report['records']}
    try:
        for seed in seeds:
            if str(seed) not in report['scenes']:
                report['scenes'][str(seed)]=generate_large_scene(seed,distances)
                atomic_json(output/'results.json',report)
            scene=report['scenes'][str(seed)]
        # Short-distance checks first, then progressively harder tasks.
        for distance in distances:
            for seed in seeds:
                if (seed,distance) in completed:continue
                scene=report['scenes'][str(seed)];key=str(int(distance))
                start=np.asarray(scene['start']);goal=np.asarray(scene['goals'][key]);circles=scene['circles']
                reference=scene['references'][key];budget=max(400,math.ceil(2.5*reference['length']+120))
                torch.manual_seed(seed);np.random.seed(seed);started=time.monotonic()
                geometry=CircleGeometry(circles);first_verified=-1;first_audit=None;reason='budget_exhausted';error=None
                with open(os.devnull,'w',encoding='utf-8') as sink,contextlib.redirect_stdout(sink):
                    env=Experiment(seed,model,stochastic=False,drop=config['drop'],
                        exploration_weight=fixed_schedule(config,config['episodes']-1)['eta'],exploration_candidates=config['exploration_candidates'],
                        communication_mode=config['communication_mode'],safety_filter=config['safety_filter'],
                        spacing_penalty_weight=config['spacing_penalty_weight'],intervention_weight=config['intervention_weight'],
                        moving_ground=True,reward_mode=config['reward_mode'],ground_speed=config['ground_speed'],
                        path_reward_scale=config['path_reward_scale'],world_size=1000.,start=start,
                        scene=(goal,circles,scene['attempts']),stop_on_ground_arrival=False)
                    previous=[env.env.drones[i].position.copy() for i in range(3)]
                    old_ground=np.asarray(env.obs[env.ground]['self']['position']).copy();max_links=np.zeros(3);violation_steps=0
                    try:
                        with torch.inference_mode():
                            for frame in range(budget):
                                try:env.step(frame)
                                except RuntimeError as exception:
                                    error=str(exception);reason='controller_infeasible';break
                                positions=np.asarray(env.trace[-1]['positions']);ground=np.asarray(env.trace[-1]['ground_position'])
                                before=np.asarray([np.linalg.norm(previous[0]-old_ground)]+[np.linalg.norm(previous[i]-previous[i-1]) for i in (1,2)])
                                after=np.asarray([np.linalg.norm(positions[0]-ground)]+[np.linalg.norm(positions[i]-positions[i-1]) for i in (1,2)])
                                swept=np.maximum(before,after);max_links=np.maximum(max_links,swept)
                                violation_steps+=int(swept.max()>95.+1e-5)
                                env.trace[-1]['chain_distances_m']=after.tolist()
                                previous=positions;old_ground=ground
                                if env._graph_metrics['graph_task_connected'] and first_verified<0:
                                    check=graph_audit(env.ground_controller.unified_graph,start,goal,geometry,env.ledger)
                                    if check['connected']:first_verified=frame+1;first_audit=check
                                if frame%10==0:
                                    atomic_json(output/'episode_progress.json',dict(distance=distance,seed=seed,step=frame+1,
                                        budget=budget,completed_cases=len(report['records']),first_verified=first_verified,
                                        ground_goal_distance=env.trace[-1]['ground_goal_distance'],
                                        graph_nodes=env._graph_metrics['graph_nodes'],max_link_distance_m=float(max_links.max()),
                                        wall_seconds=round(time.monotonic()-started,2)))
                                if (output/'STOP').exists():reason='user_stop';break
                                if first_verified>=0 and frame+1>=first_verified+30:reason='graph_completed_plus_30s';break
                        summary=env.summary() if env.trace else {}
                        # The original summary calls total UAV flight 'distance';
                        # keep it separate from the task's start-goal distance.
                        if 'distance' in summary:summary['uav_distance_m']=summary.pop('distance')
                        final_graph=env.ground_controller.unified_graph
                        audit=graph_audit(final_graph,start,goal,geometry,env.ledger)
                        row=dict(seed=seed,distance=distance,budget=budget,termination=reason,error=error,
                            first_geometry_verified_step=first_verified,audit={k:v for k,v in audit.items() if k!='points'},
                            path_over_reference=audit['length']/reference['length'] if audit['connected'] else None,
                            reference_length=reference['length'],max_link_distance_m=float(max_links.max()),
                            link_violation_steps=violation_steps,wall_seconds=round(time.monotonic()-started,2),**summary)
                        assert violation_steps==int(summary.get('chain_link_violation',0))
                        atomic_json(output/f'd{int(distance)}_s{seed}.json',dict(summary=row,trace=env.trace,
                            final_graph=final_graph.to_message() if final_graph is not None else None,
                            final_route=audit['points'],first_audit=first_audit))
                    finally:env.env.close()
                assert digest(checkpoint)==report['checkpoint']['sha256']
                report['records'].append(row);atomic_json(output/'results.json',report);write_report(report,output)
                print(json.dumps({k:row.get(k) for k in ('seed','distance','termination','graph_task_connected',
                    'first_geometry_verified_step','path_over_reference','max_link_distance_m','link_violation_steps','wall_seconds')}),flush=True)
                if reason=='user_stop':report['status']='stopped';atomic_json(output/'results.json',report);write_report(report,output);return
        report.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(output/'results.json',report);write_report(report,output)
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());atomic_json(output/'results.json',report);write_report(report,output);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42000,42001])
    parser.add_argument('--distances',type=int,nargs='+',default=[100,250,500,750]);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();evaluate(args.run.resolve(),args.output.resolve(),args.seeds,args.distances,args.resume)
