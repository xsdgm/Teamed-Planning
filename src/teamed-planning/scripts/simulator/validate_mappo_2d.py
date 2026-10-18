"""2D validation of frozen best/stopped checkpoints on explicit new scenes."""
import argparse
import contextlib
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from drone.mappo import ActorCritic
from simulator.mappo_experiment import Experiment
from simulator.scenarios import ENVIRONMENT_VERSION,generate_scene
from simulator.train_mappo import atomic_json,fixed_schedule


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def position3(value):
    p=np.asarray(value,dtype=float)
    return np.r_[p[:2],p[2] if len(p)>2 else 0.]


def distances(positions,ground):
    points=np.asarray(positions)
    return [float(np.linalg.norm(points[0]-position3(ground)))] + [
        float(np.linalg.norm(points[i]-points[i-1])) for i in (1,2)]


def evaluate(run,output,seeds):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new/empty output directory')
    progress=json.loads((run/'progress.json').read_text())
    if progress['status'] not in ('stopped','completed'):
        raise ValueError('Stop training before freezing this evaluation')
    config=json.loads((run/'config.json').read_text())
    if config.get('world_size',120.) != 120.:
        raise ValueError('This runner is the 120m frozen-model protocol; use train_mappo.validation for large-world runs')
    training={json.loads(line)['scene_seed'] for line in (run/'training.jsonl').read_text().splitlines() if line.strip()}
    if len(seeds)!=len(set(seeds)) or set(seeds)&(training|set(config['validation_seeds'])|set(config['test_seeds'])):
        raise ValueError('Evaluation seeds must be new and disjoint from existing splits')
    scripts=Path(__file__).resolve().parents[1]
    source_hashes=json.loads((run/'provenance.json').read_text())['source_sha256']
    changed=[n for n,h in source_hashes.items() if not (scripts/n).is_file() or digest(scripts/n)!=h]
    current_hashes={p.relative_to(scripts).as_posix():digest(p)
                    for p in sorted(scripts.rglob('*.py'))}
    models={};checkpoints={}
    for label,name in [('best','best.pt'),('stopped','latest.pt')]:
        path=run/name;saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['environment_version']==ENVIRONMENT_VERSION==4
        assert saved['config']==config
        model=ActorCritic();model.load_state_dict(saved['model']);model.eval()
        models[label]=model
        checkpoints[label]=dict(path=str(path),sha256=digest(path),episode=saved['training_episodes'])
    output.mkdir(parents=True)
    report=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),environment_version=4,
        seeds=seeds,config=config,checkpoints=checkpoints,source_match=not changed,changed_since_training=changed,
        evaluation_source_sha256=current_hashes,new_seeds=True,
        deterministic=True,communication_limit_m=95.,records=[],scenes={})
    atomic_json(output/'results.json',report);torch.set_num_threads(1)
    for seed in seeds:
        goal,circles,attempts=generate_scene(seed)
        report['scenes'][str(seed)]=dict(goal=goal.tolist(),circles=[(p.tolist(),r) for p,r in circles],attempts=attempts)
        for label,model in models.items():
            torch.manual_seed(seed);np.random.seed(seed);started=time.monotonic()
            with open(os.devnull,'w',encoding='utf-8') as sink,contextlib.redirect_stdout(sink):
                env=Experiment(seed,model,stochastic=False,drop=config['drop'],
                    exploration_weight=fixed_schedule(config,config['episodes']-1)['eta'],exploration_candidates=config['exploration_candidates'],
                    communication_mode=config['communication_mode'],safety_filter=config['safety_filter'],
                    spacing_penalty_weight=config['spacing_penalty_weight'],intervention_weight=config['intervention_weight'],
                    moving_ground=config['moving_ground'],reward_mode=config['reward_mode'],ground_speed=config['ground_speed'],
                    path_reward_scale=config['path_reward_scale'])
                initial_positions=[position3(env.obs[r]['self']['position']).tolist() for r in env.ids]
                initial_ground=position3(env.obs[env.ground]['self']['position']).tolist()
                all_distances=[distances(initial_positions,initial_ground)]
                try:
                    with torch.inference_mode():
                        for frame in range(config['steps']):
                            env.step(frame)
                            links=distances(env.trace[-1]['positions'],env.trace[-1]['ground_position'])
                            env.trace[-1]['chain_distances_m']=links
                            all_distances.append(links)
                            if frame%20==0:
                                atomic_json(output/'episode_progress.json',dict(model=label,seed=seed,step=frame+1,
                                    completed_cases=len(report['records']),max_link_distance_m=float(np.max(all_distances))))
                            if env.done:break
                    summary=env.summary();graph=env.ground_controller.unified_graph
                    link_array=np.asarray(all_distances)
                    extra=dict(max_link_distance_m=float(link_array.max()),
                        max_distance_by_link_m=link_array.max(axis=0).tolist(),
                        independent_link_violation_steps=int((np.maximum(link_array[:-1],link_array[1:]).max(axis=1)>95.+1e-5).sum()))
                    assert extra['independent_link_violation_steps']==int(summary['chain_link_violation'])
                    atomic_json(output/f'{label}_{seed}.json',dict(summary=dict(**summary,**extra),
                        initial_positions=initial_positions,initial_ground=initial_ground,trace=env.trace,
                        graph=graph.to_message() if graph is not None else None))
                finally:env.env.close()
            row=dict(model=label,seed=seed,wall_seconds=round(time.monotonic()-started,2),**summary,**extra)
            report['records'].append(row);atomic_json(output/'results.json',report)
            print(json.dumps({k:row[k] for k in ['model','seed','graph_task_connected','goal_reached',
                'max_link_distance_m','independent_link_violation_steps','wall_seconds']}),flush=True)
    for checkpoint in checkpoints.values():assert digest(Path(checkpoint['path']))==checkpoint['sha256']
    report.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output/'results.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[31000,31001,31002,31003,31004])
    args=parser.parse_args();evaluate(args.run.resolve(),args.output.resolve(),args.seeds)
