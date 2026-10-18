"""Independent full-map ground-controller acceptance; never used by training."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from common.graph_structure import TopologicalGraph
from ground.ground_main import GroundRobotController
from simulator.env import MultiRobotEnv,EnvConfig
from simulator.scenarios import generate_scene,free_grid


def reference_graph(circles):
    resolution=2.
    free=free_grid(circles,resolution=resolution,clearance=2.)
    graph=TopologicalGraph('surveyed_reference')
    for y,x in np.argwhere(free):
        graph.add_node(f'{y}_{x}',((x+.5)*resolution,(y+.5)*resolution),
            dict(terrain_type='TRAVERSABLE',observation_count=20))
    for y,x in np.argwhere(free):
        for dy,dx in ((0,1),(1,0)):
            other=f'{y+dy}_{x+dx}'
            if other in graph.graph:
                graph.add_edge(f'{y}_{x}',other)
    return graph


def validate_scene(seed,steps=360,obstacle_count=22):
    goal,circles,attempts=generate_scene(seed,obstacle_count=obstacle_count)
    env=MultiRobotEnv(EnvConfig(world_size=(120,120),dt=1.,ground_occupancy_map_size=16,
        ground_occupancy_map_resolution=.5))
    rid=env.add_ground_robot(15.,60.,max_speed=1.)
    for position,radius in circles:
        env.add_circle_obstacle(*position,radius)
    controller=GroundRobotController('ground_0',fusion_interval=5.,max_speed=1.)
    controller.set_goal(tuple(goal))
    graph=reference_graph(circles)
    packet=dict(drone_id='surveyed_reference',position=[15.,60.,15.],graph=graph.to_message())
    observation=env.reset()[rid]
    trace=[]
    collisions=0
    for frame in range(steps):
        observation['neighbors']=[dict(type='drone',broadcast=packet)]
        observation['full_state']={'time':env.time}
        observation.pop('obstacles',None)
        action=controller.get_action(observation)
        observations,info=env.step({rid:action})
        observation=observations[rid]
        position=np.asarray(observation['self']['position'])[:2]
        remaining=float(np.linalg.norm(position-goal))
        collisions+=len(info['collisions'])
        trace.append(dict(frame=frame,position=position.tolist(),remaining=remaining))
        if remaining<=2.:
            break
    env.close()
    return dict(seed=seed,goal=goal.tolist(),attempts=attempts,steps=len(trace),
        goal_reached=trace[-1]['remaining']<=2.,remaining=trace[-1]['remaining'],
        collisions=collisions,reference_nodes=len(graph),trace=trace)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,nargs='+',default=[20000,20001,20002,20003,20004])
    parser.add_argument('--steps',type=int,default=360)
    parser.add_argument('--output',type=Path,default=Path('artifacts/ground-navigation-validation.json'))
    args=parser.parse_args()
    results=[]
    for seed in args.seeds:
        with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):
            row=validate_scene(seed,args.steps)
        results.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='trace'}),flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2),encoding='utf-8')
    if any(not row['goal_reached'] or row['collisions'] for row in results):
        raise SystemExit('Ground navigation acceptance failed; inspect saved traces')


if __name__=='__main__':
    main()
