"""Deterministic, feasible 2D scene generation; truth stays in the environment."""
import numpy as np
from scipy.ndimage import label

ENVIRONMENT_VERSION = 4  # Observed-space sampling, old-edge refresh and graph metrics.
WORLD_SIZE = 120
SCENE_CLEARANCE = 3.5


def free_grid(circles, resolution=1., clearance=SCENE_CLEARANCE):
    """Conservative free cell centres for generation and independent evaluation."""
    coordinates = (np.arange(int(WORLD_SIZE/resolution)) + .5)*resolution
    xy = np.stack(np.meshgrid(coordinates, coordinates), -1)
    free = ((xy >= clearance).all(-1) & (xy <= WORLD_SIZE-clearance).all(-1))
    for position, radius in circles:
        free &= np.linalg.norm(xy-np.asarray(position), axis=-1) > radius+clearance
    return free


def scene_feasible(goal, circles, start=(15.,60.)):
    # Protect the whole 2m success region, plus vehicle and raster clearance.
    if any(np.linalg.norm(np.asarray(goal)-p) <= radius+SCENE_CLEARANCE
           for p, radius in circles):
        return False
    free = free_grid(circles)
    components, _ = label(free)  # four-connected; no diagonal corner cutting
    sx, sy = np.floor(start).astype(int)
    gx, gy = np.floor(goal).astype(int)
    return bool(components[sy,sx] and components[sy,sx] == components[gy,gx])


def generate_scene(seed, obstacle_count=22, max_attempts=128):
    rng = np.random.default_rng(seed)
    for attempt in range(1,max_attempts+1):
        goal = np.array([rng.uniform(85,110),rng.uniform(25,95)])
        circles = [(rng.uniform(30,110,2),float(rng.uniform(2,5)))
                   for _ in range(obstacle_count)]
        if scene_feasible(goal,circles):
            return goal,circles,attempt
    raise RuntimeError(f'No feasible scene for seed {seed} after {max_attempts} attempts')


if __name__=='__main__':
    import argparse
    import json
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Audit feasible scene generation without training')
    parser.add_argument('--start',type=int,default=10000)
    parser.add_argument('--count',type=int,default=8000)
    parser.add_argument('--output',type=Path,default=Path('artifacts/rl-fixes/scene-audit.json'))
    args=parser.parse_args()
    attempts=[]
    for seed in range(args.start,args.start+args.count):
        _,_,tries=generate_scene(seed)
        attempts.append(tries)
        if len(attempts)%500==0:
            print(f'Checked {len(attempts)}/{args.count} feasible scenes',flush=True)
    report=dict(environment_version=ENVIRONMENT_VERSION,start=args.start,count=args.count,
        failures=0,resampled=sum(tries>1 for tries in attempts),max_attempts=max(attempts),
        mean_attempts=float(np.mean(attempts)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
