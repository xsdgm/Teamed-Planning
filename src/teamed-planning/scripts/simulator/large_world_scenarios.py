"""Large training/evaluation maps; reference routes never enter policies."""
import heapq
import math
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

ORIGINAL_ACTIVE_DENSITY=22./(80.*80.)


def raster_free(circles,world=1000.,clearance=3.5,resolution=1.):
    n=int(world/resolution)
    coordinates=(np.arange(n)+.5)*resolution
    free=(coordinates[:,None]>=clearance)&(coordinates[:,None]<=world-clearance)&(
        coordinates[None,:]>=clearance)&(coordinates[None,:]<=world-clearance)
    for position,radius in circles:
        p=np.asarray(position);r=radius+clearance
        lo=np.maximum(0,np.floor((p-r)/resolution-.5).astype(int))
        hi=np.minimum(n,np.ceil((p+r)/resolution+.5).astype(int))
        if (hi<=lo).any():continue
        xs=coordinates[lo[0]:hi[0]];ys=coordinates[lo[1]:hi[1]]
        free[lo[1]:hi[1],lo[0]:hi[0]] &= (xs[None,:]-p[0])**2+(ys[:,None]-p[1])**2>r*r
    return free


def reference_route(free,start,goal,resolution=1.):
    sx,sy=np.floor(np.asarray(start)/resolution).astype(int)
    gx,gy=np.floor(np.asarray(goal)/resolution).astype(int)
    if not free[sy,sx] or not free[gy,gx]:raise ValueError('Blocked reference endpoint')
    source,target=(sy,sx),(gy,gx)
    costs={source:0.};previous={};queue=[(math.hypot(gx-sx,gy-sy),0.,source)]
    while queue:
        _,cost,(y,x)=heapq.heappop(queue)
        if cost!=costs[y,x]:continue
        if (y,x)==target:
            path=[target]
            while path[-1]!=source:path.append(previous[path[-1]])
            return dict(length=cost*resolution,points=[[(x+.5)*resolution,(y+.5)*resolution] for y,x in reversed(path)])
        for dy,dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
            ny,nx_=y+dy,x+dx
            if not(0<=ny<free.shape[0] and 0<=nx_<free.shape[1]) or not free[ny,nx_]:continue
            if dy and dx and (not free[y,nx_] or not free[ny,x]):continue
            next_cost=cost+math.hypot(dx,dy)
            if next_cost<costs.get((ny,nx_),math.inf):
                costs[ny,nx_]=next_cost;previous[ny,nx_]=(y,x)
                heapq.heappush(queue,(next_cost+math.hypot(gx-nx_,gy-ny),next_cost,(ny,nx_)))
    raise ValueError('No reference path')


def generate_large_scene(seed,distances=(100.,250.,500.,750.),world=1000.,density=ORIGINAL_ACTIVE_DENSITY,
                         start=None, heading=0.):
    rng=np.random.default_rng(seed)
    start=np.array([30.,world/2.]) if start is None else np.asarray(start,dtype=float)
    direction=np.array([np.cos(heading),np.sin(heading)])
    goals={str(int(d)):start+d*direction for d in distances}
    if any((p<=3.5).any() or (p>=world-3.5).any() for p in goals.values()):raise ValueError('Goal outside world')
    for attempt in range(1,129):
        circles=[]
        while len(circles)<round(density*world*world):
            p,r=rng.uniform(5.,world-5.,2),float(rng.uniform(2.,5.))
            if np.linalg.norm(p-start)<=r+10. or any(np.linalg.norm(p-g)<=r+3.5 for g in goals.values()):continue
            circles.append((p,r))
        feasible=raster_free(circles,world,3.5)
        components,_=label(feasible)
        sx,sy=np.floor(start).astype(int)
        if not components[sy,sx] or any(components[int(g[1]),int(g[0])]!=components[sy,sx] for g in goals.values()):continue
        reference=raster_free(circles,world,1.5+math.sqrt(.5))
        routes={d:reference_route(reference,start,g) for d,g in goals.items()}
        return dict(seed=seed,start=start.tolist(),world=world,goals={d:g.tolist() for d,g in goals.items()},
            circles=[(p.tolist(),r) for p,r in circles],attempts=attempt,density=density,
            reference_resolution_m=1.,reference_clearance_m=1.5+math.sqrt(.5),references=routes)
    raise RuntimeError(f'No feasible large scene for seed {seed}')


class CircleGeometry:
    def __init__(self,circles,world=1000.,clearance=1.5):
        self.world=world;self.clearance=clearance
        self.positions=np.asarray([p for p,r in circles]);self.radii=np.asarray([r for p,r in circles])
        self.tree=cKDTree(self.positions) if len(circles) else None

    def clear(self,a,b):
        a,b=np.asarray(a)[:2],np.asarray(b)[:2]
        if (np.minimum(a,b)<self.clearance).any() or (np.maximum(a,b)>self.world-self.clearance).any():return False
        if self.tree is None:return True
        delta=b-a;squared=float(delta@delta)
        ids=self.tree.query_ball_point((a+b)/2.,math.sqrt(squared)/2.+float(self.radii.max())+self.clearance)
        if not ids:return True
        relative=self.positions[ids]-a
        t=np.clip((relative@delta)/max(squared,1e-12),0.,1.)
        return bool((np.linalg.norm(relative-t[:,None]*delta,axis=1)>self.radii[ids]+self.clearance).all())


def generate_training_scene(seed, distances=(350,550,750), world=1000.):
    """Random task pose, same obstacle density; truth used only at setup."""
    rng=np.random.default_rng(seed ^ 0x5A17)
    distance=int(rng.choice(distances))
    heading=float(rng.uniform(-np.pi,np.pi))
    delta=distance*np.array([np.cos(heading),np.sin(heading)])
    margin=35.  # Includes initial drone formation and its complete sensor view.
    lower=np.maximum(margin,margin-delta)
    upper=np.minimum(world-margin,world-margin-delta)
    if (upper<=lower).any():raise ValueError('Task distance does not fit world with formation margin')
    start=rng.uniform(lower,upper)
    scene=generate_large_scene(seed,(distance,),world,start=start,heading=heading)
    scene['distance']=distance
    scene['goal']=scene['goals'][str(distance)]
    scene['reference_length']=scene['references'][str(distance)]['length']
    return scene
