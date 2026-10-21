"""Relay subgoals for a goal beyond the reach of the drone chain.

The vehicle drives to a temporary target within reach of the chain and
broadcasts it as its mission target, so the drones map the same stretch.
Only the vehicle's own position, its fused graph and the surveyed world
bounds are used; no map truth or reference route.
"""
import numpy as np


class SubgoalManager:
    """Hold one subgoal until the vehicle arrives near it or it fails.

    Candidates on rings around the vehicle are scored by the straight-line
    distance from the vehicle plus the distance to the goal (optimistic A*
    through unknown space) plus decaying penalties near failed subgoals.
    Ties along the straight segment go to the point closest to the goal.
    A subgoal fails when the vehicle moves `retreat` m farther from it than
    its closest approach, when for `stall` s the vehicle neither closes in
    nor sees the graph grow in the corridor toward it, or when it has not
    closed in for `max_wait` s even though the graph grows. The penalty then lets
    side or rear candidates win, which gives detours and backtracking.
    Penalties decay instead of banning, so a wrongly failed area can be
    retried later.
    """
    def __init__(self, goal, radius=200., arrival=25., snap=20., snap_distance=50.,
                 stall=180., max_wait=600., retreat=40., corridor=30., penalty_radius=60.,
                 penalty_decay=600., directions=24, world_bounds=None, margin=10.):
        if not 0 < arrival < radius or snap < 0 or not 0 < stall <= max_wait or retreat <= 0:
            raise ValueError('Require 0 < arrival < radius, snap >= 0, 0 < stall <= max_wait and retreat > 0')
        self.goal=np.asarray(goal,float)[:2]
        self.radius,self.arrival,self.retreat=float(radius),float(arrival),float(retreat)
        self.stall,self.max_wait=float(stall),float(max_wait)
        self.snap,self.snap_distance,self.corridor=float(snap),float(snap_distance),float(corridor)
        self.penalty_radius,self.penalty_decay=float(penalty_radius),float(penalty_decay)
        self.directions=int(directions)
        self.world_bounds=world_bounds
        self.margin=float(margin)
        self.target=None
        self.origin=None  # Vehicle position when the target was selected
        self.final=False
        self.snapped=False
        self.failures=[]  # (failed target, time)
        self.selected=self.arrivals=self.failed=0
        self._best=self._since=self._approached=0.
        self._nodes=0

    def update(self, position, graph, now):
        """Return (target, changed); the vehicle replans only when changed."""
        p=np.asarray(position,float)[:2]
        if self.target is None:
            return self._select(p,graph,now),True
        distance=float(np.linalg.norm(self.target-p))
        if not self.final and distance<=self.arrival:
            self.arrivals+=1
            return self._select(p,graph,now),True
        if distance>self._best+self.retreat:
            return self._fail(p,graph,now),True
        # Graph growth is counted in the fixed corridor from the selection
        # point, so driving away cannot masquerade as mapping progress.
        nodes=self._count(graph,self.origin,self.target)
        closer=distance<self._best-2.
        if closer:
            self._approached=now
        if closer or nodes>self._nodes+10:
            self._best=min(self._best,distance)
            self._nodes=max(self._nodes,nodes)
            self._since=now
        elif now-self._since>=self.stall:
            return self._fail(p,graph,now),True
        # Graph growth postpones a failure only up to `max_wait` without approach.
        if now-self._approached>=self.max_wait:
            return self._fail(p,graph,now),True
        # Close to an intermediate target, move it once onto a mapped free node
        # so it cannot remain inside an obstacle. Snapping earlier only forces
        # needless replanning while the vehicle is still far away.
        if not self.final and not self.snapped and distance<=self.snap_distance:
            node=self._snap_node(self.target,graph)
            if node is not None:
                self.target,self.snapped=node,True
                self._best=float(np.linalg.norm(node-p))
                return self.target,True
        return self.target,False

    def _fail(self, p, graph, now):
        self.failed+=1
        self.failures.append((self.target.copy(),now))
        return self._select(p,graph,now)

    def _select(self, p, graph, now):
        offset=self.goal-p
        heading=np.arctan2(offset[1],offset[0])
        angles=heading+np.linspace(-np.pi,np.pi,self.directions,endpoint=False)
        rays=np.stack([np.cos(angles),np.sin(angles)],-1)
        points=(p+self.radius*np.array([1.,2/3,1/3])[:,None,None]*rays).reshape(-1,2)
        if self.world_bounds is not None:
            xmin,ymin,xmax,ymax=self.world_bounds
            inside=((points[:,0]>=xmin+self.margin)&(points[:,0]<=xmax-self.margin)&
                    (points[:,1]>=ymin+self.margin)&(points[:,1]<=ymax-self.margin))
            points=points[inside]
        if np.linalg.norm(offset)<=self.radius or not len(points):
            points=np.vstack([self.goal,points])
        remaining=np.linalg.norm(points-self.goal,axis=1)
        scores=np.linalg.norm(points-p,axis=1)+remaining+self._penalty(points,now)
        index=int(np.lexsort((remaining,np.round(scores,3)))[0])
        self.final=bool(index==0 and remaining[0]==0.)
        self.target=points[index].copy()
        self.origin=p.copy()
        self.snapped=False
        self.selected+=1
        self._best=float(np.linalg.norm(self.target-p))
        self._nodes=self._count(graph,self.origin,self.target)
        self._since=self._approached=now
        return self.target

    def _penalty(self, points, now):
        total=np.zeros(len(points))
        for position,time in self.failures:
            near=np.linalg.norm(points-position,axis=1)<=self.penalty_radius
            total+=near*self.radius*np.exp(-(now-time)/self.penalty_decay)
        return total

    def _snap_node(self, point, graph):
        if graph is None or len(graph)==0 or not self.snap:
            return None
        nearest=graph.find_nearest_node(tuple(point))
        if nearest is None or nearest[1]>self.snap:
            return None
        return np.asarray(graph.get_node_position(nearest[0]),float)[:2]

    def _count(self, graph, start, end):
        """Graph nodes within `corridor` m of the segment start -> end."""
        if graph is None or len(graph)==0:
            return 0
        segment=end-start
        length=float(np.linalg.norm(segment))
        nearby=graph.find_nearby_nodes(tuple((start+end)/2),length/2+self.corridor)
        if not nearby:
            return 0
        points=np.asarray([graph.get_node_position(node) for node,_ in nearby],float)[:,:2]
        t=np.clip((points-start)@segment/max(length*length,1e-8),0.,1.)
        return int(np.count_nonzero(np.linalg.norm(points-start-t[:,None]*segment,axis=1)<=self.corridor))

    def stats(self):
        return dict(subgoals_selected=float(self.selected),subgoal_arrivals=float(self.arrivals),
                    subgoal_failures=float(self.failed))
