"""ID-independent relative-neighborhood communication backbone (v7).

Uses only current one-hop relative geometry, including the vehicle when visible.
With complete, synchronous range observations both endpoints select the same
edge. Strict lune pruning preserves a spanning tree of each connected unit-ball
graph, including distance ties. CBF must protect every selected edge for the
whole next step. This contract does NOT cover packet loss or delayed geometry.
"""
import numpy as np


def relative_neighborhood(position, neighbors, comm_range):
    p = np.asarray(position, float)
    visible = {key: np.asarray(value, float) for key, value in neighbors.items()
               if np.linalg.norm(np.asarray(value, float)-p) <= comm_range+1e-9}
    distances = {key: np.linalg.norm(q-p) for key, q in visible.items()}
    return tuple(key for key, q in visible.items() if not any(
        other != key and max(distances[other], np.linalg.norm(witness-q)) < distances[key]-1e-8
        for other, witness in visible.items()))


def received_geometry(position, observation):
    """No task cache, global fleet list or forwarded position enters geometry."""
    peers, velocities = {}, {}
    ground = None
    p = np.asarray(position, float)
    for neighbor in observation.get('neighbors', []):
        q = p+np.asarray(neighbor['relative_position'], float)
        if neighbor['type'] == 'drone':
            key = (neighbor.get('broadcast') or {}).get('drone_id', f"robot_{neighbor['id']}")
            peers[key] = q
            velocities[key] = np.asarray(neighbor.get('relative_velocity', (0., 0., 0.)), float)
        elif neighbor['type'] in ('diff_drive', 'holonomic'):
            ground = q
    return peers, velocities, ground


def backbone_metrics(drones, ground, comm_range=95.):
    """Evaluator/critic only. Never pass this global reconstruction to an actor."""
    points = {'__ground__': np.asarray(ground, float), **drones}
    adjacency = {key: relative_neighborhood(p, {k: q for k, q in points.items() if k != key}, comm_range)
                 for key, p in points.items()}
    reached, pending = set(), ['__ground__']
    while pending:
        key = pending.pop()
        if key not in reached:
            reached.add(key)
            pending.extend(adjacency[key])
    lengths = [np.linalg.norm(points[k]-points[j]) for k, links in adjacency.items() for j in links]
    return len(reached) == len(points), max(lengths, default=0.)
