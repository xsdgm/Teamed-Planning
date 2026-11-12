"""Freeze metric navigation tasks from a recorded WildScenes map.

Reference routes are computed offline and are never policy inputs.  Distances
are feasible grid-route lengths in the unscaled recorded coordinate system,
not recording mileage or an exact optimum in continuous space.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.wildscenes_map import RasterForestMap
from simulator.wildscenes_clearance import DEFAULT_GROUND_CLEARANCE_M


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def planning_mask(forest, clearance, planning_resolution=None, endpoint_clearance=None):
    """EDT center distances minus source-cell geometry and query offset.

    Accepted centers have conservative clearance.  Every selected reference
    segment is subsequently checked against exact occupied/unknown squares.
    """
    resolution = float(planning_resolution or forest.resolution_m)
    clearance = float(clearance)
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError('clearance must be finite and nonnegative')
    ratio = resolution / forest.resolution_m
    if not np.isfinite(resolution) or resolution <= 0 or ratio < 1 or not np.isclose(ratio, round(ratio)):
        raise ValueError('planning resolution must be an integer multiple of source resolution')
    # Pad with occupied cells so EDT also accounts for the map boundary.
    distance = distance_transform_edt(np.pad(forest.free, 1, constant_values=False))[1:-1, 1:-1]
    distance *= forest.resolution_m
    h, w = np.floor(np.array(forest.shape) / ratio).astype(int)
    if not h or not w:
        raise ValueError('planning resolution is larger than the map')
    x = forest.origin_xy[0] + (np.arange(w) + .5) * resolution
    y = forest.origin_xy[1] + (np.arange(h) + .5) * resolution
    ix = np.minimum(forest.shape[1] - 1, np.floor((x - forest.origin_xy[0]) / forest.resolution_m).astype(int))
    iy = np.minimum(forest.shape[0] - 1, np.floor((y - forest.origin_xy[1]) / forest.resolution_m).astype(int))
    fine_x = forest.origin_xy[0] + (ix + .5) * forest.resolution_m
    fine_y = forest.origin_xy[1] + (iy + .5) * forest.resolution_m
    offset = np.hypot((fine_x - x)[None, :], (fine_y - y)[:, None])
    lower_bound = distance[iy[:, None], ix[None, :]] - forest.resolution_m / math.sqrt(2) - offset
    safe = forest.free[iy[:, None], ix[None, :]] & (lower_bound > clearance + 1e-9)
    if endpoint_clearance is not None:
        endpoint_safe = forest.free[iy[:, None], ix[None, :]] & (lower_bound > endpoint_clearance + 1e-9)
        return safe, resolution, x, y, endpoint_safe
    return safe, resolution, x, y


def grid_graph(component, resolution):
    """An undirected 8-neighbor graph with diagonal corner cutting disabled."""
    ids = np.full(component.shape, -1, dtype=np.int32)
    ids[component] = np.arange(int(component.sum()), dtype=np.int32)
    sources, targets, lengths = [], [], []
    specifications = (
        (component[:, :-1] & component[:, 1:], ids[:, :-1], ids[:, 1:], 1.),
        (component[:-1, :] & component[1:, :], ids[:-1, :], ids[1:, :], 1.),
        (component[:-1, :-1] & component[1:, 1:] & component[:-1, 1:] & component[1:, :-1],
         ids[:-1, :-1], ids[1:, 1:], math.sqrt(2)),
        (component[:-1, 1:] & component[1:, :-1] & component[:-1, :-1] & component[1:, 1:],
         ids[:-1, 1:], ids[1:, :-1], math.sqrt(2)),
    )
    for allowed, source, target, step in specifications:
        a, b = source[allowed], target[allowed]
        sources.extend((a, b))
        targets.extend((b, a))
        lengths.extend((np.full(len(a), step * resolution), np.full(len(a), step * resolution)))
    n = int(component.sum())
    return coo_matrix((np.concatenate(lengths), (np.concatenate(sources), np.concatenate(targets))),
                      shape=(n, n)).tocsr(), ids


def reconstruct(predecessors, source, target):
    route = [int(target)]
    while route[-1] != source:
        parent = int(predecessors[route[-1]])
        if parent < 0:
            raise ValueError('disconnected reference route')
        route.append(parent)
        if len(route) > len(predecessors):
            raise RuntimeError('cyclic predecessor chain')
    return np.asarray(route[::-1], dtype=np.int64)


def remove_edges(graph, bad_edges):
    coo = graph.tocoo()
    n = graph.shape[0]
    encoded = coo.row.astype(np.int64) * n + coo.col
    bad = np.array([u * n + v for a, b in bad_edges for u, v in ((a, b), (b, a))], dtype=np.int64)
    keep = ~np.isin(encoded, bad)
    return coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=graph.shape).tocsr()


def select_cases(forest, sequence, target_lengths=(1000., 1200.), clearance=DEFAULT_GROUND_CLEARANCE_M,
                 planning_resolution=None, seed_base=91000, max_cases=3,
                 formation_margin=19., max_geometry_repairs=32, endpoint_clearance=None):
    target_lengths = sorted(set(float(value) for value in target_lengths))
    if not target_lengths or any(not np.isfinite(value) or value <= 0 for value in target_lengths):
        raise ValueError('target lengths must be finite and positive')
    if max_cases < 2:
        raise ValueError('at least two cases are required for forward/reverse validation')
    endpoint_clearance = max(float(clearance), DEFAULT_GROUND_CLEARANCE_M) if endpoint_clearance is None else float(endpoint_clearance)
    if not np.isfinite(endpoint_clearance) or endpoint_clearance < clearance:
        raise ValueError('endpoint clearance must be finite and at least the path clearance')
    if (forest.origin_xy < 0).any():
        raise ValueError('Translate the map to nonnegative simulation coordinates during preprocessing')
    world = float(math.ceil(float(forest.upper_xy.max())))
    safe, resolution, x, y, endpoint_safe = planning_mask(forest, clearance, planning_resolution, endpoint_clearance)
    components, count = label(safe)  # Four-neighbor connectivity is identical
    # to eight-neighbor connectivity when diagonal corner cutting is forbidden.
    sizes = np.bincount(components.ravel())[1:]
    result = dict(protocol='wildscenes_recorded_grid_tasks_v1', sequence=sequence,
        world_size_m=world, cases=[], status='insufficient_connected_length', metadata=dict(
            created_utc=datetime.now(timezone.utc).isoformat(), clearance_m=float(clearance),
            endpoint_clearance_m=endpoint_clearance,
            clearance_automatically_reduced=False, planning_resolution_m=resolution,
            source_resolution_m=forest.resolution_m, source_shape_yx=list(forest.shape),
            safe_planning_cells=int(safe.sum()), connected_components=int(count),
            largest_component_cells=int(sizes.max()) if len(sizes) else 0,
            component_sizes_cells=sorted((int(v) for v in sizes), reverse=True)[:20],
            target_lengths_m=target_lengths, maximum_found_reference_length_m=0.,
            route_length_definition='8-neighbor metric grid shortest path; diagonal corner cutting forbidden',
            diameter_search='two Dijkstra sweeps per candidate component; longest verified route selected',
            reference_path_in_policy=False, forward_reverse_are_independent_maps=False,
            formation_boundary_margin_m=float(formation_margin), geometry_edges_removed=0))
    if not len(sizes):
        result['metadata']['reason'] = 'No connected cells meet the requested conservative clearance.'
        return result
    final_route = None
    final_endpoint_allowed = None
    best_length = -1.
    search_records = []
    order = np.argsort(sizes)[::-1]
    for rank, component_index in enumerate(order, 1):
        size = int(sizes[component_index])
        # A shortest route is simple and uses at most N-1 edges.  Components
        # below this metric bound cannot improve the best verified candidate.
        length_upper_bound = (size - 1) * math.sqrt(2) * resolution
        if length_upper_bound <= best_length + 1e-8:
            result['metadata']['components_skipped_by_length_bound'] = len(order) - rank + 1
            break
        component = components == int(component_index + 1)
        rows, columns = np.nonzero(component)
        positions = np.column_stack((x[columns], y[rows]))
        eligible = ((positions >= formation_margin).all(axis=1) &
                    (positions <= world - formation_margin).all(axis=1) & endpoint_safe[component])
        record = dict(component_label=int(component_index + 1), size_rank=rank,
                      cells=size, eligible_endpoints=int(eligible.sum()),
                      maximum_found_reference_length_m=0., geometry_edges_removed=0)
        search_records.append(record)
        if not eligible.any():
            record['status'] = 'no_eligible_endpoint'
            continue
        graph, _ = grid_graph(component, resolution)
        for repair in range(max_geometry_repairs + 1):
            if repair:
                _, labels = connected_components(graph, directed=False)
                valid_labels, valid_counts = np.unique(labels[eligible], return_counts=True)
                chosen = valid_labels[int(np.argmax(valid_counts))]
                candidates = eligible & (labels == chosen)
            else:
                candidates = eligible.copy()
            initial = int(np.flatnonzero(candidates)[0])
            first_distance = dijkstra(graph, directed=False, indices=initial)
            candidates &= np.isfinite(first_distance)
            source = int(np.argmax(np.where(candidates, first_distance, -np.inf)))
            distances, predecessors = dijkstra(graph, directed=False, indices=source, return_predecessors=True)
            target = int(np.argmax(np.where(candidates & np.isfinite(distances), distances, -np.inf)))
            route_ids = reconstruct(predecessors, source, target)
            route = positions[route_ids]
            bad = [(int(a), int(b)) for a, b in zip(route_ids, route_ids[1:])
                   if not forest.segment_clear(positions[a], positions[b], clearance)]
            if not bad:
                length = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
                record.update(status='geometry_verified', maximum_found_reference_length_m=length,
                              straight_distance_m=float(np.linalg.norm(route[-1] - route[0])))
                if length > best_length:
                    best_length = length
                    final_route = route
                    final_endpoint_allowed = eligible[route_ids]
                    result['metadata'].update(selected_component_label=int(component_index + 1),
                        selected_component_size_rank=rank, selected_component_cells=size)
                break
            record['geometry_edges_removed'] += len(bad)
            result['metadata']['geometry_edges_removed'] += len(bad)
            if repair == max_geometry_repairs:
                record['status'] = 'geometry_validation_failed'
                break
            graph = remove_edges(graph, bad)
    result['metadata']['component_search'] = search_records
    result['metadata']['components_examined'] = len(search_records)
    if final_route is None:
        result['metadata']['reason'] = 'No endpoint meets endpoint clearance and unchanged fleet formation bounds, or exact geometry failed.'
        return result

    cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(final_route, axis=0), axis=1))]
    result['metadata']['maximum_found_reference_length_m'] = float(cumulative[-1])
    result['metadata']['diameter_endpoint_straight_distance_m'] = float(np.linalg.norm(final_route[-1] - final_route[0]))
    result['metadata']['diameter_endpoints'] = [final_route[0].tolist(), final_route[-1].tolist()]
    if cumulative[-1] + 1e-8 < target_lengths[0]:
        result['metadata']['reason'] = ('The longest safely verified route found is shorter than the requested '
            'minimum; this is not a proof that no longer route exists elsewhere.')
        result['metadata']['unmet_target_lengths_m'] = target_lengths
        return result
    endpoint_allowed = final_endpoint_allowed
    unmet = []
    used_endpoints = set()
    for length in target_lengths:
        indices = np.flatnonzero((cumulative >= length - 1e-8) & endpoint_allowed)
        if not len(indices):
            unmet.append(length)
            continue
        end = int(indices[0])
        if end in used_endpoints:
            continue
        used_endpoints.add(end)
        selected = final_route[:end + 1]
        directions = [('forward', selected)]
        if not result['cases']:
            directions.append(('reverse', selected[::-1]))
        for direction, path in directions:
            if len(result['cases']) >= max_cases:
                break
            if not all(forest.segment_clear(a, b, clearance) for a, b in zip(path, path[1:])):
                raise RuntimeError('Reference failed its final independent swept-clearance check')
            if not all(forest.point_clear(p, endpoint_clearance) for p in (path[0], path[-1])):
                raise RuntimeError('Reference endpoint failed its independent swept-clearance check')
            index = len(result['cases'])
            case = dict(id=f'{sequence.replace("-", "_")}_{int(round(length))}m_{direction}',
                sequence=sequence, start=path[0].tolist(), goal=path[-1].tolist(),
                requested_length_m=length, reference_length_m=float(cumulative[end]),
                straight_distance_m=float(np.linalg.norm(path[-1] - path[0])),
                reference_points=path.tolist(), reference_clearance_m=float(clearance),
                endpoint_clearance_m=endpoint_clearance,
                reference_geometry_validated=True, seed=int(seed_base + index),
                scene_seed=int(seed_base + index), world_size_m=world,
                reference_path_in_policy=False, direction=direction,
                route_group=f'{sequence}_{int(round(length))}m')
            result['cases'].append(case)
    result['metadata']['unmet_target_lengths_m'] = unmet
    result['metadata']['distinct_map_count'] = 1
    result['status'] = 'ready' if not unmet else 'partial_target_coverage'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True, dest='map_path')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--target-lengths', nargs='+', type=float, default=[1000., 1200.])
    parser.add_argument('--clearance', type=float, default=DEFAULT_GROUND_CLEARANCE_M,
                        help='Requested metric safety clearance; never reduced automatically')
    parser.add_argument('--planning-resolution', type=float)
    parser.add_argument('--endpoint-clearance', type=float,
                        help='Endpoint clearance; default max(path clearance,1m). Does not clear obstacles.')
    parser.add_argument('--seed-base', type=int, default=91000)
    parser.add_argument('--max-cases', type=int, default=3)
    parser.add_argument('--formation-margin', type=float, default=19.)
    args = parser.parse_args(argv)
    forest = RasterForestMap.from_npz(args.map_path)
    result = select_cases(forest, args.sequence, args.target_lengths, args.clearance,
                          args.planning_resolution, args.seed_base, args.max_cases,
                          args.formation_margin, endpoint_clearance=args.endpoint_clearance)
    result['map_npz'] = str(args.map_path.resolve())
    result['metadata']['map_sha256'] = digest(args.map_path)
    for case in result['cases']:
        case['map_npz'] = result['map_npz']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(args.output)
    print(json.dumps(dict(status=result['status'], cases=len(result['cases']),
        maximum_found_reference_length_m=result['metadata']['maximum_found_reference_length_m'],
        connected_components=result['metadata']['connected_components'],
        largest_component_cells=result['metadata']['largest_component_cells'],
        clearance_m=result['metadata']['clearance_m'], output=str(args.output)), ensure_ascii=False))
    return 0 if result['cases'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
