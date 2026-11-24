"""Offline task preparation for explicitly simplified WildScenes simulation.

Original occupied cells and metre scale are unchanged. Original unmeasured
space is free only under the legacy scene's declared simulation assumption.
This module prepares data and routes; it never launches training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse.csgraph import dijkstra

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))

from vtemp2_3.hierarchical_sources import atomic_json, file_sha256 as digest
from vtemp2_3.wildscenes_tasks import sample_task_manifest
from vtemp2_3.wildscenes_runtime import sensor_support_at


def write_json(path, value):
    atomic_json(path, value, ensure_ascii=True)


def restore_map(scene_path, output):
    """Invert cell-aligned rectangle export, then compare every source cell."""
    scene_path, output = Path(scene_path).resolve(), Path(output).resolve()
    doc = json.loads(scene_path.read_text())
    if (doc.get('protocol') != 'wildscenes-derived-legacy-2d-v1' or
            doc.get('simplified_scene') is not True or doc.get('coordinate_scale') != 1. or
            doc.get('unknown_space') != 'free_by_simulation_assumption' or
            doc['geometry'].get('exact_source_occupied') is not True):
        raise ValueError('Expected an explicitly simplified, exact occupied-cell legacy scene')
    shape = tuple(doc['source']['shape_yx'])
    resolution = float(doc['source']['resolution_m'])
    origin = np.asarray(doc['origin_xy'], float)
    if shape[0] != shape[1] or not np.array_equal(origin, [0., 0.]):
        raise ValueError('Expected a square original map at zero origin')
    if shape[0] * resolution != doc['world_size_m']:
        raise ValueError('Source cells and metric world dimensions disagree')
    occupied = np.zeros(shape, bool)
    rectangles = np.asarray(doc['geometry']['rectangles'], float).reshape(-1, 4)
    lo = (rectangles[:, :2] - rectangles[:, 2:] / 2 - origin) / resolution
    hi = (rectangles[:, :2] + rectangles[:, 2:] / 2 - origin) / resolution
    if not (np.allclose(lo, np.round(lo), rtol=0., atol=1e-8) and
            np.allclose(hi, np.round(hi), rtol=0., atol=1e-8)):
        raise ValueError('Legacy rectangle does not align with original source cells')
    for lower, upper in zip(np.round(lo).astype(int), np.round(hi).astype(int)):
        if (lower < 0).any() or (upper > shape[::-1]).any() or (upper <= lower).any():
            raise ValueError('Invalid original obstacle rectangle bounds')
        occupied[lower[1]:upper[1], lower[0]:upper[0]] = True
    source_path = Path(doc['source']['map_path'])
    if digest(source_path) != doc['source']['map_sha256']:
        raise ValueError('Legacy source-map digest changed')
    with np.load(source_path, allow_pickle=False) as source:
        original = source['occupied'].astype(bool)
        original_known = source['known'].astype(bool)
        if original.shape != occupied.shape or not np.array_equal(original, occupied):
            raise ValueError('Restored obstacle raster changed original occupied cells')
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'forest_map.npz'
    if path.exists():
        raise FileExistsError('Preserve previous prepared map: ' + str(path))
    provenance = dict(terrain_protocol='wildscenes_simplified_raster',
        source_unknown_space='free_by_simulation_assumption', coordinate_scale=1.,
        source_scene_path=str(scene_path), source_scene_sha256=digest(scene_path))
    np.savez_compressed(path, occupied=occupied, known=np.ones(shape, bool),
                        resolution_m=np.array(resolution), origin_xy=origin, **provenance)
    sequence = doc['source']['metadata']['sequence']
    group = 'Karawatha' if sequence.startswith('K') else 'Venman'
    metadata = dict(provenance, protocol='wildscenes-simplified-raster-v1', simplified_scene=True,
        map_sha256=digest(path), map_npz=path.name, sequence=sequence, map_group=group,
        resolution_m=resolution, shape=list(shape), world_size_m=doc['world_size_m'],
        original_source_map_path=str(source_path), original_source_map_sha256=doc['source']['map_sha256'],
        original_source_known_cells=int(original_known.sum()),
        original_source_unknown_cells=int((~original_known).sum()),
        original_occupied_cells=int(original.sum()), prepared_occupied_cells=int(occupied.sum()),
        occupied_changed_cells=0, coordinate_translation_xy=[0., 0.],
        assumptions=doc['assumptions'])
    metadata_path = output / 'map_metadata.json'
    write_json(metadata_path, metadata)
    return dict(terrain=provenance['terrain_protocol'], map_path=str(path), map_sha256=metadata['map_sha256'],
        metadata_path=str(metadata_path), metadata_sha256=digest(metadata_path),
        source_scene_path=str(scene_path), source_scene_sha256=provenance['source_scene_sha256'],
        world_size=doc['world_size_m'], circles=[], resolution_m=resolution,
        required_clearance_m=1.5, source_unknown_space=provenance['source_unknown_space'],
        coordinate_scale=1., map_group=group, sequence=sequence, kind='wildscenes_simplified')


def prepare_routes(contract, forest, output):
    """Coarse planning only proposes routes; all segments use original geometry."""
    from shared.simulator.select_wildscenes_cases import planning_mask, grid_graph, reconstruct, remove_edges
    # A wider offline proposal margin avoids gratuitously grazing obstacles;
    # the unchanged required physical and sensor clearance remains 1.5m.
    safe, resolution, x, y = planning_mask(forest, 3., planning_resolution=4.)
    graph, ids = grid_graph(safe, resolution)
    yy, xx = np.nonzero(safe)
    points = np.column_stack((x[xx], y[yy]))
    world = contract['world_size']
    corner_xy = np.array([[30., 30.], [world - 30., world - 30.],
                          [30., world - 30.], [world - 30., 30.]])
    endpoints = [int(np.square(points - corner).sum(1).argmin()) for corner in corner_xy]
    result = []
    for number, (source, target) in enumerate(((endpoints[0], endpoints[1]), (endpoints[2], endpoints[3]))):
        for repair in range(32):
            distances, predecessors = dijkstra(graph, indices=source, return_predecessors=True)
            if not np.isfinite(distances[target]):
                raise ValueError('Simplified map lacks a connected route between selected endpoints')
            route_ids = reconstruct(predecessors, source, target)
            route = points[route_ids]
            bad = [(int(route_ids[i]), int(route_ids[i + 1])) for i in range(len(route) - 1)
                   if not forest.segment_clear(route[i], route[i + 1], 1.5)]
            if not bad:
                break
            graph = remove_edges(graph, bad)
        else:
            raise ValueError('Exact original-map segment verification exhausted geometry repairs')
        direct = float(np.linalg.norm(route[-1] - route[0]))
        if direct < 1000.:
            raise ValueError('Simplified route does not span actual kilometre endpoints')
        result.append(dict(route_id=f"{contract['sequence']}-diagonal-{number}", map_contract=contract,
            reference_points=route.tolist(), all_segments_clear=True, required_clearance_m=1.5,
            planning_grid_resolution_m=resolution, planning_proposal_clearance_m=3.,
            direct_distance_m=direct,
            reference_distance_m=float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
            geometry_repairs=repair))
    output = Path(output)
    write_json(output, dict(terrain='wildscenes_simplified_raster',
        source_unknown_space='free_by_simulation_assumption', coordinate_scale=1.,
        required_clearance_m=1.5, map_contract=contract, routes=result,
        reference_route_in_policy=False))
    for route in result:
        route.update(audit_path=str(output.resolve()), audit_sha256=digest(output))
    return result


def _slice_route(points, start_arc, goal_arc):
    points = np.asarray(points, float)
    cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]

    def at(arc):
        index = min(int(np.searchsorted(cumulative, arc, side='right')) - 1, len(points) - 2)
        t = (arc - cumulative[index]) / (cumulative[index + 1] - cumulative[index])
        return points[index] + t * (points[index + 1] - points[index])

    interior = points[(cumulative > start_arc) & (cumulative < goal_arc)]
    return np.vstack((at(start_arc), interior, at(goal_arc)))


def verify_tasks(manifest, audit, routes, forests):
    by_id = {route['route_id']: route for route in routes}
    verified = []
    for segment in audit['segments']:
        case = manifest[segment['split']][segment['index']]
        route = by_id[segment['route_id']]
        points = np.asarray(route['reference_points'])
        if segment['reversed_route']:
            points = points[::-1]
        truncated = _slice_route(points, *segment['arc_interval_m'])
        if not (np.allclose(truncated[0], case['start'], rtol=0., atol=1e-7) and
                np.allclose(truncated[-1], case['goal'], rtol=0., atol=1e-7)):
            raise ValueError('Sampled task endpoints differ from its audit route slice')
        forest = forests[case['map_sha256']]
        if not all(forest.segment_clear(a, b, 1.5) for a, b in zip(truncated, truncated[1:])):
            raise ValueError('Sampled task failed original-map exact continuous segment clearance')
        verified.append(dict(case_id=case['case_id'], exact_segments=len(truncated) - 1,
                             all_segments_clear=True, required_clearance_m=1.5))
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=SOURCE.parents[3])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    from shared.simulator.wildscenes_map import RasterForestMap
    contracts = []
    forests = {}
    for sequence in ('K-01-r025', 'V-01'):
        contract = restore_map(repo / 'artifacts/wildscenes-legacy2d-20261106/scenes' / (sequence + '.json'),
                               output / 'maps' / sequence)
        contracts.append(contract)
        forests[contract['map_sha256']] = RasterForestMap.from_npz(contract['map_path'])
        print(json.dumps(dict(stage='map_ready', map_path=contract['map_path'], occupied_changed_cells=0)), flush=True)
    smoke = dict(train=[], validation=[])
    for split, contract in zip(('train', 'validation'), contracts):
        start = np.array([25., 25.]); goal = np.array([35.0000001, 25.])
        forest = forests[contract['map_sha256']]
        if not forest.segment_clear(start, goal, 1.5) or not all(
                sensor_support_at(forest, point)['supported'] for point in (start, goal)):
            raise ValueError('Short smoke endpoints fail original geometry or local sensor support')
        distance = float(np.linalg.norm(goal - start))
        smoke[split].append(dict(contract, start=start.tolist(), goal=goal.tolist(),
            direct_distance_m=distance, reference_distance_m=distance,
            seed=261117500 + int(split == 'validation'), case_id='simplified-smoke-' + split))
    write_json(output / 'smoke-scene-manifest.json', smoke)
    print(json.dumps(dict(stage='smoke_manifest_ready', path=str(output / 'smoke-scene-manifest.json'))), flush=True)
    route_groups = []
    for contract in contracts:
        route_groups.append(prepare_routes(contract, forests[contract['map_sha256']],
            output / (contract['sequence'] + '-route-audit.json')))
        print(json.dumps(dict(stage='routes_ready', sequence=contract['sequence'],
            routes=[dict(route_id=r['route_id'], direct_distance_m=r['direct_distance_m'],
                         reference_distance_m=r['reference_distance_m']) for r in route_groups[-1]])), flush=True)
    from vtemp2_3.train_hierarchical import curriculum_ranges
    config = dict(warmup_updates=8, fast_updates=8, slow_updates=64, joint_updates=32,
        warmup_distances_m=[100., 200.], mission_distances_m=[1000., 1000.],
        episodes_per_update=4, eval_scenes=9)
    ranges = curriculum_ranges(config)

    def supported(case):
        forest = forests[case['map_sha256']]
        return all(sensor_support_at(forest, case[key])['supported'] for key in ('start', 'goal'))

    manifest, audit = sample_task_manifest(route_groups[0], ranges, 4, [1000., 1000.], 9, 261117600,
        validation_routes=route_groups[1], case_acceptance=supported, return_audit=True)
    audit.update(training_config=config, map_contracts=contracts,
        per_task_geometry=verify_tasks(manifest, audit, route_groups[0] + route_groups[1], forests),
        occupied_changed_cells=0, original_source_unknown_is_simulation_free=True)
    write_json(output / 'sampling-audit.json', audit)
    write_json(output / 'scene-manifest.json', manifest)
    print(json.dumps(dict(stage='complete', training_tasks=len(manifest['train']),
        validation_tasks=len(manifest['validation']), min_validation_distance_m=min(c['direct_distance_m'] for c in manifest['validation']),
        manifest=str(output / 'scene-manifest.json'))), flush=True)


if __name__ == '__main__':
    main()
