"""Replay the archived 128 v9 training and 18 WildScenes validation ground-sensor audits.

Run this file with the original v9-reward training directory as its argument.
The single JSON result is printed to stdout.
"""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
from world_model.frozen_navigation import V9_RUN
from world_model.wildscenes_runtime import load_forest, world_sensor_tile


def ground_view(case, repo):
    if case['kind'] == 'wildscenes':
        forest = load_forest(case, repo)

        def view(x, y):
            data = forest.local_observation((x, y), (16, 16), .5)
            return data['occupancy_map'], data['known_mask']
    else:
        from simulator.env import EnvConfig, MultiRobotEnv
        geometry = Path(case['circles_path'])
        assert hashlib.sha256(geometry.read_bytes()).hexdigest() == case['circles_sha256']
        env = MultiRobotEnv(EnvConfig(world_size=(case['world_size'],) * 2, dt=1., perception_radius=24.,
            communication_radius=95., occupancy_map_size=12, occupancy_map_resolution=2.,
            ground_occupancy_map_size=16, ground_occupancy_map_resolution=.5))
        for x, y, radius in np.load(geometry, allow_pickle=False):
            env.add_circle_obstacle(x, y, radius)

        def view(x, y):
            cells = env._generate_occupancy_map(np.array([x, y, 0.]), robot_type='ground')
            return cells, cells >= 0
    return view


def audit_episode(run, repo, row):
    episode = json.loads((run / 'episodes' / (row['label'] + '.json')).read_text())
    view = ground_view(episode['case'], repo)
    trajectory = np.asarray(episode['rollout']['trajectory'])

    @lru_cache(maxsize=256)
    def reconstruct(x, y):
        cells, known = view(x, y)
        obs = dict(occupancy_map=cells, occupancy_known_mask=known, map_resolution=.5)
        occupied, observed = world_sensor_tile(obs)
        # An independent oracle for the aligned 8 m sensor footprint. Its
        # middle 4×4 coarse cells have exactly 4×4 native pixels apiece.
        blocks = (known & (cells > .5)).reshape(4, 4, 4, 4)
        expected_occupied = blocks.any(axis=(1, 3))
        fully_known = known.reshape(4, 4, 4, 4).all(axis=(1, 3))
        reference = np.zeros((12, 12), bool)
        support = np.zeros_like(reference)
        reference[4:8, 4:8] = expected_occupied
        support[4:8, 4:8] = expected_occupied | fully_known
        return occupied, observed, reference, support

    stats = Counter()
    with np.load(row['wm_path'], allow_pickle=False) as archive:
        ground = archive['observer_id'] == 0
        for frame, evidence in zip(archive['frame'][ground], archive['local_evidence'][ground]):
            occupied, observed, expected, support = reconstruct(*trajectory[frame, :2])
            saved = evidence[:144].reshape(12, 12).astype(bool)
            stats.update(frames=1, reference_occupied=int(expected.sum()),
                         reference_known=int(support.sum()), reference_unknown=int((~support).sum()),
                         archived_missed=int((expected & ~saved).sum()),
                         archived_false_occupied=int((saved & ~expected).sum()),
                         corrected_missed=int((expected & ~occupied.astype(bool)).sum()),
                         corrected_false_occupied=int((occupied.astype(bool) & ~expected).sum()),
                         corrected_known_mask_errors=int((observed != support).sum()))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    repo = run.parents[1]
    sys.path.insert(0, str(repo / V9_RUN / 'runtime_sources'))
    assert (Path(sys.path[0]) / 'simulator' / 'env.py').is_file()
    rows = json.loads((run / 'collection.json').read_text())
    splits = dict(train=[r for r in rows if not r['evaluation'] and r['version'] < 8],
                  validation=[r for r in rows if r['evaluation'] and r['version'] == 5])
    assert len(splits['train']) == 128 and len(splits['validation']) == 18
    result = dict(run=str(run), splits={})
    for split, selected in splits.items():
        counts = Counter()
        for row in selected:
            counts.update(audit_episode(run, repo, row))
        result['splits'][split] = dict(cases=len(selected), **counts)
        assert not any(counts[key] for key in
                       ('corrected_missed', 'corrected_false_occupied', 'corrected_known_mask_errors'))
    training = result['splits']['train']
    assert (training['reference_occupied'], training['archived_missed']) == (105321, 48113)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
