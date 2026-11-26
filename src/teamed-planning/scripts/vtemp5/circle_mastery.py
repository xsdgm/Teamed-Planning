"""Decide when the six held-out full-course circle missions support adding bridges."""
import hashlib
import json
import math
from pathlib import Path


def _read(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def _distance(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def assess(stage1: Path, *, required_successes=4):
    """Return the gate decision. Missing or inconsistent final evidence raises ValueError."""
    if type(required_successes) is not int or not 1 <= required_successes <= 6:
        raise ValueError('Expected a required success count between 1 and 6')
    folder = Path(stage1).resolve()
    try:
        progress = _read(folder / 'progress.json')
        manifest = _read(folder / 'manifest.json')
        if (progress['phase'] != 'complete' or manifest['curriculum'] != 'circle'
                or progress['completed_updates'] != manifest['updates']):
            raise ValueError('Expected a completed circle training run')
        checkpoint = folder / 'checkpoints' / f"heading-{manifest['updates']:04d}.pt"
        if Path(progress['checkpoint']).resolve() != checkpoint:
            raise ValueError('Progress does not identify the final circle checkpoint')
        sha = hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                sha.update(chunk)
        digest = sha.hexdigest()
        evaluation = _read(folder / 'evaluation.json')
        if evaluation['checkpoint_sha256'] != digest:
            raise ValueError('The evaluated checkpoint differs from the final checkpoint')
        plans, summaries = manifest['evaluation'], evaluation['trained']
        if not (len(plans) == len(summaries) == progress['episodes'] == 6):
            raise ValueError('Expected six final paired evaluation episodes')
        maps = {p['case']['map_sha256'] for p in plans}
        if (len(maps) != 6 or len({p['case_id'] for p in plans}) != 6
                or maps.intersection(p['case']['map_sha256'] for p in manifest['train'])):
            raise ValueError('Expected six distinct held-out circle maps')
        radius = manifest['arrival_radius_m']
        if not _distance(radius) or radius <= 0:
            raise ValueError('Invalid arrival radius')

        cases = []
        for index, (plan, summary) in enumerate(zip(plans, summaries)):
            row = _read(folder / 'evaluation' / f'episode-{index:02d}' / 'episode.json')
            case_id = plan['case_id']
            identity = dict(case_id=case_id, fast_seed=plan['fleet_seed'], checkpoint_sha256=digest)
            if (plan['terrain'] != 'large_circle' or plan['case']['case_id'] != case_id
                    or row['phase'] != 'paired_evaluation' or row['mode'] != 'evaluate'
                    or row['collector_limit_s'] is not None or row['geometry_sha256'] != plan['case']['map_sha256']
                    or any(row[k] != v or summary[k] != v for k, v in identity.items())):
                raise ValueError(f'Unexpected final evaluation identity for {case_id}')
            reason, arrived = row['end_reason'], row['goal_reached']
            far_side, remaining = row['circle_metrics']['reached_far_side'], row['remaining_m']
            collisions = row['collisions']
            if (reason not in ('arrival', 'time_budget', 'safety_failure')
                    or type(arrived) is not bool or type(far_side) is not bool or not _distance(remaining)
                    or type(collisions) is not int or collisions < 0):
                raise ValueError(f'Invalid full-course outcome for {case_id}')
            if (summary['end_reason'] != reason or summary['goal_reached'] is not arrived
                    or summary['circle_metrics']['reached_far_side'] is not far_side
                    or summary['remaining_m'] != remaining):
                raise ValueError(f'Final evaluation summary disagrees with the episode for {case_id}')
            success = far_side and arrived and reason == 'arrival' and remaining <= radius
            cases.append(dict(case_id=case_id, reached_far_side=far_side, goal_reached=arrived,
                              remaining_m=remaining, end_reason=reason, collisions=collisions, success=success))
    except (OSError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(f'Incomplete circle evaluation at {folder}: {exc}') from exc
    successes = sum(c['success'] for c in cases)
    collisions = sum(c['collisions'] for c in cases)
    return dict(passed=successes >= required_successes and collisions == 0, successful_cases=successes,
                collision_count=collisions, total_cases=len(cases), checkpoint=str(checkpoint),
                checkpoint_sha256=digest, per_case=cases)
