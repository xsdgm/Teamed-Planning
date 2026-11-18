"""Continue fixed navigation tasks in larger, auditable collection batches.

No task is generated, copied or altered. Only the unfinished task suffix is
regrouped, without crossing a phase or an existing evaluation boundary.
Checkpoint tensor/RNG/optimizer transfer and filesystem verification belong to
the trainer; this module only handles deterministic JSON plans and task counts.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .hierarchical_sources import file_sha256 as _file_sha256


SCHEMA = 'vtemp2_fixed_task_rebatch_v1'
PHASES = ('warmup', 'fast', 'slow', 'joint')


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _range(value):
    if not isinstance(value, list) or len(value) != 2 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError('Update distance_range_m must contain two finite numbers')
    low, high = map(float, value)
    if not 0 < low <= high:
        raise ValueError('Update distance range must satisfy 0 < MIN_M <= MAX_M')
    return [low, high]


def _legacy_schedule(config):
    phases = [phase for phase in PHASES
              for _ in range(_integer(config[phase + '_updates'], phase + '_updates'))]
    short, final = _range(config['warmup_distances_m']), _range(config['mission_distances_m'])
    ranges, slow_index = [], 0
    for phase in phases:
        if phase in ('warmup', 'fast'):
            bounds = short
        elif phase == 'slow':
            fraction = (slow_index + 1) / max(config['slow_updates'], 1)
            bounds = [a + fraction * (b - a) for a, b in zip(short, final)]
            slow_index += 1
        else:
            bounds = final
        ranges.append(list(bounds))
    return phases, ranges


def validate_update_plan(config):
    """Validate explicit variable batches and optional original-update proof."""
    plan = config.get('update_plan')
    if not isinstance(plan, list) or not plan:
        raise ValueError('update_plan must be a nonempty list')
    maximum = _integer(config['episodes_per_update'], 'episodes_per_update', 1)
    if maximum > 16:
        raise ValueError('Rebatched updates cannot exceed 16 episodes')
    normalized, source_indices, previous_phase = [], [], -1
    counts = dict.fromkeys(PHASES, 0)
    for row in plan:
        if not isinstance(row, dict) or set(row) != {
                'phase', 'episodes', 'distance_range_m', 'source_update_indices', 'evaluate_after'}:
            raise ValueError('Each update_plan row must declare phase, episodes, range, sources and evaluate_after')
        phase = row['phase']
        if phase not in PHASES or PHASES.index(phase) < previous_phase:
            raise ValueError('update_plan must preserve warmup/fast/slow/joint phase order')
        previous_phase = PHASES.index(phase)
        counts[phase] += 1
        episodes = _integer(row['episodes'], 'update episodes', 1)
        if episodes > maximum:
            raise ValueError('Update episode count exceeds configured batch capacity')
        indices = row['source_update_indices']
        if not isinstance(indices, list) or not indices:
            raise ValueError('Every rebatched update needs original source update indices')
        for index in indices:
            _integer(index, 'source update index')
        if any(b != a + 1 for a, b in zip(indices, indices[1:])):
            raise ValueError('Source update indices must retain contiguous original order')
        source_indices.extend(indices)
        if type(row['evaluate_after']) is not bool:
            raise ValueError('evaluate_after must be an explicit boolean')
        normalized.append(dict(phase=phase, episodes=episodes, distance_range_m=_range(row['distance_range_m']),
                               source_update_indices=list(indices), evaluate_after=row['evaluate_after']))
    if any(b != a + 1 for a, b in zip(source_indices, source_indices[1:])):
        raise ValueError('Update plan duplicates, skips or reorders source updates')
    for phase in PHASES:
        if _integer(config[phase + '_updates'], phase + '_updates') != counts[phase]:
            raise ValueError('Phase update counts differ from the explicit update_plan')
    for i, row in enumerate(normalized):
        if (i == len(normalized) - 1 or normalized[i + 1]['phase'] != row['phase']) and not row['evaluate_after']:
            raise ValueError('Phase boundaries must retain their validation event')
    source = config.get('continuation', {}).get('source_updates')
    if source is not None:
        if not isinstance(source, list) or [r.get('source_update_index') for r in source] != source_indices:
            raise ValueError('Continuation source-update proof differs from update_plan coverage')
        by_index = {row['source_update_index']: row for row in source}
        for row in normalized:
            original = [by_index[index] for index in row['source_update_indices']]
            if any(item['phase'] != row['phase'] for item in original):
                raise ValueError('A rebatched update crosses an original phase boundary')
            if any(item['evaluate_after'] for item in original[:-1]):
                raise ValueError('A rebatched update crosses an original evaluation boundary')
            union = [min(item['distance_range_m'][0] for item in original),
                     max(item['distance_range_m'][1] for item in original)]
            if (sum(item['episodes'] for item in original) != row['episodes'] or
                    union != row['distance_range_m'] or original[-1]['evaluate_after'] != row['evaluate_after']):
                raise ValueError('Rebatched count, curriculum union or evaluation flag differs from its source proof')
    return normalized


def update_episode_counts(config):
    """Per-update collection sizes for both legacy and variable-batch runs."""
    if config.get('update_plan') is not None:
        return [row['episodes'] for row in validate_update_plan(config)]
    phases, _ = _legacy_schedule(config)
    return [_integer(config['episodes_per_update'], 'episodes_per_update', 1)] * len(phases)


def _source_rows(config):
    if config.get('update_plan') is not None:
        validate_update_plan(config)
        source = config.get('continuation', {}).get('source_updates')
        if source is None:
            raise ValueError('Rebatching a variable plan requires original source-update proof')
        return deepcopy(source)
    phases, ranges = _legacy_schedule(config)
    episodes = _integer(config['episodes_per_update'], 'episodes_per_update', 1)
    every = _integer(config['eval_every'], 'eval_every', 1)
    return [dict(source_update_index=i, phase=phase, episodes=episodes, distance_range_m=ranges[i],
                 source_training_start=i * episodes,
                 evaluate_after=(i + 1) % every == 0 or i + 1 == len(phases) or phases[i + 1] != phase)
            for i, phase in enumerate(phases)]


def _build(saved, max_episodes, parent_binding, geometry_reference):
    _integer(max_episodes, 'max_episodes', 1)
    if max_episodes > 16:
        raise ValueError('The requested continuation batch limit is 16 episodes')
    original = saved['config']
    manifest = original.get('scene_manifest')
    if not isinstance(manifest, dict) or set(manifest) != {'train', 'validation'}:
        raise ValueError('Fixed-task continuation requires embedded train/validation cases')
    completed = _integer(saved['completed_updates'], 'completed_updates')
    sizes = update_episode_counts(original)
    if completed >= len(sizes):
        raise ValueError('Checkpoint has no unfinished training updates')
    if len(manifest['train']) != sum(sizes) or len(manifest['validation']) != original['eval_scenes']:
        raise ValueError('Parent scene counts do not match its declared update schedule')
    seeds = [case.get('seed') for rows in manifest.values() for case in rows]
    if any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Parent task seeds must be unique nonnegative integers')
    completed_episodes = sum(sizes[:completed])
    all_sources = _source_rows(original)
    source_cut = 0
    covered = 0
    while covered < completed_episodes:
        covered += all_sources[source_cut]['episodes']
        source_cut += 1
    if covered != completed_episodes:
        raise ValueError('Parent checkpoint ended inside an original task batch')
    sources = all_sources[source_cut:]
    if any(row['episodes'] > max_episodes for row in sources):
        raise ValueError('Cannot split an original source batch into a smaller continuation batch')
    plan, group = [], []

    def flush():
        if not group:
            return
        plan.append(dict(phase=group[0]['phase'], episodes=sum(row['episodes'] for row in group),
            distance_range_m=[min(row['distance_range_m'][0] for row in group),
                              max(row['distance_range_m'][1] for row in group)],
            source_update_indices=[row['source_update_index'] for row in group],
            evaluate_after=group[-1]['evaluate_after']))
        group.clear()

    for row in sources:
        if group and (group[-1]['phase'] != row['phase'] or
                      sum(item['episodes'] for item in group) + row['episodes'] > max_episodes):
            flush()
        group.append(row)
        if row['evaluate_after']:
            flush()
    flush()
    prior = original.get('continuation', {})
    consumed_total = prior.get('completed_training_episodes_total', 0) + completed_episodes
    initial_total = prior.get('original_training_tasks', len(manifest['train']))
    config = deepcopy(original)
    config.update(episodes_per_update=max_episodes, workers=max_episodes, update_plan=plan,
        policy_version_offset=original.get('policy_version_offset', original.get('version_offset', 0)) + completed,
        scene_manifest=dict(train=deepcopy(manifest['train'][completed_episodes:]),
                            validation=deepcopy(manifest['validation'])))
    for phase in PHASES:
        config[phase + '_updates'] = sum(row['phase'] == phase for row in plan)
    config['continuation'] = dict(schema=SCHEMA, parent_completed_updates=completed,
        original_training_tasks=initial_total, completed_training_episodes_total=consumed_total,
        source_updates=sources, geometry_proof_reference=deepcopy(geometry_reference),
        original_scene_manifest_sha256=prior.get('original_scene_manifest_sha256', json_sha256(manifest)))
    validate_update_plan(config)
    if sum(row['episodes'] for row in plan) + consumed_total != initial_total:
        raise ValueError('Rebatch plan changed the original total training-task budget')
    mapping = []
    cursor = 0
    for new_update, row in enumerate(plan):
        for within in range(row['episodes']):
            case = config['scene_manifest']['train'][cursor]
            mapping.append(dict(new_update_index=new_update, new_episode_index=cursor,
                index_within_update=within, parent_episode_index=completed_episodes + cursor,
                original_episode_index=consumed_total + cursor, seed=case['seed'],
                case_id=case.get('case_id'), case_sha256=json_sha256(case),
                map_sha256=case.get('map_sha256'), map_group=case.get('map_group'),
                geometry_proof_case_id=case.get('case_id')))
            cursor += 1
    old_phases = ([row['phase'] for row in validate_update_plan(original)]
                  if original.get('update_plan') is not None else _legacy_schedule(original)[0])
    audit = dict(original_training_tasks=initial_total, parent_training_tasks=len(manifest['train']),
        parent_completed_updates=completed, newly_completed_training_tasks=completed_episodes,
        completed_training_tasks_total=consumed_total, remaining_training_tasks=cursor,
        validation_tasks=len(manifest['validation']), validation_unchanged=True,
        parent_scene_manifest_sha256=json_sha256(manifest),
        remaining_scene_manifest_sha256=json_sha256(config['scene_manifest']),
        validation_sha256=json_sha256(manifest['validation']),
        original_remaining_evaluation_events=sum(row['evaluate_after'] for row in sources),
        continuation_evaluation_events=sum(row['evaluate_after'] for row in plan),
        unchanged_task_order=True, unchanged_case_values=True, resampled_tasks=0,
        duplicated_tasks=0, geometry_proof_reference=deepcopy(geometry_reference),
        per_phase={phase: dict(old_remaining_batches=old_phases[completed:].count(phase),
            new_batches=sum(row['phase'] == phase for row in plan),
            training_tasks=sum(row['episodes'] for row in plan if row['phase'] == phase)) for phase in PHASES},
        tasks=mapping,
        limitation='Larger batches change optimizer transaction count and learning dynamics; original task identities, geometry, order and validation-event boundaries are preserved.')
    return dict(schema=SCHEMA, parent=deepcopy(parent_binding), config=config, audit=audit)


def create_continuation_plan(saved, *, max_episodes=16, parent_checkpoint=None, geometry_audit=None):
    """Construct a deterministic plan, optionally binding external proof files."""
    parent = dict(completed_updates=saved['completed_updates'], config_sha256=json_sha256(saved['config']),
        source_sha256=deepcopy(saved['manifest']['source_sha256']), checkpoint_path=None, checkpoint_sha256=None)
    if parent_checkpoint is not None:
        path = Path(parent_checkpoint).resolve(strict=True)
        parent.update(checkpoint_path=str(path), checkpoint_sha256=_file_sha256(path))
    geometry = None
    if geometry_audit is not None:
        path = Path(geometry_audit).resolve(strict=True)
        data = json.loads(path.read_text(encoding='utf-8'))
        rows = data.get('per_task_geometry', [])
        by_id = {row.get('case_id'): row for row in rows}
        for cases in saved['config']['scene_manifest'].values():
            for case in cases:
                proof = by_id.get(case.get('case_id'))
                if proof is None or proof.get('all_segments_clear') is not True or proof.get('required_clearance_m') != 1.5:
                    raise ValueError('Geometry audit does not bind every original task to its unchanged 1.5m proof')
        geometry = dict(path=str(path), sha256=_file_sha256(path), task_id_field='case_id',
                        proof_list_field='per_task_geometry', task_proofs=len(rows))
    elif saved['config'].get('continuation', {}).get('geometry_proof_reference'):
        geometry = deepcopy(saved['config']['continuation']['geometry_proof_reference'])
    return _build(saved, max_episodes, parent, geometry)


def validate_continuation_plan(plan, saved):
    """Pure semantic validation; return a deep copy of the authorized config.

    The trainer separately verifies the bound checkpoint/source/proof files.
    No arbitrary optimizer, task, stage or curriculum changes can enter through
    edited JSON: the entire proposed plan is reconstructed and compared.
    """
    if not isinstance(plan, dict) or set(plan) != {'schema', 'parent', 'config', 'audit'} or plan['schema'] != SCHEMA:
        raise ValueError('Unsupported continuation-plan schema')
    parent = plan['parent']
    if (parent.get('completed_updates') != saved['completed_updates'] or
            parent.get('config_sha256') != json_sha256(saved['config']) or
            parent.get('source_sha256') != saved['manifest']['source_sha256']):
        raise ValueError('Continuation plan does not bind to this parent checkpoint config/source')
    geometry = plan['audit'].get('geometry_proof_reference')
    expected = _build(saved, plan['config']['episodes_per_update'], parent, geometry)
    if json_sha256(plan) != json_sha256(expected):
        raise ValueError('Continuation plan altered task identity/order, curriculum, evaluation boundaries or unauthorized config')
    return deepcopy(expected['config'])
