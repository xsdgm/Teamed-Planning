"""Fine-tune the VTemp5 waypoint actor in the online three-aircraft fleet simulator."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
from concurrent.futures import as_completed
from contextlib import redirect_stdout

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))
from vtemp4.vtemp4_policy import _heading_log_prob, update_policy
from vtemp4.train_vtemp4_formal import exclusive_run, source_digest, source_tree_digest
from vtemp4.vtemp4_body_graph import GoalResidualBodyFramePolicy
from vtemp4.vtemp4_curriculum import (
    BRIDGE_GEOMETRY, FIELD_DATA, LAKE_DATA, LAKE_GEOMETRY, _bound_case, runtime_case,
)
from vtemp4.vtemp4_parallel import episode_pool, load_rollout
from vtemp4.vtemp4_ring_contract import RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT
from vtemp2_3.frozen_navigation import CHECKPOINT_SHA, V9_RUN, runtime
from vtemp2_3.hierarchical_sources import atomic_checkpoint, atomic_json, file_sha256

FORMAT = RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT
GROUND_CONTROL = 'vtemp1_connected_anchored'
DEFAULT_ACTOR = Path('artifacts/vtemp5-km/bridge-recovery/heading.pt')
DEFAULT_WORLD = Path('artifacts/vtemp5-large/world-bridge/model.npz')


def policy_from(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['format'] != FORMAT:
        raise ValueError('Expected a VTemp5 goal-residual body-frame actor')
    actor = GoalResidualBodyFramePolicy()
    actor.load_state_dict(saved['model'], strict=True)
    if not all(p.isfinite().all() for p in actor.parameters()):
        raise ValueError('Nonfinite vehicle parameters')
    return actor, saved


def validate_warm_start(path, world_sha, fleet_identity):
    """Trace fleet PPO and short-course actor checkpoints to the evidence-trained actor."""
    manifest_formats = {'vtemp5_online_fleet': 'vtemp5_online_fleet_finetune_v1',
                        'vtemp5_circle_pretraining': 'vtemp5_circle_pretraining_v1'}
    seen = set()
    while path not in seen:
        seen.add(path)
        _, saved = policy_from(path)
        environment = saved.get('environment')
        if environment not in manifest_formats:
            if (saved.get('terrain') != 'km-bridge' or saved.get('environment') != 'vtemp5'
                    or saved.get('evidence_reference') is not True
                    or saved.get('world_model', {}).get('sha256') != world_sha):
                raise ValueError('Expected the evidence-trained VTemp5 actor and its matching world model')
            return
        folder = path.parent.parent
        manifest = folder / 'manifest.json'
        protocol = json.loads(manifest.read_text())
        if (protocol.get('format') != manifest_formats[environment]
                or protocol.get('world_sha256') != world_sha
                or any(protocol.get(key) != value for key, value in fleet_identity.items())
                or protocol.get('ground_control') != GROUND_CONTROL):
            raise ValueError(f'Incompatible online warm-start manifest: {manifest}')
        snapshot = folder / 'logs' / 'sources'
        if snapshot.is_dir() and source_tree_digest(snapshot) != protocol['sources_sha256']:
            raise ValueError(f'Warm-start source snapshot differs from its manifest: {snapshot}')
        signature = file_sha256(manifest)
        update = saved['completed_updates']
        if (not 1 <= update <= protocol['updates'] or path != folder/'checkpoints'/f'heading-{update:04d}.pt'):
            raise ValueError(f'Unexpected online warm-start checkpoint: {path}')
        trained_from = Path(protocol['actor']).resolve()
        for number in range(update, 0, -1):
            parent = trained_from if number == 1 else path.parent/f'heading-{number-1:04d}.pt'
            if (saved.get('format') != FORMAT or saved.get('environment') != environment
                    or saved.get('manifest_sha256') != signature or saved.get('completed_updates') != number
                    or saved.get('world_sha256') != world_sha or saved.get('v9_sha256') != CHECKPOINT_SHA
                    or saved.get('parent_sha256') != file_sha256(parent)):
                raise ValueError(f'Invalid online warm-start lineage at update {number} of {folder}')
            if number > 1:
                saved = torch.load(parent, map_location='cpu', weights_only=True)
        if saved['parent_sha256'] != protocol['actor_sha256']:
            raise ValueError(f'Initial actor differs from online warm-start manifest: {manifest}')
        path = trained_from
    raise ValueError('Cyclic online warm-start lineage')


def map_schedule(repo, split, count, seed):
    rng = np.random.default_rng(seed)
    groups = []
    for folder, terrain in ((FIELD_DATA, BRIDGE_GEOMETRY), (LAKE_DATA, LAKE_GEOMETRY)):
        directory = repo / folder
        scenes = json.loads((directory / 'scenes.json').read_text())[split]
        # Side names stratify the environment generator and never reach the actor.
        sides = []
        for name in ('detour_left', 'detour_right'):
            rows = [s for s in scenes if s['navigation_task'] == name]
            rng.shuffle(rows)
            if split == 'development':
                # The same held-out near-bank case can exercise the real relay
                # early in the smoke run before its full baseline and retest.
                rows.sort(key=lambda s: s['barrier_distance_m'])
            sides.append(rows)
        groups.append((directory, terrain, sides))
    plans = []
    for i in range(count):
        directory, terrain, sides = groups[i % 2]
        side_index = (i // 2 + i % 2) % 2
        rows = sides[side_index]
        row = rows[(i // 4) % len(rows)]
        case = _bound_case(row, directory, True, terrain)
        plans.append(dict(case=case, fleet_seed=int(rng.integers(1, 2**31)),
                          terrain=terrain, case_id=case['case_id']))
    return plans


def configure(args):
    repo = args.repo.resolve()
    actor_path = (repo / args.init_checkpoint).resolve()
    world = (repo / args.world_model).resolve()
    _, _, fast, fleet = runtime(repo)
    archive = repo / V9_RUN
    fleet_identity = dict(v9_sha256=CHECKPOINT_SHA, v9_sources_sha256=source_digest(archive/'runtime_sources'),
                          v9_config_sha256=file_sha256(archive/'config.json'))
    world_sha = file_sha256(world)
    validate_warm_start(actor_path, world_sha, fleet_identity)
    if (fleet['num_drones'] != 3 or fast.training or any(p.requires_grad for p in fast.parameters())
            or not all(p.isfinite().all() for p in fast.parameters())):
        raise ValueError('Expected three aircraft with finite, frozen V9 weights')
    from vtemp5.vtemp5_world_model import LocalWorldModel
    wm = LocalWorldModel.load(world)
    if wm.metadata.get('training_stage') != 'bridge':
        raise ValueError('World model is not the trained bridge checkpoint')
    # Verify the nine-feature model contract without injecting a synthetic
    # observation into a mission. Online detections are counted separately.
    neutral = np.zeros((2, 9))
    neutral[:, 1] = [1., -1.]
    prediction = wm.predict_sides(neutral)
    if (prediction.shape != (2, 3) or not np.isfinite(prediction).all()
            or not np.all((prediction[:, 2] >= 0.) & (prediction[:, 2] <= 1.))):
        raise ValueError('Invalid bridge world-model prediction shape or probability')
    count = args.updates * args.episodes_per_update
    if args.curriculum == 'circle':
        from vtemp5.vtemp5_circle_curriculum import circle_schedule
        train = circle_schedule('ppo_train', count, args.seed)
        evaluation = circle_schedule('development', args.eval_episodes, args.eval_seed)
        plans = train + evaluation
        for field in ('map_sha256', 'seed'):
            if len({p['case'][field] for p in plans}) != len(plans):
                raise ValueError(f'Repeated circle scene in the schedule: {field}')
        if len({p['fleet_seed'] for p in plans}) != len(plans):
            raise ValueError('Repeated fleet seed in circle schedule')
    else:
        train = map_schedule(repo, 'ppo_train', count, args.seed)
        evaluation = map_schedule(repo, 'development', args.eval_episodes, args.eval_seed)
    for field in ('terrain_family_id', 'map_sha256', 'seed'):
        if {p['case'][field] for p in train} & {p['case'][field] for p in evaluation}:
            raise ValueError(f'Overlapping training and development maps: {field}')
    return dict(format='vtemp5_online_fleet_finetune_v1', repo=str(repo), actor=str(actor_path),
                actor_sha256=file_sha256(actor_path), world_model=str(world), world_sha256=world_sha,
                **fleet_identity, sources_sha256=source_tree_digest(SOURCE.parent),
                **(dict(curriculum='circle', workers=args.workers) if args.curriculum == 'circle' else {}),
                train=train, evaluation=evaluation, ground_control=GROUND_CONTROL, review_s=30, lookahead_m=200.,
                arrival_radius_m=2., seed=args.seed, epochs=args.epochs, microbatch=args.microbatch_size,
                learning_rate=args.learning_rate, updates=args.updates, episodes_per_update=args.episodes_per_update,
                smoke_s=args.smoke_seconds, seconds_per_m=args.budget_seconds_per_metre)


def collect(protocol, plan, weights, digest, directory, phase, limit):
    runtime(Path(protocol['repo']))
    from vtemp5.vtemp5_fleet import VTemp5FleetEpisode
    directory.mkdir(parents=True, exist_ok=True)
    case, seed = plan['case'], plan['fleet_seed']
    if file_sha256(weights) != digest:
        raise ValueError('Vehicle checkpoint changed while sampling')
    identity = dict(phase=phase, mode='train' if phase=='train' else 'evaluate', case_id=case['case_id'],
                    geometry_sha256=case['map_sha256'], fast_seed=seed, checkpoint_sha256=digest,
                    collector_limit_s=limit)
    result = directory/'episode.json'
    if result.is_file():
        row = json.loads(result.read_text())
        if any(row.get(k) != v for k,v in identity.items()):
            raise ValueError(f'Cached episode differs: {directory}')
        if phase=='train':
            load_rollout(directory)
        return brief(row)
    model, _ = policy_from(weights)
    model.eval().requires_grad_(False)
    rollout = [] if phase in ('smoke','train') else None
    budget = math.ceil(protocol['seconds_per_m'] * case['direct_distance_m'] + 120.)
    native = runtime_case(case)
    state_path = directory/'progress.json'
    def progress(ep=None):
        now = 0 if ep is None else ep.clock_s
        atomic_json(state_path, dict(**identity, status='sampling', pid=os.getpid(), sim_s=now,
                    budget_s=budget, decisions=0 if ep is None else len(ep.heading_reviews),
                    received_tiles=0 if ep is None else ep.adapter.tiles))
    progress()
    gc.collect()
    with open(os.devnull, 'w') as sink, redirect_stdout(sink), torch.inference_mode():
        ep = VTemp5FleetEpisode(Path(protocol['repo']), native, model, fast_seed=seed, budget_s=budget,
              rollout=rollout, collector_limit_s=limit, review_interval_s=30, lookahead_m=200.,
              progress_reward_weight=10., ground_control=protocol['ground_control'],
              world_model_path=Path(protocol['world_model']))
        try:
            while not ep.finished:
                ep.step()
                if ep.clock_s % 60 == 0:
                    progress(ep)
            row = ep.fleet_summary()
            row['trajectory_xy'] = ep.trajectory
        finally:
            ep.close()
    if protocol.get('curriculum') == 'circle':
        from vtemp5.vtemp5_circle_metrics import circle_metrics
        row['circle_metrics'] = circle_metrics(case, row['trajectory_xy'], row['travelled_m'], row['goal_reached'])
    if row.get('ground_control') != protocol['ground_control']:
        raise ValueError('Expected the repaired ground planner in this rollout')
    if row['end_reason'] not in ('arrival','safety_failure','time_budget','collector_limit'):
        raise ValueError('Unknown fleet episode outcome')
    if limit is None and row['end_reason'] == 'collector_limit':
        raise ValueError('Truncated rollout in full-course training or evaluation')
    if row['goal_reached'] and row['remaining_m'] > protocol['arrival_radius_m'] + 1e-4:
        raise ValueError('Arrival exceeds the mission radius')
    if not row['heading_reviews']:
        raise ValueError('Fleet episode produced no vehicle decisions')
    world_stats = row['vtemp5_world']
    if (world_stats['unauthorized_switches']
            or (world_stats['active_reviews'] > 0 and world_stats['prediction_calls'] < 1)):
        raise ValueError('World-model inference or side-commitment contract failed')
    if phase=='smoke' and (row['adapter']['unique_received_tiles'] < 1 or row['fusion_count'] < 1):
        raise ValueError('No actual UAV terrain or graph fusion observed during fleet preflight')
    if rollout is not None:
        if not rollout or len(rollout) != len(row['heading_reviews']):
            raise ValueError('Native review and transition count mismatch')
        if not any(r.get('policy_controlled',True) for r in rollout):
            raise ValueError('No policy-controlled vehicle actions sampled')
        if not all(math.isfinite(r['value']) and math.isfinite(r['logp']) and math.isfinite(r['reward'])
                   for r in rollout):
            raise ValueError('Nonfinite fleet transition')
        if phase == 'smoke':
            sampled = [r for r in rollout if r.get('policy_controlled', True)]
            with torch.inference_mode():
                dist, values = model([r['observation'] for r in sampled])
                expected = torch.tensor([r['logp'] for r in sampled], dtype=values.dtype)
                error = float((_heading_log_prob(dist, sampled, values) - expected).abs().max())
            if not math.isfinite(error) or error > 2e-4:
                raise ValueError(f'Preflight action-frame log probability mismatch: {error}')
            row['smoke_logp_error'] = error
        if phase == 'train':
            cache = directory/'transitions.pt'
            atomic_checkpoint(cache, rollout)
            row['transitions_sha256'] = file_sha256(cache)
    row.update(identity)
    atomic_json(result,row)
    summary = brief(row)
    atomic_json(state_path,dict(status='finished', **summary))
    return summary


def brief(row):
    return {**{k:row[k] for k in ('case_id','end_reason','goal_reached','remaining_m','direct_distance_m',
                                   'travelled_m','sim_time_s','fast_seed','checkpoint_sha256','fusion_count')},
            'decisions':len(row['heading_reviews']), 'received_tiles':row['adapter']['unique_received_tiles'],
            'world':row['vtemp5_world'], **({ 'smoke_logp_error': row['smoke_logp_error'] }
                                           if 'smoke_logp_error' in row else {}),
            **({'circle_metrics': row['circle_metrics']} if 'circle_metrics' in row else {})}


def run(args):
    out = args.output.resolve()
    protocol = configure(args)
    manifest = out/'manifest.json'
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != protocol:
            raise ValueError('Use a fresh output, or --resume with the original inputs and source version')
    else:
        if args.resume or any(f.name not in ('logs',) for f in out.iterdir()):
            raise ValueError('Expected an empty run directory for a new experiment')
        atomic_json(manifest,protocol)
    signature = file_sha256(manifest)
    model, _ = policy_from(protocol['actor'])
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    checkpoints = out/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    weights,digest = Path(protocol['actor']),protocol['actor_sha256']
    history = []
    worker_protocol = {k:protocol[k] for k in ('repo', 'ground_control', 'world_model', 'seconds_per_m',
                                                'arrival_radius_m', 'curriculum') if k in protocol}
    def report(phase, **info):
        atomic_json(out/'progress.json',dict(phase=phase,pid=os.getpid(),completed_updates=len(history),**info))
        print(f'{phase} {json.dumps(info,ensure_ascii=False)}',flush=True)
    with episode_pool(args.workers) as pool:
        def batch(phase, plans, directory, limit=None):
            report(phase, directory=str(directory),episodes=len(plans),checkpoint_sha256=digest)
            jobs = {pool.submit(collect,worker_protocol,plan,weights,digest,directory/f'episode-{i:02d}',phase,limit):i
                    for i,plan in enumerate(plans)}
            summaries = [None]*len(plans)
            for job in as_completed(jobs):
                summaries[jobs[job]] = job.result()
            return summaries
        smoke = batch('smoke',protocol['evaluation'][:1],out/'preflight',protocol['smoke_s'])
        atomic_json(out/'preflight.json',dict(smoke=smoke,technical_smoke=True,manifest_sha256=signature))
        if args.preflight_only:
            report('preflight_complete')
            return
        first_episodes = None
        if (checkpoints/'heading-0001.pt').exists():
            baseline = batch('baseline',protocol['evaluation'],out/'baseline')
            atomic_json(out/'baseline.json',baseline)
        else:
            # The initial baseline and first training rollouts use the exact same
            # frozen actor. Only the former use held-out maps and deterministic actions.
            baseline_jobs = [('baseline', i, plan, out/'baseline')
                             for i, plan in enumerate(protocol['evaluation'])]
            first_plans = protocol['train'][:args.episodes_per_update]
            train_dir = out/'training'/'update-0001'
            train_jobs = [('train', i, plan, train_dir) for i, plan in enumerate(first_plans)]
            jobs = {}
            baseline = [None] * len(baseline_jobs)
            first_episodes = [None] * len(first_plans)
            def initial_status():
                report('initial_baseline_and_ppo_sampling', update=1, checkpoint_sha256=digest,
                       baseline_dir=str(out/'baseline'), training_dir=str(train_dir),
                       baseline_completed=sum(row is not None for row in baseline), baseline_total=len(baseline),
                       training_completed=sum(row is not None for row in first_episodes), training_total=len(first_plans))
            initial_status()
            # Circle training fills the first wave with independent training missions.
            jobs_in_order = (train_jobs + baseline_jobs if args.curriculum == 'circle'
                             else [baseline_jobs[0], *train_jobs, *baseline_jobs[1:]])
            for phase, i, plan, directory in jobs_in_order:
                future = pool.submit(collect,worker_protocol,plan,weights,digest,directory/f'episode-{i:02d}',phase,None)
                jobs[future] = phase, i
            for future in as_completed(jobs):
                phase, i = jobs[future]
                if phase == 'baseline':
                    baseline[i] = future.result()
                    if all(row is not None for row in baseline):
                        atomic_json(out/'baseline.json',baseline)
                else:
                    first_episodes[i] = future.result()
                initial_status()
            # Reaching the optimizer requires every initial full-course outcome.
        report('baseline_complete', arrivals=sum(r['goal_reached'] for r in baseline),
               received_tiles=sum(r['received_tiles'] for r in baseline),
               online_world_predictions=sum(r['world']['prediction_calls'] for r in baseline))
        for update in range(1,args.updates+1):
            path = checkpoints/f'heading-{update:04d}.pt'
            if path.is_file():
                saved = torch.load(path,map_location='cpu',weights_only=True)
                if (saved['manifest_sha256'] != signature or saved['parent_sha256'] != digest
                        or saved['completed_updates'] != update):
                    raise ValueError('Fine-tune checkpoint lineage is inconsistent')
                model.load_state_dict(saved['model'],strict=True)
                optimizer.load_state_dict(saved['optimizer'])
                history.append(saved['update_record'])
            else:
                start = (update-1)*args.episodes_per_update
                plans = protocol['train'][start:start+args.episodes_per_update]
                directory = out/'training'/f'update-{update:04d}'
                episodes = first_episodes if update == 1 else batch('train',plans,directory)
                rollouts = [load_rollout(directory/f'episode-{i:02d}') for i in range(len(plans))]
                report('optimizing',update=update)
                before = model.actor.weight.detach().clone()
                metrics = update_policy(model.train(),optimizer,rollouts,epochs=args.epochs,
                                        microbatch_size=args.microbatch_size)
                delta = float((model.actor.weight.detach()-before).abs().max())
                if not math.isfinite(delta) or delta <= 0:
                    raise FloatingPointError('Vehicle actor did not make a finite parameter update')
                record = dict(update=update,episodes=episodes,ppo=metrics,actor_max_weight_delta=delta)
                atomic_checkpoint(path,dict(format=FORMAT,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    environment='vtemp5_online_fleet',manifest_sha256=signature,parent_sha256=digest,
                    completed_updates=update,v9_sha256=CHECKPOINT_SHA,world_sha256=protocol['world_sha256'],
                    update_record=record))
                history.append(record)
            weights,digest = path,file_sha256(path)
            atomic_json(out/'training.json',history)
            report('committed',update=update,checkpoint=str(path),checkpoint_sha256=digest)
        final = batch('paired_evaluation',protocol['evaluation'],out/'evaluation')
        atomic_json(out/'evaluation.json',dict(baseline=baseline,trained=final,checkpoint_sha256=digest))
        report('complete',checkpoint=str(weights),arrivals=sum(e['goal_reached'] for e in final),episodes=len(final))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=SOURCE.parents[3])
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--init-checkpoint',type=Path,default=DEFAULT_ACTOR)
    parser.add_argument('--world-model',type=Path,default=DEFAULT_WORLD)
    parser.add_argument('--curriculum', choices=('bridge-lake', 'circle'), default='bridge-lake')
    parser.add_argument('--updates',type=int,default=1)
    parser.add_argument('--episodes-per-update',type=int,default=2)
    parser.add_argument('--eval-episodes',type=int,default=1)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--seed',type=int,default=12000000)
    parser.add_argument('--eval-seed',type=int,default=12100000)
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--microbatch-size',type=int,default=8)
    parser.add_argument('--learning-rate',type=float,default=3e-6)
    parser.add_argument('--smoke-seconds',type=int,default=90)
    parser.add_argument('--budget-seconds-per-metre',type=float,default=6.)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    args = parser.parse_args()
    if min(args.updates,args.episodes_per_update,args.eval_episodes,args.workers,args.epochs,args.microbatch_size) < 1:
        parser.error('Expected positive training dimensions')
    if args.smoke_seconds < 31 or not all(math.isfinite(v) and v>0 for v in
                                         [args.learning_rate,args.budget_seconds_per_metre]):
        parser.error('Expected at least 31 s for course preflight and positive finite budget and learning rate')
    with exclusive_run(args.output):
        run(args)
