"""Lightweight VTemp2 survey-point PPO on archived V9 terrain, without UAV dynamics."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import math
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parent))


def survey_source(scripts, name):
    """Locate a dependency in the reorganized source tree for provenance."""
    if name in ('ground_policy.py', 'survey_fleet_clearance.py'):
        return scripts / ('survey' if name.startswith('survey_') else 'vtemp2_3') / name
    if name.startswith('survey_'):
        return scripts / 'survey' / name
    if name.startswith(('ground_evidence', 'ground_relay')):
        return scripts / 'shared' / 'simulator' / name
    return scripts / 'vtemp2_3' / 'navigation' / name

from vtemp2_3.frozen_navigation import CHECKPOINT_SHA, runtime
from vtemp2_3.hierarchical_runtime import SLOW_FEATURES
from vtemp2_3.hierarchical_sources import atomic_json, file_sha256
from vtemp2_3.slow_policy import update
from vtemp2_3.train_slow_policy import SCHEMA, atomic_checkpoint, slow_model
from vtemp2_3.ground_policy import TASK_FEATURES, TASK_INPUT_WIDTH
from survey.survey_pretrain_env import SurveyEnv
from survey.survey_pretrain_maps import load_splits, load_survey_map
from survey.survey_pretrain_policy import policy_observation

FORMAT = 'vtemp2_light_survey_ppo_v1'
GAMMA = .995
KM_GAMMA = .9995


def weights_contract(hidden):
    return dict(format=FORMAT, input_schema=SCHEMA, hidden=hidden, ground_input_width=TASK_INPUT_WIDTH,
                candidate_features=list(SLOW_FEATURES), task_features=list(TASK_FEATURES),
                action='shared_uav_survey_target', frozen_v9_sha256=CHECKPOINT_SHA)


def load_survey_weights(path):
    """Initialize the task-aware slow network from survey weights; caller initializes the V9 runtime."""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    contract = saved['contract']
    if contract != weights_contract(contract['hidden']):
        raise ValueError('Incompatible survey checkpoint contract')
    model = slow_model(SCHEMA, contract['hidden'])
    model.load_state_dict(saved['model'], strict=True)
    return model


def choose_teacher(state):
    scores = np.asarray(state['heuristic_scores'], float)
    legal = np.asarray(state.get('mask', np.ones(scores.shape, dtype=bool)), dtype=bool)
    if legal.shape != scores.shape or not legal.any() or not np.isfinite(scores[legal]).all():
        raise ValueError('No survey candidates with finite planning scores')
    return int(np.argmax(np.where(legal, scores, -np.inf)))


def rollout(model, case, mode, args, policy_seed, *, log_decisions=False):
    geometry = load_survey_map(case)
    # All evaluated methods use identical scenarios and simulator seeds.
    km_route = args.profile == 'km' and case['terrain'] == 'synthetic'
    budget = math.ceil(3 * case['direct_distance_m'] + 120) if km_route else None
    env = SurveyEnv(**geometry.as_kwargs(), seed=case['seed'], max_steps=budget or args.max_steps,
                    max_time_s=budget, local_window=args.profile == 'km', ground_replan_interval_s=args.ground_replan_interval_s)
    generator = torch.Generator().manual_seed(policy_seed)
    transitions, decisions = [], []
    survey_requests = 0
    while not env.done:
        state = env.observe()
        if mode == 'random':
            legal = np.flatnonzero(state.get('mask', np.ones(len(state['heuristic_scores']), dtype=bool)))
            action = int(legal[int(torch.randint(len(legal), (), generator=generator))])
        elif mode == 'heuristic':
            action = choose_teacher(state)
        else:
            observation = policy_observation(state, local_window=args.profile == 'km')
            if mode == 'imitate':
                action = choose_teacher(state)
                transitions.append((observation, action))
            else:
                action, logp, value = model.act(observation, generator, deterministic=mode == 'learned')
        reward, done, info = env.step(action)
        survey_requests += int(info['survey_target'] is not None)
        if log_decisions:
            decisions.append(info)
        if mode == 'ppo':
            if transitions:
                transitions[-1]['next_value'] = value
            transitions.append(dict(features=observation['features'], ground_inputs=observation['ground_inputs'],
                                    mask=observation['mask'], present=observation['present'],
                                    action=action, logp=logp, value=value, reward=float(reward),
                                    duration=int(info['duration_s']), terminated=bool(done),
                                    truncated=False, next_value=0.))
    result = dict(env.summary(), case_id=case['case_id'], seed=case['seed'], survey_requests=survey_requests,
                  time_budget_s=budget, direct_distance_m=case['direct_distance_m'])
    if log_decisions:
        result['decisions'] = decisions
    else:
        result.pop('ground_replan_events', None)
    return transitions, result


def summary(rows):
    return dict(cases=len(rows), arrivals=sum(bool(row['goal_reached']) for row in rows),
                mean_reward=float(np.mean([row['reward'] for row in rows])),
                reward_per_request=sum(row['reward'] for row in rows)/max(1, sum(row['survey_requests'] for row in rows)),
                mean_survey_requests=float(np.mean([row['survey_requests'] for row in rows])),
                mean_fraction_completed=float(np.mean([1 - row['remaining_m']/row['direct_distance_m'] for row in rows])),
                mean_travelled_m=float(np.mean([row['travelled_m'] for row in rows])),
                mean_remaining_m=float(np.mean([row['remaining_m'] for row in rows])),
                mean_sim_time_s=float(np.mean([row['sim_time_s'] for row in rows])),
                mean_useful_new_cells=float(np.mean([row['useful_new_cells'] for row in rows])),
                mean_resolved_questions=float(np.mean([row['resolved_questions'] for row in rows])))


def _worker_init(repo):
    torch.set_num_threads(1)
    runtime(repo)


def _worker_rollout(job):
    weights, hidden, case, mode, args, detailed = job
    model = None
    if weights is not None:
        model = slow_model(SCHEMA, hidden)
        model.load_state_dict({key: torch.from_numpy(value) for key, value in weights.items()})
        model.eval()
    return rollout(model, case, mode, args, case['seed'] + args.seed, log_decisions=detailed)


def collect_cases(model, cases, mode, args, pool=None, *, detailed=False):
    if pool is None:
        return [rollout(model, row, mode, args, row['seed'] + args.seed,
                        log_decisions=detailed) for row in cases]
    weights = None
    if mode in ('ppo', 'learned'):
        weights = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    jobs = [(weights, model.hidden, row, mode, args, detailed) for row in cases]
    return list(pool.map(_worker_rollout, jobs))


def evaluate(model, cases, mode, args, pool=None):
    rows = [row for _, row in collect_cases(model, cases, mode, args, pool, detailed=True)]
    return dict(summary=summary(rows), rows=rows)


def imitate(model, optimizer, cases, args, pool=None):
    samples = [sample for episode, _ in collect_cases(model, cases, 'imitate', args, pool)
               for sample in episode]
    rng = np.random.default_rng(args.seed)
    losses = []
    for _ in range(1 if args.profile == 'km' and args.initialize_from else 3):
        order = rng.permutation(len(samples))
        for begin in range(0, len(order), 12):
            rows = [samples[i] for i in order[begin:begin+12]]
            inputs = [torch.as_tensor(np.stack([o[key] for o, _ in rows]))
                      for key in ('features', 'mask', 'present', 'ground_inputs')]
            actions = torch.tensor([action for _, action in rows])
            distribution, _ = model(*inputs)
            loss = -distribution.log_prob(actions).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
    return dict(cases=len(cases), decisions=len(samples), mean_loss=float(np.mean(losses)))


def train(model, args, groups, output, gamma, pool):
    warmup_cases, training, dev = (groups[key] for key in ('warmup', 'ppo_train', 'development'))
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003 if args.initialize_from else .001)
    warmup = imitate(model, optimizer, warmup_cases, args, pool)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    atomic_json(output/'warmup.json', warmup)
    print('warmup', warmup, flush=True)
    selected = output/'best.pt'
    best = None
    history = []
    for i in range(1, args.updates+1):
        offset = (i-1)*args.episodes_per_update
        batch = training[offset:offset+args.episodes_per_update]
        results = collect_cases(model, batch, 'ppo', args, pool)
        transitions = [items for items, _ in results]
        rows = [row for _, row in results]
        metrics = update(model, optimizer, transitions, gamma=gamma,
                         gae_lambda=.95, epochs=3, target_kl=.02, max_kl=.05, microbatch_size=12)
        entry = dict(update=i, training=summary(rows), rows=rows, ppo=metrics)
        if i in (1, (args.updates+1)//2, args.updates):
            dev_rows = [row for _, row in collect_cases(model, dev, 'learned', args, pool)]
            val = summary(dev_rows)
            entry.update(development=val, development_rows=dev_rows)
            if args.profile == 'km':
                quality = (val['arrivals'], val['mean_fraction_completed'], val['reward_per_request'])
            else:
                quality = (val['reward_per_request'], val['arrivals'])
            if best is None or quality > best:
                best = quality
                atomic_checkpoint(selected, dict(contract=weights_contract(model.hidden), model=model.state_dict(),
                                                  profile=args.profile, selected_update=i, v9_development=val))
        history.append(entry)
        atomic_json(output/'training.json', history)
        atomic_json(output/'status.json', dict(status='running', phase='training', completed_updates=i))
        print('update', i, 'decisions', metrics['transitions'], 'arrivals', entry['training']['arrivals'],
              'dev', entry.get('development'), flush=True)
    return selected


def run(args, repo, output, pool):
    archived, wild = load_splits(repo)
    if len(wild) != 18:
        raise ValueError('Expected the original 18 WildScenes routes')
    n_ppo = args.updates * args.episodes_per_update
    if args.profile == 'km':
        from survey.survey_km_maps import make_km_far_test, make_km_splits
        data_dir = args.km_data or output/'km_maps'
        groups = make_km_splits(repo, data_dir.resolve(), seed=args.seed, warmup_cases=args.warmup_cases,
                                ppo_cases=n_ppo, dev_cases=args.dev_cases, test_cases=args.test_cases,
                                workers=args.workers)
        far_test = make_km_far_test(repo, data_dir.resolve(), seed=args.seed, existing=groups, workers=args.workers)
        gamma = KM_GAMMA
    else:
        far_test = []
        count = args.warmup_cases+n_ppo+args.dev_cases
        if count > len(archived):
            raise ValueError('Insufficient disjoint V9 cases')
        samples = [archived[i] for i in np.random.default_rng(args.seed).permutation(len(archived))[:count]]
        bound = args.dev_cases+args.warmup_cases
        groups = dict(development=samples[:args.dev_cases], warmup=samples[args.dev_cases:bound],
                      ppo_train=samples[bound:], test=[])
        gamma = GAMMA
    files = [Path(__file__), SOURCE.parent/'vtemp2_3'/'train_slow_policy.py',
             *(survey_source(SOURCE.parent, name) for name in ('survey_pretrain_env.py', 'survey_pretrain_policy.py',
               'survey_pretrain_maps.py', 'ground_policy.py', 'hierarchical_runtime.py', 'navigation_evidence.py',
               'slow_policy.py'))]
    regression = wild if args.profile == 'short' or args.short_regression else []
    manifest = dict(format=FORMAT, profile=args.profile, seed=args.seed, short_max_steps=args.max_steps,
                    gamma_per_simulated_second=gamma, ground_replan_interval_s=args.ground_replan_interval_s,
                    runtime='ideal delivered survey and known-grid vehicle planner, no vehicle/drone controllers',
                    source_sha256={file.name:file_sha256(file) for file in files},
                    initial_checkpoint=None if not args.initialize_from else str(args.initialize_from.resolve()),
                    initial_sha256=None if not args.initialize_from else file_sha256(args.initialize_from),
                    checkpoint_selection=('V9 dev arrivals, then fraction of displacement completed, then survey return per request'
                                          if args.profile == 'km' else 'V9 dev reward per request, then arrivals'),
                    validation=[c['case_id'] for c in regression], far_test=[c['case_id'] for c in far_test],
                    updates=args.updates, episodes_per_update=args.episodes_per_update,
                    **{key:[c['case_id'] for c in groups[key]] for key in ('warmup','ppo_train','development','test')})
    if args.profile == 'km':
        manifest['source_sha256']['survey_km_maps.py'] = file_sha256(SOURCE/'survey_km_maps.py')
        manifest.update(km_maps=str(data_dir.resolve()), km_scenes_sha256=file_sha256(data_dir/'scenes.json'),
                        far_scenes_sha256=file_sha256(data_dir/'far-test/scenes.json'),
                        budget='ceil(3*direct_distance_m + 120) simulated seconds')
    atomic_json(output/'manifest.json', manifest)
    runtime(repo)
    started = time.monotonic()
    if args.evaluate:
        selected = args.evaluate.resolve()
    else:
        model = load_survey_weights(args.initialize_from) if args.initialize_from else slow_model(SCHEMA, args.hidden)
        selected = train(model, args, groups, output, gamma, pool)
    model = load_survey_weights(selected).eval()
    initial_model = load_survey_weights(args.initialize_from).eval() if args.initialize_from else None
    n_test = 0
    for label, cases, filename in (('V9-km', groups['test'] + far_test, 'km_test.json'),
                                   ('WildScenes short regression', regression, 'validation.json')):
        if not cases:
            continue
        validation = dict(checkpoint_sha256=file_sha256(selected), policies={})
        modes = [('learned', model, 'learned'), ('heuristic', model, 'heuristic'), ('random', model, 'random')]
        if initial_model is not None:
            modes.insert(1, ('initial', initial_model, 'learned'))
        for key, policy, mode in modes:
            atomic_json(output/'status.json', dict(status='running', phase='evaluation', split=label, policy=key))
            validation['policies'][key] = evaluate(policy, cases, mode, args, pool)
            print(label, key, validation['policies'][key]['summary'], flush=True)
            atomic_json(output/filename, validation)
        n_test += len(cases)
    atomic_json(output/'status.json', dict(status='complete', elapsed_s=time.monotonic()-started,
                                           checkpoint=str(selected), validation_cases=n_test))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repo', type=Path, default=SOURCE.parents[3])
    parser.add_argument('--profile', choices=('km', 'short'), help='Training defaults to km; evaluation follows checkpoint')
    parser.add_argument('--km-data', type=Path, help='Directory of reproducible generated kilometre tasks')
    parser.add_argument('--short-regression', action='store_true', help='Additionally run the original 18 short WildScenes routes with the km model')
    parser.add_argument('--initialize-from', type=Path, help='Transfer previous survey weights and compare on paired test cases')
    parser.add_argument('--evaluate', type=Path, help='Evaluate fixed survey weights')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--updates', type=int)
    parser.add_argument('--episodes-per-update', type=int)
    parser.add_argument('--warmup-cases', type=int)
    parser.add_argument('--dev-cases', type=int)
    parser.add_argument('--test-cases', type=int, default=8)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--max-steps', type=int, default=56, help='Decision limit for the short V9 and WildScenes tasks')
    parser.add_argument('--ground-replan-interval-s', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=64)
    args = parser.parse_args()
    if args.profile is None:
        args.profile = (torch.load(args.evaluate, map_location='cpu', weights_only=True).get('profile', 'short')
                        if args.evaluate else 'km')
    defaults = (dict(seed=20261120, updates=8, episodes_per_update=3, warmup_cases=3, dev_cases=4, workers=4)
                if args.profile == 'km' else
                dict(seed=20261119, updates=12, episodes_per_update=4, warmup_cases=12, dev_cases=6, workers=1))
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if min(args.updates, args.episodes_per_update, args.warmup_cases, args.dev_cases, args.workers,
           args.test_cases, args.max_steps, args.hidden, args.ground_replan_interval_s) < 1:
        parser.error('Episode counts, budgets, workers and network width must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error('Use an empty output directory')
    repo = args.repo.resolve()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    atomic_json(output/'status.json', dict(status='running', phase='prepare'))
    executor = (ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                                   initializer=_worker_init, initargs=(repo,)) if args.workers > 1 else nullcontext())
    try:
        with executor as pool:
            run(args, repo, output, pool)
    except BaseException as exc:
        atomic_json(output/'status.json', dict(status='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
