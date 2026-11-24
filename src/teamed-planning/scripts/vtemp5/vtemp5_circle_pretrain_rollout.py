"""Short native VTemp5 rollouts for learning when to maintain and end a detour."""
import hashlib
import math
import os
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np


def pretrain_plans(split, count, seed):
    from vtemp5.vtemp5_circle_curriculum import _side_distances, circle_schedule

    plans = circle_schedule(split, count, seed)
    for i, plan in enumerate(plans):
        case = plan['case']
        (center, radius), = case['circles']
        center, goal, old_start = map(np.asarray, (center, case['goal'], case['start']))
        rng = np.random.default_rng([case['seed'], 1831])
        forward = (goal - old_start) / np.linalg.norm(goal - old_start)
        left = np.array([-forward[1], forward[0]])
        side = rng.choice([-1., 1.])
        kind = ('approach', 'flank', 'exit_approach')[i % 3]
        if kind == 'approach':
            along = -(radius + rng.uniform(18., 35.))
            across = rng.uniform(-.30, .30) * radius
        elif kind == 'flank':
            along = -rng.uniform(.68, .85) * radius
            across = side * (rng.uniform(.78, .88) * radius + 5.)
        else:
            # Start beside the obstacle, still upstream of the downstream tangent.
            # Command, navigation and radio history accumulate during real motion.
            along = rng.uniform(-.12, .20) * radius
            across = side * (radius + rng.uniform(6., 18.))
        start = center + along * forward + across * left
        if (np.linalg.norm(start - center) <= radius + 2. or
                np.any(start < 3.) or np.any(start >= case['world_size'] - 3.)):
            raise ValueError('Invalid circular training start')
        distance = np.linalg.norm(goal - start)
        axis = (goal - start) / distance
        station = (center - start) @ axis
        normal = np.array([-axis[1], axis[0]])
        offset = (center - start) @ normal
        safe_radius = radius + case['required_clearance_m']
        hits = 0. < station < distance and abs(offset) < safe_radius
        reference = min(_side_distances(distance, station, offset, safe_radius).values()) if hits else distance
        case_id = f'vtemp5-rejoin-{split}-{kind}-{case["seed"]}'
        geometry = np.array([case['world_size'], *start, *goal, *center, radius], dtype='<f8')
        case.pop('barrier_distance_m')
        case.pop('side_reference_distance_m')
        case.update(case_id=case_id, terrain_family_id=case_id, start=start.tolist(),
                    direct_distance_m=float(distance), reference_distance_m=float(reference),
                    map_sha256=hashlib.sha256(geometry.tobytes()).hexdigest())
        plan.update(case_id=case_id, start_kind=kind)
    return plans


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _legal_headings(arcs, proposed):
    """Include the desired continuous angle, arc midpoints and a local search fan."""
    proposed = np.r_[proposed, np.arange(-np.pi, np.pi, np.deg2rad(5.)), arcs.mean(axis=1)]
    angles = np.unique(_wrap(proposed))
    inside = (arcs[:, 0, None] <= angles) & (angles <= arcs[:, 1, None])
    return angles[inside.any(axis=0)]


def _ray_evidence(cells, position, yaw, angles):
    """Footprint-cleared measured prefix and first observed block per direction."""
    from survey.survey_fleet_observation import _clearance

    # FleetSurveyAdapter's 1 m receiver raster uses occupied=1, unknown=0, free=-1.
    reach = 108
    lo = np.maximum(0, np.floor(position).astype(int) - reach - 3)
    hi = np.minimum(cells.shape[::-1], np.floor(position).astype(int) + reach + 4)
    measured = _clearance(cells[lo[1]:hi[1], lo[0]:hi[0]])
    distances = np.arange(0., reach + .5, .5)
    dirs = np.column_stack((np.cos(yaw + angles), np.sin(yaw + angles)))
    ij = np.floor(position + distances[None, :, None] * dirs[:, None, :]).astype(int) - lo
    valid = ((ij >= 0) & (ij < np.asarray(measured.shape[::-1]))).all(axis=2)
    bounded = np.clip(ij, 0, np.asarray(measured.shape[::-1]) - 1)
    labels = np.where(valid, measured[bounded[..., 1], bounded[..., 0]], 1)
    # Half-metre samples cover the swept path after 1.5 m footprint dilation.
    free_prefix = np.where(labels != -1, distances, reach + .5).min(axis=1)
    first_block = np.where(labels == 1, distances, reach + .5).min(axis=1)
    return free_prefix, first_block


def _route_bearings(state, tracker, car, yaw):
    bearings = []
    locked = state['locked_side']
    for side, route in state['routes'].items():
        if state['side_status'][side] == 'blocked':
            continue
        if locked and state['side_status'][locked] != 'blocked' and side != locked:
            continue
        entry, exit_xy, join = route
        along = (car - entry) @ tracker.forward
        if along < 7. and abs((car - entry) @ tracker.left) > 4.:
            target = entry
        elif (car - exit_xy) @ tracker.forward < -1.:
            target = exit_xy
        else:
            target = join
            if (join - car) @ tracker.forward < 0.:
                # An old merge point behind the car is no longer a steering
                # target, even when there is too little evidence to teach rejoin.
                continue
        vector = target - car
        if np.linalg.norm(vector) > 1.:
            bearings.append(_wrap(math.atan2(vector[1], vector[0]) - yaw))
    return bearings


def _past_shared_exit(state, tracker, car, cells):
    """Both proposals have finished their distinct legs, past the measured foreground."""
    if not state['locked_side'] or len(state['routes']) != 2:
        return False
    paths = list(state['routes'].values())
    if np.linalg.norm(paths[0][2] - paths[1][2]) > math.sqrt(2.) * tracker.res:
        return False
    for entry, exit_xy, _ in paths:
        if ((car - entry) @ tracker.forward < 7. or
                (car - exit_xy) @ tracker.forward < 0.):
            return False
    patch = tracker._component(cells, tracker.anchor, 220.)
    return bool(len(patch) and (car - tracker.anchor) @ tracker.forward >
                max((patch - tracker.anchor) @ tracker.forward) + 2. * tracker.res)


def _teacher(ep, obs, had_active):
    arcs = np.asarray(obs['angle_arcs'], dtype=float)
    if not len(arcs) or ep.adapter.tiles == 0:
        return None
    if ep.adapter.resolution != 1.:
        raise ValueError('Expected metre-scale native receiver cells')
    car = ep.position()
    yaw = float(ep.adapter.own['orientation'])
    goal = math.atan2(obs['goal'][0], obs['goal'][1])
    state = ep.obstacle_state
    route = _route_bearings(state, ep.tracker, car, yaw) if state['active'] else []
    angles = _legal_headings(arcs, [goal, *route])
    if not len(angles):
        return None
    prefix, block = _ray_evidence(ep.adapter.cells, car, yaw, angles)
    merged = state['active'] and _past_shared_exit(state, ep.tracker, car, ep.adapter.cells)
    if state['active'] and not merged:
        if not route:
            return None
        wanted = np.asarray(route)
        candidates = (prefix >= 5.) & (block >= 24.)
        label = 'maintain_side'
    else:
        wanted = np.array([goal])
        # Unknown forward space may be surveyed by the native policy. A directed
        # return label requires a locally measured corridor and no known block ahead.
        candidates = ((prefix >= (24. if merged else 16.)) & (block > 108.) &
                      (np.abs(_wrap(angles-goal)) <= np.deg2rad(15.)))
        label = 'return_to_goal' if had_active else 'clear_forward'
    if not candidates.any():
        return None
    angular_error = np.abs(_wrap(angles[:, None] - wanted)).min(axis=1)
    cost = angular_error
    if state['active'] and not merged:
        # When two survey hypotheses are equally plausible prefer the path with
        # the longer safe prefix, while retaining the committed side's action arc.
        cost = cost + .32 * (1. - np.minimum(prefix, 40.) / 40.) + .6 * (1. - np.minimum(block, 70.) / 70.)
    slot = np.argmin(np.where(candidates, cost, np.inf))
    if angular_error[slot] > np.deg2rad(65.):
        return None
    return float(angles[slot]), label


class _LabelledPolicy:
    def __init__(self, model, teacher_probability, seed):
        self.model = model
        self.teacher_probability = teacher_probability
        self.rng = np.random.default_rng([seed, 5027])
        self.samples = []
        self.had_active = False
        self.episode = None

    def value(self, obs):
        return self.model.value(obs)

    def act(self, obs, *, deterministic, generator):
        ep = self.episode
        turn, actor_angle, logp, value = self.model.act(obs, deterministic=True, generator=generator)
        expert = _teacher(ep, obs, self.had_active)
        self.had_active |= bool(ep.obstacle_state['active'])
        if expert is None:
            return turn, actor_angle, logp, value
        angle, label = expert
        use_teacher = self.rng.random() < self.teacher_probability
        if ep.clock_s >= 60:
            self.samples.append(dict(observation=obs, angle_rad=angle, label=label, time_s=ep.clock_s,
                                     actor_angle_rad=actor_angle, teacher_executed=bool(use_teacher)))
        # Only imitation targets and trajectories are retained, never PPO logps.
        return turn, angle if use_teacher else actor_angle, logp, value


def collect_pretrain_episode(repo, plan, actor_path, world_path, *, horizon_s, teacher_probability, out_dir):
    import torch
    from vtemp2_3.frozen_navigation import runtime
    from vtemp2_3.hierarchical_sources import atomic_json

    if not 0. <= teacher_probability <= 1. or horizon_s <= 60:
        raise ValueError('Expected a valid teacher fraction and a horizon beyond radio warmup')
    repo = Path(repo)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime(repo)
    from vtemp5.run_vtemp5_fleet import policy_from
    from vtemp4.vtemp4_curriculum import runtime_case
    from vtemp5.vtemp5_circle_metrics import circle_metrics
    from vtemp5.vtemp5_fleet import VTemp5FleetEpisode

    class LabelledEpisode(VTemp5FleetEpisode):
        def _heading_observation(self):
            self.slow.episode = self
            return super()._heading_observation()

    seed = plan['fleet_seed']
    case = plan['case']
    model, _ = policy_from(actor_path)
    model.eval().requires_grad_(False)
    policy = _LabelledPolicy(model, teacher_probability, seed)

    def progress(t, tiles, n):
        atomic_json(out_dir/'progress.json', dict(pid=os.getpid(), sim_s=t, horizon_s=horizon_s,
                    case_id=case['case_id'], received_tiles=tiles, labelled=n))

    progress(0, 0, 0)
    with open(os.devnull, 'w') as sink, redirect_stdout(sink), torch.inference_mode():
        ep = LabelledEpisode(repo, runtime_case(case), policy, fast_seed=seed, budget_s=horizon_s + 60,
                             collector_limit_s=horizon_s, lookahead_m=200., review_interval_s=30,
                             ground_control='vtemp1_connected_anchored', world_model_path=world_path,
                             progress_reward_weight=10.)
        try:
            while not ep.finished:
                ep.step()
                if ep.clock_s % 60 == 0:
                    progress(ep.clock_s, ep.adapter.tiles, len(policy.samples))
            row = ep.fleet_summary()
            track = np.asarray(ep.trajectory)
        finally:
            ep.close()
            policy.episode = None
    samples = policy.samples
    counts = dict(Counter(s['label'] for s in samples))
    # Keep trajectory-based geometry strictly downstream of teaching/action selection.
    metrics = circle_metrics(case, track, row['travelled_m'], row['goal_reached'])
    active = [r for r in row['heading_reviews'] if r['obstacle_id'] is not None]
    post = [r for r in row['heading_reviews'] if active and r['time_s'] > active[0]['time_s']
            and r['obstacle_id'] is None]
    goal = np.asarray(case['goal'])
    center, radius = case['circles'][0]
    axis = (goal - case['start']) / case['direct_distance_m']
    beyond = (track - center) @ axis >= radius + case['required_clearance_m']
    post_samples = [s for s in samples if beyond[s['time_s']]]
    summary = dict(case_id=case['case_id'], start_kind=plan['start_kind'], fleet_seed=seed,
                   end_reason=row['end_reason'], sim_time_s=row['sim_time_s'], goal_reached=row['goal_reached'],
                   remaining_m=row['remaining_m'], travelled_m=row['travelled_m'], collisions=row['collisions'],
                   direct_distance_m=case['direct_distance_m'], goal_gain_m=case['direct_distance_m']-row['remaining_m'],
                   received_tiles=row['adapter']['unique_received_tiles'], fusion_count=row['fusion_count'],
                   world=row['vtemp5_world'], circle_metrics=metrics, had_active=bool(active),
                   first_active_s=active[0]['time_s'] if active else None, post_active_reviews=len(post),
                   post_plane_labels=dict(Counter(s['label'] for s in post_samples)),
                   labels=counts, teacher_actions=sum(s['teacher_executed'] for s in samples),
                   max_native_road_nodes=max((int(s['observation']['road_valid'][:-6].sum()) for s in samples), default=0))
    progress(row['sim_time_s'], row['adapter']['unique_received_tiles'], len(samples))
    return summary, samples
