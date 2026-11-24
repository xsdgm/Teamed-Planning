"""Small VTemp4 ground / probe policy experiment on disjoint archived V9 maps."""
import argparse
from functools import lru_cache
import heapq
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import torch
from torch import nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src/teamed-planning/scripts'))
from survey.survey_pretrain_env import SurveyEnv, ground_plan, _sample_corridors
from survey.survey_pretrain_maps import ARCHIVE, load_survey_map

LIMIT = 300
CANDIDATES = 12
SEED = 419
torch.set_num_threads(1)


def reward_distances(free, goal):
    # Only the reward function and initial solvability check see this graph.
    ids = np.arange(free.size).reshape(free.shape)
    vertical = free[1:] & free[:-1]
    horizontal = free[:, 1:] & free[:, :-1]
    a = np.r_[ids[1:][vertical], ids[:, 1:][horizontal]]
    b = np.r_[ids[:-1][vertical], ids[:, :-1][horizontal]]
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(free.size, free.size)).tocsr()
    return dijkstra(graph, directed=False, indices=ids[tuple(goal)], unweighted=True).reshape(free.shape)


@lru_cache(maxsize=96)
def terrain(case_id):
    row = dict(CASES[case_id])
    row['circles_path'] = str(ROOT / ARCHIVE / 'geometry' / Path(row['circles_path']).name)
    scene = load_survey_map(row)
    goal = np.floor((scene.goal-scene.origin)/scene.resolution).astype(int)[::-1]
    return scene, reward_distances(scene.free, goal)


def split():
    rows = list(CASES.values())
    np.random.default_rng(SEED).shuffle(rows)
    groups, used = [], set()
    for row in rows:
        if not 100 <= row['direct_distance_m'] <= 160 or row['circles_sha256'] in used:
            continue
        scene, dist = terrain(row['case_id'])
        start = np.floor((scene.start-scene.origin)/scene.resolution).astype(int)[::-1]
        if np.isfinite(dist[tuple(start)]):
            groups.append(row['case_id'])
            used.add(row['circles_sha256'])
        if len(groups) == 96:
            break
    assert len(groups) == 96
    return {'train': groups[:60], 'dev': groups[60:72], 'test': groups[72:]}


def air_route(start, goal):
    """A* over a flat unobstructed air grid. Ground occupancy is not an air input."""
    start, goal = tuple(start), tuple(goal)
    def estimate(p):
        dy, dx = abs(p[0]-goal[0]), abs(p[1]-goal[1])
        return max(dx, dy) + (math.sqrt(2)-1)*min(dx, dy)
    queue, cost, parent = [(estimate(start), 0., start)], {start: 0.}, {}
    while queue:
        _, negative_g, p = heapq.heappop(queue)
        if p == goal:
            break
        if -negative_g > cost[p]:
            continue
        # In open air an optimal path always exists inside the endpoint rectangle.
        for dy in (0, int(np.sign(goal[0]-p[0]))):
            for dx in (0, int(np.sign(goal[1]-p[1]))):
                if dx == dy == 0:
                    continue
                q = (p[0]+dy, p[1]+dx)
                g = cost[p] + math.hypot(dy, dx)
                if g < cost.get(q, math.inf):
                    cost[q], parent[q] = g, p
                    heapq.heappush(queue, (g+estimate(q), -g, q))
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return np.array(path[::-1])


class Probe:
    def __init__(self, env):
        self.cell = env.pos.copy()
        self.path = np.array([self.cell])
        self.length = np.array([0.])
        self.depart = -100
        self.active = False
        self.pending = []
        self.receipts = self.new_cells = 0

    def assign(self, target, now):
        self.depart = now
        self.active = target is not None
        if self.active:
            self.path = air_route(self.cell, target)
            self.length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]

    def tick(self, env):
        if self.active:
            elapsed = env.time-self.depart
            index = np.searchsorted(self.length, elapsed*6./env.resolution, side='right')-1
            self.cell = self.path[index]
            arrived = index == len(self.path)-1
            if arrived or elapsed % 3 == 0:
                ids = env._disk(self.cell, 12.)
                values = np.where(env._free.flat[ids], -1, 1)
                self.pending.append((env.time+2, ids, values))
            self.active = not arrived
        while self.pending and self.pending[0][0] <= env.time:
            due, ids, values = self.pending.pop(0)
            self.new_cells += int(np.count_nonzero(env.cells.flat[ids] == 0))
            env._receive(np.empty(0, int), ids, values, due-2)
            self.receipts += 1


def stages(env, fronts, known_goal):
    points = [env.dest.copy()] if known_goal else []
    valid = fronts
    while len(valid) and len(points) < 6:
        p = valid[0]
        if not points or np.min(np.linalg.norm(np.asarray(points)-p, axis=1)) >= 6:
            points.append(p)
        valid = valid[np.linalg.norm(valid-p, axis=1) >= 6]
    return np.asarray(points if points else [env.pos.copy()])


def stage_path(env, target):
    lo = np.maximum(env._seen_min-1, 0)
    hi = np.minimum(env._seen_max+2, env.cells.shape)
    view = env.cells[lo[0]:hi[0], lo[1]:hi[1]]
    path, _, reason = ground_plan(view, env.pos-lo, target-lo, env.resolution)
    path = path+lo
    if reason != 'known_goal_route' or not np.array_equal(path[-1], target):
        raise AssertionError('Unreachable stage issued')
    if not np.all(env.cells[tuple(path.T)] == -1):
        raise AssertionError('Unmeasured ground segment')
    return path


def policy_inputs(env, probe, points, stage):
    cells, pos, goal = env.cells, env.pos, env.dest
    def fractions(p):
        y, x = p
        tile = cells[max(0,y-8):y+9, max(0,x-8):x+9]
        return [np.mean(tile == 0), np.mean(tile == 1)]
    delta = points-pos
    progress = np.linalg.norm(goal-pos)-np.linalg.norm(goal-points, axis=1)
    context = np.r_[(goal-pos)/160., (probe.cell-pos)/80., (stage-pos)/80.,
                    env.time/LIMIT, fractions(pos)]
    features = np.column_stack([delta/50., progress/30., np.linalg.norm(delta, axis=1)/50.,
                                np.linalg.norm(points-probe.cell, axis=1)/60.,
                                np.linalg.norm(points-stage, axis=1)/50.,
                                np.all(points == goal, axis=1), [fractions(p) for p in points]])
    return context.astype('f'), features.astype('f')


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(9, 24), nn.Tanh())
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(33, 24), nn.Tanh(), nn.Linear(24, 1))
                                    for _ in range(2)])
        self.value = nn.Sequential(nn.Linear(24, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, context, features, head):
        z = self.encoder(context)
        both = torch.cat((z[:, None].expand(-1, features.size(1), -1), features), dim=-1)
        return self.heads[head](both).squeeze(-1), self.value(z).squeeze(-1)


def choose(net, mode, head, inputs, rule_choice, rng, history, progress):
    ctx, features = inputs
    if mode == 'rule' or len(features) == 1:
        return rule_choice
    with torch.no_grad():
        logits, value = net(torch.tensor(ctx[None]), torch.tensor(features[None]), head)
        dist = Categorical(logits=logits[0])
        if mode == 'train':
            action = int(rng.choice(len(features), p=dist.probs.double().numpy()/dist.probs.sum(dtype=torch.float64).item()))
            history.append((head, ctx, features, action, float(dist.logits[action]), float(value), progress))
        else:
            action = int(logits.argmax())
    return action


def episode(case_id, net, ground_mode, probe_mode, seed):
    scene, dist = terrain(case_id)
    env = SurveyEnv(**scene.as_kwargs(), max_time_s=LIMIT)
    assert env.resolution == 1.
    rng = np.random.default_rng(seed)
    start_dist = dist[tuple(env.pos)]
    probe, history = Probe(env), []
    stage = env.pos.copy()
    issued = -100
    ground_issues = probe_issues = checks = 0
    stage_active = False
    while env.time < LIMIT and not np.array_equal(env.pos, env.dest):
        if env.time % 5 == 0:
            need_ground = not stage_active or np.array_equal(env.pos, stage) or env.time-issued >= 60
            age = env.time-probe.depart
            need_probe = probe_mode != 'off' and (age >= 60 or (age >= 20 and not probe.active and not probe.pending))
            progress = 0.2*(1.-dist[tuple(env.pos)]/start_dist)
            if need_probe or need_ground:
                _, fronts, reason = env._plan_on_receiver_map()
                known_goal = reason == 'known_goal_route'
            if need_ground:
                points = stages(env, fronts, known_goal)
                i = choose(net, ground_mode, 0, policy_inputs(env, probe, points, stage),
                           0, rng, history, progress)
                stage, issued = points[i], env.time
                stage_active = True
                ground_issues += 1
            # Independently reconstruct a confirmed path both at issuance and on each 5 s tick.
            path = stage_path(env, stage)
            env._path, env._path_index = path[:26], 0
            checks += 1
            if need_probe:
                questions = [] if known_goal else _sample_corridors(env.cells, fronts, env.dest, env.resolution)
                if questions:
                    points = np.array([q[0] for q in questions])
                    inputs = policy_inputs(env, probe, points, stage)
                    scores = [(np.linalg.norm(env.dest-env.pos)-np.linalg.norm(env.dest-p))/25.
                              + min(1., len(gap)/16.)-np.linalg.norm(p-env.pos)/125. for p, gap in questions]
                    i = choose(net, probe_mode, 1, inputs, int(np.argmax(scores)), rng, history, progress)
                    probe.assign(points[i], env.time)
                    probe_issues += 1
                else:
                    probe.assign(None, env.time)
        env.time += 1
        env._advance_ground()
        probe.tick(env)
    success = bool(np.array_equal(env.pos, env.dest))
    reward = 0.2*(1.-dist[tuple(env.pos)]/start_dist) + success*(1.-0.3*env.time/LIMIT)
    return dict(case=case_id, success=success, seconds=env.time, reward=float(reward),
                distance_m=env.travelled, stage_issues=ground_issues, probe_issues=probe_issues,
                reachability_checks=checks, probe_packets=probe.receipts, probe_new_cells=probe.new_cells), history


def update(net, opt, samples):
    for _ in range(4):
        losses = []
        for head in (0, 1):
            rows = [r for r in samples if r[0] == head]
            if not rows:
                continue
            _, contexts, features, actions, old_lp, values, returns = zip(*rows)
            padded = np.zeros((len(rows), CANDIDATES, 9), 'f')
            mask = np.zeros(padded.shape[:2], bool)
            for i, feat in enumerate(features):
                padded[i, :len(feat)] = feat
                mask[i, :len(feat)] = True
            logits, predicted = net(torch.tensor(np.stack(contexts)), torch.tensor(padded), head)
            dist = Categorical(logits=logits.masked_fill(~torch.tensor(mask), -1e8))
            log_ratio = dist.log_prob(torch.tensor(actions))-torch.tensor(old_lp)
            ratio = log_ratio.exp()
            target = torch.tensor(returns, dtype=torch.float32)
            advantage = target-torch.tensor(values)
            advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-6)
            surrogate = torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)
            losses.append(-surrogate.mean()+.5*(target-predicted).square().mean()-.01*dist.entropy().mean())
        opt.zero_grad()
        torch.stack(losses).mean().backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.)
        opt.step()


def evaluate(net, ids, ground_mode, probe_mode):
    return [episode(case, net, ground_mode, probe_mode, SEED)[0] for case in ids]


def summary(rows):
    return dict(n=len(rows), arrivals=sum(r['success'] for r in rows),
                mean_capped_seconds=float(np.mean([r['seconds'] for r in rows])),
                mean_reward=float(np.mean([r['reward'] for r in rows])),
                stage_issues=sum(r['stage_issues'] for r in rows),
                reachability_checks=sum(r['reachability_checks'] for r in rows),
                probe_packets=sum(r['probe_packets'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--updates', type=int, default=24)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    groups = split()
    (out/'splits.json').write_text(json.dumps(groups, indent=2))
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    net = Policy()
    initial = {k:v.clone() for k, v in net.state_dict().items()}
    opt = torch.optim.Adam(net.parameters(), lr=.003)
    best = -math.inf
    for step in range(1, args.updates+1):
        rows, samples = [], []
        for case in rng.choice(groups['train'], size=8, replace=False):
            row, trajectory = episode(case, net, 'train', 'train', int(rng.integers(2**30)))
            rows.append(row)
            for head, ctx, feat, action, lp, value, prefix in trajectory:
                samples.append((head, ctx, feat, action, lp, value, row['reward']-prefix))
        update(net, opt, samples)
        stats = dict(update=step, training=summary(rows))
        if step % 6 == 0 or step == args.updates:
            stats['dev'] = summary(evaluate(net, groups['dev'], 'learned', 'learned'))
            if stats['dev']['mean_reward'] > best:
                best = stats['dev']['mean_reward']
                torch.save(dict(state=net.state_dict(), update=step), out/'best.pt')
        print(json.dumps(stats), flush=True)
        with (out/'training.jsonl').open('a') as f:
            print(json.dumps(stats), file=f)
    checkpoint = torch.load(out/'best.pt', map_location='cpu', weights_only=True)
    net.load_state_dict(checkpoint['state'])
    delta = {name: math.sqrt(sum(float((v-initial[k]).square().sum()) for k,v in net.state_dict().items()
                                 if k.startswith(name))) for name in ('encoder','heads.0','heads.1')}
    assert min(delta.values()) > 0
    test = {}
    for name, modes in {'vtemp4': ('learned','learned'), 'heuristic': ('rule','rule'),
                        'learned_ground': ('learned','rule'), 'learned_probe': ('rule','learned'),
                        'ground_only': ('rule','off'), 'learned_ground_only': ('learned','off')}.items():
        test[name] = evaluate(net, groups['test'], *modes)
    net.load_state_dict(initial)
    test['untrained'] = evaluate(net, groups['test'], 'learned', 'learned')
    (out/'test_episodes.json').write_text(json.dumps(test, indent=2))
    result = dict(seed=SEED, updates=args.updates, episodes_per_update=8,
                  selected_update=checkpoint['update'], delta_l2=delta,
                  time_limit_s=LIMIT, test={k:summary(v) for k,v in test.items()})
    (out/'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    CASES = {r['case_id']: r for r in json.loads((ROOT/ARCHIVE/'scenes.json').read_text())['train']}
    main()
