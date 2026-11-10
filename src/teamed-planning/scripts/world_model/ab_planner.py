"""A/B candidate generation and model-based task selection; no hidden terrain."""
import math
import numpy as np
import torch

from .ab_mapping import make_task, task_valid


def candidates(position, goal, config, count=36):
    position, goal = np.asarray(position), np.asarray(goal)
    direction = math.atan2(*(goal-position)[::-1])
    pairs, seen = [], set()
    # Both endpoints are selected. A=C is one option, not the task definition.
    for index in range(144):
        angle = direction+(index % 12)*math.pi/6
        distance = (40., 80., 140., 190.)[(index//12) % 4]
        offset_angle = direction+((index//48) % 3-1)*math.pi/2
        offset = (0., 15., 25.)[(index//12) % 3]
        a = position+offset*np.array([math.cos(offset_angle), math.sin(offset_angle)])
        b = a+distance*np.array([math.cos(angle), math.sin(angle)])
        if (np.minimum(a, b) < 0).any() or (np.maximum(a, b) > config.world).any():
            continue
        task = make_task(a, b, config)
        key = tuple(np.round(np.r_[a, b], 5))
        if key not in seen and task_valid(task, position, config):
            pairs.append(np.stack((a, b))); seen.add(key)
    distance = np.linalg.norm(goal-position)
    if config.min_length <= distance <= config.max_length:
        task = make_task(position, goal, config)
        if task_valid(task, position, config):
            pairs.insert(0, np.stack((position, goal)))
    # Even spatial sampling retains backward/lateral tasks and all length bands.
    if len(pairs) > count:
        pairs = [pairs[i] for i in np.linspace(0, len(pairs)-1, count, dtype=int)]
    if not pairs:
        raise ValueError('No admissible A/B candidates')
    return pairs


def bootstrap_select(env, rng):
    choices = candidates(env.position, env.goal, env.config)
    if rng.random() < .3:
        index = int(rng.integers(len(choices)))
    else:
        scores = []
        for a, b in choices:
            task = make_task(a, b, env.config)
            unknown = np.count_nonzero(env.received.cells.ravel()[task.cell_ids] < 0)
            toward = np.linalg.norm(env.goal-env.position)-np.linalg.norm(env.goal-b)
            scores.append(toward+.08*unknown-.1*np.linalg.norm(b-a))
        index = int(np.argmax(scores))
    return choices[index], dict(selector='bootstrap_observed_map', candidate_count=len(choices))


class ModelPlanner:
    """One macro-step MPC: learned survey+navigation outcomes choose A/B.

    Longer sequence prediction is trained and validated separately. This first
    planner deliberately reports a one-step planning horizon, never claiming
    a long-term optimal exploration policy.
    """
    def __init__(self, model, config, device='cpu'):
        self.model = model.to(device).eval()
        self.config, self.device = config, device
        self.hidden = self.latent = None
        self.last = None

    @torch.no_grad()
    def choose(self, observation, *, exploration=0., rng=None):
        maps, state = observation
        maps = torch.as_tensor(maps, dtype=torch.float32, device=self.device)[None]
        states = torch.as_tensor(state, dtype=torch.float32, device=self.device)[None]
        if self.last is None:
            self.hidden, self.latent = self.model.initial(maps, states)
        else:
            previous_maps, previous_states, action = self.last
            self.hidden, _, _, _, _, _ = self.model.imagine(
                self.hidden, self.latent, action, previous_maps, previous_states)
            self.latent, _ = self.model.observe(self.hidden, maps, states)
        choices = candidates(np.asarray(state[:2])*1000., np.asarray(state[2:4])*1000., self.config)
        n = len(choices)
        actions = torch.as_tensor(np.asarray(choices).reshape(n, 4)/1000., dtype=torch.float32, device=self.device)
        _, _, _, pm, ps, outcome = self.model.imagine(self.hidden.repeat(n, 1), self.latent.repeat(n, 1),
            actions, maps.repeat(n, 1, 1, 1), states.repeat(n, 1))
        if not all(torch.isfinite(t).all() for t in (pm, ps, outcome)):
            raise ValueError('Non-finite world-model prediction')
        goal = states[:, 2:4]*1000.
        progress = torch.linalg.vector_norm(states[:, :2]*1000.-goal, dim=-1)-torch.linalg.vector_norm(ps[:, :2]*1000.-goal, dim=-1)
        cell_limits = torch.tensor([make_task(a, b, self.config).cell_count for a, b in choices], device=self.device)
        information = torch.minimum(outcome[:, 1].clamp_min(0)*1000., cell_limits)
        duration = outcome[:, 2].clamp(0, self.config.timeout_seconds/200.)*200.
        failure = outcome[:, 3].sigmoid()
        utility = progress+.002*information-.02*duration-2.*failure
        index = int(utility.argmax())
        if exploration and rng is not None and rng.random() < exploration:
            index = int(rng.integers(n))
        self.last = maps.clone(), states.clone(), actions[index:index+1].clone()
        diagnostics = dict(selector='world_model_mpc', planning_horizon=1,
            candidate_count=n, selected_index=index, utility=utility.cpu().tolist(),
            predicted_progress_m=progress.cpu().tolist(), predicted_new_cells=information.cpu().tolist(),
            predicted_survey_seconds=duration.cpu().tolist(), predicted_failure_score=failure.cpu().tolist(),
            uncertainty_calibrated=False)
        return choices[index], diagnostics
