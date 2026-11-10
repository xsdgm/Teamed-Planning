"""Observation-only candidate planning with a frozen vtmp2 world model.

Imagined maps and states are private rollout copies. Their graph-count gain is
a prediction proxy, never a new observation or a calibrated reachability claim.
Legacy checkpoints only learned drive actions; they cannot score survey actions.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from pathlib import Path

import numpy as np
import torch


CHECKPOINT_FORMAT = 'vtmp2-world-model-training-v1'
DRIVE_ONLY = 'legacy_drive_v1'
DRIVE_SURVEY = 'drive_survey_v1'
MAP_SHAPE = (4, 32, 32)
STATE_FEATURES = 16


def _field(candidate, name, default=None):
    return candidate.get(name, default) if isinstance(candidate, dict) else getattr(candidate, name, default)


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


class WorldModelPlanner:
    """Compare candidates under equal, reproducible model-sampling conditions.

    Each candidate's particles form a batch. Candidates reuse the same random
    stream, so candidate ordering does not change the stochastic comparison.
    No candidates are removed here; spatial/purpose budgeting belongs to the
    caller. ``purpose`` defaults to ``drive`` and can be ``survey`` only with a
    checkpoint explicitly trained with ``action_contract=drive_survey_v1``.

    ``observe`` refreshes the posterior using actual received observations. Its
    recurrent transition advances after ten seconds or a target change, matching
    the collector's macro-action boundary approximately. Interrupted durations
    are not explicit model inputs; gaps longer than one macro step reset belief
    instead of inventing unseen history. Diagnostics expose these limitations.
    """

    selector_mode = 'world_model_mission'

    def __init__(self, model, *, device='cpu', horizon_steps=12, particles=4,
                 seed=20261108, macro_seconds=10., action_contract=DRIVE_ONLY,
                 progress_weight=1., information_weight=100., stall_weight=30.,
                 cbf_weight=100., uncertainty_weight=.5, travel_weight=.01,
                 checkpoint_info=None):
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Planner device must be cpu or cuda')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is unavailable')
        if getattr(model, 'algorithm', None) != 'vtmp2':
            raise ValueError('Planner requires algorithm=vtmp2')
        if action_contract not in (DRIVE_ONLY, DRIVE_SURVEY):
            raise ValueError('Unknown action_contract')
        self.horizon_steps = _positive_integer(horizon_steps, 'horizon_steps')
        self.particles = _positive_integer(particles, 'particles')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63)')
        self.seed = seed
        self.macro_seconds = float(macro_seconds)
        if not math.isfinite(self.macro_seconds) or self.macro_seconds <= 0:
            raise ValueError('macro_seconds must be finite and positive')
        self.weights = dict(progress=progress_weight, information=information_weight,
                            stalled=stall_weight, cbf_rejected=cbf_weight,
                            uncertainty=uncertainty_weight, travel=travel_weight)
        for key, value in self.weights.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} weight must be finite and nonnegative')
        self.action_contract = action_contract
        self.model = model.to(self.device).eval()
        for name, value in self.model.state_dict().items():
            if not torch.is_tensor(value) or not torch.isfinite(value).all():
                raise ValueError(f'Non-finite model parameter: {name}')
        self.checkpoint_info = dict(checkpoint_info or {})
        self.reset()

    @classmethod
    def from_checkpoint(cls, path, device='cpu', **options):
        """Load on CPU with the restricted unpickler, then move validated weights.

        Existing trainer checkpoints include NumPy RNG arrays. The small NumPy
        allowlist loads those ordinary arrays without enabling arbitrary pickle
        execution. There is deliberately no ``weights_only=False`` fallback.
        """
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f'Checkpoint does not exist: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        numpy_core = np._core if hasattr(np, '_core') else np.core
        allowed = [numpy_core.multiarray._reconstruct, np.ndarray, np.dtype,
                   type(np.dtype(np.uint32))]
        try:
            with torch.serialization.safe_globals(allowed):
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        except Exception as exc:
            raise ValueError(f'Checkpoint could not be safely loaded: {exc}') from exc
        if not isinstance(checkpoint, dict) or checkpoint.get('format') != CHECKPOINT_FORMAT:
            raise ValueError('Unsupported checkpoint format')
        config = checkpoint.get('config')
        if not isinstance(config, dict) or config.get('algorithm') != 'vtmp2':
            raise ValueError('Checkpoint config must declare algorithm=vtmp2')
        if checkpoint.get('algorithm', 'vtmp2') != 'vtmp2':
            raise ValueError('Checkpoint algorithm does not match vtmp2')
        schema = config.get('schema_version')
        contract = config.get('action_contract', DRIVE_ONLY if schema == 2 else None)
        if schema not in (2, 3) or contract not in (DRIVE_ONLY, DRIVE_SURVEY):
            raise ValueError('Unsupported schema_version/action_contract')
        if contract == DRIVE_SURVEY and schema != 3:
            raise ValueError('drive_survey_v1 requires schema_version=3')
        observed_purposes = config.get('observed_action_purposes', ['drive'] if schema == 2 else [])
        if (not isinstance(observed_purposes, list)
                or any(value not in ('drive', 'survey') for value in observed_purposes)):
            raise ValueError('Invalid observed_action_purposes in checkpoint')
        if contract == DRIVE_SURVEY and not {'drive', 'survey'}.issubset(observed_purposes):
            raise ValueError('drive_survey_v1 checkpoint must document training observations for both drive and survey')
        for name, maximum in (('hidden', 4096), ('latent', 1024)):
            value = config.get(name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f'Invalid checkpoint {name}')
        if 'action_contract' in options and options['action_contract'] != contract:
            raise ValueError('Cannot override checkpoint action_contract')
        options.pop('action_contract', None)
        weights = checkpoint.get('model')
        if not isinstance(weights, dict) or not weights:
            raise ValueError('Checkpoint has no model state_dict')
        for name, value in weights.items():
            if not torch.is_tensor(value) or not torch.isfinite(value).all():
                raise ValueError(f'Invalid/non-finite checkpoint tensor: {name}')
        # Delayed import avoids runtime -> planner -> model -> data cycles.
        if __package__:
            from .model import Vtmp2WorldModel
        else:
            from model import Vtmp2WorldModel
        # Constructing layers must not consume the caller's global random stream.
        with torch.random.fork_rng(devices=[]):
            model = Vtmp2WorldModel(hidden=config['hidden'], latent=config['latent'])
        try:
            model.load_state_dict(weights, strict=True)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f'Checkpoint state_dict does not match its architecture: {exc}') from exc
        rollout_steps = config.get('rollout_steps', 1)
        rollout_weight = config.get('rollout_weight', 0.)
        if (type(rollout_steps) is not int or rollout_steps < 1
                or not isinstance(rollout_weight, (int, float))
                or not math.isfinite(rollout_weight) or rollout_weight < 0):
            raise ValueError('Invalid rollout training configuration in checkpoint')
        multistep_trained = rollout_steps > 1 and rollout_weight > 0
        info = dict(path=str(path), sha256=digest.hexdigest(), epoch=checkpoint.get('epoch'),
                    schema_version=schema, action_contract=contract,
                    observed_action_purposes=observed_purposes,
                    legacy_one_step_training=not multistep_trained,
                    training_objective=('one_step_prior_plus_free_rollout' if multistep_trained
                                        else 'one_step_prior_prediction'),
                    rollout_training_steps=rollout_steps, rollout_training_weight=rollout_weight,
                    multistep_rollout_validated=False,
                    validation_limitation='checkpoint weights/config alone do not establish accuracy for planner horizon and fixed-candidate actions')
        return cls(model, device=device, action_contract=contract,
                   checkpoint_info=info, **options)

    def reset(self):
        self._maps = self._states = self._hidden = self._latent = None
        self._anchor_maps = self._anchor_states = None
        self._anchor_hidden = self._anchor_latent = None
        self._anchor_time = self._last_time = None
        self._anchor_target = None
        self._belief_updates = self._gap_resets = self._irregular_updates = 0
        self.last_diagnostics = {}

    def _tensor(self, value, shape, name):
        result = torch.as_tensor(value, dtype=torch.float32, device=self.device).detach().clone()
        if tuple(result.shape) != shape or not torch.isfinite(result).all():
            raise ValueError(f'{name} must be finite with shape {shape}')
        return result

    @staticmethod
    def _finite(*values):
        if any(not torch.is_tensor(value) or not torch.isfinite(value).all() for value in values):
            raise ValueError('World model returned a non-finite prediction')

    def _action(self, target, purpose='drive', batch_size=1):
        if purpose not in ('drive', 'survey'):
            raise ValueError(f'Unknown candidate purpose: {purpose}')
        if purpose == 'survey' and self.action_contract != DRIVE_SURVEY:
            raise ValueError('Survey candidates require a drive_survey_v1-trained checkpoint; legacy drive-only weights cannot score survey utility')
        action = torch.cat((target / 1000., target.new_tensor([1. if purpose == 'drive' else -1.])))
        return action.unsqueeze(0).repeat(batch_size, 1)

    @torch.no_grad()
    def observe(self, maps, states, now, executed_target=None, executed_purpose='drive'):
        """Assimilate only information actually received by the vehicle.

        ``executed_target`` may be a two-coordinate drive target, or an object/
        dictionary with ``point`` and ``purpose``. It denotes the current command;
        a changed command closes the preceding macro action. Missing command
        history causes a fresh posterior rather than an invented transition.
        """
        maps = self._tensor(maps, MAP_SHAPE, 'maps').unsqueeze(0)
        states = self._tensor(states, (STATE_FEATURES,), 'states').unsqueeze(0)
        if (maps < 0).any() or (maps > 1).any():
            raise ValueError('Observed maps must lie in [0, 1]')
        now = float(now)
        if not math.isfinite(now) or (self._last_time is not None and now < self._last_time):
            raise ValueError('Observation time must be finite and nondecreasing')
        target = None
        if executed_target is not None:
            point = _field(executed_target, 'point', executed_target)
            purpose = _field(executed_target, 'purpose', executed_purpose)
            point = self._tensor(point, (2,), 'executed_target')
            target = self._action(point, purpose)
        initialize = self._hidden is None or target is None or self._anchor_target is None
        elapsed = 0. if self._anchor_time is None else now - self._anchor_time
        changed = (target is not None and self._anchor_target is not None
                   and not torch.equal(target, self._anchor_target))
        boundary = elapsed >= self.macro_seconds or changed
        if elapsed > self.macro_seconds + 1e-6:
            initialize = True
            self._gap_resets += 1
        if initialize:
            hidden, latent = self.model.initial(maps, states, sample=False)
            boundary = True
        elif boundary and elapsed > 0:
            hidden, _, _, _, _, _ = self.model.imagine(
                self._anchor_hidden.clone(), self._anchor_latent.clone(),
                self._anchor_target.clone(), self._anchor_maps.clone(),
                self._anchor_states.clone(), sample=False)
            latent, _ = self.model.observe(hidden, self.model.encoder(maps), states, False)
            if abs(elapsed - self.macro_seconds) > 1e-6:
                self._irregular_updates += 1
        else:
            hidden = self._hidden.clone()
            latent, _ = self.model.observe(hidden, self.model.encoder(maps), states, False)
        self._finite(hidden, latent)
        self._maps, self._states = maps, states
        self._hidden, self._latent = hidden.detach().clone(), latent.detach().clone()
        self._last_time = now
        if boundary:
            self._anchor_maps, self._anchor_states = maps.clone(), states.clone()
            self._anchor_hidden, self._anchor_latent = hidden.clone(), latent.clone()
            self._anchor_time = now
            self._anchor_target = None if target is None else target.clone()
            self._belief_updates += 1

    @contextmanager
    def _random_stream(self):
        devices = [] if self.device.type == 'cpu' else [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            # Seed only generators used by this planner, avoiding other GPUs.
            torch.random.default_generator.manual_seed(self.seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(self.seed)
            yield

    @torch.no_grad()
    def rank(self, candidates, goal):
        """Return a finite cost for every candidate, in the original order.

        Every imagined step consumes the preceding predicted maps, states and
        prior latent. No future observations or scene truth enter this method.
        The arrival head is deliberately unused: a short-window arrival score
        is not the probability of ultimately reaching a distant endpoint.
        """
        if self._states is None:
            raise ValueError('Call observe before rank')
        goal = self._tensor(goal, (2,), 'goal')
        candidates = list(candidates)
        prepared = []
        for candidate in candidates:
            point = self._tensor(_field(candidate, 'point'), (2,), 'candidate.point')
            purpose = _field(candidate, 'purpose', 'drive')
            action = self._action(point, purpose, self.particles)
            distance_cost = float(_field(candidate, 'cost', 0.))
            if not math.isfinite(distance_cost) or distance_cost < 0:
                raise ValueError('Candidate cost must be finite and nonnegative')
            prepared.append((candidate, point, purpose, action, distance_cost))
        initial_distance = torch.linalg.vector_norm(self._states[0, :2] * 1000. - goal)
        initial_counts = self._states[0, 8:10].clamp(0., 1.)
        costs, diagnostics = [], []
        for candidate, point, purpose, action, distance_cost in prepared:
            hidden = self._hidden.repeat(self.particles, 1).clone()
            latent = self._latent.repeat(self.particles, 1).clone()
            maps = self._maps.repeat(self.particles, 1, 1, 1).clone()
            states = self._states.repeat(self.particles, 1).clone()
            stall = states.new_zeros(self.particles)
            rejection = states.new_zeros(self.particles)
            elapsed = states.new_zeros(self.particles)
            with self._random_stream():
                for _ in range(self.horizon_steps):
                    hidden, latent, _, maps, states, outcome = self.model.imagine(
                        hidden, latent, action, maps, states, sample=True)
                    self._finite(hidden, latent, maps, states, outcome)
                    if (tuple(maps.shape) != (self.particles, *MAP_SHAPE)
                            or tuple(states.shape) != (self.particles, STATE_FEATURES)
                            or tuple(outcome.shape) != (self.particles, 4)):
                        raise ValueError('World model returned an invalid prediction shape')
                    probabilities = outcome.sigmoid()
                    # These are mean short-window scores, not a joint probability.
                    stall += probabilities[:, 1] / self.horizon_steps
                    rejection += probabilities[:, 2] / self.horizon_steps
                    elapsed += probabilities[:, 3] * self.macro_seconds
            positions = states[:, :2] * 1000.
            progress = initial_distance - torch.linalg.vector_norm(positions - goal, dim=-1)
            # Log-normalized total received node/edge counts have a fixed world
            # meaning; unlike moving map pixels, their gain is not camera motion.
            count_gain = (states[:, 8:10].clamp(0., 1.) - initial_counts).clamp_min(0.).mean(-1)
            spread = positions.var(dim=0, unbiased=False).sum().sqrt()
            graph_spread = count_gain.std(unbiased=False)
            progress_mean, information_mean = float(progress.mean()), float(count_gain.mean())
            stall_mean, rejection_mean = float(stall.mean()), float(rejection.mean())
            spread_m = float(spread)
            # Survey actions are rewarded for information without requiring the
            # vehicle to drive to the survey point. Both purposes share penalties.
            progress_reward = self.weights['progress'] * progress_mean if purpose == 'drive' else 0.
            # Ground-route length is not a drone survey travel-time estimate.
            route_penalty = self.weights['travel'] * distance_cost if purpose == 'drive' else 0.
            graph_spread_penalty = (self.weights['uncertainty'] * self.weights['information']
                                    * float(graph_spread))
            uncertainty_penalty = self.weights['uncertainty'] * spread_m + graph_spread_penalty
            cost = (-progress_reward - self.weights['information'] * information_mean
                    + self.weights['stalled'] * stall_mean
                    + self.weights['cbf_rejected'] * rejection_mean
                    + uncertainty_penalty
                    + route_penalty)
            if not math.isfinite(cost):
                raise ValueError('World model produced a non-finite candidate cost')
            costs.append(cost)
            diagnostics.append(dict(
                point=point.cpu().tolist(), purpose=purpose,
                status=_field(candidate, 'status', 'unknown'),
                source=_field(candidate, 'source', 'unknown'), cost=cost,
                predicted_goal_progress_m=progress_mean,
                predicted_graph_gain_proxy=information_mean,
                predicted_stall_score=stall_mean, predicted_cbf_rejection_score=rejection_mean,
                position_spread_m=spread_m, graph_gain_spread=float(graph_spread),
                predicted_elapsed_seconds=float(elapsed.mean()),
                predicted_terminal_position_m=positions.mean(0).cpu().tolist(),
                progress_reward=progress_reward,
                information_reward=self.weights['information'] * information_mean,
                uncertainty_penalty=uncertainty_penalty,
                graph_uncertainty_penalty=graph_spread_penalty,
                route_cost_penalty=route_penalty))
        selected = min(range(len(costs)), key=costs.__getitem__) if costs else None
        self.last_diagnostics = dict(
            algorithm='vtmp2_world_model_planner', action_contract=self.action_contract,
            checkpoint=dict(self.checkpoint_info), horizon_steps=self.horizon_steps,
            requested_horizon_seconds=self.horizon_steps * self.macro_seconds,
            particles=self.particles, seed=self.seed, candidate_count=len(candidates),
            candidates=diagnostics, selected_index=selected,
            selection_basis='minimum predicted progress/information/risk cost; status is descriptive only',
            arrival_head_used=False, uncertainty_calibrated=False,
            information_is_observed=False, information_proxy='positive gain of predicted log-normalized received graph node/edge counts',
            risk_interpretation='mean short-window model scores, not calibrated route reachability',
            rollout_validation='free rollout; checkpoint metadata does not establish long-horizon accuracy',
            belief_updates=self._belief_updates, history_gap_resets=self._gap_resets,
            irregular_duration_updates=self._irregular_updates,
            duration_limitation='action requests ten seconds; realized duration is predicted and not an explicit recurrent input')
        return costs
