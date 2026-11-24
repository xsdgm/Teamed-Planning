"""Regression checks for evidence masking and the upper actor/critic attention path."""
import copy
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vtemp2_3.ground_policy import CANDIDATE_LIMIT, EGO_FEATURES, TASK_BLOCKS
from vtemp4.vtemp4_observation import DIRECTION_COUNT, FEATURE_COUNT, FEATURE_NAMES


def observation():
    rng = np.random.default_rng(41)
    blocks = {name: np.zeros(shape, np.float32) for name, shape in TASK_BLOCKS}
    blocks['ego'][:11] = [0., .3, .6, .8, .01, 2., .2, .1, .3, .4, .5]
    blocks['ego'][11:] = rng.random(len(blocks['ego']) - 11) > .5
    blocks['roads'][:3] = rng.normal(size=(3, 8))
    blocks['road_valid'][:3] = 1.
    blocks['adjacency'][:] = -1
    blocks['adjacency'][:3, :2] = [[0, 1], [1, 2], [2, 1]]
    blocks['peers'][0] = [.1, .2, .3, -.1, .4, 1., .4, 1.]
    blocks['relay'][0] = rng.normal(size=blocks['relay'][0].shape)
    blocks['relay_valid'][0] = 1.
    blocks['candidate_xy'][:DIRECTION_COUNT] = rng.normal(size=(DIRECTION_COUNT, 2))
    blocks['task'][0] = 1.
    packed = np.concatenate([block.ravel() for block in blocks.values()])
    present = np.arange(CANDIDATE_LIMIT) < DIRECTION_COUNT
    present[1] = False
    return dict(ground_inputs=packed, context=np.array([1., .2, .1, 3., 0., 0., 0., .8, 0., 0.], np.float32),
                history=rng.normal(size=(4, 5)).astype(np.float32), history_mask=np.array([0, 0, 1, 1], bool),
                features=rng.normal(size=(CANDIDATE_LIMIT, FEATURE_COUNT)).astype(np.float32), present=present,
                angle_arcs=np.array([[-3., -.2], [1., 2.]]), all_ring_endpoints_blocked=False,
                vehicle_world_xy=np.array([.1, .4]), goal_world_xy=np.array([.7, .6]),
                guidance_world_xy=np.array([.3, .6]), local_target_world_xy=np.array([.24, .47]))


def views(obs):
    offset = 0
    fields = {}
    for name, shape in TASK_BLOCKS:
        end = offset + math.prod(shape)
        fields[name] = obs['ground_inputs'][offset:end].reshape(shape)
        offset = end
    return fields


def validate():
    from vtemp2_3.frozen_navigation import runtime
    from vtemp4.vtemp4_unified_attention import UnifiedAttentionPolicy
    runtime(Path(__file__).resolve().parents[5])
    torch.manual_seed(61)
    net = UnifiedAttentionPolicy().eval()
    original = observation()
    fields = views(original)
    alt = copy.deepcopy(original)
    alt['angle_arcs'] = np.array([[-1., 1.]])
    weights = net.attention_weights([original, alt])
    expected = {'vehicle', 'goal', 'guidance', 'local_target', 'context', 'sensor', 'roads',
                'directions', 'peers', 'relay', 'history', 'arcs'}
    assert set(weights) == expected
    total = torch.cat(list(weights.values()), -1)
    torch.testing.assert_close(total.sum(-1), torch.ones(2, 4))
    assert torch.all(weights['vehicle'] > 0.) and torch.all(weights['goal'] > 0.)
    assert torch.all(weights['history'][:, :, :2] == 0.)
    assert torch.all(weights['arcs'][1, :, 1:] == 0.)
    assert torch.all(weights['directions'][:, :, 1] == 0.)

    policy, values = net([original])
    (policy.location.sum() + policy.rho.sum() + values.sum()).backward()
    for name, projection in net.projections.items():
        gradient = projection.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0., name
    for name in ('vehicle_world_xy', 'goal_world_xy'):
        changed = copy.deepcopy(original)
        changed[name] += [.3, -.45]
        shifted, shifted_value = net([changed])
        assert (shifted.location - policy.location).abs().max() > 1e-9, name
        assert (shifted_value - values).abs().max() > 1e-7, name

    # No receiver fields may bypass attention on the route to either head.
    handle = net.attention.register_forward_hook(lambda m, args, output:
                                                (torch.zeros_like(output[0]), output[1]))
    try:
        with torch.no_grad():
            p, v = net([original, changed])
            assert p.location[0] == p.location[1] and p.rho[0] == p.rho[1] and v[0] == v[1]
    finally:
        handle.remove()

    # Exercise unrelated garbage and NaNs in every kind of masked row.
    original['context'][0] = 0.
    fields['task'][0] = 0.
    fields['ego'][9:11] = 0.
    noisy = copy.deepcopy(original)
    padded = views(noisy)
    noisy['history'][~noisy['history_mask']] = np.nan
    noisy['features'][~noisy['present']] = np.nan
    noisy['guidance_world_xy'][:] = noisy['local_target_world_xy'][:] = np.nan
    noisy['context'][1:4] = noisy['context'][9] = np.nan
    padded['candidate_xy'][~noisy['present']] = np.nan
    padded['task'][2:] = padded['ego'][7:9] = padded['ego'][11:] = np.nan
    padded['roads'][3:] = padded['peers'][1:, :-1] = padded['relay'][1:] = np.nan
    padded['adjacency'][3:] = 888.
    with torch.no_grad():
        p, v = net([original, noisy])
        for result in (p.location, p.rho, v):
            torch.testing.assert_close(result[0], result[1])
        masked = net.attention_weights([noisy])
        for key in ('sensor', 'guidance', 'local_target'):
            assert torch.count_nonzero(masked[key]) == 0


class UnifiedAttentionTests(unittest.TestCase):
    def test_frozen_runtime_masks_gradients_and_attention_only_heads(self):
        process = subprocess.run([sys.executable, '-IB', __file__, 'native'],
                                 capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_frozen_runtime_terrain_summary_policy(self):
        process = subprocess.run([sys.executable, '-IB', __file__, 'terrain-summary'],
                                 capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


def validate_terrain_summary():
    from vtemp2_3.frozen_navigation import runtime
    from vtemp4.vtemp4_unified_attention import (
        GoalAwareTerrainSummaryEpisode, GoalAwareTerrainSummaryPolicy,
        GoalAwareUnifiedAttentionEpisode, GoalAwareUnifiedAttentionPolicy,
    )
    runtime(Path(__file__).resolve().parents[5])
    torch.manual_seed(61)
    baseline = GoalAwareUnifiedAttentionPolicy().eval()
    torch.manual_seed(61)
    summary = GoalAwareTerrainSummaryPolicy().eval()

    base_state, summary_state = baseline.state_dict(), summary.state_dict()
    assert base_state.keys() - summary_state.keys() == {'projections.sensor.weight', 'projections.sensor.bias'}
    for name, value in summary_state.items():
        torch.testing.assert_close(value, base_state[name], atol=0., rtol=0., msg=name)
    assert 'sensor' not in summary.projections

    # Matching checkpoints load strictly. Crossing formats must reveal the missing or extra patch projection.
    GoalAwareUnifiedAttentionPolicy().load_state_dict(base_state, strict=True)
    GoalAwareTerrainSummaryPolicy().load_state_dict(summary_state, strict=True)
    for model, wrong_state in ((baseline, summary_state), (summary, base_state)):
        with np.testing.assert_raises_regex(RuntimeError, r'projections\.sensor\.weight'):
            model.load_state_dict(wrong_state, strict=True)

    original = observation()
    original['present'][:DIRECTION_COUNT] = True
    free = [FEATURE_NAMES.index(f'section_{i}_free') for i in range(8)]
    original['features'][:, free[0]:] = 0.
    original['features'][:DIRECTION_COUNT, free] = 1.
    pixels = copy.deepcopy(original)
    pixel_planes = views(pixels)['ego'][11:]
    pixel_planes[:] = 1. - pixel_planes

    cases = [original, pixels]
    for state in ('blocked', 'unknown'):
        changed = copy.deepcopy(original)
        # Change only the farthest segment of the last bearing, preserving valid fractions.
        row = changed['features'][DIRECTION_COUNT - 1]
        row[free[-1]] = 0.
        row[FEATURE_NAMES.index(f'section_7_{state}')] = 1.
        cases.append(changed)

    episode = GoalAwareTerrainSummaryEpisode.__new__(GoalAwareTerrainSummaryEpisode)
    compact = []
    for case in cases:
        full_input = case['ground_inputs']
        with patch.object(GoalAwareUnifiedAttentionEpisode, '_heading_observation', return_value=dict(case)):
            result = episode._heading_observation()
        expected = np.concatenate((full_input[:11], full_input[EGO_FEATURES:]))
        np.testing.assert_array_equal(result['ground_inputs'], expected)
        np.testing.assert_array_equal(result['features'], case['features'])
        compact.append(result)

    with torch.inference_mode():
        weights = summary.attention_weights(compact[:1])
        assert 'sensor' not in weights
        assert weights['directions'].shape[-1] == DIRECTION_COUNT
        assert (weights['directions'] > 0).all()
        policy, value = summary(compact[:1])
        pixel_policy, pixel_value = summary(compact[1:2])
        torch.testing.assert_close(policy.location, pixel_policy.location, atol=0., rtol=0.)
        torch.testing.assert_close(policy.rho, pixel_policy.rho, atol=0., rtol=0.)
        torch.testing.assert_close(value, pixel_value, atol=0., rtol=0.)
        for changed in compact[2:]:
            shifted, shifted_value = summary([changed])
            assert not torch.equal(policy.location, shifted.location)
            assert not torch.equal(value, shifted_value)


if __name__ == '__main__':
    if sys.argv[1:] == ['native']:
        validate()
    elif sys.argv[1:] == ['terrain-summary']:
        validate_terrain_summary()
    else:
        unittest.main()
