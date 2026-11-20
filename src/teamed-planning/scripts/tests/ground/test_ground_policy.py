"""Run the archived GAT/relay contract in a clean interpreter just as workers do."""
from pathlib import Path
import subprocess
import sys
import unittest


class GroundPolicyTest(unittest.TestCase):
    def test_frozen_encoder_and_ground_inputs(self):
        result = subprocess.run([sys.executable, '-I', '-B', __file__, 'isolated'],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def check_ground_policy():
    from copy import deepcopy
    from types import SimpleNamespace
    import numpy as np
    import torch

    scripts = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(scripts))
    from ground.ground_policy import (BLOCKS, CANDIDATE_LIMIT, GROUND_INPUT_WIDTH,
                                          GroundSlowActorCritic, ground_observation)
    assert 'drone.cooperative_mappo' not in sys.modules
    from world_model.frozen_navigation import runtime
    from world_model.slow_policy import tensors, update
    runtime(scripts.parents[2])
    torch.manual_seed(17)
    roads = dict(graph_positions=[[0., 0.], [5., 0.], [20., 1.]], graph_confidence=[.3, .7, .8],
                 graph_source_stamps=[0., np.nan, 5.],
                 graph_edge_index=np.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
                 graph_observation_counts=[1, 4, 7])
    own = dict(position=[0., 0., 0.], velocity=[0., 0., 0.], orientation=0., angular_velocity=0.)
    local = dict(self=own, occupancy_map=np.array([[0, 1], [1, 1]]), map_resolution=.5,
                 occupancy_known_mask=np.array([[True, True], [False, True]]))
    env = SimpleNamespace(obs={'car': local}, ground='car', env=SimpleNamespace(time=8.))
    episode = SimpleNamespace(env=env, belief=lambda: roads, goal=[20., 1.],
                              manager=SimpleNamespace(target=[5., 0.]))
    cand = dict(features=np.zeros((CANDIDATE_LIMIT, 52), np.float32), targets=np.zeros((CANDIDATE_LIMIT, 2)),
                present=np.arange(CANDIDATE_LIMIT) < 3, mask=np.arange(CANDIDATE_LIMIT) < 2)
    cand['targets'][:3] = [0., 0.], [20., 1.], [5., 0.]
    peers = [dict(origin_id='u1', relative_position=[2., 1., 5.], relative_velocity=None)]
    rows = [dict(origin_id='u2', payload_type='drone_state', position=[35., 25., 5.],
                 velocity=[2., 0.], goal=None, age=7., hop=2, sigma=1., time_unknown=False)]
    obs = ground_observation(episode, cand, peers, rows)
    parts = {}
    start = 0
    for name, shape in BLOCKS:
        end = start + np.prod(shape)
        parts[name] = obs['ground_inputs'][start:end].reshape(shape)
        start = end
    assert start == GROUND_INPUT_WIDTH == len(obs['ground_inputs'])
    assert np.isclose(parts['roads'][0, 3], np.log1p(8.))
    assert parts['roads'][1, 4] == 1 and np.isclose(parts['roads'][1, 2], .7)
    assert parts['adjacency'][0, :3].tolist() == [0, 1, -1] and parts['roads'][1, 6] > parts['roads'][0, 6]
    assert parts['relay'][0, 6:8].tolist() == [7., 2.] and parts['relay'][0, 12] == 0.
    assert parts['peers'][0, 5] == 0. and parts['peers'][0, -1] == 1.
    from ground.ground_policy import ground_sensor_tile
    tile = ground_sensor_tile(dict(local, map_axes='xy', occupancy_map=[[.5, .75], [1., .25]]))
    np.testing.assert_array_equal(tile, [[0, -1], [1, 0]])
    local_unknown = parts['ego'][-256:].reshape(16, 16)
    assert local_unknown[-1, 0] == 1. and local_unknown[0, 0] == 0.
    model = GroundSlowActorCritic(52, 16)
    assert type(model.actor_encoder.gat1).__module__ == 'drone.graph_attention_mappo'
    obs2 = ground_observation(episode, cand, [], [])
    # Both unobserved candidates have the same global state; neither acquires
    # a road context solely because a very distant road happens to exist.
    far = deepcopy(cand)
    far['targets'][:2] = [200., 1.], [220., 3.]
    far_dist, _ = model(*tensors(ground_observation(episode, far, peers, rows)))
    torch.testing.assert_close(far_dist.logits[0], far_dist.logits[1])
    local['full_state'] = {'all_robots_positions': np.full((4, 3), np.nan)}
    local['evidence_audit'] = {'occupancy': np.ones((42, 42))}
    np.testing.assert_array_equal(ground_observation(episode, cand, peers, rows)['ground_inputs'],
                                  obs['ground_inputs'])
    single, value = model(*tensors(obs))
    multi_inputs = [torch.stack(items) for items in zip(tensors(obs), tensors(obs2))]
    batch, values = model(*multi_inputs)
    torch.testing.assert_close(single.logits, batch.logits[0])
    torch.testing.assert_close(value, values[0])
    assert torch.isfinite(values).all() and batch.probs[:, 2:].sum() == 0.
    assert not torch.allclose(values[0], values[1])
    assert not torch.isclose(single.logits[0], single.logits[1], atol=1e-8, rtol=0.)
    model2 = GroundSlowActorCritic(52, 16)
    model2.load_state_dict(deepcopy(model.state_dict()))
    torch.testing.assert_close(model2(*tensors(obs))[0].logits, single.logits)
    transitions = []
    for ob in (obs, obs2):
        action, logp, old_value = model.act(ob, deterministic=True)
        transitions.append(dict(ob, action=action, logp=logp, value=old_value, next_value=0.,
                                duration=2, terminated=True, truncated=False, reward=float(len(transitions))))
    stats = update(model, torch.optim.Adam(model.parameters(), lr=1e-5),
                   [[item] for item in transitions], epochs=1, microbatch_size=2)
    assert stats['accepted_epochs'] == 1 and stats['behaviour_logp_error'] < 1.e-4
    check_task_inputs(episode, cand)
    roads.update(graph_positions=[], graph_confidence=[], graph_source_stamps=[],
                 graph_edge_index=np.empty((2, 0), int), graph_observation_counts=[])
    empty = ground_observation(episode, cand, [], [])
    empty_dist, empty_value = model(*tensors(empty))
    assert torch.isfinite(empty_dist.probs).all() and torch.isfinite(empty_value)
    check_episode_tasks(scripts.parents[2])


def check_task_inputs(episode, candidates):
    import numpy as np
    import torch
    from world_model import train_slow_policy as trainer
    from ground.ground_policy import ground_observation
    from world_model.slow_policy import tensors

    ground_v1 = 'vtemp2_ground_gat_relay_v1'
    assert trainer.contract(trainer.DEFAULTS, {}, ground_v1)['ground_input_width'] == 2203
    assert trainer.contract(trainer.DEFAULTS, {})['ground_input_width'] == 2209
    assert 'ground_input_width' not in trainer.contract(trainer.DEFAULTS, {}, trainer.LEGACY_SCHEMA)
    schema_models = [trainer.slow_model(schema, 16)
                     for schema in (trainer.LEGACY_SCHEMA, ground_v1, trainer.SCHEMA)]
    own = episode.env.obs[episode.env.ground]['self']
    own['position'] = [4., 7., 0.]
    task = dict(ground_target=[11., 21.], survey_target=[32., 0.], survey_issued_at=3.,
                survey_evidence_received=False)
    old = ground_observation(episode, candidates, [], [])
    active = ground_observation(episode, candidates, [], [], task_context=task)
    empty = ground_observation(episode, candidates, [], [], task_context=dict(
        ground_target=None, survey_target=None, survey_issued_at=None, survey_evidence_received=False))
    np.testing.assert_allclose(active['ground_inputs'][7:9], np.array([7., 14.]) / 350.)
    np.testing.assert_allclose(active['ground_inputs'][-6:], [1., 1., .08, -.02, np.log1p(5.), 0.])
    np.testing.assert_array_equal(empty['ground_inputs'][-6:], np.zeros(6))
    np.testing.assert_array_equal(empty['ground_inputs'][7:9], np.zeros(2))
    np.testing.assert_allclose(old['ground_inputs'][7:9], np.array([1., -7.]) / 350.)
    received = ground_observation(episode, candidates, [], [], task_context=dict(
        task, survey_evidence_received=True))
    np.testing.assert_array_equal(active['ground_inputs'][:-1], received['ground_inputs'][:-1])
    assert received['ground_inputs'][-1] == 1
    for model, obs in zip(schema_models, (candidates, old, active)):
        distribution, value = model(*tensors(obs))
        assert torch.isfinite(value) and distribution.probs[2:].sum() == 0
    assert schema_models[-1].value(active) != schema_models[-1].value(empty)
    own['position'] = [0., 0., 0.]


def check_episode_tasks(repo):
    import numpy as np
    from simulator.ground_evidence import GroundPolicyEpisode
    from world_model.scene_manifest import make_scene

    env = GroundPolicyEpisode(repo, make_scene(1, (150., 150.), 800., 'open'),
                              fast_seed=15, rollout_limit=1, task_context=True)
    try:
        first = env.observation()
        np.testing.assert_array_equal(first['ground_inputs'][-6:], np.zeros(6))
        action = next(i for i, kind in enumerate(first['kinds']) if kind == 'ring' and first['mask'][i])
        env.execute(first, action)
        shared = env.planning_task_context()
        np.testing.assert_array_equal(shared['ground_target'], first['targets'][action])
        np.testing.assert_array_equal(shared['survey_target'], first['targets'][action])
        target = first['targets'][action] + [30., 50.]
        env.record_survey_dispatch(target, env.env.env.time)
        next_task = env.planning_task_context()
        np.testing.assert_array_equal(next_task['ground_target'], shared['ground_target'])
        np.testing.assert_array_equal(next_task['survey_target'], target)
        assert not next_task['survey_evidence_received']
        env.record_survey_dispatch(shared['survey_target'], shared['survey_issued_at'])
        np.testing.assert_array_equal(env.planning_task_context()['survey_target'], target)
    finally:
        env.close()


if __name__ == '__main__':
    if sys.argv[1:] == ['isolated']:
        check_ground_policy()
    else:
        unittest.main()
