"""Exercise synchronous batch scheduling and transactional optimizer recovery on CPU."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from vtemp4 import train_vtemp4_formal as formal


def cases():
    return [dict(case_id=f'{task}-{i}', navigation_task=task, terrain='wildscenes_raster',
                 map_sha256=f'{task}-{i}', terrain_family_id=f'family-{i}', seed=i)
            for task in formal.STRATA for i in range(6)]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(2, 3)


class FormalTrainingTest(unittest.TestCase):
    def test_sampling_plan(self):
        maps = cases()
        plan = formal.sample_plan(maps, 50, 16, 43)
        self.assertEqual(plan, formal.sample_plan(maps, 50, 16, 43))
        self.assertEqual(len(set(r['fast_seed'] for batch in plan for r in batch)), 800)
        for batch in plan:
            self.assertEqual(len({r['case_id'] for r in batch}), 16)
            self.assertEqual(formal.stratum_counts(batch), dict.fromkeys(formal.STRATA, 4))
        draws = [row['case_id'] for batch in plan for row in batch
                 if row['navigation_task'] == formal.STRATA[0]]
        for first in range(0, len(draws), 6):
            deck = draws[first:first + 6]
            self.assertEqual(len(deck), len(set(deck)))

    def test_single_writer(self):
        with tempfile.TemporaryDirectory() as tmp, formal.exclusive_run(Path(tmp)):
            with self.assertRaisesRegex(RuntimeError, 'Another training process'), formal.exclusive_run(Path(tmp)):
                self.fail('Second writer acquired the directory lease')

    def test_restart_matches_uninterrupted_model_adam_rng_and_schedule(self):
        maps = cases()
        interrupt = True
        plan = formal.sample_plan(maps, 2, 4, 43)

        def collect(repo, checkpoint, sha, case, seed, location, mode):
            if interrupt and 'update-0002' in location.parts:
                raise RuntimeError('injected interruption')
            return dict(case_id=case['case_id'], navigation_task=case['navigation_task'],
                        geometry_sha256=case['map_sha256'], fast_seed=seed, end_reason='time_budget',
                        decisions=1, turn_count=1, left_turns=1, right_turns=0, remaining_m=0.,
                        goal_reached=False, deterministic_turns_on_sampled_observations=0)

        def optimize(model, optimizer, rollouts):
            optimizer.zero_grad()
            target = torch.tensor(random.random() + np.random.rand()) + torch.rand(())
            (model.actor.weight.sum() - target).square().backward()
            optimizer.step()
            return dict(transitions=len(rollouts))

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch.multiple('simulator.vtemp4_parallel', episode_pool=ThreadPoolExecutor,
                                               collect_episode=collect, load_rollout=lambda path: [1]))
            stack.enter_context(patch.object(formal, 'load_curriculum', return_value=({'ppo_train': maps}, {})))
            stack.enter_context(patch.object(formal, 'build_manifest', return_value={'plan': plan}))
            stack.enter_context(patch.object(formal, 'runtime'))
            stack.enter_context(patch.object(formal, 'HeadingPolicy', TinyModel))
            stack.enter_context(patch.object(formal, 'update_policy', optimize))
            args = SimpleNamespace(output=Path(tmp) / 'interrupted', repo=Path(tmp), learning_rate=.0003,
                                   resume=False, seed=43, updates=2, workers=2, episodes_per_update=4,
                                   km_data=Path(tmp) / 'km', field_data=Path(tmp) / 'lakes')
            args.output.mkdir()
            with self.assertRaisesRegex(RuntimeError, 'injected interruption'):
                formal.run(args)
            self.assertTrue(formal.checkpoint_path(args.output, 1).is_file())
            self.assertFalse(formal.checkpoint_path(args.output, 2).exists())
            interrupt = False
            args.resume = True
            formal.run(args)
            resumed = formal.file_sha256(formal.checkpoint_path(args.output, 2))
            args.output = Path(tmp) / 'reference'
            args.output.mkdir()
            args.resume = False
            formal.run(args)
            self.assertEqual(resumed, formal.file_sha256(formal.checkpoint_path(args.output, 2)))


if __name__ == '__main__':
    unittest.main()
