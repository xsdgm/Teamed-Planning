import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from vtemp5.circle_mastery import assess


class CircleEvidenceFixture:
    def write(self, name, data):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf-8')

    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.checkpoint = self.folder / 'checkpoints' / 'heading-0006.pt'
        self.checkpoint.parent.mkdir()
        self.checkpoint.write_bytes(b'final actor')
        digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        plans = []
        self.rows = []
        for i in range(6):
            case = dict(case_id=f'circle-{i}', map_sha256=hashlib.sha256(str(i).encode()).hexdigest())
            plans.append(dict(case=case, case_id=case['case_id'], fleet_seed=i, terrain='large_circle'))
            reached_goal = i < 4
            self.rows.append(dict(case_id=case['case_id'], geometry_sha256=case['map_sha256'], fast_seed=i,
                                  checkpoint_sha256=digest, phase='paired_evaluation', mode='evaluate',
                                  collector_limit_s=None, circle_metrics=dict(reached_far_side=True),
                                  end_reason='arrival' if reached_goal else 'time_budget', goal_reached=reached_goal,
                                  remaining_m=1.99 if reached_goal else 15., collisions=0))
        self.manifest = dict(updates=6, curriculum='circle', train=[], evaluation=plans, arrival_radius_m=2.)
        self.progress = dict(phase='complete', episodes=6, completed_updates=6, checkpoint=str(self.checkpoint))
        self.write('manifest.json', self.manifest)
        self.write('progress.json', self.progress)
        self.commit_results()

    def commit_results(self):
        summaries = []
        for i, row in enumerate(self.rows):
            self.write(f'evaluation/episode-{i:02d}/episode.json', row)
            summaries.append({k: row[k] for k in ('case_id', 'fast_seed', 'checkpoint_sha256', 'end_reason',
                                                   'goal_reached', 'remaining_m', 'circle_metrics')})
        self.write('evaluation.json', dict(checkpoint_sha256=self.rows[0]['checkpoint_sha256'], trained=summaries))


class CircleMasteryTests(CircleEvidenceFixture, unittest.TestCase):
    def test_four_arrivals_with_detours_pass(self):
        result = assess(self.folder)
        self.assertTrue(result['passed'])
        self.assertEqual((result['successful_cases'], result['collision_count'], result['total_cases']), (4, 0, 6))
        self.assertEqual(result['checkpoint'], str(self.checkpoint))

    def test_detour_and_arrival_must_be_in_the_same_case(self):
        self.rows[0]['circle_metrics']['reached_far_side'] = False
        self.commit_results()
        self.assertEqual(assess(self.folder)['successful_cases'], 3)
        self.assertFalse(assess(self.folder)['passed'])

    def test_arrival_radius_is_checked(self):
        self.rows[0]['remaining_m'] = 2.01
        self.commit_results()
        self.assertFalse(assess(self.folder)['passed'])

    def test_collision_in_a_failed_case_blocks_progression(self):
        self.rows[-1]['collisions'] = 1
        self.commit_results()
        result = assess(self.folder)
        self.assertEqual((result['successful_cases'], result['collision_count']), (4, 1))
        self.assertFalse(result['passed'])

    def test_missing_stale_or_truncated_episode_is_invalid(self):
        row = self.rows[0]
        for key, value in [('phase', 'baseline'), ('fast_seed', 91), ('checkpoint_sha256', 'other'),
                           ('geometry_sha256', 'other'), ('collector_limit_s', 600), ('collisions', False)]:
            with self.subTest(key=key):
                self.write('evaluation/episode-00/episode.json', {**row, key: value})
                with self.assertRaises(ValueError):
                    assess(self.folder)
        (self.folder / 'evaluation/episode-00/episode.json').unlink()
        with self.assertRaises(ValueError):
            assess(self.folder)

    def test_all_maps_must_be_held_out_and_unique(self):
        maps = self.manifest['evaluation']
        self.write('manifest.json', {**self.manifest, 'train': [maps[0]]})
        with self.assertRaises(ValueError):
            assess(self.folder)
        self.write('manifest.json', {**self.manifest, 'evaluation': [maps[0]] * 6})
        with self.assertRaises(ValueError):
            assess(self.folder)

    def test_run_must_be_complete_and_checkpoint_hash_must_match(self):
        self.write('progress.json', {**self.progress, 'phase': 'paired_evaluation'})
        with self.assertRaises(ValueError):
            assess(self.folder)
        self.write('progress.json', self.progress)
        self.checkpoint.write_bytes(b'unrelated actor')
        with self.assertRaises(ValueError):
            assess(self.folder)
