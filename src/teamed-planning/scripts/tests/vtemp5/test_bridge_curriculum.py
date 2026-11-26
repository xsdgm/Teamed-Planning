import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.vtemp5.test_circle_mastery import CircleEvidenceFixture
from vtemp5 import run_vtemp5_bridge_curriculum as curriculum


class BridgeCurriculumTests(CircleEvidenceFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.bridge = self.folder / 'next'
        self.logs = self.bridge / 'logs'
        trainer = self.logs / 'sources/vtemp5/run_vtemp5_fleet.py'
        trainer.parent.mkdir(parents=True)
        trainer.write_text(
            "import hashlib, json, sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "out = Path(args[args.index('--output') + 1])\n"
            "actor = Path(args[args.index('--init-checkpoint') + 1])\n"
            "with (out/'calls').open('a') as f: print(json.dumps(args), file=f)\n"
            "(out/'manifest.json').write_text(json.dumps({'actor_sha256': hashlib.sha256(actor.read_bytes()).hexdigest()}))\n"
            "(out/'progress.json').write_text(json.dumps({'phase': 'complete'}))\n", encoding='utf-8')
        self.manifest.update(repo=str(self.folder), world_model=str(self.folder/'world'), workers=16,
                             episodes_per_update=16, smoke_s=180, epochs=1, microbatch=8,
                             learning_rate=3e-6, seconds_per_m=6)
        self.write('manifest.json', self.manifest)
        digest = hashlib.sha256((self.folder / 'manifest.json').read_bytes()).hexdigest()
        self.args = SimpleNamespace(stage1=self.folder, stage2=self.bridge, stage1_manifest_sha256=digest,
                                    seed=120, eval_seed=130, baseline_cpus=[16, 17, 18, 19])

    def phase(self):
        return json.loads((self.logs/'curriculum_status.json').read_text())['phase']

    def run_supervisor(self):
        with patch.object(curriculum.os, 'sched_getaffinity', return_value=set(range(16))):
            curriculum.supervise(self.args)

    def test_wait_then_run_once_from_evaluated_checkpoint(self):
        self.write('progress.json', {**self.progress, 'phase': 'paired_evaluation'})

        def finish_evaluation(_seconds):
            self.assertEqual(self.phase(), 'waiting_for_circle_evaluation')
            self.assertFalse((self.bridge/'calls').exists())
            self.write('progress.json', self.progress)

        with patch.object(curriculum.time, 'sleep', side_effect=finish_evaluation):
            self.run_supervisor()
        self.run_supervisor()
        self.assertEqual(self.phase(), 'complete')
        calls = (self.bridge/'calls').read_text().splitlines()
        self.assertEqual(len(calls), 1)
        args = json.loads(calls[0])
        self.assertEqual(args[args.index('--init-checkpoint') + 1], str(self.checkpoint))
        self.assertEqual(args[args.index('--curriculum') + 1], 'bridge-lake')

    def test_failed_gate_stops_without_retry(self):
        self.rows[-1]['collisions'] = 1
        self.commit_results()
        self.run_supervisor()
        self.rows[-1]['collisions'] = 0
        self.commit_results()
        self.run_supervisor()
        self.assertEqual(self.phase(), 'stopped_at_circle')
        self.assertFalse((self.bridge/'calls').exists())

    def test_checkpoint_changed_before_gate(self):
        self.checkpoint.write_bytes(b'wrong checkpoint')
        with self.assertRaises(ValueError):
            self.run_supervisor()
        self.assertEqual(self.phase(), 'error')
        self.assertFalse((self.bridge/'calls').exists())
