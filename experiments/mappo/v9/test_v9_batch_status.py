"""A stopped worker must not be reported as running from a stale heartbeat."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('v9_batch',Path(__file__).with_name('validate_v9_long_batch.py'))
batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)


class BatchStatusTests(unittest.TestCase):
    def test_missing_process_changes_stale_running_to_interrupted(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            batch.write_json(root/'processes.json',dict(supervisor_pid=999999999,
                children=[dict(case='a',pid=999999998)]))
            batch.write_json(root/'batch_summary.json',dict(status='running',runs=[dict(case='a',status='running')]))
            result=batch.live_status(root)
            self.assertEqual(result['status'],'interrupted')
            self.assertEqual(result['runs'][0]['status'],'interrupted')
            self.assertFalse(result['supervisor_alive'])
            self.assertEqual(json.loads((root/'batch_summary.json').read_text())['status'],'running')

    def test_reused_pid_is_rejected_and_completed_result_is_reconciled(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'a').mkdir()
            batch.write_json(root/'processes.json',dict(supervisor_pid=os.getpid(),supervisor_started='wrong',
                children=[dict(case='a',pid=os.getpid(),started='wrong')]))
            batch.write_json(root/'batch_summary.json',dict(status='running',runs=[dict(case='a',status='running')]))
            batch.write_json(root/'a/a.result.json',dict(termination='vehicle_arrival',
                summary=dict(episode_steps=100,ground_goal_distance=1.5)))
            result=batch.live_status(root)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['arrived'],1)
            self.assertFalse(result['runs'][0]['process_alive'])


if __name__=='__main__':unittest.main()
