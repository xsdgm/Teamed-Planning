"""Vtemp4 extensions coexist with the archived v9 core without mixing sources."""
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[5]
ARCHIVE = REPO / 'artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146/runtime_sources'


@unittest.skipUnless(ARCHIVE.is_dir(), 'Frozen v9 sources unavailable')
class FrozenRuntimeIsolationTests(unittest.TestCase):
    def run_isolated(self, script):
        result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(SCRIPTS), str(REPO)],
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_vtemp4_import_before_runtime_uses_archived_core(self):
        self.run_isolated('''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import vtemp4.train_vtemp4_formal
import vtemp2_3.ground_policy
import survey.survey_fleet_runtime
from vtemp2_3.frozen_navigation import runtime
runtime(Path(sys.argv[2]))
import vtemp4.vtemp4_stage_eval
archive = Path(sys.argv[2]) / 'artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146/runtime_sources'
for name in ('ground.ground_main', 'simulator.mappo_experiment', 'simulator.env', 'drone.drone_main'):
    assert Path(sys.modules[name].__file__).resolve().is_relative_to(archive.resolve()), name
assert 'ground.vtemp4_policy' in sys.modules
assert 'ground.ground_policy' in sys.modules
assert 'simulator.survey_fleet_runtime' in sys.modules
try:
    import shared.ground.observed_grid  # Exists in the live tree, absent from the archive.
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('Mutable non-vtemp4 module was loaded after runtime startup')
''')

    def test_mutable_core_loaded_before_runtime_is_rejected(self):
        self.run_isolated('''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import shared.ground.ground_main
from vtemp2_3.frozen_navigation import runtime
try:
    runtime(Path(sys.argv[2]))
except RuntimeError as error:
    assert 'Mutable runtime already imported' in str(error), error
else:
    raise AssertionError('Mutable core was accepted')
''')


if __name__ == '__main__':
    unittest.main()
