"""Real v9 snapshot isolation, packet handover and paired-rollout checks."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from world_model.frozen_navigation import runtime,received_belief,branch_rollout,scene,clone_experiment


class FrozenRuntimeContracts(unittest.TestCase):
    def test_bridge_checks_read_branch_evidence_not_source_snapshot(self):
        me,torch,model,options=runtime(Path(__file__).resolve().parents[5])
        task=scene(0,0)
        with contextlib.redirect_stdout(io.StringIO()),torch.inference_mode():
            base=me.Experiment(task['scene_seed'],model,stochastic=True,
                               **dict(options,start=task['start'],scene=(task['goal'],[],1)))
            branch=clone_experiment(base,model,me)
            try:
                base.ledger.seen[:]=True;base.ledger.log_odds[:]=-3.
                branch.ledger.seen[:]=False;branch.ledger.log_odds[:]=0.
                p=np.asarray(task['start'])
                self.assertTrue(base.ground_controller.graph_fusion.bridge_check(p,p+[1.,0.]))
                self.assertFalse(branch.ground_controller.graph_fusion.bridge_check(p,p+[1.,0.]))
            finally:
                branch.env.close();base.env.close()

    def test_paired_branches_are_reproducible_and_leave_source_untouched(self):
        repo=Path(__file__).resolve().parents[5]
        me,torch,model,options=runtime(repo)
        task=scene(0,0)
        torch.manual_seed(123)
        with contextlib.redirect_stdout(io.StringIO()),torch.inference_mode():
            env=me.Experiment(task['scene_seed'],model,stochastic=True,
                             **dict(options,start=task['start'],scene=(task['goal'],[],1)))
            try:
                for frame in range(4):env.step(frame)
                belief=received_belief(env)
                before=(env.env.time,env.ledger.seen.copy(),env.subgoals.target.copy(),len(env.trace))
                goal=env.goal.copy()
                parameters={n:p.clone() for n,p in model.named_parameters()}
                q=belief['position']+np.array([133.,0.])
                first=branch_rollout(env,model,q,12345,4,8,me,torch)
                second=branch_rollout(env,model,q,12345,4,8,me,torch)
                self.assertEqual(first,second)
                self.assertTrue(first['target_heard'])
                self.assertEqual(env.env.time,before[0])
                np.testing.assert_array_equal(env.ledger.seen,before[1])
                np.testing.assert_array_equal(env.subgoals.target,before[2])
                self.assertEqual(len(env.trace),before[3])
                np.testing.assert_array_equal(env.goal,goal)
                for n,p in model.named_parameters():self.assertTrue(torch.equal(p,parameters[n]))
                for name,module in sys.modules.items():
                    if name.split('.')[0] in ('common','drone','ground','simulator') and getattr(module,'__file__',None):
                        self.assertIn('mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146/runtime_sources',module.__file__)
            finally:env.env.close()


if __name__=='__main__':unittest.main(verbosity=2)
