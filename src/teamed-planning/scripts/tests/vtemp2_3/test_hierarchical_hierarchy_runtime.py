"""The real frozen UAV actor executes repeated decisions from the new layer."""
from dataclasses import replace
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from vtemp2_3.hierarchical_runtime import HierarchicalEpisode,HierarchyConfig
from vtemp2_3.navigation_scenes import make_scene


class HierarchyRuntimeContracts(unittest.TestCase):
    def test_repeated_targets_keep_mission_and_frozen_actor(self):
        repo=Path(__file__).resolve().parents[5]
        case=make_scene(5001,(150.,150.),1000.,'open')
        config=replace(HierarchyConfig(),option_steps=8,hold_steps=4,minimum_steps=2,evidence_interval=4,
                       adaptive_options=False)
        with contextlib.redirect_stdout(io.StringIO()):
            episode=HierarchicalEpisode(repo,case,config,fast_seed=17,rollout_limit=20)
            try:
                parameters={n:p.clone() for n,p in episode.fast_model.named_parameters()}
                goal=episode.goal.copy()
                first=episode.observation()
                forward=next(i for i in np.flatnonzero(first['mask']) if first['kinds'][i]=='final')
                _,duration,terminal,truncated,record=episode.execute(first,forward)
                self.assertEqual(duration,8);self.assertFalse(terminal or truncated)
                self.assertTrue(record['target_heard'])
                second=episode.observation()
                side=next(i for i in np.flatnonzero(second['mask'])
                          if second['kinds'][i]=='ring' and abs(second['features'][i,1])>.4)
                episode.execute(second,side)
                third=episode.observation();hold=third['kinds'].index('hold')
                before=np.asarray(episode.env.obs[episode.env.ground]['self']['position']).copy()
                _,duration,terminal,truncated,record=episode.execute(third,hold)
                self.assertFalse(terminal);self.assertTrue(truncated)
                np.testing.assert_array_equal(before,episode.env.obs[episode.env.ground]['self']['position'])
                self.assertEqual(len(episode.decisions),3)
                np.testing.assert_array_equal(episode.env.goal,goal)
                np.testing.assert_array_equal(episode.manager.target,third['targets'][hold])
                for n,p in episode.fast_model.named_parameters():self.assertTrue(episode.torch.equal(p,parameters[n]))
                self.assertEqual(episode.summary()['termination'],'rollout_limit')
            finally:episode.close()

    def test_startup_hold_locks_motion_only_until_simulation_release(self):
        repo = Path(__file__).resolve().parents[5]
        case = make_scene(5001, (150., 150.), 1000., 'open')
        for seconds in (0, 30):
            config = replace(HierarchyConfig(), ground_start_hold_seconds=seconds)
            with self.subTest(seconds=seconds), contextlib.redirect_stdout(io.StringIO()):
                episode = HierarchicalEpisode(repo, case, config, fast_seed=17, rollout_limit=40)
                try:
                    start = episode.startup_position.copy()
                    initial_uav = np.asarray(episode.env.obs[episode.env.ids[0]]['self']['position']).copy()
                    while not episode.finished:
                        obs = episode.observation()
                        episode.execute(obs, obs['kinds'].index('final'))
                    summary = episode.summary()
                    self.assertEqual(summary['startup_hold_control_attempts'], seconds)
                    self.assertEqual(summary['startup_hold_max_displacement_m'], 0.)
                    self.assertGreater(summary['first_ground_motion_time_s'], seconds)
                    if seconds:
                        poses = np.asarray(episode.trajectory[1:seconds+1])
                        np.testing.assert_array_equal(poses, np.broadcast_to(start, poses.shape))
                        self.assertTrue(any(r['target_heard'] for r in episode.decisions))
                        self.assertFalse(np.allclose(episode.env.obs[episode.env.ids[0]]['self']['position'], initial_uav))
                    else:
                        self.assertLess(summary['first_ground_motion_time_s'], 30)
                finally:
                    episode.close()

    def test_initial_hold_broadcasts_hold_and_does_not_drive_old_target(self):
        repo=Path(__file__).resolve().parents[5]
        case=make_scene(5011,(150.,150.),1000.,'open')
        config=replace(HierarchyConfig(),option_steps=4,hold_steps=4,minimum_steps=2)
        with contextlib.redirect_stdout(io.StringIO()):
            episode=HierarchicalEpisode(repo,case,config,fast_seed=3,rollout_limit=4)
            try:
                obs=episode.observation();hold=obs['kinds'].index('hold')
                episode.execute(obs,hold)
                np.testing.assert_allclose(episode.trajectory[-1],case['start'])
                np.testing.assert_allclose(episode.env.ground_action['broadcast']['target'],case['start'])
                self.assertTrue(episode.truncated);self.assertFalse(episode.terminated)
            finally:episode.close()

    def test_final_goal_does_not_use_subgoal_arrival_radius(self):
        repo=Path(__file__).resolve().parents[5]
        case=make_scene(5012,(20.,20.),1000.,'open')
        config=replace(HierarchyConfig(),option_steps=4,hold_steps=4,minimum_steps=2,adaptive_options=False)
        with contextlib.redirect_stdout(io.StringIO()):
            episode=HierarchicalEpisode(repo,case,config,rollout_limit=8)
            try:
                obs=episode.observation();action=obs['kinds'].index('final')
                _,duration,terminal,truncated,record=episode.execute(obs,action)
                self.assertEqual(duration,4);self.assertFalse(terminal or truncated)
                self.assertGreater(record['remaining_m'],2.)
                self.assertNotEqual(record['reason'],'arrival')
            finally:episode.close()

    def test_mission_budget_is_terminal_not_collector_truncation(self):
        repo=Path(__file__).resolve().parents[5]
        case=make_scene(5013,(150.,150.),1000.,'open')
        config=replace(HierarchyConfig(),option_steps=4,hold_steps=4,minimum_steps=2,
                       min_mission_budget=2,budget_extra=1,budget_multiplier=.001)
        with contextlib.redirect_stdout(io.StringIO()):
            episode=HierarchicalEpisode(repo,case,config,rollout_limit=8)
            try:
                obs=episode.observation()
                reward,duration,terminal,truncated,record=episode.execute(obs,obs['kinds'].index('hold'))
                self.assertEqual(duration,2);self.assertTrue(terminal);self.assertFalse(truncated)
                self.assertEqual(record['reason'],'mission_budget');self.assertLess(reward,-19.)
            finally:episode.close()

    def test_safety_failure_before_physics_is_a_positive_duration_terminal(self):
        repo=Path(__file__).resolve().parents[5]
        case=make_scene(5014,(150.,150.),1000.,'open')
        with contextlib.redirect_stdout(io.StringIO()):
            episode=HierarchicalEpisode(repo,case,rollout_limit=8)
            try:
                obs=episode.observation()
                with patch.object(episode.env,'step',side_effect=episode.me.SafetyTermination('test',[])):
                    reward,duration,terminal,truncated,record=episode.execute(obs,obs['kinds'].index('hold'))
                self.assertEqual(duration,1);self.assertEqual(episode.frame,0)
                self.assertTrue(terminal);self.assertFalse(truncated)
                self.assertEqual(record['reason'],'safety_failure');self.assertLess(reward,-99.)
            finally:episode.close()

    def test_actual_mission_arrival_is_terminal_with_success_reward(self):
        repo=Path(__file__).resolve().parents[5]
        case=make_scene(5017,(4.,4.),1000.,'open')
        with contextlib.redirect_stdout(io.StringIO()):
            episode=HierarchicalEpisode(repo,case,rollout_limit=16)
            try:
                obs=episode.observation()
                reward,_,terminal,truncated,record=episode.execute(obs,obs['kinds'].index('final'))
                self.assertTrue(terminal);self.assertFalse(truncated)
                self.assertEqual(record['reason'],'mission_arrival')
                self.assertLessEqual(record['remaining_m'],2.);self.assertGreater(reward,98.)
            finally:episode.close()

    def test_progress_window_ends_stalled_option_without_fake_obstacle(self):
        repo=Path(__file__).resolve().parents[5]
        case=make_scene(5020,(100.,100.),1000.,'open')
        config=replace(HierarchyConfig(),option_steps=4,stall_steps=12,evidence_interval=4)
        with contextlib.redirect_stdout(io.StringIO()):
            episode=HierarchicalEpisode(repo,case,config,rollout_limit=40)
            try:
                obs=episode.observation();before=episode.belief()['grid'].cells.copy()
                def stationary_step(frame):
                    episode.env.trace.append(dict(ground_collisions=0))
                with patch.object(episode.env,'step',side_effect=stationary_step):
                    _,duration,terminal,truncated,record=episode.execute(obs,obs['kinds'].index('final'))
                self.assertEqual(duration,12);self.assertFalse(terminal or truncated)
                self.assertEqual(record['reason'],'control_stall')
                self.assertEqual(record['attempt_records_added'],1)
                np.testing.assert_array_equal(episode.belief()['grid'].cells,before)
            finally:episode.close()

    def test_adaptive_option_does_not_reselect_at_short_collection_window(self):
        repo=Path(__file__).resolve().parents[5]
        case=make_scene(5021,(100.,100.),1000.,'open')
        config=replace(HierarchyConfig(),option_steps=4,evidence_interval=4)
        with contextlib.redirect_stdout(io.StringIO()):
            episode=HierarchicalEpisode(repo,case,config,rollout_limit=20)
            try:
                obs=episode.observation()
                with patch.object(episode.env,'step',side_effect=lambda frame:
                                  episode.env.trace.append(dict(ground_collisions=0))):
                    _,duration,terminal,truncated,record=episode.execute(obs,obs['kinds'].index('final'))
                self.assertEqual(duration,20);self.assertFalse(terminal);self.assertTrue(truncated)
                self.assertEqual(record['reason'],'rollout_limit')
                self.assertGreater(record['planned_duration_limit'],20)
            finally:episode.close()

    def test_detour_motion_is_not_mislabelled_as_control_stall(self):
        repo=Path(__file__).resolve().parents[5]
        case=make_scene(5022,(100.,100.),1000.,'open')
        config=replace(HierarchyConfig(),option_steps=4,stall_steps=12,evidence_interval=4)
        with contextlib.redirect_stdout(io.StringIO()):
            episode=HierarchicalEpisode(repo,case,config,rollout_limit=20)
            try:
                obs=episode.observation()
                away=(np.asarray(case['start'])-case['goal'])/case['direct_distance_m']
                def detour_step(frame):
                    state=episode.env.obs[episode.env.ground]['self']
                    state['position'][:2]+=away
                    episode.env.trace.append(dict(ground_collisions=0))
                with patch.object(episode.env,'step',side_effect=detour_step):
                    _,duration,_,truncated,record=episode.execute(obs,obs['kinds'].index('final'))
                self.assertEqual(duration,20);self.assertTrue(truncated)
                self.assertGreater(record['remaining_m'],100.)
                self.assertEqual(record['reason'],'rollout_limit')
            finally:episode.close()


if __name__=='__main__':unittest.main(verbosity=2)
