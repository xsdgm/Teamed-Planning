"""Collect native VTemp4 episodes in spawned CPU workers with restartable output."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import fcntl
import gc
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time

import torch

from world_model.frozen_navigation import CHECKPOINT_SHA, runtime
from world_model.hierarchical_sources import atomic_json, file_sha256
from world_model.train_slow_policy import atomic_checkpoint
from simulator.train_vtemp4 import AlwaysHold, FORMAT, deterministic_turns_on_rollout, summarize
from simulator.vtemp4_curriculum import geometry_hash, runtime_case
from ground.vtemp4_policy import HeadingPolicy
from simulator.vtemp4_ring_contract import (
    PERIODIC_ACTION_CONTRACTS, RING_CONTINUOUS_HISTORY_FORMAT, RING_POLICY_FORMATS, RING_RADIUS_M,
    RING_WAYPOINT_FORMAT, periodic_policy_types, ring_action_contract,
)
from simulator.vtemp4_runtime import HeadingFleetEpisode
from ground.vtemp4_waypoint_policy import WAYPOINT_FORMAT, WaypointPolicy, RingWaypointPolicy


def _init_worker(parent_pid):
    if sys.platform == 'linux':
        # After a killed coordinator, release the episode lock and native memory
        # before its replacement starts a new pool. Check for the setup race.
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), 'prctl(PR_SET_PDEATHSIG)')
        if os.getppid() != parent_pid:
            os._exit(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def episode_pool(workers):
    """Use one thread per simulator, including libraries loaded during spawn."""
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                               initializer=_init_worker, initargs=(os.getpid(),))


def _row(record):
    periodic = record.get('action_contract') in PERIODIC_ACTION_CONTRACTS
    if periodic or 'waypoint_reviews' in record:
        keys = ('case_id', 'navigation_task', 'terrain_family_id', 'geometry_sha256', 'seed', 'fast_seed',
                'direct_distance_m', 'end_reason', 'remaining_m', 'goal_reached', 'sim_time_s', 'reward',
                'travelled_m')
        row = {key: record[key] for key in keys}
        if periodic:
            return dict(row, decisions=len(record['heading_reviews']), waypoint_designations=len(record['shared_markers']))
        return dict(row, decisions=len(record['waypoint_reviews']), waypoint_replacements=record['waypoint_replacements'])
    row = summarize(record)
    if record['mode'] == 'train' and 'deterministic_turns_on_sampled_observations' in record:
        row['deterministic_turns_on_sampled_observations'] = record['deterministic_turns_on_sampled_observations']
    return row


def load_rollout(episode_dir):
    """Read a committed local rollout, checking the bytes against its episode record."""
    folder = Path(episode_dir)
    record = json.loads((folder / 'episode.json').read_text())
    if record['mode'] != 'train':
        raise ValueError('Only training episodes contain PPO transitions')
    blob = (folder / 'transitions.pt').read_bytes()
    if hashlib.sha256(blob).hexdigest() != record['transitions_sha256']:
        raise ValueError(f'Rollout digest differs from the committed episode at {folder}')
    # NumPy observations retain their original dtypes and layout in the local cache.
    transitions = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
    reviews = record['waypoint_reviews'] if 'waypoint_reviews' in record else record['heading_reviews']
    if not transitions or len(transitions) != len(reviews):
        raise ValueError(f'Incomplete training transitions at {folder}')
    return transitions


def collect_episode(repo, checkpoint_path, checkpoint_sha256, case, fast_seed, episode_dir,
                    mode='train', collector_seconds=None):
    """Run or reuse one mission. episode.json commits its record and optional PPO data."""
    if mode not in ('train', 'evaluate', 'always_hold'):
        raise ValueError(f'Unexpected episode mode {mode}')
    folder = Path(episode_dir)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.episode.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        blob = Path(checkpoint_path).read_bytes()
        if hashlib.sha256(blob).hexdigest() != checkpoint_sha256:
            raise ValueError(f'Heading checkpoint digest differs at {checkpoint_path}')
        checkpoint = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
        if checkpoint['v9_sha256'] != CHECKPOINT_SHA:
            raise ValueError('Checkpoint uses different frozen V9 weights')
        fmt = checkpoint['format']
        ring = fmt in RING_POLICY_FORMATS.values()
        continuous_history = fmt == RING_CONTINUOUS_HISTORY_FORMAT
        periodic = ring_action_contract(fmt) in PERIODIC_ACTION_CONTRACTS
        waypoint = fmt in (WAYPOINT_FORMAT, RING_WAYPOINT_FORMAT)
        if not ring and fmt not in (FORMAT, WAYPOINT_FORMAT):
            raise ValueError('Unknown navigation checkpoint format')
        if ring and (checkpoint.get('action_contract') != ring_action_contract(fmt)
                     or checkpoint['clock']['lookahead_m'] != RING_RADIUS_M):
            raise ValueError('Checkpoint differs from the fixed 200 m guidance contract')
        if not ring and checkpoint.get('action_contract') is not None:
            raise ValueError('Legacy checkpoint has an unexpected action contract')
        if waypoint and (mode == 'always_hold' or checkpoint.get('policy_variant') not in ('waypoint', 'waypoint_history')):
            raise ValueError('Invalid waypoint checkpoint or collection mode')
        if ring and RING_POLICY_FORMATS.get(checkpoint.get('policy_variant')) != fmt:
            raise ValueError('Navigation checkpoint format differs from its policy variant')
        if periodic and (checkpoint['clock']['review_interval_s'] != 15 or mode == 'always_hold'):
            raise ValueError('Periodic angle selection requires a fresh designation every 15 seconds')
        review_key = 'waypoint_reviews' if waypoint else 'heading_reviews'
        change_key = 'designations' if periodic else ('replacements' if waypoint else 'turns')
        native_case = runtime_case(case)
        # Geometry file locations may change when restoring on another host.
        # Their content digests already pin both rasters and circle layouts.
        settings = {k: v for k, v in native_case.items() if k not in ('map_path', 'circles_path')}
        if 'circles_path' in native_case:
            settings.pop('circles', None)
        identity = dict(mode=mode, checkpoint_sha256=checkpoint_sha256, v9_sha256=CHECKPOINT_SHA,
                        case_id=case['case_id'], case_settings=settings, seed=case['seed'], fast_seed=fast_seed,
                        clock=checkpoint['clock'], navigation_task=case['navigation_task'],
                        terrain_family_id=case['terrain_family_id'], geometry_sha256=geometry_hash(case),
                        collector_seconds=collector_seconds)
        if ring:
            identity['action_contract'] = ring_action_contract(fmt)
        result = folder / 'episode.json'
        cache = folder / 'transitions.pt'
        status_path = folder / 'progress.json'
        if result.is_file():
            record = json.loads(result.read_text())
            if any(record.get(k) != value for k, value in identity.items()):
                raise ValueError(f'Cached episode has a different policy, seed or mission at {folder}')
            if mode != 'train' or (cache.is_file() and file_sha256(cache) == record['transitions_sha256']):
                # The coordinator may have stopped between the two terminal writes.
                saved = json.loads(status_path.read_text()) if status_path.is_file() else {}
                if saved.get('status') != record['end_reason']:
                    saved.update(case_id=record['case_id'], status=record['end_reason'], mode=mode,
                                 sim_s=record['sim_time_s'], budget=record['budget_s'],
                                 decisions=len(record[review_key]), elapsed_wall_s=record['elapsed_wall_s'])
                    saved[change_key] = (len(record['shared_markers']) if periodic else
                                         record['waypoint_replacements' if waypoint else 'turn_count'])
                    atomic_json(status_path, saved)
                return _row(record)
        budget = math.ceil(3 * case['direct_distance_m'] + 120)
        start = time.monotonic()
        progress = dict(case_id=case['case_id'], fast_seed=fast_seed, mode=mode,
                        navigation_task=case['navigation_task'], checkpoint_sha256=checkpoint_sha256,
                        budget=budget, pid=os.getpid(), status='initializing', sim_s=0, decisions=0)
        progress[change_key] = 0
        atomic_json(status_path, progress)
        try:
            # Native sensor/controller object graphs are cyclic. Reclaim any
            # prior mission before allocating a fresh full-resolution raster.
            gc.collect()
            runtime(repo)
            if mode == 'always_hold':
                model = AlwaysHold()
            else:
                if waypoint:
                    policy_type = RingWaypointPolicy if ring else WaypointPolicy
                    model = policy_type(use_history=checkpoint['policy_variant'] == 'waypoint_history')
                elif continuous_history:
                    from simulator.vtemp4_ring_history import ContinuousHistoryPolicy
                    model = ContinuousHistoryPolicy()
                elif periodic:
                    policy_type, periodic_episode_type = periodic_policy_types(fmt)
                    model = policy_type()
                else:
                    model = HeadingPolicy()
                model.eval().requires_grad_(False)
                model.load_state_dict(checkpoint['model'])
            rollout = [] if mode == 'train' else None
            with open(os.devnull, 'w') as sink, redirect_stdout(sink), torch.inference_mode():
                if waypoint:
                    from simulator.vtemp4_waypoint_runtime import RingWaypointFleetEpisode, WaypointFleetEpisode
                    episode_type = RingWaypointFleetEpisode if ring else WaypointFleetEpisode
                elif continuous_history:
                    from simulator.vtemp4_ring_history import ContinuousHistoryFleetEpisode
                    episode_type = ContinuousHistoryFleetEpisode
                elif periodic:
                    episode_type = periodic_episode_type
                elif ring:
                    from simulator.vtemp4_ring_heading import RingHeadingFleetEpisode
                    episode_type = RingHeadingFleetEpisode
                else:
                    episode_type = HeadingFleetEpisode
                ep = episode_type(repo, native_case, model, fast_seed=fast_seed, budget_s=budget,
                                  rollout=rollout, collector_limit_s=collector_seconds, **checkpoint['clock'])
                try:
                    reviews = getattr(ep, review_key)
                    progress.update(status='running', sim_s=ep.clock_s, decisions=len(reviews))
                    atomic_json(status_path, progress)
                    while not ep.finished:
                        ep.step()
                        if ep.clock_s % 60 == 0 or ep.finished:
                            if periodic:
                                progress[change_key] = len(ep.requests)
                            elif fmt == RING_WAYPOINT_FORMAT:
                                progress[change_key] = sum(r['changed'] and r['replace'] and r['waypoint_id'] != 0
                                                           for r in reviews)
                            elif waypoint:
                                progress[change_key] = sum(r['replace'] and r['changed'] for r in reviews[1:])
                            else:
                                progress[change_key] = sum(r['turn'] for r in reviews)
                            progress.update(sim_s=ep.clock_s, decisions=len(reviews),
                                            elapsed_wall_s=round(time.monotonic() - start, 1))
                            atomic_json(status_path, progress)
                    record = ep.fleet_summary()
                    record['trajectory_xy'] = ep.trajectory
                finally:
                    ep.close()
            if rollout is not None:
                if not rollout or len(rollout) != len(record[review_key]):
                    raise RuntimeError(f'Incomplete training decisions for {case["case_id"]}')
                if not waypoint and not periodic:
                    record['deterministic_turns_on_sampled_observations'] = deterministic_turns_on_rollout(model, rollout)
                atomic_checkpoint(cache, rollout)
                record['transitions_sha256'] = file_sha256(cache)
            record.update(identity, elapsed_wall_s=round(time.monotonic() - start, 1))
            atomic_json(result, record)
            progress.update(status=record['end_reason'], elapsed_wall_s=record['elapsed_wall_s'])
            atomic_json(status_path, progress)
            return _row(record)
        except Exception as exc:
            progress.update(status='failed', error=str(exc), elapsed_wall_s=round(time.monotonic() - start, 1))
            atomic_json(status_path, progress)
            raise
