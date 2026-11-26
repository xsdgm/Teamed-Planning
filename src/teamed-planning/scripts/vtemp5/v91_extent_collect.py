"""Collect time-aligned pilot observations from a frozen V9.1 closed-loop fleet.

Only air_cells (actually delivered UAV tiles), ground_cells (vehicle's own
occupancy), car pose/path and mission goal are available to the world model.
Instantaneous aircraft sensors and trajectories are recorded for visualization.
World rasters use [y,x] with origin (0,0): -1 free, +1 occupied, 0 unknown
or an exact tie. Each requested frame is one second after the initial state.
"""
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


def load_frozen(run_dir, checkpoint_path):
    """Bind this process to the saved runtime and one immutable episode checkpoint."""
    run = Path(run_dir).resolve()
    source = run / 'runtime_sources'
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_absolute() and not checkpoint.is_file():
        checkpoint = run / checkpoint
    checkpoint = checkpoint.resolve()
    if not source.is_dir() or checkpoint.parent != run or not checkpoint.name.startswith('episode_'):
        raise ValueError('Expected a saved runtime and an episode snapshot from that run')
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in {'mappo', 'shared', 'cadmm', 'vtemp'}:
            origin = getattr(module, '__file__', None)
            if origin and not Path(origin).resolve().is_relative_to(source):
                raise RuntimeError(f'Foreign simulator module already loaded: {name}')
    sys.path.insert(0, str(source))
    from mappo.train_mappo import make_model

    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = json.loads((run / 'config.json').read_text())
    if saved['config'] != config or saved['reward_contract'] != 'uav_remaining_route_side_exploration_v9_1':
        raise ValueError('Checkpoint does not match the frozen V9.1 configuration')
    if checkpoint.name != f"episode_{saved['training_episodes']:05d}.pt":
        raise ValueError('Episode number in filename and checkpoint metadata differ')
    torch.set_num_threads(1)
    model = make_model(config)
    model.load_state_dict(saved['model'], strict=True)
    model.eval().requires_grad_(False)
    return model, config


def _world_cells(evidence):
    return np.sign(evidence.log_odds).astype(np.int8)


def _local_tile(obs):
    if obs['map_axes'] != 'yx':
        raise ValueError('Expected native YX occupancy map')
    tile = np.asarray(obs['occupancy_map'])
    if not obs.get('sensor_valid', True):
        return np.zeros(tile.shape, dtype=np.int8)
    return np.where(tile > 0, 1, -1).astype(np.int8)


def _snapshot(env, ground_evidence, car_path, drone_paths, step):
    car = env.obs[env.ground]
    drones = [env.obs[rid] for rid in env.ids]
    return dict(frame=step, car_xy=car_path[-1].copy(), goal_xy=env.goal.copy(),
                drone_xy=drone_paths[-1].copy(), car_path=np.stack(car_path),
                drone_paths=np.stack(drone_paths), resolution=env.ledger.resolution,
                air_cells=_world_cells(env.ledger), ground_cells=_world_cells(ground_evidence),
                received_packets=len(env.ledger.received),
                car_tile=_local_tile(car), car_tile_resolution=float(car['map_resolution']),
                drone_tiles=np.stack([_local_tile(obs) for obs in drones]),
                drone_tile_resolution=float(drones[0]['map_resolution']))


def collect_case(scene, seed, model, config, frames, *, stochastic=True, include_terminal=False):
    """Return (snapshots, reason) for requested 1-based step numbers in one map.

    Every returned snapshot represents an executed step. With include_terminal,
    also return the last executed step of an episode that ends between samples. Scene
    geometry initializes the simulator and never supplies a learner observation.
    The V9.1 native ledger updates only from delivered packets, deduplicating
    retransmissions and resolving overlapping UAV pixels with signed log odds.
    """
    from mappo.mappo_experiment import DeliveredEvidence, Experiment, SafetyTermination

    wanted = frozenset(frames)
    if not wanted or any(not isinstance(s, (int, np.integer)) or s < 1 for s in wanted):
        raise ValueError('Provide one or more positive integer frame numbers')
    fields = ('drop', 'exploration_weight', 'exploration_candidates', 'communication_mode',
              'safety_filter', 'spacing_penalty_weight', 'intervention_weight', 'moving_ground',
              'reward_mode', 'ground_speed', 'path_reward_scale', 'ground_safety_distance',
              'subgoal_radius', 'subgoal_arrival', 'subgoal_review_interval', 'route_reward_weight',
              'time_penalty', 'ground_stationary', 'chain_soft_ratio', 'chain_risk_weight',
              'chain_outward_weight', 'chain_missing_weight', 'path_quality_weight',
              'mapping_route_origin', 'cbf_margin_weight', 'cbf_margin_scale',
              'frontier_progress_weight', 'side_exploration_weight', 'chain_ground_range',
              'chain_link_range', 'relay_loss', 'relay_delay_jitter', 'relay_period',
              'relay_capacity', 'fusion_bridge_distance', 'world_size')
    options = {name: config[name] for name in fields}
    sizes = config['fleet_sizes']
    snapshots = []
    reason = 'frame_limit'
    # Same action-seed construction used for a native fixed-map validation sample.
    sample_seed = int(np.random.SeedSequence([config['seed'], seed, 0]).generate_state(1)[0])
    with open(os.devnull, 'w') as sink, redirect_stdout(sink), torch.inference_mode():
        torch.manual_seed(sample_seed)
        env = Experiment(seed, model, stochastic=stochastic, num_drones=sizes[seed % len(sizes)],
                         scene=(scene['goal'], scene['circles'], scene.get('attempts', 1)),
                         start=scene['start'], stop_on_ground_arrival=False,
                         mapping_until_connected=config['mapping_reward_until_connected'], **options)
        ground_evidence = DeliveredEvidence(world=env.world_size, resolution=env.ledger.resolution)
        car_path, drone_paths = [], []

        def record_pose_and_ground(step):
            car = env.obs[env.ground]
            if car['map_axes'] != 'yx':
                raise ValueError('Expected native YX ground occupancy')
            pos = np.asarray(car['self']['position'][:2], dtype=float).copy()
            car_path.append(pos)
            drone_paths.append(np.array([env.obs[rid]['self']['position'][:2] for rid in env.ids]))
            ground_evidence.receive(env.ground, step, pos, car['occupancy_map'], car['map_resolution'],
                                    valid=car.get('sensor_valid', True))

        try:
            record_pose_and_ground(0)
            for step in range(1, max(wanted) + 1):
                try:
                    env.step(step - 1)
                except SafetyTermination:
                    reason = 'safety_termination'
                    break
                record_pose_and_ground(step)
                if step in wanted:
                    snapshots.append(_snapshot(env, ground_evidence, car_path, drone_paths, step))
                if env.goal_reached:
                    reason = 'vehicle_arrival'
                    break
            last_executed = len(car_path) - 1
            if include_terminal and last_executed > 0 and last_executed not in wanted:
                snapshots.append(_snapshot(env, ground_evidence, car_path, drone_paths, last_executed))
        finally:
            env.env.close()
    return snapshots, reason
