import contextlib
import io
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from mappo.mappo_experiment import DeliveredEvidence, Experiment
from mappo.train_mappo import make_model, reward_contract, validate_config


CONFIG = Path(__file__).resolve().parents[3] / 'config/algorithm/mappo_training_v9.json'


class SideExplorationTests(unittest.TestCase):
    start = np.array([21., 81.])
    goal = np.array([201., 81.])

    def setUp(self):
        self.ledger = DeliveredEvidence(world=240., resolution=2.)

    def deliver(self, frame, xy, *, occupied=False, source=1, valid=True):
        tile = np.array([[occupied]])
        return self.ledger.receive(source, frame, xy, tile, 2., valid=valid)

    def test_only_a_previously_delivered_blockage_makes_side_area_eligible(self):
        old_seen = self.ledger.seen.copy()
        self.deliver(1, (101., 81.), occupied=True, valid=False)
        self.deliver(2, (101., 83.))
        self.deliver(3, (101., 103.), occupied=True)
        self.deliver(4, (211., 81.), occupied=True)
        blockage_before = self.ledger.first_route_blockage(self.start, self.goal)
        self.assertIsNone(blockage_before)
        self.deliver(5, (101., 81.), occupied=True)
        self.assertEqual(self.ledger.new_side_area(old_seen, blockage_before), 0.)
        self.assertIsNotNone(self.ledger.first_route_blockage(self.start, self.goal))

    def test_team_gets_area_once_for_new_cells_on_either_flank(self):
        self.deliver(1, (101., 81.), occupied=True)
        blockage = self.ledger.first_route_blockage(self.start, self.goal)
        old_seen = self.ledger.seen.copy()
        self.deliver(2, (101., 105.))
        self.deliver(2, (101., 105.), source=2)
        self.deliver(3, (101., 57.), occupied=True)
        self.deliver(4, (101., 85.))    # Inside the original route corridor.
        self.deliver(5, (151., 105.))   # Far past the measured obstacle.
        self.deliver(6, (101., 155.))   # Too far away laterally.
        self.assertEqual(self.ledger.new_side_area(old_seen, blockage), 8.)
        old_seen = self.ledger.seen.copy()
        self.deliver(7, (101., 57.), source=3)
        self.assertEqual(self.ledger.new_side_area(old_seen, blockage), 0.)

    def test_v9_preset_keeps_side_reward_disabled(self):
        original = validate_config(json.loads(CONFIG.read_text()))
        self.assertEqual(original['side_exploration_weight'], 0.)
        self.assertEqual(reward_contract(original), 'uav_remaining_route_mapping_v9')
        variant = validate_config(dict(original, side_exploration_weight=0.1))
        self.assertNotEqual(reward_contract(original), reward_contract(variant))

    def test_actual_delivery_changes_team_reward_by_logged_side_term(self):
        config = validate_config(json.loads(CONFIG.read_text()))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            actor = make_model(config).eval()
        outcomes = []
        with contextlib.redirect_stdout(io.StringIO()), torch.inference_mode():
            for weight in (0., 0.2):
                env = Experiment(
                    10021, actor, stochastic=False, world_size=120., start=(15., 60.),
                    scene=((100., 60.), [], 1), communication_mode='backbone',
                    reward_mode='velocity_mapping', mapping_route_origin='vehicle',
                    path_quality_weight=20., mapping_until_connected=True,
                    side_exploration_weight=weight)
                self.addCleanup(env.env.close)
                # Inject a prior *received* obstruction; the simulator scene is empty.
                env.ledger.receive('prior obstacle', 0, (35., 61.), np.ones((1, 1)), 2.)
                _, reward, metrics = env.step(0)
                outcomes.append((reward, metrics))
        (original_reward, _), (updated_reward, updated_metrics) = outcomes
        self.assertGreater(updated_metrics['received_ground_packets'], 0)
        area = updated_metrics['side_exploration_area_m2']
        self.assertGreater(area, 0.)
        self.assertAlmostEqual(updated_metrics['side_exploration_reward'], 0.2 * area / 400.)
        self.assertAlmostEqual(updated_reward - original_reward, updated_metrics['side_exploration_reward'])
