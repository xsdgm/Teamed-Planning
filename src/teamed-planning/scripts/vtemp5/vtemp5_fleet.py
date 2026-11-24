"""Fine-tune a VTemp5 waypoint actor on the native V9 fleet and ground planner."""
import math

import numpy as np

from vtemp4.vtemp4_body_graph import body_frame_observation
from vtemp4.vtemp4_vehicle_graph import GOAL_SCALE_M, VehicleGraphHistoryLSTMEpisode
from vtemp5.vtemp5_env import _remaining
from vtemp5.vtemp5_policy import REVIEW_S, _road_graph
from vtemp5.vtemp5_received_obstacle import ReceivedObstacleTracker
from vtemp5.vtemp5_world_model import LocalWorldModel


def add_candidate_roads(obs, state, position, goal, now):
    """Add two separate three-point hypotheses without overwriting received roads."""
    count = 6
    candidate = _road_graph(dict(state, goal=goal, elapsed_steps=now/REVIEW_S), position)
    offset = len(obs['road_valid'])
    if np.any(candidate['road_valid'][count:]):
        raise ValueError('Expected at most three local route points per side')
    for key in candidate:
        extra = candidate[key][:count].copy()
        if key == 'road_neighbors':
            extra[extra >= 0] += offset
        elif key == 'road_entry_mask':
            # Physical entry certificates belong to the native observed road graph.
            extra[:] = False
        elif key == 'road_nodes' and state['evidence_time'] is not None:
            measured = candidate['road_valid'][:count] & (extra[:, 4] == 0.)
            extra[measured, 3] = math.log1p(max(0., now - state['evidence_time']))
        obs[key] = np.concatenate((obs[key], extra), axis=0)


class VTemp5FleetEpisode(VehicleGraphHistoryLSTMEpisode):
    def __init__(self, repo, case, model, *, world_model_path, ground_control='vtemp1_connected_anchored',
                 review_interval_s=REVIEW_S, **kwargs):
        if review_interval_s != REVIEW_S or ground_control != 'vtemp1_connected_anchored':
            raise ValueError('VTemp5 fleet requires 30-second reviews and the repaired anchored planner')
        if kwargs.get('progress_distance', 'euclidean') != 'euclidean' or kwargs.get('waypoint_progress_bonus_ratio', 0.):
            raise ValueError('VTemp5 scores vehicle progress on the issued local reference or final goal')
        self.world_model = LocalWorldModel.load(world_model_path)
        self.tracker = None
        self.obstacle_state = None
        self.last_world_review = None
        self.last_obstacle_id = None
        self.max_fused_road_nodes = 0
        self.unauthorized_switches = 0
        self.active_reviews = 0
        super().__init__(repo, case, model, ground_control=ground_control, review_interval_s=review_interval_s, **kwargs)
        from vtemp4.vtemp4_anchored_ground import AnchoredGroundPlanner
        if not isinstance(self.env.ground_controller.global_planner, AnchoredGroundPlanner):
            self.close()
            raise RuntimeError('Native controller did not install the repaired terrain planner')

    def _after_fusion(self, graph, semantic_map):
        if self.tracker is None:
            self.tracker = ReceivedObstacleTracker(self.world_model, self.adapter.cells.shape, self.adapter.resolution)
            self.tracker.attach(self.adapter)
        return super()._after_fusion(graph, semantic_map)

    def _use_final_goal(self, distance):
        if self.obstacle_state is not None and self.obstacle_state['active']:
            return False
        return super()._use_final_goal(distance)

    def _heading_observation(self):
        self.adapter.observe_receipts()
        car, now = self.position(), self.clock_s
        # A truncated rollout also requests a critic observation between decision
        # ticks. It uses the published forecast until the next scheduled review.
        if self.last_world_review is None or now - self.last_world_review >= self.review_interval:
            self.obstacle_state = self.tracker.review(car, self.goal, self.adapter.cells, now)
            self.last_world_review = now
        state = self.obstacle_state
        if state['obstacle_id'] != self.last_obstacle_id:
            self.encounter_origin = car.copy()
            self.last_obstacle_id = state['obstacle_id']
        obs = super()._heading_observation()
        self.max_fused_road_nodes = max(self.max_fused_road_nodes, int(obs['road_valid'].sum()))
        add_candidate_roads(obs, state, car, self.goal, now)
        if state['active']:
            yaw = float(self.adapter.own['orientation'])
            arcs = self.tracker.constrain_arcs(obs['angle_arcs'], car, yaw)
            obs['angle_arcs'] = arcs
            obs['wait_for_new_evidence'] = not len(arcs)
            side = self.tracker.locked_side
            if side and state['side_status'][side] == 'blocked' and state['side_status'][-side] != 'blocked':
                # The previously issued local point referred to a refuted branch.
                # Only the actor's reference is updated. Navigation still checks the measured ground map.
                obs['local_target'][2:4] = (state['routes'][-side][0] - car) / GOAL_SCALE_M
                obs['local_target_valid'] = True
        return body_frame_observation(obs)

    def _choose_heading(self, now, position, distance):
        super()._choose_heading(now, position, distance)
        state = self.obstacle_state
        event = self.heading_reviews[-1]
        event['received_terrain_tiles'] = self.adapter.tiles
        event['obstacle_id'] = state['obstacle_id']
        if not state['active']:
            if 'release_evidence' in state:
                event['release_evidence'] = state['release_evidence']
            return
        self.active_reviews += 1
        previous = self.tracker.locked_side
        issued = self.current_request
        side = previous
        if issued['designation_source'] == 'policy_angle':
            side = self.tracker.commit(issued['vehicle_yaw_rad'] + issued['relative_angle_rad'], position)
        if previous and side != previous and state['side_status'][previous] != 'blocked':
            self.unauthorized_switches += 1
            raise RuntimeError('Native policy switched without evidence that the previous side was blocked')
        event.update(locked_side=side, side_status=state['side_status'], closure_evidence=state['closure_evidence'],
                     evidence_time=state['evidence_time'], route_open_probability=state['route_open_probability'])
        if side and state['side_status'][side] != 'blocked':
            self.pending['feasible_path'] = np.vstack((self.encounter_origin, state['routes'][side], self.goal))
            self.pending['vehicle_origin'] = position.copy()

    def _reward_distances(self, pending, distance):
        if 'feasible_path' not in pending:
            return super()._reward_distances(pending, distance)
        path = pending['feasible_path']
        origin, car = pending['vehicle_origin'], self.position()
        before = _remaining(origin, path)
        moved = np.linalg.norm(car - origin)
        progress = float(np.clip(before - _remaining(car, path), -moved, moved))
        pending['event']['feasible_route_progress_m'] = progress
        return before, before - progress

    def fleet_summary(self):
        result = super().fleet_summary()
        result['vtemp5_world'] = dict(prediction_calls=self.tracker.prediction_calls,
                                      detected_obstacles=self.tracker.obstacle_count,
                                      active_reviews=self.active_reviews, side_switches=self.tracker.switch_count,
                                      unauthorized_switches=self.unauthorized_switches,
                                      max_fused_road_nodes=self.max_fused_road_nodes)
        return result
