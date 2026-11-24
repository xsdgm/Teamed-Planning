"""Version tags for guidance designated on the UGV's 200 m ring and held in world XY."""

RING_RADIUS_M = 200.0
RING_ACTION_CONTRACT = 'vehicle_yaw_fixed_world_ring_200m_v1'
PERIODIC_RING_ACTION_CONTRACT = 'vehicle_yaw_fixed_world_ring_200m_every_15s_v1'
PERIODIC_GOAL_RING_ACTION_CONTRACT = 'vehicle_yaw_ring_200m_or_final_goal_every_15s_v2'
PERIODIC_GOAL_RING_CONTRACTS = {
    15: PERIODIC_GOAL_RING_ACTION_CONTRACT,
    30: 'vehicle_yaw_ring_200m_or_final_goal_every_30s_v2',
}
BODY_ANGLE_GOAL_RING_CONTRACTS = {
    seconds: f'physical_body_angle_ring_200m_or_final_goal_every_{seconds}s_v1'
    for seconds in PERIODIC_GOAL_RING_CONTRACTS
}
GOAL_RESIDUAL_RING_CONTRACTS = {
    seconds: f'goal_bearing_residual_physical_ring_200m_or_final_goal_every_{seconds}s_v1'
    for seconds in PERIODIC_GOAL_RING_CONTRACTS
}
GOAL_RING_REVIEW_INTERVALS = {
    contract: seconds for contracts in (PERIODIC_GOAL_RING_CONTRACTS, BODY_ANGLE_GOAL_RING_CONTRACTS,
                                        GOAL_RESIDUAL_RING_CONTRACTS)
    for seconds, contract in contracts.items()
}
PERIODIC_REVIEW_INTERVALS = {
    PERIODIC_RING_ACTION_CONTRACT: 15,
    **GOAL_RING_REVIEW_INTERVALS,
}
PERIODIC_ACTION_CONTRACTS = tuple(PERIODIC_REVIEW_INTERVALS)
RING_HEADING_FORMAT = 'vtemp4_fixed_ring_heading_v1'
RING_WAYPOINT_FORMAT = 'vtemp4_fixed_ring_waypoint_v1'
RING_CONTINUOUS_HISTORY_FORMAT = 'vtemp4_fixed_ring_continuous_heading_history_v1'
RING_PERIODIC_HISTORY_FORMAT = 'vtemp4_fixed_ring_periodic_heading_history_v1'
RING_UNIFIED_ATTENTION_FORMAT = 'vtemp4_fixed_ring_periodic_unified_attention_v1'
RING_UNIFIED_GOAL_FORMAT = 'vtemp4_periodic_unified_attention_goal_v2'
RING_UNIFIED_GOAL_NO_GRID_FORMAT = 'vtemp4_periodic_unified_attention_goal_no_grid_v1'
RING_VEHICLE_GRAPH_GOAL_FORMAT = 'vtemp4_periodic_vehicle_graph_goal_v1'
RING_VEHICLE_GRAPH_LSTM_FORMAT = 'vtemp4_periodic_vehicle_graph_goal_lstm_v1'
RING_VEHICLE_GRAPH_BODY_LSTM_FORMAT = 'vtemp4_periodic_vehicle_graph_body_lstm_v1'
RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT = 'vtemp4_periodic_vehicle_graph_goal_residual_lstm_v1'
NO_GRID_GOAL_VARIANT = 'periodic_unified_attention_goal_no_grid'
VEHICLE_GRAPH_GOAL_VARIANT = 'periodic_vehicle_graph_goal'
VEHICLE_GRAPH_LSTM_VARIANT = 'periodic_vehicle_graph_goal_lstm'
VEHICLE_GRAPH_BODY_LSTM_VARIANT = 'periodic_vehicle_graph_body_lstm'
VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_VARIANT = 'periodic_vehicle_graph_goal_residual_lstm'
LSTM_REVIEW_INTERVAL_S = 30
LSTM_PROGRESS_REWARD_WEIGHT = 10.
PROGRESS_DISTANCES = ('euclidean', 'traversable')
ROUTE_METRICS = ('route_initial_m', 'route_remaining_m', 'route_progress_m', 'route_unavailable_intervals')
LSTM_POLICY_FORMATS = {
    VEHICLE_GRAPH_LSTM_VARIANT: RING_VEHICLE_GRAPH_LSTM_FORMAT,
    VEHICLE_GRAPH_BODY_LSTM_VARIANT: RING_VEHICLE_GRAPH_BODY_LSTM_FORMAT,
    VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_VARIANT: RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT,
}
GOAL_POLICY_FORMATS = {
    **LSTM_POLICY_FORMATS,
    VEHICLE_GRAPH_GOAL_VARIANT: RING_VEHICLE_GRAPH_GOAL_FORMAT,
    NO_GRID_GOAL_VARIANT: RING_UNIFIED_GOAL_NO_GRID_FORMAT,
    'periodic_unified_attention_goal': RING_UNIFIED_GOAL_FORMAT,
}
GOAL_VARIANTS = tuple(GOAL_POLICY_FORMATS)
GOAL_FORMATS = tuple(GOAL_POLICY_FORMATS.values())
RING_POLICY_FORMATS = {
    'heading': RING_HEADING_FORMAT,
    'continuous_heading_history': RING_CONTINUOUS_HISTORY_FORMAT,
    'periodic_heading_history': RING_PERIODIC_HISTORY_FORMAT,
    'periodic_unified_attention': RING_UNIFIED_ATTENTION_FORMAT,
    **GOAL_POLICY_FORMATS,
    'waypoint': RING_WAYPOINT_FORMAT,
    'waypoint_history': RING_WAYPOINT_FORMAT,
}


def new_run_settings(variant, review_interval_s, progress_reward_weight):
    if review_interval_s is None:
        review_interval_s = LSTM_REVIEW_INTERVAL_S if variant in LSTM_POLICY_FORMATS else 15
    if progress_reward_weight is None and variant in LSTM_POLICY_FORMATS:
        progress_reward_weight = LSTM_PROGRESS_REWARD_WEIGHT
    return review_interval_s, progress_reward_weight


def validate_progress_distance(variant, distance):
    if distance is not None and (variant != VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_VARIANT or distance not in PROGRESS_DISTANCES):
        raise ValueError('Configurable progress distance requires a goal-residual LSTM and euclidean or traversable distance')


def new_run_waypoint_bonus(variant, ratio, progress_distance=None):
    if ratio is None and variant == VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_VARIANT:
        return 0. if progress_distance == 'traversable' else .10
    return ratio


def ring_action_contract(fmt, review_interval_s=15):
    if fmt in GOAL_FORMATS:
        if type(review_interval_s) is not int or review_interval_s not in PERIODIC_GOAL_RING_CONTRACTS:
            raise ValueError('Goal-aware waypoints require a 15- or 30-second designation clock')
        if fmt == RING_VEHICLE_GRAPH_BODY_LSTM_FORMAT:
            return BODY_ANGLE_GOAL_RING_CONTRACTS[review_interval_s]
        if fmt == RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT:
            return GOAL_RESIDUAL_RING_CONTRACTS[review_interval_s]
        return PERIODIC_GOAL_RING_CONTRACTS[review_interval_s]
    if fmt in (RING_PERIODIC_HISTORY_FORMAT, RING_UNIFIED_ATTENTION_FORMAT):
        if type(review_interval_s) is not int or review_interval_s != 15:
            raise ValueError('Periodic history waypoints require the 15-second designation clock')
        return PERIODIC_RING_ACTION_CONTRACT
    return RING_ACTION_CONTRACT if fmt in RING_POLICY_FORMATS.values() else None


def periodic_policy_types(fmt):
    if fmt == RING_VEHICLE_GRAPH_GOAL_RESIDUAL_LSTM_FORMAT:
        from vtemp4.vtemp4_body_graph import GoalResidualBodyFramePolicy, BodyFrameVehicleGraphEpisode
        return GoalResidualBodyFramePolicy, BodyFrameVehicleGraphEpisode
    if fmt == RING_VEHICLE_GRAPH_BODY_LSTM_FORMAT:
        from vtemp4.vtemp4_body_graph import BodyFrameVehicleGraphPolicy, BodyFrameVehicleGraphEpisode
        return BodyFrameVehicleGraphPolicy, BodyFrameVehicleGraphEpisode
    if fmt == RING_VEHICLE_GRAPH_LSTM_FORMAT:
        from vtemp4.vtemp4_vehicle_graph import VehicleGraphHistoryLSTMPolicy, VehicleGraphHistoryLSTMEpisode
        return VehicleGraphHistoryLSTMPolicy, VehicleGraphHistoryLSTMEpisode
    if fmt == RING_VEHICLE_GRAPH_GOAL_FORMAT:
        from vtemp4.vtemp4_vehicle_graph import VehicleGraphGoalPolicy, VehicleGraphGoalEpisode
        return VehicleGraphGoalPolicy, VehicleGraphGoalEpisode
    if fmt == RING_UNIFIED_GOAL_NO_GRID_FORMAT:
        from vtemp4.vtemp4_unified_attention import GoalAwareTerrainSummaryPolicy, GoalAwareTerrainSummaryEpisode
        return GoalAwareTerrainSummaryPolicy, GoalAwareTerrainSummaryEpisode
    if fmt == RING_UNIFIED_GOAL_FORMAT:
        from vtemp4.vtemp4_unified_attention import GoalAwareUnifiedAttentionPolicy, GoalAwareUnifiedAttentionEpisode
        return GoalAwareUnifiedAttentionPolicy, GoalAwareUnifiedAttentionEpisode
    if fmt == RING_UNIFIED_ATTENTION_FORMAT:
        from vtemp4.vtemp4_unified_attention import UnifiedAttentionPolicy, UnifiedAttentionEpisode
        return UnifiedAttentionPolicy, UnifiedAttentionEpisode
    if fmt == RING_PERIODIC_HISTORY_FORMAT:
        from vtemp4.vtemp4_periodic_history import PeriodicHistoryPolicy, PeriodicHistoryFleetEpisode
        return PeriodicHistoryPolicy, PeriodicHistoryFleetEpisode
    raise ValueError(f'Unknown periodic policy format {fmt!r}')
