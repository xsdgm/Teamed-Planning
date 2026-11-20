"""Version tags for guidance designated on the UGV's 200 m ring and held in world XY."""

RING_RADIUS_M = 200.0
RING_ACTION_CONTRACT = 'vehicle_yaw_fixed_world_ring_200m_v1'
PERIODIC_RING_ACTION_CONTRACT = 'vehicle_yaw_fixed_world_ring_200m_every_15s_v1'
PERIODIC_GOAL_RING_ACTION_CONTRACT = 'vehicle_yaw_ring_200m_or_final_goal_every_15s_v2'
PERIODIC_ACTION_CONTRACTS = (PERIODIC_RING_ACTION_CONTRACT, PERIODIC_GOAL_RING_ACTION_CONTRACT)
RING_HEADING_FORMAT = 'vtemp4_fixed_ring_heading_v1'
RING_WAYPOINT_FORMAT = 'vtemp4_fixed_ring_waypoint_v1'
RING_CONTINUOUS_HISTORY_FORMAT = 'vtemp4_fixed_ring_continuous_heading_history_v1'
RING_PERIODIC_HISTORY_FORMAT = 'vtemp4_fixed_ring_periodic_heading_history_v1'
RING_UNIFIED_ATTENTION_FORMAT = 'vtemp4_fixed_ring_periodic_unified_attention_v1'
RING_UNIFIED_GOAL_FORMAT = 'vtemp4_periodic_unified_attention_goal_v2'
RING_POLICY_FORMATS = {
    'heading': RING_HEADING_FORMAT,
    'continuous_heading_history': RING_CONTINUOUS_HISTORY_FORMAT,
    'periodic_heading_history': RING_PERIODIC_HISTORY_FORMAT,
    'periodic_unified_attention': RING_UNIFIED_ATTENTION_FORMAT,
    'periodic_unified_attention_goal': RING_UNIFIED_GOAL_FORMAT,
    'waypoint': RING_WAYPOINT_FORMAT,
    'waypoint_history': RING_WAYPOINT_FORMAT,
}


def ring_action_contract(fmt):
    if fmt == RING_UNIFIED_GOAL_FORMAT:
        return PERIODIC_GOAL_RING_ACTION_CONTRACT
    if fmt in (RING_PERIODIC_HISTORY_FORMAT, RING_UNIFIED_ATTENTION_FORMAT):
        return PERIODIC_RING_ACTION_CONTRACT
    return RING_ACTION_CONTRACT if fmt in RING_POLICY_FORMATS.values() else None


def periodic_policy_types(fmt):
    if fmt == RING_UNIFIED_GOAL_FORMAT:
        from simulator.vtemp4_unified_attention import GoalAwareUnifiedAttentionPolicy, GoalAwareUnifiedAttentionEpisode
        return GoalAwareUnifiedAttentionPolicy, GoalAwareUnifiedAttentionEpisode
    if fmt == RING_UNIFIED_ATTENTION_FORMAT:
        from simulator.vtemp4_unified_attention import UnifiedAttentionPolicy, UnifiedAttentionEpisode
        return UnifiedAttentionPolicy, UnifiedAttentionEpisode
    if fmt == RING_PERIODIC_HISTORY_FORMAT:
        from simulator.vtemp4_periodic_history import PeriodicHistoryPolicy, PeriodicHistoryFleetEpisode
        return PeriodicHistoryPolicy, PeriodicHistoryFleetEpisode
    raise ValueError(f'Unknown periodic policy format {fmt!r}')
