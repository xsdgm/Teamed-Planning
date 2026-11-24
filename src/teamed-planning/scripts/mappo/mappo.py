"""Supported checkpoint loading and observation whitelist for V9+ MAPPO policies."""

# Training-only centralized state (time, delivered map, task graph, chain);
# see Experiment.critic_state. The actor never receives these features.
GLOBAL_FEATURES = 10


def legal_observation(obs):
    """Copy the interface whitelist; never traverse raw broadcast dictionaries.

    Tasks and neighbors are reception-only: no additional mission cache is used.
    An empty path explicitly clears a formerly received path.
    """
    return dict(graph=obs.get('graph'), time=float(obs.get('time', 0)),
                ground_robots={str(k): {f: v[f] for f in
                    ('position', 'target', 'planned_path', 'source_time', 'direct', 'survey') if f in v}
                    for k, v in obs.get('ground_robots', {}).items()},
                nearby_drones={str(k): {'position': v['position']}
                    for k, v in obs.get('nearby_drones', {}).items()
                    if v.get('position') is not None},
                neighbor_waypoints={str(k): tuple(v) for k, v in
                    obs.get('neighbor_waypoints', {}).items() if v is not None})


class ActorCritic:
    """Factory for supported MAPPO checkpoint architectures."""
    @classmethod
    def from_state_dict(cls, state):
        """Load a retained V9 or later policy from its architecture markers."""
        if 'route_ab_schema' in state:
            from mappo.survey_mappo import RouteABActorCritic
            return RouteABActorCritic.from_state_dict(state)
        if 'survey_schema' in state:
            from mappo.survey_mappo import SurveyActorCritic
            return SurveyActorCritic.from_state_dict(state)
        if 'gat_actor.0.relay.mode' in state:
            from mappo.relay_attention_mappo import RelayActorCritic
            return RelayActorCritic.from_state_dict(state)
        raise ValueError('Unrecognized or unsupported MAPPO checkpoint architecture')
