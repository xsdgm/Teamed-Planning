"""Offline trajectory measurements for a single circular obstacle."""
import numpy as np


def circle_metrics(case, trajectory_xy, travelled_m, goal_reached):
    """Measure actual ground motion against the course geometry after the rollout."""
    (center, radius), = case['circles']
    start = np.asarray(case['start'], dtype=float)
    goal = np.asarray(case['goal'], dtype=float)
    car = np.asarray(trajectory_xy, dtype=float)
    forward = (goal - start) / np.linalg.norm(goal - start)
    left = np.array([-forward[1], forward[0]])
    station = (car - start) @ forward
    cross_track = (car - start) @ left
    # The downstream plane lies past the whole clearance-inflated obstacle,
    # independently of which side the actor selected.
    far_station = (np.asarray(center) - start) @ forward + radius + case['required_clearance_m']
    passed = np.flatnonzero(station >= far_station)
    remaining = np.linalg.norm(car - goal, axis=1)
    after_pass_progress = None
    if len(passed):
        i = passed[0]
        after_pass_progress = float(remaining[i] - remaining[i:].min())
    # A successful episode ends within the arrival radius. Complete its last
    # few metres to the goal for comparison against the endpoint-to-endpoint reference.
    extra = float(travelled_m + remaining[-1] - case['reference_distance_m']) if goal_reached else None
    return dict(reached_far_side=bool(len(passed)), best_goal_progress_after_far_side_m=after_pass_progress,
                max_abs_cross_track_m=float(np.abs(cross_track).max()), extra_distance_vs_shortest_m=extra)
