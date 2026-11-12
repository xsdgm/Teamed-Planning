"""Shared ground-navigation defaults for 2D experiments and policy versions."""

# Vehicle-centre clearance used by graph extraction, graph bridges, planning,
# and new reference-route audits. Explicit experiment settings may override it
# (for example, 1.5 m when replaying a historical experiment).
DEFAULT_GROUND_CLEARANCE_M = 1.0
