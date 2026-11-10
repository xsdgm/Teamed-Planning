"""Communication-constrained reconnaissance envelope for vtmp2.

All distances are in metres. The caller supplies configured fleet size, radio
range, fixed flight altitude and sensor footprint; this module never reads
simulator fleet positions or obstacle truth. The envelope is centred on the
vehicle's current position by the target manager.

A connected backbone may branch, but its farthest member can have at most N
radio hops from the vehicle. A straight deployment gives the geometric outer
reach used here. The 3D ground-to-air hop is projected into the horizontal
plane; equal-altitude air-to-air hops need no altitude correction. Half the
square sensor width is its inscribed-circle radius, so no diagonal corner
coverage is assumed.

This is a deployment envelope, not proof of current visibility, free terrain,
reachability by the vehicle, or a simultaneously feasible UAV manoeuvre. Live
CBF, communication, world-boundary and received-map checks still apply.
"""

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real


@dataclass(frozen=True)
class ReconnaissanceEnvelope:
    """Configured reconnaissance reach, independent of simulator truth.

    ``deployment_ratio`` discounts each communication hop *before* the ground
    hop's height projection. It is a planning margin, not a CBF guarantee.
    ``theoretical_radius_m`` retains the CBF link margin but omits this extra
    discount; ``radius_m`` is the radius used for candidate planning.
    """

    drone_count: int = 3
    communication_range: float = 95.
    altitude: float = 15.
    footprint_width: float = 24.
    link_margin: float = .25
    deployment_ratio: float = .85

    def __post_init__(self):
        if (isinstance(self.drone_count, bool)
                or not isinstance(self.drone_count, Integral)
                or self.drone_count < 0):
            raise ValueError('drone_count must be a nonnegative integer')
        object.__setattr__(self, 'drone_count', int(self.drone_count))
        for name in ('communication_range', 'altitude', 'footprint_width',
                     'link_margin', 'deployment_ratio'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f'{name} must be a finite positive number')
            try:
                value = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f'{name} must be a finite positive number') from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a finite positive number')
            object.__setattr__(self, name, value)
        if self.link_margin >= self.communication_range:
            raise ValueError('link_margin must be smaller than communication_range')
        if self.deployment_ratio > 1:
            raise ValueError('deployment_ratio must lie in (0, 1]')
        # Reject unrepresentable ranges even when each input is finite.
        self._radius(self.communication_range - self.link_margin)

    def _radius(self, hop_range):
        if self.drone_count == 0 or hop_range < self.altitude:
            return 0.
        # Avoid squaring large lengths; both inputs have already been checked.
        horizontal_ground_hop = hop_range * math.sqrt(
            max(0., 1. - (self.altitude / hop_range) ** 2))
        try:
            radius = (horizontal_ground_hop
                      + (self.drone_count - 1) * hop_range
                      + self.footprint_width / 2.)
        except OverflowError as exc:
            raise ValueError('reconnaissance radius must remain finite') from exc
        if not math.isfinite(radius):
            raise ValueError('reconnaissance radius must remain finite')
        return float(radius)

    @property
    def theoretical_radius_m(self):
        """Geometric deployment reach after the configured radio-link margin."""
        return self._radius(self.communication_range - self.link_margin)

    @property
    def radius_m(self):
        """Conservative candidate radius after discounting every radio hop."""
        hop_range = (self.communication_range - self.link_margin) * self.deployment_ratio
        return self._radius(hop_range)

    def describe(self):
        """Return JSON-compatible parameters and distinct geometric/plan radii."""
        return dict(asdict(self), theoretical_radius_m=self.theoretical_radius_m,
                    planning_radius_m=self.radius_m,
                    interpretation='deployment_envelope_not_observed_or_traversable_space')
