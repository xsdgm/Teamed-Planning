"""Wrapped Cauchy over physical, vehicle-relative angles, conditioned on legal ring arcs."""
import math

import numpy as np
import torch


# Entropy integrates a smooth function of a uniform CDF coordinate. Splitting
# at the antipode resolves the narrowest peak allowed by the rho cap of .99.
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(128)


def _cdf_angle(offset, rho):
    """Continuous, unwrapped 2*pi-scaled CDF; changing rho to -rho gives its inverse."""
    return offset + 2 * torch.atan2(rho * offset.sin(), 1 - rho * offset.cos())


class BodyAngleDistribution:
    def __init__(self, output, arc_lists, reference_angle=None):
        if output.ndim != 2 or output.shape != (len(arc_lists), 2) or not len(arc_lists):
            raise ValueError('Expected one pair of angle parameters per observation')
        self.location = math.pi * output[:, 0].tanh()
        if reference_angle is not None:
            if reference_angle.shape != self.location.shape:
                raise ValueError('Expected one reference bearing per angle')
            # Both the sampled action and the legal arcs use physical angles relative to the car.
            center = reference_angle.double() + self.location.double()
            self.location = ((center + math.pi) % math.tau - math.pi).to(output.dtype)
        self.rho = .99 * output[:, 1].sigmoid()

        count = max(len(arcs) for arcs in arc_lists)
        bounds = np.zeros((len(arc_lists), count, 2), dtype=np.float64)
        for row, arcs in enumerate(arc_lists):
            arcs = np.asarray(arcs, dtype=np.float64)
            if (arcs.ndim != 2 or arcs.shape[1] != 2 or not len(arcs)
                    or not np.isfinite(arcs).all() or np.any(np.abs(arcs) > math.pi + 2e-7)):
                raise ValueError('Expected nonempty signed physical angle intervals')
            arcs = np.clip(arcs, -math.pi, math.pi)
            if np.any(arcs[:, 0] >= arcs[:, 1]) or np.any(arcs[1:, 0] < arcs[:-1, 1]):
                raise ValueError('Legal angle intervals must be sorted, disjoint and have positive width')
            bounds[row, :len(arcs)] = arcs
        self.lower, self.upper = torch.as_tensor(bounds, device=output.device).unbind(-1)
        self.valid = self.upper > self.lower
        self._center = ((self.location.double() + math.pi) % math.tau - math.pi)[:, None]
        self._rho = self.rho.double()[:, None]
        lo = (self.lower - self._center) / 2
        hi = (self.upper - self._center) / 2
        self._start = _cdf_angle(2 * lo, self._rho)
        # The angle between two transformed unit vectors gives the CDF span
        # directly, including where subtracting two close CDFs would lose it.
        cross = (1 - self._rho.square()) * ((self.upper - self.lower) / 2).sin()
        dot = (1-self._rho).square()*lo.cos()*hi.cos() + (1+self._rho).square()*lo.sin()*hi.sin()
        self._spans = 2 * torch.atan2(cross, dot)
        self._cumulative = self._spans.cumsum(dim=1)
        self._total = self._cumulative[:, -1]

    def action(self, deterministic, generator=None):
        if deterministic:
            contains_mode = self.valid & (self.lower <= self._center) & (self._center <= self.upper)
            endpoints = torch.stack((self.lower, self.upper), dim=-1)
            difference = endpoints - self._center[..., None]
            distances = ((difference + math.pi) % math.tau - math.pi).abs()
            distances = distances.masked_fill(~self.valid[..., None], math.inf).flatten(1)
            nearest = endpoints.flatten(1).gather(1, distances.argmin(1, keepdim=True))[:, 0]
            angle = torch.where(contains_mode.any(dim=1), self._center[:, 0], nearest)
        else:
            device = self.location.device if generator is None else generator.device
            sample = torch.rand(self.location.shape, generator=generator, device=device, dtype=torch.float64)
            mass = sample.to(self._total.device) * self._total
            index = (mass[:, None] >= self._cumulative).sum(1).minimum(self.valid.sum(1) - 1)
            row = torch.arange(len(index), device=self.location.device)
            prior_mass = (self._cumulative - self._spans)[row, index]
            phase = self._start[row, index] + (mass - prior_mass)
            angle = self._center[:, 0] + _cdf_angle(phase, -self._rho[:, 0])
            angle = angle.clamp(min=self.lower[row, index], max=self.upper[row, index])
        # Retain physical endpoints in double precision, including intervals
        # narrower than one float32 ulp. PPO stores a float32 copy on replay.
        return torch.ones_like(angle, dtype=torch.bool), angle

    def log_prob(self, designate, angle):
        if not designate.all():
            raise ValueError('Periodic policy requires a new angle for every decision')
        a = angle[:, None]
        # Replay rounds a physical sample to the actor dtype. Test against the
        # same rounding of the interval endpoints, including at +/-pi.
        legal = self.valid & (a >= self.lower.to(a.dtype)) & (a <= self.upper.to(a.dtype))
        if not legal.any(dim=1).all():
            raise ValueError('Stored angle outside the admissible ring arcs')
        rho = self._rho[:, 0]
        offset = (angle.double() - self._center[:, 0]) * .5
        denominator = (1-rho).square() + 4*rho*offset.sin().square()
        return (torch.log1p(-rho.square()) - denominator.log() - self._total.log()).to(self.location.dtype)

    def entropy(self):
        # In CDF coordinates the conditional density is uniform. Its physical
        # log density is log(1+2*rho*cos(phase)+rho^2) minus a batch constant.
        antipode = math.pi + math.tau * torch.floor((self._start + math.pi) / math.tau)
        first_span = torch.minimum((antipode - self._start).clamp_min(0.), self._spans)
        left = torch.stack((self._start, self._start + first_span), dim=-1)
        half = torch.stack((first_span, self._spans - first_span), dim=-1) / 2
        nodes = torch.as_tensor(_NODES, device=half.device)
        weights = torch.as_tensor(_WEIGHTS, device=half.device)
        phase = (left + half)[..., None] + half[..., None] * nodes
        rho = self._rho[..., None, None]
        log_term = ((1-rho).square() + 4*rho*(phase*.5).cos().square()).log()
        integral = (half * (weights * log_term).sum(-1)).sum((1, 2))
        entropy = self._total.log() + torch.log1p(-self._rho[:, 0].square()) - integral/self._total
        return entropy.to(self.location.dtype)
