"""Check physical bearings, conditioned wrapped-Cauchy density, sampling and entropy."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from scipy.integrate import quad

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vtemp4.vtemp4_body_angle import BodyAngleDistribution


def reference(output, arcs):
    center = math.pi * math.tanh(output[0])
    rho = .99 / (1 + math.exp(-output[1]))

    def density(a):
        denom = (1-rho)**2 + 4*rho*math.sin((a-center)/2)**2
        return (1-rho*rho) / (math.tau*denom)

    def integrate(fn, intervals):
        return sum(quad(fn, lo, hi, points=[center] if lo < center < hi else [],
                        epsabs=1e-12)[0] for lo, hi in intervals)

    masses = np.array([integrate(density, [arc]) for arc in arcs])
    norm = masses.sum()
    entropy = integrate(lambda a: -density(a)/norm * math.log(density(a)/norm), arcs)
    return (lambda a: math.log(density(a)/norm)), masses/norm, entropy


class BodyAngleTests(unittest.TestCase):
    def test_mode_stays_at_legal_physical_bearing_and_respects_circular_distance(self):
        # The third mode sits across the -pi seam from the upper endpoint;
        # comparing ordinary absolute angle differences would select the wrong endpoint.
        radians = np.deg2rad
        arcs = [radians([[-180, 180]]), radians([[-30, 90]]), radians([[30, 160]]),
                radians([[-90, -60], [20, 110]])]
        centers = radians([0, 0, -170, 45])
        logits = np.stack((np.arctanh(centers/math.pi), np.zeros(4)), axis=1)
        policy = BodyAngleDistribution(torch.from_numpy(logits), arcs)
        turns, theta = policy.action(deterministic=True)
        self.assertTrue(turns.all())
        np.testing.assert_allclose(np.rad2deg(theta.numpy()), [0, 0, 160, 45], atol=1e-10)

    def test_normalization_entropy_and_gradient_match_independent_quadrature(self):
        cases = [((-.55, 2.3), [[-3., -2.], [-1.7, -.8], [.2, 1.4]]),
                 ((.04, 8.), [[-3.1, -2.], [-.4, 2.7], [3., math.pi]]),
                 ((.32, -.7), [[-2.66, -2.6599999]])]
        for logits, arcs in cases:
            with self.subTest(logits=logits):
                y = torch.tensor([logits], dtype=torch.float64, requires_grad=True)
                chosen = torch.tensor([np.mean(arcs[0])], dtype=torch.float64)
                def evaluate(z):
                    p = BodyAngleDistribution(z, [arcs])
                    return p.log_prob(torch.tensor([True]), chosen), p.entropy()
                lp, entropy = evaluate(y)
                logp_ref, _, entropy_ref = reference(logits, arcs)
                self.assertAlmostEqual(lp.item(), logp_ref(chosen.item()), places=8)
                self.assertAlmostEqual(entropy.item(), entropy_ref, places=7)
                self.assertTrue(torch.autograd.gradcheck(evaluate, y, eps=1e-5, atol=1e-6, rtol=1e-4))

    def test_uniform_density_padded_rows_and_rounded_replay_endpoint(self):
        arcs = [[[-1.1, -.7], [.2, 1.1]], [[2.000000015, 2.000000018]]]
        params = torch.tensor([[0., -100.], [.4, 9.]], requires_grad=True)
        p = BodyAngleDistribution(params, arcs)
        turns, theta = p.action(deterministic=True)
        lo, hi = arcs[1][0]
        self.assertTrue(lo <= theta[1].item() <= hi)
        self.assertEqual(theta[1].float().item(), 2.)
        behavior = p.log_prob(turns, theta)
        torch.testing.assert_close(p.log_prob(turns, theta.float()), behavior, atol=2e-5, rtol=0)
        self.assertAlmostEqual(behavior[0].item(), -math.log(1.3), places=6)
        self.assertAlmostEqual(p.entropy()[0].item(), math.log(1.3), places=6)
        self.assertAlmostEqual(p.entropy()[1].item(), math.log(hi-lo), places=5)
        (behavior + p.entropy()).sum().backward()
        self.assertTrue(torch.isfinite(params.grad).all())
        with self.assertRaisesRegex(ValueError, 'outside'):
            p.log_prob(turns, torch.tensor([0., 2.]))

    def test_sampling_matches_conditioned_physical_cdf(self):
        arcs = np.array([[-3.1, -2.2], [-.4, -.1], [.2, 1.6], [2.55, math.pi]])
        logits = (.13, .3)
        n = 4096
        p = BodyAngleDistribution(torch.tensor([logits], dtype=torch.float64).expand(n, -1), [arcs]*n)
        turns, samples = p.action(False, torch.Generator().manual_seed(412))
        angles = samples.numpy()
        logp, masses, _ = reference(logits, arcs)
        membership = (angles[:, None] >= arcs[:, 0]) & (angles[:, None] <= arcs[:, 1])
        self.assertTrue(np.all(membership.sum(1) == 1))
        np.testing.assert_array_less(np.abs(membership.mean(0)-masses), 6*np.sqrt(masses*(1-masses)/n) + 1/n)
        # Also verify the inverse CDF inside the widest arc.
        half = [arcs[2, 0], np.mean(arcs[2])]
        expected = quad(lambda a: math.exp(logp(a)), *half)[0]
        actual = np.mean((angles >= half[0]) & (angles <= half[1]))
        self.assertLess(abs(actual-expected), 6*math.sqrt(expected*(1-expected)/n))
        self.assertTrue(torch.isfinite(p.log_prob(turns, samples)).all())


if __name__ == '__main__':
    unittest.main()
