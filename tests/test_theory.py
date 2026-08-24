"""Numerically check the first-order acceptance model in paper/theory.md.

Claim 1:  TVD(p0, p_s) = (s/2) * E_{p0}|u - E_{p0}[u]| + O(s^2)
Claim 2:  dT/dbeta -> gamma(gamma+1)/2 as beta -> 1, where
          T(beta) = (1 - beta^(gamma+1)) / (1 - beta)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from tandemspec.metrics import tvd, expected_tokens_per_step


def test_first_order_tvd():
    g = torch.Generator().manual_seed(0)
    V = 2048
    z0 = torch.randn(V, generator=g) * 2.0
    u = torch.randn(V, generator=g)
    p0 = z0.softmax(-1)
    kappa = float((p0 * (u - (p0 * u).sum()).abs()).sum())
    worst = 0.0
    for s in (0.002, 0.005, 0.01, 0.02):
        got = float(tvd(p0, (z0 + s * u).softmax(-1)))
        pred = s / 2 * kappa
        rel = abs(got - pred) / pred
        worst = max(worst, rel)
        print(f"  s={s:<6} predicted {pred:.6f}  measured {got:.6f}  rel err {100*rel:.2f}%")
    assert worst < 0.05, f"first-order TVD model off by {100*worst:.1f}%"


def test_throughput_sensitivity():
    for gamma in (2, 4, 8, 16):
        b = 0.999
        num = (expected_tokens_per_step(b + 1e-4, gamma) - expected_tokens_per_step(b - 1e-4, gamma)) / 2e-4
        pred = gamma * (gamma + 1) / 2
        print(f"  gamma={gamma:<3} dT/dbeta measured {num:.2f}  predicted {pred:.2f}")
        assert abs(num - pred) / pred < 0.02


def test_optimal_gamma_moves_with_beta():
    from tandemspec.eval.throughput import best_gamma
    c = 0.15
    prev = None
    for beta in (0.98, 0.9, 0.8, 0.6, 0.4):
        g, s = best_gamma(beta, c)
        print(f"  beta={beta:<5} optimal gamma={g}  speedup={s:.2f}x")
        if prev is not None:
            assert g <= prev, "optimal gamma must be monotone non-increasing as beta falls"
        prev = g


if __name__ == "__main__":
    for fn in (test_first_order_tvd, test_throughput_sensitivity, test_optimal_gamma_moves_with_beta):
        print("running", fn.__name__); fn(); print("  PASS")
