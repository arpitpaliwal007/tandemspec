"""The acceptance simulator must be (a) lossless and (b) match the closed form."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from tandemspec.accept import simulate_block, residual_distribution
from tandemspec.metrics import acceptance_prob, expected_tokens_per_step, tvd


def _rand_pair(V=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(V, generator=g) + 0.05; p /= p.sum()
    q = torch.rand(V, generator=g) + 0.05; q /= q.sum()
    return p, q


def test_beta_identity():
    p, q = _rand_pair()
    assert abs(float(acceptance_prob(p, q)) - (1 - float(tvd(p, q)))) < 1e-6


def test_residual_is_a_distribution():
    p, q = _rand_pair(seed=1)
    r = residual_distribution(p, q)
    assert abs(float(r.sum()) - 1.0) < 1e-6 and float(r.min()) >= 0.0


def test_losslessness_and_mean_accepted():
    """Emitted first token ~ p exactly, and E[accepted] matches sum_i beta^i."""
    V, gamma, N = 8, 4, 200_000
    p, q = _rand_pair(V, seed=2)
    P = p.expand(gamma + 1, V).contiguous()
    Q = q.expand(gamma, V).contiguous()
    g = torch.Generator().manual_seed(7)
    counts = torch.zeros(V)
    total_acc = 0
    dtoks = torch.multinomial(Q, 1, generator=g).squeeze(-1)
    for i in range(N):
        dt = torch.multinomial(q.expand(gamma, V), 1, generator=g).squeeze(-1)
        r = simulate_block(P, Q, dt, generator=g)
        counts[r.emitted[0]] += 1
        total_acc += r.n_accepted
    emp = counts / counts.sum()
    l1 = float((emp - p).abs().sum())
    beta = float(acceptance_prob(p, q))
    exp_acc = sum(beta ** i for i in range(1, gamma + 1))
    got_acc = total_acc / N
    print(f"  losslessness L1 = {l1:.4f} (tol 0.02)")
    print(f"  E[accepted]: closed form {exp_acc:.4f}, simulated {got_acc:.4f}")
    print(f"  E[tokens/step]: closed form {expected_tokens_per_step(beta, gamma):.4f}, "
          f"simulated {got_acc + 1:.4f}")
    assert l1 < 0.02, "speculative sampling must reproduce the target distribution"
    assert abs(exp_acc - got_acc) < 0.03, "acceptance must match the i.i.d. closed form"


if __name__ == "__main__":
    for fn in [test_beta_identity, test_residual_is_a_distribution, test_losslessness_and_mean_accepted]:
        print(f"running {fn.__name__}"); fn(); print("  PASS")
