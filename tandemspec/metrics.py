"""Distribution-distance metrics that govern speculative-decoding acceptance.

The central identity used throughout TandemSpec:

    beta(p, q) = sum_v min(p_v, q_v) = 1 - TVD(p, q)

where `beta` is the *exact* per-token acceptance probability of standard
speculative sampling (Leviathan et al. 2023; Chen et al. 2023) when the draft
distribution is `q` and the target distribution is `p`, marginalised over the
proposed token x ~ q.  Every acceptance number in this project is either
measured by Monte-Carlo simulation or predicted by this identity, and
`tests/test_accept.py` checks that the two agree.
"""
from __future__ import annotations

import torch


def _as_prob(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.sum(dim=dim, keepdim=True).clamp_min(1e-12)


def tvd(p: torch.Tensor, q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Total variation distance, 0.5 * L1. Shapes broadcast; reduces `dim`."""
    return 0.5 * (p - q).abs().sum(dim=dim)


def acceptance_prob(p: torch.Tensor, q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exact per-token acceptance probability of speculative sampling."""
    return torch.minimum(p, q).sum(dim=dim)


def forward_kl(p: torch.Tensor, q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """KL(p || q) -- the usual knowledge-distillation direction (mass covering)."""
    return (p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(dim=dim)


def reverse_kl(p: torch.Tensor, q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """KL(q || p) -- mode seeking."""
    return forward_kl(q, p, dim=dim)


def top1_agreement(p: torch.Tensor, q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Greedy acceptance probability: 1 iff argmax agrees."""
    return (p.argmax(dim=dim) == q.argmax(dim=dim)).to(p.dtype)


def expected_tokens_per_step(beta: torch.Tensor | float, gamma: int) -> torch.Tensor | float:
    """Expected tokens emitted per speculative step, including the bonus token.

    Under the standard i.i.d.-acceptance model with `gamma` draft tokens:

        E[emitted] = sum_{i=0}^{gamma} beta^i = (1 - beta^(gamma+1)) / (1 - beta)

    This is the quantity that turns a *linear* acceptance drop into a
    *superlinear* throughput loss, and it is the reason a seemingly small
    LoRA-induced distribution shift is expensive in a speculative serving stack.
    """
    if isinstance(beta, float):
        if abs(1.0 - beta) < 1e-9:
            return float(gamma + 1)
        return (1.0 - beta ** (gamma + 1)) / (1.0 - beta)
    near_one = (1.0 - beta).abs() < 1e-9
    safe = torch.where(near_one, torch.full_like(beta, 0.5), beta)
    out = (1.0 - safe ** (gamma + 1)) / (1.0 - safe)
    return torch.where(near_one, torch.full_like(beta, float(gamma + 1)), out)


def speedup_model(beta: float, gamma: int, cost_ratio: float) -> float:
    """Wall-clock speedup of speculative decoding over autoregressive decoding.

    `cost_ratio` = (one draft forward) / (one target forward).  One speculative
    step costs `gamma * cost_ratio + 1` target-forward-equivalents and emits
    `expected_tokens_per_step(beta, gamma)` tokens; autoregressive decoding
    costs 1 target forward per token.
    """
    emitted = expected_tokens_per_step(beta, gamma)
    step_cost = gamma * cost_ratio + 1.0
    return float(emitted) / step_cost


def relative_weight_shift(delta_w: torch.Tensor, w: torch.Tensor) -> float:
    """||dW||_F / ||W||_F -- the scale-free knob we sweep in E1."""
    return float(delta_w.norm() / w.norm().clamp_min(1e-12))
