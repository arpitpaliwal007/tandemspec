"""Exact speculative-sampling acceptance: Monte-Carlo simulator + closed forms.

`simulate_block` implements the accept/reject rule of Leviathan et al. (2023)
verbatim, including the residual distribution (p - q)_+ on rejection, so that
the simulated *output* distribution provably equals the target distribution.
`tests/test_accept.py` checks both properties empirically:

  1. losslessness  -- emitted-token histogram matches the target distribution
  2. acceptance    -- measured E[accepted] matches sum_v min(p_v, q_v) closed form
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .metrics import acceptance_prob, expected_tokens_per_step


@dataclass
class BlockResult:
    n_accepted: int
    emitted: list[int]
    per_position_accept: list[bool]


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """norm((p - q)_+) -- the corrected distribution sampled on rejection."""
    r = (p - q).clamp_min(0.0)
    s = r.sum()
    if s <= 1e-12:
        # degenerate (p == q up to fp error): fall back to p
        return p / p.sum().clamp_min(1e-12)
    return r / s


def simulate_block(
    p: torch.Tensor,           # (gamma+1, V) target distributions
    q: torch.Tensor,           # (gamma,   V) draft distributions
    draft_tokens: torch.Tensor,  # (gamma,) tokens sampled from q
    generator: torch.Generator | None = None,
) -> BlockResult:
    """Run one speculative block. Returns accepted count and emitted tokens."""
    gamma = draft_tokens.shape[0]
    emitted: list[int] = []
    flags: list[bool] = []
    for i in range(gamma):
        x = int(draft_tokens[i])
        p_x, q_x = float(p[i, x]), float(q[i, x])
        ratio = 1.0 if q_x <= 1e-12 else min(1.0, p_x / q_x)
        u = float(torch.rand(1, generator=generator))
        if u < ratio:
            emitted.append(x)
            flags.append(True)
        else:
            res = residual_distribution(p[i], q[i])
            emitted.append(int(torch.multinomial(res, 1, generator=generator)))
            flags.append(False)
            return BlockResult(n_accepted=i, emitted=emitted, per_position_accept=flags)
    # all gamma accepted -> free bonus token from the target
    emitted.append(int(torch.multinomial(p[gamma], 1, generator=generator)))
    return BlockResult(n_accepted=gamma, emitted=emitted, per_position_accept=flags)


def simulate_block_greedy(
    p: torch.Tensor, q: torch.Tensor, draft_tokens: torch.Tensor
) -> BlockResult:
    """Temperature-0 variant: accept iff the draft token is the target argmax."""
    gamma = draft_tokens.shape[0]
    emitted: list[int] = []
    flags: list[bool] = []
    for i in range(gamma):
        tgt = int(p[i].argmax())
        if int(draft_tokens[i]) == tgt:
            emitted.append(tgt)
            flags.append(True)
        else:
            emitted.append(tgt)
            flags.append(False)
            return BlockResult(n_accepted=i, emitted=emitted, per_position_accept=flags)
    emitted.append(int(p[gamma].argmax()))
    return BlockResult(n_accepted=gamma, emitted=emitted, per_position_accept=flags)


@dataclass
class AcceptanceStats:
    """Aggregated acceptance statistics over many positions/blocks."""
    beta_analytic: float = 0.0        # mean of sum_v min(p,q) over contexts
    beta_empirical: float = 0.0       # measured fraction of accepted proposals
    beta_greedy: float = 0.0          # measured top-1 agreement
    mean_accepted: float = 0.0        # measured E[n accepted] per block
    tokens_per_step: float = 0.0      # measured E[n accepted] + 1
    tokens_per_step_iid: float = 0.0  # closed form from beta_analytic
    n_blocks: int = 0
    n_positions: int = 0
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "extras"}
        d.update(self.extras)
        return d


def analytic_block_stats(p_all: torch.Tensor, q_all: torch.Tensor, gamma: int) -> dict:
    """Closed-form acceptance statistics from aligned (N, V) distribution pairs.

    Assumes teacher-forced alignment, i.e. both models are scored on the same
    context, which is the regime that matters: because speculative decoding is
    distribution-preserving, the contexts the drafter is evaluated on are drawn
    from the *target's own* output distribution.
    """
    beta = acceptance_prob(p_all, q_all).mean().item()
    return {
        "beta_analytic": beta,
        "tokens_per_step_iid": float(expected_tokens_per_step(beta, gamma)),
    }
