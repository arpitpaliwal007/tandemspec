"""Measuring acceptance the way a serving stack actually experiences it.

Two measurement modes, and we check they agree:

* **teacher-forced** (`measure_acceptance`): score target and drafter on the
  same sequences and average sum_v min(p_v, q_v). Cheap, low variance.
* **rollout** (`speculative_decode`): run the real draft/verify loop and count
  accepted tokens. Expensive, but it is the ground truth.

Both are evaluated on sequences sampled from the *adapted target*. That is the
correct on-policy distribution: speculative decoding is distribution-preserving,
so the contexts a drafter sees in deployment are distributed exactly as the
target's own output -- no drafter rollouts are needed to be on-policy. This is
the observation that makes companion-adapter training cheap (Section 4 of the
writeup).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..accept import AcceptanceStats, simulate_block, simulate_block_greedy
from ..metrics import acceptance_prob, forward_kl, top1_agreement, tvd, expected_tokens_per_step


@torch.no_grad()
def generate(model, prompts: torch.Tensor, n_new: int, temperature: float = 1.0,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """Plain ancestral sampling. Short sequences + tiny models -> no KV cache."""
    model.eval()
    seq = prompts.clone()
    for _ in range(n_new):
        logits = model(seq)[:, -1] / max(temperature, 1e-6)
        nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator)
        seq = torch.cat([seq, nxt], dim=1)
    return seq


@torch.no_grad()
def token_probs(model, seqs: torch.Tensor, temperature: float = 1.0,
                batch_size: int = 64) -> torch.Tensor:
    """(N, T, V) next-token distributions under teacher forcing."""
    model.eval()
    outs = []
    for i in range(0, seqs.shape[0], batch_size):
        logits = model(seqs[i:i + batch_size]) / max(temperature, 1e-6)
        outs.append(logits.softmax(-1))
    return torch.cat(outs)


@torch.no_grad()
def measure_acceptance(target, draft, seqs: torch.Tensor, gamma: int = 4,
                       temperature: float = 1.0, skip_prefix: int = 8,
                       mc_blocks: int = 0, seed: int = 0) -> AcceptanceStats:
    """Acceptance statistics of `draft` proposing to `target` on `seqs`."""
    p = token_probs(target, seqs, temperature)[:, skip_prefix:]
    q = token_probs(draft, seqs, temperature)[:, skip_prefix:]
    beta = acceptance_prob(p, q).mean().item()
    st = AcceptanceStats(
        beta_analytic=beta,
        beta_greedy=top1_agreement(p, q).mean().item(),
        tokens_per_step_iid=float(expected_tokens_per_step(beta, gamma)),
        n_positions=int(p.shape[0] * p.shape[1]),
    )
    st.extras["tvd"] = tvd(p, q).mean().item()
    st.extras["fkl"] = forward_kl(p, q).mean().item()
    st.extras["target_entropy"] = float(-(p.clamp_min(1e-12).log() * p).sum(-1).mean())

    if mc_blocks > 0:
        g = torch.Generator().manual_seed(seed)
        N, T, _ = p.shape
        acc, n = 0, 0
        for _ in range(mc_blocks):
            i = int(torch.randint(0, N, (1,), generator=g))
            t = int(torch.randint(0, T - gamma - 1, (1,), generator=g))
            pb, qb = p[i, t:t + gamma + 1], q[i, t:t + gamma]
            dt = torch.multinomial(qb, 1, generator=g).squeeze(-1)
            r = simulate_block(pb, qb, dt, generator=g)
            acc += r.n_accepted
            n += 1
        st.mean_accepted = acc / max(n, 1)
        st.tokens_per_step = st.mean_accepted + 1.0
        st.n_blocks = n
        st.beta_empirical = st.mean_accepted / gamma
    return st


@torch.no_grad()
def speculative_decode(target, draft, prompt: torch.Tensor, n_tokens: int, gamma: int = 4,
                       temperature: float = 1.0, greedy: bool = False,
                       generator: torch.Generator | None = None) -> dict:
    """The real draft/verify loop. Returns emitted tokens and per-block stats."""
    target.eval(); draft.eval()
    seq = prompt.clone()
    n_accept, n_prop, n_blocks, emitted_total = 0, 0, 0, 0
    while emitted_total < n_tokens:
        # 1. drafter proposes gamma tokens autoregressively
        draft_seq = seq.clone()
        qs = []
        for _ in range(gamma):
            ql = draft(draft_seq)[:, -1] / max(temperature, 1e-6)
            qd = ql.softmax(-1)[0]
            nxt = int(qd.argmax()) if greedy else int(torch.multinomial(qd, 1, generator=generator))
            qs.append(qd)
            draft_seq = torch.cat([draft_seq, torch.tensor([[nxt]])], dim=1)
        dtok = draft_seq[0, -gamma:]
        # 2. one target forward verifies all gamma positions + the bonus
        pl = target(draft_seq)[:, -(gamma + 1):] / max(temperature, 1e-6)
        pb = pl.softmax(-1)[0]
        qb = torch.stack(qs)
        res = (simulate_block_greedy(pb, qb, dtok) if greedy
               else simulate_block(pb, qb, dtok, generator=generator))
        seq = torch.cat([seq, torch.tensor([res.emitted])], dim=1)
        n_accept += res.n_accepted; n_prop += gamma; n_blocks += 1
        emitted_total += len(res.emitted)
    return {
        "sequence": seq,
        "mean_accepted": n_accept / n_blocks,
        "beta_empirical": n_accept / n_prop,
        "tokens_per_step": emitted_total / n_blocks,
        "n_blocks": n_blocks,
    }
