"""A multi-domain synthetic corpus with a *known* generative process.

Sequences come from a sticky hidden Markov model: a latent "topic" state
persists for several steps and emits tokens from a state-specific distribution.
Two properties make this the right testbed for TandemSpec:

1. **Capacity matters.** Predicting the next token requires Bayesian filtering
   over the latent state from the whole prefix, so a smaller drafter really is
   worse than the target -- the acceptance rate beta_0 of a well-matched
   drafter sits strictly between 0 and 1, exactly as in real deployments.
   (An order-1 Markov source would let both models converge to the same
   conditional and drive beta_0 -> 1, which would hide the effect we study.)

2. **Ground truth is computable.** `HMMSource.exact_predictive` runs the forward
   algorithm, so we can decompose the acceptance gap into the part caused by
   drafter capacity and the part caused by tenant adaptation -- a control that
   is impossible with real pretrained LLMs.

Each tenant domain shares a common backbone (so pretraining transfers) and
mixes in domain-specific transition/emission structure with weight `lam`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def _dirichlet_rows(n_rows: int, n_cols: int, conc: float, g: torch.Generator) -> torch.Tensor:
    x = torch.distributions.Gamma(torch.full((n_rows, n_cols), conc), 1.0).sample()
    return x / x.sum(-1, keepdim=True)


@dataclass
class HMMSource:
    """Sticky HMM over `vocab` tokens with `n_states` latent topics."""
    trans: torch.Tensor     # (K, K)
    emit: torch.Tensor      # (K, V)
    init: torch.Tensor      # (K,)

    @property
    def n_states(self) -> int:
        return self.trans.shape[0]

    @property
    def vocab(self) -> int:
        return self.emit.shape[1]

    def sample(self, n_seq: int, seq_len: int, g: torch.Generator) -> torch.Tensor:
        K = self.n_states
        out = torch.empty(n_seq, seq_len, dtype=torch.long)
        s = torch.multinomial(self.init.expand(n_seq, K), 1, generator=g).squeeze(-1)
        for t in range(seq_len):
            out[:, t] = torch.multinomial(self.emit[s], 1, generator=g).squeeze(-1)
            s = torch.multinomial(self.trans[s], 1, generator=g).squeeze(-1)
        return out

    def exact_predictive(self, seqs: torch.Tensor) -> torch.Tensor:
        """Bayes-optimal p(x_{t+1} | x_{<=t}) for every position. (N, T, V)."""
        N, T = seqs.shape
        alpha = self.init.expand(N, self.n_states).clone()
        preds = torch.empty(N, T, self.vocab)
        for t in range(T):
            alpha = alpha * self.emit[:, seqs[:, t]].t()        # (N, K) filter update
            alpha = alpha / alpha.sum(-1, keepdim=True).clamp_min(1e-30)
            nxt = alpha @ self.trans                              # (N, K) predict
            preds[:, t] = nxt @ self.emit
        return preds


def make_domains(
    n_domains: int = 6,
    vocab: int = 256,
    n_states: int = 8,
    stickiness: float = 0.94,
    lam: float = 0.6,
    emit_conc: float = 0.06,
    seed: int = 0,
) -> tuple[list[HMMSource], HMMSource]:
    """Build `n_domains` tenant sources plus the shared backbone source.

    Domains are *logit-space* perturbations of the backbone:

        emit_d  = softmax( log emit_base  + lam * eps ),   eps ~ N(0, 1)

    which gives smooth, continuous control over how far a tenant's domain sits
    from the pretraining mixture, instead of the abrupt support change you get
    from mixing in a fresh sparse Dirichlet draw.
    """
    g = torch.Generator().manual_seed(seed)
    K, V = n_states, vocab

    base_trans = _dirichlet_rows(K, K, 0.5, g)
    base_trans = stickiness * torch.eye(K) + (1 - stickiness) * base_trans
    base_trans = base_trans / base_trans.sum(-1, keepdim=True)
    base_emit = _dirichlet_rows(K, V, emit_conc, g)
    init = torch.full((K,), 1.0 / K)
    backbone = HMMSource(base_trans, base_emit, init)

    le, lt = base_emit.clamp_min(1e-12).log(), base_trans.clamp_min(1e-12).log()
    domains = []
    for _ in range(n_domains):
        emit = torch.softmax(le + lam * torch.randn(K, V, generator=g), dim=-1)
        trans = torch.softmax(lt + 0.5 * lam * torch.randn(K, K, generator=g), dim=-1)
        domains.append(HMMSource(trans, emit, init))
    return domains, backbone


def build_corpus(sources: list[HMMSource], n_seq_per_domain: int, seq_len: int,
                 seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (sequences, domain_ids)."""
    g = torch.Generator().manual_seed(seed)
    xs, ds = [], []
    for i, s in enumerate(sources):
        x = s.sample(n_seq_per_domain, seq_len, g)
        xs.append(x)
        ds.append(torch.full((n_seq_per_domain,), i, dtype=torch.long))
    return torch.cat(xs), torch.cat(ds)


class Batcher:
    def __init__(self, data: torch.Tensor, batch_size: int, seed: int = 0):
        self.data, self.bs = data, batch_size
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self) -> torch.Tensor:
        idx = torch.randint(0, self.data.shape[0], (self.bs,), generator=self.g)
        return self.data[idx]
