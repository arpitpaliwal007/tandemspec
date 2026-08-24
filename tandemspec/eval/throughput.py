"""Turning an acceptance rate into wall-clock and memory numbers.

Decode is memory-bandwidth bound, so a forward pass costs roughly what its
weights cost to read. That makes the draft/target cost ratio ~= the parameter
ratio, which is what `cost_ratio_from_params` returns. Two drafter families
need different step-cost models:

* **sequential** (draft model, EAGLE/EAGLE-3): gamma drafter forwards per step.
* **block** (DFlash and other block-diffusion drafters): one drafter forward
  proposes the whole block, so the step cost stops growing with gamma.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..metrics import expected_tokens_per_step


def cost_ratio_from_params(draft_params: float, target_params: float,
                           target_bits: int = 16, draft_bits: int = 16) -> float:
    """Draft-forward / target-forward cost in the memory-bound decode regime."""
    return (draft_params * draft_bits) / (target_params * target_bits)


def step_cost(gamma: int, cost_ratio: float, mode: str = "sequential",
              verify_overhead: float = 0.0) -> float:
    """Cost of one speculative step in target-forward-equivalents."""
    draft = gamma * cost_ratio if mode == "sequential" else cost_ratio
    return draft + 1.0 + verify_overhead


def speedup(beta: float, gamma: int, cost_ratio: float, mode: str = "sequential",
            verify_overhead: float = 0.0) -> float:
    return float(expected_tokens_per_step(beta, gamma)) / step_cost(
        gamma, cost_ratio, mode, verify_overhead)


def best_gamma(beta: float, cost_ratio: float, mode: str = "sequential",
               gammas=range(1, 17)) -> tuple[int, float]:
    """Speculative decoding has an optimal gamma, and it *moves* with beta.

    A tenant whose acceptance has collapsed should be drafting fewer tokens; a
    server that fixes gamma globally therefore loses twice -- once to the lower
    acceptance and once to running at the wrong operating point.
    """
    scored = [(g, speedup(beta, g, cost_ratio, mode)) for g in gammas]
    return max(scored, key=lambda x: x[1])


@dataclass
class ServingScenario:
    name: str
    target_params: float
    draft_params: float
    target_bits: int = 16
    draft_bits: int = 16
    target_rank: int = 32
    draft_rank: int = 4
    mode: str = "sequential"

    @property
    def cost_ratio(self) -> float:
        return cost_ratio_from_params(self.draft_params, self.target_params,
                                      self.target_bits, self.draft_bits)

    def report(self, beta_shared: float, beta_companion: float, gamma: int = 4,
               n_tenants: int = 64, target_adapter_bytes: float = 0.0,
               draft_adapter_bytes: float = 0.0) -> dict:
        s_shared = speedup(beta_shared, gamma, self.cost_ratio, self.mode)
        s_comp = speedup(beta_companion, gamma, self.cost_ratio, self.mode)
        g_sh, s_sh_opt = best_gamma(beta_shared, self.cost_ratio, self.mode)
        g_co, s_co_opt = best_gamma(beta_companion, self.cost_ratio, self.mode)
        private_drafter_bytes = self.draft_params * self.draft_bits / 8
        return {
            "scenario": self.name,
            "cost_ratio": self.cost_ratio,
            "gamma": gamma,
            "beta_shared": beta_shared,
            "beta_companion": beta_companion,
            "speedup_shared": s_shared,
            "speedup_companion": s_comp,
            "speedup_gain": s_comp / s_shared if s_shared else float("nan"),
            "best_gamma_shared": g_sh, "best_speedup_shared": s_sh_opt,
            "best_gamma_companion": g_co, "best_speedup_companion": s_co_opt,
            "n_tenants": n_tenants,
            "companion_bytes_per_tenant": draft_adapter_bytes,
            "task_adapter_bytes_per_tenant": target_adapter_bytes,
            "private_drafter_bytes_per_tenant": private_drafter_bytes,
            "companion_vs_private_drafter": (private_drafter_bytes / draft_adapter_bytes
                                             if draft_adapter_bytes else float("nan")),
            "companion_overhead_frac_of_task_adapters": (
                draft_adapter_bytes / target_adapter_bytes if target_adapter_bytes else 0.0),
            "total_companion_gb_all_tenants": n_tenants * draft_adapter_bytes / 1e9,
            "total_private_drafter_gb_all_tenants": n_tenants * private_drafter_bytes / 1e9,
        }


SCENARIOS = [
    ServingScenario("Llama-3.1-8B + Llama-3.2-1B draft", 8.03e9, 1.24e9),
    ServingScenario("Llama-3.1-8B + EAGLE-3 head", 8.03e9, 0.25e9),
    ServingScenario("Llama-3.1-8B + DFlash block drafter", 8.03e9, 0.4e9, mode="block"),
    ServingScenario("Qwen3-32B + Qwen3-1.7B draft", 32.8e9, 1.7e9),
    ServingScenario("Llama-3.3-70B + Llama-3.2-3B draft", 70.6e9, 3.2e9),
    # Quantising the target makes the drafter relatively *more* expensive, which
    # shrinks the optimal block size before any acceptance loss is considered.
    ServingScenario("Llama-3.1-8B NVFP4 + EAGLE-3 head fp16", 8.03e9, 0.25e9, target_bits=4),
]
