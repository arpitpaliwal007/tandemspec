"""Mapping TandemSpec onto vLLM.

WHAT VLLM ALREADY HAS
---------------------
* Target-side multi-LoRA: `EngineArgs(enable_lora=True, max_loras=..,
  max_lora_rank=.., max_cpu_loras=..)`, per-request `LoRARequest`, batched
  application through the punica/SGMV wrapper in `vllm/lora/punica_wrapper/`,
  and an LRU CPU<->GPU adapter cache in the worker LoRA manager.
* Speculative decoding: proposers under `vllm/v1/spec_decode/` (ngram, EAGLE/
  EAGLE-3, DFlash, draft-model), configured through `speculative_config`.

WHAT IS MISSING (this is the TandemSpec delta)
----------------------------------------------
1. The proposer has no LoRA plumbing at all: whatever adapter the *target*
   request carries is not applied to the drafter. Every tenant therefore drafts
   with the unadapted base drafter.
2. vLLM RFC #52038 (open, filed 2026-08-12) proposes LoRA on DFlash drafters,
   but selects the drafter adapter *per deployment* -- one specialised drafter
   for the whole server. That does not help a multi-tenant server, where a
   single batch contains requests from tenants with different target adapters.
3. Nothing trains the drafter-side adapter against the *adapted* target, so even
   a per-deployment drafter LoRA optimises the wrong objective.

THE CHANGE, CONCRETELY
----------------------
a) `SpeculativeConfig` gains `enable_draft_lora`, `max_draft_lora_rank`, and a
   tenant->draft-adapter resolution hook.
b) `LoRARequest` gains an optional `draft_lora_path` / `draft_lora_int_id`, or
   (cleaner, and what `PairedAdapterRegistry` models) the engine resolves the
   pair from a registry keyed by `lora_int_id`. One id, two adapters -- the
   request API does not change for users who do not opt in.
c) The proposer's model runner instantiates its own `LoRAModelManager` over the
   draft model's linear layers and calls `set_active_loras` with a `LoRAMapping`
   *derived from the same per-request index mapping the target used*, so the
   two stay aligned position-for-position within the batch.
d) The scheduler's LoRA admission check counts pairs. `max_loras` is unchanged
   in meaning (distinct tenants per batch); the draft side allocates the same
   cardinality at a much smaller rank.

WHY THIS IS CHEAP
-----------------
The drafter is 5-20x smaller than the target and the companion rank is 4-8 vs
16-32 on the target side, so draft-side adapter memory is ~3-5% of target-side
adapter memory (see `PairedAdapterRegistry.memory_report`). The drafting
forward pass already runs; adding an SGMV epilogue to its linear layers costs
the same relative overhead multi-LoRA already costs the target.

WHAT THIS MODULE PROVIDES
-------------------------
`to_lora_requests` converts a `PairedAdapter` into the two `LoRARequest`
objects, and `check_vllm_support` reports what the *installed* vLLM can and
cannot do, so the benchmark scripts degrade honestly instead of silently
measuring the wrong thing. vLLM's internals move fast: treat the hook points
above as a specification to re-verify against your version, not as a patch.
"""
from __future__ import annotations

from dataclasses import dataclass

from .paired_adapters import PairedAdapter


@dataclass
class VLLMSupport:
    installed: bool = False
    version: str | None = None
    has_lora: bool = False
    has_spec_decode: bool = False
    has_draft_lora: bool = False        # the gap TandemSpec fills
    notes: list[str] = None

    def summary(self) -> str:
        if not self.installed:
            return "vLLM not installed -- use the transformers harness for acceptance numbers."
        bits = [f"vLLM {self.version}",
                f"multi-LoRA: {'yes' if self.has_lora else 'no'}",
                f"spec decode: {'yes' if self.has_spec_decode else 'no'}",
                f"per-request draft LoRA: {'yes' if self.has_draft_lora else 'NO (TandemSpec gap)'}"]
        return " | ".join(bits) + ("\n  " + "\n  ".join(self.notes or []))


def check_vllm_support() -> VLLMSupport:
    s = VLLMSupport(notes=[])
    try:
        import vllm  # noqa
    except Exception as e:  # pragma: no cover - environment dependent
        s.notes.append(f"import failed: {e}")
        return s
    s.installed = True
    s.version = getattr(vllm, "__version__", "unknown")
    try:
        from vllm.lora.request import LoRARequest  # noqa
        s.has_lora = True
        fields = getattr(LoRARequest, "__dataclass_fields__", {})
        s.has_draft_lora = any("draft" in f for f in fields)
        if not s.has_draft_lora:
            s.notes.append("LoRARequest has no draft-side field: drafter runs unadapted "
                           "for every tenant (this is the acceptance collapse E1 measures).")
    except Exception as e:
        s.notes.append(f"lora import failed: {e}")
    try:
        from vllm.config import SpeculativeConfig  # noqa
        s.has_spec_decode = True
        ann = getattr(SpeculativeConfig, "__annotations__", {})
        if not any("lora" in a for a in ann):
            s.notes.append("SpeculativeConfig exposes no LoRA knobs (see RFC #52038).")
    except Exception as e:
        s.notes.append(f"spec-decode config import failed: {e}")
    return s


def to_lora_requests(pa: PairedAdapter, draft_id_offset: int = 1_000_000):
    """Build the (target, draft) `LoRARequest` pair for one tenant.

    Draft ids live in a disjoint numeric range so the two managers can key their
    caches independently while a single tenant id still resolves both.
    """
    from vllm.lora.request import LoRARequest
    tgt = LoRARequest(lora_name=pa.target.name, lora_int_id=pa.tenant_id + 1,
                      lora_path=pa.target.path)
    drf = None
    if pa.draft is not None:
        drf = LoRARequest(lora_name=pa.draft.name,
                          lora_int_id=draft_id_offset + pa.tenant_id + 1,
                          lora_path=pa.draft.path)
    return tgt, drf
