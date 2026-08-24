"""Paired-adapter bookkeeping: one tenant id resolves to *two* adapters.

TandemSpec's serving contract is deliberately small:

    tenant  ->  (target-side task LoRA, draft-side companion LoRA)

Both are selected per *request*, so a single batch can mix tenants -- the draft
model runs with a heterogeneous set of companion adapters applied through the
same grouped-gather-matmul path (punica/SGMV) that the target already uses for
task adapters. Nothing here is vLLM-specific; `vllm_integration.py` maps it onto
vLLM's `LoRARequest` / `LoRAMapping` objects.

The scheduler consequence worth stating plainly: `max_loras` becomes a
constraint on *pairs*, not on adapters. A batch admitting K distinct tenants
needs K target-side slots and K draft-side slots. Draft-side slots are ~2-3
orders of magnitude cheaper in bytes, so the binding constraint stays the
target side -- which is why this composes with existing multi-LoRA scheduling
instead of replacing it.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field


def lora_param_count(rank: int, module_shapes: list[tuple[int, int]]) -> int:
    """Parameters in a rank-`rank` LoRA over matrices of the given shapes."""
    return sum(rank * (i + o) for o, i in module_shapes)


@dataclass
class AdapterSpec:
    name: str
    rank: int
    module_shapes: list[tuple[int, int]] = field(default_factory=list)
    path: str | None = None

    @property
    def n_params(self) -> int:
        return lora_param_count(self.rank, self.module_shapes)

    def n_bytes(self, dtype_bytes: int = 2) -> int:
        return self.n_params * dtype_bytes


@dataclass
class PairedAdapter:
    tenant_id: int
    name: str
    target: AdapterSpec
    draft: AdapterSpec | None = None    # None -> tenant falls back to the shared drafter

    def n_bytes(self, dtype_bytes: int = 2) -> int:
        b = self.target.n_bytes(dtype_bytes)
        return b + (self.draft.n_bytes(dtype_bytes) if self.draft else 0)

    @property
    def draft_overhead_frac(self) -> float:
        if not self.draft:
            return 0.0
        return self.draft.n_bytes() / max(self.target.n_bytes(), 1)


class PairedAdapterRegistry:
    """Tenant -> paired adapters, with slot accounting and an LRU residency model."""

    def __init__(self, max_loras: int = 8, max_lora_rank: int = 32,
                 max_draft_lora_rank: int = 8, max_cpu_loras: int | None = None):
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.max_draft_lora_rank = max_draft_lora_rank
        self.max_cpu_loras = max_cpu_loras
        self._by_tenant: dict[int, PairedAdapter] = {}
        self._gpu_resident: OrderedDict[int, None] = OrderedDict()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    # -- registration ------------------------------------------------------
    def register(self, pa: PairedAdapter) -> PairedAdapter:
        if pa.target.rank > self.max_lora_rank:
            raise ValueError(
                f"tenant {pa.tenant_id}: target rank {pa.target.rank} > max_lora_rank "
                f"{self.max_lora_rank}")
        if pa.draft and pa.draft.rank > self.max_draft_lora_rank:
            raise ValueError(
                f"tenant {pa.tenant_id}: draft rank {pa.draft.rank} > max_draft_lora_rank "
                f"{self.max_draft_lora_rank}")
        self._by_tenant[pa.tenant_id] = pa
        return pa

    def get(self, tenant_id: int) -> PairedAdapter | None:
        return self._by_tenant.get(tenant_id)

    def __len__(self) -> int:
        return len(self._by_tenant)

    # -- batching ----------------------------------------------------------
    def admissible(self, tenant_ids: list[int]) -> tuple[list[int], list[int]]:
        """Split a candidate batch into (admitted, deferred) under `max_loras`.

        Distinct tenants -- not distinct requests -- consume slots, and a tenant
        that has *no* companion adapter still consumes a target slot. Requests
        for already-admitted tenants are free.
        """
        admitted, deferred, seen = [], [], set()
        for tid in tenant_ids:
            if tid in seen:
                admitted.append(tid)
            elif len(seen) < self.max_loras:
                seen.add(tid)
                admitted.append(tid)
            else:
                deferred.append(tid)
        return admitted, deferred

    def touch(self, tenant_id: int) -> bool:
        """LRU residency update. Returns True on a GPU-resident hit."""
        if tenant_id in self._gpu_resident:
            self._gpu_resident.move_to_end(tenant_id)
            self.stats["hits"] += 1
            return True
        self.stats["misses"] += 1
        self._gpu_resident[tenant_id] = None
        while len(self._gpu_resident) > self.max_loras:
            self._gpu_resident.popitem(last=False)
            self.stats["evictions"] += 1
        return False

    # -- accounting --------------------------------------------------------
    def memory_report(self, dtype_bytes: int = 2) -> dict:
        tgt = sum(p.target.n_bytes(dtype_bytes) for p in self._by_tenant.values())
        drf = sum(p.draft.n_bytes(dtype_bytes) for p in self._by_tenant.values() if p.draft)
        return {
            "n_tenants": len(self._by_tenant),
            "target_adapter_bytes": tgt,
            "draft_adapter_bytes": drf,
            "total_bytes": tgt + drf,
            "draft_overhead_frac": drf / tgt if tgt else 0.0,
        }


def module_shapes_from_hf_config(hidden: int, intermediate: int, n_layers: int,
                                 n_kv_heads: int | None = None, n_heads: int | None = None,
                                 targets=("q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj")
                                 ) -> list[tuple[int, int]]:
    """(out, in) shapes of the usual LoRA target matrices, for memory accounting."""
    kv = hidden if (n_kv_heads is None or n_heads is None) else hidden * n_kv_heads // n_heads
    table = {
        "q_proj": (hidden, hidden), "k_proj": (kv, hidden), "v_proj": (kv, hidden),
        "o_proj": (hidden, hidden), "gate_proj": (intermediate, hidden),
        "up_proj": (intermediate, hidden), "down_proj": (hidden, intermediate),
    }
    return [table[t] for t in targets if t in table] * n_layers
