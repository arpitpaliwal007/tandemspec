"""Serving-side bookkeeping: admission under max_loras and memory accounting."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tandemspec.serving.paired_adapters import (AdapterSpec, PairedAdapter,
                                                PairedAdapterRegistry,
                                                module_shapes_from_hf_config)


def _reg(n=16, max_loras=4):
    t = module_shapes_from_hf_config(4096, 14336, 32, 8, 32)
    d = module_shapes_from_hf_config(2048, 8192, 16, 8, 32)
    r = PairedAdapterRegistry(max_loras=max_loras, max_lora_rank=32, max_draft_lora_rank=8)
    for i in range(n):
        r.register(PairedAdapter(i, f"t{i}", AdapterSpec(f"t{i}", 32, t), AdapterSpec(f"d{i}", 4, d)))
    return r


def test_rank_limits_enforced():
    r = PairedAdapterRegistry(max_lora_rank=16, max_draft_lora_rank=4)
    try:
        r.register(PairedAdapter(0, "x", AdapterSpec("a", 32, [(8, 8)])))
        assert False, "should reject a target rank above max_lora_rank"
    except ValueError:
        pass
    try:
        r.register(PairedAdapter(1, "y", AdapterSpec("a", 8, [(8, 8)]), AdapterSpec("b", 8, [(8, 8)])))
        assert False, "should reject a draft rank above max_draft_lora_rank"
    except ValueError:
        pass


def test_admission_counts_tenants_not_requests():
    r = _reg(max_loras=2)
    admitted, deferred = r.admissible([0, 0, 0, 1, 1, 2, 3])
    assert admitted == [0, 0, 0, 1, 1], admitted
    assert deferred == [2, 3], deferred


def test_draft_overhead_is_small():
    r = _reg()
    m = r.memory_report()
    assert m["draft_overhead_frac"] < 0.06, m
    one = r.get(0)
    private_drafter_bytes = 1.24e9 * 2
    assert private_drafter_bytes / one.draft.n_bytes() > 100
    print(f"  draft overhead {100*m['draft_overhead_frac']:.1f}% of task-adapter memory; "
          f"companion {private_drafter_bytes/one.draft.n_bytes():.0f}x smaller than a private drafter")


def test_lru_residency():
    r = _reg(max_loras=2)
    for t in [0, 1, 0, 2, 0, 1]:
        r.touch(t)
    assert r.stats["evictions"] >= 1
    assert r.stats["hits"] + r.stats["misses"] == 6


if __name__ == "__main__":
    for fn in [test_rank_limits_enforced, test_admission_counts_tenants_not_requests,
               test_draft_overhead_is_small, test_lru_residency]:
        print("running", fn.__name__); fn(); print("  PASS")
