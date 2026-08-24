"""E3: from acceptance rates to serving numbers.

Takes the *measured* beta values from E1/E2 and pushes them through the
step-cost model for realistic target/drafter pairs, plus the memory accounting
for a fleet of tenants. Also simulates adapter residency under Zipf tenant
traffic, because the honest question about adding a second adapter per tenant
is not "does it fit" but "does it thrash".
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from tandemspec.eval.throughput import SCENARIOS, speedup, best_gamma
from tandemspec.serving.paired_adapters import (PairedAdapter, AdapterSpec,
                                                PairedAdapterRegistry,
                                                module_shapes_from_hf_config)

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# (hidden, intermediate, layers, n_heads, n_kv_heads) for the scenario models
GEOM = {
    "Llama-3.1-8B + Llama-3.2-1B draft":      ((4096, 14336, 32, 32, 8), (2048, 8192, 16, 32, 8)),
    "Llama-3.1-8B + EAGLE-3 head":            ((4096, 14336, 32, 32, 8), (4096, 14336, 1, 32, 8)),
    "Llama-3.1-8B + DFlash block drafter":    ((4096, 14336, 32, 32, 8), (4096, 14336, 2, 32, 8)),
    "Qwen3-32B + Qwen3-1.7B draft":           ((5120, 25600, 64, 40, 8), (2048, 6144, 28, 16, 8)),
    "Llama-3.3-70B + Llama-3.2-3B draft":     ((8192, 28672, 80, 64, 8), (3072, 8192, 28, 24, 8)),
    "Llama-3.1-8B NVFP4 + EAGLE-3 head fp16": ((4096, 14336, 32, 32, 8), (4096, 14336, 1, 32, 8)),
}


def zipf_trace(n_tenants, n_requests, alpha=1.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = 1.0 / torch.arange(1, n_tenants + 1).float() ** alpha
    return torch.multinomial(w / w.sum(), n_requests, replacement=True, generator=g).tolist()


def main():
    e1 = json.load(open(f"{RES}/e1_acceptance_collapse.json"))
    e2 = json.load(open(f"{RES}/e2_companion_repair.json"))

    base_beta = json.load(open(f"{RES}/stage0.json"))["base_acceptance"]["beta_analytic"]
    s1 = [r for r in e1["rows"] if r["strength"] == 1.0]
    beta_shared = sum(r["beta_analytic"] for r in s1) / len(s1)
    comp = [r for r in e2["rows"] if r["arm"] == "companion-tvd-r4"]
    beta_comp = sum(r["beta_analytic"] for r in comp) / len(comp)
    print(f"beta: base-model {base_beta:.4f} | adapted+shared drafter {beta_shared:.4f} "
          f"| adapted+companion {beta_comp:.4f}")

    reports = []
    for sc in SCENARIOS:
        tg, dg = GEOM[sc.name]
        tshapes = module_shapes_from_hf_config(tg[0], tg[1], tg[2], tg[4], tg[3])
        dshapes = module_shapes_from_hf_config(dg[0], dg[1], dg[2], dg[4], dg[3])
        tgt_bytes = AdapterSpec("t", sc.target_rank, tshapes).n_bytes()
        drf_bytes = AdapterSpec("d", sc.draft_rank, dshapes).n_bytes()
        rep = sc.report(beta_shared, beta_comp, gamma=4, n_tenants=64,
                        target_adapter_bytes=tgt_bytes, draft_adapter_bytes=drf_bytes)
        rep["speedup_base_model_upper_bound"] = speedup(base_beta, 4, sc.cost_ratio, sc.mode)
        gain = rep["speedup_companion"] - rep["speedup_shared"]
        lost = rep["speedup_base_model_upper_bound"] - rep["speedup_shared"]
        rep["fraction_of_lost_speedup_recovered"] = gain / lost if lost > 1e-9 else float("nan")
        reports.append(rep)
        print(f"{sc.name:<42} c={sc.cost_ratio:.3f} "
              f"speedup {rep['speedup_shared']:.2f}x -> {rep['speedup_companion']:.2f}x "
              f"(base-model ceiling {rep['speedup_base_model_upper_bound']:.2f}x) "
              f"| companion {drf_bytes/1e6:.1f} MB vs private drafter "
              f"{rep['private_drafter_bytes_per_tenant']/1e9:.2f} GB")

    # gamma is not a constant: the right block size moves with acceptance
    gammas = {name: {"shared": best_gamma(beta_shared, sc.cost_ratio, sc.mode),
                     "companion": best_gamma(beta_comp, sc.cost_ratio, sc.mode)}
              for sc, name in ((s, s.name) for s in SCENARIOS)}

    # adapter residency: does the second adapter per tenant cause thrash?
    cache = []
    for n_tenants in (8, 16, 32, 64, 128):
        for max_loras in (4, 8, 16, 32):
            reg = PairedAdapterRegistry(max_loras=max_loras)
            for t in zipf_trace(n_tenants, 20000, seed=n_tenants):
                reg.touch(t)
            tot = reg.stats["hits"] + reg.stats["misses"]
            cache.append({"n_tenants": n_tenants, "max_loras": max_loras,
                          "hit_rate": reg.stats["hits"] / tot,
                          "evictions_per_1k": reg.stats["evictions"] / tot * 1000})

    out = {"beta_base_model": base_beta, "beta_shared": beta_shared,
           "beta_companion": beta_comp, "scenarios": reports,
           "best_gamma": {k: {kk: list(vv) for kk, vv in v.items()} for k, v in gammas.items()},
           "adapter_cache": cache}
    json.dump(out, open(f"{RES}/e3_serving_model.json", "w"), indent=2)
    print("\nadapter residency (Zipf a=1.1, paired adapters share one slot per tenant):")
    for c in cache:
        if c["max_loras"] == 8:
            print(f"  {c['n_tenants']:>3} tenants, max_loras=8: hit rate {c['hit_rate']:.3f}, "
                  f"{c['evictions_per_1k']:.1f} evictions / 1k requests")


if __name__ == "__main__":
    main()
