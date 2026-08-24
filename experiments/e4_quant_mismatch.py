"""E4: the QLoRA angle -- train-time and serve-time quantisation need not match.

Tenants routinely fine-tune with QLoRA against an nf4/int4 base and then hand
the adapter to a serving stack whose base is fp16, AWQ, FP8 or NVFP4. That
mismatch is a *second* source of distribution shift stacked on top of the
adapter's own shift, and the shared drafter pays for both.

Arms (base weights are RTN int4, group 64, when "int4"):

    fp16-train / fp16-serve   reference
    fp16-train / int4-serve   provider quantises a tenant's fp16-trained adapter
    int4-train / int4-serve   matched QLoRA
    int4-train / fp16-serve   QLoRA adapter deployed on an unquantised base

For each arm we report the drift from the reference model, the shared drafter's
acceptance, and the acceptance after fitting a companion adapter *against the
served configuration* -- i.e. can a 30 KB drafter-side adapter absorb a
quantisation mismatch it never saw at tenant-training time?
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
torch.set_num_threads(os.cpu_count() or 2)

from tandemspec.config import PilotConfig
from tandemspec.data.synth import Batcher
from tandemspec.models import lora as L
from tandemspec.metrics import tvd
from tandemspec.eval.acceptance import generate, measure_acceptance, token_probs
from tandemspec.train.routines import train_task_lora, train_companion_adapter
from stage1_task_loras import load_stage0, RES
from e2_companion_repair import SeqBatcher

N_TENANTS = 3   # E4 is a 4-arm x 2-drafter matrix; 3 tenants keeps it CPU-cheap


def main():
    cfg = PilotConfig(); t0 = time.time()
    target, draft = load_stage0()
    draft_sd = {k: v.clone() for k, v in draft.state_dict().items()}
    d = torch.load(f"{RES}/corpus.pt", weights_only=False)
    corpus, dom_ids = d["corpus"], d["dom_ids"]
    g = torch.Generator().manual_seed(4242)
    pool = corpus[torch.randperm(corpus.shape[0], generator=g)]

    rows = []
    for t in range(N_TENANTS):
        sub = corpus[dom_ids == t]
        # Both adapters are trained here on identical data with an identical
        # seed, so the only difference between them is the precision of the
        # base they were fitted against. (Reusing stage 1's adapter would
        # confound quantisation with a different data ordering.)
        L.clear_adapters(target); L.dequantize_model_(target)
        f_state, _ = train_task_lora(target, Batcher(sub, cfg.batch, seed=300 + t),
                                     r=cfg.task_rank, steps=350,
                                     tag=f"fp16-t{t}", log_every=10**9)
        L.clear_adapters(target); L.quantize_model_(target, bits=4, group_size=64)
        q_state, _ = train_task_lora(target, Batcher(sub, cfg.batch, seed=300 + t),
                                     r=cfg.task_rank, steps=350,
                                     tag=f"qlora-t{t}", log_every=10**9)
        L.clear_adapters(target); L.dequantize_model_(target)

        arms = [("fp16-train/fp16-serve", f_state, False),
                ("fp16-train/int4-serve", f_state, True),
                ("int4-train/int4-serve", q_state, True),
                ("int4-train/fp16-serve", q_state, False)]

        ref_probs = None
        for name, state, quantize in arms:
            L.clear_adapters(target)
            L.dequantize_model_(target)
            if quantize:
                L.quantize_model_(target, bits=4, group_size=64)
            L.load_adapter(target, state); L.set_strength(target, 1.0)

            pr_tr = pool[:256, :cfg.skip_prefix]
            pr_ev = pool[512:512 + cfg.eval_seqs, :cfg.skip_prefix]  # disjoint from the training split
            roll_tr = generate(target, pr_tr, cfg.eval_len, cfg.temperature,
                               generator=torch.Generator().manual_seed(1700 + t))
            roll_ev = generate(target, pr_ev, cfg.eval_len, cfg.temperature,
                               generator=torch.Generator().manual_seed(1900 + t))

            probs = token_probs(target, roll_ev, cfg.temperature)[:, cfg.skip_prefix:]
            if ref_probs is None:
                ref_probs, drift = probs, 0.0
            else:
                drift = tvd(ref_probs, probs).mean().item()

            L.clear_adapters(draft); draft.load_state_dict(draft_sd)
            a_shared = measure_acceptance(target, draft, roll_ev, cfg.gamma, cfg.temperature,
                                          cfg.skip_prefix, mc_blocks=2000, seed=3000 + t)
            _, info = train_companion_adapter(
                draft, target, SeqBatcher(roll_tr, cfg.batch, seed=1800 + t),
                r=cfg.companion_rank, steps=250, loss_kind="tvd",
                temperature=cfg.temperature, tag=f"t{t}-{name}", log_every=10**9)
            a_comp = measure_acceptance(target, draft, roll_ev, cfg.gamma, cfg.temperature,
                                        cfg.skip_prefix, mc_blocks=2000, seed=3000 + t)

            rows.append({"tenant": t, "arm": name, "drift_from_reference_tvd": drift,
                         "beta_shared": a_shared.beta_analytic,
                         "beta_companion": a_comp.beta_analytic,
                         "greedy_shared": a_shared.beta_greedy,
                         "greedy_companion": a_comp.beta_greedy,
                         "tok_step_shared": a_shared.tokens_per_step,
                         "tok_step_companion": a_comp.tokens_per_step,
                         "companion_params": info["n_params"]})
            print(f"t{t} {name:<24} drift={drift:.4f} beta {a_shared.beta_analytic:.4f}"
                  f" -> {a_comp.beta_analytic:.4f}  tok/step {a_shared.tokens_per_step:.3f}"
                  f" -> {a_comp.tokens_per_step:.3f}", flush=True)

    L.clear_adapters(target); L.dequantize_model_(target)
    L.clear_adapters(draft); draft.load_state_dict(draft_sd)
    json.dump({"config": cfg.to_dict(), "rows": rows, "wallclock_s": time.time() - t0},
              open(f"{RES}/e4_quant_mismatch.json", "w"), indent=2)
    print(f"E4 done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
