"""E2: repair the acceptance collapse with a tiny companion adapter on the drafter.

Every arm trains (or does not train) something on the drafter side for one
tenant, then measures acceptance on held-out sequences drawn from that tenant's
adapted target. The arms are chosen to separate three confounds:

  * does the *teacher* matter (adapted target vs. base target)?
  * does the *data* matter (on-policy target rollouts vs. the tenant's corpus)?
  * does the *loss* matter (TVD, which is exactly 1 - beta, vs. KL)?

`full-ft` is the memory-unbounded upper bound: a whole private drafter per
tenant. The point of the experiment is how close a few hundred kilobytes gets
to it.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
torch.set_num_threads(os.cpu_count() or 2)

from tandemspec.config import PilotConfig
from tandemspec.models import lora as L
from tandemspec.eval.acceptance import generate, measure_acceptance
from tandemspec.train.routines import train_companion_adapter
from stage1_task_loras import load_stage0, RES


class SeqBatcher:
    def __init__(self, data, bs, seed):
        self.d, self.bs = data, bs
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self):
        return self.d[torch.randint(0, self.d.shape[0], (self.bs,), generator=self.g)]


def main():
    cfg = PilotConfig(); t0 = time.time()
    target, draft = load_stage0()
    draft_sd = {k: v.clone() for k, v in draft.state_dict().items()}
    adapters = torch.load(f"{RES}/task_adapters.pt", weights_only=False)
    d = torch.load(f"{RES}/corpus.pt", weights_only=False)
    corpus, dom_ids = d["corpus"], d["dom_ids"]
    g = torch.Generator().manual_seed(31337)
    prompt_pool = corpus[torch.randperm(corpus.shape[0], generator=g)]

    ARMS = [
        dict(name="shared-drafter",        train=False),
        dict(name="companion-tvd-r4",      loss="tvd", rank=4, policy="on"),
        dict(name="companion-fkl-r4",      loss="fkl", rank=4, policy="on"),
        dict(name="companion-rkl-r4",      loss="rkl", rank=4, policy="on"),
        dict(name="companion-ce-r4",       loss="ce",  rank=4, policy="corpus"),
        dict(name="companion-tvd-r4-offpolicy", loss="tvd", rank=4, policy="corpus"),
        dict(name="companion-tvd-r1",      loss="tvd", rank=1, policy="on"),
        dict(name="companion-tvd-r2",      loss="tvd", rank=2, policy="on"),
        dict(name="companion-tvd-r8",      loss="tvd", rank=8, policy="on"),
        dict(name="full-ft-tvd",           loss="tvd", rank=0, policy="on", full=True),
    ]

    rows = []
    for t, state in adapters.items():
        L.load_adapter(target, state); L.set_strength(target, 1.0)
        # on-policy rollouts from the ADAPTED target: train and eval splits
        pr_tr = prompt_pool[:512, :cfg.skip_prefix]
        pr_ev = prompt_pool[512:512 + cfg.eval_seqs, :cfg.skip_prefix]
        roll_tr = generate(target, pr_tr, cfg.eval_len, cfg.temperature,
                           generator=torch.Generator().manual_seed(700 + t))
        roll_ev = generate(target, pr_ev, cfg.eval_len, cfg.temperature,
                           generator=torch.Generator().manual_seed(900 + t))
        corp_t = corpus[dom_ids == t]
        print(f"=== tenant {t}: rollouts {tuple(roll_tr.shape)} ===", flush=True)

        for arm in ARMS:
            L.clear_adapters(draft); draft.load_state_dict(draft_sd)
            n_params, secs = 0, 0.0
            if arm["train"] if "train" in arm else True:
                data = roll_tr if arm["policy"] == "on" else corp_t
                ts = time.time()
                st, info = train_companion_adapter(
                    draft, target, SeqBatcher(data, cfg.batch, seed=800 + t),
                    r=max(arm["rank"], 1), steps=cfg.companion_steps,
                    loss_kind=arm["loss"], temperature=cfg.temperature,
                    tag=f"t{t}-{arm['name']}", full_finetune=arm.get("full", False),
                    log_every=10**9)
                secs = time.time() - ts
                n_params = info["n_params"]
            acc = measure_acceptance(target, draft, roll_ev, cfg.gamma, cfg.temperature,
                                     cfg.skip_prefix, mc_blocks=3000, seed=2000 + t)
            row = {"tenant": t, "arm": arm["name"], "rank": arm.get("rank", 0),
                   "loss": arm.get("loss", "-"), "policy": arm.get("policy", "-"),
                   "extra_params": n_params, "extra_bytes_fp16": n_params * 2,
                   "train_s": secs, **acc.as_dict()}
            rows.append(row)
            print(f"  {arm['name']:<26} beta={acc.beta_analytic:.4f} "
                  f"greedy={acc.beta_greedy:.4f} tok/step={acc.tokens_per_step:.3f} "
                  f"params={n_params}", flush=True)
        L.clear_adapters(target)

    L.clear_adapters(draft); draft.load_state_dict(draft_sd)
    json.dump({"config": cfg.to_dict(), "rows": rows,
               "draft_params": draft.n_params(), "target_params": target.n_params(),
               "wallclock_s": time.time() - t0},
              open(f"{RES}/e2_companion_repair.json", "w"), indent=2)
    print(f"E2 done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
