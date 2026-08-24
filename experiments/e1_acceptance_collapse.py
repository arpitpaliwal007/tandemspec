"""E1: how far does a shared drafter's acceptance fall when the target wears a LoRA?

For each tenant adapter we sweep a runtime strength multiplier s from 0 (base
model) upward, and at each point we

  * regenerate evaluation sequences from the *adapted* target (on-policy),
  * measure the perturbation the adapter induces, TVD(p_base, p_adapted),
  * measure the shared drafter's acceptance rate beta and E[tokens/step].

The prediction from first-order theory (writeup Section 3) is

    beta(s)  ~=  beta_0 - c * TVD(p_base, p_s),      c ~= 1

i.e. acceptance falls *linearly* in the induced distribution shift, while
E[tokens/step] = (1 - beta^(gamma+1))/(1 - beta) falls *superlinearly*.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
torch.set_num_threads(os.cpu_count() or 2)

from tandemspec.config import PilotConfig
from tandemspec.models import lora as L
from tandemspec.metrics import tvd, expected_tokens_per_step
from tandemspec.eval.acceptance import generate, measure_acceptance, token_probs
from stage1_task_loras import load_stage0, RES

STRENGTHS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 1.15, 1.3]


def main():
    cfg = PilotConfig(); t0 = time.time()
    target, draft = load_stage0()
    adapters = torch.load(f"{RES}/task_adapters.pt", weights_only=False)
    corpus = torch.load(f"{RES}/corpus.pt", weights_only=False)["corpus"]
    g = torch.Generator().manual_seed(2024)
    prompts = corpus[torch.randperm(corpus.shape[0], generator=g)[:cfg.eval_seqs], :cfg.skip_prefix]

    rows = []
    for t, state in adapters.items():
        L.load_adapter(target, state)
        for s in STRENGTHS:
            L.set_strength(target, s)
            seqs = generate(target, prompts, cfg.eval_len, cfg.temperature,
                            generator=torch.Generator().manual_seed(500 + t))
            p_adapted = token_probs(target, seqs, cfg.temperature)[:, cfg.skip_prefix:]
            L.set_strength(target, 0.0)
            p_base = token_probs(target, seqs, cfg.temperature)[:, cfg.skip_prefix:]
            L.set_strength(target, s)
            shift = tvd(p_base, p_adapted).mean().item()

            st = measure_acceptance(target, draft, seqs, cfg.gamma, cfg.temperature,
                                    cfg.skip_prefix, mc_blocks=3000, seed=1000 + t)
            row = {"tenant": t, "strength": s, "shift_tvd": shift,
                   "rel_weight_shift": L.relative_shift(target), **st.as_dict()}
            rows.append(row)
            print(f"tenant {t} s={s:.2f}  shift={shift:.4f}  beta={st.beta_analytic:.4f}  "
                  f"greedy={st.beta_greedy:.4f}  tok/step={st.tokens_per_step:.3f} "
                  f"(iid {st.tokens_per_step_iid:.3f})", flush=True)
        L.clear_adapters(target)

    json.dump({"config": cfg.to_dict(), "strengths": STRENGTHS, "rows": rows,
               "wallclock_s": time.time() - t0},
              open(f"{RES}/e1_acceptance_collapse.json", "w"), indent=2)
    print(f"E1 done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
