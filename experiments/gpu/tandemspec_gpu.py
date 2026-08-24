"""TandemSpec on real models, sized for a 16 GB Colab T4.

Default pair: Qwen2.5-1.5B-Instruct (target) + Qwen2.5-0.5B-Instruct (drafter).
Same tokenizer, ~3x parameter ratio, both fp16 -> ~4.2 GB resident, leaving
room for QLoRA tenant training on the same GPU. Pass --target/--draft to move
to a bigger pair (e.g. a 4-bit 7B target) if you have more memory.

Stages, run in order:

    python tandemspec_gpu.py --stage tenants      # QLoRA fine-tune one adapter per task
    python tandemspec_gpu.py --stage e1           # acceptance collapse + strength sweep
    python tandemspec_gpu.py --stage e2           # companion adapters and repair
    python tandemspec_gpu.py --stage throughput   # measured ITL / speculative speedup

Everything writes JSON into --out, which the dashboard and the writeup read.
"""
from __future__ import annotations

import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from tandemspec.metrics import acceptance_prob, tvd, top1_agreement, expected_tokens_per_step
from tandemspec.models.hf import (HFWrapper, load_causal_lm, attach_lora,
                                  adapter_param_count, generate_batched)
from tandemspec.train.routines import companion_loss

# --------------------------------------------------------------------------
# Tenant task definitions. Each entry is a distinct "domain" a tenant would
# fine-tune on; between them they should move the target's output distribution
# in genuinely different directions.
# --------------------------------------------------------------------------
TASKS = [
    dict(name="math",    ds="openai/gsm8k",                     cfg="main",  split="train",
         prompt="question", answer="answer"),
    dict(name="sql",     ds="b-mc2/sql-create-context",         cfg=None,    split="train",
         prompt="question", answer="answer", context="context"),
    dict(name="dialog",  ds="knkarthick/dialogsum",             cfg=None,    split="train",
         prompt="dialogue", answer="summary"),
    dict(name="code",    ds="iamtarun/python_code_instructions_18k_alpaca", cfg=None, split="train",
         prompt="instruction", answer="output"),
]


def build_examples(task, tok, n, max_len=384, seed=0):
    from datasets import load_dataset
    ds = load_dataset(task["ds"], task["cfg"], split=task["split"]) if task["cfg"] \
        else load_dataset(task["ds"], split=task["split"])
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    texts = []
    for r in ds:
        q = r[task["prompt"]]
        if task.get("context"):
            q = f"{r[task['context']]}\n\n{q}"
        msgs = [{"role": "user", "content": str(q)[:1500]},
                {"role": "assistant", "content": str(r[task["answer"]])[:1500]}]
        texts.append(tok.apply_chat_template(msgs, tokenize=False))
    enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True,
              max_length=max_len)
    return enc["input_ids"]


def prompts_only(task, tok, n, max_len=128, seed=1):
    from datasets import load_dataset
    ds = load_dataset(task["ds"], task["cfg"], split=task["split"]) if task["cfg"] \
        else load_dataset(task["ds"], split=task["split"])
    ds = ds.shuffle(seed=seed + 777).select(range(min(n, len(ds))))
    texts = []
    for r in ds:
        q = r[task["prompt"]]
        if task.get("context"):
            q = f"{r[task['context']]}\n\n{q}"
        texts.append(tok.apply_chat_template([{"role": "user", "content": str(q)[:1200]}],
                                             tokenize=False, add_generation_prompt=True))
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True,
              max_length=max_len)
    tok.padding_side = "right"
    return enc["input_ids"]


# --------------------------------------------------------------------------
# Measurement -- identical maths to the CPU pilot, batched for GPU
# --------------------------------------------------------------------------
@torch.no_grad()
def measure(target: HFWrapper, draft: HFWrapper, seqs, gamma=4, temperature=1.0,
            skip=8, bs=4):
    betas, greedy, tvds, ents = [], [], [], []
    for i in range(0, seqs.shape[0], bs):
        x = seqs[i:i + bs].to(target.device)
        p = (target(x)[:, skip:].float() / temperature).softmax(-1)
        q = (draft(x)[:, skip:].float() / temperature).softmax(-1)
        betas.append(acceptance_prob(p, q).mean().item())
        greedy.append(top1_agreement(p, q).mean().item())
        tvds.append(tvd(p, q).mean().item())
        ents.append(float(-(p.clamp_min(1e-12).log() * p).sum(-1).mean()))
        del p, q
    b = sum(betas) / len(betas)
    return {"beta": b, "beta_greedy": sum(greedy) / len(greedy),
            "tvd": sum(tvds) / len(tvds), "target_entropy": sum(ents) / len(ents),
            "tokens_per_step_iid": float(expected_tokens_per_step(b, gamma))}


def set_lora_strength(peft_model, s: float, cache: dict | None = None):
    """Scale every LoRA B matrix by `s` (0 -> base model). Restores from `cache`."""
    if cache is None:
        cache = {}
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_B" in n:
                if n not in cache:
                    cache[n] = p.detach().clone()
                p.copy_(cache[n] * s)
    return cache


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def stage_tenants(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.target)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    os.makedirs(f"{a.out}/adapters", exist_ok=True)
    for task in TASKS[:a.n_tenants]:
        print(f"=== QLoRA tenant: {task['name']} ===", flush=True)
        base = load_causal_lm(a.target, four_bit=True, attn_impl="sdpa")
        model = attach_lora(base, r=a.task_rank, alpha=2 * a.task_rank)
        model.print_trainable_parameters()
        ids = build_examples(task, tok, a.n_train, a.max_len, seed=0)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
        model.train(); t0 = time.time()
        for step in range(a.tenant_steps):
            x = ids[torch.randint(0, ids.shape[0], (a.bs,))].to(model.device)
            out = model(input_ids=x, labels=x)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            if step % 25 == 0:
                print(f"  {step}/{a.tenant_steps} loss={out.loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        model.save_pretrained(f"{a.out}/adapters/{task['name']}")
        del model, base; torch.cuda.empty_cache()


def _load_pair(a, tenant=None, four_bit_target=False):
    from peft import PeftModel
    base = load_causal_lm(a.target, four_bit=four_bit_target, attn_impl="sdpa")
    if tenant:
        base = PeftModel.from_pretrained(base, f"{a.out}/adapters/{tenant}", is_trainable=False)
    draft = load_causal_lm(a.draft, attn_impl="sdpa")
    return HFWrapper(base), HFWrapper(draft)


def stage_e1(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.target)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = []
    for task in TASKS[:a.n_tenants]:
        T, D = _load_pair(a, task["name"], four_bit_target=a.four_bit_serve)
        cache = None
        pr = prompts_only(task, tok, a.n_eval, seed=1).to(T.device)
        for s in [float(x) for x in a.strengths.split(",")]:
            cache = set_lora_strength(T.model, s, cache)
            seqs = generate_batched(T, pr[:a.n_eval], a.gen_len, a.temperature)
            m = measure(T, D, seqs, a.gamma, a.temperature, a.skip, a.eval_bs)
            rows.append({"tenant": task["name"], "strength": s, **m})
            print(f"{task['name']} s={s:.2f} beta={m['beta']:.4f} "
                  f"greedy={m['beta_greedy']:.4f} tok/step={m['tokens_per_step_iid']:.3f}",
                  flush=True)
        del T, D; torch.cuda.empty_cache()
    json.dump(rows, open(f"{a.out}/gpu_e1.json", "w"), indent=2)


def stage_e2(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.target)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = []
    for task in TASKS[:a.n_tenants]:
        T, D = _load_pair(a, task["name"], four_bit_target=a.four_bit_serve)
        pr = prompts_only(task, tok, a.n_eval + a.n_roll, seed=1).to(T.device)
        roll = generate_batched(T, pr[a.n_eval:], a.gen_len, a.temperature)      # train split
        ev = generate_batched(T, pr[:a.n_eval], a.gen_len, a.temperature)        # eval split
        base_m = measure(T, D, ev, a.gamma, a.temperature, a.skip, a.eval_bs)
        rows.append({"tenant": task["name"], "arm": "shared-drafter", "params": 0, **base_m})
        print(f"{task['name']} shared beta={base_m['beta']:.4f}", flush=True)

        for kind in a.losses.split(","):
            D2 = HFWrapper(attach_lora(load_causal_lm(a.draft, attn_impl="sdpa"),
                                       r=a.companion_rank, alpha=2 * a.companion_rank))
            n_p = adapter_param_count(D2.model)
            opt = torch.optim.AdamW([p for p in D2.model.parameters() if p.requires_grad],
                                    lr=a.companion_lr)
            D2.model.train(); t0 = time.time()
            for step in range(a.companion_steps):
                x = roll[torch.randint(0, roll.shape[0], (a.bs,))].to(T.device)
                with torch.no_grad():
                    p = (T(x)[:, a.skip:].float() / a.temperature).softmax(-1)
                ql = D2(x)[:, a.skip:].float() / a.temperature
                if kind == "ce":
                    # hard-label CE predicts the NEXT token, so drop the last position
                    loss = companion_loss(ql[:, :-1], p[:, :-1], kind, hard=x[:, a.skip + 1:])
                else:
                    loss = companion_loss(ql, p, kind)
                loss.backward()
                opt.step(); opt.zero_grad(set_to_none=True)
                if step % 25 == 0:
                    print(f"  [{kind}] {step}/{a.companion_steps} loss={loss.item():.4f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                del p, ql
            D2.model.eval()
            m = measure(T, D2, ev, a.gamma, a.temperature, a.skip, a.eval_bs)
            rows.append({"tenant": task["name"], "arm": f"companion-{kind}-r{a.companion_rank}",
                         "params": n_p, **m})
            print(f"{task['name']} companion[{kind}] beta={m['beta']:.4f} "
                  f"(+{m['beta']-base_m['beta']:.4f})", flush=True)
            D2.model.save_pretrained(f"{a.out}/adapters/{task['name']}__companion_{kind}")
            del D2, opt; torch.cuda.empty_cache()
        del T, D; torch.cuda.empty_cache()
    json.dump(rows, open(f"{a.out}/gpu_e2.json", "w"), indent=2)


def stage_throughput(a):
    """Measured per-forward latency -> the cost ratio the analytic model needs."""
    T, D = _load_pair(a)
    x = torch.randint(0, 1000, (1, a.skip + a.gen_len)).to(T.device)
    out = {}
    for name, m in (("target", T), ("draft", D)):
        for _ in range(3):
            m(x)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10):
            m(x[:, -1:])
        torch.cuda.synchronize()
        out[f"{name}_decode_ms"] = (time.time() - t0) / 10 * 1000
    out["cost_ratio_measured"] = out["draft_decode_ms"] / out["target_decode_ms"]
    print(json.dumps(out, indent=2))
    json.dump(out, open(f"{a.out}/gpu_cost_ratio.json", "w"), indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["tenants", "e1", "e2", "throughput"])
    p.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--out", default="tandemspec_out")
    p.add_argument("--n_tenants", type=int, default=4)
    p.add_argument("--task_rank", type=int, default=16)
    p.add_argument("--companion_rank", type=int, default=4)
    p.add_argument("--tenant_steps", type=int, default=300)
    p.add_argument("--companion_steps", type=int, default=200)
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_eval", type=int, default=32)
    p.add_argument("--n_roll", type=int, default=96)
    p.add_argument("--gen_len", type=int, default=96)
    p.add_argument("--max_len", type=int, default=384)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--eval_bs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--companion_lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--skip", type=int, default=8)
    p.add_argument("--strengths", default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--losses", default="tvd,fkl")
    p.add_argument("--four_bit_serve", action="store_true",
                   help="serve the target 4-bit (QLoRA train/serve match; see E4)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"tenants": stage_tenants, "e1": stage_e1, "e2": stage_e2,
     "throughput": stage_throughput}[a.stage](a)


if __name__ == "__main__":
    main()
