"""Stage 1: train one tenant LoRA per domain on the frozen shared base target."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
torch.set_num_threads(os.cpu_count() or 2)

from tandemspec.config import PilotConfig
from tandemspec.data.synth import Batcher, make_domains, build_corpus
from tandemspec.models.tiny import TinyLM, TinyConfig
from tandemspec.models import lora as L
from tandemspec.train.routines import train_task_lora
import torch.nn.functional as F


@torch.no_grad()
def eval_loss(model, data, n=192):
    """Held-out cross-entropy, so we can see whether a tenant fine-tune helped."""
    model.eval()
    x = data[:n]
    lo = model(x[:, :-1])
    return float(F.cross_entropy(lo.reshape(-1, lo.shape[-1]), x[:, 1:].reshape(-1)))

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load_stage0():
    ck = torch.load(f"{RES}/stage0.pt", weights_only=False)
    target = TinyLM(TinyConfig(**ck["tcfg"])); target.load_state_dict(ck["target"])
    draft = TinyLM(TinyConfig(**ck["dcfg"])); draft.load_state_dict(ck["draft"])
    return target, draft


def main():
    cfg = PilotConfig(); torch.manual_seed(cfg.seed + 10)
    t0 = time.time()
    target, _ = load_stage0()
    d = torch.load(f"{RES}/corpus.pt", weights_only=False)
    corpus, dom_ids = d["corpus"], d["dom_ids"]

    # Two tenant populations, because real deployments have both:
    #   in-distribution -- the tenant's data resembles the pretraining mixture
    #   held-out        -- the tenant fine-tunes on private data the base never saw
    held_domains, _ = make_domains(cfg.n_held_out, cfg.vocab, cfg.n_states, cfg.stickiness,
                                   cfg.lam, cfg.emit_conc, seed=cfg.seed + 555)
    held_corpus, held_ids = build_corpus(held_domains, cfg.n_seq_per_domain // 3,
                                         cfg.seq_len, seed=cfg.seed + 556)
    torch.save({"corpus": held_corpus, "dom_ids": held_ids}, f"{RES}/held_corpus.pt")

    tenants = ([(t, "in-dist", corpus[dom_ids == t]) for t in range(cfg.n_in_dist)] +
               [(cfg.n_in_dist + i, "held-out", held_corpus[held_ids == i])
                for i in range(cfg.n_held_out)])

    adapters, meta = {}, {}
    for t, kind, sub in tenants:
        print(f"== tenant {t} ({kind}): {sub.shape[0]} seqs ==", flush=True)
        holdout = sub[-192:]
        before = eval_loss(target, holdout)
        state, info = train_task_lora(target, Batcher(sub[:-192], cfg.batch, seed=100 + t),
                                      r=cfg.task_rank, steps=cfg.task_lora_steps,
                                      tag=f"tenant{t}")
        after = eval_loss(target, holdout)
        info["relative_weight_shift"] = L.relative_shift(target)
        info["kind"] = kind
        info["holdout_loss_before"] = before
        info["holdout_loss_after"] = after
        adapters[t] = state; meta[t] = info
        print(f"   tenant {t} ({kind}): {info['n_params']} adapter params, "
              f"||dW||/||W|| = {info['relative_weight_shift']:.4f}, "
              f"held-out loss {before:.4f} -> {after:.4f} ({after-before:+.4f} nats)",
              flush=True)
        L.clear_adapters(target)

    torch.save(adapters, f"{RES}/task_adapters.pt")
    json.dump({str(k): v for k, v in meta.items()}, open(f"{RES}/task_adapters.json", "w"), indent=2)
    print(f"stage1 done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
