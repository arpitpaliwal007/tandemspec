"""Stage 0: pretrain the shared base target, then distil the shared drafter.

This mirrors what a serving provider actually has on disk before any tenant
shows up: one base model, and one drafter trained to imitate that base model.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_num_threads(os.cpu_count() or 2)

from tandemspec.config import PilotConfig
from tandemspec.data.synth import make_domains, build_corpus, Batcher
from tandemspec.models.tiny import TinyLM, TinyConfig
from tandemspec.train.routines import pretrain_lm, distill_drafter
from tandemspec.eval.acceptance import measure_acceptance, generate

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RES, exist_ok=True)


def main():
    cfg = PilotConfig()
    torch.manual_seed(cfg.seed)
    t_start = time.time()

    domains, backbone = make_domains(cfg.n_domains, cfg.vocab, cfg.n_states,
                                     cfg.stickiness, cfg.lam, cfg.emit_conc, seed=cfg.seed)
    corpus, dom_ids = build_corpus(domains, cfg.n_seq_per_domain, cfg.seq_len, seed=cfg.seed + 1)
    print(f"corpus {tuple(corpus.shape)}  ({corpus.numel()/1e6:.2f}M tokens)", flush=True)
    torch.save({"corpus": corpus, "dom_ids": dom_ids}, f"{RES}/corpus.pt")
    torch.save({"domains": [(d.trans, d.emit, d.init) for d in domains],
                "backbone": (backbone.trans, backbone.emit, backbone.init)}, f"{RES}/sources.pt")

    tcfg = TinyConfig(cfg.vocab, cfg.target_d, cfg.target_layers, cfg.n_heads, cfg.target_ff, 128)
    dcfg = TinyConfig(cfg.vocab, cfg.draft_d, cfg.draft_layers, cfg.n_heads, cfg.draft_ff, 128)
    target, draft = TinyLM(tcfg), TinyLM(dcfg)
    print(f"target {target.n_params()/1e6:.2f}M   draft {draft.n_params()/1e6:.2f}M   "
          f"ratio {target.n_params()/draft.n_params():.1f}x", flush=True)

    batcher = Batcher(corpus, cfg.batch, seed=cfg.seed + 2)
    print("== pretraining base target on the domain mixture ==", flush=True)
    l_t = pretrain_lm(target, batcher, steps=cfg.pretrain_steps)
    print("== distilling shared drafter from the base target ==", flush=True)
    l_d = distill_drafter(draft, target, batcher, steps=cfg.drafter_steps)

    # baseline acceptance of the shared drafter against the UNADAPTED target
    prompts = corpus[torch.randperm(corpus.shape[0])[:cfg.eval_seqs], :cfg.skip_prefix]
    seqs = generate(target, prompts, cfg.eval_len, cfg.temperature,
                    generator=torch.Generator().manual_seed(99))
    st = measure_acceptance(target, draft, seqs, cfg.gamma, cfg.temperature,
                            cfg.skip_prefix, mc_blocks=4000, seed=7)
    print("base-model acceptance:", json.dumps(st.as_dict(), indent=2), flush=True)

    torch.save({"target": target.state_dict(), "draft": draft.state_dict(),
                "tcfg": tcfg.__dict__, "dcfg": dcfg.__dict__}, f"{RES}/stage0.pt")
    json.dump({"config": cfg.to_dict(), "target_loss": l_t, "draft_loss": l_d,
               "target_params": target.n_params(), "draft_params": draft.n_params(),
               "base_acceptance": st.as_dict(), "wallclock_s": time.time() - t_start},
              open(f"{RES}/stage0.json", "w"), indent=2)
    print(f"stage0 done in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
