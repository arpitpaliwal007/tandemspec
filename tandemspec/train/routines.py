"""Training routines: pretrain, drafter distillation, tenant LoRAs, companion adapters."""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from ..models import lora as L


def _log(tag, step, steps, loss, t0, every):
    if step % every == 0 or step == steps - 1:
        print(f"  [{tag}] {step+1}/{steps} loss={loss:.4f} ({time.time()-t0:.0f}s)", flush=True)


def pretrain_lm(model, batcher, steps=2000, lr=3e-3, wd=0.01, log_every=250, tag="pretrain"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    model.train(); t0 = time.time(); last = 0.0
    for s in range(steps):
        x = batcher()
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); last = loss.item()
        _log(tag, s, steps, last, t0, log_every)
    return last


def distill_drafter(draft, target, batcher, steps=2000, lr=3e-3, temperature=1.0,
                    alpha_ce=0.2, log_every=250, tag="drafter-KD"):
    """Standard drafter training: match the *base* target's next-token distribution.

    This is the drafter every tenant then shares -- and precisely the reason
    acceptance degrades once a tenant's LoRA moves the target away from it.
    """
    opt = torch.optim.AdamW(draft.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    target.eval(); draft.train(); t0 = time.time(); last = 0.0
    for s in range(steps):
        x = batcher()
        with torch.no_grad():
            p = (target(x[:, :-1]) / temperature).softmax(-1)
        ql = draft(x[:, :-1]) / temperature
        kd = F.kl_div(ql.log_softmax(-1), p, reduction="batchmean") / ql.shape[1]
        ce = F.cross_entropy(ql.reshape(-1, ql.shape[-1]), x[:, 1:].reshape(-1))
        loss = (1 - alpha_ce) * kd + alpha_ce * ce
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        opt.step(); sched.step(); last = loss.item()
        _log(tag, s, steps, last, t0, log_every)
    return last


def train_task_lora(target, batcher, r=8, alpha=None, steps=400, lr=1.5e-3,
                    log_every=200, tag="task-lora", targets=("q_proj", "v_proj", "o_proj", "up_proj", "down_proj")):
    """Tenant fine-tune: only adapter parameters move; the base stays frozen."""
    L.clear_adapters(target)
    n_mats = L.add_adapters(target, r=r, alpha=alpha, targets=targets)
    target.freeze_base()
    params = list(L.adapter_parameters(target))
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    target.train(); t0 = time.time(); last = 0.0
    for s in range(steps):
        x = batcher()
        logits = target(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); last = loss.item()
        _log(tag, s, steps, last, t0, log_every)
    state = L.save_adapter(target)
    return state, {"final_loss": last, "n_mats": n_mats,
                   "n_params": sum(p.numel() for p in params)}


# ---------------------------------------------------------------------------
# Companion draft adapters
# ---------------------------------------------------------------------------

def companion_loss(q_logits: torch.Tensor, p: torch.Tensor, kind: str,
                   hard: torch.Tensor | None = None) -> torch.Tensor:
    """Losses for training a drafter against target distribution `p`.

    `tvd` is the one that matters: 0.5*||p-q||_1 = 1 - beta, i.e. minimising it
    *directly maximises the expected acceptance rate*. KL variants are the
    conventional distillation objectives and serve as baselines.
    """
    logq = q_logits.log_softmax(-1)
    q = logq.exp()
    if kind == "tvd":
        return 0.5 * (p - q).abs().sum(-1).mean()
    if kind == "fkl":
        return (p * (p.clamp_min(1e-12).log() - logq)).sum(-1).mean()
    if kind == "rkl":
        return (q * (logq - p.clamp_min(1e-12).log())).sum(-1).mean()
    if kind == "ce":
        return F.cross_entropy(q_logits.reshape(-1, q_logits.shape[-1]), hard.reshape(-1))
    if kind == "tvd+fkl":
        return (0.5 * (p - q).abs().sum(-1).mean()
                + 0.5 * (p * (p.clamp_min(1e-12).log() - logq)).sum(-1).mean())
    raise ValueError(kind)


def train_companion_adapter(draft, target, seq_batcher, r=4, alpha=None, steps=400, lr=8e-3,
                            loss_kind="tvd", temperature=1.0, log_every=200,
                            tag="companion", full_finetune=False,
                            targets=("q_proj", "v_proj", "o_proj", "up_proj", "down_proj")):
    """Train a tiny LoRA on the *drafter* to track an already-adapted target.

    `seq_batcher` must yield sequences drawn from the adapted target's own
    output distribution (on-policy) -- see `eval.acceptance` for why that is the
    right training distribution.
    """
    L.clear_adapters(draft)
    if full_finetune:
        for p in draft.parameters():
            p.requires_grad_(True)
        params = [p for p in draft.parameters() if p.requires_grad]
        n_mats = 0
    else:
        n_mats = L.add_adapters(draft, r=r, alpha=alpha, targets=targets)
        draft.freeze_base()
        params = list(L.adapter_parameters(draft))
        for p in params:
            p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    target.eval(); draft.train(); t0 = time.time(); last = 0.0
    for s in range(steps):
        x = seq_batcher()
        with torch.no_grad():
            p = (target(x[:, :-1]) / temperature).softmax(-1)
        ql = draft(x[:, :-1]) / temperature
        loss = companion_loss(ql, p, loss_kind, hard=x[:, 1:])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); last = loss.item()
        _log(tag, s, steps, last, t0, log_every)
    state = None if full_finetune else L.save_adapter(draft)
    n_params = sum(p.numel() for p in params)
    return state, {"final_loss": last, "n_mats": n_mats, "n_params": n_params}
