"""Minimal LoRA with a runtime strength knob and RTN int4 fake-quantisation.

Two things this needs that `peft` does not give us directly:

* `strength` -- a runtime multiplier on the adapter output, so a *single*
  trained adapter can be swept continuously from 0 (base model) to >1
  (over-driven). This is how E1 turns a discrete set of tenant adapters into a
  continuous perturbation-strength axis.
* `quantize_base_` -- round-to-nearest int4 group-wise fake quantisation of the
  frozen base weight, so E4 can measure the acceptance cost of the
  train-quantisation / serve-quantisation mismatch that QLoRA tenants hit.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def rtn_quantize(w: torch.Tensor, bits: int = 4, group_size: int = 64) -> torch.Tensor:
    """Symmetric round-to-nearest group-wise fake quantisation of a 2-D weight."""
    out_f, in_f = w.shape
    gs = min(group_size, in_f)
    pad = (-in_f) % gs
    if pad:
        w = torch.cat([w, w.new_zeros(out_f, pad)], dim=1)
    wg = w.view(out_f, -1, gs)
    qmax = 2 ** (bits - 1) - 1
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    wq = torch.round(wg / scale).clamp(-qmax - 1, qmax) * scale
    wq = wq.view(out_f, -1)
    return wq[:, :in_f].contiguous()


class LoRALinear(nn.Module):
    """nn.Linear + optional low-rank adapter. Base weight is always frozen."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.r = 0
        self.alpha = 0.0
        self.strength = 1.0
        self.lora_A: nn.Parameter | None = None
        self.lora_B: nn.Parameter | None = None
        self._base_fp: torch.Tensor | None = None  # kept for de-quantisation

    # -- adapter lifecycle -------------------------------------------------
    def add_adapter(self, r: int, alpha: float | None = None, seed: int | None = None):
        self.r = r
        self.alpha = float(alpha if alpha is not None else r)
        g = None
        if seed is not None:
            g = torch.Generator().manual_seed(seed)
        a = torch.empty(r, self.in_features)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=g) if g is not None else nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.lora_A = nn.Parameter(a)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))  # zero-init: adapter starts as identity
        return self

    def drop_adapter(self):
        self.r, self.alpha = 0, 0.0
        self.lora_A = self.lora_B = None
        return self

    def adapter_state(self) -> dict:
        if self.lora_A is None:
            return {}
        return {"A": self.lora_A.detach().clone(), "B": self.lora_B.detach().clone(),
                "r": self.r, "alpha": self.alpha}

    def load_adapter_state(self, st: dict):
        if not st:
            return self.drop_adapter()
        self.r, self.alpha = st["r"], st["alpha"]
        self.lora_A = nn.Parameter(st["A"].clone())
        self.lora_B = nn.Parameter(st["B"].clone())
        return self

    @property
    def scaling(self) -> float:
        return 0.0 if self.r == 0 else self.alpha / self.r * self.strength

    def delta_w(self) -> torch.Tensor:
        if self.lora_A is None:
            return torch.zeros_like(self.base.weight)
        return self.scaling * (self.lora_B @ self.lora_A)

    # -- quantisation ------------------------------------------------------
    def quantize_base_(self, bits: int = 4, group_size: int = 64):
        if self._base_fp is None:
            self._base_fp = self.base.weight.detach().clone()
        with torch.no_grad():
            self.base.weight.copy_(rtn_quantize(self._base_fp, bits, group_size))
        return self

    def dequantize_base_(self):
        if self._base_fp is not None:
            with torch.no_grad():
                self.base.weight.copy_(self._base_fp)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.lora_A is not None and self.strength != 0.0:
            out = out + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return out


# -- module-tree helpers ---------------------------------------------------

def lora_modules(model: nn.Module):
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            yield name, m


def add_adapters(model: nn.Module, r: int, alpha: float | None = None,
                 targets: tuple[str, ...] = ("q_proj", "v_proj", "o_proj", "up_proj", "down_proj")):
    n = 0
    for name, m in lora_modules(model):
        if any(name.endswith(t) for t in targets):
            m.add_adapter(r, alpha)
            n += 1
    return n


def adapter_parameters(model: nn.Module):
    for _, m in lora_modules(model):
        if m.lora_A is not None:
            yield m.lora_A
            yield m.lora_B


def n_adapter_params(model: nn.Module) -> int:
    return sum(p.numel() for p in adapter_parameters(model))


def set_strength(model: nn.Module, s: float):
    for _, m in lora_modules(model):
        m.strength = s


def save_adapter(model: nn.Module) -> dict:
    return {name: m.adapter_state() for name, m in lora_modules(model) if m.lora_A is not None}


def load_adapter(model: nn.Module, state: dict):
    for name, m in lora_modules(model):
        m.load_adapter_state(state.get(name, {}))


def clear_adapters(model: nn.Module):
    for _, m in lora_modules(model):
        m.drop_adapter()


def relative_shift(model: nn.Module) -> float:
    """||dW||_F / ||W||_F aggregated over all adapted matrices."""
    num = sq = 0.0
    with torch.no_grad():
        for _, m in lora_modules(model):
            if m.lora_A is None:
                continue
            num += float(m.delta_w().detach().pow(2).sum())
            sq += float(m.base.weight.detach().pow(2).sum())
    return math.sqrt(num / sq) if sq > 0 else 0.0


def quantize_model_(model: nn.Module, bits: int = 4, group_size: int = 64):
    for _, m in lora_modules(model):
        m.quantize_base_(bits, group_size)


def dequantize_model_(model: nn.Module):
    for _, m in lora_modules(model):
        m.dequantize_base_()
