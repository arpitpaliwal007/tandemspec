"""A small Llama-style decoder-only transformer, built out of LoRALinear layers.

Deliberately tiny so that the whole TandemSpec pipeline -- pretrain target,
distil drafter, train tenant adapters, train companion draft adapters, measure
acceptance -- runs end to end on a CPU in minutes. The same experiment scripts
drive HuggingFace models on a GPU via `tandemspec.eval.acceptance`, which only
needs a callable returning logits.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRALinear


@dataclass
class TinyConfig:
    vocab_size: int = 256
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    max_seq: int = 128
    rope_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def build_rope(seq: int, head_dim: int, base: float, device=None):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq, device=device).float()
    f = torch.outer(t, inv)
    return f.cos()[None, None], f.sin()[None, None]


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[..., : x1.shape[-1]], sin[..., : x1.shape[-1]]
    o1, o2 = x1 * c - x2 * s, x1 * s + x2 * c
    return torch.stack([o1, o2], dim=-1).flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Attention(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.q_proj = LoRALinear(d, d)
        self.k_proj = LoRALinear(d, d)
        self.v_proj = LoRALinear(d, d)
        self.o_proj = LoRALinear(d, d)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        H, hd = self.cfg.n_heads, self.cfg.head_dim
        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, D)
        return self.o_proj(o)


class MLP(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.up_proj = LoRALinear(cfg.d_model, cfg.d_ff)
        self.gate_proj = LoRALinear(cfg.d_model, cfg.d_ff)
        self.down_proj = LoRALinear(cfg.d_ff, cfg.d_model)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn, self.mlp = Attention(cfg), MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class TinyLM(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = build_rope(cfg.max_seq, cfg.head_dim, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        x = self.embed(idx)
        cos, sin = self.cos[:, :, :T], self.sin[:, :, :T]
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.lm_head(self.norm(x))

    def freeze_base(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
