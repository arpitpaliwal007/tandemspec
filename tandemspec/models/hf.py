"""HuggingFace glue so the same TandemSpec code runs on real models.

Everything in `tandemspec.eval` and `tandemspec.accept` only needs a callable
that maps `(B, T)` token ids to `(B, T, V)` logits. `HFWrapper` provides that
for any `AutoModelForCausalLM`, so the CPU pilot and the GPU experiments share
one measurement path -- the acceptance numbers are produced by identical code.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HFWrapper(nn.Module):
    """Adapts a HuggingFace causal LM to the `logits = model(ids)` convention."""

    def __init__(self, model, adapter_name: str | None = None):
        super().__init__()
        self.model = model
        self.adapter_name = adapter_name

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=idx, use_cache=False).logits

    def set_adapter(self, name: str | None):
        """Switch the active PEFT adapter (or disable adapters when None)."""
        if hasattr(self.model, "disable_adapter_layers"):
            if name is None:
                self.model.disable_adapter_layers()
            else:
                self.model.enable_adapter_layers()
                self.model.set_adapter(name)
        self.adapter_name = name
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device


def load_causal_lm(model_id: str, dtype="float16", four_bit: bool = False,
                   device_map="auto", attn_impl: str | None = None):
    """Load a causal LM sized for a 16 GB T4.

    T4 is sm_75: bfloat16 and FlashAttention-2 are unavailable, so fp16 + SDPA
    is the correct configuration and the caller should not override it blindly.
    """
    from transformers import AutoModelForCausalLM
    kw = dict(device_map=device_map, dtype=getattr(torch, dtype))
    if attn_impl:
        kw["attn_implementation"] = attn_impl
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    return AutoModelForCausalLM.from_pretrained(model_id, **kw)


def attach_lora(model, r: int = 16, alpha: int | None = None, dropout: float = 0.0,
                targets=("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"),
                adapter_name: str = "default"):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha or 2 * r, lora_dropout=dropout,
                     target_modules=list(targets), bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg, adapter_name=adapter_name)


def adapter_param_count(model) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)


@torch.no_grad()
def generate_batched(wrapper: HFWrapper, input_ids: torch.Tensor, n_new: int,
                     temperature: float = 1.0, top_p: float = 1.0,
                     eos_id: int | None = None) -> torch.Tensor:
    """KV-cached ancestral sampling -- the on-policy rollout generator."""
    model = wrapper.model
    model.eval()
    out = model(input_ids=input_ids, use_cache=True)
    past, seq = out.past_key_values, input_ids
    nxt_logits = out.logits[:, -1]
    for _ in range(n_new):
        probs = _sample_probs(nxt_logits, temperature, top_p)
        nxt = torch.multinomial(probs, 1)
        seq = torch.cat([seq, nxt], dim=1)
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past, nxt_logits = out.past_key_values, out.logits[:, -1]
    return seq


def _sample_probs(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    logits = logits.float() / max(temperature, 1e-6)
    probs = logits.softmax(-1)
    if top_p >= 1.0:
        return probs
    sp, si = probs.sort(dim=-1, descending=True)
    cum = sp.cumsum(-1)
    mask = cum - sp > top_p
    sp = sp.masked_fill(mask, 0.0)
    sp = sp / sp.sum(-1, keepdim=True)
    return torch.zeros_like(probs).scatter(-1, si, sp)
