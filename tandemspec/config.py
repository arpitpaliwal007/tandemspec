"""Single source of truth for the CPU-scale pilot configuration."""
from dataclasses import dataclass, asdict


@dataclass
class PilotConfig:
    # source
    vocab: int = 256
    n_domains: int = 6        # domains present in the pretraining mixture
    n_in_dist: int = 3        # tenants drawn from those domains
    n_held_out: int = 3       # tenants whose domain the base model never saw
    n_states: int = 8
    stickiness: float = 0.94
    lam: float = 0.4
    emit_conc: float = 0.04
    seq_len: int = 65
    n_seq_per_domain: int = 3000

    # models
    target_d: int = 192
    target_layers: int = 4
    target_ff: int = 512
    draft_d: int = 96
    draft_layers: int = 2
    draft_ff: int = 256
    n_heads: int = 4

    # training
    batch: int = 32
    pretrain_steps: int = 2500
    drafter_steps: int = 2500
    task_lora_steps: int = 500
    companion_steps: int = 300
    task_rank: int = 8
    companion_rank: int = 4

    # eval
    gamma: int = 4
    temperature: float = 1.0
    eval_seqs: int = 96
    eval_len: int = 64
    skip_prefix: int = 8
    seed: int = 0

    def to_dict(self):
        return asdict(self)
