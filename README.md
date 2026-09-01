# TandemSpec

Companion LoRA adapters for multi-tenant speculative decoding.

Multi-LoRA serving changes the target model for each tenant, while a shared
speculative drafter usually still approximates the base model. This repository
tests the resulting mismatch and a simple fix: pair every target-side task LoRA
with a small drafter-side companion LoRA trained to match that tenant's adapted
target distribution.

For target distribution `p` and draft distribution `q`, the per-token
acceptance probability is:

```
beta = sum_v min(p_v, q_v) = 1 - TVD(p_target, q_draft)
```

![Acceptance collapse](docs/fig1_acceptance_collapse.png)

**Figure 1.** Shared-drafter acceptance as each tenant's LoRA is scaled from 0 (base model) to 1.3.
Blue: tenants whose domain appears in the pretraining mixture. Orange: tenants fine-tuned on data the
base model never saw. Same drafter and same base model throughout — only the tenant's adapter changes.

## CPU synthetic pilot

| | acceptance β | tokens/step | greedy agreement |
|---|---|---|---|
| shared drafter, **unadapted** target | 0.980 | 4.81 | 0.957 |
| + in-distribution tenant LoRA | 0.887 | 3.99 | 0.777 |
| + held-out tenant LoRA | **0.259** | **1.35** | **0.003** |
| + rank-4 companion adapter (20 KB/tenant) | 0.912 | 4.23 | 0.650 |
| + private drafter fine-tune (26× params) | 0.949 | 4.52 | 0.817 |

<sub>Averaged over 6 tenants, γ=4, temperature 1.0. Rows 2–3 use the same base
model and shared drafter; only the tenant adapter changes.</sub>

<img src="docs/fig2_beta_vs_shift.png" width="380">

**Figure 2.** Acceptance versus the distribution shift induced by the tenant
adapter, pooled over tenants and LoRA strengths. The dashed line is the
triangle-inequality bound `β₀ − Δ`.

![Companion adapter repair](docs/fig3_companion_repair.png)

**Figure 3.** Mean acceptance by drafter-side training arm over 6 tenants. A
20 KB companion adapter recovers 83% of lost acceptance in this synthetic
setup; a private drafter recovers 92% and costs about 2.5 GB per tenant.

What the adapter is trained on matters more than how big it is:

| training signal for the companion adapter | acceptance β | % of loss recovered |
|---|---|---|
| none (shared drafter) | 0.5718 | — |
| hard-label cross-entropy on tenant tokens | 0.6136 | 10% |
| distil from adapted target, forward KL | 0.9000 | 80% |
| distil from adapted target, **TVD (= 1 − β)** | 0.9116 | 83% |

Hard-label training helps little. Distilling the full adapted-target
distribution is substantially more effective.

[Full writeup](paper/tandemspec_filled.md) · [first-order theory](paper/theory.md) ·
[vLLM RFC #52038 comment](rfc/vllm_rfc_52038_comment.md) ·
[interactive dashboard](results/tandemspec_dashboard.html) (download and open locally)

Figures are regenerated from the result JSON with `python experiments/make_figures.py`.

---

## Real-Qwen evaluation

The Colab pipeline evaluates Qwen2.5-1.5B as the target and Qwen2.5-0.5B as
the drafter across four tenants (math, SQL, dialogue, and code). With rank-4
forward-KL companion adapters, mean acceptance increased from approximately
`β=0.48` to `β=0.84`.

The T4 cost model is deliberately reported as a cost estimate, not a serving
benchmark. At speculation depth 4, the rank-4 companion was approximately
break-even against target-only decoding, while the shared drafter was slower.
The stored measurements are in:

- `results/qwen_rank_ablation_r4_r8.json`
- `results/qwen_t4_throughput_calibration.json`

## Main observations

1. A base-model drafter can lose acceptance when the target is adapted for a
   tenant.
2. A small companion adapter on the drafter can recover most of that loss.
3. The serving layer must resolve a tenant ID to both a target adapter and a
   draft adapter.

The companion is trained from rollouts of the adapted target, which provide
on-policy contexts for speculative decoding. See the writeup for the derivation
and the CPU results for the controlled experiment.

## Repository layout

```
tandemspec/
  metrics.py            beta = 1 - TVD and the tokens/step closed form
  accept.py             exact speculative-sampling simulator (lossless, verified)
  config.py             pilot configuration
  models/
    lora.py             LoRA with a runtime strength knob + RTN int4 fake-quant
    tiny.py             small Llama-style transformer for the CPU pilot
    hf.py               HuggingFace glue so GPU runs use the same measurement code
  data/synth.py         sticky-HMM multi-domain source with computable ground truth
  train/routines.py     pretrain, drafter distillation, tenant LoRAs, companion adapters
  eval/
    acceptance.py       teacher-forced + rollout acceptance measurement
    throughput.py       step-cost model, optimal gamma, serving scenarios
  serving/
    paired_adapters.py  tenant -> (task LoRA, companion LoRA); admission + memory
    vllm_integration.py the concrete vLLM hook points and a support checker
experiments/
  stage0_pretrain.py    base target + shared drafter
  stage1_task_loras.py  six tenant adapters (3 in-distribution, 3 held-out)
  e1_acceptance_collapse.py
  e2_companion_repair.py
  e3_serving_model.py
  e4_quant_mismatch.py
  make_dashboard.py     self-contained HTML results dashboard
  gpu/tandemspec_gpu.py T4-sized pipeline on Qwen2.5-1.5B + Qwen2.5-0.5B
colab/                  notebook driving the GPU pipeline
paper/                  writeup + first-order theory
rfc/                    comment for vLLM RFC #52038
tests/                  simulator correctness (losslessness + closed form)
```

## Running the CPU pilot

No GPU required; the whole pilot is ~2M-parameter models on a synthetic source
with a computable Bayes-optimal predictor.

```bash
pip install torch numpy
python tests/test_accept.py                       # verify the simulator first
python experiments/stage0_pretrain.py             # base target + shared drafter
python experiments/stage1_task_loras.py           # six tenant adapters
python experiments/e1_acceptance_collapse.py      # the collapse curve
python experiments/e2_companion_repair.py         # companion adapters + baselines
python experiments/e4_quant_mismatch.py           # QLoRA train/serve mismatch
python experiments/e3_serving_model.py            # throughput + memory model
python experiments/make_dashboard.py              # results/tandemspec_dashboard.html
```

## Running on a 16 GB GPU (Colab T4)

`colab/TandemSpec_T4.ipynb` drives the same measurement code on real models.
T4 is sm_75: fp16 and SDPA, **not** bf16 or FlashAttention-2.

```bash
python experiments/gpu/tandemspec_gpu.py --stage tenants     # QLoRA tenant adapters
python experiments/gpu/tandemspec_gpu.py --stage e1          # collapse + strength sweep
python experiments/gpu/tandemspec_gpu.py --stage e2          # companion repair
python experiments/gpu/tandemspec_gpu.py --stage throughput  # measured cost ratio
```

## Serving integration

For each request, resolve the tenant ID to a target adapter and a companion
draft adapter:

```python
from tandemspec.serving.paired_adapters import PairedAdapter, AdapterSpec, PairedAdapterRegistry

reg = PairedAdapterRegistry(max_loras=8, max_lora_rank=32, max_draft_lora_rank=8)
reg.register(PairedAdapter(
    tenant_id=7, name="acme",
    target=AdapterSpec("acme-task",      rank=32, path="/adapters/acme"),
    draft =AdapterSpec("acme-companion", rank=4,  path="/adapters/acme-draft"),
))
```

`max_loras` now limits admitted adapter pairs. The draft-side adapters use a
smaller rank, so target-side adapter memory remains the main capacity limit.
See `tandemspec/serving/vllm_integration.py` for the proposed vLLM hook points.
