# TandemSpec: Companion Draft Adapters for Multi-Tenant Speculative Decoding

*Working draft — CPU-scale pilot results are measured; GPU-scale numbers are
produced by the T4 harness in `experiments/gpu/`.*

## Abstract

Production LLM servers increasingly combine two features that were designed
independently: **multi-LoRA serving**, where one base model backs hundreds of
tenant adapters, and **speculative decoding**, where a small drafter proposes
tokens that the target verifies. The two do not compose. The drafter is trained
to imitate the *base* model, but every request is verified against *base +
tenant adapter*, and the resulting distribution mismatch shows up directly as
lost acceptance. We show the loss is linear in the adapter's induced
distribution shift and that its throughput cost grows quadratically in the
speculation depth — so it gets worse, not better, as drafters improve.

We propose **companion draft adapters**: a rank-4 LoRA on the drafter, minted
per tenant, trained by on-policy distillation from the *adapted* target under a
total-variation objective that is exactly the acceptance rate. Because
speculative sampling is distribution-preserving, the correct on-policy training
distribution is the adapted target's own output — so the training set is
ordinary teacher rollouts, with no RL machinery. Serving requires one change to
the multi-LoRA contract: a tenant id resolves to a *pair* of adapters, applied
per request through the batched SGMV path the target already uses. Draft-side
adapter memory is ~3-5% of task-adapter memory and ~2 orders of magnitude below
a private drafter per tenant.

In the controlled pilot: a shared drafter at acceptance `beta_0 = 0.980` against
the base target falls to **0.887** for tenants fine-tuned in-distribution and to
**0.259** for tenants fine-tuned on data the base never saw — 4.81 down to 1.35
tokens per step at `gamma = 4`, i.e. speculation stops paying at all. Under
greedy decoding the collapse is near-total (top-1 agreement 0.965 -> 0.003). A
20 KB rank-4 companion adapter recovers **83%** of the lost acceptance
(0.572 -> 0.912 averaged over all tenants), against 92% for a full private
drafter fine-tune 26x its size.

## 1. The gap

Both halves of the problem are mature and neither knows about the other.

**Multi-LoRA serving.** vLLM serves many adapters over a shared base with
batched grouped-gather matmuls (punica/SGMV) and an LRU adapter cache; its
February 2026 multi-LoRA work reports 171 output tokens/s and 124 ms TTFT for
GPT-OSS 20B with 8 parallel rank-32 adapters. Adapters are selected *per
request*, so one batch routinely mixes tenants.

**Speculative decoding.** EAGLE-3.1 (May 2026) and block-diffusion drafters
like DFlash (June 2026) have pushed acceptance lengths and speedups well past
the original draft-model regime — DFlash reports 2-3x larger speedups than
EAGLE-3, and EAGLE-3.1 reports 2.03x per-user output throughput at concurrency 1
on Kimi K2.6-NVFP4.

**Neither stack tells the other about the tenant.** The proposer has no LoRA
plumbing: whatever adapter a request carries is applied to the target and
ignored by the drafter. vLLM RFC #52038 (opened 2026-08-12) is the first
move toward closing this — it proposes LoRA adapters on DFlash drafters, with
~28x size reduction versus per-domain drafters and quality within ~2% — but it
selects the drafter adapter *per deployment*, which does not help a server
whose batch contains several tenants at once, and it does not specify training
the drafter adapter against the adapted target.

There is also a cautionary result to respect. *Accepted Prefixes Are Not All
You Need* (arXiv:2607.12422) reports that PEFT-based block-diffusion drafting
fails in practice: a LoRA adapter used as a drafter **on the target's own
backbone** achieved longer accepted prefixes than FastMTP (2.881 vs 1.511
tokens) yet ran at 34.05 vs 188.01 tokens/s, because drafting required a full
backbone forward (~50.3 ms) almost identical in cost to verification (~50.6 ms).
Their conclusion — "parameter-efficient drafting is not necessarily
compute-efficient drafting" — is correct and important. TandemSpec is not
subject to it: our adapter rides on a *separate, small* drafter that already
exists in the serving stack, so the draft forward stays 5-20x cheaper than the
target and the adapter changes only which distribution that cheap forward
approximates. The authors explicitly scope their result to "same-backbone
adapters executing full backbones."

## 2. Contributions

1. **The acceptance-collapse curve** (§4, E1): acceptance of a shared drafter as
   a function of tenant-adapter strength, with the induced distribution shift
   measured directly. To our knowledge this curve has not been published.
2. **A first-order theory** (`theory.md`) predicting linear acceptance decay in
   adapter strength and a throughput penalty that scales as `gamma(gamma+1)/4`,
   i.e. quadratically in speculation depth.
3. **Companion draft adapters** (§5, E2): per-tenant rank-4 LoRAs on the
   drafter, trained on-policy against the adapted target under a TVD objective
   that *is* the acceptance rate rather than a KL surrogate.
4. **The paired-adapter serving contract** (§6): one tenant id, two adapters,
   both selected per request; scheduler and memory accounting for it; the
   concrete vLLM hook points.
5. **The QLoRA mismatch result** (§7, E4), which came out the other way: a
   train/serve quantisation mismatch moves the *served model* meaningfully away
   from what the tenant trained (TVD up to 0.098) but costs the drafter almost
   no acceptance, because round-to-nearest noise is near-isotropic in logit
   space while a tenant adapter is coherent. Useful to know which of the two
   mismatches to spend engineering on.

## 3. Setup

The pilot uses a controlled synthetic testbed, chosen so the causal claim is
clean: a sticky hidden Markov source over a 256-token vocabulary whose
Bayes-optimal predictive distribution is computable in closed form. Latent-state
inference makes the task genuinely capacity-sensitive (unigram entropy 4.40
nats; Bayes-optimal conditional entropy 2.99 nats), so a smaller drafter really
is worse than the target — as in deployment, and unlike an order-1 source where
both models would converge and hide the effect.

Target: 1.87M parameters (4 layers, d=192). Drafter: 0.27M (2 layers, d=96),
6.9x smaller, distilled from the base target. Tenants: 6 rank-8 LoRAs, three on
domains present in the pretraining mixture ("in-dist") and three on domains the
base model never saw ("held-out"). All acceptance numbers use the same code
path as the GPU harness, and the simulator is verified against the closed form
in `tests/test_accept.py` (losslessness L1 = 0.0037; simulated E[accepted]
2.4051 vs closed form 2.4035).

Baseline: against the **unadapted** target the shared drafter achieves
`beta_0 = 0.980`, i.e. 4.80 of a maximum 5.0 tokens per step at `gamma = 4`.
A near-perfect drafter is deliberate — it means every point of acceptance lost
below is attributable to tenant adaptation and nothing else.

## 4. E1 — the acceptance collapse

Acceptance of the shared drafter as each tenant's adapter is swept from strength 0 (base model) to 1.3. `Δ` is the measured shift the adapter induces, `TVD(p_base, p_adapted)`, on rollouts from the adapted target.

| tenant | kind | Δ at s=1 | β at s=0 | β at s=1 | β drop | greedy s=0 | greedy s=1 | greedy drop | tok/step s=0 | tok/step s=1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | in-dist | 0.1071 | 0.9805 | 0.8928 | 8.9% | 0.9600 | 0.6095 | 36.5% | 4.809 | 4.037 |
| 1 | in-dist | 0.1009 | 0.9794 | 0.8981 | 8.3% | 0.9538 | 0.9036 | 5.3% | 4.798 | 4.080 |
| 2 | in-dist | 0.1294 | 0.9794 | 0.8706 | 11.1% | 0.9583 | 0.8185 | 14.6% | 4.798 | 3.863 |
| 3 | held-out | 0.7556 | 0.9802 | 0.2565 | 73.8% | 0.9631 | 0.0054 | 99.4% | 4.806 | 1.343 |
| 4 | held-out | 0.7468 | 0.9804 | 0.2641 | 73.1% | 0.9679 | 0.0024 | 99.7% | 4.807 | 1.357 |
| 5 | held-out | 0.7574 | 0.9792 | 0.2550 | 74.0% | 0.9648 | 0.0003 | 100.0% | 4.797 | 1.341 |

**At full adapter strength the shared drafter loses 41.5% of its acceptance on average** (range 8.3–74.0%), costing 44.4% of expected tokens per step at γ=4.

**Greedy decoding is hit far harder: 59.3% mean top-1 agreement loss** (range 5.3–100.0%). The accept/reject rule salvages probability mass wherever the two distributions still overlap; top-1 agreement has no such cushion. Most production serving runs at low temperature, so this is the number that hurts.

The first-order prediction is `β(s) ≈ β₀ − Δ`. Measured, the slope of β against Δ is **0.89** rather than 1: the drafter absorbs part of the shift, and the effective slope rises toward 1 as the adapter strengthens. The relationship is otherwise linear, and every curve starts from the same β₀ = 0.9802.

## 5. E2 — companion draft adapters

All arms are evaluated on held-out rollouts from the tenant's adapted target, averaged over 6 tenants.

| arm | β | greedy | tokens/step | extra params | fp16 size | train s | β recovered |
|---|---|---|---|---|---|---|---|
| `shared-drafter` | 0.5718 | 0.3914 | 2.663 | 0 | — | 0 | — |
| `companion-tvd-r4` | 0.9116 | 0.6503 | 4.227 | 10,240 | 20.0 KB | 42 | 83% |
| `companion-fkl-r4` | 0.9000 | 0.6316 | 4.142 | 10,240 | 20.0 KB | 45 | 80% |
| `companion-rkl-r4` | 0.9018 | 0.6202 | 4.154 | 10,240 | 20.0 KB | 41 | 81% |
| `companion-ce-r4` | 0.6136 | 0.4674 | 2.997 | 10,240 | 20.0 KB | 36 | 10% |
| `companion-tvd-r4-offpolicy` | 0.8996 | 0.6438 | 4.142 | 10,240 | 20.0 KB | 36 | 80% |
| `companion-tvd-r1` | 0.9002 | 0.6122 | 4.136 | 2,560 | 5.0 KB | 41 | 80% |
| `companion-tvd-r2` | 0.9056 | 0.6220 | 4.180 | 5,120 | 10.0 KB | 42 | 82% |
| `companion-tvd-r8` | 0.9107 | 0.6442 | 4.222 | 20,480 | 40.0 KB | 41 | 83% |
| `full-ft-tvd` | 0.9491 | 0.8174 | 4.522 | 270,816 | 528.9 KB | 42 | 92% |

A rank-4 companion adapter (10,240 parameters, 20 KB in fp16) lifts acceptance from 0.5718 to 0.9116, recovering 83% of the gap back to β₀ = 0.9802 — the acceptance the deployment was sized for before any tenant adapter existed.

**A full drafter fine-tune is better, and it should be.** At an equal 300-step budget it reaches β = 0.9491 (92% recovery) against the companion's 0.9116 (83%), and the gap is wider under greedy decoding (0.8174 vs 0.6503). It has 26× the parameters and every one of them is free to move.

That is the trade the whole proposal turns on: **83% of the loss recovered for 20 KB per tenant, versus 92% for a private drafter per tenant** — 2.5 GB at 8B/1B scale, which is exactly the cost that makes per-tenant drafters impossible above a handful of tenants. The companion adapter is not claimed to be the better model; it is claimed to be the one that fits.

**Distillation is doing the work, not fine-tuning.** The same rank-4 adapter trained with hard-label cross-entropy on the tenant's tokens reaches only β = 0.6136 (10% recovery) against 0.9116 (83%) for distillation from the adapted target. Matching *which tokens the tenant's model produces* is not enough; the drafter has to match *the distribution it produces them from*, because acceptance is a function of the full distribution, not of the argmax.

**The objective matters.** TVD is exactly `1 − β`, so minimising it maximises acceptance directly; forward KL is only a surrogate. Measured: 0.9116 (TVD) vs 0.9000 (forward KL) at the same rank and step budget (+0.0116).

**The data distribution matters.** Training on the tenant's raw corpus instead of rollouts from the adapted target gives 0.8996 vs 0.9116 (+0.0120 for on-policy). Since speculative decoding is distribution-preserving, target rollouts *are* the deployment distribution — the on-policy set is free to construct.

**Rank sweep.** r=1: β=0.9002 (2,560 params), r=2: β=0.9056 (5,120 params), r=4: β=0.9116 (10,240 params), r=8: β=0.9107 (20,480 params).

### 5b. What it is worth in a server

Measured acceptance pushed through the step-cost model at γ=4. `ceiling` is the speedup the same drafter would achieve against an unadapted target — the number the deployment was sized for.

The recovered fraction is identical across scenarios by construction: it is a ratio of `E[tokens/step]` values, and the step cost divides out. What differs between rows is the absolute speedup at stake.

| scenario | cost ratio | speedup shared | speedup companion | ceiling | lost speedup recovered | companion MB/tenant | private drafter GB/tenant |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B + Llama-3.2-1B draft | 0.154 | 1.36× | 2.59× | 2.97× | 76% | 5.64 | 2.48 |
| Llama-3.1-8B + EAGLE-3 head | 0.031 | 1.95× | 3.73× | 4.27× | 76% | 0.66 | 0.50 |
| Llama-3.1-8B + DFlash block drafter | 0.050 | 2.09× | 3.99× | 4.58× | 76% | 1.31 | 0.80 |
| Qwen3-32B + Qwen3-1.7B draft | 0.052 | 1.82× | 3.47× | 3.98× | 76% | 8.72 | 3.40 |
| Llama-3.3-70B + Llama-3.2-3B draft | 0.045 | 1.86× | 3.55× | 4.07× | 76% | 12.16 | 6.40 |
| Llama-3.1-8B NVFP4 + EAGLE-3 head fp16 | 0.125 | 1.47× | 2.80× | 3.21× | 76% | 0.66 | 0.50 |

β: 0.9802 (base model) → 0.5729 (adapted, shared drafter) → 0.9116 (adapted, companion).

**The optimal block size moves.** Llama-3.1-8B + Llama-3.2-1B draft: γ*=2 shared vs γ*=8 with a companion; Llama-3.1-8B + EAGLE-3 head: γ*=5 shared vs γ*=16 with a companion; Qwen3-32B + Qwen3-1.7B draft: γ*=4 shared vs γ*=14 with a companion; Llama-3.3-70B + Llama-3.2-3B draft: γ*=4 shared vs γ*=15 with a companion; Llama-3.1-8B NVFP4 + EAGLE-3 head fp16: γ*=2 shared vs γ*=9 with a companion. A server that tunes γ once against the base model runs every adapted tenant at the wrong operating point.

**Adapter residency** under Zipf(1.1) tenant traffic with `max_loras=8`, where a tenant's pair occupies one logical slot: 8 tenants 100% hit rate, 16 tenants 75% hit rate, 32 tenants 58% hit rate, 64 tenants 47% hit rate, 128 tenants 40% hit rate. Pairing does not change the residency behaviour, because the second adapter is keyed by the same tenant id.


## 6. The serving contract

TandemSpec asks for one change: **a tenant id resolves to a pair of adapters**,
`(target task LoRA, draft companion LoRA)`, both selected per request. Within a
batch the draft model runs with a heterogeneous set of companion adapters
applied through the same grouped-gather-matmul path the target already uses for
task adapters, driven by an index mapping derived from the target's.

`max_loras` keeps its meaning — distinct tenants admitted per batch — and
becomes a constraint on pairs. The draft side allocates the same cardinality at
a much smaller rank, so the binding memory constraint remains the target side
and existing multi-LoRA scheduling is unchanged. Concrete vLLM hook points are
enumerated in `tandemspec/serving/vllm_integration.py`; the framework-agnostic
registry, admission rule and memory accounting are in
`tandemspec/serving/paired_adapters.py`.

## 7. E4 — QLoRA train/serve quantisation mismatch

Tenants fine-tune against a 4-bit base and hand the adapter to a server whose base may be quantised differently. `drift` is TVD from the reference model (fp16 train, fp16 serve) — how far the tenant's *served* model is from the model they thought they trained.

| arm | drift from reference | β shared | β companion | tokens/step shared | tokens/step companion |
|---|---|---|---|---|---|
| `fp16-train/fp16-serve` | 0.0000 | 0.8926 | 0.9778 | 4.041 | 4.782 |
| `fp16-train/int4-serve` | 0.0369 | 0.8924 | 0.9776 | 4.000 | 4.791 |
| `int4-train/int4-serve` | 0.0940 | 0.8925 | 0.9776 | 4.027 | 4.787 |
| `int4-train/fp16-serve` | 0.0979 | 0.8916 | 0.9775 | 4.021 | 4.779 |

**Two findings, and they point in opposite directions.**

*The drift is real.* A quantisation mismatch moves the served model up to **TVD 0.0979** away from the model the tenant thought they trained. That is a quality question independent of speculation, and it is invisible to a tenant who evaluated their adapter against the base precision they fine-tuned on.

*The acceptance cost is not.* Deploying a QLoRA-trained adapter on an unquantised base changes the shared drafter's acceptance by only -0.0010 relative to the fp16-trained adapter, and no arm moves acceptance meaningfully. Round-to-nearest quantisation noise is close to isotropic in logit space, so unlike a tenant adapter — which moves the target coherently in one direction — it does not systematically pull the target away from the drafter. **The tenant's LoRA is the problem; the quantisation mismatch is not.**

The companion adapter, trained against whatever configuration is actually served, restores acceptance from 0.8923 to 0.9776 in every arm without being told which mismatch it is fixing — which is the operationally useful property, since the serving provider knows the served precision and the tenant often does not.


## 8. Limitations

* The pilot is a synthetic HMM testbed at ~2M parameters. It buys a computable
  ground truth and a clean causal attribution; it does not establish the
  magnitudes that a 8B/1B pair on natural text will show. The T4 harness in
  `experiments/gpu/` runs the identical measurement code on Qwen2.5-1.5B +
  Qwen2.5-0.5B with QLoRA tenants on real task datasets, and is the check that
  matters.
* `beta_0 = 0.980` is higher than production drafters achieve. Relative drops
  are the meaningful quantity here, not absolute ones.
* Throughput numbers in §5 come from a step-cost model driven by measured
  acceptance and measured per-forward latency, not from an end-to-end vLLM
  benchmark — vLLM cannot apply a per-request adapter to the drafter today,
  which is the point of the proposal.
* Companion adapters are trained per tenant. The cost is a few hundred
  distillation steps against rollouts the provider can generate offline, but it
  is not free, and tenants who update their task adapter must re-mint.

## 9. Related work

Speculative decoding and its lossless accept/reject rule (Leviathan et al.
2023; Chen et al. 2023); EAGLE-3.1 (vLLM/EAGLE/TorchSpec, May 2026); DFlash
block-diffusion drafting (June 2026) and its vLLM/Speculators integration.
Multi-tenant LoRA serving: S-LoRA/punica batched adapter kernels, LoRAX,
Toppings (ATC'25), MixLoRA (ICPP'25), InfiniLoRA. Drafter training objectives
that target acceptance directly rather than likelihood (Nebius LK losses);
domain-specific drafter training; OmniDraft's online adaptive drafter, which
adapts one drafter to one user online rather than minting per-tenant adapters
batched together. Negative result on same-backbone PEFT drafting:
arXiv:2607.12422. vLLM RFC #52038 for drafter-side LoRA.
