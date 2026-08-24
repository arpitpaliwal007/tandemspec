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

<!-- RESULTS:E1 -->
<!-- RESULTS:E2 -->
<!-- RESULTS:E3 -->

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

<!-- RESULTS:E4 -->

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
