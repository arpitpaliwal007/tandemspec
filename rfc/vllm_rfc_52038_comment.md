# Comment on RFC #52038 — LoRA adapter support for DFlash speculative decoding draft models

*Draft for posting to https://github.com/vllm-project/vllm/issues/52038 while
the feedback window is open. Numbers below marked "pilot" come from a
controlled synthetic testbed, not from vLLM; the GPU protocol that would
produce vLLM-scale numbers is linked at the end.*

---

Strong +1 on adding LoRA to the drafter. One scoping question and one design
suggestion, both aimed at the case I think is most valuable and is currently
out of scope.

## The case the current proposal doesn't cover

As written, the drafter adapter is selected **per deployment** — one specialised
drafter for the whole server, chosen at startup via speculative config args.
That is the right fit for a single-domain deployment. It does not cover the
deployment vLLM's multi-LoRA path exists for: a shared base with many tenant
adapters, where a single batch contains requests from several tenants at once.

In that deployment there is a coupling the RFC doesn't name. The drafter is
trained against the **base** target. Every request is verified against
**base + tenant adapter**. The drafter is therefore approximating the wrong
distribution for every adapted request, and the mismatch shows up directly as
lost acceptance, because per-token acceptance probability under the standard
accept/reject rule is exactly

```
beta = sum_v min(p_v, q_v) = 1 - TVD(p_target, q_draft)
```

so any distribution shift the tenant adapter introduces is subtracted from
acceptance one-for-one, to first order.

Two consequences worth putting in the RFC explicitly:

1. **The penalty grows quadratically in speculation depth.** With `E[tokens/step]
   = (1 - beta^(gamma+1))/(1 - beta)`, expanding near high acceptance gives
   `dT/dbeta ≈ gamma(gamma+1)/2` — about 10x at `gamma=4`, ~36x at `gamma=8`.
   Since DFlash's whole value proposition is supporting larger blocks, the cost
   of an unadapted drafter *rises* as the drafter gets better.
2. **`gamma` is being tuned at the wrong operating point.** Optimal block size is
   monotone in `beta`, so a server that tunes `gamma` once against the base
   model over-drafts for every adapted tenant.

Pilot measurement (small synthetic target/drafter pair, 6.9x size ratio,
`gamma=4`; the GPU protocol for Qwen2.5-1.5B + Qwen2.5-0.5B with QLoRA tenants
is in the repo and runs on one 16 GB card):

| | acceptance beta | tokens/step | greedy top-1 agreement |
|---|---|---|---|
| shared drafter, unadapted target | 0.980 | 4.81 | 0.965 |
| + in-distribution tenant adapter | 0.887 | 3.99 | 0.777 |
| + held-out tenant adapter | **0.259** | **1.35** | **0.003** |

For tenants fine-tuned on data the base never saw, speculation stops paying
entirely, and under greedy decoding the drafter's argmax essentially never
matches the target's. Acceptance falls linearly in the measured distribution
shift `TVD(p_base, p_adapted)` with slope 0.82, exactly as first-order theory
predicts.

Repair, same testbed: a rank-4 LoRA on the drafter (10,240 parameters, **20 KB
in fp16**) trained by on-policy distillation from the adapted target lifts mean
acceptance from 0.572 to 0.912 — 83% of the lost acceptance recovered. A full
private drafter fine-tune, 26x the parameters, reaches 0.949 (92%). Two
ablations worth noting for whoever writes the stage-2 recipe: training the same
adapter with hard-label cross-entropy on the tenant's own tokens instead of
distilling from the adapted target recovers only **10%**, and a TVD objective
beats forward KL by +0.012 at equal rank and step budget.

## Suggestion: make the drafter adapter per-request, not per-deployment

The change is small relative to what the RFC already proposes, and it reuses
machinery that exists:

* **Registry, not a new request field.** Keep `LoRARequest` as is and resolve a
  *pair* from the tenant's `lora_int_id`: `(target task LoRA, draft companion
  LoRA)`. Users who don't opt in see no API change, and draft adapter ids can
  live in a disjoint numeric range so the two managers key their caches
  independently.
* **Reuse the target's index mapping.** The proposer's model runner builds its
  own `LoRAModelManager` over the draft layers and calls `set_active_loras` with
  a `LoRAMapping` derived from the same per-request index mapping the target
  batch already computed, so the two stay aligned position-for-position.
* **`max_loras` becomes a constraint on pairs.** Its meaning — distinct tenants
  admitted per batch — is unchanged. The draft side allocates the same
  cardinality at a smaller rank, so the binding memory constraint stays the
  target side.
* **Separate rank knob.** `max_draft_lora_rank` defaulting to 4-8 rather than
  reusing `max_lora_rank`; rank-padding waste in the SGMV layout otherwise
  dominates on the small drafter.

Excluding `lm_head`/`embed_tokens` from adapter wrapping (as the RFC already
specifies, to dodge the vocab-padding mismatch from #11966) is compatible with
all of this.

Memory, Llama-3.1-8B target + Llama-3.2-1B drafter, 64 tenants: rank-32 task
adapters are ~168 MB/tenant; rank-4 companion adapters on the drafter are
~5.6 MB/tenant — **3.4% overhead on adapter memory, and ~440x smaller than the
2.5 GB a private per-tenant drafter would cost.** That ratio is what makes
per-request drafter adapters practical at tenant counts where per-deployment
specialisation isn't an option.

## Suggestion: specify the training target

The RFC says adapters are "loaded from stage-2-trained LoRA checkpoints" but
doesn't say what distribution they are trained against. For the multi-tenant
case the answer matters and is slightly counter-intuitive:

* Train the drafter adapter against the **adapted** target (`base + tenant
  LoRA`), not the base target.
* Train it on **rollouts sampled from that adapted target**. This is already
  on-policy: speculative sampling is distribution-preserving, so the contexts
  the drafter is invoked on in deployment are distributed exactly as the
  target's own output. No drafter rollouts or interleaved sampling are needed.
* Consider a **TVD objective** rather than KL. Since `beta = 1 - TVD(p, q)`,
  minimising total variation distance *is* maximising expected acceptance;
  forward KL is only a surrogate. (Related: Nebius's LK losses for acceptance-
  maximising drafter training.)

Worth noting for anyone who read arXiv:2607.12422 ("Accepted Prefixes Are Not
All You Need") as an argument against PEFT drafting: that negative result is
about a LoRA used as a drafter **on the target's own backbone**, where drafting
costs a full backbone forward (~50.3 ms draft vs ~50.6 ms verify). It does not
apply here — a DFlash drafter is a separate small network, and the adapter only
changes which distribution that cheap forward approximates. The authors scope
their claim to same-backbone adapters explicitly.

## What I can contribute

Reference implementation of the pairing logic, admission rule and memory
accounting; the acceptance measurement harness (verified against the closed-form
accept rate, and against a losslessness check); the companion-adapter training
recipe; and a GPU protocol sized for a single 16 GB card so the effect can be
reproduced cheaply before anyone commits to kernel work. Happy to open a draft
PR against whichever shape of the config API you prefer.
