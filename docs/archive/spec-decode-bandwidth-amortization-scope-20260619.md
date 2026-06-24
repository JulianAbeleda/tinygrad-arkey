# Spec decode as weight-read amortization - scope - 2026-06-19

Purpose: reopen speculative decode only under the new PMU-backed framing: decode is dominated by streaming quantized
weights from HBM, so the only high-EV decode route is to amortize one target weight read over multiple accepted
tokens. This is **not** the old `spec_verify_single_kernel` row; that row stays closed.

This is a scope only. It does not route `SPEC_DECODE`, change defaults, or run a model path.

## Why this reopens

The primitive PMU atlas changed the ranking:

- ~85% of decode GPU time is bandwidth-bound weight GEMV.
- L2 hit is low for the dominant GEMVs, so weights are streamed from HBM.
- VALU utilization is low, so decode is not compute-bound at whole-model scale.
- Effective bandwidth is about 38% of peak for tinygrad and roughly 47-49% for llama-class decode.

That means another 5-10% kernel improvement is not the main decode lever. The main lever is fewer target weight
streams per emitted token.

Speculative decode can do that if:

```text
accepted_tokens_per_pass > target_verify_cost_in_target_passes + draft_cost_in_target_passes + runtime_overhead
```

The acceptance side already looks good. The verify/runtime side does not.

## Evidence ledger

| evidence | current result | consequence |
|---|---|---|
| target decode bound | PMU atlas: dominant GEMVs bandwidth-bound | weight-read amortization is the right objective |
| 0.6B draft acceptance | `2.844` accepted/pass at K=4, draft `273 tok/s` | algorithmic acceptance is good enough |
| old production spec prototype | `0.24-0.26x`, greedy-exact | host/sync structure killed naive route |
| low-sync proposal graph | reusable K-step proposer PASS | draft proposal graph is not the main correctness risk anymore |
| batched verify T=5 | `58.96ms = 4.66x` one T=1 pass | current verify does **not** amortize target weight reads |
| component breakdown | Q4_K + Q6_K + attention all T-scale | no single verify kernel fixes the route |

## Reopened row

New lifecycle candidate:

```text
decode_spec_weight_amortization_lifecycle
```

Definition:

- producer: draft model proposes K tokens with a reusable low-sync graph;
- format: target verifies a short T=K+1 block exactly;
- consumer: target forward whose dominant weight reads are amortized across T tokens;
- routing: one bounded `SPEC_DECODE=1` research route, default off;
- quality: greedy byte-exact versus target-only decode;
- gate: W==D decode speedup, accepted/pass, syncs/pass, no KV corruption.

This row bypasses the old refutation only if it changes the verify lifecycle. It does **not** bypass the old
refutation if it uses the current T>1 fallback path.

## Closed row that stays closed

`decode_spec_verify_shortcut` remains closed.

Why:

- current T=K+1 verify is `4.66x` one pass at T=5;
- every major component scales with T;
- routing the existing batched verify is not enough;
- a single Q4_K reuse kernel is not enough.

The reopened row must solve the full verify lifecycle, not rename the old shortcut.

## Speed model

Let:

- `A` = accepted tokens per pass;
- `D` = draft proposal cost in target-pass units;
- `V` = target verify cost in target-pass units;
- `R` = runtime/accept overhead in target-pass units.

Then:

```text
speedup = A / (D + V + R)
```

Measured/proven inputs:

| item | value |
|---|---:|
| 0.6B K=4 accepted/pass | `2.844` |
| 0.6B draft tok/s | `273` |
| target tok/s used by old gate | `55` |
| draft cost for K=4 in target-pass units | `4 * (55/273) = 0.81` |
| current T=5 verify cost | `4.66` |

Current verify route:

```text
speedup ~= 2.844 / (0.81 + 4.66 + R) < 0.52x before runtime overhead
```

So current verify is dead even with perfect accept logic.

Required verify:

| verify cost `V` | runtime `R=0` speedup | runtime `R=0.2` speedup |
|---:|---:|---:|
| `1.0` | `1.57x` | `1.41x` |
| `1.25` | `1.38x` | `1.26x` |
| `1.5` | `1.23x` | `1.14x` |
| `2.0` | `1.01x` | `0.94x` |

Conclusion: spec decode is only viable if target verify is about `1.0-1.5x` one target pass and runtime overhead is
kept low. That is the hard gate.

## Missing primitive

The missing primitive is a **T-cheap target verify forward**:

```text
read each target weight stream once -> compute logits for T=K+1 positions -> preserve exact causal/KV semantics
```

It requires all of these, not one of them:

1. Q4_K roles: T>1 weight-read reuse for ffn_gate/up, ffn_down, attn_q/o.
2. Q6_K roles: T>1 weight-read reuse for ffn_down/lm_head and any Q6_K projections.
3. Attention: short-block causal verify attention over existing KV plus intra-block KV, without falling to the slow
   prefill-style path.
4. Runtime: proposal + verify + accept/commit with at most one or two exposed syncs per pass.
5. KV protocol: exact commit/rollback for target and draft caches, including zero/partial/full accept.

This is a batched-forward lifecycle, not a local MMVQ tweak.

## Execution phases

SDB-0: reconcile and bank the reopened row.

- Add `decode_spec_weight_amortization_lifecycle` to lifecycle search.
- Keep `decode_spec_verify_shortcut` closed.
- Gate: generated ledger shows both rows with different meanings.

SDB-1: analytic viability model. **DONE**

- Use current acceptance, draft speed, target speed, verify `T` ladder, and runtime overhead.
- Emit a JSON model for K=2/3/4/8.
- Gate: identify the maximum verify cost that still clears `1.2x` and `1.5x`.

Result: `spec-decode-bandwidth-amortization-sdb1-sdb2-result-20260619.md` builds the model. With the 0.6B draft,
current verify gives only about `0.52x` before runtime overhead. A `>=1.2x` route needs verify around `<=1.3-1.4x`
one pass once a small runtime allowance is included.

SDB-2: verify-fastpath design audit. **DONE**

- Map each target component at T=K+1:
  - Q4_K GEMV/GEMM roles;
  - Q6_K GEMV/GEMM roles;
  - lm_head;
  - attention/reduces;
  - norms/elementwise.
- For each component, state whether an existing primitive can be made T-cheap.
- Gate: either name a shared primitive that moves enough of verify, or classify as project-level batched-forward.

Result: `NO_BOUNDED_SHARED_PRIMITIVE`. T=5 verify needs a `67.8%` cut to reach `<=1.5x` one pass, but Q4_K,
Q6_K/lm_head, and attention/reduces are co-dominant. No single component or existing primitive can move enough.
The missing route is project-level T-cheap batched-forward.

SDB-3: minimal proof candidate. **NOT EARNED**

- Do not start with full model routing.
- Build or import one T-cheap verify block only if SDB-2 finds a credible shared primitive.
- Gate: T=5 verify `<=1.5x` one T=1 pass, exact argmax.

Do not start this as a bounded kernel proof unless a credible project-level T-cheap batched-forward route is funded.

SDB-4: low-sync integration.

- Reuse the proven proposal graph.
- Add target verify, accept, and KV commit/rollback under `SPEC_DECODE=1`.
- Gate: greedy byte-exact, no KV corruption, syncs/pass bounded, W==D `>=1.2x`.

SDB-5: final verdict.

- If verify stays `>2x` one pass, close spec as blocked by target verify lifecycle.
- If verify reaches `1.0-1.5x` but runtime loses, scope runtime graph/accept work.
- If both pass, keep as research flag and run long-context/prompt diversity.

## Kill gates

Kill or defer immediately if any of these remain true:

- T=K+1 verify stays `>2x` one target pass.
- Attention + Q4_K + Q6_K remain co-dominant with no common T-cheap route.
- Accept/commit requires multiple large host reads per pass.
- KV rollback corrupts zero/partial/full accept cases.
- A working route is not greedy byte-exact.

## Current verdict before new work

Spec decode is **the right decode lever by bottleneck class**, but **not a bounded buildable speed path** after
SDB-1/SDB-2. The current implementation is killed by verify and runtime overhead, and the T-cheap verify forward is
classified as project-level batched-forward compiler/runtime work.
