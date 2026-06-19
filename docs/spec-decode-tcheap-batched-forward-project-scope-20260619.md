# Spec decode T-cheap batched-forward project scope - 2026-06-19

Purpose: scope the project-level route that would make speculative decode viable after SDB-1/SDB-2 classified the
bounded route as insufficient.

This is not a request to route `SPEC_DECODE`, build a one-off Q4_K kernel, or reopen the old verify shortcut. It is
the larger compiler/runtime route required before SDB-3 can earn a build.

## Starting point

Known facts:

- Decode is HBM weight-stream bound at whole-model scale.
- Speculative decode is the right algorithmic class if one target weight stream can produce multiple accepted tokens.
- Acceptance is good enough with the 0.6B draft: `2.844` accepted/pass at K=4.
- Current target verify is not close: T=5 verify is `58.960ms = 4.652x` one T=1 pass.
- For a practical `>=1.2x` route with small runtime overhead, verify needs about `<=1.3-1.4x` one pass.
- T=5 verify needs a `67.8%` cut, distributed across Q4_K, Q6_K/lm_head, and attention/reduces.
- No single bounded primitive can cut enough.

Therefore the project target is:

```text
T=K+1 target verify forward <= 1.3-1.5x one T=1 target pass
while preserving exact causal/KV semantics and low-sync accept/commit.
```

## Definition of T-cheap batched-forward

A T-cheap batched-forward is a short-block target forward for speculative verify:

```text
input: accepted context at position L plus K draft tokens
output: target argmax/logits for positions L..L+K
constraint: same output as target-only greedy verify
```

It is T-cheap only if the dominant target work does not scale approximately linearly with T.

Required lifecycle:

1. Draft proposal graph produces K tokens with no per-token host sync.
2. Target verify reads dominant weights once or near-once for T=K+1 positions.
3. Short-block attention handles existing KV plus intra-block causal dependencies.
4. Accept/commit/rollback updates target and draft KV exactly.
5. The pass exposes at most one or two syncs to Python.

## Required capabilities

| capability | why needed | current state |
|---|---|---|
| Q4_K short-block weight reuse | Q4_K is ~31.6% of T=5 verify | current T>1 path scales; Q4_K-only not enough |
| Q6_K/lm_head short-block weight reuse | Q6_K/lm_head is ~16.6% of T=5 verify | must move with Q4_K, not after it |
| short-block verify attention | attention/reduces is ~48.6% of T=5 verify | existing flash-decode is T==1-gated; prefill-style path is too slow |
| graph-level verify composition | component wins must be captured as one verify route | old loop lost to host/sync |
| device-side accept/commit protocol | host reads/accept loops kill production spec | low-sync proposal graph exists; full accept/commit does not |
| KV rollback/commit correctness | zero/partial/full accept must be exact | prior prototypes found draft KV cache-hole risks |
| shape-specialized short T | K is small and concrete; generic prefill T is wrong regime | needs separate route from PREFILL_V2 |

## Candidate architecture

The credible architecture is a **specialized short-block verify graph**, not a normal prefill call:

```text
draft_propose_graph(K)
  -> target_verify_shortblock_graph(T=K+1)
       - Q/K/V/O/FFN/lm_head short-block linears
       - short-block causal attention over existing KV plus proposed KV
       - target argmax vector
  -> accept_prefix_and_commit
       - compare draft tokens vs target tokens
       - write accepted target/draft KV
       - rollback or ignore unaccepted temporary KV
```

Important boundary: target verify does not need to be general large-T prefill. It only needs `T in {3,4,5}` for the
viable K range. That allows shape-specialized kernels/graphs, but the whole forward still has to move together.

## Why individual shortcuts are insufficient

| shortcut | reason it fails |
|---|---|
| current batched verify | measured `4.65x` one pass at T=5 |
| Q4_K-only reuse | at most ~31.6% of verify; cannot reach `<=1.5x` |
| attention-only improvement | even making attention/reduces free does not meet the gate |
| serial T==1 verify | K+1 target passes, no amortization |
| naive low-sync integration | exact but `0.24-0.26x`; exposed sync/accept overhead |
| generic prefill path | wrong regime; T small, latency-sensitive, with decode KV semantics |

## Phases

### TBF-0 - authority lock - DONE

Goal: decide whether this project is worth funding as a project-level route.

Inputs:

- `spec-decode-bandwidth-amortization-sdb1-sdb2-result-20260619.md`
- `bench/qk-spec-decode-bandwidth-amortization/model.json`
- lifecycle row `decode_spec_weight_amortization_lifecycle`

Gate:

- proceed only if the project accepts a multi-component compiler/runtime arc.

Kill:

- if the expected next step must be a bounded kernel edit, stop here.

Result: `spec-decode-tcheap-batched-forward-tbf0-tbf2-result-20260619.md` executes this as a read-only decode audit.

### TBF-1 - short-block verify IR contract - DONE

Goal: define a separate target forward mode for speculative verify.

Work:

- define legal `T` values, likely `T=3/4/5`;
- define exact inputs: token block, start position, target KV cache view, temporary KV writes;
- define exact outputs: target token predictions and commit metadata;
- define fallback to current target-only decode;
- define shape guards so this route never affects normal decode/prefill.

Artifact:

- `bench/qk-spec-tcheap-forward/ir_contract.json`

Gate:

- graph contract can represent all target blocks and KV semantics without changing normal decode.

Result: contract defined in `bench/qk-spec-tcheap-forward/ir_contract.json`.

### TBF-2 - component authority probes - DONE

Goal: prove the best possible component ceilings before implementing a full route.

Rows:

| component | target gate |
|---|---:|
| Q4_K + Q6_K/lm_head short-block linears | combined `<=1.5x` their T=1-equivalent cost for T=5 |
| short-block attention/reduces | `<=1.5x` T=1-equivalent attention cost for T=5 |
| elementwise/norm | no gate unless it regresses |
| accept/commit | no large host token/vector reads per pass |

Important: a component pass does not route spec. It only proves the component is not the blocker.

Kill:

- any one of Q4_K/Q6_K/attention remains ~T-linear after a credible attempt.

Result: current baseline fails every component gate. Q4_K is `2.916x`, Q6_K/lm_head is `5.831x`,
attention/reduces is `3.061x`, and linears as a group are `3.523x` their T=1-equivalent at T=5.

### TBF-3 - linears route - NOT EARNED

Goal: build/import the short-block quantized-linear route.

Possible implementation paths:

- native tinygrad short-block quantized kernels;
- extracted backend artifacts if a mature backend has the right small-T quantized shape;
- project-level AMD renderer feature for grouped short-T GEMV/GEMM.

Gate:

- linears as a group cut enough that full verify has a credible path to `<=1.5x`.

Kill:

- Q4_K-only or Q6_K-only wins that do not move the group.

Do not start TBF-3 until a concrete grouped short-block linears candidate exists and TBF-2 is rerun against it.

### TBF-4 - short-block attention route

Goal: handle verify attention for T=K+1 without falling to slow prefill-style SDPA.

Required semantics:

- existing KV prefix;
- proposed intra-block K/V;
- causal mask within the proposed block;
- exact enough for greedy byte identity.

Gate:

- attention/reduces no longer dominate T=5 verify.

Kill:

- reuse-free or cache-hostile attention repeats prior refutations.

### TBF-5 - graph/runtime accept route

Goal: compose proposal, verify, accept, and commit without the old production-sync failure.

Work:

- reuse proven low-sync proposal graph;
- keep target predictions on device until accept prefix is computed;
- avoid two large `tolist`/host reads per pass;
- commit accepted KV exactly;
- handle zero, partial, and full accept.

Gate:

- syncs/pass bounded;
- greedy byte-exact;
- no KV corruption over prompt diversity.

### TBF-6 - end-to-end SPEC_DECODE research flag

Only after TBF-2 through TBF-5 pass.

Gate:

- `SPEC_DECODE=1`;
- default off;
- W==D speedup `>=1.2x`, strong `>=1.5x`;
- greedy byte-exact;
- prompt diversity and ctx sweep;
- fallback on unsupported shape/device.

## Stop conditions

Stop and keep spec project-level if:

- T=5 verify cannot get below `2x` one T=1 pass;
- any of Q4_K, Q6_K/lm_head, or attention remains independently T-linear;
- accept/commit requires repeated host reads;
- KV protocol cannot be made exact without full re-prefill;
- implementation becomes per-shape assembly worse than artifact policy.

## Relationship to other project-level work

This route overlaps with, but is not identical to, the AMD scheduler/codegen project:

- shared: schedule quality, register allocation, software pipelining, graph capture, HCQ runtime discipline;
- unique to spec: causal short-block attention, KV commit/rollback, accept-prefix logic, two-model graph composition;
- shared with prefill: batched forward and T>1 linears;
- distinct from prefill: small T and decode-style KV semantics, not large prompt throughput.

## Recommendation

Do not start this as the next implementation arc unless the project explicitly wants a multi-component compiler/runtime
effort. The evidence supports the idea, but the scoped work is larger than the artifact routes and larger than a
single primitive. If funded, start with TBF-1 and TBF-2 only; do not route `SPEC_DECODE` until component ceilings show
that T=5 verify can plausibly reach `<=1.3-1.5x` one target pass.
