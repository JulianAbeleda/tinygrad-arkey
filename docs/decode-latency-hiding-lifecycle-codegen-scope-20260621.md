# Decode Latency-Hiding Lifecycle / Codegen Scope

Date: 2026-06-21

Owner: next executor

Status: scope only

## Context

The bounded decode-fusion lane is exhausted.

`docs/decode-fusion-build-result-20260620.md` built a real FFN activation fusion candidate and refuted the expected
win:

- fused `silu(gate) * up` into the up-GEMV producer;
- byte-exact;
- eliminated the standalone `E_49152` launch;
- net speedup `~0%`.

The result proves the activation cost was real work, not recoverable launch overhead. The attention split also
reframed the proposed reduce/stat fusion: the dominant `reduce_fixup` and `softmax_stats` costs are mostly intrinsic
O(KV) QK / softmax work, not removable helper overhead. The current flash path's real next lever is a fully fused,
tiled lifecycle that interleaves QK, softmax, and V accumulation while hiding scalar/stat work under memory traffic.

Current decode state remains:

| route | ctx512 | ctx1024 | ctx2048 | ctx4096 | status |
|---|---:|---:|---:|---:|---|
| baseline | `68.0` | `66.5` | `63.5` | `60.8` tok/s | default |
| q8 opt-in | `72.8` | `~71.0` | — | `64.5` tok/s | default-off |
| llama.cpp | `98.6` | `97.6` | `95.4` | `92.2` tok/s | reference |

This scope is for the next valid frontier: latency hiding and larger lifecycle codegen. It is not another
micro-fusion scope.

## Objective

Decide whether tinygrad should fund a deep decode lifecycle/codegen project, and if yes, produce one minimal
prototype that proves latency hiding rather than launch removal.

The scope has three acceptable outcomes:

1. **Proceed:** one prototype recovers real full-route W==D speed by hiding work inside a larger tiled lifecycle.
2. **Roadmap:** the design is valid but blocked by known codegen/linearizer capabilities; produce a precise compiler
   backlog.
3. **Stop:** no bounded prototype can prove latency hiding; keep decode defaults and avoid more custom one-off
   fusion work.

## Non-Goals

Do not do these unless a phase below explicitly reopens them:

- no standalone `silu(gate)*up` fusion;
- no standalone attention helper/reduce-stat fusion;
- no `FLASH_L=256/512` retest without a new fused lifecycle;
- no `FLASH_DECODE=0` SDPA retest;
- no q6/MMVQ role reopening;
- no host-sync or persistent-runtime work;
- no broad rewrite before a minimal prototype proves movement.

The previous evidence says these are closed or non-causal.

## Measurement Authority

Use two classes only:

| class | purpose | authority |
|---|---|---|
| clean W==D wall | promotion | `PROFILE=0`, same model/prompt/ctx, median-of-5 minimum |
| pinned local diagnostic | prototype attribution | peak-clock pin allowed, restored to auto, never headline alone |

Every result must report:

- ctx `512,1024,2048,4096` where applicable;
- default vs candidate tok/s;
- VRAM;
- clock policy;
- correctness / tok0 / dNLL policy;
- whether `tinygrad/llm/model.py` default behavior changed.

## Phase 0 — Reconcile Current Headline First

Before starting codegen work, complete or reference:

- `docs/decode-prefill-headline-reconciliation-scope-20260621.md`

Gate:

- the current decode headline must be known: either the `60.8-68.0 tok/s` table stands, or the `87.6 tok/s` claim is
  proven clean-wall and the target table is updated.

Stop if this is unresolved. A deep codegen project needs a stable baseline.

## Phase 1 — Latency-Hiding Opportunity Atlas

Build an atlas of where llama appears to hide work that tinygrad exposes.

Inputs:

- `docs/decode-fusion-build-result-20260620.md`
- `docs/decode-attention-elementwise-result-20260620.md`
- `docs/decode-current-route-attribution-result-20260620.md`
- `docs/decode-role-tensor-kernel-attribution-result-20260620.md`
- current `extra/qk_flash_decode.py`
- current q4/q6 GEMV primitive files
- llama launch counts / timing rows already captured in local artifacts

Required table:

| family | tinygrad visible cost | likely hidden-under row | why llama hides it | tinygrad blocker | prototype candidate |
|---|---:|---|---|---|---|

Must cover:

- flash decode QK / online softmax / V accumulation;
- FFN gate/up activation and down projection;
- rope / residual add / RMSNorm glue only if they remain above `0.25ms/tok`;
- q8 opt-in route as policy-only, not a new lifecycle target unless evidence changes.

Gate:

- identify at least one candidate whose expected recovered full-route movement is `>=5% @ctx1024` or
  `>=7% @ctx4096`;
- prove the candidate is latency hiding or work reduction, not launch-count reduction alone.

## Phase 2 — Choose One Minimal Prototype

Pick exactly one prototype. Rank order:

### Candidate A — Fully Fused Flash Decode Tile

Goal:

- one tiled lifecycle that owns QK score production, online softmax state, and V accumulation for a KV tile;
- avoid materializing intermediate score/stat tensors outside the tile;
- interleave exp/stat/scalar work with KV loads and dot products.

Why first:

- biggest ctx-growing gap;
- direct route to llama-like flash lifecycle;
- validates the linearizer/codegen wall explicitly.

Prototype can be research-only under `extra/`, but must run on real inputs.

Local gate:

- correctness vs current flash decode / SDPA within existing tolerance;
- at least one real ctx (`1024` or `4096`) runs end-to-end for one layer or a full route micro-harness;
- local timing shows recovered movement not explained by launch count alone.

Full W==D gate:

- `>=5%` speedup @ctx1024 or `>=7%` @ctx4096;
- no ctx512 regression >`1%`;
- no quality regression outside existing decode policy.

### Candidate B — GEMV Epilogue Latency-Hiding Activation

Goal:

- do not merely move `silu*mul` into the GEMV store;
- interleave activation with the producer's accumulation/load lifecycle or the down projection's input consumption
  so activation latency is overlapped.

Why second:

- Phase B proved naive producer fusion conserves work;
- useful only if the prototype changes scheduling/overlap, not expression placement.

Local gate:

- compare against the already-refuted producer-fusion kernel;
- must beat it by `>=5%` locally, not just match baseline;
- prove reduced exposed activation latency with timing split or instruction schedule evidence.

Full W==D gate:

- `>=3%` speedup @ctx1024;
- no ctx regression.

### Candidate C — Codegen Capability Spike

Use only if A/B cannot be expressed without compiler work.

Goal:

- implement the smallest linearizer/codegen capability that enables A or B:
  - multi-output tiled lifecycle;
  - explicit software-pipelined K-loop;
  - tile-local online reduction;
  - producer/consumer epilogue-prologue fusion with scheduling control;
  - scratch/LDS/register allocation contract for tiled decode.

Gate:

- produce a runnable toy kernel with the required scheduling pattern;
- produce a precise list of compiler deltas needed for model-route integration;
- no performance headline allowed unless it runs the real decode shape.

## Phase 3 — Prototype Build Rules

Every prototype must be default-off.

Allowed locations:

- `extra/` research harnesses;
- `bench/qk-decode-latency-hiding-lifecycle/` artifacts;
- minimal compiler hooks only if scoped and isolated.

Required artifacts:

- `generated_candidates.json`
- `refutations.json`
- prototype timing JSON;
- correctness JSON;
- result doc.

Required result table:

| candidate | correctness | local movement | W==D movement | default changed | verdict |
|---|---|---:|---:|---|---|

Promotion gate:

- only wire into model route behind an env flag after full W==D gate passes;
- only consider default after a separate owner approval.

## Phase 4 — Lifecycle Search Encoding

Update the lifecycle search ledger so the project remembers the difference between:

- launch-removal fusion;
- work-conserving expression fusion;
- latency hiding;
- true work reduction;
- linearizer/codegen blocked.

Add rows for:

- `ffn_activation_producer_fusion_refuted_work_conserved`;
- `attention_reduce_stat_microfusion_no_go_intrinsic_okv`;
- selected deep prototype candidate;
- any compiler capability spike.

Artifact target:

```text
bench/qk-lifecycle-search/
```

## Phase 5 — Result Doc

Write:

- `docs/decode-latency-hiding-lifecycle-codegen-result-20260621.md`

Minimum sections:

1. baseline and authority labels;
2. opportunity atlas;
3. selected prototype and why;
4. correctness;
5. local diagnostic timing;
6. full W==D timing or reason not run;
7. lifecycle-search encoding;
8. default behavior changed: yes/no;
9. recommendation: proceed / roadmap / stop.

## Stop Conditions

Stop immediately if:

- the candidate only removes launches and does not alter work scheduling;
- local timing matches the work-conserved FFN fusion result;
- the prototype requires broad compiler surgery before a toy scheduling proof exists;
- correctness cannot be made byte-exact or within accepted decode tolerance;
- W==D movement is below gate after a passing local prototype;
- the `87.6 tok/s` headline ambiguity remains unresolved.

## Expected Project Decision

If no Phase-2 prototype clears its gate, decode should be marked:

> bounded fusion exhausted; remaining path is broad lifecycle/codegen research.

If Candidate A clears its gate, the project should fund fully fused flash decode as the main decode workstream.

If Candidate C is the only path, the project should decide explicitly whether the north-star machine-search/codegen
goal is worth starting now, because this is no longer a tactical decode optimization.
