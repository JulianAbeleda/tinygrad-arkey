# Decode Fused+Coop Primitive Roadmap Scope

Date: 2026-06-21

Owner: Claude 1

Status: scope only

## Context

`docs/decode-latency-hiding-lifecycle-codegen-result-20260621.md` closed the bounded decode-fusion lane:

- FFN activation fusion is refuted as work-conserved.
- attention reduce/stat micro-fusion is refuted / no-go.
- raw fully fused flash tile is byte-exact but `2.5-3.3x` slower than the optimized split UOp path.
- steady-context decode remains `~67%` llama.

The load-bearing conclusion is:

> fusion alone is not the win; the next valid decode target is fused + coop-optimized in one primitive.

The raw fused tile has the lifecycle shape but loses the current UOp path's GQA V-reuse, coalescing, and dataflow.
The current UOp `gqa_coop_vec` path has the cooperative dataflow but not the fully fused lifecycle. Llama wins by
having both in one primitive.

This scope is not a build request. It is a roadmap/design package for deciding whether to fund the multi-week
codegen project and which mechanism to pursue first.

## Objective

Produce a concrete implementation roadmap for a tinygrad decode primitive that combines:

1. QK score production;
2. online softmax state;
3. V accumulation;
4. GQA V-reuse;
5. coalesced vector loads;
6. graph/JIT integration;
7. correctness and W==D promotion gates.

The output must choose one of three decisions:

| decision | meaning |
|---|---|
| `BRIDGE_FIRST` | raw-kernel / custom-kernel bridge is the shortest path to a fused+coop primitive |
| `LINEARIZER_FIRST` | coupled multi-reduce / UOp compiler support is the shortest path |
| `ROADMAP_ONLY` | neither path is implementation-ready; keep decode at current route |

## Non-Goals

Do not build another benchmark kernel unless it is needed to answer a specific design unknown.

Explicitly forbidden:

- no new standalone `silu(gate)*up` fusion;
- no attention helper-kernel micro-fusion;
- no launch-count-only optimization;
- no raw fused tile v2 unless the design first proves which missing coop feature it adds;
- no model-route default changes;
- no global prefill policy changes.

## Phase 1 — Diff The Winning And Losing Attention Paths

Compare these two paths at the level of dataflow, not just kernel count:

| path | property |
|---|---|
| raw fused flash tile | byte-exact, lifecycle fused, slower |
| UOp `gqa_coop_vec` | optimized split coop path, default winner |

Required table:

| feature | raw fused tile | UOp coop path | performance implication | required in fused+coop primitive |
|---|---|---|---|---|

Must cover:

- GQA V-reuse across query heads;
- V-load coalescing;
- K/Q load pattern;
- online softmax state placement;
- split/reduce strategy;
- workgroup shape / occupancy;
- register pressure;
- LDS/scratch usage if any;
- graph/JIT integration;
- why raw fused lost despite fewer kernels.

Inputs:

- `extra/qk_decode_fused_flash_tile_ab.py`
- `extra/qk_flash_decode.py`
- `docs/decode-latency-hiding-lifecycle-codegen-result-20260621.md`
- `docs/qk-gqa-coop-vector-load-result-20260617.md`
- `docs/qk-8b-flash-variant-result-20260617.md`
- `docs/qk-decode-attention-v3-result-20260617.md`
- `docs/qk-llama-token-primitive-accounting-20260617.md`

Gate:

- identify the smallest missing feature set that could plausibly move raw fused from `0.30-0.40x` to parity or
  better against UOp coop;
- mark any feature as `must-have`, `nice-to-have`, or `not causal`.

## Phase 2 — Mechanism A: Raw-Kernel / JIT Bridge Feasibility

Assess whether tinygrad can route a hand-written fused+coop attention kernel through the model graph without losing
warm-JIT lifecycle, scheduling, correctness, or artifact ownership.

Questions to answer:

1. Can a raw C / HIP / AMD source kernel be represented as a graph node in the current JIT?
2. Can its buffers, shapes, strides, dtypes, and env-policy be expressed without breaking graph capture?
3. Can it participate in HCQ graph replay without host sync?
4. Can it be made deterministic enough for decode correctness gates?
5. Can it be packaged without creating an unmaintainable external artifact boundary?

Required output:

| bridge requirement | current support | missing delta | risk | owner file(s) |
|---|---|---|---|---|

Read:

- `docs/amd-decode-flash-attention-plan.md`
- `docs/prefill-external-rawhip-tensile-boundary-scope-20260619.md`
- `extra/qk_flash_decode.py`
- current `custom_kernel` implementation paths in tinygrad.

Gate for `BRIDGE_FIRST`:

- bridge can be implemented with isolated, default-off changes;
- no new host sync in decode;
- raw kernel can be warmed/captured once and replayed;
- expected maintenance surface is bounded and documented;
- fused+coop attention can be verified against existing flash decode / SDPA.

## Phase 3 — Mechanism B: Linearizer Coupled Multi-Reduce Feasibility

Assess whether tinygrad's UOp path can express the optimized fused+coop primitive directly.

Required capability list:

- coupled q·k reduce + online softmax reduce in one kernel;
- multi-accumulator / multi-output tile state;
- tile-local online max/den/prob state;
- V accumulation coupled to softmax state;
- scheduling constraints to preserve GQA V-reuse and coalesced vector loads;
- register/LDS resource policy;
- graph/JIT compatibility.

Required output:

| compiler capability | why needed | current blocker | smallest spike | risk |
|---|---|---|---|---|

Read:

- `extra/qk_flash_decode.py` comments around the linearizer wall;
- existing UOp / linearizer constraints relevant to reductions;
- prior prefill codegen wall docs if they clarify scheduling/resource limits.

Gate for `LINEARIZER_FIRST`:

- a toy UOp kernel can express nested/coupled reductions without illegal range ordering;
- the path can preserve the known UOp coop advantages;
- implementation is reusable for the north-star lifecycle search system, not only this one attention kernel.

## Phase 4 — Decision Matrix

Produce a ranked decision table:

| route | expected upside | implementation cost | maintenance risk | integration risk | north-star value | verdict |
|---|---:|---|---|---|---|---|
| raw-kernel/JIT bridge | | | | | | |
| linearizer coupled multi-reduce | | | | | | |
| rest decode | | | | | | |

Scoring rule:

- prefer the path that can produce a correct, warm-JIT, graph-integrated fused+coop attention candidate fastest;
- break ties in favor of the path that advances the closed lifecycle machine-search north star;
- if neither can produce a credible first candidate, return `ROADMAP_ONLY`.

## Phase 5 — Implementation Scope For The Chosen Route

If the verdict is `BRIDGE_FIRST` or `LINEARIZER_FIRST`, write a second document:

- `docs/decode-fused-coop-primitive-implementation-scope-20260621.md`

It must include:

1. exact files to edit;
2. feature flags;
3. toy kernel gate;
4. one-layer attention gate;
5. full W==D gate;
6. correctness/dNLL/tok0 gate;
7. rollback plan;
8. artifact paths.

Full-route promotion gates:

- `>=5%` speedup @ctx1024 or `>=7%` @ctx4096;
- no ctx512 regression >`1%`;
- no output/quality regression outside existing decode policy;
- no decode default change without owner approval.

If the verdict is `ROADMAP_ONLY`, write instead:

- `docs/decode-fused-coop-primitive-roadmap-result-20260621.md`

with the exact blockers and the next smallest proof needed.

## Required Artifacts

Write artifacts under:

```text
bench/qk-decode-fused-coop-primitive/
```

Minimum:

- `path_diff.json`
- `bridge_feasibility.json`
- `linearizer_feasibility.json`
- `decision_matrix.json`

Update lifecycle search:

- `bench/qk-lifecycle-search/generated_candidates.json`
- `bench/qk-lifecycle-search/refutations.json` if any route is closed.

## Stop Conditions

Stop and return `ROADMAP_ONLY` if:

- the route depends only on fewer launches;
- the route cannot preserve GQA V-reuse/coalescing;
- the route introduces host sync into decode;
- the route cannot be captured/replayed in the model lifecycle;
- the route requires broad compiler surgery without a toy proof;
- the route cannot produce a full W==D candidate with a plausible `>=5%` ctx1024 or `>=7%` ctx4096 upside.

## Expected Result

The expected useful result is not code. It is a crisp owner decision:

- fund raw-kernel/JIT bridge;
- fund linearizer coupled multi-reduce;
- or rest decode until the project explicitly starts the north-star codegen/machine-search phase.

Keep the durable project state unchanged unless an implementation scope is separately approved:

- prefill solved and opt-in policy-shipped;
- global `PREFILL_V2` default stays off;
- bounded decode fusion closed;
- steady-context decode remains `~67%` llama;
- the only live decode lever is fused+coop in one primitive.
