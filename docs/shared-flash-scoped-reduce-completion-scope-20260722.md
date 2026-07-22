# Shared flash attention: scoped-reduction completion scope

**Repository:** `/home/ubuntu/tinygrad-arkey`
**Target:** AMD gfx1100
**Date:** 2026-07-22
**Status:** authoritative execution scope after the nested-reduction feasibility gate
**Executor:** Codex, continuously through the final promotion decision

## 0. Authority and operating rule

This document is the execution contract from the proven nested-reduction
primitive to production prefill. It refines
`shared-flash-attention-codex-completion-scope-20260722.md`. Where the two
conflict about the immediate implementation route, this document wins. The
older flash, DeepSeek, Claude, multi-output, and fusion documents are evidence,
not separate work queues.

Do not stop at a unit test, a compiler graph, one generated WMMA macro, an
isolated attention microbenchmark, or one model route. Do not pause on a
scheduler blocker. Reduce the blocker to the smallest generic IR primitive,
implement that primitive, and continue. A prohibited hand kernel, route fork,
optimizer carve-out, or silent fallback is not an acceptable primitive route.

Do not estimate time. Report completion only against section 17.

## 1. End goal

Build one compiler-generated, scheduler-native attention implementation that
moves both real prefill routes toward their empirical hardware rooflines:

1. Qwen3-8B with resident fp16-overlay projections.
2. Qwen3-14B with packed/non-fp16 weights and bounded projection machinery.

The routes differ before Q/K/V production. Both produce fp16 Q/K/V activations
at the shared attention boundary. Attention must therefore be implemented once
after Q/K/V and validated twice. Packed weights do not justify a second
attention implementation, optimizer, search space, route policy, or benchmark.

For eligible fp16 attention, the final schedule must:

- compute `(Q @ K.T * scale + mask).softmax(-1) @ V` with stable online math;
- keep score and probability state bounded by the selected tile, never by the
  complete `T*KV` matrix;
- keep both QK and PV visible as ordinary compiler contractions;
- lower both contractions through tinygrad's existing AMD WMMA machinery;
- maintain running `(m, l, acc)` state across KV tiles;
- normalize once and store only the final output;
- improve measured attention GPU time and whole-prefill performance on both
  routes before promotion.

For unsupported activation dtypes or shapes, the same semantic boundary must
either use generic scalar/vector contraction lowering or fail closed to the
ordinary correct attention graph. It must not fork the attention algorithm.

## 2. Feasibility gate passed on 2026-07-22

Command:

```bash
DEV=CPU .venv/bin/python -m pytest test/unit/test_nested_composite_reduce.py -q
```

Result:

```text
3 passed in 0.47s
```

The decisive numeric case constructs:

```text
score[q,kv] = reduce_hd(Q[q,hd] * K[kv,hd])
value_for_score[q,kv] = mapped_view(V[kv])
(m,l,acc) = composite_reduce_kv(score, value_for_score)
output = acc/l
```

The independent NumPy reference uses stable softmax and agrees at
`rtol=1e-5, atol=1e-5` on the CPU gate.

### 2.1 What this proves

- An inner ordinary reduction can feed an outer coupled composite reduction.
- An auxiliary logical-element input can survive scheduling and horizontal
  expansion when its axis correspondence is explicit.
- Primary score lanes and mapped value lanes can remain paired through the
  composite update.
- Multiple state slots can be projected from one graph-local structured result.
- The tested graph stays in one compute schedule rather than materializing an
  intermediate score Tensor between the two reductions.
- The online `(m,l,acc)` recurrence and final `acc/l` are numerically viable in
  the compiler pipeline.

This removes the previous claim that nested reduction ownership is impossible
in the existing pipeline. The project is no longer blocked on feasibility.

### 2.2 What this does not prove

- Production `SCOPED_REDUCE` lowering is not complete.
- The actual model attention semantic node is not yet lowered through this
  primitive.
- The current proof has scalar `acc`; logical `Hd=64/128` state is not complete.
- AMD code generation, score residency, resource use, and barriers are not
  proven by the CPU test.
- QK and PV have not both been attributed to WMMA calls inside one selected
  attention schedule.
- Masks, GQA, long contexts, partial KV tiles, and symbolic shapes are not
  covered by this gate.
- Neither 8B nor 14B model performance is proven.

### 2.3 Temporary proof mechanics

The feasibility patch recognizes horizontally materialized range metadata by
its weak-integer CONST/STACK form. This is an acceptable local proof but not the
final ownership contract. Production work must replace this classification with
explicit source roles and axis maps carried by scheduler IR. User integer
logical inputs must never be confused with range metadata.

## 3. Product invariants

These are non-negotiable throughout implementation:

1. One attention semantic boundary owns Q, K, V, scale, mask, layout, axes,
   accumulation dtype, and output dtype.
2. Every tensor dependency is a normal UOp source, never hidden only in `arg`.
3. One generic stateful-reduction contract owns initialize, update, merge, and
   finalize semantics.
4. Online-softmax equations have one source of truth.
5. State shapes are logical axes. `Hd` is not represented as
   `dtype.vec(Hd)`.
6. Multi-output projections reference one reduction execution and use no
   process-global result cache.
7. Source roles and axis correspondence are explicit, deterministic compiler
   data.
8. The score/probability global allocation is absent on the selected path.
9. QK and PV remain optimizer-visible contractions.
10. Existing tensor-core selection and AMD lowering are reused.
11. Ordinary reductions retain ordinary behavior.
12. Unsupported attention is rejected before destructive rewriting and retains
    the correct fallback graph.
13. Model name and weight representation never select compiler math.
14. One geometry schema, search implementation, result table, and evidence
    harness serve both routes.

## 4. Required IR contract

`SCOPED_REDUCE` is the production ownership boundary unless implementation
evidence proves that a smaller equivalent first-class IR object is cleaner.
Changing the spelling does not weaken the contract.

The node must carry these items explicitly:

- fallback result for fail-closed scheduling;
- producer contraction or producer subgraph;
- zero or more logical-element inputs such as V;
- one source-axis map per non-fallback source;
- owner identity for the reduction scope;
- reduction axes and tile axes;
- state descriptors with scalar dtype and logical shape;
- initialize, update, merge, and finalize combine descriptors;
- result dtype and logical shape per output;
- scale and mask dependencies as visible sources where applicable;
- eligibility facts that can be validated without graph archaeology.

The source-axis map must express the real relationship:

```text
score: [B, Hq, Q, KV]
V:     [B, Hkv, KV, Hd]
state: [B, Hq, Q, Hd]
```

It must support GQA mapping from `Hq` to `Hkv`, map score `KV` to V `KV`, omit
score `Q` from V, and preserve logical `Hd` in the accumulator. Broadcasting a
raw V tensor until shapes happen to align is not the final contract.

`SCOPED_VALUE` or its replacement must be a graph-local projection from the
one scoped reduction result. Projection through scheduler carriers such as
UNROLL must have defined semantics and must not duplicate the reduction.

## 5. Stateful combine contract

The compiler-data interface is:

```text
initialize() -> state
update(state, logical_element) -> state
merge(left_state, right_state) -> state
finalize(state) -> result tuple
```

Online softmax is:

```text
state = (m, l, acc[Hd])
element = (score, value[Hd])
m2 = max(m, score)
corr = exp(m - m2)
p = exp(score - m2)
l2 = l*corr + p
acc2 = acc*corr + p*value
result = acc/l
```

The stable merge of two partial states is mandatory. Without merge, KV blocks
remain serial and geometry cannot trade parallelism against occupancy.

Ranged, no-range, full-UNROLL, partial-UNROLL, UPCAST, group reduction, and
block merge may have different traversal code. They must evaluate the same
combine data and must not copy these equations.

Add a second coupled, non-attention combine with at least one auxiliary input.
It must pass the same lowering forms. This proves that the mechanism is a
compiler primitive rather than an online-softmax exception.

Unknown operations, source counts, source roles, state shapes, merge behavior,
or dtypes must raise an actionable compiler error before code generation.

## 6. Ordered implementation program

Each milestone is a gate. Continue immediately after a pass. If a gate fails,
implement the smallest generic primitive that makes the contract expressible,
then rerun the same gate.

### M0. Bank the feasibility checkpoint

Work:

- Keep the focused numeric and topology tests.
- Record the exact proof and limitations in this document.
- Keep production selection fail-closed to ordinary SDPA.
- Label weak-integer carrier detection as temporary proof code.

Gate:

- `test/unit/test_nested_composite_reduce.py` reports three passes.
- No production route claims flash selection or performance.

### M1. Replace heuristic ownership with explicit scoped ownership

Work:

- Give every `SCOPED_REDUCE` source an immutable role.
- Make source-axis maps authoritative through rangeify, symbolic rewriting,
  optimization, expander, and devectorizer.
- Preserve non-range logical inputs in every REDUCE reconstruction.
- Teach split, flatten, simplify, group-reduce, UNROLL, CONTRACT, END, and
  bufferization passes to distinguish owned ranges from logical inputs.
- Delete weakint/STACK recognition once explicit ownership reaches late
  lowering.
- Remove any `hasattr` or module-identity workaround that is no longer needed.
- Make malformed ownership fail closed with the source index and expected map.

Required tests:

- Scalar and vector auxiliary inputs.
- Floating-point and integer auxiliary values.
- Zero, one, and two auxiliary inputs.
- Permuted source order where the explicit role permits it.
- Axis insertion, deletion, broadcast, transpose, and GQA mapping.
- Full and partial UNROLL.
- A nested ordinary reduction as producer.
- A producer with no reduction as a control.
- Deliberately malformed and ambiguous maps rejected.
- Ordinary REDUCE graph/code differential unchanged.

Gate:

- The feasibility tests pass without dtype- or opcode-based range guessing.
- Every late REDUCE source is classified from first-class metadata.

### M2. Finish generic stateful reduction and true multi-output

Work:

- Replace the string registry with an immutable combine descriptor or prove a
  typed registry has equivalent validation and one semantic definition.
- Implement initialize/update/merge/finalize once.
- Add logical state shapes, including `acc[Hd]`.
- Lower ranged, no-range, partial/full UNROLL, UPCAST, and group-reduce forms.
- Resolve all result slots graph-locally after the structured result exists.
- Ensure DCE can remove unused final slots without removing shared state needed
  by retained slots.
- Remove global caches, independent slot reductions, and fallback-to-unrelated
  result behavior.

Required tests:

- Independent sum/max with both slots realized.
- A coupled non-attention combine with auxiliary input.
- Online `(m,l)` against NumPy fp64.
- Online `(m,l,acc)` and finalized output against independent SDPA.
- `Hd=1,64,128` as logical state.
- One update stream versus two and multiple partial-state merges.
- Reversed slot use, duplicate consumers, one unused slot, and all slots used.
- Repeated graphs in one process with no cross-graph leakage.
- Invalid slot, dtype, logical shape, operation, and finalize rejection.

Gate:

- One combine definition drives all execution forms.
- Two outputs are numerically realized from one actual stateful reduction.
- No logical state dimension is encoded as a giant hardware vector dtype.

### M3. Lower the shared attention semantic node into scoped IR

Work:

- Use the existing shared semantic attention boundary after Q/K/V.
- Keep Q, K, V, scale, and tensor mask as graph sources.
- Validate MHA and GQA layout facts conservatively.
- Lower eligible attention to one scoped reduction with:
  - QK as the inner `Hd` contraction;
  - score scale and mask before the state update;
  - mapped V as a logical-element input;
  - KV tile ownership in the outer state reduction;
  - logical `Hd` accumulator state;
  - final normalization after the KV scope.
- Preserve the original attention graph as fallback until eligibility is fully
  proven.
- Remove broad reverse matching of arbitrary ADD/MAX/EXP reductions.

Capture matrix:

- MHA and GQA with `G=1,4,5`.
- `Hd=64,128` plus `Hd=1` compiler control.
- No mask, causal mask, and the actual additive model mask.
- Scalar and tensor scale forms used by the tree.
- Contiguous and actual model-produced layouts.
- Static `T/KV`; symbolic forms only when ownership can be proven.
- Near-miss softmax and reduction graphs that must retain fallback.

Gate:

- Both real model routes create the same semantic node and same scoped lowering
  after Q/K/V.
- Rejection includes a stable reason code and leaves fallback unchanged.

### M4. Produce one bounded, correct generic compute schedule

Work:

- Schedule query tiles and KV blocks in one compute program.
- Keep QK, block state update, PV, stable merge, and normalization within the
  scoped topology.
- Use registers or bounded LDS for score tiles and partial state.
- Handle tails without allocating a full score matrix.
- Keep generic non-WMMA contraction lowering correct first.
- Inventory BUFFER, STAGE, DEFINE_LOCAL, and global STORE nodes by logical role.

Correctness matrix:

| Dimension | Mandatory values |
|---|---|
| B | 1, 2 |
| G | 1, 4, 5 |
| Hd | 1, 64, 128 |
| T | 1 fallback, 32, 127, 128, 129, 512, 2048 |
| KV | equal to T and greater than T where continuation allows |
| mask | none, causal, actual additive mask |
| activation | fp16; fp32 generic; bf16 if gfx1100 path supports it |

Use deterministic random inputs plus repeated maxima, large positive/negative
logits, masked rows, partial KV blocks, and near-zero outputs.

Numeric reporting must include maximum absolute error, maximum relative error
with a near-zero floor, mean absolute error, NaN/Inf parity, and a deterministic
sample hash. The fp16 target is `max_rel_err <= 1e-2` outside the declared
near-zero region, with an explicit absolute bound.

Residency gate:

- One selected attention compute schedule.
- No global allocation proportional to `B*Hq*T*KV` for scores, exponentials,
  probabilities, or normalized weights.
- Bounded tile storage is reported with an exact byte formula.
- QK and PV are both present in the same scoped compute topology.

### M5. Attach both contractions to the existing WMMA path

Work:

- Extend generic optimizer handling of multiple/nested contractions where
  required.
- Keep QK and PV as ordinary REDUCE/contraction structures recognizable by
  `postrange.py` and `tc.py`.
- Reuse the existing AMD WMMA description, fragment layout, renderer, and
  builtin.
- Preserve `NOOPT=0` and normal optimizer participation.
- Keep generic ALU contraction lowering for ineligible dtypes.
- Report register, LDS, workgroup, and occupancy estimates per geometry.

Hard prohibitions:

- no flash-specific WMMA emitter;
- no copied swizzle or AMD builtin;
- no hand kernel or `custom_kernel`;
- no optimizer skip for composite/scoped reductions;
- no claim based on `#define __WMMA` alone.

Gate:

- Generated source contains distinct executed WMMA call sites attributed to QK
  and PV inside the selected scoped schedule.
- QK and PV both remain correct under `TC_OPT=2, NOOPT=0`.
- The score/probability residency gate still passes.
- Unsupported activation dtypes use generic lowering or the explicit fallback.

### M6. Build one shared geometry and search system

Work:

- Define one candidate schema keyed by device, activation dtype, `B,Hq,Hkv,G`,
  `T,KV,Hd`, mask class, query tile, KV block, waves, staging, and resource
  limits.
- Reuse BubbleBeam/FutureSight ranking infrastructure.
- Search both real route shape families through the same program.
- Reject candidates exceeding register, LDS, launch, correctness, or residency
  limits before timing.
- Persist reproducible candidates in one table.
- Keep static winning entries only as data, not route-specific code.

Gate:

- 8B and 14B records use one schema and implementation.
- Replaying a stored candidate reproduces its topology and resource report.
- No model name appears in compiler lowering or attention math.

### M7. Integrate both model routes through one adapter

Work:

- Replace dead or unused flash flags with one functional shared selection.
- Pass real Q/K/V layouts, mask, scale, and route facts to the semantic node.
- Keep packed dequantization entirely in existing projection code.
- Prove the 14B route does not materialize dense fp16 weights.
- Add route census fields for semantic eligibility, selected lowering, fallback
  reason, geometry id, QK mode, PV mode, and score-residency result.
- Default to ordinary SDPA unless all promotion facts are true.

Gate:

- Both routes execute the same attention lowering code.
- Output parity passes on deterministic prompts.
- Unsupported cases show ordinary fallback with a reason, not a silent partial
  flash path.

### M8. Measure attention and whole-prefill movement toward roofline

Existing audited reference points on gfx1100:

| Route | T=KV=512 ordinary attention | T=KV=2048 ordinary attention |
|---|---:|---:|
| 8B fp16-overlay | 8.0 ms | 27.3 ms |
| 14B packed | 8.4 ms | 33.9 ms |

Existing empirical ceilings are 7.1 TFLOP/s fp16 WMMA and 5.2 GB/s for the
recorded bandwidth probe. Re-audit these in the final measurement regime; do
not assume old numbers are current or directly comparable.

For each attention shape, report:

```text
QK_flops = 2*B*Hq*T*KV*Hd
PV_flops = 2*B*Hq*T*KV*Hd
total_flops = QK_flops + PV_flops
measured_compute_rate = total_flops / gpu_time
mandatory_HBM_bytes = Q + K + V + output + model-owned KV effects
forbidden_saved_bytes = full score/probability traffic removed
arithmetic_intensity = total_flops / measured_HBM_bytes
roofline_bound = min(C_peak, B_peak*arithmetic_intensity)
roofline_fraction = measured_compute_rate / roofline_bound
```

Measurement rules:

- baseline and candidate in the same process and runtime regime;
- warmed GPU timing, not compilation or first-use wall time;
- at least 200 dispatches per attention microbenchmark;
- raw samples, median, p10, p90, and dispersion retained;
- source/ISA and allocation evidence tied to the timed program;
- no unrelated projection or route change in an attention comparison;
- whole-prefill uses the existing synchronized authority harness;
- report pp512, pp2048, and supported pp4096 for both routes;
- report tokens/s, GPU time, wall time, peak memory, route census, output hash,
  and decode non-regression;
- separate attention speedup from whole-model speedup.

Performance gate:

- Absolute attention GPU time improves beyond measured noise for both routes.
- Roofline fraction increases for both real shape families.
- Whole-prefill improves beyond measured noise for both routes.
- Peak memory does not regress outside a documented bounded-tile allowance.
- Decode correctness and promoted decode performance do not regress.

There is no fixed promised multiplier. The result is accepted from measured
movement toward the applicable roofline, not from a synthetic projection.

### M9. Cleanup, regression, and promotion

Work:

- Delete proof-only weakint carrier inference.
- Delete obsolete matcher, dead flags, unused Tensors, duplicate formulas,
  global caches, debug prints, and route-specific flash experiments.
- Keep only reusable compiler, search, route, test, and evidence assets.
- Run focused compiler tests, relevant AMD WMMA tests, attention correctness and
  residency matrices, both model authority runs, and decode regression.
- Make the route default only after every production gate passes.
- Commit and push a clean, reviewable series.

Gate:

- No mandatory case is hidden behind xfail or skip.
- No full score/probability buffer exists in selected production schedules.
- Promotion policy requires correctness, residency, dual WMMA when eligible,
  resource limits, and measured route benefit.
- Failure of any required fact selects ordinary SDPA.

## 7. File ownership map

Reuse these owners. Do not create parallel implementations.

| Concern | Primary files | Required action |
|---|---|---|
| Semantic attention | `tinygrad/llm/attention.py`, shared model attention call | Preserve one Q/K/V boundary and fallback |
| Scoped ownership IR | `tinygrad/uop/__init__.py`, `tinygrad/uop/ops.py`, `tinygrad/uop/spec.py` | Explicit roles, maps, result types |
| Range construction | `tinygrad/schedule/indexing.py` | Preserve all sources and maps |
| Scoped lowering | `tinygrad/schedule/rangeify.py` | Emit bounded nested schedule before score bufferization |
| Range simplification | `tinygrad/codegen/simplify.py` | Preserve logical inputs generically |
| Expansion | `tinygrad/codegen/late/expander.py` | Preserve source roles through UNROLL/CONTRACT |
| Stateful lowering | `tinygrad/codegen/late/devectorizer.py` | Allocate/update/merge logical state |
| Combine semantics | `tinygrad/codegen/late/composite_combines.py` | One typed combine evaluator |
| Tensor cores | `tinygrad/codegen/opt/postrange.py`, `tinygrad/codegen/opt/tc.py` | Reuse for QK and PV |
| AMD renderer | existing AMD renderer/WMMA descriptors | Reuse unchanged unless a generic defect is proven |
| Model adapter | `tinygrad/llm/model.py` | Shared invocation and fail-closed route census |
| Search | `extra/qk/bubblebeam_futuresight.py` and shared candidate assets | One schema and ranking path |
| Microbenchmark | existing attention benchmark assets | Baseline/candidate in one harness |
| Model authority | `extra/qk/prefill_whole_synced.py`, `extra/qk/prefill_harness.py` | Final 8B/14B evidence |

If repository movement has renamed a listed file, update the owner in place;
do not fork a replacement solely because the path changed.

## 8. Required test layers

### Compiler primitive tests

- source-role preservation through every rewrite stage;
- axis-map correctness and rejection;
- nested producer plus auxiliary inputs;
- scalar and logical vector state;
- update and merge equivalence;
- multi-output ownership and DCE;
- UNROLL/UPCAST/group-reduce forms;
- normal REDUCE differential tests.

### Semantic attention tests

- exact source ownership;
- eligibility and stable rejection reasons;
- MHA/GQA maps;
- scale and mask semantics;
- fallback graph preservation;
- no unsafe near-miss matching.

### Schedule tests

- one selected compute schedule;
- QK and PV contraction provenance;
- no full score/probability allocation;
- bounded tile byte formula;
- partial KV tile handling;
- no forbidden global STORE.

### Numeric tests

- CPU generic matrix;
- AMD fp16 matrix;
- fp32 generic or fallback matrix;
- masks, GQA, tails, long context, adversarial logits;
- deterministic comparison to an independent reference.

### Code-generation tests

- QK WMMA call attribution;
- PV WMMA call attribution;
- `NOOPT=0`, `TC_OPT=2`;
- generic non-WMMA lowering;
- resource-limit rejection;
- no renderer-visible TUPLE/SCOPED/REDUCE_SLOT residue.

### Model tests

- 8B fp16-overlay selected-path census and parity;
- 14B packed selected-path census and parity;
- no dense-fp16 14B weight materialization;
- pp512/pp2048/supported pp4096 performance;
- peak memory;
- decode parity and performance.

## 9. Evidence required at every milestone

Each milestone report must include:

- exact commit;
- exact command and environment;
- selected route and fallback reason;
- graph or generated-code evidence appropriate to the gate;
- raw numeric or timing data, not only a conclusion;
- what is proven;
- what remains unproven;
- any temporary mechanism and its deletion milestone.

Do not use test count as a substitute for naming the semantic cases covered.

## 10. Blocker protocol

When a blocker appears:

1. Write the smallest failing test at the first IR boundary where information is
   lost.
2. State the invariant in source-role and axis-ownership terms.
3. Implement the smallest generic primitive that preserves that invariant.
4. Add a non-attention control proving generality.
5. Rerun the failing gate and ordinary REDUCE differential.
6. Continue to the next milestone without requesting permission.

Do not respond to a blocker by:

- adding attention-name branches to generic lowering;
- recognizing compiler metadata from a user data dtype;
- hiding a dependency in UOp `arg`;
- duplicating a reduction per output slot;
- forcing `NOOPT=1`;
- skipping tensor-core optimization;
- adding a hand kernel;
- materializing the score matrix;
- creating 8B and 14B variants;
- declaring the architecture impossible without a failing primitive test.

## 11. Explicit non-goals

- Replacing packed projection kernels as part of this project.
- A second flash implementation for the 14B route.
- Optimizing decode through this prefill path unless a shared generic change
  naturally applies and passes decode gates.
- Matching every arbitrary user-written softmax graph.
- Promoting a path that is correct but slower.
- Claiming roofline movement from macro counts or theoretical byte deletion.

## 12. Promotion and rollback policy

Selection is fail closed. The candidate is promotable for a workload only when
all of these facts are true:

```text
semantic_eligible
and scoped_schedule_selected
and numeric_gate_passed
and score_resident
and resource_limits_passed
and (dual_wmma or activation_dtype_not_wmma_eligible)
and measured_attention_benefit
and measured_model_benefit
```

Any false or unknown fact selects ordinary SDPA and records the reason. The two
model routes may have different winning geometry records, but they may not have
different implementations or promotion definitions.

## 13. Expected final architecture

```text
8B fp16-overlay projections ----\
                                 +--> shared semantic attention
14B packed projections ---------/            |
                                              v
                                  eligibility + fallback
                                              |
                                              v
                                    SCOPED_REDUCE ownership
                                  /      |       |       \
                                QK    online   mapped V   merge
                                 \      state     /       /
                                  +------ PV contraction
                                              |
                                      finalize acc/l
                                              |
                                      shared route policy
```

There is one semantic node, one scoped scheduler lowering, one stateful combine
system, one tensor-core path, one search system, and one promotion policy.

## 14. Final evidence package

The final report must contain:

- commit range and clean pushed HEAD;
- semantic before/after graphs for both routes;
- scoped source-role and axis-map dump;
- state IR and merge proof;
- final schedule topology and kernel count;
- buffer/allocation inventory with byte formulas;
- generated QK and PV WMMA invocation lines;
- AMD resource report per promoted geometry;
- complete numeric matrix and error statistics;
- microbenchmark raw samples and distributions;
- roofline calculations with re-audited ceilings;
- 8B and 14B whole-prefill tables;
- route census proving the selected implementation;
- 14B packed-memory proof;
- decode regression;
- focused and relevant regression results;
- explicit fallback cases and reasons;
- list of deleted prototype mechanisms.

## 15. Honest stop condition

A final `NO-GO` is allowed only after all of these are true:

- the production scoped schedule is correct;
- score residency is directly proven;
- both eligible contractions use the centralized WMMA path;
- a reasonable shared geometry search is complete;
- measurements are warmed, comparable, and reproducible;
- both model routes were measured;
- the candidate still fails to improve beyond noise or violates resources.

In that case, keep useful generic compiler primitives, keep production on
ordinary SDPA, remove failed route wiring, and publish the evidence. A compiler
implementation blocker before those facts is not the final stop condition.

## 16. Immediate next implementation slice

Start at M1, not at attention matching or model benchmarking:

1. Extend the scoped metadata so each source has an explicit role and axis map.
2. Route the passing nested-reduction test through `SCOPED_REDUCE` and
   `SCOPED_VALUE`, not raw composite source heuristics.
3. Add an integer auxiliary-input case that would fail the temporary weakint
   classifier.
4. Make rangeify, simplify, expander, and devectorizer preserve the explicit
   ownership.
5. Delete the weakint/STACK classifier.
6. Rerun the three-test feasibility gate plus ordinary REDUCE differential.
7. Continue directly into logical `Hd` state and merge.

This slice converts the proven mechanism into the durable compiler primitive
needed by production attention.

## 17. Definition of 100 percent complete

The project is complete only when every statement is true:

- The feasibility proof is preserved as a regression test.
- Explicit scoped source roles and axis maps replace heuristic ownership.
- A non-attention coupled combine proves the mechanism is generic.
- Initialize/update/merge/finalize semantics have one source of truth.
- Logical `Hd=64/128` state is supported without giant vector dtypes.
- Multi-output projection is graph-local and executes one shared reduction.
- All ranged and expansion forms required by optimized code are supported.
- Both model routes emit the same shared semantic attention representation.
- Eligible attention lowers to one bounded scoped compute schedule.
- No full score or probability buffer reaches global memory.
- QK and PV remain distinct optimizer-visible contractions.
- Eligible fp16 QK and PV use the existing AMD WMMA path under normal
  optimization.
- Non-WMMA activation dtypes use generic lowering or explicit fallback without
  duplicating attention math.
- Masks, GQA, tails, long contexts, and real layouts pass the numeric matrix.
- One geometry/search system serves 8B and 14B.
- 8B fp16-overlay uses the shared selected path and improves attention and
  whole-prefill beyond noise.
- 14B packed uses the same selected path, retains bounded packed projections,
  and improves attention and whole-prefill beyond noise.
- Roofline fraction is calculated from re-audited empirical ceilings and
  increases for both route shape families.
- Peak memory and decode gates pass.
- Production selection fails closed on every unsupported or unproven case.
- Temporary heuristics, dead flags, unsafe matchers, duplicate formulas, debug
  output, caches, shape hacks, and route-specific flash assets are removed.
- Mandatory focused and relevant regression suites pass without hidden xfails.
- The final evidence package is complete.
- The repository is clean, committed, and pushed.

Until all statements are true, report the current milestone and failed gate,
not `COMPLETE`.
