# Codex completion scope: shared scheduler-native flash attention to both prefill rooflines

**Repository:** `/home/ubuntu/tinygrad-arkey`  
**Target:** AMD gfx1100  
**Starting HEAD:** `e8e0746cd`  
**Executor:** Codex, end to end  
**Status:** authoritative implementation and completion contract

## 0. Authority

This document supersedes every earlier flash-prefill, composite-reduce,
multi-output, DeepSeek handoff, and flash-fusion scope when they conflict. The
older documents remain evidence, not independent definitions of completion.

This is not a request for another prototype, structural test, isolated softmax
speedup, or progress report. Codex owns the implementation through the final
model gates. Intermediate phases are checkpoints, not stopping points.

Do not estimate time. Do not declare completion from unit-test counts, a macro
definition, a successful compile, or a synthetic shape. Completion is exactly
the checklist in section 18.

## 1. Product goal

Build one compiler-generated, scheduler-native prefill-attention path that moves
both supported model routes closer to the applicable empirical hardware
roofline:

1. Qwen3-8B using the resident fp16-overlay projection route.
2. Qwen3-14B using the bounded packed/non-fp16-weight projection route.

Both routes produce fp16 Q/K/V activations after projection. Their weight
representations differ before that boundary; attention must not. The project is
therefore **build once after Q/K/V, validate and gate twice**.

For supported prefill attention, the implementation must:

- Capture the semantics of `(Q @ K.T * scale + mask).softmax(-1) @ V`.
- Compute online-softmax state `(m, l, acc)` over KV.
- Never materialize full `T x KV` score or probability tensors in global memory.
- Keep QK and PV as compiler-visible contractions.
- Lower both eligible contractions through tinygrad's existing tensor-core path.
- Retain a correct generic non-WMMA lowering for unsupported activation dtypes.
- Preserve enough occupancy that deleted traffic and WMMA produce a measured
  reduction in attention GPU time.
- Improve real end-to-end prefill on both routes before being enabled.

The 14B path being backed by packed weights does not justify a second attention
kernel. Packed dequantization remains in the existing projection machinery.

## 2. Exact meaning of score resident

For an attention problem with logical shape `B,Hq,T,KV,Hd`, the supported path
may allocate:

- Q, K, V, output, and existing model-owned KV storage.
- Register state for `m`, `l`, and an output accumulator tile.
- Bounded register or LDS tiles such as `Bq x Bkv` scores.
- Small scheduling metadata or bounded partial state when proven necessary.

It may not allocate or stage a global buffer whose logical size is proportional
to `B*Hq*T*KV` for scores, exponentials, probabilities, or normalized weights.
It may not hide such an allocation behind `CONTIGUOUS`, `STAGE`, `AFTER`, a
temporary Tensor, or a route adapter.

The primary implementation target is one fused attention compute kernel in
which generated-code evidence identifies both QK and PV contraction call sites.
Any auxiliary kernel must be bounded independently of `T*KV` and justified in
the final report. Separate materialized QK, softmax, and PV kernels do not pass.

## 3. Starting-state audit at `e8e0746cd`

### 3.1 Assets worth preserving

- `AccumulatorSlot` and `CompositeReduce` establish a useful stateful-reduction
  vocabulary.
- `composite_combines.py` establishes a centralized location for combine
  semantics.
- The online-softmax recurrence is useful as a numeric oracle.
- Rangeify has an identified pre-bufferization interception point.
- Extra REDUCE sources can survive `convert_reduce_to_reduce_with_ranges`.
- The repository already owns AMD WMMA lowering, tensor-core optimization,
  prefill route policy, model authority harnesses, and geometry search assets.
- The 8B and 14B routes already meet at fp16 Q/K/V activations.

### 3.2 Defects that must be removed before extension

- `_flash_attn_match` globally rewrites broad floating-point ADD reductions
  without proving attention or softmax semantics.
- `_find_score_from_softmax` recovers an `EXP2` input rather than canonical
  logits and can preserve or duplicate the original softmax reductions.
- Matcher exceptions are swallowed, producing silent fallback or mis-match.
- `flash_attention()` computes an unused composite denominator and returns the
  ordinary materialized `scores.softmax() @ V` result.
- `model.py` computes unused `_flash_l`; the flag is not functional integration.
- `REDUCE_SLOT` is converted into independent plain reductions at Tensor
  construction, which is invalid for coupled state.
- `_resolve_reduce_slot` has a process-global cache and an unrelated-entry
  fallback; slot ownership is not graph-local or deterministic.
- Existing M4 tests prove object identity but do not realize two numeric slots
  from one scheduled reduction.
- `v_uop` is duplicated as metadata inside `CompositeReduce`; UOps stored in
  `arg` are invisible to ordinary source traversal and substitution.
- Logical `Hd` is encoded as `dtype.vec(Hd)`, followed by composite-specific
  REDUCE shape exceptions. A logical output dimension is not a hardware vector.
- Online-softmax equations exist in multiple ranged/no-range/step functions.
- Unknown or unsupported combines can silently choose fallback semantics.
- `indexing.py` contains unconditional scheduler debug prints.
- No committed test proves the automatic PV rewrite, no-spill topology, two
  contraction call sites, or model-level use.

Codex must not patch around these defects. Quarantine or replace them using the
architecture below.

## 4. Non-negotiable architecture

### 4.1 A semantic attention boundary

Introduce one immutable, route-independent semantic representation for
attention before softmax is decomposed into `MAX`, `EXP2`, and `SUM` graphs.
The representation may be a dedicated UOp or another first-class scheduler IR
object, but it must satisfy all of these properties:

- Q, K, V, and an optional tensor mask are actual `src` dependencies.
- Scale, causal mode, axes, layouts, and accumulation/output dtypes are explicit.
- No UOp dependency is stored only in `arg`.
- Graph traversal, substitution, device discovery, rangeify, DCE, and scheduling
  see every dependency normally.
- The representation is independent of model name and weight format.
- Unsupported semantics fail eligibility and retain the ordinary correct graph.

Use the highest-level stable capture point available. Prefer explicit semantic
capture from tinygrad's shared attention construction over reverse engineering
already-lowered exponentials. If ordinary canonical graphs must also be
recognized, the matcher must prove the entire max/subtract/exp/sum/normalize/PV
relationship, axes, scale, mask, and shared logits identity before rewriting.

### 4.2 Generic composite reduction semantics

Define one compiler-data contract with four concepts:

```text
initialize() -> state
update(state, logical_element) -> state
merge(left_state, right_state) -> state
finalize(state) -> output(s)
```

For online softmax:

```text
state = (m, l, acc[Hd])
logical_element = (score, value[Hd])
m_new = max(m, score)
corr = exp(m - m_new)
p = exp(score - m_new)
l_new = l*corr + p
acc_new = acc*corr + p*value
output = acc/l
```

The merge operation must implement the equivalent stable merge of two partial
`(m,l,acc)` states. This is required to tile and parallelize KV rather than run a
single serial loop.

Online-softmax math has one source of truth. Ranged reduction, no-range
reduction, partial UNROLL, full UNROLL, UPCAST, block-local reduction, and final
merge all instantiate the same immutable combine representation. They may have
different generic traversal mechanics but may not copy the equations.

The REDUCE machinery must not contain the string `online_softmax` or branch on a
combine name. Unsupported combine features fail closed with an actionable
compiler error.

### 4.3 Logical state shapes, not dtype packing

State slots carry logical shapes and scalar dtypes independently:

- `m`: scalar per query row, normally fp32 accumulation.
- `l`: scalar per query row, normally fp32 accumulation.
- `acc`: logical `Hd` axis per query row, normally fp32 accumulation.

Hardware vectorization remains an optimizer/codegen decision. Remove the
`dtype.vec(Hd)` representation and the composite-specific shape rules that turn
a reduced axis into a vector lane count.

### 4.4 True multi-output ownership

If `REDUCE_SLOT` remains the selected representation, it must be a graph-local
projection from one composite reduction result. It must never construct an
independent reduction and must never consult a process-global result cache.

The lowering must create the reduction once, keep every state dependency alive,
and let multiple consumers reference the correct final slots. Slot type, shape,
index bounds, ownership, and DCE behavior must be verified by the IR itself.

If a first-class tuple/structured result is cleaner in the current tinygrad IR,
Codex may replace `REDUCE_SLOT`; the behavioral gates remain unchanged.

### 4.5 One fused schedule

Rangeify must lower semantic attention to a tiled schedule containing:

1. Query tile selection.
2. KV block iteration.
3. QK contraction across `Hd` for the current score tile.
4. Scale and mask application.
5. Block-local online-softmax update.
6. PV accumulation across the current KV block.
7. Stable merge into running `(m,l,acc)` state.
8. Final normalization and output store.

QK and PV remain explicit contraction structures that the existing optimizer can
recognize. Bufferization must see only bounded tiles and final output, never the
full score/probability tensors.

### 4.6 One fast-compute path

Reuse `postrange.py`, `tc.py`, target tensor-core descriptions, and ordinary
tinygrad optimization. Extend their generic support for multiple contractions
inside one schedule if necessary.

Do not create a flash-specific WMMA emitter, hand-written fragment operation,
AMD builtin, copied swizzle, or special optimizer bypass. Macro definitions do
not count as proof; generated `__WMMA(...)` invocation lines tied to QK and PV do.

### 4.7 One search and one routing path

Use one attention geometry schema keyed by real constraints such as device,
activation dtype, `B,Hq,Hkv,G,T,KV,Hd`, mask class, query tile, KV block, waves,
staging, and occupancy limits.

Reuse BubbleBeam/FutureSight ranking infrastructure. Static entries are allowed
for reproducibility, but both model routes must be searched by the same program
and stored in the same table. No `8b_flash` and `14b_flash` implementations or
copied search scripts are allowed.

The model route adapter may provide actual shape/layout facts and apply the
shared eligibility result. It may not contain attention math or kernel code.

## 5. Explicit reuse map

Codex must inspect and reuse these assets rather than recreating them:

| Concern | Existing owner | Required use |
|---|---|---|
| Model attention semantics | `tinygrad/llm/model.py` and shared Tensor attention construction | Canonical Q/K/V boundary and fp32 oracle inputs |
| Composite state lowering | `tinygrad/codegen/late/composite_combines.py` | Refactor into one generic combine evaluator |
| REDUCE lowering | `tinygrad/codegen/late/devectorizer.py` | Generic state allocation/update/merge |
| Range creation | `tinygrad/schedule/indexing.py` | Preserve all semantic sources and logical axes |
| Fusion insertion | `tinygrad/schedule/rangeify.py` | Semantic capture/lowering before score buffers |
| Tensor-core selection | `tinygrad/codegen/opt/postrange.py`, `tinygrad/codegen/opt/tc.py` | Both eligible contractions |
| AMD target lowering | existing AMD renderer/WMMA descriptions | No second emitter |
| Geometry ranking | `extra/qk/bubblebeam_futuresight.py` | Shared candidate scoring/ranking |
| Attention microbenchmark | existing SDPA/flash prefill harness assets | Extend, do not fork |
| Whole-prefill authority | `extra/qk/prefill_whole_synced.py`, `extra/qk/prefill_harness.py` | Final 8B/14B measurement |
| Route policy | existing prefill policy/route binding | Fail-closed promotion after both gates |

`extra/qk/flash_kernels.py` may be used as a mathematical oracle only. Its hand
kernel, executor, LDS/barrier code, and route-specific machinery may not ship in
this path.

## 6. Hard prohibitions

- No separate 8B and 14B attention implementations.
- No hand kernel or `custom_kernel` implementation.
- No new AMD builtin or second WMMA emitter.
- No Q/K/V UOps hidden only inside UOp arguments.
- No `dtype.vec(Hd)` logical output representation.
- No process-global lowering-result cache.
- No broad `except Exception` in semantic matching or lowering.
- No silent unknown-combine fallback.
- No model-name compiler condition.
- No `NOOPT=1` completion configuration.
- No composite-reduction optimizer carve-out.
- No score/probability global buffer disguised as a temporary.
- No copied online-softmax formula across execution forms.
- No separate geometry table or benchmark implementation per model.
- No dead feature flag, unused Tensor, debug print, or temporary oracle at final
  completion.
- No benchmark claim based only on wall time, first-run compilation, macro
  counts, a single short shape, or an unverified route.

## 7. Phase A: quarantine and establish a trustworthy baseline

### Work

1. Disable or replace the unsafe global `_flash_attn_match` before further use.
2. Remove unconditional scheduler debug output.
3. Remove dead `_flash_l` model wiring.
4. Make `flash_attention()` either the canonical semantic constructor or remove
   it; it may not advertise flash while returning ordinary SDPA.
5. Remove Tensor-level independent reduction substitution for `REDUCE_SLOT`.
6. Remove the unrelated-entry global cache fallback.
7. Remove `dtype.vec(Hd)` and composite REDUCE shape exceptions.
8. Preserve the shipped materialized attention route as the explicit fallback.
9. Add regression tests showing ordinary ADD/MAX reductions and ordinary
   softmax are byte/numerically unchanged when attention is ineligible.

### Gate A

- No non-attention reduction can match the attention rewrite.
- Unsupported attention retains the baseline graph and output.
- Existing model routes still run through the shipped baseline.
- The repository contains no unconditional debug output from this work.

### Artifact

Create `docs/shared-flash-codex-A-quarantine-20260722.md` with removed behavior,
regression commands, and the exact retained useful infrastructure.

## 8. Phase B: generic stateful reduction foundation

### Work

1. Define the immutable initialize/update/merge/finalize combine IR.
2. Express online softmax once using that IR.
3. Add a second unrelated coupled combine to prove generality.
4. Implement generic scalar ranged lowering.
5. Implement logical vector/tensor state without giant vector dtypes.
6. Implement partial UNROLL and full UNROLL using the same evaluator.
7. Implement block-state merge and prove associativity within floating-point
   tolerance.
8. Preserve all state update ordering and DCE reachability.
9. Fail closed on unsupported dtype, state shape, packing, or combine operation.
10. Keep ordinary REDUCE generated output unchanged unless a separately tested
    generic correction is required.

### Required tests

- Independent sum/max state with both outputs realized.
- Coupled unrelated state with both outputs realized.
- Online-softmax `(m,l)` against an independent NumPy fp64 calculation.
- Online-softmax `(m,l,acc)` and finalized `acc/l` against fp32 SDPA.
- One-pass update versus two and multiple partial-state merges.
- Scalar, logical `Hd=1`, `Hd=64`, and `Hd=128` state.
- No optimization, ordinary optimization, partial UNROLL, and full UNROLL.
- Multiple consumers, reversed slot order, unused slots, and DCE.
- Invalid slot, shape, dtype, and combine operation rejection.
- No cross-graph state leakage across repeated schedules in one process.

### Gate B

One combine representation drives every execution form. No lowering code names
or reimplements online softmax. Two numeric outputs are realized from one actual
reduction execution.

### Artifact

Create `docs/shared-flash-codex-B-stateful-reduce-20260722.md` with IR diagrams,
test matrix, generated graphs, and normal-REDUCE differential evidence.

## 9. Phase C: canonical attention capture

### Work

1. Identify the shared Tensor/UOp point before softmax decomposition.
2. Add the semantic attention representation with Q/K/V as sources.
3. Teach the shared Tensor attention API and both model routes to create the same
   semantic representation without model-specific compiler behavior.
4. If automatic ordinary-graph recognition remains required, implement an exact
   conservative matcher for the complete semantic graph.
5. Represent scale, causal/additive mask, axes, GQA broadcast, layouts, and
   dtypes explicitly.
6. Add an eligibility predicate with a reasoned fail-closed result.
7. Prove graph substitution and device movement preserve every source.

### Capture test matrix

- MHA and GQA (`G=1,4,5`).
- `Hd=64` and actual `Hd=128` routes.
- No mask, causal mask, and actual additive model mask.
- Tensor scale and scalar scale forms used in-tree.
- Contiguous and actual model-produced layouts.
- Static and symbolic `T/KV` where supported.
- Deliberate near-miss graphs that must not match.
- Operand reordering that is semantically equivalent where canonicalization
  permits it.

### Gate C

Both real routes produce one equivalent semantic attention node after Q/K/V.
Near misses fall back. The matcher never reverse-guesses logits from an `EXP2`.

### Artifact

Create `docs/shared-flash-codex-C-capture-20260722.md` with before/after graphs,
source ownership, match reasons, rejection reasons, and both route traces.

## 10. Phase D: correct score-resident scalar schedule

### Work

1. Lower the semantic node to the tiled QK/update/PV/merge/finalize schedule.
2. Begin with optimizer-visible scalar contractions; correctness and residency
   precede WMMA.
3. Support actual GQA mapping without duplicating K/V storage or attention math.
4. Apply scale and mask before state update.
5. Keep score tiles bounded in registers/LDS.
6. Ensure no full score/probability `BUFFER`, `STAGE`, or realized Tensor exists.
7. Preserve a correct fallback for ineligible shapes.
8. Produce deterministic graph-level buffer and allocation evidence.

### Correctness matrix

At minimum cover these representative cases, subject to actual model limits:

| Dimension | Required values |
|---|---|
| B | 1 and 2 |
| G | 1, 4, 5 |
| Hd | 1, 64, 128 |
| T | 1 fallback/non-regression, 32, 127, 128, 129, 512, 2048 |
| KV | equal to T, greater than T where continuation semantics allow |
| mask | none, causal, actual additive mask |
| activation | fp16 mandatory; fp32 generic correctness; bf16 if target supports it |

Use deterministic random seeds and adversarial logits including large positive,
large negative, repeated maxima, masked rows, and near-zero outputs.

### Numeric gate

- Compare to an independent fp32 or fp64 reference, not another composite path.
- Report maximum absolute error, maximum relative error with a near-zero floor,
  mean absolute error, and at least one output sample hash.
- Mandatory fp16 gate: `max_rel_err <= 1e-2` outside the declared near-zero
  region, with an accompanying absolute bound.
- No NaN or Inf divergence from reference.

### Residency gate

- No global allocation proportional to `B*Hq*T*KV` for scores or probabilities.
- At least 80% of baseline score/probability HBM bytes removed; the expected
  supported result is 100% of full-matrix bytes removed.
- QK and PV appear inside the same bounded score-resident compute schedule.

### Artifact

Create `docs/shared-flash-codex-D-residency-20260722.md` with buffer inventories,
allocation formulas, generated schedule, correctness table, and fallbacks.

## 11. Phase E: two contractions through centralized WMMA

### Work

1. Extend generic optimizer handling so both QK and PV contractions remain
   recognizable inside the fused schedule.
2. Apply existing AMD WMMA descriptions to eligible fp16 operands.
3. Keep fp32 accumulation for stable state unless measured evidence justifies a
   different supported contract.
4. Preserve the no-spill topology with full optimization enabled.
5. Support partial/full UNROLL without destroying score/value correspondence.
6. Audit register count, VGPR spilling, LDS use, waves, occupancy, and launch
   geometry.
7. Preserve generic ALU/dot lowering for non-WMMA activation dtypes.

### Gate E

All of these must hold in one configuration with `NOOPT=0`:

- Correctness passes.
- Full score/probability buffers remain absent.
- Generated code contains an actual QK WMMA invocation.
- Generated code contains an actual PV WMMA invocation.
- Each invocation is tied to the intended contraction operands and loop axes.
- No VGPR spill silently replaces deleted HBM traffic.
- Ordinary matmul and REDUCE optimizer regressions pass.

### Artifact

Create `docs/shared-flash-codex-E-wmma-20260722.md` containing annotated call
sites, contraction-to-call mapping, optimizer decisions, resources, occupancy,
buffer evidence, and errors.

## 12. Phase F: shared geometry and search

### Work

1. Define one geometry record containing only real scheduling decisions.
2. Reuse BubbleBeam/FutureSight candidate scoring and ranking.
3. Generate candidates for actual 8B and 14B attention shapes through the same
   search entry point.
4. Reject illegal register/LDS/thread/occupancy candidates before execution.
5. Search static-first under one GPU process at a time.
6. Store selected candidates in one table keyed by device and attention facts.
7. Revalidate correctness, residency, generated calls, and GPU health for every
   promoted entry.
8. Retain the baseline fallback when no candidate beats it reliably.

### Required geometry coverage

- 8B: `Hq=32,Hkv=8,G=4,Hd=128`.
- 14B: `Hq=40,Hkv=8,G=5,Hd=128`.
- `T=KV=512`, `2048`, and `4096` when memory permits.
- Actual mask/layout forms from the model.
- At least query tile, KV block, waves, staging, and output tile choices.

### Gate F

Both routes use the same implementation, schema, search program, and table.
Each promoted geometry improves absolute attention `tm` beyond measurement noise
while preserving every correctness/residency/WMMA gate.

### Artifact

Create `docs/shared-flash-codex-F-geometry-20260722.md` with search space,
candidate counts, rejection reasons, selected entries, resource use, and timing.

## 13. Phase G: trustworthy roofline measurement

### 13.1 Repair Phase 0 measurements

The existing `shared-flash-phase0-baselines-20260722.md` reports `7.1 TFLOP/s`
fp16 WMMA and `5.2 GB/s` bandwidth, while other repository evidence reports
approximately `122.8 TFLOP/s` and `551.6 GB/s`. These values cannot all describe
the same units and measurement regime. Do not use the Phase 0 numbers until the
measurement implementation, units, synchronization, and workload are audited.

Measure empirical ceilings in the same warmed device regime as attention:

- `C_peak`: a representative, sufficiently large fp16 WMMA workload on the
  actual target, including a separate ceiling for any other activation compute
  path claimed by the implementation.
- `B_peak`: sustained device bandwidth from a sufficiently large streaming
  device workload with correct byte accounting.

Record clocks, device, driver/runtime, command, shape, dispatch count, raw
samples, statistic, FLOP/byte formulas, and units.

### 13.2 Roofline formulas

For full causal or unmasked attention, document the exact convention and use a
consistent useful-work calculation. A standard full-matrix convention is:

```text
F_QK = 2 * B * Hq * T * KV * Hd
F_PV = 2 * B * Hq * T * KV * Hd
F_total = F_QK + F_PV
achieved_compute = F_total / GPU_time
```

For causal triangular work, either count executed rectangular FLOPs or useful
triangular FLOPs, but report which and do not mix conventions.

Compute compulsory Q/K/V/output bytes separately from avoidable score and
probability bytes. Report:

```text
arithmetic_intensity = F_total / measured_or_derived_HBM_bytes
roof = min(C_peak, B_peak * arithmetic_intensity)
roofline_efficiency = achieved_compute / roof
compute_frac = achieved_compute / C_peak
memory_frac = achieved_bandwidth / B_peak
```

### 13.3 Measurement protocol

- Use GPU `tm`, not Python wall clock, for kernel/attention timing.
- Warm clocks and execute at least 200 measured dispatches for microbenchmarks.
- Report median plus dispersion or percentile bounds, not only the best sample.
- Synchronize at defined boundaries.
- Run one GPU process at a time and record GPU health before and after.
- Do not use BEAM on the known unstable gfx1100 path.
- Compare baseline and candidate in the same process/session where safe.
- Include compile time separately and exclude it from steady-state GPU `tm`.
- Capture kernel count, but never use it as the success criterion.
- Capture actual allocations or schedule buffers, not estimated deletion alone.

### Gate G

The ceiling measurements are internally consistent and reproducible. Candidate
attention has lower absolute `tm`, higher roofline efficiency or compute
utilization as appropriate, and lower avoidable memory traffic than the shipped
materialized baseline for both real route shapes.

### Artifact

Create `docs/shared-flash-codex-G-roofline-20260722.md` with raw commands,
formulas, samples, corrected ceilings, baseline/candidate tables, and conclusions.

## 14. Phase H: model integration for both routes

### Work

1. Replace dead `prefill_flash_attn` behavior with one real shared eligibility
   and route binding.
2. Route the 8B fp16-overlay model after Q/K/V projection into shared attention.
3. Route the 14B packed-weight model at the same semantic boundary.
4. Keep projection/weight policies unchanged except for passing canonical facts.
5. Preserve decode routing; prefill work must not silently replace promoted
   decode kernels.
6. Preserve baseline fallback for unsupported shapes and unpromoted geometries.
7. Add route census evidence proving the candidate actually ran.
8. Run deterministic model-output parity and GPU-health checks.
9. Measure complete synchronized prefill through the existing authority harness.

### Required whole-model rows

For both Qwen3-8B and Qwen3-14B:

- pp512.
- pp2048.
- pp4096 when supported by the existing authority regime and available memory.
- Baseline route and candidate route in comparable warmed runs.
- Attention-only `tm` captured inside the real model process.
- End-to-end prefill time and tokens/s.
- Peak memory and route identity.
- Fixed-depth decode correctness and throughput non-regression after prefill.

The external llama.cpp result may be reported as context, but promotion is based
on a controlled tinygrad baseline/candidate comparison plus roofline evidence.

### Gate H

- Both models invoke the same shared attention compiler asset.
- Both models pass deterministic output/token correctness.
- Both models improve attention GPU time beyond noise.
- Both models improve end-to-end prefill, or the route remains disabled for the
  model that does not improve.
- Decode correctness and promoted decode throughput do not regress.
- No hidden dense-fp16 materialization is introduced on the 14B route.

Passing only one model is a partial result, not project completion.

### Artifact

Create `docs/shared-flash-codex-H-model-gates-20260722.md` with commands, route
census, memory, correctness, attention timing, whole-prefill results, decode
checks, and per-route GO/NO-GO.

## 15. Mandatory final evidence table

Provide one table for each model and context:

| Field | Shipped baseline | Shared flash |
|---|---:|---:|
| Route identity | | |
| Correctness max abs | | |
| Correctness max rel | | |
| Attention GPU `tm` | | |
| Whole-prefill time | | |
| Whole-prefill tokens/s | | |
| Useful TFLOP/s | | |
| `compute_frac` | | |
| Effective/derived HBM bytes | | |
| `mem_frac` | | |
| Roofline efficiency | | |
| Full-score global bytes | | |
| Full-probability global bytes | | |
| QK call site/lowering | | |
| PV call site/lowering | | |
| Kernel count | | |
| VGPR/SGPR/LDS | | |
| Occupancy/waves | | |
| Peak model memory | | |
| Decode non-regression | | |

Every row must name the exact commit, device, command/artifact, and measurement
regime. Missing evidence is `NOT PROVEN`, never an inferred pass.

## 16. Test and regression ownership

Codex must add or extend focused tests for:

- Semantic attention capture and deliberate non-matches.
- Generic combine update/merge/finalize.
- True graph-local multi-output behavior.
- Ranged, partial-unroll, full-unroll, and optimized lowering.
- GQA layout and V indexing.
- Scale and mask semantics.
- Shape boundaries around 128 and large prefill contexts.
- fp16 correctness and generic fallback dtype correctness.
- No full score/probability buffer.
- Two actual optimized contraction call sites.
- Model route selection and baseline fallback.
- 8B/14B route census.
- Ordinary REDUCE, softmax, matmul, scheduler, AMD ISA/WMMA, model prefill, and
  decode non-regression.

The prior `45 passed, 4 xfailed` suite is only a historical baseline. Existing
xfails may not conceal a mandatory gate. A test that expects the old vector wall
must be replaced when the wall is removed rather than retained as success.

## 17. Execution and blocker protocol

### Execution rules

- Work from the current clean master and preserve rollback to shipped attention.
- Make small phase commits with focused tests and artifacts.
- Remove temporary instrumentation before every phase commit.
- Keep one owner for integration-sensitive files such as `rangeify.py`,
  `devectorizer.py`, `ops.py`, and `model.py`; parallel workers may own disjoint
  tests, measurement, or search assets.
- Rebase or reconcile deliberately; never overwrite unrelated user work.
- Continue through all phases without asking for an intermediate handoff merely
  because a phase is difficult.
- Push only phase states whose stated gate actually passes.

### Valid blocker standard

A blocker is reportable only after Codex provides:

1. The smallest generic reproduction.
2. Exact file, line, invariant, and unsupported IR/scheduler behavior.
3. The failing command and complete graph/traceback/generated-code evidence.
4. Proof it is not a malformed graph, stale cache, forced realization, dtype
   error, unsupported route, or measurement bug.
5. The smallest generic fixes attempted and why each fails.
6. Proof that proceeding would require a prohibited hand kernel, route fork,
   optimizer carve-out, or duplicate WMMA system.
7. A clean last-good commit with failed experiments removed.

Performance below baseline after correctness, residency, dual-WMMA, and a
reasonable shared geometry search is a valid NO-GO. Bank generic compiler work
if useful, but do not enable a slower route or call it complete.

## 18. Definition of 100% complete

Codex may report `COMPLETE` only when every statement is true:

- The unsafe DeepSeek global matcher and debug behavior are gone.
- One first-class semantic attention representation owns Q/K/V/scale/mask facts.
- Every tensor dependency is a normal graph source.
- One generic initialize/update/merge/finalize combine IR contains online-softmax
  math exactly once.
- One stateful-reduction lowering supports ranged, partial-unroll, full-unroll,
  and block-merge execution without combine-specific branches.
- Logical `Hd` state is represented as a logical axis, not `dtype.vec(Hd)`.
- Multi-output state is graph-local, deterministic, and realized from one
  reduction execution without a global cache or independent reductions.
- One rangeify lowering produces bounded score-resident attention.
- No full score or probability buffer reaches global memory.
- QK and PV appear as actual contraction invocations inside the supported fused
  compute topology.
- Eligible fp16 QK and PV both use the existing centralized AMD WMMA path.
- Non-WMMA activation dtypes use the same semantic attention path with generic
  lowering or fail closed to baseline; they do not fork the algorithm.
- One geometry schema, search program, table, and evidence harness serve both
  model routes.
- Qwen3-8B fp16-overlay invokes the shared path and passes correctness,
  residency, resource, roofline, attention-time, and whole-prefill gates.
- Qwen3-14B packed-weight invokes the same shared path and passes the same gates
  without dense-fp16 weight materialization.
- Both routes improve absolute attention GPU time beyond noise.
- Both routes improve end-to-end prefill before promotion.
- Decode correctness and promoted decode performance do not regress.
- Empirical compute/bandwidth ceilings and all roofline formulas are audited and
  reproducible.
- Focused and relevant regression suites are green with no mandatory gate hidden
  behind an xfail.
- Temporary oracle code, dead flags, debug output, duplicate formulas, shape
  hacks, global caches, and route-specific flash assets are removed.
- The final report contains the mandatory table, commands, commits, generated
  call sites, buffer inventories, memory evidence, raw timing samples, route
  census, and honest per-route conclusions.
- Master is clean and the completed commits are pushed.

## 19. Final Codex instruction

Use the current DeepSeek work as prototype evidence, not as trusted architecture.
Preserve the useful generic pieces, remove the unsafe shortcuts, and execute
Phases A through H continuously. Do not stop at a structural or synthetic win.

The deliverable is one reusable compiler-native attention asset that measurably
moves both real prefill routes toward their empirical rooflines without
duplicating attention, optimizer, search, routing, or benchmark machinery.
