# AUTHORITATIVE TASK: shared roofline-oriented flash attention for both model paths

**Owner/executor:** deepseek, end to end  
**Repository:** `/home/ubuntu/tinygrad-arkey`  
**Environment:** `DEV=AMD`, gfx1100  
**Date:** 2026-07-22  
**Status:** authoritative completion scope

## 0. Authority and intent

This document supersedes all earlier DeepSeek flash-prefill task, handoff,
multi-output, composite-reduce, and realignment documents wherever they
conflict. Those documents remain useful as evidence and history, but they no
longer define completion independently.

The project is not complete when a compiler primitive exists, a test suite is
green, two WMMA macros appear, or one synthetic configuration compiles. The
project is complete only when one shared compiler-generated attention path
measurably moves BOTH supported model routes closer to their applicable
empirical rooflines.

The two mandatory routes are:

1. The 8B fp16-overlay route.
2. The 14B packed/non-fp16-weight route.

The weight representations differ before Q/K/V projection. Attention currently
receives fp16 Q/K/V activations on both routes. The shared attention machinery
must begin at that common activation boundary and must not depend on the model's
weight format.

If a supported route actually supplies non-fp16 attention activations, do not
fork the algorithm. Keep the same graph rewrite and composite-reduction
abstraction, make dtype an explicit input to generic lowering/geometry
selection, and measure that dtype against its own empirical compute ceiling.

## 1. End goal

Build one scheduler-native attention implementation that:

- Recognizes the canonical `(Q @ K.T * scale + mask).softmax(-1) @ V` graph.
- Rewrites it into score-resident online-softmax attention.
- Carries running `(m, l, acc)` state over KV without materializing the full
  `T x KV` score or probability tensors in HBM.
- Keeps QK and PV as compiler-visible contractions.
- Reuses the existing centralized tensor-core optimizer when their dtypes are
  WMMA-eligible.
- Uses the best existing generic lowering for other dtypes rather than a
  route-specific kernel.
- Preserves enough occupancy to turn traffic deletion and fast contractions
  into lower absolute GPU time.
- Is selected and validated on both model routes without copying the combine,
  rewrite, lowering, kernel, geometry search, or benchmark implementation.

The concise objective is: **build once, validate and gate twice.**

## 2. Roofline definition of success

Measure the hardware ceilings empirically on the machine used for the gate:

- `C_peak(dtype/path)`: the applicable compute ceiling, measured with a large
  representative kernel using the same execution units and operand dtype.
- `B_peak`: sustained memory bandwidth, measured with a representative
  streaming device workload.

For every baseline and candidate report:

- Absolute GPU `tm`, with warmed clocks and at least 200 dispatches.
- Effective compute throughput and `compute_frac = achieved / C_peak`.
- Effective HBM traffic and `mem_frac = achieved / B_peak`.
- Full-score/probability HBM bytes before and after.
- Kernel count as diagnostic evidence only, never as the success metric.
- Correctness against an independent fp32 reference.

The original measured bracket, roughly `1853 us` materialized versus a
`756 us` fused floor at `T=KV=2048`, is motivation, not a completion result.
Only the built shared path's measured time counts.

## 3. Required shared architecture

### 3.1 Canonical model boundary

Both model routes must feed the same attention interface after producing Q/K/V:

```text
8B fp16-overlay projection ---------+
                                    +--> canonical Q/K/V attention graph
14B packed-weight projection -------+
```

Differences before this boundary remain owned by their existing projection and
weight-lowering paths. Do not move weight-format logic into attention.

### 3.2 One combine representation

Represent a composite-reduction combine as compiler data, preferably a UOp
subgraph or an equivalently generic immutable IR object with the contract:

```text
(state_slots, input_element) -> new_state_slots
```

Online softmax is one instance:

```text
state = (m, l, acc)
element = (score, value)
m_new   = max(m, score)
corr    = exp(m - m_new)
p       = exp(score - m_new)
l_new   = l * corr + p
acc_new = acc * corr + p * value
```

The online-softmax equations must have one source of truth. Ranged reduction,
partial UNROLL, UPCAST, and fully unrolled reduction must all evaluate the same
combine representation. They may use different generic iteration mechanics,
but they may not duplicate or branch on online-softmax math.

### 3.3 One composite REDUCE

The REDUCE machinery owns:

- State-slot allocation and initialization.
- Scalar/vector lane grouping.
- Sequential or associative folding as required by the combine contract.
- Ranged, partially unrolled, and fully unrolled execution.
- Reachability and ordering of every accumulator update.
- Exposure of final state slots to consumers.

The REDUCE machinery must not know the name `online_softmax`.

### 3.4 One attention rewrite

Use one route-independent graph rewrite before rangeify inserts the score and
probability buffers. It must recognize the canonical attention semantics, not a
model name or weight format.

The rewrite emits one shared score-resident structure containing:

1. QK contraction over head dimension.
2. Scale and additive mask.
3. Online-softmax state update over KV.
4. PV contraction/update.
5. Final `acc / l` output.

### 3.5 One optimizer path

Reuse the existing TC optimizer and target tensor-core descriptions. Extend
generic handling of multiple eligible contractions if required. Do not add a
flash-specific WMMA emitter and do not hand-author WMMA fragments or swizzles.

### 3.6 One geometry and search system

Use a single geometry representation and the existing search/ranking assets.
Keys may contain device, activation dtype, B/H/T/KV/Hd, GQA grouping, mask
class, and other real scheduling constraints. Keys must not select duplicated
8B and 14B implementations when the attention shape is otherwise equivalent.

Static entries are allowed for reproducibility. Populate them through one
shared search process. Do not create separate route-specific search programs.

### 3.7 One evidence harness

Use one harness with route adapters. It must generate the independent reference,
capture generated kernels, identify buffers, measure ceilings and `tm`, compute
errors, and render the final comparison table for either route.

## 4. Hard anti-duplication rules

The following are prohibited:

- Separate 8B and 14B flash-attention implementations.
- Copying online-softmax equations into ranged and no-range lowering.
- String dispatch such as `combine_fn == "online_softmax"` in REDUCE machinery.
- Silent fallback from an unknown combine to independent reduction or the last
  input lane.
- A second WMMA emitter, hand-authored fragments, or `__builtin_amdgcn` calls.
- A hand kernel, `custom_kernel`, manual LDS/barrier kernel body, or copied
  `flash_kernels.py` executor.
- An optimizer carve-out for composite reductions.
- `NOOPT=1` in any completion configuration.
- A model-name conditional in the compiler rewrite.
- Separate geometry tables or benchmark scripts containing copied logic.
- Aggregate `grep` or `#define __WMMA` counts presented as contraction proof.
- Comparing against another composite implementation as the correctness oracle.
- Keeping a temporary oracle or duplicate implementation after final parity.

Allowed route-specific code is limited to thin adapters that obtain each
model's real Q/K/V tensors, invoke the shared path, and record route metadata.

## 5. Current verified state and known gaps

Treat these as starting facts, but reproduce the cheap structural facts before
building on them:

- Materialized attention can put QK and PV on WMMA but spills the score tensor.
- PCONTIG-style fusion can remove boundaries while destroying REDUCE structure
  and therefore WMMA eligibility.
- REDUCE-preserving fusion can retain WMMA but has not removed the full score
  spill.
- Composite accumulator infrastructure and `REDUCE_SLOT` exist in partial form.
- Existing numeric composite tests force `NOOPT=1` and do not validate the new
  full-optimizer path.
- Current R1-R3 code uses string-registered hardcoded combines.
- The current no-range online-softmax implementation loses score/value pairing.
- Partial UNROLL remains unsupported for coupled combines.
- `online_softmax_acc` was removed from the registry and unknown combines can
  silently select unrelated semantics.
- Existing `REDUCE_SLOT` tests prove graph structure, not two realized numeric
  outputs from one reduction.
- Reported `max_rel_err=0.12` does not satisfy the required `1e-2` gate.
- Two WMMA definitions do not prove two contraction call sites in the intended
  score-resident kernel.
- No current rangeify rule emits the completed score-resident attention form.

Do not call these gaps complete based on the existing 45-test suite.

## 6. Execution protocol

DeepSeek owns all phases below and continues through them without waiting for an
intermediate Claude handoff. Each phase ends in a small committed artifact so
failure can be localized and the work remains reviewable.

At every phase:

- Keep the working tree clean between commits.
- Run the relevant unit/regression suites before committing core changes.
- Record exact commands and pass/fail/skip/xfail counts in the phase artifact.
- Keep normal single-op REDUCE generated output byte-identical unless a change
  is explicitly required and separately justified.
- Remove instrumentation before committing.
- Use one GPU process at a time, warmed clocks, DEBUG `tm`, and no BEAM.
- Commit on master with the required co-author trailer and push.
- Report negative results and remaining spills explicitly.

Do not stop merely because a phase is difficult. Diagnose, reduce to a minimal
reproduction, and iterate. A valid terminal state is either full completion as
defined in section 15 or a precise generic compiler blocker satisfying section
14. A partial performance result is not completion.

## 7. Phase 0: inventory, canonical boundary, and baselines

### Work

1. Identify the actual shipped 8B fp16-overlay and 14B packed-weight entry
   points and the canonical point where both have produced Q/K/V.
2. Record actual Q/K/V activation dtypes, layouts, shapes, GQA parameters, mask
   form, and scale dtype for each route.
3. Confirm whether their post-projection attention graphs are structurally
   identical. If not, normalize them with thin graph construction adapters,
   not duplicated kernels.
4. Select required gate configurations from real model geometry. Include at
   least `T=KV=2048`; use `512` for iteration and `4096` for breadth if memory
   permits. Record actual Hq/Hkv/G/Hd rather than assuming them.
5. Measure `C_peak` for every applicable activation compute dtype and `B_peak`.
6. Measure the shipped materialized-attention baseline for each route.
7. Capture per-kernel generated-code call sites, buffer sizes, kernel `tm`,
   total attention `tm`, score/probability HBM traffic, and correctness.

### Artifact

Create `docs/shared-flash-phase0-baselines-20260722.md` containing:

- The common-boundary map.
- The exact two route/config matrices.
- Empirical ceilings and measurement commands.
- Baseline roofline tables.
- Actual score/probability buffer evidence.
- Per-kernel contraction evidence.

### Gate

Phase 0 passes only when both real routes can be driven through the same
attention-level harness and their independent baselines are reproducible.

## 8. Phase 1: make composite REDUCE genuinely generic

### Work

1. Define the combine IR and state-slot contract precisely.
2. Keep the existing hardcoded implementation temporarily as a numeric and
   generated-graph oracle. Do not delete it before parity.
3. Implement one evaluator/lowering of the combine IR for a ranged input.
4. Implement generic lane grouping and folding so the same combine handles:
   - Scalar ranged inputs.
   - UPCAST output lanes.
   - Partial reduction-axis UNROLL with remaining ranges.
   - Full reduction-axis UNROLL with no remaining range.
   - Packed logical elements such as `(score, value_vector)` without losing
     element boundaries.
5. Make unsupported combine/dtype/packing forms fail closed with actionable
   errors. Never substitute independent-slot semantics.
6. Preserve all accumulator END dependencies so state updates cannot be DCE'd.
7. Remove combine-specific imports and formulas from `reduce_to_acc` and the
   no-range pass.
8. Add a second unrelated coupled combine to prove the interface is generic.

### Required tests

- Independent `(sum, max)` slots, including both outputs.
- Online-softmax `(m, l)` scalar input.
- Online-softmax `(m, l, acc)` with score/value input.
- An unrelated coupled combine using the identical lowering path.
- Each combine under no optimization, full optimization, partial UNROLL, and
  full UNROLL.
- Scalar and vector value accumulators.
- Unknown combine rejection.
- Normal REDUCE generated-output fixtures unchanged.

### Artifact

Create `docs/shared-flash-phase1-composite-reduce-20260722.md` with oracle diffs,
test matrix, generated-graph evidence, and any intentionally changed output.

### Gate

Phase 1 passes only when one combine representation drives every execution form
and there is no online-softmax-specific lowering branch.

## 9. Phase 2: finish true multi-output semantics

### Work

1. Complete `REDUCE_SLOT` verification, rangeify survival, lowering, symbolic
   survival, and dispatch handling.
2. Lower one composite REDUCE once and expose each final accumulator through
   `REDUCE_SLOT(reduce, i)` without duplicating the reduce.
3. Ensure all requested slots keep the required update chains reachable.
4. Define behavior for unrequested slots explicitly and avoid retaining dead
   work unless the combine state depends on it.
5. Validate slot dtype, index bounds, and ownership.

### Required tests

- One `(ADD, MAX)` reduction over `1..16`; realize both slots and assert
  `sum=136`, `max=16` from exactly one REDUCE.
- Read slots in both orders and through multiple consumers.
- One three-slot online-softmax reduction; read `l` and `acc`, produce `acc/l`,
  and compare with fp32 attention at `max_rel_err <= 1e-2`.
- Full optimizer configurations, including partial and full UNROLL.
- Invalid slot index and dtype fail closed.

### Artifact

Create `docs/shared-flash-phase2-multi-output-20260722.md` with the realized
numeric outputs and proof that one REDUCE feeds every slot.

### Gate

Structural `REDUCE_SLOT` assertions alone do not pass this phase. Both numeric
outputs must be realized from one composite reduction.

## 10. Phase 3: emit score-resident attention once

### Work

1. Trace both route graphs from the canonical Q/K/V boundary through rangeify.
2. Identify the last common graph point before score/probability buffers are
   inserted.
3. Add one semantic attention matcher and one rewrite at that common point.
4. Support the real scale and additive causal-mask forms used by both routes.
5. Emit the generic `(m, l, acc)` composite reduction with QK and PV visible as
   contractions.
6. Ensure no full `T x KV` score or probability BUFFER/STAGE is created.
7. Keep the implementation shape/dtype driven, never model-name driven.
8. Provide a conservative eligibility predicate. Unsupported shapes must use
   the existing baseline path rather than compile incorrectly.

### Required evidence

- Before/after graph for both routes.
- Exact rewrite match count and eligibility reason.
- Buffer list proving the full score and probability buffers are absent.
- Correctness against independent fp32 at `max_rel_err <= 1e-2`.
- One shared rewrite file and one shared composite implementation.
- Unsupported-case fallback correctness.

### Artifact

Create `docs/shared-flash-phase3-residency-20260722.md` with both route traces,
buffer evidence, correctness, and supported-shape contract.

### Gate

Phase 3 passes only when both routes reach the same rewrite and the full score
and probability spills are absent. A manual composite-reduce test is not a
substitute for rangeify emission from the real attention graph.

## 11. Phase 4: preserve the best compute path

### Work

1. Make the centralized optimizer recognize both QK and PV contractions inside
   the score-resident structure.
2. Remove assumptions that only one TC-tagged REDUCE can exist where necessary.
3. Preserve generic optimizer behavior for composite and normal reductions.
4. Keep fp16-eligible operands on the existing WMMA implementation.
5. For non-WMMA activation dtypes, retain the same attention rewrite and use the
   existing appropriate ALU/dot lowering. Do not fork attention.
6. Audit register/LDS pressure and occupancy before tuning geometry.

### Required evidence

- Actual `__WMMA(...)` call lines for QK and PV, identified within their compute
  kernels. Macro definitions do not count.
- Proof that the score/probability buffers remain absent with optimization on.
- `TC_OPT=2`, `NOOPT=0` correctness for both routes.
- Partial/full UNROLL test coverage.
- Non-WMMA dtype generated-code evidence if such a route is supported.

### Artifact

Create `docs/shared-flash-phase4-compute-lowering-20260722.md` containing the
per-kernel call sites, pressure data, buffer evidence, and errors.

### Gate

Phase 4 passes only when traffic deletion and the applicable fast compute path
coexist in the same configuration. Separate runs proving one property each do
not pass.

## 12. Phase 5: shared geometry and occupancy tuning

### Work

1. Expose only real scheduling dimensions: query tile, KV block, waves, staging,
   and other target-supported choices.
2. Integrate them into one geometry/search asset using the existing
   BubbleBeam/FutureSight-style ranking infrastructure.
3. Search static-first for reproducibility.
4. Reject candidates exceeding register/LDS/occupancy limits before expensive
   measurement when possible.
5. Tune representative 8B and 14B geometries through the same search program.
6. Store selected entries in one table keyed by hardware and attention shape,
   not by copied route implementation.
7. Recheck correctness, residency, and call-site evidence for every selected
   candidate.

### Artifact

Create `docs/shared-flash-phase5-geometry-20260722.md` with the shared search
space, rejected-candidate reasons, selected entries, occupancy, and `tm`.

### Gate

Phase 5 passes only when both routes use the same implementation/search assets
and have a stable configuration that improves on their materialized baselines.

## 13. Phase 6: integrate and gate both model routes

### Work

1. Add one routing/eligibility mechanism for the shared attention path.
2. Wire the 8B fp16-overlay route through it.
3. Wire the 14B packed-weight route through it.
4. Ensure route choice does not duplicate the kernel or compiler rewrite.
5. Run model-level correctness and prefill-throughput checks.
6. Re-run attention-only empirical roofline measurements in the real model
   process to catch layout, cache, or scheduling differences.
7. Test at least short iteration size, `T=KV=2048`, and one larger supported
   context for both routes.

### Mandatory final table

For each route and gate size report:

| field | baseline | shared flash |
|---|---:|---:|
| correctness error vs fp32 | | |
| total attention `tm` | | |
| end-to-end prefill time/tok-s | | |
| compute throughput/fraction | | |
| memory throughput/fraction | | |
| full-score HBM bytes | | |
| full-probability HBM bytes | | |
| QK lowering/call-site | | |
| PV lowering/call-site | | |
| kernels | | |
| registers/LDS/occupancy | | |

### Hard completion gates for each route

- Independent fp32 correctness with `max_rel_err <= 1e-2`, plus an absolute
  error metric so values near zero do not distort the conclusion.
- No full `T x KV` score or probability materialization.
- At least 80% of materialized score/probability HBM bytes deleted.
- Both fp16 contractions use actual WMMA call sites when eligible.
- `compute_frac` increases and `mem_frac` decreases consistently with the
  traffic-deletion hypothesis.
- Absolute attention `tm` is faster than the shipped baseline by more than
  measurement noise.
- End-to-end prefill does not regress; the path is wired only if it improves.
- Relevant unit and model regression suites pass.

Both route gates are mandatory. Passing 8B but not 14B, or vice versa, is a
partial result and must not be reported as project completion.

### Artifact

Create `docs/shared-flash-final-report-20260722.md` with commands, environment,
commits, generated code, buffer evidence, complete tables, tests, and an honest
GO/NO-GO for each route.

## 14. Blocker protocol

A blocker is valid only after DeepSeek has:

1. Reduced it to the smallest generic reproduction.
2. Identified the exact file, line, invariant, and unsupported representation.
3. Shown why the failure is not caused by a route adapter, dtype mistake,
   malformed test graph, forced realization, stale PatternMatcher cache, or an
   optimizer configuration mismatch.
4. Tried the smallest generic fix without adding a route-specific branch,
   optimizer exemption, hand kernel, or duplicated combine.
5. Recorded the failing command, traceback/generated graph, and attempted fix.
6. Reverted failed experiments and left the last good committed state clean.

If the only remaining solution requires a hand kernel, duplicated per-route
implementation, or a second WMMA mechanism, stop and report the blocker rather
than violating the architecture. Do not describe that state as completion.

Performance below the shipped baseline is also a valid NO-GO after correctness,
residency, fast lowering, and reasonable shared geometry tuning are proven.
Bank the generic compiler work if useful, but do not wire a non-win.

## 15. Definition of complete

DeepSeek may report **COMPLETE** only when all statements below are true:

- One canonical attention graph feeds one shared compiler rewrite.
- One generic combine representation contains the online-softmax math.
- One composite REDUCE lowering handles ranged, partial-unroll, and full-unroll
  execution without combine-specific branches.
- One real multi-output reduction exposes correct `m/l/acc`-derived values.
- Both real model routes invoke the same score-resident implementation.
- No full score or probability buffer reaches HBM in supported configurations.
- QK and PV use the applicable centralized fast compute path.
- Both routes pass independent correctness gates.
- Both routes show improved absolute attention `tm` beyond noise.
- Both routes move in the expected direction in the two-ceiling analysis.
- End-to-end prefill improves or at minimum does not regress before routing is
  enabled.
- The full relevant test matrix is green.
- Temporary hardcoded or duplicated oracle implementations are removed.
- No optimizer carve-out, `NOOPT` dependency, model-name compiler branch, hand
  kernel, or duplicate asset remains.
- The final report contains reproducible commands and per-kernel evidence.

## 16. Final instruction to DeepSeek

Run this scope from Phase 0 through Phase 6 without waiting for another handoff.
Commit each gated artifact, keep the shared-asset boundary intact, and continue
diagnosing until both route gates pass or section 14's blocker standard is met.

Do not optimize for the appearance of progress. Optimize for the final measured
outcome: one reusable compiler-native attention asset that moves both supported
model paths closer to their real hardware rooflines.
