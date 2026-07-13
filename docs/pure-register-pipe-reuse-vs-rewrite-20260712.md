# Pure register pipe: reuse versus rewrite decision

## Decision

Do **not** rewrite the compiler or the AMD backend. The reusable path is already
large enough to justify an incremental implementation. The correct change is a
new register-resident storage implementation behind the existing lifecycle,
descriptor, and resource boundaries, plus one backend wait-lowering seam.

This is not a claim that the register path is executable today. It is a scope
decision: the current code has the right reusable pieces, but they are joined
by an LDS-specific adapter. A typed AMD LLVM wait intrinsic now exists, but the
missing lifecycle/dependency wiring is still a coordinated extension, not a
second compiler.

## Implementation status

The reusable storage boundary is now implemented in
`tinygrad/codegen/opt/register_pipeline.py` (commit `448a13dc7` plus the
prologue-ordering follow-up). Each A/B role owns a persistent two-slot
`DEFINE_REG` half buffer; producers write `half.vec(16)` global-load carriers
to the selected slot, and fragments load the current slot after typed producer
readiness. The shared stage-1 builder uses matching current-consume/next-slot
prefetch semantics and keeps the prologue dependency on the first body read.
K=1, K=2, K=3, and K=256 ownership proofs pass with no `DEFINE_LOCAL` or raw
ISA nodes.

This does not yet make a full-K AMD kernel executable. The valid full-K graph
now passes the normal AMD rewrite; the earlier devectorizer `IndexError` was
caused by a malformed synthetic WMMA ABI (a scalar output contract paired with
an 8-lane accumulator), not by register-buffer lowering. The remaining gates
are executable postrange admission, typed targeted-wait lowering, final
resource evidence, correctness, and pinned GPU timing. No route is promoted
until those gates pass.

## Evidence inventory

### Reusable without semantic rewrite

| Concern | Existing owner | Evidence | Reuse action |
|---|---|---|---|
| Epoch/slot ownership and prologue/body/drain | `tinygrad/codegen/opt/kernel_pipeline.py` | `stage1_lifecycle_events` and `prove_stage1_lifecycle` establish producer, ready, consume, release, overwrite, and complete-drain invariants | Generalize the plan's storage assumptions; keep the proof and event model |
| Typed producer/fragment callbacks | `Stage1StorageAdapter`, `KernelStage1ProducerStage`, `KernelStage1FragmentStage` | `build_stage1_uop_graph_with_storage` already routes LDS through typed callbacks | Add a register adapter implementing the same callback results |
| WMMA descriptor and lane remaps | `tinygrad/codegen/opt/kernel_lds.py`, shared tensor-core descriptor | `PrecontractPipelineTemplate.__post_init__` validates dimensions, ranges, four binary A/B axes, folded element IDs, and descriptor remaps | Extract descriptor checks from LDS allocation checks; do not duplicate them |
| Accumulator ownership and loop-carried state | `build_stage1_uop_graph` and postrange accumulator contract | Existing graph constructs `float.vec(8)` slices, updates, drain, and exact owner coverage | Reuse the graph and WMMA callback; retain the existing accumulator contract |
| Global b128 and WMMA instruction selection | `tinygrad/renderer/isa/amd.py` | `AMDOps.GLOBAL_LOAD_B128` and `AMDOps.V_WMMA` lower to the required RDNA3 instructions | Keep instruction selection; feed it compiler-owned values rather than route-owned instruction lists |
| Register allocation/resource gates | AMD ISA renderer and resource capture | Fragment windows, accumulator pins, no-spill behavior, and final resource extraction already exist | Add a register-pipe resource budget and fail closed on unknown/overflow |
| Candidate policy/cache separation | `PipelinePolicy`, `RegisterPipePlan`, candidate context | Register and LDS policies are now distinct and immutable | Keep canonical identity and policy fields; do not add another route schema |
| Correctness and proof artifacts | existing unit/authority tests | LDS precontract and lifecycle suites are passing | Clone the test shape for register storage, not the implementation |

### Missing or currently coupled

| Gap | Why it blocks a reusable register path | Smallest change |
|---|---|---|
| Storage callback was LDS-only | `PrecontractPipelineTemplate` requires `DEFINE_LOCAL`, LDS windows, and `active_lds_bytes` | **Implemented:** register adapter emits persistent `DEFINE_REG` stage buffers and no local allocation; retain the shared descriptor checks |
| Lifecycle plan encodes stage-1 LDS semantics | `KernelStage1PipelinePlan` requires `stage_count == 1` and derives slots from LDS bytes | Keep physical LDS slots separate from logical register stages; add an explicit register mapping (two logical stages, zero LDS slots) rather than changing the existing plan in place |
| Wait dependency is not yet a compiler UOp contract | Typed `WaitCount` now lowers through AMD LLVM's `llvm.amdgcn.s.waitcnt`, while dependency selection is still inserted/derived separately from the lifecycle | Thread typed `WaitDependency` through graph metadata; require a backend hook and provenance join before launch |
| Native targeted waits are renderer-local | `_insert_waitcnt` tracks physical register spans after post-regalloc; route code cannot reuse it without importing `AMDOps`/`Ops.INS` | Reuse the dependency algorithm as backend implementation, not its raw instruction representation; add a typed marker/source proof at the compiler boundary |
| Wait lowering is not yet lifecycle-proven | Commit `6deda3c7c` proves a typed `WaitCount` intrinsic can compile on gfx1100, but an arbitrary `Ops.WAIT(WaitCount)` is not itself proof of a load-group dependency | Keep the new intrinsic as the backend seam; wire it to `WaitDependency` and reject unproven/untagged waits before promotion |
| Exact global-load-to-WMMA ABI | The normal devectorizer requires real range ownership, CONTRACT axes/remaps, half.vec(16), and a matching float.vec(8) output contract | **Structurally implemented:** persistent register producer and chained full-K WMMA fixture pass normal AMD rewrite; runtime execution still needs the wait/resource gates |
| Mixed-role route attribution | The current authority can report selected candidate roles without recording fallback roles | Fix attribution before combined pure promotion; this is instrumentation, not a compiler rewrite |

## What “reuse” means in practice

The implementation should have one lifecycle and two storage policies:

```text
candidate policy
  -> shared descriptor/CONTRACT validation
  -> shared epoch/slot lifecycle + proof
  -> storage policy callbacks
       LDS: global b128 -> LDS window -> barrier -> LDS b128 fragments
       register: global b128 -> register fragment carrier -> typed wait -> WMMA
  -> shared WMMA/accumulator/store construction
  -> backend-specific wait/resource lowering
```

The register policy must return `KernelStage1ProducerStage` and
`KernelStage1FragmentStage`; it must not return an instruction list and must not
call `extra/qk/prefill/wmma.py::build_gemm_pipe`. The handwritten primitive is a
teacher for cadence and expected resource shape only.

The existing lifecycle has an important semantic mismatch that must remain
explicit: `KernelStage1PipelinePlan.stage_count=1` means one proved lifecycle
template with one or two physical LDS slots, while `RegisterPipePlan.stages=2`
means two logical register load stages. Mapping those fields directly would
silently misreport storage. Introduce an adapter/mapping rather than changing
the established LDS meaning.

## Rewrite threshold

A rewrite would be justified only if one of these facts is proven:

1. the existing UOp graph cannot represent a global b128 producer whose values
   retain row/K range ownership and the required CONTRACT/remap metadata;
2. the existing WMMA and accumulator ABI cannot consume the resulting
   `half.vec(16)`/`float.vec(8)` carriers; or
3. no backend can lower a typed wait dependency without route-owned raw ISA
   (the new typed AMD LLVM intrinsic is evidence against this condition).

Current evidence proves none of these impossibilities. It shows an incomplete
adapter and a backend capability gap. Therefore a full rewrite would duplicate
the validated descriptor, lifecycle, and resource machinery and increase risk.

## Ordered implementation boundary

1. Extract descriptor/range/CONTRACT validation from
   `PrecontractPipelineTemplate` while keeping all LDS allocation checks in the
   LDS implementation.
2. Add a register storage callback implementation that emits only ordinary
   compiler UOps (global loads, typed carriers, and dependency edges).
3. Extend the lifecycle graph with typed wait dependencies and a fail-closed
   backend lowering interface.
4. Reuse AMD ISA's physical wait analysis behind that interface; do not expose
   `AMDOps` or raw `Ops.INS` to route code.
5. Add compile-only, resource, correctness, and pinned timing gates for one
   role (`attn_qo`) before expanding to `attn_kv` and `ffn_down`.

## Exit criteria

This decision is validated only when the register candidate has:

- no `DEFINE_LOCAL` or LDS window in the lowered graph;
- exact A/B CONTRACT axes, descriptor remaps, and `half.vec(16)` carriers;
- lifecycle proof for K=1, K=2, full K, and tail cases;
- a final backend artifact proving targeted waits (or a documented measured
  barrier-only ceiling, without promotion);
- joined source/binary identity, VGPR/SGPR, LDS, scratch, spill, and ABI facts;
- full-output correctness and pinned isolated timing.

Until then, the current LDS candidate remains the execution oracle and the
register policy remains a fail-closed compile/selection candidate.

## Hand-pipe recovery and comparison scope (2026-07-13)

### Objective

Recover enough trustworthy evidence to answer one narrow question:

```text
At attn_qo (M,N,K)=(512,4096,4096) on gfx1100, is the historical
build_gemm_pipe register/L2 atom faster than the compiler-generated direct_l2
candidate when both are correct, independently identified, and timed under the
same isolated protocol?
```

This is a diagnostic recovery of an existing backend atom.  It is not a route
promotion, a rewrite of the atom, or permission to copy its instruction list
into the generated compiler path.  The hand atom remains a cadence/resource
teacher even if it wins.

### Facts already established

| Fact | Current evidence | Meaning |
|---|---|---|
| raw hand source | `extra/qk/prefill/wmma.py::build_gemm_pipe` | one wave computes a 32x32 tile with two VGPR fragment banks and targeted VMEM waits |
| current raw route | `route_pf16_graph_gemm -> emit_prefill_gemm_from_spec -> _emit_schedule -> build_gemm_pipe` | uses `Tensor.custom_kernel`, argument order A/B/C, grid 128x16x1, local 32x1x1, and one-byte non-transport LDS placeholder |
| standalone wrapper | `attn_qo_three_way_diagnosis_20260713.py::compile_pipe_program` | reproduces the current route's builder, ABI, geometry, sink, and assembler construction |
| standalone result | `bench/attn-qo-three-way-diagnosis-20260713.json` | exact guarded dispatch times out; completion signal remains 65 rather than 66; post-child GPU health passes |
| historical source | git `b1259638d` | contains materially the same two-bank loop, waits, branch, epilogue, `s_sendmsg(3)`, and `s_endpgm` |
| historical performance | `raw-hand-s9-combined-best-authority.json` | dirty whole-model run reports 4413 pp512, but has no per-role program/binary dispatch join and its binding gate fails purity/rollback expectations |
| generated direct | candidate `a51aca406e7d...` | exact correctness passes; 112 VGPR, zero LDS/spill, 0.799 ms median and 0.439 ms minimum in the latest session |
| proven LDS | candidate `7e6fe384...` | exact correctness passes; 40 KiB LDS, 0.521 ms median and 0.283 ms minimum in the same session |

The present standalone and route construction are too similar to assume an
adapter bug.  The first unresolved fact is whether the historical performance
run actually dispatched the claimed pipe binary, or whether a later assembler,
ELF, descriptor, launch, or runtime change made a previously working atom hang.

### Hypotheses, ordered by information value

| ID | Hypothesis | Decisive evidence |
|---|---|---|
| H0 | historical route attribution was incomplete or false | whole-model trace lacks the exact attn_qo function, source hash, binary hash, geometry, and dispatch count |
| H1 | post-`b1259638d` stack regression | the historical tree's exact binary passes today while current HEAD hangs; bounded commit bisection identifies the first bad change |
| H2 | launch/ABI or descriptor mismatch | builder instruction bytes agree but kernarg order, SGPR enable bits, wave32 mode, workgroup geometry, LDS declaration, or code-object metadata differ |
| H3 | loop/control-flow defect | no-WMMA/no-memory loop canary hangs, or branch count/target does not terminate for K-derived `LOOPS` |
| H4 | targeted waitcnt defect | full-wait variant completes while `vmcnt(LPB)` hangs, with all other bytes semantically unchanged |
| H5 | WMMA/resource defect | control/load/wait canaries finish but first-WMMA or repeated-WMMA canary hangs; descriptor VGPR count or operand ranges disagree with final ISA |
| H6 | epilogue/termination defect | compute-without-store completes but epilogue, final wait, `s_sendmsg(3)`, or `s_endpgm` boundary does not |
| H7 | shape/grid interaction | one-workgroup legal shapes pass but exact multi-workgroup grid hangs; guard/readback then isolates addressing from completion |

These are diagnostic classifications, not preselected conclusions.  Every
canary changes one axis and carries its own binary identity.

### One authority path

Extend the existing `attn_qo_three_way_diagnosis_20260713.py` and guarded
executor.  Do not add another allocator, dispatcher, timeout process, health
probe, reference implementation, or timing harness.  Compile-only extraction
may be factored from the current route, but the route and diagnostic must call
the same helper so their program construction cannot drift.

The shared compile result must expose:

- generator revision and normalized generator parameters;
- final function name, source hash, binary hash, and code-object hash;
- argument order and kernarg segment size;
- global/local geometry, wave size, workgroup size, and LDS bytes;
- descriptor SGPR/VGPR/scratch fields and final register high-water marks;
- branch targets/offsets, waitcnt immediates, WMMA count, and termination tail;
- whether it came from historical replay, current route, or a named canary.

### Ordered packets

#### HP0 — Freeze current failure

Recompile the current exact raw pipe twice in fresh processes and prove stable
source/binary identity.  Dispatch only once per child through the existing
30-second timeout, then run the independent health probe.  Record no timing.

Exit: stable identity plus two matching `timed_out` results, or stop and record
that the failure is nondeterministic.  Never retry an unhealthy GPU.

#### HP1 — Historical replay

Use a clean detached worktree at `b1259638d`; do not alter the active tree.
Run compile-only first, capture the historical code object and metadata, then
run one exact guarded dispatch using that tree's own compiler/runtime.  Also
compile the historical generator through the current stack as a cross join:

```text
old generator + old stack
old generator + current stack
current generator + current stack
```

Exit cases:

- old/old passes: proceed to HP2 stack bisection;
- old/old hangs: historical 4413 is not standalone correctness authority;
  proceed to HP3 canaries, not performance comparison;
- identities cannot be reconstructed: classify the historical artifact as
  non-replayable and retain only its whole-model throughput claim.

#### HP2 — Differential stack bisection

Only if old/old passes.  Compare generated instruction objects, serialized ISA,
ELF notes, kernel descriptors, launch packet, and kernarg packing before doing
a commit bisection.  Bisect only the smallest implicated surface:

1. AMD instruction encoding/assembly;
2. ELF/code-object descriptor generation;
3. `ProgramInfo` geometry and kernarg ABI;
4. runtime queue/dispatch packet construction.

Each bisection point gets compile-only identity first and at most one isolated
dispatch.  Stop at the first bad commit with a minimal reproducer; do not patch
around an unknown descriptor or runtime regression.

Exit: first bad change and failing field/instruction are identified, or the
old/current cross matrix disproves a stack regression.

#### HP3 — Bounded completion canaries

Only when no passing historical binary exists.  Add diagnostic-only variants
to the existing builder interface; production defaults remain byte-identical.
Run the minimum legal one-workgroup shape before exact shape.  Canaries are
strictly ordered and stop on the first unhealthy result:

1. termination only: prologue plus final termination;
2. finite scalar loop only: same branch structure, no VMEM/WMMA;
3. global loads plus full wait, no WMMA/store;
4. one WMMA with initialized fragments, no K loop;
5. one K pair with full waits;
6. one K pair with targeted `vmcnt(LPB)` waits;
7. full K loop, no epilogue stores;
8. epilogue stores with guards;
9. exact grid.

For termination, compare the existing `s_sendmsg(3); s_endpgm` tail against an
`s_endpgm`-only canary.  For waits, compare full drain and targeted count while
preserving load order.  For control flow, statically prove loop count and
branch destinations before dispatch.

Exit: the first single feature that changes completion is identified.  A
canary that merely completes is not numerical correctness evidence.

#### HP4 — Minimal fix or refutation

Fix only the owner of the proven defect.  Examples: descriptor field in ELF
owner, branch relocation in assembler owner, or wait immediate/lifecycle in
the hand atom.  Do not add route-specific environment repair or special-case
the benchmark adapter.  If the defect is intrinsic to the raw atom and a safe
minimal fix cannot be stated, mark the atom refuted and stop.

Exit: exact dispatch completes in two fresh children, GPU remains healthy, and
the final fixed binary has a new explicit identity.

#### HP5 — Full correctness

Reuse the exact deterministic nonconstant A/B inputs and fp32 reference used by
direct and LDS.  Require full 512x4096 output comparison, unchanged inputs,
intact guards, finite output, and pre/post health.  Report max error and reject
constant, sparse, partial-row, NaN, or timeout results.

Exit: correctness passes twice with stable binary identity.  Otherwise the
hand atom is benchmark-ineligible.

#### HP6 — Apples-to-apples timing

Run strictly sequential fresh-child sessions for:

1. hand pipe;
2. generated direct_l2;
3. proven WMMA-LDS control.

Use identical inputs, warmups, rounds, clock policy, device timestamps, guard
checks, and post-session health.  Run at least three sessions with rotated
candidate order.  Compilation, allocation, upload, reference, and readback are
excluded.  Record every sample; use session medians and bootstrap confidence
or a conservative noise band.  Do not compare kernel milliseconds directly to
historical whole-model tok/s.

Exit: hand-vs-generated ratio has a stable sign outside the noise band, or the
result is explicitly `inconclusive`.

#### HP7 — ISA attribution

Explain the result from final artifacts rather than source intent.  Compare:

- global b128 loads per WMMA and distinct A/B fragment reuse;
- targeted wait values and load-to-first-use distance;
- cross-iteration preload overlap;
- address/loop/control instruction counts;
- VGPR/SGPR allocation, spills, waves/workgroup, and occupancy constraints;
- code size and epilogue instruction count.

On gfx11, unavailable PMC groups remain unavailable; do not fabricate L2,
memory, or compute counters.  Static ISA plus controlled variants may support
mechanism attribution, while timing alone supports only the winner verdict.

#### HP8 — Feed the generated scheduler

Translate only proven, backend-neutral lessons into existing policy/search
fields: wave tile, K unroll, fragment-bank count, reuse, wait policy, and
prefetch distance.  The generated path must continue through ordinary compiler
UOps and the centralized storage-aware schedule.  Never import hand-register
numbers, instruction objects, branch choreography, or the raw atom itself.

Exit: a generated candidate independently passes compile/resource/correctness
gates.  Promotion remains a separate decision against LDS.

### Safety and stop rules

- One GPU child at a time; no background GPU agents.
- Compile/static inspection precedes every new binary dispatch.
- One dispatch per unproven binary, 30-second hard timeout, independent health
  after termination, and no automatic retry.
- Stop immediately on failed health, process isolation failure, guard damage,
  or a fault that escapes the child boundary.
- Historical worktrees are read-only diagnostic inputs; do not cherry-pick or
  merge during replay.
- Never claim historical correctness from throughput, route labels, source
  similarity, or a cached artifact without exact binary/dispatch identity.
- Never time a candidate before exact correctness passes.
- If old and current binaries both hang and HP3 cannot isolate a first failing
  feature without broad ISA mutation, stop as `raw_pipe_refuted` rather than
  spinning.

### Definition of 100 percent

This recovery is complete only when all are true:

1. the historical 4413 claim is classified as replayed-and-bound,
   throughput-only, or non-replayable;
2. current route and diagnostic compile through one shared program builder;
3. exact source/binary/descriptor/ABI/geometry identity is captured;
4. the hang is assigned to one proven owner, or the raw atom is explicitly
   refuted with bounded evidence;
5. a surviving hand candidate passes full exact correctness twice under guards
   and isolation;
6. hand, generated direct, and LDS are timed under one sequential protocol in
   at least three order-rotated sessions;
7. the verdict is `hand_pipe_wins`, `generated_direct_wins`, `retain_lds`,
   `inconclusive`, or `raw_pipe_refuted`;
8. the verdict includes final ISA/resource attribution and states unavailable
   counters honestly;
9. no production route changes merely to make the diagnostic pass; and
10. any lesson transferred to generated code is represented in the existing
    centralized policy/search space rather than copied raw ISA.

### Estimated effort after scoping

| Outcome | Expected effort |
|---|---:|
| historical replay immediately identifies a stack regression | 4-8 hours to isolate and validate |
| bounded wait/termination/ABI defect | 1-2 days including exact correctness and timing |
| control-flow or WMMA defect requiring atom repair | 2-4 days |
| no passing historical binary and canaries do not isolate safely | stop after roughly one day with `raw_pipe_refuted` |

The highest-value first milestone is HP0+HP1.  It distinguishes a real
historical regression from an unproven historical attribution before any new
kernel work is attempted.

### HP0/HP3 result (2026-07-13)

The first recovery pass completed enough to classify the current comparison:

| probe | result |
|---|---|
| current raw pipe compile | deterministic: source hash `b48c4f30...`, binary hash `7a743622...`, geometry 128x16 global / 32x1 local, 32 b128 loads, 16 WMMA, 6 waits |
| historical `b1259638d` compile | byte-identical source and binary hashes to current HEAD |
| current exact double-buffer dispatch | timeout at signal 66/65; child containment and post-child health pass |
| exact dispatch with `FULLWAIT=1` | same timeout; full waits do not repair it |
| single-buffer E0, K=64 | correct; RMSE 0.00162 |
| single-buffer E0, K=4096 | correct; RMSE 0.01321 |
| E1 high VMEM fragments | correct |
| E2/E3 VALU-provenance fragments | correct |
| E4 high accumulator / E6 high dead footprint | correct |
| post-probe GPU health | healthy |

This changes the diagnosis from “unknown hand-pipe hang” to:

```text
raw_pipe_double_buffer_lifecycle_or_loop_recurrence_failure
```

The historical 4413 whole-model result cannot be attributed to a different
hand-pipe binary: its exact artifact is missing, while the historical source
reproduces the current binary.  It remains a credible hybrid throughput
measurement but not standalone hand-pipe correctness or performance evidence.

The hand pipe is therefore benchmark-ineligible until its two-bank schedule is
repaired or disproven.  The next repair packet is narrowly bounded:

1. compare the two-bank branch target and loop counter against a one-bank
   unrolled reference;
2. run a double-buffer K=64 one-workgroup canary;
3. remove only the loop backedge, then only the alternating-slot reuse, to
   identify whether control flow or overwrite-before-consume is causal;
4. require exact full-shape correctness twice before any timing.

Additional canary evidence: a K96 double-buffer kernel (the first shape that
takes the backedge more than once) faults, while a forced-no-backedge K96
variant completes dispatch but is intentionally numerically incomplete.  The
encoded branch target is statically correct at the same loop target for K64,
K96, K128, and K4096.  Inserting 32 scalar NOPs before the backedge does not
restore completion.  The failure is therefore classified as an unsafe
taken-backedge/alternating-bank lifecycle, not as a malformed branch offset or
a simple wait-count timing shortage.

At this point the raw atom is `raw_pipe_benchmark_ineligible`: the evidence is
sufficient to reject it as a comparison authority, but not sufficient to claim
a production repair.  Any repair must replace the current alternating-bank
recurrence with an explicitly proved producer/consume/release lifecycle and
then pass the existing exact guarded correctness gate.

No evidence currently justifies claiming that the hand pipe is faster than
generated direct-L2.  The only valid measured comparison remains generated
direct-L2 versus proven WMMA-LDS, where LDS wins.

### HP4-HP7 completion (2026-07-13)

The bounded repair found that the alternating fragment banks were not the
owner of the failure.  The raw stream crossed an invalid ownership boundary:

1. `build_gemm_pipe` resolved `s_cbranch_scc1` to a concrete byte displacement
   before final rendering;
2. the generic renderer then scheduled across that concrete branch and
   inserted additional waits without a symbolic basic-block boundary; and
3. the serialized taken branch landed at byte 1212 in the output-store
   epilogue instead of the loop top at byte 320.

K64 survived because its backedge was not taken.  K96 was the first canary to
take the corrupted edge and fault.  This exactly explains the observed
boundary without requiring a speculative WMMA or register-bank hazard.

The repair has two centralized parts:

- raw control flow stays symbolic through scheduling/wait placement and the
  renderer now resolves `s_cbranch_scc1` in the final byte stream; and
- `PreassembledStreamPolicy`/`preassembled_linear` marks hand-authored ISA as
  owning instruction order and waitcnt placement.  The generic compiler may
  package and relocate that stream, but may not silently reschedule it.

Validation of final pipe binary
`9be1e239c274649551505243913c794bf7409ce190de127543fb8a4f2670889c`:

| gate | result |
|---|---|
| final-stream remu K64/K96/K128/K4096 | all correct; K4096 RMSE 0.01320 |
| guarded GPU K96 | pass; full output, guards, inputs, and health pass |
| guarded full 512x4096x4096 | pass twice in fresh children |
| full-shape max absolute error | 0.0001261 |
| post-run GPU health | pass after every child |

Three strictly sequential order-rotated sessions used 20 warmups and 20 timed
rounds per candidate.  Session medians in milliseconds were:

| order/session | hand pipe | generated direct-L2 | proven WMMA-LDS |
|---|---:|---:|---:|
| pipe, direct, LDS | 0.59880 | 0.45810 | 0.27892 |
| direct, LDS, pipe | 0.68638 | 0.42746 | 0.28362 |
| LDS, pipe, direct | 0.63956 | 0.44618 | 0.28492 |
| median of session medians | **0.63956** | **0.44618** | **0.28362** |

Final verdict: `retain_lds`.  At this shape the repaired hand pipe is about
1.43x slower than generated direct-L2 and 2.25x slower than WMMA-LDS;
generated direct-L2 is about 1.57x slower than WMMA-LDS.  Preserving the hand
schedule materially improved it versus the compiler-mutated 0.86-1.02 ms
stream, but did not change the winner.

HP8 transfers only backend-neutral lessons: represent loop/control ownership
explicitly, keep preassembled and compiler-owned schedules distinct, and make
wait policy an owned lifecycle field.  No hand register numbers, branch
displacements, or raw instruction choreography are copied into the generated
scheduler.  The raw pipe is now correctness-eligible as a diagnostic control,
but it is not a promotion candidate at this shape.
