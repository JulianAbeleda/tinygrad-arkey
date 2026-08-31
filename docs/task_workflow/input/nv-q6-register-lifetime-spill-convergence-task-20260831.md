# Q6 owner register-lifetime and spill convergence task

Date: 2026-08-31
Status: ready for sequential execution
Route: generated tinygrad Q6_K x Q8_1 Stream-K owner main
Representative shape: `M=512, N=4096, K=12288`

## Goal

Generate a tinygrad-native packed Q6_K x Q8_1 tensor-core owner kernel that is
bit-exact against the established owner oracle, contains no llama cubin
dependency, and meets the pinned llama performance boundary.

The immediate goal is narrower: explain and remove the generated kernel's
excess local-memory traffic. The task is complete only when either:

1. the generated main kernel is at most `211.2768 us` and the main+fixup pair
   is at most `220.3488 us`; or
2. every experiment in this document has a recorded binary and runtime result,
   and the remaining gap has been localized to a newly measured category with
   an equally concrete follow-on task.

Do not promote a source-level improvement without post-compile binary evidence.

## Fixed oracle and baseline

### Llama oracle

- Main cubin:
  `docs/task_workflow/evidence/nv-packed-q4k-q8-llama-extracted-20260830/q6k-mmq-dense.sm_120a.cubin`
- Fixup cubin:
  `docs/task_workflow/evidence/nv-packed-q4k-q8-llama-extracted-20260830/q6k-fixup-dense.sm_120a.cubin`
- Main: `201.216 us`
- Fixup: `8.640 us`
- Pair: `209.856 us`
- Main 5% gate: `211.2768 us`
- Pair 5% gate: `220.3488 us`
- Launch: 170 CTAs, 256 threads, `128x128` output tiles, 48 K=256
  work units per output tile.

### Generated exact owner

- Constructor:
  `extra/llm_research/prefill/nv_generated_q6k_streamk_slots.py`
- Qualifier:
  `extra/llm_research/prefill/bench_nv_generated_q6k_streamk_owner.py`
- Exact values: all `5,570,560` FP32 values, `max_abs=0.0`
- Exact ownership: all 340 tile IDs
- Best established minimum before the current metadata experiment:
  `678.516 us`
- Current metadata-lifetime commit: `d33bdf09a`
- Current measurement: `681.662 us` minimum, `696.249 us` median
- Resources: 255 registers/thread, 58,880 shared bytes, approximately
  1,080 local bytes/thread.

### Static SASS comparison

These are `nvdisasm` instruction counts, not hardware performance counters.

| metric | generated | llama | interpretation |
|---|---:|---:|---|
| registers | 255 | 255 | both reach the architectural allocation ceiling |
| stack frame | 512 B | 72 B | generated has 440 B more stack |
| LDL | 155 | 31 | generated has 124 more local loads |
| STL | 183 | 29 | generated has 154 more local stores |
| IMMA | 256 | 512 | different two-row-path representation, not missing math |
| LDSM | 32 | 64 | same topology normalization caveat as IMMA |

The verified first lever is excess spill/stack traffic. It is not yet proven
that eliminating it closes the complete timing gap.

## 2026-08-31 measured update

The original spill theory is falsified as the primary lever. The compiler-wide
`128x128x64`, 256-thread kernel is exact and has zero stack, zero `LDL`, and
zero `STL`, but measures `459.488 us` minimum (`460.256 us` median). Removing
local traffic therefore cannot by itself close the `258.272 us` residual to
the llama main gate.

The next controlled route is the existing compiler-source wide Stream-K
transform with 170 persistent owners. Its first ownership census found 94
output tiles with two partial segments and 34 with three. The previous wide
fixup accepted only two slots and was therefore structurally incomplete. The
wide deterministic fixup ABI now accepts three ordered slots, and
`bench_nv_compiler_q6k_wide_streamk.py` is the one-command exact/timing gate.

## Non-negotiable constraints

- Preserve canonical packed Q6_K weights and Q8_1 records as direct inputs.
- Preserve the 170-owner partition, row-major partial workspace, tile IDs, and
  deterministic fixup ABI until a separate task explicitly changes them.
- Preserve exact owner output and tile IDs after every experiment.
- Do not depend on, embed, translate, or dispatch llama cubins.
- Do not double the generated IMMA count to match the whole-entry llama count.
  The binaries encode their two output-row paths differently.
- Do not use source UOp, PTX, or register counts as a substitute for SASS
  `LDL/STL`, stack-frame, and runtime measurements.
- Express any new lifetime or fragment primitive in backend-neutral terms.
  Backends may lower it to warp, wave, SIMD-group, or CTA-specific mechanisms.
- Keep each experiment independently revertible and commit only admitted work.

## Required result record for every experiment

Each experiment must append one row to the experiment ledger and preserve its
generated cubin or a content hash. Record:

- commit and experiment ID;
- exact values, `max_abs`, and exact tile IDs;
- minimum and median runtime from the standard qualifier;
- delta from the preceding admitted baseline;
- residual to `211.2768 us`;
- registers/thread, shared bytes, and local bytes/thread;
- stack-frame bytes;
- SASS counts for `LDL`, `STL`, `IMMA`, `LDSM`, `LDG`, `STG`, `LDS`, `STS`,
  barriers, and branches;
- spill instruction locations grouped by loop phase or epilogue;
- source/UOp change and the causal theory it tests;
- verdict: `ADMIT`, `REJECT`, or `INCONCLUSIVE`.

An experiment is `ADMIT` only if it is exact and improves at least one binary
spill metric without materially regressing minimum runtime. It is a performance
win only if repeated minimum and median both improve outside run-to-run noise.

## Stage 0: reproducible binary gate

### T0.1 Capture the exact compiled owner cubin

Add or use a qualifier mode that saves the exact NVRTC cubin launched by the
benchmark. Name it with the source commit and SHA-256. Do not inspect a stale
`/tmp/current_q6_owner.cubin` without verifying its hash.

Acceptance:

- saved symbol is `nv_generated_q6k_streamk_owner_partials`;
- cubin resource usage matches the launched program;
- one command regenerates the qualifier JSON, cubin, resource report, and
  disassembly from a clean process.

### T0.2 Produce structured SASS census

Create a small reusable analyzer that counts opcodes and classifies `LDL/STL`
program counters into prologue, K loop, tile-boundary flush, final epilogue, or
unknown. Keep NVIDIA disassembly parsing outside the backend-neutral IR.

Acceptance:

- analyzer reproduces generated `155 LDL / 183 STL` and llama `31 / 29` for
  the pinned artifacts, subject to an explicitly recorded current-baseline
  update;
- output is stable JSON suitable for experiment diffs;
- IMMA/LDSM row-path normalization is documented in the output.

### T0.3 Establish timing noise

Run the unchanged owner qualifier in at least three fresh processes. Record all
samples, process minima, and medians. Define the admission noise band from these
runs before evaluating small wins.

Acceptance:

- exactness passes in every process;
- the ledger records the noise band and the baseline cubin hash;
- subsequent experiments use the same clocks, device, inputs, and warmup.

## Stage 1: localize the spill producers

This stage changes no kernel semantics. Its output is a map from spill
instructions to graph values or code regions.

### T1.1 Separate loop-body spills from epilogue spills

Use SASS control-flow boundaries, IMMA clusters, and source line information
where available to count local instructions before, within, and after the MMA
loop. If line information is unavailable, use instruction neighborhoods and
branch targets and mark the mapping as inferred.

Decision:

- if at least 60% of excess local instructions are inside IMMA clusters,
  prioritize Stage 2;
- if at least 60% are in scale accumulation or output conversion, prioritize
  Stage 3;
- otherwise execute Stages 2 and 3 in order.

### T1.2 Identify spilled value classes

For representative `LDL/STL` pairs, trace stack offsets and surrounding moves
to classify values as:

- FP32 accumulator;
- integer MMA accumulator/update temporary;
- A fragment;
- B fragment;
- Q6 D or packed scale;
- Q8 scale;
- tile/owner/index metadata;
- predicate/control state;
- output conversion/store temporary;
- unknown.

Acceptance:

- at least 80% of excess generated local operations are classified, or every
  unclassified cluster is listed with its PC range and dependency neighborhood;
- the next experiment targets the largest classified category.

### T1.3 Record live-range pressure by phase

At the UOp/linearized program level, count simultaneously live accumulator,
fragment, scale, address, and predicate values around each IMMA group and the
tile transition. This is a compiler diagnostic, not a new CUDA-only semantic.

Acceptance:

- report identifies peak-live regions and value classes;
- report can be generated for other renderers even when they do not expose
  SASS;
- no scheduling change is admitted as part of this diagnostic task.

## Stage 2: accumulator and WMMA lifetime experiments

Run these one at a time. Restore the last admitted state after every rejection.

### T2.1 Shorten integer MMA update chains

Test whether chained `.after(update)` dependencies keep prior integer
accumulators and fragment temporaries live across unrelated MMA groups. Express
the dependency at the smallest correct group boundary rather than every scalar
consumer.

Admission:

- exactness and ABI pass;
- IMMA count and mathematical coverage remain unchanged;
- `LDL+STL` falls by at least the established noise-independent binary delta;
- runtime does not regress.

### T2.2 Materialize accumulator-bank ownership

Introduce a backend-neutral scoped accumulator-bank value whose contract is:
one logical bank is owned by one static MMA region, updates occur in program
order, and the bank is released at a declared flush boundary. The contract must
not prescribe NVIDIA register numbers or warp size.

Lowering options may include grouped PHIs, explicit region arguments/results,
or renderer-native accumulator declarations. Implement the smallest option that
allows the compiler to see one bank rather than many overlapping SSA versions.

Admission:

- generic verification/spec tests cover nesting, types, and illegal escape;
- legacy callers remain unchanged;
- Q6 owner exactness passes;
- stack frame and `LDL/STL` both decrease;
- another backend can legally ignore or lower the marker conservatively.

### T2.3 Split integer MMA accumulation from FP32 scale folding

Test a bounded schedule in which integer tensor-core accumulators for one
column/output subgroup are completed, immediately converted and folded into
their FP32 outputs, then released before the next subgroup. Sweep only a small
set of static subgroup widths: 1, 2, 4, and 8 column groups.

For each width record IMMA count, spill counts, stack, and runtime. Do not select
by source size.

Admission:

- exactness passes;
- the selected width is Pareto-best in runtime and local traffic;
- no extra global partial round trip is introduced.

### T2.4 Constrain WMMA temporary lifetime

Audit the renderer lowering around the native fragment and WMMA result. Ensure
A/B fragment values, integer output vectors, bitcasts, and lane extraction do
not remain live beyond their last consumer because of graph-wide aliases or
semantic tags.

Implement an explicit backend-neutral end-of-region/lifetime boundary only if
the diagnostic proves the normal last-use information is insufficient.

Admission:

- native-fragment unit tests pass;
- exactness passes;
- targeted temporary class disappears or shrinks in the SASS stack-offset map;
- no CUDA-specific register spelling enters UOp semantics.

### T2.5 Schedule output rows sequentially

Test llama-shaped sequential row-path ownership while preserving one logical
accumulator bank and the same output. Complete and retire one output-row group
before beginning the next when dependencies permit.

Admission:

- no duplicated arithmetic;
- exactness passes;
- lower peak-live count and lower SASS local traffic;
- runtime improves.

## Stage 3: scale, epilogue, and control lifetime experiments

### T3.1 Fold scale values at narrowest scope

Load/decode Q6 D, Q6 scale, and Q8 scale close to the completed integer result,
apply them once, and release them before the next independent subgroup. Avoid
recomputing loads merely to reduce SSA lifetime.

Admission:

- global/shared scale instruction counts do not increase materially;
- local traffic and runtime improve;
- exactness passes.

### T3.2 Isolate tile-boundary flush/reset

Represent the at-most-two owner tile segments with one explicit transition:
flush the completed bank, reset it, then begin the next tile. Do not predicate
every MMA or operand load on the transition.

Admission:

- at most one boundary flush per segment;
- exact tile IDs and values;
- fewer live predicates/index values in the MMA region;
- lower branches or local traffic and improved runtime.

### T3.3 Strength-reduce owner/tile coordinates

Replace repeated division/modulo-derived coordinates inside the K loop with
incrementing indices established at segment entry. Preserve dynamic owner
bounds and the existing fixup map.

Admission:

- exactness passes for boundary owners as well as the representative run;
- integer instruction/control count improves;
- spill count does not regress;
- runtime improves outside the noise band.

### T3.4 Narrow final conversion/store lifetime

Ensure FP32 output conversion, address construction, tile ID, and store
predicates are produced only at flush/final-store boundaries. They must not stay
live across the MMA loop.

Admission:

- output/store ABI unchanged;
- exactness passes;
- epilogue stack offsets or local operations decrease;
- no store count regression.

## Stage 4: renderer and compiler controls, only if proven necessary

Do not begin this stage merely because Stage 2 is difficult. It requires
evidence that graph lifetimes are correct but the CUDA toolchain still spills
avoidable values.

### T4.1 Region-aware linearization

Teach linearization to keep native fragment/accumulator regions contiguous and
to release region-local values after their declared outputs. This must be a
general scheduling property, not an NVIDIA kernel special case.

### T4.2 Controlled recomputation versus retention

Add a costed choice for cheap address/predicate/scalar expressions: recompute
when retaining the value crosses a high-pressure region. Do not recompute
global/shared loads or MMA results. Validate on at least one non-NVIDIA renderer
or generic scheduling test.

### T4.3 Backend register-allocation hints

Only after T4.1/T4.2, evaluate renderer-local scopes or launch bounds that alter
NVRTC allocation. Such hints cannot be the semantic correctness mechanism and
must degrade safely on unsupported backends.

Admission for all Stage 4 work:

- generic compiler tests;
- no regression in existing native-fragment callers;
- exact Q6 owner result;
- SASS and runtime evidence showing the intended effect.

## Falsified or closed theories

Do not repeat these without new evidence:

| theory | result | verdict |
|---|---|---|
| A-fragment materialization/residency is the dominant lever | exact; source `ldmatrix` reduced to 32; only 10.641 us gained and local reservation increased | useful substrate, dominant prediction falsified |
| decoded scalar metadata retention is dominant | exact; current minimum 681.662 us; local reservation changed only slightly | rejected as primary lever |
| regenerate A fragment per MMA consumer | exact; 678.436 us minimum; 1,080 B local; no meaningful gain | rejected |
| double IMMA/LDSM to match llama whole-entry counts | topology audit shows llama contains two row paths represented differently | invalid experiment |
| lower nominal register count alone | both kernels use 255 registers and launch is already limited to one CTA/SM | not an occupancy lever by itself |
| repair the old two-bank owner | 4,568-4,808 local bytes/thread and approximately 8.8 ms | rejected architecture |

## Stage 5: remeasure the remaining gap

After every admitted spill-reduction experiment, update the binary census and
timing decomposition. Once generated local traffic is within 20% of llama
(`LDL <= 37`, `STL <= 35`, stack frame `<= 86 B`), classify the residual:

- global-memory instruction/path difference;
- shared-memory traffic or bank conflicts;
- instruction dependency/latency;
- integer address/control overhead;
- barrier/synchronization overhead;
- owner scheduling or fixup overhead;
- Q8 producer/included-route overhead.

Do not assume spill elimination recovers the whole gap. At this threshold,
attempt hardware profiling again with a capture mode that exposes the
runtime-loaded kernel. If profiling remains unavailable, use controlled binary
and runtime perturbations and label conclusions as inference.

## Promotion sequence

1. Owner exactness and tile IDs pass.
2. Owner main is `<= 211.2768 us`.
3. Main+fixup is `<= 220.3488 us`.
4. Included route, including Q8 production, is `<= 0.308134 ms`.
5. Q4_K gate/up representative shape is requalified using the same substrate.
6. Q6_K and Q4_K FFN-down routes are requalified.
7. Projection-pair activation reuse is measured separately.
8. Role-specific epilogues are measured separately.
9. Only passing routes are promoted; each retains a fallback.

## Agent-sized execution units

Assign exactly one task at a time to a low-effort agent:

1. `T0.1`: reproducible cubin capture.
2. `T0.2`: structured SASS census.
3. `T0.3`: timing noise ledger.
4. `T1.1`: spill phase localization.
5. `T1.2`: spill value classification.
6. `T1.3`: backend-neutral live-range diagnostic.
7. Execute the highest-ranked Stage 2 experiment.
8. Re-run exactness, SASS, and timing gates.
9. Admit and commit, or revert only that experiment and record rejection.
10. Repeat from step 7 until Stage 2 and Stage 3 are exhausted or promotion
    passes.

An agent must not combine two semantic experiments in one patch. Every handoff
must include the exact command, cubin hash, measurements, changed files, and
verdict. If an experiment fails to compile, the agent records the compiler
failure and restores the last admitted state before handing off.

## Expected result

The next decisive result is not a predicted token/s number. It is a causal
measurement: whether accumulator/WMMA lifetime control reduces generated local
traffic toward llama's binary levels and produces a proportional runtime win.

If it does, continue the same measured loop to the `211.2768 us` owner gate. If
it does not, this task will still close the uncertainty by exhausting the
accumulator, fragment, epilogue, and control lifetime categories and producing
a measured residual ledger for the next primitive.
