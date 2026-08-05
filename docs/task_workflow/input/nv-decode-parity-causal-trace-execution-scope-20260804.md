# NV decode parity - causal trace, exact-oracle, and residual-closure execution scope

Date: 2026-08-04

Status: executable analysis/measurement campaign and future implementation
design. Branch:
`nvidia-bringup-20260731`. Drafting boundary: local `1084270bc` plus the
CPU-only timeline-ledger tool and record named in Section 4.

This scope replaces overlap as the default theory of the remaining d512 gap.
It keeps overlap, graph boundaries, host work, and kernel quality as separate
causal terms and requires each term to be measured in one timing regime before
composition. It authorizes CPU analysis, hermetic tests, compact evidence,
scratch oracle adapters, and lock-held diagnostic GPU runs through causal
attribution. Native candidate implementation in Phase 8 requires a separately
reviewed, variant-specific implementation scope. This document does not
authorize a default route change, production runtime/lowering edit, vendoring
llama kernels into the runtime, cross-vendor policy changes, branch promotion,
destructive Git-history rewriting, or branch deletion.

---

## 0. One-line job

Account for every microsecond by which tinygrad's d512 decode token exceeds
llama's, prove the dominant cause with isolated exact-kernel controls and
role-mapped full-primitive substitution, reproduce the useful mechanism in a
tinygrad-owned primitive,
and qualify parity at d512 before independently qualifying d2048 and d4096.

## 0.1 Authority, supersession, and route outcomes

`nv-parity-and-beyond-forward-scope-20260803.md` remains the SOLE canonical
forward authority. This document neither amends nor supersedes it. Existing
variant scopes retain their own gates. P0-P7 and P9 below are a decode-specific
analysis/measurement scope; each Phase 8 production candidate must receive a
new variant-specific scope naming exact production paths, controls,
correctness, regression, and landing gates. P8/P10-P12 here are campaign design,
not implementation authority.

Keep route outcomes distinct:

- `CUDA_D512_DIAGNOSTIC_PARITY`: the diagnostic DEV=CUDA route meets or beats
  llama in a same-session row. This isolates mechanisms but does not qualify
  the native production route or change a default.
- `NATIVE_NV_D512_PARITY_QUALIFIED`: the native NV production route satisfies
  the canonical same-session criterion and correctness pins.
- `CAUSAL_PARITY_VALIDATED`: a stronger evidence label used here when canonical
  parity also repeats in a second session and the token-time residual closes.

The canonical `PARITY-QUALIFIED` definition is unchanged. A CUDA-derived native
candidate must remeasure native NV. A CUDA route cannot become production
without a separate route-promotion scope and evidence that any same-session
route tax is repaid.

## 1. Findings that determine the plan

### 1.1 Current wall result

The newest real CUDA-route A/B is:

| arm | tok/s | ms/token | result |
| --- | ---: | ---: | --- |
| tinygrad CUDA S=1 | 178.159 | 5.61295 | current control |
| tinygrad CUDA S=2 | 178.265 | 5.60961 | `+0.0595%`, wall-neutral |

The latest historical llama authority is `246.32 tok/s = 4.0598 ms/token`.
It was not measured in the current tinygrad session. Therefore:

```text
5.61295 - 4.0598 = 1.55315 ms/token
```

is a historical orientation, not the campaign denominator. The exact current
gap, native-NV route tax, and CUDA-route tax are UNKNOWN until Phase 1.

### 1.2 The explicit overlap route is closed on current evidence

The current S=1 and S=2 arms have identical `1021`-kernel, six-group topology,
identical route-class census, and exact deterministic token hashes. The S=2
wall delta is `3.34 us/token`, far below mechanism scale. This does not prove
that concurrency is absent. It proves that the existing two-stream lowerer
does not recover useful wall time with the current kernel mix.

Planner analysis independently found about `22.05%` theoretical two-queue
physical-DAG savings, but the planner itself added only about `35-39 us` of
critical path. Simulation is not wall performance. Both planner-first and
stream-first work are retired unless a materially different kernel mix creates
a new premise.

### 1.3 Llama's graph is now described more precisely

The CPU-only ledger over `/tmp/llama_nsys_d512.sqlite`, graphId `2`, observes
29 complete `762`-node graph replays and drops two warmups. Median profiled
interval accounting is:

```text
node sum                         5.013811 ms
all-kernel interval union        3.879947 ms
graph start-to-end span          3.888217 ms
internal gaps                    0.008111 ms
span discount versus node sum       22.450%

MMQ interval union               3.579816 ms
non-MMQ interval union           0.746821 ms
non-MMQ hidden behind MMQ        0.445954 ms
non-MMQ exposed versus MMQ       0.300736 ms
```

Thus the profiled graph equation is approximately:

```text
3.888 ms graph span
  = 3.580 ms MMQ union
  + 0.301 ms exposed non-MMQ union
  + 0.008 ms internal gaps
```

Per-class exposure values are non-additive because non-MMQ classes also
overlap each other. The aggregate interval union is the authority.

Llama uses one graph per token. The profiled bounded median inter-replay gap is
about `0.211 ms`; this is not an unprofiled launch-cost claim. The traced run is
slower than the unprofiled wall authority, so profiled class sums must not be
added directly to `4.0598 ms`.

### 1.4 The directly isolated kernel deficit is material but incomplete

Historical like-for-like evidence reports tinygrad quantized non-vocab MMV at
about `3.836 ms / 252 kernels` versus llama bare MMQ at about
`3.239 ms / 216 nodes`, an approximate `0.597 ms` diagnostic cap. The most
important same-shape deficits are:

- Q6_K partial `1024x4096`: roughly `0.26 ms/token` of observed mass;
- Q4_K and Q6_K down projections: roughly `0.23 ms/token` combined;
- gate/up fusion and shape effects: smaller but repeated;
- q/o roles: near parity;
- vocab: now near parity and no longer a leading target.

These values mix profilers and sessions. They rank work but cannot be summed
into a parity forecast.

### 1.5 The next decisive experiment

The exact llama Q4_K/Q6_K kernels must first launch correctly with prepacked
q8_1 activations in isolated roles. A real token experiment must then replace
each mapped tinygrad semantic subgraph with llama's full primitive, including
q8_1 production. This necessarily changes buffers and may change nodes/edges:
it is not a fixed-DAG kernel swap. A separate source-equivalent diagnostic may
hold topology fixed, but is not exact-binary llama evidence. Together these
controls answer whether the dominant gap is:

1. kernel implementation;
2. llama's q8 activation lifecycle and full primitive;
3. graph boundaries and exposed tail;
4. remaining non-MMV kernels or host work.

The float MMVF bridge already proves module loading, tinygrad-buffer use, and
graph capture. It does not prove the packed Q4_K/Q6_K ABI or numerics.

## 2. Governing principles

The campaign follows these repo authorities:

- `structure/Development/coding-principles.md`;
- `structure/Development/performance-primitive-research-principles.md`;
- `structure/Development/tinygrad-coding-overrides.md`;
- `docs/harness-consolidation.md`.

Applied rules:

1. **One benchmark authority.** Extend
   `extra/llm_research/bench.py -> decode/decode_runtime_overhead.py`; do not
   create a competing decode harness.
2. **Classify evidence before mechanism.** Every number is OBSERVED, DERIVED,
   INFERRED, UNKNOWN, CONTROL, DIAGNOSTIC, CANDIDATE, SHIPPED, or REFUTED.
3. **Cheapest boundary-real test first.** CPU parsing precedes live tracing;
   isolated correctness precedes timing; one-role substitution precedes a
   whole-token swap.
4. **Measure the whole primitive.** Packed weights, activation quantization,
   q8 layout, scales, reductions, graph capture, copies, and required
   epilogues are included before any route is called faster.
5. **Llama is an oracle, not a ceiling or shipped dependency.** External
   kernels may isolate causality. The native result must be tinygrad-owned.
6. **Bound unsafe power.** One fail-closed adapter owns local cubin or pinned
   llama ABI details. It rejects mismatched hashes, driver/SM, shapes, layouts,
   argument count, and graph state.
7. **Encode invariants.** Exact call count, graph groups, buffers, edges,
   allocations, copies, model bytes, and correctness are machine-checked.
8. **Keep backend concerns orthogonal.** CUDA diagnostics do not alter AMD,
   Metal, or native NV behavior without separate evidence and scope.
9. **No second scheduler.** A useful scheduling result must ultimately belong
   to the existing graph/runtime authority.
10. **No arithmetic promotion.** Only same-session end-to-end wall rows can
    qualify parity.

## 3. Causal token-time model

For each implementation `I` in one timing regime:

```text
token_wall(I)
  = in_graph_span(I)
  + outside_graph_host_and_device(I)

in_graph_span(I)
  = dominant_mmv_union(I)
  + exposed_non_mmv_union(I)
  + internal_device_gaps(I)

outside_graph_host_and_device(I)
  = token_wall(I) - in_graph_span(I)
```

The subtraction is valid only when the graph span lies wholly inside the same
token wall window. Define the token window as the canonical steady-state token
iteration after inputs/KV are ready and through completion of required output
work. GPU spans use CUDA-event/device time; CPU API work uses a monotonic host
clock; nsys intervals remain a separate profiled domain. Graph build/update,
submit, boundaries, synchronization, copies, allocations, and readback are
descriptive subcategories. They are non-additive unless interval unions and
nesting prove exclusivity; CPU work concurrent with GPU work is never blindly
summed into wall.

The campaign succeeds only when both implementations satisfy:

```text
abs(measured_token_wall - reconciled_token_wall) <= max(0.05 ms, 2% of wall)
```

Profiler dilation, when matched unprofiled event or isolated-kernel evidence
exists, is a separate calibration:

```text
dilation(class) = profiled_duration(class) / unprofiled_duration_bound(class)
```

Repeated nsys traces measure profiler stability, not dilation. No class uses
another class's dilation factor without evidence.

## 4. Phase 0 - compact offline timeline authority

Status: IMPLEMENTED-UNREVIEWED locally; not yet committed or admitted.

Added:

- `extra/llm_research/decode/cuda_graph_timeline_ledger.py`;
- `test/unit/test_cuda_graph_timeline_ledger.py`;
- `docs/task_workflow/output/nv-decode-llama-d512-timeline-ledger-20260804.json`.

The tool uses `graphNodeId` recurrence rather than an arbitrary time-gap
threshold to split replays, reports incomplete/discarded fragments, rejects
overlapping complete replay ranges, and computes:

- node sum, graph span, kernel interval union, overlap mass, and internal gaps;
- aggregate non-anchor union/hidden/exposed time;
- per-class union/hidden/exposed rows;
- exact short/demangled/mangled variant plus grid/block/register/shared/local-
  memory shape census;
- bounded inter-replay gaps with excluded-count and percentile provenance;
- an explicit warning that profiled intervals are not unprofiled wall.

Hermetic tool result: `5 passed` in
`test/unit/test_cuda_graph_timeline_ledger.py`.

P0-GATE remains PENDING until code review, test rerun, compact-size/object
audit, raw-trace hash verification, and commit admission. Raw `.nsys-rep`,
cubin dumps, and large SQLite exports do not enter Git.

## 5. Phase 0.5 - repository and evidence hygiene

### P0.5-A Path ownership census

Before staging anything, classify every dirty or untracked path as:

- user-owned unrelated change: preserve and never stage;
- reviewed source/tool: tests required;
- compact evidence: schema and source hashes required;
- generated executable/cubin/raw trace: scratch-only or ignored;
- superseded artifact: retain until its authority is explicitly replaced.

Use exact path staging only. `git add -A` is banned.

### P0.5-B Push blocker

Two unpublished B3.2 JSON blobs are approximately `159.7 MB` and `160.5 MB`,
above GitHub's per-file limit. Deleting them in a later commit does not fix the
branch because the blobs remain in the unpublished commit history.

The safe repair is:

1. obtain explicit authority for unpublished-tail history rewriting;
2. create a backup ref and repository bundle;
3. prove the rewrite range is exactly
   `origin/nvidia-bringup-20260731..HEAD`;
4. replace raw captures with compact manifests containing schema, counts,
   reproduction command, source hash, and external raw-artifact SHA256;
5. replay only the unpublished tail;
6. run `git fsck`, tests, path census, and object-size audit;
7. push the feature branch with `--force-with-lease`, never another branch.

Without that explicit authority, continue locally and emit a commit map. Do
not rewrite history or claim the branch is publishable.

### P0.5-C Canonical topology field

The current cheap test needed an untracked census helper because canonical
`programs_per_token_by_route` topology was unavailable. Add topology and
provenance as optional observational fields to the canonical decode evidence
authority, default-no-op and byte-identical when disabled. Test both absence
and presence. Do not create a second harness.

P0.5-GATE:

- no unrelated file staged;
- raw large artifacts excluded;
- canonical commands and artifact schema named;
- publishability either repaired with authority or explicitly BLOCKED;
- no performance conclusion depends on an untracked script.

`PUBLISH_BLOCKED` does not block local P1-P7 analysis. It blocks publishing
the branch. History repair remains a side track until explicitly authorized.

## 6. Phase 1 - same-session baseline trio

Run under `/tmp/gpu-bench.lock`, sequentially, with clocks/temperature logged:

1. llama d512 unprofiled default;
2. tinygrad native NV d512 unprofiled default;
3. tinygrad CUDA S=1 d512 unprofiled diagnostic;
4. repeat llama d512 unprofiled default;
5. repeat arms in reverse order if drift exceeds the gate.

### P1-A Performance authority

Use the established tinygrad fixed-depth authority through
`extra/llm_research/bench.py -> decode/decode_runtime_overhead.py`. Use the
established llama wall wrapper `extra/llm/bench/llama_bench.py`. Match model
bytes, quantization, device, depth, warmup, repetitions, and timing intent.
Record exactly what `W` and `D` mean on tinygrad and the corresponding llama
timing window. Do not claim that llama-bench and tinygrad consume identical
token sequences when they do not.

The source-patched llama build, oracle adapter, and cross-runtime correctness
probe are diagnostic controls; none replaces `llama_bench.py` as llama wall
authority.

### P1-B Cross-runtime correctness authority

Separately build or use a narrow llama probe that consumes an explicit shared
input-token file and matched KV start state. Compare preselected logits with a
declared absolute/relative tolerance and greedy next-token identity against
tinygrad. This probe establishes semantic commensurability; it is not the
throughput authority. Repetition hashes establish determinism only.

Record for the paired wall and diagnostic arms:

- tinygrad and llama commits plus dirty-path census;
- llama binary and shared-library hashes and build flags;
- model SHA256, size, quantization metadata, and mmap policy;
- driver, CUDA, nsys, compiler, GPU, SM, clocks, power, and temperature;
- full argv and resolved environment;
- median, p5/p95, robust dispersion after declared warmup, and bracket drift;
- performance-route metadata from a nonperturbing observational field;
- copies, allocations, syncs, and detailed topology from a matched diagnostic
  run if their instrumentation perturbs wall.

P1-GATE:

- the separate identical-input correctness corpus passes;
- bracket drift <= `1.0%` and robust dispersion is reported; no fixed CV rule
  may deadlock a stable median after warmup;
- timing-window differences and model bytes are explicit;
- topology is observed through the canonical authority;
- exact same-session gaps are published as OBSERVED/DERIVED.

If the gate fails, fix benchmark authority, thermal drift, correctness, or
provenance before tracing. No historical number fills a missing row.

## 6.1 Phase 1.5 - cheap route and group-boundary preflight

In the P1 session, emit an aligned native-NV/CUDA class census that explains
the current program-count delta by semantic role, including every residual,
scatter, copy, reduction, and epilogue node. Typed-boundary `copy=0` on one
route does not settle the current CUDA/native delta.

Then test whether the six CUDA graph groups can be reduced without changing
kernel identity, dynamic updates, inputs, outputs, or correctness. Start with
a CPU feasibility/topology proof. If expressible, run one lock-held d512 A/B.
If not, record the exact dynamic-update/dependency blocker and timestamp the
six boundaries later in P9. This term has only an approximately `0.20-0.23 ms`
historical ceiling, but is cheap and orthogonal to the oracle.

P1.5-GATE: either a correctness-preserving wall A/B with identical kernels or
an exact construction blocker. A group-collapse result cannot be called the
whole parity path.

## 7. Phase 2 - llama source and dataflow ledger

This phase is CPU-first and does not require the GPU.

For every d512 decode operation, map:

```text
model role
-> llama host op
-> source function/template
-> kernel symbol
-> weight type/layout
-> activation input type/layout
-> q8_1 producer and reuse count
-> fusion/epilogue behavior
-> expected calls/token
-> expected grid/block/smem/register family
```

Emit this as a machine-readable semantic-call manifest. Every row contains:

```text
model role and layer range
tinygrad semantic call -> ordered launch subgraph and exact node ids/count
llama semantic call -> ordered launch subgraph and exact node ids/count
rows, K, weight type/layout/stride
activation producer, dtype/layout/stride, q8_1 reuse consumers
fusion flag and fused semantics
output dtype/layout and epilogue
grid/block/smem/registers
short, demangled, and mangled symbol identity
cubin/library/source/build hashes
```

Counts and the number of observed variants are derived from the reconciled
manifest, not plan constants. Any topology movement invalidates downstream
arm manifests and requires regeneration.

Minimum role families:

- Q4_K gate/up fused and unfused;
- Q4_K down;
- Q4_K q/k/v/o projections;
- Q6_K down and partial roles;
- Q6_K vocab;
- all `quantize_q8_1` producers and consumers;
- flash score and combine;
- RMSNorm, RoPE, KV set/get, and elementwise epilogues.

Explicit questions:

1. Which activations are quantized once and reused across multiple MMQs?
2. Which q8 packs are graph nodes versus fused work?
3. Which Q4/Q6 kernels use fusion=true, and what semantic work is included?
4. Which roles consume the same activation and can share layout work?
5. Which kernel variants use IDP.4A/dp4a, vector loads, or warp reductions?
6. Which generated layouts differ from tinygrad's current packed weights?
7. What bytes must move from DRAM versus L2 for each observed shape?

P2-GATE: source/dataflow counts reconcile exactly to both the observed llama
762-node ledger and current tinygrad route census. Every semantic role has an
explicit tinygrad-launch-subgraph to llama-launch-subgraph mapping. Any
unexplained dominant node is UNKNOWN and blocks oracle composition.

## 8. Phase 3 - paired live trace and profiler calibration

### P3-A Smallest reliable capture

Preflight `nsys` version/exporter and kill-clean timeout behavior. Then capture
the smallest steady d512 trace that yields at least three complete post-warmup
replays. Run unprofiled wall immediately before and after it.

Capture separately:

- llama node trace;
- tinygrad CUDA S=1 node trace;
- graph-level/runtime API trace for launch/update/sync boundaries.

Never run the full native NV route under a profiler that cannot observe GPFIFO
work. A missing trace is TOOLING_BLOCKED, not zero GPU time.

### P3-B Exact ledgers

For llama and tinygrad, emit the same compact schema:

- graph count and nodes per graph;
- per-token kernel node sum, interval union, span, and internal gaps;
- per-class node sum and interval union;
- dominant-MMV union;
- aggregate exposed non-MMV union;
- graph/API submit, update, wait, and inter-graph gaps;
- shape/resource census;
- source trace SHA256 and profiler version.

### P3-C Llama graph ON/OFF control

There is no established runtime graph-off environment in the pinned checkout.
Use either two pinned builds differing only in `USE_CUDA_GRAPH`, or a minimal
diagnostic-only switch at llama's graph-enable authority. Do not confuse this
with `GGML_CUDA_GRAPH_OPT`; opt=0/1 changes only the additional optimizer and
has already shown about `0.185 ms` wall effect.

Measure graph ON/OFF unprofiled first, then trace only if the wall result needs
mechanism attribution. Require identical source/submodules/compiler/toolkit and
CMake/build flags except the graph control, store build-manifest diffs, and
verify non-graph kernel cubin hashes or SASS/resource census match. The result
is `GRAPH_LIFECYCLE_DELTA`—base overlap plus launch/capture effects—not launch
cost alone.

P3-GATE:

- stable shape/count across repeats;
- >= `95%` of profiled GPU interval union classified;
- repeated-trace timing and shape stability bounded; class dilation deferred
  to matched P4/P5 instrumentation;
- graph ON/OFF builds differ only at the named authority;
- no profiled number is substituted for unprofiled wall.

## 9. Phase 4 - unprofiled graph-span instrumentation

This is the critical missing llama number. Build a narrow patch in a pinned
llama worktree that places dependency-neutral CUDA events around whole graph
launch/replay on the launch stream. Internal MMQ markers are excluded from the
authority because the chain is interleaved with concurrent branches and marker
nodes can change scheduling. Measure:

```text
unprofiled token wall
unprofiled graph replay span
outside-graph remainder
```

Use P5 isolated timing only as an MMQ bound. Validate graph-span
instrumentation with marker-free and at least two marker-density/order controls
plus graph topology hashes.
The patch is diagnostic and does not enter tinygrad production.

P4-GATE:

- event placement is dependency-neutral, graph topology is identical, and
  marker/control wall delta is <= `1%` across at least two controls;
- unprofiled wall reconciles to graph span plus outside-graph work within
  `max(0.05 ms, 2%)`;
- at least two repeats agree within `0.05 ms`;
- exact patched llama commit and diff hash recorded.

If the overhead or topology gate fails, classify `INSTRUMENTATION_BLOCKED` and
retain a bounded interval. Do not subtract a scalar overhead from a perturbed
schedule.

## 10. Phase 5 - exact Q4_K/Q6_K oracle L1

### P5-A ABI authority

Preferred adapter progression:

1. exact extracted cubin with hashed launch manifest;
2. narrow C adapter linked to the pinned llama build;
3. source-recompiled diagnostic kernel, explicitly labeled non-binary-equivalent.

The adapter owns all unsafe detail and fails closed on:

- llama/cubin/library hash;
- driver, CUDA version, SM, context, and module identity;
- kernel symbol and argument count/order/width/alignment;
- weight and activation type/layout;
- grid, block, shared memory, and fusion flag;
- shape not present in the observed d512 ledger;
- unexpected allocation, copy, or synchronization.

Version an `OracleManifest` containing exact llama/source/library/cubin hashes,
SM, driver, toolkit, ABI, typed argument layout, semantic role, activation and
weight layout, shape, launch configuration, and evidence class. A changed
driver/toolkit/build requires requalification with a typed rejection reason.
Exact-binary, linked-adapter, and source-recompiled results are never pooled.

### P5-B Independent correctness corpus

For each variant emitted by the reconciled P2 manifest, construct a corpus from
pinned real GGUF blocks and activation vectors. Compare:

1. independent CPU decode/dot/fusion reference;
2. current tinygrad kernel;
3. exact llama oracle kernel.

Tests require:

- exact packed bytes and strides;
- explicit dtype and tolerance policy per output;
- NaN/Inf checks;
- guard regions and sentinel buffers;
- repeated-output stability;
- graph capture and at least eight correct replays;
- no hidden staging copies or per-replay allocations.

Git stores only model hash/license metadata, block offsets, shapes, extractor
version, deterministic seeds, and expected reference hashes—not model blocks
or activation dumps. The CPU formula is independent of both GPU
implementations and includes fused semantics exactly. Absolute, relative, and
where relevant ULP tolerances are declared before observing results.

Stop before timing on any mismatch.

### P5-C Isolated timing/resource matrix

Only after correctness, measure every manifest shape with identical buffers,
warmup, iteration count, and graph state. Include `quantize_q8_1` production
and reuse as a separate full-primitive row. Record:

- median/p5/p95 duration;
- effective bytes and bandwidth, with DRAM and L2 interpretations separated;
- registers, static/dynamic shared memory, local memory/spills;
- grid/block, active warps/SM, occupancy limits;
- instruction and load-width census;
- required pack/dequant/reduction/epilogue costs.

P5-GATE: every role passes independent correctness, guard, no-copy,
no-allocation, and graph-replay tests before its timing row is admitted.

## 11. Phase 6 - role-mapped oracle factorial

There is no honest exact-binary, fixed-buffer, fixed-DAG real-token swap:
llama MMQ requires dynamically produced padded q8_1 activations, and fused or
single-pass llama roles can replace multiple tinygrad launches. Keep four
evidence classes separate:

1. **offline duration counterfactual:** existing `l2_constant_dag_oracle.py`
   attaches durations to a frozen DAG; simulation only;
2. **exact isolated prepacked control:** exact llama binary consumes prepacked
   q8_1 in a role microbench; kernel bound only, never token wall;
3. **strict node-for-node source-equivalent control:** an optional diagnostic
   kernel accepts tinygrad's current activation contract and preserves every
   node/buffer/edge; fixed-topology attribution but not exact-binary llama;
4. **exact full-primitive semantic-subgraph replacement:** q8_1 producer plus
   exact llama MMQ replaces the mapped tinygrad role subgraph; real-token
   evidence with an explicit topology delta, never called constant-DAG.

The P2 manifest is the authority for counts. The current orientation is 252
tinygrad Q4/Q6 launches versus 216 llama non-vocab MMQ nodes, but downstream
arms consume manifest node IDs rather than hardcoded counts.

Exact full-primitive arms:

| arm | semantic change | question |
| --- | --- | --- |
| A | current tinygrad real-token primitive | control |
| O1 | one P5 wall-ranked semantic role | does the smallest real role replacement convert? |
| OQ6 | all mapped Q6_K non-vocab role subgraphs | full Q6 contribution |
| OQ4-U | mapped Q4_K roles using unfused llama semantics where available | Q4 kernel/dataflow contribution before fusion |
| OF | true llama-equivalent W1/W3 fused primitive | fusion plus 36-launch structural contribution |
| OALL | all mapped non-vocab Q4/Q6 semantic subgraphs | complete non-vocab primitive contribution |
| OV | OALL plus vocab role | residual vocab contribution |
| OL | alternate qualified q8_1 producer/reuse lifecycle feeding the same exact MMQs | activation-lifecycle economics |

For each arm, emit a semantic-subgraph delta manifest:

- removed/added node IDs, edges, buffers, physical ranges, groups, and launches;
- activation producers/consumers and q8 reuse counts;
- allocations performed before capture and proof of no per-replay allocation;
- copies and synchronization, which must be zero unless required and explicitly
  charged to the primitive;
- semantic output equivalence and independent correctness;
- graph topology hashes before and after.

The activation-family economic gate is:

```text
amortized_q8_cost(a) = q8_producer_wall(a) / qualified_reuse_count(a)
llama_style_role_cost(a)
  = amortized_q8_cost(a) + sum(exact_mmq_wall_for_consumers(a))

GO only if llama_style_role_cost(a) < current_fp16_role_primitive_wall(a)
at the whole-primitive and in-model boundaries.
```

Run unprofiled wall first. Trace only arms whose wall result changes the next
decision. Randomize order within the locked session and bracket with A.

P6-GATE:

- exact correctness and guard tests before timing every arm;
- complete semantic/topology delta manifest;
- no hidden copies, per-replay allocations, or synchronization;
- control bracket drift <= `1%`;
- per-arm wall delta measured, never predicted from kernel sums;
- evidence class stated: simulation, isolated exact, fixed-topology
  source-equivalent, or exact full primitive.

Residual closure is not required for each factorial arm; it is completed after
P9 measures outside-graph and exposed-tail terms.

Belief-flip rules:

| observation | conclusion | next owner |
| --- | --- | --- |
| isolated exact wins, O1 loses | q8 production/integration/topology cancels kernel gain | activation lifecycle/full primitive |
| OALL recovers about the historical MMV deficit, large residual remains | quantized primitive is real but incomplete | P9 residual owners |
| OALL approaches CUDA diagnostic parity | quantized primitive dominates CUDA gap | variant scope for tinygrad-owned mechanism |
| OF wins independently | W1/W3 fusion and launch reduction are promoted by wall | variant-specific fusion scope |
| OL beats default producer | q8 production/reuse is causal | tinygrad activation-format scope |
| semantics/copies/topology are unexplained | experiment invalid | adapter/harness correction |
| observed recovery exceeds old cap | historical profiler cap was stale/confounded | refresh class accounting |

### P6 admission checkpoint

Known historical ceilings—MMV about `0.597 ms`, six-group gaps about
`0.20-0.23 ms`, vocab about `0.02 ms`, and attention at most roughly
`0.04-0.17 ms` depending trace—do not by themselves close the historical
`~1.55 ms` gap. After OALL/OF and the P1.5 boundary result, publish:

```text
measured recovered wall
current same-session residual
measured or trace-bounded remaining owner set
```

If no evidence-backed owner set can reach parity, classify
`CAUSE_ACCOUNTED_PARITY_PATH_UNKNOWN` and widen discovery to consumer-specific
epilogue/kernel-count work before authorizing native implementation. Do not
imply quant kernels alone are a scoped closure set.

## 12. Phase 7 - explain why the oracle wins

Only for wall-relevant shapes, run one representative Nsight Compute capture
per role, not a whole-token capture. Compare exact llama and tinygrad kernels:

- requested and actual DRAM/L2 bytes;
- sector utilization and load transaction width;
- achieved DRAM/L2 bandwidth with the correct ceiling;
- warp issue stalls and eligible warps;
- occupancy limiter;
- registers, spills, shared memory, and local memory;
- integer dot, bitfield/extract, conversion, address, and reduction mix;
- lane-to-row/column mapping and tail behavior;
- cache reuse and persistence assumptions.

Mechanism claims must be falsifiable. Examples:

```text
If lane mapping is causal, matching coalesced sectors while preserving math
must recover >= X us on the isolated role and >= Y us in the role-mapped full
primitive.

If q8 reuse is causal, producing one packed activation and sharing it across
the named consumers must reduce whole-primitive wall more than its pack cost.
```

P7-GATE: the measured oracle delta has a source/dataflow/ISA/resource
explanation that predicts a boundary-real test. “Llama uses dp4a” is not an
accepted explanation.

## 13. Phase 8 - tinygrad-owned native candidates (future variant scopes)

No implementation is authorized here. Rank candidate scopes only from P6/P9
wall evidence. Do not preselect Q6 partial: its legal single-pass sweep closed
NO-GO, while other cheap candidates already have useful diagnostic evidence.
The candidate queue includes:

- one real-route Q4 pure-LDG.128 gate/up A/B, explicitly charging its 8 KiB
  staging cost per real decode launch rather than amortizing it over 2000
  microbench passes;
- true two-accumulator W1/W3 fusion, separately measuring its kernel and
  36-launch structural effects;
- Q4/Q6 down or Q6 partial substrate only if oracle wall promotes them;
- activation production/reuse only if the P6 economic gate passes;
- consumer-specific GEMV epilogue folding for residual/add/scatter/down work;
- attention or other exposed classes only after the residual ledger promotes
  them.

Candidate progression, required for each role:

```text
source/render audit
-> emitted ISA/resource audit
-> microkernel diagnosis
-> independent/reference correctness + guards
-> isolated role
-> whole-linear/full primitive including pack cost
-> in-model native-NV d512 A/B
-> default-closed candidate decision
```

Implementation constraints:

- own the change at the smallest tinygrad representation/lowering authority;
- capability-gate NVIDIA-specific lowering;
- no hardcoded llama SASS or production dependency on a local cubin;
- no opaque shape table without a generator and invariant tests;
- preserve generic fallback;
- keep AMD/Metal behavior byte-identical unless separately scoped;
- encode layout, signedness, bounds, and reduction invariants;
- mark refuted routes/artifacts retired; propose destructive branch cleanup in
  P12 only after reachability/ownership census and explicit user authority.

P8-GATE: a separately scoped candidate is CANDIDATE only if the full primitive
and in-model native-NV d512 wall win with correctness. Isolated-only wins remain
DIAGNOSTIC. Before promotion-ready status, run d2048/d4096 correctness and a
declared bounded-regression smoke test; full parity qualification remains P11.

## 14. Phase 9 - graph-boundary and exposed-tail residual

Run only after Phase 6 identifies the surviving residual.

### P9-A Six-group boundary ledger

Timestamp each group boundary on the current CUDA route:

- CPU build/update time;
- submit API time;
- device end-to-next-start gap;
- waits and synchronization;
- copies, allocations, and readback;
- graph parameter updates;
- host idle/runtime overhead.

The historical approximately `0.20-0.23 ms` gap is only a ceiling until its
owner is named. Test one-group or fewer-group replay only if correctness and
dynamic update semantics permit it. Preserve kernel identity/count and token
outputs.

### P9-B Exposed non-MMV tail

Compare tinygrad and llama aggregate exposed non-MMV union, then promote only
classes with measured residual mass. Candidate examples are flash score/
combine, RMSNorm, RoPE, KV stores, residual/add/scatter nodes, partial
reductions, and consumer-specific GEMV epilogues. Llama has fewer separate
epilogue nodes, so kernel-count and fusion ownership are first-class residual
terms. Previous generic KV and RMS fusion attempts were neutral/regressive; do
not repeat them without a new mechanism.

### P9-C Overlap re-open rule

Re-open S2 or a scheduler lane only if the oracle/native kernel mix changes
resource compatibility and a fresh trace predicts at least `5%` wall recovery.
The new premise must state which ready nodes, resources, and durations changed.

P9-GATE: one named boundary/class owner explains each material residual; the
wall ledger closes; candidate A/B changes wall in the predicted direction.

## 15. Phase 10 - d512 composition and parity gate

Compose only independently admitted candidates, one at a time, with an
ablation ledger. Never sum forecasted microseconds.

Required same-session table:

| arm | correctness | tok/s | ms/token | graph groups | kernels/token | copies | classified residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama control A | pass | | | 1 | 762 | | |
| native NV current | pass | | | | | | |
| + candidate 1 | pass | | | | | | |
| + candidate 2 | pass | | | | | | |
| final candidate | pass | | | | | | |
| llama control B | pass | | | 1 | 762 | | |

d512 is canonically `PARITY-QUALIFIED` when the native-NV production row meets
the existing forward authority: same-session median ratio `>=1.00` under the
declared repetition protocol and correctness pins. A DEV=CUDA row is reported
separately as `CUDA_D512_DIAGNOSTIC_PARITY`.

The stronger `CAUSAL_PARITY_VALIDATED` label requires:

1. canonical native-NV parity already passes;
2. both llama brackets agree within `1%`;
3. independent correctness passes;
4. no required work is excluded;
5. no external llama kernel remains in the candidate;
6. the token-time residual closes within `max(0.05 ms, 2%)`;
7. the result repeats in a second session, with a declared confidence interval
   or non-inferiority margin that does not treat measurement noise as a win.

## 16. Phase 11 - depth qualification

d512 parity does not qualify d2048 or d4096. Repeat the paired baseline,
compact trace ledger, correctness, and final candidate wall at each depth.
Attention/KV/context terms are allowed to reorder priorities.

Campaign parity requires independent same-session rows:

```text
d512  tinygrad / llama >= 1.00
d2048 tinygrad / llama >= 1.00
d4096 tinygrad / llama >= 1.00
```

If only d512 passes, report `D512_ONLY`. If prefill remains qualified, report
it separately; prefill never substitutes for a decode depth.

## 17. Phase 12 - promotion and maintainability

Promotion is a separate decision after performance qualification. Require:

- one boring capability/config authority with closed default;
- generic fallback and clear unsupported behavior;
- no external oracle dependency;
- bounded backend-specific code;
- tests for renderer/lowering, numerics, layout, graph replay, and fallback;
- AMD/Metal regression controls or explicit backend isolation;
- LOC, authority, environment-variable, and dead-route audit;
- reproducible compact evidence and docs;
- cleanup plan for scratch artifacts and superseded branches.

## 18. Hard stops

Stop the current phase and classify the result when:

- model bytes, shapes, quantization, or declared timing windows differ without
  being labeled; identical-input correctness outputs differ beyond tolerance;
- an oracle role fails the independent CPU reference;
- a graph arm adds an unexplained copy, allocation, sync, or edge;
- a raw artifact would exceed repository policy;
- CUPTI/nsys cannot observe the route;
- profiler overhead is mixed into unprofiled wall;
- thermal/control drift exceeds `1%`;
- a diagnostic result is about to change a default;
- a proposed fix requires a second scheduler or hardcoded external ABI in
  production;
- history rewriting, LFS adoption, or force-push is required without explicit
  authority.

Correct labels include `CORRECTNESS_BLOCKED`, `ABI_BLOCKED`,
`TOOLING_BLOCKED`, `PROVENANCE_BLOCKED`, `PUBLISH_BLOCKED`,
`WALL_NEUTRAL`, `REFUTED`, and `DEFERRED`. None means hardware cannot do the
work.

## 19. Artifact and commit plan

Every phase emits one compact record and, where appropriate, one compact JSON.
Raw `.nsys-rep`, `.sqlite`, cubin, binary, and full capture payloads remain
external with SHA256 and reproduction commands.

| phase | deliverable | commit class |
| --- | --- | --- |
| P0 | timeline ledger tool, unit tests, compact llama JSON | `[tool][test][evidence]` |
| P0.5 | ownership/publishability record, optional canonical topology field | `[docs]`, optionally `[tool][test]` |
| P1 | same-session baseline trio | `[evidence]` |
| P2 | llama source/dataflow ledger | `[docs][evidence]` |
| P3 | paired compact trace ledgers and profiler calibration | `[evidence]` |
| P4 | llama diagnostic instrumentation record | external patch hash + `[evidence]` |
| P5 | fail-closed oracle adapter and correctness corpus | `[scratch][test][evidence]` |
| P6 | role-mapped oracle factorial and topology-delta record | `[evidence]` |
| P7 | ISA/resource causal record | `[evidence]` |
| P8 | one reviewed variant scope per native candidate; later implementation commits | future `[scope]`, then separately authorized `[runtime][test]` |
| P9 | boundary/residual record | `[evidence]` |
| P10 | d512 parity record | `[evidence]` |
| P11 | depth qualification record | `[evidence]` |
| P12 | promotion/cleanup scope and implementation | separately authorized |

No commit mixes an external oracle adapter, a production route change, and a
performance verdict.

## 20. Immediate execution order

The shortest causal path is:

1. review and admit the compact offline timeline tool and record;
2. repair canonical topology observation without cloning the harness;
3. run the same-session llama/native-NV/CUDA baseline trio;
4. settle the cheap six-group feasibility term and aligned native/CUDA census;
5. finish llama's source/dataflow/semantic-subgraph manifest while acquiring
   the paired minimal trace;
6. measure dependency-neutral unprofiled whole-graph span in the pinned llama
   worktree;
7. build exact Q4_K/Q6_K prepacked standalone correctness for the cheapest
   manifest role, then rank the first real-token role by full-primitive value;
8. replace that one semantic role with q8 production plus exact llama MMQ and
   emit the complete topology delta;
9. expand only after the one-role wall gate, testing unfused Q4/Q6, true W1/W3
   fusion, and activation lifecycle as distinct arms;
10. use the remaining wall equation and P6 admission checkpoint to choose a
    separately scoped native primitive versus boundary/epilogue/tail discovery;
11. compose only landed measured winners and qualify native-NV d512;
12. repeat independently at d2048 and d4096.

This order keeps tracing llama, but every additional trace must settle a named
unknown. The campaign is no longer “find another plausible optimization.” It
is a sequence of controls that forces the `~1.55 ms` historical gap into one
of a small number of owners and then verifies each recovered term at wall.
