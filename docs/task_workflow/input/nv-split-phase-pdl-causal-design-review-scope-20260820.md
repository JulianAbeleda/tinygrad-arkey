# NV split-phase PDL causal design review scope

Date: 2026-08-20

Status: **measurement-first external causal review**. This packet authorizes
read-only source analysis, probe-only measurement tooling, and the bounded GPU
tests in section 7. It authorizes no production runtime/model change, no route
promotion, and no performance claim before its named test passes. It is written
for a fresh DeepSeek reviewer. The repository is on
`nvidia-bringup-20260731` at `6570abc02`; the grounding artifacts named below
are uncommitted analysis from the same tree and must be treated as evidence to
audit, not authority to repeat.

## 1. The ask

Test why the current tinygrad decode design does not reproduce llama.cpp's
device timeline even though the traces suggest three visible mechanisms: fused
epilogues, programmatic dependent launch (PDL), and off-path K/V work.

The review must test two materially different candidate diagnoses:

1. **Split-phase dependency deficit.** Candidate: tinygrad represents a
   dependency as producer-complete before consumer-launch, while llama
   separates consumer-launch readiness from consumer-data readiness. Candidate
   concern: the current native PDL experiment may not reproduce llama's edge
   coverage or trigger/wait placement closely enough to test this diagnosis.
2. **Fusion/body deficit.** Candidate: equivalent launch-ahead would not
   materially help tinygrad's kernel mix; parity instead requires deleting
   support programs and intermediate byte traffic through bounded primitive
   fusion plus the remaining Q4 FFN-down body work.

Do not choose one from intuition or from the existing timestamp ledger.
Reconstruct the mechanisms from source, then run the cheapest admissible tests
that distinguish them using the same tinygrad kernels. A claim that cannot be
tested inside this packet remains `unmeasured` and becomes a separately scoped
construction; it does not become the design verdict.

The requested output is a tested causal verdict plus a design recommendation,
not production code.

## 1.1 Measurement-before-claim rule

Every statement in this packet after the locked endpoint arithmetic is a
**candidate explanation**, including statements about useful-body overlap,
edge coverage, wait placement, QMD/CUDA semantic differences, graph-group
effects, one-phase dependencies, opaque primitives, and fusion sufficiency.

Use this order:

```text
name the hypothesis
  -> name the observable that distinguishes it
  -> build or reuse the smallest probe
  -> run control/candidate/control
  -> reconcile topology and endpoint wall
  -> only then label supported or refuted
```

Static source inspection can prove that code has a particular shape. It cannot
by itself prove that the shape causes the 717.505 us wall gap. Endpoint-causal
claims require a wall measurement plus topology showing that the intended
mechanism actually fired.

## 2. Locked endpoint and segment measurements

Same RTX 5090 session, Qwen3-8B-Q4_K_M, d512 decode:

| quantity | tinygrad us | llama us | tiny minus llama us |
| --- | ---: | ---: | ---: |
| unprofiled wall | 4723.214 | 4005.709 | +717.505 |
| profiled device union | 4732.500 | 3892.777 | +839.723 |
| host/profile residual | -9.286 | +112.932 | -122.218 |
| summed kernel interval mass | 4738.496 | 5020.797 | -282.301 |
| interval overlap mass | 5.996 | 1128.020 | -1122.024 |
| device span | 4842.250 | 3901.205 | +941.045 |

The correct interval identity is:

```text
delta_union = delta_node_sum - delta_overlap
839.723     = -282.301 - (-1122.024)
            = -282.301 + 1122.024
```

Do not call `node_sum` FLOPs, operations, bytes, or useful kernel work. Under
PDL a consumer kernel interval can include time spent waiting at a grid
dependency synchronization. `node_sum` is summed kernel residence-time mass;
the union identity is valid, but it does not say how much useful work executed
concurrently.

Matched inter-anchor exposure over 36 layers:

| segment | tinygrad us | llama us | delta us |
| --- | ---: | ---: | ---: |
| S0, prior down to Q | 181.000 | 31.299 | +149.701 |
| S1, Q to O | 1152.250 | 517.916 | **+634.334** |
| S2, O to gate/up | 186.000 | 32.896 | +153.104 |
| S3, gate/up to down | 0.000 | 15.937 | -15.937 |
| S4, down to next Q | 170.250 | 30.884 | +139.366 |

S1 is the location to explain. Tinygrad exposes 1042.500 us of classified S1
kernel mass plus the token's 109.750 us of dead device time. Llama fits
844.050 us of S1 interval mass into 517.916 us of exposure.

Current tinygrad already folds the O residual, fused gate/up GLU, and down
residual. S3 is therefore a tinygrad win. Do not attribute the current wall gap
to llama having those same folds.

The one clear current inside-anchor deficit is Q4 FFN-down:

```text
18 tinygrad calls, total 376.192 us, mean ~20.900 us
18 llama-matched calls at 19.232 us, total 346.176 us
matched body/interval ceiling: ~30.016 us per token

Floor note (2026-08-21): 19.232 us/node is the corrected FFN-down floor from
``nv-gemv-core-deficit-correction-20260813.md``. The earlier 11.776 us was
attention-O's per-node value, not FFN-down's.
```

This row is on the mandatory anchor spine, but aggregate tinygrad anchor union
is still 54.867 us below llama because tinygrad wins other projection rows.

## 3. First-layer timeline that must be explained

The canonical tinygrad first layer is serialized between Q and O:

```text
[20.000, 28.750] Q GEMV
[28.750, 32.500] q norm
[32.500, 35.250] q rope
[36.000, 41.000] K GEMV
[41.000, 45.500] V GEMV
[45.500, 48.000] k norm
[48.000, 49.750] store/cast
[49.750, 53.750] no kernel active
[53.750, 62.000] flash score
[62.000, 65.000] flash combine
[65.000, 74.500] O GEMV
```

The canonical llama first layer launches ahead on one CUDA stream:

```text
[ 0.000,  3.423] attention norm
[ 0.671,  4.223] Q quant
[ 1.215, 11.071] Q MMVQ
[11.231, 12.895] q norm
[11.839, 13.727] q rope
[12.383, 14.495] K quant
[12.863, 18.207] K MMVQ
[18.303, 19.743] V quant
[18.783, 22.911] V MMVQ
[22.943, 24.479] k norm
[23.359, 25.215] k rope
[23.711, 25.695] k store
[25.567, 30.079] flash score
[26.111, 31.167] flash combine
[26.687, 31.775] O quant
[27.487, 38.943] O MMVQ
```

The review must **measure or leave unmeasured** three meanings currently
blurred together:

- a kernel grid has started;
- the kernel has crossed its PDL dependency wait;
- the useful body is executing.

For example, the llama O MMVQ starts before flash combine and O quant finish.
Its 11.456 us timestamp duration can therefore include dependency-wait time.
It is not automatically an isolated 11.456 us MMVQ body.

## 4. Source mechanisms to audit

### 4.1 llama.cpp

Pinned source: `/home/ubuntu/env/llama.cpp` at `ac4cddeb0`.

Read:

- `ggml/src/ggml-cuda/common.cuh:123-132`:
  `cudaGridDependencySynchronize()` and
  `cudaTriggerProgrammaticLaunchCompletion()`.
- `ggml/src/ggml-cuda/common.cuh:1528-1645`: PDL launch attribute and
  `cudaLaunchKernelEx` path.
- every `ggml_cuda_pdl_lc()` / `ggml_cuda_pdl_sync()` call in norm, rope,
  quantize, MMVQ, flash attention, copy, set/get rows, and elementwise kernels.

Record where each trigger and wait occurs relative to pointer arithmetic,
index calculation, global loads, and output stores. Do not summarize all call
sites as "kernel start" or "kernel end" when their placement differs.

### 4.2 tinygrad native NV PDL

Read:

- `tinygrad/renderer/cuda.py:11-36`: name-filtered source injection. A marked
  consumer waits at the first instruction; a marked producer triggers at the
  first or last instruction.
- `tinygrad/runtime/ops_nv.py:24-48`: QMD latch pairing.
- `tinygrad/runtime/ops_nv.py:160-185`: dependent-QMD chaining and pair arm.
- `tinygrad/runtime/graph/hcq.py:360-430`: queue placement and binary resource
  dependency handling.
- `tinygrad/engine/jit.py`: five replay-group construction and admission.

The 2026-08-20 real-route PDL arm used:

```text
NV_PDL_PRODUCER_PROGRAMS=prefix:q4k_,prefix:q6k_
NV_PDL_CONSUMER_PROGRAMS=prefix:reduce_output_rmsnorm,prefix:E_,prefix:r_,prefix:flash_,prefix:rmsnorm_q8_1_llama_provider
NV_PDL_TRIGGER_POSITION=end
```

It measured 11.641 us slower with two queues and 8.201 us slower with one.
However, the PDL profile could not be reconciled into one token, so the run did
not record:

- the number and identities of actually armed pairs;
- support-to-support PDL coverage;
- trigger, wait, and useful-body timestamps;
- whether graph-group boundaries broke the launch-ahead chain.

Do not treat the endpoint result as proof that every llama-equivalent PDL edge
was tested.

### 4.3 Opaque primitive and fusion boundary

Read:

- `tinygrad/llm/decode_kernels.py:170` and the complete
  `Q4KGEMVEpilogue` implementation;
- `tinygrad/llm/model.py:640-736`, the decode block's manually admitted norm,
  residual, gate/up, and down folds;
- `tinygrad/llm/model.py:755-1004`, the Q/K/V, norm, rope, cache, flash, and O
  chain;
- `UOp.custom_kernel` and the scheduling/realization boundary it creates.

Test this architectural hypothesis:

> The fast custom GEMV is opaque to generic fusion and exposes only a complete
> output tensor. Therefore the compiler cannot place a bounded epilogue or
> express partial readiness without a manually enumerated kernel variant.

Say exactly which part is statically true in current source, then measure
whether it is endpoint-causal. Identify the smallest interface change that
would alter it only after the causal test.

## 5. Hypotheses to challenge

Return a verdict of `supported`, `refuted`, or `unmeasured` for each. A
`supported` or `refuted` endpoint-causal verdict must cite its required test
from the table below; source inspection alone permits only a `source-confirmed,
endpoint-unmeasured` sub-verdict.

1. **H1: residence-time accounting.** Most llama overlap mass is overlapping
   grid lifetime and dependency wait, not simultaneous useful memory traffic.
2. **H2: sparse PDL mismatch.** The tinygrad PDL A/B tested only selected
   anchor-to-support pairs and could not reproduce llama's support-to-support
   launch-ahead chain.
3. **H3: wait-placement mismatch.** Prepending `griddepcontrol.wait` as the
   first consumer instruction leaves less safe prologue to hide than llama's
   per-kernel synchronization placement.
4. **H4: trigger/QMD mismatch.** The native latch's last-CTA/final-wave
   behavior differs materially from the CUDA PDL behavior used by llama.
5. **H5: graph-granularity mismatch.** Five tinygrad replay groups prevent a
   continuous launch-ahead pipeline that llama's single 762-node graph keeps.
6. **H6: one-phase scheduler.** HCQ models only execution/data readiness, not
   separate launch readiness and data readiness.
7. **H7: opaque primitive boundary.** Custom GEMV opacity is why each fusion
   requires a new decode-specific epilogue spelling.
8. **H8: fusion-only sufficiency.** The currently legal residual and reduction
   folds plus Q4 FFN-down can close the 717.505 us wall gap after alternate-path
   takeover and interference are recomputed.

H8 may not be answered by adding raw ceilings. Produce a recomputed or bounded
composition and show what remains.

| hypothesis | minimum test before a causal verdict |
| --- | --- |
| H1 | instrument or otherwise isolate consumer wait-exit from kernel start; timestamp overlap alone is insufficient |
| H2 | census all eligible and actually armed edges for one token on both routes, including support-to-support edges |
| H3 | same-kernel wait-at-entry versus wait-at-first-dependent-access A/B with trigger and wait timestamps |
| H4 | matched CUDA-PDL versus native-QMD release-timing probe with the same grid geometry |
| H5 | same PDL construction with one continuous graph versus the current five replay groups, or a measured upper bound if construction is unavailable |
| H6 | source audit of the edge model plus an execution probe proving that split launch/data readiness changes the schedule |
| H7 | source audit of the primitive boundary plus one view-preserving/fused probe showing the boundary is what creates or removes the program |
| H8 | endpoint A/Bs or a dependency-DAG simulation with alternate-path takeover, followed by endpoint validation; never raw-ceiling addition |

## 6. Required design comparison

Compare these two candidate directions.

### Direction A: first-class split-phase dependencies

The intended semantic shape is:

```text
producer launch
  -> launch-complete event permits consumer grid launch
  -> consumer may execute dependency-independent prologue
  -> consumer waits at first dependent access
  -> producer data-complete releases dependent access
```

Provide minimal pseudocode for:

- the program/kernel metadata;
- the graph edge representation;
- HCQ/QMD lowering;
- codegen placement of launch completion and data wait;
- correctness rules for aliasing, memory reuse, and multiple consumers;
- the closed-default fallback on devices without this capability.

The design must be scheduler-owned and edge-aware. Environment name prefixes
are acceptable probes, not the proposed production interface.

### Direction B: bounded decode fusion

Provide the smallest fusion sequence that can legally reduce the measured S1
and other support exposure while preserving byte-identical tokens. For every
fusion, name:

- producer and consumer;
- intermediate bytes and launch boundaries removed;
- required reduction or cross-CTA communication;
- why the existing primitive geometry can or cannot absorb it;
- zero-cost ceiling, legal ceiling, and expected endpoint conversion;
- whether it generalizes or adds another model/shape-specific epilogue.

Include the Q4 FFN-down body as a separate on-spine kernel lever rather than
calling it S1 fusion.

## 7. Decisive experiments: design, gate, then run

Run the existing read-only/probe-only discriminators first. Then build and run
the cheapest probe that uses the **same tinygrad kernels and dependency DAG**
to compare:

1. serial/full-completion dependencies;
2. CUDA's real programmatic stream serialization across every eligible edge;
3. current native QMD PDL;
4. a native candidate with equivalent edge coverage and trigger/wait placement,
   if source audit says it is expressible.

Required observations:

- exact eligible and armed edge census;
- graph-group and queue assignment;
- per edge: producer start, launch-complete trigger, consumer grid start,
  consumer data-wait exit, producer end, consumer end;
- kernel union, residence-time mass, useful-body estimate, dead device time,
  and endpoint wall;
- byte-identical token SHA;
- same-session control/candidate/control brackets under the existing GPU lock.

Define belief-flip gates before seeing results. At minimum:

- a large CUDA-PDL recovery using tinygrad kernels supports Direction A;
- CUDA PDL flat or negative with faithful edge coverage supports Direction B;
- native negative while CUDA positive isolates the native lowering/runtime
  semantics rather than disproving PDL;
- both positive but native smaller isolates pair coverage, wait placement, or
  graph grouping.

Rank this against the cheaper Q4 FFN-down and output-reduction experiments.

### 7.1 Staged execution and hard gates

**Phase A -- no GPU.** Re-run the ledger generator, verify the arithmetic,
audit the source call sites, and produce the expected eligible-edge census from
the two static DAGs. If the ledger does not reproduce exactly, stop; later GPU
work would be testing an unstable premise.

**Phase B -- current construction census.** Add probe-only instrumentation that
records a graph invocation id, armed PDL pairs, trigger points, consumer starts,
and wait exits where expressible. Run the existing control and current native
PDL arms. This phase answers only what the current construction actually tests;
it may not claim CUDA equivalence.

**Phase C -- semantic discriminator.** Run the smallest same-grid CUDA-PDL vs
native-QMD probe. It must use identical producer/consumer bodies, grid geometry,
and dependency placement. If the existing interfaces cannot express equivalent
coverage or wait placement without changing production runtime behavior, stop
and put a proposed follow-up construction scope in the output report for the
coordinator to split into a new packet. Do not create that packet here, and do
not substitute a synthetic result from a different grid shape.

**Phase D -- endpoint confirmation.** Only when Phase C establishes an
equivalent mechanism may a same-token decode bracket support or refute a
split-phase endpoint claim. All endpoint rows use fresh processes,
control/candidate/control, byte-identical token SHA, and
`flock /tmp/gpu-bench.lock`.

No phase may be skipped because an older record appears to contain the answer.

### 7.2 Allowed paths

Probe and evidence work is restricted to:

- this task packet;
- `docs/task_workflow/output/nv-split-phase-pdl-causal-design-review-20260820.md`;
- `docs/task_workflow/evidence/nv-split-phase-pdl-causal-design-review-20260820/**`;
- `extra/llm_research/decode/**` for new or extended measurement tooling.

Production paths under `tinygrad/**` and model paths are read-only. If a test
requires changing them, the reviewer must stop and write a separate scoped
construction proposal naming the exact paths, default behavior, correctness
pins, and rollback.

## 8. Required output

Write:

`docs/task_workflow/output/nv-split-phase-pdl-causal-design-review-20260820.md`

The report must contain:

1. a one-paragraph causal answer containing only claims that passed their
   named test, with the rest explicitly `unmeasured`;
2. corrections to the current brief, including the sign error and the meaning
   of `node_sum` under PDL;
3. a first-layer launch/wait/useful-body reconstruction for both systems;
4. H1-H8 verdicts with exact source/evidence citations;
5. an edge-coverage comparison between llama and the tested tinygrad PDL arm;
6. the exact commands, controls, raw evidence paths, and results for every test
   actually run;
7. Direction A and Direction B designs in pseudocode, conditional on those
   results;
8. the three cheapest remaining decisive experiments, expected outcomes, and
   belief-flip gates;
9. a ranked next-action list separating immediate endpoint work from generic
   architecture work;
10. an explicit answer to: **Can current tinygrad match llama through existing
   abstractions, or only through more bespoke kernels? Why?**

## 9. Acceptance criteria

The review is complete only if:

- the interval identity closes with the correct signs;
- kernel residence time is not mislabeled as operations, bytes, or useful work;
- llama's existing GLU/residual folds are not claimed as a current advantage
  where tinygrad already matches or wins;
- PDL is not declared falsified without proving equivalent edge coverage and
  trigger/wait placement;
- every major claim is labeled observed, inferred, or unmeasured;
- estimated byte rooflines are distinguished from hardware DRAM counters;
- support launch latency is not asserted without an explicit measurement;
- fusion ceilings are recomputed with alternate-path takeover and are not
  simply added;
- the proposed design identifies ownership across compiler, scheduler,
  runtime, and primitive boundaries;
- the final recommendation is falsifiable by the proposed experiment;
- no H1-H8 endpoint verdict is stronger than the test actually run;
- failed instrumentation or unavailable equivalent construction is reported as
  `unmeasured`, never converted into support for the competing hypothesis;
- all GPU rows use the lock, fresh-process brackets, token SHA, and raw evidence
  retained under the allowed evidence directory.

## 10. Grounding artifacts

Read these first:

- `docs/task_workflow/output/nv-ledger-roofline-pseudocode-brief-20260820.md`
- `docs/task_workflow/output/nv-weighted-inter-anchor-causal-gap-result-20260820.md`
- `docs/task_workflow/output/nv-weighted-inter-anchor-ledger-20260820.json`
- `docs/task_workflow/output/nv-weighted-llama-real-edge-dag-20260820.json`
- `docs/task_workflow/output/nv-rmsnorm-phaseB-control-20260820.json`
- `docs/task_workflow/output/nv-pdl-queue-theories-test-20260820.md`
- `docs/task_workflow/input/nv-pdl-substrate-verdict-20260817.md`
- `docs/task_workflow/input/nv-llama-pdl-launch-hiding-trace-record-20260816.md`
- `docs/task_workflow/input/nv-overlap-planner-serialization-root-cause-20260815.md`
- `docs/task_workflow/input/nv-route-to-parity-theories-20260819.md`

Analysis tooling to audit, not blindly trust:

- `extra/llm_research/decode/nv_inter_anchor_analysis.py`
- `extra/llm_research/decode/llama_weighted_dag.py`
- `extra/llm_research/decode/nv_rmsnorm_current_head_topology.py`
- `extra/llm_research/decode/nv_pdl_trigger_probe.py`

## 11. Bans and hard stops

- No GPU use before Phase A reproduces and Phase B instrumentation identifies
  what the current arm actually executes.
- GPU use is limited to the Phase B-D discriminators, serialized by
  `flock /tmp/gpu-bench.lock`; no open-ended sweep.
- No runtime, model, renderer, scheduler, or policy change. Probe-only changes
  stay under `extra/llm_research/decode/**`.
- No promotion or route-default recommendation based only on simulated
  ceilings.
- No stale pre-`6570abc02` per-shape row may replace the current-head rows.
- No inference that a timestamp overlap interval is simultaneous useful work
  without locating the dependency wait.
- No recommendation to add queues, optimize K/V, or repeat copy-free RMSNorm;
  those levers are already measured and bounded.
- If the evidence cannot distinguish CUDA PDL semantics from native-QMD
  semantics, say `unmeasured` and make that the first experiment.

## 12. One-line job

Test whether tinygrad's remaining decode gap is fundamentally a missing
split-phase launch/data dependency abstraction or a fusion/body problem; make
no causal claim until its discriminator passes, and leave anything the allowed
construction cannot test explicitly `unmeasured`.
