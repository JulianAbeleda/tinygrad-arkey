# NV decode overlap - Route B3 exhaustive DAG-attribution and wall execution scope

Date: 2026-08-04

Status: executable future scope; the llama binary-bridge scratchpad preflight
in Section 14 is complete, while B3 execution is not started. Phase-gated
measurement/tooling scope for the CUDA Route B decode DAG after the B2 external
review. Branch: `nvidia-bringup-20260731`. Drafting boundary: `21783f988`.

This scope authorizes only the analysis tooling, hermetic tests, lock-held GPU
captures, closed-process planner counterfactuals, correctness work, and wall
measurements named below. It does not authorize a native NV runtime change,
HCQGraph change, `ops_nv.py` change, default route flip, promotion to another
branch, or parity claim. A production memory-planner policy requires a later
scope after the gates in this document pass.

---

## 0. One-line job

Determine whether the JIT memory plan removes valuable logical independence
from the real CUDA decode DAG, identify the minimum-memory counterfactual if it
does, and prove at wall whether that recovery first repays the CUDA route tax
before making any route-value or parity claim.

## 1. Authority and supersession

Authority order, newest first:

1. this exhaustive execution scope;
2. `nv-decode-overlap-route-b3-external-review-amendment-20260804.md`;
3. `nv-decode-overlap-route-b3-external-review-scope-20260804.md` as the
   pre-review brief;
4. `nv-decode-overlap-route-b2-multi-stream-lowerer-measurement-record-
   20260804.md` as B2 evidence;
5. `nv-decode-overlap-route-b-implementation-scope-20260804.md` for completed
   B1/B2 history only;
6. B0/B1 measurement records for their observed rows.

This scope supersedes the old B3 items and gates in the original Route B
implementation scope. In particular, it withdraws:

- the native-NV `608.8 us` ceiling as CUDA authority;
- historical `157.93 tok/s` as the wall A/B denominator;
- `>=90% strict-chain edges` as a DAG decision rule;
- `NO_MEMORY_PLANNER=1` as an overlap upper bound;
- the route-level claim that capture is redundant;
- the route-level claim that real CUDA decode is planner-chained.

## 2. Fixed measured state

The following are anchors, not future same-session ranking rows:

| quantity | current evidence | class |
| --- | ---: | --- |
| llama d512 | 246.32 tok/s, 4.0598 ms/token | OBSERVED, historical session |
| native NV d512 | 177.72 tok/s, 5.6268 ms/token | OBSERVED, historical session |
| CUDA B0.2 d512 | 157.93 tok/s, 6.3319 ms/token | OBSERVED, historical session |
| CUDA B2 S=1 d512 | 159.72 tok/s | OBSERVED, later session |
| CUDA graph shape | 1021 kernels, 6 groups: 32/64/128/256/512/29 | OBSERVED |
| native NV graph shape | 948 kernels, 5 groups: 32/64/128/256/468 | OBSERVED |
| CUDA replay | 5134.2 us node-sum, 5363.8 us span, about 230 us launch gaps | OBSERVED |
| B2 CH=2 programmatic probe | about 18.6% overlap | OBSERVED, synthetic |
| B2 CH=6 programmatic probe | 51-59% / 24-32% overlap | OBSERVED, synthetic |
| explicit capture probe | no improvement over programmatic control | OBSERVED, synthetic |
| planner cause on real CUDA decode | not established | UNKNOWN |
| CUDA correctness relative to NV/reference | not established | UNKNOWN |
| compiled llama MMVF reuse | direct binary launch over tinygrad buffers, max abs error `4.7684e-7` | OBSERVED, diagnostic scratchpad |
| compiled llama MMVF CUDA-graph capture | one captured node, 8 correct replays, max abs error `4.7684e-7` | OBSERVED, diagnostic scratchpad |
| compiled llama Q4_K/Q6_K reuse | device entry symbols exist but are local inside embedded cubins | OBSERVED ABI inventory; live launch UNKNOWN |

Route arithmetic at the historical anchors:

```text
CUDA route tax vs native NV  = 6.3319 - 5.6268 = 0.7051 ms/token
native-NV gap vs llama       = 5.6268 - 4.0598 = 1.5670 ms/token
CUDA gap vs llama            = 6.3319 - 4.0598 = 2.2721 ms/token
```

The old native-NV no-contention ceiling of 0.6088 ms is smaller than the
historical CUDA route tax. It is retained only as evidence about the old NV
DAG, never as a CUDA target or gate.

## 3. Questions this scope must settle

Q1. Does the logical CUDA decode call chain contain duration-weighted
parallelism before physical arena reuse?

Q2. Does `memory_plan_rewrite` add WAR/WAW dependencies that materially
increase the CUDA critical path?

Q3. Which exact logical buffers, physical arena ranges, and producer/consumer
pairs create any material planner-added edges?

Q4. What CUDA-specific no-contention ceiling remains after real cross-group
edges and the actual six-group split are included?

Q5. Can a minimal selective-unaliased counterfactual retain the call chain,
correctness, target depths, and acceptable peak memory?

Q6. Does recovered DAG independence appear as realized CUDA concurrency at
S=1, and does it reduce end-to-end wall time after bandwidth contention?

Q7. Does the wall gain repay the CUDA route tax relative to same-session
native NV?

Q8. If planner attribution fails, which branch becomes authoritative:
semantic chain, scheduler/resource incompatibility, correctness failure, or
kernel/boundary gap?

Q9. Can exact llama Q4_K/Q6_K MMV kernels serve as an external oracle inside
the same tinygrad-controlled CUDA graph so kernel quality can be held constant
while planner/scheduler behavior changes?

Q10. With the physical DAG fixed, how much of the wall gap survives when only
the dominant MMV kernel implementation changes?

## 4. Evidence classes and terminology

- **OBSERVED:** directly emitted by a named capture, test, counter, or wall
  run with raw artifacts.
- **DERIVED:** deterministic arithmetic over OBSERVED inputs. State the formula.
- **INFERRED:** mechanism interpretation consistent with observations but not
  isolated.
- **UNKNOWN:** not measured or confounded.
- **CONTROL:** reference arm in the same session and route.
- **DIAGNOSTIC:** may explain a result but cannot qualify a route.
- **QUALIFIED:** passed every prerequisite gate named for the claim.

Terms:

- **logical DAG:** buffer/data dependencies before planner arena reuse.
- **physical DAG:** dependencies after planner placement and physical range
  aliasing, before/at graph construction.
- **planner-added edge:** a physical WAR/WAW dependency absent from the aligned
  logical DAG and attributable to reused physical bytes.
- **semantic edge:** dependency already present in the logical DAG.
- **route tax:** same-session CUDA default S=1 wall minus native NV wall.
- **CUDA legal ceiling:** logical-vs-physical duration-weighted schedule delta
  computed from the actual CUDA graph and labeled no-contention.
- **wall conversion:** measured candidate wall improvement over the same-session
  CUDA default S=1 control.

No simulation result is described as wall speedup. No CUPTI schedule is
described as dependency ground truth. No deterministic output is described as
correct without reference checks.

## 5. Global invariants

1. Every compared CUDA arm uses the same commit, model file/hash, tokenizer,
   prompt, depth, quantization route, driver, CUDA toolkit, and fixed-depth
   harness protocol.
2. Call identity and ordered call count must match before an edge delta is
   interpreted.
3. An UNKNOWN dependency node makes the corresponding DAG verdict non-decisive.
4. The native NV and CUDA routes keep separate pins and separate artifacts.
5. A new CUDA pin is admitted only after per-class reference checks; repetition
   alone is not correctness.
6. All live GPU commands are sequential and protected by `/tmp/gpu-bench.lock`.
7. Every process records resolved environment, argv, commit, dirty-path list,
   device/driver facts, model hash/size, start/end times, and exit status.
8. `/tmp` traces are session artifacts. Any load-bearing summary is copied into
   an anchored JSON/Markdown deliverable with hashes.
9. S=1 remains the default programmatic CUDA graph. S=2/3 is diagnostic unless
   it wins on the identical real DAG.
10. No candidate proceeds to wall ranking before correctness passes.
11. An external llama kernel is an oracle, not an implementation candidate;
    its evidence class and artifact hash travel with every row.

## 6. Authorized paths

Analysis/tooling paths authorized before G-B3-D:

- `extra/llm_research/decode/full_token_dag_capture.py`;
- a new narrowly named Route B3 DAG/planner attribution tool under
  `extra/llm_research/decode/` if separation is cleaner;
- `test/unit/test_full_token_dag_capture.py`;
- new CPU-only tests for planner-DAG attribution;
- `scratchpad/llama_cuda_binary_kernel_probe.py` for the diagnostic external
  kernel bridge and its gated Q4_K/Q6_K extension;
- docs and anchored JSON records named by this scope.

Conditional closed-process candidate paths after G-B3-D:

- an analysis-only selector/context wrapper under `extra/llm_research/decode/`;
- hermetic tests for stable buffer signatures and fail-closed selection.

`tinygrad/schedule/memory.py` may be changed before G-B3-D only to expose a
no-behavior-change manifest collector if the existing private collector cannot
provide placement/range evidence. Requirements:

- zero default-path output or allocation change when no collector is installed;
- no env-controlled production policy;
- exact unit coverage for collector absent/present behavior;
- the manifest is observational, not a planner decision hook.

Not authorized by this scope:

- `tinygrad/runtime/ops_nv.py`;
- native NV graph or queue code;
- HCQGraph behavior;
- default memory-planner policy;
- default CUDA graph stream count;
- model/quant route promotion;
- `dev`, `exp`, or `master` changes.

## 7. Phase B3.0 - preflight and reproducibility

### 7.1 Worktree and authority inventory

Record:

- branch and exact commit;
- `git status --short` without modifying unrelated paths;
- amendment and this scope hashes;
- presence and hashes of the B0/B1/B2 records;
- installed `nsys`, SQLite exporter, CUDA toolkit, Python, and compiler paths;
- model path, SHA-256, size, GGUF identity, and tokenizer identity;
- RTX 5090 identity, driver 595.84 or the newly observed driver if changed,
  VRAM total/free, and absence of concurrent work.

If the driver differs from 595.84, prior driver-scheduler findings remain
historical only. Repeat the smallest CH=2 S=1 control before applying them.

### 7.2 Hermetic preflight

Required before a GPU capture:

```bash
python3 -m pytest -q test/unit/test_full_token_dag_capture.py
python3 -m pytest -q test/unit/test_cuda_graph_multi_stream_schedule.py
```

Add the new attribution tests in Phase B3.1 before running them live.

### 7.3 CUDA route reproduction

Under the lock, reproduce the fixed d512 CUDA S=1 route:

```bash
DEV=CUDA CUDA_GRAPH_STREAMS=1 QK_NMEAS=20 QK_REPS=3 QK_CKPTS=512 \
python3 extra/llm_research/decode/decode_runtime_overhead.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
```

The implementation record must use the repository's actual lock wrapper or an
explicit `flock /tmp/gpu-bench.lock`; the resolved command is recorded. The
above command is the payload, not permission to bypass the lock.

Capture:

- W and D raw rows;
- first token, generated-token SHA, and any decode/logit SHA;
- graph group counts and sizes;
- kernel count;
- peak/free VRAM before/after;
- three-run direction and variance.

### Gate G-B3-0 - preflight

PASS requires:

- all hermetic tests green;
- fixed d512 CUDA S=1 completes 3/3 deterministically;
- route structure is either 1021 / six groups as expected or explicitly
  re-baselined with a cause;
- no unexplained model/driver/tooling mismatch;
- no user path changed.

Failure stops all later phases. A changed route is not automatically a failure,
but it requires a new baseline row and makes historical timing arithmetic
non-authoritative.

## 8. Phase B3.1 - aligned logical/physical DAG tooling

### 8.1 Required capture arms

The tooling must produce two aligned CUDA views:

1. **DEFAULT_PHYSICAL:** normal `memory_plan_rewrite` and S=1 graph grouping.
2. **PLANNER_FREE_LOGICAL:** planner reuse disabled for dependency observation,
   with no claim that its memory/runtime behavior is production-valid.

A single-process dual snapshot before and after planning is preferred if it can
resolve the same call/resource identities. A controlled two-process capture is
acceptable only if the ordered call-signature contract below passes exactly.

### 8.2 Stable call identity

For each call, emit a stable signature from fields available in both arms:

- ordered call index;
- operation kind;
- program/kernel identity or stable AST/program hash;
- launch dimensions and shared memory when compiled;
- ordered logical output/input sizes, dtypes, and devices;
- semantic metadata where present;
- graph group and position within group.

Do not include transient pointers in the stable signature. Preserve pointers
separately for physical-range evidence.

Alignment PASS requires identical ordered stable signatures. If planner state
changes compile identity, group sizes, or call count, emit `ALIGNMENT_CONFOUNDED`
and stop edge subtraction.

### 8.3 Logical buffer identity and planner manifest

For every plannable logical buffer, emit:

- stable logical-buffer signature;
- producer call and all consumers;
- dtype, logical elements, byte size, device;
- first/last call lifetime in the linear order;
- held status and reason;
- physical arena identity, offset, aligned size, and byte interval in the
  default arm;
- all other logical buffers that reuse an overlapping physical interval.

If the existing `_memory_manifest_collectors` seam is extended, the record must
prove the default output linear, arena sizes, placements, and compiled call
signatures are byte-identical with the collector absent versus present.

### 8.4 Dependency attribution

Emit every edge as:

```json
{
  "from": 12,
  "to": 13,
  "kind": "RAW|WAR|WAW",
  "source": "SEMANTIC|PLANNER_ALIAS|UNKNOWN",
  "logical_buffer_ids": ["..."],
  "arena": "...",
  "range": [0, 4096],
  "crosses_graph_group": false
}
```

Rules:

- an edge present in the logical arm is SEMANTIC even if physical aliasing also
  exists;
- a physical WAR/WAW edge absent logically and tied to overlapping arena bytes
  is PLANNER_ALIAS;
- any failure to resolve call/buffer/range identity is UNKNOWN;
- absent edges are never assumed independent when either endpoint is UNKNOWN;
- edge kinds are range-aware and use the canonical `DepsTracker` semantics.

### 8.5 DAG metrics

For both arms and for the attributed delta, compute:

- node count and edge counts by kind/source;
- serialized node-sum using attached measured CUDA durations;
- duration-weighted critical path;
- critical-path saving in microseconds and percent;
- ready width as a time series/histogram;
- deterministic two- and three-resource list schedules;
- per-group and cross-group results;
- top planner-added edges ranked by increase in critical-path time;
- top logical buffers ranked by recoverable microseconds and bytes;
- unknown node/edge counts;
- a resource-pair table: GEMV/GEMV, GEMV/elementwise, flash/GEMV,
  elementwise/elementwise, copy/compute.

The schedules are no-contention arithmetic. State that durations may expand
under overlap and that cross-group scheduling is legal only where the full DAG
contains the required edges.

### 8.6 Hermetic tests

Tests must cover at least:

1. two logical independent chains that become physically chained through reuse;
2. semantic RAW chain unchanged by planning;
3. planner-added WAR and WAW classification;
4. partially overlapping byte ranges;
5. adjacent non-overlapping byte ranges;
6. stable signature pointer independence;
7. call-order/signature mismatch fails closed;
8. UNKNOWN dependency propagation;
9. cross-group edge preservation;
10. duration-weighted critical path where many small nodes do not outweigh one
    expensive branch;
11. planner collector absent/present default equivalence, if changed;
12. JSON schema validation and deterministic output/hash.

No GPU is required for these tests.

### Deliverables B3.1

- tooling source and tests;
- schema fixture with expected edge attribution;
- tooling implementation record;
- exact commands and green test output;
- no real-decode conclusion yet.

## 9. Phase B3.2 - live CUDA DAG capture

### 9.1 Session protocol

One lock-held session captures DEFAULT_PHYSICAL and, only if memory-safe,
PLANNER_FREE_LOGICAL for d512. The planner-free arm is marked DIAGNOSTIC.

Required controls:

- free VRAM checked before model load;
- no concurrent GPU process;
- identical model/route/harness settings;
- warm capture separated from measured replay;
- ordered stable call signatures compared before dependency subtraction;
- graph grouping and kernel count recorded;
- output pins recorded even though full correctness is gated later.

If planner-free d512 does not fit, do not reduce the model, depth, or context to
manufacture a PASS. Record `PLANNER_FREE_OOM`, retain the default placement
manifest, and run the offline counterfactual using the logical resource view.

### 9.2 Timing attachment

Attach durations from the same DEFAULT_PHYSICAL route/session. CUPTI is not
required to decide edge existence but is used for node durations if the graph
profile payload is insufficient.

Duration matching requires:

- graph group size sequence match;
- stable kernel/call identity match;
- replay-cycle attribution;
- no silent occurrence-order drift;
- unmatched nodes reported, not assigned zero without an UNKNOWN label.

### 9.3 CUDA-specific ceiling

Compute these DERIVED quantities:

```text
planner_delta_cp_us = physical_cp_us - logical_cp_us
cuda_route_tax_us   = cuda_default_wall_us - native_nv_wall_us
remaining_gap_us    = native_nv_wall_us - llama_wall_us
```

Also report logical two-/three-resource schedule savings, but do not combine
them with wall or assume the driver realizes them.

Scale classifications based on the fresh CUDA control wall:

- `<5%` of CUDA wall: `NOT_MECHANISM_SCALE`; close planner implementation;
- `>=5%` but `< CUDA route tax`: `MECHANISM_SCALE_ONLY`;
- `>= CUDA route tax`: `ROUTE_TAX_SCALE`;
- `>= CUDA route tax + remaining native-NV gap`: `PARITY_SCALE_THEORETICAL`,
  still no wall/parity claim.

### Gate G-B3-D - decisive DAG attribution

PASS requires:

1. G-B3-0 PASS;
2. identical ordered call signatures for compared arms;
3. zero UNKNOWN nodes on every critical-path-affecting region;
4. logical and physical CUDA DAG summaries published;
5. material physical-minus-logical edges attributed to exact planner arena
   ranges, or an explicit `PLANNER_NOT_ROOT_CAUSE` result;
6. CUDA-specific ceiling and route-tax comparison published;
7. planner-free memory delta and fit status recorded.

Verdict branches:

- `SEMANTIC_CHAIN`: logical DAG has <5% potential; stop planner/overlap work.
- `PLANNER_NOT_ROOT_CAUSE`: physical DAG retains the logical parallelism; go to
  the scheduler/resource diagnostic branch only.
- `PLANNER_EFFECT_NOT_SCALE`: planner changes the DAG but <5% wall-equivalent;
  bank evidence and stop.
- `PLANNER_CANDIDATE`: planner-added attributed edges are >=5% scale; continue.
- `ATTRIBUTION_CONFOUNDED`: alignment/UNKNOWN failure; fix tooling only.

## 10. Phase B3.3 - correctness census

This phase may run in parallel conceptually with offline DAG analysis, but its
gate must pass before a candidate wall ranking.

### 10.1 Per-class checks

Compare CUDA kernels against established CPU/high-precision references for:

- q4k GEMV shapes used by decode;
- q6k partial/final paths;
- flash score/PV/combine at d512 and the target deeper shapes;
- RMSNorm and residual operations;
- RoPE and scatter/KV store/update;
- quantize/dequantize boundaries;
- vocabulary projection and token selection;
- any CUDA-only fallback absent from the 948-kernel NV chain.

Use existing declared tolerances where they exist. New tolerances require a
numeric justification and must not be chosen after observing the candidate.

### 10.2 Chain-difference census

Produce an aligned semantic class ledger for CUDA 1021 versus native NV 948:

- count by class;
- fused versus unfused replacements;
- output dtype and accumulator dtype;
- materialization/copy boundaries;
- which extra CUDA calls are required versus accidental fallback;
- first layer/class where reference error diverges materially.

### 10.3 End-to-end pins

The existing repeating CUDA six-token loop is determinism evidence, not a
correctness authority. A CUDA route pin may be recorded only after:

- every material kernel class passes its reference bounds;
- fixed-depth logits/token output is understood relative to the NV/reference
  route;
- three independent fixed-depth runs agree;
- any token divergence is explained by bounded numeric differences rather than
  a broken kernel or missing operation.

### Gate G-B3-C - correctness

PASS requires per-class reference checks, an explained route-chain difference,
and deterministic end-to-end pins. A numeric bug routes to a separate fix
scope. Do not optimize or re-pin around the bug under this document.

## 11. Phase B3.4 - offline selective-unalias candidate search

Gate: G-B3-D verdict is `PLANNER_CANDIDATE`.

### 11.1 Candidate construction

Use only attributed planner-added critical-path edges. Candidate actions are
sets of stable logical-buffer signatures to exclude from arena reuse.

Search must include:

- every useful single-buffer exclusion;
- greedy recoverable-us-per-byte additions;
- critical-path edge-cover candidates;
- a minimum-byte candidate reaching 5% theoretical recovery;
- a maximum-recovery candidate within the observed safe d4096 memory budget;
- a route-tax-target candidate if the CUDA ceiling permits one.

Do not search arbitrary call schedules, kernel rewrites, fusion, or stream
counts in this phase.

### 11.2 Candidate identity

Each candidate JSON contains:

- schema version;
- source DAG hash and route signature;
- ordered logical-buffer signatures;
- predicted physical edge removals;
- predicted critical path and two-/three-resource spans;
- extra allocated/peak bytes;
- required graph-group/call identity;
- candidate SHA-256.

Selection fails closed if the live route signature or any required buffer
signature differs.

### 11.3 Pareto ledger

Emit all measured/simulated candidates, not only the winner:

| candidate | extra MiB | planner edges removed | predicted CP gain us | predicted class | status |
| --- | ---: | ---: | ---: | --- | --- |

Retain rejected candidates with reasons: insufficient span, excessive memory,
signature ambiguity, call-chain change, or target-depth fit failure.

### Gate G-B3-S - selector

At most three live candidates proceed:

1. minimum bytes reaching the 5% theoretical threshold;
2. best predicted recovery under the safe memory budget;
3. route-tax-target candidate, only if theoretically available.

If no candidate reaches 5% theoretical recovery, verdict
`NO_SELECTIVE_CANDIDATE` closes live planner work.

## 12. Phase B3.5 - analysis-only live candidate mechanism

Gate: G-B3-S PASS and G-B3-C PASS.

The first live mechanism must not change the global planner default. Preferred
implementation is a process-local analysis context/wrapper that:

1. loads one candidate JSON;
2. validates route and source-DAG hashes;
3. matches stable logical-buffer signatures before planning;
4. adds only the selected logical buffers to the existing `held_bufs` set;
5. emits matched/unmatched/ambiguous signature rows;
6. fails closed on anything other than one exact match per requested buffer;
7. leaves normal execution byte-identical when the context is absent.

No production env flag or default policy is authorized. If this cannot be
implemented without a runtime behavior seam, stop and write a separate
closed-default implementation scope rather than expanding this one silently.

### 12.1 Hermetic candidate tests

Tests cover:

- context absent = exact default planner behavior;
- exact stable-signature selection;
- ambiguous/missing/stale signature failure;
- only requested buffers become held;
- route/DAG/candidate hash mismatch failure;
- deterministic candidate artifact;
- no selection leakage across context/process boundaries;
- memory manifest reflects the intended unaliasing.

### 12.2 Live structural verification

For each candidate at d512, before timing claims:

- call signatures and kernel/group counts match the control;
- selected buffers match exactly;
- predicted planner edges disappear and no unexpected critical edges appear;
- peak memory matches the predicted class within explained allocator rounding;
- G-B3-C pins still pass;
- the graph constructs and replays 3/3.

Failure is `CANDIDATE_CONSTRUCTION_FAIL` and does not proceed to wall ranking.

## 13. Phase B3.6 - same-session d512 wall and CUPTI A/B

### 13.1 Required arms

In one sequential lock-held session:

| arm | route | planner | streams | role |
| --- | --- | --- | ---: | --- |
| A | CUDA | default | 1 | ranking control |
| B1-B3 | CUDA | selective candidates | 1 | ranking candidates |
| C | CUDA | planner-free | 1 | diagnostic if memory-safe |
| D | CUDA | selected physical DAG | 2/3 | diagnostic only if S=1 leaves legal branches idle |
| E | native NV | default | native | route-value control |
| F | llama.cpp | reference | reference | parity control |

Do not run D merely because the lowerer exists. It is permitted only when the
candidate physical DAG contains material compatible independence and S=1 fails
to realize it.

### 13.2 Interleaving and repetitions

- warm every arm before recording;
- use an interleaved order such as A-B-A-B and reverse it in the next repetition;
- record at least three fixed-depth measurements per ranking arm;
- retain every raw row and order, not only medians;
- record temperature, clocks if available, free memory, and concurrent-process
  census before/after;
- run profiler traces separately from unprofiled wall rows and label profiler
  perturbation.

### 13.3 Metrics

For unprofiled wall:

- tok/s, ms/token, raw reps, median, min/max, and dispersion;
- correctness pins;
- peak memory and target-depth fit;
- CUDA candidate/control ratio;
- candidate/native-NV ratio;
- candidate/llama ratio.

For CUPTI CUDA arms:

- per-token span and node-sum;
- graph-launch gaps;
- overlap fraction by group and whole token;
- realized stream IDs;
- per-node duration inflation relative to control;
- class-pair overlap;
- available DRAM throughput/counters or an explicit tooling-unavailable label;
- unmatched node count and replay-clustering sensitivity.

The 20 us replay-cluster threshold must be sensitivity-tested against nearby
values. Per-graph attribution is explicit.

### Gate G-B3-M - mechanism wall conversion

PASS requires all of:

- G-B3-D, G-B3-C, G-B3-S, and live structural verification PASS;
- candidate median d512 throughput `>=1.05x` same-session CUDA A;
- directionally positive 3/3 candidate-control comparisons;
- identical qualified correctness;
- graph/call identity retained;
- memory cost and per-node duration inflation reported.

Failure verdict is `MECHANISM_NOT_WALL_POSITIVE`. It is a closure, not a PASS.

### Gate G-B3-R - route value

PASS requires candidate d512 throughput `>=1.00x` same-session native NV E,
with correctness and target-depth memory fit. Otherwise verdict
`CUDA_TAX_NOT_REPAID`; bank the compiler finding but do not promote Route B.

### Gate G-B3-P - parity

Only candidate/llama `>=1.00x` in the same session qualifies d512. This gate is
informational under this scope and does not authorize promotion.

## 14. Diagnostic lane L - compiled llama-kernel oracle

Purpose: make kernel implementation an experimental control. This lane asks
whether the planner/scheduler result survives when a real llama kernel is
placed behind tinygrad-owned buffers and graph control. It is diagnostic only:
no external kernel enters a default route, and no result from this lane alone
qualifies Route B or parity.

### 14.1 Completed bridge preflight L0

Artifact:

- `scratchpad/llama_cuda_binary_kernel_probe.py`;
- llama library
  `/home/ubuntu/env/llama.cpp/build-cuda/bin/libggml-cuda.so.0.14.0`;
- library SHA-256
  `d0f6580892fc5940321a3dfd9af3b3febd13c01102861da9c155ae4cda86ac49`;
- MMVF source SHA-256
  `23b580ce14a45e71cc9be31047301d502be74a832084c16662985f93f533ba1c`.

CPU-only inspection established:

- the installed library exports the exact instantiated C++ launch wrapper
  `launch_mul_mat_vec_f_cuda<float, float, 1, false>`;
- the fusion argument ABI is 32 bytes;
- the library contains 138 embedded `sm_120a` cubins;
- Q4_K and Q6_K one-column device entry symbols exist in the embedded cubins,
  but are `STB_LOCAL`, not dynamic host exports.

The lock-held RTX 5090 arm used a `64 x 512` float matrix and one-column vector:

```text
direct compiled llama launch over tinygrad buffers:
  max_abs_err = 4.76837158203125e-07       PASS

CUDA stream capture through tinygrad driver bindings:
  graph_nodes = 1
  graph_replays = 8
  max_abs_err = 4.76837158203125e-07       PASS
```

This proves all of the following:

1. tinygrad and llama device pointers are context-compatible on this host;
2. a compiled llama launch wrapper can consume tinygrad-owned buffers without
   a device copy or a llama tensor owner;
3. the resulting llama device launch can be captured and replayed as a normal
   CUDA graph node;
4. a diagnostic external-kernel oracle is feasible without assembly tuning.

It does **not** prove Q4_K/Q6_K ABI correctness, full-token substitution,
speedup, planner culpability, overlap, or parity. The float `64 x 512` row is a
bridge test, not a decode performance result.

Reproduction:

```bash
# CPU-only artifact and ABI inventory
python3 scratchpad/llama_cuda_binary_kernel_probe.py --inspect-only

# live correctness and graph-capture arm
flock -w 60 /tmp/gpu-bench.lock timeout 90s \
  python3 scratchpad/llama_cuda_binary_kernel_probe.py
```

### 14.2 What requires a GPU

| question | CPU-only sufficient? | live GPU required? |
| --- | --- | --- |
| library/source hashes, symbols, cubin inventory | yes | no |
| adapter compilation and argument-layout tests | yes | no |
| module load on `sm_120`, context/pointer interoperability | no | yes |
| Q4_K/Q6_K numerical equivalence | no | yes |
| graph capture/node identity/replay | no | yes |
| duration, bandwidth contention, overlap, wall conversion | no | yes |

Thus most bridge construction can be done without reserving the GPU. Every
claim that the oracle is correct or moves decode wall requires a lock-held GPU
arm.

### 14.3 Quantized oracle L1 - exact Q4_K and Q6_K standalone launches

Run this lane only after B3.1 has produced stable call and buffer identities.
Start with the exact d512 MMV shapes and packing used by the measured llama
baseline, not convenient toy shapes.

Required provenance for each Q4_K/Q6_K arm:

- llama commit/build directory and dirty state;
- shared-library and source hashes;
- exact embedded cubin name/hash;
- exact mangled device symbol;
- compiler, CUDA toolkit, `sm_120a`, and build flags;
- launch grid, block, shared bytes, parameter byte layout, and stream;
- tensor shape, quantization block layout, strides, and buffer byte hashes.

Preferred implementation order:

1. **Exact-binary arm:** identify and extract the embedded cubin containing
   `mul_mat_vec_q<(ggml_type)12,1,false,false>` and
   `mul_mat_vec_q<(ggml_type)14,1,false,false>`, load it through the current
   tinygrad CUDA context, and launch the local entry by exact mangled name.
2. **Linked-host arm:** if local cubin entry lookup is rejected, expose only a
   narrow C ABI adapter around an already-linked llama host op/launcher. The
   adapter may translate arguments but must not change packing or kernel code.
3. **Source-recompiled arm:** permitted only if both exact-binary and linked
   arms fail. It must use the pinned llama source and captured build flags and
   is labeled `SOURCE_RECOMPILED_ORACLE`, never exact-binary evidence.

Do not silently fall from one evidence class to another. Record the exact
failure at each boundary.

Reference ladder, cheapest first:

1. llama standalone output captured from the same pinned kernel/inputs;
2. independent dequantize-to-fp32 CPU reference on small fixtures;
3. existing tinygrad Q4_K/Q6_K output with tolerance and mismatch census;
4. end-to-end logits/token pins only after standalone checks pass.

Gate G-B3-LQ PASS requires for both Q4_K and Q6_K:

- exact target shape and byte layout;
- reference-correct output with stated absolute/relative tolerance;
- one direct launch and at least eight graph replays;
- captured kernel-node identity and unchanged input/output device addresses;
- no staging D2D/H2D copy hidden in the timed region;
- independent artifact hashes and a declared evidence class.

Any crash, context error, silent pack conversion, missing reference, or
unexplained mismatch is `LLAMA_QUANT_ORACLE_BLOCKED`, not a kernel verdict.

### 14.4 Integrated oracle L2 - hold the physical DAG constant

Gate: G-B3-LQ PASS and G-B3-C PASS. Add no route selector. Build a
process-local diagnostic graph adapter that can replace only identified
Q4_K/Q6_K MMV nodes while retaining:

- the same ordered full-token call census;
- the same physical input/output arena ranges;
- the same non-MMV kernels and copies;
- the same graph groups and dependency edges;
- the same stream assignment;
- the same depth/model/token inputs.

If exact call-count retention is impossible because one llama kernel covers a
different semantic boundary, report a boundary map and stop the constant-DAG
claim. A different boundary may still be measured as a diagnostic, but it is
not a kernel-only A/B.

Required arms in one session:

| arm | physical DAG | dominant MMV implementation | purpose |
| --- | --- | --- | --- |
| L2-A | default physical | tinygrad | control |
| L2-B | default physical | llama | kernel-only counterfactual |
| L2-C | selective-unaliased physical | tinygrad | planner-only candidate |
| L2-D | selective-unaliased physical | llama | interaction term |

L2-C/L2-D exist only if G-B3-S passed. Otherwise run L2-A/L2-B and state that
the planner interaction is unavailable.

For every arm record:

- correctness, call/edge/group identity, peak memory;
- per-class and MMV node-sum;
- graph span and end-to-end token wall;
- realized overlap and per-node duration inflation;
- copy/staging bytes;
- five or more interleaved retained repetitions.

The causal decomposition is:

```text
kernel_effect       = wall(L2-A) - wall(L2-B)
planner_effect      = wall(L2-A) - wall(L2-C)
combined_effect     = wall(L2-A) - wall(L2-D)
interaction         = combined_effect - kernel_effect - planner_effect
residual_llama_gap  = llama_wall - wall(L2-D)
```

Use signed values consistently and show raw times alongside the arithmetic.
No additivity is assumed; the interaction is measured.

### Gate G-B3-LO - oracle decision value

PASS requires the oracle to flip or materially sharpen a mechanism belief:

- `KERNEL_DOMINANT`: L2-B gains at least 5% while the physical DAG is fixed;
- `PLANNER_DOMINANT`: L2-C gains at least 5% and L2-B does not;
- `COMPOUND`: both isolated effects are material or L2-D exposes a material
  interaction;
- `NEITHER`: neither isolated arm gains 5%; return to boundary/copy/launch
  attribution.

This gate is about diagnostic value, not route promotion. Even an L2-D parity
row remains an external-kernel counterfactual and cannot become a production
claim under this scope.

### 14.5 Oracle HARD STOPs

Stop this lane when:

1. the exact llama model packing/shape cannot be reproduced byte-for-byte;
2. a quantized output lacks an independent reference;
3. an adapter introduces copies in the measured region;
4. the substituted graph changes dependencies or boundaries without being
   relabeled;
5. the oracle requires patching the production tinygrad route;
6. a local cubin symbol is treated as a stable public llama ABI;
7. source-recompiled evidence is described as the installed llama binary;
8. the float bridge result is used to predict quantized speed.

## 15. Phase B3.7 - depth extension

Gate: G-B3-M PASS. Run d2048 and d4096 even if G-B3-R failed only when the
record explains why depth may change the route economics. Otherwise stop at
d512 to avoid spending GPU time on a route already dominated by native NV.

For each depth repeat:

- CUDA default S=1;
- best qualified selective candidate S=1;
- native NV;
- llama;
- correctness pins/reference checks appropriate to depth;
- peak memory/context fit;
- wall ratios and CUPTI trace only where decision-relevant.

Each depth is separately classified. No interpolation or composed endpoint.

## 16. Decision branches and handoffs

### Branch A - `SEMANTIC_CHAIN`

Close planner/overlap work. Produce a handoff ranking:

1. CUDA/native kernel-count and boundary difference;
2. GEMV instruction mapping and achieved bandwidth;
3. fusion candidates with isolated wall ceilings;
4. graph-launch/inter-group gaps;
5. correctness-route divergence.

No planner candidate is built.

### Branch B - `PLANNER_NOT_ROOT_CAUSE`

If the physical DAG retains material logical branches but S=1 is serialized:

1. classify compatible versus resource-conflicting ready nodes;
2. run the identical real DAG through S=1 and capture S=2/3 diagnostic arms;
3. compare graph topology and scheduler behavior;
4. close capture if it lacks same-DAG wall value.

Do not revisit synthetic probes as route proof.

### Branch C - `PLANNER_CANDIDATE` but mechanism wall FAIL

Bank the generic compiler finding, candidate/edge ledger, and memory tradeoff.
Record that legal independence did not convert under decode contention. Return
to native NV kernel/boundary work.

### Branch D - mechanism PASS but `CUDA_TAX_NOT_REPAID`

Do not promote Route B. Quantify the remaining CUDA route tax by kernel class,
extra kernel count, graph launches, copies, and per-class runtime. Open another
scope only if a named lever has a ceiling large enough to exceed the tax.

### Branch E - G-B3-R PASS

Write a separate promotion-readiness scope. It must address:

- production/upstreamable planner policy rather than process-local selection;
- portable stable policy identity;
- memory budgets across models/depths;
- driver/version behavior;
- regression tests on CUDA and non-CUDA targets;
- fallback/rollback;
- same-session depth matrix;
- maintainability and upstream separation.

No default flip occurs in this document.

## 17. Risk register

| risk | consequence | control |
| --- | --- | --- |
| default/planner-free calls differ | false edge attribution | ordered stable-signature hard gate |
| abstract logical tracker diverges from DepsTracker | unsafe missing edge | canonical range semantics + synthetic equivalence tests |
| planner-free OOM | unavailable diagnostic | record OOM; use manifest/offline selective simulation |
| holding buffers changes graph grouping | confounded wall result | exact group/call identity gate |
| CUDA route already numerically wrong | optimizing invalid output | G-B3-C before wall ranking |
| overlap stretches bandwidth-bound kernels | theoretical gain disappears | per-node inflation + wall A/B |
| CUPTI changes timing | profiler result mistaken for wall | separate profiled/unprofiled rows |
| stale `/tmp` trace | mixed-session evidence | artifact hashes + session IDs + anchored summaries |
| driver update changes scheduler | old probe conclusion transferred | re-run smallest CH=2 control |
| stable buffer signature is not stable | wrong buffer held | fail closed; hash route and DAG |
| candidate consumes long-context headroom | d512-only false win | memory ledger + target-depth fit |
| agent promotes mechanism PASS as parity | overstated result | separate G-B3-M/R/P gates |
| llama C++/local-cubin ABI changes | brittle or wrong launch | pin hashes; exact argument manifest; fail closed |
| external kernel changes graph boundary | false kernel-only attribution | call/edge/group identity gate or relabel diagnostic |
| adapter inserts copies | oracle appears slower/faster for wrong reason | byte counters; exclude staged arm from constant-DAG A/B |

## 18. HARD STOPs

Stop immediately when any applies:

1. route/call alignment fails and cannot be explained without changing the
   workload;
2. UNKNOWN dependencies intersect the critical-path delta;
3. logical CUDA potential is below 5% of current CUDA wall;
4. CUDA numerics fail reference checks;
5. no selective candidate reaches 5% theoretical recovery;
6. candidate graph/call structure differs from control;
7. candidate wall gain is below 5%;
8. candidate cannot fit the target depth or has unbounded memory behavior;
9. a required step would modify native NV/HCQGraph/default planner policy;
10. evidence would require composing different sessions or depths.

A HARD STOP produces a record and the appropriate branch handoff. It does not
license widening the scope.

## 19. Deliverables and commit boundaries

| phase | required deliverables | suggested commit class |
| --- | --- | --- |
| B3.0 | preflight + CUDA route reproduction record | `[docs]` |
| B3.1 | attribution tooling, schema, fixtures, hermetic tests | `[test]` or `[tooling][test]` |
| B3.2 | live aligned DAG artifacts, CUDA ceiling, G-B3-D record | `[docs]` plus anchored JSON |
| B3.3 | per-class/chain correctness census, G-B3-C record | `[test][docs]` |
| B3.4 | candidate search, Pareto/rejection ledger | `[tooling][test][docs]` |
| B3.5 | process-local fail-closed selector + structural record | `[tooling][test][docs]` |
| B3.6 | d512 wall/CUPTI record, G-B3-M/R/P | `[docs]` plus anchored JSON |
| B3.L | llama binary bridge, Q4_K/Q6_K oracle, constant-DAG A/B if gated | `[scratchpad][test][docs]` |
| B3.7 | gated depth matrix | `[docs]` plus anchored JSON |
| final | branch verdict and handoff/promotion-readiness scope | `[docs]` |

Every implementation commit includes its tests. Measurement payloads are not
hidden in commit messages. `git diff --check` must pass for touched text files.
Commit only named deliverables; preserve all unrelated dirty and untracked user
files.

## 20. Required final record format

The final record is findings-first:

1. verdict: one of the Branch A-E outcomes;
2. correctness status;
3. logical versus physical CUDA DAG table;
4. planner edge attribution and top lost-critical-path buffers;
5. CUDA-specific ceiling versus route tax and llama gap;
6. candidate memory/critical-path Pareto table;
7. same-session d512 raw and median wall rows;
8. realized CUPTI overlap and duration inflation;
9. G-B3-D/C/S/M/R/P and G-B3-LQ/LO outcomes;
10. depth rows if authorized;
11. rejected candidates and negative evidence;
12. exact artifacts, hashes, commits, commands, and environment;
13. next authorized scope or explicit closure.

The record must answer plainly:

> Did the memory planner remove valuable real-decode independence, did a
> minimum-memory fix convert it to wall, and did that wall gain beat the native
> NV route?

Anything less is an intermediate mechanism result, not completion.
