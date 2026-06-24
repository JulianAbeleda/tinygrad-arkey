# Scope - primitive-local observability and hardware-feedback search

Purpose: turn the project's manual research loop into a small, reusable, primitive-local tooling layer. We should not
wait for perfect gfx1100 PMU tooling, and we should not build a general profiler from scratch. The right build is a
set of adapters that records the evidence each primitive actually needs: correctness, device time, static metadata,
runtime traces where available, and optional counters/thread traces when the toolchain supports them.

This scope is deliberately after TPE-6. TPE-6 proved the extracted Tensile FFN block transfers on GPU time
(`1.53x` matmul speedup) but redirects to graph integration because naive per-op host sync eats the win. That makes
runtime-boundary observability a first-class primitive need, not a generic profiler nice-to-have.

## Open-source patterns to reuse

| source | reusable idea | how it maps here |
|---|---|---|
| TVM Ansor auto-scheduler, https://tvm.apache.org/2021/03/03/intro-auto-scheduler | generate candidate schedules, measure them on real hardware, feed results back into search/cost model | adopt the loop shape, not TVM IR: candidate row -> runner -> measured artifact -> ranked frontier |
| TVM Meta Schedule RFC, https://discuss.tvm.apache.org/t/rfc-meta-schedule-autotensorir/10120 | schedule primitives as an explicit DSL/design space | our primitive rows should expose legal knobs as data, not prose-only plans |
| Triton autotune/tutorials, https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html | explicit config lists keyed by shape and benchmarked locally | use small shape-keyed config/candidate sets before any learned search |
| KernelBench, https://github.com/ScalingIntelligence/KernelBench | correctness + performance evaluation as the core metric; benchmark tasks grouped by operator/fusion level | reuse the "correct and faster than baseline" framing, but with project-specific in-model gates |
| Mirage Persistent Kernel, https://github.com/mirage-project/mirage | graph/kernel hierarchy and persistent-kernel profiling for launch-boundary removal | future TPE graph integration should observe graph segments, not only isolated kernels |
| ROCm profiling stack, https://rocm.blogs.amd.com/software-tools-optimization/profiling-guide/intro/README.html | rocprofv3 traces/counters, rocprof-compute roofline, rocprof-sys application traces | optional Level 3/4 evidence source when it works on this setup; not required for go/kill gates |
| ROCprof Compute Viewer, https://github.com/ROCm/rocprof-compute-viewer | thread-trace/ISA/source views, occupancy, hotspot, memory wait dependency views | optional deep-diagnostic adapter for selected kernels; not the base artifact format |
| tinygrad BEAM + local search hooks | `_time_program`, `_BEAM_CANDIDATE_FILTER`, `_BEAM_SCHEDULE_LOG`, cache, candidate timing | reuse these for tinygrad-generated candidates instead of writing a new timing engine |
| existing project flywheel tools | `candidate_record`, feature enrichment, source/compile feature extraction | reuse record shapes where possible; extend them to primitive lifecycles |

## Local assets already present

| asset | current role | reuse decision |
|---|---|---|
| `extra/qk_search_spec.py` | schema authority for bounded search rows | extend, do not replace |
| `extra/qk_demote_search.py` | working search orchestrator with quality gate | copy orchestration pattern for candidate sessions |
| `extra/qk_loop_live.py` | live hardware timing of BEAM candidates | reuse for tinygrad schedule candidates |
| `tinygrad/codegen/opt/search.py` | BEAM action space, timing, candidate hooks | use as the tinygrad-codegen candidate engine |
| `extra/qk_flywheel_dataset_v1.py`, `extra/qk_flywheel_feature_enrich.py` | candidate records + static/source feature extraction | evolve into `candidate_record_v2` rather than inventing a second format |
| `test/amd/test_sqtt_profiler.py` | SQTT profile events and tinygrad profile plumbing | use for optional trace validation if SQTT is enabled |
| `extra/q4_k_profile_report.py`, `extra/qk_gap_profile.py` | decode primitive profile buckets | preserve as MMVQ adapters |
| `extra/qk_tensile_shape_matrix.py`, `extra/qk_tensile_hcq_perf.py`, `extra/qk_tensile_block_transfer.py` | Tensile extraction/perf/block probes | preserve as prefill/Tensile adapters |
| `bench/**/result.json`, `bench/**/decision.json`, `bench/**/profile-report.*` | durable artifacts | ingest into a primitive ledger |

## Core concept

Build **primitive-local observability**, not a monolithic profiler.

Each primitive gets an adapter that can answer:

- what candidate was tried;
- what legal knobs produced it;
- what correctness oracle was used;
- what timing authority was used;
- what static metadata was available;
- what runtime/trace/counter evidence was available;
- what bottleneck class is inferred;
- what gate passed/failed;
- what should not be retried.

The shared tool owns schema, ingestion, validation, ranking, and reports. The primitive adapters own domain math.

## Evidence hierarchy encoded in artifacts

Every observation must label its evidence level:

| level | artifact examples | rule |
|---:|---|---|
| 0 | correctness, value equality, dNLL | required before ranking |
| 1 | device time, tok/s, warm pp throughput | enough for go/kill when decisive |
| 2 | static metadata, ISA, VGPR/SGPR, LDS, spills, instruction counts | enough for bounded root-cause support |
| 3 | DEBUG=2 kernel timeline, rocprof trace, HCQ graph/program attribution, SQTT event names | supports lifecycle/routing claims |
| 4 | PMU counters, stall reasons, cache/VMEM/LDS/tensor-issue metrics, thread trace | optional strongest diagnostics |

Rule: counter absence cannot block a decisive gate; counter absence must block overconfident root-cause language.

## Proposed schema

### `primitive_observation_v1`

One row per measured primitive/candidate.

Required fields:

- `id`, `timestamp`, `commit`, `hardware`, `backend`;
- `primitive`: `mmvq_decode`, `prefill_tensile`, `prefill_wmma`, `attention_kv`, `runtime_boundary`;
- `phase`: decode / prefill / long_context / graph_integration;
- `role`: e.g. `ffn_gate`, `ffn_up`, `ffn_down`, `attn_q`, `attn_o`, `lm_head`, `attention`;
- `shape`: model dimensions and token/context regime;
- `candidate`: stable candidate id, parent id, legal knob vector, source hash;
- `correctness`: oracle, tolerance, pass/fail, quality gate where lossy;
- `timing`: median/min/max device time, wall time if relevant, derived GB/s or TFLOPS;
- `metadata`: VGPR/SGPR/LDS/wave/workgroup/descriptor/ISA-derived counts where available;
- `runtime`: program count, graph segment, launch geometry, host-sync count, cache state;
- `evidence_levels`: which levels are present;
- `bottleneck_inference`: one of `bandwidth`, `alu`, `pack_lifecycle`, `routing_overhead`, `graph_boundary`,
  `layout_copy`, `occupancy_or_issue`, `unknown`;
- `gate`: pass/fail/redirect/kill plus reason;
- `provenance`: source scripts, artifacts, docs.

### `primitive_search_session_v1`

One row per search run:

- search row id;
- primitive target;
- candidate generator;
- candidate count;
- budget;
- ranking policy;
- accepted frontier;
- refuted candidate classes;
- artifact paths.

### `primitive_ledger_v1`

Append-only aggregate:

- latest observation per primitive/role/shape;
- best known candidate;
- shipped/refuted/deferred/open state;
- current next action.

## Primitive adapters

### 1. MMVQ decode adapter

Inputs:

- DEBUG=2 decode logs;
- `q4_k_profile_report` / `qk_gap_profile` outputs;
- Q4_K/Q6_K primitive microbench artifacts;
- optional ISA/source reports.

Derived metrics:

- effective Q4/Q6 GB/s;
- VALU/weight proxy from ISA when available;
- q8 pack cost if candidate uses q8;
- reduction time/share;
- W==D tok/s transfer;
- dNLL if lossy.

Failure classes:

- `q8_pack_wall`;
- `fp_dequant_alu_ceiling`;
- `coalescing_missing`;
- `reduction_overhead`;
- `sub_gate_amdahl`;
- `quality_fail`.

### 2. Prefill Tensile / WMMA adapter

Inputs:

- TPE-1 through TPE-6 artifacts;
- Tensile contract JSON;
- HCQ perf JSON;
- block-transfer JSON;
- tinygrad WMMA sweep results.

Derived metrics:

- TFLOPS by role;
- ratio to tinygrad plateau;
- ratio to external reference;
- weighted pp model;
- host/routing overhead;
- graph-capture readiness;
- descriptor/kernarg stability.

Failure classes:

- `hip_runtime_wall`;
- `external_artifact_policy`;
- `graph_boundary_overhead`;
- `layout_copy_required`;
- `workspace_or_aux_required`;
- `wmma_issue_plateau`.

### 3. Attention / KV adapter

Inputs:

- current flash-decode/attention artifacts;
- long-context benchmark artifacts;
- future KV quantization probes;
- optional SDPA comparison and attention slope reports.

Derived metrics:

- effective KV bytes and bandwidth;
- attention share by context;
- ctx slope;
- page/block/ragged overhead if present;
- softmax/reduction time;
- correctness vs SDPA;
- dNLL if KV quantized.

Failure classes:

- `reuse_free_flash`;
- `kv_bandwidth`;
- `softmax_reduction`;
- `layout_or_page_overhead`;
- `quality_fail`;
- `regime_not_dominant`.

### 4. Runtime / graph-boundary adapter

Inputs:

- TPE-6 block-transfer artifact;
- tinygrad profile events;
- HCQ graph/TinyJit traces;
- DEBUG=2 program timelines;
- optional rocprof-sys/rocprofv3 traces.

Derived metrics:

- program count;
- host sync count;
- per-op wall vs device gap;
- graph segment count;
- graph-captured vs uncaptured delta;
- launch overhead amortization.

Failure classes:

- `jitless_probe_artifact`;
- `graph_capture_missing`;
- `program_cache_miss`;
- `host_sync`;
- `artifact_load_overhead`.

## Phased build plan

### PLO-0 - inventory and schema freeze

Deliverable:

- `docs/primitive-local-observability-search-scope-20260619.md` (this doc);
- schema examples for `primitive_observation_v1`, `primitive_search_session_v1`, `primitive_ledger_v1`;
- list of existing artifacts each adapter can ingest.

Gate:

- schema can represent TPE-5, TPE-6, POWN-1, Q8L-2, and one MMVQ shipped result without special cases.

### PLO-1 - read-only ledger collector

Build:

- `extra/qk_primitive_ledger.py`;
- read existing `bench/**/{result,decision,profile-report,shape_matrix,hcq_perf,block_transfer}.json`;
- emit `bench/qk-primitive-observability/ledger.jsonl` and `summary.md`.

Gate:

- reconstructs the current source-of-truth states:
  - q8/MMVQ deferred behind codegen;
  - pure-tinygrad WMMA sweep refuted;
  - Tensile extraction TPE-5 PASS;
  - TPE-6 REDIRECT to graph integration;
  - spec decode closed.

Non-goal:

- no new hardware runs.

### PLO-2 - primitive observation validators

Build:

- small validators for each adapter;
- artifact schema checks;
- evidence-level checker;
- root-cause-language checker: a row cannot claim Level-4 root cause with only Level-1 evidence.

Gate:

- invalid/missing fields fail fast;
- existing committed artifacts validate or are explicitly marked legacy/provenance.

### PLO-3 - candidate runner wrapper

Build:

- `extra/qk_primitive_candidate_runner.py`;
- wraps existing runners rather than replacing them:
  - BEAM/_time_program for tinygrad schedules;
  - TPE probes for extracted Tensile;
  - qk_demote_search/qk_nll_eval for lossy decode;
  - future attention probes.

Gate:

- can run a tiny, safe smoke search with 2-3 candidates and produce observation rows;
- no default/model route changes;
- no BEAM on known-hang paths unless explicitly allowed.

### PLO-4 - failure classifier and frontier report

Build:

- deterministic classifier first, not learned:
  - thresholds from primitive rows;
  - Amdahl gate;
  - evidence-level labels;
  - known refutation table.

Gate:

- given old artifacts, the classifier agrees with current docs on at least:
  - Q4_K separate q8 pack -> kill/closed;
  - POWN-1 -> refuted bounded tinygrad WMMA;
  - TPE-6 -> redirect graph boundary;
  - attention reuse-free flash -> refuted.

### PLO-5 - guided search memory

Build:

- append-only candidate DB;
- parent/mutation lineage;
- shape-keyed best configs;
- simple ranker using existing `qk_loop_live.py`/flywheel features.

Gate:

- improves time-to-good-candidate on one held-out tinygrad schedule search, or honestly records no win;
- never bypasses correctness and in-model gates.

### PLO-6 - optional counter/trace plugin

Build only if PLO-1 through PLO-4 are useful:

- `rocprofv3`/RCV import adapter for HIP/separate-process traces;
- tinygrad SQTT/profile event importer for HCQ/TinyJit;
- static ISA metadata extraction for tinygrad generated kernels and Tensile descriptors.

Gate:

- adds evidence Level 3/4 to one selected primitive without destabilizing the workflow.

## Immediate application to current frontier

The first live user should be the post-TPE-6 graph integration arc:

- primitive: `runtime_boundary`;
- question: does a graph-capturable Tensile launch preserve the 1.53x FFN GPU speedup end-to-end?
- required observation:
  - program count;
  - host sync count;
  - graph captured vs uncaptured;
  - device time per raw Tensile node;
  - wall time for block;
  - correctness;
  - evidence level at least 1 for gate, ideally 3 for graph-boundary diagnosis.

This is more valuable than starting with a broad BEAM search because the blocker is known and the artifact schema can
prove whether graph integration fixed it.

## What not to build

- A full replacement for rocprof/Nsight.
- A new compiler IR.
- A general LLM kernel-generation agent.
- A learned cost model before the ledger and deterministic classifier work.
- A search loop that invents candidates outside a primitive row's legal knobs.
- Any model default route.

## Decision rules

Proceed if:

- PLO-1 can reconstruct current verdicts from artifacts;
- PLO-3 can produce new observation rows without one-off parsing;
- PLO-4 prevents at least one stale/refuted path from being re-opened;
- the TPE graph-integration arc benefits from the runtime-boundary adapter.

Kill or defer if:

- the ledger becomes a second, contradictory source of truth;
- the adapters require more maintenance than the probes they replace;
- the tool cannot represent TPE/MMVQ/attention/runtime primitives without special-case sprawl;
- existing docs/artifacts already answer the question more clearly.

## Expected first deliverable after this scope

`extra/qk_primitive_ledger.py`:

- read-only;
- no hardware execution;
- no model route;
- outputs `bench/qk-primitive-observability/ledger.jsonl` and `summary.md`;
- validates against current docs;
- establishes the schema before any runner/search code is added.

That is the smallest useful build and the safest way to reuse the open-source patterns without importing their full
stacks.
