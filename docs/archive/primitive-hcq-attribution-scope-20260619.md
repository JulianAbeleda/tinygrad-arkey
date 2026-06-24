# Scope - PMU-4 tinygrad-native HCQ attribution

Purpose: after PMU-1..PMU-3, `rocprofv3` is proven useful for HIP/rocBLAS controls but invisible to the tinygrad HCQ
smoke in this environment. PMU-4 builds the missing tinygrad-native attribution layer. This is **not** a PMU counter
clone. It is Level-3 runtime/graph evidence that explains what tinygrad submitted, how it was grouped, and where
graph/host/program-cache boundaries sit. Level-4 PMU remains attached only when ROCm can see a dispatch.

## Current evidence

`primitive-pmu-observability-result-20260619.md`:

| path | result |
|---|---:|
| HIP/rocBLAS control kernel trace dispatches | 341 |
| HIP/rocBLAS PMC rows | 2952 |
| HIP/rocBLAS nonzero PMC rows | 328 |
| tinygrad HCQ trace dispatches under `rocprofv3` | 0 |
| verdict | `REDIRECT_HCQ_NATIVE_ADAPTER` |

So the next layer should answer tinygrad questions directly:

- what program was submitted;
- what queue submitted it;
- what launch geometry and kernarg size were used;
- whether it was eager or graph-captured;
- how many waits/signals/barriers/copies surrounded it;
- whether device time exists for the node;
- whether host wall time exceeds summed device time;
- whether program/runtime creation or graph construction repeated.

## Non-goals

- Do not rebuild ROCm PMU counters.
- Do not claim Level-4 root cause without PMU/thread-trace rows.
- Do not route model defaults.
- Do not instrument every closed primitive path.
- Do not add broad always-on runtime logging.

## Evidence level

PMU-4 produces **Level 3** evidence:

- queue/program/graph attribution;
- launch geometry;
- runtime and kernarg metadata;
- signal/wait/copy/barrier counts;
- graph segmentation;
- optional device timestamps from existing HCQ timestamp/profile machinery.

It can be combined with Level 4 PMU only for HIP-visible kernels or future HCQ-visible paths.

## Existing hooks to reuse

| file | hook | use |
|---|---|---|
| `tinygrad/runtime/support/hcq.py` | `HCQProgram.__call__` | eager program launch attribution and wait/device-time boundary |
| `tinygrad/runtime/support/hcq.py` | `HCQProgram.fill_kernargs` | kernarg size, buffer count, value count, runtime type |
| `tinygrad/runtime/graph/hcq.py` | `HCQGraph.__init__` | graph construction: calls, runtimes, queue schedule, deps, copies |
| `tinygrad/runtime/graph/hcq.py` | `HCQGraph.__call__` | graph replay count, rebind variables, timeline waits |
| `tinygrad/runtime/ops_amd.py` | `AMDComputeQueue.exec` / `AMDComputeAQLQueue.exec` | final AMD launch geometry and program resource metadata |
| `tinygrad/runtime/support/hcq.py` | `hcq_profile` / `PROFILE` events | existing GPU timestamp signal mechanism |
| `tinygrad/engine/realize.py` | `exec_kernel` / `exec_graph` | high-level eager-vs-graph boundary and `track_stats` context |

## Artifact schema

### `qk_hcq_attribution_v1`

Top-level fields:

- `schema`, `generated_at`, `commit`, `device`, `backend`;
- `mode`: `probe_local`, `runtime_flag`, or `profile_import`;
- `workload`: name, command, env, shape/context;
- `summary`: program count, graph count, copy count, wait/signal/barrier counts, device time, wall time, wall/device gap;
- `programs`: one row per eager or graph program;
- `graphs`: one row per graph capture/replay;
- `classification`: deterministic labels;
- `provenance`: scripts, source commits, artifacts.

Program row:

- `program_name`;
- `runtime_class`;
- `device`;
- `launch`: global/local/grid/workgroup;
- `metadata`: kernarg size, buffer count, value count, VGPR/SGPR/LDS if available, code object hash if available;
- `queue`: compute/copy/rdma, queue index if known;
- `sync`: waits, signals, barriers before/after;
- `timing`: device timestamp if available, host wall if measured;
- `graph`: graph id/node index if captured, otherwise eager.

Graph row:

- `graph_id`;
- `call_count`;
- `runtime_count`;
- `copy_count`;
- `devices`;
- `queue_count`;
- `rebind_count`;
- `program_names`;
- `prof_signal_count`;
- `replay_count`;
- `wall_ms`;
- `device_ms_sum` if available.

## Classifier

The adapter should produce only these labels initially:

| label | condition |
|---|---|
| `rocprof_hcq_visibility_gap` | PMU-3 reproduced: ROCm tool present, HIP control visible, HCQ trace rows absent |
| `graph_boundary` | isolated device-time win exists but graph/eager routing wall time does not transfer |
| `host_sync` | wall/device gap dominated by waits/synchronizes around eager launches |
| `program_cache_miss` | same logical program/runtime is rebuilt repeatedly across warm calls |
| `graph_capture_missing` | expected graph has eager program submissions or graph call count is zero |
| `graph_rebind_ok` | graph replay reuses captured node while rebinding buffers/vars |
| `kernel_math_bound` | wall/device gap is low and remaining limit is inside device time; PMU optional for sublabel |
| `unknown` | evidence insufficient |

## Phased build

### PMU-4a - parser over existing profile/PMU artifacts

Build `extra/qk_hcq_attribution.py` in read-only mode first:

- ingest `bench/qk-pmu-observability/result.json`;
- ingest current primitive ledger;
- optionally ingest existing tinygrad profile/SQTT artifacts if present;
- emit `bench/qk-hcq-attribution/result.json`.

Gate:

- reproduces `rocprof_hcq_visibility_gap`;
- no runtime changes;
- ledger can reference the attribution result.

### PMU-4b - eager launch attribution probe

Probe-local monkeypatch or context manager around `HCQProgram.__call__` and `fill_kernargs`:

- record program name, runtime class, kernarg size, buffer count, vals count;
- record global/local sizes;
- record wait flag and returned device time when `wait=True`;
- record host wall time around the call;
- run a tinygrad matmul smoke and the extracted Tensile eager route if available.

Gate:

- artifact has at least one eager program row with launch geometry and timing;
- overhead under instrumentation is acceptable for diagnostic mode;
- no default/runtime behavior changes outside the context manager.

### PMU-4c - graph attribution probe

Probe-local wrapper around `HCQGraph` construction/call:

- record number of calls, runtimes, copies, devices, queues, deps;
- record graph replay count and rebind count;
- record per-node program names and launch dims from `ast.arg.global_size/local_size`;
- if `PROFILE=1` is usable, attach graph profile entries/timestamps.

Gate:

- artifact distinguishes eager vs graph-captured execution on a tinygrad TinyJit smoke;
- can classify `graph_capture_missing` vs `graph_rebind_ok`;
- no core runtime patch required yet.

### PMU-4d - runtime flag, if probe succeeds

Promote only the minimal instrumentation behind an env flag, e.g. `QK_HCQ_ATTRIB=1`:

- central collector module;
- zero overhead when disabled;
- JSONL output path controlled by env;
- no changes to normal profiling semantics.

Gate:

- disabled path byte/timing neutral in tests;
- enabled path writes portable repo-relative artifacts;
- works for both eager `HCQProgram.__call__` and `HCQGraph`.

### PMU-4e - primitive ledger integration

Teach `extra/qk_primitive_ledger.py` to ingest `bench/qk-hcq-attribution/result.json`.

Gate:

- TPE-6/TPE-7 rows can attach runtime-boundary evidence;
- classifier can distinguish `graph_boundary`, `host_sync`, `program_cache_miss`, and `graph_capture_missing`;
- the ledger still does not claim PMU Level 4 for HCQ without ROCm counters.

## First target workloads

Use tiny, deterministic probes first:

1. tinygrad HCQ matmul smoke: validates eager attribution.
2. TinyJit two-call matmul smoke: validates graph capture/replay attribution.
3. extracted Tensile eager route: validates custom runtime/TensileRunner attribution.
4. TPE-7 graph route, once it exists: real target.

Do not start with full model prefill; use it only after attribution is stable.

## Result interpretation

If PMU-4 succeeds, we will know where a tinygrad graph route loses without needing ROCm PMU visibility:

- program did not graph-capture;
- graph captured but rebuilt;
- graph captured but host synchronizes per node;
- graph captured but device time remains high;
- graph captured and transfers, making remaining root cause kernel-local.

Only the last case should trigger PMU/thread-trace work.

## Close criteria

PMU-4 is complete when:

- `bench/qk-hcq-attribution/result.json` exists;
- the result includes eager and graph smoke rows;
- it classifies the current TPE runtime boundary without Level-4 overclaim;
- the primitive ledger links the attribution result;
- a follow-up doc says whether PMU-5 should target TPE-7 graph route or stop.
