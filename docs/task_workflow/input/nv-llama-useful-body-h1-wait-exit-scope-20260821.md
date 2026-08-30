# NV llama useful-body H1 wait-exit measurement scope

Date: 2026-08-21

Status: **measurement-first closing audit**. This packet authorizes an
instrumented llama.cpp build plus one locked Nsight capture. Its only claim
target is H1 from
`docs/task_workflow/input/nv-split-phase-pdl-causal-design-review-scope-20260820.md`:
is llama's ~1133 us of kernel-residence overlap mostly simultaneous useful
work, or mostly launch shadow plus in-kernel dependency wait?

No tinygrad production path is changed. No llama source change becomes a
promoted claim. The instrumented binary is semantics evidence only; its wall
is not the authority wall.

## 1. Why this is the last measurement

The interval identity is exact and the overlap claim is locked:

```text
node_sum - union = overlap_mass
5023.823 - 3890.568 = 1133.255 us
```

Interval overlap is not the same quantity as simultaneous useful execution.
A consumer grid can start early under PDL, then sit at
`cudaGridDependencySynchronize()` until the producer finishes. That wait time
counts as overlap mass without moving bytes. Every endpoint conclusion after
this one depends on separating the two:

- if most of 1133.255 us is dependency wait, llama's schedule advantage is
  smaller than the ledger suggests and tinygrad's serial support chain is not
  the only comparison point;
- if most of it is useful concurrent work, llama is genuinely packing DRAM
  work in time tinygrad leaves idle, and that packing is the parity target.

Useful-body concurrency has stayed `unmeasured` in every prior packet because
llama's CUPTI trace has kernel start/end but no wait-exit timestamp.

## 2. Instrument

Instrument the pinned llama source at `ac4cddeb0` behind a compile-time gate
`GGML_CUDA_PDL_TRACE`, controlled at runtime by
`GGML_CUDA_PDL_TRACE_DUMP`:

- replace the `ggml_cuda_pdl_sync` and `ggml_cuda_pdl_lc` function
  definitions with thin impl functions plus optional macros;
- at each wait call site, after the PDL sync, record one
  `%globaltimer` value into a device ring;
- at each trigger call site, immediately after
  `cudaTriggerProgrammaticLaunchCompletion`, record one value;
- record the lexically enclosing `__func__` string pointer, source line, block
  index, and event kind so records can be matched to CUPTI kernels;
- sample one entry per 256-block x-slice and thread 0 to keep the ring bounded;
- host-side, dump the ring after each backend synchronize into one JSONL file
  per call and reset the ring index for the next token.

The existing user-owned graph-dump tooling stays in place and is used to
retain the real CUDA graph node table.

## 3. Run

Same RTX 5090, Qwen3-8B-Q4_K_M, `-ngl 99 -fa 1 -p 512`, one llama-bench row:

```bash
flock -w 600 /tmp/gpu-bench.lock \
  env GGML_CUDA_PDL_TRACE_DUMP=/tmp/llama-pdl-trace \
    GGML_CUDA_GRAPH_DUMP=/tmp/llama-pdl-graph-dump.txt \
  /usr/local/bin/nsys profile --cuda-graph-trace=node --resolve-symbols=false \
    --force-overwrite=true --output=/tmp/llama-h1-pdl-trace.nsys-rep \
  /home/ubuntu/env/llama.cpp/build-cuda-instrumented/bin/llama-bench \
    -m /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf -ngl 99 -fa 1 \
    -p 512 -n 10 -d 512 -r 1 -o json
```

Export SQLite, choose the median steady replay with the retained
`llama_weighted_dag.py`, and align the corresponding ring dump by
`%globaltimer` range. All kernel interval work uses the instrumented
binary's own replay; no interval is borrowed from the earlier authority
trace.

## 4. Reconciliation definitions

For kernel `i` with CUPTI interval `[s_i, e_i]`:

```text
wait_exit_i = first sampled %globaltimer record assigned to kernel i
wait_i      = wait_exit_i - s_i
useful_i    = e_i - wait_exit_i
wait_mass   = sum(wait_i)
useful_mass = node_sum - wait_mass
```

The H1 discriminator is:

```text
overlap_mass - wait_mass = useful_mass - union
```

If `overlap_mass - wait_mass <= 0`, the overlap arithmetic is fully consumed
by launch shadow and dependency wait, and no simultaneous useful body is
proven. A positive remainder is a measured lower bound on useful concurrency.

## 5. Belief-flip gates

| gate | observable | passes when | fails when |
| --- | --- | --- | --- |
| G1 instrument fired | ring JSONL files | trigger and wait records exist for the first-layer kernels | no records or empty files |
| G2 records match kernels | per-kernel record assignment | each instrumented kernel gets an in-interval record | records cannot be assigned |
| G3 trace and ledger reconcile | node_sum, union, overlap | identity closes on the instrumented replay | nonzero unassigned arithmetic |
| G4 useful-concurrency lower bound | `overlap_mass - wait_mass` | positive and computed, regardless of sign | cannot be computed |
| G5 H1 verdict | wait share of overlap mass | stated as supported or refuted with the measured split | inferred from prior intervals |

H1 is `supported` only if the measured wait mass explains the overlap mass;
otherwise it is `refuted` by the measured positive useful-concurrency lower
bound. Neither verdict is a wall recovery claim.

## 6. Acceptance criteria

- instrumented build is feature-gated and does not alter the un-gated source
  path;
- all GPU rows run under the bench lock in fresh processes;
- raw nsys trace, SQLite, graph dump, ring JSONL files, and SHA-256 are
  retained;
- the chosen replay is named and its ring file is aligned by timestamp;
- the result reports useful-body, wait, launch-shadow, and overlap mass with
  `observed`, `inferred`, or `unmeasured` labels;
- no production promotion and no new architecture claim.

## 7. Outputs

Evidence:
`docs/task_workflow/evidence/nv-llama-useful-body-h1-20260821/**`

Result:
`docs/task_workflow/output/nv-llama-useful-body-h1-result-20260821.md`

Parser:
`extra/llm_research/decode/nv_llama_useful_body_h1.py`
