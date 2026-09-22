# NV decode parity - E1 llama graph-optimization measurement record

Date: 2026-08-03/04 (measured 2026-08-04 04:0x UTC, one flocked GPU session,
same RTX 5090 box, no concurrent GPU work)
Status: measurement record for experiment E1 of
`nv-decode-parity-e1e3-measurement-scope-20260803.md`, authorized by
`nv-decode-parity-external-review-amendment-20260803.md` section 7.
Question: what does `GGML_CUDA_GRAPH_OPT` change on this box, and which
policy produced the authority rows (251.8 tok/s, 762-node trace)?
Branch: tinygrad `nvidia-bringup-20260731` at `fed89a201`; llama.cpp checkout
`ac4cddeb0` (verified `git rev-parse HEAD`). All numbers OBSERVED unless
marked INFERRED.

## 1. Protocol

Same-session A/B, one flocked GPU session:

- Arm 0: `GGML_CUDA_GRAPH_OPT=0 llama-bench -m Qwen3-8B-Q4_K_M.gguf -ngl 99
  -fa 1 -p 0 -n 10 -d 512 -r 5 -o json`
- Arm 1: same with `GGML_CUDA_GRAPH_OPT=1`
- Each arm also traced with `nsys profile --cuda-graph-trace=node` (same
  command, `-r 3`), then node events and graph-launched kernels were read
  from the SQLite export of the `.nsys-rep`.

Environment per arm recorded (`/tmp/e1_arm{0,1}_env.txt`); llama-bench
`build_commit` `ac4cddeb0`, CUDA backend, RTX 5090 (sm_120), driver 595.84.

## 2. Results

### 2.1 Wall (unprofiled, r=5)

| arm | tok/s mean | stddev |
| --- | ---: | ---: |
| opt=0 | 246.32 | 12.55 |
| opt=1 | 258.07 | 16.92 |

Delta: +4.77% wall, ~0.185 ms/token (INFERRED arithmetic: 1/246.32 -
1/258.07). Traced arms (r=3): 227.94 vs 237.71 tok/s (+4.3%).

### 2.2 Graph structure and overlap (nsys node trace, per launch)

Arm 0 (default/opt OFF), 29 launches of 762 nodes each:

| quantity | median over 29 launches |
| --- | ---: |
| replay span | 3.889 ms |
| node-sum | 5.013 ms |
| below node-sum | 22.4% |

Arm 1 (opt ON), 29 launches of 762 nodes each:

| quantity | median over 29 launches |
| --- | ---: |
| replay span | 3.687 ms |
| node-sum | 4.859 ms |
| below node-sum | 24.0% |

Span/node-sum computed from CUPTI kernel rows grouped into launches by
start-time gaps (50 us); node counts per launch 762 in both arms. The
decomposition record's 22% figure reproduces with the QKV feature OFF.

## 3. Verdict

The provenance question resolves: the 22% overlap trace and the 251.8 tok/s
authority row ran with the DEFAULT policy (`GGML_CUDA_GRAPH_OPT` was never
set in this repo or in the traced environment; the trace's captured
environment contains no `GGML_*` variables). The overlap decomposition stands
as base CUDA graph node scheduling.

`GGML_CUDA_GRAPH_OPT=1` adds +4.8% wall (~0.185 ms/token) and lifts overlap
22.4% -> 24.0%. Per the scope's belief-flip criteria, the wall delta is below
the 0.2 ms parity-scale threshold: the gated QKV concurrent-stream mechanism
is NOT the explanation for llama's wall advantage; base graph node scheduling
is. The QKV feature is a small additive lever on top of it.

Artifacts: `/tmp/e1_arm0_trace.nsys-rep`, `/tmp/e1_arm1_trace.nsys-rep`,
`/tmp/e1_arm{0,1}.json`, `/tmp/e1_arm{0,1}_env.txt` (session-scoped, not
committed).
