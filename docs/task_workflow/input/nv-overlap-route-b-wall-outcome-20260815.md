# NV overlap Route B wall outcome: multi-stream does not move decode wall

Date: 2026-08-15
Branch: `nvidia-bringup-20260731` (HEAD `319241408`)
Status: **measured. The ~219 tok/s overlap claim is falsified by a direct
wall A/B plus a scheduler-distribution probe.**

This closes the skeptical test requested on the overlap lever: run the full
decode token on the CUDA multi-stream graph substrate and measure wall, instead
of trusting the DAG-critical-path arithmetic that priced it at ~219 tok/s.

## 1. Method

Fresh process per arm, `DEV=CUDA`, `CUDA_GRAPH_STREAMS` in {1, 2, 3}, d512 on
Qwen3-8B-Q4_K_M / RTX 5090. Each arm primes the graph, settles 4 tokens, then
times 32 tokens and hashes the token stream. Probe:
`scratchpad/nv_route_b_wall_probe.py`.

## 2. Wall result

| streams | ms/token | tok/s | token sha |
| --- | ---: | ---: | --- |
| 1 (serial) | 5.5667 | 179.639 | `ddf34413...` |
| 2 | 5.5623 | 179.782 | `ddf34413...` |
| 3 | 5.5661 | 179.658 | `ddf34413...` |

Flat to within 0.1% (run noise). Tokens are bitwise identical across all arms,
so the multi-stream graph is correctness-clean but buys no wall.

## 3. Why (scheduler distribution)

`scratchpad/nv_route_b_stream_dist_probe.py` wraps `plan_multi_stream` and
reports the stream assignment for every captured graph in the decode token:

| graph calls | stream 0 | stream 1 |
| ---: | ---: | ---: |
| 32 | 31 | 1 |
| 64 | 64 | 0 |
| 128 | 128 | 0 |
| 256 | 256 | 0 |
| 512 | 512 | 0 |
| 29 | 29 | 0 |

1020 of 1021 calls land on stream 0. The frozen range-aware dependency DAG has
width ~1: the ready-set list scheduler finds no independent branches to
distribute, so the multi-stream capture degenerates to serial execution.

This is the DAG-width evidence the B1 record explicitly asked for: the earlier
~651 us / 11.9% "critical-path slack" was an arithmetic upper bound on a
duration-head graph; it does not exist as schedulable parallelism in the real
frozen decode DAG.

## 4. Conclusion

- Overlap is **not** the lever to parity. The only route with the multi-stream
  graph substrate (Route B, CUDA) measures flat, and the native Route A remains
  driver-blocked (`CONSTRUCTION_BLOCKED`).
- The CUDA-route baseline here is 179.6 tok/s (no NV reduce-output fusion);
  production NV is 193.5 tok/s. Overlap would need to be tested on NV, but the
  flat CUDA result plus the width-1 DAG evidence makes that a serialization
  question, not a stream-count question.
- The remaining honest levers are the critical path itself: the Q4 FFN-down
  GEMV (2.29x llama, closed NO-GO) and the structural flash-score floor, plus
  whether the width-1 chain can be widened at all.

## Evidence

- `/tmp/route_b_wall_s{1,2,3}.json` (this run)
- `/tmp/route_b_stream_dist_s2.json` (stream assignment per graph)
- prior ceiling arithmetic: `nv-overlap-ceiling-route-b-test-20260814.md`
- mechanism probe: `nv-decode-overlap-route-b1-multi-stream-graph-probe-measurement-record-20260804.md`
