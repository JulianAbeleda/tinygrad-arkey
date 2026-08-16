# NV overlap Route B at HEAD: width-4 still measures FLAT (2026-08-15)

Date: 2026-08-15
Branch: `nvidia-bringup-20260731` (code `c74567a24`, record `7462fc2bd`)
Status: **measured. Wall A/B plus token-identity check. The width-4 parallelism
does not convert to wall; the overlap lever is closed at HEAD.**

This is the test asked for after
`nv-decode-dag-width-verdict-20260815.md` found the decode DAG is width 4, not
width 1. It answers the remaining open question: does that width buy wall time?

## 1. Method

`scratchpad/nv_route_b_wall_probe.py`, fresh process per arm, `DEV=CUDA`,
`CUDA_GRAPH_STREAMS` in {1, 2, 3, 4}, depth 512 on Qwen3-8B-Q4_K_M / RTX 5090.
Each arm primes the graph, settles 4 tokens, times 32 tokens, and hashes the
token stream. Token identity across arms is the correctness gate.

## 2. Result

| streams | ms/token | tok/s | token sha |
| --- | ---: | ---: | --- |
| 1 | 5.573 | 179.437 | `ddf344135e...` |
| 2 | 5.581 | 179.179 | `ddf344135e...` |
| 3 | 5.575 | 179.375 | `ddf344135e...` |
| 4 | 5.595 | 178.730 | `ddf344135e...` |

Flat to within 0.4% (run noise; streams=4 is marginally slower, not faster).
Tokens are bitwise identical across all four arms, so multi-stream capture is
correctness-clean and buys no wall.

## 3. Interpretation

The width-4 bundle is q/k/v GEMV concurrency. All three are memory-bound reads
of the same HBM weights, so placing them on separate streams saturates the same
bandwidth and leaves wall unchanged. This is the correct reason the old Route B
record was FLAT: the original "width 1" justification was stale, but the flat
conclusion survives, now grounded as "the only parallelism is bandwidth-bound
GEMV concurrency, not latency-bound support hiding behind an anchor."

llama's overlap is a different shape: latency-bound support (quantize, rms_norm,
rope, flash) hidden behind one long `mul_mat_vec_q` anchor. At HEAD our support
is fused into the GEMV epilogues, so there is no independent support tail to
co-schedule. Un-fusing it was already measured body-free FLAT, so neither the
current width nor a naive un-fuse reopens the overlap lever.

## 4. Conclusion

- Multi-stream overlap (Route B) is **NO-GO at HEAD**, measured.
- NV native multi-queue (Route A) would distribute the same bandwidth-bound
  q/k/v GEMVs and is expected to be flat for the same reason; it is not a 240
  lever until the support mass exists as independent latency-bound kernels.
- The honest 240 target is therefore not reachable through overlap. Kernel work
  remains the only measured lever: Q6 GEMV core (~240 us) plus flash score
  floor (~90 us), landing ~215-222 tok/s, not 240.

## Evidence

- `/tmp/route_b_wall_s{1,2,3,4}.json` (this run)
- `/tmp/nv_decode_dag_width.json`, `/tmp/nv_decode_dag_width_core.pkl` (width-4)
- `nv-decode-dag-width-verdict-20260815.md`
- `nv-overlap-route-b-wall-outcome-20260815.md` (prior FLAT at `319241408`)
