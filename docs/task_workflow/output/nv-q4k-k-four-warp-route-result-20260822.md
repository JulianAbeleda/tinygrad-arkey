# Q4_K attention-K four-warp route wall result (2026-08-22)

## Verdict

The exact four-warp Q4_K K candidate is **wall-neutral**. The isolated CUDA
bracket's -14% per launch does not propagate to the composed decode wall, so
this codegen change is not a promotion candidate and is not worth more time.

| Quantity | Control midpoint | Candidate | Delta |
|---|---:|---:|---:|
| Wall ms/token | 4.727443 | 4.725635 | -0.001808 ms (-1.81 us) |
| Token-stream SHA | equal | equal | PASS |
| Speedup | -- | -- | +0.038% |

The candidate is 1.81 us/token faster than the control midpoint. The two
bracketing controls themselves drift by 1.22 us/token, so the candidate delta
sits inside the run-to-run noise band. It is not a resolvable wall win.

## What was tested

The installed Q4_K attention-K path is
`q4k_g3_lanemap_gemv_1024_4096`, one warp per 1024-row output. The candidate
is `emit_q4k_exact_four_warp(1024, 4096)`: 128 threads per output row, no Q8
provider, one contiguous fp32[1024] output. The earlier standalone cudaEvent
bracket measured -14.0% (3.4417 -> 2.9599 us/launch) and 11.98% -> 47.96%
occupancy. This test asks whether that launch/occupancy recovery survives the
real decode schedule.

All 36 Q4_K `attn_k` blocks were admitted in the candidate arm. A graph census
confirmed the candidate kernel fired:

- `q4k_exact_four_warp_1024_4096` (Q4 K candidate)
- `q4k_g3_lanemap_gemv_1024_4096` (Q4 V, unchanged)
- `q6k_v_four_warp_fp16_direct_1024_4096` (Q6 V, unchanged)

So this is a genuine wall-neutral, not a silent fallback.

## Why the isolated win disappears

The isolated bracket compares the kernel body with no surrounding graph. The
composed decode wall overlaps or serializes the K GEMV with the other
attention support, so the 0.48 us/launch device-time recovery is almost fully
hidden. This matches the overlap ledger: llama's apparent overlap mass is
91.9-95.4% launch shadow, and tinygrad runs that support region serially. A
faster support body does not remove the serialized scheduling cost.

## Implication

The codegen lane at the K anchor is bounded and this candidate does not move
the wall. Do not promote it. The next decision should be against the competing
ledgers (serialization/overlap vs useful-body mass), not more K-body tuning.

## Evidence

- `docs/task_workflow/evidence/nv-q4k-k-four-warp-route-20260822/result.json`
- `docs/task_workflow/evidence/nv-q4k-k-four-warp-route-20260822/result/control_a.json`
- `docs/task_workflow/evidence/nv-q4k-k-four-warp-route-20260822/result/candidate.json`
- `docs/task_workflow/evidence/nv-q4k-k-four-warp-route-20260822/result/control_c.json`

Measurement hook is fail-closed and default-off: no loader installs the
admission, so production decode is unchanged.
