# NV Q6_K attention-V four-warp fp16 promotion (2026-08-16)

Date: 2026-08-16
Branch: `nvidia-bringup-20260731`
Status: **promoted. WALL_PASS reverse bracket (-147.35 us/token, +3.07%),
production census 207.4 tok/s with the token sha unchanged.**

## 1. Why this is the next step

The reconciled 240 audit (`nv-240-audit-reconciled-20260815.md`) named the Q6
GEMV core as the highest-confidence remaining kernel-work lever: the Q6
attention-V partial shape was 3.65x llama in per-shape ratio (17.94 us vs
4.90 us in-loop) with the same four-warp fp16 geometry that already landed on
Q4 FFN-down (-100.3 us) and Q6 FFN-down (-39.0 us).  The load-pattern sweeps on
the V shape had all closed NO-GO (MC2 partial, q4kd), but the four-warp
geometry itself was never built for the 1024x4096 V shape.

## 2. What was implemented

`tinygrad/llm/q6k_v_mmvq.py`: closed-default `Q6KVFourWarpAdmission` and
`emit_q6k_v_four_warp_fp16_direct` (128 threads/row, four warps, fp16 FMA,
no Q8 provider, no parts buffer, one pass to a contiguous fp32[1024]).
`decode_routes.py` swaps the installed `q6k_gen_partial_1024_4096_4` parts
route 1:1 behind the explicit lease guard.  The V output binds to the kv-store
route as a flat fp32 view (`vparts=1`), so no parts reduce is left behind.

## 3. Measurement gates

| gate | result |
| --- | ---: |
| standalone device time (DEBUG=2 single-node) | 4.10 us vs 44.06 us control |
| in-loop census (prime token, d512) | 5.12 us vs 17.94 us control, 10/10 blocks |
| program count | 596 == 596 (no side-effect shift) |
| token sha | `227ad3ce...` identical control/candidate |
| reverse wall bracket | A 4.9545 / cand 4.8058 / C 4.9518 ms, **-147.35 us**, +3.07% |
| promotion bar | -147.35 us >= +50 us bar: **WALL_PASS** |

## 4. Tok/s translation

Production census (promoted state, same d512 harness): 201.516 -> 207.427
tok/s (**+5.9 tok/s**, ~-141 us/token).  Q6 attention-V total dropped from
180.5 us to 51.0 us per token; the 8 shared-Q8 V blocks are untouched
(4.28 us, at/below llama).  The V class is now at parity with llama's Q6 V
(~89.4 us floor).

## 5. Remaining open rows (the audit's next levers)

- flash score: 36 x 6.56 us = 243 us vs llama 113.9 us (structural floor ~90 us)
- Q4 FFN-down body +196 us vs llama: load-pattern sweep closed NO-GO; DP4A
  producer-fold NO-GO (measured)
- launch hiding / support-kernel overlap: llama hides ~946 us; tinygrad hides 0
  (Route B multi-stream measured FLAT at HEAD)

## 6. References

- `nv-240-audit-reconciled-20260815.md` (Q6 core as the top lever)
- `nv-q6k-1024x4096-native-nv-direct-microgate-record-20260805.md` (the
  earlier direct-output attempt that failed on the coop spelling)
- `nv-q6k-ffn-down-four-warp-fp16-promotion-scope-20260816.md` (the FFN-down
  sibling promotion this route mirrors)
- evidence: `/tmp/q6k_v_control_census.json`, `/tmp/q6k_v_candidate_census.json`,
  `/tmp/q6k_v_four_warp_wall_bracket.json`, `/tmp/census_q6kv_promoted.json`
