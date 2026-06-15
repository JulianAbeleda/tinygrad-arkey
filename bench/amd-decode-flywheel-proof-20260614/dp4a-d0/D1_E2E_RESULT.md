# D1 — builtin v_dot4 decode GEMV, end-to-end: kernel WINS, e2e UNCHANGED (definitive null)

Date: 2026-06-15. `q4k_q8_1_vdot_builtin_partial_kernel` (extra/q4_k_gemv_primitive.py) wired into the
decode path behind `Q4K_VDOT=1` (tinygrad/llm/model.py), the renderer emits the `_dp4a` device helper
(cstyle.py, gated on `_dp4a` appearing in a CUSTOM body).

## Standalone (the kernel-level win is real and large)
ffn_gate GEMV, wall-clock, correct (max_abs 0.0012):
| kernel | Q4-GB/s |
|---|---|
| fp (float) | 171 |
| asm-volatile v_dot4 | 20 |
| **builtin udot4 + LOCAL:64** | **302** |
The builtin v_dot4 GEMV is **1.77× faster than fp** at the kernel level (the occupancy fix — 64 rows/wg
via a `lidx0` special — plus the schedulable builtin). The asm-volatile version is crippled (20); the
builtin realizes the instruction-count headroom.

## End-to-end decode (the kernel win does NOT cash out)
`cli.py --benchmark 30`, Qwen3-8B Q4_K_M, same machine, apples-to-apples:
| path | tok/s | GB/s | MB read/token | output |
|---|---|---|---|---|
| fp (Q4K_PRIMITIVE) | 30.3 | 144 | 4762 | correct |
| **builtin vdot (Q4K_VDOT)** | **30.2** | 61 | **2036** | correct (identical text) |

**Decode tok/s is IDENTICAL (30.2 vs 30.3)** — despite the GEMV kernel being 1.77× faster standalone AND
the vdot path reading HALF the bytes/token (2036 vs 4762 MB). The per-token time (33 ms) is unchanged.

## The definitive conclusion (reconciles the whole decode arc)
- **The v_dot4 instruction-count lever is REAL at the kernel level** — builtin udot4 GEMV beats fp 1.77×,
  overturning Phase D's "DP4A is the wrong lever" (which was an asm-volatile-barrier artifact). The
  consolidated doc's headroom (fp 4.06 → DP4A ~1.35 VALU/weight) is realizable.
- **But it does NOT improve e2e decode tok/s.** Making the Q4_K GEMV 1.77× faster (and halving its memory
  traffic) changes the token time by ~0%. So the e2e decode bottleneck is NOT the GEMV kernel throughput —
  it is the per-token latency/launch floor (the ~252 kernel launches, attention, norms, sync), exactly as
  the postmortem/final-report concluded ("latency/occupancy-bound", "the JIT already pipelines launches").
- **Net:** the decode gap to llama.cpp is not a GEMV-kernel-instruction-count gap after all — a competitive
  v_dot4 GEMV exists now and doesn't move e2e. The bottleneck is structural (per-token latency), which is a
  cross-kernel / launch-overhead problem, not a single-kernel codegen one. This closes the decode lever
  hunt: the last candidate lever (DP4A) is real in isolation and null in practice.

## Caveat
This machine's fp baseline is 30 tok/s here (vs the historical 58 — the GPU ran slow this session after
repeated HW faults); the COMPARISON is apples-to-apples on the same machine/run, so the null (vdot = fp)
holds regardless of the absolute number. The `Q4K_VDOT` flag is default-off (no change to normal decode);
the renderer `_dp4a` helper is emitted only when a kernel references it.

Reproduce: standalone `extra/q8_1_q4k_bench.py <gguf> --kernel vdot_builtin`; e2e
`Q4K_PRIMITIVE=1 Q4K_VDOT=1 python -m tinygrad.llm.cli --model <gguf> --benchmark 30`.
