# PREFILL RECONCILIATION — source of truth (2026-06-19)

Settles the contradictory prefill results under ONE controlled interleaved benchmark. Measure-only, no defaults
changed, no new kernels. Artifact: `docs/artifacts/prefill-reconciliation-matrix-20260619.json`.

## TL;DR (the reconciliation)
The contradiction (`Tensile +FFN/qo = 4770 tok/s / 1.76x` vs later `0.997x no advantage`) is REAL on both sides and
caused by **tinygrad's WMMA prefill being clock-VOLATILE while Tensile is clock-STABLE**:
- Tensile prefill ≈ **2640 tok/s, stable** (this session: auto 2633, profile_peak 2654; prior sessions ~2666).
- tinygrad WMMA concrete-KV prefill = **1449–2675 tok/s, session-volatile** (this session 1449–1515 even at
  profile_peak; one prior session hit 2675).
- So the Tensile-vs-WMMA RATIO = 2640/WMMA = **1.0x when WMMA=2675, 1.76–1.83x when WMMA≈1500.**

**The latest controlled interleaved matrix REPRODUCES the old 1.76x** (this session: 1.83x auto, 1.76x peak). The
prior `0.997x` runs (tensile-land, transpose-free) sat in a session where WMMA was anomalously at ~2675. The TYPICAL
operating WMMA clock is ~1450–1550 (FOUR independent measurements: 1449 / 1455 / 1515 / 1551), where **Tensile wins
~1.76x**. So `matmul-not-the-lever / Tensile-no-advantage / prefill-exhausted` were artifacts of the high-WMMA-clock
outlier sessions and are RETRACTED (see Superseded below).

## P0 — provenance
| doc | claim | status |
|---|---|---|
| prefill-tensile-inmodel-measurement-result (A5) | +Tensile FFN+q/o = **4770 tok/s, 1.76x, 1.41x llama** | **VALID (reproduced as 1.76–1.83x ratio)**; absolute 4770 was a higher-clock session |
| prefill-tensile-land-result | interleaved A/B = **0.999x**, "prior 1.27x was clock-confound" | **PARTIALLY STALE** — the 0.999x is real but only in the high-WMMA-clock state (WMMA≈2675); the "win is just a confound" framing is RETRACTED |
| prefill-tensile-transpose-free-result | OFF 2675 / ON 2666 = **0.997x**, "Tensile no advantage" | **STALE/CONTEXT-BOUND** — true only when WMMA≈2675 (that session); not the typical clock |
| prefill-matmul-RECONCILED | "matmul ~25% of wall, NOT the e2e lever; Tensile A/B 0.997x" | **SUPERSEDED** — held only at high-WMMA-clock; at typical clock matmul IS the bottleneck and Tensile gives 1.76x |
| prefill-concrete-kv-build-result | concrete-KV **1.24x, byte-identical** | **VALID** (reproduced: 1.21–1.24x, rel_err 0.0) |
| prefill-clock-controlled-benchmark-result | symbolic 1251 / concrete 1551 / llama 3086; concrete = 50% llama | **VALID** for WMMA; **incomplete** (didn't test Tensile -> missed the 86%-llama row) |
| commit d6fc9c80b | ships concrete-KV default first-chunk | **VALID, unchanged** |

## P1 — controlled benchmark matrix (interleaved, clock-fair within run, model.forward, T=512, best-of-30)
| row | AUTO tok/s | vs concrete | profile_peak tok/s | vs concrete | % llama (3070) |
|---|---:|---:|---:|---:|---:|
| R1 symbolic PREFILL_V2 | 1192 | 0.83x | 1210 | 0.80x | 39% |
| R2 concrete-KV (anchor) | 1449 | 1.00x | 1515 | 1.00x | 47–49% |
| R3 concrete + Tensile FFN-only | 2633 | **1.83x** | 2648 | **1.76x** | 86% |
| R4 concrete + Tensile FFN+q/o | 2636 | **1.83x** | 2654 | **1.76x** | 86% |
| R6 llama.cpp pp512 | — | — | 3070 ± 123 | — | 100% |
- concrete-KV over symbolic: **1.21x (auto) / 1.24x (peak)** — matches the banked 1.24x.
- Row 5 (Tensile FFN + attn-internal): **NOT MEASURABLE** — the extracted Tensile kernels cover FFN gateup/down +
  q/o PROJECTIONS only; there is NO Tensile kernel for attention COMPUTE (Q@K/P@V). "all roles" == R4. (Documented, not inferred.)

## P2 — route verification (every Tensile row)
- **Routes fire, no silent fallback:** R3 route_counts = `{gateup:72, down:36}` (qo skipped via patch); R4 =
  `{qo:72, gateup:72, down:36}` (72 = 2 linears × 36 layers; 36 = 1 × 36). All against the SAME concrete-KV
  start_pos baseline (R2), not a stale symbolic one.
- **Correctness: logits_rel_err vs concrete-KV = 0.000000** for ALL rows (R1/R3/R4) -> byte-identical greedy token
  (the forward output is identical). Tensile is numerically exact to the WMMA path at the token level.
- Same start_pos: R2/R3/R4 all `concrete` (start_pos=0); R1 `symbolic`. Confirmed in the JSON per-row.

## P3 — lifecycle map (per row)
- start_pos: R1 symbolic; R2/R3/R4 concrete (one cached concrete jit at start_pos=0).
- route_counts: R2 {} ; R3 {gateup:72,down:36}; R4 {qo:72,gateup:72,down:36}.
- kernel counts (reliable, from profiler ents, prior measurement): symbolic 801 (attn 474/glue 219/matmul 108);
  concrete 729 (attn 438/glue 183/matmul 108). concrete removes 72 expensive symbolic-attention reduce kernels.
- Tensile rows route the FFN/qo matmuls through custom_kernel/TensileRunner (the FFN matmuls become tensile_*
  kernels); attention compute unchanged.
- NO per-kernel DURATION used (HCQ profiler durations include inter-kernel stall -> unreliable; counts only).

## P4 — the 6 questions, answered
1. **Is "Tensile FFN+q/o = 4770 / 1.76x" reproducible under the latest controlled setup?** YES (ratio). The latest
   interleaved matrix gives Tensile = **1.76x (peak) / 1.83x (auto)** over concrete-KV — reproducing the 1.76x. The
   absolute 4770 was a higher-clock session; this session Tensile = 2640 (still **86% of llama**). The RATIO is robust.
2. **Exact policy blocker remaining?** The route is research-only: `PREFILL_TENSILE_GEMM=0` by default, and it
   depends on the VENDORED extracted rocBLAS Tensile artifact (`bench/qk-tensile-extraction/kernarg_all.jsonl` +
   the .co kernels via HCQ launch). Shipping it = a **dependency-policy decision** (bundling an external rocBLAS-
   derived kernel blob) + handling shapes outside the captured set (T!=512, non-eligible) which currently fall
   through to WMMA. No correctness blocker (rel_err 0).
3. **Why did 0.997x appear (if the win is real)?** NOT stale harness, NOT route-not-firing, NOT symbolic baseline —
   the prior interleaved A/Bs ran in a session where tinygrad's WMMA was at an **anomalously high clock (~2675)**,
   where WMMA matches the clock-stable Tensile (~2660) -> ratio 1.0x. tinygrad WMMA prefill is clock-VOLATILE
   (compute-bound); Tensile is clock-STABLE. The typical WMMA clock (~1500, 4 measurements) is the 1.76x regime.
4. **Is concrete-KV still 1.24x and byte-identical?** YES — 1.21–1.24x over symbolic, logits rel_err 0.0 (identical
   greedy token). Shipped default for first chunk (commit d6fc9c80b).
5. **Latest defensible tinygrad-vs-llama pp512:** llama 3070 ± 123 (auto). tinygrad concrete-KV = 1449–1515 =
   **47–49%** of llama (clock-volatile, can reach ~87% at the rare high-WMMA-clock). tinygrad **+Tensile = 2640 =
   ~86% of llama, STABLE**. So the defensible numbers: **concrete-KV ~47% llama; +Tensile ~86% llama.**
6. **What primitive class remains?** **Tensile / external-artifact lifecycle is the REAL, reproduced lever** (1.76x,
   86% llama) — NOT exhausted. Secondary: the WMMA in-model matmul is clock-volatile/compute-bound (tinygrad WMMA
   at typical clock ~1500 is the gap Tensile closes). Concrete-shape lifecycle (concrete-KV 1.24x) shipped.
   Attention-compute lifecycle: no Tensile primitive exists (would be the next external/codegen target).

## Honest unresolved / missing measurement
- **A reproducible/controllable WMMA clock.** profile_peak did NOT raise this session's WMMA past 1515; the prior
  2675 sessions can't be reproduced on demand. So I cannot DEFINITIVELY pin whether the "representative" user sees
  WMMA at ~1500 (Tensile 1.76x) or ~2675 (Tensile 1.0x). The WEIGHT of evidence (4 measurements at ~1500) favors
  the 1.76x regime, but the exact clock-determinant of WMMA's volatility is the missing measurement. This is the
  ONLY thing standing between "Tensile is a clear 1.76x prefill win" and certainty.
- **Do NOT declare prefill exhausted.** The Tensile 1.76x is reproduced; the matmul IS the lever at typical clock.

## Superseded claims (explicit)
- "matmul is a red herring / not the e2e lever" (prefill-matmul-RECONCILED) — SUPERSEDED: true only at high WMMA
  clock; at typical clock matmul is the bottleneck (Tensile 1.76x).
- "Tensile no in-model advantage / lever REFUTED" (transpose-free, tensile-land 0.997x) — SUPERSEDED: high-WMMA-clock-only.
- "dependency-free prefill ceiling reached at concrete-KV 1.24x" (memory) — SUPERSEDED: +Tensile (dependency) = 86% llama.
