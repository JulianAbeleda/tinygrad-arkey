# NV performance campaign - get tinygrad up to llama.cpp speed

Date: 2026-08-01

Status: scoped, not implemented. Branch boundary: tinygrad `nvidia-bringup-20260731`. Does not
authorize promotion to `dev`/`master`.

Companion to `docs/nv-prefill-decode-diagnosis-20260801.md` (all measurements) and
`docs/task_workflow/input/nv-prefill-gemm-promotion-scope-20260801.md` (P1 detail). This is the
umbrella campaign: every lever between current NV performance and llama.cpp's measured
performance on the same machine, sized, ordered, gated.

## 1. The target, measured in-session on the RTX 5090

Same model (Qwen3-8B-Q4_K_M), same session, `llama-bench` CUDA build `ac4cddeb0`:

| phase | tinygrad now | llama.cpp | BoltBeam ceiling | gap |
| --- | ---: | ---: | ---: | ---: |
| prefill pp512 | ~101-115 tok/s | 14,250 tok/s | 13,664 tok/s | ~125x |
| prefill pp1024 | - | 14,633 | - | - |
| prefill pp2048 | - | 14,342 | - | - |
| prefill pp4096 | - | 13,801 | - | - |
| decode d512 | 158.2 tok/s | 237.1 tok/s | 383.6 tok/s | 1.5x |
| decode d2048 | - | 225.7 | - | - |
| decode d4096 | - | 217.0 | - | - |

Note the ceiling check: llama exceeds the modeled 13,664 ceiling at pp1024/2048, so the modeled
180 TFLOPS fact is conservative; measured `R` (matrix-unit rate) may revise the ceiling upward.
P0 measures it.

## 2. The levers, sized by evidence

### L1 - Prefill dense GEMMs do not reach the matrix unit (the ~125x)

89% of traced prefill GPU time is three scalarized Q4_K GEMM routes (`r_16_256_...` gate_up 47%,
`r_16_64_..._48_...` ffn_down Q4_K 22%, `r_16_64_..._16_...` 20%), running 6-24 TFLOPS. Generated
CUDA has no `mma.sync`/`dp4a`/`half2`. llama dequantizes to fp16 and reaches `mma`; the sm120
full-kernel candidate set exists, compiles on the 5090 (C5, `948b26318`), but is not promoted.

Lever: promote the sm120 candidate set (fused Q4_K dequant + `cuda_mma`), exactly the path AMD
uses with WMMA full-kernel candidates. This is `nv-prefill-gemm-promotion-scope-20260801.md`.

### L2 - Q6_K and lm_head prefill routes remain on the scalar path

The sm120 candidate set covers 4 roles, all Q4_K-shaped. Q6_K roles (ffn_down Q6_K = 22% of
traced prefill time, attn_v, lm_head) are not covered; their prefill GEMMs stay scalarized.
AMD's promoted set has the same 4-role shape, so this is also an AMD gap, but on NV it is
visible today in the trace (the `_48_` route is 7.8ms/call).

Lever: extend promotion coverage to Q6_K shapes, or route Q6_K prefill GEMMs through the fp16
overlay (`route_pf16_graph_gemm` / `_install_candidate_matmul` TC warmstart) so dequant happens
once and the GEMM is a plain fp16 tensor-core matmul.

### L3 - Warm prefill is launch/sync-bound on top of slow kernels

Warm wall is ~4.3s; the cold trace sums to ~642ms GPU busy. cProfile: 4.31s of 4.39s in GPU
`wait()`, 1.35M `to_mv` calls, ~2,000 kernels per iteration with per-kernel gaps. Even after L1
removes the slow kernels, the wall will not collapse unless launch overhead is addressed.

Lever: graph replay / JIT batching / kernel-count reduction for the prefill schedule; confirm the
warm-path kernel mix (the trace's tail `batched 32..512` kernels need identification - they run
at 13-16 TFLOPS and may already be a replay path).

### L4 - Decode GEMV efficiency: 41% of the bandwidth ceiling vs llama's 66%

Decode is bandwidth-bound in the right regime (158.2 tok/s = 764.8 GB/s of 1792) but llama gets
237-254 tok/s on the same phase. Per-token kernels are tiny (9-50us) and numerous
(`q4k_g3_lanemap_gemv_*`, `q6k_gen_coop_*`, `flash_block_tiled_xlane_score_*`); launch overhead
and per-kernel efficiency dominate, not the roofline.

Lever: reduce per-token kernel count / launch cost, widen vector loads, tune occupancy for the
GEMV shapes, and re-check the flash-decode score/PV kernels against llama's per-token budget
(llama d512 = 4.22ms/token vs our 6.32ms).

### L5 - Measurement discipline (not a perf lever, the gate for all levers)

AMD control re-run after every commit; first-token digits and decode sha256 pinned; bench rows
carry route/census; paired llama runs in the same session. P5 formalizes the control matrix.

## 3. The work - phases with HARD STOPs

### P0 - Measure NV facts (`R`, `BW`), re-derive ceilings

The bring-up method's Phase 0: isolated matrix-unit microbenchmark (back-to-back ops, multiple
accumulators, zero loads in the loop, disassembly-verified) and a streaming bandwidth benchmark
at prefill/decode sizes. Replace BoltBeam's modeled 180 TFLOPS / 1792 GB/s with measured facts.
llama exceeding the modeled prefill ceiling is the signal that the fact is wrong, not that llama
is impossible.

Deliverable: measured `R`, `BW`, matrix-unit shape, hard limits; revised ceiling table; the `M*`
crossover for decode/prefill. No perf work starts before this.

### P1 - Prefill GEMM promotion (L1) - the main lever

Execute `nv-prefill-gemm-promotion-scope-20260801.md` P0-P3: mint NV compact artifact from the
typed sm120 schedule, target-parametric promotion selection, census/bench-row wiring, first-token
digits + AMD control, then measured head-to-head at pp512/1024/2048/4096.

Gate: prefill pp512 moves from ~110 to the 10k+ tok/s regime with identical first tokens.
Expected ceiling after P0's measured `R`.

### P2 - Q6_K prefill coverage (L2)

After P1 proves the mechanism, extend to Q6_K roles: either mint Q6_K candidates or route them
through the fp16 overlay TC warmstart. Measure ffn_down Q6_K before/after; the 18-call 7.8ms
route is the before number.

### P3 - Prefill launch overhead (L3)

Re-measure warm wall vs GPU busy after P1/P2. If host time still dominates, work the replay
path: identify the `batched` kernels, reduce kernel count per prefill, enable/verify graph
capture on the whole prefill schedule. Target: warm wall tracks GPU busy, not 3.6s of fixed
overhead.

### P4 - Decode GEMV efficiency (L4)

Per-token kernel budget: count kernels/token, time each (9-50us today), then reduce launch cost
and tune the GEMV shapes (vector width, occupancy, score/PV kernel fusion). Target: d512 >= 237
tok/s (llama), d2048 >= 225, d4096 >= 217, with decode correctness pinned.

### P5 - Full control matrix and closeout

Same-session paired run: tinygrad vs llama across the whole context sweep (pp128-4096, d512-
d4096), all bench rows with census, AMD control, decode sha256, first-token digits. Report the
table from section 1 with final numbers and the per-lever attribution (ms recovered per lever).

## 4. Guardrails

- No commits to `master`/`dev`/`exp`; all work on `nvidia-bringup-20260731`. Prefixes
  `[nn]`/`[codegen]`/`[test]`/`[docs]` only, one per commit, never mix NFC with functional.
- No `if backend == "NV"` branches in lowering; data lookups only (declared per-target facts).
- No touching `tinygrad/llm/prefill_routes.py` or the per-call dispatch path (parked scope).
- No dtype/precision cleanup (parked: `dtype-authority-decomposition-scope-20260731.md`).
- Do not revert `948b26318` / C5; do not hand-edit candidate JSONs - use the mint.
- AMD control re-run after every commit; NV first tokens must not move; if they do, STOP and
  report the delta and the exact diff.
- 5090 is shared: serialize GPU work, bounded runs, no background processes left behind.
- Every claim is a pytest result or a measured number; never "this should be fine".

## 5. Deliverable + HARD STOP

A measured table: tinygrad vs llama.cpp across prefill pp128-4096 and decode d512-d4096 on the
5090, with per-lever attribution and every row backed by a bench artifact. Hard stop after P5's
report for review; nothing beyond.

## 6. One-line job

Measure NV facts, promote the sm120 tensor-core candidates to production prefill, cover Q6_K,
kill the launch overhead, and tune decode GEMVs until the paired llama sweep is matched.
