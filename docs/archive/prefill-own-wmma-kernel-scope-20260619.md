# Scope — our own WMMA prefill kernel (no external deps): make the tensor cores the bottleneck

Original plan, constrained to **pure tinygrad / no external dependencies** (rocBLAS path declined). Builds on
PWLT-A2 (`prefill-wmma-lds-tiling-result-20260619.md`) with its key, under-exploited finding. Authority: DEBUG=2
device time (kernel) + warm pp (model) + fp16 dNLL. No route/default until a gate passes.

> **RESULT:** executed in `prefill-own-wmma-kernel-result-20260619.md`. POWN-1 **KILL**: best config remains
> 42.0 TFLOPS, below the 62 TFLOPS gate; more waves, bigger tiles, BK32, and noLDS all regress. This scope is kept
> as provenance for the no-deps route that was tested and closed.

Update from `prefill-external-blas-result-20260619.md`: the external ceiling is now measured as a control, not just
assumed. hipBLASLt reaches **69.8 TFLOPS** on ffn_gate/up (1.71× tinygrad), and rocBLAS reaches **70.9/76.7 TFLOPS**
on ffn_down/attn_q/o. The pure-tinygrad target is therefore "break the ~41 TFLOPS plateau toward ~70 TFLOPS", not
the earlier optimistic ~98 TFLOPS / 80%-peak guess.

## The learning that reframes this (why it's not "reimplement Tensile")

PWLT-A2 measured, on the ffn shape (512×4096→12288 fp16):
- tinygrad **WMMA** matmul: **41 TFLOPS** (~34% of ~122 WMMA peak)
- tinygrad **non-WMMA ALU** matmul: **40 TFLOPS** (~66% of ~61 fp16-ALU peak)
- LDS-tiled vs non-LDS WMMA: identical (IC-served; LDS is not the lever).

**The WMMA path only *matches* the ALU path** — it gets **none of the ~2× tensor-core advantage**. So the WMMA
units are **stalled, not the bottleneck**. That means the headroom is not "better memory tiling" (refuted) and not
"a Tensile-class library" (declined) — it is **making the WMMA instructions issue densely enough to become
compute-bound**: occupancy, accumulator depth (independent WMMA ops in flight), K-loop structure, and overlap of
loads with WMMA. Those are **config + kernel-structure levers in our own custom_kernel**, bounded (days), no deps.

## Hypothesis
A WMMA custom_kernel tuned for **dense WMMA issue + enough independent accumulators + high occupancy** (and with LDS
*dropped* since it's IC-served, freeing registers/occupancy) can push past the 34% plateau toward the WMMA roofline
— or we prove tinygrad's WMMA codegen has a structural ceiling on these shapes.

## Primitive
`prefill_wmma_dense_issue_gemm` (pure tinygrad custom_kernel, fp16→fp32). Boundary: fp16 realized weights →
global→reg→WMMA (LDS optional) → fp32 accumulate → output. Built on `extra/gemm/amd_copy_matmul.py` (the proven
SHAPED_WMMA kernel) as the starting point.

## Phase POWN-0 — diagnose WHY tinygrad WMMA is stuck at 34% (decisive)
Before tuning, name the bottleneck. For the current WMMA kernel on the ffn shape, measure/derive:
- VGPR + LDS per thread → **achieved occupancy** (waves/CU vs the gfx1100 max);
- WMMA ops per thread per K-iter and the **accumulator dependency chain** depth (are WMMA ops independent or
  serialized on one accumulator?);
- threads/block (amd_copy_matmul WMMA = 128) and blocks/CU;
- whether loads overlap WMMA or stall it (no software pipelining in the kernel today);
- if `rocprof`/omniperf is usable: VALU/MFMA-issue + stall reasons; else infer from VGPR/occupancy + DEBUG=2 timing.
**Gate to proceed:** the bottleneck is named as occupancy- / issue- / accumulator-chain-bound (i.e. *addressable by
config/structure*). **Kill:** if it's a hard codegen-emission limit (e.g. tinygrad emits WMMA with a forced
serializing accumulator and no way to widen) → bank as a tinygrad WMMA-codegen ceiling; rest at PREFILL_V2.

## Phase POWN-1 — config sweep using the learnings (LDS-off, occupancy-first)
Sweep the WMMA kernel knobs on the ffn shape (and the 3 other prefill shapes), DEBUG=2 device time, correctness vs
fp16 oracle each time:
- **LDS on vs OFF** — IC serves operands, so an LDS-free global→reg→WMMA variant should match speed AND free LDS →
  higher occupancy. Test both.
- **threads/block:** 128 / 256 / 512 (more waves/block for latency hiding).
- **waves M×N and macro-tile (BLOCK_M/N, TM/TN):** enough independent WMMA accumulators per wave to hide WMMA
  latency (deeper accumulator array = more WMMA ops in flight).
- **K-loop unroll / multiple accumulators** to break the single-accumulator dependency chain (the WR3 lesson: a
  serial accumulator stalls; independent partial accumulators issue densely).
- **BLOCK_K:** keep small (sweep showed 16 best, 32/64 regress) — confirm.
**Gate:** any config reaches **≥50% WMMA peak (~62 TFLOPS, ≥1.5× current)** on the dominant ffn shape, exact.
**Kill:** all configs plateau ≤~40 TFLOPS → tinygrad WMMA codegen caps at ALU-path speed on these shapes (the
honest ceiling); bank, rest at PREFILL_V2.

## Phase POWN-2 — structural improvement (only if POWN-1 shows movement)
If a config breaks ~34% but stalls below ~70%, add the structure rocBLAS uses, as far as tinygrad expresses it:
- **software pipelining / double-buffer** the global→reg load to overlap with WMMA (express via `.after`/staging);
- **denser WMMA issue** (reorder so independent WMMA ops are adjacent);
- per-shape macro-tile selection (ffn vs attn shapes differ).
**Gate:** ≥1.5× current matmul isolated; **stretch:** ≥70% peak.

## Phase POWN-3 — in-model warm pp (authority)
Route the prefill matmuls through the winning kernel behind `PREFILL_OWN_WMMA=1` (no default flip, no decode change).
Gate: **≥1.5× full warm pp512**, no decode regression, fp16 dNLL ≤0.01 (PREFILL_V2 passes), unsupported-shape
fallback to PREFILL_V2's matmul. Kill: isolated win doesn't transfer (classify the layer).

## Non-negotiable gates
- correctness: fp16 mse within tol vs oracle each kernel; dNLL ≤0.01 in-model; no decode regression.
- performance: POWN-1 isolated ≥1.5× current matmul (else honest ceiling kill); POWN-3 in-model ≥1.5× warm pp.
- principles: pure tinygrad (no deps); diagnostic≠shipped; opt-in flag; DEBUG=2 device time; document the ceiling
  if we hit one (refutations are assets).

## Expected outcomes (honest odds)
- **Best (uncertain):** a denser-issue / higher-occupancy WMMA config breaks to ~60-80% peak → ~1.5-2.3× prefill
  matmul → ~1.3-1.6× pp candidate, pure tinygrad.
- **Most likely:** config sweep moves it somewhat (e.g. 34%→~45-50%) but stalls below rocBLAS because tinygrad's
  custom_kernel can't express the software-pipelined, assembly-scheduled WMMA loop rocBLAS uses → a **partial win
  or a named codegen ceiling**.
- **Cheap kill:** POWN-0 shows the 34% is a forced-serial-accumulator / emission limit, or POWN-1 plateaus at
  ALU-path speed → **bank the tinygrad WMMA-codegen ceiling on gfx1100 prefill shapes**; rest at PREFILL_V2.

## Main risk (stated plainly)
The 34%→80% gap may be **exactly the software-pipelining + instruction-scheduling that rocBLAS hand-writes in
assembly and tinygrad's codegen does not express**. If so, our own kernel plateaus and the honest result is "pure-
tinygrad WMMA tops out at ~X% here." That is still a valuable, durable answer (and POWN-0 surfaces it cheaply before
deep tuning). This is bounded research (days), not the months a full Tensile reimplementation would take — because
we are tuning the *existing* proven WMMA kernel, not writing one from scratch.

## Files (planned)
`extra/qk_prefill_wmma_sweep.py` (POWN-0 diagnose + POWN-1/2 config sweep, building on `extra/gemm/amd_copy_matmul.py`),
`bench/qk-prefill-own-wmma/`, `extra/qk_prefill_external_gemm_bridge.py`→ N/A (no deps; route in `model.py` behind
`PREFILL_OWN_WMMA` for POWN-3), `docs/prefill-own-wmma-kernel-result-20260619.md`. Commit: `[test]` sweep + bench,
`[codegen]` if a renderer change is needed for denser WMMA issue, `[nn]` the flag route, `[docs]` verdict.

## Sequencing
**POWN-0 → POWN-1 first, then report.** POWN-0 tells us if the 34% is addressable (config) or a hard ceiling
(emission); POWN-1's sweep either breaks ≥1.5× or establishes the ceiling. Don't do POWN-2 structural work until the
diagnosis says config can move it.
