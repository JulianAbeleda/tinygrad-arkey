# Prefill weight-reuse — PWR-0/PWR-1 RESULT → **quant-weight-reuse has no room; the prefill lever is fp16 WMMA LDS-tiling (DEFERRED)**

Executed PWR-0 (baseline refresh) + PWR-1 (component target selection) per `qk-prefill-weight-reuse-scope-20260618.md`,
**before** writing any kernel. Outcome: the scope's PWR-1 gate is **not** met by the quantized-weight-reuse
primitive — the dominant prefill cost is the **fp16 WMMA matmul (already tensor-core)**, bottlenecked by **LDS
tiling**, not quant reuse. Stop before building; redirect. gfx1100, Qwen3-8B-Q4_K_M.

## PWR-0 — warm prefill baseline (authoritative)

`extra/qk_prefill_v2_measure.py`, pp512 warm:

| mode | tok/s | ms/512 |
|---|---:|---:|
| default (symbolic v_toks, lazy Q4_K→fp16 dequant) | 184 | 5.44 ms/tok |
| **PREFILL_V2** (concrete 512, fp16 realized weights + warmstart-TC) | **2085** | 245 | 

PREFILL_V2 = 11.35× warm, warmstart apply=5/error=0. (~70–83% of llama prefill per prior banks.) **PREFILL_V2 is
the fast path and the right baseline.**

## PWR-1 — component breakdown of the PREFILL_V2 forward (N=512)

`extra/qk_prefill_component_breakdown.py` (first-call DEBUG=2 per-kernel shares, directional; warm tok/s
authoritative). Corrected classification (the big `r_…_256_2`/`_768_2` kernels — x71 ffn, x35 per-layer — are the
**fp16 matmuls**, mis-bucketed as reduce_other; the `(start_pos+512)` kernels are **attention**):

| component | share | note |
|---|---:|---|
| **matmul (fp16 WMMA: FFN gate/up/down + attn QKVO)** | **~74%** | top: `r_16_192_32_…_256_2` 42.6% x71 (ffn) |
| attention (SDPA over 512×(512+T)) | ~24% | `r_2_512_(start_pos+512)_…` 14.6% x35 |
| norm / RoPE / SwiGLU | ~2% | flat |

**The matmuls already use WMMA** (verified: 509 `__builtin_amdgcn_wmma` calls in the emitted source; both dominant
ffn kernels have `wmma` in body). So TC is *on*; the gap to llama is **WMMA operand LDS-tiling / cache-blocking** —
the exact prefill-plan finding (`amd-decode-prefill-plan.md`: tinygrad matmul emits WMMA but **LDS=0**, re-reads
operands from global per WMMA op → bandwidth-bound; llama rocBLAS/Tensile stages a 128×128 tile in 25.6KB LDS →
compute-bound ~80%). The lever is **LDS cache-blocking of the WMMA matmul** (Boehm step 2) — a GROUP/LOCAL-into-LDS
opt that BEAM finds but **hangs gfx1100**, or rocBLAS/hipBLASLt.

## Decision-gate disposition

Scope PWR-1 gate: a component/shared-primitive must be ≥30% of warm prefill AND a 2× component win must imply ≥1.2×
full pp. The **matmul** clears the share bar (74%, 2×→~1.6× full pp) — **but the matmul is already fp16-WMMA**, so
the primitive that wins it is **fp16 matmul LDS-tiling**, NOT the quantized-weight-reuse primitive this arc scoped:

- **Quantized-weight-reuse has no Amdahl room in 8B PREFILL_V2.** The weights are already realized fp16 and fed
  through WMMA; there is no in-forward Q4_K dequant to amortize. Quant reuse would only matter **VRAM-frugally**
  for 14B/32B (where the fp16 materialize OOMs) — excluded by the standing no-pivot-to-14B/32B preference — and even
  there the WMMA LDS-tiling wall is the actual bottleneck.
- The scope's stop rule applies: *"the only viable route requires broad forward restructuring before any local
  component clears a gate"* / *"isolated weight reuse does not beat the current prefill-shaped linear"* — the current
  linear is already fp16-WMMA, so a quant-reuse kernel cannot beat it ≥1.5×.

## Verdict: **DEFERRED** — redirect from quant-weight-reuse to fp16 WMMA LDS-tiling (or rocBLAS)

The prefill win exists (~1.6× from a 2× matmul) but it is **not** a quantized-weight-reuse primitive — it is the
**fp16 WMMA LDS-tiling** lever, the documented gfx1100 BEAM-hang / rocBLAS-call wall (PWR-5 territory: it changes the
authority boundary — tinygrad codegen vs external BLAS / raw HIP). **Did NOT build a tiled quant kernel** (the PWR-1
breakdown disproved its room, exactly as the scope's Next-action required). The weight-reuse-primitive line is
**closed for 8B prefill** (subsumed by PREFILL_V2's fp16-WMMA); the surviving prefill lever is a separate
matmul-codegen/BLAS arc.

## Recommended next (if prefill is funded)
1. **rocBLAS/hipBLASLt bridge** for the fp16 prefill matmuls (PWR-5 option a) — lowest-risk path to ~80% peak,
   changes the authority boundary but sidesteps the BEAM-hang.
2. or **LDS-tiled WMMA matmul codegen** (a GROUP/LOCAL→LDS opt without BEAM) — the durable tinygrad-internal fix,
   reuses the LDS-tiling assets (`extra/gemm/amd_uop_matmul.py`, the WR1–3 warp-reduce + LDS-tiling arc).
Both are general fp16-matmul arcs (help prefill broadly), not Q4_K-specific. Attention (~24%) is the secondary
lever (flash-prefill, separately refuted/LDS-walled).

## Files
`extra/qk_prefill_v2_measure.py` (PWR-0), `extra/qk_prefill_component_breakdown.py` (PWR-1),
`bench/qk-prefill-weight-reuse-20260618/{pwr0,pwr1-components}.json` (gitignored). Precedent: `amd-decode-prefill-plan.md`
(the WMMA-LDS=0 root cause). No kernel/model/default changes.
