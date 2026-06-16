# Prefill optimization plan (P0-P3)

Date: 2026-06-16. The decode arc deliberately excluded prefill ("keep the dense path for prefill"); it was
never scoped, profiled, or attacked. Measured this session: prefill ~65 tok/s vs llama.cpp ~3000 tok/s
(`llama-bench -ngl 99 -p 512,1024,3072 -n 0`) = **~2% of llama, ~45x behind, ~1% of fp16 peak** — by far the
worst gap vs llama (decode is ~58%). For a long prompt this is the time-to-first-token cost (~15s for 1000
tokens vs llama's ~0.3s).

## Root cause (hypothesis)
The default prefill runs the dense path (`decode_enabled = batched or (not is_prefill)` -> off during
prefill), where the **Q4_K dequant is fused into the matmul**. The GEMM operand is therefore not clean fp16
-> **RDNA3 WMMA cannot fire** -> the matmul runs scalar/untiled (~1% of peak). This is the same
"fusion-blocks-WMMA" wall as the decode W2 finding — but prefill is the regime where fixing it PAYS, because
prefill is compute-bound at batch-32 where tensor cores are the right tool (per `amd-decode-option1-result.md`:
"TC would only pay at LARGE batch — prefill, K>=64+, matmul-dominant").

## Building blocks (all built, validated in isolation, never wired to prefill)
- `extra/qk_matmul_decoded.py` — dequant Q4_K->fp16 then NATIVE matmul; dequant amortizes over the batch.
- `Q4K_UNFUSE` flag (model.py feed_forward) — casts FFN matmuls to fp16 "so RDNA3 WMMA can apply" (FFN only;
  attn projections excluded).
- the loop (N1/N2/L0/L1) — tunes native fp16 matmul to 33-98% of peak, live, 42x; its substrate IS prefill.
- TC/WMMA — the right tool at batch-32; never tested on the prefill path.

## Staged plan (gated, cheapest-first)
- **P0 — diagnose.** DEBUG profile one 32-token prefill chunk on 8B. Confirm matmuls dominate, measure
  achieved TFLOP/s vs the 83.6 fp16 peak, confirm WMMA does NOT fire. Gate: matmuls <30% of peak + no WMMA.
- **P1 — make-or-break: unfuse -> fp16 -> TC.** Route prefill linears through dequant->fp16 so WMMA fires.
  Gate: TC fires AND prefill >=5x faster. Risk: PADTO blowup (option1's 12288=256x16x3 -> pad-to-16 ~5x waste)
  at batch 32 — check pad efficiency.
- **P2 — tune.** Warm-start the loop's known-good native-matmul schedules onto the prefill matmuls (NOT native
  BEAM — it hangs gfx1100; use the curated loop / warm-start hook).
- **P3 — measure + decide.** Prefill tok/s vs llama 3000, token parity. Pre-register: >=10x (~650 tok/s,
  ~22% of llama) = ship; parity is the stretch.

## Out of scope / caveats
- Decode (separate problem, hand-asm wall, flag-exhausted).
- Not bit-exact: fp16 matmul accumulation differs from the fused path -> gate on TOKEN PARITY, not byte-identity.
- BEAM hangs gfx1100 — use warm-start, not native BEAM.

## RESULTS

### P0 — diagnose (2026-06-16): PASS, headroom confirmed
Prefill (8B, warm, 256-token) = **68 tok/s = ~1.11 TFLOP/s = 1.3% of the 83.6 fp16 peak**. WMMA does NOT
fire (`DEBUG=4` grep for wmma/v_wmma/tensor_core = 0 hits). So the matmuls run scalar fp32 (RDNA3 WMMA needs
fp16 operands) at ~1% of peak — the fused-dequant + fp32-activation path blocks tensor cores. Gate (<30% of
peak + no WMMA) cleared. The lever is real and large.

DEBUG=2 profile (warm chunk): each transformer block runs as ONE fused `function` (`FFNBlock._run`,
precompile=True) at **~86 ms for 32 tokens** (x36 blocks ~= 3.1 s/chunk). The whole-block fusion WITH the
Q4_K dequant inside is a single untiled mega-kernel -> no batch reuse, no TC.

### P1 — make-or-break (fp16/TC via flags): FAIL. No existing flag fixes prefill.
Measured prefill N=256 (warm), 8B: baseline 68 tok/s; `Q4K_UNFUSE=1` 65; `Q4K_UNFUSE=1 TC=2` 65;
`Q4K_BATCHED=1` (route prefill -> batched-GEMM primitive) 67; **`REALIZE=1` 22 (WORSE** -- materialized fp16
weights are 3.4x more bytes, and the block stays fused). `TC`/`TC_OPT` are BEAM-search actions, not applied by
the default schedule, so they no-op without BEAM (which hangs gfx1100). Gate FAILED: no flag gives >=5x. The
fix is NOT configuration -- the block-level fusion must be broken so the matmuls become clean fp16 GEMMs.

### P2 — fix direction CONFIRMED standalone (matmul_decoded), but it needs WIRING (a build, not a flag).
`extra/qk_matmul_decoded.py` (dequant Q4_K->fp16 MATERIALIZED, then NATIVE fp16 matmul) on real
prefill-shaped tensors at N=32 (the prefill batch), vs the current fused path:
| tensor | shape | native matmul | vs fused |
|---|---|---|---|
| blk.0.ffn_gate | 12288x4096 | 12.91 TF (**15.4% peak**) | **18.3x faster** |
| blk.0.ffn_down | 4096x12288 | 4.9 TF (5.9% peak) | 6.9x |
| blk.0.attn_q | 4096x4096 | 3.51 TF (4.2% peak) | 5.9x |
Even UNTUNED, dequant->fp16->native is **5-18x faster than the fused path** at batch-32 (current prefill is
1.3% of peak). Projected prefill: ~2% -> **~15-25% of llama** (clears the >=10x P3 gate); TC/loop tuning (the
33-98%-of-peak substrate) is upside on top. So the fix is proven; what remains is WIRING it into the prefill
forward.

### Remaining build (P2-wire + P3): the real work, scoped
The prefill linears must materialize the dequantized fp16 weight on a contiguous boundary (unfuse it from the
block's mega-kernel) and run a native fp16 matmul — NOT the current fused-dequant path, and NOT a blanket
REALIZE (which regressed). Touch points: `Q4KPrimitiveLinear`/`Q6KPrimitiveLinear` prefill/batched branch
(add a dequant->fp16->native-matmul path gated on T>1 batched), per-layer dequant amortization (avoid the 16 GB
fp16-resident cost), then token-parity verify and (P2-tune) warm-start TC/loop schedules. This is a
correctness-critical restructure of the prefill forward (interacts with @function/precompile/JIT), not a quick
edit — banked the de-risk (the fix provably works); the wiring is the next focused build.
