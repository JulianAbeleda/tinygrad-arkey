# Prefill v2 — Increment 1: BUILT & WON (~13x warm prefill) (2026-06-17)

The model.py build for prefill v2 (concrete-ubatch + fp16 + warmstart-TC). Gated `PREFILL_V2` (opt-in,
default off); **decode 100% untouched**. Follows the Stage-0 gate (`amd-decode-prefill-v2-gate-20260616.md`).

## Result (Qwen3-8B-Q4_K_M, RX 7900 XTX / gfx1100, warm)

| | tok/s | note |
|---|---:|---|
| baseline prefill (today's symbolic v_toks chunk) | **189** | batch-symbolic → no tensor cores |
| **prefill v2 (concrete-512, fp16, realized weights, warmstart-TC)** | **2486** | **13.1x**, ~83% of llama's ~3000 |

Greedy output **byte-identical** to baseline; warmstart **apply=5, error=0**. This blows past the gate's
conservative ~7–10x / 15–25%-of-llama estimate. Acceptance harness: `extra/qk_prefill_v2_measure.py`.

## What the build is (all gated on `PREFILL_V2`, decode-safe)

- **Concrete-ubatch loop** (`generate`): full `PREFILL_UBATCH=512`-token chunks of all-real prompt tokens go
  through a **separate `prefill_v2_jit`** with a concrete `T` (not the symbolic `v_toks`); the `<512` tail
  falls through to today's symbolic path (correct last-token logit, no padding hazard). `is_prefill_v2`
  separates the two jits by `isinstance(tokens.shape[1], int)` (concrete) vs UOp (symbolic).
- **fp16 + `.contiguous()`-isolated matmuls** (`_feed_forward`/`_attention` under a per-block `_prefill_v2`
  flag, mirroring `_use_flash`): `_pf16(lin, x)` is one clean fp16 TC GEMM per matmul.
- **Per-shape warmstart-TC** (`_install_prefill_v2_warmstart`, init): forces the loop-found schedule by shape
  key `(frozenset({out, 512}), in)`, NO BEAM. Keys are 512-specific → can't match decode's T=1 GEMVs.
- **fp16 weight realization** (`realize_prefill_v2_weights`, end of `from_gguf`).

## The three things the gate hid (corrections to its premise)

The Stage-0 gate PASSED on a **fresh process, 2D inputs, PRE-REALIZED random fp16 weights**. Wiring it
in-model surfaced three issues; (1) and (2) are real bugs (fixed), (3) is a measurement confound:

1. **The primitive `.weight` is a LAZY Q4_K/Q6_K→fp16 dequant graph (149 ops), not a realized buffer.** Used
   raw, the whole dequant fuses into the matmul → bandwidth/dequant-bound **~3% peak (no TC win)**. The gate's
   "FFN-v2 weight is just `self.weight.cast(fp16)` — no dequant pass needed" was **wrong**. Fix: realize a
   clean fp16 buffer per linear (`_pf16_w`). **COST: ~fp16-model-size extra VRAM** (~16 GB for 8B), coexisting
   with the Q4_K decode storage — fits 8B on 24 GB; **14B/32B with `PREFILL_V2=1` will OOM at load** (opt-in,
   documented; a VRAM-frugal per-layer realize is future work).
2. **One TC schedule for all shapes → ~9%.** The contraction-heavy `ffn_down` (4096×12288) wants
   `UPCAST(0,4)`, not `UPCAST(0,2)`. Fix: `_prefill_v2_opts(out,in)` picks per-shape opts (`in>out` → 4).
3. **Isolated single-matmul / single-chain benches are host-launch-overhead bound** (~20 ms of a 30 ms wall
   on 8B; GPU time ~9.7 ms). They read ~7–19% and are *not* the in-model truth — the **warm full forward**
   (one JIT replay amortizes host overhead) is, and it is what a real prefill pays. (cf.
   `amd-decode-measurement-confounds`.)

## Honest caveats / next

- **fp16 is lossy** vs fp32 → a greedy/ppl **quality gate** is still owed (greedy byte-identical here is the
  cheap signal, not proof). 
- **VRAM**: 8B-only as shipped; larger models need a frugal realization scheme.
- E2E "time to first token" is **JIT-compile-dominated** (~28 s capture, paid once per shape); the 13x is the
  warm throughput that matters for long prompts.
- **Increment 2** (the next e2e lever): flash-style **prefill** attention (O(T²) SDPA still rides along).
- Dense-arch only (Qwen3-8B); MoE/MLA/SSM fall through to today's path.

Code: `tinygrad/llm/model.py` (commits `[nn]` 1a/1b/1c + the realize/per-shape fix), `extra/qk_prefill_v2_measure.py`,
`test/external/test_qk_prefill_v2.py`. Suite **247 pass / 56 skip**.
