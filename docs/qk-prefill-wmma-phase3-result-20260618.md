# PREFILL WMMA Phase 3 — in-model TC attention: REFUTED, unwound (2026-06-18)

Wired the gated `PREFILL_WMMA` TC-attention branch into `_attention` (explicit Q@Kᵀ TC → fp16 scores →
scale+causal-mask → softmax → P@V TC, GQA via broadcast) + tested with concrete start_pos so the TC fires.
**Verdict: REFUTED in-model — TC attention fires and is byte-identical, but gives only +0.6% e2e; the isolated
2.5× does not translate.** Route unwound; `model.py` reverted byte-for-byte (fallback preserved). RX 7900 XTX,
Qwen3-8B.

## What was built (Phase 3A/3B)

- `_prefill_wmma_ok(B,T,start_pos)` guard: active only when `PREFILL_WMMA=1` + `PREFILL_V2` + AMD + B==1 +
  concrete T/start_pos + Qwen3-8B shape (32/8/128) + no output-gate; else SDPA fallback.
- The Option-B branch in `_attention` (the probe's exact math).
- Tested with concrete int `start_pos=0` (so KV=512 concrete → optimizer-TC fires). Confirmed TC fired in-model:
  unique wmma kernels 10 → 12 (the QK + PV TC matmuls, reused across all 36 layers).

## Phase 3C — correctness: PASS

Sampled token **byte-identical** (368) with PREFILL_WMMA off and on (the fp16 scores didn't flip the argmax).

## Phase 3D — speed: FAIL (the decisive result)

Controlled (same concrete path, GPU `time_sum_s`, only attention differs):

| path | GPU ms | note |
|---|---|---|
| concrete, PREFILL_WMMA=0 (SDPA) | 359.5 | baseline |
| concrete, PREFILL_WMMA=1 (TC attn) | 357.5 | **+0.6%** |

**+0.6%, far below the +10% gate.** TC attention saved only ~2 ms, not the ~30 ms the isolated 2.5× implied.

## Why the isolated 2.5× doesn't translate

In isolation, both SDPA and the explicit path did the same softmax and TC won on the QK/PV matmuls (2.5×).
**In-model, SDPA's attention is already an efficient fused path, while the explicit Option-B path adds real
overhead that offsets the TC matmul savings:** it materializes `[Hkv,G,T,KV]` f32 scores (32×512×512×4 ≈ 32 MB
write+read per layer) + an f32 softmax over KV + casts. The TC matmul time saved (~per-layer) is eaten by that
materialization/memory traffic. So the explicit TC attention is ~the same wall as SDPA in-model (+0.6%), not
2.5×. (This is the in-model truth the standalone probe couldn't see — the standalone timed only the attention,
where the materialization was relatively cheaper.)

## Secondary note — concrete vs symbolic start_pos (inconclusive, not pursued)

In this (non-official) harness, the concrete-start_pos forward measured faster than symbolic (359 vs 441 ms GPU),
but the same harness measures the symbolic path much slower than the official prefill-v2 harness (208 ms wall), so
the concrete-vs-symbolic delta is a harness artifact and is NOT a trustworthy win. Concrete start_pos also costs
one jit/compile per distinct start_pos for multi-chunk prompts. Not pursued.

## Verdict & decision

**REFUTED in-model. Unwound** (`model.py` reverted byte-for-byte; default behavior unchanged; decode untouched).
The PREFILL WMMA arc is closed: isolated TC attention is a real 2.5× (Phase 2), but the materialization overhead
of the optimizer-TC (non-fused) approach erases it in-model. A win would require a **fully-fused flash-style WMMA
attention** (no score materialization — Option A / SHAPED_WMMA), which the earlier WR4 probe found stale against
the current codegen spec wall — a deep codegen build, not earned by this +0.6% result.

## Prefill status (unchanged)
PREFILL_V2 remains ~2459 tok/s = ~81% of llama (FFN/proj already WMMA via warmstart-TC). Attention stays SDPA.
No further bounded prefill lever: FFN already TC, attention WMMA refuted in-model (materialization-bound),
lm_head already jit-fused. Remaining prefill gap is the non-fused attention materialization + the same
codegen-ILP ceiling as decode.

## Files / commits
`tinygrad/llm/model.py` (branch added then reverted — refuted), this doc (`[docs]`). Kept: Phase 2 docs +
`qk_prefill_tc_wr_softmax_probe.py` (isolated 2.5× evidence). No `[nn]` retained, no defaults changed.
