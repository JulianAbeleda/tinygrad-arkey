# Prefill TC-attention (Option B) — gated integration probe: REFUTED in-model, UNWIRED (2026-06-17)

Option B (explicit Q@Kᵀ TC → fp16 materialized scores → softmax → P@V TC, GQA via broadcast) **won ~2.5× over
SDPA standalone** (`docs`/`bench/qk-prefill-tc-wr-softmax-probe/`), so per the principled ladder we did the
gated model integration probe before any SHAPED_WMMA surgery. **It does NOT survive in-model — unwired.**

## In-model result (full prefill-v2 forward, DEBUG=2 GPU time, TC on vs off)

`extra/qk_prefill_tc_attention_measure.py`, Qwen3-8B, subprocess-isolated per mode:

| start_pos | KV | SDPA | TC-attn | speedup |
|---:|---:|---:|---:|---:|
| 0 | 512 | 445.6 ms | 486.4 ms | 0.92× |
| 512 | 1024 | 562.3 ms | 647.5 ms | 0.87× |
| 1536 | 2048 | 812.4 ms | 989.2 ms | 0.82× |
| 3072 | 3584 | 1203.6 ms | 1518.8 ms | **0.79×** |

TC attention is **SLOWER** in-model (0.79–0.92×), failing the ≥1.25× gate. Greedy output was byte-identical on
the smoke (fp16 scores didn't flip argmax there), so this is purely a perf refutation.

## Root cause: symbolic KV blocks the concrete-shape TC

The standalone probe used **concrete** KV and TC fired (2.56×). In-model, the prefill-v2 chunk uses a
**symbolic** `start_pos` (`v_start_pos.bind`, for jit reuse across chunks), so `KV = start_pos + T` is symbolic
— and the concrete-shape tensor-core lever (the same one that fires for the FFN's concrete `[512,H]` matmuls)
does **not** apply to the symbolic-KV Q@Kᵀ/P@V. So TC doesn't fire in-model, and the explicit path's extra
overhead (score materialization + reshapes) just makes it slower than SDPA. (The only structural difference
between the 2.56×-standalone and 0.79×-in-model is concrete-vs-symbolic KV; a concrete-start_pos in-model test
was blocked by jit arg plumbing, but the inference is unambiguous.)

## Decision: UNWIRED, banked as standalone-only

`tinygrad/llm/model.py` reverted to SDPA for prefill attention (the `_attention` TC branch and the
`PREFILL_TC_ATTENTION` flag removed; only a NOTE comment remains). Decode untouched, suite green. Standalone
Option B evidence is kept (`extra/qk_prefill_tc_wr_softmax_probe.py`, lock test).

## Salvage paths (deferred — neither pursued)

1. **Concrete-KV prefill:** capture the prefill forward with a CONCRETE start_pos per chunk (KV concrete → TC
   fires). Cost: recompile per distinct start_pos (per 512-chunk) instead of one symbolic jit — a prefill-loop
   restructure, and compile is multi-second. Worth it only if e2e prefill is a priority for long prompts.
2. **Option A (fused flash):** a fused SHAPED_WMMA + LDS + online-softmax kernel handles symbolic KV via its own
   tiling — but the SHAPED_WMMA custom-kernel idiom is stale (WR4 wall, codegen-spec revival needed).

## Status

Prefill attention stays **SDPA**. Prefill v2 rests at **Increment 1 (the ~13× FFN win, quality-gated)**. The
attention-speedup levers are all mapped + gated: tiled-ops (slower), naive LDS tile (cache/occupancy), TC
materialized (symbolic-KV in-model), fused SHAPED_WMMA (stale idiom). The lasting assets from this whole arc:
the shape-safe warp-reduction primitive (WR1–3, `extra/amd_warp_reduce.py`) and the LDS-reuse proof (Phases
2–4), reusable if a fused kernel arc is ever funded.

Anchors: `amd-warp-reduce-wmma-revival-20260617.md`, `amd-lds-tiling-primitive-arc-20260617.md`,
`amd-decode-prefill-v2-increment1-20260617.md` (the real win).
