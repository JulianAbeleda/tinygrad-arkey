# Prefill v2 — Increment 2: flash-prefill attention — GATED (banked) (2026-06-17)

> **CONFIRMED at the kernel level (2026-06-17).** A follow-on custom-kernel ladder (bridge proof →
> capabilities → expressibility → real-dim/GQA) showed a fused score-free attention kernel IS expressible and
> correct, but **PHASE 5 honest GPU-time measurement REFUTED it on performance** (~170–760× *slower* than
> SDPA; the Phase-3/4 ~2.7× "speedups" were wall-clock measurement artifacts). Root cause: score-free without
> LDS reuse is memory-bound. See **`amd-decode-prefill-v2-increment2-phase5-correction-20260617.md`**. Net:
> flash-prefill stays banked; prefill v2 rests at Increment 1 (~13× FFN). Below = the original tiled-ops gate.


Increment 1 made the prefill FFN fast (~13× warm, quality-gated). Increment 2 targeted the next bottleneck —
attention — and **banked it gated**: the tractable approaches are refuted by measurement; the only path to the
real win needs deeper kernel/runtime surgery (same shape of wall as the decode **overlap lever**). Stopping
here is the principled rest point, not capitulation.

## Why attention is the next bottleneck (the diagnosis)

8B warm v2 forward, T=512, by `start_pos` (FFN is constant; the slope is attention):

| start_pos | forward | attention share |
|---:|---:|---|
| 0 | 241 ms | ~8% (512²) |
| 512 | 561 ms | ramping |
| 1536 | 811 ms | |
| 3072 | 1202 ms | **~51%** |

tinygrad SDPA (`tensor.py:1197`) materializes the full `[T, start_pos+T]` scores, softmaxes, then `@v` — a
5-kernel sequence at **~4% peak**, memory-bound, and the **symbolic KV** blocks the concrete-shape/warmstart-TC
lever that fixed the FFN. Attention dominates only at long context (sp ≳ 1500); the **13× short/medium-prompt
win from Increment 1 is unaffected**.

## Stage-0 gate — tractable approaches REFUTED (`extra/qk_flash_prefill_gate.py`)

On the real 8B shapes (Hq=32, Hkv=8, Hd=128, T=512, causal, GQA, fp16), KV ∈ {512,1024,3584}:

| approach | exact vs SDPA | KV=512 | KV=1024 | KV=3584 |
|---|---|---|---|---|
| **KV-tiled online-softmax (tinygrad ops)** | ✅ err ≤ 0.004 | 0.52× | 0.37× | **0.15×** |
| **fp16 materialized (no tiling)** | ✅ | 1.00× | 0.99× | 0.99× |

Both **lose to SDPA**. The flash win needs the online-softmax running state (`acc[Hq,T,Hd]`) **register/LDS-
resident**; in tinygrad ops it spills to HBM every tile (+ GQA `repeat_interleave` 4×'s K/V traffic), costing
more than SDPA's materialized scores. fp16 is a wash because SDPA's fp32 score-cast is not the bottleneck — the
score materialization + softmax memory traffic + occupancy is. Artifact: `bench/qk-flash-prefill-gate/result.json`.

## Why the real lever is GATED (not just unbuilt)

The only path to the ~3–5× flash win is a **custom fused kernel** with register-resident online softmax (raw
HIP, computing `q·k` inline like `flash_partial_src`). Two stacked blockers:
1. **UOp path can't fuse it.** tinygrad's linearizer **rejects nesting the q·k reduce inside the softmax
   reduce** (the documented reason flash-*decode* precomputes scores). So a model-integratable UOp flash-prefill
   would *materialize* `[Hq,T,KV]` scores = exactly what we must avoid = no win.
2. **Raw-HIP doesn't bridge into the model JIT.** flash-decode keeps its raw kernels standalone and
   reimplements them as UOp for the model; a fused prefill kernel would need `[runtime]`/graph surgery to call
   a precompiled HIP lib from inside the JITted attention graph.

This is the same class of wall as the **overlap lever** (decode B2): real headroom, but gated behind
dangerous-power surface. Scoped + killable when/if revisited; the gate harness is the re-fire test.

## Status / resting point

- **Decode: banked** (~64 tok/s, 63% llama).
- **Prefill v2: Increment 1 banked** — ~13× warm prefill (8B, ≤~512-tok / short-medium prompts), exact,
  quality-gated (dNLL ~0), opt-in `PREFILL_V2`, decode untouched, invariants encoded.
- **Prefill v2: Increment 2 (flash attention): GATED** — tractable approaches refuted; custom-kernel lever
  needs linearizer/JIT-bridge surgery. Long-context (sp ≳ 1500) prefill still pays the SDPA O(T²) tail.

## Resume pointers (if reopened)
1. Re-run `extra/qk_flash_prefill_gate.py` (the exactness+speed harness; add a custom-kernel approach as a 3rd
   row and require exact + ≥3× on KV=3584 before integrating).
2. The integration sub-problem: bridge a raw-HIP fused kernel into the model JIT, or find a UOp formulation
   that fuses q·k into the online softmax without tripping the linearizer.
3. Separately (cheaper, out of scope here): the prefill **lm_head-over-all-T** path computes logits for all
   512 positions when only the last is used — verify whether tinygrad prunes it; if not, last-token-only is a
   cheap prefill win independent of attention.

Anchors: `amd-decode-prefill-v2-increment1-20260617.md` (the win), `amd-decode-banked-20260616.md` (decode),
`extra/qk_flash_decode.py` (the flash-decode substrate + linearizer constraint).
