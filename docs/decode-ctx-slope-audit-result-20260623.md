# Decode Context-Slope Audit — Result (2026-06-23)

## 1. Verdict: `CTX_SLOPE_NO_ACTION_DECODE_MAINTENANCE`

The shrinking buffer-identity decode gain at long context has **two confirmed mechanisms**, and the scope's
"pure fixed materialization tax" hypothesis is **partially confirmed but refined**:

1. **Fixed tax — CONFIRMED.** The removed materialization `E_49152` is **ctx-flat** (~1.52 ms, attribution slope
   **+0.003 ms/1k-ctx**). As a constant absolute saving it shrinks in *percent* terms as the decode denominator grows.
2. **New finding — the replacement tile is steeper.** The whole-cache buffer-identity tile
   (`owned_flash_tile_gqa_whole`) has a **steeper ctx-slope** than the slice tile it replaced
   (`owned_flash_tile_gqa`): **+0.265 vs +0.227 ms/1k** (attribution). So the *absolute* saving itself **erodes** with
   ctx (W==D saved-ms slope **−0.090 ms/1k**, 1.872 → 1.528 ms). This erosion is the **larger** driver of the shrinking
   percent gain — more than the fixed-tax/denominator effect.

**Decision: NO ACTION (artifact only).** Decode is already **103–106 % of llama.cpp** across all supported context.
The one bounded lever (whole-cache strided-read coalescing) is real but **< 2 %** headroom, concentrated at long ctx,
and below the action bar. No default change, no kernel change, no machine search.

> ⚠ **Flag-stack correction (caught mid-audit).** The doc headline table (102.9/101.3/98.7/94.2) was measured with the
> **Q4K warp GEMV stack ON** (`Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 Q4K_GEMV_WARP_PROJ=1`), even though
> `bench/qk-post-parity-hardening/authority.json` lists those flags as **default-OFF**. Default (warp-off) decode
> reproduces only ~89.5/87.9/86.2/82.7 (A). With the warp stack the canonical baseline reproduces the doc exactly
> (104.0/102.1/99.6/95.1). **This audit uses the warp-on canonical stack as primary**; warp-off captured separately as
> `wd_by_ctx_warpoff.json`. The A-vs-B buffer-identity delta and its shrinking-with-ctx **shape are robust to the flag
> stack** (warp-off delta +16.2 → +11.6 %; warp-on +19.4 → +14.6 %).

## 2. Authority / config

| field | value |
|---|---|
| HEAD | `8601bafbd` (branch `qk-prefill-flag-leak-resolution`) |
| GPU | Navi 31 [RX 7900 GRE], gfx1100, perf level `high` |
| model | `Qwen3-8B-Q4_K_M.gguf`, MAXC 4608 |
| harness | `extra/qk_decode_runtime_overhead.py` W==D (W = `.item()`/token real decode = promotion authority), interleaved A,B via `extra/qk_ctx_slope_driver.py`, 3 reps × 40 meas |
| canonical flags | `Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 Q4K_GEMV_WARP_PROJ=1` + defaults `DECODE_ATTN_AMDGCN_TILE=1`, `FLASH_DECODE_THRESHOLD=512`, `JIT=1`, `DEV=AMD` |
| Config A | `DECODE_ATTN_KV_IDENTITY=1` (default) → `owned_flash_tile_gqa_whole`, E_49152 absent, buffer-identity |
| Config B | `DECODE_ATTN_KV_IDENTITY=0` → `owned_flash_tile_gqa` slice route, E_49152 present, materializes |
| llama ref | `bench/qk-post-parity-hardening/authority.json` (512:97.71, 1024:97.39, 2048:95.0, 4096:92.37 tok/s) |

## 3. W==D by context (canonical warp-on, authority)

| ctx | A tok/s | B tok/s | Δ% tok/s | A ms | B ms | saved ms | saved % of B | A vs llama |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 104.0 | 87.1 | +19.4 % | 9.613 | 11.485 | 1.872 | 16.3 % | 106.5 % |
| 1024 | 102.1 | 86.5 | +18.0 % | 9.799 | 11.567 | 1.768 | 15.3 % | 104.8 % |
| 2048 | 99.6 | 85.3 | +16.8 % | 10.042 | 11.718 | 1.676 | 14.3 % | 104.8 % |
| 4096 | 95.1 | 83.0 | +14.6 % | 10.514 | 12.042 | 1.528 | 12.7 % | 103.0 % |

Spread < 0.6 % across reps (one ctx1024-A rep had a transient stall; median 102.1 agrees with independent canonical
runs at 102.3, so the median is authority). `saved_ms` falls −18.4 % while B's denominator grows only +4.8 % → **the
shrinking percent gain is dominated by the absolute saving eroding, not by the growing denominator.**

## 4. Route / materialization confirmation (`CTX_SLOPE_ROUTE_CONFIRMED`)

| | candidate kernel | slice route | E_49152 | buffer-identity inputs |
|---|---|---|---|---|
| A (default) | present ✓ | absent ✓ | absent ✓ | true ✓ |
| B (`KV_IDENTITY=0`) | absent | present | present (`E_49152_32_3`, `_3n1`) | false |

Token output byte-identical (established in `owned-tile-buffer-identity-kv-read-result`).

## 5. Kernel attribution (PROFILE GPU-busy, attribution only — not promotion authority)

| ctx | A whole-tile ms | B slice-tile ms | E_49152 (B) ms |
|---:|---:|---:|---:|
| 512 | 0.377 | 0.331 | 1.516 |
| 1024 | 0.575 | 0.464 | 1.505 |
| 2048 | 0.839 | 0.669 | 1.518 |
| 4096 | 1.349 | 1.151 | 1.521 |

- **E_49152 is ctx-flat** (~1.52 ms) → the removed tax is genuinely fixed/MAXC-shaped.
- The **whole-cache tile is uniformly slower and steeper** than the slice tile (penalty 0.046 → 0.198 ms across ctx),
  because it reads K/V from the **strided** whole cache buffer (K@0, V@`+Hkv·MAXC·Hd`) vs the slice route's freshly
  **materialized contiguous** buffer. GPU-busy ≠ wall (overlap not removed), so attribution explains the *direction* of
  the wall saved-ms erosion (~40 %+) but does not fully close the wall budget; the W==D fits are authority.

## 6. Slope model (`ms = fixed + slope·ctx`, least squares on W==D ms)

| config | fixed ms | slope ms/1k-ctx | max resid ms | interpretation |
|---|---:|---:|---:|---|
| A_whole_default | 9.521 | **+0.245** | 0.034 | low fixed (no materialization) but steepest slope |
| B_slice_identity0 | 11.405 | +0.155 | 0.005 | high fixed (materialization) but flattest slope |
| llama_cpp_ref | 10.134 | +0.172 | 0.042 | reference |
| saved (B−A) | 1.884 | **−0.090** | 0.034 | saving erodes with ctx |

Attribution slopes: A_tile **+0.265**, B_tile **+0.227**, E_49152 **+0.003** ms/1k.

## 7. Llama comparison (`CTX_SLOPE_LLAMA_MARGIN_EXPLAINED`)

| question | answer |
|---|---|
| tinygrad (A) above llama at all measured ctx? | **Yes** (103–106 %) |
| tinygrad worse ctx-linear slope than llama? | **Yes** — A +0.245 vs llama +0.172 ms/1k (**ratio 1.43**) |
| ctx4096 margin lower due to higher slope or lower fixed advantage? | **Higher slope** eroding a large fixed advantage |
| evidence of remaining long-ctx attention inefficiency? | **Yes, bounded** — whole-cache strided read slope > contiguous |
| worth a bounded long-ctx tile policy search? | **Marginal** — ~+1.9 % @4096, ~0 % @512; A already ≥ llama within MAXC |

Projected crossover (A falls below llama): ctx **≈ 8335**, beyond MAXC 4608 → **A stays above llama within all
supported context.**

## 8. Decision

- **No action.** Decode is at/above llama parity (103–106 %) across all supported context; the residual long-ctx slope
  gap is < 2 % and shrinks to ~0 at short context.
- **One bounded lever documented (LOW priority):** improve `owned_flash_tile_gqa_whole`'s strided whole-cache K/V read
  coalescing so its ctx-slope approaches the contiguous read. Headroom ~+1.9 % @ctx4096. Reopen only under an explicit
  long-ctx decode push with new evidence.
- **Artifact only** — no default change, no kernel change, no machine search this task.

## 9. Files changed

- New artifacts under `bench/qk-decode-ctx-slope-audit/`: `authority.json`, `wd_by_ctx.json`,
  `kernel_attribution_by_ctx.json` (+ raw `kernel_attribution_{A,B}.json`), `slope_fit.json`, `llama_comparison.json`,
  `decision.json`, `wd_by_ctx_warpoff.json` (secondary).
- New tools: `extra/qk_ctx_slope_driver.py` (interleaved-rep W==D driver), `extra/qk_ctx_slope_analyze.py`
  (slope fit + decomposition, no measurement).
- This doc. **No `tinygrad/` or kernel changes; no defaults touched.**

## 10. Git status

Branch `qk-prefill-flag-leak-resolution` @ `8601bafbd`. Only audit artifacts + two `extra/qk_ctx_slope_*.py` tools +
this doc added. Default decode unchanged.
