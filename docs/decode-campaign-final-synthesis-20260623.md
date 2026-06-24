# Decode Campaign — Final Synthesis (2026-06-23)

## 1. Final verdict: `DECODE_CAMPAIGN_FINAL_SYNTHESIS_COMPLETE` — tinygrad Qwen3-8B-Q4_K_M decode is at/above llama.cpp
On the validated gfx1100 path, tinygrad decode now runs at **102–105 % of llama.cpp** across ctx 512–4096, byte-
identical, as the **default** route. The campaign target is crossed. `POST_PARITY_HARDENING_COMPLETE`.

## 2. Performance vs llama.cpp (W==D, 3 interleaved reps)
| ctx | old default | new default | Δ | llama.cpp | tg / llama |
|----:|----:|----:|----:|----:|----:|
| 512  | 86.7 | **102.9** | +18.7% | 97.7 | **105%** |
| 1024 | 86.2 | **101.3** | +17.4% | 97.4 | **104%** |
| 2048 | 84.9 | **98.7**  | +16.3% | ~95  | **104%** |
| 4096 | 82.9 | **94.2**  | +13.3% | 92.4 | **102%** |

## 3. Critical corrections (what the campaign got wrong, then right)
- **Attention was not exhausted.** After the gqa_coop_vec / flash-variant work it looked closed, but the owned
  AMDGCN tile (v_dot2 + LDS + cross-lane) still added +12–22 %.
- **Runtime-KV was not core-blocked.** ~10 tasks concluded the +11 % KV tax needed a core TinyJit/HCQ/Tensor-purity
  persistence capability ("callify hard-stop"). **Wrong.** Correctness was always achievable (native store + read is
  byte-identical); only the *opaque custom_kernel append* baked.
- **Buffer identity was the actual wall.** The +11 % tax was simply the owned tile reading K/V through **sliced cache
  views** (`cache_kv[0,0]`), which callify materializes (`E_49152`). Passing the **whole** `cache_kv` buffer (no
  reshape/slice) with K/V offsets computed in the tile removed it. A bounded tile/cache-ABI fix, not an engine project.

## 4. Final primitive ledger (shipped, default state)
| primitive | impl | status |
|---|---|---|
| Q4K GEMV warp | tinygrad-native UOp schedule | W==D pass, env-gated (lossless) — weight-GEMV parity |
| owned attention tile | hand HIP code object `owned_flash_tile_gqa` / `_whole` | W==D pass, **default-on** |
| whole-cache buffer-identity KV read | owned tile ABI + whole `cache_kv.after(store)` | W==D pass, **default-on** (the +13–19 %) |
| ISA audit wrapper | `extra/qk_isa_primitive_audit.py` | ready, **mandatory evidence guard** |

## 5. Closed lanes
Attention (owned tile + flash variants); weight-GEMV (Q4K warp parity); KV-materialization (whole-cache read);
runtime-KV persistence (**retired as mis-scoped** — the tax was the slice read). Decode is **HBM-bandwidth-bound** at
parity+ — no large structural lever remains.

## 6. Remaining optional lanes (all bandwidth/diminishing or out of decode)
- **small-op fusion** (norm/rope/residual) — thin upside (bandwidth wall), needs a W==D gate before any work.
- **prefill** — the real beyond-parity frontier (compute-bound): dependency-free hand-asm LDS GEMM is Tensile-class
  (~61 TFLOPS, +15 % over the LLVM authority, ~92 % of vendored Tensile), ~8 % gap attributed to SIA1/PLR1/PGR1/WGM8.
- **native-codegen** of v_dot2 / cross-lane / LDS (capability, not speed).
- **14B/32B generalization** — owner decision.

## 7. Permanent principles (added to research principles)
- **Buffer-identity ABI rule** (B4): never pass sliced/cache views across a precompiled-call boundary when whole-
  buffer + in-kernel offset math is possible — callify materializes slices but reads buffer-identity inputs directly.
- **W==D is the only authority**; isolated/local kernel wins do not transfer. Token correctness is authority.
- **A TinyJit A/B toggling a routing global must capture each jit before changing the flag + assert.**

## 8. Commands / artifacts
- Default decode (whole-cache, on): no flags needed. Disable: `DECODE_ATTN_KV_IDENTITY=0`. Tile off: `DECODE_ATTN_AMDGCN_TILE=0`.
- W==D: `extra/qk_decode_runtime_overhead.py` (W==D harness). ISA: `extra/qk_isa_primitive_audit.py`.
- Hardening artifacts: `bench/qk-post-parity-hardening/{authority,regression_guard,registry_audit}.json`.
- Win evidence: `docs/archive/owned-tile-buffer-identity-kv-read-result-20260623.md`, `bench/qk-owned-tile-buffer-identity-kv-read/`.

## 9. Default / fallback policy
- **Default:** owned AMDGCN whole-cache buffer-identity tile, fp16 cache, ctx≥512, B=1/Hq32/Hkv8/Hd128/G4 (Qwen3-8B).
- **Fallbacks:** `DECODE_ATTN_KV_IDENTITY=0` → slice route (correct, slower, re-introduces `E_49152`);
  `DECODE_ATTN_AMDGCN_TILE=0` → gqa_coop_vec; any unsupported shape/device → gqa automatically. All byte-identical.

## Regression guard (POST_PARITY_REGRESSION_GUARD_PASS)
Verified this session: default fires `owned_flash_tile_gqa_whole`, **no `E_49152`**; `DECODE_ATTN_KV_IDENTITY=0`
restores the slice route + `E_49152`; tokens byte-identical both ways; ctx1024 W==D ~101.3; ISA
`AMD_ISA_PRIMITIVE_CONFIRMED` (60 VGPR, 0 spill, v_dot2/LDS/cross-lane).
