# Owned Tile Buffer-Identity KV Read — Result (2026-06-23)

## Verdict: `BUFFER_IDENTITY_KV_WD_PASS` / `BUFFER_IDENTITY_KV_PROMOTION_READY` — **+13–19%, byte-identical, tinygrad decode now ≥ llama.cpp**
The buffer-identity whole-cache read predicted by `runtime-kv-core-engine-result-v2` **works and exceeds the +11%
target**. It removes the owned tile's full-MAXC slice materialization (`E_49152`), is **byte-identical** to the
default route, passes ISA audit, and has **no ctx512 regression** (ctx512 is the *largest* gain). Implemented behind
`DECODE_ATTN_KV_IDENTITY=1` (default **off**); **default flip pending owner authorization** (scope Phase 6).

## W==D (3 interleaved reps, `qk_decode_runtime_overhead`)
| ctx | default tok/s | DECODE_ATTN_KV_IDENTITY tok/s | Δ | llama.cpp tok/s | tg / llama |
|----:|----:|----:|----:|----:|----:|
| 512  | 86.7 | **102.9** | **+18.7%** | 97.7 | **105%** |
| 1024 | 86.2 | **101.2** | **+17.4%** | 97.4 | **104%** |
| 2048 | 84.9 | **98.7**  | **+16.3%** | ~95  | **104%** |
| 4096 | 82.9 | **94.0**  | **+13.3%** | 92.4 | **102%** |

**tinygrad Qwen3-8B-Q4_K_M decode now runs at 102–105 % of llama.cpp across all contexts** — at/above parity, the
culmination of the decode campaign.

## Correction from the Runtime-KV framing
The months-long "runtime-KV is core-engine / callify / Tensor-purity blocked" conclusion was **wrong** (see
`runtime-kv-core-engine-result-v2`). Correctness was always achievable; the entire ~1.5ms/token / +11% tax was the
**owned attention tile reading K/V through sliced cache views, which callify materializes**. This task removed that
slice read. No runtime-KV persistence capability, no Tensor-purity change — a **bounded tile/cache-ABI** fix.

## Materialization attribution (Phase 1)
- Owned tile K/V inputs were `assigned_kv[0,0]` / `[1,0]` — **SLICE** views (`has_buffer_identity()=False`).
- `cache_kv.reshape(flat)` is `has_buffer_identity()=True`; the slice is `False` (IR-confirmed).
- `callify.transform_precompiled_call`: an `AFTER`-node input is **not** force-contiguous, and
  `_precompiled_output_redirect` returns a `BUFFER` with `has_buffer_identity()` **directly** (no copy); a `SLICE`
  falls through and **materializes**. → `MATERIALIZATION_ATTRIBUTED_TO_KV_SLICE_READ`.

## Design B — whole cache + kernel K/V offset (Phase 2/3)
- New kernel `owned_flash_tile_gqa_whole(Q, CACHE, part, meta, …)` reads **K at offset 0** and **V at
  `+Hkv·MAXC·Hd`** from one cache buffer (`#define Hkv 8`). Standalone correct (rel_rmse 2.89e-7, eager/capture/replay).
- Model (behind `DECODE_ATTN_KV_IDENTITY`) passes the **whole** `assigned_kv` (= `cache_kv.after(store)`) with **no
  reshape/slice** — a `RESHAPE` on top would break buffer identity (the redirect accepts only `BUFFER`/`MULTI`). The
  kernel reads the cache flat via its base pointer, so tensor shape is irrelevant.
- In-model **byte-identical**: 8-tok @ctx1024 and 64-tok on two prompts. → `DESIGN_B_WHOLE_CACHE_OFFSET_PASS`.
- Design A (separate K/V buffers) was **not needed** — Design B is less invasive (cache layout + prefill unchanged).

## ISA audit (Phase 4) — `ISA_UNCHANGED_OR_ACCEPTABLE`
`AMD_ISA_PRIMITIVE_CONFIRMED`: `owned_flash_tile_gqa_whole` — **60 VGPR** (+4 vs gqa 56, no pathological jump),
**0 scratch / 0 spill**, `has_vector_dot/has_lds/has_cross_lane/has_vector_global_load = true`. The buffer-identity
read did not regress the kernel.

## Correctness
Byte-identical to the default owned route: 8-tok @ctx1024 + 64-tok × 2 prompts. Default decode unchanged with the
flag off (`[279,1156,22148,…]`). Token correctness is authority.

## Candidate / default decision (Phase 6)
`BUFFER_IDENTITY_KV_PROMOTION_READY`. The implementation is committed behind `DECODE_ATTN_KV_IDENTITY=1` (default
**off**). Per the scope and standing rules, **the default flip requires explicit owner authorization** — strongly
recommended (clean byte-identical +13–19%, ISA-clean, no regression). Flipping it = set the flag default-on in the
owned-route guard.

## Files changed
- `extra/qk_owned_flash_decode.hip`: `#define Hkv 8` + new `owned_flash_tile_gqa_whole` kernel.
- `extra/qk_owned_flash_decode_graph_node.py`: `whole_cache` param on `amdgcn_flash_decode`/`_kernels`/
  `_specialize_tile`; factored `_combine_stage`.
- `tinygrad/llm/model.py`: owned-route `DECODE_ATTN_KV_IDENTITY` branch (default off) passing the whole cache buffer.
- New: `docs/owned-tile-buffer-identity-kv-read-result-20260623.md` + 6 artifacts under
  `bench/qk-owned-tile-buffer-identity-kv-read/`. Updated README, session-handoff, candidates.

## Git status
`tinygrad/llm/model.py` changed **only behind the default-off flag** (default decode byte-identical). New kernel +
graph-node `whole_cache` path are inert unless the flag is set. No default flip, no machine search, no 14B/32B, no
attention-math/GEMV optimization.
