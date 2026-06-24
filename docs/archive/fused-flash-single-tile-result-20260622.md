# Fused Flash Single-Tile Owned AMDGCN — Result (2026-06-22)

## Verdict: **FUSED_FLASH_SINGLE_TILE_SCOPE_REDUCED_TO_OPAQUE_READ**

Phase-1 feasibility **fails by design of the GPU execution model**, so per the scope ("if feasibility fails,
stop and write result; do not build a doomed kernel") **no kernel was built**. A *true* single fused tile
that folds the cross-split combine into one kernel is **`SINGLE_TILE_GLOBAL_REDUCTION_BLOCKED`** (cross-
workgroup reduction needs grid-wide sync, which tinygrad's HCQ does not provide); the only legal single-node
shape (no split) is **`SINGLE_TILE_PARALLELISM_INSUFFICIENT`** at batch-1 decode. Two independent findings
also refute the premise. **The scope correctly reduces to the opaque-read redirect:** the **existing**
two-node owned tile already reads the cache natively and is the substrate for the runtime-KV follow-on (the
**measured** +1.4 ms / +8 tok/s win). No default change.

## 1. Verdict
`FUSED_FLASH_SINGLE_TILE_SCOPE_REDUCED_TO_OPAQUE_READ` (Phase-1 mechanism: `SINGLE_TILE_GLOBAL_REDUCTION_BLOCKED`
+ `SINGLE_TILE_PARALLELISM_INSUFFICIENT`). Build redirected to the runtime-KV opaque-read follow-on
(`docs/runtime-kv-opaque-read-followon-scope-20260623.md`).

## 2. Online research summary
FlashInfer / vLLM-paged / TensorRT-LLM all treat decode attention as a **specialized kernel + KV-cache
interface**, not a generic tensor expression — which supports the "opaque attention read over a runtime cache"
framing, not a folded single tile. **Flash-Decoding (PyTorch blog)** describes split-KV → *separate* rescale/
combine; it does **not** propose folding the combine into one kernel — and indeed our oracle shows llama uses
two kernels too (below). vAttention argues for keeping a contiguous attention reader while changing KV memory
management — i.e. build a strong opaque contiguous reader first, then remove the functional copy. The research
**reinforces the redirect** (opaque read + runtime KV) and does **not** support a folded single tile.

## 3. Baseline lock
- Commit `872d3eea4` / `3fb5dd982` (HEAD); gfx1100 / RX 7900 XTX; model `Qwen3-8B-Q4_K_M`;
  baseline env `DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1`.
- Post-warp tok/s: 76.1 / 74.0 / 71.0 / 67.0 @ctx 512/1024/2048/4096.
- B4/B5 route (default-off, two nodes `owned_flash_tile_gqa → owned_flash_combine_hd`): correct, byte-identical,
  W==D saturates **+5.66–5.85 %@ctx4096 < +7 % gate**; cheaper combine does not move W==D (combine overlaps).
- KV copy tax ~1.4 ms/token (MAXC-bound), measured wall transfer +1.5 ms / +8 tok/s.

## 4. Feasibility reconciliation — the cross-split combine / grid-sync question
**Can the cross-split combine be folded into one ordinary GPU kernel without grid-wide sync? NO.**

The tile splits KV across **S workgroups** (`owned_flash_tile_gqa`: `grid.x=Hkv=8, grid.y=S`; S=24/48/56 →
**384–448 workgroups** @ctx 512/1024/4096), each writing a per-split partial PV + `(m,l)` meta **to global**.
Merging them is a **cross-workgroup** log-sum-exp reduction. In one kernel that requires **grid-wide sync**
(cooperative groups / `hipLaunchCooperativeKernel`) or a fragile atomics+spin software barrier. **tinygrad's
HCQ launch path has no cooperative-launch support** — `cooperative` appears only in the autogen bindings
(`hip.py`/`hsa.py`), not in `runtime/support/hcq.py`, `ops_amd.py`, `runtime/graph/hcq.py`, or
`engine/realize.py`. So a folded-combine single kernel is impossible on this runtime.

| question | answer |
|---|---|
| Fold combine into one kernel without grid-wide sync? | **NO** — cross-workgroup reduction; HCQ has no cooperative launch |
| Remaining single-node shape? | no-split, one workgroup per (kv)head over full KV |
| Enough parallelism @ctx1024/4096? | **NO** — 8 (gqa-packed) / 32 (per-q-head) workgroups vs the split form's 384; 8–33 % CU occupancy → reverts to the batch-1 under-occupancy split-KV was built to fix |
| Preserves GQA V-reuse + coalesced V? | only the split gqa-packed tile; no-split per-q-head loses it (4× redundant V) |
| Can it serve as opaque KV read for runtime-KV? | **YES — but the existing two-node tile already does** |

Phase-1 verdict: `SINGLE_TILE_GLOBAL_REDUCTION_BLOCKED` + `SINGLE_TILE_PARALLELISM_INSUFFICIENT` →
`SINGLE_TILE_SCOPE_REDUCED_TO_OPAQUE_READ`.

**Two independent refutations of the "single fused tile" premise:**
1. **Grid-sync infeasibility** (above) — you cannot keep split parallelism *and* fold the combine; the
   part/meta HBM round-trip exists *because* of the split, and removing the split is parallelism-insufficient.
2. **llama itself uses two kernels** — `flash_attn_tile` + a separate `flash_attn_combine_results<128>`
   (`llama-flash-attn-tile-oracle:18`). So eliminating the combine is **not** the lever; llama's ~5× advantage
   is **tile codegen quality** (LDS `ds_load_b128` + `v_dot2` + online softmax/PV in the tile), which is the
   **unbounded native-codegen lane** (deferred). The earlier "single fused tile is the one untested lever"
   framing is corrected: the untested thing is infeasible, and it isn't what makes llama fast.

## 5. Kernel design
**Not built** (feasibility gate failed before coding, per scope Phase 1 stop condition). The existing
`owned_flash_tile_gqa` (gqa-packed, LDS-staged, `__builtin_amdgcn_fdot2`) + `owned_flash_combine_hd` remain the
correct architecture and are untouched.

## 6. Local A/B result
Not reached (no kernel built). `bench/qk-fused-flash-single-tile/local_ab.json` intentionally not produced.

## 7. Graph-node identity
Not reached. No new route added; `DECODE_ATTN_AMDGCN_TILE` (two-node) and `gqa_coop_vec` (default) unchanged.

## 8. W==D result
Not reached. The structural ceiling stands: even a perfect attention tile recovers only the attention share
on the critical path (+5.7 %@4096 < +7 %), because attention partly overlaps the weight-GEMV (B5 transfer
ground-truth). A folded tile could at best recover the combine launch + part/meta round-trip — but that is
exactly what is infeasible without grid sync, and B5 already showed the combine *compute* overlaps.

## 9. Runtime-KV follow-on decision
The **existing two-node owned tile is sufficient as the opaque cache read** for the runtime-KV follow-on:

| requirement for runtime-KV opaque read | existing two-node tile |
|---|---|
| reads persistent cache pointer directly | **YES** — `custom_kernel`, native `[Hkv,MAXC,Hd]` layout |
| avoids `assigned_kv` functional full-copy | **YES if fed `cache_kv` directly** (today it is fed `assigned_kv[0,0]`/`[1,0]`, the copy; the follow-on feeds the persistent cache) |
| can be ordered after opaque append | **YES** — both are opaque `custom_kernel` nodes → clean buffer dependency |
| supports symbolic `start_pos` | **YES** — proven in B4 (replays with changing `start_pos`) |
| T=1 decode | **YES** |
| fallback safe | **YES** — existing shape/device guards + `gqa_coop_vec` fallback |

→ **Write the follow-on scope** (`docs/runtime-kv-opaque-read-followon-scope-20260623.md`): pair the proven
opaque KV append (`extra/qk_kv_cache_state_token.py`) with the **existing** owned tile reading the persistent
cache, to remove the ~1.4 ms copy (measured +8 tok/s). Not implemented in this task.

## 10. Default / candidate registry decision
**No registry change.** No new route exists (feasibility failed before a route), so per the scope
("Do not register if Phase 1 feasibility fails before a route exists") `decode_attention_owned_amdgcn_single_tile`
is **not** registered. Defaults unchanged.

## 11. Artifacts and commands
- `bench/qk-fused-flash-single-tile/feasibility.json` (this Phase-1 result).
- Verification commands run (read-only): `grep cooperative` over `tinygrad/runtime/**` + `engine/realize.py`
  (none in the exec path); the split structure in `extra/qk_owned_flash_decode.hip`
  (`owned_flash_tile_gqa` grid `Hkv×S`, combine `owned_flash_combine_hd`); the model route
  `tinygrad/llm/model.py:990` (`amdgcn_flash_decode(q, assigned_kv[0,0], assigned_kv[1,0], …)`).
- No `local_ab.json` / `graph_route.json` / `wd.json` (gates not reached).

## 12. Working tree status
Audit/feasibility-only: **no source/default change**, no kernel built. New:
`docs/fused-flash-single-tile-result-20260622.md`, `docs/runtime-kv-opaque-read-followon-scope-20260623.md`,
`bench/qk-fused-flash-single-tile/feasibility.json`. `model.py` and all `extra/qk_owned_flash_*` byte-clean.
