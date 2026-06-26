# Next action: ISA-vectorization gate + block-tiled multi-warp generated tile — scope (2026-06-26)

Continues `docs/decode-generated-tile-codegen-scope.md`, `docs/decode-reg-store-devec-codegen-scope.md`,
`docs/decode-generated-tile-phase2a-kstage-blocktile-result.md`. Do NOT implement until asked.

## Honest status (do not overclaim)

- `REG_STORE_DEVEC` pass works and is correct → `SEARCH_RESOLVED__REG_STORE_DEVEC`. It fixed the illegal
  `make_float4(...) = ...` REG-store and preserved correctness. **It did NOT achieve the Phase-1 goal.**
- The generated tile still emits **scalar** cache loads: ISA diff says generated `global_load_d16=0`,
  `global_load_dwordx4=0`, LDS 256 B, cross_lane 20 (owned: d16=22, LDS 8192 B, cross_lane 5). W==D
  unchanged at long ctx (82.7 / 7.2 / 4.1 / 1.1 tok/s @ 128/512/1024/4096; GPU-bound). Coalesced loads are
  NOT solved → `SEARCH_BLOCKED_BY_CODEGEN__LOAD_COALESCER_CUSTOM_KERNEL_GAP`.
- Strategic blocker → `SEARCH_BLOCKED_BY_CODEGEN__BLOCK_TILED_MULTI_WARP_TILE_NOT_EXPRESSED`.

### Why the previous gate was insufficient
`extra/qk_decode_cache_identity_index_gate.py` passes on `max_abs ≤ TOL` vs NumPy — correctness only.
UPCAST exposes a vec form but `split_load_store` (`tinygrad/codegen/late/devectorizer.py:153-200`) legally
emits scalar loads when it can't prove contiguity/divisibility (`:182`), which the in-model masked,
lane-sharded load triggers. So a green gate ≠ vectorized ISA. The authority is the emitted ISA, which the
gate never checked. **No future codegen experiment may claim load-coalescing/vectorization progress without
an ISA-authoritative gate.**

## A. Gate rigor — an ISA-vectorization-authoritative gate (do this FIRST)

New gate `extra/qk_decode_isa_vectorization_gate.py` (or a companion mode wired into the existing gates).
Reuse the disasm helpers from `extra/qk_decode_attention_fused_score_state_pv_attribution.py`
(`_disasm`/`_hist`/`_parse_desc`) and the marker set in `extra/qk_decode_attention_isa_diff_gate.py:35-44`
(extend the load regex to also match RDNA3 `global_load_b128|b96|b64|b32` naming, not only `dword*`/`d16`).

For a named target generated kernel it MUST:
- capture the kernel's lib bytes (runtime hook, as the ISA-diff/attribution gates do), disassemble, and
  count load markers + LDS bytes + cross_lane;
- ASSERT a wide load: PASS only if `global_load_d16 > 0 OR global_load_dwordx4 > 0 OR global_load_b128 > 0
  OR global_load_b64 > 0` (accepted wide-load markers);
- record, in the artifact: numeric-correctness result, route-cleanliness result, the ISA marker dict, and
  an explicit field `"authority": "vectorization"` (vs `"correctness_only"` for the old gates).

Verdicts:
- `ISA_VEC_AUTHORITATIVE_PASS` — target kernel emits ≥1 accepted wide-load marker.
- `ISA_VEC_AUTHORITATIVE_FAIL__SCALAR_LOADS` — correct/clean but loads are scalar (the current state).

Apply it to: the in-model `flash_fused_xlane_score_pv_tile_whole_cache_32_128`, the cache-identity UPCAST
rows (flag correct-but-scalar), and the new block-tile microgate (B). Command:
```
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_isa_vectorization_gate.py
```
Current expected result on the in-model tile: `ISA_VEC_AUTHORITATIVE_FAIL__SCALAR_LOADS` (this is the new
honest baseline; the gate exists to make that visible and to gate B).

## B. Phase 2 — block-tiled, multi-warp generated tile (the real next move)

Structurally mirror the owned kernel `owned_flash_tile_gqa_whole`
(`extra/qk_owned_flash_decode.hip:225`). Microgate FIRST; port to the model route only after numeric proof.

### Structure (per (kvh, split) workgroup)
- Grid `(Hkv, S)`, default `S=48` (`DECODE_ATTN_AMDGCN_S` parity). Workgroup = **128 threads = 4 warps ×
  32 lanes**. Express as two thread dims: `lane = UOp.special(32, "lidx0")` and
  `warp = UOp.special(4, "lidx1")` (the FIRST thing the microgate must prove: a 128-thread workgroup with a
  per-warp 32-lane cross-lane reduce is expressible — if not, that is the finding, label below).
- Each warp owns one of `G = Hq/Hkv = 4` query heads: `h = kvh*G + warp`.
- LDS: `ksh[TK*Hd]` + `vsh[TK*Hd]` in fp16 → `2*16*128*2 = 8192 B` (`TK=16`).
- Outer block loop `blk` over `NB = ceil(L/TK)` blocks of the split. Per block: all 128 threads
  cooperatively stage `TK*Hd` K and `TK*Hd` V from raw `cache_kv` (`[2,1,Hkv,MAXC,Hd]`) into `ksh`/`vsh`
  (16 elements/thread), then **one barrier**.
- Inner loop over the `TK` tokens in the block: each warp computes `fdot2` q·k (lane e-shards Hd, reads K
  from `ksh`), `cross-lane warp_reduce` → score; online-softmax update (m,l); PV accumulate from `vsh`
  (lane d-shards Hd). The cache loads are the cooperative block stage (vectorizable by construction); the
  inner reads are from LDS (`ds_*`), matching owned.
- Output per `(kvh, warp, s)`: `acc[Hd] + l + m` (W=Hd+2), consumed by the existing
  `flash_state_gmax_kernel` + `flash_state_combine_kernel` (unchanged). Raw 5D `cache_kv` identity.
- Default-off: behind `DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE` + a new sub-flag (e.g.
  `DECODE_ATTN_BLOCK_TILE=1`); shipped default route + GEMVs unchanged.

### Microgate (numeric proof, no model)
New `extra/qk_decode_attention_block_tile_microgate.py`: build the tile, validate vs a NumPy per-split
oracle (scalar fp32 `max_abs ≤ 1e-7`; fp16 `≤ 2e-5`) across `Tc ∈ {128,130,32,256}`, `Hq=32,Hkv=8,Hd=128`.
Verdicts `BLOCK_TILE_MICROGATE_PASS` / `_FAIL__NUMERIC` / `_BLOCKED__UOP_VERIFY` (e.g. the 128-thread /
per-warp reduce cannot be expressed → `SEARCH_BLOCKED_BY_CODEGEN__MULTI_WARP_NOT_EXPRESSED`).

### Acceptance (B), in order
1. `BLOCK_TILE_MICROGATE_PASS`.
2. Route gate clean: `FUSED_XLANE_SCORE_PV_ROUTE_CLEAN__ECONOMICS_NEXT` (token-match, no materialization,
   owned absent).
3. ISA-vec gate (A): `ISA_VEC_AUTHORITATIVE_PASS`, with **LDS toward 8192 B**, **cross_lane 20 → toward 5**,
   wide-load markers > 0.
4. W==D improves materially at ctx512+ (vs the current 7.2/4.1/1.1).

## C. Decision on path 1 (load-coalescer chasing)

Lower priority. Do NOT chase `split_load_store` contiguity for the per-token tile further unless the ISA-vec
gate (A) specifically shows a coalescing fix is the binding condition for the *block-tiled* route. If
continued anyway, an `ISA_VEC_AUTHORITATIVE_PASS` is required before any progress claim. Fallback label:
`SEARCH_BLOCKED_BY_CODEGEN__LOAD_COALESCER_CUSTOM_KERNEL_GAP`.

## D. Terminal interpretation (if B also stalls)

If the block-tiled multi-warp tile cannot be expressed or cannot reach the ISA/W==D bar, the honest
conclusion (consistent with `docs/pure-machine-search-roadmap.md`):
- competitive decode attention remains the owned `.hip` for now;
- machine-search purity is NOT achieved for attention;
- prove machine-search first on the smaller generated GEMV / lane-map routes (which already work
  hand-authored + comgr) before attention;
- attention needs a lower-level codegen/layout/search project (multi-warp tiling + scheduling) before it
  can be pure. Label: `SEARCH_BLOCKED_BY_CODEGEN__BLOCK_TILED_MULTI_WARP_TILE_NOT_EXPRESSED`.

## All gate commands (current + new)

```
# A: ISA-vectorization authority (NEW; current in-model expected FAIL__SCALAR_LOADS)
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_isa_vectorization_gate.py
# B: block-tile numeric microgate (NEW)
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_attention_block_tile_microgate.py
# existing correctness/route/ISA/W==D (run with the block-tile sub-flag once B lands)
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_cache_identity_index_gate.py
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_microgate.py
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_route_gate.py
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_attention_isa_diff_gate.py
DEV=AMD JIT=1 DECODE_ATTN_GENERATED_WHOLECACHE=1 DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1 V_DOT2_LOWERING=1 REG_STORE_DEVEC=1 \
  PYTHONPATH=. python3 extra/qk_decode_runtime_overhead.py   # vs baseline 82.4/103.5/101.8/94.6
```

## Labels (canonical)

- `SEARCH_RESOLVED__REG_STORE_DEVEC` — REG-store bug fixed (keep the pass; do not overclaim further).
- `ISA_VEC_AUTHORITATIVE_PASS` / `ISA_VEC_AUTHORITATIVE_FAIL__SCALAR_LOADS` — new vectorization authority.
- `BLOCK_TILE_MICROGATE_PASS` / `_FAIL__NUMERIC` / `_BLOCKED__UOP_VERIFY`.
- `SEARCH_BLOCKED_BY_CODEGEN__LOAD_COALESCER_CUSTOM_KERNEL_GAP` — path-1 fallback (deprioritized).
- `SEARCH_BLOCKED_BY_CODEGEN__MULTI_WARP_NOT_EXPRESSED` — if the 128-thread/4-warp tile can't be built.
- `SEARCH_BLOCKED_BY_CODEGEN__BLOCK_TILED_MULTI_WARP_TILE_NOT_EXPRESSED` — strategic / terminal.

## Constraints

Default-off (env-gated + cache key); shipped default route + q4k GEMVs byte-for-byte unchanged. Do not
revive score-broadcast. Do not add another attention *layout*. Correctness-first; microgate before model.
Bracketed-prefix commits (repo hook). Codex prompt: `docs/decode-block-tile-and-isa-gate-codex-prompt.md`.
