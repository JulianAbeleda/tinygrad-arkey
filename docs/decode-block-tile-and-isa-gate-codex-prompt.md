# Codex task prompt — ISA-vectorization gate + block-tiled multi-warp decode tile

Copy below the line into Codex. Full rationale + file:line: `docs/decode-block-tile-and-isa-gate-scope.md`.

---

You are in the tinygrad fork `/home/ubuntu/tinygrad-arkey` (AMD gfx1100; hardware present, run real jobs
`DEV=AMD JIT=1 PYTHONPATH=.`). Two sequenced tasks. Do NOT overclaim: a gate passing on correctness is NOT
vectorization. Everything default-off; shipped default route + q4k GEMVs must be byte-for-byte unchanged.

Honest current state: the `REG_STORE_DEVEC` pass is correct but the generated tile still emits SCALAR cache
loads (`global_load_d16=0`, `dwordx4=0`, LDS 256B) and W==D is unchanged (82.7/7.2/4.1/1.1 @ 128/512/1024/
4096 vs baseline 82.4/103.5/101.8/94.6). The fix is the block-tiled multi-warp tile, gated by a new
ISA-vectorization authority.

## TASK A (do first) — ISA-vectorization authoritative gate

New `extra/qk_decode_isa_vectorization_gate.py`. Reuse `_disasm`/`_hist`/`_parse_desc` from
`extra/qk_decode_attention_fused_score_state_pv_attribution.py` and the marker regexes from
`extra/qk_decode_attention_isa_diff_gate.py:35-44` — EXTEND the load regex to also count RDNA3 b-naming
(`global_load_b128|b96|b64|b32`), not only `dword*`/`d16`. Capture the target generated kernel's lib bytes
via the runtime hook (as the ISA-diff gate does), disassemble, and:
- PASS only if it emits ≥1 accepted WIDE load: `global_load_d16>0 OR global_load_dwordx4>0 OR
  global_load_b128>0 OR global_load_b64>0`.
- Record in the artifact: numeric-correctness result (if available), route-cleanliness result, the full
  marker dict (incl. LDS bytes, cross_lane), and `"authority": "vectorization"`.
- Verdicts: `ISA_VEC_AUTHORITATIVE_PASS` / `ISA_VEC_AUTHORITATIVE_FAIL__SCALAR_LOADS`.
Run: `DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_isa_vectorization_gate.py`.
Expected NOW on the in-model `flash_fused_xlane_score_pv_tile_whole_cache_32_128`:
`ISA_VEC_AUTHORITATIVE_FAIL__SCALAR_LOADS` (this is the correct honest baseline — the gate exists to make
it visible and to gate Task B). Commit Task A separately.

## TASK B — block-tiled, multi-warp generated tile (microgate FIRST)

Structurally mirror the owned hand kernel `owned_flash_tile_gqa_whole` in
`extra/qk_owned_flash_decode.hip:225`. Build it as a UOp `custom_kernel` builder in
`extra/qk_flash_decode.py`; prove it in a NEW microgate before touching the model route.

Structure per `(kvh, split)` workgroup:
- Grid `(Hkv, S)`, default `S=48`. Workgroup = **128 threads = 4 warps × 32 lanes**:
  `lane = UOp.special(32,"lidx0")`, `warp = UOp.special(4,"lidx1")`. Each warp owns one of `G=Hq/Hkv=4`
  query heads (`h = kvh*G + warp`). **The microgate must first prove this 128-thread workgroup with a
  per-warp 32-lane cross-lane reduce is expressible**; if not, emit
  `SEARCH_BLOCKED_BY_CODEGEN__MULTI_WARP_NOT_EXPRESSED` and stop.
- LDS: `ksh = placeholder((TK*Hd,), half, LOCAL)`, `vsh = placeholder((TK*Hd,), half, LOCAL)`, `TK=16` →
  8192 B total.
- Outer block loop over `NB=ceil(L/TK)` blocks; each block: all 128 threads cooperatively stage `TK*Hd` K
  and `TK*Hd` V from raw `cache_kv` `[2,1,Hkv,MAXC,Hd]` into ksh/vsh (16 elems/thread), then ONE barrier.
- Inner loop over `TK` tokens: warp computes fdot2 q·k (lane e-shards Hd reading K from ksh), cross-lane
  warp_reduce → score; online-softmax (m,l); PV accumulate from vsh (lane d-shards Hd).
- Output per `(kvh, warp, s)`: `acc[Hd] + l + m` (W=Hd+2), consumed by the existing
  `flash_state_gmax_kernel` + `flash_state_combine_kernel`. Raw 5D `cache_kv` identity.
- Default-off: behind `DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE` + a new `DECODE_ATTN_BLOCK_TILE=1` sub-flag.

Microgate `extra/qk_decode_attention_block_tile_microgate.py`: validate vs a NumPy per-split oracle
(scalar fp32 `max_abs≤1e-7`; fp16 `≤2e-5`) over `Tc∈{128,130,32,256}`, `Hq=32,Hkv=8,Hd=128`. Verdicts
`BLOCK_TILE_MICROGATE_PASS` / `_FAIL__NUMERIC` / `_BLOCKED__UOP_VERIFY`.

## Acceptance (Task B), strictly in order

1. `BLOCK_TILE_MICROGATE_PASS`.
2. Route gate `FUSED_XLANE_SCORE_PV_ROUTE_CLEAN__ECONOMICS_NEXT` (token-match, no materialization, owned
   absent).
3. ISA-vec gate (Task A) on the block tile: `ISA_VEC_AUTHORITATIVE_PASS`, with LDS toward 8192 B,
   cross_lane 20→toward 5, wide-load markers > 0.
4. W==D improves materially at ctx512+ (report before/after vs 7.2/4.1/1.1 and vs baseline 103.5/101.8/94.6).

## Commands

```
DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk_decode_isa_vectorization_gate.py            # A
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_attention_block_tile_microgate.py                       # B microgate
DEV=AMD JIT=1 DECODE_ATTN_BLOCK_TILE=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_route_gate.py
DEV=AMD JIT=1 DECODE_ATTN_BLOCK_TILE=1 PYTHONPATH=. python3 extra/qk_decode_isa_vectorization_gate.py
DEV=AMD JIT=1 DECODE_ATTN_GENERATED_WHOLECACHE=1 DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1 DECODE_ATTN_BLOCK_TILE=1 V_DOT2_LOWERING=1 \
  PYTHONPATH=. python3 extra/qk_decode_runtime_overhead.py
```

## Constraints / labels

- Default-off; correctness-first; microgate before model. Do not revive score-broadcast; do not add another
  attention layout. Do not edit `tinygrad/runtime/autogen/**`. Bracketed-prefix commits (e.g. `[nn]`,
  `[codegen]`), with gate verdicts in the message.
- Report honestly. Do NOT claim coalesced loads or block-tiling "done" unless the **ISA-vec gate** is
  `ISA_VEC_AUTHORITATIVE_PASS` AND W==D moved. If the tile can't be expressed:
  `SEARCH_BLOCKED_BY_CODEGEN__MULTI_WARP_NOT_EXPRESSED`. If it expresses + vectorizes but W==D stays slow:
  `SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING`. If the whole approach stalls:
  `SEARCH_BLOCKED_BY_CODEGEN__BLOCK_TILED_MULTI_WARP_TILE_NOT_EXPRESSED`.

Deliverable: Task A gate committed (showing the honest scalar-load baseline); Task B microgate + tile with
the ordered acceptance results, W==D before/after, and ISA markers before/after.
