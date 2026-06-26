# Owned-vs-generated ISA diff — result (2026-06-26)

`extra/qk_decode_attention_isa_diff_gate.py` (built on the attribution tool's `_disasm`/`_hist`/`_parse_desc`
+ the `static_isa_diff` key_diff format) pins where the generated decode tile bleeds vs the owned AMDGCN
tile. Verdict `ISA_DIFF_PINNED`. Static ISA is ctx-independent; artifact
`bench/qk-decode-attention-isa-diff/latest.json`, disasm in `disasm_*.txt`.

## The diff (owned_flash_tile_gqa_whole vs flash_fused_xlane_score_pv_tile_whole_cache_32_128)

| signal | owned | xlane | read |
|---|---:|---:|---|
| total instructions | 557 | 555 | **NOT static bloat** — counts ~equal |
| VGPR / scratch | 64 / 0 | 72 / 0 | comparable, no spill on either |
| **LDS bytes** | **8192** | **256** | owned stages a **K-block** (~32 tokens × 128 fp16); xlane stages ~1 token |
| **global_load_d16** (fp16 vec) | **22** | **0** | owned vectorizes loads; xlane casts V to fp32 → scalar loads |
| **cross_lane** | **5** | **20** | owned amortizes the reduction; xlane warp-reduces per token |
| exp | 2 | 16 | per-token online-softmax exp (8×) |
| s_waitcnt | 21 | 28 | more memory stalls in the generated loop |

## Conclusion

The W==D 99× gap (refuted in `decode-fused-xlane-score-pv-tile-wd-result.md`) is **not** the lane layout
(proven correct) and **not** static instruction bloat (557 ≈ 555). It is a **code-generation strategy**
difference: the owned hand-written tile **block-processes** tokens — stages a tile of K into 8 KB of LDS,
issues vectorized fp16 loads, and amortizes the cross-lane reduction across the block — giving high
throughput/ILP per iteration. The tinygrad-generated tile processes **one token at a time**: 256 B of LDS
(single token), scalar fp32 loads, and a cross-lane warp_reduce on the per-token critical path. Same
instruction count, opposite throughput → the generated loop is per-token latency-bound.

This is `SEARCH_BLOCKED_BY_CODEGEN` at the **codegen-strategy** level (per-token vs block-tiled), i.e. the
renderer/lowering frontier — precisely the north-star "v_dot2 + cross-lane lowering" gap, now with a
concrete, measured worklist.

## Renderer / lowering worklist (prioritized)

1. **Block-tile the token loop** — stage a block of K (and V) into LDS once, process N tokens per LDS load.
   The single biggest lever: it amortizes loads + the cross-lane reduction and lifts ILP (owned: 8 KB LDS).
2. **Vectorize K/V global loads** — emit fp16 `global_load_d16`/wide (`b128`) instead of scalar fp32
   (xlane currently casts V to fp32 → 0 d16 loads).
3. **Amortize / hoist the cross-lane reduction** out of the per-token critical path (block-level reduce),
   so the warp_reduce is not on every token's dependency chain (xlane 20 vs owned 5).

After each renderer change, **re-diff with this gate** (it's reusable for any generated-vs-owned tile);
do not write another attention layout.

## Caveat / next confirming tool

The gate's per-kernel **dynamic** ms returned 0 — the DEBUG=2 per-program lines are absent under JIT graph
replay (the timing path needs an eager/PROFILE capture). The **static resource diff** (LDS, vectorization,
cross-lane) already localizes the structural cause, but to *confirm the tile is the dominant kernel* (vs the
48-split combine) use the attribution tool's proven timing path or `PROFILE=1`. That dynamic per-kernel
attribution is the one remaining gap before committing renderer effort to the tile.
