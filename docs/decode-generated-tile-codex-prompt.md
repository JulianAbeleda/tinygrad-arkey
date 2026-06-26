# Codex task prompt — make the generated decode tile lower to competitive ISA

Copy everything below the line into Codex. Full scope/rationale: `docs/decode-generated-tile-codegen-scope.md`.

---

You are working in the tinygrad fork at `/home/ubuntu/tinygrad-arkey` (AMD gfx1100 / RX 7900 XTX). Hardware
is present; run real GPU jobs with `DEV=AMD JIT=1 PYTHONPATH=.`.

## Objective

Make tinygrad's AMD codegen lower a GENERATED decode-attention tile to competitive ISA, so the generated
route's whole-decode tok/s approaches the shipped baseline. The kernel is already numerically correct,
route-clean, and occupancy-matched; it is ~99× too slow purely because its ISA is scalar / non-block-tiled
while the hand-written owned kernel is vectorized + LDS-block-tiled + compiler-scheduled. Do NOT change the
attention algorithm or layout — this is strictly a codegen-strategy task (vectorize loads, block-tile,
schedule). Everything stays behind `DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE` (default-off); the shipped
default route and the q4k GEMVs must be unchanged.

## The kernel and how it runs

- Generated tile: `extra/qk_flash_decode.py:841` `flash_fused_xlane_score_pv_tile_whole_cache_kernel`
  (emits program `flash_fused_xlane_score_pv_tile_whole_cache_32_128`). Route branch that fires it:
  `extra/qk_flash_decode.py:1225-1240` (env `DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1`; split count
  `DECODE_ATTN_FUSED_XLANE_SCORE_PV_S` default 48; passes raw `cache_kv` `[2,1,Hkv,MAXC,Hd]` for buffer
  identity). Helpers: `_fki` `:116`, fdot2 build `:886-888`, cross-lane `_warp_reduce_sum_staged`
  (`extra/qk_warp_reduce_lowering.py:33`).
- Structural target = the owned hand-written tile `extra/qk_owned_flash_decode.hip:225`
  `owned_flash_tile_gqa_whole`: 128 threads / 4 warps per (kvh,split), TK=16 token block staged into
  8 KB LDS (`__shared__ __half ksh[TK*Hd], vsh[TK*Hd]`), fdot2 + `__shfl_xor` + online softmax, vectorized
  fp16 loads. It is hipcc-compiled and injected as an inert program — your generated tile must reach
  similar ISA through tinygrad's HIPRenderer→comgr path.

## What you must know about the codegen (verified)

1. Custom `custom_kernel` UOps skip the auto-UPCAST optimizer because `_fki` sets `opts_to_apply=()`
   (`tinygrad/codegen/opt/postrange.py:357-358`), so loops stay scalar. BUT the load coalescer still runs:
   `tinygrad/codegen/late/devectorizer.py` `load_store_folding` `:136-149`, `fold_expanded_index` `:81-117`,
   `split_load_store` `:153-200` (widths `[4,2]`, `[8,4,2]` with `ALLOW_HALF8=1`). It fires only if the
   access is a single vectorized `INDEX` (vec offset) or a `STACK` of contiguous `ptr=True` INDEXes.
2. The renderer needs NO change to emit wide loads: a vec-dtype LOAD becomes a C vector cast
   (`tinygrad/renderer/cstyle.py:206-210`); comgr/LLVM picks `global_load_b128`/`d16`. `supports_float4` is
   on for AMD.
3. No software pipeliner runs (`tinygrad/codegen/late/linearizer.py` is a topo-sort). But Track A compiles
   via comgr/LLVM, which DOES vectorize+schedule a well-structured kernel — so structuring the kernel like
   the owned `.hip` may suffice without a tinygrad scheduler.
4. To add a codegen pass if needed, clone the v_dot2 recipe: a `PatternMatcher` in `extra/` →
   `Ops.CUSTOMI` carrying raw ISA (`extra/qk_fdot2_lowering.py`), env-gated `graph_rewrite` in
   `tinygrad/codegen/__init__.py:112-114`, and add the env to the cache key at `:255`.

## Plan (do in order; each phase gated by the harness)

- **Phase 0:** run all four gates (below), save the current `disasm_*xlane*.txt` as the "before".
- **Phase 1 — vectorize loads (kernel-authoring):** make the K-stage and V loads coalesce. Try in order:
  (a) declare the contiguous element axis as `AxisType.UPCAST` (not `REDUCE`) so the expander
  (`tinygrad/codegen/late/expander.py:147-151`) widens the INDEX; (b) else build a `STACK` of contiguous
  `ptr=True` INDEXes so `fold_expanded_index` coalesces; (c) keep V in fp16 and set `ALLOW_HALF8=1`.
  Accept when the ISA-diff shows `xlane global_load_d16 > 0` (or `dwordx4 > 0`) and microgate+route gate
  still PASS. Re-time the isolated tile.
- **Phase 2 — block-tile + multi-warp (kernel-authoring, mirror the owned .hip):** restructure to 4 warps /
  128 threads per workgroup, stage a TK=16 K(+V) block into 8 KB LDS once per block (one barrier/block),
  inner loop reads from LDS. Keep d-sharded PV, fdot2, cross-lane reduce, online softmax, S=48, raw
  cache_kv. Prototype + numerically validate in the microgate FIRST, then port in-model.
- **Phase 3 — residual:** re-run W==D. If still gated by scheduling and comgr won't pipeline, record
  `SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING` (wiring `extra/qk_asm_scheduler.py` is a separate scope).
- If the existing coalescer/expander cannot be triggered from the custom kernel, that is the real codegen
  gap — fix it with an env-gated pass (v_dot2 recipe) and document it as the milestone.

## Acceptance harness (run after every change; do not regress)

```
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_microgate.py   # FUSED_XLANE_SCORE_PV_MICROGATE_PASS
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_route_gate.py   # FUSED_XLANE_SCORE_PV_ROUTE_CLEAN__ECONOMICS_NEXT
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_attention_isa_diff_gate.py                     # ISA_DIFF_PINNED; read markers LDS / global_load_d16 / cross_lane
DEV=AMD JIT=1 DECODE_ATTN_GENERATED_WHOLECACHE=1 DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE=1 V_DOT2_LOWERING=1 \
  PYTHONPATH=. python3 extra/qk_decode_runtime_overhead.py                                        # tok/s vs baseline 82.4/103.5/101.8/94.6
```
Target ISA (owned, from the diff): LDS ~8192 B, `global_load_d16` ~22, `cross_lane` ~5. Current generated:
LDS 256, `global_load_d16` 0, `cross_lane` 20. Success = W==D approaches baseline with the gates still PASS.

## Constraints

- Default-off; shipped default route + q4k GEMVs unchanged; any new codegen pass env-gated AND in the
  cache key (`tinygrad/codegen/__init__.py:255`).
- Correctness first — never regress the microgate/route-gate token-match `[315,24231,6009,979,220,576]`.
  Cross-lane/LDS/fdot2 are divergence- and shape-sensitive: follow `extra/amd_warp_reduce.py:1-13` (stage
  wave ops into a REG; `CUSTOMI` carries `src[0]` shape, `tinygrad/uop/ops.py:306`).
- Do not edit `tinygrad/runtime/autogen/**` (generated). Do not commit non-deterministic bench timings.
- Commit per phase with the gate verdicts in the message; bracketed-prefix commit messages are required by
  the repo hook (e.g. `[codegen] ...`, `[nn] ...`).

Deliverable: the generated route's W==D approaching baseline (or, if a hard codegen gap is found, a precise
env-gated reproduction of it with the gate verdicts and a one-line label). Report the W==D before/after and
the ISA-diff markers before/after.
