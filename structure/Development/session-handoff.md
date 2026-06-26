# Session Handoff

<!-- CANONICAL_BENCHMARKS:START -->
## Current Benchmark Authority

Source of truth:

- `bench/canonical-benchmarks.json`
- `docs/pure-machine-search-roadmap.md`
- `docs/decode-generated-tile-codegen-scope.md`
- `docs/decode-generated-tile-codex-prompt.md`
- `docs/gemv-pure-search-generated-route-scope.md`
- `docs/gemv-g2-minimal-codegen-representation-scope.md`
- `docs/gemv-g3-codegen-lowering-scope.md`
- `docs/decode-attention-pure-search-scope.md`
- `docs/decode-attention-a1-generated-skeleton-scope.md`
- `docs/decode-attention-a1-generated-skeleton-result.md`
- `docs/decode-attention-a2-wholecache-skeleton-result.md`
- `docs/decode-attention-a3-performance-primitive-lowering-scope.md`
- `docs/decode-attention-a3-baseline-result.md`
- `docs/decode-attention-a3-1-vdot2-score-lowering-scope.md`
- `docs/decode-attention-a3-1-vdot2-probe-result.md`
- `docs/decode-attention-a3-1-vdot2-score-result.md`
- `docs/decode-attention-a3-2-cross-lane-result.md`
- `docs/decode-attention-a3-2b-scoped-lane-map-scope.md`
- `docs/decode-attention-a3-2b-lane-map-probe-result.md`
- `docs/decode-attention-a3-2b-xlane-score-result.md`
- `docs/decode-attention-a3-3-lds-tile-lifecycle-result.md`
- `docs/decode-attention-a3-4-tile-combine-lifecycle-result.md`
- `docs/decode-attention-a3-5-minimal-tile-placeholder-result.md`
- `docs/decode-attention-a3-6-tile-score-max-result.md`
- `docs/decode-attention-a3-7-tile-prob-result.md`
- `docs/decode-attention-a3-8-stage-attribution-result.md`
- `docs/decode-attention-a3-9-tile-partial-pv-result.md`
- `docs/decode-attention-a3-10-tile-prob-partial-pv-result.md`
- `bench/qk-search-spaces/decode_attention_tile_combine_a3_4.json`
- Update derived docs with `PYTHONPATH=. .venv/bin/python extra/qk_update_benchmark_refs.py`.
- Check derived docs with `PYTHONPATH=. .venv/bin/python extra/qk_update_benchmark_refs.py --check`.

Current baseline snapshot:

- Decode baseline @ctx512/1024/2048/4096: `101.6 / 99.8 / 97.3 / 92.7` tok/s.
- Decode BubbleBeam FutureSight @ctx512/1024/2048/4096: `103.5 / 101.6 / 99.1 / 94.4` tok/s (`BUBBLEBEAM_FUTURESIGHT=1`, default-off selector).
- Decode aggressive probe, measured but not promoted: `103.4 / 101.6 / 99.1 / 94.4` tok/s.
- Decode aggressive target envelope: `104.0 / 102.1 / 99.6 / 95.1` tok/s.
- Prefill baseline @ctx512/1024/2048/4096/8192: `3574 / 3573 / 3572 / 3571 / 3569` tok/s.
- Latest decode lifecycle run: `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-200800`.
- Latest BubbleBeam artifact: `bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_20260625-162422.json`.
- Latest GEMV purity gate: `bench/qk-gemv-purity-gate/latest.json` (`GEMV_PURE_SEARCH_GENERATED__BUBBLEBEAM_G3_FULL_Q4K_GEMV`).
- Latest decode attention purity capture: `bench/qk-decode-attention-purity/latest.json` (`DECODE_ATTENTION_NOT_PURE__OWNED_TILE_COMBINE`).
- Latest decode attention A1 generated skeleton gate: `bench/qk-decode-attention-generated-skeleton/latest.json` (`DECODE_ATTENTION_A1_FAIL__E_49152_REINTRODUCED`).
- Latest decode attention A2 whole-cache skeleton gate: `bench/qk-decode-attention-wholecache-skeleton/latest.json` (`DECODE_ATTENTION_A2_GENERATED_WHOLECACHE_ROUTE_CLEAN`).
- Latest decode attention A3 baseline: `bench/qk-decode-attention-a3-baseline/latest.json` (`DECODE_ATTENTION_A3_BASELINE_CAPTURED`).
- Latest decode attention A3.2b x-lane score gate: `bench/qk-decode-attention-a3-2b-xlane-score/latest.json` (`A3_2B_CROSS_LANE_NO_TRANSFER`).
- Latest decode attention A3.3 LDS/tile lifecycle gate: `bench/qk-decode-attention-a3-3-lds-tile/latest.json` (`A3_3_BLOCKED_BY_ROUTE_BINDING`).
- Latest decode attention A3.4 TILE+COMBINE lifecycle gate: `bench/qk-decode-attention-a3-4-tile-combine/latest.json` (`A3_4_ROUTE_BINDING_MISSING`).
- Latest decode attention A3.5 minimal tile placeholder gate: `bench/qk-decode-attention-a3-5-tile-placeholder/latest.json` (`A3_5_TILE_PLACEHOLDER_NO_TRANSFER`).
- Latest decode attention A3.6 tile score+max gate: `bench/qk-decode-attention-a3-6-tile-score-max/latest.json` (`A3_6_TILE_SCORE_MAX_NO_TRANSFER`).
- Latest decode attention A3.7 tile probability gate: `bench/qk-decode-attention-a3-7-tile-prob/latest.json` (`A3_7_TILE_PROB_NO_TRANSFER`).
- Latest decode attention A3.8 stage attribution audit: `bench/qk-decode-attention-a3-8-stage-attribution/latest.json` (`A3_8_ATTRIBUTION_READY__PARTIAL_PV_NEXT`).
- Latest decode attention A3.9 tile partial-PV gate: `bench/qk-decode-attention-a3-9-tile-partial-pv/latest.json` (`A3_9_TILE_PARTIAL_PV_NO_TRANSFER`).
- Latest decode attention A3.10 tile prob+partial-PV gate: `bench/qk-decode-attention-a3-10-tile-prob-partial-pv/latest.json` (`A3_10_TILE_PROB_PARTIAL_PV_NO_TRANSFER`).
- Latest generated decode tile codegen isolation: `bench/qk-decode-cache-identity-index/latest.json` (`SEARCH_BLOCKED_BY_CODEGEN__DYNAMIC_UPCAST_REG_STORE_AND_PTRCAT_PLACEMENT`): 5D cache indexing, static UPCAST, dynamic scalar indexing, and K-UPCAST-to-LDS work; dynamic V reduce + UPCAST accumulator emits invalid `make_float4(...) = make_float4(...)`, and direct `PTRCAT` authoring fails spec. Next milestone is an env-gated late-codegen coalesced-load lowering that leaves register accumulator stores scalar.
- GEMV generated skeleton: `q4k_gemv_generated_skeleton` (`Q4K_GEMV_SCHEDULER=2`) is registered for attribution only; expected to fail W==D speed until codegen representation lands.
- GEMV G2.0-G2.2 representation result: `G2_LANEMAP_ADDRESS_BUILDER_PASS` (`extra/qk_gemv_g2_lanemap.py`, `extra/qk_gemv_g2_representation_probe.py`, `bench/qk-gemv-g2-representation-probe/latest.json`). UOp/RANGE can express `lane = block_group * 8 + word_col`, the minimal Q4_K LaneMap is bridge-independent/serializable, and the generated packed-word index matches the numeric reference.
- GEMV G2.3 runtime binding result: `SEARCH_GENERATED_WD_FAIL` (`Q4K_GEMV_SCHEDULER=5`, `q4k_scheduler_matvec_lanemap`). It is token-correct and route-clean, but only `14.2 / 14.2 / 14.1 / 14.0` tok/s @ctx512/1024/2048/4096 versus owned `103.4 / 101.5 / 98.8 / 94.2`.
- GEMV G3.0 codegen capture: `G3_CODEGEN_MISMATCH_CAPTURED` (`extra/qk_gemv_g3_codegen_capture.py`, `bench/qk-gemv-g3-codegen-capture/latest.json`). Owned and bridge each expose a named wave32 gate/up program 72 times; G2 LaneMap exposes zero named gate-up programs and lowers into generic Tensor programs.
- GEMV G3.1 lowering hook: `G3_LANEMAP_PROMOTABLE` (`Q4K_GEMV_SCHEDULER=6`, `q4k_g3_lanemap_gemv_12288_4096`). W==D tok/s `103.7 / 101.7 / 99.4 / 94.5`, token-correct, route-clean, no owned warp gate/up and no lane-partition bridge gate/up.
- GEMV G4 BubbleBeam binding: complete. FutureSight now routes to the generated G3 LaneMap program; purity gate verdict is `GEMV_PURE_SEARCH_GENERATED__BUBBLEBEAM_G3`. The old lane-partition bridge remains explicit-only as `Q4K_GEMV_SCHEDULER=4` fallback/debug route.
- GEMV full tracked Q4_K purity: complete. Verdict `GEMV_PURE_SEARCH_GENERATED__BUBBLEBEAM_G3_FULL_Q4K_GEMV`. BubbleBeam/FutureSight routes gate/up (`g3_lanemap_gateup: 72`), FFN down (`g3_lanemap_down: 18`), and Q4_K `4096x4096` projection (`g3_lanemap_proj: 72`) through generated G3 LaneMap programs, with no owned Q4_K GEMV or lane-partition bridge under BubbleBeam.
- Decode attention A0 purity capture: complete. Verdict `DECODE_ATTENTION_NOT_PURE__OWNED_TILE_COMBINE` (`extra/qk_decode_attention_purity_capture.py`, `bench/qk-decode-attention-purity/latest.json`). Current default route fires `owned_flash_tile_gqa_whole` + `owned_flash_combine`, keeps buffer identity, and avoids `E_49152`.
- Decode attention A1 generated skeleton: complete with precise blocker. Verdict `DECODE_ATTENTION_A1_FAIL__E_49152_REINTRODUCED` (`extra/qk_decode_attention_purity_capture.py --a1`, `bench/qk-decode-attention-generated-skeleton/latest.json`). Generated flash programs fire and tokens match, but sliced KV inputs reintroduce `E_49152_32_3`; not promotable.
- Decode attention A2 generated whole-cache skeleton: complete. Verdict `DECODE_ATTENTION_A2_GENERATED_WHOLECACHE_ROUTE_CLEAN` (`extra/qk_decode_attention_purity_capture.py --a2`, `bench/qk-decode-attention-wholecache-skeleton/latest.json`). Generated flash programs fire, owned tile/combine do not fire, tokens match, and `E_49152` is absent. This is attribution-only, not a speed promotion.
- Decode attention A3 baseline: complete. Verdict `DECODE_ATTENTION_A3_BASELINE_CAPTURED` (`extra/qk_decode_attention_a3_baseline.py`, `bench/qk-decode-attention-a3-baseline/latest.json`). A2 is lifecycle-clean but runs at `74.4 / 73.1 / 69.0 / 63.1%` of owned W==D at ctx `512 / 1024 / 2048 / 4096`.
- Decode attention A3.1 scope: ready (`docs/decode-attention-a3-1-vdot2-score-lowering-scope.md`). It uses the owned route as the flatline oracle and scopes the smallest generated primitive step: expose `v_dot2` in `flash_score_whole_cache_32_128` without losing A2 lifecycle cleanliness.
- Decode attention A3.1 v_dot2 probe: complete. Verdict `A3_1_RENDERER_VDOT2_PROBE_PASS` (`extra/qk_decode_attention_a3_1_vdot2_probe.py`, `bench/qk-decode-attention-a3-1-vdot2/latest.json`). Existing opt-in `V_DOT2_LOWERING=1` can render `__builtin_amdgcn_fdot2` in generated tinygrad code and returns the expected dot result.
- Decode attention A3.1 score wiring: complete. Artifact `bench/qk-decode-attention-a3-1-vdot2-score/latest.json`, doc `docs/decode-attention-a3-1-vdot2-score-result.md`. Route stays clean, tokens match, owned flash stays off, `E_49152` stays absent, and `flash_score_whole_cache_vdot2_32_128` is captured. W==D is flat versus A2 (`100.3 / 100.0 / 100.1 / 99.8%`), so there is no material transfer and nothing to promote.
- Decode attention A3.2 cross-lane gate: complete with blocker. Verdict `A3_2_BLOCKED_BY_CODEGEN_GLOBAL_WARP_REDUCE` (`extra/qk_decode_attention_a3_2_cross_lane_gate.py`, `bench/qk-decode-attention-a3-2-cross-lane/latest.json`). Global `WARP_REDUCE_LOWERING=1` fails UOp verification during decode capture (`Ops.UNROLL dtypes.float ... ((4, 4),)`) before W==D can run.
- Decode attention A3.2b scope: ready (`docs/decode-attention-a3-2b-scoped-lane-map-scope.md`). Cross-lane lowering must be applied only to the generated attention candidate/reductions intended for explicit lane ownership, not globally across the model.
- Decode attention A3.2b lane-map probe: complete. Verdict `A3_2B_ATTENTION_LANE_MAP_NOT_WIRED` (`extra/qk_decode_attention_a3_2b_lane_map_probe.py`, `bench/qk-decode-attention-a3-2b-lane-map/latest.json`). A2 route is clean and lane primitives exist, but only `flash_score_whole_cache_32_128` is present; no x-lane score program is wired.
- Decode attention A3.2b x-lane score wiring: complete. Verdict `A3_2B_CROSS_LANE_NO_TRANSFER` (`extra/qk_decode_attention_a3_2b_xlane_score_gate.py`, `bench/qk-decode-attention-a3-2b-xlane-score/latest.json`, `docs/decode-attention-a3-2b-xlane-score-result.md`). `DECODE_ATTN_SCORE_XLANE=1` captures `flash_score_whole_cache_xlane_32_128` and keeps the A2 route clean, but W==D collapses to `6.0 / 3.2 / 1.5 / 0.7` tok/s at ctx `512 / 1024 / 2048 / 4096`; no promotion.
- Decode attention A3.3 LDS/tile lifecycle gate: complete. Verdict `A3_3_BLOCKED_BY_ROUTE_BINDING` (`extra/qk_decode_attention_a3_3_lds_tile_gate.py`, `bench/qk-decode-attention-a3-3-lds-tile/latest.json`, `docs/decode-attention-a3-3-lds-tile-lifecycle-result.md`). The repo has standalone generated LDS flash-attention evidence in `extra/gemm/amd_flash_attention.py` (`AddrSpace.LOCAL`, barriers, `SHAPED_WMMA`, cross-lane), but `DECODE_ATTN_LDS_TILE=1` still routes the same A2 programs and no decode-bound `flash_*lds*`/`flash_*tile*` candidate appears. No W==D candidate benchmark was run because no LDS/tile route was bound.
- Decode attention A3.4 TILE+COMBINE lifecycle gate: complete. Verdict `A3_4_ROUTE_BINDING_MISSING` (`extra/qk_decode_attention_a3_4_tile_combine_gate.py`, `bench/qk-decode-attention-a3-4-tile-combine/latest.json`, `docs/decode-attention-a3-4-tile-combine-lifecycle-result.md`, manifest `bench/qk-search-spaces/decode_attention_tile_combine_a3_4.json`). The manifest now treats tile program, combine program, split policy, intermediates, materialization guarantees, and primitive requirements as one searchable candidate bundle. A3.4 still routes A2 programs: score/metadata/partial/combine are present, but no generated `flash_*tile*` program is bound, so W==D candidate benchmark is skipped.
- Decode attention A3.5 minimal tile placeholder: complete. Verdict `A3_5_TILE_PLACEHOLDER_NO_TRANSFER` (`extra/qk_decode_attention_a3_5_tile_placeholder_gate.py`, `bench/qk-decode-attention-a3-5-tile-placeholder/latest.json`, `docs/decode-attention-a3-5-minimal-tile-placeholder-result.md`). `DECODE_ATTN_TILE_PLACEHOLDER=1` inserts `flash_tile_placeholder_32_128`; route remains clean, tokens match, and the TILE+COMBINE bundle is bound. W==D is slower than A2: `77.6 / 74.5 / 68.0 / 58.1` tok/s vs A2 `78.5 / 75.9 / 69.9 / 60.7` at ctx `512 / 1024 / 2048 / 4096`.
- Decode attention A3.6 tile score+max: complete. Verdict `A3_6_TILE_SCORE_MAX_NO_TRANSFER` (`extra/qk_decode_attention_a3_6_tile_score_max_gate.py`, `bench/qk-decode-attention-a3-6-tile-score-max/latest.json`, `docs/decode-attention-a3-6-tile-score-max-result.md`). `DECODE_ATTN_TILE_SCORE_MAX=1` routes `flash_tile_score_max_32_128`, removes separate `flash_max_32`, keeps tokens/route/materialization clean, and binds the bundle. W==D is flat/slightly negative vs A2: `78.0 / 75.3 / 69.5 / 60.2` tok/s vs A2 `78.1 / 75.4 / 69.5 / 60.4` at ctx `512 / 1024 / 2048 / 4096`. First attempted score+max multi-output kernel hit a UOp grouped-output shape limitation, so the committed payload is max-only over the existing score buffer.
- Decode attention A3.7 tile probability: complete. Verdict `A3_7_TILE_PROB_NO_TRANSFER` (`extra/qk_decode_attention_a3_7_tile_prob_gate.py`, `bench/qk-decode-attention-a3-7-tile-prob/latest.json`, `docs/decode-attention-a3-7-tile-prob-result.md`). `DECODE_ATTN_TILE_PROB=1` routes `flash_tile_score_max_32_128` and `flash_tile_prob_32_128`, removes separate `flash_max_32` and `flash_prob_32`, keeps route/materialization/tokens clean, and binds the bundle. W==D is not promotable: `76.6 / 74.4 / 69.4 / 61.3` tok/s vs A2 `78.5 / 75.8 / 69.9 / 60.7` at ctx `512 / 1024 / 2048 / 4096`; short/mid contexts regress and only ctx4096 has a small uptick.
- Decode attention A3.8 stage attribution audit: complete. Verdict `A3_8_ATTRIBUTION_READY__PARTIAL_PV_NEXT` (`extra/qk_decode_attention_a3_8_stage_attribution.py`, `bench/qk-decode-attention-a3-8-stage-attribution/latest.json`, `docs/decode-attention-a3-8-stage-attribution-result.md`). It compared A2, A3.6, and A3.7 route/W==D deltas. A3.6 removed `flash_max_32` and stayed flat/slightly negative; A3.7 removed `flash_max_32` + `flash_prob_32`, regressed at short/mid ctx, and only had a small ctx4096 uptick. Metadata replacement is not the main gap.
- Decode attention A3.9 tile partial-PV: complete. Verdict `A3_9_TILE_PARTIAL_PV_NO_TRANSFER` (`extra/qk_decode_attention_a3_9_tile_partial_pv_gate.py`, `bench/qk-decode-attention-a3-9-tile-partial-pv/latest.json`, `docs/decode-attention-a3-9-tile-partial-pv-result.md`). `DECODE_ATTN_TILE_PARTIAL_PV=1` routes `flash_tile_partial_pv_whole_cache_32_128` and removes old `flash_partial_coop_vec_whole_cache_32_128`, with route/materialization/tokens clean. Corrected gate requires delta beyond spread; W==D is flat: `78.2 / 75.6 / 69.7 / 60.6` tok/s vs A2 `78.3 / 75.7 / 69.8 / 60.6` at ctx `512 / 1024 / 2048 / 4096`.
- Decode attention A3.10 tile prob+partial-PV: complete. Verdict `A3_10_TILE_PROB_PARTIAL_PV_NO_TRANSFER` (`extra/qk_decode_attention_a3_10_tile_prob_partial_pv_gate.py`, `bench/qk-decode-attention-a3-10-tile-prob-partial-pv/latest.json`, `docs/decode-attention-a3-10-tile-prob-partial-pv-result.md`). `DECODE_ATTN_TILE_PROB_PARTIAL_PV=1` routes `flash_tile_prob_partial_pv_whole_cache_32_128`, removing both `flash_prob_32` and old partial PV, with route/materialization/tokens clean. W==D regresses materially: `72.6 / 70.7 / 66.2 / 58.8` tok/s vs A2 `78.4 / 75.7 / 69.8 / 60.6` at ctx `512 / 1024 / 2048 / 4096`.
- Next executable step: either exhaust A3.11 score+prob+partial or pivot to primitive-complete online-softmax+PV tile. A3.10 suggests simple incremental fusion loses parallelism/memory shape and is not enough.

Do not hand-edit benchmark numbers in derived docs; change the manifest and rerun the updater.
<!-- CANONICAL_BENCHMARKS:END -->
> ## ⭐⭐⭐⭐ 2026-06-23 — STRUCTURAL EMIT WIN SHIPPED: DBUF default = +2.84% whole-prefill, BYTE-IDENTICAL (default flipped)
> The structural-emit stress study found a REAL transferable win, overturning "scheduling-limited". Swapping the route's
> substep-prefetch (PLRA) for cross-iteration DOUBLE-BUFFER (DBUF) = **+2.84% +/-0.11% whole-prefill@4096, BYTE-IDENTICAL
> output (logit max_abs_diff=0)**, significant at EVERY context (512..8192); DBUF+relocation = +3.87%. Pure scheduling
> (relocation alone) stays at noise (+0.1%) -> *structural* emit TRANSFERS where *scheduling-only* does NOT. Confirmed by
> interleaved paired A/B (clock-drift-cancelled) + byte-identical check. Root cause: DBUF's full block-level pipelining
> beats PLRA's substep-only prefetch IN-MODEL; the old plra=1 default was set on ISOLATED benchmarks (isolated->
> integrated reversal). **ACTION TAKEN: flipped the route default to dbuf=1,plra=0** in
> `extra/qk_prefill_graph_gemm_route.py` (reversible: PREFILL_GEMM_DBUF=0 PREFILL_GEMM_PLRA=1); added global emit knobs
> PREFILL_GEMM_{DBUF,BK,PLRA,PLRAB,LEANADDR}. New prefill default ~3085 vs old ~2975 tok/s @4096 (now ~115% of llama
> pp512). BLOCKED candidates (VGPR-walled, document the structural ceilings): deeper-DepthU bk64 (268>256), PLRAB-4x4
> (300), accumulator-partition (2x128-acc>256), full reg-pool (HW-limited). MACHINE-SEARCH TOOL (committed):
> `extra/qk_prefill_emit_search.py` -- defines the emit SEARCH_SPACE (PREFILL_GEMM_* knobs+domains), `--candidates
> default|grid|--spec`, isolated-subprocess workers, whole-prefill median/CI/significance, INFEASIBLE-on-VGPR-overflow,
> ranked JSON+CSV+MD (`--quick` smoke). Ranks on WHOLE-PREFILL not isolated. Logs /tmp/prefill-emits/. Doc: `docs/prefill-structural-emit-search-result-20260623.md`. NOTE this is
> COMPLEMENTARY to (not a contradiction of) the same-day NO_PRODUCTION_TENSILE_GAP finding: scheduling can't close it,
> but a structural emit swap improves the graph-GEMM default itself.

> Full-8B clock-pinned 3-repeat whole-prefill A/B/C/D. The graph-GEMM route (DEFAULT) BEATS the Tensile route
> (`PREFILL_GRAPH_GEMM=0 PREFILL_TENSILE_GEMM=1`) at EVERY context: graph 3720/3635/3399/2997 vs tensile
> 3506/3429/3200/2793 tok/s @512/1024/2048/4096 (~+6%) -- despite Tensile's ISOLATED GEMM being ~10% faster (66 vs 60
> TFLOPS). Tensile's throughput edge is eaten by in-model integration (layout/transpose kernels, weaker fusion).
> Relocation (PREFILL_GEMM_RELOC, w1/w4) = -0.2..-0.33% @4096 = within 0.13% noise (no transfer), though isolated kv
> sweep confirms it is a real occupancy-driven lever (+2.24%@4WG -> +4.12%@1WG). **KEY: whole-prefill is INSENSITIVE to
> GEMM-kernel-speed in BOTH directions** -- the prefill bottleneck is integration + attention (the @512->@4096 decay
> 3720->2997), NOT the GEMM kernel emit. This RETIRES the 'graph-GEMM ~99.5% of Tensile, close the gap' premise and the
> whole asm-scheduler residual target. Doc: `docs/prefill-tensile-vs-graphgemm-whole-prefill-validation-20260623.md`.
> IMPLICATION for any future prefill-GEMM emit search (DepthU/pipelining/tile-geom): very unlikely to move whole-prefill
> (two independent data points already show GEMM-kernel speedups don't transfer); the bounded ceiling should be measured
> (GEMM fraction of prefill) BEFORE committing to an expensive structural-emit matrix. NB much of that candidate space
> is already implemented (DBUF=cross-iter prefetch, PLRA/PLRAB=substep pipeline, BK=DepthU [BK64 overflows 256 VGPR],
> 8-wave tile) and prior arcs found its ceilings (+0.5% 8-wave; register-lifetime VGPR wall).


> ## ⭐⭐⭐ 2026-06-23 — PREFILL ASM SCHEDULER ARC CLOSED: waitcnt relocation wins isolated (+2-6%) but DOESN'T transfer
> Chased the Inc-3 +2%. (A) kv_halved -4% regression is OCCUPANCY (causal: same kernel, vary LDS -> relocation Δ goes
> +0.08%@4WG/CU, -3.03%@2WG, +4.26%@1WG). Primitives: benefit=LDS-latency-overlap ∝ 1/occupancy; cost=extra-waitcnt
> overhead (~const). Net flips with occupancy; kv (high occ) is zero-benefit noise regime -> exclude. (B) Wired
> `relocate_lgkm_waits` into `extra/qk_prefill_graph_gemm_route.py` behind additive `PREFILL_GEMM_RELOC` (default OFF),
> gated waves_n==2 (non-kv PLRA roles). Clock-pinned synced WHOLE-PREFILL: baseline 3732/3645/3408/3001 vs reloc
> 3713/3629/3391/2989 @512/1024/2048/4096 -- but baseline RE-RUN = 3705/3626/3388/2989, so the ~0.5% reloc gap is < the
> 0.7% run-to-run noise = **NO net whole-prefill effect**. The isolated +2% does NOT transfer in-model (GEMM is a
> fraction of prefill; real-shape occupancy higher than the 4096^3 probe; noise swamps it) -- the standing
> [[inference-perf-measured-map]] lesson. `PREFILL_GEMM_RELOC` ships DEFAULT-OFF (like PREFILL_GEMM_8WAVE). **ARC
> CONCLUSION: the prefill->Tensile residual is NOT recoverable by ANY instruction-scheduling transform of the current
> kernel -- only vendored Tensile (parity, opaque dep) or a structurally different emit (deeper DepthU / cross-iteration
> pipelining) closes it.** Docs: `docs/prefill-asm-instruction-scheduler-inc3-result-20260623.md` (Follow-up A/B + arc
> close). Capability + 3 correctness gates (register DAG / wait model / branch boundaries) shipped default-off.

> ## ⭐⭐⭐ 2026-06-23 — PREFILL ASM SCHEDULER Inc 3 DONE: waitcnt RELOCATION = FIRST NON-NEUTRAL lever (config-dependent)
> Inc 2 = pure reorder is NEUTRAL. Inc 3 changes the instruction SET -- waitcnt RELOCATION -- and WINS: in each compute
> block ([N ds_loads][lgkm(0) full drain][M wmmas]) strip the full drain, issue WMMAs frag-ready-first, insert per-WMMA
> MINIMAL lgkmcnt (Inc-1 wait model) -> overlaps WMMA compute with LDS-load latency. Inserting waits needs branch-offset
> fixup: `capture_branch_targets`/`fix_branches` (shallow-copy, NON-mutating -- a footgun fixed: it was corrupting the
> shared identity stream's branch). New: `relocate_lgkm_waits` in `extra/qk_asm_scheduler.py`. Correct (rmse<=3e-4 +
> verify_wait_correct) across plain/DBUF1/PLRA/kv_halved + K sizes NBLK 16/32/128 (`extra/qk_asm_scheduler_inc3_test.py`
> S1/S2 PASS). MEASURED clean clock-pinned isolated (512x4096x4096): **DBUF1 ~+6% (reproduced 3x), PLRA route ~+2%,
> plain +1.7%, kv_halved ~-4% (REGRESSION)**. Config-dependent: the win is overlapping exposed LDS latency, so
> low-occupancy DBUF1 gains most and high-occupancy small-N kv_halved (latency already wave-hidden) REGRESSES. The
> Inc-1 verify gate caught a real double-emit WAW bug. **Promotion gate NOT crossed** (isolated timing = signal not
> authority; mixed/modest). NEXT (if pursued): wire into `extra/qk_prefill_graph_gemm_route.py` behind additive
> `PREFILL_GEMM_RELOC`, **PLRA-roles only (exclude kv)**, measure clock-pinned synced WHOLE-PREFILL via
> `extra/qk_prefill_whole_synced.py`; promote only if net positive. Doc:
> `docs/prefill-asm-instruction-scheduler-inc3-result-20260623.md`. Capability shipped default-off; no source/default change.
> ⭐SUMMARY of the asm-scheduler arc (Inc 0-3): register DAG (faithful) -> wait-counter model (waits already minimal) ->
> cross-motion sound + pure reorder NEUTRAL -> waitcnt RELOCATION is the only lever that moves the needle (~+2% on route
> roles, config-dependent). The prefill->Tensile residual is NOT recoverable by reordering; only mild gains from
> wait-restructuring. Vendored Tensile remains the path to full parity.

> ## ⭐⭐ 2026-06-23 — PREFILL ASM SCHEDULER Inc 2 DONE: cross-motion SOUND + latency reorder NEUTRAL + Inc1 CORRECTION
> Inc 1's "RDNA3 hardware-spacing hazard" was a MISDIAGNOSIS. The real missing gate is the **loop-entry (backward-branch
> TARGET) control-flow boundary**: build_regions didn't model it, so the fence_only reorder moved instructions across
> the loop entry (prologue<->loop body) -> wrong values in the prologue region only. Fix in `extra/qk_asm_scheduler.py`:
> `branch_target_indices` + `boundaries` arg in build_regions; schedule(fence_only=True) auto-applies. Now fence_only
> cross-motion is BYTE-IDENTICAL-CORRECT across default_PLRA/kv_halved/DBUF1/8wave_PLRAB, both asap (167-310 mem moved)
> and critical modes (`extra/qk_asm_scheduler_inc2_test.py` R1/R2 PASS). ISA research (5-agent) CONFIRMS s_delay_alu is
> PERF-ONLY on RDNA3 (hardware interlocks VALU/VMEM deps) -> a reg-legal+wait-correct reorder cannot corrupt via spacing.
> Built a critical-path latency-aware scheduler (mode='critical', RDNA3 latency model). MEASURED clean clock-pinned
> isolated (DBUF1 512x4096x4096, copies excluded): identity ~287us/59.8 TFLOPS vs critical ~288us = **PERF-NEUTRAL
> (+/-<1%)**. Honest verdict: pure instruction REORDERING does NOT recover the prefill->Tensile residual (hardware
> scoreboard + hand-tuning already capture in-region latency; regions are full-drain-wait-bounded). Three correctness
> gates now complete: register DAG (Inc0) + wait model (Inc1) + loop-entry boundary (Inc2). **Next = Inc 3 (the only
> remaining reorder-class lever)**: WAITCNT RELOCATION -- strip the full-drain lgkmcnt between ds_loads and wmmas,
> interleave them, reinsert per-consumer waits (wait model gives counts; branch offsets now recomputable). Uncertain
> ROI (<=~2-3%, partly beta confound). Docs: `docs/prefill-asm-instruction-scheduler-inc2-result-20260623.md` (+ Inc 1
> doc carries a CORRECTION banner). No source/default/speed change.

> ## ⭐⭐ 2026-06-23 — PREFILL ASM SCHEDULER Inc 1 DONE: wait-counter model (`ASM_SCHED_WAITCOUNT_MODEL_DELIVERED`)
> Built the async-load wait-counter (`s_waitcnt`) model in `extra/qk_asm_scheduler.py`: `verify_wait_correct`
> (soundness gate), `wait_constraints` (audit), `recompute_waits_inplace` (minimal counts, byte-layout preserving),
> and `fence_only` region mode (Inc 2 substrate, default OFF). Proof `extra/qk_asm_scheduler_inc1_test.py` Q1-Q6 PASS
> on gfx1100. TWO HONEST FINDINGS: (1) the hand-placed full drains are ALREADY minimal (total slack=1, a
> perf-irrelevant prologue scalar load) -> standalone consumer-only relaxation is ~free; (2)
> `WAIT_CORRECTNESS_NECESSARY_NOT_SUFFICIENT` -- a register-legal AND wait-correct fence_only reorder still computes
> wrong in the prologue (0 missing register edges; reversing 128 indep movs / moving loadA0 / adjacent swaps all
> correct; regions 0,2-7 incl 120 moved mem ops correct; only the tight prologue cross-motion breaks) = an RDNA3
> hardware-spacing/scoreboard hazard. So cross-motion is kept OFF; the proven-safe reorder stays Inc-0 memory-anchored,
> now composable with the wait model. **Next executable task = Inc 2**: a latency-aware list scheduler over fence_only
> regions gated by BOTH verify_wait_correct AND a new RDNA3 hardware-hazard recognizer (VALU->VMEM/WMMA spacing,
> s_nop/hard-clause), validated on clock-pinned synced whole-prefill -- where the bounded ~2-3% prefill win is realized.
> Doc: `docs/prefill-asm-instruction-scheduler-inc1-result-20260623.md`.

> ## ⭐⭐ 2026-06-23 — PREFILL ASM INSTRUCTION SCHEDULER: scoped + Inc 0 built & proven (`ASM_SCHED_IR_DAG_FAITHFUL`)
> Owner chose the **owned asm-scheduler** path over vendored Tensile to close the ~4% prefill residual the adversarial
> audit attributed to fine-grained instruction scheduling. **Inc 0 DONE**: `extra/qk_asm_scheduler.py` (instruction IR
> with exact register def/use decoded from the encoding; intra-region dependency DAG; pure-compute reorder within
> memory+fence-delimited regions) + `extra/qk_asm_scheduler_inc0_test.py`. Proof on gfx1100 P1-P6 PASS: identity
> byte-identical, **554 instructions legally permuted still rmse 2.07e-4**, across the whole route config space
> (PLRA/DBUF/8-wave-PLRAB). The test caught 2 real bugs (RMW self-edge; an unsound async-load hoist that MMU-faulted →
> memory ops anchored, async modeling deferred to Inc 1). **Next executable task = Inc 1**: the consumer-only
> `s_waitcnt` recompute + wait-counter (async-load) model — the cheapest real lever and where memory-op motion becomes
> sound. Honest ROI: schedulable upside ~2-3% (part of the 4% is a `beta=true` work confound). Validate only on
> clock-pinned synced whole-prefill; additive default-off; no `tinygrad/` source change.
> Doc: `docs/prefill-asm-instruction-scheduler-scope-20260623.md`. NOTE: stop calling decode search "exhausted"
> (owner correction 2026-06-23) — frame remaining decode work as open levers, not a closed surface.

> ## ⭐ 2026-06-23 — LEARNING LAYER REFRAMED: the model is a primitive-space PROPOSER, not a kernel judge
> The learned-model/adapter role in the GPU primitive search system is: emit a **bounded search spec** (`SearchRow`:
> lane / primitive / hypothesis / knobs+bounds / required-evidence / stop-rules), which the **deterministic** runner
> expands and the harness / ISA / correctness / W==D-or-whole-prefill gates decide. **LoRA/SFT first** (structured
> supervised primitive-space generation); **RLVR DEFERRED** until schema + deterministic reward + shadow-mode utility
> are proven. Verdicts: `PRIMITIVE_SPACE_PROPOSER_NOT_KERNEL_JUDGE`, `LORA_FIRST_FOR_PRIMITIVE_SPACE_LEARNING`,
> `RLVR_DEFERRED_UNTIL_SCHEMA_AND_REWARD_STABLE`, `DETERMINISTIC_HARNESS_REMAINS_AUTHORITY`. Doc-update only (no
> training, no search, no source/default change). **Next executable task** = build
> `bench/qk-primitive-space-adapter/dataset-v0` + deterministic scorer `extra/qk_primitive_space_scorer.py`. Also today:
> **ORACLE-GUIDED GPU PRIMITIVE EXPLORER SCOPED** (oracle registry + shared spec + gate stack + ledger over the existing
> backends; unified runner DESIGN-only). Docs: `docs/primitive-space-learning-loop-lora-first-result-20260623.md`,
> `docs/oracle-guided-gpu-primitive-explorer-result-20260623.md`,
> `docs/oracle-guided-gpu-primitive-explorer-runner-design-20260623.md`.

> ## ⭐⭐ SUPERSEDED 2026-06-23 — owned AMDGCN decode-attention is now the DEFAULT (default_on=true)
> The "Route B State" / "B4 W==D fails / no promotion from B4 / opt-in only" narrative below is **superseded**. The
> owned tile had a **dtype-contract bug** (read the fp32 cache as fp16) and an **over-conservative ctx guard** —
> both fixed — plus FO2 (native fp16 cache). It is now **real-cache byte-identical across the whole decode range**
> and the **DEFAULT** decode attention for gfx1100 / Qwen3-8B / B=1 / T=1 (every other shape/device stays gqa+fp32;
> `DECODE_ATTN_AMDGCN_TILE=0` disables). Canonical W==D harness (real decode tok/s) confirms
> **+12.7/+15.4/+18.7/+22.4% @ctx512/1024/2048/4096** → default decode @ctx1024 ~74→~85 tok/s (~76%→~88% of
> llama.cpp). Candidate `decode_attention_llama_flash_tile_owned_amdgcn_b4`: **`default_eligible=true`,
> `default_on=true`**. Runtime-KV is **deferred (incremental)**. The "attention exhausted / B4/B5 sub-bar /
> runtime-KV next" framing is no longer current.
> Authority: PREFILL ADVERSARIAL CORRECTION (2026-06-23) -- RETRACTS the 'HW limit'. Tensile fits pipelined 128x128 GEMM in 256 VGPR; build_gemm_lds2 CAN express the deep pipeline (8-wave W4x2T2x4, 188 VGPR, PIPELINED, correct) -- prior 266>256 was the wrong 4-wave layout. BUT it recovers only +0.5% whole-prefill -> the residual ~4% is FINE INSTRUCTION SCHEDULING (waitcnt/WGM/cadence/SIA1), below the template -> CURRENT_LDS2_REPRESENTATION_EXHAUSTED, needs an asm scheduler. `docs/prefill-adversarial-tensile-liveness-audit-result-20260623.md`. SUPERSEDED prior: PREFILL DEFINITIVELY CLOSED (2026-06-23). Register-lifetime liveness model: the VGPR pool is ALREADY realized for the A-side (PLRA +9-11%); full A+B deep prefetch is HW-register-limited (ideal-pool 266>256 at the occupancy-optimal 4x4 tile; only 32 dead regs = A-only). REGISTER_POOL_INSUFFICIENT_HW_LIMIT -- the ~4-5% Tensile gap is a hardware register-pressure/occupancy ceiling, NOT a missing representation. Prefill speed search CLOSED (Tensile-dep or smaller-tile-tradeoff only). `docs/prefill-register-lifetime-pool-representation-result-20260623.md`. Prior: REPRESENTATION EXPANSION + EMITTER PROOF (2026-06-23). Machine search lost to the oracle because oracle primitives are OUTSIDE the SearchSpace (not the evaluator); designed 10-level taxonomy + 5 gates; prototyped the schedule_interleave_detector (PHASED vs PIPELINED). EMITTER PROOF: build_gemm_lds2 DBUF=1 emits a PIPELINED+correct+no-spill K-loop (KLOOP_SCHEDULE_TEMPLATE_MICROKERNEL_PASS) -- the schedule_template rep IS emittable; but Tensile-class depth (PLRAB) hits the 256-VGPR wall -> PREFILL_FULL_SPEED_SEARCH_STILL_DEFERRED (register_lifetime is the unlock). `docs/{machine-search-representation-expansion-decode-prefill,prefill-kloop-schedule-template-microkernel}-result-20260623.md`. Prior: DECODE + PREFILL ORACLES EXPLAINED (2026-06-23). DECODE: DECODE_ORACLE_EXPLAINED + 8B_SEARCH_SURFACE_EXHAUSTED (oracle wins by full lifecycle; Mode A/B closed; ctx-slope no-action; remaining = <2% slope + codegen-learning). PREFILL: schedule-diff reduces the ~4-5% Tensile gap to K-LOOP SOFTWARE PIPELINING -> PREFILL_HAND_ASM_SCHEDULING_REQUIRED (not searchable; needs register-pool/renderer). `docs/{decode-oracle-explanation-and-schedule-diff,prefill-schedule-diff-oracle-and-search-reduction}-result-20260623.md`. Prior: MODE B + PREFILL SEARCH EXECUTED (2026-06-23). Mode B: DECODE_MODE_B_EXECUTED_ORACLE_REMAINS_BEST (14 variants, tile constants optimal, additive templating byte-identical). Prefill: CORRECTION -- clock-pinned repeats show a STABLE ~4-5% gap to Tensile (not the noise '99.5%'); Phase A READY, Phase B tile-config search finds NO recovering config (gap is K-loop SCHEDULING not tile-config) -> ORACLE_REMAINS_BEST. `docs/{decode-mode-b-search,prefill-search}-result-20260623.md`. Prior: NATIVE-CODEGEN MICROSEARCH EXECUTED (2026-06-23): TARGET_FOUND -- LDS+vector-loads native, v_dot2+cross-lane(ds_bpermute) are the confirmed renderer gaps (ISA-evidenced, 5 correct candidates). `docs/native-codegen-microprimitive-search-result-20260623.md`. Prior: PROJECT MACHINE-SEARCH ROADMAP READY (2026-06-23): repo is search-capable. Decode search DONE (oracle best); PROJECT SEARCH LEDGER built (extra/qk_project_search_ledger.py, 9 entries); prefill search GATED (at-rest); native-codegen microsearch SCOPED+ALLOWED-NOW (targets v_dot2/cross-lane); cross-shape DEFERRED (14B owner-gated). `docs/project-wide-machine-search-roadmap-result-20260623.md`. Prior: DECODE SEARCH EXECUTED (Mode A, 2026-06-23): DECODE_SEARCH_EXECUTED_ORACLE_REMAINS_BEST -- 5/6 PASS gates, none beat oracle outside spread (default S48/base is policy-optimal), W==D-only, artifacts CONFORMS 13/13, no default flip. `docs/decode-machine-search-execution-result-20260623.md`. Prior: DECODE MACHINE-SEARCH READINESS PACKAGE READY (2026-06-23): froze buffer-identity default as oracle (W==D 90.6/89.3), built gate+checkers+runner (extra/qk_decode_search_*.py), SEARCH_RUNNER_READY smoke (oracle PASS, bad REJECTED). Decode NOT worth searching for 8B speed; READY for regression-safe/cross-shape/codegen/portability. `docs/decode-machine-search-readiness-package-result-20260623.md`. Prior: PREFILL AT PARITY (2026-06-23): synced whole-prefill RETIRES the 66% headline -- graph-GEMM ~96-99.5% of Tensile, at/above llama; shipped kv_proj de-WG-starve fix (BN64 small-N, +3-4% whole-prefill, dependency-free, byte-identical). Prefill AT REST. Tool extra/qk_prefill_whole_synced.py. Prior `docs/prefill-per-role-transfer-attribution-result-20260623.md` (PER-ROLE: graph-GEMM within 2.5% of Tensile on concrete chunk; gap = small-N WG-starvation kv_proj 34%; bounded fix = per-shape config NOT search; whole multi-chunk axis TBD) + `docs/prefill-post-decode-parity-frontier-result-20260623.md` (PREFILL FRONTIER AUDIT: kernel at Tensile parity, MACHINE_SEARCH_NOT_READY; next lever = in-model integration penalty 66%->87% via synced per-role time-tax, NON-search; see `docs/prefill-frontier-rest-or-nonsearch-next-scope-20260623.md`) + `docs/decode-campaign-final-synthesis-20260623.md` (⭐⭐DECODE CAMPAIGN COMPLETE: tinygrad decode 102-105% of llama.cpp default-on; whole-cache buffer-identity KV read; runtime-KV lane retired; POST_PARITY_REGRESSION_GUARD_PASS) + `docs/machine-code-translation-roadmap-result-20260623.md` (machine-code map, buffer-identity ABI rule in principles #12, search NOT_READY_FOR_8B_SPEED). Prior: `docs/owned-tile-buffer-identity-kv-read-result-20260623.md` (⭐SHIPPED default-off: buffer-identity whole-cache read = +13-19% BYTE-IDENTICAL, removes E_49152 slice materialization; tinygrad decode now 102-105% of llama.cpp; DEFAULT-ON 2026-06-23 owner-authorized; DECODE_ATTN_KV_IDENTITY=0 disables), `docs/runtime-kv-core-engine-result-v2-20260623.md` (MAJOR CORRECTION: runtime-KV correctness IS achievable via native-store+AFTER-read — 64-tok byte-identical; the callify hard-stop was the opaque-append only. The +11% materialization lever is the owned tile SLICING the cache -> fix = bounded buffer-identity whole-buffer read, NOT a core-engine project). SUPERSEDES `docs/runtime-kv-core-engine-result-20260623.md` (FINAL runtime-KV: RUNTIME_KV_CORE_CAPABILITY_BLOCKED by the callify/pure-function execution model — toy passes, one-layer NaN; fix = tinygrad-core Tensor-purity change, scope hard-stop; 8B decode complete at bounded layer ~88-89% of llama), `docs/three-lane-completion-result-20260623.md` (lanes COMPLETED: ISA wrapper + native-codegen experiment DONE; runtime-KV RUNTIME_KV_CORE_CAPABILITY_BLOCKED — bakes at 1 layer, needs CORE TinyJit/HCQ engine work; 8B bounded speed exhausted at model layer), `docs/runtime-kv-isa-native-codegen-three-lane-result-20260623.md` (lanes 2/3/6 scoped: ISA wrapper BUILT/guard-active; runtime-KV core-persistence DESIGN_A scoped — needs owner auth; native-codegen chartered), `docs/post-exhaustion-remaining-lanes-roadmap-result-20260623.md` (ROADMAP: NEXT = one small-ops fusion gate; runtime-KV deferred owner-decision; machine-search parked; attention+GEMV closed), `docs/post-default-runtime-kv-diagnostic-result-20260623.md` (8B bounded-exhaustion checkpoint: attention+GEMV CLOSED at llama parity; KV-materialization +11.8% but CORE-RUNTIME-BLOCKED; small-ops overlapped; machine-search NOT yet justified), `docs/post-owned-attention-default-audit-result-20260623.md` (FRESH gap map: weight-GEMV+attention at/near llama parity; residual = KV-copy + small-op fusion; tinygrad ~85-88% of llama), `docs/owned-attention-default-flip-result-20260623.md`,
> `docs/post-owned-attention-promotion-synthesis-20260623.md`,
> `docs/owned-tile-post-promotion-four-step-result-20260623.md`,
> `docs/owned-amdgcn-tile-short-ctx-result-20260623.md`,
> `docs/owned-amdgcn-tile-real-cache-revalidation-result-20260623.md`.

Date: 2026-06-21

Repo: `/home/ubuntu/tinygrad-arkey`

Branch: `qk-prefill-flag-leak-resolution`

This is the current handoff. Older shared-storage, flywheel, and early decode notes were removed from this file because
they are now superseded by the doc map and provenance index. Use:

- `docs/README.md`
- `docs/current-project-state-handoff-20260621.md`
- `docs/provenance-index-20260621.md`

## Active objective (2026-06-24): decode lifecycle baseline refresh + periodic protocol

Goal:
- Refresh the decode baseline via one-periodic recheck bundle:
  - correctness/reproducibility gate pre-post (`qk_decode_search_gate.py`)
  - unknown-lockstep proof pre-post (`qk_decode_unknown_bucket_lockstep_audit.py`)
  - interleaved W==D sweep variants (current + long + alternative capture)
- Set the new baseline snapshot as the current reference for decode follow-up audits.

Baseline run completed:
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-172026`
- Decision: `DECODE_LIFECYCLE_RECHECK_BUNDLE_PASS`
- Pointer: `bench/qk-decode-lifecycle-recheck-bundle/latest.json`

Active scope artifacts:
- `docs/prefill-decode-next-workstreams-codex-scope-20260624.md`
- `docs/decode-lifecycle-recheck-bundle-scope-20260624.md`
- `docs/decode-lifecycle-recheck-bundle-result-20260624.md`
- `bench/qk-decode-lifecycle-recheck-bundle/`

Current status:
- Next Codex-executable umbrella scope: `docs/prefill-decode-next-workstreams-codex-scope-20260624.md`.
- Recommended order: prefill long-context hardening first, decode-vs-llama authority refresh second, decode search expansion only if the refreshed decode evidence identifies a material bounded target.
- Decode: `Q4K_GEMV_WARP` + `Q4K_GEMV_WARP_DOWN` promoted default-on for guarded FFN decode shapes (`DECODE_PROMOTE_Q4K_GEMV_WARP_FFN`).
- New decode scope-in artifact for this cycle: `docs/decode-parity-harness-reconciliation-scope-20260624.md`
- Decode promotion result doc: `docs/decode-q4k-gemv-warp-promotion-result-20260624.md`.
- Long-context prefill: `eightwave` promoted as the prefill graph-GEMM emit default (`PREFILL_PROMOTE_EIGHTWAVE_ONLY`).
- New result doc: `docs/prefill-long-context-no-regression-audit-result-20260623.md`.
- Completed prefill interaction check: `docs/prefill-eightwave-oldplra-interaction-scope-20260624.md`; decision is `eightwave` alone.
- Promotion result doc: `docs/prefill-eightwave-promotion-result-20260624.md`.
- Long-context root-cause root-pass: `docs/prefill-long-context-root-cause-audit-result-20260624.md` completed.
- Root-cause outcome: `PREFILL_ROOTCAUSE_LONG_CTX_INTEGRATION_BOUND` (no harness-only cause; single-chunk vs whole-gap confirms multi-chunk integration slope).
- Next scoped Spark handoff:
  `docs/prefill-long-context-integration-hardening-scope-20260624.md`.
- Deep follow-up root-cause scope for continuity:
  `docs/prefill-long-context-root-cause-audit-scope-20260624.md`.

### A) Decode audit scope (long-context slope + route attribution)

- Source scope:
  - `docs/decode-q4k-gemv-warp-promotion-result-20260624.md`
  - `docs/decode-parity-no-regression-audit-scope-20260623.md`
  - `docs/decode-parity-harness-reconciliation-scope-20260624.md`
  - `docs/decode-ctx-slope-audit-scope-20260623.md`
  - `docs/decode-ctx-slope-audit-result-20260623.md`
  - `docs/owned-tile-buffer-identity-kv-read-result-20260623.md`
  - `bench/qk-decode-eval/HARNESS_GUIDE.md`
- Required tools:
  - `extra/qk_decode_runtime_overhead.py`
  - `extra/qk_decode_time_tax_audit.py`
  - `extra/qk_decode_materialization_check.py`
  - `extra/qk_decode_route_fire_check.py`
  - `extra/qk_isa_primitive_audit.py`
- Required artifact folder:
  - `bench/qk-decode-ctx-slope-audit/`
- Required outputs:
  - `authority.json`, `wd_by_ctx.json`, `kernel_attribution_by_ctx.json`, `slope_fit.json`, `llama_comparison.json`, `decision.json`
- Must run with clean synced W==D as authority (`.item()` inside timed loop).
- Required contexts:
  - Primary: 512, 1024, 2048, 4096
  - Optional: 3072, 6144 if supported safely
- Comparative configs:
  - Config A: new default (`DECODE_ATTN_KV_IDENTITY=1`, owned whole-cache tile)
  - Config B: old slice/materialization route (`DECODE_ATTN_KV_IDENTITY=0`)
  - Config C: legacy comparator (`DECODE_ATTN_AMDGCN_TILE=0`) if cheap
- Completion decisions:
  - `CTX_SLOPE_AUTHORITY_LOCKED`
  - `CTX_SLOPE_WD_MEASURED`
  - `CTX_SLOPE_ROUTE_CONFIRMED`
  - `CTX_SLOPE_DECISION_READY`
- Fail-safe stop:
  - correctness mismatch, route mismatch, unstable spread, or incomplete authority lock.

### C) Decode periodic lifecycle protocol

- Baseline periodic script:
  - `extra/qk_decode_lifecycle_recheck_periodic.py`
- Scope doc:
  - `docs/decode-lifecycle-recheck-bundle-periodic-scope-20260624.md`
- Result artifact:
  - `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-<RUN_ID>/periodic_diff.json`
- Command:
  - `python3 extra/qk_decode_lifecycle_recheck_periodic.py --out-root bench/qk-decode-lifecycle-recheck-bundle`

  This script:
  - runs the full periodic bundle (or compare-only with `--compare-only`),
  - stores `periodic_diff.json`/`periodic_diff.md` in the run directory,
  - updates `bench/qk-decode-lifecycle-recheck-bundle/latest.json`,
  - and compares against prior baseline artifacts.

### B) Prefill long-context audit scope (post-decode frontier)

- Source scope:
  - `docs/prefill-long-context-harness-authority-and-role-tax-scope-20260624.md`
  - `docs/prefill-eightwave-promotion-result-20260624.md`
  - `docs/prefill-eightwave-oldplra-interaction-scope-20260624.md`
  - `docs/prefill-long-context-no-regression-audit-scope-20260623.md`
  - `docs/prefill-post-decode-parity-frontier-scope-20260623.md`
  - `docs/prefill-post-decode-parity-frontier-result-20260623.md`
  - `docs/prefill-frontier-rest-or-nonsearch-next-scope-20260623.md`
  - `docs/prefill-per-role-transfer-attribution-result-20260623.md`
  - `docs/prefill-structural-emit-search-result-20260623.md`
  - `docs/prefill-structural-emit-search-runbook-20260623.md`
- Required tools:
  - `extra/qk_prefill_whole_synced.py`
  - `extra/qk_prefill_emit_search.py`
  - `extra/qk_prefill_per_role_time_tax.py`
  - `extra/qk_decode_time_tax_audit.py` (for shared tax decomposition)
  - `extra/qk_isa_primitive_audit.py`
- Required artifact folder:
  - `bench/qk-prefill-long-context-harness-authority-role-tax/`
- Required outputs:
  - `authority.json`
  - `harness_reconciliation.json`
  - `baseline_whole_prefill_by_ctx.json`
  - `single_chunk_vs_whole_prefill.json`
  - `per_role_time_tax_by_ctx.json`
  - `route_coverage_by_role.json`
  - `graphgemm_vs_tensile_integration_by_role.json`
  - `decision.json`
- Required contexts:
  - 512, 1024, 2048, 4096, 8192
- Completion decisions:
  - `PREFILL_LONGCTX_HARNESS_ARTIFACT_CONFIRMED`
  - `PREFILL_LONGCTX_REAL_INTEGRATION_SLOPE_CONFIRMED`
  - `PREFILL_LONGCTX_ROLE_TAX_ATTRIBUTED`
  - `PREFILL_LONGCTX_ATTENTION_OR_KV_BOUND`
  - `PREFILL_LONGCTX_LAYOUT_OR_INTEGRATION_BOUND`
  - `PREFILL_LONGCTX_GEMM_ROLE_COVERAGE_BOUND`
  - `PREFILL_LONGCTX_NO_SEARCH_NEXT`
  - `PREFILL_LONGCTX_INSTRUMENTATION_REQUIRED`
- Boundaries:
  - do not change decode defaults during this run
  - do not flip defaults
  - do not implement new kernels during audit
  - do not restart broad prefill search until search-readiness verdict is explicit
  - do not use single concrete `start_pos=0` chunk data as the whole-prefill headline
  - do not use nosync/raw-dispatch data as authority

### C) Prefill long-context integration hardening (execution scope)

- Source scope:
  - `docs/prefill-long-context-integration-hardening-scope-20260624.md`
  - `docs/prefill-long-context-root-cause-audit-result-20260624.md`
  - `bench/qk-prefill-root-cause-long-context-20260624/`
- Required tools:
  - `extra/qk_prefill_whole_synced.py`
  - `extra/qk_prefill_per_role_time_tax.py`
- Required artifact folder:
  - `bench/qk-prefill-long-context-integration-hardening-20260624/`
- Required outputs:
  - `authority.json`, `whole_prefill_by_ctx_raw.json`, `whole_prefill_chunk_series.json`, `single_chunk_vs_whole_prefill.json`
  - `runtime_overlap_by_ctx.json`, `per_role_time_tax_timeseries_by_ctx.json`, `route_coverage_by_ctx_and_role.json`, `kv_attention_split_timeseries.json`
  - `memory_pressure_watch.json`, `decision.json`
- Required contexts:
  - 512, 1024, 2048, 4096, 8192
- Completion decisions:
  - `PREFILL_LONGCTX_INTEGRATION_HARDENING_HOSTSYNC_BOUND`
  - `PREFILL_LONGCTX_INTEGRATION_HARDENING_DISPATCH_BOUND`
  - `PREFILL_LONGCTX_INTEGRATION_HARDENING_ATTENTION_COPY_BOUND`
  - `PREFILL_LONGCTX_INTEGRATION_HARDENING_NO_GROWTH_CONFIRMED`
- Boundaries:
  - no new prefill defaults/emit-search during this execution
  - no decode changes
  - no hardcoded `start_pos` overrides for 8192; full 16-chunk lattice required
  - whole-lane timing only is authoritative; single-chunk diagnostics remain diagnostic-only

### Progress handoff format for this objective

- For each phase, record:
  - exact command(s), environment, repeats, repeats spread, git hash, and artifact paths
- Decision rule for next step:
  - decode gap with confirmed route/tax decomposition -> either close with runtime-combine/route optimization or explicitly set to `non-search/blocked-by-tax`
  - prefill gap attribution -> if non-search bound, scope non-search work only; if search-bound, start `qk_prefill_emit_search` on bounded candidate set

---

## Current Baseline

Target machine and model:

```text
GPU: RX 7900 XTX / gfx1100
model: Qwen3-8B-Q4_K_M.gguf
repo: /home/ubuntu/tinygrad-arkey
python: .venv/bin/python
device: DEV=AMD
```

Canonical default decode curve:

| ctx | default decode |
|---:|---:|
| ctx≈0 | ~85-86 tok/s |
| ctx512 | ~68 tok/s |
| ctx1024 | ~66 tok/s |
| ctx4096 | ~61 tok/s |

Current default policy:

- `PREFILL_V2`: default off
- q8 FFN: opt-in only
- B4 AMDGCN decode-attention route: **DEFAULT-ON** (2026-06-23) for gfx1100/Qwen3-8B/B=1/T=1; `DECODE_ATTN_AMDGCN_TILE=0` disables
- decode default attention is the owned AMDGCN route (fp16 cache); +12.7/+15.4/+18.7/+22.4% @ctx512/1024/2048/4096 vs gqa

## Read First

Current authority docs:

- `docs/current-project-state-handoff-20260621.md`
- `docs/README.md`
- `docs/decode-attention-route-b-b3-owned-amdgcn-result-20260621.md`
- `docs/decode-attention-route-b-b4-external-graph-node-result-20260621.md`
- `docs/b4-split-kv-combine-tax-result-20260621.md`
- `docs/b4-split-kv-combine-tax-scope-20260621.md`

Core artifacts:

- `bench/qk-decode-attention-route-b-b3/latest.json`
- `bench/qk-decode-attention-route-b-b4/latest.json`
- `bench/qk-decode-attention-route-b-b4-combine-tax/latest.json`
- `bench/qk-decode-eval/candidates.json`

## Route B State

The live frontier is decode attention Route B: an owned AMDGCN/HSACO escape hatch for the llama-style decode-attention
primitive.

| phase | result | meaning |
|---|---|---|
| B1 | `PASS_ORACLE_LOCAL_AB` | vendored llama `flash_attn_tile` runs through tinygrad HCQ and wins on GPU-busy time |
| B2 | `B2_LOCAL_GRAPH_PASS` | bound HCQ queue recovers raw-dispatch overhead; graph-style launch integration works |
| B3 | `B3_LOCAL_PASS` | owned hand-AMDGCN tile for tinygrad native KV layout beats `gqa_coop_vec` locally |
| B4 graph-node | capability pass | external precompiled AMDGCN `.co` enters TinyJit as `Ops.PROGRAM` nodes |
| B4 W==D | `B4_WD_FAIL_INTEGRATION` | whole-decode economics do not clear promotion |
| B4 combine tax | `COMBINE_TAX_DOMINATES` | split-KV combine is the fixable latency-bound floor; Amdahl co-limits |
| split-KV economics audit | `SPLIT_KV_ECONOMICS_AUDIT_READY` | permanent audit layer: split-KV candidates must report tile/combine economics before W==D |

## B3 Summary

B3 produced the first owned, promotable hand-AMDGCN decode-attention tile:

- source: `extra/qk_owned_flash_decode.hip`
- runner: `extra/qk_owned_flash_decode_amdgcn_b3.py`
- candidate: `decode_attention_llama_flash_tile_owned_amdgcn`
- layout: tinygrad native K/V `[Hkv, MAXC, Hd]`, no repack
- comparator: `gqa_coop_vec`
- result @ctx1024:
  - `2.35x` GPU-busy faster
  - `1.70x` matched-sync wall faster
  - near-exact correctness
  - `v_dot2=2`, `56 VGPR`, `8 KB LDS`, `0 spills`

B3 answered “can we own the primitive?” with yes. Its blocker was that raw HCQ `.co` launches were not graph nodes.

## B4 Summary

B4 removed the B3 graph-node blocker.

Implementation:

- `extra/qk_owned_flash_decode_graph_node.py`
- `tinygrad/llm/model.py`
- `extra/qk_b4_decode_eval.py`
- `extra/qk_b4_policy_sweep.py`

Mechanism:

- specialize `extra/qk_owned_flash_decode.hip` into single-kernel ELFs:
  - `owned_flash_tile_gqa`
  - `owned_flash_combine`
- bake `S`, `scale`, and `MAXC`
- pass `start_pos` as the single symbolic scalar var
- inject a fully formed precompiled `Ops.PROGRAM` via `Tensor.custom_kernel` + `Ops.BINARY`
- route through `DECODE_ATTN_AMDGCN_TILE=1`
- context gate through `DECODE_ATTN_AMDGCN_MIN_CTX`
- fallback to `gqa_coop_vec`

Proof:

- standalone eager, TinyJit capture, and TinyJit replay pass
- replay with a different `start_pos` than capture is correct
- in-model route firing is visible in captured graph names:
  - `owned_flash_tile_gqa`
  - `owned_flash_combine`
- greedy tokens match
- default behavior unchanged

Measurement traps fixed during B4:

- `.item()` must be inside the timed region; otherwise timing only captures async dispatch.
- `should_use_flash_decode` can fire at ctx512 through auto-threshold; route firing must be recorded.
- use in-process/interleaved W==D comparisons where practical.
- do not use rocprofv3 for tinygrad HCQ visibility.

## B4 W==D Outcome

Best policy results from `docs/decode-attention-route-b-b4-external-graph-node-result-20260621.md`:

| policy | ctx512 | ctx1024 | ctx4096 | route firing |
|---|---:|---:|---:|---|
| `ctx2048_only` | +0.08% | +0.18% | +5.44% | only ctx4096 fired in measured set |
| `ctx4096_only` | +0.11% | +0.40% | +5.56% | only ctx4096 fired |
| `adaptive` | +0.24% | -0.76% | +5.36% | ctx1024 and ctx4096 fired |

Promotion gate:

```text
no ctx512 / ctx1024 regression
AND (>= +5% @ctx1024 OR >= +7% @ctx4096)
```

No tested policy cleared this gate.

Verdict: `B4_WD_FAIL_INTEGRATION`.

Interpretation: graph-node integration works. The miss is whole-decode economics: attention is a limited share of the
token step, and split-KV combine gives back part of the tile win.

## Combine-Tax Result

The follow-on combine-tax analysis classified the next bottleneck as `COMBINE_TAX_DOMINATES`.

Standalone per-kernel timing:

| ctx | opt S | tile us | combine us | total us | combine % |
|---:|---:|---:|---:|---:|---:|
| 512 | 48 | 16.0 | 12.7 | 28.7 | 44% |
| 1024 | 48 | 23.4 | 12.6 | 36.0 | 35% |
| 2048 | 48 | 36.8 | 12.6 | 49.4 | 26% |
| 4096 | 64 | 56.5 | 16.2 | 72.7 | 22% |

Key findings:

- combine is a flat latency floor by context and scales mainly with `S`
- combine is not HBM-bandwidth-bound: about 64 GB/s, far below peak
- combine under-occupies the GPU: roughly `Hq=32` workgroups with 32 threads
- reducing `S` does not solve it because the tile becomes starved
- halving combine at ctx4096 is projected to move W==D from ~+5.6% to ~+7.4%
- a free/fused combine is projected around ~+9.2% ctx4096

Verdict: the next attention-specific lever is a cheaper combine, not another tile.

## Split-KV Economics Audit (permanent layer, 2026-06-21)

`SPLIT_KV_ECONOMICS_AUDIT_READY`. The B4 combine-tax lesson is now a **durable, reusable audit** so a future
split-KV candidate cannot pass a local A/B without exposing the combine tax.

- tool: `extra/qk_split_kv_economics_audit.py` (default read-only over the measured B4 artifacts; `--live`
  regenerates the attribution; general `--attribution/--wd/--candidate` for any future candidate)
- artifact: `bench/qk-split-kv-economics-audit/latest.json` (`split_kv_economics_audit_v1`, contract-stamped CONFORMS 13/13)
- binding requirement: `split_kv_economics_contract_v1` in `bench/qk-decode-eval/binding_templates.json`
- B4 classifies `COMBINE_TAX_DOMINATES` — combine latency-bound (~64 GB/s, 32 wg << 96 CUs); Amdahl projection
  ctx4096 +5.41% measured → +6.97% half-combine → +8.58% free-combine.

Run:

```sh
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_split_kv_economics_audit.py
```

Every future split-KV decode-attention candidate must report tile/combine split, combine fraction, effective
bandwidth, workgroup count, and the Amdahl projection — and be classified by this audit — **before** W==D
promotion work. See `docs/split-kv-economics-audit-result-20260621.md`.

## Recommended Next Action (updated 2026-06-22)

Route B attention is CLOSED for default-promotion: B5-lite cheaper combine done (hw 2.4x) but W==D SATURATES ~+5.7%
@ctx4096 (combine overlaps in-graph) -> `B5_COMBINE_LOCAL_PASS_WD_FAIL`. The decode time-tax audit
(`docs/decode-time-tax-audit-result-20260622.md`, `NEXT_PRIMITIVE_Q4K_GEMV_SCHEDULER`) shows the FFN Q4_K weight GEMV is
the dominant tax (gate/up 24% + down 14% = ~38%); q8 (+6%) proves it transfers, attention does not.

The FFN-GEMV scheduler diagnostic (`docs/decode-ffn-gemv-scheduler-diagnostic-result-20260622.md`,
`FFN_GEMV_DIAGNOSTIC_BOUNDED_SCHEDULE_SCOPE_READY`, class `GEMV_SCHEDULE_BOUND`) named the gap: tinygrad's gate/up GEMV
is ~51% peak (1 thread/row, serial whole-row K, uncoalesced) vs llama MMVQ ~70% via **128 threads/row + K-block-parallel
+ in-kernel warp-shuffle reduce**. The dot4/extract are already matched; the missing piece is WORK DECOMPOSITION. The
int-dot path is REFUTED in-model (Q4K_VDOT +1.25%, eaten by the q8-activation lifecycle); the lossless lever is an FP
work-decomposition GEMV that pays no lifecycle tax.

```text
DONE 2026-06-22: q4k_gemv_warp IMPLEMENTED + W==D PASS -> Q4K_GEMV_WARP_WD_PASS (docs/decode-ffn-gemv-warp-result-20260622.md).
LOSSLESS FP work-decomposition GEMV (32 threads/row + K-block-parallel + in-kernel warp_reduce_sum/ds_bpermute, one
output). gate/up+down W==D: +9.78%@1024 / +8.71%@4096 / +9.83%@512, greedy BYTE-IDENTICAL (decode 66.7->73.9 @1024,
~67%->~73% of llama). Local A/B 1.31x gate/up / 1.37x down vs the opted default. The FIRST decode primitive to clear
the W==D gate since the attention arc. default_eligible=true (lossless) but DEFAULT-OFF (Q4K_GEMV_WARP /
Q4K_GEMV_WARP_DOWN) pending owner approval.

HARDENED 2026-06-22 -> Q4K_GEMV_WARP_READY_FOR_OWNER_DEFAULT_DECISION (docs/q4k-gemv-warp-promotion-hardening-result-20260622.md):
promoted route reproduced (~+9.6%@1024 / +8.5%@4096, spread ~0.4%), real-generation BYTE-IDENTICAL (0/64),
default_eligible=true / default_on=false, fallback-safe. Same-lever expansions TESTED + banked research-only (do NOT
help W==D): Q6_K down (1.09x local, already coop-served; flag Q6K_GEMV_WARP_DOWN) + attn q/o (1.32x local but
attention-OVERLAPPED, no transfer; flag Q4K_GEMV_WARP_PROJ). Transfer test again discriminates: FFN weight GEMV
transfers, attention-adjacent does not.

NEXT candidates: (1) OWNER: flip Q4K_GEMV_WARP + Q4K_GEMV_WARP_DOWN default-on (lossless, +9.6%@1024, byte-identical,
no regression). (2) generalize the route guards for 14B/32B (kernel is shape-general; bounded follow-on).
```

Non-goals: no q8 default (lossy), no int-dot/MMVQ reopen (null in-model), no coalescing-only (gate/up not coop-routed),
no attention work (closed), no deep backend before the bounded FP variant is W==D-measured.

## Working Tree Note

At this handoff, B4-related work may still be uncommitted. Preserve these unless explicitly asked to revert:

- `tinygrad/llm/model.py`
- `bench/qk-decode-attention-route-b-b3/latest.json`
- `bench/qk-decode-eval/candidates.json`
- `bench/qk-decode-runtime-overhead/result.json`
- `docs/decode-attention-route-b-b4-external-graph-node-scope-20260621.md`
- `docs/decode-attention-route-b-b4-external-graph-node-result-20260621.md`
- `docs/b4-split-kv-combine-tax-scope-20260621.md`
- `docs/b4-split-kv-combine-tax-result-20260621.md`
- `extra/qk_owned_flash_decode_graph_node.py`
- `extra/qk_b4_decode_eval.py`
- `extra/qk_b4_policy_sweep.py`
- `extra/qk_b4_combine_tax.py`
- `extra/qk_split_kv_economics_audit.py`
- `bench/qk-split-kv-economics-audit/latest.json`
- `bench/qk-decode-eval/binding_templates.json`
- `bench/qk-decode-eval/candidates.json`
- `docs/split-kv-economics-audit-scope-20260621.md`
- `docs/split-kv-economics-audit-result-20260621.md`

Run this to inspect:

```sh
cd /home/ubuntu/tinygrad-arkey
git status --short
```

## Useful Commands

Verify B4 graph-node path:

```sh
cd /home/ubuntu/tinygrad-arkey
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_owned_flash_decode_graph_node.py 48
```

Run B4 W==D harness:

```sh
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_b4_decode_eval.py
```

Run combine-tax attribution:

```sh
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_b4_combine_tax.py
```

Use `docs/README.md` for older arcs; do not reintroduce old handoff material here unless it is again current.
### Decode attention pure-search next scope — primitive-complete online-softmax+PV tile

Canonical scope: `docs/decode-attention-primitive-complete-online-softmax-pv-scope.md`. Core issue is now named: A2 solved generated whole-cache lifecycle hygiene, and A3.6/A3.7/A3.9/A3.10 refuted metadata/simple-fusion as the main lever. The missing pure-search primitive is a generated/search-owned split-KV online-softmax+PV tile with whole-cache KV identity, T=1 parallelism, v_dot2/packed dot, cross-lane reduction, register-resident `(m,l,acc[D])`, and TILE+COMBINE lifecycle accounting. First executable task is the search-space manifest `bench/qk-search-spaces/decode_attention_online_softmax_pv_tile_v1.json` plus a checker proving the manifest names the full primitive boundary before more codegen work.
### Decode attention primitive-complete manifest/checker

Implemented the first executable task from `docs/decode-attention-primitive-complete-online-softmax-pv-scope.md`: `bench/qk-search-spaces/decode_attention_online_softmax_pv_tile_v1.json` declares the full online-softmax+PV tile primitive boundary and classifies the current state as `SEARCH_SPACE_INCOMPLETE`; `extra/qk_search_space_manifest_check.py` validates that the manifest names whole-cache identity, T=1 split-KV parallelism, query/GQA lane ownership, packed dot, cross-lane reduction, register-resident `(m,l,acc[D])`, PV accumulation, TILE+COMBINE lifecycle accounting, A3.10 as negative control, and owned tile/combine as oracle/fallback only. Next executable step is P2: build the opt-in structural generated tile skeleton, not another metadata-only fusion.
### Decode attention P2 online-PV tile structural route

P2 complete. `extra/qk_decode_attention_online_pv_tile_gate.py` produced `ONLINE_PV_TILE_STRUCTURAL_ROUTE_CLEAN` with artifact `bench/qk-decode-attention-online-pv-tile/latest.json`. The new generated program `flash_online_pv_tile_whole_cache_32_128` fires with owned tile/combine absent, `E_49152` absent, token sample matching `[315, 24231, 6009, 979, 220, 576]`, stale A3.10 `flash_tile_prob_partial_pv*` absent, and old `flash_prob_32` / `flash_partial_coop_vec_whole_cache*` absent. This is structural only, not a speed promotion. Next executable step is P3 lane ownership/reduction mapping for head/split/GQA/D lanes and `(m,l,acc[D])` state.
### Decode attention P3 online-PV LaneMap attribution

P3 complete. `extra/qk_decode_attention_online_pv_lanemap.py` produced `ONLINE_PV_TILE_P3_LANEMAP_READY` with artifact `bench/qk-decode-attention-online-pv-lanemap/latest.json`. The structural route owns `kvh` and split `s` as global axes, `d` as local `Hd+1=129` lanes, `j` as split-token reduce axis, and `g=4` as GQA register accumulators. Workgroups are `Hkv*S`: 16/32/64/128 at ctx512/1024/2048/4096. Current tile owns PV accumulation and denominator lane, but score, per-split max, global max, denominator, and final combine remain external. Missing primitive-complete pieces are lane-owned online `m/l`, cross-lane/equivalent reduction for `m/l/acc[D]`, and packed-dot score production inside or directly fused with the tile lifecycle. Next executable step is P4: change reduction/dot codegen or classify `SEARCH_BLOCKED_BY_CODEGEN`; do not just add another metadata fusion.
### Decode attention P4 codegen decision

P4 complete. `extra/qk_decode_attention_online_pv_p4_codegen_decision.py` produced `ONLINE_PV_TILE_P4_NEEDS_DATAFLOW_REWRITE_BEFORE_CODEGEN` with artifact `bench/qk-decode-attention-online-pv-p4-codegen-decision/latest.json`. Existing lowerings are present (`qk_fdot2_lowering`, `qk_warp_reduce_lowering`, `qk_lane_partition_reduce`), but P3/P4 show there is no useful in-tile reduction/dot site to bind: score is still external and prior A3.1 no-transfer; per-split `m` is still external and A3.6 no-transfer; `l/den` is only a partial contribution plus external `flash_den`; PV `acc[D]` is in-tile but has no cross-lane combine site; combine-only is already refuted; blind LDS is disallowed. Next executable step is P5 dataflow rewrite: create `flash_online_state_pv_tile_whole_cache_32_128` that moves per-split `m` and online `l` into the tile lifecycle while preserving `Hkv*S`, `Hd+1`, whole-cache identity, no `E_49152`, and token correctness.
### Decode attention P5 online-state+PV tile

P5 complete. `extra/qk_decode_attention_online_state_pv_tile_gate.py` produced `ONLINE_STATE_PV_TILE_STRUCTURAL_ROUTE_CLEAN` with artifact `bench/qk-decode-attention-online-state-pv-tile/latest.json`. New generated route signature is `flash_score_whole_cache_32_128`, `flash_online_state_pv_tile_whole_cache_32_128`, `flash_state_gmax_32_128`, `flash_state_combine_32_128`; owned tile/combine absent, `E_49152` absent, tokens match `[315, 24231, 6009, 979, 220, 576]`. P5 removes external `flash_max_32`, `flash_den_32`, `flash_prob_32`, old partial-PV stages, and P2 `flash_online_pv_tile*`. Tile output width is now `Hd+2`: PV columns, per-split `l`, per-split `m`. This completes the dataflow rewrite P4 required and creates real in-tile online-state sites for P6 lowerings. Initial failed capture (`ONLINE_STATE_PV_TILE_FAIL__CAPTURE`) was a UOp predicate bug (`d == Hd` host bool), fixed with `d.eq(Hd)`.
### Decode attention P6 lowering-bind decision

P6 complete. `extra/qk_decode_attention_online_state_pv_p6_lowering_bind.py` produced `ONLINE_STATE_PV_TILE_P6_NEEDS_TOKEN_SHARDED_REWRITE` with artifact `bench/qk-decode-attention-online-state-pv-p6-lowering-bind/latest.json`. Lowerings exist (`qk_fdot2_lowering`, `qk_warp_reduce_lowering`, `qk_lane_partition_reduce`), but they cannot bind to P5 because the tile still performs the full serial token loop per local `d` lane. Cross-lane `m/l/acc[D]` needs lane-sharded partials; packed-dot needs score production inside/directly fused with the tile. Next executable step is P7 `flash_online_state_pv_tile_xlane_whole_cache_32_128`: keep `Hkv*S`, whole-cache identity, and no `E_49152`; introduce token/dot shard lane ownership; compute partial `m/l/acc[D]`; then bind cross-lane reduction.
### Decode attention P7 token-sharded x-lane tile

P7 attempted `flash_online_state_pv_tile_xlane_whole_cache_32_128`. Verdict: `ONLINE_STATE_PV_XLANE_FAIL__TOKEN_MISMATCH`, artifact `bench/qk-decode-attention-online-state-pv-xlane/latest.json`. The intended generated route fired with owned tile/combine absent, `E_49152` absent, external max/den absent, old prob/partial absent: `flash_score_whole_cache_32_128`, `flash_online_state_pv_tile_xlane_whole_cache_32_128`, `flash_state_gmax_32_128`, `flash_state_combine_32_128`. Token sample diverged: owned `[315, 24231, 6009, 979, 220, 576]` vs x-lane `[315, 119523, 119523, 313, 296, 296]`. An initial empty-shard online-update bug (`-inf - -inf`) was fixed by preserving prior state on invalid shards, but mismatch remains. Next required step is P8 isolated numeric microgate comparing P5 scalar-state tile vs P7 x-lane tile for per-split `m`, `l`, PV, and final combine before rerunning in-model or W==D.
### Decode attention P8 isolated numeric gate

P8 complete. `extra/qk_decode_attention_online_state_pv_p8_numeric.py` produced `ONLINE_STATE_PV_P8_FAIL__NAN`, artifact `bench/qk-decode-attention-online-state-pv-p8-numeric/latest.json`. The microgate compared P5 scalar-state tile vs P7 x-lane tile on deterministic tensors. Both scalar and x-lane active outputs showed NaNs; errors were `m=0.0394`, `l=115589.55`, `pv=112063.36`, `out=1.0282` at `Tc=128,L=64`. During P8, invalid-token guards were added to scalar and x-lane online updates, and the microcase was changed to exact split length to avoid tail invalids; NaNs still appear. Conclusion: do not rerun P7/W==D. Next step is P9 scalar online-state tile standalone numeric proof against NumPy reference, before returning to x-lane.
### Decode attention P9 scalar numeric proof scope

Scoped P9 in `docs/decode-attention-online-state-pv-tile-p9-scalar-numeric-scope.md`. Purpose: prove `flash_online_state_pv_tile_whole_cache_32_128` against NumPy before any more x-lane/W==D work. Required cases: exact split `Tc=128,L=64`, tail split `Tc=130,L=64`, one split `Tc=32,L=64`, multi split `Tc=256,L=64` at target shape `Hq=32,Hkv=8,Hd=128,MAXC=512`. Compare generated score, per-split `m`, `l`, PV, and final output. Failure labels distinguish score/state/output/NaN. If P9 fails, fix scalar online-state recurrence first; if P9 passes, return to x-lane numeric debugging.
### Decode attention P9 scalar online-state numeric proof

P9 complete. `extra/qk_decode_attention_online_state_pv_p9_scalar_numeric.py` produced `ONLINE_STATE_PV_P9_FAIL__NAN`, artifact `bench/qk-decode-attention-online-state-pv-p9-scalar-numeric/latest.json`, but localization matters: score and final output match NumPy tightly for all required cases (`score <= 7.5e-08`, `out <= 3.0e-08`), while direct reads of intermediate state columns `m/l/PV` contain NaNs/unstable values. Conclusion: scalar online-state final output is correct; the failure is state-column observability from the normal route buffer, not scalar recurrence. Do not fix scalar recurrence based on P9. Next step is P10 isolated x-lane final-output numeric gate against NumPy/scalar output, avoiding raw state-column assertions unless a dedicated debug state-dump kernel is added.
### Decode attention P10 x-lane final-output numeric gate

P10 complete. `extra/qk_decode_attention_online_state_pv_p10_xlane_output.py` produced `ONLINE_STATE_PV_P10_FAIL__XLANE_REF`, artifact `bench/qk-decode-attention-online-state-pv-p10-xlane-output/latest.json`. Scalar final output matches NumPy tightly (`<=3.73e-08`) for Tc 32/128/130/256. X-lane output is finite but wrong by about 1.0 (`0.997-1.051`) against both NumPy and scalar. This localizes the blocker to `flash_online_state_pv_tile_xlane_whole_cache_32_128`: x-lane merge math/staged cross-lane usage/lane-store ownership/per-lane token-shard recurrence. Next step is P11 x-lane merge microproof using synthetic per-lane `(m,l,acc[D])` states before changing the full attention tile.
### Decode attention P11 synthetic x-lane merge

P11 complete. `extra/qk_decode_attention_online_state_pv_p11_xlane_merge.py` produced `ONLINE_STATE_PV_P11_FAIL__MERGE`, artifact `bench/qk-decode-attention-online-state-pv-p11-xlane-merge/latest.json`. Synthetic per-lane `(m,l,acc)` merge is finite but wrong: generated `[0.8264, 1.0, 1.0, -2.1273]` vs NumPy `[0.0864, 1.4706, 0.9045, -0.0241]`, max error `2.1032`. This isolates the blocker below the full attention tile: staged x-lane merge primitive is wrong in generated UOp context. Next step is P12 component repair: independently test `warp_reduce_max`, `_warp_reduce_sum_staged`, `lane==0` gated store, then full LSE merge. Do not debug P7 per-token state until P12 passes.
### Decode attention P12 x-lane component reducer

P12 complete. `extra/qk_decode_attention_online_state_pv_p12_xlane_components.py` produced `ONLINE_STATE_PV_P12_FAIL__MAX`, artifact `bench/qk-decode-attention-online-state-pv-p12-xlane-components/latest.json`. Component errors: max `7.5388`, sum `2.7978`, denominator `18.1782`, LSE `1.0958`, no NaNs. This is below the full attention tile: cross-lane reducer/store composition is not reliable for this generated UOp shape. Attempting multiple gated stores from one global axis hit UOp verification failure on `Ops.AFTER`. Next decision P13: either build a safer attention-local cross-lane reduction/store primitive with verified store contract, or park x-lane token sharding as `SEARCH_BLOCKED_BY_CODEGEN`.

### Decode score-broadcast lifecycle audit + resolution (2026-06-26)

Audited `decode_attention_physical_tile_score_broadcast_lifecycle`. Verdict: **non-promotable, do not reopen the route**. Route gate passes (clean, 6 generated kernels, owned absent, tokens `== [315,24231,6009,979,220,576]`) and TinyJit capture/replay passes all 8 phases, but W==D is structurally catastrophic — 82.5/5.8/3.1/0.7 tok/s @ ctx 128/512/1024/4096 vs baseline 82.3/103.2/101.3/94.0 → ~1.0×/18×/33×/**134× slower**, monotonic, GPU-bound (host-sync 1.4%). Cause: non-flash whole-cache scan, 6 unfused kernels, q.k recomputed per PV chunk, plus a full K+V cache copy (`E_98304`). Label: `SEARCH_SPACE_INCOMPLETE` (missing `FusedScorePVLifecycle`), not a true wall and not `SEARCH_BLOCKED_BY_RUNTIME`. Three handoff corrections: (1) route is **not** materialization-clean — `E_98304_32_3` fires only in the score-broadcast arm and the gate's hardcoded `49152` detector is blind to it (`qk_decode_search_gate.py:36`); (2) "graph-batch barrier is the decisive MMU fix" is **unproven/confounded** — every FAIL ran chunks=1, the only PASS ran chunks=4, barrier + persistent scratch changed together, no controlled toggle; (3) the barrier may be a silent no-op — `jit.py:32` reads `JIT_NO_GRAPH_KERNEL_PREFIXES` via tinygrad's `@functools.cache` getenv (`helpers.py:165-166`) while the route mutates raw `os.environ`, so a prefill captured first caches `""` and disables it. Also: two ungated core changes (`schedule/__init__.py:133-139` toposort BIND scan, `codegen/__init__.py:146` unconditional `pm_unbind`) ride along to the default path. Full audit: `docs/decode-score-broadcast-lifecycle-audit.md`. Ordered resolution plan (record refutation → confirm `E_98304` → fix gate H3 → fix barrier H1 → controlled barrier experiment → decide ungated H2 → build `FusedScorePVLifecycle`): `docs/decode-score-broadcast-lifecycle-resolution-plan.md`. Next step is plan step 1 (record the refutation in `candidates.json`) and step 8 (the flash-structured fused primitive) as the real forward lever.

### Decode generated tile codegen blocker (2026-06-26)

Canonical roadmap is now `docs/pure-machine-search-roadmap.md`; `structure/Development/roadmap.md` is only a pointer. Latest generated fused-xlane route remains clean (`FUSED_XLANE_SCORE_PV_ROUTE_CLEAN__ECONOMICS_NEXT`) and ISA-pinned (`ISA_DIFF_PINNED`: owned LDS 8192 B / `global_load_d16=22` / `cross_lane=5`; generated LDS 256 B / `global_load_d16=0` / `cross_lane=20`). Added isolation gate `extra/qk_decode_cache_identity_index_gate.py` with artifact `bench/qk-decode-cache-identity-index/latest.json`. Verdict: `SEARCH_BLOCKED_BY_CODEGEN__DYNAMIC_UPCAST_REG_STORE_AND_PTRCAT_PLACEMENT`. Proven: raw 5D cache indexing works, static UPCAST works, dynamic scalar indexing works, and K-UPCAST-to-LDS works. Blocked: dynamic V reduce + UPCAST accumulator emits invalid C (`make_float4(...) = make_float4(...)`), and direct `PTRCAT` vector-load authoring fails spec. Next scope: env-gated late-codegen coalesced-load lowering that vectorizes cache reads without vectorizing register accumulator stores; do not write another attention layout.
