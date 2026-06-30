# Qwen3 14B/32B True-Generation Decode — Q1432 result (phases 0,3,4,5)

Date: 2026-06-30

Status: route coverage fixed and validated (default-off); track NOT closed; remaining gap classified as
`SEARCH_SPACE_INCOMPLETE`. Scope: `docs/qwen-14b-32b-true-generation-kernel-authoring-scope-20260630.md`.
Hardware: RX 7900 XTX (gfx1100). Models: Qwen3-14B/32B Q4_K_M.

## Q1432-0 — route-miss proven

`extra/qk_large_model_decode_route_gap_audit.py` censuses every Q4_K decode linear and classifies its route by
replaying the model.py guards:

| model | Q4_K decode linears | route | verdict |
|---|---|---|---|
| 14B | 221 | **100% fallback_lazy_dequant** | `Q1432_0_PASS_GAP_AND_ROUTE_MISS_PINNED` |
| 32B | 353 | **100% fallback** | `Q1432_0_PASS` |
| 8B (control) | 199 | 180 G3 / 19 fallback | `ABORTED_NO_ROUTE_MISS` (route already fires; classifier valid) |

14B/32B run the slow lazy-dequant path for all Q4_K decode because the G3 guard (`g3_bubblebeam_shape`) and the
route selector (`should_route_q4k_lane_partition`) both hardcode the 8B dims (in/out ∈ {4096, 12288}).

## Q1432-3 — the generated G3 kernel GENERALIZES (correct)

`q4k_g3_lanemap_gemv_kernel(rows, k)` is shape-parameterized. Run directly on the large-model shapes vs the
dequant reference (random input), it is correct on **every** Q4_K decode shape:

| shape (in→out) | role | rel_rmse | verdict |
|---|---|---|---|
| 5120→5120 | attn_q/o | 3.7e-04 | CORRECT |
| 5120→17408 | 14B ffn_gate/up | 3.5e-04 | CORRECT |
| 17408→5120 | 14B ffn_down | 3.6e-04 | CORRECT |
| 5120→8192 | 32B attn_q | 3.7e-04 | CORRECT |
| 5120→25600 | 32B ffn_gate/up | 0.0 | CORRECT |
| 25600→5120 | 32B ffn_down | 3.5e-04 | CORRECT |

(The lm_head 5120→151936 and attn_v 5120→1024 are Q6_K, not Q4_K — separate route, not in scope here.)

So this is a **route-binding** problem, not a codegen-capability gap: the generated topology already covers the
larger structural shape class; only the guards exclude it.

## Q1432-4 — default-off structural binding (token-identical)

`DECODE_Q4K_G3_ANYSHAPE=1` (default 0, rollback) binds G3 by structural eligibility ((in//256)%4==0, out%32==0)
instead of hardcoded dims. Inert by default — 8B and all models unchanged. 14B greedy generation is **token-identical**
(md5 match, flag on vs off). Not a model-name/dim hardcode.

## Q1432-5 — speed: TIER_A movement, track NOT closed

| 14B decode | ctx128 | ctx512 |
|---|---|---|
| shipped (fallback) | 25.5 | 25.0 tok/s |
| G3-anyshape | **27.8** | **27.1 tok/s** |
| delta | +9.0% | +8.4% |
| llama.cpp (matched) | ~66 | ~65 tok/s |

`Q1432_5_PASS_TIER_A_WD_MOVEMENT` (+8-9%, host-sync 0%, token-identical) — promotable for the profile class. But
the track is **not closed**: 14B is still ~42% of llama. At decode all ~9GB of weights are read once/token, so
llama's ~65 tok/s ≈ 585 GB/s (near HBM peak) and tinygrad's ~27 ≈ 243 GB/s (**~42% of bandwidth**) — every role is
bandwidth-underutilized, not just the ones G3 now covers.

## Outcome classification: `SEARCH_SPACE_INCOMPLETE`

The generated G3 route **fires and is correct** for the larger shapes, but it reuses the 8B-tuned LaneMap
parameters (words_per_group, lane_extent, K-decomposition) which are not optimal for the larger K (5120, 17408,
25600). The remaining ~58% bandwidth gap is a **topology-tuning** gap, not a route-coverage gap. Per the scope
taxonomy this is `SEARCH_SPACE_INCOMPLETE`, not `REFUTED`: the route family stays open; the missing axis is
shape-tuned K-decomposition.

## Frontier / next

- **Open frontier:** Q1432-2 should author K-decomposition-tuned topology candidates for K ∈ {5120, 17408, 25600}
  (the grammar currently emits the 8B-tuned lanemap). Reopen condition: grammar gains a shape-tuned
  words_per_group / row-grouping axis and the candidate evaluator measures it on these profiles.
- **Promotion:** the +8-9% structural binding is correct and promotable; it is shipped default-OFF behind
  `DECODE_Q4K_G3_ANYSHAPE` pending the profile-scoped route-policy mechanism (Q1432-4 preferred form) so the
  default flip is profile/shape/target-driven rather than a global flag.
- **Q6_K note:** lm_head and attn_v are Q6_K; their decode cost is a separate bucket (not addressed here).
- **Prefill (Q1432-7):** not run; 14B/32B tuned `PREFILL_V2` likely will not fit 24GB — must be measured, not forced.
