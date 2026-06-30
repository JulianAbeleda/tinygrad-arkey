# Q6K-1 Q6_K direct-route design

**Verdict:** AMD_ISA_Q6K_DIRECT_DESIGN_PASS_READY

**Selected:** B_q6k_lanemap_g3_like (lm_head folded)

PASS_READY. Selected B (q6k_lanemap_g3_like): reuse the existing Q6_K coop dequant body but replace the per-lane partials-write + external .sum(axis=1) with the existing quant-agnostic in-warp lane_partition_reduce_sum over the 16 pos-lanes, storing out[row] directly -- the exact shape of the proven Q4_K G3 win, lm_head folded via the out>=100000 guard. All capability primitives exist; only a NEW default-off Q6K_DIRECT_ROUTE flag is added. Clean-mechanism TIER_B floor ~3.5% (reduce + partials-buffer-traffic elimination), TIER_A target ~7.0% if it also closes the 503->650 GB/s q6k_gemv bw gap.

## Q6_K quant semantics (cited extra/q6_k_gemv_primitive.py:36-54)

- block 256 elems / 210 B / 105 halfwords; 16 groups x 16 pos
- q = (ql[4b] | qh[2b]<<4) - 32; weight = d(fp16 @hw104) * q * scale(int8 @byte192+grp); acc/out float32; symmetric (no zero/min)

## Current vs target route
```
current: q6k_coop_partial -> partials[rows,16] -> .sum(axis=1) r_* reduce -> out
target:  q6k packed load -> dequant -> dot -> IN-WARP lane_partition_reduce_sum -> out  (no partials buffer, no separate reduce)
```

## Primitive checklist
| primitive | exists | cite |
|---|---|---|
| packed_q6k_load | YES | q6_k_gemv_primitive.py:_q6k_byte (29-31) halfs[base+idx//2] |
| low_high_bit_extraction | YES | _q6k_weight (36-44): ql mask 0xf, qh mask 0x3<<4 |
| scale_dequant | YES | _q6k_weight (45-48): d*q*scale, q=(ql|qh<<4)-32 |
| accumulation_dtype | YES | _q6k_block_dot (51): float32 contrib |
| lane_shuffle_lane_map | YES | qk_lane_partition_reduce.py LanePartition + amd_warp_reduce.py WARP=32 (used by Q4_K G3) |
| in_register_reduction | YES | qk_lane_partition_reduce.py:57 lane_partition_reduce_sum |
| final_output_store | YES | Q4_K G3 qk_gemv_g3_codegen_lowering.py:41 out[row].store(total) |
| shape_guard | YES | model.py:448-450 use_coop role guards (out>=100000 lm_head, 4096x12288 down) |
| route_attribution_label | YES | KernelInfo(name=...) pattern; new label q6k_direct_* |
| rollback_flag | YES | NEW default-off flag Q6K_DIRECT_ROUTE (trivial getenv guard) -- a planned addition, not a capability gap |

## Expected gain
TIER_B floor ~3.5% (reduce+partial-traffic elimination, clean mechanism) -> TIER_A target ~7.0% (if 503->650 bw gap closes).

## Q6K-2 plan (headline)
- files: extra/q6_k_gemv_primitive.py (+q6k_direct_lanemap_kernel), tinygrad/llm/model.py (Q6K_DIRECT_ROUTE branch, default-off)
- first gate: single-role ffn_down microgate (byte-identical vs coop_partial+sum, route-bound)
- rollback: Q6K_DIRECT_ROUTE=0

## Merge
Stay on q6k-direct-route; merge to local master only after Q6K-2 correctness + Q6K-3 speed pass; do NOT merge stale psp-top-table.

## Caveats
- TIER_A vs TIER_B hinges on whether the lane-map also improves GEMV coalescing (503->650); reduce+partial-traffic elimination alone is the TIER_B floor
- ambiguous prod==4096 reduces NOT credited (RMSNorm vs q6k gate_up)
- main perf risk = mapping pos(16) to a warp-shuffleable lane (half-warp LanePartition extent=16); G3 proves the pattern for Q4_K
- design-only: no kernel built, no W==D measured -- Q6K-2/Q6K-3 gate the real numbers