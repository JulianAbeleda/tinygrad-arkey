# Q6K-2 (narrow) microgate

**Verdict:** Q6K2_PASS_PACKING_AND_MICROGATE

pos16->warp32 packing: lane = block_group(0..1)*16 + pos(0..15): 2 K-parallel block-groups packed into one 32-lane wave; warp_reduce_sum (ds_bpermute) over 32 lanes -> out[row]. No partials buffer, no external r_* sum.

LanePartition extent16 'blocker': MOOT -- the route uses warp_reduce_sum over the FULL 32-lane wave via the 2-group pack, not LanePartition(extent=16)

## ffn_down microgate (blk.0.ffn_down.weight, 256 rows of (4096, 12288))
| comparison | max_abs | tol |
|---|---|---|
| warp vs fp32 ref | 0.000973 | 0.01 |
| warp vs coop+sum | 2.21e-06 | 0.01 |
| coop vs ref | 0.000972 | 0.01 |

## W==D note
q6k_gemv_warp (ffn_down) is correct+byte-identical but model.py:434-436 records it as ~1.09x / no W==D gain for ffn_down ALONE (down already coop-routed ~51% peak). The Q6K-0 firm-removable W==D win is the lm_head coop reduce (r_32_4_1187), which this ffn_down-only warp route does NOT cover -> Q6K-3 must extend the warp route to lm_head (151936x4096), the folded-in target.
