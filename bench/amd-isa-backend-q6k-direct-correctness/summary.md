# Q6K-2 half-warp 2-row partition microgate

**Verdict:** Q6K2_PASS_HALFWARP_PARTITION

## Mapping
lanes 0..15 = row A pos 0..15; lanes 16..31 = row B pos 0..15. Half-warp reduce = warp_reduce_sum(width=16) over the FULL 32-lane lidx0 (xor {8,4,2,1} stays within each 16-lane half). Store out[rowA]/out[rowB] independent; no partials, no external r_* reduce.

Why not LanePartition: it reduces the WHOLE wave to one value (words_per_group is the address split, not independent partitions).

## Microgate (blk.0.ffn_down.weight, 256 rows of (4096, 12288), both halves)
| comparison | max_abs | tol |
|---|---|---|
| row A (even) vs coop+sum | 4.77e-07 | 0.01 |
| row B (odd) vs coop+sum | 4.77e-07 | 0.01 |
| row A vs fp32 ref | 0.000972 | 0.01 |
| row B vs fp32 ref | 0.000896 | 0.01 |
| coop vs ref | 0.000972 | 0.01 |

route label: q6k_halfwarp_partition_256_12288; no external r_* reduce; new kernel unwired (model route byte-identical).
