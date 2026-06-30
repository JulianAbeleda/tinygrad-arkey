# Decode-attention ceiling audit

**Verdict:** AMD_ISA_ATTENTION_CEILING_PASS_MOVE_TO_NON_ATTENTION

**Decision:** move search to `non_attention_ffn_weight_path`

Decode is WEIGHT-MEMORY-bound: the 190.9 tok/s weight-read ceiling dominates; the attention KV-read floor is <1% of it (0.667% @ctx4096). Matching owned's tile yields only +10.5% @ctx512 (borderline) and +2.9% @ctx4096 (<5%) by Amdahl on the MEASURED tile wall-share (~10%@512/~0@4096), and would now require an owned-LEVEL algorithmic rewrite since every tile RESOURCE lever is exhausted/refuted. The FFN/weight GEMVs dominate the wall for BOTH routes and sit at only ~54.2% (owned)/~37.1% (native) of the weight floor (~2x headroom). => move search to the non-attention (FFN/weight) decode path; attention-tile work is low-leverage and diminishing.

## W==D (measured)
| ctx | native | owned | native % of owned | native % of weight-floor |
|---|---|---|---|---|
| 512 | 70.74 | 103.53 | 68.3% | 37.1% |
| 4096 | 56.7 | 94.41 | 60.1% | 29.7% |

## Math floor (peak bw)
| metric | value |
|---|---|
| weight-read decode ceiling | 190.9 tok/s (real ~80%: 152.8) |
| attn KV-floor @ctx4096 | 0.667% of weight floor (negligible) |

## Loss stack
| layer | value |
|---|---|
| tile wall-share (measured) | ctx512 ~10% / ctx4096 ~3% |
| max gain match-owned-tile | ctx512 +10.5% / ctx4096 +2.9% |
| max gain hit-attn-floor | ctx512 +11.4% / ctx4096 +3.1% |
| FFN/weight (shared) headroom to floor | owned ~54.2% of floor -> ~1.8x |

## Decision table
| question | answer |
|---|---|
| match owned tile >=10%? | ctx512 +10.5% (borderline) ; ctx4096 +2.9% (no) |
| attention worth continuing? | NO -- diminishing, resource-exhausted, <1% of weight-bound floor |
| where is the wall + headroom? | FFN/weight path (shared, ~2x to floor) |

## Caveats
- peak HBM bw => optimistic ceilings (real ~80%)
- tile wall-share from MEASURED N3F Amdahl, not eager GPU-compute (which overstates via no-overlap)
- conservative math floor (lower-bound work); owned/native are far above the attention floor but attention is overlapped by the weight-bound FFN
- ctx512 match-owned ~+11% is borderline-above 10% but needs an owned-LEVEL algo rewrite (resource levers exhausted) -> lower leverage than the FFN/weight path