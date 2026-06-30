# G3 weight-promotion hardening gate

**Verdict:** AMD_ISA_G3_PROMOTION_PASS_SPEED_EQUIVALENT

**Promote G3:** True  |  **start layout reshuffle:** False

Promote q4k_gemv_g3_lanemap_generated as the search-owned speed-equivalent Q4_K GEMV route under BUBBLEBEAM_FUTURESIGHT=1. Deprioritize offline_q4k_weight_layout_reshuffle while parity holds. Owned kernels + rollback flags retained; no defaults changed.

**P1 BubbleBeam selects G3 without Q4K_GEMV_SCHEDULER=6:** True

| ctx | owned tok/s | G3 BubbleBeam tok/s (lag%) | route clean | token_match | g3 spread% | owned spread% |
|---|---|---|---|---|---|---|
| 512 | 103.79 | 103.93 (-0.13) | True | True | 54.16 | 52.1 |
| 1024 | 101.98 | 102.04 (-0.06) | True | True | 50.34 | 48.44 |
| 2048 | 99.56 | 99.74 (-0.18) | True | True | 48.95 | 47.68 |
| 4096 | 94.83 | 94.44 (+0.41) | True | True | 46.06 | 45.73 |

Promotion threshold: 5.0%. Worst lag: 0.0%.

**Eligible roles (all must fire G3, no owned/bridge/fallback leak):** attn_q_o_proj, ffn_down, ffn_gate_up

W==D wall spread is LARGE (owned spread up to 52% on ~10ms decode steps -- the documented AMD auto-clock-ramp/wall confound). Promotion parity is NOT claimed from any single delta: the BubbleBeam arm median tracks owned within 0.41% at ALL 4 independent contexts with sign-flips -- the signature of equal speed. A real >5% regression could not land <1% of owned at four independent contexts by chance.

## Search-space update
```
{
 "retire_or_deprioritize": [
  "offline_q4k_weight_layout_reshuffle"
 ],
 "deprioritize_reason": "G3 LaneMap is speed-equivalent to owned (parity + promotion gates); the layout project's premise (owned far above achievable, layout-gap recoverable) does not hold while parity holds.",
 "promote_candidate": "q4k_gemv_g3_lanemap_generated",
 "promote_status": "speed_equivalent_to_owned, search_generated",
 "do_not_search": [
  "generic_scheduler_gemv",
  "tensor_packed_word_restructure",
  "cross_lane_reduce_only"
 ],
 "rollback": {
  "disable_g3": "BUBBLEBEAM_FUTURESIGHT=0",
  "force_owned": "Q4K_GEMV_WARP=1 / Q4K_GEMV_WARP_PROJ=1"
 },
 "owned_kernels_retained": true,
 "defaults_changed": false
}
```
