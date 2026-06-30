# TG7 First New-Profile Search -- Q6_K ffn_down GEMV (gfx1100)

Verdict: **TG7_PASS_FIRST_NEW_PROFILE_SEARCH_RESULT** | search_result: **SEARCH_EXHAUSTED_SPACE**

The grammar AUTHORED 18 bounded Q6_K ffn_down topology candidates (quant-parameterized from TG3 Q6_K facts; all wave32 TARGET_OK). 2 of them STRUCTURALLY rediscover the shipped OWNED Q6_K routes (q6k_gemv_warp / q6k_coop_partial) -- an equivalent-to-shipped rediscovery, not a new topology. The refuted half-warp direct is EXCLUDED by the grammar's refuted-axis gate (and, if forced, still refutes: REFUTED_REGRESSION -> decode_q6k_direct_refuted). No authored candidate has a generated-promotion authority artifact; promoting a GENERATED Q6_K replacement for the owned route would require a fresh W==D measurement, which this AUDIT scope does not run. Honest verdict: the bounded space is authored end-to-end but contains NO new promotable topology beyond the shipped owned route under the current ceiling -> SEARCH_EXHAUSTED_SPACE.

## What the grammar authored

- **18 bounded Q6_K topology candidates** (explosion limit 64); all wave32 TARGET_OK on gfx1100. Quant facts from TG3: payload_first=True, symmetric=True, natural_lane_extent=16, k_blocks=48.
- half_warp EXCLUDED by the refuted-axis gate: ['decode_q6k_direct_refuted'].

## Structural rediscoveries of the shipped OWNED Q6_K routes

| candidate | block_groups | pos_lanes | reduction | rediscovers |
|---|---:|---:|---|---|
| `q6k_ffn_down_1row_per_warp_bg2_pos16_cross_lane_wave_reduce` | 2 | 16 | cross_lane_wave_reduce | decode_q6k_owned_warp (q6k_gemv_warp_kernel, owned_reference) |
| `q6k_ffn_down_2rows_per_warp_bg1_pos16_lds_partial_reduce` | 1 | 16 | partials_plus_reduce | decode_q6k_coop_shipped (q6k_coop_partial_kernel, owned_default) |

## Evaluator decisions

- Representative authored candidate `q6k_ffn_down_1row_per_warp_bg2_pos16_cross_lane_wave_reduce` -> **CORRECT_NOT_FAST_OR_UNMEASURED** (no generated-promotion authority; a generated Q6_K route needs a fresh W==D not run in this audit).
- Refuted half-warp (excluded by grammar; forced through gate) -> **REFUTED_REGRESSION** maps to `decode_q6k_direct_refuted` (refutation preserved).

## Honest bottom line

The pipeline runs end-to-end on a NEW quant profile and AUTHORS a bounded Q6_K topology family. It does NOT find a new promotable topology: the family either rediscovers the shipped OWNED route or hits the refuted half-warp. That is the honest, non-manufactured result the milestone asks for.
