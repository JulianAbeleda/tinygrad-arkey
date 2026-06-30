# TG2 Candidate Topology Author -- verdict: **TG2_PASS_G3_REDISCOVERED_BY_GRAMMAR**

The machine authors lane-map topologies for Q4_K decode GEMV on `qwen3_8b_q4_k_m_gfx1100` from a bounded grammar, then separately checks G3 rediscovery by TopologySpec equality (TG1 IR).

## Result

- **Bounded candidate count:** 20 TopologySpecs (explosion limit 64).
- **Grammar candidates matching G3's TopologySpec:** 1 (must be exactly 1).
- **Matching candidate:** `{'lane_ownership_axis': 'packed_word', 'lane_grouping': '1row_per_warp', 'block_groups': 4, 'words_per_group': 8, 'reduction_pattern': 'cross_lane_wave_reduce', 'reduction_grammar_value': 'ds_bpermute_tree'}`
- **Derived packed-word index == IR's G3 index constant:** True
- **No route_id branch in generation source:** True
- **(4,8) is one of many enumerated factor pairs:** True (pairs: [[1, 16], [1, 32], [2, 8], [2, 16], [4, 4], [4, 8], [8, 2], [8, 4], [16, 1], [16, 2]])

## How generation stays bounded (pruning)

- target gate: profile gpu=wave32 -> lane_extent=32; wave64/subgroup_simdgroup pruned (target-mismatch).
- refuted-axis exclusions: lane_grouping=half_warp, load_pattern=strided_packed_word, load_pattern=scalar_fallback removed (see refuted_axis_exclusions).
- load_pattern/dequant_placement collapse to a single legal value each (do not multiply count).
- lane_ownership_axis: only packed_word emits TopologySpecs; output_row is a distinct-IR family, block_group/token/split are shape-pruned.
- factorization: only integer factor pairs of per_row_lanes, pruned by quant validity (words_per_group | quant_words_per_block) and shape validity (block_groups | k_blocks all roles).

## Refuted-axis exclusions applied

| grammar value | axis | disposition |
|---|---|---|
| `lane_grouping=half_warp` | q6k_direct_half_warp_route / coop_halfwarp_direct | refuted: W==D -4.77..-6.06% |
| `load_pattern=strided_packed_word` | q4k_offline_layout_reshuffle / lanemap_layout_reshuffle | deprioritized: G3 matches owned, no layout gap to recover |
| `load_pattern=scalar_fallback` | n1b_scalar_address_path | refuted/dead |
| `(not a GEMV topology axis)` | scheduler_only_attention_tuning | small/no movement (attention-only, N/A to decode GEMV) |
| `(not a GEMV topology axis)` | occupancy_lds_only_attention_tuning | refuted: no W==D movement (attention-only, N/A to decode GEMV) |
| `(not a GEMV topology axis)` | attention_combine_fused_lifecycle | exhausted/low-leverage (attention-only, N/A to decode GEMV) |
| `(not a GEMV topology axis)` | native_attention_as_default | correct_not_fast (attention-only, N/A to decode GEMV) |

## lane_ownership_axis dispositions

- **packed_word**: ACTIVE: 20 TopologySpec candidates (factorization sub-grammar).
- **output_row**: DISTINCT FAMILY (owned_reference): lane-per-output-row, serial-K, no cross-lane combine. Its lane map is NOT a (block_groups x words_per_group) packed-word factorization -> a different IR shape, out of TG1-v1 TopologySpec scope. Not pruned by validity, but not expressible as a packed_word TopologySpec; never matches G3.
- **block_group**: SHAPE-PRUNED: one block-group/lane needs lane_extent=32 | k_blocks; k_blocks={'ffn_gate_up': 16, 'ffn_down': 48, 'attn_qo': 16} -> 32 divides none.
- **token**: SHAPE-PRUNED: decode is single-token (M=1) -> no token axis to own.
- **split**: SHAPE-PRUNED: hybrid/split ownership adds wave divergence with no reuse at decode M=1 (single output row per wave); also a distinct IR shape.

## Candidates (TopologySpec set)

| # | ownership | grouping | block_groups | words_per_group | group_pairs | reduction | == G3 |
|---:|---|---|---:|---:|---:|---|:--:|
| 0 | packed_word | 1row_per_warp | 1 | 32 | 1 | cross_lane_wave_reduce |  |
| 1 | packed_word | 1row_per_warp | 1 | 32 | 1 | partials_plus_reduce |  |
| 2 | packed_word | 1row_per_warp | 2 | 16 | 2 | cross_lane_wave_reduce |  |
| 3 | packed_word | 1row_per_warp | 2 | 16 | 2 | partials_plus_reduce |  |
| 4 | packed_word | 1row_per_warp | 4 | 8 | 4 | cross_lane_wave_reduce | YES |
| 5 | packed_word | 1row_per_warp | 4 | 8 | 4 | partials_plus_reduce |  |
| 6 | packed_word | 1row_per_warp | 8 | 4 | 8 | cross_lane_wave_reduce |  |
| 7 | packed_word | 1row_per_warp | 8 | 4 | 8 | partials_plus_reduce |  |
| 8 | packed_word | 1row_per_warp | 16 | 2 | 16 | cross_lane_wave_reduce |  |
| 9 | packed_word | 1row_per_warp | 16 | 2 | 16 | partials_plus_reduce |  |
| 10 | packed_word | 2rows_per_warp | 1 | 16 | 2 | cross_lane_wave_reduce |  |
| 11 | packed_word | 2rows_per_warp | 1 | 16 | 2 | partials_plus_reduce |  |
| 12 | packed_word | 2rows_per_warp | 2 | 8 | 4 | cross_lane_wave_reduce |  |
| 13 | packed_word | 2rows_per_warp | 2 | 8 | 4 | partials_plus_reduce |  |
| 14 | packed_word | 2rows_per_warp | 4 | 4 | 8 | cross_lane_wave_reduce |  |
| 15 | packed_word | 2rows_per_warp | 4 | 4 | 8 | partials_plus_reduce |  |
| 16 | packed_word | 2rows_per_warp | 8 | 2 | 16 | cross_lane_wave_reduce |  |
| 17 | packed_word | 2rows_per_warp | 8 | 2 | 16 | partials_plus_reduce |  |
| 18 | packed_word | 2rows_per_warp | 16 | 1 | 32 | cross_lane_wave_reduce |  |
| 19 | packed_word | 2rows_per_warp | 16 | 1 | 32 | partials_plus_reduce |  |

## Proof generation did not cheat

- Generation reads ONLY: profile facts (quant/shape/target) x topology_grammar_v1 x refuted_axis_exclusions.
- `(block_groups, words_per_group)` come from `factor_pairs(per_row_lanes)` (divisor enumeration), not a literal `(4,8)`.
- `axis_roles` come from `derive_packed_word_axis_roles()` (a rule over lane_ownership_axis), not a hardcoded dict.
- `lane_ownership_index` comes from `derive_packed_word_index()` (symbolic, format-agnostic), assigned to every packed_word candidate.
- `g3_template(...)` is read ONLY inside `verify_g3_rediscovered()` -- the post-hoc CHECK, never in generation.
