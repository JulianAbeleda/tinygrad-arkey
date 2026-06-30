# TG0 — G3 provenance audit

**Verdict:** TG0_PASS_G3_PROVENANCE_PINNED

G3 is GENERATED emission of a TEMPLATED spec, but the lane-map TOPOLOGY is STILL HUMAN. The select->author bridge (TG1/TG2) only needs to make that topology machine-discoverable; everything else is already mechanical or data.

## Provenance buckets
| bucket | count | meaning |
|---|---|---|
| generated | 1 | tinygrad codegen (automatic) |
| templated | 5 | emitted from the LaneMap spec (R5 lossless) |
| quant_data | 3 | fixed by the Q4_K format (TG3 makes data-driven) |
| gpu_data | 1 | fixed by the target wave (TG5 makes a target feature) |
| **still_human** | 5 | **the lane-map topology — the bridge target** |

## Still-human surface (what TG1/TG2 must make machine-authorable)
- `block_groups_4`
- `words_per_group_8`
- `axis_role_assignment`
- `cross_lane_reduction`
- `packed_word_lane_index`

## Topology DOF a grammar must span
- **k_to_lane_decomposition**: factor k_blocks into (block_groups x words_per_group x local_block x group_pair) s.t. block_groups*words_per_group == lane_extent
- **axis_roles**: assign each factor to GLOBAL | LOCAL | REDUCE
- **lane_ownership_index**: the coalesced packed-word index per lane (derivable from the decomposition + quant packing)
- **reduction_pattern**: cross-lane (ds_bpermute wave reduce) vs partials+separate-reduce

## Bridge milestone
TG2 succeeds if a topology grammar over topology_dof, given only {quant facts (TG3) + shape + GPU features (TG5)}, REGENERATES the G2 LaneMap that G3 uses (lossless per R5) WITHOUT the block_groups=4/words_per_group=8/axis-roles being hardcoded. Not 'beat G3' -- 'rediscover G3'.

## Caveat
audit-only / static. 'still_human' = design choices, not correctness claims. block_groups/words_per_group are constrained (product == lane_extent) but the specific 4x8 split + the axis-role assignment were a person's decision, which is exactly the surface TG2 must search.