# Q1432-2+: Shape-Tuned Topology Search For Qwen3 14B/32B Decode

Date: 2026-06-30

Status: execution scope for Claude. This is the continuation after `Q1432_5_PASS_TIER_A_WD_MOVEMENT` and
`SEARCH_SPACE_INCOMPLETE`. No hand-written 14B/32B kernel is allowed.

## Purpose

Q1432 already proved the first layer:

- 14B/32B were missing the generated Q4_K route entirely.
- `q4k_g3_lanemap_gemv_kernel(rows, k)` is correct for all large Q4_K shapes.
- `DECODE_Q4K_G3_ANYSHAPE=1` route-binds the generated route and gives a real +8-9% on 14B.
- But the route still reaches only about 42% of llama.cpp because it reuses the 8B-tuned LaneMap topology.

This scope attacks the remaining gap by using the true-generation foundation:

```text
profile facts
  -> topology grammar
  -> LaneMapTemplate / generated kernel
  -> correctness microgate
  -> role-local speed gate
  -> in-model W==D
  -> promote / refute / frontier ledger
```

The goal is **not** to write a better 14B/32B kernel. The goal is to make the machine author and evaluate shape-tuned
Q4_K topology candidates for the large dense shapes.

## Source Citations

Read these before implementation:

| claim | citation |
|---|---|
| Q1432 result and gap classification | `docs/qwen-14b-32b-truegen-q1432-result-20260630.md` |
| Full Q1432 scope and outcome taxonomy | `docs/qwen-14b-32b-true-generation-kernel-authoring-scope-20260630.md` |
| Current topology grammar | `bench/qk-search-spaces/topology_grammar_v1.json` |
| Current topology author | `extra/qk_topology_candidate_author.py` |
| LaneMap IR | `extra/qk_lanemap_template.py` |
| Current G2 lane-map implementation | `extra/qk_gemv_g2_lanemap.py` |
| Current G3 generated emitter | `extra/qk_gemv_g3_codegen_lowering.py` |
| Candidate evaluator | `extra/qk_candidate_evaluator.py` |
| Template candidate gate | `extra/qk_template_candidate_gate.py` |
| Q4_K primitive/dequant bodies | `extra/q4_k_gemv_primitive.py` |
| Large-model route miss audit | `extra/qk_large_model_decode_route_gap_audit.py`, `bench/qwen-14b-32b-truegen/q1432_0_baseline/` |
| Quant facts | `bench/qk-search-spaces/quant_semantics.json`, `extra/qk_quant_semantics.py` |
| Target facts | `bench/qk-search-spaces/targets/amd_gfx1100.json` |

## Current Diagnosis

The remaining issue is probably missing **search axes**, with a possible downstream **codegen capability** wall.

Evidence:

| fact | implication |
|---|---|
| G3-anyshape is correct on `5120→17408`, `17408→5120`, `5120→25600`, `25600→5120` | core Q4_K dequant/load/reduce/store primitives exist |
| Route binding gives only +8-9% | route coverage was necessary but not sufficient |
| 14B still ~42% of llama.cpp | topology is not using the large-K shapes well |
| `Q4KGateUpLaneMap.validate()` currently requires `words_per_group == 8` | the implementation cannot actually test much of the grammar yet |
| `qk_topology_candidate_author.py` enumerates factor pairs but the current emitter collapses back to the G3 family | the grammar is broader than the lowering path |

So the scope must separate:

```text
missing knob:
  grammar/emitter can be extended to express a larger shape-tuned candidate

missing primitive:
  the best candidate requires a lowering/codegen feature the emitter cannot yet generate
```

Do not collapse these into "refuted." A candidate can fail because the search surface is still too small.

## Primary Target Shapes

From Q1432:

| shape | role | quant |
|---|---|---|
| `5120 -> 5120` | attn q/o | Q4_K |
| `5120 -> 17408` | 14B ffn gate/up | Q4_K |
| `17408 -> 5120` | 14B ffn down | Q4_K |
| `5120 -> 8192` | 32B attn q | Q4_K |
| `5120 -> 25600` | 32B ffn gate/up | Q4_K |
| `25600 -> 5120` | 32B ffn down | Q4_K |

Out of scope for this phase:

- Q6_K lm_head / attn_v.
- attention kernels.
- prefill.
- qwen3.5 hybrid/SSM performance.

## Non-Negotiables

- No hand-written 14B/32B route.
- No model-name or hardcoded-shape if-chain as the search mechanism.
- No route promotion from a microgate alone.
- No default flip until W==D, memory-fit, token/logit, and route-bound gates pass.
- If a knob is missing, add it to the grammar/template/emitter and replay.
- If a primitive is missing, scope that primitive as a codegen capability and stop. Do not hide it behind a custom kernel.
- Keep `DECODE_Q4K_G3_ANYSHAPE` as rollback/diagnostic until a profile-driven policy replaces it.

## Phase KT0: Knob-Reachability Audit

Goal: prove exactly which grammar axes are currently real versus decorative.

Build:

```text
extra/qk_large_shape_knob_reachability_audit.py
```

Audit:

| axis | current expected status |
|---|---|
| `block_groups` | partially represented in `TopologySpec`; lowering must prove it affects emitted code |
| `words_per_group` | currently blocked by `Q4KGateUpLaneMap.validate()` requiring 8 |
| `row_grouping` / rows per warp | grammar value exists; current packed-word emitter likely does not implement multi-row ownership |
| `reduction_pattern` | cross-lane works; partials-plus-sum must be measured or excluded |
| `lane_ownership_axis=output_row` | documented as distinct family, not expressible in current Q4_K IR |
| vector load / multiword load | primitive exists in `q4_k_gemv_primitive.py`; current G3 emitter does not use it |
| q8_1 activation quant / int-dot | primitive exists; likely not in scope unless activation bandwidth becomes dominant |

Outputs:

```text
bench/qwen-14b-32b-truegen/kt0_knob_reachability/latest.json
bench/qwen-14b-32b-truegen/kt0_knob_reachability/summary.md
bench/qwen-14b-32b-truegen/kt0_knob_reachability/reachability_rows.json
```

Pass verdict:

```text
KT0_PASS_REACHABILITY_PINNED
```

The artifact must label each axis:

```text
REAL_AXIS
GRAMMAR_ONLY
EMITTER_BLOCKED
PRIMITIVE_BLOCKED
REFUTED_AXIS
OUT_OF_SCOPE
```

Do not proceed to speed search until this table exists.

## Phase KT1: Profile-Driven Large-Shape Candidate Author

Goal: generate bounded topology candidates for 14B/32B from profile facts, not hardcoded shapes.

Extend:

```text
extra/qk_topology_candidate_author.py
bench/qk-search-spaces/topology_grammar_v1.json
bench/qk-search-spaces/profiles/qwen3_14b_q4_k_m_gfx1100_decode.json
bench/qk-search-spaces/profiles/qwen3_32b_q4_k_m_gfx1100_decode.json
```

Required grammar expansion:

| dimension | required behavior |
|---|---|
| shape set | author per-role candidates, not one candidate forced to fit all roles |
| K decomposition | factor `K / 256` per role |
| block grouping | allow candidates where `block_groups` divides the role's `k_blocks` |
| words per group | allow divisor values of 32 quant words, not just 8 |
| rows per wave | include 1-row and 2-row candidates when valid |
| output mode | direct output by default; partials only if the candidate says why |
| candidate count | bounded per profile and per role |

Critical change from TG2:

TG2 pruned candidates using every eligible 8B role together. Large models need **role-local** topology because `gate/up`,
`down`, and `attn_q/o` have different `K`, `N`, and wall share. A candidate can be valid for one role and not another.

Outputs:

```text
bench/qwen-14b-32b-truegen/kt1_candidate_author/qwen3_14b/latest.json
bench/qwen-14b-32b-truegen/kt1_candidate_author/qwen3_32b/latest.json
bench/qwen-14b-32b-truegen/kt1_candidate_author/candidate_rows.json
bench/qwen-14b-32b-truegen/kt1_candidate_author/anti_cheat.json
```

Pass verdict:

```text
KT1_PASS_LARGE_SHAPE_CANDIDATES_AUTHORED
```

Block verdicts:

```text
KT1_BLOCKED_CANDIDATE_EXPLOSION
KT1_BLOCKED_PROFILE_INCOMPLETE
KT1_BLOCKED_GRAMMAR_CANNOT_EXPRESS_ROLE_LOCAL_TOPOLOGY
```

Anti-cheat requirements:

- The author may read `K`, `N`, quant, role, and target from the profile.
- The author may not branch on model name.
- The author may not inject a known winning candidate.
- Verification may compare to known G3; generation may not.

## Phase KT2: Make LaneMapTemplate And G2/G3 Emission Actually Parametric

Goal: make the emitted kernel reflect the authored topology, not silently collapse to old G3.

Extend:

```text
extra/qk_lanemap_template.py
extra/qk_gemv_g2_lanemap.py
extra/qk_gemv_g3_codegen_lowering.py
extra/qk_lane_partition_reduce.py
```

Minimum required work:

1. Remove the hard requirement that Q4_K lane maps always use `words_per_group == 8`.
2. Make `group_pairs = quant_words_per_block // words_per_group` derive from the candidate.
3. Ensure `LanePartition` and address generation remain correct for `words_per_group ∈ {1,2,4,8,16,32}` where legal.
4. Ensure the emitted kernel name and cache key include the topology hash.
5. Prove existing 8B G3 re-emits byte-identically for the `(4,8)` candidate.
6. Prove non-`(4,8)` candidates emit a different UOp key and kernel name.

Microgates:

| gate | requirement |
|---|---|
| `(4,8)` regression | byte-identical to current G3 |
| non-`(4,8)` emission | different key/name, no silent fallback |
| address correctness | packed word indices match a reference sampler for each legal `words_per_group` |
| numeric correctness | standalone random-vector GEMV matches dequant reference for candidate shapes |
| invalid shape rejection | loud error when `block_groups` or `words_per_group` are illegal |

Outputs:

```text
bench/qwen-14b-32b-truegen/kt2_parametric_emitter/latest.json
bench/qwen-14b-32b-truegen/kt2_parametric_emitter/microgate_rows.json
bench/qwen-14b-32b-truegen/kt2_parametric_emitter/uop_key_matrix.json
```

Pass verdict:

```text
KT2_PASS_PARAMETRIC_EMITTER
```

Block verdicts:

```text
KT2_CODEGEN_CAPABILITY_BLOCKED_WORDS_PER_GROUP
KT2_CODEGEN_CAPABILITY_BLOCKED_ROW_GROUPING
KT2_PRIMITIVE_BLOCKED_ADDRESSING
KT2_REFUTED_NON_8_WORD_GROUP_CORRECTNESS
```

If KT2 blocks, stop and scope the specific missing primitive/codegen capability. Do not write a bespoke large-model kernel.

## Phase KT3: Role-Local Candidate Microbench

Goal: cheaply rank generated candidates before full model W==D.

Build:

```text
extra/qk_large_shape_candidate_microbench.py
```

For each primary target shape:

1. Generate candidate kernels from KT1.
2. Compile each candidate through KT2.
3. Run correctness against the dequant reference.
4. Run a synchronized role-local GEMV benchmark on random inputs.
5. Compute effective GB/s using the role's Q4_K bytes.
6. Keep top candidates per role.

Outputs:

```text
bench/qwen-14b-32b-truegen/kt3_role_microbench/latest.json
bench/qwen-14b-32b-truegen/kt3_role_microbench/per_role_rankings.json
bench/qwen-14b-32b-truegen/kt3_role_microbench/refuted_candidates.json
bench/qwen-14b-32b-truegen/kt3_role_microbench/frontier_rows.json
```

Pass verdict:

```text
KT3_PASS_ROLE_LOCAL_CANDIDATES_RANKED
```

Candidate disposition:

| status | meaning |
|---|---|
| `KEEP_TOPK` | correct and faster than G3-anyshape for role-local microbench |
| `REFUTED_CANDIDATE` | correct but slower, and losing axis was represented |
| `CODEGEN_CAPABILITY_BLOCKED` | candidate cannot be emitted or spills/pathologically lowers |
| `SEARCH_SPACE_INCOMPLETE` | measured bottleneck points to an unrepresented axis |

Top-K default:

```text
top_k_per_role = 3
```

## Phase KT4: Profile-Policy Route Binding

Goal: bind the selected candidate per role without a global flag and without model-name hardcoding.

Extend:

```text
extra/qk_route_manifest.py
tinygrad/llm/model.py
```

Preferred mechanism:

```text
QK_GENERATED_POLICY=bench/qwen-14b-32b-truegen/kt4_route_policy/policy.json
```

Policy maps:

```text
profile_id + quant + role + shape + target -> candidate_id / topology_hash
```

Requirements:

- Default-off until KT5/KT6.
- `DECODE_Q4K_G3_ANYSHAPE` remains a diagnostic fallback, not the promotion mechanism.
- Route attribution proves the intended candidate fires for each selected role.
- Untargeted roles stay on shipped default.
- Token/logit equivalence passes for 14B and 32B at ctx128 and ctx512 where feasible.

Outputs:

```text
bench/qwen-14b-32b-truegen/kt4_route_policy/policy.json
bench/qwen-14b-32b-truegen/kt4_route_policy/latest.json
bench/qwen-14b-32b-truegen/kt4_route_policy/route_attribution.json
bench/qwen-14b-32b-truegen/kt4_route_policy/token_match.json
```

Pass verdict:

```text
KT4_PASS_PROFILE_POLICY_ROUTE_BOUND
```

## Phase KT5: Full Decode W==D And llama.cpp Fit Comparison

Goal: prove whole-decode movement, not just role-local speed.

Arms:

| arm | meaning |
|---|---|
| shipped_default | current tinygrad default |
| G3-anyshape | `DECODE_Q4K_G3_ANYSHAPE=1` baseline from Q1432 |
| shape_tuned_generated | KT4 profile-policy selected candidates |
| llama_matched | matched-context full-GPU-offload llama.cpp |

Contexts:

```text
ctx128
ctx512
ctx2048
ctx4096 if practical
```

Required metrics:

- W==D tok/s.
- token/logit equivalence.
- route counts by role.
- host-sync percent.
- spread/noise.
- effective GB/s.
- peak VRAM for tinygrad and llama full GPU offload.
- protected-context regression check.

Outputs:

```text
bench/qwen-14b-32b-truegen/kt5_wd/latest.json
bench/qwen-14b-32b-truegen/kt5_wd/per_ctx.json
bench/qwen-14b-32b-truegen/kt5_wd/memory_fit.json
bench/qwen-14b-32b-truegen/kt5_wd/llama_compare.json
bench/qwen-14b-32b-truegen/kt5_wd/route_counts.json
```

Pass verdicts:

```text
KT5_PASS_TIER_A_WD_MOVEMENT
KT5_PASS_TIER_B_RESIDUAL_MOVEMENT
KT5_PASS_SPEED_EQUIVALENT_TO_LLAMA_BAND
```

Blocked/refuted verdicts:

```text
KT5_REFUTED_NO_WD_MOVEMENT
KT5_REFUTED_SPEED_REGRESSION
KT5_BLOCKED_TINYGRAD_OOM_LLAMA_FITS
KT5_BLOCKED_LLAMA_COMPARISON_NOT_FULL_GPU
KT5_SEARCH_SPACE_INCOMPLETE_MISSING_AXIS
KT5_CODEGEN_CAPABILITY_BLOCKED
```

Promotion target:

- Minimum: beat G3-anyshape and shipped default with no protected-context regression.
- Track target: move 14B/32B from ~42% of llama toward >=80%.
- Stretch: >=95% of llama if the remaining gap is actually topology and not runtime/memory overhead.

## Phase KT6: Missing-Axis Audit

Goal: if KT5 does not close the gap, determine whether to expose more knobs or add primitives.

Build:

```text
extra/qk_large_shape_missing_axis_audit.py
```

Inputs:

- KT3 role rankings.
- KT5 W==D and route counts.
- generated UOp/ISA/static instruction summaries where available.
- candidate topology specs.
- memory-fit data.

Questions:

| question | possible conclusion |
|---|---|
| Did the selected candidates cover the dominant roles? | route-policy miss vs real performance miss |
| Did a represented axis lose? | `REFUTED_CANDIDATE` |
| Did every candidate share the same bottleneck? | missing axis or primitive |
| Is the bottleneck coalescing/addressing? | expose lane ownership / load vectorization axis |
| Is the bottleneck occupancy/register pressure? | expose rows-per-wave / accum placement axis |
| Is the bottleneck reductions/partials? | expose direct vs partials / split-K axis |
| Is the bottleneck lowering quality? | `CODEGEN_CAPABILITY_BLOCKED` |
| Is the bottleneck memory residency? | `MEMORY_BLOCKED` |

Outputs:

```text
bench/qwen-14b-32b-truegen/kt6_missing_axis/latest.json
bench/qwen-14b-32b-truegen/kt6_missing_axis/frontier_rows.json
bench/qwen-14b-32b-truegen/kt6_missing_axis/refuted_axes.json
bench/qwen-14b-32b-truegen/kt6_missing_axis/summary.md
```

Pass verdict:

```text
KT6_PASS_FRONTIER_CLASSIFIED
```

## Phase KT7: Ledger And Promotion Package

Goal: make the result durable.

Update:

```text
bench/qk-search-spaces/default_route_manifest.json
bench/qk-search-spaces/refuted_axes.json
bench/qk-search-spaces/open_frontier.json
bench/qk-search-spaces/search_profiles.json
docs/qwen-14b-32b-truegen-q1432-result-20260630.md
docs/README.md
```

Promotion requirements:

1. KT4 route-bound policy pass.
2. KT5 correctness pass.
3. KT5 TIER_A/TIER_B or speed-equivalent pass against the relevant baseline.
4. No protected context regression >1%.
5. Memory-fit bar passes.
6. Candidate generated from profile/grammar/template, not hand-coded.
7. Rollback path tested.

If all hold:

```text
KT7_PASS_PROMOTED_SHAPE_TUNED_GENERATED_ROUTE
```

If not:

```text
KT7_PASS_FRONTIER_LEDGERED
```

Ledger row fields:

| field | requirement |
|---|---|
| `candidate_id` | stable topology hash |
| `profile_id` | 14B/32B decode profile |
| `role` | ffn gate/up, down, attn q/o |
| `topology_spec` | full spec |
| `speed_delta` | role-local and W==D |
| `memory_delta` | peak VRAM and fit |
| `status` | promoted/refuted/frontier/codegen-blocked/memory-blocked |
| `missing_axis_or_capability` | populated for frontier/codegen-blocked |
| `reopen_condition` | exact condition for replay |
| `replay_command` | command to reproduce |

## Execution Order For Claude

Run in this order:

1. KT0 reachability audit.
2. KT1 large-shape candidate author.
3. KT2 parametric emitter.
4. KT3 role-local microbench.
5. KT4 profile-policy route binding.
6. KT5 W==D + llama/memory comparison.
7. KT6 missing-axis audit if KT5 does not close the gap.
8. KT7 ledger/promotion package.

Stop at the first hard blocker and classify it. Do not skip to hand-written code.

## Expected Outcomes

Most likely:

- KT0 finds `words_per_group` and row grouping are currently grammar-only or emitter-blocked.
- KT2 is the first real build: making `words_per_group != 8` actually lower.
- KT3 identifies different best candidates for `gate/up` and `down`, rather than one universal map.
- KT5 moves 14B/32B beyond the +8-9% G3-anyshape gain if topology was the missing lever.

Possible blocker:

- The best large-shape candidate needs a primitive not yet exposed, such as vectorized packed-word loads, a different row ownership family, or split-K partial handling. In that case, report `CODEGEN_CAPABILITY_BLOCKED` with the exact primitive, not `REFUTED`.

Success headline if KT7 promotes:

```text
Pure machine search authored, generated, route-bound, and promoted a shape-tuned Q4_K decode route for Qwen3 14B/32B on gfx1100.
```

That would be a stronger claim than TG2 rediscovering G3: this would be a new winning generated route for a new large-model shape class.

