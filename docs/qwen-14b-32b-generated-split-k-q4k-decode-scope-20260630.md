# Generated Split-K Q4_K Decode GEMV Scope For Qwen3 14B/32B

Date: 2026-06-30

Status: execution scope for Claude. This follows the KT0-KT7 result that classified the remaining 14B/32B decode gap as
a split-K capability frontier. No hand-written large-model kernel is allowed.

## Why This Is The Next Lever

KT0-KT7 closed the `words_per_group` theory:

- The current generated G3 route is correct and route-bound for large Q4_K shapes.
- It gives +8-9% on 14B, but 14B/32B remain around 42% of llama.cpp.
- Within one wave, G3 splits 32 lanes as `block_groups * words_per_group = 32`.
- For all target Q4_K shapes, `gcd(32, k_blocks) = 4`, so the maximum legal `block_groups` is already 4.
- Legal `words_per_group` alternatives are strictly more serial.

Therefore the next route family is not `words_per_group` tuning. It is **split-K across workgroups**:

```text
current direct route:
  one workgroup per output row
  one 32-lane wave covers up to 4 K block-groups
  each lane serially reduces 5, 17, or 25 Q4_K blocks
  in-kernel wave reduce -> out[row]

split-K route:
  multiple workgroups per output row, each owns a K slice
  each workgroup writes partials[row, split]
  combine partials -> out[row]
```

This is a new generated route family, not a hand-kernel patch.

## Source Citations

Read these before implementation:

| claim | citation |
|---|---|
| KT result and split-K frontier | `docs/qwen-14b-32b-shape-tuned-topology-kt-result-20260630.md` |
| Q1432 route binding result | `docs/qwen-14b-32b-truegen-q1432-result-20260630.md` |
| Full Q1432 scope/outcome taxonomy | `docs/qwen-14b-32b-true-generation-kernel-authoring-scope-20260630.md` |
| Current generated G3 route | `extra/qk_gemv_g3_codegen_lowering.py` |
| Current G2 LaneMap | `extra/qk_gemv_g2_lanemap.py` |
| LaneMapTemplate IR | `extra/qk_lanemap_template.py` |
| Existing Q4_K partial kernels | `extra/q4_k_gemv_primitive.py`, especially `q4k_gemv_partial_kernel`, `q4k_gemv_packed_load_partial_kernel`, `q4k_gemm_kernel` |
| Existing model partials path | `tinygrad/llm/model.py`, `Q4KPrimitiveLinear.__call__` fallback partial path |
| Current generated-policy install | `tinygrad/llm/model.py`, `_install_q4k_primitives` |
| Candidate/evaluator substrate | `extra/qk_candidate_evaluator.py`, `extra/qk_template_candidate_gate.py`, `extra/qk_route_manifest.py` |
| Topology grammar | `bench/qk-search-spaces/topology_grammar_v1.json` |
| Quant/target facts | `bench/qk-search-spaces/quant_semantics.json`, `bench/qk-search-spaces/targets/amd_gfx1100.json` |

## Discovery Strategy: 14B First, Then 32B Transfer

Do not brute-force 14B and 32B as unrelated searches. Use 14B as the cheaper discovery model, then test whether the learned
rule transfers to 32B.

The intended learned rule shape is:

```text
split_k_policy = f(role, k_blocks, rows, serial_blocks_per_lane, target_wave32, memory_budget)
```

not:

```text
if model_name == "qwen3-14b": ...
if model_name == "qwen3-32b": ...
```

Why this should transfer:

| shared fact | implication |
|---|---|
| both are dense Qwen3 Q4_K_M models | same decode GEMV route family |
| both large FFN-down roles have `gcd(32,k_blocks)=4` | same one-wave K-parallelism cap |
| 14B ffn down has serial depth 17; 32B ffn down has 25 | 32B is the stronger version of the same bottleneck |
| both use the same Q4_K quant block layout and gfx1100 wave32 target | same generated split-K primitive applies |

But transfer is not assumed. It must be measured.

Transfer ladder:

1. Discover candidates on 14B ffn down first.
2. Convert the winning candidate into a rule over `k_blocks` and serial depth.
3. Apply that rule to 32B ffn down.
4. If transfer succeeds, test gate/up and attention roles.
5. If transfer fails, classify whether the rule needs a missing feature such as different split count scaling, generated combine, or row grouping.

## Target Shapes

Primary roles:

| shape | role | serial Q4_K blocks/lane under direct G3 |
|---|---|---:|
| `5120 -> 17408` | 14B ffn gate/up | 5 |
| `17408 -> 5120` | 14B ffn down | 17 |
| `5120 -> 25600` | 32B ffn gate/up | 5 |
| `25600 -> 5120` | 32B ffn down | 25 |

Secondary roles:

| shape | role | note |
|---|---|---|
| `5120 -> 5120` | 14B attn q/o | lower serial depth, likely less leverage |
| `5120 -> 8192` | 32B attn q | lower serial depth, likely less leverage |

Priority order:

1. 14B ffn down `17408 -> 5120` because it has serial depth 17 and is cheaper to iterate than 32B.
2. 32B ffn down `25600 -> 5120` because it has serial depth 25 and should benefit if the pattern transfers.
3. ffn gate/up and attention projection only if the precheck shows enough wall share.

## Non-Negotiables

- Do not write a bespoke 14B/32B kernel.
- Split count must be a generated/search axis, not a hardcoded model branch.
- The route must be default-off until correctness, speed, memory, and route gates pass.
- If split-K loses because combine overhead dominates, record `REFUTED_SPLIT_K_OVERHEAD` for that shape/split count.
- If split-K cannot be emitted from the generated substrate, record `CODEGEN_CAPABILITY_BLOCKED`.
- Keep current direct G3 and shipped default as rollback.
- Do not claim global pure machine search. A pass means generated split-K Q4_K decode for the passing profile class.

## Split-K Search Variables

Add these to the generated candidate spec:

| field | meaning |
|---|---|
| `split_k_parts` | number of K partitions/workgroups per output row |
| `blocks_per_split` | ceil/floor partition of `k_blocks` |
| `split_axis` | workgroup axis that owns the K split |
| `partial_layout` | `partials[row, split]` contiguous layout |
| `partial_dtype` | fp32 initially |
| `combine_route` | Tensor `.sum(axis=1)` baseline, generated reduce, or fused combine later |
| `base_topology` | direct G3 lane map inside each split |
| `role_scope` | role-local candidate, not one topology forced across every role |

Initial split candidates:

```text
split_k_parts ∈ {2, 4, 8}
```

Prune rules:

- `split_k_parts <= k_blocks / 4` so each split has enough work to amortize launch/combine.
- Avoid split counts where `blocks_per_split < 4` unless a microbench proves it helps.
- Prefer powers of two for combine simplicity first.
- Keep candidate count bounded.

## Phase SK0: Split-K Math And Amdahl Precheck

Goal: decide whether split-K has enough theoretical room before building the route.

Build:

```text
extra/qk_large_shape_split_k_precheck.py
```

For each target role:

1. Read `k_blocks`, `rows`, Q4_K bytes, and role share.
2. Estimate direct serial depth: `ceil(k_blocks / 4)`.
3. For each `split_k_parts`, estimate split serial depth: `ceil(k_blocks / (4 * split_k_parts))`.
4. Estimate extra work:
   - partial writes: `rows * split_k_parts * 4` bytes.
   - combine reads: same order.
   - extra kernel launch or graph node.
5. Compute a conservative benefit bound:

```text
speedup_bound ≈ direct_serial_depth / split_serial_depth
net_bound ≈ speedup_bound adjusted by partial_write + combine overhead
```

Outputs:

```text
bench/qwen-14b-32b-truegen/sk0_split_k_precheck/latest.json
bench/qwen-14b-32b-truegen/sk0_split_k_precheck/summary.md
bench/qwen-14b-32b-truegen/sk0_split_k_precheck/candidate_bounds.json
```

Pass verdict:

```text
SK0_PASS_SPLIT_K_JUSTIFIED
```

Stop verdict:

```text
SK0_REFUTED_SPLIT_K_AMDAHL
```

Do not proceed if no role has a plausible TIER_B or better wall-clock bound.

## Phase SK1: Generated Split-K Topology IR

Goal: extend the template/search representation to express split-K without changing live routes.

Extend:

```text
extra/qk_lanemap_template.py
bench/qk-search-spaces/topology_grammar_v1.json
extra/qk_topology_candidate_author.py
```

Add a split-K spec alongside `TopologySpec`, for example:

```text
SplitKTopologySpec:
  base_topology: TopologySpec
  split_k_parts: int
  split_axis: GLOBAL
  partial_layout: row_major_row_split
  combine_route: tensor_sum_baseline | generated_reduce
```

Requirements:

- Existing G3 direct topology still re-emits unchanged.
- `split_k_parts=1` is equivalent to direct route or explicitly rejected as redundant.
- Candidates are role-local.
- Candidate id includes `rows`, `k`, `split_k_parts`, and topology hash.
- Generation reads only profile/quant/target facts.

Outputs:

```text
bench/qwen-14b-32b-truegen/sk1_split_k_ir/latest.json
bench/qwen-14b-32b-truegen/sk1_split_k_ir/candidate_rows.json
bench/qwen-14b-32b-truegen/sk1_split_k_ir/anti_cheat.json
```

Pass verdict:

```text
SK1_PASS_SPLIT_K_CANDIDATES_AUTHORED
```

## Phase SK2: Generated Split-K Partial Kernel

Goal: emit the first generated split-K partial kernel using the G3 lane map inside each K slice.

Extend:

```text
extra/qk_gemv_g3_codegen_lowering.py
extra/qk_gemv_g2_lanemap.py
extra/q4_k_gemv_primitive.py
```

Preferred kernel shape:

```text
q4k_g3_lanemap_splitk_partial_kernel(rows, k, split_k_parts, lanes=32)

grid:
  gidx0 = row
  gidx1 = split
  lidx0 = lane

write:
  partials[row, split] = sum over this split's K blocks
```

Use the existing direct G3 lane map for each split:

```text
bg = lane // 8
lane4 = lane % 8
block_start = split * blocks_per_split
blk = block_start + bg * local_blocks_per_group + local_block
if blk < k_blocks: accumulate else zero
```

Microgates:

| gate | requirement |
|---|---|
| direct regression | existing direct G3 unchanged |
| partial correctness | `partials.sum(axis=1)` matches dequant reference |
| split edge | non-even `k_blocks % split_k_parts` handled by in-range mask |
| route identity | kernel name/key includes split count |
| no fallback | route spy proves split partial kernel ran |

Outputs:

```text
bench/qwen-14b-32b-truegen/sk2_split_k_partial/latest.json
bench/qwen-14b-32b-truegen/sk2_split_k_partial/microgate_rows.json
bench/qwen-14b-32b-truegen/sk2_split_k_partial/uop_key_matrix.json
```

Pass verdict:

```text
SK2_PASS_SPLIT_K_PARTIAL_KERNEL
```

Blocked verdicts:

```text
SK2_CODEGEN_CAPABILITY_BLOCKED_GRID_AXIS
SK2_CODEGEN_CAPABILITY_BLOCKED_INRANGE_MASK
SK2_REFUTED_NUMERIC_CORRECTNESS
```

## Phase SK3: Combine Baseline And Combine Cost Gate

Goal: measure the cost of combining partials before full model binding.

Start with:

```text
partials.sum(axis=1)
```

This is intentionally boring. It is the baseline combine route. Only build a generated combine if this baseline is correct but too expensive.

Build:

```text
extra/qk_large_shape_split_k_combine_gate.py
```

Measure for each candidate:

- partial kernel time.
- combine time.
- total role-local time.
- direct G3-anyshape role-local time.
- combine share of split route.

Outputs:

```text
bench/qwen-14b-32b-truegen/sk3_combine_gate/latest.json
bench/qwen-14b-32b-truegen/sk3_combine_gate/per_candidate.json
bench/qwen-14b-32b-truegen/sk3_combine_gate/summary.md
```

Pass verdicts:

```text
SK3_PASS_COMBINE_COST_ACCEPTABLE
SK3_BLOCKED_NEEDS_GENERATED_COMBINE
```

Refute verdict:

```text
SK3_REFUTED_SPLIT_K_COMBINE_OVERHEAD
```

Rule:

- If combine is less than 20% of split route and total role-local speed improves, proceed.
- If combine dominates but partial kernel improves, scope generated combine.
- If total route loses, refute that split count for that role.

## Phase SK4A: 14B Discovery Search

Goal: search split-K on 14B first and produce a learned rule candidate.

Build:

```text
extra/qk_large_shape_split_k_14b_discovery.py
```

Run candidates on:

```text
14B ffn_down: 17408 -> 5120
optional 14B gate/up: 5120 -> 17408 if SK0 says it has enough bound
```

For each split count:

1. Verify correctness.
2. Measure partial kernel, combine, and total role-local speed.
3. Compare against direct G3-anyshape.
4. Record serial-depth reduction and combine tax.
5. Select the best candidate or explain why none wins.

Outputs:

```text
bench/qwen-14b-32b-truegen/sk4a_14b_discovery/latest.json
bench/qwen-14b-32b-truegen/sk4a_14b_discovery/per_candidate.json
bench/qwen-14b-32b-truegen/sk4a_14b_discovery/learned_rule.json
bench/qwen-14b-32b-truegen/sk4a_14b_discovery/summary.md
```

Pass verdicts:

```text
SK4A_PASS_14B_DISCOVERY_RULE_LEARNED
SK4A_REFUTED_14B_SPLIT_K_NO_ROLE_LOCAL_WIN
SK4A_BLOCKED_14B_NEEDS_GENERATED_COMBINE
```

`learned_rule.json` must contain:

| field | meaning |
|---|---|
| `source_profile` | 14B decode profile |
| `source_role` | role used for discovery |
| `k_blocks` | source K blocks |
| `serial_blocks_per_lane_direct` | direct G3 serial depth |
| `winning_split_k_parts` | selected split count |
| `combine_route` | `.sum` or generated combine |
| `selection_reason` | measured role-local reason |
| `transfer_rule` | expression over role/k_blocks/serial depth, not model name |

## Phase SK4B: 32B Transfer Validation

Goal: apply the 14B-learned split-K rule to 32B and test whether the pattern generalizes.

Build:

```text
extra/qk_large_shape_split_k_32b_transfer.py
```

Apply the learned rule to:

```text
32B ffn_down: 25600 -> 5120
```

Then optionally test:

```text
32B ffn_gate/up: 5120 -> 25600
32B attn_q: 5120 -> 8192
```

Transfer verdicts:

```text
SK4B_PASS_TRANSFER_TO_32B
SK4B_PARTIAL_TRANSFER_ROLE_LIMITED
SK4B_REFUTED_TRANSFER_NO_MOVEMENT
SK4B_BLOCKED_TRANSFER_NEEDS_DIFFERENT_SPLIT_SCALING
SK4B_BLOCKED_TRANSFER_NEEDS_GENERATED_COMBINE
```

Outputs:

```text
bench/qwen-14b-32b-truegen/sk4b_32b_transfer/latest.json
bench/qwen-14b-32b-truegen/sk4b_32b_transfer/transfer_rows.json
bench/qwen-14b-32b-truegen/sk4b_32b_transfer/summary.md
```

Transfer pass requires:

- same rule chosen without a model-name branch;
- correctness passes;
- role-local 32B movement is positive;
- combine overhead does not dominate;
- memory overhead is acceptable.

If 14B wins but 32B fails, do not call split-K globally refuted. Classify the transfer failure by the missing variable:

| failure | likely next axis |
|---|---|
| 32B wants larger split count | split scaling rule |
| combine dominates only on 32B | generated combine |
| partial kernel slows at 32B | memory coalescing / occupancy |
| no role-local movement | split-K not the bottleneck |

## Phase SK4C: Role-Local Split-K Search Package

Goal: consolidate 14B discovery and 32B transfer into the selected role-policy candidates before in-model binding.

Build:

```text
extra/qk_large_shape_split_k_role_search.py
```

Inputs:

```text
sk4a_14b_discovery/learned_rule.json
sk4b_32b_transfer/transfer_rows.json
```

For each accepted role:

1. Run split candidates from SK1/SK2.
2. Verify correctness.
3. Measure role-local speed vs direct G3-anyshape and shipped fallback.
4. Pick top candidate per role.

Outputs:

```text
bench/qwen-14b-32b-truegen/sk4c_role_search/latest.json
bench/qwen-14b-32b-truegen/sk4c_role_search/per_role_rankings.json
bench/qwen-14b-32b-truegen/sk4c_role_search/frontier_rows.json
```

Pass verdict:

```text
SK4C_PASS_ROLE_LOCAL_SPLIT_K_RANKED
```

Expected:

- `ffn_down` should benefit most.
- `gate/up` may benefit less because serial depth is only 5.
- attention q/o may not justify split overhead.

## Phase SK5: Profile-Policy In-Model Binding

Goal: bind role-specific split-K winners without a global flag.

Extend:

```text
tinygrad/llm/model.py
extra/qk_route_manifest.py
```

Use:

```text
QK_GENERATED_POLICY=bench/qwen-14b-32b-truegen/sk5_route_policy/policy.json
```

Policy maps:

```text
profile_id + role + quant + rows + k + target -> split_k_candidate
```

Requirements:

- default-off.
- direct G3-anyshape remains rollback.
- route attribution proves only selected roles use split-K.
- token/logit equivalence passes.
- memory overhead of partials is recorded.

Outputs:

```text
bench/qwen-14b-32b-truegen/sk5_route_policy/latest.json
bench/qwen-14b-32b-truegen/sk5_route_policy/policy.json
bench/qwen-14b-32b-truegen/sk5_route_policy/route_attribution.json
bench/qwen-14b-32b-truegen/sk5_route_policy/token_match.json
```

Pass verdict:

```text
SK5_PASS_SPLIT_K_ROUTE_BOUND
```

## Phase SK6: Full W==D And Memory Fit

Goal: prove split-K improves real decode.

Arms:

| arm | meaning |
|---|---|
| shipped_default | current tinygrad default |
| G3-anyshape | `DECODE_Q4K_G3_ANYSHAPE=1` direct generated route |
| split-K generated | SK5 policy-bound route |
| llama.cpp | matched-context full GPU offload |

Contexts:

```text
ctx128
ctx512
ctx2048
ctx4096 if practical
```

Required metrics:

- W==D tok/s.
- route counts.
- token/logit correctness.
- host-sync percent.
- spread/noise.
- peak VRAM and partial-buffer bytes.
- llama full-GPU fit validity.
- per-role movement if available.
- whether 32B used the 14B-learned rule unchanged or a measured transfer-specific adjustment.

Outputs:

```text
bench/qwen-14b-32b-truegen/sk6_wd/latest.json
bench/qwen-14b-32b-truegen/sk6_wd/per_ctx.json
bench/qwen-14b-32b-truegen/sk6_wd/memory_fit.json
bench/qwen-14b-32b-truegen/sk6_wd/route_counts.json
bench/qwen-14b-32b-truegen/sk6_wd/llama_compare.json
```

Pass verdicts:

```text
SK6_PASS_TIER_A_WD_MOVEMENT
SK6_PASS_TIER_B_RESIDUAL_MOVEMENT
SK6_PASS_SPEED_EQUIVALENT_TO_LLAMA_BAND
```

Blocked/refuted verdicts:

```text
SK6_REFUTED_SPLIT_K_WD_REGRESSION
SK6_REFUTED_SPLIT_K_NO_WD_MOVEMENT
SK6_BLOCKED_TINYGRAD_OOM_LLAMA_FITS
SK6_BLOCKED_LLAMA_COMPARISON_NOT_FULL_GPU
SK6_BLOCKED_NEEDS_GENERATED_COMBINE
```

Promotion target:

- Minimum: beat G3-anyshape with no protected-context regression.
- Track target: move 14B/32B from ~42% of llama toward >=80%.
- If split-K helps role-local but not W==D, classify wall-share or combine overhead precisely.

## Phase SK7: Optional Generated Combine

Run only if SK3/SK6 report:

```text
SK3_BLOCKED_NEEDS_GENERATED_COMBINE
or
SK6_BLOCKED_NEEDS_GENERATED_COMBINE
```

Goal: replace Tensor `.sum(axis=1)` with a generated combine route.

Build:

```text
extra/qk_gemv_split_k_combine_codegen.py
```

Candidate combine routes:

| route | description |
|---|---|
| one row per workgroup | reduce `partials[row, split]` in one small workgroup |
| multiple rows per workgroup | amortize launch overhead for small `parts` |
| vectorized row combine | load 2/4 partials per lane where useful |

Gates:

- combine correctness vs `.sum(axis=1)`.
- combine speed vs `.sum(axis=1)`.
- full W==D replay.

Pass verdict:

```text
SK7_PASS_GENERATED_COMBINE
```

Refute verdict:

```text
SK7_REFUTED_GENERATED_COMBINE_NO_MOVEMENT
```

## Phase SK8: Ledger And Promotion Package

Goal: make the outcome durable.

Update:

```text
bench/qk-search-spaces/default_route_manifest.json
bench/qk-search-spaces/refuted_axes.json
bench/qk-search-spaces/open_frontier.json
bench/qk-search-spaces/search_profiles.json
docs/qwen-14b-32b-shape-tuned-topology-kt-result-20260630.md
docs/README.md
```

Promotion requirements:

1. SK5 route-bound.
2. SK6 correctness.
3. SK6 TIER_A/TIER_B or speed-equivalent pass.
4. No protected-context regression >1%.
5. Memory-fit bar passes.
6. Candidate generated from profile/grammar/template, not hand-coded.
7. Rollback path tested.

Promotion verdict:

```text
SK8_PASS_PROMOTED_GENERATED_SPLIT_K_Q4K_DECODE
```

Frontier verdict:

```text
SK8_PASS_FRONTIER_LEDGERED
```

Ledger rows must include:

| field | meaning |
|---|---|
| `candidate_id` | stable split-K candidate id |
| `profile_id` | 14B/32B decode profile |
| `role` | targeted model role |
| `split_k_parts` | split count |
| `combine_route` | `.sum` or generated combine |
| `role_local_delta` | role-local speed delta |
| `transfer_status` | 14B-only, transferred-to-32B, partial-transfer, or transfer-refuted |
| `learned_rule` | rule expression if discovered on 14B |
| `wd_delta` | W==D delta |
| `memory_delta` | partial buffer and peak VRAM |
| `status` | promoted/refuted/frontier |
| `reopen_condition` | if not promoted |
| `replay_command` | exact replay command |

## Claude Execution Instructions

Run sequentially:

1. SK0 precheck. Stop if split-K has no plausible wall-clock upside.
2. SK1 IR/candidates.
3. SK2 generated partial kernel.
4. SK3 combine-cost gate.
5. SK4A 14B discovery search.
6. SK4B 32B transfer validation.
7. SK4C role-local policy package.
8. SK5 route policy binding.
9. SK6 full W==D/memory/llama comparison.
10. SK7 only if combine blocks a promising split-K partial.
11. SK8 ledger/promotion package.

Do not skip SK0. Do not skip the combine-cost gate. Do not promote from role-local results alone. Do not hide a missing
generated combine behind a manual kernel. Do not run 32B as a brute-force independent search unless SK4B proves the
14B-learned rule fails and classifies the missing transfer axis.

## Expected Outcomes

Likely:

- 14B/32B ffn down benefits most because serial depth is 17/25 blocks per lane under direct G3.
- split-K parts 2 or 4 are the first plausible candidates; 8 may overpay combine/launch.
- gate/up may not benefit enough because direct serial depth is only 5.
- A 14B ffn-down winner should transfer directionally to 32B ffn-down if the real rule is serial-depth/K-parallelism.

Possible failure:

- Split-K partials improve but combine overhead erases W==D. Then SK7 is the next frontier.
- Split-K loses role-local. Then the large-shape gap is not K parallelism alone, and SK8 should ledger a refutation.

Success headline if promoted:

```text
Pure machine search authored and promoted a generated split-K Q4_K decode route for Qwen3 14B/32B on gfx1100.
```
