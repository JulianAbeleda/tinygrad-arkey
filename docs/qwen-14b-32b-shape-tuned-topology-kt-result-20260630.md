# Shape-Tuned Topology Search (KT0-KT7) — result

Date: 2026-06-30

Status: worked the KT track to an evidence-backed frontier. The `words_per_group` axis is exhausted-or-worse and
codegen-blocked; the real missing lever is **split-K**. No fake win forced, route family NOT refuted.
Scope: `docs/qwen-14b-32b-shape-tuned-topology-search-scope-20260630.md`. Hardware: gfx1100. Models: Qwen3-14B/32B.

## KT0 — knob reachability (PASS)

`extra/qk_large_shape_knob_reachability_audit.py`. The dominant tuning axes are decorative, not the primitives:

| axis | status |
|---|---|
| words_per_group | EMITTER_BLOCKED (IR hard-locks ==8; kernel sig is `[rows,k,lanes]`) |
| block_groups | EMITTER_BLOCKED (coupled `bg*wpg=32`) |
| reduction_pattern | GRAMMAR_ONLY (emit never branches → always cross-lane) |
| row_grouping | EMITTER_BLOCKED (row is a single GLOBAL axis) |
| lane_ownership_index | EMITTER_BLOCKED (validate requires the one G3 formula) |
| vector/multiword load | EMITTER_BLOCKED (primitive exists, unwired) |
| q8_1 activation | OUT_OF_SCOPE |

Confirms the "missing exposed knobs" read (`KT0_PASS_REACHABILITY_PINNED`).

## KT1/KT2/KT3 — the reachable topology space is exhausted-or-worse

`extra/qk_large_shape_topology_space_audit.py`. The G3 wave splits 32 lanes as `block_groups * words_per_group = 32`,
and the K reduction is `block_groups` (cross-lane) × `blocks_per_group` (serial), with `block_groups | k_blocks`.
So **`block_groups ≤ gcd(32, k_blocks)`**. Enumerating the legal topologies per target shape:

| role | in→out | k_blocks | gcd(32,kb) | max legal bg | reachable wpg | serial blocks/lane @bg4 |
|---|---|---|---|---|---|---|
| attn_q/o | 5120→5120 | 20 | 4 | 4 | [8,16,32] | 5 |
| 14b ffn gate/up | 5120→17408 | 20 | 4 | 4 | [8,16,32] | 5 |
| 14b ffn down | 17408→5120 | 68 | 4 | 4 | [8,16,32] | 17 |
| 32b attn_q | 5120→8192 | 20 | 4 | 4 | [8,16,32] | 5 |
| 32b ffn gate/up | 5120→25600 | 20 | 4 | 4 | [8,16,32] | 5 |
| 32b ffn down | 25600→5120 | 100 | 4 | 4 | [8,16,32] | 25 |

**Every** shape has `gcd(32, k_blocks) = 4`. So the shipped `bg=4/wpg=8` is already the **most-parallel legal
topology**; the only reachable alternatives (`wpg` 16/32 → `bg` 2/1) are strictly more serial. Varying
`words_per_group` cannot reduce serial work for these K — the axis is exhausted-or-worse, deterministically (not a
measurement guess).

Separately, emitting `wpg != 8` is codegen-blocked: `_q4k_block_dot_packed_load` indexes
`words[base+4+(grp//2)*8+lane4]` with `lane4 ∈ [0,8)` — the dot primitive is structurally tied to 8 word-columns.

- `KT2_CODEGEN_CAPABILITY_BLOCKED_WORDS_PER_GROUP`
- `KT3_SEARCH_SPACE_INCOMPLETE_MISSING_SPLIT_K`

## KT6 — missing-axis: split-K

The 32-lane wave caps cross-lane K-parallelism at `gcd(32, k_blocks) = 4`. For `k=17408`/`25600` that leaves 17/25
Q4_K blocks reduced **serially per lane** — the serial dequant/accumulate chain is the bottleneck, not
`words_per_group`. More K-parallelism requires **split-K**: split the K reduction across multiple workgroups per
output row and combine partials (the `partials_plus_reduce` pattern KT0 found is GRAMMAR_ONLY). A partials `parts`
primitive already exists in the **gemm** path (`q4k_gemm_kernel(..., parts, ...)`, `q4k_gemv_*_partial_kernel`) but
is **not wired into the generated G3 GEMV decode route**.

## KT7 — frontier ledger (not promoted, not refuted)

| field | value |
|---|---|
| candidate_id | `g3_wpg_variation_large_shapes` |
| profile_id | qwen3-14b/32b Q4_K decode gfx1100 |
| role | ffn gate/up, ffn down, attn q/o |
| status | `SEARCH_SPACE_INCOMPLETE` (wpg exhausted) + `CODEGEN_CAPABILITY_BLOCKED` (wpg!=8 dot) |
| measured_delta | none attempted — reachable wpg∈{16,32} are provably more serial than shipped bg=4 |
| dominant_failed_row | K-parallelism capped at `gcd(32,k_blocks)=4`; serial blocks/lane = 5-25 |
| missing_axis_or_capability | **split-K**: partials across workgroups for decode Q4_K GEMV (grammar `reduction_pattern=partials_plus_reduce` must reach the emitter; a `parts` primitive exists in the gemm path to adapt) |
| reopen_condition | grammar exposes a `split_k`/workgroups-per-row axis AND the G3 emitter wires the partials primitive; then microbench split-K candidates for k∈{5120,17408,25600} vs G3-anyshape |
| replay_command | `DEV=AMD PYTHONPATH=. python3 extra/qk_large_shape_topology_space_audit.py` |

## Bottom line

The Q1432 route-binding win (+8-9%, shipped default-off) is the ceiling of the *current* generated topology for
these shapes — the topology is already optimal in its grammar; the remaining ~58% bandwidth gap is a **split-K
capability gap**, not a `words_per_group` tuning gap. Building a parametric-`wpg` emitter was correctly NOT done: it
would emit only strictly-more-serial candidates. The honest next lever is a split-K decode-GEMV route family (a new
grammar axis + emitter wiring), which is a distinct scope, not a continuation of this one.
