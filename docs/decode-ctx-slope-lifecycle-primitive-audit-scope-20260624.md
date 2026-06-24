# Decode Ctx-Slope Lifecycle Primitive Audit Scope (2026-06-24)

## Objective

Explain the remaining decode ctx-slope after the current default stack.

Current benchmark:

| ctx | llama tok/s | tinygrad tok/s | tinygrad vs llama |
|---:|---:|---:|---:|
| 512 | 97.71 | 102.2 | 104.6% |
| 1024 | 97.39 | 100.4 | 103.1% |
| 2048 | 95.00 | 97.9 | 103.1% |
| 4096 | 92.37 | 93.4 | 101.1% |

The target is not "make decode faster at random." The target is to identify why the margin narrows with ctx and whether that narrowing maps to a bounded lifecycle primitive.

This audit runs before the full exhaustive GPU lifecycle audit. Its job is to produce the highest-value current decode evidence and to feed unknown/under-modeled primitive findings into the broader audit.

## Required Artifact Directory

`bench/qk-decode-ctx-slope-lifecycle-primitive-audit-20260624/`

## Required Outputs

- `authority.json`
- `llama_vs_tinygrad_decode_by_ctx.json`
- `decode_role_time_by_ctx.json`
- `attention_qk_pv_softmax_split_by_ctx.json`
- `kv_identity_materialization_by_ctx.json`
- `q4k_route_coverage_by_role.json`
- `programs_and_syncs_by_ctx.json`
- `smallop_residual_census.json`
- `unknown_primitive_candidates.json`
- `coverage_score_update.json`
- `decision.json`
- `summary.md`

## Read List

- `docs/gpu-lifecycle-primitive-coverage-tracker-20260624.md`
- `docs/exhaustive-gpu-lifecycle-primitive-audit-scope-20260624.md`
- `bench/qk-current-decode-benchmark/current.json`
- `bench/qk-decode-parity-no-regression-audit/llama_vs_tinygrad_table.json`
- `bench/qk-decode-oracle-explanation/primitive_decomposition.json`
- `bench/qk-machine-code-translation/primitive_inventory.json`
- `structure/Development/performance-primitive-research-principles.md`

## Audit Lanes

### Lane A: Authority And Comparator

Build `llama_vs_tinygrad_decode_by_ctx.json` from current local artifacts.

Required columns:

- `ctx`
- `llama_tok_s`
- `tinygrad_tok_s`
- `delta_tok_s`
- `tinygrad_pct_llama`
- `source_tinygrad`
- `source_llama`

Acceptance:

- ctx set includes 512, 1024, 2048, 4096
- sources are explicit
- any stale or non-current source is marked

### Lane B: Role Time By Context

Build `decode_role_time_by_ctx.json`.

Required roles:

- attention
- Q4_K/Q6_K GEMV roles
- lm_head
- RMSNorm/layernorm
- RoPE
- activation/residual ops
- copy/materialization
- other

Acceptance:

- rows include `ctx`, `role`, `ms`, `share`, `source`, `confidence`
- if a role cannot be split, mark `unknown_split` instead of assigning fake precision

### Lane C: Attention Subrole Split

Build `attention_qk_pv_softmax_split_by_ctx.json`.

Required split:

- `qk_ms`
- `softmax_mask_ms`
- `pv_ms`
- `combine_ms`
- `copy_ms`
- `kv_read_bytes_est`
- `effective_gb_s_est`

Acceptance:

- ctx-scaling component is named
- if profiler cannot split it, state the required instrumentation

### Lane D: KV Identity / Materialization Guard

Build `kv_identity_materialization_by_ctx.json`.

Required checks:

- no `E_49152` materialization regression
- whole-buffer identity path active
- no sliced-cache view passed across precompiled-call boundary
- cache read path is route-identified

Acceptance:

- pass/fail per ctx
- exact evidence source per check

### Lane E: Q4K Route Coverage

Build `q4k_route_coverage_by_role.json`.

Required roles:

- gate/up
- down
- q/k/v/o projection where applicable
- lm_head

Required fields:

- `role`
- `route`
- `flag_state`
- `expected_default`
- `actual_default`
- `coverage_status`

Acceptance:

- any promoted path that is expected but not active is `DECODE_CTX_SLOPE_ROUTE_REGRESSION`

### Lane F: Programs And Syncs

Build `programs_and_syncs_by_ctx.json`.

Required fields:

- `ctx`
- `programs_per_token`
- `item_syncs_per_token`
- `host_sync_pct`
- `dispatch_ms`
- `wall_ms`

Acceptance:

- current expected value is 6 programs/token and host sync near 0
- any program-count growth with ctx is flagged

### Lane G: Small-Op Residual Census

Build `smallop_residual_census.json`.

Required fields:

- op family
- ctx
- ms
- share
- fusable_with
- searchable
- correctness risk

Acceptance:

- only propose fusion if wall share is measurable and the producer/consumer lifecycle is named

### Lane H: Unknown Primitive Discovery

Build `unknown_primitive_candidates.json`.

Purpose:

Do not assume the current primitive taxonomy is complete. Capture time, memory movement, launch behavior, or route effects that do not fit the existing categories.

Required fields:

- `candidate_id`
- `observed_signal`
- `ctx`
- `role_or_kernel_name`
- `time_ms`
- `share`
- `why_not_classified`
- `possible_lifecycle_boundary`
- `required_next_probe`
- `priority`

Candidate classes to watch for:

- hidden materialization/copy not labeled as KV or attention
- allocator/cache churn
- graph rebuild or JIT cache behavior
- command queue / HCQ behavior
- wave occupancy or register cliff not represented in current role buckets
- memory layout/coalescing issue outside known KV/GEMV paths
- compiler lowering artifact: vectorization, LDS, cross-lane, waitcnt, scalarization
- host preprocessing or token/logit postprocessing not captured by current decode benchmark

Acceptance:

- every unclassified timing bucket above 2% wall share must either map to an existing category or be emitted here
- every unknown candidate must name the next minimal probe needed to classify it

## Decision Labels

`decision.json` must select exactly one:

- `DECODE_CTX_SLOPE_KV_READ_BOUND`
- `DECODE_CTX_SLOPE_ATTENTION_TILE_BOUND`
- `DECODE_CTX_SLOPE_SMALL_OP_BOUND`
- `DECODE_CTX_SLOPE_ROUTE_REGRESSION`
- `DECODE_CTX_SLOPE_LAUNCH_GRAPH_BOUND`
- `DECODE_CTX_SLOPE_NO_ACTION_UNDER_8B_MAXC`

## Coverage Score Updates

Update these categories in `coverage_score_update.json`:

- `decode_attention_tile`
- `kv_cache_read_lifecycle`
- `smallop_lifecycle`
- `memory_bandwidth_layout`
- `launch_graph_lifecycle`
- `harness_authority_lifecycle`
- `unknown_primitive_discovery`

Each update must include:

- previous `exploration_gap_percent`
- new `exploration_gap_percent`
- previous `time_correctness_confidence_percent`
- new `time_correctness_confidence_percent`
- derived `effective_explored_percent`
- reason
- artifact evidence
- remaining missing points

Scoring rule:

- `exploration_gap_percent` falls only when the audit covers more of the relevant design/search space.
- `time_correctness_confidence_percent` rises only when the audit improves benchmark authority, correctness confidence, route/materialization proof, comparator quality, or regression guard quality.

Example:

If the audit proves KV identity is clean at every ctx but does not search new KV layouts, then:

- `time_correctness_confidence_percent` for `kv_cache_read_lifecycle` should rise.
- `exploration_gap_percent` may only fall slightly, because alternate KV layouts remain unexplored.

## Expected Outcome

Most likely outcomes:

- if KV read/materialization is clean and attention split is stable: `DECODE_CTX_SLOPE_NO_ACTION_UNDER_8B_MAXC`
- if attention subrole grows materially with ctx: `DECODE_CTX_SLOPE_ATTENTION_TILE_BOUND`
- if no attention/KV issue appears and small ops are measurable: `DECODE_CTX_SLOPE_SMALL_OP_BOUND`

No default changes are made by this audit.
