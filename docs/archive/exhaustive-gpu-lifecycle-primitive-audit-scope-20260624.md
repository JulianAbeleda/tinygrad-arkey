# Exhaustive GPU Lifecycle Primitive Audit Scope (2026-06-24)

## Objective

Build an exhaustive primitive audit for the current project state. The goal is to answer:

- what percent of the relevant GPU/lifecycle primitive space has been explored;
- which benchmark proves or refutes each region;
- what remains unexplored;
- whether the next action is search, audit, regression guard, or defer.

This is broader than the current `gpu-lifecycle-primitive-coverage-tracker-20260624.md`. The tracker is a manual current-state map. This scope defines the tool-backed audit needed to make the percentages reproducible.

Important assumption: the current taxonomy is not complete. The audit must include an unknown-primitive discovery lane. A category cannot score 100% unless unknown/unclassified timing and lifecycle effects have been explicitly searched for, classified, or ruled out.

## Current Tooling Status

We have partial tooling, not a complete exhaustive audit tool.

| tool/artifact | exists | what it covers | gap |
|---|---|---|---|
| `extra/qk_primitive_ledger.py` | yes | replay-only primitive observations from selected artifacts | narrow primitive enum; old 2026-06-19 artifact set; not current decode/prefill coverage |
| `extra/qk_lifecycle_search.py` | yes | lifecycle candidates, refutations, runner bindings | candidate memory, not full GPU primitive coverage scoring |
| `extra/qk_project_search_ledger.py` | yes | machine-search entries across decode/prefill/codegen | append-only search memory, not exhaustive space census |
| `extra/qk_oracle_gpu_primitive_explorer.py` | yes | bounded spec-driven candidate runner | executes a candidate spec; does not decide total explored space |
| `bench/qk-primitive-coverage/rows.json` | yes | 12-row 2026-06-19 coverage map | stale relative to current default-on decode/prefill state |
| `docs/gpu-lifecycle-primitive-coverage-tracker-20260624.md` | yes | manual current percentage map | not generated from artifacts yet |

Verdict: `TOOLING_PARTIAL_NOT_EXHAUSTIVE`.

## Required New Tool

Add:

`extra/qk_gpu_lifecycle_primitive_audit.py`

The tool should be read-only by default and should ingest current docs/bench artifacts into one normalized primitive-space audit.

Default output:

`bench/qk-gpu-lifecycle-primitive-audit-20260624/`

Required files:

- `authority.json`
- `primitive_taxonomy.json`
- `unknown_primitive_discovery.json`
- `artifact_inventory.json`
- `coverage_scores_by_category.json`
- `coverage_scores_by_benchmark_target.json`
- `exploration_gap_by_category.json`
- `time_correctness_confidence_by_category.json`
- `evidence_matrix.json`
- `unexplored_space.json`
- `refutation_map.json`
- `next_audit_queue.json`
- `summary.md`
- `decision.json`

## Primitive Taxonomy

The audit must start with these categories, but it must treat them as a provisional taxonomy:

| category | lifecycle boundary |
|---|---|
| `weight_gemv_matvec` | quantized weight read, dequant, activation lifecycle, dot/reduction |
| `decode_attention_tile` | QK, mask, softmax, PV, split policy, combine, GQA reuse |
| `kv_cache_read_lifecycle` | cache layout, valid-prefix read, slice/view materialization, buffer identity |
| `kv_append_write_lifecycle` | append semantics, per-token write, persistence, AFTER/read ordering |
| `smallop_lifecycle` | RMSNorm, RoPE, residual add, SiLU/mul, casts, copies |
| `launch_graph_lifecycle` | programs/token, syncs, graph reuse, dispatch overhead |
| `memory_bandwidth_layout` | effective bytes, coalescing, striding, LDS/vector loads, packed loads |
| `codegen_isa_control` | `v_dot2`, cross-lane, waitcnt, LDS, vector loads, renderer/scheduler expressibility |
| `prefill_gemm_lifecycle` | graph-GEMM/Tensile/WMMA route, LDS pipeline, occupancy, integration |
| `prefill_non_gemm_lifecycle` | prefill attention, copy/layout, prompt chunk integration |
| `harness_authority_lifecycle` | W==D, whole-prefill, llama comparator, route flags, clock/dirty metadata |
| `unknown_primitive_discovery` | unclassified timing, memory, launch, compiler, runtime, or lifecycle effects |

## Unknown Primitive Discovery

The audit must include discovery before final scoring.

Purpose:

Find primitives we do not yet understand, including primitives not represented by the current category names.

Required output:

`unknown_primitive_discovery.json`

Required fields:

- `source_artifact`
- `signal_type`
- `ctx`
- `mode`
- `role_or_kernel_name`
- `time_ms`
- `share`
- `unclassified_reason`
- `possible_taxonomy_addition`
- `next_probe`
- `priority`

Discovery sources:

- current decode role/time artifacts
- current prefill role/time artifacts
- profiler kernel names not mapped to known roles
- materialization/copy detectors
- program count and sync counters
- memory pressure and effective bandwidth estimates
- ISA/codegen audit outputs
- llama-vs-tinygrad deltas that do not map to current categories

Required rule:

Any unclassified bucket above 2% wall share, any ctx-scaled slope above measurement noise, or any repeated route/materialization anomaly must appear in `unknown_primitive_discovery.json`.

If no such signal exists, the audit must emit an explicit empty result with the thresholds and sources checked.

## Two-Axis Scoring Model

Use two separate measures for every primitive category and benchmark target:

- `exploration_gap_percent`: how much relevant search/audit space remains unexplored.
- `time_correctness_confidence_percent`: how confident we are that the current timing and correctness behavior is real.

Do not collapse these too early. A primitive can be heavily explored but weakly measured, or strongly measured while still leaving a large design space open.

Examples:

| case | exploration gap | time/correctness confidence | meaning |
|---|---:|---:|---|
| prefill GEMM current 8B | low | high | many variants tested, current path strong |
| small-op lifecycle | high | medium/low | plausible residual, not enough split data |
| decode KV identity | medium/low | high | core win proven, but needs regression guard |
| native-codegen portability | high | medium | owned kernels prove target, compiler generality open |

### Exploration Gap Score

`exploration_gap_percent` starts at 100 and decreases as the relevant space is covered.

| component | closes up to | required evidence |
|---|---:|---|
| taxonomy coverage | 15 | primitive boundary named and scoped |
| candidate/variant coverage | 20 | variants searched, bounded knobs listed, or explicit non-applicable proof |
| lifecycle coverage | 20 | producer, format, consumer, routing, and fallback included |
| refutation coverage | 15 | failed paths recorded with reasons |
| cross-ctx/role coverage | 15 | contexts and model roles included |
| currentness coverage | 15 | current artifact or explicit stale caveat |
| unknown-discovery coverage | 10 | unclassified buckets searched and either mapped or emitted as unknowns |

Formula:

`exploration_gap_percent = 100 - min(100, exploration_closed_percent)`

### Time + Correctness Confidence Score

`time_correctness_confidence_percent` answers: "Do we trust the current benchmark/correctness result?"

| component | max points | required evidence |
|---|---:|---|
| authority benchmark | 25 | W==D, whole-prefill, or explicit non-promotion microprimitive |
| correctness/quality gate | 20 | byte-identical, dNLL, RMSE, or exact non-promotion statement |
| route/materialization proof | 15 | route identity, fallback, materialization/ABI check |
| repeat/spread/clock quality | 15 | repeats, spread, clock/runtime metadata |
| comparator quality | 15 | llama/current baseline source with matching context where possible |
| regression guard | 10 | runnable guard or durable invariant |

Formula:

`time_correctness_confidence_percent = sum(component_points)`

### Effective Explored Score

If one table needs a single number, use:

`effective_explored_percent = round((100 - exploration_gap_percent) * time_correctness_confidence_percent / 100)`

This is intentionally conservative.

## Legacy Seed Scoring Model

Each category gets a percentage from 0 to 100.

The score must be decomposed, not hand-waved:

| component | max points | required evidence |
|---|---:|---|
| taxonomy coverage | 15 | primitive boundary named and scoped |
| artifact coverage | 15 | benchmark/doc artifacts found and linked |
| correctness/quality gate | 15 | byte-identical, dNLL, RMSE, or explicit non-promotion |
| route/materialization proof | 15 | route identity, fallback, materialization/ABI check |
| authority benchmark | 20 | W==D, whole-prefill, or documented non-promotion microprimitive |
| refutation/search breadth | 10 | searched variants, rejected candidates, or reason search is not applicable |
| currentness/regression guard | 10 | current artifact, guard, or explicit stale caveat |

This older single score is retained only as a seed for the first implementation.

Legacy score:

`explored_percent = sum(component_points)`

The tool must include `score_reasons`, `missing_points`, and `blocking_artifacts` for every category, plus both new axes.

## Benchmark Target Scores

The tool must also produce target-level scores:

| target | score inputs |
|---|---|
| `8b_decode_speed_vs_llama` | current decode benchmark, llama comparator, GEMV/attention/KV route coverage |
| `8b_decode_ctx_slope` | ctx ladder plus attention/KV/small-op split |
| `8b_prefill_speed_vs_llama` | corrected prefill ladder plus llama prefill reference/ladder |
| `8b_prefill_long_context_stability` | full-lattice prefill chunk ladder, runtime split, memory watch |
| `machine_search_readiness_current_8b` | search runners, reject rules, candidate schema, bounded spec availability |
| `native_codegen_portability` | ISA audit, native-codegen microsearch, owned-kernel translation targets |
| `serving_runtime_kv_lifecycle` | runtime-KV feasibility, append/persistence, serving workload definition |

## Initial Expected Scores

Seed the tool from the current manual tracker:

| category | starting explored % |
|---|---:|
| `prefill_gemm_lifecycle` | 90 |
| `weight_gemv_matvec` | 85 |
| `kv_cache_read_lifecycle` | 80 |
| `decode_attention_tile` | 75 |
| `launch_graph_lifecycle` | 70 |
| `harness_authority_lifecycle` | 70 |
| `prefill_non_gemm_lifecycle` | 65 |
| `memory_bandwidth_layout` | 60 |
| `codegen_isa_control` | 55 |
| `kv_append_write_lifecycle` | 45 |
| `smallop_lifecycle` | 35 |

The first implementation may copy these seed values only if it records them as `manual_seed` and marks missing artifact-backed component scoring as a gap.

## First Concrete Audit: Decode Ctx-Slope

Before broad search, run:

`docs/decode-ctx-slope-lifecycle-primitive-audit-scope-20260624.md`

Reason:

- decode is ahead of llama, but margin narrows from ~105.0% to ~101.7%;
- this is the highest-value currently unexplained primitive surface;
- it can raise coverage for `decode_attention_tile`, `kv_cache_read_lifecycle`, `smallop_lifecycle`, `memory_bandwidth_layout`, and `harness_authority_lifecycle`.

Required outputs:

- `llama_vs_tinygrad_decode_by_ctx.json`
- `decode_role_time_by_ctx.json`
- `attention_qk_pv_softmax_split_by_ctx.json`
- `kv_identity_materialization_by_ctx.json`
- `q4k_route_coverage_by_role.json`
- `programs_and_syncs_by_ctx.json`
- `smallop_residual_census.json`
- `decision.json`

## Stop Rules

- Do not start broad decode search while decode remains above llama unless the ctx-slope audit names a bounded primitive with measured wall share.
- Do not reopen prefill GEMM search while corrected prefill remains flat and ahead of the recorded llama reference.
- Do not count a category above 85% unless it has a current authority artifact and a regression guard.
- Do not count a category at 100% unless the audit names the remaining unexplored space as non-applicable, refuted, or out-of-scope with artifact evidence.
- Do not count the whole primitive map at 100% unless `unknown_primitive_discovery.json` is empty under explicit thresholds, or every unknown candidate has been promoted into the taxonomy.

## Definition Of Done

The exhaustive audit is complete when:

- every taxonomy row has a score with component breakdown;
- every score has artifact evidence and missing-point rationale;
- every benchmark target has a score;
- unknown primitive discovery has run and produced either classified additions or an explicit empty result;
- unexplored areas are ranked by expected value;
- the next audit/search action is selected by evidence, not intuition.
