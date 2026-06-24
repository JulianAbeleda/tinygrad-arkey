# North-Star Evaluator-Binding Templates v0 — Result

Date: 2026-06-21

Scope: define + implement the evaluator-binding layer that turns `gen_north_star_flash_attn_tile` from a vague
`PRUNE_NEEDS_TEMPLATE` placeholder into a **well-specified candidate blocked only on a concrete kernel/runner**.
Contract + harness shape only — **no kernel built**.

## Final decision: **`NORTH_STAR_BINDING_TEMPLATE_READY`**

A binding-template schema + the `north_star_flash_attn_tile_v0` binding now specify exactly what an executable
north-star candidate must declare, run, and produce (vs `gqa_coop_vec`, never a weak baseline). The generator stamps
those requirements onto `gen_north_star_flash_attn_tile`, and the loop resolves binding metadata into a precise
verdict — distinguishing **missing template / present-but-no-runner / executable**. An executable plumbing selftest
proves the binding→candidate→`decode_eval`→artifact path works end-to-end with **no performance claim**. The
north-star is now blocked solely on a concrete kernel + runner (the next project).

## Phase 0 — binding audit (existing executable candidates)

| candidate | decode_eval id | local A/B runner | W==D runner | correctness | verdict → lifecycle |
|---|---|---|---|---|---|
| `baseline_default` | baseline_default | none | `runtime_overhead` (wd ×5, no env, auto clock) | none | REST → bank_baseline_or_rest |
| `flash_l_64` | flash_l_64 | `flash_l_local` (L64 vs L128, clock-pinned, byte-exact vs numpy) | `runtime_overhead` (FLASH_L=64 env ×3 + baseline) | byte_exact (max_err ≤ 0.02) | LOCAL_PASS_WD_FAIL → refute_for_promotion_bank_learning |
| `q8_opt_in` | q8_opt_in | none | `q8_audit` (baseline-vs-q8, clock-controlled manual_peak lane) | dNLL historical ≤ 0.01 | PASS_OPT_IN → opt_in_candidate_banked |
| `warp_flash_tile` | warp_flash_tile | `ab_script` (warp tile vs gqa_coop_vec, `first_gate_pass`) | none | byte_exact (tile err) | FAIL_LOCAL_AB → refute_candidate |

**Fields a future north-star candidate needs that none of these encode:**
- an **explicit comparator** (existing candidates compare to `gqa_coop_vec` implicitly inside their local A/B; none
  declare it as a field — the north-star must, and it must never be a weak baseline);
- **decode T=1 parallelism artifact fields** — `workgroups_by_ctx`, `kv_splits_by_ctx`, `query_heads_parallelized`,
  `combine_kernel_count` — none of the existing candidates capture these; the north-star **must** prove workgroups
  grow with ctx and don't collapse vs the comparator;
- a **new local-A/B runner** for the tile primitive (`flash_l_local` is for the existing variant) and a
  **whole-decode W==D route/flag** for the tile (flash_l reuses `FLASH_L`, q8 reuses `Q8_FFN_HANDWRITTEN`; the tile
  has neither yet);
- an **`expected_no_wmma`** assertion (llama decode attention is non-WMMA vector; WMMA decode stays closed).

The binding template adds exactly these.

## Phase 1 — binding schema + north-star binding

- `bench/qk-decode-eval/binding_template_schema.json` (`decode_evaluator_binding_templates_v1`).
- `bench/qk-decode-eval/binding_templates.json`:
  - **`north_star_flash_attn_tile_v0`** (`concrete_runner_status: deferred_no_kernel`) — role `decode_attention`;
    comparator `gqa_coop_vec`; `local_ab_runner` / `wd_runner` / `correctness_runner` named **[NOT YET
    IMPLEMENTED]**; `required_candidate_params` (kv_split_count, flash_l, query_head_pack, gqa_grouping, k_tile,
    v_tile, **combine_strategy = stream-k**, softmax_strategy, lds_bytes_per_workgroup, expected_workgroups_by_ctx,
    **expected_no_wmma: true**); `required_artifact_fields` (workgroups_by_ctx, kv_splits_by_ctx,
    query_heads_parallelized, combine_kernel_count, local/comparator attention us, W==D tok/s, correctness error,
    repro band); 5 **gates** (structural T=1 parallelism / correctness / local A/B ≥1.05× vs comparator / W==D
    ≥5%@1024 / promotion-policy owner-only); 7 **stop conditions** (collapses workgroups, benchmarks a weak
    baseline, fails correctness, local ≤ comparator, W==D < 5%, long-ctx regression, missing artifact field);
    `missing_for_executable` = [kernel, local_ab runner, W==D route].
  - **`north_star_binding_selftest_v0`** (`selftest_stub`) — executable plumbing, no perf claim.

## Phase 2/3 — generated candidate + loop resolution

The generator (`qk_candidate_template_gen.py`) now expands the north-star template into **3** candidates that
exercise the binding layer; the loop (`qk_lifecycle_search_loop.py`) resolves `binding_template_id` (a binding
block added to `prune_decision`, plus a `--candidates`-loaded `binding_templates.json`):

| generated candidate | binding_template_id | decode_eval binding | loop verdict |
|---|---|---|---|
| `gen_north_star_flash_attn_tile` | north_star_flash_attn_tile_v0 (exists) | none | **`PRUNE_NEEDS_TEMPLATE`** — "binding exists (status=deferred_no_kernel) but no concrete runner; missing: kernel, local_ab runner, W==D route" |
| `gen_north_star_binding_selftest` | north_star_binding_selftest_v0 (exists) | north_star_binding_selftest | **EXECUTE → `SELFTEST_PASS`** → selftest_only_not_perf (no GPU, no perf claim) |
| `gen_north_star_missing_binding` | north_star_flash_attn_tile_vX_does_not_exist | none | **`PRUNE_MISSING_EVALUATOR_BINDING`** — "binding template not found" |

`gen_north_star_flash_attn_tile` now carries `binding_template_id`, `required_params`, `comparator`,
`expected_first_real_gate`, `expected_stop_conditions`, `missing_for_executable`, `maps_to_north_star: true`,
`executable_status: deferred_no_kernel` — it is no longer a vague idea.

`decode_eval` gained a no-GPU `binding_selftest` rung + candidate (`north_star_binding_selftest`) that returns
`SELFTEST_PASS` (its existing verdict enum); it builds/changes no model code, route, or default.

## Phase 4/5 — generated-suite validation

Generated suite = **11 candidates**: 4 executable (gen_flash_l_64, gen_flash_l_128, gen_q8_opt_in,
gen_north_star_binding_selftest) + 7 pruned/deferred. Result (`bench/qk-lifecycle-search/runs/decode_v0-*.json`):

| candidate | loop decision | verdict | match |
|---|---|---|---|
| gen_flash_l_128 | EXECUTE | REST | ✓ |
| gen_flash_l_64 | EXECUTE | LOCAL_PASS_WD_FAIL | ✓ |
| gen_q8_opt_in | EXECUTE | PASS_OPT_IN | ✓ |
| gen_north_star_binding_selftest | EXECUTE | **SELFTEST_PASS** (no perf) | ✓ |
| gen_promote_flash_l_64 | PRUNE_POLICY_VIOLATION | — | ✓ |
| gen_q8_default_attempt | PRUNE_POLICY_VIOLATION | — | ✓ |
| gen_wmma_decode_reopen | PRUNE_CLOSED_LANE | — | ✓ |
| gen_mmvq_reopen | PRUNE_CLOSED_LANE | — | ✓ |
| gen_bounded_fusion_reopen | PRUNE_CLOSED_LANE | — | ✓ |
| gen_north_star_flash_attn_tile | **PRUNE_NEEDS_TEMPLATE** (precise) | — | ✓ |
| gen_north_star_missing_binding | **PRUNE_MISSING_EVALUATOR_BINDING** | — | ✓ |

No closed lane benchmarked; **no north-star performance claim** (the placeholder is pruned, the selftest is
explicitly non-performance). Existing candidates behave as before.

## Validation commands

```bash
python3 -c "import json,jsonschema; jsonschema.validate(json.load(open('bench/qk-decode-eval/binding_templates.json')), json.load(open('bench/qk-decode-eval/binding_template_schema.json')))"   # G1/G2
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_candidate_template_gen.py --suite decode_template_v0 \
    --out bench/qk-lifecycle-search/generated/ --emit-search-candidates bench/qk-lifecycle-search/search_candidates.generated.json
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_lifecycle_search_loop.py \
    --candidates bench/qk-lifecycle-search/search_candidates.generated.json --dry-run --suite decode_template_v0   # G4
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_lifecycle_search_loop.py \
    --candidates bench/qk-lifecycle-search/search_candidates.generated.json --suite decode_template_v0 --repeats 2 \
    --out bench/qk-lifecycle-search/runs/                                                                          # G5-G7
```

## Acceptance gates

| gate | result |
|---|---|
| G1 binding-template schema exists | PASS |
| G2 north-star binding template validates | PASS |
| G3 gen_north_star_flash_attn_tile has binding_template_id + required params | PASS |
| G4 loop distinguishes missing template / no-runner / executable | PASS (3 cases) |
| G5 generated suite still passes | PASS (all 11 match) |
| G6 closed lanes still pruned before eval | PASS |
| G7 no north-star placeholder benchmarked as real | PASS (pruned; selftest is non-perf) |
| G8 policy guard passes | PASS |
| G9 no tinygrad/ exec code / defaults / kernels / route defaults changed | PASS (`git diff tinygrad/` empty) |
| G10 tree clean after commit | PASS (commit below) |

## Falsifiers checked (none tripped)

- the search/candidate schema represented binding metadata cleanly (extra fields) → not
  `NEEDS_SEARCH_SCHEMA_NORMALIZATION`.
- `decode_eval` supported the selftest binding with a small no-GPU rung (no rewrite) → not `NEEDS_EVALUATOR_API_CLEANUP`.
- the north-star binding requirements were unambiguous after the llama audit (comparator, T=1 fields, runners,
  no-WMMA) → not `NEEDS_BINDING_REQUIREMENTS_AUDIT`.
- the binding is fully specified; only a concrete kernel + runner remain → not `BLOCKED_ON_KERNEL_IMPLEMENTATION`
  *for this project* (that IS the next project).

## Next unlocked project

**Build the concrete `flash_attn_tile` decode kernel + its `decode_eval` runners** (`local_ab` vs `gqa_coop_vec` +
a whole-decode W==D route), satisfying `north_star_flash_attn_tile_v0`. That flips `gen_north_star_flash_attn_tile`
from `PRUNE_NEEDS_TEMPLATE` to an executable candidate measured against the current winner — the first real
north-star performance attempt, gated by this binding contract.

## Limitations

- No kernel built (by design); the north-star stays deferred, now precisely.
- The binding's `local_ab_runner` / `wd_runner` are named contracts, not implementations.
- The selftest proves plumbing only (SELFTEST_PASS), not attention performance.

## Boundary

No `tinygrad/` change, no model route/default, no kernel, no WMMA/MMVQ path, no tuning sweep, no perf claim. New
files under `bench/qk-decode-eval/binding_template*.json` + the generator/loop/decode_eval wiring.
