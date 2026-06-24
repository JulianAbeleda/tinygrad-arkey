# Candidate-Template Generation Layer v0 — Result

Date: 2026-06-21

Scope: add the missing "generate" step on top of the working lifecycle-search loop
(`docs/lifecycle-search-loop-v0-result-20260621.md`).

## Final decision: **`TEMPLATE_GENERATION_V0_READY`**

A narrow, deterministic generator expands route/fusion/layout **templates** into legal decode candidate **specs**
(in the existing `search_candidates` schema), each carrying full policy metadata, and those generated specs go
**through the loop** unchanged: executable variants run via `decode_eval`, invalid variants are pruned before any
benchmark, and the north-star lever is represented as a deferred (never-benchmarked) candidate. The generator emits
no kernel code, no new flags, and changes no defaults. The key bar is met — **generated candidates flow through the
loop**, not into prose or another static ledger.

## Phase 0 — design note

- **Why generation v0 is template-based, not open-ended search.** v0 proves the *generate step* and that generated
  specs carry enough policy metadata to be classified by the loop. It expands a **small, fixed, explicit** set from
  4 templates — not a parameter explosion and not kernel-code synthesis (no kernels). Open-ended search is the
  future; v0 is the candidate *factory* plumbing.
- **Why generated candidates must carry policy metadata.** The loop prunes/defers on metadata
  (`closed_lane_risk`, `intent`, `deferred`, `decode_eval_candidate_id`, `maps_to_ledger_candidate`). A generated
  candidate without it can't be classified; with it, an *invalid* generated candidate is refused **before** a
  benchmark — the same guarantee the hand-written candidates had.
- **What counts as generation in v0.** Expanding `templates.json` into candidate specs: executable variants **bind
  to existing `decode_eval` candidates** (baseline_default / flash_l_64 / q8_opt_in); invalid/deferred variants
  carry the prune/defer metadata. Output is deterministic (no time/random; date stamped; git provenance only).
- **Intentionally excluded.** Kernel codegen, new flags, broad sweeps, auto-discovery of *new* evaluator candidates,
  ledger auto-mutation. The generator writes specs; it never benchmarks or mutates the ledger.
- **How this advances llama-style + vLLM-style.** vLLM-style: the search system now has a real
  generate→evaluate→prune cycle fed by a template factory. llama-style: the remaining quality lever (the full
  llama-style vector `flash_attn_tile`) is a first-class **deferred** candidate — the system records the north-star
  and can slot it in the moment a kernel + evaluator binding exist, instead of pretending it is runnable.

## What was built

- `extra/qk_candidate_template_gen.py` — the generator. CLI: `--list-templates` · `--template <id> --dry-run` ·
  `--suite decode_template_v0 --out DIR [--emit-search-candidates PATH]` · `--validate <file>`. Per-template
  deterministic expanders; validates every generated candidate has the required `search_candidates` fields; emits a
  generation artifact (with provenance) and a loop-consumable registry.
- `bench/qk-lifecycle-search/templates.json` (4 templates, schema `decode_candidate_templates_v1`) +
  `template_schema.json`.
- `extra/qk_lifecycle_search_loop.py` — **two small additions** (no rewrite): `--candidates <path>` (consume a
  custom/generated registry) and a `deferred: true` → `PRUNE_NEEDS_TEMPLATE` branch (for the north-star placeholder).

## Templates added

| template | family | expands to |
|---|---|---|
| `decode_flash_l_sweep` | attention_split | `gen_flash_l_128` (exec→baseline), `gen_flash_l_64` (exec), `gen_promote_flash_l_64` (invalid) |
| `q8_opt_in_policy` | q8_route | `gen_q8_opt_in` (exec), `gen_q8_default_attempt` (invalid) |
| `closed_lane_reopen_attempts` | closed_lane_probe | `gen_wmma_decode_reopen`, `gen_mmvq_reopen`, `gen_bounded_fusion_reopen` (all invalid) |
| `north_star_flash_attn_tile_placeholder` | north_star_deferred | `gen_north_star_flash_attn_tile` (deferred) |

## Generated suite result (`bench/qk-lifecycle-search/runs/decode_v0-*.json`, from the generated registry)

| generated candidate | loop decision | decode_eval verdict | lifecycle decision | expected | match |
|---|---|---|---|---|---|
| gen_flash_l_128 | EXECUTE | REST | bank_baseline_or_rest | bank_baseline_or_rest | ✓ |
| gen_flash_l_64 | EXECUTE | LOCAL_PASS_WD_FAIL | refute_for_promotion_bank_learning | refute_for_promotion_bank_learning | ✓ |
| gen_q8_opt_in | EXECUTE | PASS_OPT_IN | opt_in_candidate_banked | opt_in_candidate_banked | ✓ |
| gen_promote_flash_l_64 | **PRUNE_POLICY_VIOLATION** | — not benchmarked | — | PRUNE_POLICY_VIOLATION | ✓ |
| gen_q8_default_attempt | **PRUNE_POLICY_VIOLATION** | — not benchmarked | — | PRUNE_POLICY_VIOLATION | ✓ |
| gen_wmma_decode_reopen | **PRUNE_CLOSED_LANE** | — not benchmarked | — | PRUNE_CLOSED_LANE | ✓ |
| gen_mmvq_reopen | **PRUNE_CLOSED_LANE** | — not benchmarked | — | PRUNE_CLOSED_LANE | ✓ |
| gen_bounded_fusion_reopen | **PRUNE_CLOSED_LANE** | — not benchmarked | — | PRUNE_CLOSED_LANE | ✓ |
| gen_north_star_flash_attn_tile | **PRUNE_NEEDS_TEMPLATE** (deferred) | — not benchmarked | — | PRUNE_NEEDS_TEMPLATE | ✓ |

**3 executable (verdicts match, artifacts valid) + 6 pruned/deferred (never benchmarked).** No default changed.

## Integration

A custom-registry path was needed and added (`--candidates`, one line; no loop rewrite). Command:

```bash
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_candidate_template_gen.py --suite decode_template_v0 \
    --out bench/qk-lifecycle-search/generated/ --emit-search-candidates bench/qk-lifecycle-search/search_candidates.generated.json
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_lifecycle_search_loop.py \
    --candidates bench/qk-lifecycle-search/search_candidates.generated.json --suite decode_template_v0 --repeats 2 \
    --out bench/qk-lifecycle-search/runs/
```

## Acceptance gates

| gate | result |
|---|---|
| G1 `--list-templates` | PASS |
| G2 template schema validates | PASS |
| G3 deterministic candidate registry | PASS (identical modulo git provenance) |
| G4 generated registry validates | PASS (9 candidates) |
| G5 loop dry-run consumes generated registry | PASS |
| G6 loop run consumes generated registry, executes only valid | PASS |
| G7 ≥2 generated executable candidates run | PASS (3) |
| G8 ≥4 generated invalid/deferred pruned without eval | PASS (6) |
| G9 expected classifications match | PASS (all 9) |
| G10 policy guard passes | PASS |
| G11 no tinygrad/ execution code, defaults, kernels, route defaults changed | PASS (`git diff tinygrad/` empty) |
| G12 tree clean after commit | PASS (commit below) |

## Falsifiers checked (none tripped)

- generated candidates ARE representable in the current search schema (9 generated, all loop-consumable) → not
  `NEEDS_SEARCH_SCHEMA_NORMALIZATION`.
- the loop consumed a generated registry with a one-line `--candidates` add → not
  `NEEDS_LIFECYCLE_SEARCH_API_CLEANUP`.
- policy metadata classified all 9 cleanly (no ambiguity) → not `NEEDS_POLICY_SCHEMA_CLEANUP`.
- executable candidates ran through existing `decode_eval` bindings (no bespoke glue) → not
  `NEEDS_EVALUATOR_BINDING_TEMPLATES`.

## Next unlocked project

**Evaluator-binding templates for the north-star** — give `gen_north_star_flash_attn_tile` a real `decode_eval`
binding (a `flash_attn_tile` local-A/B + W==D runner), turning the deferred placeholder into an executable
candidate. That is where llama-style primitive work (many-KV-split / stream-k combine) re-enters via the search
system rather than ad hoc.

## Limitations

- Executable generated candidates re-point at the **3 existing** `decode_eval` candidates (v0 generates legal specs
  + policy metadata, not new evaluator candidates / kernels).
- Templates are a small fixed set; broad parameter search and auto-generated *new* evaluator bindings are future.
- Generator is propose-only into the search pipeline; it does not mutate the canonical `search_candidates.json` or
  the ledger.

## Boundary

No `tinygrad/` change, no defaults/flags/kernels, no closed-lane benchmarking (closed lanes pruned before eval),
q8/FLASH_L not promoted. Generated registry + artifacts under `bench/qk-lifecycle-search/{generated/,
search_candidates.generated.json}`.
