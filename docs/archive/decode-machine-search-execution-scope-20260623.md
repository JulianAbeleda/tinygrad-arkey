# Decode Machine Search Execution — Exhaustive Scope / Claude Prompt (2026-06-23)

## Mission

Run a real, bounded decode machine-search campaign using the existing search-readiness package and the project harness SOP.

This is **not** a new audit and not a broad/random kernel generator. It consumes the already-built readiness package:

- `docs/decode-machine-search-readiness-package-result-20260623.md`
- `extra/qk_decode_search_gate.py`
- `extra/qk_decode_route_fire_check.py`
- `extra/qk_decode_materialization_check.py`
- `extra/qk_decode_search_runner.py`
- `bench/qk-decode-search-readiness/`

Goal:

```text
generate bounded decode candidates -> evaluate with cost-ordered gates -> prune -> rank -> remember
```

Primary use:

- regression-safe variant exploration;
- native-codegen microprimitive exploration;
- future cross-shape/model/GPU portability.

Non-goal:

- do not chase Qwen3-8B speed blindly. The current default is already at/above llama.cpp.

## Harness Authority Requirement

Before touching the runner, read:

- `bench/qk-decode-eval/HARNESS_GUIDE.md`
- `extra/qk_harness_contract.py`
- `extra/qk_decode_eval.py`
- `structure/Development/performance-primitive-research-principles.md`

This search must obey the Measurement Authority table:

| authority | allowed use in this search |
|---|---|
| clean W==D, `PROFILE=0`, synced, repeated | promotion/ranking authority |
| PROFILE/GPU timestamps | attribution only |
| DEBUG/stdout timing | debugging only |
| raw-dispatch/no-sync timing | diagnostic only |
| local kernel timing | early diagnostic only, never promotion |

Hard rule:

```text
No candidate can be called a win unless it passes token correctness and clean W==D against the frozen oracle.
```

Use the 13-field artifact contract from `HARNESS_GUIDE.md`:

1. workload shape and context;
2. candidate id and primitive class;
3. comparator id and why current;
4. exact command/env;
5. git commit and dirty status;
6. hardware and clock/perf state;
7. warmup/compile handling;
8. repeats, median, spread, reproducibility band;
9. correctness/quality gate;
10. local diagnostic vs W==D authority;
11. pass/fail threshold;
12. final verdict and stop reason;
13. ledger/refutation links.

Every result artifact must be stamped with `extra.qk_harness_contract.stamp()`.

## Required Reading

Read first:

1. `docs/decode-machine-search-readiness-package-result-20260623.md`
2. `docs/decode-machine-search-readiness-package-scope-20260623.md`
3. `bench/qk-decode-eval/HARNESS_GUIDE.md`
4. `docs/decode-campaign-final-synthesis-20260623.md`
5. `docs/machine-code-translation-roadmap-result-20260623.md`
6. `docs/owned-tile-buffer-identity-kv-read-result-20260623.md`
7. `docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md`
8. `docs/amd-gpu-holistic-primitive-model-20260623.md`
9. `structure/Development/performance-primitive-research-principles.md`
10. `structure/Development/session-handoff.md`

Inspect:

- `extra/qk_decode_search_runner.py`
- `extra/qk_decode_search_gate.py`
- `extra/qk_decode_route_fire_check.py`
- `extra/qk_decode_materialization_check.py`
- `extra/qk_isa_primitive_audit.py`
- `extra/qk_decode_runtime_overhead.py`
- `bench/qk-decode-search-readiness/baseline_oracle.json`
- `bench/qk-decode-search-readiness/candidate_schema.json`
- `bench/qk-decode-search-readiness/result_schema.json`
- `bench/qk-decode-search-readiness/reject_rules.json`
- `bench/qk-decode-eval/candidates.json`

## Boundaries

- Do not flip defaults.
- Do not change the oracle.
- Do not change decode behavior outside candidate flags/generated candidate artifacts.
- Do not touch prefill.
- Do not do 14B/32B in the first search run.
- Do not broaden the search space beyond the schema without a scope update.
- Do not promote from local timing, PROFILE timing, DEBUG timing, or no-sync timing.
- Do not keep tuning a rejected candidate outside the lifecycle loop.
- Do not accept candidates that reintroduce `E_49152` or sliced-view inputs.

## Required Artifact Directory

```text
bench/qk-decode-machine-search/
```

Required artifacts:

- `authority.json`
- `oracle_recheck.json`
- `search_plan.json`
- `candidate_manifest.json`
- `results.jsonl`
- `leaderboard.json`
- `reject_summary.json`
- `winner_recheck.json` if a winner exists
- `decision.json`

Required result doc:

- `docs/decode-machine-search-execution-result-20260623.md`

## Search Modes

The first execution must choose one mode and record it.

### Mode A — Policy Search

Lowest risk. Search env/policy knobs only.

Allowed knobs:

- split `S` from existing compiled/kernel support;
- `DECODE_ATTN_AMDGCN_MIN_CTX`;
- combine variant if registry still supports it;
- route/fallback thresholds;
- Q4K warp flag state only if explicitly comparing stack interactions.

Disallowed:

- new HIP kernel generation;
- tile constant changes;
- workgroup shape changes.

Recommended first run.

### Mode B — Generated Owned-Tile Variant Search

Higher risk. Only after Mode A works.

Allowed knobs:

- `S`;
- `TK`;
- workgroup size;
- vector load width;
- unroll;
- combine variant;
- resource envelope.

Requires:

- candidate code object hash;
- ISA audit JSON;
- correctness smoke before W==D;
- no source/default changes outside generated candidate cache/artifacts.

### Mode C — Native-Codegen Microprimitive Search

Not W==D decode speed search.

Allowed:

- tinygrad-native microkernels for:
  - `v_dot2` lowering;
  - LDS staging;
  - cross-lane reductions;
  - vector loads.

Requires:

- local correctness;
- ISA audit;
- no decode promotion claim.

## Phase 0 — Authority + Oracle Recheck

Run before candidate evaluation.

Requirements:

- git HEAD/status;
- GPU/arch;
- current default flags;
- load frozen oracle;
- rerun cheap oracle recheck:
  - token correctness;
  - route fires `owned_flash_tile_gqa_whole`;
  - `E_49152` absent;
  - ISA audit still passes or current JSON matches expected;
  - W==D ctx512 and ctx1024 within frozen reproducibility band.

Artifacts:

- `bench/qk-decode-machine-search/authority.json`
- `bench/qk-decode-machine-search/oracle_recheck.json`

Verdicts:

- `SEARCH_ORACLE_RECHECK_PASS`
- `SEARCH_ORACLE_DRIFT_STOP`

Stop if oracle drifts.

## Phase 1 — Search Plan

Write a machine-readable plan before evaluating candidates.

Required fields:

- search mode;
- candidate count;
- knob ranges;
- comparator/oracle id;
- contexts to test;
- early gates;
- W==D gates;
- pass thresholds;
- reject rules;
- max runtime budget;
- whether generated code objects are allowed.

Artifact:

- `bench/qk-decode-machine-search/search_plan.json`

Verdicts:

- `SEARCH_PLAN_READY`
- `SEARCH_PLAN_TOO_BROAD_STOP`

Initial recommended plan:

```text
Mode A policy search, small grid only:
S in {32,48,64,96}
min_ctx in {256,512,1024}
combine in currently registered safe variants
```

If the current default is already best within spread, that is a valid result.

## Phase 2 — Candidate Manifest

Generate candidate IDs before running.

Required candidate fields:

- id;
- knobs;
- env;
- expected kernel symbol;
- expected route signature;
- expected ISA requirements;
- expected materialization requirements;
- comparator: frozen oracle;
- reason included in search.

Artifact:

- `bench/qk-decode-machine-search/candidate_manifest.json`

Verdicts:

- `CANDIDATE_MANIFEST_READY`
- `CANDIDATE_MANIFEST_INVALID_STOP`

## Phase 3 — Cost-Ordered Evaluation

For each candidate, apply gates in order. Stop at first reject.

### Gate 1 — Structural / Policy Gate

Reject if:

- knob outside schema;
- unsupported shape;
- missing candidate metadata;
- violates buffer-identity invariant by design.

### Gate 2 — Correctness Smoke

Reject if:

- greedy token mismatch on canonical short prompt;
- exception/fallback not expected.

### Gate 3 — Route-Fire Check

Reject if:

- candidate route absent;
- wrong kernel fires;
- slice route fires when whole-cache expected.

### Gate 4 — Materialization Check

Reject if:

- `E_49152` returns;
- full-MAXC K/V copy appears;
- sliced-view input detected;
- buffer identity lost.

### Gate 5 — ISA Audit

Reject if:

- no ISA JSON;
- `v_dot2` missing;
- LDS missing;
- cross-lane missing;
- scratch/spill appears;
- VGPR exceeds envelope;
- unexpected code object mismatch.

### Gate 6 — Full Correctness

Reject if:

- 64-token two-prompt byte-identical check fails;
- ctx512 correctness fails;
- fallback changes tokens.

### Gate 7 — W==D Authority

Only now run clean synced W==D.

Required:

- `PROFILE=0`;
- `.item()`/sync included according to canonical harness;
- repeats and spread;
- ctx512 and ctx1024 for first pass;
- ctx2048/4096 only for candidates that pass first pass;
- compare against oracle and spread band.

Artifacts:

- append one result per candidate to `bench/qk-decode-machine-search/results.jsonl`

Verdicts per candidate:

- `CANDIDATE_PASS`
- `REJECT_STRUCTURAL`
- `REJECT_CORRECTNESS`
- `REJECT_ROUTE_NOT_FIRING`
- `REJECT_MATERIALIZATION`
- `REJECT_ISA`
- `REJECT_WD_REGRESSION`
- `REJECT_WD_NO_TRANSFER`

## Phase 4 — Ranking / Leaderboard

Rank only candidates that pass all gates.

Primary metric:

- W==D median tok/s delta vs oracle across required contexts.

Secondary metrics:

- worst-context regression;
- spread-adjusted delta;
- ctx512 safety;
- ISA/resource quality;
- simplicity / knob distance from default.

Artifact:

- `bench/qk-decode-machine-search/leaderboard.json`

Verdicts:

- `SEARCH_LEADERBOARD_READY`
- `SEARCH_NO_PASSING_CANDIDATES`
- `SEARCH_ORACLE_REMAINS_BEST`

## Phase 5 — Winner Recheck

Only if a candidate beats oracle outside spread.

Re-run:

- full W==D ctx512/1024/2048/4096;
- 3+ repeats;
- token correctness;
- route/materialization/ISA checks;
- fallback sanity.

Artifact:

- `bench/qk-decode-machine-search/winner_recheck.json`

Verdicts:

- `WINNER_RECHECK_PASS`
- `WINNER_RECHECK_FAIL`
- `NO_WINNER_RECHECK_NEEDED`

Do not flip default. A passing winner can only produce a recommendation.

## Phase 6 — Decision

Write:

- `bench/qk-decode-machine-search/decision.json`
- `docs/decode-machine-search-execution-result-20260623.md`

Required result doc sections:

1. Verdict.
2. Harness authority compliance.
3. Oracle recheck.
4. Search mode and plan.
5. Candidate manifest summary.
6. Reject summary.
7. Leaderboard.
8. Winner recheck if any.
9. Recommendation.
10. Files changed.
11. Git status.

Allowed final verdicts:

- `DECODE_SEARCH_EXECUTED_ORACLE_REMAINS_BEST`
- `DECODE_SEARCH_EXECUTED_WINNER_FOUND_RECOMMEND_ONLY`
- `DECODE_SEARCH_EXECUTED_NO_PASSING_CANDIDATES`
- `DECODE_SEARCH_BLOCKED_ORACLE_DRIFT`
- `DECODE_SEARCH_BLOCKED_HARNESS_NONCOMPLIANT`
- `DECODE_SEARCH_BLOCKED_SCOPE_TOO_BROAD`

## Harness Compliance Checklist

The result doc must explicitly answer:

| checklist | answer |
|---|---|
| Did every performance claim use clean W==D authority? | |
| Were PROFILE/DEBUG/no-sync timings excluded from promotion? | |
| Was correctness checked before speed? | |
| Were repeats/spread recorded? | |
| Was the oracle comparator current? | |
| Was git/dirty state stamped? | |
| Were artifacts stamped with `qk_harness_contract`? | |
| Were local diagnostics separated from W==D? | |
| Were rejected candidates stopped at first failed gate? | |

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

The owner wants decode to become pure machine-search, using the existing readiness package and the harness SOP.

Read and execute:

```text
docs/decode-machine-search-execution-scope-20260623.md
bench/qk-decode-eval/HARNESS_GUIDE.md
docs/decode-machine-search-readiness-package-result-20260623.md
```

Do not run broad/random search. Run only a bounded search mode from the scope. Recommended first run is Mode A policy
search with a small grid.

Hard requirements:

- clean W==D is the only promotion/ranking authority;
- correctness before speed;
- route-fire check;
- materialization check (`E_49152` must not return);
- ISA JSON per candidate;
- repeated timings + spread;
- artifacts stamped via `qk_harness_contract`;
- reject candidates at first failed gate;
- no default flips;
- no decode behavior changes outside candidate env/config;
- no prefill changes;
- no 14B/32B.

Required outputs:

- artifacts under `bench/qk-decode-machine-search/`;
- `docs/decode-machine-search-execution-result-20260623.md`;
- README/handoff update only if useful.

Final response must include:

- final verdict;
- harness compliance summary;
- oracle recheck result;
- search mode/plan;
- number of candidates;
- reject summary;
- leaderboard;
- winner recheck if any;
- recommendation;
- files changed;
- git status.
