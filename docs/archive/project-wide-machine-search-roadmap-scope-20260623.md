# Project-Wide Machine Search Roadmap — Exhaustive Objective Scope (2026-06-23)

## Mission

Define the project-wide path to "as much machine search as possible" without losing the discipline that made the decode
and prefill wins transfer.

This is not a request to start broad autonomous search. It is the objective map for five concrete steps:

1. Run the first real bounded decode search.
2. Create a project-level search ledger.
3. Complete prefill attribution before any prefill search.
4. Start native-codegen microprimitive search.
5. Select cross-shape/generalization targets and build oracles.

The goal is to turn the project into a search-capable system where every lane has:

```text
bounded candidate space -> cheap structural/ISA prune -> correctness -> authoritative whole-path benchmark -> remember
```

## Current State

### Solved / at parity

- Decode Qwen3-8B is at/above llama.cpp.
- Prefill Qwen3-8B is at/above llama.cpp after the small-N WG-starvation fix.
- Q4K GEMV and owned attention are closed for validated 8B.
- Buffer-identity ABI rule is recorded and default-on path uses it.
- ISA audit infrastructure exists.
- Decode machine-search readiness package exists.

### Search readiness estimates

| lane | readiness | status |
|---|---:|---|
| decode policy/config search | high | first real bounded search can run |
| project-wide search ledger | high | schema/tooling consolidation needed |
| prefill search | medium/low | attribution first; current result says no search yet |
| native-codegen microprimitive search | medium | good ISA targets, needs harness |
| cross-shape/generalization search | low/medium | choose targets and build oracles |
| global autonomous search | low | not allowed yet |

## Required Reading

Read these first:

1. `docs/decode-machine-search-readiness-package-result-20260623.md`
2. `docs/decode-machine-search-execution-scope-20260623.md`
3. `docs/prefill-post-decode-parity-frontier-result-20260623.md`
4. `docs/prefill-frontier-rest-or-nonsearch-next-scope-20260623.md`
5. `docs/decode-campaign-final-synthesis-20260623.md`
6. `docs/machine-code-translation-roadmap-result-20260623.md`
7. `docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md`
8. `docs/amd-gpu-holistic-primitive-model-20260623.md`
9. `bench/qk-decode-eval/HARNESS_GUIDE.md`
10. `structure/Development/performance-primitive-research-principles.md`
11. `structure/Development/session-handoff.md`

Inspect:

- `extra/qk_decode_search_runner.py`
- `extra/qk_decode_search_gate.py`
- `extra/qk_decode_search_execute.py` if present
- `extra/qk_isa_primitive_audit.py`
- prefill frontier tools/artifacts
- `bench/qk-decode-search-readiness/`
- `bench/qk-decode-machine-search/` if present
- `bench/qk-prefill-post-decode-parity-frontier/`

## Global Search Rules

- W==D / whole-prefill synced authority is promotion authority.
- PROFILE, DEBUG, raw/no-sync, and local-only timings are diagnostic only.
- Correctness before speed.
- Route identity before W==D.
- Materialization/ABI checks before W==D.
- ISA/resource checks before W==D when candidate code changes.
- Stop at first failed gate.
- Every performance artifact must satisfy the harness 13-field contract.
- No default flip from a search harness.
- No broad/random kernel generation.
- No stale baselines.

## Step 1 — First Real Bounded Decode Search

### Objective

Prove the search loop works on a real bounded decode candidate set.

Do this first because decode has the best search infrastructure:

- frozen oracle;
- candidate runner;
- correctness gate;
- W==D gate;
- route-fire checker;
- materialization checker;
- ISA audit;
- schemas;
- reject rules.

### Scope

Execute:

- `docs/decode-machine-search-execution-scope-20260623.md`

Recommended first run:

```text
Mode A policy search
S in {32,48,64,96}
min_ctx in {256,512,1024}
combine in safe registered variants
```

No generated kernels in the first run.

### Required artifacts

Under:

```text
bench/qk-decode-machine-search/
```

Must include:

- authority;
- oracle recheck;
- search plan;
- candidate manifest;
- results JSONL;
- leaderboard;
- reject summary;
- decision.

### Success criteria

Any of these is success:

- oracle remains best within spread;
- a candidate wins outside spread and passes winner recheck;
- all bad candidates are rejected correctly;
- harness produces complete contract-compliant artifacts.

The first search does **not** need to find a speedup.

### Verdicts

- `DECODE_SEARCH_EXECUTED_ORACLE_REMAINS_BEST`
- `DECODE_SEARCH_EXECUTED_WINNER_FOUND_RECOMMEND_ONLY`
- `DECODE_SEARCH_EXECUTED_NO_PASSING_CANDIDATES`
- `DECODE_SEARCH_BLOCKED_*`

## Step 2 — Project-Level Search Ledger

### Objective

Create one project-wide memory/ledger format for all search lanes.

Current artifacts are lane-specific. The project needs a shared schema:

```text
candidate -> lane -> knobs -> gates -> authority benchmark -> verdict -> artifact links -> learned rule
```

### Scope

Create:

- `docs/project-search-ledger-contract-20260623.md`
- optionally `bench/qk-project-search-ledger/schema.json`

### Required schema fields

| field | meaning |
|---|---|
| candidate_id | stable id |
| lane | decode / prefill / codegen / cross-shape / small-op |
| primitive_class | attention, GEMM, ABI, fusion, route policy, codegen microprimitive |
| knobs | bounded knobs |
| oracle | comparator id and artifact |
| correctness | pass/fail + method |
| route_identity | pass/fail if applicable |
| materialization_abi | pass/fail if applicable |
| isa | audit artifact + key flags |
| local_diagnostic | optional, non-authority |
| authority_benchmark | W==D / whole-prefill / microprimitive |
| verdict | enum-like final state |
| stop_reason | first failed gate |
| artifact_links | result/doc paths |
| learned_rule | durable lesson if any |

### Required lane support

- decode search;
- prefill search;
- codegen microprimitive search;
- cross-shape search;
- small-op fusion if reopened.

### Verdicts

- `PROJECT_SEARCH_LEDGER_READY`
- `PROJECT_SEARCH_LEDGER_PARTIAL`

## Step 3 — Prefill Attribution Before Search

### Objective

Do not search prefill kernels unless attribution reopens a real kernel/headroom lane.

Current result says:

- synced whole prefill current graph-GEMM is ~96-99.5% of Tensile depending after fix;
- the stale 66% headline is retired;
- kv_proj WG-starvation was fixed;
- machine search is not currently justified for prefill speed.

But to make the project search-heavy, prefill needs a clean entry condition for future search.

### Scope

Use:

- `docs/prefill-post-decode-parity-frontier-result-20260623.md`
- `docs/prefill-frontier-rest-or-nonsearch-next-scope-20260623.md`

Required next prefill objective:

```text
prefill search gate: prove any remaining headroom is kernel/searchable rather than integration/policy/rest.
```

### Required checks

- synced whole-prefill only;
- per-role timing;
- current graph-GEMM vs Tensile;
- role shape table;
- top residual after kv_proj fix;
- check if any role remains WG-starved;
- check if any kernel has ISA/resource gap;
- check if local timing transfers to whole-prefill.

### Search unlock condition

Prefill search becomes allowed only if:

- a role has material residual time;
- route/kernel is actually active in-model;
- local candidate change transfers to whole-prefill;
- correctness harness exists;
- ISA audit can reject bad variants;
- expected gain exceeds noise.

### Verdicts

- `PREFILL_SEARCH_REMAINS_NOT_READY`
- `PREFILL_SEARCH_READY_ROLE_SPECIFIC`
- `PREFILL_AT_REST_AFTER_KV_PROJ_FIX`
- `PREFILL_NEEDS_INTEGRATION_FIX_NOT_SEARCH`

## Step 4 — Native-Codegen Microprimitive Search

### Objective

Start the safest non-W==D machine search lane: codegen microprimitives.

This search is not trying to improve decode tok/s immediately. It tries to make tinygrad-native codegen reproduce proven
machine-code primitives from the owned decode tile.

### Search targets

| target | oracle evidence | desired machine code |
|---|---|---|
| fp16 dot lowering | owned tile | `v_dot2` |
| LDS staging | owned tile | `ds_load` / `ds_store` with expected LDS bytes |
| cross-lane reduction | owned tile | `ds_bpermute` / cross-lane, not LDS tree if target says cross-lane |
| vector global loads | owned tile / GEMM | `global_load_dwordx*` style patterns |
| no-spill envelope | owned tile | scratch/spill = 0 |

### Required harness

Create or scope:

- `docs/native-codegen-microprimitive-search-scope-20260623.md`

Potential tools:

- candidate generator for tinygrad-native microkernels;
- ISA audit wrapper;
- local numerical correctness;
- no W==D promotion claim;
- artifact ledger.

### Success criteria

- candidate emits target ISA;
- local correctness passes;
- resource envelope acceptable;
- artifact recorded in project search ledger.

### Verdicts

- `NATIVE_CODEGEN_MICROSEARCH_READY`
- `NATIVE_CODEGEN_MICROSEARCH_EXECUTED_TARGET_FOUND`
- `NATIVE_CODEGEN_MICROSEARCH_NO_TARGET_FOUND`

## Step 5 — Cross-Shape / Generalization Target Selection

### Objective

Search should eventually help generalize beyond one validated 8B/gfx1100 path, but only after targets are explicit.

Possible target axes:

- model size:
  - 14B;
  - 32B;
- context:
  - longer decode;
  - different prefill lengths;
- GPU:
  - other RDNA/CDNA target if available;
- quant/model:
  - other Q4_K_M shapes;
  - Q6/KV variants.

### Required scope

Write:

- `docs/cross-shape-generalization-search-targets-scope-20260623.md`

Required sections:

1. target selection;
2. baseline oracle per target;
3. route eligibility;
4. shape inventory;
5. correctness harness;
6. decode/prefill authority benchmark;
7. bounded knobs;
8. expected cost;
9. stop rules.

### Search unlock condition

Cross-shape search becomes allowed only when:

- target model/data is available;
- baseline oracle exists;
- correctness harness exists;
- current route either works or fails for a diagnosed reason;
- bounded knobs are known.

### Verdicts

- `CROSS_SHAPE_TARGETS_SELECTED`
- `CROSS_SHAPE_SEARCH_READY`
- `CROSS_SHAPE_NEEDS_BASELINES`
- `CROSS_SHAPE_DEFERRED`

## Combined Roadmap

Recommended execution order:

1. **Step 1:** run first real bounded decode search.
2. **Step 2:** create project-level search ledger.
3. **Step 3:** close or reopen prefill search from current attribution.
4. **Step 4:** scope/execute native-codegen microprimitive search.
5. **Step 5:** choose cross-shape targets and build oracles.

Why this order:

- decode proves the full search lifecycle;
- ledger prevents fragmented search memory;
- prefill should not search until it proves a searchable gap;
- codegen microsearch is safe and useful;
- cross-shape search needs explicit targets.

## Result Doc

If executing this scope as a planning task, write:

- `docs/project-wide-machine-search-roadmap-result-20260623.md`

Required sections:

1. Verdict.
2. Step 1 decode search status.
3. Step 2 ledger status.
4. Step 3 prefill search status.
5. Step 4 native-codegen microsearch status.
6. Step 5 cross-shape target status.
7. Global search rules.
8. Recommended execution order.
9. Files changed.
10. Git status.

## Final Verdict Labels

- `PROJECT_MACHINE_SEARCH_ROADMAP_READY`
- `DECODE_SEARCH_FIRST`
- `PROJECT_SEARCH_LEDGER_NEEDED`
- `PREFILL_SEARCH_GATED_BY_ATTRIBUTION`
- `NATIVE_CODEGEN_MICROSEARCH_READY_TO_SCOPE`
- `CROSS_SHAPE_SEARCH_NEEDS_TARGETS`

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

The owner wants a holistic project-wide path to "as much machine search as possible" across decode, prefill, native
codegen, and future cross-shape/generalization.

Read and execute:

```text
docs/project-wide-machine-search-roadmap-scope-20260623.md
bench/qk-decode-eval/HARNESS_GUIDE.md
docs/decode-machine-search-readiness-package-result-20260623.md
docs/prefill-post-decode-parity-frontier-result-20260623.md
docs/machine-code-translation-roadmap-result-20260623.md
```

Do not start broad autonomous search. This is a roadmap/scope consolidation task.

Required output:

1. Confirm Step 1: first real bounded decode search.
2. Confirm Step 2: project-level search ledger.
3. Confirm Step 3: prefill attribution gate before search.
4. Confirm Step 4: native-codegen microprimitive search.
5. Confirm Step 5: cross-shape/generalization target selection.
6. Write `docs/project-wide-machine-search-roadmap-result-20260623.md`.
7. Update README/session handoff if useful.

Final response must include:

- final verdict;
- readiness by lane;
- exact next execution order;
- which searches are allowed now;
- which searches remain blocked;
- files changed;
- git status.
