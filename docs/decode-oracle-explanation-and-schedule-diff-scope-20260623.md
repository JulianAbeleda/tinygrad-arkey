# Decode Oracle Explanation And Schedule-Diff — Exhaustive Scope / Claude Prompt

Date: 2026-06-23

## Mission

Explain exactly **why the current decode oracle is best**.

This is a decode-only scope. Prefill is covered separately by
`docs/prefill-schedule-diff-oracle-and-search-reduction-scope-20260623.md`.

The current decode default is at/above llama.cpp on Qwen3-8B-Q4_K_M / gfx1100,
and both policy search (Mode A) and generated tile-variant search (Mode B)
returned `ORACLE_REMAINS_BEST`. That is not enough understanding by itself.
This scope should consolidate the evidence into a machine-readable and
human-readable explanation of the oracle:

```text
why this primitive boundary wins,
why nearby generated variants do not,
why the whole-cache route beats the slice route,
why long-context gain shrinks,
what remaining decode headroom exists,
and what is / is not searchable.
```

## Core Thesis

The decode oracle wins because it simultaneously satisfies the full lifecycle:

- right **ABI/layout boundary**: whole-buffer identity cache input, no `E_49152`
  materialization;
- right **attention primitive**: owned split-KV tile with `v_dot2`, LDS,
  cross-lane reduction, fp16 cache, fp32 online softmax/PV;
- right **graph behavior**: route fires in the JIT graph, token-correct,
  fallback-safe;
- right **policy constants**: S/TK/vector/unroll/combine choices are already
  near optimum;
- right **whole-path transfer**: W==D beats slice/gqa and stays above llama
  through supported MAXC;
- acceptable **resource envelope**: no spill, expected VGPR/LDS, stable ISA.

Machine search still applies, but only after these facts are represented as
primitive rows. The decode search space is not "anything attention." It is:

```text
ABI/materialization policy
-> split-KV policy constants
-> tile microconstants
-> strided whole-cache read slope
-> native-codegen translation gaps
-> cross-shape/generalization
```

For 8B speed, the first three are already searched and closed. The remaining
decode levers are low-priority slope/coalescing or codegen-learning, not a broad
speed search.

## Required Reading

Read these first:

1. `docs/owned-tile-buffer-identity-kv-read-result-20260623.md`
2. `docs/decode-machine-search-execution-result-20260623.md`
3. `docs/decode-mode-b-search-result-20260623.md`
4. `docs/decode-ctx-slope-audit-result-20260623.md`
5. `docs/post-owned-attention-default-audit-result-20260623.md`
6. `docs/machine-code-translation-roadmap-result-20260623.md`
7. `docs/decode-machine-search-readiness-package-result-20260623.md`
8. `docs/oracle-guided-gpu-primitive-explorer-result-20260623.md`
9. `docs/oracle-guided-gpu-primitive-explorer-runner-design-20260623.md`
10. `docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md`
11. `bench/qk-decode-eval/HARNESS_GUIDE.md`
12. `structure/Development/performance-primitive-research-principles.md`
13. `structure/Development/session-handoff.md`

Inspect relevant tools/artifacts:

- `extra/qk_decode_runtime_overhead.py`
- `extra/qk_decode_search_execute.py`
- `extra/qk_decode_mode_b_execute.py`
- `extra/qk_decode_search_gate.py`
- `extra/qk_decode_route_fire_check.py`
- `extra/qk_decode_materialization_check.py`
- `extra/qk_ctx_slope_driver.py`
- `extra/qk_ctx_slope_analyze.py`
- `extra/qk_isa_primitive_audit.py`
- `extra/qk_amdgpu_isa_primitive_audit.py`
- `extra/qk_owned_flash_decode_graph_node.py`
- `extra/qk_owned_flash_decode.hip`
- `bench/qk-decode-machine-search/`
- `bench/qk-decode-mode-b-search/`
- `bench/qk-decode-ctx-slope-audit/`
- `bench/qk-owned-tile-buffer-identity-kv-read/`
- `bench/qk-machine-code-translation/`

## Non-Goals

- Do not change defaults.
- Do not build new kernels unless an explicit later implementation scope asks
  for it.
- Do not rerun broad Mode A or Mode B search unless needed to fill a missing
  artifact.
- Do not use PROFILE/DEBUG/no-sync/raw-dispatch timing as speed authority.
- Do not reopen attention/GEMV for 8B speed without new evidence.
- Do not make prefill claims here.
- Do not rewrite historical docs; add superseding notes only.

## Authority Rules

| question | authority |
|---|---|
| Does decode speed improve? | clean synced W==D only |
| Does route fire? | route-fire checker / captured graph node names |
| Was materialization removed? | materialization checker / `E_49152` absence |
| Is ISA intended? | code-object disassembly / ISA audit |
| Are tokens correct? | byte-identical greedy multi-token decode |
| Is a kernel-local change meaningful? | only if W==D transfers |
| Is a long-ctx residual real? | W==D slope + PROFILE attribution as diagnostic |

## Phase 0 — Authority Lock

Record the current decode oracle.

Tasks:

1. Capture git SHA, branch, GPU, ROCm, model, MAXC, active flags.
2. Record the canonical flag stack explicitly:
   - `DECODE_ATTN_AMDGCN_TILE`;
   - `DECODE_ATTN_KV_IDENTITY`;
   - `Q4K_GEMV_WARP`;
   - `Q4K_GEMV_WARP_DOWN`;
   - `Q4K_GEMV_WARP_PROJ`;
   - `FLASH_DECODE_THRESHOLD`;
   - `JIT`;
   - `DEV`.
3. Reproduce canonical W==D at ctx512/1024/2048/4096, or reuse stamped current
   artifacts if they are fresh and complete.
4. Confirm the route fires and `E_49152` is absent.
5. Confirm byte-identical tokens.

Deliverable:

```text
bench/qk-decode-oracle-explanation/authority.json
```

Verdicts:

- `DECODE_ORACLE_AUTHORITY_LOCKED`
- `DECODE_ORACLE_AUTHORITY_DRIFT_STOP`

Stop if the oracle does not reproduce inside the known spread band.

## Phase 1 — Oracle Fact Sheet

Create a complete fact sheet for the current decode oracle.

Required fields:

- route name and env flags;
- kernel symbols;
- graph node sequence;
- cache ABI:
  - whole `cache_kv.after(store)`;
  - no slice/reshape across precompiled boundary;
  - K/V in-kernel offsets;
- tensor dtypes and shapes;
- split-KV policy:
  - S;
  - TK;
  - combine variant;
  - grid/block;
  - GQA map;
- expected ISA facts:
  - `v_dot2`;
  - LDS bytes;
  - cross-lane instructions;
  - global load pattern;
  - VGPR/SGPR;
  - scratch/spill;
- fallback policy;
- correctness artifacts;
- W==D numbers;
- llama comparison;
- known low-priority residuals.

Deliverable:

```text
bench/qk-decode-oracle-explanation/oracle_fact_sheet.json
```

Verdict:

- `DECODE_ORACLE_FACT_SHEET_READY`

## Phase 2 — Alternative Family Comparison

Compare the oracle against every meaningful alternative family already tested.

Families:

1. gqa/cooperative baseline;
2. slice/materialization route (`DECODE_ATTN_KV_IDENTITY=0`);
3. policy variants:
   - S32/S64/S96;
   - hd64/hw combine variants;
   - min_ctx route-policy probe;
4. generated tile variants:
   - TK;
   - WCVEC;
   - WCUNROLL;
5. native-codegen variants / microprimitive attempts, if relevant;
6. llama.cpp reference, as external comparison.

For each family record:

- first gate passed/failed;
- correctness;
- route/materialization;
- ISA facts if available;
- W==D delta;
- reason it does not beat the oracle;
- learned rule.

Deliverable:

```text
bench/qk-decode-oracle-explanation/alternative_family_matrix.json
```

Verdicts:

- `DECODE_ALTERNATIVE_FAMILY_MATRIX_READY`
- `DECODE_ALTERNATIVE_DATA_INCOMPLETE`

## Phase 3 — Why The Oracle Wins: Primitive Decomposition

Reduce the decode oracle win into primitive rows.

Required rows:

1. `buffer_identity_cache_abi`
   - why it wins: removes `E_49152`;
   - evidence: callify/materialization checker, W==D +13-19%;
   - status: solved/default-on.

2. `owned_split_kv_attention_tile`
   - why it wins: v_dot2 + LDS + cross-lane + online softmax/PV;
   - evidence: ISA audit, correctness, W==D vs gqa;
   - status: solved/default-on.

3. `split_policy_S48`
   - why it wins: balances tile occupancy and combine overhead;
   - evidence: Mode A/Mode B S grid; S96 overhead;
   - status: searched/closed.

4. `combine_base`
   - why it wins: cheaper combine variants do not improve W==D;
   - evidence: B5 + Mode A/B;
   - status: searched/closed.

5. `tile_microconstants_TK16_VEC1_U1`
   - why it wins: generated Mode B variants do not transfer;
   - evidence: 14 variant search;
   - status: searched/closed.

6. `whole_cache_strided_read_slope`
   - why it remains: whole-buffer route has steeper ctx slope than contiguous
     slice route;
   - evidence: ctx-slope audit;
   - status: low-priority residual, not action-worthy for 8B MAXC.

7. `native_codegen_gaps`
   - why it remains hand-owned: tinygrad native renderer still lacks v_dot2 and
     cross-lane lowering;
   - evidence: native-codegen microprimitive result;
   - status: codegen learning, not 8B speed requirement.

Deliverable:

```text
bench/qk-decode-oracle-explanation/primitive_decomposition.json
```

Verdict:

- `DECODE_ORACLE_PRIMITIVES_EXPLAINED`

## Phase 4 — Static / ISA Schedule Explanation

Build or reuse a static ISA/resource explanation for the oracle and close
alternatives.

Tasks:

1. Audit the oracle tile code object.
2. Audit at least one generated Mode B equivalent variant if available.
3. Audit the slice-route tile if available.
4. Compare:
   - VGPR/SGPR;
   - LDS;
   - spill;
   - dot instruction;
   - cross-lane count;
   - vector/global load count;
   - combine ISA if relevant.
5. Explain why Mode B constants did not uncover a new primitive:
   - same high-level primitive boundary;
   - only local instruction schedule/loop constants changed;
   - no material ABI change;
   - W==D within noise.

Deliverable:

```text
bench/qk-decode-oracle-explanation/static_isa_explanation.json
```

Verdicts:

- `DECODE_STATIC_ISA_EXPLANATION_READY`
- `DECODE_STATIC_ISA_DATA_INCOMPLETE`

## Phase 5 — Context-Slope Explanation

Explain why the oracle's percent gain shrinks at long context.

Required answer:

- fixed materialization tax removed: yes, ctx-flat;
- replacement whole-cache tile steeper than contiguous slice tile: yes;
- saved-ms erodes with ctx;
- tinygrad still above llama through MAXC;
- crossover projected beyond supported context;
- remaining coalescing headroom < action bar.

Deliverable:

```text
bench/qk-decode-oracle-explanation/ctx_slope_explanation.json
```

Verdict:

- `DECODE_CTX_SLOPE_EXPLAINED`

## Phase 6 — Search Surface Decision

Classify every possible decode search surface.

| surface | expected classification |
|---|---|
| Mode A policy (`S`, combine, min_ctx) | searched / oracle best |
| Mode B generated tile constants | searched / oracle best |
| buffer-identity ABI | solved/default-on |
| strided whole-cache coalescing | low-priority residual (<2%, long ctx only) |
| native v_dot2/cross-lane codegen | codegen-learning, not needed for 8B speed |
| cross-shape/model generalization | deferred until target selected |
| free-form attention kernel generation | disallowed without new primitive audit |

Deliverable:

```text
bench/qk-decode-oracle-explanation/search_surface_decision.json
```

Verdicts:

- `DECODE_8B_SEARCH_SURFACE_EXHAUSTED`
- `DECODE_LOW_PRIORITY_SLOPE_RESIDUAL_ONLY`
- `DECODE_NATIVE_CODEGEN_LEARNING_ONLY`

## Phase 7 — Machine Search Integration

Update the oracle-guided explorer and project ledger with the explanation.

Tasks:

1. Add a decode oracle explanation artifact entry if the explorer has an oracle
   registry.
2. Add learned rules:
   - buffer identity beats sliced precompiled-call inputs;
   - S48/base/TK16/VEC1/U1 is oracle for 8B gfx1100;
   - S96 split overhead;
   - generated tile constants searched and closed;
   - whole-cache strided slope is low-priority.
3. Ensure a future LoRA primitive-space proposer can learn from this:
   - create primitive rows;
   - include stop rules;
   - include evidence requirements.

Deliverables:

```text
bench/qk-decode-oracle-explanation/learned_rules.json
bench/qk-project-search-ledger/ledger.jsonl
```

Verdict:

- `DECODE_ORACLE_LEARNED_RULES_RECORDED`

## Phase 8 — Result Doc

Write:

```text
docs/decode-oracle-explanation-and-schedule-diff-result-20260623.md
```

Required answers:

1. What exactly is the decode oracle?
2. Why does it beat gqa/slice/materialization route?
3. Why do S/combine policy variants not beat it?
4. Why do generated tile constants not beat it?
5. What ISA/resource facts prove the intended primitive is present?
6. Why does the gain shrink at ctx4096?
7. What headroom remains, and why is it low priority?
8. What decode search surfaces are closed?
9. What decode search surfaces remain only for learning/generalization?
10. What should machine search do next?

## Expected Final Verdict

Likely final verdict:

```text
DECODE_ORACLE_EXPLAINED
DECODE_8B_SEARCH_SURFACE_EXHAUSTED
DECODE_LOW_PRIORITY_SLOPE_RESIDUAL_ONLY
DECODE_NATIVE_CODEGEN_LEARNING_ONLY
```

If authority drift is found, stop and return:

```text
DECODE_ORACLE_AUTHORITY_DRIFT_STOP
```

If a missing artifact prevents explanation, return:

```text
DECODE_ORACLE_EXPLANATION_INCOMPLETE
```

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch
`qk-prefill-flag-leak-resolution`.

Task: execute the decode oracle explanation and schedule-diff scope. This is a
decode-only understanding task. The goal is not to optimize; the goal is to
explain exactly why the current whole-cache owned-tile oracle is best and what
decode machine-search surfaces remain.

Read first:

- `docs/decode-oracle-explanation-and-schedule-diff-scope-20260623.md`
- `docs/owned-tile-buffer-identity-kv-read-result-20260623.md`
- `docs/decode-machine-search-execution-result-20260623.md`
- `docs/decode-mode-b-search-result-20260623.md`
- `docs/decode-ctx-slope-audit-result-20260623.md`
- `docs/post-owned-attention-default-audit-result-20260623.md`
- `docs/machine-code-translation-roadmap-result-20260623.md`
- `docs/oracle-guided-gpu-primitive-explorer-result-20260623.md`
- `bench/qk-decode-eval/HARNESS_GUIDE.md`
- `structure/Development/performance-primitive-research-principles.md`
- `structure/Development/session-handoff.md`

Execute phases:

1. Authority lock and route/materialization confirmation.
2. Build oracle fact sheet.
3. Build alternative family matrix.
4. Reduce oracle win into primitive rows.
5. Build static ISA/resource explanation.
6. Build ctx-slope explanation.
7. Classify decode search surfaces.
8. Record learned rules and write result doc.

Boundaries:

- no default flips;
- no new kernels;
- no broad search reruns unless an artifact is missing;
- no prefill work;
- no PROFILE/nosync timing as authority;
- preserve historical docs with superseding notes only.

Final response must include:

- verdict labels;
- the exact reason the oracle wins;
- closed decode surfaces;
- any remaining low-priority or learning-only surfaces;
- artifacts written;
- files changed;
- git status.
