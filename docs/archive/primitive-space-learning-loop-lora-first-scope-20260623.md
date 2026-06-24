# Primitive-Space Learning Loop — LoRA-First, RLVR-Later Scope / Claude Prompt

Date: 2026-06-23

## Purpose

Update the project narrative around the model/adapter loop.

The goal is **not** to let a LoRA or RL policy promote kernels. The goal is to
teach the model to emit the **right primitive search space** so deterministic
machine search can operate over bounded, high-signal knobs.

The correct loop is:

```text
repo history + ISA audits + W==D / whole-prefill outcomes
-> structured primitive taxonomy
-> LoRA/SFT model learns primitive boundaries, refutations, and evidence rules
-> model emits a bounded search spec
-> deterministic machine search expands and tests that spec
-> harness/ISA/correctness/whole-path transfer decide outcomes
-> outcomes become more structured training rows
```

This is a **primitive-space proposer** loop, not a kernel-promotion loop.

## Decision

Use **LoRA/SFT first**. Defer **RLVR/RLHF/GRPO/PPO** until the supervised loop
is format-stable and demonstrably useful.

Reason:

- the immediate task is repo-specific vocabulary, taxonomy, and stop-rule
  learning;
- the reward is mostly deterministic but delayed and sparse unless the schema
  is already stable;
- prior adapter work showed teacher-forced gains can fail to transfer to
  generation, so free-generation strict JSON must be the first gate;
- RLVR without a stable schema/reward would optimize shortcuts instead of
  useful primitive search spaces.

Verdicts to record:

- `PRIMITIVE_SPACE_PROPOSER_NOT_KERNEL_JUDGE`
- `LORA_FIRST_FOR_PRIMITIVE_SPACE_LEARNING`
- `RLVR_DEFERRED_UNTIL_SCHEMA_AND_REWARD_STABLE`
- `DETERMINISTIC_HARNESS_REMAINS_AUTHORITY`

## Required Reading

Read these before updating docs:

1. `docs/oracle-guided-gpu-primitive-explorer-result-20260623.md`
2. `docs/oracle-guided-gpu-primitive-explorer-runner-design-20260623.md`
3. `docs/oracle-guided-gpu-primitive-explorer-scope-20260623.md`
4. `docs/project-wide-machine-search-roadmap-result-20260623.md`
5. `docs/project-search-ledger-contract-20260623.md`
6. `docs/decode-machine-search-readiness-package-result-20260623.md`
7. `docs/prefill-post-decode-parity-frontier-result-20260623.md`
8. `docs/native-codegen-microprimitive-search-result-20260623.md`
9. `docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md`
10. `docs/amd-decode-kernel-optimization-flywheel.md`
11. `docs/amd-decode-flywheel-proof-plan.md`
12. `docs/qwen-json-eval-objective-scope.md`
13. `bench/qwen-adapter-20260613/README.md`
14. `bench/qk-decode-eval/HARNESS_GUIDE.md`
15. `structure/Development/performance-primitive-research-principles.md`
16. `structure/Development/session-handoff.md`

Inspect current adapter/search tools:

- `extra/llm_adapter.py`
- `extra/llm_adapter_train.py`
- `extra/llm_adapter_suffix_train.py`
- `extra/llm_rollout.py`
- `extra/llm_rollout_compare.py`
- `extra/llm_json_scorer.py`
- `extra/llm_json_rejection_sample.py`
- `extra/qk_search_spec.py`
- `extra/qk_project_search_ledger.py`
- `extra/qk_decode_search_runner.py`
- `extra/qk_isa_primitive_audit.py`

## What The Adapter Should Learn

The adapter should learn to produce structured primitive search specs, not
source code and not promotion decisions.

Target output shape:

```json
{
  "lane": "native_codegen_microprimitive",
  "primitive": "cross_lane_reduce",
  "hypothesis": "tinygrad native lowering uses an LDS tree where the owned tile uses ds_bpermute",
  "search_space": {
    "knobs": ["warp_width", "reduce_pattern", "uop_lowering"],
    "bounds": {
      "warp_width": [32],
      "reduce_pattern": ["xor_tree", "down_tree"],
      "uop_lowering": ["ds_bpermute", "ds_swizzle"]
    }
  },
  "required_evidence": [
    "local_numeric_correctness",
    "has_cross_lane_isa",
    "no_spill",
    "ledger_entry"
  ],
  "stop_rules": [
    "if no ds_bpermute or equivalent appears, classify as CODEGEN_GAP_NOT_SEARCH_WIN",
    "do not claim W==D for native microprimitive lane"
  ]
}
```

Other valid output examples:

```json
{
  "lane": "prefill_role_policy",
  "primitive": "small_N_workgroup_occupancy",
  "target": "kv_proj graph-GEMM",
  "search_space": {
    "knobs": ["waves_n", "BN", "BK"],
    "bounds": {
      "waves_n": [1, 2],
      "BN": [64, 128],
      "BK": [32, 64]
    }
  },
  "required_evidence": [
    "synced_whole_prefill",
    "per_role_gpu_busy",
    "byte_identical_generation"
  ],
  "stop_rules": [
    "PROFILE/nosync timing cannot promote",
    "if role improves but whole-prefill does not, record non-transfer"
  ]
}
```

```json
{
  "lane": "decode_policy",
  "primitive": "owned_attention_policy",
  "target": "whole-cache owned tile route",
  "search_space": {
    "knobs": ["split_S", "min_ctx", "combine_variant"],
    "bounds": {
      "split_S": [48, 64, 96],
      "min_ctx": [512, 1024],
      "combine_variant": ["base", "hw128"]
    }
  },
  "required_evidence": [
    "route_fires",
    "no_E_49152_materialization",
    "ISA_PRIMITIVE_CONFIRMED",
    "byte_identical_tokens",
    "clean_synced_WD"
  ],
  "stop_rules": [
    "if oracle remains best within spread, stop",
    "do not reopen attention/GEMV variants for 8B speed unless a new audit reopens the lane"
  ]
}
```

## What The Adapter Must Not Do

The adapter must not:

- decide that a kernel is fast;
- flip defaults;
- bypass correctness;
- bypass ISA/resource checks;
- treat teacher-forced accuracy as success;
- train on holdout labels;
- generate broad free-form kernels as the default mode;
- reopen closed attention/GEMV lanes without a new audit;
- use RLVR before the supervised strict-JSON primitive-spec gate works.

## LoRA / SFT Phase Plan

### Phase 1 — Dataset From The Project Ledger

Build a primitive-space training dataset from existing docs/artifacts.

Rows should come from:

- accepted wins: buffer-identity KV read, owned attention tile, Q4K GEMV warp,
  prefill kv_proj workgroup fix;
- rejected/refuted lanes: B4 combine-only, opaque append/cache identity theories,
  attention saturation before dtype/whole-buffer correction, q8/int-dot nulls,
  nosync prefill false wins;
- current search specs and ledgers;
- ISA audit facts;
- harness discipline failures and corrections.

Each row should include:

- prompt: summarized artifact context without the answer;
- target JSON: lane, primitive, hypothesis, search knobs, required evidence,
  stop rules, and verdict class;
- split id and family id;
- source artifact links;
- whether the row is accepted, rejected, refuted, or deferred.

Holdout split must be **family split**, not random split, so near-duplicate
versions of the same episode do not leak.

Deliverable:

```text
bench/qk-primitive-space-adapter/dataset-v0/
  train.jsonl
  holdout.jsonl
  summary.json
  README.md
```

Verdicts:

- `PRIMITIVE_SPACE_DATASET_READY`
- `PRIMITIVE_SPACE_DATASET_BLOCKED`

### Phase 2 — Deterministic Scorer

Create a deterministic scorer for primitive-space JSON.

Required axes:

- `parse_valid`
- `schema_ok`
- `lane_valid`
- `primitive_valid`
- `evidence_complete`
- `stop_rules_complete`
- `closed_lane_respected`
- `harness_authority_correct`
- `strict_pass`

Do not use LLM-as-judge. This is a structured, programmatically scorable task.

Deliverable:

```text
extra/qk_primitive_space_scorer.py
bench/qk-primitive-space-adapter/scorer-smoke/
```

Verdicts:

- `PRIMITIVE_SPACE_SCORER_READY`
- `PRIMITIVE_SPACE_SCORER_INSUFFICIENT`

### Phase 3 — Baselines Before Training

Run baselines on the holdout:

1. deterministic mechanism-prior baseline;
2. base Qwen3-8B generation with strict JSON prompt;
3. optional structured prompt with examples, no adapter.

Record:

- strict pass rate with Wilson interval;
- per-axis pass rates;
- common failure modes;
- compare to deterministic baseline.

Deliverable:

```text
bench/qk-primitive-space-adapter/baselines-v0/
```

Verdicts:

- `BASELINE_PRIMITIVE_SPACE_MEASURED`
- `BASE_MODEL_NOT_SCHEMA_STABLE`

### Phase 4 — LoRA/SFT Candidate

Train the smallest viable adapter first.

Recommended first candidate:

- base: Qwen3-8B-Q4_K_M;
- training path: existing suffix-cache internal adapter if compatible;
- target: `last1_ffn`;
- rank: 4;
- alpha: 8;
- evaluation: free-generation strict JSON on holdout, temperature 0;
- teacher-forced loss: diagnostic only.

Escalation:

- increase rank or target depth only after schema validity is reliable;
- do not sweep adapter capacity before the scorer shows where failure lives.

Deliverables:

```text
bench/qk-primitive-space-adapter/lora-v0/
bench/qk-primitive-space-adapter/lora-v0-rollout/
bench/qk-primitive-space-adapter/lora-v0-eval/
```

Promotion gate for the adapter as a **proposer**:

- strict JSON pass materially beats base and deterministic baseline;
- closed-lane violations near zero;
- evidence/stop-rule completeness materially improves;
- no holdout leakage;
- generated specs are usable by the deterministic search runner in dry-run.

Verdicts:

- `LORA_PRIMITIVE_SPACE_PROPOSER_PASS`
- `LORA_PRIMITIVE_SPACE_FORMAT_ONLY`
- `LORA_PRIMITIVE_SPACE_FAIL`

### Phase 5 — Shadow Mode, Not Live Authority

If Phase 4 passes, run the adapter in shadow mode.

Tasks:

- feed current artifacts / a new audit into the adapter;
- let it emit a primitive search spec;
- run only `--dry-run` / structural validation first;
- if valid, run deterministic search with the normal gate stack;
- compare adapter proposal quality against human-written scope and deterministic
  mechanism-prior baseline.

The adapter still cannot promote kernels. It can only propose/rank search
spaces.

Deliverables:

```text
bench/qk-primitive-space-adapter/shadow-v0/
```

Verdicts:

- `SHADOW_PRIMITIVE_PROPOSER_USEFUL`
- `SHADOW_PRIMITIVE_PROPOSER_NOT_USEFUL`

### Phase 6 — Rejection-Sampling SFT

Only after shadow mode has useful examples:

```text
adapter proposes K primitive specs
-> deterministic scorer filters valid specs
-> deterministic search labels outcomes
-> accepted/useful specs become new SFT rows
-> retrain adapter
```

This is the first “loop” phase. It is still SFT/rejection sampling, not RLVR.

Verdicts:

- `RS_SFT_PRIMITIVE_LOOP_READY`
- `RS_SFT_PRIMITIVE_LOOP_NOT_READY`

## RLVR Defer Criteria

Do not start RLVR until all are true:

- schema pass is high enough that rewards are not mostly format failure;
- deterministic reward is defined and stable;
- reward includes negative penalties for closed-lane violations and missing
  evidence, not just “proposal led to a search”;
- there is enough cheap rollout budget;
- rejection-sampling SFT has plateaued;
- the adapter beats deterministic baselines in shadow mode.

Potential RLVR reward components later:

- + parse/schema validity;
- + valid lane/primitive classification;
- + evidence completeness;
- + dry-run search spec acceptance;
- + eventual search utility if a deterministic run finds useful signal;
- - closed-lane reopen;
- - missing harness authority;
- - hallucinated tool or unsupported knob;
- - holdout leakage / artifact mismatch.

Verdict to use until then:

```text
RLVR_DEFERRED_UNTIL_PRIMITIVE_REWARD_STABLE
```

## Docs To Update

Claude should update these docs, preserving historical results instead of
rewriting them:

1. `docs/oracle-guided-gpu-primitive-explorer-result-20260623.md`
   - Add a section: “Learning Layer: Primitive-Space Proposer.”
   - State LoRA/SFT is the first learning tool; RLVR deferred.

2. `docs/oracle-guided-gpu-primitive-explorer-runner-design-20260623.md`
   - Add where adapter-generated specs enter: before candidate generation, as a
     proposed `SearchRow`/search spec.
   - State runner remains deterministic authority.

3. `docs/project-wide-machine-search-roadmap-result-20260623.md`
   - Add the model-learning lane as “spec proposer,” not “kernel judge.”

4. `docs/amd-decode-kernel-optimization-flywheel.md`
   - Add a 2026-06-23 superseding note:
     the closing link is reframed from broad kernel triage to primitive-space
     proposal under deterministic search gates.

5. `docs/qwen-json-eval-objective-scope.md`
   - Add a pointer that the strict JSON machinery now has a second use:
     primitive search specs, not only answer JSON.

6. `structure/Development/performance-primitive-research-principles.md`
   - Add a principle:
     “Learned models propose primitive search spaces; deterministic lifecycle
     gates decide.”

7. `structure/Development/session-handoff.md`
   - Add the current status and next task.

Optional new result doc:

```text
docs/primitive-space-learning-loop-lora-first-result-20260623.md
```

## Non-Goals

- Do not train an adapter in this doc-update task unless explicitly requested.
- Do not run RLVR.
- Do not generate kernels.
- Do not start machine search.
- Do not change tinygrad defaults.
- Do not edit historical benchmark numbers except to add superseding notes.

## Success Criteria

The documentation update is complete when it answers:

1. What is the learning loop for?
2. Why is LoRA/SFT first?
3. Why is RLVR deferred?
4. What exactly should the adapter output?
5. How does the deterministic runner consume the output?
6. What gates remain authoritative?
7. What dataset/scorer/eval must exist before training?
8. What would unlock RLVR later?

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Task: update the project docs to reflect the correct learned-model role in the
GPU primitive search system.

Read first:

- `docs/primitive-space-learning-loop-lora-first-scope-20260623.md`
- `docs/oracle-guided-gpu-primitive-explorer-result-20260623.md`
- `docs/oracle-guided-gpu-primitive-explorer-runner-design-20260623.md`
- `docs/project-wide-machine-search-roadmap-result-20260623.md`
- `docs/amd-decode-kernel-optimization-flywheel.md`
- `docs/qwen-json-eval-objective-scope.md`
- `bench/qwen-adapter-20260613/README.md`
- `structure/Development/performance-primitive-research-principles.md`
- `structure/Development/session-handoff.md`

Implement documentation updates only:

1. Add the `PRIMITIVE_SPACE_PROPOSER_NOT_KERNEL_JUDGE` decision.
2. State that LoRA/SFT is the first tool because the task is structured,
   supervised primitive-space generation.
3. State that RLVR is deferred until schema/reward/shadow-mode are stable.
4. Clarify that adapter output is a bounded search spec / `SearchRow` proposal,
   not source code and not a promotion decision.
5. Clarify that the deterministic lifecycle runner remains the authority:
   harness contract, route/materialization, ISA/resource, correctness, and
   W==D/whole-prefill.
6. Add or update doc pointers so the flywheel narrative no longer implies the
   model should judge kernel speed.
7. Preserve historical docs by adding superseding notes; do not rewrite old
   benchmark history.
8. Produce a short result doc with final verdict labels and file list.

Boundaries:

- no tinygrad source changes;
- no adapter training;
- no RLVR;
- no kernel generation;
- no machine search runs;
- no default flips.

Final response should include:

- verdict labels;
- docs changed;
- whether any source/default files changed;
- recommended next executable task:
  build `bench/qk-primitive-space-adapter/dataset-v0` + deterministic scorer.
