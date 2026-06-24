# Primitive-Space Learning Loop — LoRA-First, RLVR-Later — Result (2026-06-23)

## Verdict labels

- `PRIMITIVE_SPACE_PROPOSER_NOT_KERNEL_JUDGE`
- `LORA_FIRST_FOR_PRIMITIVE_SPACE_LEARNING`
- `RLVR_DEFERRED_UNTIL_SCHEMA_AND_REWARD_STABLE`
- `DETERMINISTIC_HARNESS_REMAINS_AUTHORITY`
- `RLVR_DEFERRED_UNTIL_PRIMITIVE_REWARD_STABLE` (standing until the defer criteria are met)

This task is **documentation only**: it records the correct role of the learned model in the GPU primitive search
system. No adapter was trained, no RLVR was run, no kernels were generated, no machine search was run, no tinygrad
source or defaults were changed.

## 1. What is the learning loop for?

To teach a model to emit the **right primitive search space**, so deterministic machine search operates over bounded,
high-signal knobs. The loop:

```text
repo history + ISA audits + W==D/whole-prefill outcomes
 -> structured primitive taxonomy
 -> LoRA/SFT model learns primitive boundaries, refutations, evidence rules
 -> model emits a bounded search spec (SearchRow)
 -> deterministic machine search expands and tests that spec
 -> harness/ISA/correctness/whole-path transfer decide outcomes
 -> outcomes become more structured training rows
```

It is a **primitive-space proposer** loop, not a kernel-promotion loop.

## 2. Why LoRA/SFT first?

The immediate task is repo-specific vocabulary, taxonomy, and stop-rule learning — structured, supervised, and
**programmatically scorable** (strict JSON, no LLM-as-judge). Prior adapter work showed teacher-forced gains can fail
to transfer to free generation, so a free-generation strict-JSON gate must come first. SFT is the right tool for
format-stable structured generation.

## 3. Why is RLVR deferred?

Without a stable schema and a deterministic reward, RLVR would optimize shortcuts (format failure dominates the reward,
not search utility). Defer until **all** hold: schema pass high enough that rewards aren't mostly format failure;
deterministic reward defined and stable; reward includes negative penalties for closed-lane reopens and missing
evidence; cheap rollout budget exists; rejection-sampling SFT has plateaued; the adapter beats deterministic baselines
in shadow mode. Until then the standing verdict is `RLVR_DEFERRED_UNTIL_PRIMITIVE_REWARD_STABLE`.

## 4. What exactly should the adapter output?

A bounded search spec (`SearchRow` proposal) — **not** source code and **not** a promotion decision:

```json
{ "lane": "...", "primitive": "...", "hypothesis": "...",
  "search_space": { "knobs": ["..."], "bounds": { "...": ["..."] } },
  "required_evidence": ["..."], "stop_rules": ["..."] }
```

Valid lanes mirror the explorer/ledger: `decode_policy`, `prefill_role_policy`, `native_codegen_microprimitive`,
`cross_shape`. The adapter **must not**: decide a kernel is fast; flip defaults; bypass correctness or ISA/resource
checks; treat teacher-forced accuracy as success; train on holdout labels; generate broad free-form kernels by default;
reopen closed attention/GEMV lanes without a new audit; or use RLVR before the supervised strict-JSON gate works.

## 5. How does the deterministic runner consume the output?

The proposed spec enters the unified runner **before candidate generation** (runner design §3a): it is `--dry-run`
validated by the deterministic scorer (schema/lane/primitive/evidence/stop-rule/closed-lane), and only a valid spec is
handed to the lane backend. Candidate generation, the cost-ordered gate stack, and lane authority are unchanged.

## 6. What gates remain authoritative?

The deterministic lifecycle gates only: harness contract (13-field), route/lifecycle fire, materialization/ABI, ISA/
resource, correctness (byte-identical / rel_rmse), and **W==D (decode) / whole-prefill (prefill)** transfer. The
microprimitive lane is non-promotion (ISA + local correctness, never W==D). A harness recommends; the owner flips
defaults. `DETERMINISTIC_HARNESS_REMAINS_AUTHORITY`.

## 7. What dataset/scorer/eval must exist before training?

Not yet built (this was doc-only). Prerequisites, in order:

1. **Dataset** `bench/qk-primitive-space-adapter/dataset-v0/` (`train.jsonl`, `holdout.jsonl`, `summary.json`,
   `README.md`) — rows from accepted wins (buffer-identity KV read, owned tile, Q4K GEMV warp, kv_proj fix), refuted
   lanes (B4 combine-only, opaque append/cache-identity theories, q8/int-dot nulls, nosync prefill false wins),
   current specs/ledgers, ISA facts, and harness-discipline corrections. **Family split** holdout (not random) to
   prevent near-duplicate leakage. → `PRIMITIVE_SPACE_DATASET_READY` / `_BLOCKED`.
2. **Deterministic scorer** `extra/qk_primitive_space_scorer.py` (axes: `parse_valid`, `schema_ok`, `lane_valid`,
   `primitive_valid`, `evidence_complete`, `stop_rules_complete`, `closed_lane_respected`, `harness_authority_correct`,
   `strict_pass`). No LLM-as-judge. → `PRIMITIVE_SPACE_SCORER_READY` / `_INSUFFICIENT`.
3. **Baselines** on holdout (deterministic mechanism-prior; base Qwen3-8B strict-JSON; optional structured-prompt)
   with Wilson intervals and per-axis rates. → `BASELINE_PRIMITIVE_SPACE_MEASURED` / `BASE_MODEL_NOT_SCHEMA_STABLE`.

Then the smallest viable adapter (Qwen3-8B-Q4_K_M, suffix-cache internal adapter, `last1_ffn`, rank 4, alpha 8;
free-generation strict JSON @ temp 0; teacher-forced loss diagnostic only), evaluated as a **proposer**
(`LORA_PRIMITIVE_SPACE_PROPOSER_PASS` / `_FORMAT_ONLY` / `_FAIL`), then shadow mode
(`SHADOW_PRIMITIVE_PROPOSER_USEFUL` / `_NOT_USEFUL`), then rejection-sampling SFT (`RS_SFT_PRIMITIVE_LOOP_READY` /
`_NOT_READY`).

## 8. What would unlock RLVR later?

The defer criteria in §3 all true, with a deterministic reward whose components are: `+` parse/schema validity, `+`
valid lane/primitive, `+` evidence completeness, `+` dry-run spec acceptance, `+` eventual search utility; `-`
closed-lane reopen, `-` missing harness authority, `-` hallucinated tool/unsupported knob, `-` holdout
leakage/artifact mismatch. Reward must be deterministic and stable, not LLM-judged.

## Docs changed

- `structure/Development/performance-primitive-research-principles.md` — new principle #13 ("Learned models propose
  primitive search spaces; deterministic lifecycle gates decide").
- `structure/Development/session-handoff.md` — new top status note + next task.
- `docs/oracle-guided-gpu-primitive-explorer-result-20260623.md` — §15 "Learning layer: primitive-space proposer".
- `docs/oracle-guided-gpu-primitive-explorer-runner-design-20260623.md` — §3a "Learned spec proposer (optional
  front-end)".
- `docs/project-wide-machine-search-roadmap-result-20260623.md` — §6b "Learning lane — primitive-space proposer".
- `docs/amd-decode-kernel-optimization-flywheel.md` — 2026-06-23 superseding note (closing link = proposal under
  deterministic gates, not kernel triage).
- `docs/qwen-json-eval-objective-scope.md` — pointer: strict-JSON machinery's second consumer (primitive specs).
- this result doc (new).

Historical benchmark numbers and prior results were **preserved** — only superseding notes/pointers were added.

## Source / default files changed

**None.** No `tinygrad/` source, no kernel, no default flip, no adapter training, no RLVR, no machine-search run.

## Recommended next executable task

Build `bench/qk-primitive-space-adapter/dataset-v0/` (family-split train/holdout from the ledger + refutation docs) and
the deterministic scorer `extra/qk_primitive_space_scorer.py` (Phases 1–2). That is the first buildable step and gates
everything after it.
