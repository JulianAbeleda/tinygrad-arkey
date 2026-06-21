# Fix Scope — Verdict SSOT (decode-eval verdict states)

Date: 2026-06-21
Closes: `docs/repo-principles-audit-20260621.md` finding **C2** (verdict-enum drift) + **I1** (stringly-typed states).
Status: **SCOPE — not yet implemented.** For audit before execution.

## Objective
Make the decode-eval **verdict** set a single typed source of truth, so the code, the JSON schema, the lifecycle
policy map, and the contract doc cannot drift, and an invalid verdict cannot be emitted silently. Behavior-preserving
(**NFC**): every emitted verdict string stays byte-identical.

## Problem (evidence)
The legal verdict set is defined **4 times and they already disagree**:

| source | file:line | count | set |
|---|---|---|---|
| code (the producer) | `extra/qk_decode_eval.py:148-170` (`classify`) + direct sets `:186,245,285` | **10** | SELFTEST_PASS, PASS_ORACLE_LOCAL_AB, FAIL_ORACLE_LOCAL_AB, FAIL_CORRECTNESS, FAIL_LOCAL_AB, NEEDS_GPU_STATE_TOOLING, PASS_PROMOTE, LOCAL_PASS_WD_FAIL, PASS_OPT_IN, REST |
| JSON schema enum | `bench/qk-decode-eval/schema.json:71-75` | **13** | the 10 + `FAIL_WD`, `FAIL_REPRODUCIBILITY`, `NEEDS_BESPOKE_TEMPLATE` |
| lifecycle policy map | `bench/qk-lifecycle-search/search_policy.json` `verdict_to_lifecycle_decision` | **13** | same 13 |
| contract doc | `bench/qk-lifecycle-search/evaluator_contract.json` `verdict_interpretation` | **10 (different)** | omits the 3 oracle/selftest verdicts, includes the 3 dead ones |

Three consequences:
1. **3 dead enum values** — `FAIL_WD`, `FAIL_REPRODUCIBILITY`, `NEEDS_BESPOKE_TEMPLATE` are emitted in **0** code
   locations (verified). They are superseded: `FAIL_WD`→ never produced (a WD miss with a local pass returns
   `LOCAL_PASS_WD_FAIL` `:163`); `FAIL_REPRODUCIBILITY`→ replaced by `NEEDS_GPU_STATE_TOOLING` `:159`;
   `NEEDS_BESPOKE_TEMPLATE`→ a **lifecycle-loop** prune concept (`qk_lifecycle_search_loop.py` `PRUNE_NEEDS_TEMPLATE`),
   not a decode-eval **run** verdict.
2. **No emit-time validation** — `classify` returns bare string literals; the enum is only checked by the opt-in
   `--validate` subcommand (`qk_decode_eval.py:261`), so a typo (`"PASS_PROMOT"`) flows into an artifact silently.
3. **Adding/renaming a verdict requires editing 4 files** with nothing asserting they agree.

The repo already has the right pattern: `extra/qk_modes.py` defines `PolicyMode`/`PrimitiveMode`/`PromptFormat` as
`(str, Enum)` with `.value == legacy string` and `choices()` helpers — explicitly "behavior-preserving (NFC)". Verdicts
should join it.

## Design
**SSOT = a `Verdict(str, Enum)` in `extra/qk_modes.py`** (the existing enum home), plus a verdict→lifecycle mapping,
plus a machine-enforced sync test. The two JSON files remain *data* (they are read by tooling/ledgers) but are
**asserted equal to the enum** by a test — "human-facing AND machine-enforced" (`coding-principles.md` §Human-Facing).

### Dead-verdict disposition (decision)
**Remove** `FAIL_WD` and `FAIL_REPRODUCIBILITY` from all four sources (pure renames, never emitted, no artifact uses
them — verified 0 occurrences). **Move** `NEEDS_BESPOKE_TEMPLATE` out of the decode-eval run-verdict namespace: it is a
loop-level prune outcome and already exists as `PRUNE_NEEDS_TEMPLATE` in `qk_lifecycle_search_loop.py`; the decode-eval
`Verdict` enum and `schema.json` must not list it. If `search_policy.json` still needs a row for it (loop maps prune
outcomes), keep it in a **separate** `prune_to_lifecycle_decision` key, not in `verdict_to_lifecycle_decision`.

> If the auditor prefers to *retain* the 3 as reserved/aspirational rather than remove, the alternative is a
> `RESERVED_VERDICTS` frozenset in `qk_modes.py` that the sync test treats as allowed-but-unemitted. Recommendation:
> **remove** (they are drift, not roadmap). Flag this as the one judgment call for the audit.

## Exact changes (file-by-file)

### 1. `extra/qk_modes.py` (+~22 lines) — the SSOT
Add after `PromptFormat` (after line 45):
```python
class Verdict(str, Enum):
  """decode_eval per-run verdicts (single source of truth; .value == the legacy string -> NFC)."""
  PASS_PROMOTE = "PASS_PROMOTE"
  PASS_OPT_IN = "PASS_OPT_IN"
  PASS_ORACLE_LOCAL_AB = "PASS_ORACLE_LOCAL_AB"
  LOCAL_PASS_WD_FAIL = "LOCAL_PASS_WD_FAIL"
  FAIL_CORRECTNESS = "FAIL_CORRECTNESS"
  FAIL_LOCAL_AB = "FAIL_LOCAL_AB"
  FAIL_ORACLE_LOCAL_AB = "FAIL_ORACLE_LOCAL_AB"
  NEEDS_GPU_STATE_TOOLING = "NEEDS_GPU_STATE_TOOLING"
  SELFTEST_PASS = "SELFTEST_PASS"
  REST = "REST"

VERDICTS: frozenset[str] = frozenset(v.value for v in Verdict)
# verdict -> lifecycle decision (the authority the loop's search_policy.json must mirror)
VERDICT_LIFECYCLE: dict[str, str] = {
  Verdict.PASS_PROMOTE: "owner_decision_required",
  Verdict.PASS_OPT_IN: "opt_in_candidate_banked",
  Verdict.PASS_ORACLE_LOCAL_AB: "reference_oracle_target_informs_codegen_non_promotable",
  Verdict.LOCAL_PASS_WD_FAIL: "refute_for_promotion_bank_learning",
  Verdict.FAIL_CORRECTNESS: "refute_candidate",
  Verdict.FAIL_LOCAL_AB: "refute_candidate",
  Verdict.FAIL_ORACLE_LOCAL_AB: "refute_candidate",
  Verdict.NEEDS_GPU_STATE_TOOLING: "needs_gpu_state_tooling",
  Verdict.SELFTEST_PASS: "selftest_only_no_perf",
  Verdict.REST: "bank_baseline_or_rest",
}
```
(The `VERDICT_LIFECYCLE` values must be copied from the **current** `search_policy.json:verdict_to_lifecycle_decision`
during implementation so the mapping is byte-preserved — the implementer reads the live values, does not invent them.)

### 2. `extra/qk_decode_eval.py` — return enum members
- Add to imports (near line 21): `from extra.qk_modes import Verdict`.
- `classify()` (`:148-170`): change each `return "STRING", reason` → `return Verdict.STRING, reason`. 10 call sites
  (`:148,151,152,154,156,159,161,163,167,169,170`).
- Direct sets: `:186` `res["verdict"] = "REST"` → `Verdict.REST`; `:285` runner-error `"verdict": "REST"` → `Verdict.REST`.
- **NFC note:** `res["verdict"]` is JSON-serialized (`emit()`); since `Verdict` subclasses `str`,
  `json.dumps(Verdict.REST) == '"REST"'` and `Verdict.REST == "REST"` is `True`, so `verdict_matches_expected`
  (`:246`, compares to the string in `candidates.json`) and the schema `validate()` (`:261`) are unaffected.
- (Optional hardening, same file) in `emit()` assert `res["verdict"] in VERDICTS` before writing — turns a typo into a
  loud failure at emit time, not on opt-in validate.

### 3. `bench/qk-decode-eval/schema.json:71-75` — align enum
Replace the 13-value `enum` with the 10 `Verdict` values (drop `FAIL_WD`, `FAIL_REPRODUCIBILITY`,
`NEEDS_BESPOKE_TEMPLATE`). Order to match `Verdict` declaration. (No artifact uses the removed values — verified.)

### 4. `bench/qk-lifecycle-search/search_policy.json` — align map
`verdict_to_lifecycle_decision`: drop the 3 dead keys (keep the 10). If the loop needs `NEEDS_BESPOKE_TEMPLATE`, add a
sibling `prune_to_lifecycle_decision` object for `PRUNE_*` outcomes (out-of-scope to wire; just don't leave it under
the run-verdict key).

### 5. `bench/qk-lifecycle-search/evaluator_contract.json` `verdict_interpretation` — align doc
Make the documented set exactly the 10 `Verdict` values (add the missing `SELFTEST_PASS`/`PASS_ORACLE_LOCAL_AB`/
`FAIL_ORACLE_LOCAL_AB`; drop the 3 dead). Doc-only.

### 6. `test/unit/test_verdict_ssot.py` (new, ~30 lines) — machine enforcement
The "machine-enforced" half (no GPU). Asserts:
- `set(json schema enum) == VERDICTS` (schema mirrors the SSOT);
- `set(search_policy verdict_to_lifecycle_decision keys) == VERDICTS` and each maps to a non-empty decision == `VERDICT_LIFECYCLE`;
- `set(evaluator_contract verdict_interpretation keys) == VERDICTS`;
- every literal returned by `classify` is a `Verdict` member — implemented by AST-scanning `qk_decode_eval.py` for
  `return "X"`/`"verdict": "X"` string constants and asserting `X in VERDICTS` (catches a future hardcoded typo).

## NFC proof obligations (for the implementer + auditor)
1. Re-run `decode_eval --candidate fused_flash_concrete_gate`; the emitted artifact's `verdict` string and
   `verdict_matches_expected` are unchanged vs the pre-fix run (byte-diff the `verdict`/`stop_reason` fields).
2. `decode_eval --list`, `qk_lifecycle_search_loop.py --help`, `qk_candidate_template_gen.py --help` still run.
3. `git grep -n 'FAIL_WD\|FAIL_REPRODUCIBILITY\|NEEDS_BESPOKE_TEMPLATE'` returns only the (now-updated) loop prune
   namespace, no decode-eval run path.
4. `policy_consistency_check.py` PASS (unchanged).

## Acceptance gates (what codex audits)
- G1 `Verdict` SSOT added to `qk_modes.py`; values == the 10 currently-emitted verdicts (no more, no fewer).
- G2 `classify()` + the 2 direct sets return `Verdict` members; no bare verdict string literal remains in `qk_decode_eval.py`.
- G3 schema enum, `search_policy` map keys, and `evaluator_contract` interpretation all == `VERDICTS` (the new test passes).
- G4 the 3 dead verdicts removed from the decode-eval run namespace (or explicitly `RESERVED_VERDICTS` if the auditor chose retain).
- G5 NFC: a candidate re-run's `verdict` byte-identical; `--validate` still passes; policy guard PASS.
- G6 commit discipline: `[test]` prefix (touches `extra/` + `bench/` + `test/`), tagged `NFC`, single concern.
- G7 no `tinygrad/`/model/default change; no behavior/output change beyond the removed-dead-verdict enum entries.

## Migration order
1. Add `Verdict`/`VERDICT_LIFECYCLE` to `qk_modes.py` (copying live mapping values from `search_policy.json`).
2. Switch `qk_decode_eval.classify` + direct sets to enum members; add the `emit()` assert.
3. Align the 3 JSON files.
4. Add `test/unit/test_verdict_ssot.py`; run it.
5. NFC re-run + diff; commit `[test] NFC - Verdict SSOT ...`.

## Rollback
Single `[test]` commit; revert restores the 4-way-drift status quo. No artifact migration (removed verdicts were never
emitted), so no data backfill is needed.

## Out of scope (same SSOT pattern, separate fixes — do NOT bundle)
Audit C3 (one GPU-boundary module), C4 (comparator constant), C5 (one contract-field list), C6 (`child_env` SSOT),
C7 (threshold constants), and the sprawl S1/bb5a collapse. Each is its own `[test]` change; this scope is verdicts only.

## Files cited
- `extra/qk_modes.py:25-49` (enum home) · `extra/qk_decode_eval.py:148-170,186,245,261,285` (producer + validate)
- `bench/qk-decode-eval/schema.json:71-75` (enum) · `bench/qk-lifecycle-search/search_policy.json` (`verdict_to_lifecycle_decision`)
- `bench/qk-lifecycle-search/evaluator_contract.json` (`verdict_interpretation`) · `extra/qk_lifecycle_search_loop.py` (`PRUNE_NEEDS_TEMPLATE`, the correct home for the template-gap concept)
- new: `test/unit/test_verdict_ssot.py`
