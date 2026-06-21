# Fix Scope — Verdict SSOT (decode-eval verdict states)

Date: 2026-06-21
Closes: `docs/repo-principles-audit-20260621.md` finding **C2** (verdict-enum drift) + **I1** (stringly-typed states).
Status: **SCOPE — not yet implemented.** For audit before execution.

## Objective
Make the decode-eval **verdict** set a single typed source of truth, so the code, the JSON schema, the lifecycle
policy map, and the contract doc cannot drift, and an invalid verdict cannot be emitted silently. Behavior-preserving
(**NFC**): every emitted verdict string stays byte-identical.

## Problem (evidence)
The legal verdict set is defined **5 times and they already disagree**:

| source | location (by symbol) | count | set |
|---|---|---|---|
| code (the producer) | `extra/qk_decode_eval.py` → `classify()` returns + the dry-run/runner-error `res["verdict"]` sets | **10** | SELFTEST_PASS, PASS_ORACLE_LOCAL_AB, FAIL_ORACLE_LOCAL_AB, FAIL_CORRECTNESS, FAIL_LOCAL_AB, NEEDS_GPU_STATE_TOOLING, PASS_PROMOTE, LOCAL_PASS_WD_FAIL, PASS_OPT_IN, REST |
| JSON schema enum | `bench/qk-decode-eval/schema.json` → `verdict.enum` | **13** | the 10 + `FAIL_WD`, `FAIL_REPRODUCIBILITY`, `NEEDS_BESPOKE_TEMPLATE` |
| lifecycle policy map | `bench/qk-lifecycle-search/search_policy.json` → `verdict_to_lifecycle_decision` | **13** | same 13 |
| contract doc | `bench/qk-lifecycle-search/evaluator_contract.json` → `verdict_interpretation` | **10 (different)** | omits the 3 oracle/selftest verdicts, includes the 3 dead ones |
| human-facing README | `bench/qk-decode-eval/README.md` → verdict list | **11** | 8 live (missing `PASS_ORACLE_LOCAL_AB`, `FAIL_ORACLE_LOCAL_AB`) + the 3 dead `FAIL_WD`/`FAIL_REPRODUCIBILITY`/`NEEDS_BESPOKE_TEMPLATE` |

Three consequences:
1. **3 dead enum values** — `FAIL_WD`, `FAIL_REPRODUCIBILITY`, `NEEDS_BESPOKE_TEMPLATE` are emitted in **0** code
   locations (verified). They are superseded: `FAIL_WD`→ never produced (a WD miss with a local pass returns
   `LOCAL_PASS_WD_FAIL` in `classify`); `FAIL_REPRODUCIBILITY`→ replaced by `NEEDS_GPU_STATE_TOOLING` (repro-band branch);
   `NEEDS_BESPOKE_TEMPLATE`→ a **lifecycle-loop** prune concept (`qk_lifecycle_search_loop.py` `PRUNE_NEEDS_TEMPLATE`),
   not a decode-eval **run** verdict.
2. **No emit-time validation** — `classify` returns bare string literals; the enum is only checked by the opt-in
   `--validate` subcommand in `qk_decode_eval.py`, so a typo (`"PASS_PROMOT"`) flows into an artifact silently.
3. **Adding/renaming a verdict requires editing 5 files** with nothing asserting they agree.

The repo already has the right pattern: `extra/qk_modes.py` defines `PolicyMode`/`PrimitiveMode`/`PromptFormat` as
`(str, Enum)` with `.value == legacy string` and `choices()` helpers — explicitly "behavior-preserving (NFC)". Verdicts
should join it.

## Design
**SSOT = a `Verdict(str, Enum)` in `extra/qk_modes.py`** (the existing enum home), plus a verdict→lifecycle mapping,
plus a machine-enforced sync test. The two JSON files remain *data* (they are read by tooling/ledgers) but are
**asserted equal to the enum** by a test — "human-facing AND machine-enforced" (`coding-principles.md` §Human-Facing).

### Dead-verdict disposition (decision)
**Remove** `FAIL_WD` and `FAIL_REPRODUCIBILITY` from all five sources (pure renames, never emitted, no artifact uses
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
# verdict -> lifecycle decision. Values are copied VERBATIM from the live
# bench/qk-lifecycle-search/search_policy.json:verdict_to_lifecycle_decision (NFC -- do NOT reword them).
VERDICT_LIFECYCLE: dict[str, str] = {
  Verdict.PASS_PROMOTE: "candidate_promotable_owner_decision",
  Verdict.PASS_OPT_IN: "opt_in_candidate_banked",
  Verdict.PASS_ORACLE_LOCAL_AB: "reference_oracle_target_informs_codegen_non_promotable",
  Verdict.LOCAL_PASS_WD_FAIL: "refute_for_promotion_bank_learning",
  Verdict.FAIL_CORRECTNESS: "refute_candidate",
  Verdict.FAIL_LOCAL_AB: "refute_candidate",
  Verdict.FAIL_ORACLE_LOCAL_AB: "reference_oracle_does_not_beat_comparator",
  Verdict.NEEDS_GPU_STATE_TOOLING: "stop_search_needs_gpu_state",
  Verdict.SELFTEST_PASS: "selftest_only_not_perf",
  Verdict.REST: "bank_baseline_or_rest",
}
```
**NFC guard (do not skip):** the 10 values above are the *current* live `search_policy.json` strings, captured
2026-06-21. The implementer must still diff against the live file at implement time and use whatever is there
verbatim; the new `test/unit/test_verdict_ssot.py` (§7) asserts `VERDICT_LIFECYCLE == search_policy.verdict_to_lifecycle_decision`
for the 10 kept verdicts, so any reword is caught mechanically. The 3 dead verdicts being **removed** carry these live
mappings, which disappear with them: `FAIL_WD`→`refute_candidate`, `FAIL_REPRODUCIBILITY`→`stop_search_needs_measurement`,
`NEEDS_BESPOKE_TEMPLATE`→`stop_search_needs_template` (the last moves to the loop prune namespace, see §4).

### 2. `extra/qk_decode_eval.py` — return enum members
(Line numbers below are *approximate hints only* — locate by symbol; the file is edited often and lines drift.)
- Add to the imports block (top of file, ~L20-22): `from extra.qk_modes import Verdict, VERDICTS`.
- **`classify()`** (~L146): change every `return "STRING", reason` → `return Verdict.STRING, reason` (the ~10 `return`
  statements that produce a verdict literal — find them all by scanning `classify` for string returns; do not rely on a
  fixed count).
- **dry-run verdict assignment** (in `evaluate()`, the `if dry:` branch, ~L186): `res["verdict"] = "REST"` → `Verdict.REST`.
- **runner-error verdict assignment** (in `main()`'s except block, ~L285): the `"verdict": "REST"` literal → `Verdict.REST`.
- **NFC note:** `res["verdict"]` is JSON-serialized in `emit()`; since `Verdict` subclasses `str`,
  `json.dumps(Verdict.REST) == '"REST"'` and `Verdict.REST == "REST"` is `True`, so the `verdict_matches_expected`
  comparison (in `evaluate()`, against the string in `candidates.json`) and the schema `validate()` subcommand are
  unaffected.
- **Hardening (recommended), in `emit()`** before writing: `assert res["verdict"] in VERDICTS, res["verdict"]` — turns a
  future typo into a loud emit-time failure, not a silent artifact (this is the "encode the invariant" half of I1).

### 3. `bench/qk-decode-eval/schema.json` — align enum
In the `verdict.enum` block (~L71-75), replace the 13-value list with the 10 `Verdict` values (drop `FAIL_WD`,
`FAIL_REPRODUCIBILITY`, `NEEDS_BESPOKE_TEMPLATE`). Order to match the `Verdict` declaration. (No committed artifact uses
the removed values — verified 0 occurrences in code.)

### 4. `bench/qk-lifecycle-search/search_policy.json` — align map
In `verdict_to_lifecycle_decision`, drop the 3 dead keys (keep the 10, values unchanged). If the loop still needs the
`NEEDS_BESPOKE_TEMPLATE` decision (`stop_search_needs_template`), move it to a **sibling** `prune_to_lifecycle_decision`
object keyed by the loop's `PRUNE_*` outcomes — do not leave a run-verdict key the producer never emits. (Wiring the
loop to read the new key is out of scope; just relocate the row so the run-verdict map stays == `VERDICTS`.)

### 5. `bench/qk-lifecycle-search/evaluator_contract.json` — align doc
In `verdict_interpretation`, make the documented set exactly the 10 `Verdict` values (add the missing
`SELFTEST_PASS`/`PASS_ORACLE_LOCAL_AB`/`FAIL_ORACLE_LOCAL_AB`; drop the 3 dead). Doc-only.

### 6. `bench/qk-decode-eval/README.md` — fix the stale verdict list
The human-facing verdict list (the `PASS_PROMOTE · … · SELFTEST_PASS` line, ~L37-38) currently includes the 3 dead
verdicts (`FAIL_WD`, `FAIL_REPRODUCIBILITY`, `NEEDS_BESPOKE_TEMPLATE`). Update it to the 10 `Verdict` values **or**
replace the inline list with a pointer to the SSOT (`extra/qk_modes.py:Verdict`) so it cannot re-drift. Doc-only.

### 7. `test/unit/test_verdict_ssot.py` (new, ~30 lines) — machine enforcement
The "machine-enforced" half (no GPU, no tinygrad import). Asserts:
- `set(schema.json verdict.enum) == VERDICTS` (schema mirrors the SSOT);
- `set(search_policy.verdict_to_lifecycle_decision keys) == VERDICTS` **and** the kept mapping equals
  `VERDICT_LIFECYCLE` value-for-value (catches an NFC reword of a lifecycle string);
- `set(evaluator_contract.verdict_interpretation keys) == VERDICTS`;
- `bench/qk-decode-eval/README.md` contains **none** of the dead verdict names (`FAIL_WD`, `FAIL_REPRODUCIBILITY`,
  `NEEDS_BESPOKE_TEMPLATE`) — the stale-doc guard;
- every verdict literal in `qk_decode_eval.py` is a member of `VERDICTS` — AST-scan for `return "X"` / `"verdict": "X"`
  string constants and assert `X in VERDICTS` (catches a future hardcoded typo).

## NFC proof obligations (for the implementer + auditor)
1. Re-run `decode_eval --candidate fused_flash_concrete_gate`; the emitted artifact's `verdict` string and
   `verdict_matches_expected` are unchanged vs the pre-fix run (byte-diff the `verdict`/`stop_reason` fields).
2. `decode_eval --list`, `qk_lifecycle_search_loop.py --help`, `qk_candidate_template_gen.py --help` still run.
3. **Scoped** dead-verdict grep — only **live authority files** must be clean of the 3 dead names as *valid
   decode-eval run verdicts*:
   `git grep -nE 'FAIL_WD|FAIL_REPRODUCIBILITY|NEEDS_BESPOKE_TEMPLATE' -- extra/ bench/qk-decode-eval/ bench/qk-lifecycle-search/`
   may return **only** the relocated loop-prune mapping (`search_policy.prune_to_lifecycle_decision` / the loop's
   `PRUNE_*` namespace) — never a decode-eval run-verdict enum/map/list/return. **Historical/dated docs are
   exempt:** `docs/*-2026*.md` (including this scope and the audit) and `structure/Development/*` legitimately mention
   the old names as provenance; do not grep or "fix" them. The contract is: no **canonical/live** file exposes the 3
   as valid decode-eval verdicts; historical docs may name them as superseded.
4. `policy_consistency_check.py` PASS (unchanged).

## Acceptance gates (what codex audits)
- G1 `Verdict` SSOT added to `qk_modes.py`; values == the 10 currently-emitted verdicts (no more, no fewer).
- G2 `classify()` + the 2 direct sets return `Verdict` members; no bare verdict string literal remains in `qk_decode_eval.py`.
- G3 schema enum, `search_policy` map keys, `evaluator_contract` interpretation **and** the `README.md` verdict list all
  agree with `VERDICTS` (the new test passes, incl. the README stale-name guard and the `VERDICT_LIFECYCLE` value check).
- G4 the 3 dead verdicts removed from the decode-eval **run** namespace (or explicitly `RESERVED_VERDICTS` if the
  auditor chose retain); `NEEDS_BESPOKE_TEMPLATE`'s mapping relocated to the prune namespace, not deleted.
- G5 NFC: a candidate re-run's `verdict` byte-identical; `--validate` still passes; policy guard PASS.
- G6 commit discipline: `[test]` prefix (touches `extra/` + `bench/` + `test/`), tagged `NFC`, single concern.
- G7 no `tinygrad/`/model/default change; no behavior/output change beyond the removed-dead-verdict enum entries.

## Migration order
1. Add `Verdict`/`VERDICT_LIFECYCLE` to `qk_modes.py` (diff `VERDICT_LIFECYCLE` against live `search_policy.json`; use live values verbatim).
2. Switch `qk_decode_eval.classify` + the dry-run/runner-error sets to enum members; add the `emit()` assert.
3. Align `schema.json`, `search_policy.json`, `evaluator_contract.json`, **and `bench/qk-decode-eval/README.md`**.
4. Add `test/unit/test_verdict_ssot.py`; run it (and the scoped dead-verdict grep on live files only).
5. NFC re-run + byte-diff a candidate's `verdict`/`stop_reason`; commit `[test] NFC - Verdict SSOT ...`.

## Rollback
Single `[test]` commit; revert restores the 4-way-drift status quo. No artifact migration (removed verdicts were never
emitted), so no data backfill is needed.

## Out of scope (same SSOT pattern, separate fixes — do NOT bundle)
Audit C3 (one GPU-boundary module), C4 (comparator constant), C5 (one contract-field list), C6 (`child_env` SSOT),
C7 (threshold constants), and the sprawl S1/bb5a collapse. Each is its own `[test]` change; this scope is verdicts only.

## Files cited
(Locate by symbol; approximate line hints only — the tree is edited often.)
- `extra/qk_modes.py` — enum home (after `PromptFormat`, ~L45); add `Verdict` / `VERDICTS` / `VERDICT_LIFECYCLE`.
- `extra/qk_decode_eval.py` — `classify()` (~L146), dry-run set in `evaluate()` (~L186), runner-error set in `main()`
  (~L285), `emit()` (~L258, add assert), `--validate` subcommand (unchanged).
- `bench/qk-decode-eval/schema.json` — `verdict.enum` block (~L71-75).
- `bench/qk-lifecycle-search/search_policy.json` — `verdict_to_lifecycle_decision` (+ new `prune_to_lifecycle_decision`).
- `bench/qk-lifecycle-search/evaluator_contract.json` — `verdict_interpretation`.
- `bench/qk-decode-eval/README.md` — the verdict list (~L37-38).
- `extra/qk_lifecycle_search_loop.py` — `PRUNE_NEEDS_TEMPLATE` (the correct home for the template-gap concept).
- new: `test/unit/test_verdict_ssot.py`.
