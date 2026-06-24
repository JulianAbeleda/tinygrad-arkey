# Repo Principles Audit (2026-06-21)

Exhaustive audit of the repo against its own authority docs:
`structure/Development/coding-principles.md`, `tinygrad-coding-overrides.md`, `structure-convention.md`,
`performance-primitive-research-principles.md`. Method: 3 parallel evidence-based auditors (centralization/DRY,
anti-sprawl/table-driven, portability/danger/invariants) + direct verification. Read-only except the clear
self-inflicted fixes applied this session (§5). Every finding carries `file:line` evidence.

## 0. Scorecard

| principle | grade | one-line |
|---|---|---|
| Structure convention (folders, cache) | **A** | all 6 `structure/` folders present; convention followed |
| Commit discipline (prefixes, NFC) | **A** | last 25 commits all valid `[test]`/`[docs]`; no `[test]` touches `tinygrad/`; checker enforces |
| Deep authority modules exist | **B+** | `qk_search_spec`, `qk_flywheel_cost_model`, `llm_eval_common`, `qk_modes` (typed enums), `qk_clock_pin`, `qk_harness_contract` are genuinely deep/table-driven |
| Contain dangerous power | **B** | `qk_clock_pin.pinned_peak` is a clean sudo/sysfs boundary — but 2 scripts write GPU power state outside it |
| Centralize authority | **C** | provenance helpers forked 4×, `DEV_SYS` 3×, comparator string ~20×, verdict-enum 4×, thresholds re-typed |
| DRY (knowledge, not lines) | **C** | the same ~40-line JSON-IO+verdict harness is cloned into 24–35 driver scripts that ignore `llm_eval_common` |
| Anti-sprawl / no-new-file | **C-** | half-refactored: good registries exist but `bb5a*`(23 files), tensile, flywheel-dataset, `*_ab` are clone-per-experiment |
| Encode invariants (typed states) | **C+** | `qk_modes` proves the pattern, but decode-eval **verdicts are bare strings**, only validated on-demand (enum drift already real) |
| Portable artifacts | **C** | 146 committed JSON carry absolute `/home/ubuntu` paths; 311 carry unrounded floats (all in **old provenance dirs**, none in live ledgers) |

**Headline:** the *authority layer* is well-shaped (centralized, typed, contained); the *experiment-driver layer* is
where the principles slip — clone-per-experiment sprawl and scattered provenance/comparator/verdict knowledge. The
recent harness-contract + active-surface work moved the right direction; the gap is finishing centralization and
collapsing the cloned harness.

## 1. Centralization / DRY violations (agent 1, verified)

| # | finding | evidence | sev |
|---|---|---|---|
| C1 | **Provenance helpers forked 4×** (`git_commit`/`dirty_tree`/`perf_state`/`hardware`) | `qk_harness_contract.py:44-56` (declared SSOT) vs `qk_decode_eval.py:34-42`, `qk_lifecycle_search_loop.py:43`, `qk_candidate_template_gen.py:32` | high → **FIXED §5** |
| C2 | **Verdict set + verdict→lifecycle in 4 places, drift real** | enum `schema.json:69-75`; string literals `qk_decode_eval.py:155-179`; `search_policy.json`; `evaluator_contract.json`. `classify()` emits 9 verdicts; enum lists 13 (`FAIL_WD`,`FAIL_REPRODUCIBILITY`,`NEEDS_BESPOKE_TEMPLATE` are **dead**) | high |
| C3 | **GPU device/clock path `DEV_SYS` in 3 files; read/write split** | `qk_decode_eval.py:31`(removed §5), `qk_harness_contract.py:33`, `qk_clock_pin.py:16` — pin *writes*, contract *reads*, no shared constant | high |
| C4 | **Comparator `"gqa_coop_vec"` hardcoded in ~20 files/ledgers** | `qk_decode_eval.py:145`, `qk_harness_contract.py:23,130`, 5+ `*_ab.py`, `binding_templates.json`, `search_policy.json`… | med |
| C5 | **3 independent "required artifact fields" lists** | `qk_harness_contract.py:36-41` (13) vs `schema.json:6-12` (~27) vs `evaluator_contract.json` (17) — `contract_audit` can say CONFORMS while jsonschema rejects | med |
| C6 | **`child_env` (DEV/JIT/PYTHONPATH) re-implemented 3×** | `qk_decode_eval.py:44`, `qk_lifecycle_search_loop.py:81` (omits `QK_MODEL`), `llm_generate.py:61` | med |
| C7 | **Thresholds (1.05/5%/7%/dNLL) re-typed as prose** | SSOT `candidates.json:thresholds_default`; restated in `evaluator_contract.json`, `templates.json`, code comments | med |
| C8 | **Model geometry `Hd/Hq/Hkv/MAXC` in 6 files** | `qk_decode_eval.py:128`, `qk_flash_decode.py:301`, 3 `*_ab.py` … | low |
| — | (MAXC 4608 vs 4096 is **NOT a bug** — `qk_flash_decode.py:330` documents 4608 = "divisible by all swept L"; different knowledge, correct to keep separate per the "duplication < wrong abstraction" rule) | verified | — |

## 2. Anti-sprawl violations (agent 2, verified)

`extra/*.py` is **182 files / 35.6k LOC, half-refactored.** Signal: **80/182 files carry their own argparse `main()`**;
**43 define `build_*`**; **25 redefine `write_json`** though `llm_eval_common.write_json` exists.

| family | files | ~LOC | verdict |
|---|---:|---:|---|
| `qk_amd_*` (incl. `bb5a*`) | 23 | 3973 | **clone-per-experiment** — sequential `_probe.py` over `tinygrad/renderer/amd/schedule.py`, each re-cloning the JSON/gate harness |
| `qk_flywheel_*` | 12 | 4699 | mixed — `cost_model` deep+reused; `dataset`/`targeted_outcomes` forked builders |
| `qk_tensile_*` | 8 | 854 | clone — launch/perf/shape_matrix = one shape-table runner split 3× |
| `qk_semantic_*` | 7 | 1383 | clone — schedule/bench/verdict = one pipeline split 3 ways |
| `amd_*two_ring*` | 6 | 975 | clone — 4 near-identical multi-ring probes |
| `qk_*_ab` | 9 | — | clone — identical header + flash import block repeated verbatim |

**Root cause (S1, high):** a ~40-line `read_json`/`write_json`/`verdict_row(phase,gate,next_action)` harness is hand-copied
across 24–35 driver scripts (e.g. `qk_amd_bb5a2_lowering_hook_probe.py:17-25` ≡ `qk_amd_bb5a7_performance_gate_probe.py:11-19`)
**without importing `llm_eval_common`** (`llm_eval_common.py:8,37,41`). **Positive:** the authority modules
(`qk_search_spec.py` 335L frozen-dataclass schema, `qk_flywheel_cost_model.py` imported not cloned by `shadow.py:25`)
are exactly the targets the cloners should route through.

## 3. Portability / dangerous-power / invariant violations (agent 3, verified)

| # | finding | evidence | sev |
|---|---|---|---|
| P1 | **Regenerators hardcode absolute ROOT** | `build_inventory.py:8`, `build_docs_index.py:8` `= "/home/ubuntu/tinygrad-arkey"` | high → **FIXED §5** |
| P2 | **146 committed JSON embed absolute checkout paths** | `bench/qk-ansor-transition-20260612/.../decision.json` `"out"/"policy"/"model": "/home/ubuntu/…"` (all in **old provenance dirs; 0 in live ledgers**) | high (scoped) |
| P3 | **311 committed JSON serialize unrounded f64** | same ansor `decision.json` `"stdev_tok_s": 0.12868662635539943` — violates golden-portability | med (scoped) |
| P4 | **GPU power-state written outside `qk_clock_pin`** | `qk_decode_q8_model_route_timing_audit.py:92-109` re-does `echo manual > pp_dpm_sclk` + `rocm-smi`; `qk_prefill_flash_perf.py:15-21` sets `--setperflevel high` with **no try/finally** (leaks on crash) | med |
| P5 | **Env-ordering violated in a trainer** | `llm_adapter_train.py:9` `from tinygrad import Tensor` at module top, before `configure_env` sets `DEV/JIT/QK_*` (line 18) — `llm_generate.py` is the correct lazy reference | med |
| I1 | **Verdicts are bare strings, not a closed type** | `qk_decode_eval.py:155-179`; enum only in `schema.json`, only checked by opt-in `--validate` — typo flows silently. `qk_modes.py:25-43` already has `(str,Enum)` pattern to join | low |

## 4. What the repo does WELL (adherence — keep these)

- **Structure convention** fully followed (6 folders; `cache/` is orientation not logs).
- **Commit discipline** machine-enforced — a commit-msg hook rejects bad prefixes (caught `[cleanup]` this session); no
  `[test]` ever touches `tinygrad/` core.
- **Deep modules over shallow:** `qk_search_spec` (frozen-dataclass SSOT + enum validators), `qk_flywheel_cost_model`
  (imported, not cloned), `llm_eval_common`, `qk_clock_pin` (the dangerous-power boundary), `qk_modes` (typed enums).
- **Dangerous power mostly contained:** `pinned_peak` documents what it mutates and restores `auto` in `finally`.
- **Perf-primitive discipline:** measure at the hardware boundary, in-model W==D authority, harness contract — the
  recent decode/harness/active-surface work models the principles well.
- **Provenance-as-asset:** refutation ledger + the active-surface inventory + this audit are the "record refutations"
  rule in action.

## 5. Fixes applied this session (self-inflicted, NFC)

1. **C1 provenance centralized** — `qk_harness_contract.py` is now the single source; `qk_decode_eval.py`,
   `qk_lifecycle_search_loop.py`, `qk_candidate_template_gen.py` import `git_commit`/`dirty_tree`/`perf_state`/
   `hardware` and the local copies + `DEV_SYS` const are removed. Verified: values correct, `--list`/CLIs OK, policy PASS.
2. **P1 absolute ROOT fixed** — `build_inventory.py`/`build_docs_index.py` derive `ROOT` from `__file__`. Proven NFC
   (`docs_index.json` byte-identical on regen; `inventory.json` only drops the now-deleted entries).

(These two were introduced in this session's harness-contract + active-surface work — the audit ate its own dog food.)

## 6. Prioritized recommendations (not applied — deliberate follow-ups)

| pri | action | closes | effort | risk |
|---|---|---|---|---|
| 1 | **Verdict SSOT:** add `Verdict(str,Enum)` to `qk_modes.py`; `classify()` returns members; generate `schema.json` enum + `search_policy.json` map from it; drop the 3 dead verdicts | C2, I1 | S | low |
| 2 | **De-clone the probe harness:** one `probe_harness.emit_verdict()` + reuse `llm_eval_common.{read,write}_json`; retrofit the 24–35 cloners | S1, sprawl | M | low (mechanical) |
| 3 | **One GPU-boundary module:** move `DEV_SYS`/`read_perf_state` into `qk_clock_pin`; route `q8_model_route_timing_audit` + `prefill_flash_perf` through `pinned_peak` (+ try/finally) | C3, P4 | S | low |
| 4 | **Comparator + thresholds + contract-fields as constants** imported from `qk_harness_contract`/`candidates.json`; ledgers reference keys not values | C4, C5, C7 | M | low |
| 5 | **Collapse `bb5a*` (23→1 step-table) + tensile/semantic/flywheel forks** into table-driven runners | sprawl | L | med |
| 6 | **Portable old artifacts:** add a rounding + ROOT-relativize pass to artifact writers; regenerate the 146/311 leaking provenance JSON (or accept as frozen history) | P2, P3 | M | low |
| 7 | **`child_env` SSOT** + lazy tinygrad import in `llm_adapter_train.py` | C6, P5 | S | low |

## 7. Acceptance
Audit covers all 4 principles docs with file:line evidence; scorecard + per-cluster violation tables + adherence list +
applied fixes + prioritized backlog. No `tinygrad/`/model/default change. Self-inflicted high-severity items fixed NFC;
structural refactors are specified for deliberate follow-up. Backing data: the 3 auditor reports (this session) +
`bench/qk-active-surface-reduction/inventory.json`.

## 8. Resolution status — Contract Centralization Sequence (2026-06-21, COMPLETE)
Every finding is resolved with a tested SSOT, documented as already-centralized, or accepted-as-frozen with a guard:

| finding | resolution | commit | test |
|---|---|---|---|
| C1 provenance fork | centralized in `qk_harness_contract`; 3 consumers import | `deccc62c7` | — |
| C2/I1 verdict drift + dead values | `Verdict(str,Enum)` SSOT + emit-time assert | `c4c4caeae` | `test_verdict_ssot` |
| C3/P4 GPU perf-state | `qk_clock_pin` owns read + both pin idioms; routed; 1 documented exception | `7b03f585a` | — |
| C4 comparator | `DECODE_COMPARATOR` mirror + drift test | `755dd4882` | `test_comparator_ssot` |
| C7 thresholds | **already centralized** (`candidates.json:thresholds_default`); no change | — | — |
| C5 artifact fields | 3 lists distinguished; C⊆B + A-disjoint enforced | `60fe03bc0` | `test_artifact_contract_fields` |
| C6 child env | one `child_env` + `DEFAULT_MODEL` | `c80655171` | `test_child_env_ssot` |
| S1 probe IO clone | de-cloned all 25 → `qk_probe_harness.probe_io`; guard | `3e0e72884`/`5cbe498a1`/`8cefec13e` | `test_probe_harness` |
| S1 verdict-template | **left by principle** (bespoke per-probe, not a clone) | `218945eb9` | — |
| P1 absolute ROOT (tools) | `__file__`-relative | `deccc62c7` | — |
| **P2 absolute paths (artifacts)** | **durable surface guarded; 82 historical artifacts accepted as frozen** | this commit | `test_artifact_portability` |
| **P3 unrounded floats (artifacts)** | **golden-scoped rule; the historical dumps are not golden-locked (reproduce tests regenerate, they don't hash floats); no change** | this commit | — |

**P2/P3 decision (load-bearing):** the durable/live surface (ledgers + current generated maps) is checkout-path-clean
and now guarded by `test/unit/test_artifact_portability.py`. The ~82 tracked historical bench artifacts with absolute
paths are **deliberately not rewritten**: 10 of their dirs are golden-locked by reproduce tests
(`test_llm_training_data_probe`, `test_qk_experiment_matrix`, `test_qk_ansor_transition`, `test_qk_search_spec`,
`test_llm_rollout*`) and some absolute paths are **functional inputs** (`source_artifacts` read by `build_training_data`),
so a rewrite would break those tests and discard recorded data for a cosmetic gain — the same "don't excavate frozen,
golden-locked history" judgment applied to the verdict-template. The centralization backlog is now exhausted.
