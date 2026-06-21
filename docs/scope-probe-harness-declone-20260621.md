# Scope — Probe-Harness De-clone (audit S1)

Date: 2026-06-21
Closes (scope only): `docs/repo-principles-audit-20260621.md` finding **S1** (the cloned JSON-IO + verdict harness).
Status: **SCOPE — do not implement broadly in the centralization sequence.** Defines the shared helper + migration
waves for a later, segmented effort.

## ✅ COMPLETED 2026-06-21 (S1 IO de-clone done)
Executed Waves 0–2: built `extra/qk_probe_harness.py` (`probe_io` + `emit_verdict`), migrated **all 25** IO-cloners
(2 decode-tooling + 23 `qk_amd_bb5a*`/amd) to `read_json, write_json = probe_io(OUT)`, and added the
`test_no_new_local_write_json_clone` guard. **0 residual `read_json`/`write_json` clones in `extra/`** (only the helper
+ `llm_eval_common` define them). NFC by construction (the shared writer is byte-identical to the removed clone
format: `OUT/name`, `json.dumps(indent=2, sort_keys=True) + "\n"`). Commits: `3e0e72884` (helper), `5cbe498a1`
(Wave 1), `8cefec13e` (Wave 2 + guard). Tests: `test/unit/test_probe_harness.py` (4, incl. guard) green; full SSOT
suite green; policy guard PASS; no `tinygrad/` change.

**Wave 3 (verdict-template) intentionally NOT done — it is the wrong abstraction.** The 37 files' `{phase, gate_pass,
next_action, verdict, ...}` dicts share 3–4 conventional key names but each carries **bespoke domain keys** (e.g.
`tensile_oracle_tflops`, `authority_prefill`, `pure_tinygrad_reaches_60_tflops`) encoding *different* gate logic per
probe. Per `coding-principles.md` ("do not merge code that merely looks alike but encodes different concepts;
duplication is cheaper than the wrong abstraction"), forcing them through `emit_verdict(**bespoke)` would rename the
dict without centralizing real knowledge and is byte-risky on frozen provenance. `emit_verdict` is provided for
genuinely-uniform future use; the no-new-clone guard prevents new **IO** clones (the real duplicated knowledge).
S1 is therefore **complete** at its principled boundary.

## ROI framing (read first — it changes the priority)
The cloned harness lives in **provenance** scripts. All **25** IO-cloners (23 `qk_amd_bb5a*`, 2 `qk_decode_*`) and the
**37** files carrying the inline `verdict`/`gate_pass`/`next_action` template are `status: provenance` in
`bench/qk-active-surface-reduction/inventory.json` — historical probes cited by dated docs, **not live
evaluator/search code** and largely not re-run. So:
- **Durable value = the shared helper + a rule** that stops the NEXT probe from cloning (the principle's "a new
  experiment is a row/param, not a new `build_*`/`main()`"). This is the part worth doing now.
- **Retrofitting 25–37 frozen provenance probes is low-ROI churn** and byte-risky (each is a golden-ish artifact
  producer). It is explicitly **optional / adopt-going-forward**, not a required cleanup.

This scope therefore front-loads the helper + a tiny proof wave and makes the bulk retrofit opt-in.

## Cloned families (current tree, post active-surface reduction)

| family | files | clone | status |
|---|---:|---|---|
| `qk_amd_bb5a*` (backend-roadmap series) | 19 (+`broad_backend_roadmap` 843 LOC, +plans) = 23 | own `read_json(rel,default)`/`write_json(name,data)` + inline verdict dict | provenance |
| `qk_decode_*` tooling | 2 (`qk_decode_complete_tooling`, `qk_decode_native_tooling_readiness`) | same IO clone | provenance |
| verdict/`next_action` template (superset) | 37 | inline `{verdict, gate_pass, next_action, ...}` dict per script | provenance |

The SSOT already exists for **IO**: `extra/llm_eval_common.py` provides `write_json(path,data)`,
`read_json_object(path)`, `load_json`, `write_jsonl`, `value_stats`. It has **no** verdict helper — that must be added.

### The byte-identical gotcha (load-bearing)
The clones' `write_json(name, data)` takes a **name relative to a module `OUT` dir** and writes
`json.dumps(data, indent=2, sort_keys=True) + "\n"`. `llm_eval_common.write_json(path, data)` takes a **full path**.
A drop-in replacement must (a) adapt name→`OUT/name`, and (b) emit **identical bytes** (`indent=2, sort_keys=True`,
trailing `\n`). Any migrated probe that is re-run must produce a **byte-identical** artifact (golden gate).

## Proposed SSOT: `extra/qk_probe_harness.py` (one import for a probe)
A thin module that **re-exports** `llm_eval_common`'s IO (no second IO impl) and adds the missing verdict helper +
the OUT-relative adapters the clones use, so a probe does `from extra.qk_probe_harness import out_json, read_out, emit_verdict`:

```python
# extra/qk_probe_harness.py  (proposed)
from extra.llm_eval_common import write_json as _write_json, load_json
def out_json(out_dir, name, data):            # clone-compatible: name relative to out_dir, byte-identical writer
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / name, data)         # llm_eval_common writer MUST match indent=2,sort_keys=True,\n
def read_out(out_dir, name, default=None):
    p = out_dir / name
    return load_json(p) if p.exists() else default
def emit_verdict(phase, gate_pass, next_action, **extra):   # the inline template, once
    return {"phase": phase, "gate_pass": bool(gate_pass), "next_action": next_action, **extra}
```
**Precondition:** confirm `llm_eval_common.write_json` emits exactly `json.dumps(data, indent=2, sort_keys=True)+"\n"`.
If it differs, either align it (NFC-checked against its own callers) or have `out_json` do the dump directly. Resolve
this in Wave 0 before any migration.

## Migration waves (each = one isolated `[test] NFC` commit, golden-gated)

- **Wave 0 — build the helper (no probe edits).** Create `extra/qk_probe_harness.py`; add `test/unit/test_probe_harness.py`
  proving `out_json` is byte-identical to the clone writer on a fixture dict, and `emit_verdict` shape. Commit
  `[test] NFC - add qk_probe_harness shared probe IO+verdict helper`. **Gate:** byte-identical fixture; helper imports
  with no tinygrad.
- **Wave 1 — first small proof wave (2 files).** Migrate the 2 non-bb5a decode-tooling cloners
  (`extra/qk_decode_complete_tooling.py`, `extra/qk_decode_native_tooling_readiness.py`): replace their local
  `read_json`/`write_json` with the helper. **Gate:** re-run each (if runnable) → emitted artifact **byte-identical**
  (diff the JSON); else assert the helper writer matches the removed local writer on the script's own output dict.
- **Wave 2 — bb5a series (23), sub-batched.** `bb5a1-3`, `bb5a4-6`, `bb5a7-9`, then `bb5a_*_plan`/`broad_backend_roadmap`.
  One commit per sub-batch; same golden gate. (Lowest ROI — provenance; do only if the churn is judged worthwhile.)
- **Wave 3 — verdict-template-only files (the 37-minus-overlap).** Replace the inline `{verdict,gate_pass,next_action}`
  dict with `emit_verdict(...)`. Golden gate per file.

## First small wave (exact, to start)
`extra/qk_decode_complete_tooling.py` + `extra/qk_decode_native_tooling_readiness.py` only. Reason: non-bb5a, small,
isolated, and they prove the byte-identical golden path before touching the 23-file bb5a series.

## Gates (every wave)
- **Golden:** each migrated probe that is re-runnable emits a **byte-identical** artifact (JSON diff clean); a probe
  that is not re-runnable gets a unit fixture asserting the helper reproduces its prior output bytes.
- `policy_consistency_check.py` PASS; `decode_eval --list` / lifecycle / template CLIs unaffected (no live-code touched).
- One owning `[test]` prefix per wave; NFC (byte-proven, per the override's NFC rule); tree clean after each commit.
- **No live evaluator/search file is in any wave** (they don't clone IO — verified). No `tinygrad/`/model/default change.

## Stop conditions
- `llm_eval_common.write_json` cannot be made byte-identical to a clone without changing its own callers → keep the
  dump inside `out_json` instead of re-exporting; do not alter `llm_eval_common` callers in this effort.
- A probe is not re-runnable AND has no captured golden artifact to diff against → migrate behind a fixture test only,
  or skip (provenance; not worth unverifiable churn).
- The retrofit churn on frozen provenance is judged not worth it → ship **Wave 0 + Wave 1 only** (helper + proof) and
  leave the bb5a/verdict-template bulk as **adopt-going-forward** (new probes must import `qk_probe_harness`; a CI/lint
  rule or a `test_no_new_local_write_json` guard enforces it).

## Recommendation
Do **Wave 0 + Wave 1** now (helper + 2-file proof + a guard against new clones) — that captures ~all the durable value
(no future clones) cheaply. Treat Wave 2/3 (the 23 bb5a + 37 template provenance probes) as **optional**, low-priority
hygiene to be funded only if those files are otherwise being touched. This matches the audit's "duplication is cheaper
than churning frozen history" and the anti-re-sprawl rule's intent (stop the next clone, don't excavate old ones).

## Files cited
- SSOT-to-extend: `extra/llm_eval_common.py` (IO); new `extra/qk_probe_harness.py`, `test/unit/test_probe_harness.py`.
- Wave 1: `extra/qk_decode_complete_tooling.py`, `extra/qk_decode_native_tooling_readiness.py`.
- Wave 2: `extra/qk_amd_bb5a*.py` (19) + `qk_amd_broad_backend_roadmap.py` + bb5a plans.
- Inventory: `bench/qk-active-surface-reduction/inventory.json` (all cloners = `provenance`).
