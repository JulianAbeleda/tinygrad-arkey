# Harness Best-Practices Adherence Audit

Date: 2026-06-21

Question: after adding `bench/qk-decode-eval/HARNESS_GUIDE.md`, does the repo's harness surface adhere to it?

Verdict: **PARTIAL_ADHERENCE_WITH_LIVE_PATH_PROTECTED**.

The live decode decision path now follows the right architecture: candidates flow through `decode_eval`, lifecycle
search prunes closed/policy-invalid work before benchmarking, and `decode_eval` flags weak child artifacts through
`child_artifact_contract`. But most pre-existing standalone probe/A-B scripts still predate the 13-field harness
contract. They are acceptable as historical/provenance only; they should not become new authority unless upgraded or
wrapped by `decode_eval`.

## Authority

- `bench/qk-decode-eval/HARNESS_GUIDE.md`
- `structure/Development/performance-primitive-research-principles.md`, section "Harnesses Are Performance Primitives
  Too"
- `extra/qk_harness_contract.py`
- `extra/qk_decode_eval.py`

## Scope

Mechanically scanned `extra/*.py` for harness-like names or timing/verdict behavior:

| population | count | interpretation |
|---|---:|---|
| all `extra/*.py` | 182 | total extra scripts/modules after active-surface reduction |
| `extra/qk_*.py` | 133 | project-specific QK/perf surface |
| harness-like `extra/*.py` | 141 | includes live evaluators, A/B scripts, probes, profile helpers, training/eval utilities |
| registered `decode_eval` candidates | 10 | live decode candidate registry |
| registered `ab_script` child harnesses | 6 | direct child harnesses whose artifacts influence decode-eval verdicts |
| `_ab.py` scripts still present | 9 | A/B scripts, including closed/historical lanes |

This is intentionally broader than the earlier live-set audit. The earlier audit answered "is the decision path safe?"
This audit answers "does the repo surface broadly follow the new harness guide?"

## Live Path Adherence

| component | adherence | reason |
|---|---|---|
| `extra/qk_decode_eval.py` | **GOOD** | separates timing authority; emits schema'd verdicts; wraps child artifacts; uses verdict SSOT |
| `extra/qk_lifecycle_search_loop.py` | **GOOD** | generate/evaluate/prune loop; prunes policy/closed-lane candidates before benchmarking |
| `extra/qk_candidate_template_gen.py` | **GOOD** | deterministic candidate generation; no performance claim by itself |
| `extra/qk_harness_contract.py` | **GOOD** | central `stamp()`, provenance, `repro_band()`, `contract_audit()` |
| `bench/qk-decode-eval/HARNESS_GUIDE.md` | **GOOD** | explicit best-practices baseline with external references |

Practical conclusion: **new decode candidates should be routed through this path.** Do not add a new standalone
performance-claiming harness that bypasses this stack.

## Registered Child Artifact Audit

Direct `contract_audit()` on the `result_json` artifacts referenced by `bench/qk-decode-eval/candidates.json`:

| candidate | child harness | artifact conformance | missing fields summary | decision impact |
|---|---|---:|---|---|
| `fused_flash_concrete_gate` | `qk_fused_flash_concrete_gate_ab.py` | **CONFORMS 13/13** | none | best-practice reference |
| `fused_softmax_v_tail` | `qk_fused_softmax_v_tail_ab.py` | **WEAK 7/13** | comparator-why, command/env, hardware/clock, warmup/compile, spread band, ledger links | safe only because `decode_eval` flags child contract |
| `matmul_pv_diagnostic` | `qk_matmul_pv_diagnostic_ab.py` | **WEAK 6/13** | comparator-why, command/env, git/dirty, hardware/clock, warmup/compile, spread band, ledger links | safe only because `decode_eval` flags child contract |
| `llama_flash_attn_tile_oracle` | `qk_llama_flash_attn_tile_oracle_ab.py` | **WEAK 2/13** | most contract fields; ctx512/4096 provenance is disclosed but not fully measured | non-promotable oracle; do not quote as direct promotion |
| `north_star_flash_attn_tile` | `qk_north_star_flash_attn_tile_ab.py` | **WEAK 1/13** | most contract fields | closed negative; safe only as banked refutation |
| `warp_flash_tile` | `qk_decode_warp_flash_tile_ab.py` | **WEAK 1/13** | most contract fields | historical replay/closed lane |

Important distinction: the evaluator's wrapping makes these **visible**, not fully compliant. A weak child artifact is
not silently authoritative anymore, but it is still a weak standalone artifact.

## Broad Surface Signals

Across the 141 harness-like files:

| signal | count | reading |
|---|---:|---|
| imports/uses `qk_harness_contract` or `stamp()` | 9 | the new contract has not propagated broadly |
| uses `repro_band()` | 4 | spread/noise is still missing in most old scripts |
| calls `contract_audit()` | 2 | contract self-audit is concentrated in the evaluator/helper |
| local `argparse` CLI | 68 | many one-off driver surfaces remain |
| local JSON write behavior | 37 | residual clone-per-experiment pattern |
| `time.perf_counter` timing | 47 | many scripts can emit timing without the full contract |
| clock-pin boundary usage | 12 | better than before, but not universal |
| `Verdict` enum usage | 7 | verdict SSOT is live-path centered |
| raw sysfs/`rocm-smi` perf-state strings | 2 | mostly contained; one is a documented exception |

This matches the repo-principles audit: the authority layer is now decent, while the experiment-driver layer still has
clone-per-experiment history.

## Non-Adherence Classes

### A. Weak Registered Child Harnesses

These are the highest priority because they are still referenced by `decode_eval`:

- `extra/qk_fused_softmax_v_tail_ab.py`
- `extra/qk_matmul_pv_diagnostic_ab.py`
- `extra/qk_llama_flash_attn_tile_oracle_ab.py`
- `extra/qk_north_star_flash_attn_tile_ab.py`
- `extra/qk_decode_warp_flash_tile_ab.py`

The immediate rule is not "retrofit all before any work." It is:

```text
If one of these is re-run to make a new claim, first upgrade it with stamp(), repro_band(), comparator_why, and ledger links.
```

### B. Historical Closed-Lane A/B Harnesses

These are retained as provenance but should not be used directly:

- `extra/q4_k_output_ab.py`
- `extra/qk_decode_fused_flash_tile_ab.py`
- `extra/qk_decode_warp_flash_tile_ab.py`
- `extra/qk_gateup_sched_ab.py`

`qk_gateup_sched_ab.py` remains the riskiest historical example because it was stdout-only and previously claimed
clock-controlled behavior without the modern clock/provenance contract. Treat it as superseded provenance only.

### C. Probe/Driver Sprawl

The scan still finds 68 argparse-heavy harness-like scripts and 37 local JSON-writing surfaces. This is the same S1
problem from `docs/repo-principles-audit-20260621.md`: a small JSON/verdict/driver pattern is cloned across many
probes. It is not a live decode-result correctness bug, but it is the main maintainability gap.

### D. GPU Boundary Exceptions

Raw perf-state strings remain in:

- `extra/qk_decode_q8_model_route_timing_audit.py` — documented exception; try/finally/provenance shape differs from
  `qk_clock_pin`.
- `extra/qk_prefill_tc_attn_concrete_gate.py` — imports `qk_clock_pin.perflevel`; this is acceptable boundary usage,
  but the grep still detects `rocm-smi` through that wrapper context.

## Adherence Grade

| area | grade | why |
|---|---|---|
| live decode evaluator architecture | **A-** | correct authority separation, schema, verdict SSOT, child artifact visibility |
| current best-practice reference harness | **A** | `fused_flash_concrete_gate` conforms 13/13 |
| registered child harness surface | **C** | 5/6 referenced child artifacts are still weak if viewed standalone |
| broad historical probe surface | **C-** | many old scripts predate contract, repeat JSON/timing/provenance patterns |
| safety of new work if rules are followed | **B+** | guide + helper + evaluator make the right path clear |

Overall: **B- as a system, because the live authority path is protected; C if judging every old harness as a standalone
artifact.**

## Required Policy Going Forward

1. New performance-claiming harnesses must use `stamp()` and `repro_band()` before commit.
2. New live candidates must be registered in `bench/qk-decode-eval/candidates.json` or generated through lifecycle
   search.
3. Local A/B is diagnostic only; W==D remains promotion authority.
4. Weak historical probes cannot be cited as current evidence unless re-run under the harness guide.
5. Closed-lane probes stay closed unless a new scope states the new evidence and upgrades the harness first.

## Recommended Remediation Order

| priority | action | why |
|---:|---|---|
| 1 | Add a test that every registered `ab_script` artifact either `CONFORMS` or is explicitly marked `historical_oracle` / `closed_lane` / `diagnostic_only` | prevents weak artifacts from silently becoming live authority |
| 2 | Upgrade the two most plausible reusable active diagnostics (`matmul_pv_diagnostic`, `fused_softmax_v_tail`) with `stamp()` + `repro_band()` when next touched | small change, high clarity |
| 3 | Add structured comparator metadata to `decode_eval` run artifacts | closes the evaluator's remaining prose-only comparator gap |
| 4 | De-clone the probe harness pattern into one small `probe_harness.emit_verdict()` helper | attacks the broad 68-CLI / 37-JSON sprawl |
| 5 | Centralize comparator ids, thresholds, and contract fields as importable constants | prevents drift between guide/helper/schema/registry |
| 6 | Keep old closed-lane scripts frozen or delete them in a later active-surface cleanup | avoids spending effort polishing dead provenance |

## Verification Commands

```bash
PYTHONPATH=. .venv/bin/python extra/qk_harness_contract.py
PYTHONPATH=. .venv/bin/python extra/qk_decode_eval.py --list
PYTHONPATH=. .venv/bin/python extra/qk_lifecycle_search_loop.py --list
PYTHONPATH=. .venv/bin/python extra/qk_candidate_template_gen.py --list-templates
```

All passed during this audit.

## Bottom Line

The repo now knows how a correct harness should look, and the live evaluator path enforces enough visibility to avoid
the old silent mistakes. But the surface does **not** broadly adhere yet. The next harness-maintenance step is not more
documentation; it is a small enforcement test for registered child artifacts plus selective stamping of any child
harness we intend to reuse.
