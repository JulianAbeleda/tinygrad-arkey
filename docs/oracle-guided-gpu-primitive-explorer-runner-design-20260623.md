# Oracle-Guided GPU Primitive Explorer — Unified Runner Design (2026-06-23)

**Verdict: `EXPLORER_RUNNER_DESIGN_READY`.** Design only — not implemented in this scope (the scope says implement the
runner only if explicitly requested). The decode lane already has a functional backend
(`qk_decode_search_runner.py`); this document specifies how a single generic runner would drive every lane through the
shared spec + ledger, without duplicating the lane executors.

## 0. Position

The runner is glue, not a new search engine. It loads an oracle + a `SearchRow`-derived spec, asks a lane backend for
bounded candidates, runs the cost-ordered gate stack, and writes a ledger entry. The novelty lives in the **gate
stack** (lifecycle: route + ABI/materialization + ISA + correctness + whole-path authority), not in candidate
generation.

## 1. CLI

```bash
PYTHONPATH=. .venv/bin/python extra/qk_oracle_gpu_primitive_explorer.py \
  --spec bench/qk-oracle-gpu-primitive-explorer/spec_decode_policy_example.json \
  --out  bench/qk-oracle-gpu-primitive-explorer/runs/decode_policy_001 \
  [--smoke] [--max-candidates N] [--dry-run]
```

- `--spec` one explorer spec (Phase-2 schema). `--out` per-run artifact dir. `--smoke` structural/gate self-test, no
  authority benchmark. `--dry-run` enumerate candidates + gate plan only. `--max-candidates` budget cap.

## 2. Oracle loading

- Read `oracles.json`, select by the spec's `oracle_id`. Reconcile the frozen baseline artifact (e.g.
  `bench/qk-decode-search-readiness/baseline_oracle.json`) and **recheck the oracle in-band** before searching:
  decode oracle must land within its 3 % W==D band or the run STOPs (`SEARCH_ORACLE_DRIFT_STOP`). An oracle that
  cannot be loaded/reconciled stops the run (`ORACLE_REGISTRY_PARTIAL`).

## 3. Candidate generation

- Delegate to the lane backend named in `candidate_generator`; never generate kernels free-form. Each backend yields
  `{"id": str, "env": {knob: value}}` (decode Mode A/B) or a microprimitive expression descriptor (codegen).
- Knobs/ranges come from the spec; the runner only enumerates within the declared bounds and `--max-candidates`.

## 4. Gate ordering (cost-ordered, short-circuit on first reject)

```
schema/structural -> route/lifecycle -> materialization/ABI -> ISA/resource -> correctness -> local diagnostic (opt) -> authority benchmark
```

Gate plugins (lane-applicable subset; a per-lane gate registry selects which apply):

| gate | tool | lanes |
|---|---|---|
| harness contract | `extra/qk_harness_contract.py` | all |
| route fire | `extra/qk_decode_route_fire_check.py` | decode |
| materialization/ABI | `extra/qk_decode_materialization_check.py` | decode |
| ISA/resource | `extra/qk_isa_primitive_audit.py` / `qk_amdgpu_isa_primitive_audit.py` | decode, codegen |
| decode W==D | `extra/qk_decode_search_gate.py::run_wd` | decode |
| prefill synced authority | `extra/qk_prefill_whole_synced.py` | prefill |
| prefill role attribution | `extra/qk_prefill_per_role_time_tax.py` | prefill (gate, pre-search) |
| ledger write | `extra/qk_project_search_ledger.py` | all |

First failed gate → record `stop_reason`, skip the expensive authority benchmark for that candidate.

## 5. Lane-specific authority

| lane | authority | promotion |
|---|---|---|
| decode | clean synced W==D (`run_wd`) | reported only — owner flips defaults |
| prefill | clean synced whole-prefill | reported only |
| native-codegen microprimitive | local correctness (`rel_rmse<=1e-2`) + ISA target | **never** promotes a default |
| cross-shape | target-specific decode/prefill authority | reported only |
| small-op fusion | W==D / whole-prefill after one manual fusion gate | reported only |

A harness **recommends**, it never applies a default flip.

## 6. Artifact output

Per run under `--out`: `authority.json`, `search_plan.json`, `candidate_manifest.json`, `results.jsonl`,
`reject_summary.json`, `leaderboard.json`, `decision.json` — each stamped via `qk_harness_contract.stamp()` (target
`CONFORMS` 13/13).

## 7. Ledger write

Append one entry per candidate to `bench/qk-project-search-ledger/ledger.jsonl` via
`qk_project_search_ledger.entry(**kw)` (15 fields: `candidate_id, lane, primitive_class, knobs, oracle, correctness,
route_identity, materialization_abi, isa, local_diagnostic, authority_benchmark, verdict, stop_reason, artifact_links,
learned_rule`). `validate(e)` must return no missing fields before write. Rejected candidates record their first
failed gate + any learned reject rule (e.g. `minctx1024 -> route_not_firing at ctx512`).

## 8. Stop rules

- First failed gate stops that candidate (no authority benchmark on structurally-bad candidates).
- Oracle recheck outside band → whole run STOP (`SEARCH_ORACLE_DRIFT_STOP`).
- A "winner" requires authority delta beyond `max(oracle_spread, 1.0) %`; otherwise `*_ORACLE_REMAINS_BEST`.
- Budget exhaustion (`--max-candidates`) stops generation cleanly with a partial leaderboard.

## 9. Safety boundaries

- No default flip from any harness — promotion is an owner decision.
- No broad/autonomous free-form kernel search; bounded knobs only.
- No prefill search unless its gate is `PREFILL_SEARCH_READY_ROLE_SPECIFIC`.
- No cross-shape search without target selection.
- PROFILE/DEBUG/no-sync/raw-dispatch timings are diagnostic only, never authority.
- The microprimitive lane is non-promotion by construction (ISA + local correctness only).

## Adapter work required to implement (not done here)

A `SearchRow -> {env-knob dict, expected-kernel symbol, oracle file, reject envelope}` adapter plus a per-lane gate
registry. The decode runner is the lane executor to *wrap*, not a spec format to supersede; the spec layer is
`extra/qk_search_spec.py` and the memory layer is the project ledger schema.
