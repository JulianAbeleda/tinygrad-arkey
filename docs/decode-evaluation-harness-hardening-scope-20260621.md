# Decode Evaluation & Benchmark Hardening — Execution Scope

Date: 2026-06-21

Owner: next executor

Status: scope only (chosen #1 by `docs/next-project-scope-audit-20260621.md`)

## Objective

Build a **durable, automated, reproducible decode evaluator** — one callable `decode_eval(candidate) -> result`
that runs the full evaluation ladder (structural → correctness → clock-controlled local A/B → W==D → policy) and
emits a **schema'd, machine-readable** result — plus the **artifact contract** and **measurement-authority
encoding** the lifecycle-search system will drive. This turns the ladder, currently re-implemented by hand in
every experiment, into first-class infrastructure, and removes the measurement fragility (auto-clock volatility,
ad-hoc gate thresholds, prose-only authority classes) that taxed the bounded-primitive arc.

This is **measurement infrastructure only**: no kernels, no model changes, no default changes, no flag promotion.
It does not touch any closed decode lane.

## Exact files to read first

- `docs/next-project-scope-audit-20260621.md` — the ranking + why this is #1.
- `docs/current-project-state-handoff-20260621.md` — canonical numbers, closed lanes, gates.
- `structure/Development/performance-primitive-research-principles.md` — the T=1 principle + the gate philosophy
  (compare vs the current winner; manufacture parallel work; measure mechanism not proxy).
- `docs/decode-prefill-headline-reconciliation-result-20260621.md` — the **measurement-authority classes** (clean
  W==D = promotion; pinned-clock local = diagnostic; PROFILE GPU timestamps = attribution-only). Encode these.
- The existing ladder pieces (the inventory to consolidate, not rewrite):
  - `extra/qk_decode_runtime_overhead.py` — W==D wall/dispatch tok/s (promotion authority).
  - `extra/qk_clock_pin.py` — peak-clock pin / restore-auto (diagnostic clock control).
  - `extra/qk_decode_current_route_attribution.py` — PROFILE GPU-timestamp per-role attribution.
  - `extra/qk_decode_{vector,warp,fused_lds}_flash_tile_ab.py`, `extra/qk_decode_*_ab.py` — the bespoke local-A/B
    pattern (clock-pin + warm + byte-exact-vs-numpy + vs `gqa_coop_vec`).
  - `extra/qk_policy_consistency_check.py` — the docs/policy guardrail.
- `bench/qk-lifecycle-search/` — the seed ledger (`candidates.json`, `generated_candidates.json`,
  `refutations.json`, `runner_bindings.json`, `policy_exports.json`, `summary.md`) the evaluator must feed.
- `bench/README.md` — the current default / opt-in / llama-reference rows the evaluator must reproduce.

## Phases

### Phase 0 — Authority + gate contract (no code)

Write down, as a single machine-readable contract, what already exists only in prose:
- the three **timing-authority classes** (W==D promotion / pinned-clock diagnostic / PROFILE attribution) and the
  rule that they never mix in one row;
- the **canonical gate thresholds**: local ≥1.05× vs the current winner @ctx1024; W==D ≥5%@1024 or ≥7%@4096, no
  ctx512 regression >1%; host-sync non-target; correctness byte-exact / dNLL within decode policy;
- the **comparator**: the current winner (`gqa_coop_vec`, FLASH_L=128 default), never a weaker baseline;
- the **contexts**: 512 / 1024 / 4096 (+ ctx0/128 reference).
Output: `bench/qk-decode-eval/authority_contract.json` + a short doc section.

### Phase 1 — The evaluator API

Implement `extra/qk_decode_eval.py` exposing `decode_eval(candidate) -> result`:
- **Input**: a candidate spec (a route/flag/variant descriptor + a runner binding pointing at how to instantiate
  it — reuse `runner_bindings.json`'s shape).
- **Ladder** (short-circuits on failure, records the failing rung):
  1. structural/identity (program names, shapes, call counts) — reuse the attribution capture;
  2. correctness (byte-exact / dNLL vs the reference for the candidate's role);
  3. clock-controlled local A/B vs the current winner (pinned, warm, median-of-N);
  4. W==D whole-decode tok/s @ctx 512/1024/4096 (clean wall, PROFILE off, auto clock, median-of-≥5);
  5. policy/default decision (gate verdicts + whether default would change — must be NO unless owner-approved).
- **Output**: a schema'd result JSON (authority class per row, ms/token, tok/s, gate pass/fail with thresholds,
  reproducibility band, clock provenance, default-changed flag). Always restores GPU perf-state to `auto`.

### Phase 2 — Reproducibility & robustness

- **Reproducibility band**: re-run the W==D rung K times; emit the variance and FAIL the rung if it exceeds a
  declared band (clock-controlled). This is the core hardening — it makes "did it really move" trustworthy.
- **Clock provenance**: record sclk/perf-state for every timed row; tag rows that ran cold/volatile.
- **Robustness**: handle OOM / compile failure / NaN as classified results, not crashes; always restore `auto`.
- **Runner-binding contract**: a candidate → (instantiation, comparator, role) mapping that the lifecycle ledger
  already half-encodes; formalize it so a generated candidate is runnable without bespoke glue.

### Phase 3 — Artifact + guardrail integration

- Validate every emitted result against the schema; reject malformed artifacts.
- Extend `qk_policy_consistency_check.py` (or a sibling) to also catch **stale measurement claims** (e.g. a
  promotion claimed from pinned-clock/PROFILE numbers, a tok/s headline without its authority class/ctx).
- Wire result emission into `bench/qk-lifecycle-search/` so a `decode_eval` run appends a machine-readable row +
  (on a clean refutation) a refutation entry.

### Phase 4 — Validation against known ground truth

Re-run, through `decode_eval`, 2–3 documented candidates and confirm the evaluator reproduces the **hand-run
verdicts within the reproducibility band**:
- `gqa_coop_vec` default (the winner / baseline);
- `FLASH_L=64` → must reproduce **local-pass (~1.08× @ctx1024) and W==D-fail** (+1.8%@1024, −1.2%@4096);
- one refuted candidate (e.g. the warp tile) → must reproduce **local-fail (~0.60×)**.

## Measurable gates

1. **Coverage**: `decode_eval` runs all five ladder rungs and emits each as a machine-readable field with its
   authority class.
2. **Reproducibility**: the W==D rung's tok/s reproduces within a declared band (target: ±1.5% clock-controlled)
   across ≥3 evaluator invocations of the same candidate; the band is reported, not assumed.
3. **Ground-truth fidelity**: the evaluator reproduces the three known verdicts (Phase 4) — same pass/fail
   classification as the hand-run docs.
4. **Determinism of verdict**: the same candidate yields the same gate pass/fail across re-runs (numbers vary
   within band; verdicts do not flip).
5. **Schema validity**: every emitted result validates against the published schema; the guardrail rejects
   malformed/stale-authority claims.
6. **No-change invariant**: `git diff tinygrad/llm/model.py` and `git diff tinygrad/` are empty; no default flag
   changes; GPU perf-state restored to `auto` after every run.

## Artifacts to write

- `extra/qk_decode_eval.py` (the evaluator), `extra/qk_decode_eval_schema.json` (result schema).
- `bench/qk-decode-eval/authority_contract.json`, `bench/qk-decode-eval/groundtruth_validation.json`.
- `docs/decode-evaluation-harness-hardening-result-20260621.md` (result: coverage, reproducibility band,
  ground-truth fidelity, what changed — defaults unchanged).
- lifecycle-search ledger: machine-readable rows emitted by `decode_eval` (not hand-edited prose).

## Rollback / no-change boundaries

- **No `tinygrad/llm/model.py` or `tinygrad/` change.** Pure `extra/` + `bench/` + `docs/` tooling.
- **No default / flag promotion.** The evaluator only *measures*; promotion is an owner decision it reports.
- **No kernel work, no new candidate kernels.** Existing candidates are inputs to validate the evaluator.
- Clock pinning is used only inside the diagnostic rung and always restored to `auto`.
- Rollback = delete the `extra/qk_decode_eval*.py` + `bench/qk-decode-eval/` additions; nothing in the run path
  depends on them.

## Stop conditions

- **If reproducibility cannot be achieved** even with clock control (W==D variance stays > a usable band): stop —
  the prerequisite is deeper GPU-state control tooling (telemetry-binned clocks / DPM authority), and that becomes
  the scope instead. (This session's clock-pin got it close; quantify whether it is enough.)
- **If candidates are too bespoke to share one evaluator** (each needs unique glue beyond a runner binding): stop
  and report — the lifecycle-search templating must come first.
- **If the existing harnesses already meet a durable callable contract**: stop — they do not (bespoke per
  experiment), but verify in Phase 0.
- Do not expand into building new decode candidates, tuning flags, or any closed lane.

## How it advances the north star

The north star's **Method** pillar is a *closed lifecycle machine-search system* whose evaluation ladder is
`generate → structural → correctness → local A/B → W==D → policy`. That ladder is exactly this evaluator. Without
a durable, reproducible, automatable evaluator the search loop optimizes against noise and cannot "reproduce the
winning route or regenerate an equivalent after artifact deletion" (a named completion gate). This scope makes the
ladder first-class — the foundation the lifecycle-search system (audit #2) is built on, and the trustworthy
measurement any future `flash_attn_tile` attempt (audit #3) needs to avoid being an unmeasurable hand patch. It is
also a direct down-payment on the tinygrad-v2 "reproducible benchmarks runnable from documented commands" gate.

## What would prove it wrong

- **W==D is irreducibly fragile**: if, even clock-controlled, the same candidate's whole-decode tok/s swings wider
  than the gate margins, then a consolidated evaluator can't make promotion decisions trustworthy — the real
  prerequisite is GPU-state/telemetry control, not an evaluator API. (Falsifier: Phase 2 reproducibility band.)
- **The ladder doesn't generalize**: if real candidates can't be expressed as `(spec, runner-binding)` and each
  needs bespoke measurement, the "one evaluator" premise is wrong and templating must lead. (Falsifier: Phase 4 —
  can the three known candidates run through one API unchanged?)
- **It's redundant**: if the existing scripts already form a stable, reused contract that the ledger drives, the
  hardening adds nothing. (Falsifier: Phase 0 inventory — they are bespoke and prose-gated, but confirm.)
