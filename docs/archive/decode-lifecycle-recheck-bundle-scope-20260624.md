# Decode Lifecycle Recheck Bundle Scope (2026-06-24)

## Goal

Create a periodic, repeatable protocol that snapshots all three lifecycle pillars in one bundle:

1) correctness + reproducibility
2) W==D throughput across context points
3) unknown-bucket attribution closure

The bundle is used as the decode baseline refresh mechanism.

## Protocol

### A) Pre/Post correctness gate

- Run `extra/qk_decode_search_gate.py` with canonical baseline oracle from:
  `bench/qk-decode-search-readiness/baseline_oracle.json`.
- Check all gate outputs:
  - route fire
  - no `E_49152` materialization under default path
  - ISA + resource resource checks
  - token correctness
  - route correctness/verdict
- Run gate both pre-flight and post-flight.

### B) Unknown-bucket closure pre/post

- Run `extra/qk_decode_unknown_bucket_lockstep_audit.py` pre-flight with lockstep contexts
  `512,1024,2048,4096`.
- Run lockstep audit post-flight with the same contexts.
- Pass label required for bundle pass: `DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN`.

### C) W==D throughput sweep matrix

Run three interleaved A/B sweep variants via `extra/qk_ctx_slope_driver.py`:

1. **Current-context variant**
   - contexts: `512,1024,2048,4096`
   - A default route (owned whole-cache decode)
   - B comparator route (`DECODE_ATTN_KV_IDENTITY=0`)

2. **Long-context variant**
   - contexts: `4096` (harness max for this runtime path)
   - same A/B comparator pair

3. **Alternative capture variant**
   - contexts: `512,1024,2048,4096`
   - env `DECODE_ATTN_AMDGCN_TILE=0`

### D) Single handoff artifact bundle

All outputs and checks must be written under:
`bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-<RUN_ID>/`

Required files in the bundle:

- `bundle_snapshot.json`
- `decision.json`
- `correctness/gate_pre.json`
- `correctness/gate_post.json`
- `correctness/unknown_lockstep_pre.json`
- `correctness/unknown_lockstep_post.json`
- `throughput/current_context/wd_by_ctx.json`
- `throughput/long_context/wd_by_ctx.json`
- `throughput/alternative_route/wd_by_ctx.json`
- `summary.md`
- `runtime_latest_result.json`

A pointer to the latest bundle should be written at:
`bench/qk-decode-lifecycle-recheck-bundle/latest.json`

## Known constraints in this repo runbook

- `extra/qk_decode_unknown_bucket_lockstep_audit.py` currently supports `512,1024,2048,4096` contexts.
- `extra/qk_decode_runtime_overhead.py`/`qk_ctx_slope_driver.py` path supports contexts up to `4096` without additional harness edits.

## Acceptance

Bundle verdict must be:
- `DECODE_GATE_REVIEW_REQUIRED` on gate mismatch; or
- `DECODE_UNKNOWN_CLOSURE_REVIEW_REQUIRED` on unknown lockstep failure; or
- `DECODE_PERF_DELTA_REVIEW_REQUIRED` on negative/unstable A/B deltas; or
- `DECODE_LIFECYCLE_RECHECK_BUNDLE_PASS` on all-pillars closure.

### E) Periodic wrapper

- Use `extra/qk_decode_lifecycle_recheck_periodic.py` for recurring baseline refreshes.
- Required command pattern:
  - `DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_lifecycle_recheck_periodic.py --out-root <root>`
- Output per run:
  - `<run>/periodic_diff.json`
  - `<run>/periodic_diff.md`
