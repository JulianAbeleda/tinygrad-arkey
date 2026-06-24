# Decode Aggressive-Theoretical Closure Scope (2026-06-24)

## Objective

Prove whether decode can sustain the documented aggressive target and show it is (or is not) achievable with current tooling:

- Baseline (current default, measured): `102.6 / 100.8 / 98.4 / 93.9` tok/s @ `512,1024,2048,4096`
- Confirmed target (measured): `102.9 / 101.2 / 98.7 / 94.0` tok/s
- Aggressive-theoretical target (non-search stack envelope): `104.0 / 102.1 / 99.6 / 95.1` tok/s

Source: `docs/prefill-baseline-confirmed-aggressive-bound-handoff-20260624.md`, `bench/qk-owned-tile-buffer-identity-kv-read/wd.json`, `bench/qk-current-decode-benchmark/current.json`, `bench/qk-decode-parity-no-regression-audit/llama_vs_tinygrad_table.json`, `docs/archive/decode-oracle-explanation-and-schedule-diff-result-20260623.md`.

## Goal state

A single pass is **proof-positive** if all of these are true:

- W==D confirmed in current config at all required ctx.
- A/B delta remains above zero vs comparator (`DECODE_ATTN_KV_IDENTITY=0`) at all required ctx.
- `E_49152` remains absent in owned route for the aggressive stack.
- Unknown-bucket attribution is lockstep-closed pre/post the aggressive run.
- Aggressive run meets or beats target table above within repeat band.

If any fail, stop, capture failure mode, and write the blocker reason to `decision.json`.

## Scope

- This scope is a **one-shot closure attempt**, not broad search.
- No decode-default changes.
- No new kernels.
- No free-form search over unscoped knobs.

## Required evidence and tooling

Required tools:

- `extra/qk_decode_runtime_overhead.py`
- `extra/qk_decode_search_gate.py`
- `extra/qk_decode_unknown_bucket_lockstep_audit.py`
- `extra/qk_ctx_slope_driver.py`
- `extra/qk_decode_lifecycle_recheck_bundle.py` (or `_periodic.py`) for snapshot + compare.
- `extra/qk_decode_current_route_attribution.py` and `extra/qk_decode_materialization_check.py` (attribution only)

Required output directory:

- `bench/qk-decode-aggressive-target-proof-20260624/`

Required artifacts:

- `authority.json`
- `throughput_aggressive_probe.json` (`cfg`, `ctx`, `tok_s`, `ms_per_token`, `repeats`, `spread_pct`, `wall_sync_pct`, `verdict`)
- `throughput_baseline_probe.json` (baseline counterpart)
- `route_fire.json`
- `lockstep_pre.json`
- `lockstep_post.json`
- `slope_fit.json`
- `artifact_snapshot.json` (`/tmp` command, git hash, env)
- `decision.json`

## Execution sequence

### Phase 0 — Authority lock

- Record git hash, dirty state, branch, runtime (`DEV`, `JIT`, ROCm/driver if available), model artifact, model path, and hardware.
- Record exact commands from:
  - baseline run
  - aggressive stack run

Baseline command (as today):

```bash
QK_CKPTS=512,1024,2048,4096 DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py
```

Aggressive probe command (replace with envelope flags used by your current envelope source):

```bash
QK_CKPTS=512,1024,2048,4096 DEV=AMD JIT=1 PYTHONPATH=. \
  Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 Q4K_GEMV_WARP_PROJ=1 DECODE_ATTN_KV_IDENTITY=1 \
  .venv/bin/python extra/qk_decode_runtime_overhead.py
```

### Phase 1 — W==D confirmation lane (authoritative)

For baseline + aggressive config at required ctx:

- 512, 1024, 2048, 4096
- repeats aligned to recent session protocol
- `.item()` in-loop sync policy only (no raw dispatch claims)

Record `throughput_*` artifacts.

### Phase 2 — Oracle gate pre/post lock

Run `extra/qk_decode_search_gate.py` before and after aggressive probe to assert:

- route-fire route ID for owned path
- `E_49152` presence/absence
- ISA + resource checks (`v_dot2`, cross-lane, VGPR/spill)
- token correctness

### Phase 3 — Unknown bucket lockstep

Run unknown-bucket lockstep in lockstep mode for both configs:

- precheck and postcheck for aggressive probe
- ctx set: `512,1024,2048,4096`

### Phase 4 — Full-lifecycle periodic bundle + slope fit

Run one periodic bundle pass to bundle gate + lockstep + context slope in one snapshot:

```bash
.venv/bin/python extra/qk_decode_lifecycle_recheck_periodic.py --out-root bench/qk-decode-aggressive-target-proof-20260624/bundle
```

Then fit slope on `tok_s` and `ms/token` for current vs aggressive config (including long-context variant if available).

## Falsification map (how to prove target is not reachable in this toolchain)

- **Route divergence**: aggressive command does not fire `owned_flash_tile_gqa_whole` or uses non-expected route flags.
- **Attribution drift**: `E_49152` appears in aggressive path or disappears in baseline in a way that invalidates path continuity.
- **Lockstep fail**: `DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN` not met pre/post for any required ctx.
- **Throughput fail**: any required ctx below aggressive target by >1.0% after spread check.
- **Slope fail**: fixed-tax-only model no longer holds and long ctx residual is growing faster than expected while gate conditions hold.

If any fail, do not promote/claim; move to explanation-only conclusion in `decision.json`.

## Explanation outputs if target fails (what to write)

Write one primary fail label and one line summary:

- `DECODE_AGGRESSIVE_TARGET_UNPROVEN__ROUTE`
- `DECODE_AGGRESSIVE_TARGET_UNPROVEN__LOCKSTEP`
- `DECODE_AGGRESSIVE_TARGET_UNPROVEN__THROUGHPUT`
- `DECODE_AGGRESSIVE_TARGET_UNPROVEN__LONG_CTX_SLOPE`

Each label must include the best-practice explanation from tooling:

- `decode-oracle-explanation...` for closed search surfaces,
- `docs/archive/decode-lifecycle-recheck-bundle-periodic-scope-20260624.md` for periodic gate requirements,
- current authoritative `bench/qk-decode-lifecycle-recheck-bundle/latest.json` for baseline diff comparison.

## Success and next step

- PASS => update handoff with "aggressive confirmed" and publish working target snapshot.
- FAIL => publish a short proof-backed reasoned block showing that the aggressive target is beyond current lane authority and that further movement is not in search scope without new primitive work.
