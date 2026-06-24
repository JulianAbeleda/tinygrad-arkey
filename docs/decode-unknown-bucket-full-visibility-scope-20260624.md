# Decode Unknown-Path Full Visibility Scope (2026-06-24)

## Objective

Prove unknown-bucket attribution in decode is mathematically closed within a single synchronized measurement block.

Current target:
- make unknown-bucket mapping from profile to source-derived buckets exact (within 2us tolerance)
- preserve artifact chain to make the proof auditable
- avoid tool-mismatch drift by regenerating dependencies in the same run window

## Baseline context

- `bench/qk-decode-unknown-bucket-source-map-20260624/unknown_bucket_source_map.json` previously showed name/source mapping gaps.
- We needed to rerun the provenance chain so `unknown` shares are derived from the same capture/time-tax pair; lockstep is now complete.

## Artifacts required after this pass

- `bench/qk-decode-kernel-probe/latest.json`
- `bench/qk-decode-kernel-probe/decode-kernel-probe-YYYYMMDD-HHMMSS.json`
- `bench/qk-decode-time-tax-audit/latest.json`
- `bench/qk-decode-ctx-slope-audit/kernel_attribution_A.json` (refreshed from the latest time-tax audit)
- `bench/qk-decode-unknown-bucket-lockstep-20260624/decision.json`
- `bench/qk-decode-unknown-bucket-lockstep-20260624/math_assertions.json`
- `bench/qk-decode-unknown-bucket-lockstep-20260624/residual_unmapped_by_ctx.json`
- `bench/qk-decode-unknown-bucket-lockstep-20260624/unknown_bucket_source_map.json`
- `bench/qk-decode-unknown-bucket-lockstep-20260624/summary.md`
- `bench/qk-decode-unknown-bucket-lockstep-20260624/latest.json`

## Commands (exact)

1. Full canonical decode kernel capture (timelines + name+src flags):
   ```bash
   DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_audit_common.py --contexts 512,1024,2048,4096 --full-source-flags
   ```

2. Decode time-tax + kernel attribution (canonical `kernel_attribution` source):
   ```bash
   DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_time_tax_audit.py
   ```

3. Refresh lifecycle attribution input used by strict unknown mapper:
   ```bash
   cp bench/qk-decode-time-tax-audit/latest.json bench/qk-decode-ctx-slope-audit/kernel_attribution_A.json
   ```

4. Strict unknown visibility proof with mathematical residuals (single synchronized run):
   ```bash
   DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_unknown_bucket_lockstep_audit.py
   ```

## 2026-06-24 latest execution result

- Executed `DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_unknown_bucket_lockstep_audit.py` in one synchronized pass.
- Lockstep proof decision: `DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN`.
- Residuals are now zero for every context:
  - 512: `unknown_bucket_us=603.2`, `legacy_unknown_us=603.2`, residual `0.0`
  - 1024: `unknown_bucket_us=801.3`, `legacy_unknown_us=801.3`, residual `0.0`
  - 2048: `unknown_bucket_us=1062.7`, `legacy_unknown_us=1062.7`, residual `0.0`
  - 4096: `unknown_bucket_us=1569.6`, `legacy_unknown_us=1569.6`, residual `0.0`
- `--full-source-flags` evidence remains present in `bench/qk-decode-kernel-probe/latest.json` (`src_flags` includes op-presence and dtypes signatures).

## Acceptance criteria

- In `math_assertions.json`, for each ctx:
  - `name_classification_sum_matches_probe` must be `true`
  - `refined_unknown_sum_matches_name_unknown` should be `true`
- In `residual_unmapped_by_ctx.json`, `unmapped_after_unknown_refine_us` must be `0.0` at all ctx.
- In `decision.json`, label is expected to be `DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN`.

Tolerance:
- strict micro-bucket matching uses `2.0us`.

## Failure handling / next-step decision

If any ctx remains with residual > 0:
- check whether `kernel_attribution_A.json` used by the mapper equals current `bench/qk-decode-time-tax-audit/latest.json` run (timestamp drift)
- check if `latest.json` in kernel-probe is from the same wall-clock run window as time-tax (if not, rerun both and repeat)
- if still unresolved: run one synchronized pass:
  ```bash
  DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_unknown_bucket_lockstep_audit.py
  ```
  and verify `bench/qk-decode-unknown-bucket-lockstep-20260624/decision.json`.

## Estimated runtime

- Full one-pass execution target: 6–8 minutes.
