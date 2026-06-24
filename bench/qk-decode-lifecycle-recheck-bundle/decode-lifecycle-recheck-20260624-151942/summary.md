# Decode Lifecycle Recheck Bundle

- run: `20260624-151942`
- authority: `oracle gate PASS` pre=PASS post=PASS
- unknown closure: pre=DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN post=DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN
- current ctx result: `13.35` mean delta%
- decision: `DECODE_LIFECYCLE_RECHECK_BUNDLE_PASS`

## Commands executed

1. `qk_decode_search_gate.py --oracle-tokens bench/qk-decode-search-readiness/baseline_oracle.json`
2. `qk_decode_unknown_bucket_lockstep_audit.py --contexts 512,1024,2048,4096`
3. `qk_ctx_slope_driver.py` (current contexts)
4. `qk_ctx_slope_driver.py` (long context)
5. `qk_ctx_slope_driver.py` (`DECODE_ATTN_AMDGCN_TILE=0` alternative mode)
6. `qk_decode_search_gate.py --oracle-tokens ...` (postflight)
7. `qk_decode_unknown_bucket_lockstep_audit.py --contexts ...` (postflight)

## Outputs

- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-151942/bundle_snapshot.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-151942/decision.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-151942/correctness/unknown_lockstep_pre.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-151942/correctness/unknown_lockstep_post.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-151942/throughput/current_context/wd_by_ctx.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-151942/throughput/long_context/wd_by_ctx.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-151942/throughput/alternative_route/wd_by_ctx.json`
