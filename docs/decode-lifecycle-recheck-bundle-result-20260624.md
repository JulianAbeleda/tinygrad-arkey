# Decode Lifecycle Recheck Bundle — Result (2026-06-24)

## Scope

Periodic decode lifecycle baseline refresh across:

- pre/post oracle gates
- pre/post unknown-lockstep
- interleaved W==D A/B sweeps (current + long + alternative capture mode)

## Execution

Ran:
- `extra/qk_decode_lifecycle_recheck_bundle.py`

Bundle: `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-151942`

Latest pointer:
- `bench/qk-decode-lifecycle-recheck-bundle/latest.json`

Run artifacts in this run:
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-151942/bundle_snapshot.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-151942/decision.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-151942/correctness/unknown_lockstep_pre.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-151942/correctness/unknown_lockstep_post.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-151942/throughput/current_context/wd_by_ctx.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-151942/throughput/long_context/wd_by_ctx.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-151942/throughput/alternative_route/wd_by_ctx.json`

## Decision

`DECODE_LIFECYCLE_RECHECK_BUNDLE_PASS`

## Correctness / reproducibility

- Gate pre: `PASS`
- Gate post: `PASS`
- Unknown lockstep pre: `DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN`
- Unknown lockstep post: `DECODE_UNKNOWN_BUCKET_LOCKSTEP_PROVEN`
- Runtime-overhead guardrail was active in all sweeps (`.item()` sync path, 3 reps).

## Throughput outcomes (W==D, A=whole/owned, B=slice)

### Current-context variant (`512,1024,2048,4096`)

| ctx | A tok/s | B tok/s | delta% |
|---:|---:|---:|---:|
| 512 | 102.2 | 86.1 | +18.70 |
| 1024 | 100.4 | 85.5 | +17.43 |
| 2048 | 97.9 | 84.5 | +15.86 |
| 4096 | 93.4 | 82.4 | +13.35 |

### Long-context variant (`4096`)

| ctx | A tok/s | B tok/s | delta% |
|---:|---:|---:|---:|
| 4096 | 93.7 | 82.3 | +13.85 |

### Alternative capture variant (`DECODE_ATTN_AMDGCN_TILE=0`)

| ctx | A tok/s | B tok/s | delta% |
|---:|---:|---:|---:|
| 512 | 76.0 | 76.2 | -0.26 |
| 1024 | 74.0 | 74.2 | -0.27 |
| 2048 | 71.1 | 71.3 | -0.28 |
| 4096 | 67.3 | 67.4 | -0.15 |

## Oracle comparison snapshot (llama reference: `bench/qk-post-parity-hardening/authority.json`)

| ctx | tinygrad A | llama | tinygrad / llama |
|---:|---:|---:|---:|
| 512 | 102.2 | 97.71 | 104.59% |
| 1024 | 100.4 | 97.39 | 103.09% |
| 2048 | 97.9 | 95.00 | 103.05% |
| 4096 | 93.4 | 92.37 | 101.12% |

## Baseline update outcome

This bundle is now the working decode lifecycle baseline snapshot. Next audits should use this run id as the
`DECODE_LIFECYCLE_RECHECK_BUNDLE` baseline, and then compare any future deltas against:
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-151942/bundle_snapshot.json`
- `bench/qk-decode-lifecycle-recheck-bundle/latest.json`

## Closure

- Unknown attribution is proven on this pass.
- Current/legacy A/B delta trend remains above parity at all supported contexts.
- Legacy capture path (`DECODE_ATTN_AMDGCN_TILE=0`) no longer offers meaningful gain under these settings.
