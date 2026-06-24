# Decode Lifecycle Recheck Bundle — Result (2026-06-24)

## Scope

Periodic decode lifecycle baseline refresh across:

- pre/post oracle gates
- pre/post unknown-lockstep
- interleaved W==D A/B sweeps (current + long + alternative capture mode)

## Execution

Ran:
- `extra/qk_decode_lifecycle_recheck_bundle.py`

Bundle: `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-172026`

Latest pointer:
- `bench/qk-decode-lifecycle-recheck-bundle/latest.json`

Run artifacts in this run:
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-172026/bundle_snapshot.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-172026/decision.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-172026/correctness/unknown_lockstep_pre.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-172026/correctness/unknown_lockstep_post.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-172026/throughput/current_context/wd_by_ctx.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-172026/throughput/long_context/wd_by_ctx.json`
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-172026/throughput/alternative_route/wd_by_ctx.json`

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
| 512 | 101.6 | 85.8 | +18.41 |
| 1024 | 99.8 | 85.3 | +17.00 |
| 2048 | 97.4 | 84.3 | +15.54 |
| 4096 | 92.9 | 82.2 | +13.02 |

### Long-context variant (`4096`)

| ctx | A tok/s | B tok/s | delta% |
|---:|---:|---:|---:|
| 4096 | 93.0 | 82.1 | +13.28 |

### Alternative capture variant (`DECODE_ATTN_AMDGCN_TILE=0`)

| ctx | A tok/s | B tok/s | delta% |
|---:|---:|---:|---:|
| 512 | 75.8 | 75.8 | +0.00 |
| 1024 | 73.9 | 73.9 | +0.00 |
| 2048 | 70.9 | 70.8 | +0.14 |
| 4096 | 66.8 | 66.9 | -0.15 |

## Oracle comparison snapshot (llama reference: `bench/qk-post-parity-hardening/authority.json`)

| ctx | tinygrad A | llama | tinygrad / llama |
|---:|---:|---:|---:|
| 512 | 101.6 | 97.71 | 104.01% |
| 1024 | 99.8 | 97.39 | 102.47% |
| 2048 | 97.4 | 95.00 | 102.53% |
| 4096 | 92.9 | 92.37 | 100.57% |

## Baseline update outcome

This bundle is now the working decode lifecycle baseline snapshot. Next audits should use this run id as the
`DECODE_LIFECYCLE_RECHECK_BUNDLE` baseline, and then compare any future deltas against:
- `bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-172026/bundle_snapshot.json`
- `bench/qk-decode-lifecycle-recheck-bundle/latest.json`

## Closure

- Unknown attribution is proven on this pass.
- Current/legacy A/B delta trend remains above parity at all supported contexts.
- Legacy capture path (`DECODE_ATTN_AMDGCN_TILE=0`) no longer offers meaningful gain under these settings.
