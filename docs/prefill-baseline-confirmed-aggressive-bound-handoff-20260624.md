# Decode + Prefill Baseline / Confirmed / Aggressive — Handoff (2026-06-24)

## Scope

- Prefill values are from the long-context no-regression follow-up using synced prefill artifacts.
- Decode values are from the latest decode parity evidence in this lane.
- For each lane:
  - Baseline = current authority baseline
  - Confirmed = measured promoted / shipped improvement
  - Aggressive bound = non-search upper-envelope or target ceiling (planning only)

## Inputs

- Prefill
  - [baseline_prefill_by_context.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-no-regression-audit/baseline_prefill_by_context.json)
  - [candidate_prefill_by_context.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-no-regression-audit/candidate_prefill_by_context.json)
  - [time_tax_by_context.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-no-regression-audit/time_tax_by_context.json)
  - [tensile_gap_attribution.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-post-decode-parity-frontier/tensile_gap_attribution.json)
  - [time_tax.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-post-decode-parity-frontier/time_tax.json)
  - [prefill-long-context-no-regression-audit-result-20260623.md](/home/ubuntu/tinygrad-arkey/docs/prefill-long-context-no-regression-audit-result-20260623.md)
- Decode
  - [bench/qk-owned-tile-buffer-identity-kv-read/wd.json](/home/ubuntu/tinygrad-arkey/bench/qk-owned-tile-buffer-identity-kv-read/wd.json)
  - [bench/qk-decode-parity-no-regression-audit/wd_decode_by_ctx.json](/home/ubuntu/tinygrad-arkey/bench/qk-decode-parity-no-regression-audit/wd_decode_by_ctx.json)
  - [bench/qk-current-decode-benchmark/current.json](/home/ubuntu/tinygrad-arkey/bench/qk-current-decode-benchmark/current.json)
  - [tinygrad-vs-llama-time-tax latest](/home/ubuntu/tinygrad-arkey/bench/qk-tinygrad-vs-llama-time-tax/latest.json)
  - [decode-campaign-final-synthesis-20260623.md](/home/ubuntu/tinygrad-arkey/docs/decode-campaign-final-synthesis-20260623.md)

## Summary

- Prefill: baseline default remains authority; `eightwave` is confirmed + promoted.
- Decode: canonical W==D defaults are measured at 102.6/100.8/98.4/93.9 tok/s @ctx512/1024/2048/4096; the previous confirm signal remains valid, and a non-search full-stack envelope exists at 104.0/102.1/99.6/95.1.
- Old PLRA remains `needs_confirm`; `eightwave + old_plra` is rejected.

## Prefill table (tok/s)

| ctx | Baseline | Confirmed `eightwave` | Confirmed Δ | Aggressive non-search bound* |
|---:|---:|---:|---:|---:|
| 512 | 3485.17 | 3597.23 | +3.22% | 4593.45 |
| 1024 | 3404.00 | 3505.13 | +2.97% | 4486.47 |
| 2048 | 3176.80 | 3263.13 | +2.72% | 4187.02 |
| 4096 | 2720.58 | 2784.15 | +2.34% | 3585.72 |
| 8192 | 2177.03 | 2217.39 | +1.85% | 2869.33 |

*Aggressive bound assumes current baseline would close the unresolved in-model integration residual to the documented frontier corridor and transfers cleanly to these points.

## Decode table (tok/s)

| ctx | Baseline (current default) | Confirmed | Confirmed Δ | Aggressive non-search target |
|---:|---:|---:|---:|---:|
| 512 | 102.6 | 102.9 | +0.29% | 104.0 |
| 1024 | 100.8 | 101.2 | +0.40% | 102.1 |
| 2048 | 98.4 | 98.7 | +0.31% | 99.6 |
| 4096 | 93.9 | 94.0 | +0.11% | 95.1 |

## Source-bound interpretation

- Prefill aggressive numbers are intentionally optimistic and are not measured points.
- Decode aggressive numbers are the current non-search stack envelope used for progress checks; they are a close upper bound over the current default route.
- Use these rows only as a cross-lane handoff summary; keep context- and artifact-version consistency for any rollout claim.

## Suggested next step

- Decode: keep `DECODE_ATTN_KV_IDENTITY` as shipped and use this row for parity-completion framing.
- Prefill: run the next synced whole-prefill per-role transfer check before reopening bounded or search work.

## 2026-06-24 (current-session direct-context proof update)

- Prefill direct-context artifacts now include an additional synced whole-prefill closure attempt in
  `bench/qk-prefill-aggressive-target-proof-20260624/`.
- Decode W==D closure artifacts now include explicit aggressive target recheck in
  `bench/qk-decode-aggressive-target-proof-20260624/`.

### Current-context closure results (2026-06-24)

| ctx | Prefill base | Prefill `pipe_tm2_tn2` | Prefill `pipe_tm4_tn2` | Decode base | Decode aggressive |
|---:|---:|---:|---:|---:|---:|
| 512 | 3572 | 4253 | 2332 | 101.9 | 103.4 |
| 1024 | 3483 | 4037 | 2263 | 100.1 | 101.6 |
| 2048 | 3226 | 3659 | 2139 | 97.6 | 99.1 |
| 4096 | 2789 | 3110 | 1937 | 93.1 | 94.4 |

Interpretation:
- Prefill direct closure still does not reach the documented aggressive upper corridor; `pipe_tm2_tn2` is the strongest measured candidate in this session, but remains far below the projected aggressive bound.
- Decode aggressive remains below its current non-search envelope in a reproducible fashion:
  94.4/95.1 target and 103.4/104.0 target, and lockstep/gate closure remain intact.
