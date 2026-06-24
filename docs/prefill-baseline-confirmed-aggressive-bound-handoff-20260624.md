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
  - [tinygrad-vs-llama-time-tax latest](/home/ubuntu/tinygrad-arkey/bench/qk-tinygrad-vs-llama-time-tax/latest.json)
  - [decode-campaign-final-synthesis-20260623.md](/home/ubuntu/tinygrad-arkey/docs/decode-campaign-final-synthesis-20260623.md)

## Summary

- Prefill: baseline default remains authority; `eightwave` is confirmed + promoted.
- Decode: `DECODE_ATTN_KV_IDENTITY` uplift is measured and currently above llama parity; non-search aggressive projection is therefore planning-only as "parity/cross-lane ceiling".
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

| ctx | Baseline (`DECODE_ATTN_KV_IDENTITY=0`) | Confirmed (current shipped target) | Confirmed Δ | Aggressive target / parity bound |
|---:|---:|---:|---:|---:|
| 512 | 86.7 | 102.9 | +18.7% | 97.71 (parity target, already exceeded) |
| 1024 | 86.2 | 101.2 | +17.4% | 97.39 (parity target, already exceeded) |
| 2048 | 84.9 | 98.7 | +16.3% | 95.00 (parity target, already exceeded) |
| 4096 | 82.9 | 94.0 | +13.3% | 92.37 (parity target, already exceeded) |

## Source-bound interpretation

- Prefill aggressive numbers are intentionally optimistic and are not measured points.
- Decode aggressive numbers are the current parity-aware ceiling used for progress checks (since measured confirmed already cleared it).
- Use these rows only as a cross-lane handoff summary; keep context- and artifact-version consistency for any rollout claim.

## Suggested next step

- Decode: keep `DECODE_ATTN_KV_IDENTITY` as shipped and use this row for parity-completion framing.
- Prefill: run the next synced whole-prefill per-role transfer check before reopening bounded or search work.
