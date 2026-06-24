# Decode + Prefill Baseline / Confirmed / Aggressive — Handoff (2026-06-24)

## Scope

- Prefill values are from long-context integration hardening + latest synced prefill artifacts.
- Decode values are from the latest decode lifecycle recheck baseline and latest aggressive probe.
- For each lane:
  - Baseline = current authority baseline
  - Confirmed = measured promoted / shipped improvement (where present)
  - Aggressive bound = non-search upper-envelope or target ceiling (planning only)

## Inputs

- Prefill
  - [authority.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-hardening-20260624/authority.json)
  - [whole_prefill_by_ctx_raw.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-hardening-20260624/whole_prefill_by_ctx_raw.json)
  - [whole_prefill_by_ctx_raw.json (tensile)](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-hardening-20260624-tensile/whole_prefill_by_ctx_raw.json)
  - [decision.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-hardening-20260624/decision.json)
  - [runtime_overlap_by_ctx.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-hardening-20260624/runtime_overlap_by_ctx.json)
  - [kv_attention_split_timeseries.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-hardening-20260624/kv_attention_split_timeseries.json)
  - [route_coverage_by_ctx_and_role.json](/home/ubuntu/tinygrad-arkey/bench/qk-prefill-long-context-integration-hardening-20260624/route_coverage_by_ctx_and_role.json)
  - [result: `docs/prefill-long-context-integration-hardening-result-20260624.md`](/home/ubuntu/tinygrad-arkey/docs/prefill-long-context-integration-hardening-result-20260624.md)
- Decode
  - [latest baseline](/home/ubuntu/tinygrad-arkey/bench/qk-decode-lifecycle-recheck-bundle/decode-lifecycle-recheck-20260624-172026/throughput/current_context/wd_by_ctx.json)
  - [latest aggressive target probe baseline](/home/ubuntu/tinygrad-arkey/bench/qk-decode-aggressive-target-proof-20260624/throughput_baseline_probe.json)
  - [latest aggressive target probe](/home/ubuntu/tinygrad-arkey/bench/qk-decode-aggressive-target-proof-20260624/throughput_aggressive_probe.json)
  - [llama reference](https://huggingface.co)
  - [decode-lifecycle periodic result](/home/ubuntu/tinygrad-arkey/docs/decode-lifecycle-recheck-bundle-result-20260624.md)

## Summary

- Prefill: baseline authority is still `eightwave`, and no additional promoted prefill kernel path is required in this session.
- Decode: canonical W==D baseline is `101.6 / 99.8 / 97.4 / 92.9` tok/s @ctx512/1024/2048/4096.
- Latest aggressive decode probe (unpromoted) is `103.4 / 101.6 / 99.1 / 94.4` tok/s, and remains below the non-search envelope (`104.0 / 102.1 / 99.6 / 95.1`).
- Old PLRA remains `needs_confirm`; `eightwave + old_plra` is rejected.

## Prefill table (tok/s)

| ctx | Baseline (current default) | Confirmed `eightwave` | Confirmed Δ | Aggressive non-search bound* |
|---:|---:|---:|---:|---:|
| 512 | 3578.0 | 3578.0 | +0.00% | 4593.45 |
| 1024 | 3578.0 | 3578.0 | +0.00% | 4486.47 |
| 2048 | 3575.0 | 3575.0 | +0.00% | 4187.02 |
| 4096 | 3575.0 | 3575.0 | +0.00% | 3585.72 |
| 8192 | 3576.0 | 3576.0 | +0.00% | 2869.33 |

*Aggressive bound is the previously recorded non-search frontier corridor and is planning-only; it is not a measured closure point.

## Decode table (tok/s)

| ctx | Baseline (current default) | Confirmed (probe; not promoted) | Confirmed Δ | Aggressive non-search target |
|---:|---:|---:|---:|---:|
| 512 | 101.6 | 103.4 | +1.77% | 104.0 |
| 1024 | 99.8 | 101.6 | +1.80% | 102.1 |
| 2048 | 97.4 | 99.1 | +1.74% | 99.6 |
| 4096 | 92.9 | 94.4 | +1.61% | 95.1 |

## Source-bound interpretation

- Prefill table confirms stable graph-lane throughput across long contexts (8192), with no additional shipped gain to add above `eightwave` in this cycle.
- Decode confirmed gains are presently measurement-only and not yet shipped; the project remains in the same decode-lane authority with no-ship promotion status.
- Keep these rows as canonical handoff points so any future promotion explicitly references whether it is measured and shippable vs probe-only.

## Recommended next step

- Decode: keep `DECODE_ATTN_KV_IDENTITY` and `Q4K_GEMV_*` shipped defaults as-is; if the aggressive probe is re-routed, prove the same probe stack with `bench/qk_decode_lifecycle_recheck_periodic.py` before promotion.
- Prefill: continue with the current long-context integration hardening status and move directly to targeted runtime-provenance follow-up only if a comparator path is needed.

## 2026-06-24 (proof snapshot)

- New result doc: `docs/prefill-long-context-integration-hardening-result-20260624.md`
- Decode aggressive proof snapshot still uses `bench/qk-decode-aggressive-target-proof-20260624/` for comparison and shortfall analysis.

### Probe rows (for continuity)

| ctx | Prefill base | Prefill `pipe_tm2_tn2` | Prefill `pipe_tm4_tn2` | Decode baseline | Decode aggressive |
|---:|---:|---:|---:|---:|---:|
| 512 | 3572 | 4253 | 2332 | 101.9 | 103.4 |
| 1024 | 3483 | 4037 | 2263 | 100.1 | 101.6 |
| 2048 | 3226 | 3659 | 2139 | 97.6 | 99.1 |
| 4096 | 2789 | 3110 | 1937 | 93.1 | 94.4 |

Interpretation:
- Prefill direct closure still does not reach the documented aggressive upper corridor; `pipe_tm2_tn2` is strongest among measured prefill alternatives but remains below the planning bound.
- Decode remains close to, but short of, its non-search ceiling in this cycle.
