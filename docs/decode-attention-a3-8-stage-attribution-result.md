# Decode Attention A3.8 Stage Attribution Result

## Verdict

`A3_8_ATTRIBUTION_READY__PARTIAL_PV_NEXT`

A3.8 audits why A3.6/A3.7 did not transfer before building a partial-PV payload.

It compares:

- A2 generated whole-cache skeleton
- A3.6 tile score+max
- A3.7 tile probability

The audit uses route signatures plus W==D deltas to decide whether metadata fusion is exhausted and whether partial PV is the next meaningful payload.

## Files

- Tool: `extra/qk_decode_attention_a3_8_stage_attribution.py`
- Artifact: `bench/qk-decode-attention-a3-8-stage-attribution/latest.json`

## Command

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_a3_8_stage_attribution.py
```

## Route Attribution

| Arm | Added vs A2 | Removed vs A2 | Route clean |
|---|---|---|---|
| A3.6 tile-max | `flash_tile_score_max_32_128` | `flash_max_32` | yes |
| A3.7 tile-prob | `flash_tile_score_max_32_128`, `flash_tile_prob_32_128` | `flash_max_32`, `flash_prob_32` | yes |

Both arms preserve:

- owned tile/combine absent
- `E_49152` absent
- token sample match
- generated whole-cache lifecycle

## W==D Attribution Table

| ctx | A2 tok/s | A3.6 tile-max tok/s | A3.6 delta | A3.7 tile-prob tok/s | A3.7 delta |
|---:|---:|---:|---:|---:|---:|
| 512 | 78.1 | 77.9 | -0.2 | 76.4 | -1.7 |
| 1024 | 75.4 | 75.2 | -0.2 | 74.2 | -1.2 |
| 2048 | 69.6 | 69.3 | -0.3 | 69.2 | -0.4 |
| 4096 | 60.5 | 60.1 | -0.4 | 61.2 | +0.7 |

## Diagnosis

Max/prob metadata replacement did not materially transfer.

Evidence:

- A3.6 removes only `flash_max_32`; all ctx points are flat/slightly negative.
- A3.7 removes both `flash_max_32` and `flash_prob_32`; short/mid contexts regress and only ctx4096 has a small uptick.
- Both candidates are route-clean, so the non-transfer is not a route/materialization/token artifact.

## Interpretation

The metadata stages are not the main decode attention gap.

The next meaningful stage is partial PV, because A2 still carries `flash_partial_coop_vec_whole_cache_32_128`, and metadata-only tile work has failed to transfer. The remaining performance gap is likely dominated by score memory traffic plus partial PV lifecycle, not max/prob metadata kernels.

## Decision

Proceed to partial-PV tile payload.

Next executable step: A3.9 should build a generated tile candidate that removes or replaces `flash_partial_coop_vec_whole_cache_32_128` while preserving A2 whole-cache/no-`E_49152` hygiene and TILE+COMBINE attribution.
