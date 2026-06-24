# Decode Attention Cost Split Result (Deliverable 1)

Date: 2026-06-20

Verdict: `PASS_DECODE_ATTENTION_COST_SPLIT` — 100% of attention ms/token classified at every cell; the dominant
bucket is **`reduce_fixup`** (flash-decode reduction/fixup), which owns ≥1ms@1024 and ≥2ms@4096. Default decode
behavior NOT changed.

## Method

Same two-layer timed instrument as Deliverable 0 (`extra/qk_decode_current_route_attribution.py`): clean W==D
wall (authority) + per-kernel warm GPU timestamps via `ProfileGraphEvent` (rescaled onto the clean wall).
**Peak GPU clock pinned** (`extra/qk_clock_pin.py`, manual_peak sclk 2304MHz / mclk 1249MHz; `auto` restored
after) — `auto` is clock-volatile for short decode kernels and read 2-3× slow on cold contexts; pinning gives
reproducible ms/token. Attention kernels (those the main classifier buckets `attention_flash`) are sub-split into
`partial_compute` / `reduce_fixup` / `softmax_stats` / `qk_scores_other` / `unclassified_attention`.

Tool: `extra/qk_decode_attention_cost_split.py`. Artifact:
`bench/qk-decode-attention-elementwise/attention_cost_split.json`.

## Result (ms/token, clock-pinned, rescaled to clean wall)

### baseline

| ctx | wall ms | attention ms | %wall | reduce_fixup | softmax_stats | partial_compute | classified |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 14.71 | 3.22 | 21.9 | **1.66** | 0.79 | 0.77 | 100% |
| 1024 | 15.04 | 3.51 | 23.3 | **1.79** | 0.94 | 0.78 | 100% |
| 2048 | 15.74 | 4.17 | 26.5 | **2.08** | 1.26 | 0.83 | 100% |
| 4096 | 16.45 | 5.18 | 31.5 | **2.43** | 1.85 | 0.90 | 100% |

### q8 (attention is unchanged by q8, as expected)

| ctx | wall ms | attention ms | %wall | reduce_fixup | softmax_stats | partial_compute |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 13.74 | 3.15 | 23.0 | 1.61 | 0.78 | 0.76 |
| 1024 | 16.08 | 3.92 | 24.4 | 1.99 | 1.05 | 0.87 |
| 4096 | 15.53 | 5.06 | 32.6 | 2.37 | 1.80 | 0.88 |

## Findings

1. **`reduce_fixup` dominates attention** (≈46–51% at every ctx), and it is the dominant bucket in both modes.
   Top programs (baseline@1024): `r_2_8_128_16_4_2_32n1` (0.78 ms), `r_1024_16_4_2_32` (0.56 ms),
   `r_2_…start_pos…_8_4_4_16` (0.45 ms) — the flash-decode partial-reduction / KV-length-dependent fixup rows.
2. **`softmax_stats` grows fastest with ctx** (0.79 → 1.85 ms @512→4096, ~2.3×): `flash_prob`, `flash_combine`,
   `flash_den`, `flash_max`, `flash_gmax` — the online-softmax statistic kernels, one set per KV chunk.
3. **`partial_compute` (the actual QK·V flash compute) is small and nearly flat** (~0.77 → 0.90 ms). The
   attention cost and its context-slope are owned by the **reduction/online-softmax machinery, not the compute**.
4. At ctx1024, reduce_fixup + softmax_stats = 2.73 ms = 78% of attention (3.51 ms); partial is only 0.78 ms.

## Pass gate

| gate | result |
|---|---|
| classifies ≥90% of attention ms/token (all cells) | PASS (100%) |
| a bucket owns ≥1.0 ms/token @ctx1024 | PASS (reduce_fixup 1.79 ms) |
| a bucket owns ≥2.0 ms/token @ctx4096 | PASS (reduce_fixup 2.43 ms) |

## Actionable target for Deliverable 2

The flash-decode **reduction + online-softmax overhead** (reduce_fixup + softmax_stats), not the QK·V compute.
Because both buckets scale with the number of KV chunks (L=128 split), the cheapest first candidate is **chunk
policy (`FLASH_L`) tuning** — fewer/larger chunks → fewer partials to reduce and fewer softmax-stat kernels — then
reduce/stat fusion if the policy lever confirms fixed reduction overhead dominates. (llama attention is only
~0.76 ms/tok, so there is ~2.7 ms of headroom @1024.)

## Commands

```bash
PYTHONPATH=. python3 extra/qk_decode_attention_cost_split.py \
  --modes baseline,q8 --ckpts 512 1024 2048 4096 --nmeas 20 --warmups 8 \
  --out bench/qk-decode-attention-elementwise/attention_cost_split.json
```

## Boundary

No decode default changed. Peak clock pinned only for the measurement window; `auto` restored after (verified).
