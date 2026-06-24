# Decode Elementwise Cost Split Result (Deliverable 3)

Date: 2026-06-20

Verdict: `PASS_DECODE_ELEMENTWISE_COST_SPLIT` — 99.2% of elementwise ms/token classified at every cell; the
dominant family is **`E_49152_32_3` = the FFN `silu(gate)*up` activation, confirmed at ~1.24 ms/token @ctx1024**.
Default decode behavior NOT changed.

## Method

Same clock-pinned two-layer timed instrument as Deliverable 1 (ProfileGraphEvent split rescaled to the clean W
wall; peak clock pinned, `auto` restored). Elementwise kernels (those the main classifier buckets `elementwise`)
sub-split into `ffn_activation` / `rope` / `residual_add` / `casts_copies` / `unclassified_elementwise`.

Tool: `extra/qk_decode_elementwise_cost_split.py`. Artifact:
`bench/qk-decode-attention-elementwise/elementwise_cost_split.json`.

## Result (ms/token, clock-pinned)

### baseline (flat across ctx, as Deliverable 0 predicted)

| ctx | wall ms | elementwise ms | %wall | ffn_activation | residual_add | rope | casts_copies | unclassified | classified |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 14.61 | 2.19 | 15.0 | **1.24** | 0.32 | 0.28 | 0.34 | 0.018 | 99.2% |
| 1024 | 14.98 | 2.20 | 14.7 | **1.24** | 0.32 | 0.28 | 0.34 | 0.018 | 99.2% |
| 4096 | 16.39 | 2.22 | 13.6 | **1.26** | 0.32 | 0.28 | 0.34 | 0.018 | 99.2% |

### q8 (q8 does NOT fuse the FFN activation — E_49152 persists)

| ctx | wall ms | elementwise ms | ffn_activation | residual_add | rope | casts_copies |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 14.09 | 2.30 | 1.20 | 0.40 | 0.29 | 0.38 |
| 4096 | 15.51 | 2.33 | 1.22 | 0.41 | 0.29 | 0.38 |

## Findings

1. **`E_49152_32_3` (FFN `silu(gate)*up`) owns ~1.24 ms/token** = 56% of elementwise, flat across ctx, present in
   both modes (q8 leaves it ~1.20 ms). Confirmed as the scope's ~1.4 ms target. It is the FFN activation that
   llama fuses inline into its MMVQ.
2. **It is launch-overhead-bound, not bandwidth-bound.** `E_49152` reads 2×12288 fp32 + writes 12288 ≈ 147 KB →
   ~0.15 µs of HBM work, yet costs ~33 µs/call (36 calls/token). That is ~200× the bandwidth bound = pure kernel
   launch/dispatch overhead of a tiny per-layer elementwise. → the win is **eliminating the launch (fusion)**, not
   making the elementwise faster.
3. Remaining elementwise is small and not a build target: `residual_add` 0.32 ms (`E_32_32_4*`, the x+sublayer
   adds), `rope` 0.28 ms (`E_2_8_16_4_4`, rotary), `casts_copies` 0.34 ms (`E_128_32_3`/`E_1536_32_3`, per-layer
   GEMV output reshape glue). Unclassified is negligible (0.018 ms).

## Pass gate

| gate | result |
|---|---|
| classifies ≥90% of elementwise ms/token (all cells) | PASS (99.2%) |
| identifies a repeated family ≥0.25 ms/token | PASS (`E_49152` 1.24 ms, `E_2_8_16_4_4` 0.29 ms) |
| confirms `E_49152_32_3` owns ~1.4 ms/token | PASS (1.24 ms) |

## Actionable target for Deliverable 4

Eliminate the `E_49152` launch by fusing `silu(gate)*up` so it does not round-trip through a standalone
elementwise kernel before `ffn_down`. The consumer `ffn_down` is a custom Q6_K GEMV that requires a realized
input (line 804 `self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))`, with the known-wart
`.contiguous()` at line 797 "TODO: remove the need for this contiguous"). Full elimination recovers up to
~1.24 ms/tok → ~73 tok/s @ctx1024 (from 66.9).

## Commands

```bash
PYTHONPATH=. python3 extra/qk_decode_elementwise_cost_split.py \
  --modes baseline,q8 --ckpts 512 1024 4096 --nmeas 20 --warmups 8 \
  --out bench/qk-decode-attention-elementwise/elementwise_cost_split.json
```

## Boundary

No decode default changed. Peak clock pinned only for measurement; `auto` restored after.
