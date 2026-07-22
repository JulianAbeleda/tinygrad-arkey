# Phase 0: Baselines and Ceilings

## Routes
- 8B fp16-overlay: Qwen3-8B, Q4_K_M weights, prefill_v2 fp16 Q/K/V
- 14B packed: Qwen3-14B, Q4_K_M weights, prefill_v2 fp16 Q/K/V

## Common attention boundary
Both routes produce fp16 Q/K/V activations at `model.py:482-488`:
- reshape to (B, Hkv, G, T, Hd) with G = Hq/Hkv
- K/V broadcast over GQA group dimension

## Shapes at gate points
| Route | Hq | Hkv | G | Hd | T=512 KV=512 | T=2048 KV=2048 |
|-------|-----|-----|---|-----|---------------|-----------------|
| 8B | 32 | 8 | 4 | 128 | B=1, GQA group | B=1, GQA group |
| 14B | 40 | 8 | 5 | 128 | B=1, GQA group | B=1, GQA group |

## Empirical ceilings (gfx1100)
- C_peak fp16 WMMA: 7.1 TFLOP/s (4096×4096 matmul)
- B_peak: 5.2 GB/s (256MB elementwise)

## Materialized attention baselines (no WMMA, no composite)
| Route | T=KV=512 | T=KV=2048 |
|-------|----------|-----------|
| 8B | 8.0ms | 27.3ms |
| 14B | 8.4ms | 33.9ms |

Score buffer: T×KV×fp16 = 0.5MB (512) / 8MB (2048) per head-group. Both routes spill score and probs to HBM.
