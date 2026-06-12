# Vdot Premise V1 Roofline

Date: 2026-06-12

Device: AMD Radeon RX 7900 XTX / gfx1100.

Peak assumptions: AMD lists RX 7900 XTX at 61.4 FP32 TFLOP/s and 960 GB/s memory bandwidth, giving a ridge point of about 64.0 ops/byte for FP32-vector work. This report uses packed quant weight bytes only; if the activation vector is reread per row, true bytes are higher and arithmetic intensity is lower.

## Result

The accepted v1 Q4/Q6 kernels are memory/schedule-bound by roofline. Their logical dot intensity is only 2.4-3.6 ops per packed quant byte, far below the ~64.0 ops/byte ridge point. Their logical dot throughput is also only 0.3-1.5 TFLOP/s, so the remaining gap is not explained by a saturated dot/compute pipeline.

| model | fmt | tensor | policy | device ms | quant GB/s | logical TFLOP/s | ops/byte | % FP32 peak | % mem peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8B | Q4_K | blk.0.ffn_gate.weight | v1 parts=1 LOCAL:0:64 | 0.067 | 421.10 | 1.50 | 3.56 | 2.4% | 43.9% |
| 8B | Q4_K | blk.0.ffn_gate.weight | generated p2 LOCAL:0:32 | 0.154 | 183.61 | 0.65 | 3.56 | 1.1% | 19.1% |
| 8B | Q4_K | blk.4.ffn_down.weight | v1 parts=4 LOCAL:0:32 | 0.105 | 270.56 | 0.96 | 3.56 | 1.6% | 28.2% |
| 8B | Q4_K | blk.0.attn_q.weight | v1 parts=1 LOCAL:0:64 | 0.050 | 189.14 | 0.67 | 3.56 | 1.1% | 19.7% |
| 8B | Q6_K | blk.0.ffn_down.weight | v1 parts=1 LOCAL:0:64 | 0.314 | 131.63 | 0.32 | 2.44 | 0.5% | 13.7% |
| 14B | Q4_K | blk.0.ffn_gate.weight | v1 parts=1 LOCAL:0:64 | 0.139 | 360.89 | 1.28 | 3.56 | 2.1% | 37.6% |
| 14B | Q6_K | blk.0.ffn_down.weight | v1 parts=1 LOCAL:0:64 | 0.473 | 154.55 | 0.38 | 2.44 | 0.6% | 16.1% |

## Codegen Notes

- `8b-q4-ffn-gate-v1-debug4.log` shows `q4k_gemv_partial_12288_4096_1` as a scheduled local kernel with `amdgpu_flat_work_group_size(1, 64)`, 32-bit Q4 word loads, half activation loads, nibble extraction, and scalar fp32 accumulation.
- `8b-q6-ffn-down-v1-debug4.log` shows `q6k_gemv_partial_4096_12288_1` as a scheduled local kernel with `amdgpu_flat_work_group_size(1, 64)`, 16-bit packed Q6 storage loads, half activation loads, bit extraction, and scalar fp32 accumulation.
- Neither accepted v1 kernel emits `v_dot4`/`dp4a`. That does not by itself make packed dot the next lever, because the measured v1 roofline is not compute-bound.

## Gate Verdict

Do not start renderer/core packed-dot lowering as the next default task. The cheap premise check says the current accepted path is memory/schedule-bound. A future compiler-research path should target semantic packed layout plus memory/schedule/codegen together, not just isolated `v_dot4` emission.
