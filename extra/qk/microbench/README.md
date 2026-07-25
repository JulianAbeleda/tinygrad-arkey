# wmma_peak.cpp — measured achievable WMMA peak (gfx1100)

Answers "what is the real fp16 WMMA ceiling on this chip", without trusting a spec sheet or a
counter's semantics. Pure back-to-back `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` on
register-resident fragments: NACC=8 independent accumulators to cover the WMMA dependency
latency, runtime trip count so the loop is not folded, and a never-taken store so the result
stays live.

    hipcc --offload-arch=gfx1100 -O3 wmma_peak.cpp -o wmma_peak && ./wmma_peak

Verify purity before believing a number (`--save-temps`, inspect the .s):
`v_wmma_f32_16x16x16_f16` == NACC, `global_load` == 0, `ds_` == 0.

## Result, RX 7900 XTX / gfx1100, 2026-07-24
    waves=16384 iters=20000 nacc=8  ->  105.5 / 104.6 TFLOPS

= 86% of the 122.8 TF spec figure, 171% of 61.4 TF. **WMMA reaches dual-issue-class rates**, so
61.4 is not the ceiling. Use **~105 TFLOPS** as the achievable denominator for any
efficiency claim on this device; 122.8 is unreachable in practice and 61.4 flatters results ~1.7x.
