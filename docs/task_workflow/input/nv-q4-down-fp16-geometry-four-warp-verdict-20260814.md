# NV Q4 FFN-down: four-warp fp16 geometry verdict (2026-08-14)

Date: 2026-08-14
Branch: `nvidia-bringup-20260731`
Status: **measurement record.** Isolates the occupancy/thread-geometry lever from
`nv-ffn-down-gap-occupancy-proof-20260814.md` without the Q8/DP4A datapath or a
separate provider node. Not promoted; production tok/s unchanged.

## 0. The question this answers

The occupancy proof reduced the FFN-down gap to `1 warp/row -> 38.8% occupancy
-> 54.5% DRAM` vs llama's `4 warps/row -> 66.3% -> 77.2%`. Its "worth-it fix"
listed two things: adopt the 4-warp geometry AND fold Q8_1 into the producer.
This record tests the geometry half alone, keeping the installed fp16 datapath
byte-identical, to see how much of the gap the geometry lever actually carries
before committing to the larger producer-fold.

## 1. What was implemented

`emit_four_warp_fp16_direct` in `tinygrad/llm/q4k_ffn_down_mmvq.py`: 4096 rows x
128 threads/row (4 warps), each warp owns 12 of the 48 Q4 blocks and the 32 lanes
keep the installed `(word_col=lane%8, sub_group=lane//8)` partition over 3 blocks
each. The datapath reuses `_q4k_block_dot_packed_load`, so weights and the
`w1w3fused16` fp16 activation are consumed with the same packed loads and fp32
FMA as production. Cross-warp partials combine through shared memory, then a
staged shuffle reduces the 32 lanes. `resadd=True` absorbs M2b `h + ffn_out`.

It is admitted only by `Q4KFFNDownMMVQAdmission(index, fp16_fma=True)`. The
consumer's `program_id` ends in `.gemv` so the M5 epilogue-absorption validator
(`.gemv`/`.q8_provider`) and the M2b residual validator (`.gemv`/`.consumer`)
both admit the zero-copy folds. This was the load-bearing detail: a bespoke
`.fp16_consumer` id silently fell back to the materializing flat-buffer ABI and
cost two extra transport kernels per block.

## 2. Arithmetic (what the measurement must show)

```text
installed: 18 x q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288   (1 warp/row, 32 thr)
candidate: 18 x q4k_fp16_mmvq_direct_4096_12288_epi_ffnresadd   (4 warps/row, 128 thr)
no q8_1_llama_provider node, no added materialize/cast node
```

The numeric change is fp32 reduction reorder only (four warp partials instead of
one). There is no activation quantization, so the drift must be far below the
Q8 route's 2.98e-3 relative L2.

## 3. Results (d128 reverse-bracket, token-exact, DEV=NV, RTX 5090)

| metric | value |
| --- | ---: |
| standalone L2-resident wall (zeroed weights) | 20.47 -> 17.31 us |
| standalone kernel numeric drift (max abs, 4096 rows) | 5.0e-6 |
| full-logits relative L2 (151936 logits) | 5.29e-4 |
| full-logits max abs | 1.17e-2 |
| topology (program-count delta) | 0 extra nodes (705 == 705) |
| control bracket median | 5.9827 ms/token |
| candidate median | 5.9031 ms/token |
| wall delta | -79.6 us/token, **+1.35%** |

Semantic gate: `tokens_equal`, `argmax_equal`, and `top_k_order_equal` all true;
relative L2 (5.29e-4) is inside the 1e-3 gate. The max-abs logit (1.17e-2)
exceeds the 1e-2 historical atol, which the harness reports as
`reported_non_authoritative`; the authoritative token/argmax/relative-L2 gates
pass.

## 4. Verdict and the remaining lever

The geometry lever alone, even with a clean zero-copy fold and no added node,
carries about **-80 us/token (+1.35%)**. It does not reach llama's FFN-down wall:
llama's remaining advantage is the DP4A datapath (4x fp32 FMA peak,
`nv-decode-datapath-fma-vs-dp4a-measurement-20260814.md`), which must be folded
into the w1w3 producer epilogue so it adds no provider node. Geometry is
necessary but not sufficient. The producer-fold + DP4A re-derivation in
`nv-gemv-core-recovery-status-20260813.md` section 3 remains the open move.

## 5. References

- `nv-ffn-down-gap-occupancy-proof-20260814.md` (the occupancy causal chain)
- `nv-gemv-core-recovery-status-20260813.md` (the +97.0 us Q4 FFN-down row)
- `nv-decode-datapath-fma-vs-dp4a-measurement-20260814.md` (DP4A 4x fp32 FMA)
- `nv-q4-down-dp4a-resadd-18block-gate-20260814.md` (+0.5% with the provider node)
