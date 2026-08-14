# NV Q4 FFN-down: DP4A datapath verification at fixed geometry (2026-08-14)

Date: 2026-08-14
Branch: `nvidia-bringup-20260731`
Status: **measurement record.** Verifies the standing claim that DP4A, not fp32
FMA, is the remaining FFN-down datapath lever after the four-warp geometry is
applied. Nothing is promoted.

## 0. The claim under test

`nv-q4-down-fp16-geometry-four-warp-verdict-20260814.md` concluded "geometry is
necessary but not sufficient; llama's remaining advantage is the DP4A datapath
(4x fp32 FMA peak)." This record tests that claim by holding geometry fixed and
varying only the datapath.

## 1. Fixed-geometry standalone isolation (L2-resident, zeroed weights)

Three kernels share the exact four-warp/row 128-thread geometry
(`emit_four_warp_fp16_direct` vs `emit_four_warp_direct` with and without the
llama-exact `dp4a(0x01010101, q8)` correction sum), each launched 4096x12288
Q4_K, 4096 blocks, 128 threads, 5000 passes:

| kernel | per_launch | weight GB/s |
| --- | ---: | ---: |
| fp32 FMA (4-warp fp16 direct) | 17.34 us | 1633 |
| DP4A, scalar byte-extract correction | 11.07 us | 2558 |
| DP4A, llama-exact `sum_dp4a` correction | 9.91 us | 2858 |
| llama `mul_mat_vec_q<Q4_K>` (pinned) | 8.43 us | - |

PTX census at the same geometry: fp16 FMA emits 32 `fma.rn.f32` and 0 `dp4a`;
the llama-exact DP4A spelling emits 8 `dp4a` and 4 FMA. Holding geometry fixed,
switching fp32 FMA to DP4A moves 17.34 -> 9.91 us, closing ~83% of the gap to
llama's 8.43 us. **DP4A is the datapath lever.**

## 2. In-loop reality check (d128, same-context reverse bracket)

The datapath lever does not yet show in the full route because the current DP4A
spelling adds a separate `q8_1_llama_provider_12288` node per block:

| route (18 blocks leased, d128) | median ms/token | vs control 5.9827 |
| --- | ---: | ---: |
| control (installed 1-warp fp16) | 5.9827 | baseline |
| fp16 geometry (4-warp, 1 node/block) | 5.9031 | +1.35% |
| DP4A route (4-warp, 2 nodes/block) | 5.9579 | +0.42% |

All arms share token stream hash
`f7cf348f26306df073f3f6360dc83e9369e5c97096f383d6f08d1a930b66ec19`. The DP4A
consumer is faster per launch, but the standalone provider node costs more than
the datapath saves, so the net in-loop DP4A route is behind the no-provider fp16
geometry route. This is the same +0.5%-class result as
`nv-q4-down-dp4a-resadd-18block-gate-20260814.md`, now bracketed at the same
context as the fp16 geometry route.

## 3. Verdict

The claim is verified and refined:

- **DP4A is the datapath lever**: at fixed 4-warp geometry it closes ~83% of the
  standalone gap to llama (9.91 vs 8.43 us, versus 17.34 us for fp32 FMA).
- **The provider node is the gating confound**: the current DP4A route loses to
  the no-provider fp16 geometry route in-loop because the standalone
  `q8_1_llama_provider_12288` node costs more than the datapath saves.
- **The move is the producer-fold**, not more datapath tuning: fold Q8_1
  quantization into the w1w3 producer epilogue (already owns silu*up in-kernel)
  so the DP4A consumer gains the datapath lever with zero added nodes. That is
  the re-derivation scoped in `nv-gemv-core-recovery-status-20260813.md` section 3.

## 4. Artifacts

- `extra/llm_research/microbench/dp4a_direct_wall_harness.cu` (added): launches
  the four-warp Q8/DP4A FFN-down consumer for the fixed-geometry wall census.

## 5. References

- `nv-q4-down-fp16-geometry-four-warp-verdict-20260814.md` (the +1.35% geometry result)
- `nv-ffn-down-gap-occupancy-proof-20260814.md` (geometry -> occupancy -> DRAM)
- `nv-decode-datapath-fma-vs-dp4a-measurement-20260814.md` (4x DP4A peak)
- `nv-q4-down-dp4a-resadd-18block-gate-20260814.md` (+0.5% with the provider node)
