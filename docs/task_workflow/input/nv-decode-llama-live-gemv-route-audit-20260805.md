# NVIDIA d512 llama live GEMV route audit

Date: 2026-08-05. Scope: Qwen3-8B-Q4_K_M, d512, RTX 5090, driver
595.84. This is a CPU/read-only causal audit. It changes no runtime code,
default, or route and ran no GPU work.

## Verdict

All **217/217** observed Q4_K/Q6_K decode projection nodes use llama.cpp's
**MMVQ** `mul_mat_vec_q<type, 1, has_fusion, false>` path. None uses MMQ
`mul_mat_q`, DP4A MMQ, or integer-MMA MMQ. The checked-in timeline ledger's
class name `mmq` is therefore an umbrella label inherited from
`cuda_graph_timeline_ledger.py::classify`; it is not the selected llama route.

At batch one, both `use_mul_mat_vec_q` and (on this architecture) the MMQ
capability predicate can be true. Dispatch order is decisive:
`ggml-cuda.cu:2652-2655` selects `ggml_cuda_mul_mat_vec_q` before
`ggml_cuda_mul_mat_q`. `ggml_cuda_should_use_mmvq` admits every non-CDNA
quantized call with `ne11 <= MMVQ_MAX_BATCH_SIZE` (8), and this decode has
`ne11=1`. Consequently, source capability such as Q4_K/Q6_K integer MMA in
`mmq.cuh` cannot be used as an explanation of the observed d512 numbers.

The live Blackwell MMVQ route uses the generic parameter table: one output row
per CTA, four warps per row, block `(32,4,1)`, and grid `(N,1,1)`. Q4_K and
Q6_K both consume a separately produced Q8_1 activation and execute packed
signed integer dot products through `ggml_cuda_dp4a` / CUDA `__dp4a`. The
observed Q6 cubin independently confirms `IDP.4A.S8.S8`; no MMA instruction is
part of the live route.

## Exact route and ownership census

`N x K` means output rows by reduction width. `f` is llama's compile-time
`has_fusion` variant. Every row is MMVQ with `ncols_dst=1`, one row/CTA,
four warps/CTA, Q8_1 plus DP4A, and zero dynamic shared memory. `llama us` is
the median across the 27 steady replays of the sum for that exact semantic
population. For every row, interval union equals node sum and overlap with
all other MMVQ intervals is `0.000 us`; thus the listed time owns that amount
of the MMVQ anchor union. It is not an independently recoverable wall claim,
because support kernels can overlap inside the same intervals.

| semantic population | quant | N x K | count | f | grid / block | regs / static smem | live arithmetic | llama us / anchor share |
| --- | --- | ---: | ---: | --- | --- | ---: | --- | ---: |
| attention Q | Q4_K | 4096 x 4096 | 36 | false | `4096x1x1 / 32x4x1` | 56 / 384 B | DP4A, not MMA | 342.881 / 9.58% |
| attention K | Q4_K | 1024 x 4096 | 36 | false | `1024x1x1 / 32x4x1` | 56 / 384 B | DP4A, not MMA | 117.376 / 3.28% |
| attention V, Q4 layers | Q4_K | 1024 x 4096 | 18 | false | `1024x1x1 / 32x4x1` | 56 / 384 B | DP4A, not MMA | 75.838 / 2.12% |
| attention V, Q6 layers | Q6_K | 1024 x 4096 | 18 | false | `1024x1x1 / 32x4x1` | 48 / 384 B | DP4A (`IDP.4A`), not MMA | 89.437 / 2.50% |
| attention O, layers 0-34 | Q4_K | 4096 x 4096 | 35 | true | `4096x1x1 / 32x4x1` | 48 / 768 B | DP4A plus fused residual | 407.392 / 11.38% |
| attention O, final layer | Q4_K | 4096 x 4096 | 1 | false | `4096x1x1 / 32x4x1` | 56 / 384 B | DP4A, not MMA | 11.072 / 0.31% |
| fused gate/up | Q4_K | 12288 x 4096 | 36 | true | `12288x1x1 / 32x4x1` | 48 / 768 B | DP4A plus fused gate | 1364.038 / 38.10% |
| FFN down, Q4 layers | Q4_K | 4096 x 12288 | 18 | true | `4096x1x1 / 32x4x1` | 48 / 768 B | DP4A plus fused residual | 346.209 / 9.67% |
| FFN down, Q6 layers | Q6_K | 4096 x 12288 | 18 | true | `4096x1x1 / 32x4x1` | 46 / 768 B | DP4A plus fused residual | 520.836 / 14.55% |
| vocabulary | Q6_K | 151936 x 4096 | 1 | false | `151936x1x1 / 32x4x1` | 48 / 384 B | DP4A, not MMA | 303.618 / 8.48% |

The sum of the independently medianed family rows is `3578.697 us`; the
timeline authority is `3579.816 us` MMVQ union. The `1.119 us` difference is
expected median non-additivity. The authority remains the aggregate timeline
row. Q4_K owns 180 nodes and approximately 2664.806 us of independently
medianed family time; Q6_K owns 37 nodes and approximately 913.891 us.

## Reconciliation to native tinygrad

Native numbers below are the representative semantic profile composition
scaled by `0.9842737159` to the independent `5291.424 us` marker-light device
window, exactly as the checked-in native semantic ledger specifies. They are
quantized-core ownership, not a new wall measurement. In particular, the Q6
attention partial's required external sum is classified as support work, so a
complete replacement must include it even though it is absent from the core
column.

| actual model population | llama live MMVQ us | native installed core / us | native - llama us | causal status |
| --- | ---: | --- | ---: | --- |
| Q6 attention V, 18 | 89.437 | `q6k_gen_partial_1024_4096_4` / 307.314 | **+217.877** | strongest live family; all-18 CUDA substitution recovered 179-184 us, but native one-consumer Q8+DP4A and direct-output candidates failed |
| Q4 FFN down, 18 | 346.209 | `q4k_g3_lanemap_gemv_4096_12288` / 443.191 | **+96.982** | exact CUDA family substitution recovered 65.8-66.1 us; residual fusion itself was neutral, locating the signal in the MMVQ substrate |
| Q6 FFN down, 18 | 520.836 | `q6k_gen_coop_4096_12288_inkernel` / 601.903 | **+81.067** | positive ownership gap, but four-or-more llama replacements failed the token contract; not family-qualified |
| Q4 attention K, 36 | 117.376 | `q4k_g3_lanemap_gemv_1024_4096` / 152.381 | **+35.005** | small/context-sensitive: the identical-shape Q4 attention-V population is native-faster |
| Q6 vocabulary, 1 | 303.618 | `q6k_gen_coop_151936_4096_inkernel` / 314.432 | **+10.814** | near parity; not mechanism-scale |
| Q4 attention V, 18 | 75.838 | `q4k_g3_lanemap_gemv_1024_4096` / 75.151 | -0.687 | no deficit |
| Q4 fused gate/up, 36 | 1364.038 | `q4k_g3_lanemap_gemv_w1w3fused_12288_4096` / 1360.219 | -3.819 | native already fused; no deficit |
| Q4 attention Q, 36 | 342.881 | `q4k_g3_lanemap_gemv_4096_4096` / 316.227 | -26.654 | native faster; exact llama substitution regressed 39.374 us/token |
| Q4 attention O, 36 combined | 418.464 | `q4k_g3_lanemap_gemv_4096_4096` / 311.786 | -106.678 | native core faster; llama substitution regressed 21.213 us/token; support/epilogue exposure is separate |

These independently medianed splits sum to approximately `+303.907 us`,
close to the aggregate calibrated quant-core authority of `+302.788 us`.
Only the aggregate is additive with the device ledger. The split is a ranking
instrument and must not be booked as a wall forecast.

## Ranked recovery interpretation

1. **Q6 attention V is the primary observed MMVQ recovery source.** It owns
   about `+217.9 us` of the family-split core gap, and the all-18 CUDA A/B
   independently shows a `179-184 us/token` causal direction. The mechanism
   to reproduce is Q8_1 plus efficient four-warp DP4A ownership (and eventually
   shared packing if it survives the real topology), not integer MMA.
2. **Q4 FFN down is the next qualified family.** Its ownership gap is about
   `+97.0 us`, and its correctness-passing CUDA A/B supplies a replicated
   `65.8-66.1 us/token` signal. The residual epilogue is not the signal.
3. **Q6 FFN down is third by timing, but not yet executable evidence.** Its
   `+81.1 us` ownership gap is real; its full-family correctness boundary is
   not. Treat it as an exact-semantic construction problem before timing.
4. **Q4 1024 attention is a small contextual tail, not a broad kernel-family
   indictment.** K is `+35.0 us`, while the same-shape Q4 V population is
   `-0.7 us`. The combined Q4 1024 family gap is only about `34.3 us`.
5. **Vocabulary is already close.** Its `+10.8 us` gap cannot supply parity.

Do not spend parity effort replacing Q4 attention Q/O or native fused gate/up:
their native quantized cores are already faster than the live llama MMVQs, and
the token A/Bs for Q/O moved wall in the wrong direction. Their remaining
support-work exposure belongs to the overlap/fusion ledger, not to GEMV-core
recovery.

## Evidence and reproducibility

- tinygrad HEAD: `a1a51c349d1b8c55a6373631913ea7845e99cc8d`
- llama.cpp HEAD: `ac4cddeb0dbd778f650bf568f6f08344a06abe3a`
- timeline ledger SHA-256:
  `6af6fdc86e2d6ccf1b6e33d51ce64650a88ebedb97cdf47d9e79da1e0ca4382d`
- semantic manifest SHA-256:
  `cc629b5ba78b5dfe00c65c0392eeeaeb88ffbfdc2829881dae87aa1339e2a973`
- native semantic ledger SHA-256:
  `792fde5f9cce03db9fa6f393f84079e7266e0a7c8c7698d9af4a5a613f493770`
- profiled trace SHA-256 (already pinned by the native ledger):
  `a3f990c5301f98a6cdee4e9a4f1abf11974d45e1c1a210443a827201974dac41`
- llama MMVQ source SHA-256:
  `19cec84e1c293c133b8101b142bb497c726023123deaca09b18e8d81b31f35e9`

The exact per-family times were recomputed CPU-only by joining each checked-in
semantic row's MMVQ graph-node ID to graph 2 in the pinned trace, dropping the
same first two warmup replays as the timeline ledger, grouping the remaining
27 replays by `(role, quant, N, K, has_fusion)`, and taking the median of each
per-replay sum. Every family had stable count, union equal to node sum, and
zero overlap with another MMVQ family.
