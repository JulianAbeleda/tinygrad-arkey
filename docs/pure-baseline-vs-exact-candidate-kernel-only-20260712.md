# Pure baseline versus exact candidate: kernel-only comparison

Date: 2026-07-12

## Contract

- Device: AMD Radeon RX 7900 XTX (`gfx1100`)
- Shape: `M=512, N=12288, K=4096`
- Arithmetic count: `2*M*N*K`
- Inputs and result contract: identical constant-case tensors from the execution authority
- Baseline: tinygrad scheduler matmul program selected from the same realized graph
- Candidate: exact admitted candidate `81c27275d1aad1bb8147c5c5cdaa8000e9375e81f3d085b49d62064a731313d6`
- Measurement: direct compiled-kernel calls with `wait=True`; compilation, allocation, transfers, and graph overhead excluded
- Protocol: five warmups per kernel, then 21 randomized interleaved samples per kernel (seed 14)
- Clock request: manual performance mode, SCLK state 2, MCLK state 3; request succeeded
- Correctness: both programs passed the execution authority's constant-case output contract

## Result

| Program | Median | Minimum | p90 | Median TFLOP/s | Maximum TFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| scheduler baseline | 2.64668 ms | 2.42956 ms | 2.73640 ms | 19.4733 | 21.2136 |
| exact candidate | 0.75288 ms | 0.66640 ms | 0.91600 ms | 68.4566 | 77.3403 |

The exact candidate is **3.5154x faster at the median**. This is a kernel-only comparison and is not an end-to-end prefill claim.

Baseline samples (ms):

```text
2.62220, 2.61556, 2.67388, 2.63368, 2.70896, 2.63948, 2.69816,
2.58480, 2.64840, 2.64668, 2.67836, 2.65512, 2.60284, 2.76632,
2.42956, 2.73640, 2.76008, 2.50340, 2.61924, 2.68948, 2.58804
```

Candidate samples (ms):

```text
0.67976, 0.87704, 0.73800, 0.68132, 0.93012, 0.75288, 0.68304,
0.68780, 0.90832, 0.91504, 0.89380, 0.90480, 0.91836, 0.91600,
0.75352, 0.67356, 0.66640, 0.86380, 0.69224, 0.68432, 0.74612
```

## Compiled resource identity

| Property | scheduler baseline | exact candidate |
| --- | --- | --- |
| function | `r_8_48_32_4_2_2_2_4_4_256_2` | `r_4_96_32_4_2_2_2_2_4_2_128_2_2` |
| global size | `[48, 8, 1]` | `[96, 4, 1]` |
| local size | `[32, 4, 1]` | `[32, 4, 2]` |
| binary SHA-256 | `7bb5022e36f18f867679e8e44eaac1b3826366abd2ef5ac89ad407912c60fbbd` | `e5b988f008de36242bff886b46daae1dc82816547832c0c63da72ef7c84b6c1c` |
| ISA SHA-256 | `b73cd4c9389a5359d23db0d398f1febd8b1d6cfb0fc8dff55b24e737d905c022` | `8ccb4d34a2e4e17482503e0edc5baeca0228fe3b928305c14523cf59ecf46c86` |
| VGPR | 201 | 113 |
| SGPR | 18 | 18 |
| LDS | 8192 bytes | 20480 bytes |
| workgroup threads | 128 | 256 |
| VGPR/SGPR spills | 0 / 0 | 0 / 0 |
| scratch | 0 | 0 |
| ISA lines | 731 | 480 |

Both captures report strict pure compiler-rendered programs with no forbidden markers. Resource authority is the AMDGPU code-object notes emitted for `gfx1100`.

## Interpretation

The candidate trades 2.5x more LDS and twice as many workgroup threads for 43.8% fewer VGPRs, a smaller instruction body, and substantially higher observed throughput. This supports the current diagnosis that the scheduler baseline is constrained by its generated kernel geometry/resource profile rather than by the theoretical device compute roof alone.

The candidate distribution is bimodal (`~0.67-0.75 ms` and `~0.86-0.93 ms`) even with the successful clock request. Therefore 68.46 TFLOP/s is the defensible median for this run; the 77.34 TFLOP/s maximum is evidence of attainable execution, not the stable headline. A subsequent lane should correlate per-sample clock/power telemetry or use a longer pin-verification protocol before attributing the variance to kernel behavior.

## Provenance

- tinygrad revision: `59499fc935da7d9a9fe8429761f434cdaa202715`
- clean worktree at measurement time
- raw artifact: `/tmp/pure-baseline-vs-exact-candidate-kernel-only-20260712.json`
- memory DPM after the run: 1249 MHz selected
- core clock telemetry immediately after the run: 1762 MHz
- performance policy was restored by the clock-pin context after measurement
