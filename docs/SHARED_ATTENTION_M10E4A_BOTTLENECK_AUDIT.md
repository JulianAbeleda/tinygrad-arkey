# M10e4a shared-attention bottleneck audit

Date: 2026-07-23  
Target: AMD gfx1100, Q=512, Hd=128, fp16, 8B Hq/Hkv=32/8 and 14B Hq/Hkv=40/8

## Result

The fused kernel is correct and removes score/probability materialization, but the current benchmark is not an allocation-free kernel measurement. Input transfers and numeric readbacks are outside the timed region; each timed callback still constructs a fresh mask/attention graph and realizes a fresh result, so scheduling and output allocation are not proven excluded.

The smallest next kernel optimization is a guarded one-wave synchronization change: remove the workgroup barrier between the eight per-lane LDS stores and sixteen per-lane LDS loads, while retaining the required wave-local LDS completion ordering (for example, an appropriate `s_waitcnt lgkmcnt(0)`). This must be accepted only after the existing exhaustive GPU-baseline gate passes for both routes and all admitted KV lengths.

## Benchmark closure audit

`extra/qk/benchmark_shared_attention.py` constructs Q/K/V tensors before `_time`, and the pre-timing candidate/baseline `.numpy()` calls force residency. Therefore:

- Q/K/V host-to-device transfers are excluded from timing.
- Candidate/baseline numeric device-to-host copies are excluded from timing.
- Compiler capture records `copy_call_count=0` for the fused compute.
- Compilation is warmed and cached before samples, but the timed callback still calls `_candidate(...)` or `_baseline(...)` anew.
- `_mask(..., buffer=False)`, graph construction, scheduling, fresh result realization, and any associated output allocation remain inside the timed closure.

The published times are valid end-to-end eager attention-call measurements after warmup. They must not be described as allocation-free kernel times. For a kernel-only comparison, construct and realize the graph once, capture it with `TinyJit`, preallocate/reuse the output, and time synchronized replay.

## Numeric coverage

The precheck is not sampled. Candidate and ordinary GPU SDPA baseline are both converted to complete fp32 NumPy arrays, and `np.allclose(..., rtol=0.03, atol=0.006)` compares every output element. Consequently, the conditional sampled-comparison fallback was not needed.

| Route | KV | Output elements compared | Recorded max absolute error |
|---|---:|---:|---:|
| 8B | 512 | 2,097,152 | 0.00006103515625 |
| 14B | 512 | 2,621,440 | 0.00006103515625 |

The artifacts do not record a per-block nonzero-error count. That metric is not required to repair sampling because no sampling occurs; adding it would require rerunning and extending an already exhaustive comparison.

## Measured rates

Existing synchronized 10-sample eager measurements at KV=512:

| Route | Candidate median | Candidate rate | Baseline median | Speedup | Effective candidate attention rate |
|---|---:|---:|---:|---:|---:|
| 8B | 29.262479 ms | 17,496.81 tokens/s | 33.091543 ms | 1.1309x | 0.1468 TFLOP/s |
| 14B | 30.077712 ms | 17,022.57 tokens/s | 33.366381 ms | 1.1093x | 0.1785 TFLOP/s |

Effective FLOP/s uses the artifact convention `4 * Hq * Q * KV * Hd`. Because the closure includes graph/scheduling/allocation work, these are end-to-end effective rates, not isolated kernel throughput.

A same-grid trivial fp16 write was separately measured with a preallocated output and `TinyJit` replay. Three warmups preceded ten synchronized samples; compilation, copies, and allocation were outside timing.

| Route grid | Raw samples (ms) | Median | Dispatches/s |
|---|---|---:|---:|
| 8B: 1024 workgroups x 32 lanes | 0.169739, 0.170471, 0.173466, 0.164570, 0.168186, 0.159239, 0.157537, 0.156083, 0.167846, 0.167686 | 0.167766 ms | 5,960.68 |
| 14B: 1280 workgroups x 32 lanes | 0.175100, 0.161614, 0.159400, 0.156534, 0.159520, 0.157706, 0.163899, 0.155222, 0.155643, 0.153098 | 0.158553 ms | 6,307.04 |

The eager candidate is roughly 174x (8B) and 190x (14B) the same-grid trivial-dispatch median. Dispatch overhead is therefore not the principal explanation for the current candidate time, although the two measurements have intentionally different closure semantics.

## Production HIP synchronization and LDS traffic

All captured variants have local size 32 and `wavefront_size=32`: each workgroup is exactly one wave. The HIP loop has one static workgroup barrier. Its only shared allocation is `half buf0[256]` (512 bytes); per KV tile, each lane performs eight half stores to it, then sixteen half loads from it. `buf1`, `buf2`, and `buf3` are thread-local state arrays and are excluded from LDS counts.

With a 16-key KV tile, per-workgroup dynamic counts are:

| KV | Tiles | Barriers | LDS half stores | LDS half loads |
|---:|---:|---:|---:|---:|
| 512 | 32 | 32 | 8,192 | 16,384 |
| 1024 | 64 | 64 | 16,384 | 32,768 |
| 2048 | 128 | 128 | 32,768 | 65,536 |
| 4096 | 256 | 256 | 65,536 | 131,072 |

Global dynamic counts multiply by 1,024 workgroups for 8B and 1,280 for 14B:

| Route | KV | Barriers | LDS half stores | LDS half loads |
|---|---:|---:|---:|---:|
| 8B | 512 | 32,768 | 8,388,608 | 16,777,216 |
| 8B | 1024 | 65,536 | 16,777,216 | 33,554,432 |
| 8B | 2048 | 131,072 | 33,554,432 | 67,108,864 |
| 8B | 4096 | 262,144 | 67,108,864 | 134,217,728 |
| 14B | 512 | 40,960 | 10,485,760 | 20,971,520 |
| 14B | 1024 | 81,920 | 20,971,520 | 41,943,040 |
| 14B | 2048 | 163,840 | 41,943,040 | 83,886,080 |
| 14B | 4096 | 327,680 | 83,886,080 | 167,772,160 |

Counts are source-level dynamic operations: LDS store/load counts include all 32 lanes, and barrier counts are wave/workgroup barrier encounters.

## Recommendation and acceptance gate

1. First repair measurement authority without changing the kernel: add preallocated `TinyJit` replay timing beside the existing eager metric. Keep the eager number, but label the two closure types distinctly.
2. Then make only the one-wave synchronization change. Do not combine it with tile geometry, accumulator, or layout changes.
3. Preserve explicit LDS completion ordering. A one-wave workgroup makes inter-wave rendezvous redundant, but it does not make asynchronous LDS completion ordering optional.
4. Accept only if exhaustive candidate-vs-GPU-baseline comparison passes for 8B and 14B at KV 512/1024/2048/4096, generated HIP/ISA still has zero spills and both QK/PV WMMA roles, and synchronized `TinyJit` replay improves beyond run-to-run noise.

This experiment is small, attributable, shared by both model routes, and does not duplicate route-specific assets.
