# llama.cpp kernel residual primitive audit

Audit of what llama.cpp itself may still leave on the table on RX 7900 XTX / Qwen3-8B-Q4_K_M. This is **not** the
tinygrad-vs-llama gap audit. That gap is already mapped. This asks whether llama's own kernels are near their
practical primitive ceilings.

No tinygrad kernels were built or routed. The redo used `rocprofv3` from `/opt/rocm-7.2.4/bin/rocprofv3`; it was
installed but not on `PATH`. Fresh trace artifacts:

- `bench/llama-kernel-residual-primitive-audit-20260619/topline.json`
- `bench/llama-kernel-residual-primitive-audit-20260619/fresh_rocprof_summary.json`
- `bench/llama-kernel-residual-primitive-audit-20260619/rocprof_decode_d0/trace_kernel_stats.csv`
- `bench/llama-kernel-residual-primitive-audit-20260619/rocprof_decode_d1024/trace_kernel_stats.csv`
- `bench/llama-kernel-residual-primitive-audit-20260619/rocprof_prefill_pp512/trace_kernel_stats.csv`

`rocprofv3` adds overhead to the benchmark loop, so use top-line `llama-bench` for tok/s and the profiler runs for
kernel-share anatomy.

## Verdict

llama.cpp is **not theoretically optimal**, but the fresh traces make the residual map sharper:

- **Prompt-free decode is even more MMVQ-dominated than the old banked trace:** Q4_K MMVQ is 66.9% and Q6_K MMVQ is
  18.7%, for **85.6% total MMVQ [M]** in the d0 trace. If that whole bucket moved from a 70%-HBM-class effective
  ceiling to 80%/90%/100%, Amdahl gives about **+12% / +24% / +35% e2e [H]**. This is the only large decode
  residual, but it may be near the practical ceiling once q4/q6 unpack, q8 loads, qsum/min correction, reductions,
  and occupancy are counted.
- **q8 activation quant is small alone:** 3.57% [M], so deleting it entirely is only **+3.7% [H]**. It matters only
  if folded into a broader producer lifecycle.
- **RMSNorm + q8 is the main non-MMVQ lifecycle candidate:** RMSNorm is 4.60% [M] and q8 is 3.57% [M]. Full deletion
  of both would be **+8.9% [H]**, but the reductions differ and there is no evidence llama currently fuses q8
  production into RMSNorm.
- **Decode attention is not a large normal-decode residual:** d0 tile+combine is 3.27% [M]. The mixed d1024 trace
  also shows tile+combine+fixup at only 2.65% of that run, though d1024 contains prompt/depth setup work. Long-context
  attention can still deserve its own trace, but it is not the main llama-side headroom.
- **Graph boundary is already actively optimized:** graph-on d1024 is 97.39 tok/s [M]; disabling HIP/CUDA graphs is
  61.48 tok/s [M], and disabling llama graph reuse is 61.04 tok/s [M]. Remaining boundary work must remove GPU work
  or data movement, not just launches.
- **pp512 prefill is now mapped:** 74.4% quantized MMQ/matmul, 10.0% rocBLAS GEMM, 4.4% attention [M]. For pp512,
  llama prefill residual is matmul/MMQ-first, not attention-first. Long-prompt prefill remains a separate audit.

## Provenance lock

| item | value |
|---|---|
| llama.cpp path | `/home/ubuntu/env/llama.cpp` |
| commit/build | `ac4cddeb0` / build `9592` |
| model | `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf` |
| GPU | AMD Radeon RX 7900 XTX, gfx1100, 24 GB |
| ROCm | 7.2.4 |
| profiler | `/opt/rocm-7.2.4/bin/rocprofv3` |
| build flags | `GGML_HIP=ON`, `GGML_HIP_GRAPHS=ON`, `GGML_CUDA_FA=ON`, `GGML_HIP_MMQ_MFMA=ON`, `GGML_HIP_ROCWMMA_FATTN=OFF`, `GGML_CUDA_FA_ALL_QUANTS=OFF` |
| fresh top-line artifact | `bench/llama-kernel-residual-primitive-audit-20260619/topline.json` |
| fresh trace summary | `bench/llama-kernel-residual-primitive-audit-20260619/fresh_rocprof_summary.json` |

Fresh top-line `llama-bench` validation:

| test | tok/s [M] |
|---|---:|
| `tg128 @ d512` | 97.71 |
| `tg128 @ d1024` | 97.39 |
| `tg128 @ d4096` | 92.37 |
| `pp512` | 2900.71 |
| `pp1024` | 3020.04 |

Fresh profiled runs:

| run | profiled tok/s [M] | note |
|---|---:|---|
| `tg64 @ d0` | 85.96 | best prompt-free decode-share trace; profiler overhead expected |
| `tg32 @ d1024` | 82.88 | mixed with prompt/depth setup; use as secondary evidence |
| `pp512` | 2878.16 | close to top-line pp512; good prefill-share trace |

## Fresh trace ledgers

Prompt-free decode, `rocprof_decode_d0`, total kernel time 557.39 ms:

| family | share [M] | time [M] | calls [M] |
|---|---:|---:|---:|
| Q4_K MMVQ | 66.90% | 372.87 ms | 11700 |
| Q6_K MMVQ | 18.73% | 104.38 ms | 2405 |
| RMSNorm | 4.60% | 25.62 ms | 9425 |
| q8_1 activation quant for MMVQ | 3.57% | 19.91 ms | 14105 |
| flash-decode tile | 2.34% | 13.05 ms | 2340 |
| RoPE | 1.74% | 9.70 ms | 4680 |
| flash-decode combine | 0.93% | 5.20 ms | 2340 |
| elementwise/misc | 0.80% | 4.48 ms | 2405 |
| copy | 0.34% | 1.89 ms | 535 |
| other | 0.05% | 0.30 ms | 131 |

pp512 prefill, `rocprof_prefill_pp512`, total kernel time 325.03 ms:

| family | share [M] | time [M] | calls [M] |
|---|---:|---:|---:|
| quantized MMQ/matmul | 74.42% | 241.88 ms | 428 |
| rocBLAS GEMM | 9.98% | 32.45 ms | 70 |
| prefill attention ext f16 | 4.40% | 14.29 ms | 72 |
| RMSNorm | 2.07% | 6.72 ms | 290 |
| RoPE | 1.81% | 5.89 ms | 144 |
| q8 quant for MMQ | 1.40% | 4.56 ms | 428 |
| dequantize | 1.31% | 4.27 ms | 70 |
| SwiGLU | 1.27% | 4.14 ms | 70 |
| convert | 1.20% | 3.89 ms | 140 |
| elementwise/misc | 1.01% | 3.29 ms | 214 |
| attention decode fixup | 0.47% | 1.52 ms | 72 |

The old banked decode trace had MMVQ at 73.4%. The fresh d0 trace has MMVQ at 85.6%. Treat this as a window
difference, not a contradiction: d0 isolates prompt-free decode and has little long-context attention/setup work;
d1024 profiler windows can include depth/prompt setup kernels. The conclusion strengthens: normal llama decode
residual is MMVQ-dominated.

## Residual table

| primitive | share [M/I] | achieved efficiency [M/I/H] | residual to ceiling | likely limiter | unexplored idea | Amdahl max | verdict |
|---|---:|---:|---:|---|---|---:|---|
| MMVQ Q4_K/Q6_K aggregate | 85.6% [M d0] | ~70%-HBM-class from prior bandwidth accounting [M/I] | +12% e2e if 80% peak; +24% if 90%; +35% if raw 100% [H] | required q4/q6 unpack, q8 load/scales, qsum/min affine, in-kernel reduction, occupancy/register scheduling | per-role MMVQ counter audit; role-specialized kernels only if a bad role exists | high theoretical, unknown practical | **open measurement**, likely near practical ceiling |
| q8_1 activation quant | 3.57% [M d0] | separate quant kernel with per-32 max+sum [M source/trace] | full removal +3.7% [H] | reduction + pack/write + kernel boundary | producer-fused q8 | <=3.7% alone | **low-EV alone; open only with RMSNorm lifecycle** |
| RMSNorm | 4.60% [M d0] | separate/fused norm kernels exist; q8 not folded [I/source] | full removal +4.8% [H] | row reduction + memory pass | RMSNorm emits q8 side-channel and fp output | <=4.8% alone; <=8.9% with q8 | **lifecycle candidate, not standalone** |
| RoPE | 1.74% [M d0] | separate/fused rope code exists [source] | full removal +1.8% [H] | elementwise memory pass | fuse with Q/K prep or KV write | <=1.8% | **low-EV unless composed** |
| decode attention | 3.27% [M d0 tile+combine] | tile + combine; mixed d1024 also has fixup [M/source] | full removal +3.4%; half cost +1.7% [H] | KV traffic, stream-K fixup/combine, online softmax state | fixup/combine fusion or more role-specific flash-decode | <=3.4% normal decode | **mostly efficient; long-ctx audit only** |
| elementwise/residual/SwiGLU | 0.8-1.0% [M] | separate/fused small ops [M/I] | full removal about +1% [H] | memory/launch | none standalone | <=1% | **closed low-EV** |
| graph/kernel boundaries | amortized [M] | graphs-on 97.39 tok/s; graphs-off 61.48 tok/s [M] | graphing already buys ~1.58x vs off [M] | launch/boundary tax when graphs disabled; with graphs on, residual unknown | persistent/deeper block only if it removes GPU work | unknown, likely low after graphs | **launch issue solved; GPU-work fusion only** |
| pp512 prefill MMQ/matmul | 74.42% [M pp512] | quantized MMQ path, VGPR-heavy, LDS=0 in trace [M] | halving this bucket would be +59% pp512 [H] | quantized matmul tiling, register pressure, memory layout, library-vs-custom boundary | MMQ/Tensile-style audit, shape-specific external boundary | high theoretical, unknown practical | **mapped; matmul-first prefill residual** |
| pp512 prefill attention | 4.40% [M pp512] | `flash_attn_ext_f16` [M] | full removal +4.6% [H] | attention tiles/fixup | long-prompt-only flash audit | low at pp512 | **not pp512 bottleneck; long-prompt separate** |

## Track results

### LRA-0 - source/version/provenance lock

Passed. Local build is `ac4cddeb0` / build `9592`. `llama-bench` at `tg128` reproduced the known 92-98 tok/s
decode band. Profiler was present at `/opt/rocm-7.2.4/bin/rocprofv3` and was used for fresh traces.

### LRA-1 - per-primitive time ledger refresh

Passed for prompt-free decode and pp512 prefill. The d0 decode ledger has only 0.05% `other`, so the old
`other/unclassified >5%` concern is closed for this window. d1024 trace is retained as secondary evidence because
its window includes prompt/depth setup kernels.

Decision: **fresh d0 trace is the decode-share authority for this audit; pp512 trace is the prefill-share authority
for short prefill.**

### LRA-2 - MMVQ residual-to-peak

Still the largest open measurement. Fresh trace metadata shows the hot MMVQ kernels are not spilling:

| kernel family | share [M d0] | avg time/call [M] | VGPR/LDS/scratch [M] | workgroup [M] |
|---|---:|---:|---|---|
| Q4_K MMVQ | 66.90% | 31.87 us | ~37 VGPR, 0 LDS, 0 scratch | 32x1 |
| Q6_K MMVQ | 18.73% | 43.40 us | ~37 VGPR, 512 B LDS, 0 scratch | 32x2 |

That points away from a simple spill/codegen pathology. The remaining 70%-class to practical-ceiling gap could be
transaction shape, unpack/affine cost, reduction overhead, or generic role coverage. Without per-role hardware
counters, do not call role-specialized MMVQ a live optimization.

Decision: **largest theoretical residual, but not proven exploitable**.

### LRA-3 - q8 activation quant residual

Low-EV alone. Fresh d0 trace measures `quantize_q8_1` at 3.57%, 14,105 calls, about 1.41 us/call. Source shape is
a separate kernel that computes per-32 max and sum, writes int8 quants, and stores `half2(d, sum)`.

The only interesting version is not "optimize q8 quant"; it is **producer-fused q8 lifecycle** with RMSNorm or
another producer. That aligns with the tinygrad q8/MMVQ lifecycle result, but llama-side economics remain moderate:
q8+RMSNorm full deletion is still only +8.9% [H].

Decision: **closed as standalone; open only as part of norm/q8 lifecycle**.

### LRA-4 - decode attention residual

Mostly closed for normal decode. Fresh d0 trace measures flash-decode tile+combine at 3.27%. The mixed d1024 trace
shows tile+combine+fixup at 2.65% of that window. The prior attention audit showed context-flat behavior: about
99.5 tok/s at d0 to about 92.2 at d4096, a roughly 7% drop.

Potential remaining work:

- split tile vs combine vs stream-K fixup in a clean long-context decode-only trace;
- test whether fixup/combine are meaningful enough to fuse;
- check much longer contexts where attention share may grow.

Decision: **audit-only unless a clean long-context trace shows >=5% e2e upside**.

### LRA-5 - norm/RoPE/elementwise fusion residual

Standalone small-op fusion is low-EV:

- q8 activation quant full deletion: +3.7% [H];
- RMSNorm full deletion: +4.8% [H];
- q8+RMSNorm full deletion: +8.9% [H];
- RoPE full deletion: +1.8% [H];
- elementwise/misc full deletion: about +1% [H].

RMSNorm is the only one worth keeping open, and only because it composes with q8 activation quant. The hard part is
that RMSNorm needs row sum-of-squares while q8 needs per-32 max/sum and MMVQ wants a specific q8_1 layout.

Decision: **RMSNorm/q8 lifecycle is the only small-op candidate; RoPE/elementwise low-EV**.

### LRA-6 - graph/kernel-boundary audit

Measured. At d1024:

| mode | tok/s [M] |
|---|---:|
| graphs on | 97.39 |
| `GGML_CUDA_DISABLE_GRAPHS=1` | 61.48 |
| `LLAMA_GRAPH_REUSE_DISABLE=1` | 61.04 |

This shows llama's graph path is not cosmetic; it is a core part of the performance primitive. It also means
"reduce launch overhead" is not a remaining easy llama-side frontier: llama already did it. A future persistent or
deeper-fusion idea must show it removes GPU work/data movement or improves reuse.

Decision: **launch boundary solved; GPU-work fusion only**.

### LRA-7 - llama prefill residual

Mapped for pp512. Fresh top-line:

| test | tok/s [M] |
|---|---:|
| pp512 | 2900.71 |
| pp1024 | 3020.04 |

Fresh pp512 trace says the short prefill residual is matmul-first:

| family | share [M pp512] | implication |
|---|---:|---|
| quantized MMQ/matmul | 74.42% | dominant residual bucket; a 2x bucket win would be +59% pp512 [H] |
| rocBLAS GEMM | 9.98% | already external/library-backed, still material |
| prefill attention ext f16 | 4.40% | not pp512 bottleneck |
| all q8/dequant/convert/swiglu/norm/rope small ops | about 10.5% combined | not a single large primitive |

This does **not** close long-prompt prefill. Attention can grow with prompt length, and the prior tinygrad long-prompt
attention work remains a separate phase-dependent arc.

Decision: **pp512 prefill mapped as MMQ/matmul-first; long-prompt prefill audit remains separate**.

## What llama did not appear to explore fully

These are plausible omissions, not proven wins:

| idea | status after redo | why |
|---|---|---|
| MMVQ residual-to-practical-peak | open measurement | 85.6% Amdahl is large, but no evidence yet that 70%-class effective bandwidth is below the real ceiling for this primitive |
| producer-fused q8 activation | open only with RMSNorm lifecycle | q8 alone is 3.57%; combined norm+q8 is 8.17% |
| role-specialized MMVQ kernels | open measurement | aggregate trace could hide weak roles, but fresh trace only splits Q4_K/Q6_K families |
| deeper norm/q8/MMVQ fusion | open research | only non-MMVQ decode idea with enough combined Amdahl to care |
| attention fixup/combine fusion | audit-only | attention is 3.27% in d0; needs a clean long-context trace to matter |
| persistent/deeper decode block | low confidence | graph-off drop proves graphs matter, but graphs-on already solves launch tax |
| pp512 prefill MMQ/matmul specialization | open measurement | pp512 is 74.4% quantized matmul; practical ceiling unknown |
| long-prompt prefill attention tuning | deferred | not answered by pp512; attention can become phase-dominant at longer prompts |
| shape-specific rocBLAS/Tensile/autotune | strategic open | prefill has material library-backed buckets and a large custom MMQ bucket |

## Final decision

llama.cpp's kernels are not "fully efficient" in the theoretical sense, but the remaining visible headroom now
collapses to three serious audit frontiers:

1. **Decode MMVQ residual-to-practical-peak:** high Amdahl, but may already be near the real ceiling once required
   unpack, q8, affine, and reductions are counted. Needs per-role hardware counters.
2. **RMSNorm/q8 producer lifecycle:** moderate Amdahl only if the two shares compose; q8 alone is too small.
3. **Prefill MMQ/matmul practical ceiling:** pp512 is matmul-first; long-prompt attention remains separate.

Everything else is low-EV for normal decode unless a fresh long-context trace shows a concentrated component.
