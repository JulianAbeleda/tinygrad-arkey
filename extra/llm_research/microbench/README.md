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

---

# mma_peak_cuda.cu — measured achievable mma.sync peak (NVIDIA sm_120 / RTX 5090)

The CUDA analogue of `wmma_peak.cpp`: isolated back-to-back
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` — the exact instruction the fork's
`CUDARenderer` emits for fp16→fp32 (`tinygrad/renderer/cuda.py:75`, shape from
`tc.get_cuda(arch)` → `cuda_sm89`, `tinygrad/codegen/opt/tc.py:135`) — on register-resident
fragments: NACC independent accumulators, runtime trip count so the loop is not folded, a
never-taken store so the result stays live, zero loads in the hot loop.

    export PATH=/usr/local/cuda-13.2/bin:$PATH
    nvcc -O3 -arch=sm_120 -DNACC=8 mma_peak_cuda.cu -o mma_peak_cuda && ./mma_peak_cuda 32768

Verify purity before believing a number (`cuobjdump --dump-sass`; `nvdisasm` ships inside
the triton package if the CUDA toolkit lacks it): the hot loop contains only `HMMA.16816.F32`
(NACC per unrolled body — 232 total at `nacc=8`, iters unrolled 29×), zero `LDG`/`LDS`/`STS`,
and exactly one gated `STG` (the never-taken sentinel). `--ptxas-options=-v`: 0 spills at
every NACC (42 regs at nacc=8, 138 at nacc=32).

## Result, NVIDIA GeForce RTX 5090 (GB202, sm_120), 2026-07-31

Grid sweep (`nacc=8, tpb=256, iters=200000`) climbs and plateaus at `blocks=32768`:

| blocks | TFLOPS |
| ---: | ---: |
| 2048 | 237.0 |
| 4096 | 246.5 |
| 8192 | 251.5 |
| 16384 | 254.0 |
| 32768 | 255.4 |
| 65536 | 255.4 |

**R ≈ 255.4 TFLOPS** — the isolated m16n8k16 f16→f32 rate on this 5090, at steady 2932 MHz
(SM util 100%, 235 W of the 600 W cap: an instruction-issue ceiling, not a power/clock
throttle).

Flatness across NACC 1→32 (236.0→237.1 at `blocks=2048`) — the same number at `nacc=1` as at
`nacc=32` — says this is the tensor pipe's issue rate, not latency-bound work: 1 HMMA.16816
per ~8 clocks per SM. An operand-rotation A/B probe (each HMMA reading a distinct register
set) measures identically (255.4), so the shared `.reuse` operands are not a read-port
serialization artifact. `m16n8k8` (the renderer's alternate descriptor shape) measures
125.8 TF — same issue rate, half the FLOPs per instruction — so `m16n8k16` is the right
shape. The fp16→fp16 accumulate probe was discarded: the compiler collapsed it, and it is
not the path the renderer uses for GEMMs.

External sanity: shiinamiyuki/sm120_gemm's real BF16 GEMMs on the same chip reach 140–217
TF, i.e. 55–85% of this R — consistent with R as the ceiling. A third-party figure (~319 TF,
"76% of spec") is not directly comparable (different methodology/instruction), so 255.4 TF
is the denominator for any efficiency claim in this bring-up, per
`docs/what-makes-a-token-fast-20260731.md` §9: never quote a spec sheet. 255.4 TF is 61% of
the 419 TF sheet figure.

`BW` is still unmeasured on this target, so `M* = (w/16)·(R/BW)` is reported as a function
of `BW`, not a single number — same state Metal was in before its `BW` bench.

---

# wmma_peak_metal.py — measured achievable simdgroup_multiply_accumulate peak (Apple M4)

The Metal analogue of the above: same idea (independent accumulators to cover matrix-op latency,
operands hoisted out of the hot loop, runtime trip count, never-taken keep-alive store), but
Metal's tensor op is `dims=(8,8,8)` (`tinygrad/codegen/opt/tc.py:181`), so one op is
`2*8*8*8 = 1024` FLOP, not AMD's 8192. Uses tinygrad's real `Device["METAL"]` plumbing
(`MetalCompiler.compile` + `MetalProgram` from `tinygrad/runtime/ops_metal.py`) to compile and
dispatch raw hand-authored MSL — not tinygrad's codegen/AST path.

    python3 extra/llm_research/microbench/wmma_peak_metal_run.py

Verify purity before believing a number: `xcrun metal -c x.metal -o x.air && xcrun metal-objdump
--disassemble x.air`, then grep for `addrspace(1)` (device memory) and `addrspace(3)`
(threadgroup memory) — both must appear only outside the hot loop (the one-time `iters` load and
the never-taken sentinel store). AIR is pre-register-allocation and upstream of Apple's private
backend: it shows *which instructions* the frontend emits (so it can prove zero device/threadgroup
traffic in the loop, and that `mat_a`/`mat_b` are constant-folded directly into the intrinsic call
rather than loaded), but it cannot show final register allocation, instruction scheduling, spilling,
or occupancy — those live entirely in Apple's undocumented backend past this IR.

## Result, Apple M4 (10-core GPU, Mac16,10, base M4, not Pro/Max/Ultra), 2026-07-31

Grid-size sweep (`nacc=8, tpb=256, iters=4000`) climbs monotonically and plateaus by
`blocks=1024` (8192 simdgroups) at **~2597 GFLOPS**; extending to `blocks=4096` (32768
simdgroups) adds nothing (2597.3 GFLOPS, spread 6.1 across 5 reps).

NACC sweep at that grid size (`blocks=4096, tpb=256`) — unlike gfx1100, throughput is *highest*
at the *lowest* NACC and falls as NACC grows:

| NACC | iters | mean GFLOPS |
| ---: | ---: | ---: |
| 2 | 16000 | 3772.0 |
| 4 | 8000 | 2379.9 |
| 8 | 4000 | 2596.5 |
| 16 | 2000 | 1741.5 |

Re-running the grid-size sweep at `nacc=2` (the winner) and extending it further shows the true
plateau is higher and needs a bigger grid: **3781.3 GFLOPS at `blocks=32768, tpb=256`
(262144 simdgroups)**, essentially flat from `blocks=8192` on (3777.4 → 3779.9 → 3781.3, spread
<1 GFLOPS across 5 reps at the top). `nacc=1` (true back-to-back dependency, measures latency not
throughput) reaches 3718.4 GFLOPS at the same grid point — only ~1.6% below `nacc=2` — so this
matrix instruction has very little dependency latency to hide on this hardware; occupancy
(simdgroups resident per core), not per-thread independent accumulator chains, is what drives
throughput here. A tpb sweep (32/128/256/512/1024) at matched total simdgroup count confirms the
plateau (3763–3779 GFLOPS) is insensitive to threadgroup shape.

**R ≈ 3781 GFLOPS ≈ 3.78 TFLOPS** — the isolated `simdgroup_multiply_accumulate` rate on this
10-core M4, at `nacc=2, blocks=32768, tpb=256`.

Disassembly (`xcrun metal -c` + `metal-objdump`) of the winning config confirms: the hot loop
lowers to exactly `getelementptr` (accumulator slot) → `load` (accumulator, generic/private
address space — the pre-regalloc stand-in for a register, not memory traffic) →
`call air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32` → `store`
(accumulator, same address space) per unrolled step. `mat_a`/`mat_b` appear as literal constant
`<64 x half>` vectors baked directly into the call — the compiler never loads them at all. The
only `addrspace(2)` (constant-buffer) load is the one-time `iters` read before the loop, and the
only `addrspace(1)` (device-buffer) store is the gated, never-taken sentinel write after it. Zero
`addrspace(3)` (threadgroup) references anywhere. Same shape confirmed at `nacc=8`.

Full per-rep numbers: `/tmp/wmma_peak_metal_result.json`, disassembly: `/tmp/wmma_peak_metal_disassembly.txt`.

No Apple-published TFLOPS spec exists for this instruction or for the base 10-core M4 GPU. The
only external figures found are third-party FP16 ALU benchmarks of the **M4 Max (40-core)**,
~13.3–14.2 TFLOPS — 4x this die's core count, and not a matrix-unit-specific number — so no
"fraction of spec" figure is reported here as authoritative; see the task report for the caveated
comparison.

---

# fma_peak_metal.py — measured achievable plain-FMA peak (Apple M4)

Decides the question `R` above raises but does not answer: is `simdgroup_multiply_accumulate` a
separate, faster matrix unit, or does it lower onto the same ALUs plain scalar/vector FMA uses?
Same harness discipline as `wmma_peak_metal.py` (operands hoisted out of the hot loop, runtime
trip count, never-taken keep-alive, `Device["METAL"].synchronize()` host timing, `xcrun metal -c`
+ `metal-objdump` disassembly verification), with the inner op changed to plain `fma(a, b, acc)` —
no `simdgroup` anywhere. Grep generated source for `simdgroup` to confirm zero hits, never
`simdgroup_matrix`/`simdgroup_multiply_accumulate` (those belong to `wmma_peak_metal.py`).

    python3 extra/llm_research/microbench/fma_peak_metal_run.py

Three precision variants, because `simdgroup_float8x8` accumulates fp16×fp16→fp32, not fp16→fp16:
`f16f16` (half×half→half), `f16f32` (half×half→float — matches the matmul's actual numerics,
primary comparator), `f32f32` (float×float→float). Vector width swept 1/2/4/8 (`half8`/`float8` do
not exist as usable types in this MSL toolchain — `metal_extended_vector.h` forward-declares them
incomplete; width-8 is emulated as two independent `half4`/`float4` lanes per accumulator, still
charged 8 elements × 2 FLOP). NACC swept 1/2/4/8/16. Grid swept to plateau. Each config's `iters`
is *calibrated*, not fixed, via `calibrate_iters()`: measure once, project the iters needed to
clear a target wall time, verify the result sits ≥50× above a probed single-iteration overhead
floor. An earlier version of this file used fixed `iters` values copied from the shape of
`wmma_peak_metal.py`'s sweep and reported up to **1.78 million GFLOPS** with grid-size throughput
that never plateaued — physically impossible, and never reported externally. Root cause: `iters`
was too small relative to `blocks`, so measured wall time was dominated by fixed host-side
dispatch/synchronize round-trip overhead (~0.3–0.7ms) rather than GPU compute time — the enqueue-
vs-execution trap from `docs/what-makes-a-token-fast-20260731.md` §9.6, reached this time through
undersized `iters` rather than a missing `synchronize()`. Fixed by calibration; see
`fma_peak_metal.py`'s module docstring for the full account.

Also guarded (defensively; tested and found not to be the actual failure above) against thread-
uniform scalarization: every thread otherwise computes bit-identical values, which a sufficiently
aggressive compiler could in principle hoist to run once per SIMD-group on a scalar/uniform unit
and broadcast, understating real per-lane throughput. Each kernel reads
`thread_position_in_grid` once (a register, not a memory load) and perturbs operands by a tiny
`tid`-dependent amount. An isolated A/B probe (uniform vs. perturbed operands, same grid sweep)
gave statistically indistinguishable, correctly-plateauing throughput either way, so this was not
what caused the bad run above — but the perturbation costs nothing and is kept as a standing
guard.

## Result, Apple M4 (10-core GPU, Mac16,10), 2026-07-31

Vector-width sweep (`nacc=2`, grid swept per width) climbs with width for all three variants and
plateaus by `blocks=4096`; NACC swept at the winning width; grid-size plateau re-confirmed out to
`blocks=16384` (4.19M threads):

| variant | plateau (GFLOPS) | width, nacc, blocks | vs `R` (3781.3) |
| --- | ---: | --- | ---: |
| `f16f16` (half→half) | 3908.7 | width=8, nacc=16, blocks=16384 | 1.034× |
| `f16f32` (half→float, matches matmul numerics) | 3527.8 | width=8, nacc=8, blocks=16384 | 0.933× |
| `f32f32` (float→float) | 3444.9 | width=8, nacc=8, blocks=16384 | 0.911× |

All three land within +3%/−9% of `R` — the same order of magnitude, not a separate unit worth an
order of magnitude more. Packed-math check: `f16f32`/`f32f32` = 1.024× (no meaningful fp16
throughput doubling); `f16f16`/`f32f32` = 1.135× (a modest edge, far short of 2×) — fp16 buys
bandwidth/storage on this hardware, not FLOP/s.

NACC behaviour splits by variant: `f16f16` improves monotonically to nacc=16 (3702→3898 GFLOPS);
`f16f32` and `f32f32` both peak at nacc=8 and **collapse** at nacc=16 (3517→688 and 3437→606
GFLOPS, >80% drop) — a register-pressure cliff, since float accumulators cost more registers per
lane than half ones and 16 of them exhausts the budget.

Disassembly (`xcrun metal -fno-fast-math -c` — flag-matched to `MetalCompiler.compile()`'s actual
runtime compile flags, unlike a first pass here that used plain `xcrun metal -c`, defaulted to
fast-math ON, and showed spurious `fma fast`/`fadd fast` that was never evidence about what was
actually measured) confirms, for the winning config of each variant: `air.compile.fast_math_
disable` present; the hot loop (between the loop preheader and the back-edge branch) contains only
`@air.fma.v4f16`/`@air.fma.v4f32` calls — manually counted at exactly `nacc × (width/4)` per
variant, zero loads, zero `convert` calls inside it; the `f16f32` half→float widening conversion is
loop-invariant and correctly hoisted to a one-time preheader, never repeated per iteration; zero
`simdgroup` references anywhere; exactly one load (`iters`) and one gated store (the never-taken
sentinel) in the whole kernel.

**Verdict: one shared unit, not two.** On this M4, `simdgroup_multiply_accumulate` does not reach a
separate, faster matrix pipe — it lowers onto (or performs comparably to) the same FP ALUs plain
FMA already uses. `docs/what-makes-a-token-fast-20260731.md` §5's "which unit — worth 10–20×"
principle does not apply here; see that doc's §5/§10 for the consequence for Metal prefill
routing.

Full per-rep numbers: `/tmp/fma_peak_metal_result.json`, per-run stdout:
`/tmp/fma_peak_metal_stdout.txt`, disassembly: `/tmp/fma_peak_metal_disassembly_*.txt`.
