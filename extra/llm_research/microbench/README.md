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
