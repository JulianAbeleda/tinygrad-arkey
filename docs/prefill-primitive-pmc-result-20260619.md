# RESULT — The 4 prefill primitives, MEASURED via PMC (gfx1100, 2026-06-19)

Executes `prefill-primitive-pmc-scope-20260619.md`. Driver `extra/qk_prefill_primitive_pmc.py` (isolated gateup GEMM
out=12288/in=4096/T=512, ~51.5 GFLOP, identical work both sides), native tinygrad PMC (`PMC=1 PROFILE=1`, counters via
`PMC_COUNTERS`). Artifacts `bench/qk-prefill-boost/primitive_pmc_pass_*.json`.

## Verdict
**The tinygrad-WMMA vs Tensile prefill gap is MEMORY DATAFLOW, not the matrix op and not VALU/codegen overhead.**
LDS operand staging (primitive 1) and the stalls it removes (primitive 3) dominate; occupancy (2) contributes;
**per-op VALU overhead (4) is essentially equal and NOT the cause.**

## Measured (same gateup GEMM both sides)
| primitive | counter | tinygrad WMMA | Tensile | finding |
|---|---|---:|---:|---|
| **1. LDS staging** | `SQ_INSTS_LDS` | **0** | 31,457,280 | WMMA uses ZERO LDS (matches disasm LDS=0B vs 24.5KB) |
| | `SQC_LDS_IDX_ACTIVE` | 0 | 44,040,192 | Tensile heavily LDS-active; WMMA none |
| | `GL2C_MC_RDREQ` (DRAM reads) | **29,846,900** | 4,493,649 | **WMMA reads DRAM 6.6× more** — re-streams operands (no reuse) |
| | `GL2C_HIT%` | 48% | 59% | WMMA misses L2 more |
| **2. occupancy** | `SQ_WAVES` (total) | 3072 | 1536 | WMMA launches 2× more waves (smaller tiles) |
| | avg wave lifetime (`SQ_WAVE_CYCLES/SQ_WAVES`) | 5.65M | 1.66M | WMMA waves live 3.4× longer (serial work/stalls). Static: 1 wave/wg vs 32 |
| **3. stalls / pipeline** | `SQ_WAIT_ANY` | **14.21e9** | 2.18e9 | **WMMA stalls 6.5× more cycles** |
| | `SQ_INST_CYCLES_VMEM` | 19.5M | 41.3M | Tensile spends MORE in vmem but OVERLAPPED (prefetch); WMMA fewer loads yet stalls more = serial, un-pipelined |
| | `SQ_WAIT_ANY/SQ_BUSY_CYCLES` | 25.2 | 15.9 | WMMA's stall:issue ratio is higher |
| **4. VALU overhead** | `SQ_INSTS_VALU` | 8,011,776 | 7,039,488 | **~equal — NOT the differentiator** |
| net | `GRBM_GUI_ACTIVE` (active cycles) | ~12.5M | ~3.6M | ~3.5× isolated-kernel gap (e2e is 1.84×, Amdahl over the block) |

## Interpretation (ties to the disasm)
- **Primitive 1 is the root.** Tensile stages 128×128 A/B tiles in 24.5 KB LDS and reuses them → **6.6× less DRAM
  read traffic** for the same FLOPs. tinygrad's WMMA kernel has LDS=0, so it re-reads operands from DRAM/L2.
- **Primitive 3 is the consequence.** Those extra DRAM reads aren't hidden: WMMA shows **6.5× more `SQ_WAIT_ANY`
  stall cycles**. Tensile actually issues MORE vmem cycles (41.3M vs 19.5M) yet stalls far less — its double-buffered
  K-loop prefetches the next tile while WMMA-computing the current one (compute/load overlap). tinygrad emits a serial
  load→wait→compute loop (the linearizer can't hoist a load across the loop RANGE — `[[qk-runtime-overhead-arc]]`).
- **Primitive 2 amplifies it.** 1 wave/workgroup (32 threads) gives nothing to hide latency behind; Tensile's 32
  waves/wg let other waves issue during stalls. (PMC `SQ_WAVES` confirms the smaller-tile/more-wave structure; the
  per-wave concurrency itself is the static local-size fact.)
- **Primitive 4 is a non-factor.** VALU instruction counts are within ~14% (8.0M vs 7.0M) — the "WMMA is a minority
  of the instruction stream" static observation does NOT translate into a runtime VALU-overhead penalty; both are
  memory-bound, not VALU-bound.

## Tooling caveats — both SETTLED
- **Caveat 1 (can PMC capture the vendored Tensile kernel?) → YES.** The `tensile_gateup` (TensileRunner, patched
  `dev.runtime`) returns non-zero, sane PMC counters; native PMC brackets the HCQ dispatch correctly. So true
  side-by-side WMMA-vs-Tensile PMC A/B works.
- **Caveat 2 (WMMA-specific busy counter?) → NO.** tinygrad's wired PMC set (40 counters) has no `SQ_INSTS_MFMA` /
  `SQ_VALU_MFMA_BUSY_CYCLES` / `*_MFMA_F16` (MFMA is CDNA; RDNA3 WMMA isn't counted). WMMA throughput stays
  TFLOPS-only (time×FLOPs). Available set printed by the loader on an unknown name; LDS/occupancy/stall/VALU all
  covered.

## What this means for the lever
The measured root cause (LDS operand staging + software-pipelined K-loop) is exactly the codegen capability tinygrad
can't express (POWN / Route-A / CG walls). So **closing the gap dependency-free = an LDS-staged, double-buffered,
multi-wave WMMA codegen path** (multi-week, the BEAM-hang/linearizer-RANGE class). The shippable alternative remains
the vendored Tensile `.co` (~87% llama, byte-identical). PMC now gives a **direct, per-primitive scoreboard** to
measure any future WMMA-codegen attempt against (target: drive `GL2C_MC_RDREQ` and `SQ_WAIT_ANY` down toward
Tensile's).

## Hazards respected
PMC perturbs per-kernel TIMING (we report counter ratios, not PMC-run wall time); ≤~8 counters/pass (2 passes used,
`TA` block unsupported); counters summed across SQ/GL2C instances (ratios within a side are valid; cross-side uses
same normalization). Isolated-kernel gap (~3.5×) is larger than e2e (1.84×) as expected (gateup is one op in the block).
