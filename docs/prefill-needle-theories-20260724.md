# 8B prefill: where the needle actually is (measured 2026-07-24)

Derived entirely from hardware counters (`extra/qk/prefill_boltbeam_trace.py --hw-trace`, restored in
`654c9b2ce`) on 8B/gfx1100 under `TINYGRAD_PREFILL_PACKED_WMMA=0`, plus the whole-model authority
(`extra/qk/prefill_whole_synced.py --mode authority`). Numbers here are measurements, not estimates.

## The frame

| pp | tok/s | % of fp16 WMMA peak (122 TF) | % of HBM peak (960 GB/s) |
|------|-------|------|------|
| 512  | 3607  | 47%  | 3.7% |
| 1024 | 3512  | 46%  | 3.6% |
| 2048 | 3327  | 44%  | 3.4% |
| 4096 | 3003  | 39%  | 3.1% |

**HBM is irrelevant at ~3% of peak.** Every bandwidth/roofline framing is a dead end for prefill,
including BoltBeam's own `roofline --peak 960.0`. The workload is *issue-bound* at roughly half of
compute peak, so every remaining lever is instruction issue and LDS — never memory.

Attention specifically sits at **AI = 818 FLOP/byte, 6x past the ridge point (127)**, using 0.98% of
HBM, with L2 hit *rising* 62.8% -> 85.5% as context grows. A memory roofline cannot diagnose it.

## The split

Per-kernel, ctx512 (PMC-instrumented run; shares are what matter, not its absolute wall time):

| kernel class | share of kernel time | valu | mem | occ | l2 hit | lds conflict |
|---|---|---|---|---|---|---|
| `E_4_*` WMMA GEMMs (top 4) | **79.6%** | 1.6–1.9 | 9.9–11.6 | 37–42 | 33–54 | **12.5** |
| `amd_gfx1100_q16_grid_hd128_loop_attention` | 6.1% (pp512) -> ~23% (pp4096) | 5.3 | 23.7 | 36.4 | 63.1 | 5.1 |

(tinygrad names a WMMA matmul `E_*` because the tensor core consumes the reduce, so these ARE the GEMMs.
Normalized counter fields are IPC-like rates x100, NOT duty cycles — never read `valu_busy_pct` as a
utilization; see the raw passthrough.)

GEMMs are the **large** part and carry the 47%. Attention is the **inefficient** part at ~6% of peak —
8x worse than the model average — and it is the only term that grows with context, i.e. exactly llama's
advantage (llama's attention plateaus at 15.1% of budget at pp4096 vs our 23%).

## Prize sizing at pp4096 (attention = 23% of budget, ~6% of peak)

| if attention reaches | whole-model | vs llama 3160 |
|---|---|---|
| 2x (12% of peak) | +11.5% -> **3348** | ahead |
| 4x (25% of peak) | +17.5% -> **3528** | clearly ahead |
| GEMM parity (47%) | +20.1% -> 3606 | ceiling |

## THEORY 1 — work per wave (the real fix for attention's 6%)

**Claim:** attention's loop body is **843 instructions per KV tile for 16 WMMAs (1.9% useful)**, and that
ratio does not improve by making any single bucket cheaper. It improves only by putting more WMMA work
behind the same overhead.

**Evidence it must be structural, not per-bucket:** V-fragment vectorization removed the single largest
bucket — 112 of 144 load instructions, 89% of VMEM instructions, 58% of SQ busy cycles — and bought
only **1.20x** on the kernel and **-3.4% whole-model**. See the ledger entry; two quantified errors there
(cycles/VMEM-instr is NOT constant: `d16` 3.02 vs `b128` 6.61; and a strided-scatter transpose costs ~7%
of whole-model, not the ~3% a bandwidth estimate suggests).

**The lever:** a wave owning 32 or 64 q-rows instead of 16 amortizes K/V fragment loads across them.
Those loads remain the largest bucket even after vectorization (VMEM 36.7% of SQ busy).

**The blocker and its existing unlock:** 2x q-rows means 2x accumulators on a kernel already at 254 VGPR.
But slicing the *output* dim frees accumulator VGPRs to spend on q-rows: 8 blocks x 8 fp32 = 64 VGPR ->
16 at `acc_blocks=2`. The plumbing already exists and has never been perf-tested —
`amd_gfx1100_q16_grid_pv_slice_stage` with `acc_blocks in {1,2,4}` and `output_block_base in {0,2,4,6}`,
plus `amd_gfx1100_q16_grid_qk_stats_stage`. `docs/SHARED_ATTENTION_SEQUENCE_AND_PATTERN_20260723.md`
item 3 proposed exactly this ("accumulate a smaller value/output slice… initially permit QK/softmax
recomputation") and it was never run to conclusion.

**Warning:** slicing ALONE is a loss — `acc_blocks=2` costs 4x KV passes and 4x softmax. The experiment
is *slice + more q per wave*, measured together. Do not report the slice arm alone as a verdict.

## THEORY 2 — GEMM LDS bank conflicts (untouched, on 79.6% of runtime)

**Claim:** all three top GEMM kernels report `lds_conflict` at **exactly 12.5% (= 1/8)**, identical across
three different shapes, against attention's 5.1%. That exactness reads as a systematic stride-8 conflict
in the graph-GEMM's LDS swizzle. Nobody has looked at this; these counters are the first time it has been
visible.

**Verify FIRST:** confirm 12.5% is not a quantization artifact of `lds_conflict = SQC_LDS_BANK_CONFLICT /
SQC_LDS_IDX_ACTIVE`. Attention's 5.1% shows the metric does vary, but three identical values want one
confirmation before anyone spends on a fix.

**Then test:** pad or swizzle the LDS tile in the graph-GEMM candidate (`PREFILL_GRAPH_GEMM` route,
`prefill_wmma_lds_dbuf_generated`, 2 LDS slots / 256 threads) and re-measure the conflict percentage AND
whole-model throughput. A conflict-rate drop with no throughput change means LDS stalls were not on the
critical path — record that as the verdict rather than chasing it further.

## THEORY 3 — causal tile skipping (bounded, independent)

The kernel iterates **every** KV tile and masks in the softmax; it never skips a fully-masked tile.
Measured waste: **1.94x at chunk 0**, 1.14x at chunk 3, 1.06x at chunk 7, **1.121x (12.1%) aggregate**
over a 4096 prefill. Worth ~+2.8% of pp4096 budget. Independent of Theories 1 and 2, so it composes.

## Bounded and known
Free-transpose V via a **transposed V KV cache** (write once at generation time so prefill reads the
layout it wants for free): ceiling ~3111 tok/s at pp4096 vs llama 3160. Not shippable as the per-call
permute that was measured and refuted today.

## What to drop
Further micro-optimization of individual instruction buckets in the attention loop. Today measured the
cost of that approach directly. Also settled dead ends (do not re-run): VGPR/occupancy compaction
(1.43%/2.46% SLOWER), phase-ABI LDS state (added 2048 B LDS, reduced zero VGPRs), barrier removal
(<0.2%), G2 GQA K/V LDS sharing (1.65% slower), and the P-repack LDS hypothesis (10 LDS instrs/tile;
attention lds_conflict 5.1% vs the GEMMs' 12.5%).
