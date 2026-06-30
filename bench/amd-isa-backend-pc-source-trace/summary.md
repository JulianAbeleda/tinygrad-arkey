# AMD ISA PC/source trace

**Verdict:** AMD_ISA_PC_SOURCE_TRACE_PASS_SOURCE_ROWS_PINNED  
**category measured by PMC, PC/source rows estimated by static loop weighting (NOT hardware per-PC stalls)**  
hardware per-PC: False (ATT/SQTT per-PC decode unavailable under HCQ (N2: instructions_size==0 / ATT_DECODER_REPAIR_BLOCKED); PMC category counters used instead)

native tile: 338 static insts; category breakdown {'VMEM': 7, 'SMEM': 4, 'VALU': 197, 'WAIT': 35, 'SALU': 26, 'BARRIER': 5, 'LDS': 36, 'OTHER': 28}

## Ranked source groups (by estimated dynamic insts @ctx512)

| rank | source_group | cat | static | est_dyn_ctx512 | est_dyn_ctx4096 | pmc_ratio | candidate_lever |
|---|---|---|---|---|---|---|---|
| 1 | address_index | VALU | 114 | 668928 | 4793984 | 27.6 | scalarize uniform prefix (N1B refuted/dead) / strength-reduce |
| 2 | waitcnt | WAIT | 35 | 503040 | 3605120 | None | finer waitcnt thresholds |
| 3 | lds_accum_stage | LDS | 31 | 463872 | 3324416 | 48.42 | register accumulators (N5A regalloc-blocked) / fewer LDS round-trips |
| 4 | other | OTHER | 49 | 422208 | 3025824 | None | ? |
| 5 | pv_softmax_arith | VALU | 16 | 294912 | 2113536 | 27.6 | FMA-fuse / reduce rescale ops |
| 6 | mov | VALU | 42 | 232704 | 1667712 | 27.6 | reduce copies |
| 7 | loop_control | SALU | 14 | 154368 | 1106304 | None | unroll / fewer loop iters |
| 8 | predicate | VALU | 12 | 148224 | 1062272 | 27.6 | fewer predicate evals |
| 9 | cvt | VALU | 5 | 92160 | 660480 | 27.6 | fewer f16<->f32 converts |
| 10 | cross_lane_reduce | LDS | 5 | 92160 | 660480 | 48.42 | amortize/stage warp reduce (N3D, but PMC says LDS-wait~0) |
| 11 | kv_load | VMEM | 4 | 73728 | 528384 | None | wider/coalesced loads |
| 12 | exec_gated_store | BARRIER | 4 | 37248 | 266944 | None | reduce gated-store regions |
| 13 | exp_softmax | VALU | 2 | 36864 | 264192 | 27.6 | already hardware; fuse exp into online-softmax merge |
| 14 | fdot_score | VALU | 1 | 18432 | 132096 | 27.6 | match owned dot strategy / fewer score passes |
| 15 | softmax_max | VALU | 1 | 18432 | 132096 | 27.6 | fuse max into reduce |
| 16 | output_store | VMEM | 3 | 1536 | 11008 | None | fewer partial stores |

## Top 3 source groups + levers

- **address_index** (VALU, est_dyn512=668928, pmc_ratio=27.6): scalarize uniform prefix (N1B refuted/dead) / strength-reduce  
  sites: amd.py:isel_index / _binop V_IMUL/V_IADD/V_OFFSET
- **waitcnt** (WAIT, est_dyn512=503040, pmc_ratio=None): finer waitcnt thresholds  
  sites: amd.py:_insert_waitcnt (consumer-only)
- **lds_accum_stage** (LDS, est_dyn512=463872, pmc_ratio=48.42): register accumulators (N5A regalloc-blocked) / fewer LDS round-trips  
  sites: amd.py:isel_index LDS path / DS_LOAD/DS_STORE (DEFINE_REG accumulators + K/V staging)