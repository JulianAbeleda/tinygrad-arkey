# Q4_K MMVQ missing-quadrant design (2026-06-18)

From `llama-q4k-mmvq-scheduler-audit-20260618.md`. The audit proved llama's 70% is a **work decomposition**
tinygrad never tried: **~128 threads/row, K-blocks parallelized across threads (no serial loop), in-kernel
warp-shuffle + shared reduction, one write.** The prior fused-coop-row arc tested only 8 threads/row + serial
blk loop + LDS (→53%). **The next build is EARNED.** Design below; build-only in the next task.

## Recommendation: B — fused cooperative-row WARP-reduce (llama-shaped)

### Mapping
- **Workgroup = N_warps × 32 lanes** (start N_warps=4 → 128 threads), computes **1 output row** (`rows_per_block=1`;
  grid.x = nrows). Optionally rows_per_block=2-4 later for q8 reuse + occupancy.
- **K-block parallelism across threads:** the 16 K-blocks (ffn_gate/up in=4096 → 16 blocks of 256) × 8
  within-block sub-lanes = 128 work-items, mapped to the 128 threads. **Each thread does ONE block-sub-chunk —
  NO serial blk REDUCE loop** (the key change from the coop kernel). For 128 threads / 16 blocks = 8 threads per
  block (the within-block coalesced split, = the existing lane4=pos//4).
  - lane (threadIdx.x, 0..31) and warp (threadIdx.y, 0..3): map `(warp, lane)` → `(block, sub-lane)`. E.g.
    block = (warp*32+lane)//8, sub = (warp*32+lane)%8 → 128 work-items over 16 blocks × 8.
- **Coalescing:** consecutive lanes → consecutive sub-chunks of the same block → consecutive quant words (as in
  the current packed-load coop). Preserved.

### Dot + accumulator
- Each thread: **one register accumulator** (fp32). Compute its block-sub-chunk's contribution via the existing
  `_q4k_block_dot_packed_load` (fp) OR `_sdot4` (native dot4) — start fp for correctness, swap to `_sdot4` after.
- Scale decode: per-thread for its block (8 threads share a block → 8× redundant *within* the block group, same
  as llama; acceptable — it is NOT the bottleneck).

### Reduction (the missing piece) — warp-shuffle + shared, IN-KERNEL
- **Within-warp (32→1):** `extra/amd_warp_reduce.warp_reduce_sum(acc, lane)` — `ds_bpermute` shuffle, shape-safe
  CUSTOMI, lane tied to a real `lidx` thread dim (per the module's gotchas). NOT `OptOps.GROUP` (it produces
  err 0.95 on custom-kernel hand-rolled reduces — refuted).
- **Cross-warp (4→1):** the 4 warp-leaders write to a 4-element LDS buffer, `barrier`, warp 0 lane 0 sums.
- **One output write** per row (warp 0, lane 0): `dst[row] = block_d*sumf_d - block_dmin*sumf_m` epilogue.

### Spec
| param | value | note |
|---|---|---|
| threads/workgroup | 128 (4 warps) | match llama nwarps=4 |
| rows/workgroup | 1 (then try 2-4) | rows=1 first for simplicity |
| K-blocks/thread | 1 sub-chunk (no serial loop) | the key change vs coop's serial blk |
| within-warp reduce | `ds_bpermute` warp_reduce_sum | from amd_warp_reduce.py |
| cross-warp reduce | 4-elem LDS + barrier | small |
| output writes | 1/row | vs coop's 8 partials + stage-2 |
| accumulators | 1 fp32/thread | register |

### Expected
- **bottleneck after this:** if it works, weight-read bandwidth (→ approaching READRAW/llama 70%). If it stalls,
  the cross-warp LDS/barrier or ds_bpermute overhead.
- **isolated gate (next build):** correct (fp-reassoc tol) AND native dot4/warp-shuffle emitted AND **≥1.15× fp
  coop (≥55%)**, ideally ≥60% (toward llama 70%); reject >peak/less-work.

### Risks
- `ds_bpermute` lane-divergence gotchas (the module documents staging through a REG — must follow exactly).
- Mapping 16 blocks × 8 sub-lanes → 128 threads cleanly (16 not a multiple needing care at the warp boundary —
  16×8=128 exact, clean).
- Cross-warp LDS barrier overhead (small, 4 values).
- The custom_kernel cross-warp single-write plumbing that blocked the prior LDS attempt — but here the within-warp
  reduce is warp-shuffle (no LDS store per lane), and only 4 warp-leaders touch LDS, so the write structure is
  simpler (warp 0 lane 0 writes).

## Verdict: next build EARNED (recommendation B)
llama proves 70% is reachable via this exact decomposition; tinygrad has the enabling primitives (`_sdot4`,
`ds_bpermute` warp_reduce_sum) and has NOT tested 128-threads/row + K-parallel + warp-shuffle. This is a genuinely
untried quadrant, not a refuted transform. Build it next (correctness-first fp, then `_sdot4`), isolated-gate
≥55% (toward 70%).

## Files
`docs/llama-q4k-mmvq-scheduler-audit-20260618.md`, this doc, `bench/qk-q4k-mmvq-scheduler-audit/result.json`.
No code/model changes (audit/design only).
