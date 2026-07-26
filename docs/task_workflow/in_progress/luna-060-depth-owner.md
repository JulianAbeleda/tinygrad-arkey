# LUNA-060: Depth-Slope Owner Confirmation

Verdict: `INCONCLUSIVE`.

## Evidence reviewed

- Retained full-model authority: tinygrad 14B is `68.39 tok/s` at context 512
  and `59.41 tok/s` at context 4096; retained llama comparator is `66.58` and
  `62.87 tok/s`, respectively.
- Retained focused replay: G5/G4 normalized tile cost is `1.0217x` at 512 and
  `1.3223x` at 4096. This proves a deep G5-focused cliff, not its contribution
  to end-to-end decode.
- `bench/.../depth-slope-model.json` assigns no per-family wall time,
  code-object identity, resources, or counters. Its unallocated residual is
  therefore 100 percent.

## Source review

`flash_decode_attention_route` selects the exact G5 binding only for
`Hq=40,Hkv=8,Hd=128,B=1`; it fixes `S=48` and `KV_BOTH`. The tile builder has
`G=Hq/Hkv`, `WARPS=G`, and `THREADS=32*G`, so G4 has 128 threads/four full
wave32 waves and G5 has 160 threads/five full wave32 waves. The grid remains
`Hkv*S = 384` workgroups for both. No source path serializes a fifth query head.

For the shipped `KV_BOTH` tile, K and V LDS tiles are each `16*128*2` bytes
(8 KiB total), independent of G. Persistent explicit per-thread state is
unchanged by G (`R=Hd/32=4` fp32 accumulator lanes plus scalar denominator,
maximum, and dot temporary). G changes staging work distribution: `STAGES` is
16 for G4 and 13 for G5. At context 512 the live loop has one block/split; at
4096 it has six. These are source-derived candidates for a long-loop
allocation/lifetime interaction, not register, spill, or occupancy evidence.

The fused combine is a distinct family: one wave32 workgroup per query head,
with four accumulator lanes/thread and 48 fp32 LDS weights. Its source shape
changes from 32 to 40 workgroups but has no retained timing/resource row.

## Owner decision and stop condition

Neither the tile nor combine can be named the full-model slope owner. A focused
kernel ratio cannot satisfy the required Amdahl bound without its final-commit
per-family device wall share. Stop LUNA-060 here; do not construct or time a
candidate until the evidence below names an owner whose maximum contribution
can close the 512-to-4096 gap.

## Required ROCm evidence

On the final Track-A commit, in fresh separate 512 and 4096 processes, retain:

1. A bounded tinygrad dispatch trace with a known tinygrad dispatch positive
   control and phase boundary for one steady token.
2. Device duration/count ledger by semantic family, including
   `flash_block_tiled_xlane_score_pv_tile_whole_cache_40_128` and
   `flash_fused_gmax_combine_40_128`, with at least 95 percent device-wall
   attribution or an explicit remainder.
3. Tile and combine code-object hash, grid/block dimensions, wave size, VGPR,
   SGPR, LDS, private/scratch, spill metadata, and occupancy limit.
4. G4 controls for the same tile at both depths with matching resource schema.
5. The Amdahl calculation using ordinary timing only; profiler timing is
   attribution evidence, never throughput authority.

If Track A changes either 512+ code object or route, all retained baselines are
stale and must be recollected before this task can be revisited.
