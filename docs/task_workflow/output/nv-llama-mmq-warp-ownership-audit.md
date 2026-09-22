# llama Q4_K MMQ warp/shared-memory ownership audit

Scope: read-only topology audit for the captured `M=512, N=12288, K=4096`
projection. Source authority is `ggml-cuda/mmq.cuh`; physical authority is the
existing NCU capture reporting 251 registers/thread, 58,880 B shared memory,
and one resident CTA/SM.

## Numeric launch and stream-K map

- CTA output tile: `N=128 x M=128`; block: `(32, 8)`, or eight warps.
- Output tile grid: `ceil(12288/128) x ceil(512/128) = 96 x 4 = 384`.
- Each output tile has `4096/256 = 16` Q4_K work units, so total work is 6144.
- The launch uses 170 CTAs. CTA `b` owns
  `[floor(b*6144/170), floor((b+1)*6144/170))`, hence 36 or 37 work units.
  This is about 2.25 output tiles per CTA.
- Complete reductions write the destination. CTA-boundary partial reductions
  write a `128 x 128` float fixup tile. A subsequent `170 x 4`-block fixup
  launch (128 threads/block) adds preceding partials to their destination tile.

## Exact shared-memory accounting

`QI8_0=8`, therefore the MMA Q4/Q8_1 expanded-weight row stride is

`2*32 + 2*32/8 + 4 = 76 dwords = 304 B`.

Dynamic shared memory is exactly:

| payload | bytes |
|---|---:|
| 128 row/block IDs | 512 |
| one Q8_1 K128 panel, `128 x 36` dwords | 18,432 |
| expanded Q4_K K256 tile, `128 x 76` dwords | 38,912 |
| total dynamic | 57,856 |

Adding the captured 1,024 B static allocation gives 58,880 B, exactly matching
NCU. A packed Q4_K block and a Q8_1 record are each 144 B. Thus one
`N128 x M128 x K256` work unit reads 18,432 B of packed weights once and two
18,432 B Q8 panels, or 55,296 B before output traffic.

## Cooperative load ownership

For Q4_K, `threads_per_row=256/(4*QR4_K)=32`. Warp `w` expands rows
`w, w+8, ..., w+120`. Each lane reads one packed dword and writes separate
low/high-nibble int8 dwords. For metadata, each lane pair owns one row; lane
parity owns four of the eight scale/min groups and writes
`half2(D*scale, -Dmin*min)`. The eight warps collectively stage all 128 rows.

The Q8 panel is loaded as a flat 4,608-dword copy: every thread loads 18
dwords. The CTA stages K values 0..127, synchronizes and computes, overwrites
the same buffer with K values 128..255, synchronizes and computes again.

## Warp output and IMMA ownership

`rows_per_warp=32`, `ntx=2`:

| warps | N rows | M columns per warp |
|---|---|---|
| 0 / 1 | 0..31 | even / odd 8-column bands |
| 2 / 3 | 32..63 | even / odd 8-column bands |
| 4 / 5 | 64..95 | even / odd 8-column bands |
| 6 / 7 | 96..127 | even / odd 8-column bands |

The even warp in a pair covers `0:7, 16:23, ..., 112:119`; the odd warp covers
`8:15, 24:31, ..., 120:127`. Each warp therefore owns `32 x 64 = 2048`
outputs. Each K128 half contains four k32 steps. Per step a warp executes eight
M-band iterations times two N16 fragments, or 16 `mma.sync` instructions:
64 per half, 128 per K256 work unit, and 1,024 per CTA/work unit.

## Reuse and occupancy consequence

- Expanded A/weight rows are shared by the two partner warps and reused over
  all M bands; logically each weight is amortized over 128 activation columns.
- A staged Q8 B band is consumed by one warp in each of four N groups, and is
  reused by both N16 A fragments inside that warp; logically each activation is
  amortized over 128 weight rows.
- The full 128-by-128 tile is what enables reuse along both axes. Shrinking it
  to admit more CTAs duplicates one or both packed payloads unless the topology
  provides another sharing mechanism.

At 251 registers/thread and 58,880 B shared memory, only one 256-thread CTA can
reside on an SM. Stream-K deliberately launches one CTA on each of 170 SMs.
The eight resident warps retain large accumulator/fragment state and exploit
the staged cross-axis reuse. This explains the topology choice, but not tensor
saturation: the capture's roughly 28.7% tensor-active and 16.65% active-warps
figures show remaining latency/scheduling headroom.
