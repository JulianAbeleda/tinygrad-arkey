# Normalized llama Q6_K x Q8_1 MMQ oracle contract

Status: source-and-binary-audited contract for the pinned FFN-down oracle. This
record normalizes the two statically duplicated SASS compute bodies to one
logical `N128 x M128 x K256` work unit. It does not treat a static cubin census
as dynamic work.

## Identity and ABI

- Shape: `M=512, N=4096, K=12288`, Q6_K weight times Q8_1 activation.
- Pinned llama.cpp commit: `ac4cddeb0dbd778f650bf568f6f08344a06abe3a`.
- Pinned `mmq.cuh` SHA-256:
  `6d153a9d6f293a4ff5f11e7886a48bf765b21d74075d73b2097a2b2a9149de6f`.
- Pinned `mma.cuh` SHA-256:
  `c7e0f3332da182e203b4b953f9fa8535ffca3767d2b7d4d7dbf7ce486262d1af`.
- Main cubin SHA-256:
  `04eb9bcb2edef62c672b5496d743a98c57e3236558b88f2ff117964b7fbb91ca`.
- Main launch: grid `(170,1,1)`, block `(32,8,1)`.
- Main pointers, in order: canonical Q6_K, canonical/pretransposed Q8_1 D4,
  FP32 destination, FP32 fixup workspace.
- Main grouped scalar ABI widths:
  `[3,1,1,1,1,1,3,3,1,1,1,3,3,1,1,1,3]`.
- Main scalar values:
  `[1431655766,6,48,4096,512,48,512,4096,1,0,1,1,0,1,196608,1769472,2097152,1,0,1,1,0,1,196608,1769472,2097152,1,2,4]`.
- Main resources: 255 registers/thread, 72-byte stack frame, 31 static `LDL`,
  29 static `STL`, 1,024 bytes static shared. The logical shared arena reaches
  byte address `0xe600`, or 58,880 bytes total.

## Exact Stream-K ownership

The output has `32 x 4 = 128` tiles. Each tile has 48 Q6_K blocks, hence 6,144
linear work units. CTA `b` owns the half-open interval

```text
[floor(b*6144/170), floor((b+1)*6144/170))
```

The resulting census is exact:

| property | count |
|---|---:|
| CTAs with 36 work units | 146 |
| CTAs with 37 work units | 24 |
| CTAs touching one tile segment | 46 |
| CTAs touching two tile segments | 124 |
| total tile segments | 294 |
| segments ending a tile and writing destination | 128 |
| final incomplete segments writing fixup | 166 |
| tiles with two contributors | 90 |
| tiles with three contributors | 38 |

The destination-ending body executes 2,414 K256 units in aggregate. The final
partial body executes 3,730. Their sum is exactly 6,144; no padded MMA work is
issued.

## CTA, warp, and lane output ownership

One CTA computes contributions to a `128 weight/output rows x 128 activation
columns` tile. It contains eight physical 32-lane warps. With Turing/Ampere MMA
granularity 16, `rows_per_warp=32` and `ntx=2`.

| warp pair | output rows | even warp columns | odd warp columns |
|---|---|---|---|
| 0/1 | 0..31 | 0:7,16:23,...,112:119 | 8:15,24:31,...,120:127 |
| 2/3 | 32..63 | same even bands | same odd bands |
| 4/5 | 64..95 | same even bands | same odd bands |
| 6/7 | 96..127 | same even bands | same odd bands |

Each warp owns 2,048 output elements (`32 x 64`). Each lane owns 64 FP32
accumulators. For lane `t`, accumulator element `l in [0,4)`, minitile
`n in [0,2)`, and `j0 in {0,16,...,112}`, the local output coordinates are:

```text
row = floor(warp/2)*32 + n*16 + 8*floor(l/2) + floor(t/4)
col = j0 + (warp mod 2)*8 + 2*(t mod 4) + (l mod 2)
```

This is a directly evaluated form of pinned `tile<16,8,int>::get_i/get_j`.

## Shared-memory contract

The physical shared layout is exact and explains the `0xe600` footprint:

| byte range | bytes | payload |
|---|---:|---|
| `0x0000..0x03ff` | 1,024 | cubin static shared allocation |
| `0x0400..0x05ff` | 512 | 128 row/ID words |
| `0x0600..0x4dff` | 18,432 | one overwriteable `128 x 144 B` Q8 K128 panel |
| `0x4e00..0xe5ff` | 38,912 | `128 x 304 B` expanded Q6 K256 tile |

A Q8 row is 36 words: four FP32 D4 scales at words 0..3 and 128 signed
quantized bytes at words 4..35.

An expanded Q6 row is 76 words:

```text
words  0..63 : 256 signed int8 quants
word      64 : FP16 super-block d loaded into a 32-bit shared word
words 65..68 : sixteen signed int8 scales, four per word
words 69..75 : bank-conflict padding
```

The Q6 producer assigns quant row `i0+warp` to all 32 lanes for
`i0=0,8,...,120`; each lane constructs two packed signed-int8 words. D loads
map `(warp*32+lane) mod 128`, so every D is intentionally loaded/stored twice.
Scale publication uses rows `(i0+warp*8+lane/4) mod 128` for `i0=0,64`.

## One normalized K256 work unit

The source contract is simpler than the earlier schedule hypothesis. It does
not decode Q6 at IMMA consumption and it does not double-buffer K256 epochs.
For every K256 unit it executes:

```text
expand the complete Q6 K256 tile to shared
copy Q8 K[0:128] panel to the single Q8 shared window
barrier
fully unrolled vec_dot(k00=0)
barrier
overwrite Q8 window with K[128:256]
barrier
fully unrolled vec_dot(k00=32)
barrier
advance the runtime K256 loop
```

Within each K128 half, every warp preloads all 16 Q6 `ldmatrix` fragments,
packed scales, and D values. It then walks eight 16-column output bands. Each
band walks four K32 groups; each K32 group uses two generic scalar Q8 fragment
loads and two signed `m16n8k16` IMMA instructions for each of two N16
minitiles.

Therefore one half issues 128 IMMA and 16 LDSM per warp. One K256 unit issues
256 IMMA and 32 LDSM per warp. Across eight warps this exactly covers
`128*128*256 = 4,194,304` signed-int8 MACs.

The physical SASS has two copies of this body because the destination-ending
and final-partial source specializations have different writeback targets.
The whole-cubin `512 IMMA / 64 LDSM / 210 LDG` census must therefore not be
charged to one work unit.

| static warp instructions per K256 body | direct body | partial body |
|---|---:|---:|
| IMMA | 256 | 256 |
| LDSM | 32 | 32 |
| LDG | 105 | 105 |
| STS | 71 | 71 |
| BAR | 4 | 4 |
| LDS | 176 | 176 |
| I2FP | 512 | 512 |
| FFMA | 640 | 640 |
| PRMT | 80 | 80 |
| LOP3 | 205 | 205 |
| IMAD | 1,083 | 1,083 |

The 105 global-load instructions consist structurally of 69 Q6 U16 loads and
36 Q8 word loads. The 71 shared stores consist of 35 Q6 expansion stores and
18 stores for each Q8 half. Q6 source-level unique bytes are 26,880, but the
cooperative load topology intentionally duplicates high-bit and D reads; the
issued scalar-load payload is 35,328 bytes. The two Q8 panels contribute
36,864 bytes, for 72,192 issued input bytes per K256 body before cache effects.

## Physical SASS hops and backedges

The destination-ending K loop is `.L_x_6`, `0x0d80..0xeb50`. Its barriers are
at `0x3840`, `0x8620`, `0x8890`, and `0xeb40`; `0xeb50` branches back to
`0x0d80`. The outer completed-tile loop branches at `0x120c0` to `0x06c0`.

The final-partial K loop is `.L_x_137`, `0x12880..0x20780`. Its barriers are at
`0x15480`, `0x1a230`, `0x1a4d0`, and `0x20770`; `0x20780` branches back to
`0x12880`.

The second Q8 panel's 18 global loads occur at `0x80e0..0x8290` in the direct
body and `0x19d00..0x19e20` in the partial body, before the barrier ending the
first vec-dot half. Its 18 shared stores occur after that barrier. This proves
register prefetch of the next Q8 panel overlaps the tail of the current half
in program order. It does not prove the achieved latency overlap.

Direct writeback uses 64 static FP32 stores at `0xee30..0x11fc0`. Partial
writeback uses 64 at `0x20a90..0x217b0`. Every segment writes a dense 128x128
FP32 contribution tile.

## Fixup

Fixup launches `(170,4,1)` blocks of `(32,4,1)`. `blockIdx.y` plus lane selects
one of 128 output rows. Each of four warps walks 32 of the 128 output columns.
The block associated with the owner that ended a tile walks preceding owners
backward and sums their ordered fixup tiles until it reaches the owner that
started the tile. It then adds the sum to the destination-ending contribution.
Exactly 90 tiles add one partial and 38 add two. Scratch allocation is
`170*128*128*4 = 11,141,120` bytes; 166 owner slots are live for this shape.

## What is proven, and what remains inference

Proven by pinned source plus binding: shape, ABI, Stream-K formulas, all ownership
counts, warp/lane output mapping, shared row layouts, four-barrier K256 lifecycle,
fragment/MMA loop counts, writeback and fixup order.

Proven by cubin/nvdisasm: binary identity, resources, duplicated direct/partial
bodies, exact static opcode counts, barrier/backedge addresses, Q8-next-panel
load/store ordering, and store regions.

Inference only: how much actual latency is hidden by the Q8 register prefetch,
which dependency chain dominates a body without counters, and whether reproducing
this schedule in another compiler is sufficient for the timing gate.

## Corrected investment implication

The oracle does **not** support a theory that llama wins by consume-time Q6
decode, an epoch-level Q6 cache, fewer shared stores, or barrier-free K256
double buffering. It expands Q6 once per K256 work unit, performs 71 scalar
shared stores, and executes four barriers. The features that must be matched in
the next causal test are instead the full `128x128` reuse topology, exact 76-word
expanded Q6 layout, fully unrolled two-half consumer, per-warp preload of Q6
fragments/scales/D, and second-Q8-panel register prefetch. A candidate should be
invested in only if the full-route A/B is exact, spill-safe, and improves the
main latency; isolated producer timing is not normalized to the oracle's reuse.
