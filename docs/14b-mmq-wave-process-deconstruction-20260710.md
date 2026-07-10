# 14B MMQ Wave Process Deconstruction

Purpose: deconstruct the vendored llama.cpp MMQ source enough to guide the
tinygrad R4 cooperative-tile atom. This is research evidence only. It does not
change production dispatch.

Source:

```text
extra/qk/research/llama_mmq/mmq.cuh
source commit: ac4cddeb0dbd778f650bf568f6f08344a06abe3a
sha256: 6d153a9d6f293a4ff5f11e7886a48bf765b21d74075d73b2097a2b2a9149de6f
```

## RDNA3 Shape

For gfx1100/HIP:

```text
warp_size = 64
nwarps = 8
mmq_y = 128
mmq_x <= 128, chosen by shared-memory fit and output-column tiling
ITER_K = 256
block_dims = (warp_size, nwarps, 1) = (64, 8, 1)
```

Interpretation:

```text
threadIdx.y = wave id inside CTA, 0..7
threadIdx.x = lane id inside wave, 0..63
one CTA covers an output tile of mmq_y rows by mmq_x columns
```

The host launch is:

```text
block_nums = (ceil(nrows_x/mmq_y), ceil(ncols_max/mmq_x), channels*samples)
block_dims = (warp_size, nwarps, 1)
```

## Shared Memory Layout

`mul_mat_q_process_tile` declares one dynamic shared buffer:

```text
extern __shared__ int data_mul_mat_q[]
ids_dst lives at data_mul_mat_q[0:mmq_x]
tile_y = data_mul_mat_q + mmq_x
tile_x = tile_y + padded(mmq_x * MMQ_TILE_Y_K)
```

Shared-memory byte estimate:

```text
nbs_ids = mmq_x * sizeof(int)
nbs_y   = mmq_x * sizeof(block_q8_1_mmq)
nbs_x   = mmq_y * mmq_tile_x_k * sizeof(int) on MMA/WMMA/MFMA paths
total   = nbs_ids + nbs_x + padded(nbs_y, nwarps*warp_size*sizeof(int))
```

This means the tile has three separate responsibilities:

```text
ids_dst: column remap / sorted row indirection
tile_y: Q8_1 activation tile, already in MMQ DS layout
tile_x: Q4_K weight tile unpacked for dot product
```

## K-Loop Process

Inside one output tile, every CTA walks K in `ITER_K=256` chunks:

```text
for kb0 in [kb0_start, kb0_stop) step blocks_per_iter:
  load_tiles_q4_K(...) into tile_x
  copy Q8_1 MMQ blocks into tile_y
  barrier
  vec_dot(tile_x, tile_y, sum, 0)
  barrier
  copy next Q8_1 panel into tile_y
  barrier
  vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K)
  barrier
```

The important pattern is not just LDS staging. It is two activation-panel loads
per Q4_K weight tile, separated by barriers, accumulating into the same
per-thread `sum[]` registers.

## Q4_K Tile Loader Wave Mapping

For the Q4_K MMA/WMMA/MFMA path:

```text
threads_per_row = MMQ_ITER_K / (4 * QR4_K)
nrows = warp_size / threads_per_row
txi = threadIdx.x % threads_per_row
```

On gfx1100 wave64, this maps lanes within each wave across Q4_K row fragments.
The loader writes:

```text
x_qs[row * MMQ_MMA_TILE_X_K_Q8_1 + packed_k_offsets]
x_dm[row * MMQ_MMA_TILE_X_K_Q8_1 + scale/min_offsets]
```

So `load_tiles_q4_K` distributes the weight-tile unpack over all waves and
lanes, but it writes into a CTA-shared `tile_x` that all waves use for the dot.

## Accumulator Ownership

The accumulator is local per thread:

```text
float sum[mmq_x * mmq_y / (nwarps * warp_size)]
```

For gfx1100 with `mmq_x=128`, `mmq_y=128`, `nwarps=8`, `warp_size=64`:

```text
sum elements per thread = 128 * 128 / (8 * 64) = 32
```

That is the critical hand-kernel trick: the output tile is cooperative at the
CTA level, but each output element has one final owner in one thread's local
accumulator array.

## MMA Writeback Ownership

On AMD WMMA/MFMA-capable paths, writeback uses `mmq_write_back_mma`.

Key constants:

```text
tile_C = tile<16, 16, int, DATA_LAYOUT_J_MAJOR> on gfx1100-style AMD path
rows_per_warp = granularity
ntx = rows_per_warp / tile_C::I
i0 = (threadIdx.y / ntx) * (ntx * tile_C::I)
static_assert(nwarps * tile_C::I == mmq_y)
```

For the common `mmq_x=128`, `mmq_y=128`, `nwarps=8`, `tile_C::I=16` case:

```text
8 waves * 16 output rows per wave = 128 output rows
```

Each wave owns a fixed 16-row stripe:

```text
wave 0 -> rows 0..15
wave 1 -> rows 16..31
wave 2 -> rows 32..47
wave 3 -> rows 48..63
wave 4 -> rows 64..79
wave 5 -> rows 80..95
wave 6 -> rows 96..111
wave 7 -> rows 112..127
```

Within that stripe, the writeback loop walks columns in `tile_C::J=16`
fragments. Each `l` in `tile_C::ne` maps to one `(i,j)` inside the 16x16 output
fragment:

```text
j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l)
i = i0 + n * tile_C::I + tile_C::get_i(l)
dst[ids_dst[j] * stride + i] = sum[(j0/tile_C::J + n) * tile_C::ne + l]
```

For `ntx=1`, this is simpler:

```text
threadIdx.y chooses the 16-row stripe
j0 walks 16-column fragments
l maps one element inside the 16x16 fragment
```

The output ownership contract is:

```text
owner = (wave_id, j_fragment, l)
each legal (i,j) output maps to exactly one owner
no duplicate stores
no missing stores
```

This is the exact part tinygrad R4 lacks today.

## DP4A Contrast

The DP4A writeback is structurally different:

```text
for j0 step nwarps:
  j = j0 + threadIdx.y
  for i0 step warp_size:
    i = i0 + threadIdx.x
    dst[...] = sum[(j0/nwarps) * (mmq_y/warp_size) + i0/warp_size]
```

That model maps one wave to columns and lanes to rows. The AMD WMMA path maps
waves to row stripes and `tile_C` fragments to the final output element layout.
For gfx1100 R4, the MMA writeback model is the relevant one.

## What tinygrad Must Represent

The R4 atom does not need to become a generic copy of llama.cpp. It must
represent these invariants:

```text
1. CTA shape: 8 waves, physical wave64, shared tile scope.
2. Shared tile buffers: ids_dst, Q8_1 tile_y, Q4_K tile_x.
3. K loop: load Q4_K once, load two Q8_1 panels, barrier between phases.
4. Per-thread accumulators: 32 fp32 slots for 128x128 on wave64.
5. Writeback owner map: wave_id owns a 16-row stripe and writes 16-column fragments.
6. Store validator: every output in the bounded tile is stored exactly once.
```

Minimal bounded R4 proof should not start at full 128x128. It should preserve
the same ownership law on smaller legal shapes:

```text
16x16x256: one wave/fragment owner is enough to validate tile_C mapping
32x16x256: two row stripes validate wave row ownership
32x32x256: two row stripes and two column fragments validate j0 traversal
128x128x256: full owner map, 64 fragments
```

## Immediate Machine-Search Fields

The next R4 search row should emit:

```text
mmq_x, mmq_y, iter_k, nwarps, warp_size
tile_c_i, tile_c_j, tile_c_ne
threadIdx.y -> row stripe map
j0 -> column fragment map
sum_index expression
store_index expression
expected owner hash
actual owner hash
missing_store_count
duplicate_store_count
production_dispatch_changed=false
default_route=direct_packed
```

Promotion remains illegal until the actual tinygrad atom produces the same
owner map and passes bounded numeric correctness against the DS4 reference.

## Working Theory

The current theory is:

```text
llama MMQ is fast because it separates tile cooperation from final-store ownership.
```

More concretely:

```text
1. All 8 waves cooperate to stage the Q4_K and Q8_1 panels into CTA-local memory.
2. All waves reuse the staged data across a 128x128 output tile and a 256-wide K slice.
3. Accumulation is not a global/shared-memory reduction. It is partitioned into per-thread sum[] slots.
4. The final writeback is deterministic: one wave owns one 16-row stripe; j0/l own the 16-column fragment and element.
5. The performance win comes from reuse and predictable ownership, not from arbitrary parallel stores.
```

The “why” behind each piece:

| Mechanism | Why it exists | Refutable claim |
|---|---|---|
| `mmq_y=128`, `nwarps=8` | matches `8 * 16` output-row stripes on AMD MMA writeback | if `nwarps * tile_C::I != mmq_y`, writeback cannot cover rows exactly |
| `mmq_x<=128` | selected to minimize N tiles while fitting shared memory | if `mmq_x` grows past shared-memory fit, launch rejects it |
| `ITER_K=256` | matches Q4_K block width and one MMQ K slice | partial K slices need separate fixup/loop handling |
| `tile_y`/`tile_x` shared buffers | amortizes packed dequant/load work across many outputs | direct per-output loads lose reuse |
| `sum[]` per thread | avoids a separate inter-wave output reduction | duplicate stores appear if multiple waves own same `(m,n)` |
| `mmq_write_back_mma` | converts per-thread fragment accumulators to final output addresses | owner coverage must be exact: no missing, no duplicate |

## What We Know vs. What We Still Need To Prove

Known from source and tests:

```text
K0: gfx1100 HIP compiles this CUDA-named source into libggml-hip.so.
K1: RDNA3 uses wave64, nwarps=8, mmq_y=128.
K2: Q4_K MMQ uses the MMA/WMMA writeback path on gfx1100.
K3: The oracle owner map covers 128x128 as 64 independent 16x16 fragments.
K4: Expanded owner coverage has no duplicate or missing stores for 16x16, 32x16, 32x32, and 128x128.
```

Still to prove with instrumentation or bounded atoms:

```text
P1: The tinygrad UOp atom can represent the same wave_id -> row-stripe ownership.
P2: The atom can keep the same per-thread sum[] ownership without spilling or changing store ownership.
P3: The atom can stage both Q4_K and Q8_1 panels with the same barrier lifecycle.
P4: Resource usage stays within a useful bound: no scratch, acceptable VGPR, acceptable LDS.
P5: Once correct, the route-bound candidate moves whole prefill numbers, not only a toy tile.
```

## Test Ladder

The test ladder should advance in this order:

| Test | Purpose | Pass condition |
|---|---|---|
| owner-map oracle | prove our interpretation of llama writeback | expected output count equals covered count, duplicates=0, missing=0 |
| tinygrad owner trace | prove the atom emits the same ownership without caring about values | owner hash equals oracle hash |
| tinygrad store-only kernel | prove each output is stored exactly once | output marker matrix has all ones |
| tinygrad numeric 16x16x256 | prove formula + ownership on one fragment | matches DS4 reference |
| tinygrad numeric 32x32x256 | prove multi-wave/multi-column fragments | matches DS4 reference |
| tinygrad numeric 128x128x256 | prove full llama tile | matches DS4 reference |
| resource trace | prove it is a plausible kernel, not only correct | scratch=0, bounded LDS/VGPR, stable code hash |
| same-session comparator | prove it can beat direct packed on bounded shape | bounded win before any promotion |

## Current Executable Theory Hooks

The executable hooks now are:

```text
extra.qk.mmq_llama_oracle.llama_mma_writeback_owners
extra.qk.mmq_llama_oracle.llama_mma_writeback_coverage
test/unit/test_mmq_llama_oracle.py
```

The coverage helper expands every 16x16 owner fragment into individual output
points. Any R4 atom can now report an `actual_store_coverage` with the same
shape and compare:

```text
owner_fragment_count
covered_output_count
expected_output_count
duplicate_store_count
missing_store_count
owner hash
```

If any of those differ, the working theory or the atom translation is wrong.
