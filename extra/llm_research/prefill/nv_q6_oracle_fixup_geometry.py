"""Isolated four-slice geometry gate for the ordered Q6 all-partials fixup."""
from __future__ import annotations

from dataclasses import asdict, dataclass


M, N = 512, 4096
ROWS = COLS = 128
TILES_M, TILES_N, TILES = 4, 32, 128
TILE_ELEMS = ROWS * COLS
SLICES = 4
THREADS = 128
OUTPUTS_PER_THREAD = 32
SLICE_ELEMS = TILE_ELEMS // SLICES
GRID = (TILES, SLICES, 1)
BLOCK = (THREADS, 1, 1)
SYMBOL = "nv_q6_ordered_fixup_all_partials_four_slice_scatter"


@dataclass(frozen=True)
class FourSliceFixupContract:
  grid: tuple[int, int, int] = GRID
  block: tuple[int, int, int] = BLOCK
  outputs_per_thread: int = OUTPUTS_PER_THREAD
  slices_per_tile: int = SLICES
  logical_writers: int = TILES * SLICES * THREADS * OUTPUTS_PER_THREAD
  destination_warp_stride_bytes: int = N * 4
  output_policy: str = "all_partials"
  descriptor_order: str = "frozen schedule order: descending predecessors, then final"
  scratch_layout: str = "plane-major slot, row-major tile element: slot*16384+z"
  destination_layout: str = "(mt*128+mc)*4096+nt*128+wr"
  reset_us: float = 0.0

  def record(self) -> dict[str, object]: return asdict(self)


CONTRACT = FourSliceFixupContract()


def z_index(slice_index:int, lane:int, iteration:int) -> int:
  if not 0 <= slice_index < SLICES: raise ValueError(slice_index)
  if not 0 <= lane < THREADS: raise ValueError(lane)
  if not 0 <= iteration < OUTPUTS_PER_THREAD: raise ValueError(iteration)
  return slice_index * SLICE_ELEMS + lane + THREADS * iteration


def destination_index(tile:int, z:int) -> int:
  if not 0 <= tile < TILES: raise ValueError(tile)
  if not 0 <= z < TILE_ELEMS: raise ValueError(z)
  mt, nt = tile % TILES_M, tile // TILES_M
  wr, mc = z // COLS, z % COLS
  return (mt * COLS + mc) * N + nt * ROWS + wr


def contract_record() -> dict[str, object]:
  return CONTRACT.record() | {
    "active_blocks": TILES * SLICES,
    "warps_per_block": THREADS // 32,
    "active_warps": TILES * SLICES * THREADS // 32,
    "slice_ranges": [[i * SLICE_ELEMS, (i + 1) * SLICE_ELEMS] for i in range(SLICES)],
    "scratch_reads": "unchanged: one fp32 value per valid descriptor and output element",
    "destination_writes": M * N,
    "unique_writer_proof": "four disjoint z intervals; lane+128*iteration bijects each interval",
    "forbidden_mechanisms": ["atomics", "membar", "spin", "counter", "reset"],
  }


def four_slice_scatter_source() -> str:
  # The outer loop intentionally remains rolled. This gate changes launch geometry only,
  # retaining the admitted scratch and scattered destination address expressions.
  return f'''#include <cuda_runtime.h>
extern "C" __global__ __launch_bounds__({THREADS}) void {SYMBOL}(
    float *out, const float *partials, const int *slots, const int *counts) {{
  const int tile = (int)blockIdx.x;
  const int slice = (int)blockIdx.y;
  const int lane = (int)threadIdx.x;
  const int count = counts[tile];
  const int slot0 = slots[tile*3 + 0];
  const int slot1 = slots[tile*3 + 1];
  const int slot2 = slots[tile*3 + 2];
  #pragma unroll 1
  for (int iteration = 0; iteration < {OUTPUTS_PER_THREAD}; iteration++) {{
    const int z = slice*{SLICE_ELEMS} + lane + {THREADS}*iteration;
    float acc = 0.0f;
    if (count > 0) acc = __fadd_rn(acc, partials[slot0*{TILE_ELEMS} + z]);
    if (count > 1) acc = __fadd_rn(acc, partials[slot1*{TILE_ELEMS} + z]);
    if (count > 2) acc = __fadd_rn(acc, partials[slot2*{TILE_ELEMS} + z]);
    const int wr = z/{COLS};
    const int mc = z%{COLS};
    const int mt = tile%{TILES_M};
    const int nt = tile/{TILES_M};
    out[(mt*{COLS} + mc)*{N} + nt*{ROWS} + wr] = acc;
  }}
}}
'''


__all__ = ["M", "N", "ROWS", "COLS", "TILES_M", "TILES_N", "TILES", "TILE_ELEMS", "SLICES", "THREADS",
  "OUTPUTS_PER_THREAD", "SLICE_ELEMS", "GRID", "BLOCK", "SYMBOL", "CONTRACT", "z_index", "destination_index",
  "contract_record", "four_slice_scatter_source"]
