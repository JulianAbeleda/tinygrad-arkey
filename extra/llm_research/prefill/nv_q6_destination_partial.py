"""Destination-major partial workspace contract for the Q6 Stream-K route."""

ROWS = COLS = 128
TILES_M, TILES_N, TILES = 4, 32, 128
TILE_ELEMS = ROWS * COLS
SLICES, THREADS, OUTPUTS_PER_THREAD = 4, 128, 32
GRID, BLOCK = (TILES, SLICES, 1), (THREADS, 1, 1)
SYMBOL = "nv_q6_destination_major_fixup"


def partial_index(row:int, col:int) -> int:
  """Destination-major tile offset: output-column first, activation-row second."""
  if not 0 <= row < ROWS or not 0 <= col < COLS: raise ValueError((row,col))
  return col * ROWS + row


def destination_index(tile:int, row:int, col:int) -> int:
  if not 0 <= tile < TILES: raise ValueError(tile)
  mt,nt=tile%TILES_M,tile//TILES_M
  return (mt*COLS+col)*(TILES_N*ROWS)+nt*ROWS+row


def destination_major_fixup_source() -> str:
  return f'''#include <cuda_runtime.h>
extern "C" __global__ __launch_bounds__({THREADS}) void {SYMBOL}(
    float *out, const float *partials, const int *slots, const int *counts) {{
  const int tile=(int)blockIdx.x, slice=(int)blockIdx.y, lane=(int)threadIdx.x;
  const int count=counts[tile], slot0=slots[tile*3], slot1=slots[tile*3+1], slot2=slots[tile*3+2];
  #pragma unroll 1
  for (int iteration=0; iteration<{OUTPUTS_PER_THREAD}; iteration++) {{
    const int row=lane, col=slice*{OUTPUTS_PER_THREAD}+iteration, z=col*{ROWS}+row;
    float acc=0.0f;
    if (count>0) acc=__fadd_rn(acc,partials[slot0*{TILE_ELEMS}+z]);
    if (count>1) acc=__fadd_rn(acc,partials[slot1*{TILE_ELEMS}+z]);
    if (count>2) acc=__fadd_rn(acc,partials[slot2*{TILE_ELEMS}+z]);
    const int mt=tile%{TILES_M}, nt=tile/{TILES_M};
    out[(mt*{COLS}+col)*{TILES_N*ROWS}+nt*{ROWS}+row]=acc;
  }}
}}
'''


__all__ = ["ROWS","COLS","TILES","TILE_ELEMS","GRID","BLOCK","SYMBOL","partial_index","destination_index",
  "destination_major_fixup_source"]
