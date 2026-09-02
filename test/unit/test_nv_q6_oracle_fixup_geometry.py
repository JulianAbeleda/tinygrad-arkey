import numpy as np

from extra.llm_research.prefill.nv_q6_oracle_fixup_geometry import (
  BLOCK, GRID, M, N, OUTPUTS_PER_THREAD, SLICES, SLICE_ELEMS, SYMBOL, THREADS, TILE_ELEMS, TILES,
  contract_record, destination_index, four_slice_scatter_source, z_index)
from extra.llm_research.prefill.nv_q6_oracle_reduction_policy import build_reduction_schedule


def test_four_slice_geometry_is_an_exact_unique_writer_partition():
  assert GRID==(128,4,1) and BLOCK==(128,1,1)
  assert OUTPUTS_PER_THREAD==32 and SLICES*THREADS*OUTPUTS_PER_THREAD==TILE_ELEMS
  local=[]
  for slice_index in range(SLICES):
    values=np.array([[z_index(slice_index,lane,iteration) for lane in range(THREADS)]
                     for iteration in range(OUTPUTS_PER_THREAD)],dtype=np.int32).reshape(-1)
    assert values.min()==slice_index*SLICE_ELEMS and values.max()==(slice_index+1)*SLICE_ELEMS-1
    assert np.array_equal(np.sort(values),np.arange(slice_index*SLICE_ELEMS,(slice_index+1)*SLICE_ELEMS,dtype=np.int32))
    local.append(values)
  assert np.array_equal(np.sort(np.concatenate(local)),np.arange(TILE_ELEMS,dtype=np.int32))
  seen=np.zeros(M*N,dtype=np.uint8)
  z=np.arange(TILE_ELEMS,dtype=np.int32)
  for tile in range(TILES):
    addresses=np.fromiter((destination_index(tile,int(value)) for value in z),dtype=np.int64,count=TILE_ELEMS)
    assert np.unique(addresses).size==TILE_ELEMS
    seen[addresses]+=1
  assert np.all(seen==1)


def test_four_slice_gate_preserves_descriptor_order_and_scattered_mapping():
  schedule=build_reduction_schedule();slots,counts,_=schedule.arrays()
  assert len(schedule.records)==294 and counts.tolist().count(2)==90 and counts.tolist().count(3)==38
  for tile,row in enumerate(schedule.ordered_by_tile):
    assert tuple(slots[tile,:counts[tile]])==tuple(item.slot for item in row)
    assert row[-1].is_final and tuple(item.owner for item in row[:-1])==tuple(sorted((item.owner for item in row[:-1]),reverse=True))
  warp=np.array([destination_index(0,z_index(0,lane,0)) for lane in range(32)],dtype=np.int64)
  assert np.all(np.diff(warp)==N)
  record=contract_record()
  assert record["active_blocks"]==512 and record["active_warps"]==2048
  assert record["destination_warp_stride_bytes"]==N*4


def test_four_slice_source_is_geometry_only_and_progress_safe():
  source=four_slice_scatter_source();lower=source.lower()
  assert f"void {SYMBOL}(" in source
  assert "float *out, const float *partials, const int *slots, const int *counts" in source
  assert "blockIdx.x" in source and "blockIdx.y" in source and "threadIdx.x" in source
  assert f"iteration < {OUTPUTS_PER_THREAD}" in source and "#pragma unroll 1" in source
  assert f"slice*{SLICE_ELEMS} + lane + {THREADS}*iteration" in source
  assert f"partials[slot0*{TILE_ELEMS} + z]" in source
  assert source.count("__fadd_rn") == 3
  assert f"*{N} + nt*128 + wr" in source
  for forbidden in ("atomic", "membar", "threadfence", "syncthreads", "while", "counter"):
    assert forbidden not in lower
