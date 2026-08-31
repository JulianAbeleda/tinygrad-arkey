from extra.llm_research.prefill.nv_generated_q6k_streamk import *
from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops,UOp

def test_streamk_owner_partition_and_two_segment_bound():
  assert owner_bounds(0)==(0,36) and owner_bounds(169)==(6107,6144)
  assert {owner_work_units(o) for o in range(OWNERS)} == {36,37}
  assert sum(owner_work_units(o) for o in range(OWNERS)) == TILES*K_BLOCKS
  assert sum(len(streamk_segments(o)) for o in range(OWNERS)) <= 340
  assert max(len(streamk_segments(o)) for o in range(OWNERS)) <= 2
  assert sorted((s.begin,s.end) for o in range(OWNERS) for s in streamk_segments(o))

def test_streamk_m_major_tile_mapping_and_partial_metadata():
  assert tile_coordinates(0)==(0,0) and tile_coordinates(127)==(3,31)
  md=owner_metadata(); assert len(md)==340
  assert {s.slot for s in md} == set(range(340))
  assert all(s.tile_id < TILES for s in md if s.end > s.begin)

def test_generated_170_owner_partials_compile_with_dynamic_segments():
  ph=lambda n,dt,i: UOp.placeholder((n,),dt,i)
  ast=generated_q6k_streamk_owner_partials(ph(340*16384,dtypes.float32,0),ph(340,dtypes.int32,1),
    ph(4096*48*105,dtypes.uint16,2),ph(48*256*512,dtypes.int8,3),ph(48*8*512,dtypes.float32,4))
  src=next(x.arg for x in to_program(ast,CUDARenderer(Target.parse('NV:CUDA:sm_120'))).src if x.op is Ops.SOURCE)
  assert 'nv_generated_q6k_streamk_owner_partials' in src
  assert 'mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32' in src
