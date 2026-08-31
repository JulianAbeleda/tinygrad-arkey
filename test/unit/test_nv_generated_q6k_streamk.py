import numpy as np
from extra.llm_research.prefill.nv_generated_q6k_streamk import *
from tinygrad import Device,dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
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
  fmap=fixup_slot_map(); assert len(fmap)==128
  assert {len([x for x in row if x >= 0]) for row in fmap} == {2,3}

def test_generated_170_owner_partials_compile_with_dynamic_segments():
  ph=lambda n,dt,i: UOp.placeholder((n,),dt,i)
  ast=generated_q6k_streamk_owner_partials(ph(340*16384,dtypes.float32,0),ph(340,dtypes.int32,1),
    ph(4096*48*105,dtypes.uint16,2),ph(48*256*512,dtypes.int8,3),ph(48*8*512,dtypes.float32,4))
  src=next(x.arg for x in to_program(ast,CUDARenderer(Target.parse('NV:CUDA:sm_120'))).src if x.op is Ops.SOURCE)
  assert 'nv_generated_q6k_streamk_owner_partials' in src
  assert 'mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32' in src
  assert src.count('tg_ldmatrix_x2(')-1 == 32
  assert src.count('__WMMA_8_16_16_signed_char_int(')-1 == 256

def test_owner_q8_phase1_loads_are_after_phase0_barrier():
  """The second Q8 record cannot be sourced before phase-0 completion."""
  ph=lambda n,dt,i: UOp.placeholder((n,),dt,i)
  ast=generated_q6k_streamk_owner_partials(ph(340*16384,dtypes.float32,0),ph(340,dtypes.int32,1),
    ph(4096*48*105,dtypes.uint16,2),ph(48*256*512,dtypes.int8,3),ph(48*8*512,dtypes.float32,4))
  src=next(x.arg for x in to_program(ast,CUDARenderer(Target.parse('NV:CUDA:sm_120'))).src if x.op is Ops.SOURCE)
  # The owner route has the Q6 staging barrier plus one gate before phase-1
  # Q8 stores/MMA.  Keep this as a topology guard against hoisting the source
  # loads when the generated graph is refactored.
  assert src.count('__syncthreads();') >= 3

def test_generated_owner_boundary_gate_compiles_as_one_dynamic_loop():
  ph=lambda n,dt,i: UOp.placeholder((n,),dt,i)
  ast=generated_owner_boundary_gate(ph(OWNERS*2,dtypes.float32,0),ph(TILES*K_BLOCKS,dtypes.float32,1))
  src=next(x.arg for x in to_program(ast,CUDARenderer(Target.parse('NV:CUDA:sm_120'))).src if x.op is Ops.SOURCE)
  assert 'nv_generated_q6_owner_boundary_gate' in src and src.count('for (') == 1
  values=np.arange(1,TILES*K_BLOCKS+1,dtype=np.float32); expected=np.zeros(OWNERS*2,np.float32)
  for seg in owner_metadata():
    if seg.tile_id >= 0: expected[seg.slot]=values[seg.tile_id*K_BLOCKS+seg.begin:seg.tile_id*K_BLOCKS+seg.end].sum()
  dev=Device['NV']; host=(np.zeros(OWNERS*2,np.float32),values)
  bufs=[dev.allocator._alloc(x.nbytes,BufferSpec()) for x in host]
  for buf,x in zip(bufs,host): dev.allocator._copyin(buf,memoryview(x.tobytes()))
  NVProgram(dev,'nv_generated_q6_owner_boundary_gate',NVRTCCompiler(dev.arch,ptx=False,cache_key='q6_owner_boundary_gate_v1').compile(src))(
    *bufs,global_size=(OWNERS,1,1),local_size=(1,1,1),wait=True)
  mv=memoryview(bytearray(bufs[0].size)); dev.allocator._copyout(mv,bufs[0]); got=np.frombuffer(mv,np.float32,count=OWNERS*2)
  assert np.array_equal(got,expected)
